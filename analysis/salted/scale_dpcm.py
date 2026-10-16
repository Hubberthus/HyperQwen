"""Approach A spike: cross-layer scale DPCM (delta coding) — scale stream compression.

Prior probe (crosslayer_probe) showed H(scale | same-position scale in prev layer)
~7.80-7.83 bits vs pooled marginal ~8.0 -> small but real cross-layer structure.
This spike measures what a per-family delta scheme actually saves:
  Arm 1: d = bin(s_N) - bin(s_{N-1}) mod 256, entropy over 256-symbol alphabet;
         stream = layer-0 scales at full 8 bits + 63 layers at H(delta) bits/scale.
  Arm 2: delta-of-delta (same binning), stream = 8 + H(d1)*1 + 62*H(d2).
  Recon: layer-0 binned codes + mod-256 deltas must reproduce layer N binned
         codes exactly (checked at L1 and L63, full arrays).

Binning: 256 equal-frequency bins, quantiles pooled across all 64 layers per
family (same recipe as crosslayer_probe). Deltas are cyclic mod 256, so the
stream is losslessly reconstructable given layer-0 codes.

Readout: bits/scale saved vs the 8-bit baseline, and MB saved on the full
checkpoint at fp16 scale storage (2 bytes/scale).

Run: venv/bin/python analysis/salted/scale_dpcm.py   (budget 15 min CPU)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "venv" / "lib" / "python3.12" / "site-packages"))
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32  # noqa: E402

CKPT = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
OUT = REPO / "analysis" / "salted" / "scale_dpcm.json"
NL = 64
BINS = 256
BYTES_PER_SCALE_FP16 = 2

_INDEX = None


def shard_for(key):
    global _INDEX
    if _INDEX is None:
        _INDEX = json.load(open(CKPT / "model.safetensors.index.json"))["weight_map"]
    return CKPT / _INDEX[key]


_FILES = {}


def get_file(key):
    path = shard_for(key)
    if path not in _FILES:
        _FILES[path] = safe_open(path, framework="pt", device="cpu")
    return _FILES[path]


def get_scales(fam, layer):
    key = f"model.language_model.layers.{layer}.mlp.{fam}.weight_scale"
    return get_file(key).get_tensor(key).to(torch.float64).numpy()


def entropy(b, n_sym):
    counts = np.bincount(b.ravel(), minlength=n_sym).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def pin_check():
    key = "model.language_model.layers.31.mlp.gate_proj.weight_packed"
    f = get_file(key)
    packed = f.get_tensor(key)
    shape = tuple(f.get_tensor("model.language_model.layers.31.mlp.gate_proj.weight_shape").tolist())
    ref = unpack_from_int32(packed, 4, shape)
    rows, cols = shape
    p = packed.to(torch.int64).numpy()
    up = np.empty((rows, cols), dtype=np.int64)
    for i in range(8):
        up[:, i::8] = (p >> (4 * i)) & 15
    diff = int(np.abs((up - 8) - ref.numpy()).max())
    print(f"  PIN L31 gate_proj: max|manual-ct| = {diff} over {ref.numel()} codes")
    assert diff == 0
    return diff


def main():
    pin = pin_check()
    results = {"pin_max_diff": pin, "families": {}}

    for fam in ["gate_proj", "down_proj"]:
        n_val = int(get_scales(fam, 0).size)
        all_flat = np.concatenate([get_scales(fam, L).ravel() for L in range(NL)])
        qs = np.quantile(all_flat, np.linspace(0, 1, BINS + 1))
        del all_flat
        codes = np.empty((NL, n_val), dtype=np.uint8)
        for L in range(NL):
            codes[L] = (np.searchsorted(qs, get_scales(fam, L).ravel(), side="right") - 1).astype(np.uint8)
        c = codes.astype(np.int16)

        # Arm 1: mod-256 deltas
        d1 = (c[1:] - c[:-1]) % BINS  # (63, n)
        H1 = entropy(d1, BINS)

        # Arm 2: delta of delta
        d2 = (d1[1:] - d1[:-1]) % BINS  # (62, n)
        H2 = entropy(d2, BINS)

        bits_arm1 = 8.0 + 63.0 * H1
        bits_arm2 = 8.0 + H1 + 62.0 * H2
        bits_base = 8.0 * NL

        # Reconstruction check: layer-0 codes + stored deltas -> exact codes
        recon_diffs = {}
        for L in (1, 63):
            rec = c[0].copy()
            for l in range(1, L + 1):
                rec = (rec + (c[l] - c[l - 1]) % BINS) % BINS
            recon_diffs[str(L)] = int(np.abs(rec - c[L]).max())
        assert all(v == 0 for v in recon_diffs.values())

        # Byte savings on full checkpoint (fp16 scale storage)
        base_bytes = NL * n_val * BYTES_PER_SCALE_FP16
        save1 = (bits_base - bits_arm1) / 8.0 * n_val * BYTES_PER_SCALE_FP16
        save2 = (bits_base - bits_arm2) / 8.0 * n_val * BYTES_PER_SCALE_FP16

        results["families"][fam] = {
            "scales_per_layer": n_val,
            "H_delta_bits_per_scale": round(H1, 4),
            "H_delta2_bits_per_scale": round(H2, 4),
            "stream_bits_arm1": round(bits_arm1, 2),
            "stream_bits_arm2": round(bits_arm2, 2),
            "stream_bits_baseline": bits_base,
            "baseline_MB": round(base_bytes / 1e6, 2),
            "saved_MB_arm1": round(save1 / 1e6, 2),
            "saved_MB_arm2": round(save2 / 1e6, 2),
            "recon_max_diff": recon_diffs,
        }
        print(f"[{fam}] n/layer={n_val}  H(d1)={H1:.4f}  H(d2)={H2:.4f}  "
              f"arm1 {bits_arm1:.0f} vs {bits_base:.0f} bits  saved {save1/1e6:.2f} MB  "
              f"recon {recon_diffs}")

    total1 = sum(results["families"][f]["saved_MB_arm1"] for f in results["families"])
    total2 = sum(results["families"][f]["saved_MB_arm2"] for f in results["families"])
    results["total_saved_MB_arm1"] = round(total1, 2)
    results["total_saved_MB_arm2"] = round(total2, 2)
    results["verdict"] = "TBD"
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT}   total arm1={total1:.2f} MB arm2={total2:.2f} MB")


if __name__ == "__main__":
    main()
