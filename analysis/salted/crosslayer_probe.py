"""Cross-layer structure probe: do layer-N scales/codes predict layer N+1?

Last untested axis of the revival round: all prior sequence probes were
within-tensor. Arms:
  A. H(scale | same-position scale in previous layer) vs marginal, 256
     equal-frequency bins pooled per family — ALL 63 adjacent layer pairs,
     gate_proj + down_proj (scales only: ~0.7M/2.4M values per layer).
  B. H(int4 code | same-position code previous layer) vs marginal ~3.57,
     sampled adjacent pairs {0-1, 31-32, 62-63} per family.
  C. pooled-across-64-layers marginal scale entropy vs mean per-layer marginal
     (pooled lower => a shared cross-layer codebook saves bits).

Readout: gap > 0.1 bits/element = worth a delta-coding probe.
Gate: pin check manual unpack vs compressed_tensors.unpack_from_int32 == 0.

Run: venv/bin/python analysis/salted/crosslayer_probe.py   (budget 20 min CPU)
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
OUT = REPO / "analysis" / "salted" / "crosslayer_probe.json"
NL = 64
CODE_PAIRS = [(0, 1), (31, 32), (62, 63)]
BINS = 256


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


def cond_entropy(arr, pred, n_a, n_p):
    """H(arr|pred) from exact joint bincounts."""
    joint = np.bincount((arr * n_p + pred).ravel(), minlength=n_a * n_p).astype(np.float64)
    joint = joint.reshape(n_a, n_p)
    n = joint.sum()
    nz = joint[joint > 0]
    H_joint = -(nz / n * np.log2(nz / n)).sum()
    pm = joint.sum(axis=1)
    nzp = pm[pm > 0]
    H_pred = -(nzp / n * np.log2(nzp / n)).sum()
    return float(H_joint - H_pred), H_pred  # conditional, marginal-of-pred


def marginal_from_bins(b):
    n = b.size
    counts = np.bincount(b.ravel(), minlength=BINS).astype(np.float64)
    p = counts / n
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def bin_scales(s_all_flat, qs):
    return np.searchsorted(qs, s_all_flat, side="right") - 1


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
        # --- A + C: scales across all layers ---
        per_layer = {}
        layer_scales = {}
        for L in range(NL):
            s = get_scales(fam, L)
            layer_scales[L] = s.ravel()
            per_layer[L] = marginal_from_bins(np.searchsorted(
                np.quantile(s, np.linspace(0, 1, BINS + 1)), s.ravel(), side="right") - 1)
        pooled_flat = np.concatenate([layer_scales[L] for L in range(NL)])
        qs = np.quantile(pooled_flat, np.linspace(0, 1, BINS + 1))
        H_pool = marginal_from_bins(bin_scales(pooled_flat, qs))

        cond = []
        for L in range(1, NL):
            a = bin_scales(layer_scales[L], qs)
            p = bin_scales(layer_scales[L - 1], qs)
            n_a = int(a.max()) + 1
            n_p = int(p.max()) + 1
            Hc, _ = cond_entropy(a, p, n_a, n_p)
            cond.append(Hc)
        mean_cond = float(np.mean(cond))
        mean_marg = float(np.mean(list(per_layer.values())))
        results["families"][fam] = {
            "A_mean_cond_bits": mean_cond,
            "A_marginal_bits": H_pool,
            "A_gap_bits": H_pool - mean_cond,
            "per_pair_cond": {str(L): round(c, 4) for L, c in zip(range(1, NL), cond)},
            "C_pooled_bits": H_pool,
            "C_mean_perlayer_bits": mean_marg,
            "C_gain_bits": mean_marg - H_pool,
        }
        print(f"[{fam}] A: H(scale|prev)={mean_cond:.4f} vs pooled marginal {H_pool:.4f} "
              f"gap={H_pool - mean_cond:+.4f}  C: pooled {H_pool:.4f} vs mean-per-layer {mean_marg:.4f}")

        # --- B: codes on sampled pairs ---
        if fam == "gate_proj":  # codes once per family-pair; reuse for both? codes differ per family — do both
            pass
        code_res = {}
        for L0, L1 in CODE_PAIRS:
            up = {}
            for L in (L0, L1):
                k_p = f"model.language_model.layers.{L}.mlp.{fam}.weight_packed"
                k_s = f"model.language_model.layers.{L}.mlp.{fam}.weight_shape"
                f = get_file(k_p)
                packed = f.get_tensor(k_p)
                shape = tuple(f.get_tensor(k_s).tolist())
                rows, cols = shape
                p64 = packed.to(torch.int64).numpy()
                u = np.empty((rows, cols), dtype=np.int64)
                for i in range(8):
                    u[:, i::8] = (p64 >> (4 * i)) & 15
                up[L] = u
            a, p = up[L1], up[L0]
            H_marg = marginal_from_bins(a)  # 16-symbol marginal via same fn (BINS>=16 ok)
            Hc, _ = cond_entropy(a, p, 16, 16)
            code_res[f"{L0}-{L1}"] = {"marginal": round(H_marg, 4), "cond": round(Hc, 4),
                                      "gap": round(H_marg - Hc, 4)}
            print(f"[{fam}] B {L0}-{L1}: marginal {H_marg:.4f} cond {Hc:.4f} gap {H_marg - Hc:+.4f}")
        results["families"][fam]["B_code_pairs"] = code_res

    results["verdict"] = (
        "TBD"
    )
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
