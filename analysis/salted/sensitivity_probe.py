"""Probe: per-group-128 sensitivity map under the GPTQ/AWQ-style activation-exposure proxy.

Question: how concentrated is quantization-relevant sensitivity across the
group-128 lattice of this checkpoint, and does group SCALE magnitude track
sensitivity (i.e. could the stored scales alone serve as a sensitivity map)?

Definition (proxy, labeled):
  sens_ij ~= w_ij^2 * exposure_j^2
  exposure_j = ||W[:, j]||_2  (per-column weight L2 norm) — WEAKER PROXY for
  the expected activation magnitude of input channel j. Checked first: NO
  cached calibration hidden states exist under analysis/ or bench/ (the only
  cache found was bench/results/kv_spectrum_grams.pt = KV-cache Gram matrices,
  not activations), so the proxy is used and clearly labeled.
  Group score: S[i, g] = sum_{j in group g} sens_ij (groups along input dim).

Tensors: gate_proj & down_proj at layers 0 and 63 (full), plus lm_head.
lm_head (248320x5120 = 1.27B elements) is STREAMED in row chunks — no row
subsample needed; full group statistics are exact.

SCHEME FINDING (documented): lm_head is packed as 8-bit in this checkpoint,
although quantization_config.json claims num_bits=4 and lists lm_head in
"ignore". Evidence: unpack_from_int32 at num_bits=8 reproduces the original
bf16 lm_head (shard7.bak) to max err 3.6e-4 = one fp16-scale rounding step;
num_bits=4 does not (err ~0.09). Layer MLP tensors are standard W4A16.

Pin gates (both asserted):
  1. manual unpack vs compressed_tensors.unpack_from_int32 -> max|diff| == 0
  2. factored group score vs direct elementwise recomputation (rel err gate)

Run:  venv/bin/python analysis/salted/sensitivity_probe.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent.parent  # ~/qwen-serving
sys.path.insert(0, str(REPO / "venv" / "lib" / "python3.12" / "site-packages"))

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32  # noqa: E402

CKPT = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
OUT = REPO / "analysis" / "salted" / "sensitivity_probe.json"
GROUP = 128
CHUNK = 4096

# (base name, num_bits). lm_head is 8-bit in this checkpoint — see docstring.
TARGETS = [
    ("model.language_model.layers.0.mlp.gate_proj", 4),
    ("model.language_model.layers.0.mlp.down_proj", 4),
    ("model.language_model.layers.63.mlp.gate_proj", 4),
    ("model.language_model.layers.63.mlp.down_proj", 4),
    ("lm_head", 8),
]


def find_shard(base):
    shards = sorted(CKPT.glob("*.safetensors"))
    for shard in shards:
        f = safe_open(CKPT / shard, framework="pt", device="cpu")
        if f"{base}.weight_packed" in list(f.keys()):
            shape = tuple(f.get_tensor(f"{base}.weight_shape").tolist())
            return f, f"{base}.weight_packed", f"{base}.weight_scale", shape, shard.name
    raise FileNotFoundError(f"{base}.weight_packed not found")


def manual_unpack(packed_np, num_bits):
    """int32 [rows, n32] -> unsigned codes uint8 [rows, cols] (0 .. 2^b-1)."""
    factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    rows, n32 = packed_np.shape
    out = np.empty((rows, n32 * factor), dtype=np.uint8)
    for i in range(factor):
        out[:, i::factor] = ((packed_np >> (num_bits * i)) & mask).astype(np.uint8)
    return out


def concentration(S):
    """Smallest fraction of groups (desc order) carrying 50/80/95% of total."""
    s = S.ravel().astype(np.float64)
    total = s.sum()
    assert total > 0
    cum = np.cumsum(np.sort(s)[::-1]) / total
    out = {}
    for t in (0.5, 0.8, 0.95):
        k = int(np.searchsorted(cum, t, side="left")) + 1
        out[f"frac_groups_{int(t * 100)}pct"] = k / s.size
    return out


def probe(base, num_bits, results):
    t0 = time.time()
    f, pk, sk, shape, shard = find_shard(base)
    rows, cols = shape
    assert cols % GROUP == 0
    groups = cols // GROUP
    offset = 1 << (num_bits - 1)
    res = {"shape": list(shape), "num_bits": num_bits, "shard": shard,
           "n_groups": rows * groups, "group_size": GROUP}
    print(f"\n=== {base} {shape} bits={num_bits} groups={rows * groups:,} shard={shard}")

    # ---- pass 1: pin check + per-column exposure proxy (sum_i w_ij^2) ----
    pin_max, pin_codes = 0, 0
    exp_acc = np.zeros((groups, GROUP), dtype=np.float64)
    dequant_gate_max = 0.0  # factored vs direct-dequant exposure on first chunk
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        packed_np = np.asarray(f.get_slice(pk)[start:stop]).astype(np.int32)
        sc = np.asarray(f.get_slice(sk)[start:stop]).astype(np.float32)  # [n, groups]

        # pin: manual vs compressed_tensors
        ref = unpack_from_int32(torch.from_numpy(packed_np), num_bits, (stop - start, cols))
        man = manual_unpack(packed_np, num_bits).astype(np.int16) - offset
        d = int(np.abs(man.astype(np.int32) - ref.numpy().astype(np.int32)).max())
        pin_max = max(pin_max, d)
        pin_codes += man.size

        codes3 = manual_unpack(packed_np, num_bits).reshape(-1, groups, GROUP)
        codes_sq = (codes3.astype(np.int32) - offset) ** 2   # signed codes squared
        sc_sq = (sc ** 2).astype(np.float32)               # [n, groups]
        # factored: sum_i w_ij^2 = sum_i code_ij^2 * scale_ig^2
        factored = (codes_sq.astype(np.float32) * sc_sq[:, :, None]).sum(axis=0, dtype=np.float64)
        exp_acc += factored
        # direct-dequant gate on first chunk: w = code * scale, sum w^2 per column
        if start == 0:
            signed = codes3.reshape(-1, cols).astype(np.float32) - offset
            W = signed * np.repeat(sc, GROUP, axis=1)  # scale tiled across group
            direct = (W.astype(np.float64) ** 2).sum(axis=0).reshape(groups, GROUP)
            dequant_gate_max = float(np.abs(factored - direct).max() / max(direct.max(), 1e-30))

    assert pin_max == 0, f"unpack pin failed for {base}: max|manual-ct|={pin_max}"
    res["pin_check"] = {"max_abs_diff": 0, "codes_checked": pin_codes}
    res["exposure_dequant_gate"] = {"max_rel_err": dequant_gate_max}
    assert dequant_gate_max < 1e-5, "exposure factoring mismatch vs direct dequant"

    # ---- pass 2: per-group sensitivity S[i,g] = scale_ig^2 * sum_{j in g} code_ij^2 * exp_j^2 ----
    exp_sq = exp_acc.astype(np.float32)                    # [groups, 128]
    S = np.zeros((rows, groups), dtype=np.float32)
    direct_gate_max = 0.0
    rng = np.random.default_rng(0)
    check_pairs = set(zip(rng.integers(0, min(rows, 256), size=8),
                          rng.integers(0, groups, size=8)))
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        packed_np = np.asarray(f.get_slice(pk)[start:stop]).astype(np.int32)
        sc = np.asarray(f.get_slice(sk)[start:stop]).astype(np.float32)
        codes3 = manual_unpack(packed_np, num_bits).reshape(-1, groups, GROUP)
        codes_sq = (codes3.astype(np.int32) - offset) ** 2   # signed codes squared
        sc_sq = (sc ** 2).astype(np.float32)
        chunk_S = (codes_sq.astype(np.float32) * exp_sq[None]).sum(axis=2, dtype=np.float32) * sc_sq
        S[start:stop] = chunk_S
        # direct elementwise gate on a few random groups
        for i, g in list(check_pairs):
            if start <= i < stop:
                w = (codes3[i - start, g].astype(np.float32) - offset) * sc[i - start, g]
                direct = float((w.astype(np.float64) ** 2 * exp_sq[g].astype(np.float64)).sum())
                factored = float(chunk_S[i - start, g])
                direct_gate_max = max(direct_gate_max,
                                      abs(factored - direct) / max(direct, 1e-30))
    res["direct_score_gate"] = {"max_rel_err": direct_gate_max}
    assert direct_gate_max < 1e-4, "factored vs direct group score mismatch"

    # ---- stats ----
    res["concentration"] = concentration(S)
    rho = spearmanr(S.ravel().astype(np.float64),
                    np.asarray(f.get_tensor(sk)).astype(np.float64).ravel())
    res["spearman_sens_vs_scale"] = {"rho": float(rho.statistic), "pvalue": float(rho.pvalue),
                                     "n": int(S.size)}
    res["total_sensitivity"] = float(S.astype(np.float64).sum())
    res["runtime_s"] = round(time.time() - t0, 1)
    print(f"  pin: {pin_codes:,} codes, max|diff| = 0")
    print(f"  concentration: {json.dumps({k: round(v, 5) for k, v in res['concentration'].items()})}")
    print(f"  spearman sens vs scale: rho = {res['spearman_sens_vs_scale']['rho']:.4f}")
    print(f"  runtime: {res['runtime_s']}s")
    results[base] = res
    return res


def main():
    t0 = time.time()
    results = {
        "probe": "sensitivity_map (GPTQ/AWQ-style activation-exposure proxy, group-128)",
        "date": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CKPT.relative_to(REPO)),
        "definition": ("sens_ij = w_ij^2 * exposure_j^2; group score S[i,g] = "
                       "sum_{j in g} sens_ij; groups along input dim"),
        "activation_source": (
            "PROXY (weaker): no cached calibration hidden states found under "
            "analysis/ or bench/ (only bench/results/kv_spectrum_grams.pt = KV-cache "
            "Gram matrices, not activations). exposure_j = per-column weight L2 norm "
            "||W[:,j]||_2 stands in for expected activation magnitude."),
        "scheme_note": (
            "Layer MLP tensors: W4A16 int4 symmetric group-128. lm_head: packed as "
            "8-bit (unpack_from_int32 num_bits=8 reproduces original bf16 in "
            "model-00007-of-00007.safetensors.bak to 3.6e-4 = fp16-scale rounding; "
            "num_bits=4 fails), despite quantization_config.json claiming 4-bit and "
            "listing lm_head in ignore. Documented in this probe."),
        "tensors": {},
    }
    for base, bits in TARGETS:
        probe(base, bits, results["tensors"])
    results["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT} (total {results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
