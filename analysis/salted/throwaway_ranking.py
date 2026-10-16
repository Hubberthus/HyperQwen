"""Throwaway-weight probe, Stage 1: candidate ranking on CPU.

For the 4 MLP tensors from sensitivity_probe.py (gate_proj & down_proj at
model.language_model.layers.0 and .63): rank group-128 groups by the EXISTING
sensitivity score S[i,g] = sc_ig^2 * sum_{j in g} (code_ij - offset)^2 * exp_j^2
(exposure_j = per-column weight L2 norm proxy, documented in sensitivity_probe.py)
and report, for the bottom K% of groups (K = 5, 10, 20, 30, 50), the fraction of
TOTAL sensitivity those bottom-K groups carry. Same curve when groups are ranked
by group scale magnitude instead (is magnitude a cheap equivalent?).

Verification vs analysis/salted/sensitivity_probe.json (direct comparison):
  - total_sensitivity must match to < 1e-9 relative error (S is recomputed with
    identical ops/chunking, so it is expected to be bitwise identical);
  - the descending concentration curve is re-derived from the same S, so the
    50/80/95% group-fractions in the probe must also reproduce.
Gate: pin check (manual unpack vs compressed_tensors.unpack_from_int32 == 0)
must pass before any numbers are reported/trusted.

Run:  venv/bin/python analysis/salted/throwaway_ranking.py   (<10 min CPU)
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent  # ~/qwen-serving
sys.path.insert(0, str(REPO / "venv" / "lib" / "python3.12" / "site-packages"))

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32  # noqa: E402

CKPT = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
PROBE_JSON = REPO / "analysis" / "salted" / "sensitivity_probe.json"
OUT = REPO / "analysis" / "salted" / "throwaway_ranking.json"
GROUP = 128
CHUNK = 4096
KS = (5, 10, 20, 30, 50)

TARGETS = [
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.63.mlp.gate_proj",
    "model.language_model.layers.63.mlp.down_proj",
]


def find_shard(base):
    for shard in sorted(CKPT.glob("*.safetensors")):
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


def bottom_k_fraction(scores, ks):
    """Fraction of total carried by the bottom K% of groups (ascending rank)."""
    s = scores.ravel().astype(np.float64)
    total = s.sum()
    assert total > 0
    asc = np.sort(s)
    cum = np.cumsum(asc) / total
    out = {}
    for k in ks:
        n = max(1, int(round(k / 100.0 * s.size)))
        # exact: sum of the n smallest scores
        out[f"bottom_{k}pct"] = float(asc[:n].sum() / total)
    return out


def probe(base, results):
    t0 = time.time()
    f, pk, sk, shape, shard = find_shard(base)
    rows, cols = shape
    assert cols % GROUP == 0
    groups = cols // GROUP
    num_bits = 4
    offset = 1 << (num_bits - 1)
    res = {"shape": list(shape), "num_bits": num_bits, "shard": shard,
           "n_groups": rows * groups, "group_size": GROUP}
    print(f"\n=== {base} {shape} groups={rows * groups:,} shard={shard}")

    # ---- pass 1: pin check + per-column exposure proxy (identical to probe) ----
    pin_max, pin_codes = 0, 0
    exp_acc = np.zeros((groups, GROUP), dtype=np.float64)
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        packed_np = np.asarray(f.get_slice(pk)[start:stop]).astype(np.int32)
        sc = np.asarray(f.get_slice(sk)[start:stop]).astype(np.float32)

        ref = unpack_from_int32(torch.from_numpy(packed_np), num_bits, (stop - start, cols))
        man = manual_unpack(packed_np, num_bits).astype(np.int16) - offset
        d = int(np.abs(man.astype(np.int32) - ref.numpy().astype(np.int32)).max())
        pin_max = max(pin_max, d)
        pin_codes += man.size

        codes3 = manual_unpack(packed_np, num_bits).reshape(-1, groups, GROUP)
        codes_sq = (codes3.astype(np.int32) - offset) ** 2
        sc_sq = (sc ** 2).astype(np.float32)
        exp_acc += (codes_sq.astype(np.float32) * sc_sq[:, :, None]).sum(axis=0, dtype=np.float64)

    assert pin_max == 0, f"unpack pin failed for {base}: max|manual-ct|={pin_max}"
    res["pin_check"] = {"max_abs_diff": 0, "codes_checked": pin_codes}
    print(f"  pin: {pin_codes:,} codes, max|diff| = 0  (gate PASSED)")

    # ---- pass 2: group sensitivity S (identical ops/chunking to probe) ----
    exp_sq = exp_acc.astype(np.float32)
    S = np.zeros((rows, groups), dtype=np.float32)
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        packed_np = np.asarray(f.get_slice(pk)[start:stop]).astype(np.int32)
        sc = np.asarray(f.get_slice(sk)[start:stop]).astype(np.float32)
        codes3 = manual_unpack(packed_np, num_bits).reshape(-1, groups, GROUP)
        codes_sq = (codes3.astype(np.int32) - offset) ** 2
        sc_sq = (sc ** 2).astype(np.float32)
        S[start:stop] = (codes_sq.astype(np.float32) * exp_sq[None]).sum(axis=2, dtype=np.float32) * sc_sq

    total = float(S.astype(np.float64).sum())
    res["total_sensitivity"] = total

    # ---- verification vs sensitivity_probe.json ----
    ref = json.loads(PROBE_JSON.read_text())["tensors"][base]
    rel_total = abs(total - ref["total_sensitivity"]) / abs(ref["total_sensitivity"])
    res["verify_vs_probe"] = {
        "total_sensitivity_rel_err": rel_total,
        "recomputed_concentration": concentration_desc(S),
        "probe_concentration": ref["concentration"],
    }
    conc_match = all(
        abs(res["verify_vs_probe"]["recomputed_concentration"][k] - v) < 1e-9
        for k, v in ref["concentration"].items())
    res["verify_vs_probe"]["concentration_match"] = conc_match
    assert rel_total < 1e-9, f"total_sensitivity mismatch vs probe: rel={rel_total}"
    assert conc_match, "concentration curve mismatch vs probe"

    # ---- bottom-K curves ----
    res["bottom_k_by_sensitivity"] = bottom_k_fraction(S, KS)
    # magnitude ranking: group scale magnitude |sc[i,g]| (one scalar per group).
    # NOTE: AutoRound scales are SIGNED (min<0), so take abs() before ranking —
    # raw scales sum to a negative total, which tripped `assert total > 0`.
    scales = np.asarray(f.get_tensor(sk)).astype(np.float64)  # [rows, groups]
    assert scales.shape == (rows, groups)
    print(f"  scales: shape={scales.shape} dtype={scales.dtype} "
          f"min={scales.min():.6g} max={scales.max():.6g} "
          f"sum={scales.sum():.6g} abs_sum={np.abs(scales).sum():.6g}")
    res["bottom_k_by_scale"] = bottom_k_fraction(np.abs(scales), KS)
    rho = float(np.corrcoef(np.log(np.abs(scales.ravel()) + 1e-30),
                            np.log(S.ravel().astype(np.float64) + 1e-30))[0, 1])
    res["pearson_log_sens_vs_log_scale"] = rho

    res["runtime_s"] = round(time.time() - t0, 1)
    print(f"  bottom-K (sens): {fmt(res['bottom_k_by_sensitivity'])}")
    print(f"  bottom-K (scale): {fmt(res['bottom_k_by_scale'])}")
    print(f"  pearson log-log: {rho:.4f}  runtime: {res['runtime_s']}s")
    results[base] = res
    return res


def concentration_desc(S):
    """Descending-order curve (same as probe.concentration, for verification)."""
    s = S.ravel().astype(np.float64)
    cum = np.cumsum(np.sort(s)[::-1]) / s.sum()
    out = {}
    for t in (0.5, 0.8, 0.95):
        k = int(np.searchsorted(cum, t, side="left")) + 1
        out[f"frac_groups_{int(t * 100)}pct"] = k / s.size
    return out


def fmt(d):
    return {k: f"{v:.4%}" for k, v in d.items()}


def main():
    t0 = time.time()
    results = {
        "probe": "throwaway-weight stage 1: bottom-K% sensitivity concentration ranking (CPU)",
        "date": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CKPT.relative_to(REPO)),
        "definition": ("sens_ij = w_ij^2 * exposure_j^2 (exposure_j = per-column weight "
                       "L2-norm proxy, per sensitivity_probe.py); group score S[i,g] = "
                       "sum_{j in g} sens_ij; groups along input dim. bottom-K% = the K% "
                       "of groups with the SMALLEST scores."),
        "gate": "pin check manual_unpack vs unpack_from_int32 == 0 must pass first",
        "k_values": list(KS),
        "tensors": {},
    }
    for base in TARGETS:
        probe(base, results["tensors"])
    results["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT} (total {results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
