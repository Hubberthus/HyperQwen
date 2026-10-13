"""Task 2 on the ORIGINAL (unquantized) weights: position-heterogeneity eta^2.

Same sweep as analysis/salted/heterogeneity.py — 9 tensors x {linear, row, col,
row+col} % N, N in {4,8,16,32,64} — but reads the plain BF16/FP16 originals
from models-fp16/Qwen3.8-27B. No unpack/dequant step exists on this source.
The StreamingEta accumulator, SALT_DEFS/NS, and the synthetic gates are
REUSED UNCHANGED from heterogeneity.py (imported, not copied).

The dequant pin has no analogue here; it is replaced by a load-sanity check
(null-is-guilty protocol): the loaded originals must show weight-like
statistics (dtype, std, percentiles) before any eta^2 number is trusted.

Run:  venv/bin/python analysis/salted/fp16/heterogeneity_fp16.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SALTED = Path(__file__).resolve().parent.parent          # analysis/salted
sys.path.insert(0, str(SALTED))
sys.path.insert(0, str(SALTED / "fp16"))
import heterogeneity as hz      # noqa: E402  (StreamingEta, SALT_DEFS, NS, synthetic_test)
import fp16_common as fc         # noqa: E402

SAMPLES = [  # identical tensors to the baseline sweep (layers 0/27/63 + lm_head)
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.27.self_attn.q_proj",
    "model.language_model.layers.27.mlp.gate_proj",
    "model.language_model.layers.27.mlp.down_proj",
    "model.language_model.layers.63.self_attn.q_proj",
    "model.language_model.layers.63.mlp.gate_proj",
    "model.language_model.layers.63.mlp.down_proj",
    "lm_head",
]


def run_all_salts(base):
    """Stream the original tensor once into all 20 (salt, N) accumulators."""
    accs = None
    for w, r0 in fc.tensor_chunks(base):
        if accs is None:
            accs = {(sd, n): hz.StreamingEta(sd, n, w.shape[1])
                    for sd in hz.SALT_DEFS for n in hz.NS}
        for acc in accs.values():
            acc.add_chunk(w, r0)
    return {f"{sd}%{n}": accs[(sd, n)].result() for sd in hz.SALT_DEFS for n in hz.NS}


def load_sanity(base):
    """Null-is-guilty check: weights must look like weights before trusting eta^2."""
    shape, dtype = fc.tensor_meta(base)
    v = fc.load_tensor(base).ravel()
    stats = {
        "dtype_on_disk": dtype,
        "shape": list(shape),
        "std": float(v.std()),
        "p1": float(np.percentile(v, 1)),
        "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
        "pct_positive": float((v > 0).mean() * 100),
    }
    print(f"  load sanity {base}: dtype={dtype} std={stats['std']:.5f} "
          f"p1={stats['p1']:.4f} p50={stats['p50']:.4f} p99={stats['p99']:.4f} "
          f"%pos={stats['pct_positive']:.1f}")
    return stats


def noise_floor():
    """iid Gaussian eta^2 floor at the REAL projection shapes (the same 3
    shapes the baseline measured 5.5-9.2e-7 on). At/below-floor values are
    'at noise floor', never signal."""
    rng = np.random.default_rng(0)
    floor = {}
    for rows, cols in [(15360, 5120), (17408, 5120), (5120, 17408)]:
        w = rng.standard_normal((rows, cols)).astype(np.float32)
        worst, worst_key = 0.0, ""
        for sd in hz.SALT_DEFS:
            for n in hz.NS:
                e = hz.StreamingEta(sd, n, cols)
                e.add_chunk(w, 0)
                v = e.result()["eta_squared"]
                if v > worst:
                    worst, worst_key = v, f"{sd}%{n}"
        floor[f"{rows}x{cols}"] = {"worst_eta2": worst, "at": worst_key}
        print(f"  iid floor {rows}x{cols}: worst eta2 = {worst:.3e} ({worst_key})")
    return floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    if not args.skip_synthetic and not hz.synthetic_test():
        print("SYNTHETIC TESTS FAILED - aborting real run")
        sys.exit(1)

    print("\n== Load sanity + iid noise floor ==")
    sanity = load_sanity("model.language_model.layers.0.mlp.gate_proj")
    floor = noise_floor()

    baseline = json.loads((SALTED / "heterogeneity.json").read_text())
    results = {"_sanity": sanity, "_iid_floor": floor}
    print("\n== eta^2 on ORIGINAL weights, side-by-side vs dequant baseline ==")
    print(f"{'tensor':50s} {'best(orig)':>12s} {'eta2':>10s}   "
          f"{'best(base)':>12s} {'eta2':>10s}")
    for base in SAMPLES:
        tkey = base.split("model.language_model.")[-1] if "layers" in base else base
        res = run_all_salts(base)
        best = max(res.items(), key=lambda kv: kv[1]["eta_squared"])
        results[tkey] = {
            "desc": base,
            "salt": {k: v["eta_squared"] for k, v in res.items()},
            "best": {"salt": best[0], "eta_squared": best[1]["eta_squared"],
                     "total_var": best[1]["total_var"],
                     "class_mean": best[1]["class_mean"],
                     "class_std": best[1]["class_std"]},
        }
        b0 = baseline[tkey]["best"]
        print(f"{tkey:50s} {best[0]:>12s} {best[1]['eta_squared']:10.6f}   "
              f"{b0['salt']:>12s} {b0['eta_squared']:10.6f}")
        for sd in hz.SALT_DEFS:
            print(f"    {sd:8s}: " + " ".join(
                f"N{n}={results[tkey]['salt'][f'{sd}%{n}']:.4f}" for n in hz.NS))
        sys.stdout.flush()

    (fc.FP16 / "heterogeneity.json").write_text(json.dumps(results, indent=1))
    print("\nWrote analysis/salted/fp16/heterogeneity.json")


if __name__ == "__main__":
    main()
