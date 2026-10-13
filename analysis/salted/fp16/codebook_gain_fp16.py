"""Task 3 on the ORIGINAL (unquantized) weights: salted vs global Lloyd-Max
codebook gain at equal bits/weight (k = 4/8/16 for 2/3/4 bits).

Machinery REUSED UNCHANGED from analysis/salted/codebook_gain.py:
lloyd_max, run_analysis, class_flat, quantize_generic, quantize_generic_scale,
unit_tests, KS / BITS_LIST / SUB_ROW_STRIDE(_5120), META. Only the chunk
source differs: fp16_common.tensor_chunks (plain BF16/FP16 originals) instead
of the W4A16 dequant path (codebook_gain.tensor_chunks).

Gates, all must pass before the real run:
  1. synthetic unit tests (reused verbatim: row/col/row+col-structured + iid,
     real projection shapes)
  2. Lloyd-Max theory check: a global 4-level book fit on iid Gaussian must
     reach the theoretical optimum MSE = 0.1175 * var (probes reference)
  3. --regression: the W4A16 baseline cells re-run through THIS script's own
     wiring (chunks_fn = codebook_gain.tensor_chunks, salts from the baseline
     heterogeneity.json, same fit strides) must reproduce EVERY scalar cell of
     analysis/salted/codebook_gain.json — the final reviewed baseline the
     findings doc cites — at 1e-12 relative, zero exceptions. (The plan named
     codebook_gain.pre_fix.json; the current codebook_gain.json is the final
     post-review run, so it is the stricter and correct target.)

The real run reads its salts from analysis/salted/fp16/heterogeneity.json
(Task 2 on the originals) and writes analysis/salted/fp16/codebook_gain.json
with a side-by-side baseline comparison per cell.

Run:  venv/bin/python analysis/salted/fp16/codebook_gain_fp16.py --regression
      venv/bin/python analysis/salted/fp16/codebook_gain_fp16.py
"""

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

SALTED = Path(__file__).resolve().parent.parent          # analysis/salted
sys.path.insert(0, str(SALTED))
sys.path.insert(0, str(SALTED / "fp16"))
import codebook_gain as cg      # noqa: E402  (lloyd_max, run_analysis, unit_tests, SAMPLES, ...)
import fp16_common as fc         # noqa: E402


def theory_check():
    """Global 4-level Lloyd-Max on iid Gaussian must hit 0.1175 * var."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2_000_000)
    c = cg.lloyd_max(x, 4)
    mid = 0.5 * (c[:-1] + c[1:])
    q = c[np.searchsorted(mid, x)]
    mse = float(np.square(x - q).mean())
    ratio = mse / float(x.var())
    ok = abs(ratio - 0.1175) < 0.001
    print(f"  Lloyd-Max theory: 4-level iid Gaussian MSE/var = {ratio:.5f} "
          f"(theory 0.1175) -> {'PASS' if ok else 'FAIL'}")
    return ok, ratio


def _worker(job):
    """One tensor, all arms/bits, wholly inside one worker (deterministic)."""
    kind, tkey, base, shard, bits, salt_str, salt_def, n, shape, stride = job
    # spawn workers re-import this module; cap torch's thread pools (numerics
    # unaffected: elementwise ops + exact per-chunk float64 reductions)
    torch.set_num_threads(1)
    t0 = time.time()
    if kind == "fp16":
        chunks_fn = lambda: fc.tensor_chunks(base)
    else:  # regression: the unchanged W4A16 dequant chunk path
        chunks_fn = lambda: cg.tensor_chunks(shard, base, bits)
    res = cg.run_analysis(chunks_fn, shape, salt_def, n,
                          [cg.KS[b] for b in cg.BITS_LIST], stride=stride)
    per_bits = {str(b): res[cg.KS[b]] for b in cg.BITS_LIST}
    rec = {
        "desc": base, "source": kind,
        "salt": salt_str, "N": n,
        "absmax_per_class": [float(v) for v in res["_scales_absmax"]],
        "fit_samples_per_class": res["_fit_counts_per_class"],
        "n_fit_samples": res["_n_fit_samples"],
        "fit_stride": int(stride),
        "bits": per_bits,
        "runtime_s": round(time.time() - t0, 1),
    }
    return tkey, rec


def build_jobs(kind):
    het = json.loads(((fc.FP16 if kind == "fp16" else SALTED)
                      / "heterogeneity.json").read_text())
    jobs = []
    for base, shard, bits in cg.SAMPLES:
        tkey = base.split("model.language_model.")[-1] if "layers" in base else base
        salt_str = het[tkey]["best"]["salt"]
        salt_def, n = salt_str.split("%")
        n = int(n)
        if kind == "fp16":
            shape = fc.tensor_shape(base)
        else:
            shape = cg.tensor_shape(shard, base)
        stride = (cg.SUB_ROW_STRIDE_5120
                  if (salt_def == "row" and shape[0] == 5120) else cg.SUB_ROW_STRIDE)
        jobs.append((kind, tkey, base, shard, bits, salt_str, salt_def, n, shape, stride))
    return jobs


def run_sweep(jobs, label):
    workers = min(9, os.cpu_count() - 2)
    # spawn (not fork): forking a multi-threaded torch/jemalloc parent deadlocks
    # the workers on a futex (probes reference); spawn re-imports this module
    mp_spawn = multiprocessing.get_context("spawn")
    print(f"\n== {label}: {len(jobs)} tensors, {workers} workers "
          f"(cpu_count={os.cpu_count()}, spawn) ==")
    t0 = time.time()
    done = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_spawn) as ex:
        futs = {ex.submit(_worker, job): job[1] for job in jobs}
        for fut in as_completed(futs):
            tkey, rec = fut.result()
            done[tkey] = rec
            print(f"  [done] {tkey:38s} worker {rec['runtime_s']:6.1f}s "
                  f"(wall {time.time() - t0:6.1f}s)", flush=True)
    return done, round(time.time() - t0, 1), workers


def regression_gate(done):
    """Every scalar cell must reproduce the final baseline JSON at 1e-12."""
    base_json = json.loads((SALTED / "codebook_gain.json").read_text())
    keys = ("mse_global", "mse_salted", "ratio_salted",
            "mse_salted_scale", "ratio_salted_scale", "max_abs_err_global")
    print("\n== Regression gate vs codebook_gain.json (final baseline, "
          "tol 1e-12 relative, zero exceptions) ==")
    mismatches = []
    for tkey, rec in done.items():
        for b in cg.BITS_LIST:
            o = base_json[tkey]["bits"][str(b)]
            nw = rec["bits"][str(b)]
            bad = []
            for key in keys:
                rel = abs(nw[key] - o[key]) / max(abs(o[key]), 1e-300)
                if rel > 1e-12:
                    mismatches.append((tkey, b, key, o[key], nw[key], rel))
                    bad.append(key)
            oa = np.array(base_json[tkey]["absmax_per_class"], dtype=np.float64)
            na = np.array(rec["absmax_per_class"], dtype=np.float64)
            if not np.allclose(na, oa, rtol=1e-12, atol=0):
                mismatches.append((tkey, b, "absmax_per_class", None, None, None))
                bad.append("absmax_per_class")
            print(f"  {tkey:38s} {b}b  r_s {o['ratio_salted']:.6f} == "
                  f"{nw['ratio_salted']:.6f}  "
                  f"{'MISMATCH: ' + ','.join(bad) if bad else 'ok'}")
    if mismatches:
        print("REGRESSION GATE FAILED:")
        for m in mismatches:
            print(f"  {m[0]} {m[1]}b {m[2]}: old={m[3]!r} new={m[4]!r} rel={m[5]}")
        return False
    print("  all cells reproduce the final baseline (rel diff <= 1e-12)")
    return True


def main():
    regression_only = "--regression" in sys.argv

    print("== Unit tests (reused from codebook_gain.py) ==")
    ok, unit = cg.unit_tests()
    if not ok:
        print("UNIT TESTS FAILED - aborting")
        sys.exit(1)
    print("\n== Lloyd-Max theory check ==")
    tok, theory_ratio = theory_check()
    if not tok:
        print("THEORY CHECK FAILED - aborting")
        sys.exit(1)

    if regression_only:
        jobs = build_jobs("baseline")
        done, wall, workers = run_sweep(
            jobs, "REGRESSION (W4A16 baseline through the fp16-script wiring)")
        if not regression_gate(done):
            print("REGRESSION GATE FAILED - aborting")
            sys.exit(1)
        print(f"REGRESSION GATE PASSED (wall {wall}s)")
        return

    jobs = build_jobs("fp16")
    done, wall, workers = run_sweep(jobs, "Real data: ORIGINAL weights, salted vs global")

    baseline = json.loads((SALTED / "codebook_gain.json").read_text())
    results = {
        "_unit_tests": unit,
        "_theory_check": {"mse_over_var": theory_ratio, "theory": 0.1175},
        "_meta": {**cg.META,
                  "computed_by": "analysis/salted/fp16/codebook_gain_fp16.py (this file)",
                  "source_checkpoint": ("models-fp16/Qwen3.8-27B "
                                        "(original unquantized weights)")},
        "_parallel": {"workers": workers, "wall_clock_s": wall},
    }

    print("\n== Table: originals vs dequant baseline "
          "(r_s = MSE_salted/MSE_global, r_ss = MSE_salted+scale/MSE_global) ==")
    print(f"{'tensor':38s} {'bits':>4s} {'r_s orig':>9s} {'r_s base':>9s} "
          f"{'r_ss orig':>9s} {'r_ss base':>9s}  verdict")
    for job in jobs:
        tkey = job[1]
        rec = done[tkey]
        cmp = {}
        for b in cg.BITS_LIST:
            r = rec["bits"][str(b)]
            ob = baseline[tkey]["bits"][str(b)]
            cmp[str(b)] = {
                "r_s_orig": r["ratio_salted"], "r_s_baseline": ob["ratio_salted"],
                "r_ss_orig": r["ratio_salted_scale"],
                "r_ss_baseline": ob["ratio_salted_scale"],
            }
            cm = r["class_max_err_salted"]
            print(f"{tkey:38s} {b:4d} {r['ratio_salted']:9.4f} {ob['ratio_salted']:9.4f} "
                  f"{r['ratio_salted_scale']:9.4f} {ob['ratio_salted_scale']:9.4f}  "
                  f"{'SALTED WINS' if r['ratio_salted'] < 0.9 else 'no gain'}")
            print(f"{'':38s}      MSE_g={r['mse_global']:.4e} MSE_s={r['mse_salted']:.4e} "
                  f"maxerr salted={max(cm):.4e} global={r['max_abs_err_global']:.4e}")
        rec["baseline_compare"] = cmp
        beats = [b for b in cg.BITS_LIST if rec["bits"][str(b)]["ratio_salted"] < 0.9]
        rec["verdict"] = (f"salted beats global (r_s<0.9) at bits {beats}" if beats
                          else "salted does NOT beat global at any bit level (all r_s >= 0.9)")
        results[tkey] = rec

    print("\n== Spec headline check: salted 2-bit MSE vs global 4-bit MSE "
          "(spec claims salted INT2 ~ matches INT4; ratio ~1.0, <1 = salted wins) ==")
    headline = {}
    for job in jobs:
        tkey = job[1]
        rec = done[tkey]
        o = rec["bits"]["2"]["mse_salted"] / rec["bits"]["4"]["mse_global"]
        bb = (baseline[tkey]["bits"]["2"]["mse_salted"]
              / baseline[tkey]["bits"]["4"]["mse_global"])
        headline[tkey] = {"salted2_vs_global4_orig": o, "salted2_vs_global4_baseline": bb}
        print(f"  {tkey:38s} orig={o:.4f}  baseline={bb:.4f}")
    results["_spec_headline"] = headline

    print("\n== Verdicts (originals) ==")
    for job in jobs:
        print(f"  {job[1]:38s} {done[job[1]]['verdict']}")
    print(f"\nParallel sweep wall clock: {wall}s for {len(jobs)} tensors ({workers} workers)")

    (fc.FP16 / "codebook_gain.json").write_text(json.dumps(results, indent=1))
    print("Wrote analysis/salted/fp16/codebook_gain.json")


if __name__ == "__main__":
    main()
