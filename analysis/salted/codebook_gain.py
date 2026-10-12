"""Task 3: Codebook gain — salted vs global quantization (salted-quantization spike).

Question: at equal bits/weight, do per-salt-class Lloyd-Max codebooks beat one
global Lloyd-Max codebook on the dequantized weights?

FAIR-COMPARISON / ENTRY-COUNT PARITY (explicit, per plan):
  GLOBAL arm        : ONE codebook of k entries; every weight uses it.
  SALTED arms       : N per-class codebooks of k entries each = N*k entries.
  Bits per weight   : log2(k) in EVERY arm. The class index is derived from the
                      weight's POSITION (salt def), which the spec grants as
                      free side information — it is NOT stored per weight.
  Codebook size k is identical across arms at each bit level (k = 4/8/16 for
  2/3/4 bits). The salted arm therefore holds MORE total entries (N*k vs k),
  but bits-per-weight — the quantity that determines kernel cost — is equal,
  so the comparison is fair. (In a real kernel the N*k entries are N small
  tables indexed by the free position-derived class id.)

Lloyd-Max procedure (identical for every arm and every class):
  1-D k-means (Lloyd) on weight values; deterministic equal-percentile
  (equal-mass) init: centroids = np.percentile(x, (i + 0.5) * 100 / k);
  update = per-cluster MEAN (the "Max" centroid = conditional mean); <= 20
  iterations, early stop when centroids move < 1e-9 relative; final centroids
  sorted. float32 data, float64 arithmetic. No RNG anywhere -> reproducible.

Fitting data (deterministic subsample, documented):
  rows with global row index r % 65 == 0. 65 % 64 == 1, so sampled rows hit
  EVERY row-class residue mod 64 exactly (row k of the sample has r % 64 == k
  for k = 0..63); ~1/65 of the tensor (~15k-305k samples per class).
  (Stride 97 was tried first but leaves 11 of 64 row classes empty on the
  5120-row down_proj — those classes then fall back to the global book,
  handicapping the salted arm; 65 avoids that.)
  The global book is fit on the union of exactly the same subsample.
  Exact MSE and per-class max error are then computed by streaming the FULL
  tensor (fit on subsample, evaluate on everything).

  Task 3 review fix (#3, fit adequacy): the 5120-row down_proj with a row salt
  (layers.63.mlp.down_proj) was FIT-NOISE-DOMINATED at the 1/65 fraction —
  probe evidence: r%65-fit ratio 1.1330, r%33-fit 1.0623, all-rows fit 0.9973
  (~/probe_down_all2.py; per-class fit data there was only ~1-2 rows x 17408
  cols). For 5120-row tensors with a row salt the fit fraction is bumped to
  rows r%17 == 0: 301 sampled rows, gcd(17, 64) == 1 so every row-class
  residue is still covered (4-5 rows/class, ~70k-87k samples/class), and the
  cell is re-derived. All other tensors keep r%65. No positive result is being
  chased — the null stands; the goal is that every reported cell has adequate
  fit data or is labeled noisy.

SALTED+SCALE (spec's 2-step, mechanism REPLACED per Task 3 review #1):
  Per-class scale = absmax of that class over the FULL tensor (exact streaming
  max); ONE shared k-entry codebook fit in NORMALIZED space: every salt class
  is divided by its own absmax and a single Lloyd-Max book of k entries is fit
  over the union of all normalized classes; quantization = class scale +
  shared book (recon = centroid * class_scale). Total entries k + N — the
  codebook itself now has FULL entry-count parity with the global arm's k
  entries (bits/weight = log2(k), unchanged).
  History / why the old variant was a no-op (review #1): the ORIGINAL variant
  fit a separate k-entry book PER CLASS on w/absmax. That is provably
  identical to raw per-class books under Lloyd-Max: the mean update is scale
  equivariant, lloyd_max(x/s)*s == lloyd_max(x) for any s > 0 (the
  equal-percentile init commutes with positive scaling and the mean update is
  homogeneous), so per-class books fit on w/scale were numerically the same as
  raw books and r_ss == r_s was FORCED BY CONSTRUCTION (measured agreement
  ~1e-9, float32 noise, on all 27 cells). The replacement tests what the
  spec's 2-step actually constrains: one shared book + per-class scale, which
  is DISTINCT from raw per-class books only if the classes differ in SHAPE
  (distribution), not merely in scale.
  Reviewer note: ~50.7% of weight_scale entries in this AutoRound checkpoint
  are NEGATIVE. The dequant path multiplies the signed scale so weight signs
  are already correct; the absmax scale (>= 0) is sign-agnostic.

MAX-ERROR CAVEAT FOR TASK 4 (measured on the pre-fix sweep, 9 tensors x 3 bits):
  The blanket "no max-error win" claim is FALSE: salted per-class MAX error is
  LOWER than global in 14/27 cells (e.g. lm_head 4-bit 0.258 vs 0.304, 1.18x
  lower) — outlier-heavy classes benefit from per-class books even where MSE
  shows no gain. Task 4 must not repeat the blanket claim; report the
  per-cell max errors (class_max_err_salted vs max_abs_err_global) as-is.

Dequantization is REUSED from Task 2's analysis/salted/heterogeneity.py
(load_packed / dequant_chunk / CHUNK_ROWS streaming). verify_pin() from that
module is re-run at start; the script aborts unless max_abs_diff == 0.0.

Run:  venv/bin/python analysis/salted/codebook_gain.py [--synthetic-only]
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent  # ~/qwen-serving
SALTED = REPO / "analysis" / "salted"
sys.path.insert(0, str(SALTED))
import heterogeneity as hz  # noqa: E402  (Task 2 module: dequant + pin)

KM_ITERS = 20
BITS_LIST = [2, 3, 4]           # -> codebook sizes k = 4, 8, 16
KS = {2: 4, 3: 8, 4: 16}
SUB_ROW_STRIDE = 65             # fit subsample: rows r % 65 == 0 (65 % 64 == 1 -> all residues)
SUB_ROW_STRIDE_5120 = 17        # fit bump for 5120-row tensors with a row salt (review fix #3):
                                # 301 sampled rows, gcd(17,64)==1 -> all 64 residues still covered
BLOCK = 2_000_000               # generic-path block (bounds (b,k) temporaries)

SAMPLES = [  # identical to Task 2's list (layers 0/27/63 + lm_head)
    ("model.language_model.layers.0.mlp.gate_proj", "model-00001-of-00007.safetensors", 4),
    ("model.language_model.layers.0.mlp.down_proj", "model-00001-of-00007.safetensors", 4),
    ("model.language_model.layers.27.self_attn.q_proj", "model-00002-of-00007.safetensors", 4),
    ("model.language_model.layers.27.mlp.gate_proj", "model-00002-of-00007.safetensors", 4),
    ("model.language_model.layers.27.mlp.down_proj", "model-00002-of-00007.safetensors", 4),
    ("model.language_model.layers.63.self_attn.q_proj", "model-00004-of-00007.safetensors", 4),
    ("model.language_model.layers.63.mlp.gate_proj", "model-00004-of-00007.safetensors", 4),
    ("model.language_model.layers.63.mlp.down_proj", "model-00004-of-00007.safetensors", 4),
    ("lm_head", "model-00007-of-00007.safetensors", 8),
]

META = {
    "computed_by": "analysis/salted/codebook_gain.py (this file)",
    "question": "salted per-class Lloyd-Max codebooks vs one global codebook at equal bits/weight",
    "parity_note": (
        "bits/weight = log2(k) in every arm (k=4/8/16 at 2/3/4 bits, identical across arms). "
        "The salt-class index is derived from weight position (free side information per the "
        "spec), not stored per weight, so the salted arm's N*k total entries do NOT cost extra "
        "bits per weight. Entry-count parity is intentionally NOT imposed on total entries; "
        "parity is imposed on codebook size k and on the Lloyd-Max procedure."
    ),
    "lloyd_max": (
        "1-D Lloyd k-means, per-cluster mean update, deterministic equal-percentile init "
        "(np.percentile at (i+0.5)/k quantiles), <=20 iters, 1e-9 relative early stop, "
        "centroids sorted; fit float64, quantize float32"
    ),
    "subsample": (
        "codebooks fit on rows r%65==0 (65%64==1 -> sampled rows cover every row-class residue "
        "mod 64 exactly; ~1/65 of tensor, 15k-305k samples/class), EXCEPT 5120-row tensors with "
        "a row salt use rows r%17==0 (review fix #3: down L63 was fit-noise-dominated at r%65 — "
        "probe: r%65 1.1330 / r%33 1.0623 / all-rows 0.9973; r%17 = 301 rows, 4-5 rows/class, "
        "gcd(17,64)==1 so all residues still covered); global book fit on the union of the same "
        "subsample; MSE and per-class max error computed exactly over the FULL tensor. Per-class "
        "absmax tables and per-class fit-sample counts are persisted per tensor in this JSON."
    ),
    "salted_scale": (
        "spec 2-step (mechanism REPLACED per review #1): per-class scale = absmax over the FULL "
        "tensor (exact streaming max); ONE shared k-entry codebook fit on the union of all "
        "classes normalized by their own absmax; recon = shared_centroid * class_scale; "
        "k + N total entries (full entry-count parity with the global arm's k-entry book)"
    ),
    "salted_scale_history": (
        "the ORIGINAL variant (per-class books fit on w/absmax) was a mathematical no-op: "
        "Lloyd-Max with the mean update is scale-equivariant, lloyd_max(x/s)*s == lloyd_max(x), "
        "so per-class normalized books were provably identical to raw per-class books and "
        "r_ss == r_s was forced by construction (measured ~1e-9 agreement on all 27 cells). "
        "The replacement is the distinct 2-step mechanism the spec intended."
    ),
    "max_error_note": (
        "measured on the pre-fix sweep: salted per-class MAX error is LOWER than global in "
        "14/27 cells (lm_head 4-bit 0.258 vs 0.304, 1.18x lower) — Task 4 must NOT repeat the "
        "blanket no-max-error-win claim; see class_max_err_salted vs max_abs_err_global per cell"
    ),
    "negative_scales": (
        "~50.7% of weight_scale entries are negative (AutoRound); dequant path multiplies the "
        "signed scale (reused from heterogeneity.py), absmax class scale is sign-agnostic"
    ),
}


# ---------------------------------------------------------------- Lloyd-Max
def lloyd_max(x, k, max_iter=KM_ITERS):
    """1-D Lloyd k-means with mean update; deterministic equal-percentile init."""
    x = np.asarray(x, dtype=np.float64)
    c = np.percentile(x, (np.arange(k) + 0.5) * (100.0 / k))
    for _ in range(max_iter):
        mid = 0.5 * (c[:-1] + c[1:])
        idx = np.searchsorted(mid, x)
        cnt = np.bincount(idx, minlength=k)
        tot = np.bincount(idx, weights=x, minlength=k)
        new = np.where(cnt > 0, tot / np.maximum(cnt, 1), c)  # empty cluster keeps centroid
        if np.max(np.abs(new - c)) <= 1e-9 * (np.max(np.abs(c)) + 1e-30):
            c = new
            break
        c = new
    return np.sort(c)


# ---------------------------------------------------------------- streaming
def tensor_chunks(shard_file, base, num_bits):
    """Yield (dequantized float32 chunk [r0:r1], row0) — reuses Task 2's path."""
    packed, scale, shape = hz.load_packed(shard_file, base)
    gs = shape[1] // scale.shape[1]
    sc = scale.to(torch.float32).repeat_interleave(gs, dim=1).numpy()
    off = 2 ** (num_bits - 1)
    for r0 in range(0, shape[0], hz.CHUNK_ROWS):
        r1 = min(r0 + hz.CHUNK_ROWS, shape[0])
        up = hz.dequant_chunk(packed[r0:r1], num_bits, shape[1])
        w = (up - off).astype(np.float32) * sc[r0:r1]
        yield w, r0


def tensor_shape(shard_file, base):
    f = hz.safe_open(hz.CKPT / shard_file, framework="pt", device="cpu")
    return tuple(f.get_tensor(f"{base}.weight_shape").tolist())


def array_chunks(w, chunk_rows=hz.CHUNK_ROWS):
    for r0 in range(0, w.shape[0], chunk_rows):
        yield w[r0:r0 + chunk_rows], r0


def diag_reduce(m2, n, c):
    """Per-class max from M2[j, c] = max over block-rows of |value| at
    (row0 + b*N + j, col c), with row0 % N == 0 and C % N == 0.
    class(j, c) = (j + c) % N."""
    cm = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        cm[j] = m2[j].reshape(c // n, n).max(axis=0)  # index m <-> c % N == m
    p = np.stack([np.roll(cm[j], j) for j in range(n)])  # p[t, j] = cm[j][(t - j) % n]
    return p.max(axis=1)


def class_flat(salt_def, n, rc, row0, c):
    rr = np.arange(row0, row0 + rc, dtype=np.int64)
    cc = np.arange(c, dtype=np.int64)
    if salt_def == "row":
        g = np.repeat((rr % n)[:, None], c, axis=1)
    elif salt_def == "col":
        g = np.broadcast_to((cc % n)[None, :], (rc, c))
    elif salt_def == "linear":
        # linear-as-col shortcut: (r*c_total + c) % n == c % n ONLY if n | c_total
        # (review fix #2: guard the shortcut instead of relying on it silently)
        assert c % n == 0, (f"linear-as-col shortcut invalid: C={c} not divisible "
                            f"by N={n} (true linear class would be (row*C+col) % n)")
        g = np.broadcast_to((cc % n)[None, :], (rc, c))
    elif salt_def == "row+col":
        g = (rr[:, None] + cc[None, :]) % n
    else:
        raise ValueError(salt_def)
    return g.astype(np.int32).ravel()


def quantize_generic(wf, cf, centers_cls, block=BLOCK):
    """Nearest-centroid error with a per-element codebook (generic fallback)."""
    n = wf.size
    err = np.empty(n, dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        cb = centers_cls[cf[s:e]]                      # (b, k)
        d = np.abs(wf[s:e, None] - cb)
        idx = np.argmin(d, axis=1)
        best = np.take_along_axis(cb, idx[:, None], axis=1)[:, 0]
        err[s:e] = wf[s:e] - best
    return err


def quantize_generic_scale(wf, cf, centers_n_cls, scales_cls, block=BLOCK):
    err = np.empty(wf.size, dtype=np.float32)
    n = wf.size
    for s in range(0, n, block):
        e = min(s + block, n)
        cfs = cf[s:e]
        sc = scales_cls[cfs]
        cb = centers_n_cls[cfs]
        wn = wf[s:e] / sc
        d = np.abs(wn[:, None] - cb)
        idx = np.argmin(d, axis=1)
        recon = np.take_along_axis(cb, idx[:, None], axis=1)[:, 0] * sc
        err[s:e] = wf[s:e] - recon
    return err


# ---------------------------------------------------------------- core
def run_analysis(chunks_fn, shape, salt_def, n, ks, stride=SUB_ROW_STRIDE, verbose=False):
    """Full pipeline for one (tensor, salt) — used by unit tests AND real run."""
    r, c = shape
    assert c % n == 0, f"C={c} not divisible by N={n} (needed for col/row+col reductions)"
    assert r % n == 0 and hz.CHUNK_ROWS % n == 0

    # ---- pass 1: per-class absmax (exact, full tensor) + deterministic subsample
    lists = [[] for _ in range(n)]
    amax = np.zeros(n, dtype=np.float64)
    n_sub = 0
    for w, r0 in chunks_fn():
        rc = w.shape[0]
        assert rc % n == 0, f"chunk rows {rc} not divisible by N={n}"
        if salt_def == "row":
            for j in range(n):
                amax[j] = max(amax[j], float(np.abs(w[j::n]).max()))
        elif salt_def in ("col", "linear"):
            for j in range(n):
                amax[j] = max(amax[j], float(np.abs(w[:, j::n]).max()))
        else:  # row+col
            m2 = np.abs(w).reshape(rc // n, n, c).max(axis=0)
            amax = np.maximum(amax, diag_reduce(m2, n, c))
        rr = np.arange(r0, r0 + rc)
        mask = (rr % stride) == 0
        sub = w[mask]
        rs = sub.shape[0]
        if rs:
            rr_s = rr[mask]
            if salt_def == "row":
                fc = np.repeat((rr_s % n), c)
            elif salt_def in ("col", "linear"):
                fc = np.tile(np.arange(c) % n, rs)
            else:
                fc = ((rr_s[:, None] + np.arange(c)[None, :]) % n).ravel()
            fv = sub.ravel()
            order = np.argsort(fc, kind="stable")
            fc_s = fc[order]
            bnd = np.searchsorted(fc_s, np.arange(n + 1))
            for j in range(n):
                seg = fv[order[bnd[j]:bnd[j + 1]]]
                if seg.size:
                    lists[j].append(seg)
            n_sub += rs
    per_class_x = [np.concatenate(l) if l else np.zeros(0, np.float32) for l in lists]
    xg = np.concatenate(per_class_x)
    if verbose:
        print(f"  fit samples: total={xg.size:,}  per-class min={min(x.size for x in per_class_x):,}")

    # ---- fit: global book on union; per-class raw books; ONE shared normalized book
    scales = np.where(amax > 0, amax, 1.0)          # per-class absmax scale (guard 0)
    fit_counts = [int(x.size) for x in per_class_x]
    centers_g, centers_c, centers_n = {}, {}, {}
    for k in ks:
        centers_g[k] = lloyd_max(xg, k)
        cc = np.zeros((n, k))
        for j in range(n):
            xj = per_class_x[j] if per_class_x[j].size else xg  # guard: empty class
            cc[j] = lloyd_max(xj, k)
        centers_c[k] = cc
        # salted_scale (review fix #1): ONE shared k-entry book fit on the union of
        # all classes normalized by their OWN absmax — the old per-class normalized
        # books were provably identical to cc (scale equivariance of Lloyd-Max).
        # lloyd_max sorts centroids, so the union order does not matter.
        xn = np.concatenate([
            (per_class_x[j].astype(np.float64) / scales[j]) if per_class_x[j].size
            else (xg.astype(np.float64) / scales[j])
            for j in range(n)])
        centers_n[k] = np.tile(lloyd_max(xn, k), (n, 1))  # (n, k), rows identical

    # ---- pass 2: exact quantize over the FULL tensor, all arms
    sse = {k: {"global": 0.0, "salted": 0.0, "salted_scale": 0.0} for k in ks}
    gmax = {k: 0.0 for k in ks}
    cmax_s = {k: np.zeros(n) for k in ks}
    cmax_ss = {k: np.zeros(n) for k in ks}
    total = 0
    nchunks = 0
    generic = salt_def == "row+col"
    for w, r0 in chunks_fn():
        rc = w.shape[0]
        wf = w.ravel()
        total += wf.size
        cf = class_flat(salt_def, n, rc, r0, c) if generic else None
        for k in ks:
            cg = centers_g[k].astype(np.float32)
            midg = 0.5 * (cg[:-1] + cg[1:])
            eg = wf - cg[np.searchsorted(midg, wf)]
            sse[k]["global"] += float(np.square(eg.astype(np.float64)).sum())
            gmax[k] = max(gmax[k], float(np.abs(eg).max()))
            cc = centers_c[k].astype(np.float32)
            cn = centers_n[k].astype(np.float32)
            sc32 = scales.astype(np.float32)
            if not generic:
                for j in range(n):
                    blk = w[j::n] if salt_def == "row" else w[:, j::n]
                    mid_c = 0.5 * (cc[j, :-1] + cc[j, 1:])
                    ib = np.searchsorted(mid_c, blk.ravel())
                    eb = blk - cc[j][ib].reshape(blk.shape)
                    sse[k]["salted"] += float(np.square(eb.astype(np.float64)).sum())
                    cmax_s[k][j] = max(cmax_s[k][j], float(np.abs(eb).max()))
                    mid_n = 0.5 * (cn[j, :-1] + cn[j, 1:])
                    wn = blk.ravel().astype(np.float32) / sc32[j]
                    i2 = np.searchsorted(mid_n, wn)
                    recon = cn[j][i2].reshape(blk.shape) * sc32[j]
                    e2 = blk - recon
                    sse[k]["salted_scale"] += float(np.square(e2.astype(np.float64)).sum())
                    cmax_ss[k][j] = max(cmax_ss[k][j], float(np.abs(e2).max()))
            else:
                eb = quantize_generic(wf, cf, cc)
                sse[k]["salted"] += float(np.square(eb.astype(np.float64)).sum())
                ae = np.abs(eb).reshape(rc // n, n, c).max(axis=0)
                cmax_s[k] = np.maximum(cmax_s[k], diag_reduce(ae, n, c))
                e2 = quantize_generic_scale(wf, cf, cn, sc32)
                sse[k]["salted_scale"] += float(np.square(e2.astype(np.float64)).sum())
                ae2 = np.abs(e2).reshape(rc // n, n, c).max(axis=0)
                cmax_ss[k] = np.maximum(cmax_ss[k], diag_reduce(ae2, n, c))
        if verbose:
            nchunks += 1
            if nchunks % 16 == 0:
                print(f"    quantized {total:,} elems", flush=True)

    out = {}
    for k in ks:
        mg = sse[k]["global"] / total
        ms = sse[k]["salted"] / total
        mss = sse[k]["salted_scale"] / total
        out[k] = {
            "k": k,
            "entries": {"global": k, "salted": n * k, "salted_scale": k + n},
            "mse_global": mg, "mse_salted": ms, "mse_salted_scale": mss,
            "ratio_salted": ms / mg, "ratio_salted_scale": mss / mg,
            "max_abs_err_global": gmax[k],
            "class_max_err_salted": [float(v) for v in cmax_s[k]],
            "class_max_err_salted_scale": [float(v) for v in cmax_ss[k]],
        }
    out["_scales_absmax"] = [float(v) for v in scales]
    out["_n_total"] = total
    out["_n_fit_samples"] = int(xg.size)
    out["_fit_counts_per_class"] = fit_counts
    out["_fit_stride"] = int(stride)
    return out


# ---------------------------------------------------------------- unit tests
def unit_tests():
    print("== Unit tests (synthetic, real projection shapes, before real data) ==")
    rows, cols = 17408, 5120  # gate_proj shape
    rng = np.random.default_rng(0)
    res = {}

    # 1) row-structured: same construction as heterogeneity.py (r % 8 * 0.5 offsets)
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    w += ((np.arange(rows) % 8) * 0.5).astype(np.float32)[:, None]
    r = run_analysis(lambda: array_chunks(w), (rows, cols), "row", 64, [KS[2]])[KS[2]]
    r1 = r["ratio_salted"]
    ok1 = r1 < 0.9
    print(f"  row-structured {rows}x{cols} salt=row%64 2-bit: "
          f"MSE_g={r['mse_global']:.5f} MSE_s={r['mse_salted']:.5f} "
          f"ratio={r1:.4f} -> {'PASS' if ok1 else 'FAIL'} (expect < 0.9)")
    res["structured_ratio"] = float(r1)
    del w

    # 2) col-structured (review fix #2: covers the col/linear class-index branches,
    #    previously only the row path was exercised): per-col mean offsets, and the
    #    col%8 pattern SURVIVES at N=64 because 8 | 64 (a col class c%64 fixes c%8).
    w2 = rng.standard_normal((rows, cols)).astype(np.float32)
    w2 += ((np.arange(cols) % 8) * 0.5).astype(np.float32)[None, :]
    rc = run_analysis(lambda: array_chunks(w2), (rows, cols), "col", 64, [KS[2]])[KS[2]]
    rr = run_analysis(lambda: array_chunks(w2), (rows, cols), "row", 64, [KS[2]])[KS[2]]
    r3, r4 = rc["ratio_salted"], rr["ratio_salted"]
    ok3 = r3 < 0.9
    ok4 = r4 >= 0.9
    print(f"  col-structured {rows}x{cols} salt=col%64 2-bit: "
          f"MSE_g={rc['mse_global']:.5f} MSE_s={rc['mse_salted']:.5f} "
          f"ratio={r3:.4f} -> {'PASS' if ok3 else 'FAIL'} (expect < 0.9)")
    print(f"  col-structured {rows}x{cols} salt=row%64 2-bit (wrong salt): "
          f"MSE_s={rr['mse_salted']:.5f} ratio={r4:.4f} -> {'PASS' if ok4 else 'FAIL'} "
          f"(expect >= 0.9: row classes carry no col structure)")
    res["col_structured_ratio"] = float(r3)
    res["col_structured_wrong_salt_row_ratio"] = float(r4)
    del w2

    # 3) row+col structured (covers the generic row+col class-index branch):
    #    offsets on BOTH axes; class (r+c)%64 fixes (r%8 + c%8) mod 8 -> per-class
    #    distribution differs from global -> salted must beat global.
    w3 = rng.standard_normal((rows, cols)).astype(np.float32)
    w3 += (((np.arange(rows) % 8) * 0.5)[:, None]
           + ((np.arange(cols) % 8) * 0.5)[None, :]).astype(np.float32)
    rg = run_analysis(lambda: array_chunks(w3), (rows, cols), "row+col", 64, [KS[2]])[KS[2]]
    r5 = rg["ratio_salted"]
    ok5 = r5 < 0.9
    print(f"  row+col-structured {rows}x{cols} salt=row+col%64 2-bit: "
          f"MSE_g={rg['mse_global']:.5f} MSE_s={rg['mse_salted']:.5f} "
          f"ratio={r5:.4f} -> {'PASS' if ok5 else 'FAIL'} (expect < 0.9)")
    res["rowcol_structured_ratio"] = float(r5)
    del w3

    w4 = rng.standard_normal((rows, cols)).astype(np.float32)
    ri = run_analysis(lambda: array_chunks(w4), (rows, cols), "row", 64, [KS[2]])[KS[2]]
    r6 = ri["ratio_salted"]
    ok6 = 0.9 <= r6 <= 1.1
    print(f"  iid Gaussian   {rows}x{cols} salt=row%64 2-bit: "
          f"MSE_g={ri['mse_global']:.5f} MSE_s={ri['mse_salted']:.5f} "
          f"ratio={r6:.4f} -> {'PASS' if ok6 else 'FAIL'} (expect ~1.0)")
    res["iid_ratio"] = float(r6)
    return all([ok1, ok3, ok4, ok5, ok6]), res


# ---------------------------------------------------------------- main
def _worker_tensor(job):
    """Run ALL arms/bits for ONE tensor wholly inside one worker process.

    Deterministic, no shared mutable state: the worker dequantizes its own
    shards, fits every arm/bit, and computes every MSE cell itself, so the 9
    tensors can run fully concurrently (review parallelization requirement).
    """
    tkey, base, shard, bits, salt_str, salt_def, n, shape, stride = job
    # spawn workers re-import this module; cap torch's thread pools so 9 workers
    # don't oversubscribe the CPU (numerics unaffected: elementwise ops + exact
    # per-chunk numpy float64 reductions are deterministic regardless of threads)
    torch.set_num_threads(1)
    t0 = time.time()
    res = run_analysis(lambda: tensor_chunks(shard, base, bits), shape,
                       salt_def, n, [KS[b] for b in BITS_LIST], stride=stride)
    per_bits = {}
    beats = []
    for b in BITS_LIST:
        per_bits[str(b)] = res[KS[b]]
        if res[KS[b]]["ratio_salted"] < 0.9:
            beats.append(b)
    rec = {
        "desc": base, "num_bits_ckpt": bits, "salt": salt_str, "N": n,
        # persisted per tensor (review fix #3): per-class absmax of the salt used
        # (= the best salt from Task 2) + per-class fit-sample counts
        "absmax_per_class": [float(v) for v in res["_scales_absmax"]],
        "fit_samples_per_class": res["_fit_counts_per_class"],
        "n_fit_samples": res["_n_fit_samples"],
        "fit_stride": int(stride),
        "bits": per_bits,
        "verdict": (f"salted beats global (ratio<0.9) at bits {beats}" if beats
                    else "salted does NOT beat global at any bit level (all ratios >= 0.9)"),
        "runtime_s": round(time.time() - t0, 1),
    }
    if tkey == "layers.63.mlp.down_proj":
        rec["note"] = ("fit-noise-dominated at the pre-fix r%65 fit (old 3-bit r_s=1.1330; probe "
                       "evidence ~/probe_down_all2.py: r%65-fit 1.1330, r%33-fit 1.0623, all-rows "
                       "fit 0.9973); re-derived here at fit stride r%17 per review fix #3")
        rec["bits"]["3"]["note"] = ("fit-noise-dominated at pre-fix r%65 fit (r_s was 1.1330; "
                                    "probe: r%65 1.1330 / r%33 1.0623 / all-rows 0.9973); "
                                    "re-derived at r%17")
    return tkey, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-only", action="store_true")
    args = ap.parse_args()

    print("== Pin check (re-run of Task 2's verify_pin) ==")
    pin = hz.verify_pin("model.language_model.layers.0.mlp.gate_proj",
                        "model-00001-of-00007.safetensors", 4)
    if pin["max_abs_diff"] != 0.0:
        print("PIN FAILED - dequantization mismatch; aborting")
        sys.exit(1)

    ok, unit = unit_tests()
    if not ok:
        print("UNIT TESTS FAILED - aborting real run")
        sys.exit(1)
    if args.synthetic_only:
        print("\n--synthetic-only: stopping before real data")
        return

    het = json.loads((SALTED / "heterogeneity.json").read_text())
    # keep the pre-fix JSON for the regression comparison + as the sequential timing baseline
    old = json.loads((SALTED / "codebook_gain.json").read_text())
    (SALTED / "codebook_gain.pre_fix.json").write_text(json.dumps(old, indent=1))
    prior_sequential_s = round(sum(v["runtime_s"] for k, v in old.items() if not k.startswith("_")), 1)

    # build one job per tensor: wholly independent (own shards, own fit, own MSE cells)
    jobs = []
    for base, shard, bits in SAMPLES:
        tkey = base.split("model.language_model.")[-1] if "layers" in base else base
        salt_str = het[tkey]["best"]["salt"]          # e.g. "linear%64"
        salt_def, n = salt_str.split("%")
        n = int(n)
        shape = tensor_shape(shard, base)
        stride = SUB_ROW_STRIDE_5120 if (salt_def == "row" and shape[0] == 5120) else SUB_ROW_STRIDE
        jobs.append((tkey, base, shard, bits, salt_str, salt_def, n, shape, stride))

    workers = min(9, os.cpu_count() - 2)
    # spawn (not fork): forking a multi-threaded torch/jemalloc process deadlocks
    # the workers on a futex held by a thread that does not exist in the child
    # (observed: all 9 fork workers stuck in futex_wait at 0% CPU); spawn workers
    # re-import this module fresh (main() is guarded) and never inherit the pools
    mp_spawn = multiprocessing.get_context("spawn")
    print(f"\n== Real data: salted vs global codebook gain "
          f"(ratio = MSE_salted / MSE_global), PARALLEL over {len(jobs)} tensors "
          f"with {workers} workers (cpu_count={os.cpu_count()}, spawn context) ==")
    t_sweep = time.time()
    done = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_spawn) as ex:
        futs = {ex.submit(_worker_tensor, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            tkey, rec = fut.result()
            done[tkey] = rec
            print(f"  [done] {tkey:38s} worker runtime {rec['runtime_s']:6.1f}s "
                  f"(wall {time.time() - t_sweep:6.1f}s)", flush=True)
    wall_parallel = time.time() - t_sweep
    results = {"_pin": pin, "_unit_tests": unit, "_meta": META}
    for tkey, base, shard, bits, salt_str, salt_def, n, shape, stride in jobs:
        results[tkey] = done[tkey]
    results["_parallel"] = {
        "workers": workers,
        "wall_clock_parallel_s": round(wall_parallel, 1),
        "prior_sequential_s": prior_sequential_s,
        "prior_sequential_source": ("sum of per-tensor runtime_s in codebook_gain.pre_fix.json "
                                    "(pre-fix run, same arms except salted_scale)"),
        "speedup_vs_prior_sequential": round(prior_sequential_s / wall_parallel, 2),
    }

    # ---- regression check (review-fix acceptance): global + salted arms are
    # UNCHANGED by the fix, so every r_s/mse cell must reproduce the pre-fix JSON
    # to float64 noise — EXCEPT the explicitly re-derived cells (down L63: its fit
    # stride changed r%65 -> r%17, which re-fits global+salted there by design).
    REDERIVED = {"layers.63.mlp.down_proj"}
    print("\n== Regression check vs codebook_gain.pre_fix.json "
          "(mse_global, mse_salted, ratio_salted; tol 1e-12 relative) ==")
    mismatches = []
    for tkey, *_ in jobs:
        for b in BITS_LIST:
            o = old[tkey]["bits"][str(b)]
            nw = results[tkey]["bits"][str(b)]
            if tkey in REDERIVED:
                print(f"  {tkey:38s} {b}b  RE-DERIVED (fit stride r%65 -> r%17): "
                      f"r_s {o['ratio_salted']:.6f} -> {nw['ratio_salted']:.6f}")
                continue
            for key in ("mse_global", "mse_salted", "ratio_salted"):
                rel = abs(nw[key] - o[key]) / max(abs(o[key]), 1e-300)
                if rel > 1e-12:
                    mismatches.append((tkey, b, key, o[key], nw[key], rel))
            print(f"  {tkey:38s} {b}b  r_s {o['ratio_salted']:.6f} == "
                  f"{nw['ratio_salted']:.6f}  "
                  f"{'MISMATCH' if any(m[0] == tkey and m[1] == b for m in mismatches) else 'ok'}")
    if mismatches:
        print("REGRESSION CHECK FAILED (global/salted arms must be unchanged):")
        for m in mismatches:
            print(f"  {m[0]} {m[1]}b {m[2]}: old={m[3]!r} new={m[4]!r} rel={m[5]:.3e}")
        sys.exit(1)
    print("  all unchanged cells reproduce the pre-fix values (rel diff <= 1e-12)")

    print("\n== Table (r_s = MSE_salted/MSE_global, r_ss = MSE_salted+scale/MSE_global) ==")
    print(f"{'tensor':38s} {'bits':>4s} {'k':>3s} {'MSE_global':>12s} {'MSE_salted':>12s} "
          f"{'r_s':>7s} {'MSE_salt+scale':>14s} {'r_ss':>7s}  verdict")
    for tkey, *_ in jobs:
        for b in BITS_LIST:
            r = results[tkey]["bits"][str(b)]
            cm = r["class_max_err_salted"]
            print(f"{tkey:38s} {b:4d} {r['k']:3d} {r['mse_global']:12.4e} {r['mse_salted']:12.4e} "
                  f"{r['ratio_salted']:7.4f} {r['mse_salted_scale']:14.4e} "
                  f"{r['ratio_salted_scale']:7.4f}  "
                  f"{'SALTED WINS' if r['ratio_salted'] < 0.9 else 'no gain'}")
            print(f"{'':38s}      per-class max err (salted): max={max(cm):.4e} "
                  f"median={float(np.median(cm)):.4e}  global max={r['max_abs_err_global']:.4e}")

    (SALTED / "codebook_gain.json").write_text(json.dumps(results, indent=1))
    print("\n== Verdicts ==")
    for tkey, *_ in jobs:
        print(f"  {tkey:38s} {results[tkey]['verdict']}")
    print(f"\nParallel sweep wall clock: {wall_parallel:.1f}s for {len(jobs)} tensors "
          f"({workers} workers) vs {prior_sequential_s:.1f}s sequential pre-fix run "
          f"(speedup {prior_sequential_s / wall_parallel:.2f}x)")
    print("Wrote analysis/salted/codebook_gain.json (+ codebook_gain.pre_fix.json backup)")


if __name__ == "__main__":
    main()
