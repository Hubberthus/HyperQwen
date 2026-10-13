"""Block-diagonal Hadamard rotation simulation on BF16 originals (NEW experiment).

The salted spike never rotated. This simulates, CPU-only, what input-side
Hadamard rotation does to group-128 uniform quantization quality:

Tensors (original BF16, models-fp16/Qwen3.8-27B):
  lm_head              [248320, 5120]  -> configs: 40 blocks x 128, 20 blocks x 256
  L63 mlp down_proj    [5120, 17408]   -> config:  136 blocks x 128

Method:
  H = block-diagonal orthonormal Hadamard on the INPUT dim (Sylvester
  construction, H_b = H_{2b} recursion, normalized H @ H^T = I).
  W' = W @ H (block-wise matmul; H is symmetric so H^T = H).
  Quantizer: symmetric uniform group-128, per-group scale = absmax / 2^(bits-1),
  round-to-nearest, clip to [-(2^(bits-1)-1), 2^(bits-1)-1].
  Rotated result is rotated BACK to the original frame (Wq @ H^T) for all
  comparisons, because the served checkpoint lives in the original frame.
  (MSE vs the BF16 original is frame-invariant under the orthogonal rotation.)

Per config: MSE vs BF16 original, MSE vs current-checkpoint dequant,
max abs elementwise error (both references).
lm_head extra: per-row (vocab) mean abs error of rotated INT4 vs checkpoint.

Checkpoint dequant uses the PINNED path from analysis/salted/heterogeneity.py
(load_packed + dequant_chunk, 0.0 max-diff pin vs compressed_tensors) — the
only import of spike code, as explicitly allowed; all rotation/quant code here
is fresh.

Run:  venv/bin/python analysis/salted/rotation/rotation_sim.py
Writes: analysis/salted/rotation/rotation_sim.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
CKPT_FP16 = REPO / "models-fp16" / "Qwen3.8-27B"
SALTED = REPO / "analysis" / "salted"
OUTDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SALTED))
import heterogeneity as hz  # pinned checkpoint dequant only

LM = "lm_head"
DP = "model.language_model.layers.63.mlp.down_proj"


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------- hadamard
def sylvester(n):
    """Sylvester construction of the (symmetric) Hadamard, normalized to I."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def block_apply(W, H):
    """W @ block_diag(H, ..., H) via broadcast matmul (no dense block diag)."""
    r, c = W.shape
    b = H.shape[0]
    return (W.reshape(r, c // b, b) @ H).reshape(r, c)


# ------------------------------------------------------------- metrics
def struct_metrics(W):
    r, c = W.shape
    amax = np.abs(W.reshape(r, c // 128, 128)).max(axis=2).ravel()
    med = float(np.median(amax))
    gmax = float(np.abs(W).max())
    step = max(1, W.size // 10_000_000)
    p9999 = float(np.percentile(np.abs(W).ravel()[::step], 99.99))
    return {
        "group128_absmax_p50": med,
        "group128_absmax_max": float(amax.max()),
        "group128_absmax_spread": float(amax.max() / med),
        "global_max": gmax,
        "p9999_over_max": p9999 / gmax,
    }


def mse_between(A, B, chunk=16384):
    s = 0.0
    for r0 in range(0, A.shape[0], chunk):
        d = A[r0:r0 + chunk] - B[r0:r0 + chunk]
        s += float((d * d).sum())
    return s / A.size


# ------------------------------------------------------------- quant
def quant_group128(W, bits):
    """Symmetric uniform group-128: scale = absmax / 2^(bits-1), clip, dequant."""
    r, c = W.shape
    g = c // 128
    Wr = W.reshape(r, g, 128)
    amax = np.abs(Wr).max(axis=2)
    s = np.where(amax > 0, amax / (2.0 ** (bits - 1)), 1.0)
    lim = 2 ** (bits - 1) - 1
    q = np.round(Wr / s[:, :, None]).clip(-lim, lim)
    return (q * s[:, :, None]).reshape(r, c)


def eval_quant(Win, Wd, Hback, bits, per_row=False, chunk=16384, Worig=None):
    """Quantize Win (group-128, `bits`), rotate back with Hback if given,
    and accumulate MSE/max-abs vs the BF16 original (Worig, original frame;
    defaults to Win for the unrotated case) and vs the checkpoint dequant Wd
    (both references in the original frame)."""
    if Worig is None:
        Worig = Win
    rows, cols = Win.shape
    nback = cols // Hback.shape[0] if Hback is not None else None
    sq_o = sq_c = 0.0
    mx_o = mx_c = 0.0
    prow = np.zeros(rows, np.float64) if per_row else None
    for r0 in range(0, rows, chunk):
        r1 = min(r0 + chunk, rows)
        wc = Win[r0:r1]
        wq = quant_group128(wc, bits)
        if Hback is not None:
            wq = (wq.reshape(r1 - r0, nback, Hback.shape[0]) @ Hback
                  ).reshape(r1 - r0, cols)
        do = wq - Worig[r0:r1]
        dc = wq - Wd[r0:r1]
        sq_o += float((do * do).sum())
        mx_o = max(mx_o, float(np.abs(do).max()))
        sq_c += float((dc * dc).sum())
        mx_c = max(mx_c, float(np.abs(dc).max()))
        if prow is not None:
            prow[r0:r1] = np.abs(dc).mean(1)
    n = Win.size
    out = {
        "mse_vs_orig": sq_o / n,
        "mse_vs_ckpt": sq_c / n,
        "max_abs_vs_orig": mx_o,
        "max_abs_vs_ckpt": mx_c,
    }
    if prow is not None:
        out["per_row_vs_ckpt"] = {
            "mean": float(prow.mean()), "p50": float(np.median(prow)),
            "p99": float(np.percentile(prow, 99)), "max": float(prow.max()),
            "top10_rows": [[int(i), float(prow[i])]
                            for i in np.argsort(prow)[::-1][:10]],
        }
    return out


# ------------------------------------------------------------- loading
def load_orig(name):
    idx = json.loads((CKPT_FP16 / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(CKPT_FP16 / idx[name], framework="pt", device="cpu") as f:
        w = f.get_tensor(name)
    assert str(w.dtype) == "torch.bfloat16", f"{name}: {w.dtype}"
    return w.to(torch.float32).numpy()


def dequant_ckpt(base, shard, bits):
    """Pinned tiled unpack+dequant (heterogeneity.py) -> full float32."""
    packed, scale, shape = hz.load_packed(shard, base)
    gs = shape[1] // scale.shape[1]
    assert gs == 128, gs
    sc = scale.to(torch.float32).repeat_interleave(gs, dim=1).numpy()
    out = np.empty(shape, np.float32)
    for r0 in range(0, shape[0], 4096):
        r1 = min(r0 + 4096, shape[0])
        up = hz.dequant_chunk(packed[r0:r1], bits, shape[1])
        out[r0:r1] = (up - 2 ** (bits - 1)).astype(np.float32) * sc[r0:r1]
    return out


# ------------------------------------------------------------- main
def main():
    t0 = time.time()
    H128 = sylvester(128)
    H256 = sylvester(256)
    assert np.abs(H128 @ H128.T - np.eye(128)).max() < 1e-6
    assert np.abs(H256 @ H256.T - np.eye(256)).max() < 1e-6
    log("Hadamard H128/H256 built, orthonormality verified")

    results = {}

    # ================= lm_head =================
    log("loading lm_head original (BF16) ...")
    W = load_orig(LM + ".weight")
    log(f"lm_head original {W.shape} loaded ({W.nbytes / 1e9:.2f} GB)")
    log("dequantizing current lm_head checkpoint (INT8, pinned path) ...")
    Wd = dequant_ckpt(LM, "model-00007-of-00007.safetensors", 8)
    ckpt_lm = {
        "mse_vs_orig": mse_between(Wd, W),
        "max_abs_vs_orig": float(np.abs(Wd - W).max()),
    }
    log(f"lm_head checkpoint dequant: mse_vs_orig = {ckpt_lm['mse_vs_orig']:.6e}")

    log("rotating lm_head (40x128 and 20x256) ...")
    Wrot128 = block_apply(W, H128)
    Wrot256 = block_apply(W, H256)

    struct = {
        "unrotated": struct_metrics(W),
        "rot128": struct_metrics(Wrot128),
        "rot256": struct_metrics(Wrot256),
    }
    del Wrot256
    for k, v in struct.items():
        log(f"lm_head {k}: absmax spread {v['group128_absmax_spread']:.2f}  "
            f"max|w| {v['global_max']:.4f}  p99.99/max {v['p9999_over_max']:.4f}")

    log("quantizing lm_head configs ...")
    cfgs = {
        "unrot_int4": eval_quant(W, Wd, None, 4),
        "unrot_int8": eval_quant(W, Wd, None, 8),
        "rot128_int4": eval_quant(Wrot128, Wd, H128, 4, per_row=True, Worig=W),
        "rot128_int8": eval_quant(Wrot128, Wd, H128, 8, Worig=W),
    }
    del Wrot128
    Wrot256 = block_apply(W, H256)
    cfgs["rot256_int4"] = eval_quant(Wrot256, Wd, H256, 4, Worig=W)
    del Wrot256
    for k, v in cfgs.items():
        log(f"lm_head {k}: mse_vs_orig {v['mse_vs_orig']:.6e}  "
            f"mse_vs_ckpt {v['mse_vs_ckpt']:.6e}  max_abs {v['max_abs_vs_orig']:.4f}")

    results["lm_head"] = {
        "shape": list(W.shape),
        "structural": struct,
        "current_checkpoint": ckpt_lm,
        "configs": cfgs,
    }
    del W, Wd

    # ================= down_proj L63 =================
    log("loading L63 down_proj original (BF16) ...")
    V = load_orig(DP + ".weight")
    log(f"down_proj original {V.shape} loaded")
    log("dequantizing current L63 down_proj checkpoint (INT4, pinned path) ...")
    Vd = dequant_ckpt(DP, "model-00004-of-00007.safetensors", 4)
    ckpt_dp = {
        "mse_vs_orig": mse_between(Vd, V),
        "max_abs_vs_orig": float(np.abs(Vd - V).max()),
    }
    log(f"down_proj checkpoint dequant: mse_vs_orig = {ckpt_dp['mse_vs_orig']:.6e}")

    log("rotating down_proj (136x128) ...")
    Vrot = block_apply(V, H128)
    struct_v = {
        "unrotated": struct_metrics(V),
        "rot128": struct_metrics(Vrot),
    }
    for k, v in struct_v.items():
        log(f"down_proj {k}: absmax spread {v['group128_absmax_spread']:.2f}  "
            f"max|w| {v['global_max']:.4f}  p99.99/max {v['p9999_over_max']:.4f}")

    log("quantizing down_proj configs ...")
    cfgs_v = {
        "unrot_int4": eval_quant(V, Vd, None, 4, chunk=5120),
        "unrot_int8": eval_quant(V, Vd, None, 8, chunk=5120),
        "rot128_int4": eval_quant(Vrot, Vd, H128, 4, chunk=5120, Worig=V),
        "rot128_int8": eval_quant(Vrot, Vd, H128, 8, chunk=5120, Worig=V),
    }
    for k, v in cfgs_v.items():
        log(f"down_proj {k}: mse_vs_orig {v['mse_vs_orig']:.6e}  "
            f"mse_vs_ckpt {v['mse_vs_ckpt']:.6e}  max_abs {v['max_abs_vs_orig']:.4f}")

    results["down_proj_l63"] = {
        "shape": list(V.shape),
        "structural": struct_v,
        "current_checkpoint": ckpt_dp,
        "configs": cfgs_v,
    }

    # embed null re-derivation if present
    nf = OUTDIR / "nulls.json"
    if nf.exists():
        results["nulls"] = json.loads(nf.read_text())

    results["runtime_s"] = round(time.time() - t0, 1)
    (OUTDIR / "rotation_sim.json").write_text(json.dumps(results, indent=1))
    log(f"wrote {OUTDIR / 'rotation_sim.json'}  (wall {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
