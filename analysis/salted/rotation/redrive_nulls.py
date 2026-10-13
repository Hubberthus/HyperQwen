"""Fresh re-derivation of the salted-spike null numbers from the on-disk BF16 originals.

NO import of the spike's analysis code (fresh re-derivation, per task contract).
Tensors (models-fp16/Qwen3.8-27B, dtype asserted BFloat16 from headers):
  lm_head                      [248320, 5120]
  L63 mlp down_proj            [5120, 17408]   (row=output, col=input)

Nulls re-derived:
  (a) lm_head eta^2, salt col%64
      eta^2 = between-class var / total var; classes = col % 64,
      between = sum_c n_c (mu_c - mu)^2 / N, total = between + sum_c n_c var_c / N.
      (For cols=5120, linear%64 == col%64 exactly, so this is the spike's
      "linear%64 = 4.318e-4" number.)
  (b) lm_head residue-class mean-|w|^2 deviation vs iid floor, col%128 / col%64
      dev = (max_c ms_c - min_c ms_c) / mean_c ms_c,  ms_c = mean w^2 in class c.
  (c) lm_head per-group-128 absmax: p50 and max of the [rows, cols/128] table.
  (d) L63 down_proj col%128 residue deviation ratio.

iid floor (floor_layout.py was a lost scratch script; reconstructed here, fresh):
  floor(N) = C(N) * sqrt(Var(w^2) / N_c) / E[w^2]
  C(N) = E[max - min of N iid N(0,1)] via Monte Carlo (fixed seed 1234),
  Var(w^2)/E[w^2] estimated exactly (streaming float64) from the tensor itself,
  so the floor matches the tensor's real marginal (heavy tails included).

Run:  venv/bin/python analysis/salted/rotation/redrive_nulls.py
Writes: analysis/salted/rotation/nulls.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
CKPT = REPO / "models-fp16" / "Qwen3.8-27B"
OUTDIR = Path(__file__).resolve().parent

LM = "lm_head.weight"
DP = "model.language_model.layers.63.mlp.down_proj.weight"
CHUNK = 16384


def log(msg):
    print(msg, flush=True)


def load_full(name):
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(CKPT / idx[name], framework="pt", device="cpu") as f:
        w = f.get_tensor(name)
    assert str(w.dtype) == "torch.bfloat16", f"{name}: {w.dtype}"
    return w.to(torch.float32).numpy()


class ColEta:
    """eta^2 for salt col%N accumulated over row chunks. Fresh code."""

    def __init__(self, N, cols):
        self.N, self.cols = N, cols
        self.n = np.zeros(N, np.int64)
        self.s = np.zeros(N, np.float64)
        self.ss = np.zeros(N, np.float64)
        self.total = 0
        self._cls = (np.arange(cols) % N).astype(np.int32)

    def add(self, w):
        r, c = w.shape
        flat = np.broadcast_to(self._cls, (r, c)).ravel()
        v = w.ravel().astype(np.float64)
        self.n += np.bincount(flat, minlength=self.N)
        self.s += np.bincount(flat, weights=v, minlength=self.N)
        self.ss += np.bincount(flat, weights=v * v, minlength=self.N)
        self.total += v.size

    def result(self):
        mu = self.s / self.n
        var = self.ss / self.n - mu * mu
        mu_all = self.s.sum() / self.total
        between = float(((mu - mu_all) ** 2 * self.n).sum() / self.total)
        total = between + float((var * self.n).sum() / self.total)
        return between / total


class ColMs:
    """Per-residue-class mean of w^2 for col%N, over row chunks. Fresh code."""

    def __init__(self, N, cols):
        self.N, self.cols = N, cols
        self.s2 = np.zeros(N, np.float64)
        self.total = 0
        self._cls = np.arange(cols) % N

    def add(self, w):
        r, c = w.shape
        w64 = w.astype(np.float64)
        css = (w64 * w64).sum(0)
        self.s2 += np.bincount(self._cls, weights=css, minlength=self.N)
        self.total += w.size

    def ms(self):
        return self.s2 / (self.total // self.N)


def group128_absmax(W):
    r, c = W.shape
    return np.abs(W.reshape(r, c // 128, 128)).max(axis=2)


def range_coef(N, draws=2_000_000, seed=1234):
    """E[max - min of N iid standard normals], Monte Carlo, fixed seed."""
    rng = np.random.default_rng(seed)
    block = 200_000
    acc = 0.0
    for _ in range(draws // block):
        x = rng.standard_normal((block, N))
        acc += float((x.max(axis=1) - x.min(axis=1)).sum())
    return acc / draws


def moments(W, chunk=CHUNK):
    """Exact (E[w^2], Var(w^2)) in float64, streamed over row chunks."""
    s2 = s4 = 0.0
    for r0 in range(0, W.shape[0], chunk):
        x = W[r0:r0 + chunk].ravel().astype(np.float64)
        s2 += float((x * x).sum())
        s4 += float((x * x * x * x).sum())
    n = W.size
    m2 = s2 / n
    return m2, s4 / n - m2 * m2


def res_dev_floor(W, N, C):
    acc = ColMs(N, W.shape[1])
    for r0 in range(0, W.shape[0], CHUNK):
        acc.add(W[r0:r0 + CHUNK])
    ms = acc.ms()
    dev = float((ms.max() - ms.min()) / ms.mean())
    M2, V2 = moments(W)
    Nc = W.size // N
    floor = C * np.sqrt(V2 / Nc) / M2
    return dev, float(floor), dev / floor, float(ms.max()), float(ms.min())


def main():
    t0 = time.time()
    C64 = range_coef(64)
    C128 = range_coef(128)
    log(f"range coefs: C(64)={C64:.5f}  C(128)={C128:.5f}")

    # ---------------- lm_head ----------------
    log("loading lm_head (BF16) ...")
    W = load_full(LM)
    log(f"lm_head shape {W.shape} dtype {W.dtype}  ({W.nbytes / 1e9:.2f} GB f32)")

    eta = ColEta(64, W.shape[1])
    for r0 in range(0, W.shape[0], CHUNK):
        eta.add(W[r0:r0 + CHUNK])
    eta2_64 = eta.result()

    dev128, fl128, rat128, ms128max, ms128min = res_dev_floor(W, 128, C128)
    dev64, fl64, rat64, ms64max, ms64min = res_dev_floor(W, 64, C64)
    log(f"lm_head eta2 col%64 = {eta2_64:.6e}")
    log(f"lm_head res128 dev = {dev128:.4f}  floor = {fl128:.4e}  ratio = {rat128:.1f}")
    log(f"lm_head res64  dev = {dev64:.4f}  floor = {fl64:.4e}  ratio = {rat64:.1f}")

    amax = group128_absmax(W)
    p50, amax_max = float(np.median(amax)), float(amax.max())
    del W, amax
    log(f"lm_head group128 absmax: p50 = {p50:.4e}  max = {amax_max:.4e}  max/p50 = {amax_max / p50:.2f}")

    # ---------------- down_proj L63 ----------------
    log("loading L63 down_proj (BF16) ...")
    V = load_full(DP)
    log(f"down_proj shape {V.shape}")
    vdev128, vfl128, vrat128, vmsmax, vmsmin = res_dev_floor(V, 128, C128)
    log(f"down_proj res128 dev = {vdev128:.4f}  floor = {vfl128:.4e}  ratio = {vrat128:.1f}")
    del V

    out = {
        "range_coef": {"C64": C64, "C128": C128},
        "lm_head": {
            "eta2_col64": eta2_64,
            "res128": {"dev": dev128, "iid_floor": fl128, "ratio": rat128,
                        "class_ms_max": ms128max, "class_ms_min": ms128min},
            "res64": {"dev": dev64, "iid_floor": fl64, "ratio": rat64,
                       "class_ms_max": ms64max, "class_ms_min": ms64min},
            "group128_absmax": {"p50": p50, "max": amax_max, "max_over_p50": amax_max / p50},
        },
        "down_proj_l63": {
            "res128": {"dev": vdev128, "iid_floor": vfl128, "ratio": vrat128,
                        "class_ms_max": vmsmax, "class_ms_min": vmsmin},
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUTDIR / "nulls.json").write_text(json.dumps(out, indent=1))
    log(f"wrote {OUTDIR / 'nulls.json'}  (wall {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
