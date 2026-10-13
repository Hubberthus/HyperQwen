"""Weight-layout plots: original BF16 vs dequantized W4A16 (salted spike follow-up).

Question: the salted-codebook null (eta2 <= 4.3e-4 on both baselines) measured
between-class MEAN variance only. A position-dependent codebook can also exploit
between-class SCALE/SHAPE differences, which appear as position-dependent variance.
This script plots the value distribution and the positional layout of the weights
to find any pattern worth exploiting.

Sources:
  - original:  models-fp16/Qwen3.8-27B (BF16; dtype verified from safetensors
    headers, not config prose)
  - hyperqwen: models/Qwen3.8-27B-W4A16-AutoRound, dequantized (INT4/INT8, group-128)

Dequant path is the manual tiled unpack+dequant from heterogeneity.py, pinned
against compressed_tensors (unpack_from_int32 + lifecycle.forward.dequantize);
the pin result is persisted to weight_layout.json and the run aborts if != 0.

Null-is-guilty checks: per tensor/source std, % positive, percentiles (weights
must look like weights); dequant grid check wd/scale on the exact signed integer
grid {-8..7} (INT4) / {-128..127} (INT8) on every tile (float32 round-trip
tolerance 1e-3); original dtype asserted BF16 from the header.

Note: cols = 5120 or 17408, both divisible by 8/64/128, so flat%N == col%N for
every residue probe below — "linear" residue classes are column classes.

Outputs: analysis/salted/plots/*.png + analysis/salted/weight_layout.json
Run:  venv/bin/python analysis/salted/weight_layout.py
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
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent
SALTED = REPO / "analysis" / "salted"
sys.path.insert(0, str(SALTED))
sys.path.insert(0, str(SALTED / "fp16"))

import heterogeneity as hz   # W4A16 dequant: load_packed, dequant_chunk, verify_pin
import fp16_common as fc    # original BF16 chunked reader

OUT = SALTED / "plots"
CHUNK_ROWS = 4096
GROUP = 128

VALUE_LO, VALUE_HI, VALUE_BINS = -0.2, 0.2, 4000
ABS_LO, ABS_HI, ABS_BINS = 1e-6, 1.0, 200   # log-spaced bins
HEX_POINTS = 1_000_000
PCT_POINTS = 1_000_000
RES_NS = (8, 64, 128)

# (tkey, base, shard, num_bits) — identical sample list to Task 2/3
SAMPLES = [
    ("L0.gate",  "model.language_model.layers.0.mlp.gate_proj",    "model-00001-of-00007.safetensors", 4),
    ("L0.down",  "model.language_model.layers.0.mlp.down_proj",    "model-00001-of-00007.safetensors", 4),
    ("L27.q",    "model.language_model.layers.27.self_attn.q_proj", "model-00002-of-00007.safetensors", 4),
    ("L27.gate", "model.language_model.layers.27.mlp.gate_proj",    "model-00002-of-00007.safetensors", 4),
    ("L27.down", "model.language_model.layers.27.mlp.down_proj",    "model-00002-of-00007.safetensors", 4),
    ("L63.q",    "model.language_model.layers.63.self_attn.q_proj", "model-00004-of-00007.safetensors", 4),
    ("L63.gate", "model.language_model.layers.63.mlp.gate_proj",    "model-00004-of-00007.safetensors", 4),
    ("L63.down", "model.language_model.layers.63.mlp.down_proj",    "model-00004-of-00007.safetensors", 4),
    ("lm_head",  "lm_head",                                         "model-00007-of-00007.safetensors", 8),
]
DEEP = ("L27.gate", "lm_head")
SLICE = (4096, 4608, 1024, 2048)  # r0, r1, c0, c1

ABS_EDGES = np.logspace(np.log10(ABS_LO), np.log10(ABS_HI), ABS_BINS + 1)
V_EDGES = np.linspace(VALUE_LO, VALUE_HI, VALUE_BINS + 1)


class SourceStats:
    """Exact streaming moments + fixed-bin histograms. Never per-chunk averages."""

    def __init__(self, rows, cols):
        assert cols % max(RES_NS) == 0, "residue probes assume cols % N == 0"
        self.rows, self.cols = rows, cols
        self.n = 0
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.pos = 0
        self.vhist = np.zeros(VALUE_BINS, dtype=np.int64)
        self.v_under = self.v_over = 0
        self.ahist = np.zeros(ABS_BINS, dtype=np.int64)
        self.a_under = self.a_over = 0
        self.row_s = np.zeros(rows, dtype=np.float64)
        self.row_ss = np.zeros(rows, dtype=np.float64)
        self.col_s = np.zeros(cols, dtype=np.float64)
        self.col_ss = np.zeros(cols, dtype=np.float64)
        self.res = {N: [np.zeros(N, dtype=np.float64),
                        np.zeros(N, dtype=np.float64)] for N in RES_NS}
        self.colcls = {N: np.arange(cols, dtype=np.int64) % N for N in RES_NS}
        ngroups = rows * (cols // GROUP)
        self.group_e = np.zeros(ngroups, dtype=np.float64)
        self.group_n = np.zeros(ngroups, dtype=np.int64)
        total = rows * cols
        self.hex_stride = max(1, total // HEX_POINTS)
        self.pct_stride = max(1, total // PCT_POINTS)
        self.hex_v, self.hex_p, self.pct_v = [], [], []

    def add_chunk(self, w, row0):
        r, c = w.shape
        w64 = w.astype(np.float64)
        self.n += w.size
        self.s1 += w64.sum()
        self.s2 += (w64 * w64).sum()
        self.s3 += (w64 ** 3).sum()
        self.s4 += (w64 ** 4).sum()
        self.pos += int((w > 0).sum())
        h, _ = np.histogram(w, bins=VALUE_BINS, range=(VALUE_LO, VALUE_HI))
        self.vhist += h
        self.v_under += int((w < VALUE_LO).sum())
        self.v_over += int((w > VALUE_HI).sum())
        aw = np.abs(w)
        h2, _ = np.histogram(aw, bins=ABS_EDGES)
        self.ahist += h2
        self.a_under += int((aw < ABS_LO).sum())
        self.a_over += int((aw > ABS_HI).sum())
        # per-row / per-col exact sums (chunks are row-aligned, rows never split)
        self.row_s[row0:row0 + r] += w64.sum(1)
        self.row_ss[row0:row0 + r] += (w64 * w64).sum(1)
        cs = w64.sum(0)
        css = (w64 * w64).sum(0)
        self.col_s += cs
        self.col_ss += css
        # group energy (groups never cross rows: cols % GROUP == 0)
        ge = (w64 * w64).reshape(r, c // GROUP, GROUP).mean(-1)
        g0 = row0 * (c // GROUP)
        self.group_e[g0:g0 + r * (c // GROUP)] += ge.ravel()
        self.group_n[g0:g0 + r * (c // GROUP)] += GROUP
        # residue-class sums: flat%N == col%N, so bincount over the col sums
        for N in RES_NS:
            self.res[N][0] += np.bincount(self.colcls[N], weights=cs, minlength=N)
            self.res[N][1] += np.bincount(self.colcls[N], weights=css, minlength=N)
        # deterministic subsamples for hexbin + percentiles
        v = w.ravel()
        idx = np.arange(0, v.size, self.hex_stride)
        self.hex_v.append(v[idx].astype(np.float32))
        self.hex_p.append((row0 * c + idx).astype(np.int64))
        self.pct_v.append(v[:: self.pct_stride].astype(np.float32))

    def summary(self):
        n = self.n
        mean = self.s1 / n
        var = self.s2 / n - mean * mean
        sd = var ** 0.5
        m3 = self.s3 / n - 3 * mean * self.s2 / n + 2 * mean ** 3
        m4 = (self.s4 / n - 4 * mean * self.s3 / n
              + 6 * mean * mean * self.s2 / n - 3 * mean ** 4)
        row_mean = self.row_s / self.cols
        row_e = self.row_ss / self.cols                       # mean |w|^2 per row
        row_std = np.sqrt(np.maximum(self.row_ss / self.cols - row_mean ** 2, 0))
        order = np.argsort(row_e)[::-1]
        top1 = float(row_e[order[: max(1, self.rows // 100)]].sum() / row_e.sum())
        col_mean = self.col_s / self.rows
        col_std = np.sqrt(np.maximum(self.col_ss / self.rows - col_mean ** 2, 0))
        pv = np.concatenate(self.pct_v)
        p1, p50, p99, p999 = (float(x) for x in np.percentile(pv, [1, 50, 99, 99.9]))
        ge = self.group_e / self.group_n
        res = {}
        for N in RES_NS:
            ms = self.res[N][1] / (n // N)                     # mean square per class
            res[f"res{N}_dev"] = float((ms.max() - ms.min()) / ms.mean())
            res[f"res{N}_profile"] = ms.tolist()
        stats = {
            "n": int(n), "mean": float(mean), "std": float(sd),
            "skew": float(m3 / sd ** 3), "excess_kurtosis": float(m4 / sd ** 4 - 3.0),
            "pct_positive": float(self.pos / n),
            "p1": p1, "p50": p50, "p99": p99, "p99.9": p999,
            "absmax": float(np.abs(pv).max()),
            "row_std_min": float(row_std.min()), "row_std_med": float(np.median(row_std)),
            "row_std_max": float(row_std.max()),
            "row_energy_top1pct_share": top1,
            "col_std_min": float(col_std.min()), "col_std_med": float(np.median(col_std)),
            "col_std_max": float(col_std.max()),
            "group_energy_mean": float(ge.mean()), "group_energy_min": float(ge.min()),
            "group_energy_max": float(ge.max()),
            "group_energy_max_over_mean": float(ge.max() / ge.mean()),
            "vhist": self.vhist.tolist(), "v_under": int(self.v_under),
            "v_over": int(self.v_over),
            "ahist": self.ahist.tolist(), "a_under": int(self.a_under),
            "a_over": int(self.a_over),
            **res,
        }
        return stats, row_std, col_std, ge, row_e


def process_tensor(tkey, base, shard, bits):
    torch.set_num_threads(1)
    t0 = time.time()
    packed, scale, shape = hz.load_packed(shard, base)
    rows, cols = shape
    gs = cols // scale.shape[1]
    assert gs == GROUP, f"{tkey}: group size {gs} != {GROUP}"
    sc = scale.to(torch.float32).repeat_interleave(gs, dim=1).numpy()
    K = 1 << bits
    off = K // 2

    so = SourceStats(rows, cols)
    sd = SourceStats(rows, cols)
    code_hist = np.zeros(K, dtype=np.int64)
    code_lin64 = np.zeros((64, K), dtype=np.int64)
    code_row64 = np.zeros((64, K), dtype=np.int64)
    x_od = 0.0            # sum w_orig * w_deq (cross-source correlation)
    grid_ok = True
    dtype_orig = fc.tensor_meta(base)[1]

    it = fc.tensor_chunks(base)
    for r0 in range(0, rows, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, rows)
        wo, r0o = next(it)
        assert r0o == r0 and wo.shape[0] == r1 - r0
        up = hz.dequant_chunk(packed[r0:r1], bits, cols)
        wd = (up - off).astype(np.float32) * sc[r0:r1]
        # grid check: wd/scale must sit on the signed integer grid (float32
        # round-trip tolerance 1e-3; a wrong unpack is off by >= 1 code level)
        q = wd / sc[r0:r1]
        rq = np.round(q)
        if not (np.all(np.abs(q - rq) < 1e-3)
                and rq.min() >= -off and rq.max() <= off - 1):
            grid_ok = False
        code_hist += np.bincount(up.ravel(), minlength=K)
        for c64 in range(64):  # flat%64 == col%64 (cols divisible by 64)
            code_lin64[c64] += np.bincount(up[:, c64::64].ravel(), minlength=K)
        for c64 in range(64):  # row%64 (rows and CHUNK_ROWS divisible by 64)
            code_row64[c64] += np.bincount(up[c64::64].ravel(), minlength=K)
        so.add_chunk(wo, r0)
        sd.add_chunk(wd, r0)
        a = wo.astype(np.float64).ravel()
        b = wd.astype(np.float64).ravel()
        x_od += float((a * b).sum())
    assert next(it, None) is None, f"{tkey}: original iterator had extra chunks"

    so_sum, row_std_o, col_std_o, ge_o, row_e_o = so.summary()
    sd_sum, row_std_d, col_std_d, ge_d, row_e_d = sd.summary()
    n = so.n
    cov = x_od / n - so_sum["mean"] * sd_sum["mean"]
    corr = cov / (so_sum["std"] * sd_sum["std"])

    # scale stats + scale/energy correlation (dequant side)
    scf = scale.to(torch.float32).numpy()          # [rows, cols//GROUP]
    scale_stats = {
        "min": float(scf.min()), "p50": float(np.median(scf)),
        "p99": float(np.percentile(scf, 99)), "max": float(scf.max()),
        "frac_negative": float((scf < 0).mean()),
        "corr_scale_energy": float(np.corrcoef(scf.ravel(), ge_d)[0, 1]),
        "corr_absscale_energy": float(np.corrcoef(np.abs(scf).ravel(), ge_d)[0, 1]),
    }
    p = code_hist / code_hist.sum()
    code_entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    out = {
        "tkey": tkey, "base": base, "bits": bits, "shape": [rows, cols],
        "dtype_orig": dtype_orig, "grid_ok": bool(grid_ok),
        "corr_orig_dequant": float(corr),
        "orig": so_sum, "dequant": sd_sum,
        "row_std_orig": row_std_o, "row_std_dequant": row_std_d,
        "col_std_orig": col_std_o, "col_std_dequant": col_std_d,
        "row_energy_orig": row_e_o, "row_energy_dequant": row_e_d,
        "group_energy_orig": ge_o, "group_energy_dequant": ge_d,
        "code_hist": code_hist, "code_lin64": code_lin64, "code_row64": code_row64,
        "code_entropy_bits": code_entropy, "code_max_bits": float(np.log2(K)),
        "scale_stats": scale_stats,
        "hex_orig": (np.concatenate(so.hex_v), np.concatenate(so.hex_p)),
        "hex_dequant": (np.concatenate(sd.hex_v), np.concatenate(sd.hex_p)),
        "runtime_s": round(time.time() - t0, 1),
    }
    # subsampled scale/energy pairs for the scatter plot
    sub = np.arange(0, scf.size, max(1, scf.size // 200_000))
    out["scale_sample"] = scf.ravel()[sub]
    out["ge_sample"] = ge_d[sub]
    if tkey in DEEP:
        r0, r1, c0, c1 = SLICE
        up = hz.dequant_chunk(packed[r0:r1], bits, cols)
        out["slice_dequant"] = ((up - off).astype(np.float32) * sc[r0:r1])[:, c0:c1]
        with safe_open(fc.CKPT_FP16 / fc.shard_for(base), framework="pt", device="cpu") as f:
            sl = f.get_slice(base + ".weight")
            out["slice_orig"] = sl[r0:r1].to(torch.float32).numpy()[:, c0:c1]
    return out


# ------------------------------------------------------------------ plotting
C_O, C_D = "#1f77b4", "#ff7f0e"


def _dens(counts, edges, n):
    return counts / (n * np.diff(edges))


def plot_all(results):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    deep = [r for r in results if r["tkey"] in DEEP]

    # ---- 01 overall value distribution (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, r in zip(axes.ravel(), results):
        no, nd = r["orig"]["n"], r["dequant"]["n"]
        ax.stairs(_dens(np.array(r["orig"]["vhist"]), V_EDGES, no), V_EDGES,
                  fill=False, color=C_O, lw=1.2,
                  label=f"original (std {r['orig']['std']:.4f})")
        ax.stairs(_dens(np.array(r["dequant"]["vhist"]), V_EDGES, nd), V_EDGES,
                  fill=False, color=C_D, lw=1.2,
                  label=f"dequant (std {r['dequant']['std']:.4f})")
        ax.set_yscale("log")
        ax.set_title(r["tkey"], fontsize=10)
    axes[0, 0].set_ylabel("density (log)")
    axes[2, 1].set_xlabel("weight value")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Weight-value distribution — original BF16 vs dequantized W4A16 (9 tensors)")
    fig.tight_layout()
    fig.savefig(OUT / "01_overall_value_dist.png", dpi=100)
    plt.close(fig)

    # ---- 02 |w| tail on log-x (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, r in zip(axes.ravel(), results):
        no, nd = r["orig"]["n"], r["dequant"]["n"]
        ax.stairs(_dens(np.array(r["orig"]["ahist"]), ABS_EDGES, no), ABS_EDGES,
                  fill=False, color=C_O, lw=1.2, label="original")
        ax.stairs(_dens(np.array(r["dequant"]["ahist"]), ABS_EDGES, nd), ABS_EDGES,
                  fill=False, color=C_D, lw=1.2, label="dequant")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(r["tkey"], fontsize=10)
    axes[0, 0].set_ylabel("density (log)")
    axes[2, 1].set_xlabel("abs weight (log)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Absolute-weight tail — original vs dequantized (heavy-hitter structure)")
    fig.tight_layout()
    fig.savefig(OUT / "02_abs_tail.png", dpi=100)
    plt.close(fig)

    # ---- 03 raw-value slice heatmaps (deep tensors)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for i, r in enumerate(deep):
        so_, sd_ = r["slice_orig"], r["slice_dequant"]
        vlim = float(max(np.percentile(np.abs(so_), 99.9),
                        np.percentile(np.abs(sd_), 99.9)))
        for j, (mat, name) in enumerate(((so_, "original BF16"),
                                         (sd_, "dequantized W4A16"))):
            ax = axes[i, j]
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                           interpolation="nearest")
            ax.set_title(f"{r['tkey']} — {name}  "
                         f"rows {SLICE[0]}:{SLICE[1]}, cols {SLICE[2]}:{SLICE[3]}")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Raw weight values (slice) — spatial layout")
    fig.tight_layout()
    fig.savefig(OUT / "03_slice_heatmaps.png", dpi=100)
    plt.close(fig)

    # ---- 04 per-row / per-col std profiles (deep tensors)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for i, r in enumerate(deep):
        rows, cols = r["shape"]
        step = max(1, rows // 4000)
        ax = axes[i, 0]
        ax.plot(np.arange(0, rows, step), r["row_std_orig"][::step], color=C_O,
                lw=0.8, label=f"original (min/med/max {r['orig']['row_std_min']:.4f}/"
                              f"{r['orig']['row_std_med']:.4f}/{r['orig']['row_std_max']:.4f})")
        ax.plot(np.arange(0, rows, step), r["row_std_dequant"][::step], color=C_D,
                lw=0.8, label=f"dequant (min/med/max {r['dequant']['row_std_min']:.4f}/"
                              f"{r['dequant']['row_std_med']:.4f}/{r['dequant']['row_std_max']:.4f})")
        ax.set_title(f"{r['tkey']} — per-row std (row = output channel)")
        ax.legend(fontsize=7)
        ax.set_xlabel("row")
        ax = axes[i, 1]
        ax.plot(np.arange(cols), r["col_std_orig"], color=C_O, lw=0.8,
                label=f"original (min/med/max {r['orig']['col_std_min']:.4f}/"
                      f"{r['orig']['col_std_med']:.4f}/{r['orig']['col_std_max']:.4f})")
        ax.plot(np.arange(cols), r["col_std_dequant"], color=C_D, lw=0.8,
                label=f"dequant (min/med/max {r['dequant']['col_std_min']:.4f}/"
                      f"{r['dequant']['col_std_med']:.4f}/{r['dequant']['col_std_max']:.4f})")
        ax.set_title(f"{r['tkey']} — per-col std (col = input channel)")
        ax.legend(fontsize=7)
        ax.set_xlabel("col")
    fig.suptitle("Positions with higher energy — per-row / per-col std profiles")
    fig.tight_layout()
    fig.savefig(OUT / "04_rowcol_profiles.png", dpi=100)
    plt.close(fig)

    # ---- 05 sorted per-row energy, all tensors (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, r in zip(axes.ravel(), results):
        rows = r["shape"][0]
        rank = np.arange(1, rows + 1)
        for src, color in (("orig", C_O), ("dequant", C_D)):
            re_ = np.sort(r[f"row_energy_{src}"])[::-1]
            ax.plot(rank, re_, color=color, lw=1.0,
                    label=f"{src} (top-1% rows hold {r[src]['row_energy_top1pct_share']:.3f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(r["tkey"], fontsize=10)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("mean |w|^2 per row (log)")
    axes[2, 1].set_xlabel("row rank, sorted desc (log)")
    fig.suptitle("Per-row energy concentration — sorted profiles (heavy-row structure)")
    fig.tight_layout()
    fig.savefig(OUT / "05_row_energy_sorted.png", dpi=100)
    plt.close(fig)

    # ---- 06 group structure (deep tensors): trace, residue profile, scale scatter
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, r in enumerate(deep):
        ax = axes[i, 0]
        for src, color in (("orig", C_O), ("dequant", C_D)):
            ge = r[f"group_energy_{src}"]
            step = max(1, ge.size // 50_000)
            ax.plot(np.arange(0, ge.size, step), ge[::step], color=color, lw=0.7,
                    label=src)
        ax.set_title(f"{r['tkey']} — mean |w|^2 per 128-group (trace)")
        ax.legend(fontsize=7)
        ax.set_xlabel("group index (row-major)")
        ax = axes[i, 1]
        x = np.arange(GROUP)
        for src, color in (("orig", C_O), ("dequant", C_D)):
            prof = np.array(r[src]["res128_profile"])
            ax.plot(x, prof, color=color, lw=1.0,
                    label=f"{src} (max-min dev {r[src]['res128_dev']:.2e})")
        ax.set_title(f"{r['tkey']} — mean |w|^2 per col%128 class (periodicity probe)")
        ax.legend(fontsize=7)
        ax.set_xlabel("position in group (col % 128)")
        ax = axes[i, 2]
        ax.scatter(np.abs(r["scale_sample"]), r["ge_sample"], s=2, alpha=0.3, color=C_D)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ss = r["scale_stats"]
        ax.set_title(f"{r['tkey']} — |scale| vs group energy (dequant)\n"
                     f"corr(|s|,E)={ss['corr_absscale_energy']:.4f}  "
                     f"frac neg scales {ss['frac_negative']:.3f}")
        ax.set_xlabel("|group scale| (log)")
        ax.set_ylabel("group mean |w|^2 (log)")
    fig.suptitle("Group-128 structure — energy trace, within-group periodicity, scale vs energy")
    fig.tight_layout()
    fig.savefig(OUT / "06_group_structure.png", dpi=100)
    plt.close(fig)

    # ---- 07 code usage (deep tensors)
    gate = next(r for r in results if r["tkey"] == "L27.gate")
    lm = next(r for r in results if r["tkey"] == "lm_head")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.bar(np.arange(16) - 8, gate["code_hist"], color=C_D)
    ax.set_yscale("log")
    ax.set_title(f"L27.gate INT4 code usage (entropy {gate['code_entropy_bits']:.3f}"
                 f" / {gate['code_max_bits']:.0f} bits)")
    ax.set_xlabel("signed code")
    ax = axes[0, 1]
    ax.bar(np.arange(256) - 128, lm["code_hist"], color=C_D)
    ax.set_yscale("log")
    ax.set_title(f"lm_head INT8 code usage (entropy {lm['code_entropy_bits']:.3f}"
                 f" / {lm['code_max_bits']:.0f} bits)")
    ax.set_xlabel("signed code")
    for ax, r in ((axes[1, 0], gate), (axes[1, 1], lm)):
        m = r["code_lin64"].astype(np.float64)          # [64 classes, K codes]
        K = m.shape[1]
        if K > 16:                                       # bin 256 codes -> 16 bins
            m = m.reshape(64, 16, K // 16).sum(-1)
        mn = m / m.sum(1, keepdims=True)                 # per-class normalized
        im = ax.imshow(mn.T, aspect="auto", cmap="viridis", vmin=0)
        ax.set_title(f"{r['tkey']} — code usage by col%64 class (per-class normalized)")
        ax.set_xlabel("col % 64 class")
        ax.set_ylabel("code bin (low -> high)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Quantization code usage — global and per position class")
    fig.tight_layout()
    fig.savefig(OUT / "07_code_usage.png", dpi=100)
    plt.close(fig)

    # ---- 08 value vs flattened position (hexbin, deep tensors)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for i, r in enumerate(deep):
        for j, src in enumerate(("orig", "dequant")):
            v, p = r[f"hex_{src}"]
            ax = axes[i, j]
            hb = ax.hexbin(p, v, gridsize=120, mincnt=1, bins="log", cmap="magma")
            ax.set_title(f"{r['tkey']} — {src}: weight value vs flattened position")
            ax.set_xlabel("flattened position")
            ax.set_ylabel("w")
            ax.set_ylim(-0.1, 0.1)
            fig.colorbar(hb, ax=ax, fraction=0.046)
    fig.suptitle("Density of weight values across positions (log-count hexbin)")
    fig.tight_layout()
    fig.savefig(OUT / "08_value_vs_position.png", dpi=100)
    plt.close(fig)


def main():
    t0 = time.time()
    print("== Dequant pin check (manual tiled vs compressed-tensors library) ==")
    pin = hz.verify_pin("model.language_model.layers.0.mlp.gate_proj",
                        "model-00001-of-00007.safetensors", 4)
    if pin["max_abs_diff"] != 0.0:
        print("PIN FAILED - aborting")
        sys.exit(1)

    workers = min(len(SAMPLES), max(2, (os.cpu_count() or 4) - 2))
    print(f"== Streaming {len(SAMPLES)} tensors x 2 sources ({workers} workers) ==")
    results = []
    ctx = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    futs = {ctx.submit(process_tensor, *s): s[0] for s in SAMPLES}
    for fut in as_completed(futs):
        r = fut.result()
        results.append(r)
        print(f"  {r['tkey']:9s} done in {r['runtime_s']:6.1f}s  "
              f"orig std {r['orig']['std']:.5f} / deq std {r['dequant']['std']:.5f}  "
              f"corr {r['corr_orig_dequant']:.6f}  grid_ok {r['grid_ok']}  "
              f"dtype {r['dtype_orig']}")
        sys.stdout.flush()
    ctx.shutdown()
    order = {s[0]: i for i, s in enumerate(SAMPLES)}
    results.sort(key=lambda r: order[r["tkey"]])

    bad = [r["tkey"] for r in results if not r["grid_ok"] or r["dtype_orig"] != "BF16"]
    if bad:
        print(f"SANITY FAILED (grid/dtype) on {bad} - aborting before plotting")
        sys.exit(1)

    plot_all(results)

    skip = ("hex_orig", "hex_dequant", "slice_orig", "slice_dequant",
            "row_std_orig", "row_std_dequant", "col_std_orig", "col_std_dequant",
            "row_energy_orig", "row_energy_dequant",
            "group_energy_orig", "group_energy_dequant",
            "scale_sample", "ge_sample")

    def _jsonable(v):
        return v.tolist() if isinstance(v, np.ndarray) else v

    payload = {
        "pin": pin,
        "slice": list(SLICE),
        "value_hist_range": [VALUE_LO, VALUE_HI, VALUE_BINS],
        "abs_hist_range": [ABS_LO, ABS_HI, ABS_BINS],
        "tensors": {r["tkey"]: {k: _jsonable(v) for k, v in r.items() if k not in skip}
                    for r in results},
        "runtime_s": round(time.time() - t0, 1),
    }
    (SALTED / "weight_layout.json").write_text(json.dumps(payload, indent=1))
    print(f"\nWrote {OUT}/*.png and analysis/salted/weight_layout.json "
          f"(wall {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
