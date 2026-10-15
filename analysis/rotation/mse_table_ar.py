#!/usr/bin/env python
"""mse_table_ar.py — per-tensor weight-space MSE table for the 50 patch tensors.

For each of the 50 selected INT4 linears, dequantizes (manual dequant, identical
decoder everywhere) and computes MSE vs the BF16 original (fp16 source) for:
  current : models/Qwen3.8-27B-W4A16-AutoRound            (baseline, no rotation)
  naive-rot : models/Qwen3.8-27B-W4A16-rot-patch          (Hadamard fold, RTN)
  rot-AR  : models/Qwen3.8-27B-W4A16-rot-patch-AR         (Hadamard fold, AutoRound)
MSE is computed in float64 on the full [out,in] matrix (pure weight space; the
rotation's real effect is input-activation-weighted and is covered separately by
the tail-fidelity floor). Writes analysis/rotation/mse_table_ar.json and a
printed summary. Read-only w.r.t. the model dirs.
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp
from safetensors import safe_open

SRC_DIR = bp.REPO / "models"
DIRS = {
    "current": SRC_DIR / "Qwen3.8-27B-W4A16-AutoRound",
    "naive-rot": SRC_DIR / "Qwen3.8-27B-W4A16-rot-patch",
    "rot-AR": SRC_DIR / "Qwen3.8-27B-W4A16-rot-patch-AR",
}
OUT = HERE / "mse_table_ar.json"

_, patch = bp.build_patch_set()
assert len(patch) == 50, len(patch)
bases = sorted(patch)
shards = {}
for d in DIRS.values():
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"] \
        if (d / "model.safetensors.index.json").exists() else None

def shard_of(d, key):
    pk = key[:-len(".weight")] + ".weight_packed"
    if (d / "model.safetensors.index.json").exists():
        return json.loads((d / "model.safetensors.index.json").read_text())["weight_map"][pk]
    for p in sorted(d.glob("*.safetensors")):
        h = bp.read_st_header(p)
        if pk in h[1]:
            return p.name
    raise KeyError(pk)

def load_pair(d, key):
    sh = shard_of(d, key)
    with safe_open(d / sh, framework="pt", device="cpu") as f:
        return f.get_tensor(key + "_packed"), f.get_tensor(key + "_scale")

t0 = time.time()
rows = {}
for b in bases:
    key = b + ".weight"
    W_orig = bp.load_fp16(b).to(torch.float64).numpy() if False else None
    import torch
    W_orig = bp.load_fp16(b)
    W64 = W_orig.to(torch.float64).numpy()
    shape = W64.shape
    row = {"shape": [int(shape[0]), int(shape[1])], "elements": int(shape[0] * shape[1])}
    for name, d in DIRS.items():
        pk, sc = load_pair(d, key)
        deq = bp.manual_dequant(pk, sc, shape).astype(np.float64)
        row[name + "_mse"] = float(((deq - W64) ** 2).mean())
        row[name + "_maxdiff"] = float(np.abs(deq - W64).max())
        del deq
    # relative improvements
    row["naive_vs_current_pct"] = round(100 * (1 - row["naive-rot_mse"] / row["current_mse"]), 3)
    row["ar_vs_current_pct"] = round(100 * (1 - row["rot-AR_mse"] / row["current_mse"]), 3)
    row["ar_vs_naive_pct"] = round(100 * (1 - row["rot-AR_mse"] / row["naive-rot_mse"]), 3)
    rows[b] = row
    print("%-55s cur=%.4e naive=%.4e ar=%.4e | ar_vs_naive %+7.2f%% (%.1fs)" %
          (b.split("layers.")[-1], row["current_mse"], row["naive-rot_mse"], row["rot-AR_mse"],
           row["ar_vs_naive_pct"], time.time() - t0), flush=True)
    del W64
    import gc; gc.collect()

import torch  # noqa: E402 (bp imports torch already)

cur_sum = sum(r["current_mse"] * r["elements"] for r in rows.values())
nai_sum = sum(r["naive-rot_mse"] * r["elements"] for r in rows.values())
ar_sum = sum(r["rot-AR_mse"] * r["elements"] for r in rows.values())
n_el = sum(r["elements"] for r in rows.values())
summary = {
    "n_tensors": len(rows),
    "macro_avg_mse": {"current": cur_sum / n_el, "naive-rot": nai_sum / n_el, "rot-AR": ar_sum / n_el},
    "ar_vs_naive_pct": round(100 * (1 - ar_sum / nai_sum), 3),
    "ar_vs_current_pct": round(100 * (1 - ar_sum / cur_sum), 3),
    "naive_vs_current_pct": round(100 * (1 - nai_sum / cur_sum), 3),
    "n_ar_worse_than_naive": sum(1 for r in rows.values() if r["rot-AR_mse"] > r["naive-rot_mse"]),
    "n_ar_worse_than_current": sum(1 for r in rows.values() if r["rot-AR_mse"] > r["current_mse"]),
}
OUT.write_text(json.dumps({"summary": summary, "per_tensor": rows}, indent=1))
print("\nSUMMARY:", json.dumps(summary, indent=1))
print("mse_table_ar.json written (%.1fs)" % (time.time() - t0))
