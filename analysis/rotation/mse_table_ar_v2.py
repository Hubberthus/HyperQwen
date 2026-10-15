#!/usr/bin/env python
"""mse_table_ar_v2.py — corrected per-tensor effective-weight MSE table.

Metric (parent-verified): for the 38 ROTATED tensors the stored weight is W@H, so the
effective-weight error is deq vs the folded target W@H — because
|| deq(W@H) @ H - W ||^2 = || deq(W@H) - W@H ||^2 (H orthonormal => Frobenius-norm
preserving). Comparing deq vs unrotated W (v1) was dominated by the representation
change itself and is meaningless for those. For the 12 PLAIN in_proj tensors
(unrotated, valid as-is) the target is W itself; both build options are reported:
  option A: keep AR plain in_proj weights (as spliced)  -> mse = deq_AR vs W
  option B: keep CURRENT unpatched in_proj weights      -> mse = deq_current vs W
Writes analysis/rotation/mse_table_ar.json (v2 overwrites the v1 table).
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp

AR_DIR = bp.REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-AR"
NAIVE_DIR = bp.REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch"
SRC_DIR = bp.SRC
PLAIN_TSUF = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")
OUT = HERE / "mse_table_ar.json"

H128 = bp.sylvester(128)
_, patch = bp.build_patch_set()
assert len(patch) == 50
inv = json.loads(bp.INVENTORY.read_text())
tname = {x["name"]: x for x in inv["tensors"]}

def load_pair(d, b):
    entry = tname[b + ".weight_packed"]
    shard = entry["file"]
    with safe_open(d / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(b + ".weight_packed"), f.get_tensor(b + ".weight_scale")

t0 = time.time()
rows = {}
for b in sorted(patch):
    tsuf = b.split("layers.")[1].split(".", 1)[1]
    plain = tsuf in PLAIN_TSUF
    entry = tname[b + ".weight_packed"]
    shard = entry["file"]
    pk_shape = entry["shape"]
    out, inp = int(pk_shape[0]), int(pk_shape[1]) * 8
    G = inp // bp.GROUP
    shape = (out, inp)
    W64 = bp.load_fp16(b).to(torch.float64).numpy()
    if plain:
        target = W64
    else:
        target = bp.block_apply(W64, H128)  # W @ H folded target (float64)
    row = {"shape": [out, inp], "elements": out * inp, "rotated": (not plain),
           "mse_target": "W@H (folded)" if not plain else "W (plain)"}
    for name, d in [("current", SRC_DIR), ("naive-rot", NAIVE_DIR), ("rot-AR", AR_DIR)]:
        pk, sc = load_pair(d, b)
        deq = bp.manual_dequant(pk, sc, shape).astype(np.float64)
        row[name + "_mse"] = float(((deq - target) ** 2).mean())
        row[name + "_maxdiff"] = float(np.abs(deq - target).max())
        del deq
    if plain:
        # option B keeps the CURRENT (unpatched) weight: its error is exactly
        # current_mse; rot-AR / naive-rot are candidate replacements (vs W).
        row["optionA_mse"] = row["rot-AR_mse"]      # keep AR plain
        row["optionB_mse"] = row["current_mse"]     # keep current (restore baseline bytes)
        row["optionA_vs_B_pct"] = round(100 * (row["optionA_mse"] / row["optionB_mse"] - 1), 2)
        row["ar_vs_current_pct"] = round(100 * (1 - row["rot-AR_mse"] / row["current_mse"]), 2)
    else:
        row["ar_vs_naive_pct"] = round(100 * (1 - row["rot-AR_mse"] / row["naive-rot_mse"]), 3)
        row["ar_vs_current_equiv_pct"] = round(100 * (1 - row["rot-AR_mse"] / row["current_mse"]), 3)
    rows[b] = row
    tag = "plain" if plain else "rot  "
    print("%s %-50s cur=%.4e naive=%.4e ar=%.4e %s"
          % (tag, b.split("layers.")[-1], row["current_mse"], row["naive-rot_mse"],
             row["rot-AR_mse"],
             ("optA_vs_B %+.1f%%" % row.get("optionA_vs_B_pct", 0)) if plain
             else ("ar_vs_naive %+6.2f%%" % row.get("ar_vs_naive_pct", 0))), flush=True)
    del W64, target
    gc.collect()

def agg(key, pred):
    n = sum(rows[b]["elements"] for b in rows if pred(b))
    return sum(rows[b][key] * rows[b]["elements"] for b in rows if pred(b)) / n

rot_b = [b for b in rows if rows[b]["rotated"]]
plain_b = [b for b in rows if not rows[b]["rotated"]]
summary = {
    "n_tensors": 50, "n_rotated": len(rot_b), "n_plain": len(plain_b),
    "metric_rotated": "deq vs W@H folded target (effective-weight; H orthonormal preserves Frobenius norm)",
    "metric_plain": "deq vs W (both options reported)",
    "rotated_macro": {"current_mse": agg("current_mse", lambda b: rows[b]["rotated"]),
                      "naive-rot_mse": agg("naive-rot_mse", lambda b: rows[b]["rotated"]),
                      "rot-AR_mse": agg("rot-AR_mse", lambda b: rows[b]["rotated"])},
    "plain_optionA_macro": {"rot-AR_mse": agg("rot-AR_mse", lambda b: not rows[b]["rotated"])},
    "plain_optionB_macro": {"current_mse": agg("current_mse", lambda b: not rows[b]["rotated"])},
    "rotated_ar_vs_naive_pct": round(100 * (1 - agg("rot-AR_mse", lambda b: rows[b]["rotated"]) /
                                            agg("naive-rot_mse", lambda b: rows[b]["rotated"])), 3),
    "rotated_ar_vs_current_equiv_pct": round(100 * (1 - agg("rot-AR_mse", lambda b: rows[b]["rotated"]) /
                                                    agg("current_mse", lambda b: rows[b]["rotated"])), 3),
    "plain_optionA_vs_optionB_pct": round(100 * (agg("rot-AR_mse", lambda b: not rows[b]["rotated"]) /
                                                 agg("current_mse", lambda b: not rows[b]["rotated"]) - 1), 2),
    "n_plain_optionA_regressed": sum(1 for b in plain_b if rows[b]["rot-AR_mse"] > rows[b]["current_mse"]),
    "n_rotated_ar_better_than_naive": sum(1 for b in rot_b if rows[b]["rot-AR_mse"] < rows[b]["naive-rot_mse"]),
    "n_rotated_ar_worse_than_naive": sum(1 for b in rot_b if rows[b]["rot-AR_mse"] > rows[b]["naive-rot_mse"]),
}
OUT.write_text(json.dumps({"summary": summary, "per_tensor": rows}, indent=1))
print("\nSUMMARY:", json.dumps(summary, indent=1))
print("mse_table_ar.json (v2) written (%.1fs)" % (time.time() - t0))
