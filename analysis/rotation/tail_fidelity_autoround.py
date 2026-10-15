#!/usr/bin/env python
"""tail_fidelity_autoround.py — exact tail-fidelity floor remeasure for the AR patch.

Three series per tensor, same definitions as build_patch.py STEP 4 (verbatim):
  orig    : BF16 original weight (fp16 source, float32)
  current : baseline W4A16-AutoRound dequant (naive rounding, no rotation)
  rot-AR  : models/Qwen3.8-27B-W4A16-rot-patch-AR dequant (fold + AutoRound)
t = 0.1 * median(per-row 128-group absmax of the BF16 ORIGINAL); same t for all 3
series per tensor. zeros = count(x == 0.0); band_mass = sum(|x| for |x| <= t);
retention = band_mass / band_mass[orig]. Floor per tensor: rot-AR zeros <= current
zeros AND rot-AR retention >= current retention.
Writes analysis/rotation/tail_fidelity_autoround.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp

AR_DIR = bp.REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-AR"
OUT = HERE / "tail_fidelity_autoround.json"
GROUP = bp.GROUP

_, patch = bp.build_patch_set()
inv = json.loads(bp.INVENTORY.read_text())
tname = {x["name"]: x for x in inv["tensors"]}

def load_shard(d, shard, key):
    with safe_open(d / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)

tf = {"t_def": "t = 0.1 * median(per-row 128-group absmax of the BF16 ORIGINAL); same t for all 3 series per tensor",
      "band_def": "band_mass = sum(|w| for |w| <= t); retention = band_mass / band_mass[orig]",
      "series": ["orig(BF16)", "current(baseline W4A16 dequant)", "rot-AR(patch-AR dequant)"],
      "tensors": {}, "summary": {}}
t0 = time.time()
for b in sorted(patch):
    entry = tname[b + ".weight_packed"]
    shard = entry["file"]
    pk_shape = entry["shape"]
    out, inp = int(pk_shape[0]), int(pk_shape[1]) * 8
    shape = (out, inp)
    G = inp // GROUP
    # t from the BF16 ORIGINAL (float64, same as quantize_rotated's amax_orig)
    Wo64 = bp.load_fp16(b).to(torch.float64).numpy().reshape(out, G, GROUP)
    t = 0.1 * float(np.median(np.abs(Wo64).max(axis=2)))
    W = bp.load_fp16(b)
    Wo = W.to(torch.float32).numpy()
    cur = bp.manual_dequant(load_shard(bp.SRC, shard, b + ".weight_packed"),
                            load_shard(bp.SRC, shard, b + ".weight_scale"), shape)
    rot = bp.manual_dequant(load_shard(AR_DIR, shard, b + ".weight_packed"),
                            load_shard(AR_DIR, shard, b + ".weight_scale"), shape)
    def metrics(x):
        ax = np.abs(x)
        return int((x == 0.0).sum()), float(ax[ax <= t].sum())
    zo, bo = metrics(Wo)
    zc, bc = metrics(cur)
    zr, br = metrics(rot)
    tf["tensors"][b] = {
        "t": t,
        "orig": {"zeros": zo, "band_mass": bo},
        "current": {"zeros": zc, "band_mass": bc, "retention": (bc / bo) if bo else None},
        "rot-AR": {"zeros": zr, "band_mass": br, "retention": (br / bo) if bo else None},
    }
    r = tf["tensors"][b]
    ok = (zr <= zc) and (r["rot-AR"]["retention"] >= (r["current"]["retention"] or 0))
    print("%-55s t=%.5g zeros c/r=%d/%d retain c/r=%.4f/%.4f -> %s"
          % (b.split("layers.")[-1], t, zc, zr, r["current"]["retention"] or -1,
             r["rot-AR"]["retention"] or -1, "FLOOR_OK" if ok else "regress"),
          flush=True)
    del W, Wo64, cur, rot

beats_zeros = beats_band = floor = 0
for b, d in tf["tensors"].items():
    cz, rz = d["current"]["zeros"], d["rot-AR"]["zeros"]
    cr = d["current"]["retention"] or 0.0
    rr = d["rot-AR"]["retention"] or 0.0
    if rz < cz: beats_zeros += 1
    if rr > cr: beats_band += 1
    if rz <= cz and rr >= cr: floor += 1
tf["summary"] = {"n_tensors": len(tf["tensors"]),
                 "ar_beats_current_on_zeros": beats_zeros,
                 "ar_beats_current_on_band_retention": beats_band,
                 "floor_satisfied_no_regression_both": floor,
                 "floor_tensors": [b.split("layers.")[-1] for b, d in tf["tensors"].items()
                                   if d["rot-AR"]["zeros"] <= d["current"]["zeros"]
                                   and (d["rot-AR"]["retention"] or 0) >= (d["current"]["retention"] or 0)]}
OUT.write_text(json.dumps(tf, indent=1))
print("\nSUMMARY:", json.dumps(tf["summary"], indent=1))
print("tail_fidelity_autoround.json written (%.1fs)" % (time.time() - t0))
