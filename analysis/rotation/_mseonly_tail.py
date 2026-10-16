#!/usr/bin/env python
"""_mseonly_tail.py — low-band floor re-measure for the MSE-only selective patch.

Same t / band definitions as tail_fidelity_selective.py (verbatim):
  orig      : BF16 original weight
  current   : baseline W4A16-AutoRound dequant
  selective : models/Qwen3.8-27B-W4A16-rot-patch-selective-mse dequant
t = 0.1 * median(per-row 128-group absmax of the ORIGINAL); same t for all 3
series. Floor measured ON THE KEPT SET (38 rotated tensors) — expected to
regress on many; recorded factually, NOT a gate this round.

Writes analysis/rotation/tail_fidelity_selective_mse.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp

REPO = bp.REPO
SEL_DIR = REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-selective-mse"
OUT = HERE / "tail_fidelity_selective_mse.json"
GROUP = bp.GROUP

sel = json.loads((HERE / "selective_selection_mse.json").read_text())
kept = sel["kept"]
_, patch_set = bp.build_patch_set()
assert set(kept) <= set(patch_set)
inv = json.loads(bp.INVENTORY.read_text())
tname = {x["name"]: x for x in inv["tensors"]}


def load_shard(d, shard, key):
    with safe_open(d / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


tf = {"t_def": "t = 0.1 * median(per-row 128-group absmax of the BF16 ORIGINAL); same t for all 3 series per tensor",
      "band_def": "band_mass = sum(|w| for |w| <= t); retention = band_mass / band_mass[orig]",
      "series": ["orig(BF16)", "current(baseline W4A16 dequant)",
                 "selective(MSE-only selective patch dequant)"],
      "scope": "measured on the KEPT set only (MSE-only round; low-band floor "
               "recorded factually, NOT a criterion)",
      "tensors": {}, "summary": {}}
t0 = time.time()
for b in sorted(kept):
    entry = tname[b + ".weight_packed"]
    shard = entry["file"]
    pk_shape = entry["shape"]
    out, inp = int(pk_shape[0]), int(pk_shape[1]) * 8
    shape = (out, inp)
    G = inp // GROUP
    Wo64 = bp.load_fp16(b).to(torch.float64).numpy().reshape(out, G, GROUP)
    t = 0.1 * float(np.median(np.abs(Wo64).max(axis=2)))
    W = bp.load_fp16(b)
    Wo = W.to(torch.float32).numpy()
    cur = bp.manual_dequant(load_shard(bp.SRC, shard, b + ".weight_packed"),
                            load_shard(bp.SRC, shard, b + ".weight_scale"), shape)
    sl = bp.manual_dequant(load_shard(SEL_DIR, shard, b + ".weight_packed"),
                           load_shard(SEL_DIR, shard, b + ".weight_scale"), shape)

    def metrics(x):
        ax = np.abs(x)
        return int((x == 0.0).sum()), float(ax[ax <= t].sum())

    zo, bo = metrics(Wo)
    zc, bc = metrics(cur)
    zs, bs = metrics(sl)
    tf["tensors"][b] = {
        "t": t,
        "orig": {"zeros": zo, "band_mass": bo},
        "current": {"zeros": zc, "band_mass": bc, "retention": (bc / bo) if bo else None},
        "selective": {"zeros": zs, "band_mass": bs, "retention": (bs / bo) if bo else None},
        "decision": "KEEP",
    }
    r = tf["tensors"][b]
    ok = (zs <= zc) and (r["selective"]["retention"] >= (r["current"]["retention"] or 0))
    print("%-55s t=%.5g zeros c/s=%d/%d retain c/s=%.4f/%.4f -> %s"
          % (b.split("layers.")[-1], t, zc, zs, r["current"]["retention"] or -1,
             r["selective"]["retention"] or -1, "FLOOR_OK" if ok else "regress"),
          flush=True)
    del W, Wo64, cur, sl

beats_zeros = beats_band = floor = 0
floor_tensors = []
for b, d in tf["tensors"].items():
    cz, zs = d["current"]["zeros"], d["selective"]["zeros"]
    cr = d["current"]["retention"] or 0.0
    sr = d["selective"]["retention"] or 0.0
    if zs < cz:
        beats_zeros += 1
    if sr > cr:
        beats_band += 1
    if zs <= cz and sr >= cr:
        floor += 1
        floor_tensors.append(b)
tf["summary"] = {"n_tensors": len(tf["tensors"]),
                 "selective_beats_current_on_zeros": beats_zeros,
                 "selective_beats_current_on_band_retention": beats_band,
                 "floor_satisfied_no_regression_both": floor,
                 "floor_tensors": [b.split("layers.")[-1] for b in floor_tensors],
                 "regressing_tensors": [b.split("layers.")[-1]
                                        for b in sorted(tf["tensors"])
                                        if b not in floor_tensors],
                 "note": "low-band regressions on kept tensors are EXPECTED this "
                         "round (MSE-only criterion) and are recorded, not gated",
                 "n_kept": sel["n_kept"], "n_deselected": sel["n_deselected"]}
OUT.write_text(json.dumps(tf, indent=1))
print("\nSUMMARY:", json.dumps(tf["summary"], indent=1), flush=True)
print("written (%.1fs)" % (time.time() - t0), flush=True)
