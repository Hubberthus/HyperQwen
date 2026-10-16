#!/usr/bin/env python
"""tail_fidelity_selective.py — floor re-measure for the selective patch.

Three series per tensor, identical t and band definitions as
tail_fidelity_autoround.py (verbatim):
  orig      : BF16 original weight (fp16 source, float32)
  current   : baseline W4A16-AutoRound dequant (naive rounding, no rotation)
  selective : models/Qwen3.8-27B-W4A16-rot-patch-selective dequant
t = 0.1 * median(per-row 128-group absmax of the BF16 ORIGINAL); same t for all
3 series per tensor. zeros = count(x == 0.0); band_mass = sum(|x| for |x| <= t);
retention = band_mass / band_mass[orig]. Floor per tensor: selective zeros <=
current zeros AND selective retention >= current retention.

Also: sha256 sweep of the selective dir vs rot-patch-AR (byte-identity for
untouched files).

Writes analysis/rotation/tail_fidelity_selective.json.
CPU-only.
"""
import hashlib
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

AR_DIR = bp.REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-AR"
SEL_DIR = bp.REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-selective"
OUT = HERE / "tail_fidelity_selective.json"
GROUP = bp.GROUP

sel = json.loads((HERE / "selective_selection.json").read_text())
_, patch_set = bp.build_patch_set()
assert set(patch_set) == set(sel["per_tensor"]) | set(
    k for k in json.loads((HERE / "mse_table_ar.json").read_text())["per_tensor"]
    if not json.loads((HERE / "mse_table_ar.json").read_text())["per_tensor"][k].get("rotated"))
inv = json.loads(bp.INVENTORY.read_text())
tname = {x["name"]: x for x in inv["tensors"]}


def load_shard(d, shard, key):
    with safe_open(d / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


# ---------------- sha256 sweep (untouched byte identity vs rot-patch-AR)
print("=== sha256 sweep selective vs rot-patch-AR ===", flush=True)
untouched, changed = [], []
for p in sorted(SEL_DIR.iterdir()):
    if not p.is_file():
        continue
    a = hashlib.sha256((SEL_DIR / p.name).read_bytes()).hexdigest()
    b = hashlib.sha256((AR_DIR / p.name).read_bytes()).hexdigest()
    (untouched if a == b else changed).append(p.name)
    print("  %-45s %s" % (p.name, "IDENTICAL" if a == b else "CHANGED"), flush=True)
print("untouched identical: %d / files: %d" % (len(untouched), len(untouched) + len(changed)), flush=True)
expect_changed = {"quantization_config.json"} | set(v["shard"] for v in sel["restore_pins"].values())
assert set(changed) == expect_changed, (set(changed) ^ expect_changed)

# ---------------- floor
tf = {"t_def": "t = 0.1 * median(per-row 128-group absmax of the BF16 ORIGINAL); same t for all 3 series per tensor",
      "band_def": "band_mass = sum(|w| for |w| <= t); retention = band_mass / band_mass[orig]",
      "series": ["orig(BF16)", "current(baseline W4A16 dequant)", "selective(selective patch dequant)"],
      "sha256": {"untouched_identical": sorted(untouched), "changed": sorted(changed)},
      "tensors": {}, "summary": {}}
t0 = time.time()
for b in sorted(patch_set):
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
        "decision": sel["per_tensor"].get(b, {}).get("decision", "PLAIN_KEEP_CURRENT"),
    }
    r = tf["tensors"][b]
    ok = (zs <= zc) and (r["selective"]["retention"] >= (r["current"]["retention"] or 0))
    print("%-55s t=%.5g zeros c/s=%d/%d retain c/s=%.4f/%.4f -> %s"
          % (b.split("layers.")[-1], t, zc, zs, r["current"]["retention"] or -1,
             r["selective"]["retention"] or -1, "FLOOR_OK" if ok else "regress"), flush=True)
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
                 "floor_tensors_full": floor_tensors,
                 "n_kept": sel["n_kept"],
                 "n_deselected": sel["n_deselected"],
                 "kept_are_floor": [b.split("layers.")[-1] for b in sel["kept"]
                                    if b in floor_tensors]}
OUT.write_text(json.dumps(tf, indent=1))
print("\nSUMMARY:", json.dumps(tf["summary"], indent=1), flush=True)
print("tail_fidelity_selective.json written (%.1fs)" % (time.time() - t0), flush=True)
