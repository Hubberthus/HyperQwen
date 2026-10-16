#!/usr/bin/env python
"""build_selective.py — Step 1 selective rotation patch.

Builds models/Qwen3.8-27B-W4A16-rot-patch-selective from
models/Qwen3.8-27B-W4A16-rot-patch-AR:

  KEEP rotated  (rot-AR packed/scale stay): per-tensor winners where rot-AR has
  BOTH lower corrected-MSE than the current baseline AND no low-band tail
  regression (retention >= current), subject to fused-module granularity.
  DESELECT: baseline AutoRound packed+scale bytes restored by in-place
  overwrite (same shape/dtype -> header untouched), and the tensor removed
  from transform_config targets.

Fused-module rule (verified against installed vLLM
vllm/model_executor/layers/quantization/compressed_tensors/transform/linear.py):
for a fused parent (packed_modules_mapping: qkv_proj -> [q,k,v],
gate_up_proj -> [gate,up], in_proj_qkvz -> [in_proj_qkv,in_proj_z],
in_proj_ba -> [in_proj_b,in_proj_a]) a single input-side Hadamard transform
wraps the WHOLE fused module input; get_linear_transform_schemes assigns
input_tfms per partition and _validate_tfm_schemes raises
ValueError("Must have same input") when partitions differ. A partial target
set on a fused parent is therefore invalid -> if any member of a fused parent
must be deselected, the whole parent is deselected (all members restored to
baseline weights, all removed from transform targets).

Writes analysis/rotation/selective_selection.json.
CPU-only, read-only w.r.t. all source dirs.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp

from safetensors import safe_open

AR_DIR = bp.REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-AR"
DST = bp.REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-selective"
OUT = HERE / "selective_selection.json"

# ------------------------------------------------------------- decisions
mse = json.loads((HERE / "mse_table_ar.json").read_text())["per_tensor"]
tail = json.loads((HERE / "tail_fidelity_autoround.json").read_text())["tensors"]

ROTATED = sorted(k for k, v in mse.items() if v.get("rotated"))
assert len(ROTATED) == 38, len(ROTATED)

rows = {}
for b in ROTATED:
    m, t = mse[b], tail[b]
    mse_ok = m["rot-AR_mse"] < m["current_mse"]
    tail_ok = (t["rot-AR"]["retention"] or 0.0) >= (t["current"]["retention"] or 0.0)
    rows[b] = {"mse_ok": mse_ok, "tail_ok": tail_ok,
               "win_strict": mse_ok and tail_ok}

# ------------------------------------------------------------- fused map
def parent_of(b):
    """-> (parent_name, [qkv-part names for packed_modules_mapping]) or None."""
    parts = b.split(".")
    leaf = ".".join(parts[-2:])
    L = parts[-3]
    if leaf == "self_attn.q_proj":
        return (f"layers.{L}.self_attn.qkv_proj", ["q_proj", "k_proj", "v_proj"])
    if leaf == "self_attn.k_proj":
        return (f"layers.{L}.self_attn.qkv_proj", ["k_proj", "q_proj", "v_proj"])
    if leaf == "self_attn.v_proj":
        return (f"layers.{L}.self_attn.qkv_proj", ["v_proj", "q_proj", "k_proj"])
    if leaf == "mlp.gate_proj":
        return (f"layers.{L}.mlp.gate_up_proj", ["gate_proj", "up_proj"])
    if leaf == "mlp.up_proj":
        return (f"layers.{L}.mlp.gate_up_proj", ["gate_proj", "up_proj"])
    if leaf == "linear_attn.in_proj_qkv":
        return (f"layers.{L}.linear_attn.in_proj_qkvz", ["in_proj_qkv", "in_proj_z"])
    if leaf == "linear_attn.in_proj_z":
        return (f"layers.{L}.linear_attn.in_proj_qkvz", ["in_proj_qkv", "in_proj_z"])
    return None


fused_members = {}
for b in ROTATED:
    p = parent_of(b)
    if p:
        fused_members.setdefault(p[0], set()).add(b)

fused_actions = {}
keep = set(b for b in ROTATED if rows[b]["win_strict"])
for pname, members in sorted(fused_members.items()):
    wins = members & keep
    if not wins:
        fused_actions[pname] = "uniform_deselect"
    elif len(wins) == len(members):
        fused_actions[pname] = "uniform_keep"
    else:
        fused_actions[pname] = "mixed->whole_parent_deselected"
        keep -= members
        for b in members:
            rows[b]["fused_forced_deselect"] = pname

kept = sorted(keep)
# PLAIN in_proj tensors: spec says they "already keep current weights", but the
# AR dir holds AutoRound re-quantized (option-A) weights for all 12, and option-A
# loses to option-B (current) on BOTH corrected MSE (+12..24% worse) and tail
# retention. Same decision rule applied -> restore to baseline (fused parents
# in_proj_qkvz are uniformly deselected). Deviation recorded in the JSON.
PLAIN = sorted(k for k, v in mse.items() if not v.get("rotated"))
assert len(PLAIN) == 12
plain_plain_rows = {}
for b in PLAIN:
    m, t = mse[b], tail[b]
    plain_mse_ok = m["optionA_mse"] < m["optionB_mse"]
    plain_tail_ok = (t["rot-AR"]["retention"] or 0.0) >= (t["current"]["retention"] or 0.0)
    rows[b] = {"mse_ok": plain_mse_ok, "tail_ok": plain_tail_ok, "win_strict": False,
               "plain": True}
    plain_plain_rows[b] = (plain_mse_ok, plain_tail_ok)
    assert not plain_mse_ok and not plain_tail_ok, b
    p = parent_of(b)
    assert p and p[0].endswith("in_proj_qkvz"), b
deselected = sorted(b for b in ROTATED if b not in keep) + PLAIN
print("KEEP  : %d" % len(kept))
for b in kept:
    print("   ", b)
print("DESEL : %d" % len(deselected))
print("fused actions:", json.dumps(fused_actions, indent=1))

# ------------------------------------------------------------- copy
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)
src_files = sorted(p for p in AR_DIR.iterdir() if p.is_file())
t0 = time.time()
for p in src_files:
    shutil.copy2(p, DST / p.name)
print("copied %d files (%.1fs)" % (len(src_files), time.time() - t0))

# ------------------------------------------------------------- restore
ar_idx = json.loads((AR_DIR / "model.safetensors.index.json").read_text())["weight_map"]
hdr = {}
by_shard = {}
for b in deselected:
    by_shard.setdefault(ar_idx[b + ".weight_packed"], []).append(b)

restored = {}
for shard in sorted(by_shard):
    with safe_open(AR_DIR / shard, framework="pt", device="cpu") as f:
        hlen_ar = None
    hlen, h_ar = bp.read_st_header(AR_DIR / shard)
    hlen_b, h_b = bp.read_st_header(bp.SRC / shard)
    for b in sorted(by_shard[shard]):
        for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
            k = b + suf
            assert k in h_ar and k in h_b, k
            assert h_ar[k]["data_offsets"] == h_b[k]["data_offsets"], (k, "offset mismatch")
            assert h_ar[k]["dtype"] == h_b[k]["dtype"], (k, "dtype mismatch")
        shape = tuple(bp.load_src_tensor(shard, b + ".weight_shape").tolist())
        pk_b = bp.load_src_tensor(shard, b + ".weight_packed")
        sc_b = bp.load_src_tensor(shard, b + ".weight_scale")
        # sanity: rot-AR bytes really differ from baseline for deselected tensors
        with safe_open(AR_DIR / shard, framework="pt", device="cpu") as f:
            pk_a = f.get_tensor(b + ".weight_packed")
            sc_a = f.get_tensor(b + ".weight_scale")
        assert pk_a.dtype == pk_b.dtype and tuple(pk_a.shape) == tuple(pk_b.shape), b
        changed = bool((pk_a.to(torch.int32) != pk_b.to(torch.int32)).any()) or \
            bool((sc_a.to(torch.float32) != sc_b.to(torch.float32)).any())
        # in-place overwrite with baseline bytes
        hlen_d, h_d = bp.read_st_header(DST / shard)
        n1 = bp.st_write_at(DST / shard, hlen_d, h_d, b + ".weight_packed", pk_b)[1]
        n2 = bp.st_write_at(DST / shard, hlen_d, h_d, b + ".weight_scale", sc_b)[1]
        # dequant pin: manual vs compressed-tensors == 0.0, and == baseline dequant
        with safe_open(DST / shard, framework="pt", device="cpu") as f:
            np_ = f.get_tensor(b + ".weight_packed")
            ns = f.get_tensor(b + ".weight_scale")
        md = bp.manual_dequant(np_, ns, shape)
        cd = bp.ct_dequant(np_, ns, shape).numpy()
        pin = float(np.abs(md - cd).max())
        bd = bp.manual_dequant(pk_b, sc_b, shape)
        same = bool(np.array_equal(md, bd))
        restored[b] = {"shard": shard, "pin_manual_vs_ct": pin,
                       "dequant_equals_baseline": same, "ar_had_different_bytes": changed}
        print("  restored %-58s pin=%.3e same_as_baseline=%s ar_differed=%s"
              % (b.replace("model.language_model.layers.", "L"), pin, same, changed),
              flush=True)
        assert pin == 0.0, (b, pin)
        assert same, (b, "restored dequant != baseline dequant")
        del pk_b, sc_b, pk_a, sc_a, md, cd, bd

all_pins_zero = all(v["pin_manual_vs_ct"] == 0.0 for v in restored.values())
print("dequant pin: %s (%d touched tensors)" %
      ("PASS (all 0.0)" if all_pins_zero else "FAIL", len(restored)))

# ------------------------------------------------------------- config
qcfg_path = DST / "quantization_config.json"
qcfg = json.loads(qcfg_path.read_text())
tc = qcfg["transform_config"]["config_groups"]["rot"]["apply"]
old_targets = list(tc[0]["targets"])
new_targets = ["language_model.model." + b.removeprefix("model.language_model.") for b in kept]
tc[0]["targets"] = new_targets
qcfg_path.write_text(json.dumps(qcfg, indent=1))
print("transform targets: %d -> %d" % (len(old_targets), len(new_targets)))

# ------------------------------------------------------------- selection json
def short(b):
    return b.removeprefix("model.language_model.layers.")


sel_rows = {}
for b in ROTATED:
    m, t, r = mse[b], tail[b], rows[b]
    rot_ar = t["rot-AR"]
    cur = t["current"]
    sel_rows[b] = {
        "rotated": True,
        "shape": m["shape"],
        "mse_current": m["current_mse"],
        "mse_naive_rot": m["naive-rot_mse"],
        "mse_rot_ar": m["rot-AR_mse"],
        "mse_rot_ar_beats_current": r["mse_ok"],
        "t": t["t"],
        "zeros_current": cur["zeros"],
        "zeros_rot_ar": rot_ar["zeros"],
        "retention_current": cur["retention"],
        "retention_rot_ar": rot_ar["retention"],
        "tail_no_regression": r["tail_ok"],
        "decision": "KEEP" if b in keep else "DESELECT",
        "fused_parent": parent_of(b)[0] if parent_of(b) else None,
        "fused_action": fused_actions.get(parent_of(b)[0]) if parent_of(b) else None,
        "rationale": (
            ("wins both criteria" if r["win_strict"] else
             "loses MSE criterion" if not r["mse_ok"] else
             "loses tail criterion (retention drop %.2f%%)" %
             (100 * (rot_ar["retention"] - cur["retention"]) / cur["retention"]))
            + ("" if parent_of(b) is None or b in keep else
               "; forced by fused parent %s (%s)" %
               (parent_of(b)[0], fused_actions[parent_of(b)[0]]))
        ),
    }

for b in PLAIN:
    m, t = mse[b], tail[b]
    ok_m, ok_t = plain_plain_rows[b]
    p = parent_of(b)
    sel_rows[b] = {
        "rotated": False,
        "shape": m["shape"],
        "mse_current_optionB": m["optionB_mse"],
        "mse_rot_ar_optionA": m["optionA_mse"],
        "mse_rot_ar_beats_current": ok_m,
        "t": t["t"],
        "zeros_current": t["current"]["zeros"],
        "zeros_rot_ar": t["rot-AR"]["zeros"],
        "retention_current": t["current"]["retention"],
        "retention_rot_ar": t["rot-AR"]["retention"],
        "tail_no_regression": ok_t,
        "decision": "RESTORE_BASELINE",
        "fused_parent": p[0],
        "fused_action": "uniform_deselect",
        "rationale": "plain in_proj tensor held AutoRound option-A re-quantized weights in "
                     "rot-patch-AR (NOT current weights as spec assumed); option-A loses to "
                     "current on both MSE (%+.1f%%) and tail retention -> restored to baseline "
                     "(fused in_proj_qkvz parent uniformly deselected)" % (100 * (m["optionA_mse"] / m["optionB_mse"] - 1)),
    }

macro = {
    "rotated_macro_from_mse_table": {
        "current": json.loads((HERE / "mse_table_ar.json").read_text())["summary"]["rotated_macro"]["current_mse"],
        "naive_rot": json.loads((HERE / "mse_table_ar.json").read_text())["summary"]["rotated_macro"]["naive-rot_mse"],
        "rot_ar": json.loads((HERE / "mse_table_ar.json").read_text())["summary"]["rotated_macro"]["rot-AR_mse"],
    },
    "selective_kept_macro_rot_ar_mse": float(np.mean([mse[b]["rot-AR_mse"] for b in kept])),
    "selective_kept_macro_current_mse": float(np.mean([mse[b]["current_mse"] for b in kept])),
}

sel = {
    "decision_rule": "KEEP rotated iff (a) rot-AR corrected-MSE < current baseline MSE AND "
                     "(b) rot-AR low-band retention >= current retention (same t/band defs as "
                     "tail_fidelity_autoround.json); fused parents decided at whole-module "
                     "granularity (mixed parent -> whole parent deselected).",
    "fused_rule_basis": "vLLM compressed_tensors transform/linear.py: get_linear_transform_schemes "
                        "assigns per-partition input_tfms; _validate_tfm_schemes raises "
                        "ValueError('Must have same input') on mismatch; a single input transform "
                        "wraps the whole fused module input, so partial targeting is invalid.",
    "fused_actions": fused_actions,
    "n_rotated": len(ROTATED),
    "n_plain_in_proj": len(PLAIN),
    "n_kept": len(kept),
    "n_deselected": len(deselected),
    "spec_deviations": [
        "The 12 plain linear_attn in_proj tensors were NOT holding current baseline "
        "weights in rot-patch-AR as the task spec assumed: they hold AutoRound "
        "option-A re-quantized weights (verified byte-difference vs baseline and via "
        "mse_table_ar.json optionA vs optionB). Since option-A loses on BOTH corrected "
        "MSE (+12..24% worse) and low-band retention, they were restored to baseline "
        "under the same decision rule (fused in_proj_qkvz parents uniformly deselected).",
    ],
    "kept": kept,
    "deselected": deselected,
    "macro": macro,
    "restore_pins": restored,
    "per_tensor": sel_rows,
}
OUT.write_text(json.dumps(sel, indent=1))
print("wrote", OUT)
print("TOTAL selective dir bytes:", sum(p.stat().st_size for p in DST.iterdir() if p.is_file()))
