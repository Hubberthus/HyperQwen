#!/usr/bin/env python
"""_mseonly_build.py — MSE-ONLY selective rotation patch.

Builds models/Qwen3.8-27B-W4A16-rot-patch-selective-mse FRESH from the
BASELINE checkpoint (cp -a), then splices kept tensors:

  Decision (MSE-only this round, low-band tail NOT a criterion):
    per rotated tensor: winner = argmin(naive-rot_mse, rot-AR_mse);
    KEEP iff winner_mse < current_mse. Winner weights spliced from its
    source dir (naive -> rot-patch, AR -> rot-patch-AR).
    Fused qkv_proj/gate_up_proj parents decided whole-parent: keep only if
    EVERY member wins MSE-only (partial transform invalid in vLLM
    _validate_tlm_schemes).
    The 12 plain linear_attn in_proj tensors keep CURRENT baseline weights
    (option B; baseline copy already holds them, no splice) and get NO
    transform target.

  transform_config: copied structurally from rot-patch-AR
  (hadamard / head_dim 128 / location input / inverse false), targets
  swapped to exactly the kept set.

Pins: dequant manual-vs-ct == 0.0 on every touched tensor; spliced dequant
identical to its source dir dequant; sha256 of untouched files == baseline.

Writes analysis/rotation/selective_selection_mse.json.
CPU-only, source dirs read-only.
"""
import hashlib
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

REPO = bp.REPO
BASE = bp.SRC                                    # Qwen3.8-27B-W4A16-AutoRound
NAIVE = REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch"
ARDIR = REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-AR"
DST = REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-selective-mse"
OUT = HERE / "selective_selection_mse.json"

log = lambda m: print(m, flush=True)
t0 = time.time()

# ============================================================ decisions
msetab = json.loads((HERE / "mse_table_ar.json").read_text())["per_tensor"]
ROT = sorted(k for k, v in msetab.items() if v.get("rotated"))
PLAIN = sorted(k for k, v in msetab.items() if not v.get("rotated"))
assert len(ROT) == 38 and len(PLAIN) == 12, (len(ROT), len(PLAIN))


def parent_of(b):
    parts = b.split(".")
    L, leaf = parts[-3], ".".join(parts[-2:])
    if leaf in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        return f"layers.{L}.self_attn.qkv_proj"
    if leaf in ("mlp.gate_proj", "mlp.up_proj"):
        return f"layers.{L}.mlp.gate_up_proj"
    return None


rows = {}
for b in ROT:
    m = msetab[b]
    naive, ar, cur = m["naive-rot_mse"], m["rot-AR_mse"], m["current_mse"]
    var = "naive" if naive < ar else "ar"
    best = min(naive, ar)
    rows[b] = {"best_variant": var, "best_mse": best,
               "wins_mse_only": best < cur}

fused_members = {}
for b in ROT:
    p = parent_of(b)
    if p:
        fused_members.setdefault(p, set()).add(b)

keep = set(b for b in ROT if rows[b]["wins_mse_only"])
fused_actions = {}
for pname, mem in sorted(fused_members.items()):
    wins = mem & keep
    if wins == mem:
        fused_actions[pname] = "uniform_keep"
    elif not wins:
        fused_actions[pname] = "uniform_deselect"
    else:
        fused_actions[pname] = "mixed->whole_parent_deselected"
        keep -= mem
        for b in mem:
            rows[b]["fused_forced_deselect"] = pname
kept = sorted(keep)
deselected = sorted(set(ROT) - keep)
for b in PLAIN:
    rows[b] = {"decision": "KEEP_CURRENT_BASELINE_WEIGHTS", "plain": True}

assert "model.language_model.layers.63.self_attn.o_proj" in kept, "canary lost!"
log("KEPT %d / DESELECTED %d" % (len(kept), len(deselected)))
log("fused actions: %s" % json.dumps(fused_actions))
if not kept:
    log("kept set EMPTY -> STOP (no-transform copy not emitted)")
    sys.exit(2)

# ============================================================ copy baseline
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)
src_files = sorted(p for p in BASE.iterdir() if p.is_file())
for p in src_files:
    shutil.copy2(p, DST / p.name)
base_total = sum(p.stat().st_size for p in src_files)
dst_total = sum((DST / p.name).stat().st_size for p in src_files)
log("copied baseline: %d files, %d bytes (match=%s)"
    % (len(src_files), dst_total, base_total == dst_total))
assert base_total == dst_total

# ============================================================ splice
# index: kept base tensor -> (shard, source dir)
inv = json.loads(bp.INVENTORY.read_text())
tname = {x["name"]: x for x in inv["tensors"]}
SRC_FOR = {b: (NAIVE if rows[b]["best_variant"] == "naive" else ARDIR) for b in kept}

by_shard = {}
for b in kept:
    shard = tname[b + ".weight_packed"]["file"]
    by_shard.setdefault(shard, []).append(b)

log("\n=== splicing %d kept tensors ===" % len(kept))
spliced = {}
for shard in sorted(by_shard):
    hlen_b, h_b = bp.read_st_header(BASE / shard)
    hlen_p, h_p = bp.read_st_header((SRC_FOR[by_shard[shard][0]]) / shard)
    hlen_d, h_d = bp.read_st_header(DST / shard)
    assert hlen_d == hlen_b and h_d == h_b, ("dst header != baseline", shard)
    for b in sorted(by_shard[shard]):
        sdir = SRC_FOR[b]
        hlen_s, h_s = bp.read_st_header(sdir / shard)
        shape = tuple(bp.load_src_tensor(shard, b + ".weight_shape").tolist())
        for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
            kb = b + suf
            assert kb in h_s and kb in h_b, kb
            assert h_s[kb]["dtype"] == h_b[kb]["dtype"], (kb, "dtype")
            assert h_s[kb]["shape"] == h_b[kb]["shape"], (kb, "shape")
            assert h_s[kb]["data_offsets"] == h_b[kb]["data_offsets"], (kb, "offset")
        with safe_open(sdir / shard, framework="pt", device="cpu") as f:
            pk = f.get_tensor(b + ".weight_packed")
            sc = f.get_tensor(b + ".weight_scale")
        with safe_open(BASE / shard, framework="pt", device="cpu") as f:
            pk0 = f.get_tensor(b + ".weight_packed")
            sc0 = f.get_tensor(b + ".weight_scale")
        changed = bool((pk.to(torch.int32) != pk0.to(torch.int32)).any()) or \
            bool((sc.to(torch.float32) != sc0.to(torch.float32)).any())
        assert changed, (b, "source bytes equal baseline — nothing to splice")
        del pk0, sc0
        bp.st_write_at(DST / shard, hlen_d, h_d, b + ".weight_packed", pk)
        bp.st_write_at(DST / shard, hlen_d, h_d, b + ".weight_scale", sc)
        # pins: manual vs ct == 0.0, and dequant identity vs source dir
        with safe_open(DST / shard, framework="pt", device="cpu") as f:
            npk = f.get_tensor(b + ".weight_packed")
            nsc = f.get_tensor(b + ".weight_scale")
        md = bp.manual_dequant(npk, nsc, shape)
        cd = bp.ct_dequant(npk, nsc, shape).numpy()
        pin = float(np.abs(md - cd).max())
        sdeq = bp.manual_dequant(pk, sc, shape)
        ident = bool(np.array_equal(md, sdeq))
        assert pin == 0.0, (b, pin)
        assert ident, (b, "spliced dequant != source dequant")
        spliced[b] = {"shard": shard, "source_dir": str(sdir), "pin": pin,
                      "dequant_equals_source": ident}
        log("  spliced %-58s src=%s pin=%.3e"
            % (b.replace("model.language_model.layers.", "L"),
               "naive" if rows[b]["best_variant"] == "naive" else "AR", pin))
        del pk, sc, npk, nsc, md, cd, sdeq

log("dequant pin: PASS (all 0.0, %d touched tensors)" % len(spliced))

# ============================================================ transform_config
qcfg = json.loads((DST / "quantization_config.json").read_text())
assert "transform_config" not in qcfg
ar_q = json.loads((ARDIR / "quantization_config.json").read_text())
tc = json.loads(json.dumps(ar_q["transform_config"]))  # deep copy of structure
targets = ["language_model.model." + b.removeprefix("model.language_model.")
           for b in kept]
tc["config_groups"]["rot"]["apply"][0]["targets"] = targets
qcfg["transform_config"] = tc
(DST / "quantization_config.json").write_text(json.dumps(qcfg, indent=1))
log("transform_config written: hadamard/head_dim=%s, %d targets"
    % (tc["config_groups"]["rot"]["head_dim"], len(targets)))

# ============================================================ byte identity
log("\n=== sha256 sweep vs baseline ===")
untouched, changed = [], []
touched_shards = set(v["shard"] for v in spliced.values())
expect_changed = touched_shards | {"quantization_config.json"}
for p in src_files:
    a = bp.sha256(BASE / p.name)
    b = bp.sha256(DST / p.name)
    (untouched if a == b else changed).append(p.name)
    status = "IDENTICAL" if a == b else "CHANGED"
    log("  %-45s %s" % (p.name, status))
assert set(changed) == expect_changed, (set(changed) ^ expect_changed)
log("byte identity: %d untouched identical, %d changed (expected: %d touched shards + quantization_config.json)"
    % (len(untouched), len(changed), len(touched_shards)))

dst_total = sum(p.stat().st_size for p in DST.iterdir() if p.is_file())
log("TOTAL dst bytes: %d (baseline %d)" % (dst_total, base_total))

# ============================================================ selection json
import numpy as np  # noqa: E402

macro_kept_cur = float(np.mean([msetab[b]["current_mse"] for b in kept]))
macro_kept_rot = float(np.mean([msetab[b]["rot-AR_mse"] for b in kept]))
macro_kept_naive = float(np.mean([msetab[b]["naive-rot_mse"] for b in kept]))
wins = [{"tensor": b, "current": msetab[b]["current_mse"],
         "naive": msetab[b]["naive-rot_mse"], "ar": msetab[b]["rot-AR_mse"]}
        for b in kept]
wins.sort(key=lambda x: x["current"] / min(x["naive"], x["ar"]), reverse=True)

per_tensor = {}
for b in ROT:
    m = msetab[b]
    per_tensor[b] = {
        "rotated": True, "shape": m["shape"],
        "mse_current": m["current_mse"], "mse_naive": m["naive-rot_mse"],
        "mse_ar": m["rot-AR_mse"],
        "best_variant": rows[b]["best_variant"],
        "best_variant_mse": rows[b]["best_mse"],
        "wins_mse_only": rows[b]["wins_mse_only"],
        "decision": "KEEP" if b in keep else "DESELECT",
        "fused_parent": parent_of(b),
        "fused_action": fused_actions.get(parent_of(b)),
        "rationale": (
            "best rotated variant %s (%.4e) vs current (%.4e) -> %s"
            % (rows[b]["best_variant"], rows[b]["best_mse"],
               m["current_mse"], "kept (MSE-only)"
               if b in keep else "deselected")
            + ("" if parent_of(b) is None or b in keep else
               "; forced by fused parent %s" % parent_of(b))),
    }
for b in PLAIN:
    m = msetab[b]
    per_tensor[b] = {
        "rotated": False, "shape": m["shape"],
        "mse_current_optionB": m["optionB_mse"], "mse_ar_optionA": m["optionA_mse"],
        "decision": "KEEP_CURRENT_BASELINE_WEIGHTS",
        "fused_parent": parent_of(b),
        "rationale": "plain in_proj tensor (option B established): baseline copy "
                     "already holds option-B weights; no splice, no transform target.",
    }

sel = {
    "decision_rule": "MSE-ONLY: keep rotated tensor iff best rotated variant "
                     "(argmin of naive-rot / rot-AR corrected MSE) beats the "
                     "current baseline MSE; low-band tail NOT a criterion this "
                     "round. Fused qkv_proj/gate_up_proj parents decided "
                     "whole-parent (every member must win, else parent deselected).",
    "fused_rule_basis": "vLLM compressed_tensors transform/linear.py: single input "
                        "transform wraps the whole fused input; _validate_tlm_schemes "
                        "raises on partial targeting.",
    "plain_in_proj_policy": "12 plain linear_attn in_proj tensors keep current "
                            "baseline weights (option B; fresh baseline copy holds "
                            "them by construction) and get no transform target.",
    "n_rotated": len(ROT),
    "n_plain": len(PLAIN),
    "n_kept": len(kept),
    "n_deselected": len(deselected),
    "n_fused_parents_kept": sum(1 for a in fused_actions.values() if a == "uniform_keep"),
    "fused_actions": fused_actions,
    "kept": kept,
    "deselected": deselected,
    "macro_mse": {"kept_current": macro_kept_cur,
                  "kept_naive": macro_kept_naive,
                  "kept_rot_ar": macro_kept_rot,
                  "all38_current": json.loads((HERE / "mse_table_ar.json")
                                              .read_text())["summary"]["rotated_macro"]["current_mse"]},
    "largest_mse_wins": wins[:3],
    "splice_pins": spliced,
    "transform_config": {"type": "hadamard", "head_dim":
                         tc["config_groups"]["rot"]["head_dim"],
                         "location": "input", "inverse": False,
                         "n_targets": len(targets)},
    "dir": str(DST),
    "total_size_bytes": dst_total,
    "baseline_total_bytes": base_total,
    "byte_identity": {"untouched_identical": untouched, "changed": changed},
    "canary_63_o_proj_kept": "model.language_model.layers.63.self_attn.o_proj" in kept,
    "per_tensor": per_tensor,
}
OUT.write_text(json.dumps(sel, indent=1))
log("wrote %s" % OUT)
log("DONE splice phase in %.1fs" % (time.time() - t0))
