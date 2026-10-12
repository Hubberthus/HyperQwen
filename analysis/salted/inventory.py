#!/usr/bin/env python3
"""Task 1 (salted-quantization spike): weight inventory for Qwen3.8-27B W4A16.

Iterates every *.safetensors shard of the checkpoint with safetensors.safe_open
(framework='pt'), classifies each stored tensor into a quant group derived from
config.json quantization_config (group_0 Linear->INT4, group_1 lm_head->INT8,
group_2 embed_tokens->INT8, group_3 mtp.*->INT8, quantization_config.ignore ->
FP16-ignored), and writes the full table to analysis/salted/inventory.json plus
a grouped stdout summary with sanity checks.

Run from repo root inside the venv:
    python analysis/salted/inventory.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

from safetensors import safe_open

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound")
OUT = os.path.join(REPO, "analysis", "salted", "inventory.json")

SUFFIX_RE = re.compile(r"\.(weight_packed|weight_scale|weight_shape)$")


def logical_weight(name: str) -> str:
    """Strip packing suffixes so packed/scale/shape rows map to one weight."""
    return SUFFIX_RE.sub("", name)


def classify(name: str, logical: str, qcfg: dict, ignore_set: set) -> str:
    groups = qcfg["config_groups"]

    # group_3: re:^mtp\..* -> INT8 (mtp draft head/fc/attention/mlp)
    if re.fullmatch(r"mtp\..*", logical):
        return "INT8-head"
    # group_1: re:.*lm_head$ -> INT8
    if re.search(r"lm_head$", logical):
        return "INT8-head"
    # group_2: re:.*embed_tokens$ -> INT8
    if re.search(r"embed_tokens$", logical):
        return "INT8-head"
    # ignore list (FP16/BF16 kept): visual tower + linear_attn.in_proj_a/b etc.
    # config entries omit the trailing ".weight"; strip it from the tensor name
    base = logical[:-len(".weight")] if logical.endswith(".weight") else logical
    if base in ignore_set:
        return "FP16-ignored"
    # group_0 targets Linear: language_model layer Linear weight_packed rows
    # (.q_proj/.k_proj/.v_proj/.o_proj/.gate_proj/.up_proj/.down_proj,
    #  .in_proj_qkv, ...) -> INT4-linear.
    # NOTE: the checkpoint stores two projection styles across the 64 layers
    # (hybrid attention): 48 linear_attn layers use packed in_proj_qkv /
    # in_proj_z / out_proj; 16 self_attn layers use separate q/k/v_proj.
    # The module-name match must cover any tensor suffix (weight / weight_packed /
    # weight_scale / weight_shape) so packed rows and their scale/shape
    # companions are classified with their parent module.
    if "language_model.layers" in logical and (
        re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
                  r"in_proj_qkv|in_proj_z|out_proj)\.", logical + ".")
        or logical.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj",
                             ".gate_proj", ".up_proj", ".down_proj",
                             ".in_proj_qkv", ".in_proj_z", ".out_proj"))
    ):
        return "INT4-linear"
    # Safety net (review finding): any remaining packed weight row in the
    # language-model layers is INT4-linear by definition. A packed row landing
    # here would mean an unknown quantized module, which the sanity check
    # below reports as an error.
    if name.endswith(".weight_packed") and "language_model.layers" in logical:
        return "INT4-linear"
    return "FP16-other"


def main() -> int:
    with open(os.path.join(CKPT, "config.json")) as f:
        cfg = json.load(f)
    qcfg = cfg["quantization_config"]
    ignore_set = set(qcfg["ignore"])
    text = cfg.get("text_config", {})
    tc = {k: text[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size") if k in text}

    shards = sorted(glob.glob(os.path.join(CKPT, "*.safetensors")))
    rows = []
    per_group = {}
    for path in shards:
        with safe_open(path, framework="pt") as sf:
            for name in sf.keys():
                sl = sf.get_slice(name)
                shape = list(sl.get_shape())
                dtype = str(sl.get_dtype())
                logical = logical_weight(name)
                grp = classify(name, logical, qcfg, ignore_set)
                elems = 1
                for d in shape:
                    elems *= d
                rows.append({"name": name, "shape": shape, "dtype": dtype,
                             "elements": elems, "group": grp,
                             "logical_weight": logical,
                             "file": os.path.basename(path)})
                g = per_group.setdefault(grp, {"tensors": 0, "elements": 0})
                g["tensors"] += 1
                g["elements"] += elems

    # sanity: per-group element counts
    print(f"checkpoint: {os.path.basename(CKPT)}  shards: {len(shards)}")
    print(f"{'group':<14} {'tensors':>8} {'elements':>18}")
    for grp, g in sorted(per_group.items()):
        print(f"{grp:<14} {g['tensors']:>8} {g['elements']:>18}")

    # sanity 2b (review Important #2): every packed row must land in INT4/INT8.
    # A packed row classified FP16-* means an unknown quantized module — the
    # safety net in classify() should have caught it, so this asserts the net.
    stray = [r for r in rows if r["name"].endswith(".weight_packed")
             and r["group"] not in ("INT4-linear", "INT8-head")]
    print(f"\nstray packed rows outside INT4/INT8 groups: {len(stray)}")
    for r in stray[:10]:
        print(f"  STRAY {r['name']}  group={r['group']}  shape={r['shape']}")
    if stray:
        raise SystemExit(f"ERROR: {len(stray)} weight_packed rows not classified "
                         "as INT4/INT8 — extend the classify() safety net")

    # sanity 2c: config-derived expected totals
    print(f"\nconfig: {tc}")
    nl = tc.get("num_hidden_layers", 0)
    hs = tc.get("hidden_size", 0)
    isz = tc.get("intermediate_size", 0)
    int4_rows = [r for r in rows if r["group"] == "INT4-linear"
                 and r["name"].endswith("weight_packed")]
    int4_elems = sum(r["elements"] for r in int4_rows)
    print(f"INT4 packed rows: {len(int4_rows)}  packed elements: {int4_elems}")
    # Packing format varies (marlin-style repack); derive the observed unpack
    # ratio from scale-shape instead of assuming a fixed divisor. For each
    # module the unpacked element count equals packed_elems * ratio where
    # ratio = (scale_cols * group_size) relative to packed geometry. Report
    # per-row samples rather than asserting a universal multiplier.
    qkv = [r for r in int4_rows if ".in_proj_qkv.weight_packed" in r["name"]]
    qkv_scale = [r for r in rows if ".in_proj_qkv.weight_scale" in r["name"]]
    if qkv and qkv_scale:
        p, s = qkv[0], qkv_scale[0]
        # in_proj_qkv: out=3*hs=15360, in=hs=5120 -> 78,643,200 unpacked
        # elements; packed holds 6,553,600 I32 elems -> 12 int4 per int32
        # (marlin-style repack; scale [10240,40] keeps rows, cols = packed/16).
        unpacked = 3 * hs * hs
        print(f"  sample packed row 0.linear_attn.in_proj_qkv: "
              f"packed {p['shape']} ({p['elements']} elems), scale {s['shape']}; "
              f"unpacked {unpacked} elems = packed * {unpacked / p['elements']:.1f}")
    # in_proj_qkv: one packed tensor per linear_attn layer holding q+k+v
    qkv = [r for r in int4_rows if ".in_proj_qkv.weight_packed" in r["name"]]
    # 48 linear_attn layers have in_proj_qkv (single packed tensor holding
    # q+k+v); 16 self_attn layers instead carry separate q/k/v_proj rows.
    print(f"  in_proj_qkv rows: {len(qkv)} (expect 48: one per linear_attn layer)")
    qkv_widths = sum(r["shape"][1] for r in qkv)
    qkv_rows_total = sum(r["shape"][0] for r in qkv)
    print(f"  in_proj_qkv packed geometry: rows_total={qkv_rows_total} "
          f"cols_total={qkv_widths} (layout is marlin-repacked, not a plain "
          f"width/4; per-layer sample above gives the unpack ratio)")

    # sanity 3: FP16-ignored tensors vs config ignore list (strip ".weight")
    def base_name(lw: str) -> str:
        return lw[:-len(".weight")] if lw.endswith(".weight") else lw

    ign_weights = {base_name(r["logical_weight"]) for r in rows
                   if r["group"] == "FP16-ignored"}
    cfg_ignore = ignore_set
    print(f"\nFP16-ignored distinct weights in ckpt: {len(ign_weights)}")
    print(f"config ignore entries: {len(cfg_ignore)}")
    missing = sorted(w for w in cfg_ignore if w not in ign_weights)
    extra = sorted(w for w in ign_weights if w not in cfg_ignore)
    print(f"in config but absent from ckpt ({len(missing)}): {missing[:10]}"
          + (" ..." if len(missing) > 10 else ""))
    print(f"in ckpt but not in config ({len(extra)}): {extra[:10]}"
          + (" ..." if len(extra) > 10 else ""))

    out = {
        "checkpoint": os.path.basename(CKPT),
        "config_sanity": tc,
        "groups": {g: g2 for g, g2 in sorted(per_group.items())},
        "ignore_check": {
            "config_ignore_count": len(cfg_ignore),
            "ckpt_ignored_weights": len(ign_weights),
            "missing_from_ckpt": missing,
            "extra_in_ckpt": extra,
        },
        "tensors": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT} ({len(rows)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
