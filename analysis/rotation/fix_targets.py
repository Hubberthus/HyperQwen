#!/usr/bin/env python
"""Fix transform_config targets in the AR patch dir: use exact vLLM module names.

vLLM module paths are `language_model.model.layers.N.<...>` (no `model.` prefix),
and is_match() is exact-string match (match_name) against full part names
(fused modules resolve via packed_modules_mapping: in_proj_qkvz -> [in_proj_qkv,
in_proj_z], qkv_proj -> [q/k/v_proj], gate_up_proj -> [gate_proj, up_proj]).
"""
import json, sys
from pathlib import Path

REPO = Path("/home/hubbe/qwen-serving")
AR = REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-AR"

LIN_LAYERS = [0, 38, 45, 56, 58, 60]
FULL_LAYERS = [27, 63]
targets = []
for L in LIN_LAYERS:
    for t in ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        targets.append("language_model.model.layers.%d.%s" % (L, t))
for L in FULL_LAYERS:
    for t in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        targets.append("language_model.model.layers.%d.%s" % (L, t))
assert len(targets) == 50, len(targets)

tcfg = {"config_groups": {"rot": {"type": "hadamard", "head_dim": 128,
        "apply": [{"targets": targets, "location": "input", "inverse": False}]}}}

for name in ["quantization_config.json", "config.json"]:
    p = AR / name
    d = json.loads(p.read_text())
    if name == "config.json":
        d["quantization_config"]["transform_config"] = tcfg
    else:
        d["transform_config"] = tcfg
    p.write_text(json.dumps(d, indent=2))
    print("updated", p)
print("targets:", len(targets), "first:", targets[0], "last:", targets[-1])
