#!/usr/bin/env python
"""Probe config attrs to determine the correct cos/sin head-dim slice for rope."""
import json
import transformers

cfg = transformers.AutoConfig.from_pretrained(
    "models/Qwen3.8-27B-W4A16-AutoRound", trust_remote_code=True)
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
keys = ("head_size", "head_dim", "num_attention_heads", "num_key_value_heads",
        "hidden_size", "rope_scaling", "partial_rotary_factor", "rotary_dim")
out = {k: getattr(tc, k, None) for k in keys}
# rope_parameters lives in the raw json
raw = json.load(open("models/Qwen3.8-27B-W4A16-AutoRound/config.json"))
tc_raw = raw.get("text_config", raw)
out["rope_parameters"] = tc_raw.get("rope_parameters")
print(json.dumps(out, indent=1, default=str))
