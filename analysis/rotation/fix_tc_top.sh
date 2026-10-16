#!/bin/bash
cd /home/hubbe/qwen-serving
venv/bin/python - <<'EOF'
import json
ar = json.load(open("models/Qwen3.8-27B-W4A16-rot-patch-AR/config.json"))
print("AR top-level transform_config:", "transform_config" in ar,
      "targets:", len(ar["transform_config"]["config_groups"]["rot"]["apply"][0]["targets"]) if "transform_config" in ar else "-")
mse = "models/Qwen3.8-27B-W4A16-rot-patch-selective-mse"
cfg = json.load(open(f"{mse}/config.json"))
# remove the wrong nesting, add at top level
tc_in = cfg["quantization_config"].pop("transform_config", None)
tc = tc_in or json.load(open(f"{mse}/quantization_config.json"))["transform_config"]
cfg["transform_config"] = tc
json.dump(cfg, open(f"{mse}/config.json", "w"), indent=2, sort_keys=True)
print("mse: top-level transform_config set, targets:",
      len(tc["config_groups"]["rot"]["apply"][0]["targets"]))
EOF
