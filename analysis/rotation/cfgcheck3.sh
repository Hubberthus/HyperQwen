#!/bin/bash
cd /home/hubbe/qwen-serving
ls models/Qwen3.8-27B-W4A16-rot-patch-selective-mse/
echo "== quantization_config.json (mse)"
venv/bin/python - <<'EOF'
import json
c = json.load(open("models/Qwen3.8-27B-W4A16-rot-patch-selective-mse/quantization_config.json"))
tc = c.get("transform_config")
if tc:
    g = tc["config_groups"]["rot"]
    a = g["apply"][0]
    print("rot group:", g.get("type"), g.get("head_dim"), "n_targets:", len(a["targets"]))
    print("first/last targets:", a["targets"][:2], a["targets"][-2:])
else:
    print("no transform_config in quantization_config.json")
# compare with AR's, inside config.json
b = json.load(open("models/Qwen3.8-27B-W4A16-rot-patch-AR/config.json"))["quantization_config"]
tb = b.get("transform_config")
print("AR in-config tc targets:", len(tb["config_groups"]["rot"]["apply"][0]["targets"]) if tb else None)
EOF
