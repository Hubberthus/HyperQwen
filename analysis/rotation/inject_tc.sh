#!/bin/bash
# Inject transform_config from the sidecar quantization_config.json into
# config.json's quantization_config, mirroring the serving-proven
# rot-patch-AR layout. Backs up config.json first. Idempotent.
set -e
cd /home/hubbe/qwen-serving
D=models/Qwen3.8-27B-W4A16-rot-patch-selective-mse
cp -n $D/config.json $D/config.json.bak-pre-tc-inject || true
venv/bin/python - <<'EOF'
import json
d = "models/Qwen3.8-27B-W4A16-rot-patch-selective-mse"
side = json.load(open(f"{d}/quantization_config.json"))
tc = side.get("transform_config")
assert tc and tc["config_groups"]["rot"]["type"] == "hadamard", "sidecar tc missing"
n = len(tc["config_groups"]["rot"]["apply"][0]["targets"])
cfg = json.load(open(f"{d}/config.json"))
assert "transform_config" not in cfg["quantization_config"], "already present"
cfg["quantization_config"]["transform_config"] = tc
json.dump(cfg, open(f"{d}/config.json", "w"), indent=2, sort_keys=True)
print(f"injected transform_config ({n} targets) into config.json")
EOF
