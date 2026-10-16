#!/bin/bash
cd /home/hubbe/qwen-serving
venv/bin/python - <<'EOF'
import json
mse = "models/Qwen3.8-27B-W4A16-rot-patch-selective-mse"
cfg = json.load(open(f"{mse}/config.json"))
tc = cfg.pop("transform_config", None) or cfg["quantization_config"].get("transform_config")
cfg["quantization_config"]["transform_config"] = tc
json.dump(cfg, open(f"{mse}/config.json", "w"), indent=2, sort_keys=True)
print("restored tc into quantization_config;",
      "top-level removed:", "transform_config" not in cfg,
      "; targets:", len(tc["config_groups"]["rot"]["apply"][0]["targets"]))
EOF
venv/bin/python analysis/rotation/smoke_ab.py \
  models/Qwen3.8-27B-W4A16-rot-patch-selective-mse \
  analysis/rotation/smoke_mse.json --pins || true
