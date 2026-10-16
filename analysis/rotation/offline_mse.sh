#!/bin/bash
set -e
cd /home/hubbe/qwen-serving
bash analysis/rotation/shutdown.sh
echo "=== offline smoke: -mse (tc now top-level AND in quantization_config? no - top-level only now)"
venv/bin/python analysis/rotation/smoke_ab.py \
  models/Qwen3.8-27B-W4A16-rot-patch-selective-mse \
  analysis/rotation/smoke_mse.json --pins || true
