#!/usr/bin/env bash
# Negative-control copy: models/Qwen3.8-27B-W4A16-rot-patch -> analysis/rotation/tmp_negctrl
# (original naive dir stays read-only; we serve only the copy, then delete it)
set -e
cd /home/hubbe/qwen-serving
if [ ! -d analysis/rotation/tmp_negctrl ]; then
  mkdir -p analysis/rotation/tmp_negctrl
  cp -a models/Qwen3.8-27B-W4A16-rot-patch/. analysis/rotation/tmp_negctrl/
fi
echo "negctrl copy ready: $(du -sb analysis/rotation/tmp_negctrl | cut -f1) bytes"

echo "=== SMOKE 1: rot-patch-AR (with transform_config) + pins ==="
venv/bin/python analysis/rotation/smoke_ab.py models/Qwen3.8-27B-W4A16-rot-patch-AR \
    analysis/rotation/smoke_ar.json --pins
echo "SMOKE1_EXIT=$?"

echo "=== SMOKE 2: baseline (reference) ==="
venv/bin/python analysis/rotation/smoke_ab.py models/Qwen3.8-27B-W4A16-AutoRound \
    analysis/rotation/smoke_baseline.json
echo "SMOKE2_EXIT=$?"

echo "=== SMOKE 3: naive rot-patch copy (negative control, NO transform_config) ==="
venv/bin/python analysis/rotation/smoke_ab.py analysis/rotation/tmp_negctrl \
    analysis/rotation/smoke_negctrl.json
echo "SMOKE3_EXIT=$?"
