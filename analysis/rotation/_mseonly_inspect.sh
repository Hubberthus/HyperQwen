#!/bin/bash
for d in Qwen3.8-27B-W4A16-rot-patch Qwen3.8-27B-W4A16-rot-patch-AR Qwen3.8-27B-W4A16-AutoRound; do
echo "== $d"
python3 /home/hubbe/qwen-serving/analysis/rotation/_mseonly_cfg.py "$d"
done
