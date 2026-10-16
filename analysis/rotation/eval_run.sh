#!/bin/bash
TAG=$1
exec > /home/hubbe/qwen-serving/analysis/rotation/logs/eval_$TAG.log 2>&1
cd /home/hubbe/qwen-serving
PID=$(pgrep -f "venv/bin/vllm serve" | head -1)
echo "server_pid=$PID"
grep -E "GPU KV cache size" analysis/rotation/logs/boot_$TAG.log | tail -1
export SERVER_PID=$PID PORT=18020
./venv/bin/python analysis/rotation/ppl_eval.py $TAG
echo EVAL_EXIT=$?
