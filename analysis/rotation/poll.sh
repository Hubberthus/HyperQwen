#!/bin/bash
# poll.sh <tag> <model_abs_dir> : wait for /health, then print pid, cmdline
# and the KV-pool line from the boot log so the caller can verify a FRESH boot.
REPO=/home/hubbe/qwen-serving
TAG="$1"; MODEL="$2"
LOG="$REPO/analysis/rotation/logs/boot_$TAG.log"
for i in $(seq 1 180); do
  if curl -sf -o /dev/null http://127.0.0.1:18020/health; then
    PID=$(pgrep -f "venv/bin/vllm serve $MODEL" | head -1)
    echo "HEALTHY tag=$TAG server_pid=$PID"
    echo "CMDLINE: $(tr '\0' ' ' < /proc/$PID/cmdline)"
    grep -E "KV cache is configured|KV cache size|Available KV cache|maximum concurrency|memory utilization" "$LOG" | tail -4
    exit 0
  fi
  sleep 5
done
echo "TIMEOUT waiting for health, tag=$TAG; log tail:" >&2
tail -30 "$LOG" >&2
exit 1
