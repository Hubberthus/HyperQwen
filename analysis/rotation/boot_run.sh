#!/bin/bash
# boot+smoke one arm, fully serialized. usage: boot_run.sh <model_rel> <tag>
REPO=/home/hubbe/qwen-serving
cd "$REPO" || exit 1
bash analysis/rotation/shutdown.sh
LOG=analysis/rotation/logs/boot_$2.log; rm -f "$LOG"
setsid env MODEL=$REPO/$1 PORT=18020 WARMUP=0 \
  bash single-user/qwen-server.sh > "$LOG" 2>&1 &
WPID=$!
echo "wrapper_pid=$WPID tag=$2"
for i in $(seq 1 70); do
  sleep 5
  curl -sf -o /dev/null http://127.0.0.1:18020/health 2>/dev/null && { echo "HEALTH OK at $((i*5))s"; break; }
  kill -0 $WPID 2>/dev/null || { echo "WRAPPER DIED at $((i*5))s"; tail -5 "$LOG"; exit 1; }
done
PID=$(pgrep -f "venv/bin/vllm serve" | head -1)
echo "server_pid=$PID"
echo "cmdline: $(tr '\0' ' ' < /proc/$PID/cmdline | cut -c1-200)"
grep -E "GPU KV cache size|Available KV cache" "$LOG" | tail -2
# smoke only when SMOKE=1 (first boot of an arm)
if [ "${SMOKE:-0}" = 1 ]; then bash analysis/rotation/smoke_serve.sh || exit 1; fi
