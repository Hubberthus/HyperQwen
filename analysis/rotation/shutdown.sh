#!/bin/bash
# shutdown.sh : TERM the running vllm server (found by its venv path, never a
# blanket pkill), wait until it is gone and the GPU drains. Verifies as-found.
PIDS=$(pgrep -f "qwen-serving/venv/bin/vllm serve" || true)
[ -z "$PIDS" ] && { echo "no server running"; exit 0; }
echo "killing: $PIDS"
kill -TERM $PIDS 2>/dev/null || true
for i in $(seq 1 90); do
  pgrep -f "qwen-serving/venv/bin/vllm serve" >/dev/null || break
  sleep 2
done
if pgrep -f "qwen-serving/venv/bin/vllm serve" >/dev/null; then
  echo "still alive after 180s; KILL"
  pkill -KILL -f "qwen-serving/venv/bin/vllm serve"
  sleep 5
fi
pgrep -f "qwen-serving/venv/bin/vllm serve" >/dev/null && echo "STILL ALIVE" || echo "server down"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
