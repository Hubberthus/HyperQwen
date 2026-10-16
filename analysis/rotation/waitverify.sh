#!/bin/bash
L=/home/hubbe/qwen-serving/analysis/rotation/logs/boot_patch1.log
for i in $(seq 1 100); do
  if curl -sf -o /dev/null http://127.0.0.1:18020/health 2>/dev/null; then echo UP; break; fi
  sleep 5
done
echo "LinearTransformMethod count: $(grep -c LinearTransformMethod "$L")"
grep -nE "hadamard|LinearTransformMethod" "$L" | head -3
curl -sf -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:18020/health
PID=$(pgrep -f "venv/bin/vllm serve" | head -1)
echo "server_pid=$PID"
grep -E "KV cache size|Available KV cache" "$L" | tail -2
