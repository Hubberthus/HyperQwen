#!/bin/bash
# Reboot the patch (tc restored to quantization_config) and smoke it.
cd /home/hubbe/qwen-serving
bash analysis/rotation/shutdown.sh
LOG=analysis/rotation/logs/boot_p1.log; rm -f "$LOG"
setsid env MODEL=$PWD/models/Qwen3.8-27B-W4A16-rot-patch-selective-mse PORT=18020 WARMUP=0 \
  bash single-user/qwen-server.sh > "$LOG" 2>&1 &
WPID=$!
for i in $(seq 1 70); do
  sleep 5
  curl -sf -o /dev/null http://127.0.0.1:18020/health 2>/dev/null && { echo "HEALTH OK at $((i*5))s"; break; }
  kill -0 $WPID 2>/dev/null || { echo "WRAPPER DIED at $((i*5))s"; tail -5 "$LOG"; exit 1; }
done
bash analysis/rotation/smoke_serve.sh
