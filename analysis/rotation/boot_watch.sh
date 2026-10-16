#!/bin/bash
# One-shot: boot, then watch the log until health OK or the engine dies.
REPO=/home/hubbe/qwen-serving
cd "$REPO" || exit 1
LOG=$REPO/analysis/rotation/logs/boot_p1.log
rm -f "$LOG"
setsid env MODEL=$REPO/models/Qwen3.8-27B-W4A16-rot-patch-selective-mse PORT=18020 WARMUP=0 \
  bash single-user/qwen-server.sh > "$LOG" 2>&1 &
WPID=$!
echo "wrapper_pid=$WPID"
for i in $(seq 1 75); do
  sleep 5
  if ! kill -0 $WPID 2>/dev/null; then
    echo "WRAPPER DIED at $((i*5))s"; break
  fi
  if curl -sf -o /dev/null http://127.0.0.1:18020/health 2>/dev/null; then
    echo "HEALTH OK at $((i*5))s"
    break
  fi
done
echo "=== log size: $(stat -c %s "$LOG" 2>/dev/null) bytes"
echo "=== transforms: $(grep -c LinearTransformMethod "$LOG")"
tail -8 "$LOG"
