#!/bin/bash
# boot.sh <model_dir_rel_to_repo> <tag> : cold-boot the single-user server on
# port 18020 with default bool knobs (SPEC=mtp CTX=fast PREFIX_CACHE off,
# WARMUP=0), log to analysis/rotation/logs/boot_<tag>.log. Returns the
# wrapper's pid; health-wait.sh polls until /health is 200.
REPO=/home/hubbe/qwen-serving
MODEL="$REPO/$1"; TAG="$2"
mkdir -p "$REPO/analysis/rotation/logs"
cd "$REPO" || exit 1
setsid env MODEL="$MODEL" PORT=18020 WARMUP=0 \
  bash single-user/qwen-server.sh > "analysis/rotation/logs/boot_$TAG.log" 2>&1 &
echo "wrapper_pid=$!"
