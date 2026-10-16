#!/bin/bash
cd /home/hubbe/qwen-serving
PID=$(pgrep -f "venv/bin/vllm serve" | head -1)
echo "server_pid=$PID"
echo "cmdline_head: $(tr '\0' ' ' < /proc/$PID/cmdline | cut -c1-160)"
grep -E "KV cache size|Available KV cache" analysis/rotation/logs/boot_p1.log | tail -2
export SERVER_PID=$PID PORT=18020
./venv/bin/python analysis/rotation/ppl_eval.py patch1
