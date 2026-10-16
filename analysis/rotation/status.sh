#!/bin/bash
L=/home/hubbe/qwen-serving/analysis/rotation/logs/boot_patch1.log
ls -la "$L"
tail -2 "$L"
echo "transforms: $(grep -c LinearTransformMethod "$L")"
curl -sf -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:18020/health
pgrep -f "venv/bin/vllm serve" | head -1
