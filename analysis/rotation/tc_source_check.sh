#!/bin/bash
cd /home/hubbe/qwen-serving
V=venv/lib/python3.12/site-packages/vllm
echo "=== grep transform_config in vllm source"
grep -rln "transform_config" $V/model_executor/layers/quantization $V/model_executor/models 2>/dev/null | head
grep -rn "LinearTransformMethod" $V --include="*.py" -l 2>/dev/null | head
echo "=== AR serving evidence: any boot logs with LinearTransformMethod"
grep -rlc "LinearTransformMethod" analysis/rotation/logs/ 2>/dev/null
grep -rn "transform" analysis/rotation/logs/*.log 2>/dev/null | grep -iv "bytecode\|Dynamo\|Transformer" | head -5
