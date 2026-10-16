#!/bin/bash
exec > /home/hubbe/qwen-serving/analysis/rotation/logs/eval_current.log 2>&1
bash /home/hubbe/qwen-serving/analysis/rotation/run_eval.sh
echo EVAL_EXIT=$?
