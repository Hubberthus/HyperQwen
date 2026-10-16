#!/bin/bash
# Kill any running server, then boot-watch the patch again.
bash /home/hubbe/qwen-serving/analysis/rotation/shutdown.sh
bash /home/hubbe/qwen-serving/analysis/rotation/boot_watch.sh
