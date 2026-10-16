#!/bin/bash
ps -eo pid,ppid,lstart,cmd | grep -E "boot.sh|poll.sh|wsl|vllm|qwen-server|start_qwen" | grep -v grep
