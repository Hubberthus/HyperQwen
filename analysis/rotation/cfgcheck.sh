#!/bin/bash
cd /home/hubbe/qwen-serving
for m in models/Qwen3.8-27B-W4A16-rot-patch-selective-mse models/Qwen3.8-27B-W4A16-rot-patch-AR models/Qwen3.8-27B-W4A16-AutoRound; do
  echo "== $m"
  venv/bin/python - "$m" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
print("top keys:", sorted(c.keys()))
tc = c.get("transform_config") or (c.get("quantization_config") or {}).get("transform_config")
print("transform_config:", json.dumps(tc)[:400] if tc else None)
EOF
done
grep -n "transform" analysis/rotation/_mseonly_cfg.py | head -30
