#!/bin/bash
cd /home/hubbe/qwen-serving
for f in config.json config.json.bak-quant config.json.bak-mtp; do
  p="models/Qwen3.8-27B-W4A16-rot-patch-selective-mse/$f"
  echo "== $f"
  venv/bin/python - "$p" <<'EOF'
import json, sys, os
if not os.path.exists(sys.argv[1]):
    print("MISSING"); raise SystemExit
c = json.load(open(sys.argv[1]))
tc = c.get("transform_config") or (c.get("quantization_config") or {}).get("transform_config")
if tc:
    groups = tc.get("config_groups", {})
    for name, g in groups.items():
        print("group", name, "type", g.get("type"), "n_apply_targets", sum(len(a.get("targets", [])) for a in g.get("apply", [])))
else:
    print("no transform_config")
EOF
done
echo "=== _mseonly_cfg.py (build-time transform writer)"
cat analysis/rotation/_mseonly_cfg.py
