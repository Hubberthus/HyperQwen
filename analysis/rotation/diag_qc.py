#!/usr/bin/env python
import os
MODEL = "/home/hubbe/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound"
from compressed_tensors.quantization import QuantizationConfig
qc = QuantizationConfig.from_pretrained(MODEL)
print("qc.ignore type:", type(qc.ignore))
print("qc.ignore repr[:200]:", repr(qc.ignore)[:200])
print("qc fields:", list(qc.model_fields.keys()) if hasattr(qc,'model_fields') else "n/a")

# find the ignore field annotation
import inspect
try:
    ann = qc.model_fields["ignore"].annotation
    print("ignore field annotation:", ann)
except Exception as e:
    print("annotation err:", e)

# find where ignore is consumed (named_modules) in compressed_tensors
CT = "/home/hubbe/qwen-serving/venv/lib/python3.12/site-packages/compressed_tensors"
print("\n=== code consuming 'ignore' ===")
for root, _, files in os.walk(CT):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(root, fn)
        txt = open(p, encoding="utf-8", errors="ignore").read()
        for i, l in enumerate(txt.splitlines(), 1):
            s = l.strip()
            if "ignore" in s and ("named_modules" in s or "in name" in s or "re." in s or "match" in s.lower() or "for " in s):
                print("%s:%d: %s" % (p.replace(CT, "CT"), i, s[:110]))
