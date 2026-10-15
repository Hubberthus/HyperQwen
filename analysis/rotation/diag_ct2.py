#!/usr/bin/env python
import os
CT = "/home/hubbe/qwen-serving/venv/lib/python3.12/site-packages/compressed_tensors"
print("=== def is_module_quantized ===")
for root, _, files in os.walk(CT):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(root, fn)
        txt = open(p, encoding="utf-8", errors="ignore").read()
        for i, l in enumerate(txt.splitlines(), 1):
            if "def is_module_quantized" in l:
                lines = txt.splitlines()
                for j in range(i - 1, min(i + 18, len(lines))):
                    print("%s:%d: %s" % (p.replace(CT, "CT"), j + 1, lines[j]))
                print("---")
print("\n=== imports of is_module_quantized in model_compressor ===")
mc = os.path.join(CT, "compressors/model_compressors/model_compressor.py")
for i, l in enumerate(open(mc).read().splitlines(), 1):
    if "is_module_quantized" in l or "compress_module" in l:
        print("%d: %s" % (i, l))
print("\n=== pack_quantized/base.py compress (40-95) ===")
pq = os.path.join(CT, "compressors/pack_quantized/base.py")
lines = open(pq).read().splitlines()
for j in range(39, min(95, len(lines))):
    print("%d: %s" % (j + 1, lines[j]))
