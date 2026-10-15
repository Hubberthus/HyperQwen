#!/usr/bin/env python
"""Locate the compressed-tensors compression code that fails on the Embedding."""
import os, re
CT = "/home/hubbe/qwen-serving/venv/lib/python3.12/site-packages/compressed_tensors"
hits = []
for root, _, files in os.walk(CT):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(root, fn)
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if "Compressing model" in line or "def compress" in line.lower():
                hits.append((p, i, line.strip()))
print("=== compress entrypoints ===")
for p, i, l in hits:
    print("%s:%d: %s" % (p.replace(CT, "CT"), i, l))

print("\n=== .weight access in lifecycle/forward (compression) ===")
for root, _, files in os.walk(CT + "/quantization"):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(root, fn)
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            s = line.strip()
            if ".weight" in s and ("module" in s.lower() or "for " in s.lower() or "getattr" in s.lower()):
                print("%s:%d: %s" % (p.replace(CT, "CT"), i, s[:120]))
