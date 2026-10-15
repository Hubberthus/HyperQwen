#!/usr/bin/env python
"""Probe: can vLLM 0.31 load the INT4 baseline in-process (V1 multiprocessing off),
expose the model object, and fire a forward hook on a target linear during generate?"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import sys, time, torch

MODEL = "/home/hubbe/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound"

import vllm
print("vllm", vllm.__version__, flush=True)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

t0 = time.time()
llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.92,
          max_model_len=2112, dtype="bfloat16", seed=0)
print("LLM loaded in %.1fs" % (time.time() - t0), flush=True)
print("alloc GB %.2f" % (torch.cuda.memory_allocated() / 1e9), flush=True)

import torch.nn as nn

found = []
def is_target_model(m):
    try:
        names = [n for n, _ in m.named_modules()]
    except Exception:
        return False
    return any(n.endswith("layers.0.linear_attn.in_proj_qkv") for n in names)

def walk(obj, path, depth, seen):
    if depth > 10 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (str, int, float, bytes, bool)) or obj is None:
        return
    if isinstance(obj, torch.Tensor):
        return
    if isinstance(obj, nn.Module):
        if is_target_model(obj):
            found.append((path, obj))
            return
        for n, c in obj.named_children():
            walk(c, path + "." + n, depth + 1, seen)
        return
    if isinstance(obj, (list, tuple, set)):
        for i, c in enumerate(obj):
            walk(c, path + "[%d]" % i, depth + 1, seen)
        return
    if isinstance(obj, dict):
        for k, c in obj.items():
            walk(c, path + "[%r]" % (k,), depth + 1, seen)
        return
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for k, c in d.items():
            walk(c, path + "." + k, depth + 1, seen)
    for k in (getattr(obj, "__slots__", None) or []):
        try:
            walk(getattr(obj, k), path + "." + k, depth + 1, seen)
        except Exception:
            pass

seen = set()
walk(llm, "llm", 0, seen)
print("FOUND %d candidate model objects" % len(found), flush=True)
for p, m in found[:5]:
    print("  path=%s type=%s" % (p, type(m).__name__), flush=True)
if not found:
    print("NO MODEL FOUND - hooks impossible via this path")
    sys.exit(2)
model = found[0][1]

mods = dict(model.named_modules())
target_name = "model.language_model.layers.0.linear_attn.in_proj_qkv"
if target_name not in mods:
    cands = [n for n in mods if n.endswith("in_proj_qkv")]
    print("target missing; in_proj_qkv candidates:", cands[:8], flush=True)
    target_name = cands[0]
m = mods[target_name]
print("target module: %s.%s" % (target_name, type(m).__name__), flush=True)
print("module attrs: in=%s out=%s" % (getattr(m, "in_features", None) or getattr(m, "input_size", None),
                                      getattr(m, "out_features", None) or getattr(m, "output_size", None)), flush=True)

cap = []
def hook(module, args, output):
    x = args[0] if args else None
    if x is not None:
        cap.append(x.detach().to("cpu").float())

h = m.register_forward_hook(hook)

prompts = [TokensPrompt(prompt_token_ids=[(i * 7 + j) % 248000 + 10 for j in range(64)]) for i in range(2)]
sp = SamplingParams(max_tokens=1, temperature=0.0)
t0 = time.time()
outs = llm.generate(prompts, sp)
print("generate done in %.1fs" % (time.time() - t0), flush=True)
print("captured %d hook firings; shapes %s" % (len(cap), [tuple(c.shape) for c in cap[:4]]), flush=True)
if cap:
    x = cap[0]
    print("X stats: shape %s mean %.4f std %.4f" % (tuple(x.shape), float(x.mean()), float(x.std())), flush=True)
h.remove()
print("PEAK GPU GB %.2f" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
print("PROBE_OK", flush=True)
