#!/usr/bin/env python
"""Probe 2: find the vLLM model object (gc scan fallback), enumerate target-layer
module names, verify hook firing + input sharing for the 50 target linears."""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import sys, time, gc, torch

MODEL = "/home/hubbe/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound"

import vllm
print("vllm", vllm.__version__, flush=True)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

t0 = time.time()
llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.92,
          max_model_len=2112, dtype="bfloat16", seed=0)
print("LLM loaded in %.1fs" % (time.time() - t0), flush=True)

import torch.nn as nn

def has_targets(m):
    try:
        names = [n for n, _ in m.named_modules()]
    except Exception:
        return False
    return any(n.endswith("layers.0.linear_attn.in_proj_qkv") for n in names)

# strategy 1: known attribute paths
cands = []
for path in [
    "llm_engine.engine_core.engine_core.model_executor.driver_worker.model",
    "llm_engine.engine_core.model_executor.driver_worker.model",
    "llm_engine.model_executor.driver_worker.model",
    "llm_engine.engine_core.engine.model_executor.driver_worker.model",
    "llm_engine.engine_core.engine_core_worker.model",
    "llm_engine.engine_core.engine_core.model",
]:
    obj = llm
    ok = True
    for attr in path.split(".")[1:]:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            ok = False
            break
    if ok:
        cands.append((path, obj))
        print("PATH OK:", path, type(obj).__name__, flush=True)

# strategy 2: gc scan
print("gc scan...", flush=True)
found = []
for o in gc.get_objects():
    if isinstance(o, nn.Module) and has_targets(o):
        found.append(o)
        if len(found) >= 3:
            break
print("gc found %d" % len(found), flush=True)
for o in found:
    cands.append(("gc", o))

models = []
for p, o in cands:
    if any(o is m for m in models):
        continue
    models.append(o)
print("distinct model objects: %d" % len(models), flush=True)
if not models:
    print("NO MODEL FOUND")
    sys.exit(2)
model = models[0]

mods = dict(model.named_modules())
print("total modules: %d" % len(mods), flush=True)
for L in (0, 27):
    print("=== layer %d linear-ish modules ===" % L, flush=True)
    for n, m in mods.items():
        if n.startswith("model.language_model.layers.%d." % L) or n.startswith("model.layers.%d." % L):
            if isinstance(m, nn.Linear) or "Linear" in type(m).__name__ or "proj" in n:
                print("   %-70s %s" % (n, type(m).__name__), flush=True)

# hook the candidate input modules, record fingerprints
watch = {}
for L in (0, 27):
    for suffix in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                   "self_attn.qkv_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.o_proj", "mlp.gate_up_proj", "mlp.gate_proj", "mlp.up_proj",
                   "mlp.down_proj"):
        for prefix in ("model.language_model.layers.%d." % L, "model.layers.%d." % L):
            n = prefix + suffix
            if n in mods:
                watch[n] = []
print("watching %d modules" % len(watch), flush=True)

def mk(n):
    def hook(module, args, output):
        x = args[0] if args else None
        if x is None:
            return
        d = x.detach()
        watch[n].append((tuple(d.shape), d.data_ptr(), float(d.float().sum())))
    return hook

hs = []
for n in watch:
    hs.append(mods[n].register_forward_hook(mk(n)))

prompts = [TokensPrompt(prompt_token_ids=[(i * 13 + j) % 248000 + 10 for j in range(512)]) for i in range(4)]
sp = SamplingParams(max_tokens=1, temperature=0.0)
t0 = time.time()
llm.generate(prompts, sp)
print("generate done in %.1fs" % (time.time() - t0), flush=True)
for h in hs:
    h.remove()

for n in sorted(watch):
    rec = watch[n]
    print("%-72s firings=%d first=%s" % (n, len(rec), rec[0] if rec else None), flush=True)

# sharing check: same (shape, ptr, sum) sequence?
names = sorted(watch)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = watch[names[i]], watch[names[j]]
        if len(a) == len(b) and len(a) > 0 and a == b:
            print("SHARED INPUT: %s <-> %s" % (names[i], names[j]), flush=True)
print("PEAK GPU GB %.2f" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
print("PROBE2_OK", flush=True)
