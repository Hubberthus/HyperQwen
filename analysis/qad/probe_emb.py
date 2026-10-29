#!/usr/bin/env python
"""Probe the loaded student's embed_tokens module structure (one-off debug)."""
import json
import inspect
import torch
import torch.nn as nn
from pathlib import Path

REPO = Path("/home/hubbe/qwen-serving")
MODEL = REPO / "models/Qwen3.8-27B-W4A16-AutoRound"

import compressed_tensors.quantization.utils.helpers as _cthelp
import compressed_tensors.compressors.model_compressors.model_compressor as _ctmc
_orig = _cthelp.is_module_quantized


def _safe(module):
    w = getattr(module, "weight", None)
    if w is None or not isinstance(w, nn.Parameter):
        return False
    return _orig(module)


_cthelp.is_module_quantized = _safe
_ctmc.is_module_quantized = _safe
import transformers.modeling_utils as _tmu

_tmu.PreTrainedModel._initialize_missing_keys = lambda self, is_quantized=False: None

from transformers.utils.quantization_config import CompressedTensorsConfig

qc = CompressedTensorsConfig(**json.load(open(MODEL / "quantization_config.json")))
ig = list(qc.quantization_config.ignore or [])
for extra in ["embed_tokens", "model.language_model.embed_tokens", "model.embed_tokens",
              "mtp", "draft_lm_head"]:
    if extra not in ig:
        ig.append(extra)
qc.quantization_config.ignore = ig

from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    str(MODEL), quantization_config=qc, torch_dtype=torch.bfloat16,
    device_map="cuda:0", low_cpu_mem_usage=True)
tm = model.model.language_model
emb = tm.embed_tokens
print("emb type:", type(emb).__name__, type(emb).__mro__)
print("emb attrs:", [a for a in dir(emb) if not a.startswith("_")][:40])
print("has weight attr:", hasattr(emb, "weight"))
print("named_parameters:", [(n, tuple(p.shape)) for n, p in emb.named_parameters()][:5])
print("named_buffers:", [(n, tuple(b.shape) if hasattr(b, "shape") else b)
                         for n, b in emb.named_buffers()][:8])
try:
    print("forward src:")
    print(inspect.getsource(type(emb).forward)[:800])
except Exception as e:
    print("src err:", e)
