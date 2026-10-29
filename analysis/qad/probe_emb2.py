#!/usr/bin/env python
"""Probe: load student WITHOUT the is_module_quantized patch; check embed decompression."""
import json, inspect, torch
from pathlib import Path

REPO = Path("/home/hubbe/qwen-serving")
MODEL = REPO / "models/Qwen3.8-27B-W4A16-AutoRound"

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
print("emb type:", type(emb).__name__)
print("quantization_status:", getattr(emb, "quantization_status", None))
print("named_parameters:", [(n, tuple(p.shape)) for n, p in emb.named_parameters()][:5])
print("has weight:", hasattr(emb, "weight"))
if hasattr(emb, "weight"):
    print("weight shape:", tuple(emb.weight.shape), emb.weight.dtype)
    gp = tm.layers[0].mlp.gate_proj
    print("gate_proj type:", type(gp).__name__, "has weight:", hasattr(gp, "weight"),
          "params:", [(n, tuple(p.shape)) for n, p in gp.named_parameters()][:6])
    # quick forward
    ids = torch.randint(1000, 200000, (1, 64), device="cuda")
    with torch.no_grad():
        out = tm(input_ids=ids, use_cache=False)
    print("forward OK, last_hidden:", tuple(out.last_hidden_state.shape))
