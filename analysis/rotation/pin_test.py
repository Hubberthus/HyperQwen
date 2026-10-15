#!/usr/bin/env python
"""Pin-only test: load AR patch dir in vLLM offline, count CompressedTensorsLinearTransformMethod."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch_ar as ar

from vllm import LLM
llm = LLM(model="models/Qwen3.8-27B-W4A16-rot-patch-AR", dtype="bfloat16", max_model_len=64,
          enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.85,
          enable_prefix_caching=False, seed=0)
model = ar.find_model_object(llm)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (
    CompressedTensorsLinearTransformMethod)
n_trans, n_linear, miss = 0, 0, []
want = set()
for L, rows in ar.build_targets().items():
    for base, tsuf in rows:
        want.add("layers.%d.%s" % (L, tsuf))
for name, mod in model.named_modules():
    if any(name.endswith("layers.%d.%s" % (L, tsuf)) for L in ar.LIN_LAYERS + ar.FULL_LAYERS
           for _, tsuf in []):
        pass
# count per-target: resolve fused part names
FUSED_PARTS = {}
for L, rows in ar.build_targets().items():
    for base, tsuf in rows:
        full = tsuf.split(".")[0]
        FUSED_PARTS.setdefault(L, set()).add(tsuf)
nm = dict(model.named_modules())
per_target = {}
for L, rows in ar.build_targets().items():
    for base, tsuf in rows:
        # exact part names in vLLM naming (no model. prefix)
        parts = {"linear_attn.in_proj_qkv": "in_proj_qkv", "linear_attn.in_proj_z": "in_proj_z",
                 "linear_attn.out_proj": "out_proj", "mlp.gate_proj": "gate_proj",
                 "mlp.up_proj": "up_proj", "mlp.down_proj": "down_proj",
                 "self_attn.q_proj": "q_proj", "self_attn.k_proj": "k_proj",
                 "self_attn.v_proj": "v_proj", "self_attn.o_proj": "o_proj"}[tsuf]
        matches = [n for n in nm if n.endswith("layers.%d.%s" % (L, parts))]
        assert len(matches) == 1, (tsuf, matches)
        cls = type(nm[matches[0]].quant_method).__name__
        per_target["%d.%s" % (L, tsuf)] = cls
        if cls == "CompressedTensorsLinearTransformMethod":
            n_trans += 1
        else:
            miss.append("%d.%s=%s" % (L, tsuf, cls))
n_linear = sum(1 for v in per_target.values() if v == "CompressedTensorsLinearMethod")
print("transform_methods=%d linear_methods=%d missing=%s" % (n_trans, n_linear, miss[:10]))
json.dump({"n_transform": n_trans, "n_linear": n_linear, "missing": miss},
          open(HERE / "pin_test.json", "w"), indent=1)
assert n_trans == 50, "expected 50 transform methods, got %d" % n_trans
print("PIN TEST PASS")
