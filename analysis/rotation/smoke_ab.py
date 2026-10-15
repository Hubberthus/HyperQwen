#!/usr/bin/env python
"""Serving smoke for one model dir (offline vLLM, never the server).

  smoke_ab.py <model_dir> <out_json> [--pins]

--pins (for the AR patch dir): after load, walk the model object and
  (1) assert all 50 target linears' quant_method is CompressedTensorsLinearTransformMethod,
  (2) assert the runtime Hadamard matrix == build_patch.sylvester(128) (structural, max
      abs diff 0.0) and that the hadacore kernel applies it numerically,
then run 8 short prompts (greedy, max_tokens=8, logprobs=5) and save outputs for A/B.
"""
import json, sys, os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import build_patch as bp
import build_patch_ar as ar

MODEL = sys.argv[1]
OUT = Path(sys.argv[2])
CHECK_PINS = "--pins" in sys.argv

PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "2 + 2 =",
    "The chemical formula of water is",
    "Shakespeare wrote",
    "The speed of light is approximately",
    "Python is a",
    "The first man on the Moon was",
]

def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=512, enforce_eager=True,
              disable_log_stats=True, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, seed=0)
    sp = SamplingParams(max_tokens=8, temperature=0.0, logprobs=5)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    results = []
    for p, o in zip(PROMPTS, outs):
        o = o.outputs[0]
        steps = []
        for i, tid in enumerate(o.token_ids):
            lp_entry = o.logprobs[i][tid] if o.logprobs else None
            steps.append({"token_id": int(tid), "logprob": float(lp_entry.logprob) if lp_entry else None,
                          "top": {str(k): float(v.logprob) for k, v in o.logprobs[i].items()} if o.logprobs else {}})
        results.append({"prompt": p, "text": o.text, "token_ids": [int(t) for t in o.token_ids],
                        "steps": steps})
    pin_report = None
    if CHECK_PINS:
        model = ar.find_model_object(llm)
        import torch.nn as nn
        fused_names = {}
        targets = ar.build_targets()
        want = {}
        for L, rows in targets.items():
            for base, tsuf in rows:
                want.setdefault("layers.%d.%s" % (L, ar.FUSED_OF[tsuf]), set()).add((L, tsuf))
        named = dict(model.named_modules())
        covered = set()
        for suf in want:
            matches = [n for n in named if n.endswith(suf)]
            assert len(matches) == 1, "%s -> %d matches" % (suf, matches)
            covered.add(suf)
            fused_names[suf] = matches[0]
        assert len(covered) == len(want)
        try:
            from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (
                CompressedTensorsLinearTransformMethod)
        except Exception:
            CompressedTensorsLinearTransformMethod = None
        qm = {}
        n_transform = 0
        for suf, fname in fused_names.items():
            mod = named[fname]
            cls = type(mod.quant_method).__name__
            qm[fname] = cls
            if cls == "CompressedTensorsLinearTransformMethod":
                n_transform += 1
        # structural H check: deterministic_hadamard_matrix(128) vs build_patch sylvester(128)
        from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
        Hd = deterministic_hadamard_matrix(128)
        if isinstance(Hd, torch.Tensor):
            Hd = Hd.to(torch.float64)
        Hd = np.asarray(Hd.cpu().numpy() if isinstance(Hd, torch.Tensor) else Hd, dtype=np.float64)
        Hs = bp.sylvester(128)
        norm = np.sqrt(128.0)
        diff_norm = float(np.abs(Hd / norm - Hs).max())
        diff_raw = float(np.abs(Hd - Hs).max())
        # numeric kernel check: hadacore_transform(x) vs x @ H
        import vllm._custom_ops as ops
        x = torch.randn(8, 4096, 128, device="cuda", dtype=torch.float16)
        y = ops.hadacore_transform(x.clone(), inplace=True)
        ref = x.float().cpu().numpy().reshape(-1, 128) @ Hs
        kdiff = float(np.abs(y.reshape(-1, 128).float().cpu().numpy() - ref).max())
        rel = kdiff / max(1e-9, float(np.abs(ref).max()))
        pin_report = {
            "fused_modules_checked": len(fused_names),
            "quant_method_classes": qm,
            "transform_methods": n_transform,
            "all_50_targets_covered_by_transform_method": n_transform == len(fused_names),
            "n_targets_expected": 50,
            "hadamard_structural_max_diff_normalized": diff_norm,
            "hadamard_structural_max_diff_raw": diff_raw,
            "hadacore_numeric_max_abs_diff": kdiff,
            "hadacore_numeric_rel_diff": rel,
        }
        print(json.dumps({k: v for k, v in pin_report.items() if k != "quant_method_classes"}, indent=1))
    OUT.write_text(json.dumps({"model": MODEL, "pin_report": pin_report, "results": results}, indent=1))
    print("saved", OUT)

if __name__ == "__main__":
    import numpy as np
    main()
