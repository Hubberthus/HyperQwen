#!/usr/bin/env python
"""Probe: can transformers load the compressed-tensors qwen3_5 INT4 checkpoint on GPU
and fire a forward hook on a target linear? Memory-frugal (14GB model, ~15GB headroom)."""
import sys, time, json, traceback
import torch
import torch.nn as nn

MODEL = "/home/hubbe/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound"

# --- defensive patch: a module with no `.weight` Parameter cannot be a quantized
# linear; keep it out of the compressed-tensors compression/decompression set.
# (Workaround for a custom Embedding module in qwen3_5 that carries a weight
#  scheme but has no `.weight` attribute -> 'Embedding' object has no attr 'weight'.)
import compressed_tensors.quantization.utils.helpers as _cthelp
import compressed_tensors.compressors.model_compressors.model_compressor as _ctmc
_orig_isq = _cthelp.is_module_quantized
def _isq_safe(module):
    w = getattr(module, "weight", None)
    if w is None or not isinstance(w, nn.Parameter):
        return False
    return _orig_isq(module)
_cthelp.is_module_quantized = _isq_safe
_ctmc.is_module_quantized = _isq_safe

# --- the checkpoint is COMPLETE (works in vLLM). The only "missing" keys are
# arch-mismatch artifacts (mtp.*, bare-LLM name mismatch). Skip missing-key init,
# which otherwise calls qwen3_5._init_weights -> base _init_weights -> crashes on
# the custom Embedding (no .weight). Language-model weights all load from disk.
import transformers.modeling_utils as _tmu
_tmu.PreTrainedModel._initialize_missing_keys = lambda self, is_quantized=False: None

def main():
    print("cuda:", torch.cuda.is_available(),
          "total_mem_GB: %.2f" % (torch.cuda.get_device_properties(0).total_memory / 1e9), flush=True)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    print("model_type:", cfg.model_type, "arch:", cfg.architectures, flush=True)

    # Try candidate auto-classes in order.
    import transformers
    candidates = []
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            candidates.append((name, cls))

    # Root-cause fix: the config's ignore list omits embed_tokens (and mtp), so
    # compressed-tensors quantizes+packs the embedding, consuming its .weight and
    # wrapping it with quantized_forward -> forward crashes. Add them to ignore.
    from compressed_tensors.quantization import QuantizationConfig
    qc = QuantizationConfig.from_pretrained(MODEL)
    base_ig = list(qc.ignore) if qc.ignore else []
    for extra in ["embed_tokens", "model.language_model.embed_tokens",
                  "model.embed_tokens", "mtp", "draft_lm_head"]:
        if extra not in base_ig:
            base_ig.append(extra)
    qc.ignore = base_ig
    print("ignore entries now: %d (added embed_tokens/mtp)" % len(base_ig), flush=True)

    model = None
    for name, cls in candidates:
        try:
            t0 = time.time()
            model = cls.from_pretrained(
                MODEL, quantization_config=qc, torch_dtype=torch.bfloat16,
                device_map="cuda:0", low_cpu_mem_usage=True)
            print("LOADED via %s in %.1fs  GPU_mem_GB=%.2f" % (name, time.time() - t0,
                                                            torch.cuda.memory_allocated() / 1e9), flush=True)
            break
        except Exception as e:
            print("FAILED %s: %s: %s" % (name, type(e).__name__, str(e)[:160]), flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
    if model is None:
        print("ALL LOAD ATTEMPTS FAILED"); sys.exit(2)
    model.eval()

    # locate a target linear
    target = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    m = model
    for part in target.split("."):
        m = getattr(m, part)
    print("target:", type(m).__name__,
          "in_features=", getattr(m, "in_features", None),
          "out_features=", getattr(m, "out_features", None), flush=True)

    cap = []
    def hook(mod, args, kwargs, output):
        x = args[0] if args else kwargs.get("input")
        cap.append(x.detach().to("cpu").float())
    h = m.register_forward_hook(hook)

    ids = torch.arange(1, 9, device="cuda").view(1, -1)
    with torch.no_grad():
        try:
            out = model(input_ids=ids)
        except Exception as e:
            print("forward failed:", type(e).__name__, str(e)[:200], flush=True)
            raise
    print("forward OK; captured shapes:", [tuple(c.shape) for c in cap], flush=True)
    if cap:
        H = cap[0].T @ cap[0]
        print("H (X^T X) shape:", tuple(H.shape), "diag_sum=%.4g" % float(H.diagonal().sum()), flush=True)
    print("PEAK_GPU_GB=%.2f" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print("PROBE_OK", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
