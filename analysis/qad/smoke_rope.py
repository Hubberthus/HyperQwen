#!/usr/bin/env python
"""Smoke test: verify tm.rotary_emb fix resolves the 512 vs 1024 rope mismatch.

Loads the student model, runs a single forward through L0 with the model's own
rotary_emb (instead of a hand-rolled instance), and checks the cos/sin shapes
and the final hidden state shape. No shard checks, no LoRA wrapping — just the
rope path.
"""
import json
import torch
from pathlib import Path

REPO = Path("/home/hubbe/qwen-serving")
MODEL = REPO / "models/Qwen3.8-27B-W4A16-AutoRound"


def patch_ct():
    import torch.nn as nn
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


def load_student():
    import json as _json
    from transformers.utils.quantization_config import CompressedTensorsConfig
    qc = CompressedTensorsConfig(**_json.load(open(MODEL / "quantization_config.json")))
    base_ig = list(qc.quantization_config.ignore or [])
    for extra in ["embed_tokens", "model.language_model.embed_tokens", "model.embed_tokens",
                  "mtp", "draft_lm_head"]:
        if extra not in base_ig:
            base_ig.append(extra)
    qc.quantization_config.ignore = base_ig
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL), quantization_config=qc, torch_dtype=torch.bfloat16,
        device_map="cuda:0", low_cpu_mem_usage=True)
    return model


def decompress_packed_modules(model):
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization import QuantizationStatus
    n_done = 0
    with torch.no_grad():
        for name, m in model.named_modules():
            if not hasattr(m, "weight_packed") or hasattr(m, "weight"):
                continue
            sch = getattr(m, "quantization_scheme", None)
            args = getattr(sch, "weights", None) if sch is not None else None
            if args is None:
                continue
            num_bits = args.num_bits
            group_size = args.group_size
            shape = tuple(m.weight_shape.tolist())
            rows, cols = shape
            codes = unpack_from_int32(m.weight_packed.data, num_bits, shape, packed_dim=1)
            offset = 1 << (num_bits - 1)
            signed = codes.to(torch.float32) - offset
            groups = cols // group_size
            scale = m.weight_scale.data.to(torch.float32)
            w = signed.view(rows, groups, group_size) * scale.view(rows, groups, 1)
            w = w.reshape(rows, cols).to(torch.bfloat16)
            m.weight = torch.nn.Parameter(w, requires_grad=False)
            del m.weight_packed, m.weight_scale, m.weight_shape
            m.__dict__.pop("forward", None)
            m.quantization_status = QuantizationStatus.DECOMPRESSED
            n_done += 1
    print(f"decompress_packed_modules: {n_done} modules", flush=True)


def main():
    patch_ct()
    model = load_student()
    model.eval()
    tm = model.model.language_model
    print("tm children:", [n for n, _ in tm.named_children()], flush=True)
    decompress_packed_modules(model)

    rot = tm.rotary_emb
    print("tm.rotary_emb type:", type(rot).__name__, flush=True)

    torch.manual_seed(0)
    ids = torch.randint(1000, 200000, (1, 128), device="cuda")

    # Single forward through L0 only, with the model's own rotary_emb.
    with torch.no_grad():
        h = tm.embed_tokens(ids)
        print("embed_tokens out shape:", tuple(h.shape), flush=True)
        pos2 = torch.arange(ids.shape[1], device="cuda").view(1, -1).expand(ids.shape[0], -1)
        pos3 = pos2.reshape(1, 1, -1).expand(3, ids.shape[0], -1)
        cos, sin = rot(h, pos3)
        print("cos shape:", tuple(cos.shape), "sin shape:", tuple(sin.shape), flush=True)
        layer = tm.layers[0]
        try:
            out = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                        position_ids=pos2, past_key_values=None, use_cache=False)
            print("L0 forward OK, out shape:", tuple(out.shape), flush=True)
        except Exception as e:
            print("L0 forward FAILED:", type(e).__name__, str(e)[:400], flush=True)
            raise


if __name__ == "__main__":
    main()
