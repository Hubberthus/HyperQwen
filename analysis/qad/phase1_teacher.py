#!/usr/bin/env python
"""QAD pilot Phase 1: teacher target generation (analysis/qad/phase1_teacher.py).

The BF16 teacher (models-fp16/Qwen3.8-27B, 52 GB over 18 shards) cannot sit in
RAM (31 GB) or VRAM (24 GB), so it never loads whole. Instead a layer-major
streaming loop:

  1. Load the STUDENT (W4A16, 14 GB) once with the validated machinery from
     layerwise_validate.py, capture the exact position_embeddings (cos, sin)
     its own forward passes to layer 0 for 2048 positions, save to disk.
  2. Build the teacher text model skeleton on the meta device (0 bytes), then
     for each decoder layer: materialize on cuda, load its weights from the
     safetensors shards, run the batch through it, capture the layer output,
     write it to disk as safetensors, free the layer.
  3. Embedding is streamed the same way (it is a plain bf16 tensor here).

Calibration: 256 sequences x 2048 tokens, windows cut from the SAME corpus the
ppl harness uses (wikitext-2-raw-v1 test parquet, joined, student tokenizer,
add_special_tokens=False). The corpus is 297,053 tokens < 256*2048, so windows
overlap: 256 window starts spread evenly across the corpus (linspace).

Outputs (analysis/qad/):
  calib_ids.safetensors        (256, 2048) int64 token ids
  pos_emb.safetensors          cos, sin as captured from the student forward
  teacher_targets/L{ii}_b{bb}.safetensors  layer ii output, batch bb (16 seqs)
  phase1_teacher.json          run record (conventions: json next to script)
"""
import json, time
from pathlib import Path
import torch

REPO = Path("/home/hubbe/qwen-serving")
TEACHER = REPO / "models-fp16/Qwen3.8-27B"
STUDENT = REPO / "models/Qwen3.8-27B-W4A16-AutoRound"
OUT = REPO / "analysis" / "qad"
TT = OUT / "teacher_targets"
N_SEQ, N_TOK, BATCH = 256, 2048, 16


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
    qc = CompressedTensorsConfig(**_json.load(open(STUDENT / "quantization_config.json")))
    base_ig = list(qc.quantization_config.ignore or [])
    for extra in ["embed_tokens", "model.language_model.embed_tokens", "model.embed_tokens",
                  "mtp", "draft_lm_head"]:
        if extra not in base_ig:
            base_ig.append(extra)
    qc.quantization_config.ignore = base_ig
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        str(STUDENT), quantization_config=qc, torch_dtype=torch.bfloat16,
        device_map="cuda:0", low_cpu_mem_usage=True)
    # decompress packed embeddings (from layerwise_validate.py)
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization import QuantizationStatus
    with torch.no_grad():
        for name, m in model.named_modules():
            if not isinstance(m, torch.nn.Embedding) or hasattr(m, "weight"):
                continue
            sch = getattr(m, "quantization_scheme", None)
            args = getattr(sch, "weights", None) if sch is not None else None
            if args is None:
                continue
            nb, gs = args.num_bits, args.group_size
            rows, cols = (int(v) for v in m.weight_shape.tolist())
            codes = unpack_from_int32(m.weight_packed.data, nb, (rows, cols), packed_dim=1)
            # this checkpoint's 8-bit tensors (embed_tokens, lm_head) are
            # ASYMMETRIC with zero-point 0 — validated against the FP16 teacher
            # (corr=1.0000 unsigned vs 0.6547 with the symmetric offset). The
            # 4-bit linears are symmetric (offset 2^(nb-1), pin_check==0).
            # Apply the offset only for symmetric schemes.
            if getattr(args, "symmetric", True) and nb <= 4:
                signed = codes.to(torch.float32) - (1 << (nb - 1))
            else:
                signed = codes.to(torch.float32)
            groups = cols // gs
            scale = m.weight_scale.data.to(torch.float32)
            w = (signed.view(rows, groups, gs) * scale.view(rows, groups, 1))
            w = w.reshape(rows, cols).to(torch.bfloat16)
            m.weight = torch.nn.Parameter(w, requires_grad=False)
            del m.weight_packed, m.weight_scale, m.weight_shape
            m.__dict__.pop("forward", None)
            m.quantization_status = QuantizationStatus.DECOMPRESSED
    # wrap packed linears with the differentiable PackedLinearFn
    import types
    from layerwise_validate import PackedLinearFn
    n = 0
    for name, m in model.named_modules():
        if not hasattr(m, "weight_packed") or not hasattr(m, "quantization_scheme"):
            continue
        sch = m.quantization_scheme
        if sch is None or sch.weights is None:
            continue

        def fwd(self, x, _nb=sch.weights.num_bits, _gs=sch.weights.group_size):
            return PackedLinearFn.apply(x, self.weight_packed, self.weight_scale,
                                        self.weight_shape, getattr(self, "bias", None),
                                        _nb, _gs)
        m.forward = types.MethodType(fwd, m)
        n += 1
    print(f"student: wrapped {n} packed linears", flush=True)
    return model


def build_calib():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(STUDENT))
    txt = "".join(pq.read_table(
        REPO / "bench/quality-data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    ).column("text").to_pylist())
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    L = len(ids)
    # 256 window starts spread evenly (linspace) across the corpus
    import numpy as np
    starts = np.linspace(0, L - N_TOK, N_SEQ).astype(int).tolist()
    assert len(set(starts)) == N_SEQ
    win = torch.tensor([ids[s:s + N_TOK] for s in starts], dtype=torch.long)
    from safetensors.torch import save_file
    save_file({"ids": win}, str(OUT / "calib_ids.safetensors"))
    print(f"calib: {N_SEQ} x {N_TOK} windows from {L} corpus tokens "
          f"(first start {starts[0]}, last {starts[-1]})", flush=True)
    return win


def capture_pos_emb(student):
    """Capture (cos, sin) exactly as the student's own forward hands them to
    layer 0 — never hand-rolled (partial rotary; see layerwise_validate.py)."""
    tm = student.model.language_model
    cap = {}
    def pos_hook(mod, args, kwargs):
        cap["pe"] = kwargs.get("position_embeddings")
    ph = tm.layers[0].register_forward_pre_hook(pos_hook, with_kwargs=True)
    ids = torch.zeros((1, N_TOK), dtype=torch.long, device="cuda")
    with torch.no_grad():
        tm(input_ids=ids, use_cache=False)
    ph.remove()
    cos, sin = cap["pe"]
    from safetensors.torch import save_file
    save_file({"cos": cos.detach().cpu(), "sin": sin.detach().cpu()},
              str(OUT / "pos_emb.safetensors"))
    print("pos_emb shapes:", cos.shape, sin.shape, cos.dtype, flush=True)
    return cos, sin


def teacher_layer_stream():
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = AutoConfig.from_pretrained(str(TEACHER))
    scfg = AutoConfig.from_pretrained(str(STUDENT))
    tcfg = getattr(cfg, "text_config", cfg)
    record = {"teacher_cfg": str(tcfg.__class__.__name__)}
    # the text configs must match: layer count / dims / rope / GDN settings
    keys = ["num_hidden_layers", "hidden_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "intermediate_size",
            "rope_theta", "rope_scaling", "partial_rotary_factor"]
    diffs = {}
    for k in keys:
        a, b = getattr(tcfg, k, None), getattr(scfg.text_config if hasattr(scfg, "text_config") else scfg, k, None)
        if a != b:
            diffs[k] = [a, b]
    record["text_cfg_diffs"] = diffs
    print("text cfg diffs vs student:", diffs, flush=True)
    assert not diffs, f"teacher/student text configs differ: {diffs}"

    with init_empty_weights():
        full = AutoModelForImageTextToText.from_config(cfg, torch_dtype=torch.bfloat16)
    tm = full.model.language_model
    n_layers = len(tm.layers)
    print("teacher text model:", type(tm).__name__, n_layers, "layers", flush=True)

    wmap = json.load(open(TEACHER / "model.safetensors.index.json"))["weight_map"]

    def load_into(module, prefix, dst="cuda"):
        sd = {}
        shards = sorted(set(wmap[k] for k in wmap if k.startswith(prefix)))
        for sh in shards:
            with safe_open(str(TEACHER / sh), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        sd[k[len(prefix):]] = f.get_tensor(k)
        missing, unexpected = module.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if "inv_freq" not in k]
        assert not real_missing, f"{prefix}: missing {real_missing[:5]}"
        assert not unexpected, f"{prefix}: unexpected {unexpected[:5]}"
        return len(sd)

    cos, sin = None, None
    from safetensors.torch import load_file
    p = load_file(str(OUT / "pos_emb.safetensors"), device="cuda")
    cos, sin = p["cos"], p["sin"]
    ids = load_file(str(OUT / "calib_ids.safetensors"), device="cuda")["ids"]

    t0 = time.time()
    n_batches = N_SEQ // BATCH
    TT.mkdir(parents=True, exist_ok=True)
    emb_prefix = "model.language_model.embed_tokens."
    tm.embed_tokens.to_empty(device="cuda")
    print("embed loaded keys:", load_into(tm.embed_tokens, emb_prefix), flush=True)
    pos2 = torch.arange(N_TOK, device="cuda").view(1, -1).expand(BATCH, -1)
    for b in range(n_batches):
        with torch.no_grad():
            h = tm.embed_tokens(ids[b * BATCH:(b + 1) * BATCH].cuda())
            for i in range(n_layers):
                pre = f"model.language_model.layers.{i}."
                lay = tm.layers[i]
                lay.to_empty(device="cuda")
                nk = load_into(lay, pre)
                h = lay(h, position_embeddings=(cos, sin), attention_mask=None,
                        position_ids=pos2, past_key_values=None, use_cache=False)
                if isinstance(h, tuple):
                    h = h[0]
                save_file({"h": h.detach().to(torch.bfloat16).cpu()},
                          str(TT / f"L{i:02d}_b{b:02d}.safetensors"))
                # restore the layer to meta (correct shapes) so to_empty can
                # re-materialize it for the next batch
                lay._apply(lambda t: torch.empty(t.shape, dtype=t.dtype, device="meta"),
                           recurse=True)
                if i % 16 == 0:
                    print(f"batch {b} layer {i} live {torch.cuda.memory_allocated()/1e9:.1f} GB "
                          f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB",
                          flush=True)
            torch.cuda.empty_cache()
    record["runtime_s"] = round(time.time() - t0, 1)
    record["n_batches"], record["n_layers"] = n_batches, n_layers
    record["peak_gpu_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (OUT / "phase1_teacher.json").write_text(json.dumps(record, indent=1))
    print("phase1 done:", json.dumps(record), flush=True)


def main():
    patch_ct()
    t0 = time.time()
    win = build_calib() if not (OUT / "calib_ids.safetensors").exists() else None
    student = load_student()
    student.eval()
    capture_pos_emb(student)
    del student
    torch.cuda.empty_cache()
    teacher_layer_stream()
    print("total runtime_s:", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
