#!/usr/bin/env python
"""QAD pilot pre-flight (analysis/qad/layerwise_validate.py).

Validates the machinery the teacher-target capture will rely on, using the
STUDENT checkpoint (14 GB, fits on GPU) as a proxy for the teacher (55.6 GB,
cannot be loaded whole):

  (a) manual layer-major forward loop (own rope + layer calls) reproduces the
      text model's own forward on the student -> hidden states at layer 0 and
      layer 63 outputs must match the hooked reference,
  (b) mmap shard materialization: a decoder layer rebuilt from raw safetensors
      shards must match the same layer of the loaded student bitwise
      (packed codes + scales),
  (c) differentiability smoke test: LoRA-wrapped compressed-tensors linear
      must backprop into the LoRA params with the base frozen,
  (d) prints module types / shapes for the LoRA target modules.

Writes analysis/qad/layerwise_validate.json.
"""
import json, sys, time
from pathlib import Path
import torch

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

REPO = Path("/home/hubbe/qwen-serving")
OUT = REPO / "analysis" / "qad" / "layerwise_validate.json"
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
    # The checkpoint declares quant_method=compressed-tensors in config.json, so
    # transformers requires the same class: CompressedTensorsConfig (from
    # transformers.utils.quantization_config). Construct it directly from the
    # quantization_config.json dict (it carries config_groups + ignore) instead
    # of a plain compressed_tensors.QuantizationConfig (class-mismatch error).
    import json as _json
    from transformers.utils.quantization_config import CompressedTensorsConfig
    qc = CompressedTensorsConfig(**_json.load(open(MODEL / "quantization_config.json")))
    base_ig = list(qc.quantization_config.ignore or [])
    for extra in ["embed_tokens", "model.language_model.embed_tokens", "model.embed_tokens",
                  "mtp", "draft_lm_head"]:
        if extra not in base_ig:
            base_ig.append(extra)
    qc.quantization_config.ignore = base_ig
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM
    last = None
    model = None
    for name, cls in (("AutoModelForImageTextToText", AutoModelForImageTextToText),
                      ("AutoModelForCausalLM", AutoModelForCausalLM)):
        try:
            model = cls.from_pretrained(str(MODEL), quantization_config=qc, torch_dtype=torch.bfloat16,
                                        device_map="cuda:0", low_cpu_mem_usage=True)
            print("LOADED via", name, flush=True)
            break
        except Exception as e:
            print("FAILED", name, type(e).__name__, str(e)[:200], flush=True)
            last = e
            torch.cuda.empty_cache()
    if model is None:
        raise RuntimeError(f"student load failed: {last}")
    decompress_packed_modules(model)
    wrap_packed_linears(model)
    return model


def decompress_packed_modules(model):
    """Walk modules; any with weight_packed/weight_scale/weight_shape but no plain
    `weight` param (plain nn.Embedding / nn.Linear kept packed by compressed-tensors
    at load) is unpacked + dequantized into a fresh bf16 `weight` Parameter, and the
    packed params + instance-level quantized_forward override are removed. Only the
    ~5 GB of embed/lm_head-class modules are decompressed — quantized Linear layers
    keep their packed layout (decompressing everything would need ~55 GB)."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization import QuantizationStatus
    n_done = 0
    gb = 0.0
    with torch.no_grad():
        for name, m in model.named_modules():
            # Only Embedding-class modules lack a working packed forward path
            # (their quantized_forward does self.weight unconditionally). The
            # quantized Linear layers have functioning packed quantized_forward
            # paths — decompressing them all needs ~55 GB and OOMs a 24 GB card.
            if not isinstance(m, torch.nn.Embedding) or hasattr(m, "weight"):
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
            scale = m.weight_scale.data.to(torch.float32)  # [rows, groups]
            w = signed.view(rows, groups, group_size) * scale.view(rows, groups, 1)
            w = w.reshape(rows, cols).to(torch.bfloat16)
            m.weight = torch.nn.Parameter(w, requires_grad=False)
            del m.weight_packed, m.weight_scale, m.weight_shape
            m.__dict__.pop("forward", None)  # drop instance-level quantized_forward
            m.quantization_status = QuantizationStatus.DECOMPRESSED
            n_done += 1
            gb += w.numel() * 2 / 1e9
    print(f"decompress_packed_modules: {n_done} modules, {gb:.1f} GB", flush=True)


def wrap_packed_linears(model):
    """Give every pack-quantized Linear an instance-level forward that runs the
    differentiable PackedLinearFn (unpack + dequantize on the fly). The native
    quantized_forward requires a plain `.weight` the packed checkpoint does not
    carry (its decompress path never fires without dequantize=True -> 55 GB),
    so without this ANY forward through the student crashes. The decompressed
    Embedding (no weight_packed left) is skipped."""
    import types
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
    print(f"wrap_packed_linears: {n} modules -> PackedLinearFn", flush=True)
    return n


def materialize(module, device="cuda:0"):
    for p in module.parameters():
        p.data = torch.empty(tuple(p.shape), dtype=p.dtype, device=device)
    for b in module.buffers():
        b.data = torch.empty(tuple(b.shape), dtype=b.dtype, device=device)


class PackedLinearFn(torch.autograd.Function):
    """Differentiable forward for compressed-tensors pack-quantized (int4, symmetric,
    group-wise) Linear modules. Forward unpacks + dequantizes the packed codes and
    computes out = x @ W.T (+ bias). Backward returns grad wrt x only — the base
    weights are frozen; LoRA params live in the wrapping module and receive grads
    through the input path."""

    @staticmethod
    def forward(ctx, x, w_packed, w_scale, w_shape, bias, num_bits, group_size):
        rows, cols = (int(v) for v in w_shape.tolist())
        codes = unpack_from_int32(w_packed, num_bits, (rows, cols), packed_dim=1)
        offset = 1 << (num_bits - 1)
        signed = codes.to(torch.float32) - offset
        groups = cols // group_size
        scale = w_scale.to(torch.float32)
        w = (signed.view(rows, groups, group_size) * scale.view(rows, groups, 1))
        w = w.reshape(rows, cols).to(torch.bfloat16)
        ctx.save_for_backward(w)
        out = x @ w.t()
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (w,) = ctx.saved_tensors
        grad_x = grad_out @ w
        return grad_x, None, None, None, None, None, None


class PackedLinear(torch.nn.Module):
    """Wraps a packed compressed-tensors Linear; bypasses its native
    quantized_forward (which requires a plain `.weight` the checkpoint does not
    carry, and whose on-the-fly decompress path is opaque under autograd)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        sch = base.quantization_scheme
        self.num_bits = sch.weights.num_bits
        self.group_size = sch.weights.group_size

    def forward(self, x):
        bias = getattr(self.base, "bias", None)
        return PackedLinearFn.apply(x, self.base.weight_packed, self.base.weight_scale,
                                    self.base.weight_shape, bias, self.num_bits, self.group_size)


def main():
    patch_ct()
    t0 = time.time()
    res = {}
    model = load_student()
    model.eval()
    tm = model.model.language_model
    print("text model:", type(tm).__name__, "n_layers:", len(tm.layers), flush=True)
    print("model children:", [n for n, _ in model.named_children()], flush=True)
    print("tm children:", [n for n, _ in tm.named_children()], flush=True)

    # Position embeddings are captured from the model's own forward (see the
    # pos_hook below) — hand-rolling tm.rotary_emb produced cos/sin at twice
    # the head dim, causing 'size of tensor a (512) must match b (1024)' in
    # apply_rotary_pos_emb.

    torch.manual_seed(0)
    ids = torch.randint(1000, 200000, (2, 512), device="cuda")

    # ---- (a) reference: text model forward with output hooks
    cap = {}
    def mk(i):
        def hook(mod, args, kwargs, out):
            cap[i] = out.detach().clone()
        return hook
    h0 = tm.layers[0].register_forward_hook(mk(0), with_kwargs=True)
    h1 = tm.layers[63].register_forward_hook(mk(63), with_kwargs=True)
    with torch.no_grad():
        tm(input_ids=ids, use_cache=False)
    h0.remove(); h1.remove()
    ref0, ref63 = cap[0], cap[63]

    # ---- manual layer-major loop
    # Capture the exact position_embeddings the model's own forward passes to
    # layer 0 via a forward_pre_hook, and replay those in the manual loop —
    # constructing (cos, sin) by calling tm.rotary_emb by hand produced a
    # dim-2 mismatch (512 vs 1024) inside apply_rotary_pos_emb.
    cap_pos = {}
    def pos_hook(mod, args, kwargs):
        cap_pos["pe"] = kwargs.get("position_embeddings")
    ph = tm.layers[0].register_forward_pre_hook(pos_hook, with_kwargs=True)
    with torch.no_grad():
        tm(input_ids=ids, use_cache=False)
    ph.remove()
    with torch.no_grad():
        h = tm.embed_tokens(ids)
        pos2 = torch.arange(ids.shape[1], device="cuda").view(1, -1).expand(ids.shape[0], -1)
        cos, sin = cap_pos["pe"]
        mine = {}
        for i, layer in enumerate(tm.layers):
            h = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                      position_ids=pos2, past_key_values=None, use_cache=False)
            if i in (0, 63):
                mine[i] = h.detach().clone()
    for i, ref in ((0, ref0), (63, ref63)):
        d = (mine[i].float() - ref.float()).abs()
        rel = (d / ref.float().abs().clamp_min(1e-6)).max().item()
        print(f"layer {i}: max_abs_diff={d.max().item():.3e} mean_abs={d.mean().item():.3e} "
              f"max_rel={rel:.3e}", flush=True)
        res[f"layer{i}_loop"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                 "max_rel": float(rel)}

    # ---- (b) shard materialization of student layer 0
    from safetensors import safe_open
    idx = json.load(open(MODEL / "model.safetensors.index.json"))
    wmap = idx["weight_map"]
    sample = [k for k in wmap if "layers.0." in k][:6]
    print("sample shard keys:", sample, flush=True)
    res["sample_shard_keys"] = sample
    prefix = "model.language_model.layers.0."
    keys = [k for k in wmap if k.startswith(prefix)]
    sd = {}
    for sh in sorted(set(wmap[k] for k in keys)):
        with safe_open(str(MODEL / sh), framework="pt", device="cuda") as f:
            for k in keys:
                sd[k[len(prefix):]] = f.get_tensor(k)
    lay = tm.layers[0]
    params = dict(lay.named_parameters()); bufs = dict(lay.named_buffers())
    exact, mismatch, missing = 0, [], []
    for k, v in sd.items():
        if k in params:
            ok = torch.equal(v, params[k].data)
        elif k in bufs:
            ok = torch.equal(v, bufs[k].data)
        else:
            missing.append(k); continue
        if ok:
            exact += 1
        else:
            mismatch.append(k)
    print(f"shard materialization: exact={exact}/{len(sd)} mismatch={mismatch[:5]} "
          f"missing={missing[:5]}", flush=True)
    res["shard_materialization"] = {"n": len(sd), "exact": exact, "mismatch": mismatch,
                                    "missing": missing}

    # ---- target packed Linears are already PackedLinearFn-wrapped at load
    # (wrap_packed_linears in load_student) — record their status here.
    wrapped = []
    for li in (0, 63):
        mlp = tm.layers[li].mlp
        for pname in ("gate_proj", "down_proj"):
            base_mod = getattr(mlp, pname)
            assert hasattr(base_mod, "weight_packed"), f"L{li}.{pname} not packed?"
            assert getattr(base_mod, "forward", None) is not None
            wrapped.append(f"L{li}.{pname}")
    print("target linears (PackedLinearFn-wrapped):", wrapped, flush=True)
    res["wrapped_modules"] = wrapped

    # ---- (c)/(d) LoRA target modules + differentiability smoke
    gp = tm.layers[0].mlp.gate_proj
    dp = tm.layers[0].mlp.down_proj
    res["target_modules"] = {
        "layers.0.mlp.gate_proj": {"type": type(gp).__name__,
                                   "in": getattr(gp, "in_features", None),
                                   "out": getattr(gp, "out_features", None)},
        "layers.0.mlp.down_proj": {"type": type(dp).__name__,
                                   "in": getattr(dp, "in_features", None),
                                   "out": getattr(dp, "out_features", None)},
    }
    print("gate_proj type:", type(gp).__name__, getattr(gp, "in_features", None),
          getattr(gp, "out_features", None), flush=True)
    print("down_proj type:", type(dp).__name__, getattr(dp, "in_features", None),
          getattr(dp, "out_features", None), flush=True)

    class LoRA(torch.nn.Module):
        def __init__(self, base, r=16, alpha=32.0):
            super().__init__()
            self.base = base
            in_f = getattr(base, "in_features", None) or base.weight_shape[1].item()
            out_f = getattr(base, "out_features", None) or base.weight_shape[0].item()
            self.A = torch.nn.Parameter(torch.empty(r, in_f, device="cuda", dtype=torch.bfloat16))
            self.B = torch.nn.Parameter(torch.zeros(out_f, r, device="cuda", dtype=torch.bfloat16))
            torch.nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            self.scaling = alpha / r
        def forward(self, x):
            return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

    parent = tm.layers[0].mlp
    orig_gp = parent.gate_proj
    parent.gate_proj = LoRA(orig_gp, r=16)
    for p in model.parameters():
        p.requires_grad_(False)
    parent.gate_proj.A.requires_grad_(True)
    parent.gate_proj.B.requires_grad_(True)
    x = torch.randn(1, 256, 5120, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cos = torch.randn(1, 256, 64, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(1, 256, 64, device="cuda", dtype=torch.bfloat16)
    with torch.enable_grad():
        # through the decoder layer itself: the real training path (accelerate's
        # per-layer device-alignment hooks fire on this call, unlike a direct
        # parent.gate_proj(x) which bypasses them and finds weights on CPU)
        out = tm.layers[0](x, position_embeddings=(cos, sin), attention_mask=None,
                           position_ids=None, past_key_values=None, use_cache=False)
        loss = out.float().square().mean()
        loss.backward()
    gA = parent.gate_proj.A.grad
    gB = parent.gate_proj.B.grad
    print("grad A:", None if gA is None else (gA.abs().max().item(), gA.abs().mean().item()),
          "grad B:", None if gB is None else (gB.abs().max().item(), gB.abs().mean().item()),
          flush=True)
    res["lora_smoke"] = {"gradA_max": None if gA is None else float(gA.abs().max()),
                         "gradB_max": None if gB is None else float(gB.abs().max()),
                         "gradA_finite": bool(torch.isfinite(gA).all()) if gA is not None else False,
                         "gradB_finite": bool(torch.isfinite(gB).all()) if gB is not None else False}
    parent.gate_proj = orig_gp
    res["runtime_s"] = round(time.time() - t0, 1)
    res["peak_gpu_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
