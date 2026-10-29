import sys, math, torch
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from pilot import build_windows, PPL_W

P1.patch_ct()
full = P1.load_student()
tm = full.model.language_model
lh = dict(full.named_modules())["lm_head"]
print("lm_head type:", type(lh).__name__)
for a in ("weight", "weight_packed", "weight_scale", "weight_shape", "bias", "quantization_scheme"):
    v = getattr(lh, a, None)
    if v is None:
        print(f"  {a}: None")
    elif a == "quantization_scheme":
        print(f"  {a}: bits={v.weights.num_bits} gs={v.weights.group_size}")
    elif torch.is_tensor(v):
        print(f"  {a}: shape={tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"  {a}: {type(v).__name__}")

# try dequantizing and re-test
try:
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    sch = lh.quantization_scheme
    nb, gs = sch.weights.num_bits, sch.weights.group_size
    rows, cols = (int(x) for x in lh.weight_shape.tolist())
    codes = unpack_from_int32(lh.weight_packed, nb, (rows, cols), packed_dim=1)
    off = 1 << (nb - 1)
    groups = cols // gs
    w = ((codes.to(torch.float32) - off).view(rows, groups, gs)
         * lh.weight_scale.to(torch.float32).view(rows, groups, 1)).reshape(rows, cols).to(torch.bfloat16)
    lh.weight = torch.nn.Parameter(w, requires_grad=False)
    print("dequantized lm_head ok")

    tok, ids, starts = build_windows()
    inp = torch.tensor([ids[s:s+PPL_W] for s in starts[:2]], device="cuda")
    with torch.no_grad():
        h = tm(input_ids=inp, use_cache=False)
        h = h[0] if isinstance(h, tuple) else h.last_hidden_state
        logits = lh(h)[:, :-1]
        tgt = inp[:, 1:]
        lp = torch.log_softmax(logits.float(), dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"after fix: logits mean={logits.mean():.3f} std={logits.std():.3f} "
              f"nll_mean={nll.mean():.3f} ppl={math.exp(nll.mean().item()):.4f}")
except Exception as e:
    import traceback; traceback.print_exc()
