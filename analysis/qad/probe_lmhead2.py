import sys, math, torch
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from pilot import build_windows, PPL_W

P1.patch_ct()
full = P1.load_student()
tm = full.model.language_model
lh = dict(full.named_modules())["lm_head"]
print("type:", type(lh).__name__, "| mro:", [c.__name__ for c in type(lh).__mro__][:4])
print("attrs:", [a for a in dir(lh) if "weight" in a.lower() or "fn" in a.lower() or "forward" in a.lower()])

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
sch = lh.quantization_scheme
nb, gs = sch.weights.num_bits, sch.weights.group_size
rows, cols = (int(x) for x in lh.weight_shape.tolist())
print(f"shape rows={rows} cols={cols} bits={nb} gs={gs}")

codes = unpack_from_int32(lh.weight_packed, nb, (rows, cols), packed_dim=1)
off = 1 << (nb - 1)
groups = cols // gs
w = ((codes.to(torch.float32) - off).view(rows, groups, gs)
     * lh.weight_scale.to(torch.float32).view(rows, groups, 1)).reshape(rows, cols).to(torch.bfloat16)
print(f"my W stats: mean={w.mean():.4f} std={w.std():.4f} max|.|={w.abs().max():.4f}")
print(f"scale stats: mean={lh.weight_scale.float().mean():.6f} max={lh.weight_scale.float().max():.4f}")

tok, ids, starts = build_windows()
inp = torch.tensor([ids[s:s+PPL_W] for s in starts[:2]], device="cuda")
with torch.no_grad():
    h = tm(input_ids=inp, use_cache=False)
    h = h[0] if isinstance(h, tuple) else h.last_hidden_state
    y1 = lh(h)[:, :-1]                      # module's own (packed) path
    y2 = h @ w.t()                          # my dequant
    d = (y1.float() - y2.float()).abs().max().item()
    print(f"y1 vs y2 max_abs_diff={d:.4e}")
    for name, lg in (("packed path", y1), ("my dequant", y2)):
        lp = torch.log_softmax(lg.float(), dim=-1)
        tgt = inp[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"  {name}: nll={nll.mean():.3f} ppl={math.exp(min(nll.mean().item(),60)):.3e}")
    # also: compare against embed_tokens weight (untied, but similar distribution expected)
    emb = tm.embed_tokens
    ew = emb.weight if hasattr(emb, "weight") else None
    if ew is not None:
        print(f"embed_tokens W: mean={ew.float().mean():.4f} std={ew.float().std():.4f} shape={tuple(ew.shape)}")
