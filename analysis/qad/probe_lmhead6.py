import sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from safetensors import safe_open
from pilot import build_windows, PPL_W

P1.patch_ct()
full = P1.load_student()
tm = full.model.language_model
lh = dict(full.named_modules())["lm_head"]
print("hooks: fwd_pre=%d fwd=%d | bias=%s" % (
    len(lh._forward_pre_hooks), len(lh._forward_hooks),
    None if lh.bias is None else tuple(lh.bias.shape)))
for k, hk in list(lh._forward_pre_hooks.items()) + list(lh._forward_hooks.items()):
    print("  hook:", getattr(hk, "__qualname__", hk))

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
sch = lh.quantization_scheme
nb, gs = sch.weights.num_bits, sch.weights.group_size
rows, cols = (int(x) for x in lh.weight_shape.tolist())
codes = unpack_from_int32(lh.weight_packed, nb, (rows, cols), packed_dim=1)
groups = cols // gs
Wmine = (codes.to(torch.float32).view(rows, groups, gs)
         * lh.weight_scale.to(torch.float32).view(rows, groups, 1)).reshape(rows, cols).to(torch.bfloat16)

TEACHER = "/home/hubbe/qwen-serving/models-fp16/Qwen3.8-27B"
import json
idx = json.load(open(TEACHER + "/model.safetensors.index.json"))["weight_map"]
with safe_open(TEACHER + "/" + idx["lm_head.weight"], framework="pt") as f:
    Wt = f.get_tensor("lm_head.weight").to("cuda", torch.float32)
sl = slice(0, 4096)  # avoid OOM on full compare
d = (Wmine[sl].float() - Wt[sl]).abs()
print(f"W_mine vs W_teacher (4k rows): mean_abs={d.mean():.6f} max_abs={d.max():.4f}")

tok, ids, starts = build_windows()
inp = torch.tensor([ids[s:s+PPL_W] for s in starts[:2]], device="cuda")
with torch.no_grad():
    h = tm(input_ids=inp, use_cache=False)
    h = h[0] if isinstance(h, tuple) else h.last_hidden_state
    hs = h[:, :-1, :]
    tgt = inp[:, 1:]
    print(f"h stats: mean={h.mean():.4f} std={h.std():.4f} max|.|={h.abs().max():.4f}")
    print("instance forward shadow in lh.__dict__:", "forward" in lh.__dict__)
    if "forward" in lh.__dict__:
        print("  shadowed forward:", lh.__dict__["forward"])
    Wmb = Wmine.to(torch.bfloat16)
    Wtb = Wt.to(torch.bfloat16)
    hs2 = hs.to(torch.bfloat16)
    for name, W in (("W_mine", Wmb), ("W_teacher", Wtb)):
        lg = F.linear(hs2, W)
        lp = torch.log_softmax(lg.float(), dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"  F.linear {name}: nll={nll.mean():.3f} ppl={math.exp(min(nll.mean().item(),60)):.3e}")
    # does deleting the instance shadow make lh() use the class forward?
    if "forward" in lh.__dict__:
        saved = lh.__dict__["forward"]
        del lh.__dict__["forward"]
        lg2 = lh(hs2)
        lp2 = torch.log_softmax(lg2.float(), dim=-1)
        nll2 = -lp2.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"  lh() after del shadow: nll={nll2.mean():.3f} ppl={math.exp(min(nll2.mean().item(),60)):.3e}")
        d2 = (lg2.float() - F.linear(hs2, Wmb).float()).abs().max().item()
        print(f"  lh() vs F.linear(W_mine) after del: max_abs_diff={d2:.4e}")

    # embed_tokens sanity: materialize from packed and compare to teacher embedding
    emb = tm.embed_tokens
    print("embed type:", type(emb).__name__, "| has weight:", hasattr(emb, "weight") and emb.weight is not None)
    if hasattr(emb, "weight_packed"):
        esch = emb.quantization_scheme
        enb, egs = esch.weights.num_bits, esch.weights.group_size
        er, ec = (int(x) for x in emb.weight_shape.tolist())
        ecodes = unpack_from_int32(emb.weight_packed, enb, (er, ec), packed_dim=1)
        eg = ec // egs
        # try symmetric offset
        Wsym = ((ecodes.to(torch.float32) - (1 << (enb - 1))).view(er, eg, egs)
                * emb.weight_scale.to(torch.float32).view(er, eg, 1)).reshape(er, ec).to(torch.bfloat16)
        eidx = json.load(open(TEACHER + "/model.safetensors.index.json"))["weight_map"]
        ekey = [k for k in eidx if k.endswith("embed_tokens.weight")]
        with safe_open(TEACHER + "/" + eidx[ekey[0]], framework="pt") as f:
            Wte = f.get_tensor(ekey[0]).to("cuda", torch.float32)
        ds = (Wsym[sl].float() - Wte[sl]).abs()
        c = torch.corrcoef(torch.stack([Wsym[sl].float().flatten(), Wte[sl].float().flatten()]))[0, 1].item()
        print(f"  embed sym-offset vs teacher: mean_abs={ds.mean():.6f} corr={c:.4f} "
              f"(teacher std={Wte[sl].std():.5f}, mine std={Wsym[sl].float().std():.5f})")
