import sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from safetensors.torch import load_file
from pilot import build_windows, PPL_W, make_pos, pos2_ids, layer_fwd, ADIR, TT

P1.patch_ct()
full = P1.load_student()
tm = full.model.language_model
# lm_head: dense unsigned + shadow popped (mirror pilot.py fix)
lh = dict(full.named_modules())["lm_head"]
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
sch = lh.quantization_scheme
nb, gs = sch.weights.num_bits, sch.weights.group_size
rows, cols = (int(x) for x in lh.weight_shape.tolist())
codes = unpack_from_int32(lh.weight_packed, nb, (rows, cols), packed_dim=1)
groups = cols // gs
Wlh = (codes.to(torch.float32).view(rows, groups, gs)
       * lh.weight_scale.to(torch.float32).view(rows, groups, 1)).reshape(rows, cols).to(torch.bfloat16)
lh.weight = torch.nn.Parameter(Wlh.to("cuda"), requires_grad=False)
lh.__dict__.pop("forward", None)

cos, sin = make_pos()
tok, ids, starts = build_windows()
inp = torch.tensor([ids[s:s+PPL_W] for s in starts[:2]], device="cuda")

# teacher targets for these seqs (batch files: seqs 0,1 are in b00)
tt63 = load_file(str(TT / "L63_b00.safetensors"))["h"][:2].cuda()
tt0 = load_file(str(TT / "L00_b00.safetensors"))["h"][:2].cuda()

def mse(a, b):
    return F.mse_loss(a.float(), b.float()).item()

with torch.no_grad():
    # path A: tm(input_ids) as ppl_run does
    hA = tm(input_ids=inp, use_cache=False)
    hA = hA[0] if isinstance(hA, tuple) else hA.last_hidden_state
    # path B: manual, explicit pos embeddings, layer by layer
    hb = tm.embed_tokens(inp)
    h0 = hb.clone()
    for i, layer in enumerate(tm.layers):
        hb = layer_fwd(layer, hb, cos, sin, pos2_ids(inp.shape[0], PPL_W))
    print(f"embed stats: mean={tm.embed_tokens.weight.mean():.5f} std={tm.embed_tokens.weight.std():.5f}")
    print(f"pathA L63-out mse vs teacher: {mse(hA, tt63):.2f}")
    print(f"pathB L63-out mse vs teacher: {mse(hb, tt63):.2f}")
    # L0 check via hook on path A: capture layer0 output
    outs = {}
    def hook(m, i, o):
        oo = o[0] if isinstance(o, tuple) else o
        outs["l0"] = oo
    hndl = tm.layers[0].register_forward_hook(hook)
    _ = tm(input_ids=inp, use_cache=False)
    hndl.remove()
    print(f"pathA L0-out mse vs teacher: {mse(outs['l0'], tt0):.2f}")
    h0b = layer_fwd(tm.layers[0], h0, cos, sin, pos2_ids(inp.shape[0], PPL_W))
    print(f"pathB L0-out mse vs teacher: {mse(h0b, tt0):.2f}")
    # nll with teacher-dequant lm_head both ways
    tgt = inp[:, 1:]
    for name, hh in (("A", hA), ("B", hb)):
        lg = lh(hh[:, :-1, :])
        lp = torch.log_softmax(lg.float(), dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"  nll path {name}: {nll.mean():.3f} ppl={math.exp(min(nll.mean().item(),60)):.3e}")
