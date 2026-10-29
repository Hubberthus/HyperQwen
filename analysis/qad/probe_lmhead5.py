import sys, inspect, math, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from pilot import build_windows, PPL_W

P1.patch_ct()
full = P1.load_student()
tm = full.model.language_model
lh = dict(full.named_modules())["lm_head"]
print("class:", type(lh))
try:
    print(inspect.getsource(type(lh).forward))
except Exception as e:
    print("no source:", e)

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
sch = lh.quantization_scheme
nb, gs = sch.weights.num_bits, sch.weights.group_size
rows, cols = (int(x) for x in lh.weight_shape.tolist())
codes = unpack_from_int32(lh.weight_packed, nb, (rows, cols), packed_dim=1)
groups = cols // gs
w = (codes.to(torch.float32).view(rows, groups, gs)
     * lh.weight_scale.to(torch.float32).view(rows, groups, 1)).reshape(rows, cols).to(torch.bfloat16)
lh.weight = torch.nn.Parameter(w.to("cuda"), requires_grad=False)
print("assigned: lh.weight is", type(lh.weight).__name__, tuple(lh.weight.shape))

tok, ids, starts = build_windows()
inp = torch.tensor([ids[s:s+PPL_W] for s in starts[:1]], device="cuda")
with torch.no_grad():
    h = tm(input_ids=inp, use_cache=False)
    h = h[0] if isinstance(h, tuple) else h.last_hidden_state
    hs = h[0, :8, :]                      # 8 token positions
    a = F.linear(hs, lh.weight)           # explicit dense path with assigned W
    b = lh(hs)                            # module call
    d = (a - b).abs().max().item()
    print(f"F.linear vs lh(): max_abs_diff={d:.4e}")
    tgt = inp[0, 1:9]
    for name, lg in (("F.linear", a), ("lh()", b)):
        lp = torch.log_softmax(lg.float(), dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"  {name}: nll={nll.mean():.3f}")
