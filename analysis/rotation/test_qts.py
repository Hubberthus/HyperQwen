import torch
from auto_round.data_type.int import quant_tensor_sym
from auto_round.data_type.utils import round_ste
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from torch.optim.lr_scheduler import LinearLR

torch.manual_seed(0)
out, inn = 64, 256
G = inn // 128
W = (torch.randn(out, inn, dtype=torch.float32) * 0.05)
W[0, :5] = 0.0

# v in the reshaped [out*G, 128] layout (matches reshape_pad_tensor_by_group_size output)
v0 = torch.zeros(out * G, 128)
qdq, scale, maxq = quant_tensor_sym(W, bits=4, group_size=128, v=v0, scale_dtype=torch.float16)
print("qdq", tuple(qdq.shape), qdq.dtype, "scale", tuple(scale.shape), scale.dtype, "maxq", maxq)
s = scale.float()
print("scale row0:", s.view(out, -1)[0].tolist())
amax = W.abs().view(out, -1, 128).amax(dim=-1)
print("absmax/8 row0:", (amax / 8)[0].tolist())
print("sign matches |min|>|max| rule:", bool((s.view(out,-1)[0] < 0).sum()))

# dequant error with v=0
print("max |qdq-W| (v=0):", (qdq - W).abs().max().item())

# with a nonzero per-element v (shape [out, in])
v2 = torch.full((out * G, 128), 0.3)
qdq2, scale2, _ = quant_tensor_sym(W, bits=4, group_size=128, v=v2, scale_dtype=torch.float16)
print("max |qdq2-W| (v=0.3):", (qdq2 - W).abs().max().item())
print("scale identical:", torch.equal(scale, scale2))

# recover codes
s3 = scale2.float().view(out, G, 1)
codes = (qdq2.float().view(out, G, 128) / s3)
print("codes unique:", codes.unique().tolist())
print("codes integral:", bool((codes == codes.round()).all()))
print("codes in [-8,7]:", bool((codes >= -8).all() and (codes <= 7).all()))
print("qdq == codes*s exactly:", bool((qdq2.float().view(out, G, 128) - codes * s3).abs().max() == 0))

# also test v with 3D shape [out, G, 128]
v3 = torch.full((out, inn // 128, 128), 0.3)
try:
    qdq3, scale3, _ = quant_tensor_sym(W, bits=4, group_size=128, v=v3, scale_dtype=torch.float16)
    print("v 3D works, same result:", bool((qdq3 - qdq2).abs().max() == 0))
except Exception as e:
    print("v 3D FAILED:", type(e).__name__, str(e)[:200])

# SignSGD smoke test: sign update
p = torch.nn.Parameter(torch.zeros(5))
opt = SignSGD([p], lr=0.005)
sched = LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=200)
for i in range(3):
    loss = (p * torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])).sum()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print("iter", i, "p", p.detach().tolist(), "lr", opt.param_groups[0]["lr"])
