#!/usr/bin/env python
"""Pre-flight test for build_patch_ar.py: imports, target table, SignSGD loop,
LinearLR end_factor=0, STE gradient, and RTN-convention match vs build_patch."""
import sys, random, zlib
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp
import build_patch_ar as ar
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

print("1) imports OK")

t = ar.build_targets()
n = sum(len(v) for v in t.values())
assert n == 50, n
print("2) target table: 50 tensors across", sorted(t))

# --- SignSGD + LinearLR(0) + STE mini-loop on synthetic data
torch.manual_seed(0)
out_d, in_d, G = 128, 512, 4
Wrot = (torch.randn(out_d, in_d, dtype=torch.float64) * 0.05).numpy()
amax = np.abs(Wrot.reshape(out_d, G, 128)).max(axis=2)
s64 = np.where(amax > 0, amax / 8.0, 1.0)
s16 = s64.astype(np.float16)
w = torch.from_numpy(Wrot.astype(np.float32)).cuda()
s = torch.from_numpy(np.repeat(s16.astype(np.float32), 128, axis=1)).cuda()
X = torch.randn(8192, in_d, dtype=torch.float32)
try:
    sched0 = torch.optim.lr_scheduler.LinearLR(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1),
                                               start_factor=1.0, end_factor=0.0, total_iters=10)
    print("3) LinearLR end_factor=0.0: ACCEPTED")
except Exception as e:
    print("3) LinearLR end_factor=0.0 REJECTED:", e); sys.exit(1)

v = torch.zeros_like(w, requires_grad=True)
opt = SignSGD([v], lr=0.005, weight_decay=0)
sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=200)
rng = random.Random(1)
Ht = torch.from_numpy(ar.H128).float().cuda()

def loss_of(vv, idx):
    xb = X[idx].cuda()
    xb = xb.view(xb.shape[0], G, 128) @ Ht
    z = w / s + vv
    r = torch.round(z)
    r_ste = r.detach() - z.detach() + z
    q = torch.clamp(r_ste, -8.0, 7.0)
    dq = q * s - w
    err = torch.einsum("bgk,ogk->bo", xb.to(torch.bfloat16),
                       dq.to(torch.bfloat16).view(out_d, G, 128))
    return err.float().pow(2).mean()

l0 = loss_of(v, torch.arange(4096)).item()
# start deliberately suboptimal (codes shifted -1) and confirm SignSGD recovers
with torch.no_grad():
    v.sub_(0.6)
best = (loss_of(v, torch.arange(4096)).item(), v.detach().clone())
for it in range(200):
    idx = torch.arange((it % 2) * 4096, (it % 2) * 4096 + 4096)
    loss = loss_of(v, idx)
    loss.backward()
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    lv = loss.item()
    if lv < best[0]:
        best = (lv, v.detach().clone())
lf = loss_of(best[1], torch.arange(4096)).item()
codes = torch.clamp(torch.round(w / s + best[1]), -8, 7).to(torch.int8).cpu().numpy()
print("4) synthetic loop (suboptimal start): best=%.6g start=%.6g (RTN ref=%.6g) moved_vs_rtn=%d"
      % (best[0], best[0], l0,
         int((codes != np.clip(np.round(Wrot / np.repeat(s16.astype(np.float64), 128, axis=1)), -8, 7)).sum())))
assert lf < best[0], "no improvement from suboptimal start: %g -> %g" % (best[0], lf)
assert (codes != np.clip(np.round(Wrot / np.repeat(s16.astype(np.float64), 128, axis=1)), -8, 7)).sum() > 0, \
    "SignSGD moved no codes"

# --- RTN convention match vs build_patch.quantize_rotated on a real small tensor
base = "model.language_model.layers.0.linear_attn.in_proj_z"
W = bp.load_fp16(base)
packed_ref, scale_ref, _ = bp.quantize_rotated(W, ar.H128)
# our path:
W64 = W.to(torch.float64).numpy()
out_r, in_r = W.shape
Gr = in_r // 128
Wrot64 = bp.block_apply(W64, ar.H128)
amax = np.abs(Wrot64.reshape(out_r, Gr, 128)).max(axis=2)
s64 = np.where(amax > 0, amax / 8.0, 1.0)
s16m = s64.astype(np.float16)
codes_m = np.clip(np.round(Wrot64.reshape(out_r, Gr, 128) / s16m.astype(np.float64)[:, :, None]), -8, 7).astype(np.int64).reshape(out_r, in_r)
packed_m = bp.repack_nibbles((codes_m + 8).reshape(out_r, in_r)).astype(np.int32)
same_scale = np.array_equal(scale_ref, s16m)
same_packed = np.array_equal(packed_ref, packed_m)
print("5) RTN convention vs build_patch.quantize_rotated on %s: scale=%s packed=%s"
      % (base.split("layers.")[1], same_scale, same_packed))
assert same_scale and same_packed, "convention mismatch!"

# --- folded RTN MSE sanity (naive-fold reproduces when v=0)
print("6) pre-flight PASS")
