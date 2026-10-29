import sys, math, torch, json
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from safetensors import safe_open
from pilot import build_windows, PPL_W, MODEL

# --- teacher lm_head (ground truth)
TEACHER = "/home/hubbe/qwen-serving/models-fp16/Qwen3.8-27B"
idx = json.load(open(TEACHER + "/model.safetensors.index.json"))["weight_map"]
key = [k for k in idx if k == "lm_head.weight" or k.endswith("lm_head.weight")]
print("teacher lm_head key:", key, "->", idx[key[0]])
with safe_open(TEACHER + "/" + idx[key[0]], framework="pt") as f:
    Wt = f.get_tensor(key[0]).to("cuda", torch.float32)
print(f"teacher W: shape={tuple(Wt.shape)} mean={Wt.mean():.5f} std={Wt.std():.5f}")
row_means = Wt.mean(dim=1)
print(f"teacher row-means: mean={row_means.mean():.5f} std={row_means.std():.5f}")

P1.patch_ct()
full = P1.load_student()
tm = full.model.language_model
lh = dict(full.named_modules())["lm_head"]
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
sch = lh.quantization_scheme
nb, gs = sch.weights.num_bits, sch.weights.group_size
rows, cols = (int(x) for x in lh.weight_shape.tolist())
codes = unpack_from_int32(lh.weight_packed, nb, (rows, cols), packed_dim=1)
off = 1 << (nb - 1)
groups = cols // gs
Wmine = ((codes.to(torch.float32) - off).view(rows, groups, gs)
         * lh.weight_scale.to(torch.float32).view(rows, groups, 1)).reshape(rows, cols).to(torch.bfloat16)
d = (Wmine.float() - Wt).abs()
rel = d / Wt.abs().clamp_min(1e-3)
print(f"my W vs teacher: max_abs={d.max():.4f} mean_abs={d.mean():.5f} "
      f"median_rel={rel.median():.4f} corr={torch.corrcoef(torch.stack([Wmine.float().flatten(), Wt.flatten()]))[0,1]:.4f}")

tok, ids, starts = build_windows()
inp = torch.tensor([ids[s:s+PPL_W] for s in starts[:2]], device="cuda")
with torch.no_grad():
    h = tm(input_ids=inp, use_cache=False)
    h = h[0] if isinstance(h, tuple) else h.last_hidden_state
    hs = h[:, :-1, :]
    tgt = inp[:, 1:]
    y1 = lh(h)[:, :-1, :]
    y2 = hs @ Wmine.t()
    y3 = hs @ Wt.t()
    print(f"y1 vs y2 (packed vs my dequant): max_abs={(y1.float()-y2.float()).abs().max():.4e}")
    print(f"y2 vs y3 (my dequant vs teacher): max_abs={(y2.float()-y3.float()).abs().max():.4e}")
    for name, lg in (("packed path", y1), ("my dequant", y2), ("teacher W", y3)):
        lp = torch.log_softmax(lg.float(), dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"  {name}: nll={nll.mean():.3f} ppl={math.exp(min(nll.mean().item(),60)):.3e}")
