import sys, torch, json
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from safetensors import safe_open
from pilot import MODEL

P1.patch_ct()
full = P1.load_student()
lh = dict(full.named_modules())["lm_head"]
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
sch = lh.quantization_scheme
nb, gs = sch.weights.num_bits, sch.weights.group_size
rows, cols = (int(x) for x in lh.weight_shape.tolist())
print(f"lm_head: rows={rows} cols={cols} bits={nb} gs={gs} packed={tuple(lh.weight_packed.shape)} scale={tuple(lh.weight_scale.shape)}")

# raw code distribution from packed ints
p = lh.weight_packed[:1000].cpu().to(torch.int64)
factor = 32 // nb
codes = torch.stack([(p >> (nb * i)) & ((1 << nb) - 1) for i in range(factor)], dim=-1).flatten()
print(f"packed-path codes (first 1k ints): mean={codes.float().mean():.2f} std={codes.float().std():.2f} "
      f"min={codes.min()} max={codes.max()} p10={codes.float().quantile(0.1):.0f} p90={codes.float().quantile(0.9):.0f}")

# reference: a working 4-bit linear (gate_proj L0)
tm = full.model.language_model
gp = tm.layers[0].mlp.gate_proj
print("gate_proj type:", type(gp).__name__)
if hasattr(gp, "weight_packed"):
    q = gp.weight_packed[:1000].cpu().to(torch.int64)
    f4 = 32 // 4
    c4 = torch.stack([(q >> (4 * i)) & 15 for i in range(f4)], dim=-1).flatten()
    print(f"gate_proj codes (4bit): mean={c4.float().mean():.2f} std={c4.float().std():.2f} min={c4.min()} max={c4.max()}")

# teacher comparison on a small slice
TEACHER = "/home/hubbe/qwen-serving/models-fp16/Qwen3.8-27B"
idx = json.load(open(TEACHER + "/model.safetensors.index.json"))["weight_map"]
with safe_open(TEACHER + "/" + idx["lm_head.weight"], framework="pt") as f:
    Wt = f.get_tensor("lm_head.weight").to("cuda", torch.float32)

def try_layout(name, W):
    Wc = W[:64].float()
    Tc = Wt[:64]
    corr = torch.corrcoef(torch.stack([Wc.flatten(), Tc.flatten()]))[0, 1].item()
    print(f"  {name}: mean={W.mean():.5f} std={W.std():.5f} corr_vs_teacher(64 rows)={corr:.4f}")

s = lh.weight_scale.float()
sc = unpack_from_int32(lh.weight_packed, nb, (rows, cols), packed_dim=1).float()
try_layout("L1 sym off=128 row-groups", ((sc - 128).view(rows, cols // gs, gs) * s.view(rows, cols // gs, 1)).reshape(rows, cols))
try_layout("L2 unsigned no-off", (sc.view(rows, cols // gs, gs) * s.view(rows, cols // gs, 1)).reshape(rows, cols))
try_layout("L3 sym off=128 col-groups", ((sc - 128).view(rows, gs, cols // gs) * s.view(rows, 1, cols // gs).expand(rows, gs, cols // gs)).reshape(rows, cols))
try_layout("L4 sym off=128 groups-first", ((sc - 128).view(cols // gs, gs, rows).permute(2, 0, 1) * s.view(cols // gs, 1, rows).permute(2, 0, 1)).reshape(rows, cols))
