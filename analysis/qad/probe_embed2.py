import json, torch
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

CKPT = "/home/hubbe/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound"
idx = json.load(open(CKPT + "/model.safetensors.index.json"))["weight_map"]
shard = idx["model.language_model.embed_tokens.weight_packed"]
with safe_open(CKPT + "/" + shard, framework="pt") as f:
    pk = f.get_tensor("model.language_model.embed_tokens.weight_packed")
    sc = f.get_tensor("model.language_model.embed_tokens.weight_scale")
    sh = f.get_tensor("model.language_model.embed_tokens.weight_shape")
rows, cols = (int(v) for v in sh.tolist())
print(f"packed={tuple(pk.shape)} scale={tuple(sc.shape)} shape={rows}x{cols}")
nb = 32 // (pk.shape[1] * 8 // cols) if False else None
# infer bits: cols / packed_cols = codes per int32
cpint = cols // pk.shape[1]
nb = {8: 4, 4: 8, 2: 16, 1: 32}[cpint]
print(f"inferred num_bits={nb} (codes/int32={cpint}), scale rows match: {sc.shape[0]==rows}, groups/row={sc.shape[1]}")
gs = cols // sc.shape[1]

TEACHER = "/home/hubbe/qwen-serving/models-fp16/Qwen3.8-27B"
tidx = json.load(open(TEACHER + "/model.safetensors.index.json"))["weight_map"]
tkey = [k for k in tidx if k.endswith("embed_tokens.weight")][0]
with safe_open(TEACHER + "/" + tidx[tkey], framework="pt") as f:
    Wte = f.get_tensor(tkey).to("cuda", torch.float32)
print("teacher:", tuple(Wte.shape), "std=%.5f" % Wte.std())

codes = unpack_from_int32(pk, nb, (rows, cols), packed_dim=1).float().to("cuda")
sc = sc.to("cuda")
groups = cols // gs
sl = slice(0, 4096)
for name, W in (
    ("sym off", ((codes - (1 << (nb - 1))).view(rows, groups, gs) * sc.view(rows, groups, 1)).reshape(rows, cols)),
    ("unsigned", (codes.view(rows, groups, gs) * sc.view(rows, groups, 1)).reshape(rows, cols)),
    ("col-groups sym", ((codes - (1 << (nb - 1))).view(rows, gs, groups) * sc.view(rows, 1, groups)).reshape(rows, cols)),
):
    Wb = W.to(torch.bfloat16)
    d = (Wb[sl].float() - Wte[sl]).abs().mean().item()
    c = torch.corrcoef(torch.stack([Wb[sl].float().flatten(), Wte[sl].flatten()]))[0, 1].item()
    print(f"  {name}: mean_abs={d:.6f} corr={c:.4f} std={Wb[sl].float().std():.5f}")

# also print the full quantization_config.json groups for embed/lm_head symmetric flag
qc = json.load(open(CKPT + "/quantization_config.json"))
for g, v in qc.get("config_groups", qc).items() if isinstance(qc.get("config_groups", qc), dict) else []:
    if isinstance(v, dict):
        w = v.get("weights", {})
        print(f"group {g}: targets={v.get('targets')} bits={w.get('num_bits')} symmetric={w.get('symmetric')} zp={w.get('zp_dtype')}")
