import sys, math, torch, json
import torch.nn.functional as F
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from pilot import build_windows, PPL_W

P1.patch_ct()
full = P1.load_student()
tm = full.model.language_model

# inventory: how many packed modules per num_bits?
from collections import Counter
cnt = Counter()
emb = tm.embed_tokens
print("embed:", type(emb).__name__, "has dense weight:", getattr(emb, "weight", None) is not None,
      "| has packed:", hasattr(emb, "weight_packed"))
for nm, m in full.named_modules():
    if hasattr(m, "weight_packed") and hasattr(m, "quantization_scheme"):
        try:
            cnt[m.quantization_scheme.weights.num_bits] += 1
        except Exception:
            pass
print("packed modules by num_bits:", dict(cnt))

# teacher embed
TEACHER = "/home/hubbe/qwen-serving/models-fp16/Qwen3.8-27B"
idx = json.load(open(TEACHER + "/model.safetensors.index.json"))["weight_map"]
ekey = [k for k in idx if k.endswith("embed_tokens.weight")][0]
with safe_open(TEACHER + "/" + idx[ekey], framework="pt") as f:
    Wte = f.get_tensor(ekey).to("cuda", torch.float32)
print("teacher embed:", tuple(Wte.shape), "std=%.5f" % Wte.std())
sl = slice(0, 4096)
cur = emb.weight
dc = (cur[sl].float() - Wte[sl]).abs().mean().item()
cc = torch.corrcoef(torch.stack([cur[sl].float().flatten(), Wte[sl].flatten()]))[0, 1].item()
print(f"current emb.weight vs teacher: mean_abs={dc:.6f} corr={cc:.4f} "
      f"std mine={cur[sl].float().std():.5f} teacher={Wte[sl].std():.5f}")

