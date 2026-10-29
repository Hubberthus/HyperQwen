#!/usr/bin/env python
"""Smoke test: verify the head-dim slicing fix resolves the 512 vs 1024 rope mismatch.

Runs a single forward through L0 with the sliced cos/sin and confirms no
apply_rotary_pos_emb size mismatch occurs.
"""
import sys, time
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import torch
from layerwise_validate import patch_ct, load_student, MODEL

patch_ct()
t0 = time.time()
model = load_student()
model.eval()
tm = model.model.language_model
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

# --- capture position_embeddings from the text model's own forward (reference path)
cap_pos = {}
def pos_hook(mod, args, kwargs):
    cap_pos["pe"] = kwargs.get("position_embeddings")
ph = tm.layers[0].register_forward_pre_hook(pos_hook, with_kwargs=True)
torch.manual_seed(0)
ids = torch.randint(1000, 200000, (2, 512), device="cuda")
with torch.no_grad():
    tm(input_ids=ids, use_cache=False)
ph.remove()
cos, sin = cap_pos["pe"]
print("cos/sin shapes:", tuple(cos.shape), tuple(sin.shape), flush=True)

# --- single forward through L0 with the captured cos/sin (already correct shape)
layer = tm.layers[0]
with torch.no_grad():
    h = tm.embed_tokens(ids)
    try:
        out = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                    position_ids=None, past_key_values=None, use_cache=False)
        print("L0 forward OK, out shape:", tuple(out.shape), flush=True)
    except Exception as e:
        print("L0 forward FAILED:", type(e).__name__, str(e)[:400], flush=True)
        raise
