#!/usr/bin/env python
"""Probe: why does ppl_run produce garbage logits? Compare lm_head candidates."""
import sys, math, torch
sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1
from pilot import build_windows, PPL_W

P1.patch_ct()
full = P1.load_student()
tm = full.model.language_model

# what does the lm_head search find?
cands = [(nm, m) for nm, m in full.named_modules() if nm.endswith("lm_head")]
print("lm_head candidates:", [(nm, type(m).__name__) for nm, m in cands])
oe = full.get_output_embeddings()
print("get_output_embeddings:", type(oe).__name__ if oe is not None else None)

tok, ids, starts = build_windows()
inp = torch.tensor([ids[s:s+PPL_W] for s in starts[:2]], device="cuda")
with torch.no_grad():
    h = tm(input_ids=inp, use_cache=False)
    h = h[0] if isinstance(h, tuple) else h.last_hidden_state
    print("hidden stats: mean=%.4f std=%.4f max|.|=%.4f" % (h.mean(), h.std(), h.abs().max()))
    tgt = inp[:, 1:]
    for nm, m in cands:
        logits = m(h)[:, :-1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"  cand {nm}: logits mean={logits.mean():.3f} std={logits.std():.3f} "
              f"nll_mean={nll.mean():.3f} ppl={math.exp(min(nll.mean().item(), 50)):.4f}")
    # also try the full CausalLM/CondGen forward path
    try:
        out = full(input_ids=inp, use_cache=False)
        lg = out.logits[:, :-1]
        lp = torch.log_softmax(lg.float(), dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        print(f"  full forward: logits mean={lg.mean():.3f} std={lg.std():.3f} "
              f"nll_mean={nll.mean():.3f} ppl={math.exp(min(nll.mean().item(), 50)):.4f}")
    except Exception as e:
        print("  full forward failed:", e)
