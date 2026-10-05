"""Token coverage of each id list over a prompt cohort (CPU-only, #196).

Distinct from --stats, which counts vocabulary rows. This tokenizes the actual
eval prompts and asks how many of the tokens the model would emit are in each
list -- the quantity that bounds acceptance, since a token outside the list is
a guaranteed rejection that also ends the chain (gotcha 11).
"""
import json, sys, collections
from transformers import AutoTokenizer

MODEL = "/home/hubbe/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound-fast"
LISTS = {
    "shipped":  "/home/hubbe/qwen-serving/prepare/draft_vocab_ids.json",
    "cjk":      "/tmp/arms/ids_cjk.json",
    "cyrillic": "/tmp/arms/ids_cyrillic.json",
}
tok = AutoTokenizer.from_pretrained(MODEL)
sets = {k: set(json.load(open(v))) for k, v in LISTS.items()}
for k, v in sets.items():
    print(f"{k}: {len(v)} ids")

cohorts = {
    "han": "/home/hubbe/qwen-serving/bench/prompts_han.jsonl",
    "cyrillic": "/home/hubbe/qwen-serving/bench/prompts_cyrillic.jsonl",
    "english": "/home/hubbe/qwen-serving/bench/prompts_real.jsonl",
}

print("\n%-10s %8s  %s" % ("cohort", "tokens", "  ".join(f"{k:>10}" for k in sets)))
for cname, path in cohorts.items():
    texts = [json.loads(l)["prompt"] for l in open(path) if l.strip()]
    ids = []
    for t in texts:
        ids += tok(t, add_special_tokens=False).input_ids
    cov = {k: 100 * sum(1 for i in ids if i in s) / len(ids) for k, s in sets.items()}
    print("%-10s %8d  %s" % (cname, len(ids), "  ".join(f"{cov[k]:9.1f}%" for k in sets)))

# Per-prompt spread for the cohort that matters, so a mean cannot hide a bad prompt.
print("\nper-prompt coverage, %:")
for cname in ("han", "cyrillic"):
    texts = [json.loads(l)["prompt"] for l in open(cohorts[cname]) if l.strip()]
    print(f"  {cname}:")
    for n, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False).input_ids
        cov = {k: 100 * sum(1 for i in ids if i in s) / len(ids) for k, s in sets.items()}
        print("   p%d n=%4d  %s" % (n + 1, len(ids), "  ".join(f"{k}={cov[k]:5.1f}%" for k in sets)))