#!/usr/bin/env python3
"""Minimal output-tied A/B perplexity eval (Task 2 ppl gate).

Corpus: wikitext-2-raw-v1 test split (raw, joined). Windows of 2048 tokens,
stride 256, all windows fit fully in the corpus. Metric: mean NLL over every
predicted prompt token (all windows), reported as exp(-mean) perplexity.
Identical for every run: same corpus, windows, token budget, request shape;
only the served model path differs between arms.

Data: prompt_logprobs=0 through the live server (bench/harness.py), token ids
sent explicitly so no tokenizer drift can enter. bench/quality_battery.py was
the preferred tool but cannot run: bench/quality-data/ ships only wikitext
(no fineweb2/gsm8k parquets), so its docs() raises. This script follows the
task's fallback protocol.
usage: ppl_eval.py <tag>   -> writes analysis/rotation/ppl_runs/<tag>.json
"""
import json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

REPO = "/home/hubbe/qwen-serving"
sys.path.insert(0, REPO + "/bench")
import harness  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tag = sys.argv[1]
tok = AutoTokenizer.from_pretrained(REPO + "/models/Qwen3.8-27B-W4A16-AutoRound")
txt = "".join(pq.read_table(
    REPO + "/bench/quality-data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
).column("text").to_pylist())
ids = tok(txt, add_special_tokens=False)["input_ids"]

W, S = 2048, 256
starts = list(range(0, len(ids) - W + 1, S))
assert len(ids) >= W and starts, "corpus smaller than one window"

def one(a):
    win = ids[a:a + W]
    r = harness.post("/v1/completions", {"prompt": win, "max_tokens": 1,
                                         "temperature": 0, "prompt_logprobs": 0},
                     timeout=900)
    pl = r["choices"][0]["prompt_logprobs"]
    lps = []
    for e in pl[1:]:
        if e is None:
            continue
        v = list(e.values())[0]
        lps.append(v["logprob"] if isinstance(v, dict) else v)
    if len(lps) != W - 1:
        raise RuntimeError(f"window@{a}: {len(lps)} of {W-1} prompt logprobs")
    return sum(lps), len(lps)

t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(one, starts))
wall = time.time() - t0
s = sum(x[0] for x in res); n = sum(x[1] for x in res)
out = {"tag": tag, "corpus": "wikitext-2-raw-v1/test (raw, joined)",
       "window_tokens": W, "stride": S, "n_windows": len(starts),
       "corpus_tokens": len(ids), "n_tokens_predicted": n,
       "mean_nll": -s / n, "ppl": math.exp(-s / n),
       "wall_s": round(wall, 1),
       "server_pid": os.environ.get("SERVER_PID"),
       "port": os.environ.get("PORT", "18020")}
os.makedirs(REPO + "/analysis/rotation/ppl_runs", exist_ok=True)
with open(REPO + f"/analysis/rotation/ppl_runs/{tag}.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out))
