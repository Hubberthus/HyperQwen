#!/usr/bin/env python3
"""Aggregate ppl_runs/*.json into analysis/rotation/ppl.json with noise + verdict."""
import json, math, os, glob

REPO = "/home/hubbe/qwen-serving"
runs = {}
for p in sorted(glob.glob(REPO + "/analysis/rotation/ppl_runs/*.json")):
    d = json.load(open(p))
    arm = "patch" if d["tag"].startswith("patch") else "baseline"
    runs.setdefault(arm, []).append(d)

out = {"metric": "perplexity = exp(-mean NLL) over ALL predicted prompt tokens; "
                 "wikitext-2-raw-v1 test (raw, joined), 2048-token windows, stride 256, "
                 "identical script/corpus/token budget both arms; measured via prompt_logprobs "
                 "through the live vLLM server (SPEC=mtp, CTX=fast, PREFIX_CACHE off)",
       "runs": {a: [{"tag": d["tag"], "ppl": d["ppl"], "mean_nll": d["mean_nll"],
                     "n_tokens_predicted": d["n_tokens_predicted"], "n_windows": d["n_windows"],
                     "wall_s": d["wall_s"], "server_pid": d["server_pid"], "port": d["port"]}
                    for d in v] for a, v in runs.items()}}

stats = {}
for arm, v in runs.items():
    ppls = [d["ppl"] for d in v]
    mean = sum(ppls) / len(ppls)
    spread = max(ppls) - min(ppls)
    stats[arm] = {"n_runs": len(v), "ppl_mean": mean, "ppl_min": min(ppls),
                  "ppl_max": max(ppls), "duplicate_spread": spread}
out["per_arm"] = stats

b, p = stats.get("baseline"), stats.get("patch")
if b and p:
    noise = max(b["duplicate_spread"], p["duplicate_spread"])
    delta = p["ppl_mean"] - b["ppl_mean"]
    out["noise_estimate_duplicate_spread"] = noise
    out["patch_minus_baseline_ppl"] = delta
    if delta <= noise:
        out["verdict"] = "PATCH-OK"
        out["verdict_detail"] = (f"patch ppl {p['ppl_mean']:.4f} vs baseline {b['ppl_mean']:.4f}: "
                                 f"delta {delta:+.4f} within duplicate noise {noise:.4f}")
    else:
        out["verdict"] = "PATCH-REGRESSION"
        out["verdict_detail"] = (f"patch ppl {p['ppl_mean']:.4f} vs baseline {b['ppl_mean']:.4f}: "
                                 f"delta {delta:+.4f} exceeds duplicate noise {noise:.4f}")
    out["verdict_detail"] += ("; delta is " +
                              ("better (patch lower)" if delta < 0 else
                               "within noise" if delta <= noise else "worse (patch higher)"))

with open(REPO + "/analysis/rotation/ppl.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
