#!/bin/bash
# smoke.sh: a few real prompts through the live server; fail if output is
# empty, or obviously garbage (no letters / repeated single char).
set -e
cd /home/hubbe/qwen-serving
PY=analysis/rotation/ppl_smoke_out
mkdir -p $PY
venv/bin/python - <<'EOF'
import json, sys
sys.path.insert(0, "bench")
import harness
prompts = [
    "What is the capital of France? Answer in one sentence.",
    "Write one short Python function that returns the sum of a list of integers.",
    "Briefly explain why the sky is blue (2 sentences max).",
]
ok = True
for i, p in enumerate(prompts):
    r = harness.post("/v1/chat/completions",
                     {"messages": [{"role": "user", "content": p}],
                      "max_tokens": 120, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False}})
    txt = r["choices"][0]["message"]["content"] or ""
    alpha = sum(c.isalpha() for c in txt)
    uniq = len(set(txt))
    print(f"--- prompt {i}: {p}\n{txt[:300]}\n[len={len(txt)} alpha={alpha} uniq_chars={uniq}]")
    if len(txt) < 20 or alpha < 0.5 * len(txt) or uniq < 10:
        ok = False
print("SMOKE_OK" if ok else "SMOKE_FAIL")
EOF
