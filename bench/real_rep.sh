#!/bin/bash
# The C1 row of bench/run_benchmarks.sh, repeated: 8 realistic prompts, 1,024 tokens each,
# one at a time. Prints tokens/step and ms/step per repeat, which is what to compare when
# two configurations look different (greedy e2e moves between sessions; tokens/step does not).
#
#   bash bench/real_rep.sh <tag> [reps] [temperature]     # temperature 0 = greedy
#   PROMPTS=<file.jsonl> bash bench/real_rep.sh <tag>      # another cohort
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
export PATH="$REPO/venv/bin:$PATH"
# One precedence chain for every caller (#113): an explicit OPENAI_API_KEY
# wins, else the server key, else api_key.txt, else the EMPTY placeholder --
# never a bare "Bearer " against a server that bound a key.
source "$REPO/resolve_api_key.sh"
resolve_client_key
M=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}; TAG=$1; N=${2:-3}; T=${3:-}
PROMPTS=${PROMPTS:-$HERE/prompts_real.jsonl}
# --model is the served name (the /tokenize alignment probe posts it as the
# request's model; a checkpoint path 404s there); --tokenizer loads locally.
B="venv/bin/vllm bench serve --host 127.0.0.1 --port 18020 --model qwen3.8-27b --tokenizer $M --served-model-name qwen3.8-27b"
# RLOGS must not be /tmp: a WSL restart wipes it and takes the measurements with
# it (#196 campaign, two lost runs). One log per rep, written as it completes.
RLOGS=${RLOGS:-/tmp}; mkdir -p "$RLOGS"
metrics() { curl -s http://127.0.0.1:18020/metrics -H "Authorization: Bearer $OPENAI_API_KEY"; }
snap() { metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total" | grep -v created | awk '{print $NF}' | tr "\n" " "; }
for i in $(seq 1 $N); do
  S0=$(snap)
  [ -n "$T" ] && TA="--temperature $T" || TA=""
  $B --dataset-name custom --dataset-path $PROMPTS --custom-output-len 1024 --num-prompts 8 --max-concurrency 1 $TA > $RLOGS/rr_$TAG_$i.log 2>&1
  S1=$(snap)
  OUT=$(awk '/Total generated tokens/ {print $4}' $RLOGS/rr_$TAG_$i.log); DUR=$(awk '/Benchmark duration/ {print $4}' $RLOGS/rr_$TAG_$i.log); E2E=$(awk '/Output token throughput/ {print $5}' $RLOGS/rr_$TAG_$i.log); TPOT=$(awk '/Mean TPOT/ {print $4}' $RLOGS/rr_$TAG_$i.log)
  python3 - "$S0" "$S1" "$TAG" "$i" "$OUT" "$DUR" "$E2E" "$TPOT" <<PY
import sys
a=[float(x) for x in sys.argv[1].split()]; b=[float(x) for x in sys.argv[2].split()]
d=[y-x for x,y in zip(a,b)]  # drafts, accepted
steps=d[0]; acc=d[1]
# The counters are cumulative and per-process. steps+acc is the tokens this run
# emitted, so it must land near OUT; when it does not, the snapshot was taken
# across something else (a warmup, a stray request) and the tok/step below is
# not this cohort's. Flag it rather than reporting a number that cannot be true.
out=float(sys.argv[5]); tot=steps+acc
bad = abs(tot-out) > 0.02*out
print(f"REP {sys.argv[3]} #{sys.argv[4]} out={sys.argv[5]} dur={sys.argv[6]}s e2e={sys.argv[7]} tok/s decode={1000/float(sys.argv[8]):.1f} tok/s | steps={steps:.0f} tok/step={1+acc/steps:.2f} ms/step={1000*float(sys.argv[6])/steps:.1f}"
      + (f" COUNTER-MISMATCH emitted={tot:.0f} vs out={out:.0f}" if bad else ""))
PY
done
