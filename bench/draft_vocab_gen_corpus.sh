#!/bin/bash
# Boot the shipped arm, generate a Chinese ranking corpus, shut down.
# Separate from the arm runner because this is setup, not a measurement.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
M=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
SNAPS=${SNAPS:-$HOME/arm-snapshots}
OUT=${1:-$REPO/bench/testdata/zh_corpus_model.txt}
LOG=$SNAPS/gencorpus.log
mkdir -p "$SNAPS"

# Nothing else may hold 18020: measuring (or generating) against a server that is
# not the one we started is how an arm silently reports the wrong head.
if ss -ltn 2>/dev/null | grep -q ':18020 '; then
  echo "port 18020 already in use; refusing to start" >&2; exit 1
fi

: > $LOG
SPEC=mtp CTX=long MODEL=$M MAX_SEQS=2 bash single-user/start_qwen.sh >> $LOG 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; wait $SRV 2>/dev/null' EXIT

code=000
for i in $(seq 1 120); do
  code=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:18020/health || true)
  [ "$code" = "200" ] && break
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED:"; tail -20 $LOG; exit 1; }
  sleep 5
done
[ "$code" = "200" ] || { echo "SERVER NOT UP:"; tail -20 $LOG; exit 1; }
echo "server up"

# Warmup before anything else touches the counters: the boot profile runs only
# profiling dummies, so early requests still pay first-batch allocation and can JIT.
bash bench/warmup.sh && echo "warmup ok"

source "$REPO/resolve_api_key.sh"; resolve_client_key
venv/bin/python bench/draft_vocab_gen_corpus.py --out "$OUT" --key "$OPENAI_API_KEY"
RC=$?
kill $SRV 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; wait $SRV 2>/dev/null
trap - EXIT
echo "gen corpus rc=$RC -> $OUT"
exit $RC