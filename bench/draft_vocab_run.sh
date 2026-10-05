#!/bin/bash
# The full draft-vocab arm campaign (#196), 8 arms. Appends to $RESULTS as it
# goes, so an interrupted run keeps whatever finished. Safe to re-run: it skips
# an arm whose REP lines are already present in $RESULTS.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
N=${1:-2}
export SNAPS=${SNAPS:-$HOME/arm-snapshots}
export RESULTS=${RESULTS:-$SNAPS/results.txt}
mkdir -p "$SNAPS"

have() { grep -q "REP $1 #" "$RESULTS" 2>/dev/null; }

# fullhead is the control #196 asks for: truncated head vs the full lm_head.
# Mismatched arm/cohort pairs are skipped -- a CJK head on Cyrillic text would
# only measure the head being wrong for the language.
run() {  # arm cohort
  local tag="${1}_${2}"
  if have "$tag"; then echo "skip $tag (already in results)"; return 0; fi
  case $2 in
    han)      C=$HERE/prompts_han.jsonl;;
    cyrillic) C=$HERE/prompts_cyrillic.jsonl;;
    english)  C=$HERE/prompts_real.jsonl;;
  esac
  echo "=== $1 / $2"
  bash bench/draft_vocab_arm.sh "$1" "$C" "$tag" "$N" || echo "=== $tag FAILED rc=$?"
}

run shipped  han
run cjk      han
run fullhead han
run shipped  cyrillic
run cyrillic cyrillic
run fullhead cyrillic
run shipped  english
run fullhead english

# Leave the shipped install in place whatever happened above.
M=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
cp "$SNAPS/shipped_extras.safetensors" $M/model_extra_tensors.safetensors
cp "$SNAPS/shipped_ids.pt"             $M/mtp_draft_vocab_ids.pt
cp "$SNAPS/shipped_index.json"         $M/model.safetensors.index.json
echo "=== campaign finished; shipped arm restored; results in $RESULTS"