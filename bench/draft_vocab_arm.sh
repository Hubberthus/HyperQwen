#!/bin/bash
# Draft-vocab arm campaign (#196): swap the installed head, boot, measure.
#
#   bash bench/draft_vocab_arm.sh <arm> <cohort.jsonl> <tag> [reps]
#
# Arms: shipped | cjk | cyrillic | fullhead
#   shipped|cjk|cyrillic swap model_extra_tensors + mtp_draft_vocab_ids.pt + index
#   fullhead keeps the shipped files and sets MTP_DRAFT_VOCAB=0 (the control
#   #196 asks for: truncated head vs the full lm_head at CTX=long)
#
# Nothing here lives in /tmp. A WSL restart wipes /tmp, and that cost two runs of
# this campaign: the first was killed mid-flight, the second left no recoverable
# output at all. Snapshots go to $SNAPS (under $HOME) and every REP line is
# appended to $RESULTS as it is produced, so a killed run still leaves behind the
# reps that finished.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
M=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
SNAPS=${SNAPS:-$HOME/arm-snapshots}
RESULTS=${RESULTS:-$SNAPS/results.txt}
ARM=$1; COHORT=$2; TAG=$3; N=${4:-2}
LOG=$SNAPS/server_$TAG.log
mkdir -p "$SNAPS"
echo "$(date -Is) ARM=$ARM COHORT=$COHORT TAG=$TAG REPS=$N" >> $RESULTS

install_arm() {
  case "$1" in
    shipped) P=shipped;; cjk) P=cjk;; cyrillic) P=cyrillic;;
    *) echo "unknown arm $1" >&2; exit 2;;
  esac
  for f in extras.safetensors ids.pt index.json; do
    [ -f "$SNAPS/${P}_$f" ] || { echo "missing snapshot $SNAPS/${P}_$f" >&2; exit 3; }
  done
  cp "$SNAPS/${P}_extras.safetensors" $M/model_extra_tensors.safetensors
  cp "$SNAPS/${P}_ids.pt"             $M/mtp_draft_vocab_ids.pt
  cp "$SNAPS/${P}_index.json"         $M/model.safetensors.index.json
  echo "$(date -Is) installed arm=$1"
}

case "$ARM" in
  shipped|cjk|cyrillic) install_arm "$ARM" >> $RESULTS;;
  fullhead) echo "$(date -Is) arm=fullhead (MTP_DRAFT_VOCAB=0, shipped files)" >> $RESULTS;;
  *) echo "unknown arm $ARM" >&2; exit 2;;
esac

: > $LOG
if [ "$ARM" = "fullhead" ]; then export MTP_DRAFT_VOCAB=0; else export MTP_DRAFT_VOCAB=1; fi
SPEC=mtp CTX=long MODEL=$M MAX_SEQS=2 bash single-user/start_qwen.sh >> $LOG 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT

code=000
for i in $(seq 1 120); do
  code=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:18020/health || true)
  [ "$code" = "200" ] && break
  kill -0 $SRV 2>/dev/null || { echo "$(date -Is) $ARM $TAG SERVER DIED" >> $RESULTS; tail -20 $LOG; exit 1; }
  sleep 5
done
[ "$code" = "200" ] || { echo "$(date -Is) $ARM $TAG SERVER NOT UP" >> $RESULTS; tail -20 $LOG; exit 1; }
IDS=$(venv/bin/python -c "import torch;print(torch.load('$M/mtp_draft_vocab_ids.pt',map_location='cpu').numel())")
echo "$(date -Is) $ARM $TAG server up draft_ids=$IDS MTP_DRAFT_VOCAB=$MTP_DRAFT_VOCAB" >> $RESULTS

# Line-buffered, and appended per line: a buffering pipe that dies with the
# process takes everything it had not yet flushed with it. That is how the first
# two runs lost their numbers.
MODEL=$M PROMPTS=$COHORT RLOGS=$SNAPS bash bench/real_rep.sh "$TAG" "$N" 0 \
  | stdbuf -oL grep -E "^REP" \
  | while IFS= read -r l; do echo "$l" >> $RESULTS; echo "$l"; done
RC=${PIPESTATUS[0]}
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
trap - EXIT
echo "$(date -Is) $ARM $TAG done rc=$RC" >> $RESULTS
exit $RC