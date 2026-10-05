#!/bin/bash
# Rebuild the arm snapshots under $SNAPS (default ~/arm-snapshots, NOT /tmp).
# The shipped arm is snapshotted from whatever is currently installed; the cjk
# and cyrillic arms are rebuilt from the variant id lists, which are a pure
# function of the base list + corpus + budget, so the rebuild is deterministic.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
M=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
SNAPS=${SNAPS:-$HOME/arm-snapshots}
mkdir -p "$SNAPS"

snap_current() {  # $1 = name
  cp $M/model_extra_tensors.safetensors "$SNAPS/$1_extras.safetensors"
  cp $M/mtp_draft_vocab_ids.pt             "$SNAPS/$1_ids.pt"
  cp $M/model.safetensors.index.json         "$SNAPS/$1_index.json"
  echo "snapshotted $1: $(venv/bin/python -c "import torch;print(torch.load('$M/mtp_draft_vocab_ids.pt',map_location='cpu').numel())") ids"
}

echo "=== current install -> shipped"
snap_current shipped

for L in "cjk zh bench/testdata/zh_corpus.txt" "cyrillic ru bench/testdata/ru_corpus.txt"; do
  set -- $L; ARM=$1; CORPUS=$3
  echo "=== building $ARM variant ids from $CORPUS"
  venv/bin/python prepare/build_draft_vocab.py $M --ids prepare/draft_vocab_ids.json \
      --corpus "$CORPUS" --add-scripts "$ARM" --script-budget 16384 \
      --out "$SNAPS/ids_$ARM.json" | grep -E "union|written"
  echo "=== building $ARM variant head"
  venv/bin/python prepare/build_draft_vocab.py $M --ids "$SNAPS/ids_$ARM.json" | tail -2
  snap_current "$ARM"
done

echo "=== rebuilding the shipped head from the shipped id list"
# NOT snap_current here: by this point the install holds whichever variant was
# built last, so re-snapshotting would label that variant "shipped". The head
# slice is an index_select over the ids, so rebuilding from
# prepare/draft_vocab_ids.json is deterministic and restores the 40,960-row head.
venv/bin/python prepare/build_draft_vocab.py $M --ids prepare/draft_vocab_ids.json | tail -2
snap_current shipped

echo "=== restoring the shipped install"
cp "$SNAPS/shipped_extras.safetensors" $M/model_extra_tensors.safetensors
cp "$SNAPS/shipped_ids.pt"             $M/mtp_draft_vocab_ids.pt
cp "$SNAPS/shipped_index.json"         $M/model.safetensors.index.json
echo "installed now: $(venv/bin/python -c "import torch;print(torch.load('$M/mtp_draft_vocab_ids.pt',map_location='cpu').numel())") ids"
echo "done. snapshots in $SNAPS"
ls -la "$SNAPS"