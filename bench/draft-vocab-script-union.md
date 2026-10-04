# A per-script draft-vocabulary union (a capped variant of the shipped list)

Branch for [syv-ai/HyperQwen#196](https://github.com/syv-ai/HyperQwen/issues/196). Nothing here
changes the shipped list or the default path: `prepare/draft_vocab_ids.json` is untouched, and
`--add-scripts` writes a *separate* variant file that you build a head from on purpose.

## What the CPU-only measurement says

`--stats`, on `Qwen3.8-27B-W4A16-AutoRound-fast` (248,044 rows; 247,100 whole-UTF-8, 944 BPE
fragments):

| script | vocab rows | in the shipped 40,960 | % of script |
|---|---:|---:|---:|
| han | 55,328 | **3** | 0.0% |
| kana | 5,476 | **2** | 0.0% |
| hangul | 6,807 | **4** | 0.1% |
| cjk_punct | 231 | **1** | 0.4% |
| cyrillic | 18,580 | **1** | 0.0% |
| latin_ext | 14,282 | 354 | 2.5% |
| **CJK total** | **67,842** | **10** | **0.0%** |

Ten CJK ids out of 67,842. This is type-coverage (rows), the lower bound on #196's 5.4-8.0%,
which is token-weighted over running text. The two are consistent, not in conflict.

## Why a union is capped rather than complete

A head row on this checkpoint is **2,560 B** (`(40960, 640)` int32). So:

| variant | ids | added rows | added VRAM |
|---|---:|---:|---:|
| complete CJK union | 106,880 | +65,920 | **+168.8 MB** |
| complete CJK + Cyrillic | 125,459 | +84,499 | **+216.3 MB** |
| latin_ext union | 54,888 | +13,928 | +35.6 MB |
| **CJK, capped at 16,384 rows** | **57,344** | **+16,384** | **+41.9 MB** |

The complete union is 2.6x the shipped head. The cap keeps "the shortlist is short" — the
property the whole optimisation rests on — and moves the VRAM decision into an explicit knob.
At the 16,384 cap on a ~200-token Chinese sample, Han coverage goes **0.0% -> 28.7%** and CJK
overall **0.0% -> 24.2%**, for +41.9 MB. Tighter budgets buy less coverage; there is no
setting that reaches completeness without the VRAM.

## Use

```bash
# 1. measure the shipped list against this checkpoint (CPU, no GPU)
venv/bin/python prepare/build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound-fast \
    --ids prepare/draft_vocab_ids.json --stats

# 2. write a variant id list (writes one file; the model dir is unchanged)
venv/bin/python prepare/build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound-fast \
    --ids prepare/draft_vocab_ids.json --corpus your_traffic.txt \
    --add-scripts cjk --script-budget 16384 --out prepare/draft_vocab_ids_cjk.json

# 3. build the variant head from it
venv/bin/python prepare/build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound-fast \
    --ids prepare/draft_vocab_ids_cjk.json
```

`--add-scripts` takes a comma list of `han,kana,hangul,cjk_punct,cjk,cyrillic,latin_ext`, and
`cjk` is the first four as a group. The corpus is what ranks the added rows, so **pass your own
traffic**: a cap over a corpus that does not resemble your output spends rows on tokens you
never emit.

## Invariants, checked

- the base ids stay an ordered prefix, so the variant list is a superset and the superset and
  `verify.sh` byte-compare checks in #196 still hold for the base file;
- added ids are sorted, unique, and disjoint from the base;
- ties break by id, so a rebuild from the same corpus is byte-identical (verified: two runs,
  same sha256).

## Two things this does not settle

**No acceptance or tok/s numbers.** Those need a GPU run on Han and Cyrillic prompt sets, with
the variant against the shipped list at `CTX=long`, which is the arm #196 says to trust. The
3090 here is the box, but the card is not free; nothing in this branch has run a decode step.
The claim being tested is narrow: at equal coverage a truncated head beats the full head, and
more CJK coverage at bounded VRAM beats less.

**The corpus is a ~200-token fixture**, not traffic. It exercises the ranking and the cap, and
it is why this branch's coverage figures are a smoke test rather than a measurement. Real
numbers need a real corpus.

## Also fixed on the way

`--corpus` took every argument after it, so any flag written after it landed in the file list
and became a filename (`FileNotFoundError: '--out'`). It now stops at the next flag.

`--add-scripts` ranks on both corpus splits, not just the 90% one. The held-out tenth exists to
report unbiased *coverage*; a corpus small enough to be one text lands entirely in it, which
would leave the ranking with nothing to rank by.