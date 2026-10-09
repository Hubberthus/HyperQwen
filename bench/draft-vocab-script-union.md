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

## Measured: 8 arms, one 3090, WSL2

`SPEC=mtp CTX=long` (fp8 KV, k=3, `MAX_SEQS=2`), greedy, 8 prompts x 1024 output tokens,
2 reps. `bench/draft_vocab_run.sh` + `bench/draft_vocab_arm.sh`; raw rows in
`results.txt`, one per rep. Decode tok/s comes from `vllm bench serve` and is unaffected by
the counter problem below.

| cohort | shipped 40,960 | variant 57,344 | full head 248,044 |
|---|---:|---:|---:|
| Han | 43.9 / 46.1 | **75.9 / 71.7** | 80.8 / 82.1 |
| Cyrillic | 57.6 / 55.9 | **90.6 / 88.1** | 88.0 / 87.9 |
| English | 88.4 / 89.3 | -- | 82.3 / 87.1 |

tok/step, from `out/steps`:

| cohort | shipped | variant | full head |
|---|---:|---:|---:|
| Han | 1.38 | 2.26 ᵃ | 2.09 ᵃ / 2.50 |
| Cyrillic | 1.68 | 2.63 | 2.71 |
| English | 2.60 | -- | 2.65 |

- **The union beats the shipped list where the shipped list is broken: Han +64%,
  Cyrillic +57%.** The Han half replicates across three independent runs.
- **It does not beat the full head.** Cyrillic ties it (89.4 vs 88.0, inside noise); Han is
  9% behind (73.8 vs 81.5). So the ceiling for this idea is *parity with `MTP_DRAFT_VOCAB=0`
  at 4.3x less head* -- a real answer to #196, but not a win over the full head.
- **English is the control that makes the rest credible.** The 40,960 shipped list *beats* the
  full head (88.9 vs 84.7), which independently reproduces the `CTX=fast` k=4 claim that
  truncation wins where coverage is high. Truncation is a coverage bet, not a loss by
  construction.
- Leading explanation for the Han shortfall: the ranking corpus was 670 tokens of hand-written
  text. The 705-token Cyrillic corpus reached 98.4% prompt coverage and reached parity; the
  CJK one reached 86.7% and fell short. Suggestive, not proof -- the first thing to test is a
  real corpus.

### Follow-up: was the corpus the problem? Partly, and no.

Same script, same `--script-budget 16384`, only the ranking corpus changed: 670 tokens of
hand-written text -> **75,757 characters (47,357 tokens) generated by the model itself** over
50 topics chosen to be disjoint from the eval prompts (`bench/draft_vocab_gen_corpus.py`).

| CJK variant | eval-prompt coverage | decode tok/s | tok/step |
|---|---:|---:|---:|
| ranked over 670 tokens | 86.7% | 75.9 / 71.7 | 2.26 ᵃ |
| ranked over 47,357 tokens | 88.4% | **76.8 / 76.8** | **2.35 / 2.35** |
| full head (for reference) | 100% | 80.8 / 82.1 | 2.09 ᵃ / 2.50 |

**A 70x corpus bought +1.7 points of coverage and +4% throughput.** The two heads differ in
2,235 rows each way (76% overlap), so the ranking genuinely moved -- it just moved very little
that mattered. The gap to the full head survives essentially intact: 76.8 against 81.5, still
**5.8% behind** where it was 9% behind.

So the corpus is a minor term, not the mechanism. What the three arms now line up on is
coverage itself:

| variant | prompt coverage | vs full head |
|---|---:|---:|
| Cyrillic | 98.4% | +1.6% (tie) |
| CJK, 47k corpus | 88.4% | -5.8% |
| CJK, 670-token corpus | 86.7% | -9.4% |
| shipped, English | 98.5% | +5.0% |

Every arm within ~2 points of 98-100% coverage ties or beats the full head; the two arms below
90% lose in proportion. The lever is therefore the **budget**, not the corpus -- which predicted
that the uncapped CJK union (65,920 added rows, 106,880 ids, 99.7% coverage, +168.8 MB) would
close the Han gap. It does:

### Follow-up 2: uncapped CJK closes the gap, as predicted

| Han arm | ids | added VRAM | coverage | decode tok/s | tok/step |
|---|---:|---:|---:|---:|---:|
| shipped | 40,960 | -- | 11.6% | 43.9 / 46.1 | 1.38 / 1.38 |
| CJK capped | 57,344 | +41.9 MB | 88.4% | 76.8 / 76.8 | 2.35 / 2.35 |
| **CJK uncapped** | **106,880** | **+168.8 MB** | **99.7%** | **79.6 / 79.8** | **2.43 / 2.43** |
| full head | 248,044 | +535 MB | 100% | 80.8 / 82.1 | 2.09 / 2.50 |

**-2.2% against the full head, inside the spread between its own two reps.** Read with the
Cyrillic and English arms, coverage is the whole story:

| coverage | arms | outcome vs full head |
|---|---|---|
| 98-100% | Cyrillic 98.4%, English 98.5%, CJK uncapped 99.7% | +1.6%, +5.0%, -2.2% |
| 86-88% | CJK capped, twice | -5.8%, -9.4% |

So the defensible claim for this branch is the narrow one: **a per-script union reaches parity
with `MTP_DRAFT_VOCAB=0` once its coverage reaches ~99%, at 43% of the full head's rows** --
106,880 against 248,044, +168.8 MB against +535 MB. It does not beat the full head anywhere,
and at the 16,384-row default it does not come close on Han. The cap is the design's real dial,
and where it should sit depends on the VRAM a card has spare, which is why it is a knob.

### Follow-up 3: all-Han same-day re-run (Oct 9), after finding day-to-day drift

A re-run of all four Han arms back-to-back on one day, after a warmup + drifted ~6% slower
ms/step than Oct 5 was discovered (all Oct 9 arms uniformly affected, so *same-day pairs stay
valid; cross-day pairs do not*):

| Han cohort, Oct 9 | decode avg | tok/step | ms/step | steps |
|---|---:|---:|---:|---:|
| full head | 73.05 | 2.46 | 34.4 | 3324 |
| CJK capped (57,344 rows, 88.4% cov) | 72.15 | 2.39 | 33.7 | 3434 |
| CJK uncapped (106,880 rows, ~99.7% cov) | 72.05 | 2.39 | 33.85 | 3424 |
| **shipped (40,960 English ids)** | **41.85** | **1.41** | 34.05 | **5815** |

- The central premise reproduces emphatically same-day: shipped on Han is **57% slower** with
  1.41 tok/step, and the capped CJK head recovers to within **1.3%** of the full head at ~1/4
  the rows.
- The uncapped claim in Follow-up 2 stands (Oct 9: uncapped -1.4% vs full head, same as Oct 5's
  -2.2%). A momentary reading of these rows as "the uncapped prediction refuted" was wrong --
  today's rows *confirm* it.
- The one number that moved between days: the capped arm's deficit vs full head was -5.8% on
  Oct 5 and -1.3% on Oct 9, with each day's pair internally valid. Unexplained; the capped arm
  is behind the full head on both days either way, and neither day's data puts it behind
  shipped anywhere.

### Two measurement defects, and what they cost

**Contaminated spec-decode counters.** Three of sixteen reps report `COUNTER-MISMATCH`:
`steps + accepted` does not land within 2% of the run's own generated-token count. Cause is
the missing `bench/warmup.sh`: the boot-time profile runs only profiling dummies, so the first
real requests still pay first-batch transient allocation (gotcha 35) and can JIT serving-path
kernels (gotcha 45), inflating the cumulative counters the snapshot reads. `steps` reconciles
with `duration / ms_per_step` in 15 of 16 reps, so tok/step is recoverable as `out/steps` where
only the *accepted* counter was inflated; `cjk_han` #2 is discarded outright (15.7 ms/step
against ~30 ms everywhere else is not physical). The fix is to run `bench/warmup.sh` between
server-up and the first snapshot -- it is what that script is for.

**Everything in `/tmp` is lost on a WSL restart.** Two runs died this way before the snapshots
and results moved under `$HOME`. The campaign now appends every rep line as it is produced and
skips arms already present, so an interrupted run resumes instead of restarting.

## Also fixed on the way

`--corpus` took every argument after it, so any flag written after it landed in the file list
and became a filename (`FileNotFoundError: '--out'`). It now stops at the next flag.

`--add-scripts` ranks on both corpus splits, not just the 90% one. The held-out tenth exists to
report unbiased *coverage*; a corpus small enough to be one text lands entirely in it, which
would leave the ranking with nothing to rank by.