# Salted Quantization Spike — Originals (BF16) Probe — Findings

**Date:** 2026-10-12 · **Branch:** `spike/salted-quantization`
**Plan:** `docs/superpowers/plans/2026-10-12-salted-quantization-fp16-originals.md`
**Question:** Does the original (unquantized) Qwen3.8-27B checkpoint carry position-dependent
variance structure that the dequantized W4A16 baseline hid — i.e. is η² on the originals
materially above the dequant baseline (max 4.318e-4), enough to revive salted quantization?

## Verdict: hypothesis FALSIFIED — the null reproduces on the originals

Testing on the original weights gives the **same view** as testing on the dequantized
weights: η² per tensor is within 0.95–1.02× of the baseline on all 9 sampled tensors, and
the max anywhere is identical to 4 significant figures (4.3182e-4 originals vs 4.3180e-4
baseline, lm_head `linear%64`). The AutoRound group-128 scales were not hiding positional
structure — the original BF16 weights are themselves near-iid per position. There is nothing
for a salted codebook to exploit, on either baseline.

Salted quantization is dead for this model on **both** baselines. Revival path #1 from the
no-go findings is exhausted; the spike should close. Remaining revival paths (coarse/no
group scales, output-tied metrics, outlier pairing) target different quantization regimes
and need new measurements in those regimes — they are not re-runs of this probe.

## Dtype correction (read before quoting)

The no-go findings say "FP16 originals". The original checkpoint is **BF16**: safetensors
headers report BF16 for every sampled tensor, and the original config declares
`"dtype": "bfloat16"` nested in `text_config` (there is no top-level `dtype` — the top-level
one in the W4A16 config was written by the quantizer). `tie_word_embeddings: false`
confirmed on the original. All measurements below ran on the BF16 originals; "FP16" in the
earlier doc should be read as "the unquantized 2-byte weights".

## Task 0 — fetch + verify (PASS)

- Source: `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, 32 files, 55.6 GB,
  18 safetensors shards → `models-fp16/Qwen3.8-27B/` (gitignored; `models/` untouched).
  Download wall 495.1 s. (`analysis/salted/fp16/fetch_originals.py`, `fetch.json`,
  `expected.json`)
- All 9 sampled tensors present; shapes match the W4A16 `weight_shape` exactly.
- Cross-check vs the dequant baseline (layer 27 q/gate/down, streaming over full tensors):
  corr = 0.994252 / 0.994687 / 0.994561, rel MSE ≈ 1.1e-2 of weight variance — this is the
  checkpoint the AutoRound quant was made from. (A wrong repo/revision would show corr ≪ 0.99.)

## Task 1 — loader swap + regression gate (PASSED)

- `analysis/salted/fp16/`: `fp16_common.py` (index-based shard mapping from the original's
  own `model.safetensors.index.json`; chunked BF16→float32 reader), `heterogeneity_fp16.py`,
  `codebook_gain_fp16.py`. The StreamingEta accumulator and the Lloyd-Max / streaming-MSE
  machinery are IMPORTED UNCHANGED from the baseline scripts; only the chunk source differs
  (no unpack/dequant exists on this source).
- Synthetic gates: PASS (row-structured η² = 0.5675–0.5676; iid worst ≈ 0 over all 20 defs
  at 3 real shapes). Lloyd-Max theory check: 4-level iid Gaussian MSE/var = 0.11771
  (theory 0.1175).
- Regression gate: the W4A16 baseline cells re-run through the fp16 scripts' own wiring
  reproduce EVERY scalar cell of the final `analysis/salted/codebook_gain.json` at 1e-12
  relative, zero exceptions (wall 393.6 s). Note: the plan named `codebook_gain.pre_fix.json`
  as the target; the final `codebook_gain.json` is the post-review baseline the no-go
  findings cite, so it is the stricter and correct target. (`regression_gate.json`)

## Task 2 — η² on the originals (the core probe)

Load sanity (null-is-guilty check): BF16, std 0.01017, p1 −0.0238 / p50 0.0000 / p99 0.0238,
50.0 % positive — real weight statistics, not a degenerate load. iid noise floor at the real
projection shapes: 7.3e-7 – 9.7e-7 (baseline measured 5.5–9.2e-7 — same ballpark).

Best salt per tensor, originals vs dequant baseline (full 20-cell tables in
`analysis/salted/fp16/heterogeneity.json`):

| tensor | best salt | η² originals | η² baseline | ratio |
|---|---|---|---|---|
| gate L0 | linear%64 | 4.2e-5 | 4.1e-5 | 1.02 |
| down L0 | row+col%64 | 1e-6 | 1e-6 | 0.98 |
| q L27 | linear%64 | 6.1e-5 | 6.0e-5 | 1.01 |
| gate L27 | linear%64 | 9.2e-5 | 9.1e-5 | 1.01 |
| down L27 | linear%64 | 1e-6 | 1e-6 | 0.95 |
| q L63 | linear%64 | 6.6e-5 | 6.6e-5 | 1.00 |
| gate L63 | linear%64 | 1.53e-4 | 1.51e-4 | 1.01 |
| down L63 | row%64 | 1e-6 | 1e-6 | 1.02 |
| lm_head | linear%64 | **4.3182e-4** | **4.3180e-4** | 1.00 |

- Max anywhere: 4.3182e-4 (originals) vs 4.3180e-4 (baseline) — identical to 4 sig figs.
- `row%N` above 10× floor: **0 of 9 tensors** on the originals — and also 0 of 9 on the
  baseline. Values at ~1e-6 are at the noise floor, never signal.
- The only above-floor signal anywhere is lm_head `linear%64` ≈ 4.3e-4 (≈450× floor),
  present identically in both sources. Its consequence was already measured on the
  baseline: r_s = 0.9106 at 4-bit — the only sub-1.0 cell, still above the 0.9
  "meaningful gain" bar.

Two independent loaders (the pinned compressed-tensors dequant path and the plain BF16
reader) agree on η² to within 2 % per tensor — the null is cross-validated, not an artifact
of either path.

## Task 3 — codebook gain on the originals: SKIPPED per the plan's gate

Gate (plan, Task 2 Step 4): proceed only if max η² > 10× the baseline max (> ~4.3e-3) or
any `row%N` cell above floor. Measured: max = 1.00006× baseline; `row%N` above floor 0/9.
**Gate fails → skipped** (`analysis/salted/fp16/task3_skip.json`).

Why the skip is sound, not a shortcut: the baseline Task 3 established empirically that
r_s tracks η² (lm_head, largest η², gets the only real benefit; everything else is within
fit noise of 1.0). With η² unchanged to 4 significant figures, r_s on the originals is a
determined quantity — re-running the sweep would re-measure it, not produce new evidence.
The spec's headline claim ("salted INT2 matches INT4") was already falsified on the
baseline (best salted result anywhere: 8.9 % MSE reduction, lm_head 4-bit); unchanged η²
means unchanged headroom.

## What this means

- The original hypothesis — "AutoRound's group-128 scales absorbed the position information
  the salt would exploit; the originals may still have it" — is **falsified**. The scales had
  nothing positional to absorb; the originals are near-iid per position, just like the
  dequantized weights.
- The no-go stands on both baselines. A salted INT2/INT3 kernel for this checkpoint has no
  measured headroom: the best salted result anywhere on either weight source is an 8.9 % MSE
  reduction on one tensor (lm_head, 4-bit), far below the spec's headline claim.
- Revival paths 2–4 from the no-go findings (coarse/no group scales; output-tied metrics;
  Bonsai-2-style outlier pairing) target different quantization regimes and would need new
  measurements in those regimes — they are not re-runs of this probe.

## Artifacts & reproduction

- `analysis/salted/fp16/fetch_originals.py` + `fetch.json` + `expected.json` — Task 0
  (`venv/bin/python analysis/salted/fp16/fetch_originals.py [--skip-download]`)
- `analysis/salted/fp16/fp16_common.py`, `heterogeneity_fp16.py` + `heterogeneity.json` +
  `heterogeneity_fp16.log` — Tasks 1–2
- `analysis/salted/fp16/codebook_gain_fp16.py` (+ `--regression`), `regression_gate.json`,
  `task3_skip.json` — Task 1 gate + Task 3 skip
- `analysis/salted/fp16/watch_gui.py` — tkinter progress monitor for the whole probe
  (`python analysis\salted\fp16\watch_gui.py` on Windows; `--once` = headless smoke test)
- `models-fp16/Qwen3.8-27B/` — 55.6 GB originals (gitignored; delete freely)
- SDD ledger: `.superpowers/sdd/2026-10-12-salted-quantization-spike/progress.md`
- Baseline record (unchanged): `docs/superpowers/plans/2026-10-12-salted-quantization-findings.md`
