# Salted Quantization Spike — Originals (FP16/BF16) Probe Plan

> **Status (2026-10-13): COMPLETE — hypothesis FALSIFIED.** The null reproduces on the BF16
> originals (max η² identical to 4 sig figs on both baselines); see
> 2026-10-12-salted-quantization-fp16-findings.md. Revival path #1 exhausted.

> **Project class: Spike (follow-up).** Output is an answer (numbers + recommendation), not shipped code.
> Executes revival path #1 from the Task 4 findings: test on the pre-quantization weights.

**Question:** Does position-dependent variance structure exist in the *original* (unquantized)
Qwen3.8-27B weights — structure that AutoRound's group-128 scales absorbed? I.e. is η² on the
originals materially above the dequantized baseline, and if so does salted beat global at equal
bits/weight?

**Baseline for comparison (cite, do not re-derive):**
`docs/superpowers/plans/2026-10-12-salted-quantization-findings.md`
- max η² = 4.318e-4 (lm_head `linear%64`); `row%N` flat 0.0000 on every tensor
- best salted/global MSE ratio r_s = 0.9106 (lm_head, 4-bit); no tensor crosses 0.9

**Tech stack:** same `venv/` (Python 3.12, torch, safetensors, numpy; `huggingface_hub` 1.33.0
confirmed installed). CPU only, RAM-tiled ≤4 GB. No GPU. Probe protocol per the spike skill's
`references/quantized-weights-probes.md` (exact streaming η², deterministic Lloyd-Max,
regression gating, spawn-based parallel sweep).

**Source checkpoint:** `Qwen/Qwen3.8-27B` @ huggingface.co (repo README references it; confirm
revision against the quantized checkpoint). ~54 GB at 2 bytes/param. Disk gate already passed:
821 GB free on the WSL fs (checked 2026-10-12).

**Dtype note:** the findings doc says "FP16 originals", but the serving checkpoint's config
declares `"dtype": "bfloat16"` (`models/Qwen3.8-27B-W4A16-AutoRound/config.json:5` and `:319`).
Expect the original to ship BF16; record the dtype from the safetensors headers and measure
whatever ships — label it correctly in the report.

## Global Constraints
- Every metric cites its computation; numbers as measured; a failed hypothesis is retracted plainly.
- Scratch outputs → `analysis/salted/fp16/` (new subdir). Do NOT clobber the dequant-baseline
  JSONs — `analysis/salted/codebook_gain.pre_fix.json` is the regression baseline for Task 1.
- Do not touch serving configs, the Docker image, or anything under `models/`. The download goes
  to a NEW top-level dir `models-fp16/Qwen3.8-27B/`; add `models-fp16/` to `.gitignore` (current
  .gitignore covers `models/` only).
- Same salt defs (`linear/row/col/row+col % N`, N ∈ {4,8,16,32,64}), same 9 sampled tensors
  (L0/L27/L63 q,gate,down + lm_head), same Lloyd-Max settings (equal-percentile init, ≤20 iters,
  deterministic, float32 data / float64 math), same fit-subsample rules (r%65; r%17 for the
  5120-row down_proj row-salt cell) — every number must be directly comparable to the findings table.
- No serving-path changes at any point.

---
### Task 0: Fetch + verify the originals (gate)
**Files:** `analysis/salted/fp16/fetch_originals.py`
**Produces:** `analysis/salted/fp16/fetch.json`
- [ ] Step 1: pin source repo + revision. Expect `Qwen/Qwen3.8-27B`; confirm arch
      `Qwen3_5ForConditionalGeneration` and the 64-layer / 5120-hidden geometry match the W4A16
      config. Record the exact revision sha in `fetch.json`.
- [ ] Step 2: download via `huggingface_hub.snapshot_download(..., revision=<sha>)` into
      `models-fp16/Qwen3.8-27B/`. ~54 GB — the long pole; report download wall time.
- [ ] Step 3: verify shard count + total bytes; record dtype from safetensors headers (expect
      BF16 per config — see dtype note); record `tie_word_embeddings` from the original's config
      (the W4A16 config says `false` at config.json:421 and :425 — verify it holds on the original).
- [ ] Step 4: cross-check against the baseline: load one full-attention layer's q/gate/down both
      ways (original vs dequantized W4A16) and record per-tensor correlation + MSE. Correlation
      ≳0.99 confirms we fetched the checkpoint the quant was made from; materially lower → wrong
      repo/revision, abort.
- [ ] Step 5: add `models-fp16/` to `.gitignore`.

### Task 1: Loader swap (reuse, don't rewrite)
**Files:** `analysis/salted/fp16/heterogeneity_fp16.py`, `analysis/salted/fp16/codebook_gain_fp16.py`
(or a `--source` flag on the existing scripts — pick one; keep the old entry points runnable)
**Produces:** refactored scripts + regression gate result
- [ ] Step 1: extract the streaming η² accumulator and the Lloyd-Max + streaming-MSE code
      UNCHANGED; replace only the weight source: a plain safetensors reader for the original's
      BF16/FP16 tensors, no unpack/dequant step. The `load_packed`/`verify_pin` machinery stays
      with the baseline scripts only.
- [ ] Step 2: synthetic unit gates must still pass: heterogeneous-by-row synthetic → η² ≫ 0 for
      `row % N`; iid Gaussian → η² ≈ 0 for all defs; Lloyd-Max on iid matches the 0.1175·σ²
      optimum; salted beats global on row-structured synthetic, ties (≈1.0006) on iid.
- [ ] Step 3: regression gate: re-run the dequant-baseline sweep's unchanged cells and require
      exact reproduction of `codebook_gain.pre_fix.json` (1e-12 tolerance). Abort on mismatch.
- [ ] Step 4: axis semantics unchanged — stored Linear weights are W[out, in], row = output
      channel (gate: MLP-N on rows; down: MLP-N on cols). Verify one tensor's original shape
      against the W4A16 dequantized shape.

### Task 2: Position heterogeneity on the originals (the core probe)
**Files:** `analysis/salted/fp16/heterogeneity_fp16.py`
**Produces:** `analysis/salted/fp16/heterogeneity.json` + printed table
- [ ] Step 1: same sweep: 9 tensors × {linear,row,col,row+col} × N ∈ {4,8,16,32,64}, exact
      streaming η² on the raw original weights, ≤4 GB CPU tiles.
- [ ] Step 2: measure the iid noise floor at the REAL shapes (baseline measured 5.5–9.2e-7);
      report at/below-floor values as "at noise floor", never as signal.
- [ ] Step 3: verdict line per tensor: best def + N + value, printed SIDE-BY-SIDE with the
      dequant-baseline value from the findings doc.
- [ ] Step 4: gate for Task 3 — proceed only if the originals show material signal: suggested bar
      max η² > 10× the baseline max (> ~4.3e-3), or any `row%N` cell above floor. Otherwise record
      the skip and jump to Task 4.

### Task 3: Codebook gain on the originals (conditional on the Task 2 gate)
**Files:** `analysis/salted/fp16/codebook_gain_fp16.py`
**Produces:** `analysis/salted/fp16/codebook_gain.json` + log
- [ ] Step 1: global vs per-salt Lloyd-Max at k = 4/8/16 (2/3/4 bits), exact streaming MSE over
      full tensors; fit subsample per the documented rules (r%65; r%17 for the 5120-row down_proj
      row-salt cell); assert every salt class non-empty before fitting.
- [ ] Step 2: report r_s per tensor × bits, next to the baseline table from the findings doc.
- [ ] Step 3: run the honest 2-step arm (r_ss: per-class absmax + one shared k-entry book) only if
      some r_s < 1.0; otherwise skip and say why.
- [ ] Step 4: same bars as the findings: "meaningful gain" = r_s < 0.9; also check the spec's
      headline directly — salted 2-bit MSE vs global 4-bit MSE per tensor.

### Task 4: Interpretation gate + report
**Files:** `docs/superpowers/plans/2026-10-12-salted-quantization-fp16-findings.md`
- [ ] Step 1: comparison table — η² and r_s, originals vs dequantized baseline, same cells.
- [ ] Step 2: verdict. If any tensor crosses r_s < 0.9 or salted-2-bit ≈ global-4-bit → theory
      revived on the originals; state the stakes explicitly: serving salted means re-quantizing
      from the originals (a new checkpoint + kernel), NOT patching the current W4A16.
- [ ] Step 3: if the null holds on the originals too → salted quantization is dead for this model
      on both baselines; close the spike. Further revival paths (2–4 in the findings doc) need new
      evidence, not more analysis of the same weights.
- [ ] Step 4: update the SDD ledger
      `.superpowers/sdd/2026-10-12-salted-quantization-spike/progress.md`.
      The no-go findings doc stays as the baseline record — do not edit its numbers.

## Review Focus (what this probe can get wrong if nobody checks)
1. **Wrong source repo/revision** — a different checkpoint than the one AutoRound quantized makes
   the comparison meaningless; the Task 0 Step 4 correlation check is the guard.
2. **BF16 vs FP16 mislabel** — the findings doc says "FP16 originals", the config says bfloat16;
   measure and label what actually ships.
3. **Weight tying** — config says `tie_word_embeddings: false`; verify on the original so the
   lm_head row is unambiguous.
4. **Loader regression** — the refactor must not touch accumulator/Lloyd-Max math; the
   `pre_fix.json` 1e-12 regression check is the guard.
5. **Null is guilty until proven innocent** — if η² ≈ 0 on the originals too, prove it is not
   instrumentation: realistic weight statistics on the loaded originals (std, dtype, percentiles),
   synthetic noise floor at real shapes, one headline cell re-derived through an independent path.
6. **Comparability** — same tensors/defs/N/k/fit rules; any deviation must be called out in the
   report or the side-by-side table is invalid.
7. **Disk/git hygiene** — 54 GB lands in `models-fp16/`, gitignored; never inside `models/`, never
   committed.
