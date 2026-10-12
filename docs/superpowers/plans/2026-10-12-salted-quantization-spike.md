# Salted Quantization Spike — Probe Plan

> **Project class: Spike.** Output is an answer (numbers + recommendation), not shipped code.
> Anything built here is throwaway unless the findings justify a follow-up (bounded/architectural) task.

**Question:** Does weight *position* in Qwen3.8-27B carry enough information that position-dependent
("salted") codebooks beat a single global codebook at the same bits/weight — enough to justify
a salted INT2/INT3 kernel effort?

**Tech stack:** Python 3.12 + torch 2.13.0+cu130 inside the repo `venv/` (WSL Ubuntu-24.04), safetensors, numpy.
No GPU needed for tasks 1–4 (CPU RAM-tile the weights); GPU only if we later measure perplexity.

**Model baseline (measured 2026-10-12):**
- `models/Qwen3.8-27B-W4A16-AutoRound`, 22 GB, 8 safetensors shards.
- Quantization: `compressed-tensors` pack-quantized. **All Linear = INT4 symmetric, group_size 128**
  (memoryless minmax observer); `lm_head`/`embed_tokens`/`mtp.*` = INT8 group-128; visual blocks 0–26
  and `linear_attn.in_proj_a/b` on odd-ish layers are FP16 (ignored list).
- Arch: `Qwen3_5ForConditionalGeneration` (hybrid linear-attention layers + MTP + vision tower).
- **FP16 originals are not on disk.** The best available "original" is the dequantized W4A16/INT8
  weight (what the serving kernel reconstructs). All analysis runs on that baseline — i.e. the spike
  answers "salted vs the weights HyperQwen actually serves."

**Spec:** conversation export `C:\Users\hubbe\AppData\Local\hermes\cache\scratch\conv\extracted.md`
(the salted-quantization proposal + Bonsai 2 comparison; proposal §5–6 defines per-salt k-means codebooks
and the N search).

## Global Constraints
- Every metric must cite its computation (file:line or printed table row) — no plausibility claims.
- Report numbers as measured; a hypothesis that fails is retracted plainly, not rescued.
- Scratch outputs → `analysis/salted/` (throwaway dir); the report goes to `docs/`.
- Do not touch serving configs, `models/`, or the Docker image in this spike.
- Salt candidate definitions (from spec §5): `linear % N`, `row % N`, `col % N`, `(row+col) % N`; N ∈ {4, 8, 16, 32, 64}.
- Bits: 2, 3, 4 per weight (per-salt codebooks of 4/8/16 entries).

---

### Task 1: Weight inventory

**Files:**
- Create: `analysis/salted/inventory.py`

**Produces:** a table (stdout + `analysis/salted/inventory.json`) of every weight tensor in the 8 shards:
name, shape, dtype, quant group (INT4-linear / INT8-head / FP16-ignored), element count.
Sanity: total element counts match config (`num_hidden_layers`, `hidden_size`, `intermediate_size`).

- [ ] Step 1: implement via `safetensors.safe_open` (framework="pt") shard-by-shard; print grouped table
- [ ] Step 2: run `wsl python analysis/salted/inventory.py`; confirm all 4 quant groups appear and FP16-ignored set matches the config ignore list count

### Task 2: Position-heterogeneity measurement (the core probe)

**Files:**
- Create: `analysis/salted/heterogeneity.py`

**Produces:** per sampled layer × salt definition × N, the **η² ratio** (between-salt-class variance /
total variance, computed on dequantized weights) plus per-salt-class mean/std table.
η² ≈ 0 means position carries no information (salted quantization useless); higher η² = more gain.

Sampled tensors: `layer 0 & 27 & last`: one Q proj, one gate proj, one down proj; plus `lm_head`.
Tile to CPU in ≤4 GB RAM chunks (only per-class sums needed — streaming mean/var, no full copy).

- [ ] Step 1: streaming accumulator that takes (weight_2d, salt_def, N) and yields η²
- [ ] Step 2: unit check on synthetic: heterogeneous-by-row synthetic matrix (row means differ) must give
      η² ≫ 0 for `row % N`; iid Gaussian must give η² ≈ 0 for all defs
- [ ] Step 3: run over sampled tensors × {linear,row,col,row+col} × N; write `heterogeneity.json`
- [ ] Step 4: verdict line: for each tensor, which salt def + N maximizes η², and the top value

### Task 3: Codebook gain — salted vs global quantization

**Files:**
- Create: `analysis/salted/codebook_gain.py`

**Produces:** for each sampled tensor, MSE of 2/3/4-bit quantization three ways:
(a) single Lloyd-Max codebook (global), (b) per-salt codebooks at the best salt def/N from Task 2,
(c) per-salt with per-salt scale (spec's 2-step). Report salted-vs-global MSE ratio
(the direct measure of "quality at same bits").

- [ ] Step 1: implement Lloyd-Max (k-means on weight values per class, ≤20 iters, float32)
- [ ] Step 2: unit test: salted must beat global on the heterogeneous synthetic, tie on iid Gaussian
- [ ] Step 3: run and print the ratio table (tensor × bits); note where ratio < 0.9 (salted meaningfully better)

### Task 4: Interpretation gate

**Files:** none (analysis only)

- [ ] Decide from Task 3 numbers: if salted INT2 ≈ global INT4 MSE (the spec's headline claim,
      "salted INT2 matches INT4"), the theory is alive → recommend kernel-spike follow-up.
      If gain is <10% MSE at all bit levels, the position-heterogeneity is already absorbed by
      group-128 scales → salted adds nothing on this baseline; say so.
- [ ] Write `docs/superpowers/plans/2026-10-12-salted-quantization-findings.md` with the numbers,
      which salt def won, and the go/no-go recommendation.

---

## Review Focus (what the probe can get wrong if nobody checks)
1. **Dequantization correctness** — compressed-tensors pack format (nibble order, symmetric zero-point)
   must be reproduced exactly or every downstream number is garbage; pin by cross-checking one tensor
   against vLLM's own dequant path.
2. **Axis mix-up** — row/col semantics differ per projection (gate: N=out, K=in; down: N=in, K=out).
   Wrong axis silently halves the effect; the synthetic test must use the real shapes.
3. **Streaming-accumulator bug** — η² computed per-chunk and averaged instead of over all elements
   underestimates heterogeneity; the accumulator must be exact (Welford over classes).
4. **Tail weights dominating MSE** — the few large weights dominate quantization MSE; report MSE and
   per-class max error, or a single outlier layer hides the real picture.
