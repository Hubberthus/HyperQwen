# Rotation Re-Quant — Step 1 Findings: Layer-Patch Perplexity Gate

**Date:** 2026-10-16 · **Branch:** `spike/salted-quantization` · **Commits:** `f868035` (Task 1b) → `f62321d` (step-2 artifacts)
Executes Task 2 + Task 3 of `2026-10-13-rotation-step1-plan.md`.

## Verdict: NO-GO — rotation loses at output level

The ppl gate **FAILS**: the MSE-only selective rot-patch is measurably worse than the
AutoRound baseline it replaces, by an amount ~10,000× the duplicate-run noise. The 179×
weight-MSE win of the Hadamard fold does not survive contact with an output-tied metric —
exactly the check the original salted findings doc flagged as missing (its revival path #3).

## Measured evidence

### PPL gate (`analysis/rotation/ppl.json`, commit `f62321d`)

Metric: perplexity = exp(−mean NLL) over ALL predicted prompt tokens; wikitext-2-raw-v1
test (raw, joined), 2048-token windows at 256 stride, 1153 windows, 2,360,191 tokens
predicted per run, identical script/corpus/token budget both arms, via prompt_logprobs
through the live vLLM server (SPEC=mtp, CTX=fast, PREFIX_CACHE off). ×2 runs per arm.

| tag | ppl | mean NLL | wall s | pid |
|---|---|---|---|---|
| base1 | 6.8183772 | 1.91962150 | 2491.8 | 6221 |
| base2 | 6.8183723 | 1.91962077 | 5606.3 | 15284 |
| patch1 | 6.8774678 | 1.92825053 | 5023.8 | 90685 |
| patch2 | 6.8774737 | 1.92825139 | 6270.2 | 123404 |

- Duplicate spread: baseline 4.95e-6, patch 5.89e-6 — measurement noise ≈ 6e-6 ppl.
- **patch − baseline = +0.05910 ppl** (patch worse). ~10⁴× the noise band.
- Per-boot hygiene held: new pid verified per boot, KV pool line confirmed per boot
  (`logs/boot_*.log`), patch post-load smoke passed before the full runs
  (`smoke_mse.json` — 26 rotated parents wrapped in `CompressedTensorsLinearTransformMethod`,
  plain `in_proj_qkvz` unwrapped `CompressedTensorsLinearMethod`).

### What the patch was (`analysis/rotation/selective_selection_mse.json`)

MSE-only selection (the relaxed gate — this run deliberately tested whether the user's
tail-fidelity floor was stricter than output quality demands): keep a tensor rotated iff
the better rotated variant (naive-RTN vs rot-AR) beats current corrected-MSE. **All 38
rotated tensors won MSE-only** (naive fold wins every tensor; macro 3.81e-4 → 2.12e-6,
179×), so the patch = `rot-patch-selective-mse`: 38 naive-fold splices (hash-verified
against sources), 12 plain `in_proj` at baseline weights (option B — the AR dir's
option-A in_proj were worse and were NOT reused), `transform_config` with exactly 38
hadamard/128/input targets. Fused `qkv_proj`/`gate_up_proj` parents kept whole (every
member won; partial targeting is invalid — `_validate_tfm_schemes`).

### Floor on the kept set (`analysis/rotation/tail_fidelity_selective_mse.json`)

Accepted by design this round, recorded for the mechanism: 4/38 tensors satisfy the full
low-band floor (`27.k`, `63.k`, `63.o`, `63.v`); 34/38 regress band retention. The
regression is rotation+group-absmax scale redistribution, not rounding (Task 1b falsified
the "optimized rounder recovers the floor" hypothesis: AR made corrected-MSE 6.4% worse
than naive RTN).

### Latency (runtime Hadamard cost)

Patch walls 5024/6270 s vs clean baseline 2492 s → **~2.0–2.5× per-window latency** for
the runtime input transform (base2's 5606 s wall was contended and excluded from the
ratio). Even if quality had passed, this is a real serving cost, not a free fold.

## Decision

- Keep `models/Qwen3.8-27B-W4A16-AutoRound` served. The rot-patch checkpoints
  (`-rot-patch`, unservable; `-rot-patch-AR`, `-rot-patch-selective`, `-rot-patch-selective-mse`)
  are measurement artifacts — dominated on BOTH quality and latency.
- Rotation direction (block-128 Hadamard fold + group-absmax INT4, W4A16, this checkpoint):
  **CLOSED at output level.** The failure mechanism is measured, not assumed: the fold
  trades low-band L1 mass for heavy-tail L2 MSE; AutoRound's low-band plateau is what
  output quality rests on, and no rounding scheme tested recovers it after a fold.

## What would revive it (in order of value)

1. **A fold objective that preserves low-band mass** — quantize the rotated weights with
   an error signal weighted toward the low-|w| band (the plain MSE objective is
   tail-dominated and blind to exactly what the floor measures). Untested.
2. **Fewer, highest-structure layers only** — the strict-gate result (1/38) and the
   per-row spread survey (`survey.json`) suggest structure is concentrated; a patch of
   only o_proj/out_proj-class tensors might pass ppl, but the upside is small by
   construction.
3. **Activation-side rotation (W4A4)** — out of scope here (OptRot interaction risk, and
   W4A16 is the settled footprint).

## Non-goals / status

- lm_head INT4 stays dead (~250× MSE, rotation doc §2). Salted codebooks stay closed
  (findings doc 2026-10-12 + fp16 falsification). Power cap untouched.
- Step 2 (full 3.23 B-element re-quant) is **not** triggered — its gate failed.

## Artifacts

- `analysis/rotation/ppl.json` + `ppl_runs/{base1,base2,patch1,patch2}.json` + `logs/` (commit `f62321d`)
- `analysis/rotation/selective_selection{,_mse}.json`, `tail_fidelity_selective{,_mse}.json`, `smoke_mse.json` (same commit)
- Checkpoints under `models/` (gitignored, on disk): the four rot-patch variants + baseline
- Parent verification: splice provenance hash-check (`_verify_mseonly_parent.py`); baseline
  doc mislabels recorded in-session (8 shards / 23,132,043,335 B; "+5,488 B" is vs the naive patch).
