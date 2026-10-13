# Salted Quantization Spike — Findings (Task 4)

**Date:** 2026-10-12 · **Branch:** `spike/salted-quantization` (base `main` @ `95ee3b6`)
**Question:** Does weight *position* in Qwen3.8-27B carry enough information that position-dependent
("salted") codebooks beat a single global codebook at the same bits/weight — enough to justify a
salted INT2/INT3 kernel effort?

## Verdict: NO-GO on this checkpoint's baseline

Salted quantization shows **no measurable gain** over a single global codebook at equal bits/weight
on the dequantized weights of `models/Qwen3.8-27B-W4A16-AutoRound`. The best salted result anywhere
is a 8.9 % MSE reduction (lm_head, 4-bit) — far below the spec's headline claim ("salted INT2
matches INT4"). The reason is measured, not assumed: position-dependent variance structure in these
weights is essentially zero (η² ≤ 4.3e-4), because the AutoRound checkpoint's **group-128 scales
already absorb the position information** the salt would exploit.

## Baseline caveat (read before quoting these numbers)

Tasks 1–4 ran on the **dequantized W4A16/INT8 weights** — exactly what the serving kernel
reconstructs. Conclusions hold for "salted vs the weights this repo actually serves". The pre-
quantization BF16 originals are now on disk at `models-fp16/Qwen3.8-27B` (used by the 2026-10-13
probe round below for layout plots); a full Tasks 2–3 re-run on the originals remains open (see
"What would revive the theory", item 1).

## Measured evidence

### Task 1 — inventory (commits `fa78c42`→`6b1f399`, `analysis/salted/inventory.py`)

| group | tensors | elements |
|---|---|---|
| INT4-linear | 1200 | 3,230,925,600 |
| INT8-head | 40 | 819,144,214 |
| FP16-ignored | 206 | 479,490,048 |
| FP16-other | 576 | 7,478,512 |

Zero stray packed rows outside INT4/INT8 groups. Checkpoint geometry: marlin-style repack
(`in_proj_qkv.weight_packed` [10240, 640] I32 = 12 int4/int32, not a plain /4 layout).

### Task 2 — position heterogeneity η² (commits `32b4e06`, `ac1cbf6`, `analysis/salted/heterogeneity.py`)

η² = between-salt-class variance / total variance over 9 tensors × {linear, row, col, row+col} ×
N ∈ {4,8,16,32,64}:

- **All η² ≈ 0.** Max = lm_head `linear%64` = **4.318e-4**. `row%N` is flat 0.0000 on every tensor.
- Measured iid noise floor at these shapes: η² ≈ 5.5–9.2e-7. Most real values sit **at** the floor.
  The only above-floor signal (gate_proj `col%64` ≈ 4.1e-5) is 0.004 % of total variance.
- Dequant pin: manual tiled unpack+dequant vs the venv's own `compressed_tensors`
  (`unpack_from_int32` + `lifecycle.forward.dequantize`) = **0.000e+00** over 89,128,960 elems;
  independently reproduced by the reviewer, who also re-derived the max η² cell to 7.6e-19.
- Reviewer's independent checks ruled out a manufactured null: dequantized values show real weight
  statistics (std 0.0102; exactly {−8…7}×scale per group), and 50.7 % of gate_proj scales are
  negative (AutoRound property — handled by the signed path, not "fixed").

### Task 3 — codebook gain, salted vs global Lloyd-Max (commits `693a3da`, `01a294d`, `analysis/salted/codebook_gain.py`)

Lloyd-Max (equal-percentile init, ≤20 iters, deterministic), identical k = 4/8/16 entries per arm,
bits/weight = log2(k) everywhere (salt class index is position-derived, stored nowhere — parity
reasoning in the docstring). MSE computed exactly by streaming over full tensors.

Salted/global MSE ratio (r_s; r_ss = spec's 2-step: per-class absmax scale + one shared k-entry
codebook fit in normalized space — full entry-count parity):

| tensor (best salt from Task 2) | r_s 2/3/4-bit | r_ss 2/3/4-bit |
|---|---|---|
| gate L0 (linear%64) | 0.9995 / 1.0010 / 1.0042 | 1.2986 / 1.2926 / 1.1897 |
| down L0 (row+col%64) | 1.0002 / 1.0009 / 1.0013 | 2.4469 / 1.6474 / 1.2379 |
| q L27 (linear%64) | 0.9999 / 1.0020 / 1.0186 | 1.1198 / 1.1054 / 1.0399 |
| gate L27 (linear%64) | 0.9992 / 1.0004 / 1.0055 | 1.2127 / 1.2298 / 1.1844 |
| down L27 (linear%64) | 1.0003 / 1.0008 / 1.0049 | 1.6835 / 1.6759 / 1.5476 |
| q L63 (linear%64) | 0.9996 / 0.9993 / 1.0145 | 1.1665 / 1.1051 / 1.0302 |
| gate L63 (linear%64) | 0.9985 / 1.0004 / 1.0287 | 1.3664 / 1.3185 / 1.1689 |
| down L63 (row%64, fit r%17) | 1.0012 / 1.0137 / 1.0150 | 1.4025 / 1.3914 / 1.1295 |
| lm_head (linear%64) | **0.9921 / 0.9596 / 0.9106** | 1.2043 / 1.1705 / 1.0272 |

- No tensor crosses the 0.9 "meaningful gain" threshold at any bit level. The gain tracks η²:
  lm_head (largest η²) gets the only real benefit; everything else is within fit noise of 1.0.
- The honest 2-step arm is **worse** (1.03–2.45×): per-class absmax scaling + a shared book loses
  per-class shape information. (The previously reported r_ss ≡ r_s was a mathematical artifact:
  Lloyd-Max is exactly scale-equivariant, so per-class books on w/absmax scaled back are provably
  identical to raw per-class books. Documented in the code; the variant now run is distinct.)
- Down L63's pre-fix 3-bit 1.1330 was fit-subsample noise, re-derived at fit stride r%17 → 1.0137
  (probe evidence: r%65 1.1330 / r%33 1.0623 / all-rows 0.9973).
- Regression check: all unchanged cells reproduce the pre-fix JSON exactly (1e-12 tolerance).
- Method validity: salted beats global 0.49× on row-structured synthetic, 0.73× on row+col, ties
  (1.0006) on iid Gaussian; global 4-level Lloyd-Max MSE matches the theoretical optimum
  0.1175·σ² (0.11751 measured).

### One nuance that is NOT an MSE win (report it accurately)

Salted's per-class **max error** is lower than global's in 14/27 cells — most notably lm_head at
all bit levels, up to 1.18× lower at 4-bit (0.258 vs 0.304). A worst-case-error argument for
salted quantization exists in principle, but it is unproven at the output level and small in
magnitude; it does not rescue the MSE verdict.

## Why the null is expected

The salt hypothesis grants position as free side information. But the checkpoint was produced by
AutoRound with symmetric per-group (128) scales: within each group the quantized values are near
uniform over the 16/256 code levels, so any positional structure left after scale absorption is
~0.04 % of variance at most. There is nothing for a salted codebook to compress.

## Probe round — 2026-10-13 (`analysis/salted/weight_layout.py` + follow-ups)

A layout/visualization pass over the same 9 tensors (original BF16 `models-fp16/Qwen3.8-27B` vs
dequantized W4A16) produced 8 plots + a JSON (`weight_layout.json`, wall 323 s, 9 spawn workers),
then three numeric follow-ups — `floor_layout.py`, `check_lmhist.py`, and scale-stats probes of the
JSON — archived as `analysis/salted/noise_floor.txt`. Two early *visual* readings of the plots were
wrong and are corrected here; the numeric probes replace them.

### Noise floor of residue-class structure — quantization adds nothing

`floor_layout.py` measures mean-|w|² deviation by residue class (col%128 / col%64) against an iid
noise floor. Ratio over floor, original vs dequant (`analysis/salted/noise_floor.txt`):

| tensor | r128 orig | r128 deq | r64 orig | r64 deq |
|---|---|---|---|---|
| L0.gate | 4.3 | 4.2 | 4.1 | 4.1 |
| L0.down | 1.3 | 1.3 | 1.3 | 1.2 |
| L27.q | 4.8 | 4.5 | 4.5 | 4.3 |
| L27.gate | 7.3 | 7.4 | 6.0 | 6.0 |
| L27.down | 5.4 | 5.3 | 4.4 | 4.3 |
| L63.q | 20.0 | 19.3 | 13.7 | 13.2 |
| L63.gate | 9.1 | 8.6 | 7.0 | 6.7 |
| L63.down | 15.1 | 15.4 | 17.3 | 17.5 |
| lm_head | 217.2 | 217.2 | 164.0 | 164.0 |

Original and dequant land on the **same** ratio per tensor: quantization contributes zero
positional structure. The 1.3–20× over-floor values are genuine weight structure (deeper layers
more structured) — already shown in Tasks 2/3 to be absorbed by group scales + row conditioning.
Caveats: the iid floor does not condition on group scale, so these ratios overstate residual class
structure; orig≈dequant holds because quantization noise is small relative to the real structure
(the equality is the informative fact). lm_head's 217× is its per-group scale spread (below), not
a residue signal.

### lm_head — two earlier visual readings, corrected

An earlier vision analysis of `01_overall_value_dist.png` claimed the lm_head dequant curve was
"drastically broader and flatter" with mass "smeared across the range" (i.e. lossy tails). Measured
on the full tensor (`check_lmhist.py`, 155 M-element histogram):

- Tail mass **identical** original vs dequant: |w|>0.1 = 1.60e-05 both, |w|>0.15 = 1.84e-06 both;
  10 elements beyond 0.2 in each. No tail smearing.
- Dequant peak density 95.2 at 0.0 vs original 34.8 at +0.0043 — the zero-code spike, as expected
  for 8-bit symmetric quant (code spacing ≈ 3 histogram bins at scale p50 3.057e-4). The "broader"
  look was the 256-level comb rendered dense at plot resolution.
- corr(orig, dequant) = **1.0000**; code entropy 7.48 of 8.0 max bits. lm_head at INT8 is
  essentially lossless at distribution level. The "particularly lossy" call is **retracted**.

### Group-scale coupling — none on the 4-bit tensors

`weight_layout.json` → `scale_stats`, corr(|scale|, group mean-|w|²) per tensor: L0.gate −0.0013,
L0.down −0.0063, L27.q −0.0006, L27.gate −0.0022, L27.down −0.0008, L63.q −0.0143, L63.gate
−0.0020, L63.down −0.0063, lm_head +0.6099. Scales do not couple to group energy anywhere on the
INT4 side (lm_head's 0.61 is its 9× scale range: p50 3.057e-4, max 2.691e-3, all-positive scales).
Code entropy 3.54–3.64 bits of 4.0 max on every INT4 tensor — near-uniform code usage, no code-map
regularity to exploit. (The `06_group_structure.png` panels backing these numbers could not be
vision-read twice — provider 503; the JSON numbers stand in for them.)

### Probe-round conclusion

No positional pattern survives measurement: the residue-class deviations are real weight structure
already handled by group scales, quantization adds none, and both anomalies that motivated the pass
(lm_head "smearing", group-scale coupling) were measurement artifacts or expected quantizer
behavior. **NO-GO stands unchanged.**

## What would revive the theory (in order of value)

1. **Test on FP16 originals** (the pre-quantization weights). Heterogeneity may exist before
   AutoRound's group scales absorb it. Requires the unquantized checkpoint (~54 GB download).
2. **Coarse/no group scales**: salted may help where scale granularity is per-channel or per-tensor
   (INT2/3 without group-128), i.e. exactly the regime a salted kernel would target. Re-run
   Task 3's arms on group-1 (per-channel) or group-64 variants.
3. **Output-tied error metrics**: weight MSE is a proxy; measure per-channel output error and
   perplexity deltas for salted vs global at 2-bit on a small slice (needs GPU, unlike Tasks 1–4).
4. **Structured-outlier pairing**: combine heavy-hitter outlier handling (Bonsai-2-style) with a
   salted residual — the spec's comparison target — rather than plain salted-everything.

## Artifacts & reproduction

- `analysis/salted/inventory.py|json` — Task 1. Run: `venv/bin/python analysis/salted/inventory.py`
- `analysis/salted/heterogeneity.py|json` — Task 2 (pin + synthetic gates abort on failure).
- `analysis/salted/codebook_gain.py|json|log` (+ `codebook_gain.pre_fix.json` regression baseline)
  — Task 3, parallel sweep (9 workers, spawn context; regression check aborts on mismatch).
- `analysis/salted/weight_layout.py|json` + `analysis/salted/plots/*.png` — probe-round layout
  pass (wall 323 s; dequant pin + dequant-grid + BF16-dtype checks abort on failure).
- `analysis/salted/noise_floor.txt` — probe-round residue-class iid-floor table (floor_layout.py).
- `check_lmhist.py` / scale-stats probes — probe-round lm_head histogram and scale-coupling checks
  (scratch scripts; the numbers in the probe-round section are their full output, not excerpts).
- SDD ledger: `.superpowers/sdd/2026-10-12-salted-quantization-spike/progress.md`
- Spec (the theory under test): extracted conversation at
  `C:\Users\hubbe\AppData\Local\hermes\cache\scratch\conv\extracted.md` §5–6.
- Every number above cites its computation (scripts) and was independently reviewed (Task 2 and
  Task 3 reviews reproduced key cells exactly); throwaway analysis code, not serving code.
