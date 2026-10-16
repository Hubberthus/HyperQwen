# Compression revival round — probe ledger (2026-10-16)

**Branch:** `spike/salted-quantization` · Status: probe #1 done (NO-GO), probe #2 in flight.

## Context: where the revival started (2026-10-16)

Salted quantization (Task 4, 2026-10-12) and Hadamard rotation (Step 1, 2026-10-16)
are both CLOSED NO-GO at output level. The user reopened the original compression
idea with three angles:

1. **Find what must be protected** — the output-tied structure that MSE is blind to
   (rotation proved: low-band mass is what quality rests on; L2-MSE lost 6.818→6.877 ppl).
2. **Previous-value prediction** (the never-tested second half of the original idea) —
   delta-coding codes/scales from a neighbor value.
3. **Codec transplant** — borrow audio/video rate-distortion machinery (perceptual
   weighting matrices, predictors) for weight "noise".

Ordering decision: probe #2 (previous-value, cheapest, answers "is there sequence
structure?") first; the sensitivity map (#1) is the prerequisite that makes #3
(codec) meaningful rather than a rotation repeat.

## Probe 1 — conditional-entropy scan on down_proj L63 (DONE 2026-10-16, NO-GO)

**Question:** is there exploitable *sequence* structure in quantized codes or group
scales? (Salted proved position-in-distribution null; sequence never measured.)

**Artifact:** `analysis/salted/cond_entropy_probe.py` + `.json` (untracked).
Run: `venv/bin/python analysis/salted/cond_entropy_probe.py` (~10 min, CPU).

**Tensor:** `model.language_model.layers.63.mlp.down_proj` [5120, 17408], INT4
group-128 → 136 scales/row. Note the checkpoint key prefix is `model.language_model.`
(vLLM-fork remap) — a first run failed on `model.layers.` before this fix.
**Pin:** manual unpack vs `compressed_tensors.unpack_from_int32` — max|diff| = 0 over
89,128,960 codes.

| Predictor | H(bits) | Gain vs marginal 3.5673 |
|---|---|---|
| A. prev column (horizontal delta) | 3.5667 | +0.0007 (noise) |
| B. prev row (vertical delta) | 3.5657 | +0.0017 (noise) |
| C. Paeth 2-D (left/up/left-up min) | 3.4879 | **+0.0794** (real, sub-threshold) |
| D. scale \| prev row (256 equal-freq bins) | 7.9253 | ≈0.0006 bits/weight |
| E. scale \| prev group | 7.8777 | ≈0.0010 bits/weight |

**Verdict:** NO-GO below the 0.1 bits/weight threshold. Simple spatial predictors are
noise; Paeth's 0.079 is a genuine but weak 2-D residue; scale prediction negligible.
A learned NN predictor could beat 0.08 but the ceiling is tiny at 3.57/4 bits.

**Strategic reframe:** weight *values* are the wrong target — the exploitable dimension
is *sensitivity* (which weights move the output). Codec transplant becomes viable only
as a perceptual-weighting (rate-distortion) encoder, which needs the sensitivity map.

## Probe 2 — proxy sensitivity map (IN FLIGHT, delegated)

**Question (per tensor):** which weight elements/groups actually move the output —
i.e. what must a compression encoder protect vs. throw away?

**Design (user-approved):** proxy first (activation-magnitude exposure, GPTQ/AWQ
insight), group granularity, 3–4 diverse tensors: gate_proj/down_proj at L0 and L63 +
lm_head subsample. GPU end-to-end ppl validation only if the proxy shows concentration.

**Success criteria:** a per-group sensitivity score per tensor; concentration readout
(e.g. what fraction of groups carry X% of total sensitivity). Concentrated → design a
perceptually-weighted encoder; uniform → codec idea joins the closed pile cheaply.

## Probe 3 — throwaway-weight stage 1: candidate ranking (DONE 2026-10-16)

**Question:** does a bottom-K subset of groups exist that carries <5% of total
sensitivity (headroom for demotion)? Stage 2 (demotion simulation) and the GPU
ppl gate follow only if the curve is flat enough.

**Artifacts:** `analysis/salted/throwaway_ranking.py` + `.json` (untracked).
Runtime ~1.2 s/tensor. Gates: pin = 0; `verify_vs_probe` reproduces
`sensitivity_probe.json` exactly (rel_err 0.0) before ranking. Note: AutoRound
scales are **signed** — the scale-ranking arm ranks on `abs(scales)` (raw
`scales.sum()` is negative; first run asserted on this).

**Bottom-K% of groups by sensitivity → fraction of TOTAL sensitivity:**

| K | L0 gate | L0 down | L63 gate | L63 down |
|---|---|---|---|---|
| 5% | 3.39% | 3.48% | 2.37% | 3.12% |
| 10% | 7.17% | 7.32% | 5.30% | 6.78% |
| 20% | 15.26% | 15.48% | 12.09% | 14.75% |
| 30% | 23.87% | 24.11% | 19.82% | 23.34% |

**By |scale| (cheaper ranking):** materially worse but close — e.g. L63 gate
K=5: 3.21% vs 2.37% (log-log Pearson 0.60–0.77). Magnitude ranking is a decent
proxy, not an equivalent.

**Stage 2 — demotion simulation (DONE 2026-10-16): NO-GO.**
`analysis/salted/throwaway_demote.py` + `.json` (untracked; ~3 s/tensor, in-memory
only, RTN at the group's existing 4-bit scale, 2-bit signed codes). Gates: pin = 0
(89M codes/tensor), reconstruction diff = 0. Per K (5/10/20%), the sensitivity-weighted
distortion fraction is essentially **equal** to the raw-L2 fraction on all four tensors
(e.g. L0 gate K=10: 0.0302 raw vs 0.0302 sens; L63 gate K=10: 0.0219 vs 0.0218).
The sensitivity weighting provides no demotion headroom — distortion grows linearly
with K because the map is diffuse. Verdict: NOT favorable; the end-to-end GPU ppl
gate is not earned. **Throwaway-weight identification joins the closed pile.**

**Round-3 standing:** of the user's three revival ideas, #2 (previous-value) and #1
(throwaway/sensitivity) are now measured NO-GO on this checkpoint. Remaining: #3
(codec transplant as perceptual weighting — weakened by the diffuse finding) and the
untested corner the round hasn't touched: cross-layer structure (the sequence probes
were within-tensor only).

## Probe 4 — cross-layer structure (DONE 2026-10-16)

**Question:** do layer-N scales/codes predict layer N+1, and does a shared
cross-layer codebook save bits? First cross-tensor probe of the round.

**Artifacts:** `analysis/salted/crosslayer_probe.py` + `.json` (untracked; 37 s CPU,
all 63 adjacent layer pairs on scales, codes on pairs {0-1, 31-32, 62-63}).
Gate: pin = 0 (89M codes, L31 gate_proj).

**Results:**

| Arm | gate_proj | down_proj |
|---|---|---|
| A: H(scale\|prev layer) vs pooled marginal 8.0 | 7.833 → **+0.165** | 7.798 → **+0.199** |
| B: H(code\|prev layer code) vs ~3.6 marginal | +0.020 / −0.003 / −0.015 | −0.001 / −0.007 / −0.065 |
| C: pooled vs mean per-layer scale entropy | 7.9973 vs 7.9963 (nil) | 7.9962 vs 7.9960 (nil) |

**Verdict: MIXED — first positive signal of the round, but tiny in absolute terms.**
Scales DO predict across layers (+0.16/+0.20 bits/scale, above the 0.1 threshold) —
the only temporal dependency found in this whole round. Codes do NOT (negative gaps =
noise). Sharing a codebook across depth saves nothing (C nil). Practical value:
scale delta coding saves ~0.17 bits/scale over ~2.4M scales/layer × 63 layers ≈
**~16 MB total** (0.07% of the checkpoint) — real but marginal.

**Next (user-approved):** Approach A spike — cross-layer scale DPCM (delta coding),
applied as the codec's inter-frame prediction analogue. If it lands, it validates the
codec direction on a small, cheap surface.



