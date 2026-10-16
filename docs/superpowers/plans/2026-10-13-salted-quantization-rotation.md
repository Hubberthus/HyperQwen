# Salted quantization — rotation follow-up (2026-10-13)

Status: CLOSED NO-GO (2026-10-16) — the Step-1 ppl gate failed: the MSE-only selective
rot-patch (all 38 rotated kept, floor accepted-lost) is +0.05910 ppl WORSE than the
AutoRound baseline at ~10⁴× duplicate noise, and costs ~2–2.5× per-window latency.
Findings: `2026-10-16-rotation-step1-findings.md`. Step 2 (full re-quant) NOT triggered.
Branch: `spike/salted-quantization`. Three subagents ran this round: A = prior-work survey
(`analysis/salted/prior_work_rotation.md`), B = fresh re-derivation + rotation simulation
(`analysis/salted/rotation/`), C = records cross-check.

## 1. What was verified this round

### 1.1 Salted nulls reproduce (fresh code, B)

`analysis/salted/rotation/redrive_nulls.py` re-derived all headline nulls from the BF16
originals with independent code:

| null | measured | previously reported | deviation |
|---|---|---|---|
| lm_head eta2, salt col%64 | 4.31822e-4 | 4.3182e-4 | 0.0004% |
| lm_head col%128 residue-class deviation | 209.3x floor | 217x | -3.6% |
| lm_head col%64 residue-class deviation | 175.0x floor | 164x | +6.7% |
| down_proj L63 col%128 deviation | 14.6x | 15-17x | -2.8% |

Verdict: the NO-GO stands; the nulls are a weight property, not a pipeline artifact.

### 1.2 Records cross-check (C)

- **lm_head is INT8** (symmetric group-128) in the serving checkpoint. On-disk proof:
  packed `[248320,1280]` I32 + scale `[248320,40]` F16 in
  `models/Qwen3.8-27B-W4A16-AutoRound/model-00007-of-00007.safetensors`; ~1.29 GB.
  The earlier spec-conversation table ("LM head INT4 ~640 MB") is WRONG.
- The 40 INT8-head tensor keys = lm_head, `embed_tokens`, `mtp.draft_lm_head` [40960x5120],
  `mtp.fc` [5120x10240], and MTP draft layer 0 (q/k/v/o + 7 norms). The checkpoint's
  `quantization_config.json` describes NONE of these (lm_head is in its ignore list, L238) —
  the config under-describes the on-disk reality; serving/analysis read the packed tensors.
- Minor inconsistencies (no bearing on conclusions): "155 M-element histogram" in the main
  findings doc (full lm_head is 1.27 B elements); "12 int4s per int32" in
  findings+ledger (impossible, max 8; on-disk in_proj_qkv is 8 int4/int32, 52.4 M elems/layer);
  duplicated "Task 3 REVIEW FIXES" section in the SDD ledger; ledger has no CLOSED line and
  never recorded the 2026-10-13 probe round.
- `models-fp16/Qwen3.8-27B/` confirmed on disk: 18 shards, 32 files, 55.6 GB.

### 1.3 Prior work (A — full detail in `analysis/salted/prior_work_rotation.md`)

- **QuIP#** (arXiv 2402.04396): randomized Hadamard transform + E8 VQ, weight-only, folded
  into weights at quant time → **no runtime activation-side transform**. 4-bit ≈ GPTQ/AWQ on
  LLaMA-2; rotation leverage is largest at 2-3 bit.
- **SpinQuant** (arXiv 2405.16406): LEARNED block-diagonal orthogonal rotations; headline
  wins are W4A4KV4, not weight-only. Compute-heavy.
- **Bonsai 2** (Prism ML, Sept 2026, HF `prism-ml/Ternary-Bonsai-2-27B-gguf`): REAL — ternary
  1.76 bpw / 5.9 GB re-quant of THIS exact model, Hadamard basis folded offline + runtime
  activation FWHT. 98.2% retention is VENDOR-RUN on their own 20-bench suite (not
  independent); llama.cpp-fork + MLX-fork only, no vLLM path (open ggml-org/llama.cpp#27779).
  It is ternary, not INT4 — do not extrapolate to our W4A16 target.
- **vLLM integration**: `compressed-tensors` (this box's own quant lib) ALREADY has
  transform support (Hadamard / random Hadamard / random matrix) + a `transform_config`
  field; Quark / llm-compressor 0.7.0 ship a "rotation"/"quarot" pre-quant path into vLLM.
  A weight-only fold needs NO per-forward patch; only the W4A4 activation-side FWHT is
  non-upstream.
- **Runtime cost**: block-diagonal Hadamard is O(d log d) ≈ 0.2% of the preceding matmul
  FLOPs; "no extra overhead" measured at decode on sm89 (PALU, arXiv 2404.00456). sm86
  unmeasured (extrapolation).
- **Contraindications**: OptRot (arXiv 2512.24124) — rotation can HURT once activations are
  also 4-bit; most headline rotation gains are activation-side (unreachable at W4A16);
  Qwen3.8-27B is 75% linear-attention (Gated DeltaNet) — rotation behavior on the
  recurrent-state path is less studied than standard MHA.

## 2. New experiment: rotation simulation (B)

Artifacts: `analysis/salted/rotation/{redrive_nulls.py, rotation_sim.py, nulls.json,
rotation_sim.json}`. Block-diagonal orthonormal Hadamard on the input dim (128-blocks;
256-blocks for lm_head), rotated W' = W @ H, symmetric group-128 uniform quant.

**Structure:** rotation compresses the outlier signature — lm_head group-128 absmax spread
8.81x → 2.84x, global max|w| 0.342 → 0.108; down_proj L63 spread 14.5x → 5.0x, max|w|
0.50 → 0.161.

**MSE (weight-level, vs BF16 original):**

lm_head [248320x5120, served at INT8]:

| config | MSE vs orig | MSE vs current ckpt | max abs err |
|---|---|---|---|
| current checkpoint INT8 | 8.98e-9 | — | 1.34e-3 |
| unrot INT4 | 2.51e-6 | 2.52e-6 | 4.27e-2 |
| rot128 INT4 | 2.21e-6 | 2.22e-6 | 1.48e-2 |
| rot128 INT8 | 8.50e-9 | 1.75e-8 | 9.18e-4 |

down_proj L63 [5120x17408, served at INT4/AutoRound]:

| config | MSE vs orig | MSE vs current ckpt | max abs err |
|---|---|---|---|
| current checkpoint INT4 | 1.72e-6 | — | 1.12e-1 |
| unrot INT4 (uniform) | 1.94e-6 | 2.47e-6 | 6.25e-2 |
| rot128 INT4 | 1.58e-6 | 3.27e-6 | 1.89e-2 |

**Verdicts:**

- **lm_head: DEAD.** INT4 is ~250x worse in weight MSE than the INT8 we already serve;
  rotation cuts max-abs error 3x but cannot close that gap. Rot-INT8 gains 5% MSE at zero
  VRAM saving — not worth it.
- **Body: PROMISING.** rot-INT4 beats the current AutoRound INT4 by **8.3%** in weight MSE
  on the sampled tensor, with a deterministic, calibration-free fold (no AutoRound
  optimization, no calibration data).

## 3. What is exploitable (and what is not)

- Exploitable: body outlier structure (13-20x over floor, measured) addressed by the
  Hadamard fold; one-tensor evidence says rot-INT4 ≥ AutoRound-INT4 in weight MSE.
  Missing: output-tied validation (perplexity) — exactly revival path #3 the original
  findings doc flagged as absent.
- Not exploitable at W4A16: lm_head VRAM (dead, see above); most of rotation's headline
  gains (activation-side, W4A4 territory); Bonsai 2's ternary footprint (separate goal,
  not quality at the current footprint).

## 4. Proposal — next phase

- **Step 0 (docs, ~15 min):** ledger — record this round + add the spike's CLOSED line;
  fix the minor inconsistencies of §1.2; add the rotation revival path; mark lm_head-INT4
  dead with the 250x MSE evidence.
- **Step 1 (decisive, GPU, ~half day):** rotation-INT4 patch of 4-8 layers on a COPY of the
  current checkpoint (layer set from a cheap all-1200-tensor absmax-spread survey — patch
  where structure is strongest; must include at least one linear-attention layer and one
  full-attention layer; include L63). Metric: perplexity on a standard corpus vs the current
  checkpoint. GATE: no ppl regression on patched layers (improvement = bonus). Use
  compressed-tensors' built-in transform_config/Hadamard path; first verify parity with B's
  hand-rolled fold on the pinned tensor.
- **Step 2 (gated on Step 1 pass):** full 3.23 B-element rotation re-quant →
  `W4A16-rot` checkpoint → A/B against current via the bench harness
  (ab-measurement-campaign discipline).
- **Step 3 (optional):** MTP drafter INT8 tensors (`draft_lm_head`, `mtp.fc`, draft layer 0)
  → rotation; metric = speculative acceptance rate.
- **Non-goals:** lm_head INT4, salted codebooks, W4A4 activation quant (OptRot interaction
  risk), any power-limit change.

## 5. Open items

- `inventory.json` INT8-head element total (819,144,214) cannot include lm_head
  (1,271,398,400) — an inventory.py counting quirk; per-tensor on-disk verification is
  solid, the group total is not trusted.
- sm86 runtime for an activation-side transform: unmeasured; only relevant if W4A4 is ever
  pursued (weight-only fold has no runtime transform).
- Gated-DeltaNet linear-attention path under rotation: unmeasured — covered by the
  Step-1 layer-set requirement above.
