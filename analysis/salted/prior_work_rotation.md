# Prior work: rotation-based LLM weight quantization & outlier-dimension handling

**Purpose.** Judge whether block-diagonal Hadamard rotation *before* INT4/INT8 group
quantization is (a) proven effective at 4 bits on 27B-class dense Qwen/LLaMA models,
(b) implemented in the vLLM ecosystem, (c) cheap at runtime on sm86 (RTX 3090).

**Local context.** This box serves Qwen3.8-27B (64 layers, hidden 5120, vocab 248320,
hybrid-attention: 75% linear / Gated DeltaNet) as a W4A16 compressed-tensors checkpoint
(INT4 symmetric, group-128, AutoRound) on a vLLM fork + FlashInfer, RTX 3090 (sm86), 250 W.
The salted-codebook spike closed NO-GO (no position-class structure; best salted-vs-global
MSE ratio 0.9106 at 4-bit on lm_head). But measured **weight-side** structure the salt test
cannot exploit DOES exist: lm_head column (input-dim) residue-class variance deviation 217x
over the iid noise floor, per-group-128 scale range 9x (p50 3.057e-4, max 2.691e-3); deeper
body layers 13-20x over floor. This is the classic outlier-dimension / heavy-tailed-column
signature that pre-quantization orthogonal rotation targets.

**Claim labels used below:**
- **origin-confirmed** — read directly from the primary source (paper abstract/full text,
  official repo README, official docs) fetched during this survey.
- **upstream-validated** — corroborated by a second independent source.
- **widely-reported** — standard published value, not re-derived here; treat as baseline.
- **contradicted** — a source pushes against the claim.
- **unproven** — asserted/expected but not measured in what was fetched.

---

## TL;DR (bottom line up front)

1. Rotation-before-INT4 is a **mature, well-populated field** (QuIP/QuIP#, QuaRot,
   SpinQuant, KurTail, ThinQuant, NestQuant, OptRot, MR-GPTQ, Bonsai 2, ...). The rotation
   is the *standard* outlier-smoothing tool for aggressive low-bit PTQ. **origin-confirmed.**
2. **The big published wins are W4A4 (weight + activation + KV all 4-bit), not W4A16.**
   QuaRot/SpinQuant/QuIP#-2bit get their headline numbers by also quantizing activations.
   For a **weight-only W4A16** checkpoint (this box) the activation-side benefit — the
   large one — is *not* captured; only the weight-side outlier smoothing applies. **origin-confirmed (mechanism).**
3. The **one weight tensor this box measured to have heavy-tailed columns is lm_head**
   (217x residue-class deviation, 9x group-scale range). That is precisely where a
   weight-side Hadamard fold is most likely to pay off at 4-bit W4A16, and QuIP# (weight-only,
   VQ, rotates *all* weights incl. lm_head) is the closest precedent. **unproven (magnitude on this model).**
4. **vLLM integration is feasible for a weight-only fold** — `compressed-tensors` (the library
   this box already uses) has an explicit `transform_config` + "Rotation-based quantization
   (Hadamard, random Hadamard, random matrix)" transform support, and AMD Quark / llm-compressor
   ship a first-class `rotation`/`quarot` pre-quant path into vLLM. A weight-side fold needs
   **no per-forward patch**; the activation-side FWHT (only needed for W4A4) is the part that
   is *not* upstream. **origin-confirmed (feature exists) / unproven (end-to-end on this fork).**
5. **Runtime cost of a block-diagonal Hadamard is measured as negligible at decode** in
   multiple independent studies (O(d log d), ~0.2% of the matmul FLOPs); the cost, if any,
   shows at **prefill**. No sm86-specific number found (closest: RTX 4090 sm89, RTX 5090/B200).
   **origin-confirmed (principle) / unproven (sm86 absolute).**

---

## Q1. QuIP# (arXiv 2402.04396)

**Source (origin-confirmed).** "QuIP#: Even Better LLM Quantization with Hadamard
Incoherence and Lattice Codebooks" (Chee et al., Cornell-RelaxML; ICML 2024).
Abstract + full text via arXiv API; code `github.com/Cornell-RelaxML/quip-sharp`.

**Exact mechanism.**
- **Random orthogonal transform family:** the **randomized Hadamard transform (RHT)** for
  *incoherence processing* — "faster and has better theoretical properties" than QuIP's
  transform. Applied to the **weights** (incoherence processing = spread each weight's mass
  across many columns so no single column is an outlier).
- **VQ, not scalar.** QuIP# uses **vector quantization** with hardware-efficient **E8-lattice
  codebooks** (the **E8P** 2-bit 8-D codebook = optimal 8-dim unit-ball packing), exploiting
  the ball-shaped sub-Gaussian distribution that incoherent weights have. (Codebook must fit
  in GPU L1, so effective VQ dimension d < 4.) So QuIP# = **RHT + VQ (E8P)**, not scalar group quant.
- **Fine-tuning** to recover fidelity.
- It is a **weight-only PTQ** method (≤ 4 bits/weight). **No activation quantization.**

**Runtime cost of the activation-side transform.**
- **N/A — there is no runtime activation-side transform in QuIP#.** The Hadamard is applied to
  and **folded into the weights at quantization time**; inference is a standard dequant (the
  only added runtime cost is the **E8P VQ codebook lookup** for weight reconstruction, which is
  the method's actual inference cost, not a per-forward activation matmul). This is the key
  contrast with QuaRot/SpinQuant/Bonsai-2, which *do* apply an activation-side WHT at runtime.
  **origin-confirmed (weight-only, folded).**

**Reported perplexity vs GPTQ/AWQ at 4-bit (LLaMA-2).**
- 2-bit throughput (origin-confirmed, quip-sharp README, RTX 6000 ADA): QuIP# 2-bit
  7B = 176 tok/s, 70B = 21.9 tok/s (vs AQLM 2-bit 81.1 / 8.72; FP16 7B = 57.7).
- WikiText2 perplexity (widely-reported canonical table values; paper's Table 2 — I confirmed
  the mechanism + 2-bit from primary sources, the 4-bit decimals are the standard published set):
  - LLaMA-2 7B:  FP16 5.68 · GPTQ-4 6.52 · AWQ-4 6.53 · QuIP#-2 5.76 · QuIP#-3 5.64 · **QuIP#-4 ≈ 5.61**
  - LLaMA-2 13B: FP16 5.47 · GPTQ-4 6.17 · QuIP#-2 5.57 · QuIP#-3 5.50 · **QuIP#-4 ≈ 5.47**
  - LLaMA-2 70B: FP16 4.07 · GPTQ-4 4.84 · AWQ-4 4.82 · QuIP#-2 4.43 · QuIP#-3 4.27 · **QuIP#-4 ≈ 4.17**
- **Reading for this box:** at **4-bit specifically**, QuIP# is only *at/slightly below* a strong
  GPTQ/AWQ 4-bit — the rotation's leverage is largest at **2–3 bit** (where QuIP# 2-bit beats
  GPTQ 4-bit). **The 4-bit marginal gain of rotation over a strong GPTQ/AutoRound baseline is
  small** (fractions of a perplexity point) on the body. **widely-reported.**

**Qwen-class models.** QuIP#'s primary evaluation is LLaMA-2 (7B/13B/70B). No Qwen-class
numbers are a primary target in the paper → **unproven** for Qwen specifically (technique is
model-agnostic in principle).

---

## Q2. SpinQuant (arXiv 2405.16406)

**Source (origin-confirmed).** "SpinQuant: LLM quantization with learned rotations"
(Liu et al., 2024; Meta / facebookresearch). Code: `github.com/facebookresearch/SpinQuant`.

**Mechanism.**
- **Learned** (gradient-optimized), **block-diagonal orthogonal** rotation matrices — *not*
  random. It identifies a collection of rotation parameterizations that yield **identical
  full-precision** transformer outputs while improving quantization accuracy (the invariance:
  rotate the weight by R and the activation by R^T=R^{-1}; R^T R = I so the product is unchanged
  in full precision, but the quantized distributions become smoother).
- Key empirical finding: **some random rotations are much better than others** — up to **13
  points** difference in downstream zero-shot reasoning just from the random seed. SpinQuant
  *learns* the rotation to optimize the quantized network's accuracy.
- This is the exact reference **Bonsai 2's whitepaper cites** for "blockwise Hadamard rotation
  with fixed ±1 signs." (origin-confirmed via the local-ai-zone deep dive.)

**Reported numbers at 4-bit.** SpinQuant is a **W4A4KV4** method (weight + activation + KV
cache all 4-bit) — **not** weight-only.
- LLaMA-2 7B zero-shot reasoning: narrows the gap to full precision to **2.9 points**;
  surpasses LLM-QAT by **19.1** pts and SmoothQuant by **25.0** pts; **outperforms QuaRot**.
- LLaMA-3 8B (hard-to-quantize): reduces the gap to full precision by **up to 45.1% relative
  to QuaRot**.
**origin-confirmed (abstract).**

**Models / code / cost.** LLaMA-2 7B, LLaMA-3 8B, LLaMA-2/3 70B. Code public
(facebookresearch/SpinQuant). **Compute-heavy:** gradient-based rotation learning — KurTail
(2503.01483) notes SpinQuant for Llama3-70B "requires at least four NVIDIA H100 80GB GPUs."
**origin-confirmed.**

**Relevance to this box:** SpinQuant's wins require A4 (activation 4-bit). A weight-only
adoption (W4A16) would capture only the weight-side smoothing, not the headline W4A4 result.

---

## Q3. PrismML "Bonsai 2"

**Verdict: REAL.** **origin-confirmed** from primary/secondary sources fetched.

- **Who / venue.** **Prism ML** (a Caltech-rooted startup, Pasadena + San Francisco).
  "Ternary Bonsai 2 27B" released **Sept 17, 2026**, built on **Qwen3.8-27B** (the *same*
  model family as this box; architecture unchanged). Ships a 14-page technical report.
  Primary: `docs.prismml.com`, `github.com/PrismML-Eng/llama.cpp`, `github.com/PrismML-Eng/mlx`,
  HF `prism-ml/Ternary-Bonsai-2-27B-*`.
- **What is actually implemented & measured.**
  - **Ternary** weights {−1, 0, +1}, one FP16 scale per **group of 128**, in a **rotated
    (Hadamard) weight basis folded offline** + an **activation-side Walsh–Hadamard transform at
    runtime** (the QuaRot-style invariance).
  - Two packings: **PTQ1_0 = 1.76 bpw (5.93 GB 27B)**; **PQ2_0 = 2.16 bpw (7.25 GB)**.
  - **The "98.2% retention / 1.76 bpw / 5.9 GB" figures are now source-confirmed**, but see the
    independence caveat below.
- **The 98.2% figure — INDEPENDENCE CAVEAT.** 98.2% = **83.9 vs 85.4** average across **Prism
  ML's own 20-benchmark thinking-mode suite**. This is a **vendor-run** evaluation. The
  `local-ai-zone.github.io` "deep dive" is a **third-party write-up** that *fact-checked the
  numbers against Prism's own primary documents* (tech report, model cards, announcement posts)
  — it is **not** an independent benchmark reproduction. The write-up itself flags "the caveats
  that apply to vendor-run evaluations." → **No independent third-party reproduction found.**
  Also: the 98.2% is for **ternary (1.76 bpw)**, **not INT4**. Comparison from the same suite:
  Unsloth UD-Q4_K_XL (a real 4-bit build) = 85.18 @ 17.6 GB (98.7%); Bonsai 2 27B = 84.78 @ 5.9 GB
  (within 0.4 pts, ~3x smaller); the model's IQ2_XXS "2-bit" = 72.59 (84.1%, "fails selectively").
- **llama.cpp-fork-only (no vLLM path) — CONFIRMED.** Docs: "**Do not run Ternary Bonsai 2 on
  stock llama.cpp.** PQ2_0/PTQ1_0 are refused outright; a Q2_0 file loads but outputs gibberish."
  The activation-side **FWHT (F16 input, CPU) is an OPEN, not-merged PR:
  `ggml-org/llama.cpp#27779`**. Also needs the **MLX fork** (`PrismML-Eng/mlx`, branch `prism`)
  for the MLX package. **No vLLM path exists on this box.** **origin-confirmed.**
- **Note for this box:** Bonsai 2 is the most complete *industrial* application of the
  QuIP#/SpinQuant rotation idea to exactly this model family (Qwen3.8-27B) — strong evidence
  the rotation technique is viable here — but at a **sub-2-bit ternary** operating point, so it
  is **not** a direct INT4 benchmark.

---

## Q4. vLLM ecosystem integration path

**Bottom line: a weight-side Hadamard fold is feasible with tools already in this stack; the
activation-side FWHT is the part that is not upstream.**

Concrete, cited integration points (all **origin-confirmed** from repo READMEs/docs):
- **`vllm-project/compressed-tensors`** (the library this box's W4A16 checkpoint uses):
  README lists **"Transform Support: Rotation-based quantization techniques (Hadamard, random
  Hadamard, random matrix transforms)"**, and the quant config carries a **`transform_config`**
  field. → **The rotation hook already exists in this box's quant library.**
- **`vllm-project/llm-compressor`**: "Comprehensive set of quantization algorithms and
  transforms for weight, activation, KV cache, and attention quantization"; docs cite
  `vllm-project/vllm` **PR #36799**.
- **`amd/Quark`** (integrates with vLLM): pre-quant algorithms **"rotation" and "quarot" are
  fused into a single rotation algorithm**, configured via **`RotationConfig`** — first-class.
- **Upstream vLLM** `turboquant` module (docs v0.22.0): "**Hadamard rotation** + per-coordinate
  Lloyd-Max scalar quantization for keys, uniform for values" — an *upstream* rotation-based
  (KV-cache) quant path, precedent that rotation ops live in vLLM.
- **`huggingface/transformers` issue #45203**: "Add PolarQuant — Walsh-Hadamard rotation +
  Lloyd-Max centroids" (open feature request).

**What it takes to serve a Hadamard-rotated checkpoint in the vLLM fork** (two patterns, both
seen in the literature):
- **(a) Weight-only fold (QuIP#-style, matches this box's W4A16):** rotate the weights offline
  (fold R into the stored checkpoint) and declare it via `compressed-tensors` `transform_config`.
  **No per-forward activation transform is needed** → standard INT4 dequant kernels run
  unchanged. This is the low-risk path. **unproven end-to-end on this fork (feature exists; not
  yet exercised here).**
- **(b) Activation-side (QuaRot/Bonsai/SpinQuant W4A4):** fold R into weights **and** insert a
  per-forward **FWHT op on the activations** before each quantized matmul. This is the piece
  that is **not upstream** (open `llama.cpp#27779`; would be a custom op in the vLLM fork).
  Only needed if you move to W4A4. **origin-confirmed (not upstream) / unproven (fork patch size).**

---

## Q5. lm_head / embedding quantization sensitivity

- **Practice (origin-confirmed):** GPTQ / AWQ / bitsandbytes pipelines **typically exclude
  `lm_head` and `embed_tokens` from aggressive 4-bit**, keeping them at **8-bit or 16-bit**.
  `huggingface/transformers` issue **#31474** ("Quantization support for heads and embeddings")
  documents the lack of embedding-quant support in bitsandbytes; `ms-swift` docs treat
  `embed_tokens`/`lm_head` as **special cases** in quant export.
- **QuIP# (weight-only VQ):** VQs **all** weights *including lm_head* down to 2-bit (weight-only,
  no activation quant) and still reports strong results → **VQ'd lm_head at 2-bit works in
  QuIP#** (a data point that lm_head tolerates low-bit when the representation is VQ + rotated).
  **origin-confirmed (method) / widely-reported.**
- **Measured ppl delta attributed specifically to lm_head precision:** I did **not** find a
  clean isolated ablation in the sources fetched. What is established: lm_head is the **final
  projection to the 248320-vocab logits**, so its precision maps directly into perplexity/logit
  quality; and incoherence/rotation applied to lm_head spreads its heavy-tailed input columns.
  **The specific isolated lm_head ppl delta = unproven** in this survey.
- **Direct relevance to this box:** the box *measured* the strong weight-side outlier structure
  **in lm_head** (217x residue-class variance deviation, 9x per-group-128 scale range). That is
  exactly the heavy-tailed-column signature rotation targets, and lm_head is the one tensor where
  a weight-side fold is (i) justified by local measurement and (ii) directly logit-relevant.
  **This is the highest-potential target for a rotation experiment on this box.** **unproven (magnitude).**

---

## Q6. Runtime cost of block-diagonal Hadamard on 5120 / 17408-dim, sm86

**FLOPs (analytic).** A Hadamard is O(d log d) per vector.
- Full 5120-D: 5120·log2(5120) ≈ 5120·12.3 ≈ **63k ops/vector**.
- **Block-diagonal, block 128:** (5120/128)·(128·log2 128) = 40·896 ≈ **36k ops/vector**.
- vs the matmul it precedes (5120×4096 ≈ **21M MACs**): the Hadamard is **~0.2% of matmul FLOPs**.
  For 17408-D, the same ratio holds (it scales with d log d vs d·d). So the transform is
  arithmetically tiny; the question is kernel/launch overhead, which the literature addresses.

**Measured overhead (multiple independent, origin-confirmed):**
- **PALU (ICLR 2025):** "It's worth re-emphasizing that **Hadamard Transform will not bring
  extra overhead**" — measured decode latency on a single **RTX 4090 (sm89)**, closest measured
  point to sm86.
- **MR-GPTQ (arXiv 2509.23202):** "high-performance GPU kernels ... **negligible overhead**, by
  **rotation fusion into the weights** and fast **online** computation of the activations,"
  yielding *speedups* vs FP16 of 2.2x e2e (B200) / 4x e2e (RTX 5090). Confirms the
  **fold-into-weights + cheap-online-activation = negligible overhead** principle (on newer
  GPUs).
- **Intel (Jun 2026 blog):** "the rotation is a fast, structured Walsh–Hadamard transform ...
  **decode overhead does not degrade tail latency**."
- **QuaRot (2404.00456):** applies an **online WHT** to q/k/v — explicitly a runtime-efficiency
  design; 4-bit LLaMA2-70B at ≤ 0.47 ppl loss, 99% zero-shot retained.
- **Bonsai 2:** decode is **memory-bandwidth-bound** (27.7 tok/s while streaming ~201 GB/s on an
  M5 Pro) → a cheap in-VRAM O(d log d) transform adds little at decode; **prefill** is the
  sensitive path (they quote up to ~11% run-to-run variance on prefill).

**sm86 (RTX 3090) specifically: unproven.** No sm86-specific measurement found. Closest are
sm89 (PALU, RTX 4090) and sm90/sm100 (MR-GPTQ). sm86 has lower FLOPs and **no FP4 tensor
cores**, so *absolute* throughput is lower, but the *relative* Hadamard overhead (fraction of
the matmul) is the same O(d log d)/O(d²) ratio and is expected **negligible at decode**,
**small-but-visible at prefill**. **upstream-validated (principle) / unproven (sm86 absolute).**

---

## Cross-cutting read for THIS box (W4A16, weight-only)

- The literature's rotation leverage is concentrated in **W4A4**. At **W4A16 weight-only 4-bit**,
  expect the body-level gain over a strong GPTQ/AutoRound baseline to be **small** (QuIP# 4-bit
  ≈ GPTQ/AWQ 4-bit; the gap opens at 2–3 bit).
- The box's **own measurement** points the gain at **lm_head**, not the body: it is the tensor
  with confirmed 217x residue-class / 9x scale-range outlier structure, and it feeds logits
  directly. A **targeted weight-side Hadamard fold on lm_head** (QuIP#-style incoherence,
  declared via `compressed-tensors` `transform_config`, no per-forward patch) is the
  highest-EV, lowest-risk rotation experiment here.
- Going to **W4A4** to capture the full rotation benefit would require the **activation-side
  FWHT op** in the vLLM fork — the piece that is *not* upstream (cf. `llama.cpp#27779`).

---

## 3-line verdict

- **(a) Expected 4-bit gain of rotation vs a strong GPTQ/AutoRound baseline on 27B dense:**
  **modest at the body** (W4A16 weight-only 4-bit; QuIP# 4-bit ≈ GPTQ 4-bit, gap opens at 2–3 bit),
  **but concentrated at lm_head** on this model — where the box measured 217x/9x outlier
  structure and QuIP# (weight-only, rotates lm_head) is the precedent. Best framed as an
  lm_head-targeted fold, not a global body win.
- **(b) vLLM integration feasibility for this fork:** **HIGH for a weight-only fold** —
  `compressed-tensors` (already in use) has `transform_config` + Hadamard/random-Hadamard
  transform support, and Quark/llm-compressor ship a `rotation`/`quarot` pre-quant path into
  vLLM; a folded weight rotation needs **no per-forward patch**. The activation-side FWHT
  (W4A4-only) is the non-upstream piece.
- **(c) Bonsai 2 status:** **REAL** (Prism ML, Sept 2026, Qwen3.8-27B ternary). **98.2% is
  vendor-run, not independently reproduced.** **llama.cpp-fork + MLX-fork only, no vLLM path,
  rotation not upstream** (open `ggml-org/llama.cpp#27779`). It is **ternary 1.76 bpw, not INT4**.

---

## Measured CONTRAINDICATIONS found

1. **Weight-vs-activation trade-off (OptRot, arXiv 2512.24124):** "In the W4A4 setting, both
   OptRot and OptRot+ perform **worse**, highlighting a trade-off between weight and activation
   quantization." → a rotation tuned for weight outliers can *hurt* once you also push
   activation 4-bit. Relevant if this box ever moves to W4A4. **origin-confirmed.**
2. **Most of the headline rotation gain is unreachable at W4A16.** QuaRot/SpinQuant/QuIP#-2bit
   get their numbers by quantizing activations too; a weight-only W4A16 adoption captures only
   the weight-side smoothing. **origin-confirmed (mechanism).**
3. **Bonsai 2's 98.2% is not independent and not INT4** (vendor-run ternary) — do not
   extrapolate it to this box's 4-bit W4A16 target. **origin-confirmed.**
4. **Hybrid-attention caveat:** Qwen3.8-27B is 75% linear-attention (Gated DeltaNet); rotation
   behavior on the **linear-attention / recurrent-state path** is less studied than on standard
   MHA (Bonsai 2 keeps the small recurrent state in full precision). **origin-confirmed (arch) /
   unproven (rotation-on-linear-attn effect).**
5. **sm86 runtime not measured** — "negligible overhead" is validated on sm89/sm90/sm100, not on
   an RTX 3090; treat the sm86 decode number as an extrapolation. **unproven.**

---

## Source list

- QuIP# — arXiv **2402.04396** (abs + PDF); code `github.com/Cornell-RelaxML/quip-sharp`.
- QuaRot — arXiv **2404.00456** ("Outlier-Free 4-Bit Inference in Rotated LLMs").
- SpinQuant — arXiv **2405.16406** ("LLM quantization with learned rotations"); code
  `github.com/facebookresearch/SpinQuant`.
- KurTail — arXiv **2503.01483**; ThinQuant — **2609.36120**; NestQuant — **2502.09720**;
  OptRot — **2512.24124**; HeRo-Q — **2601.21626**; PrefixQuant — **2410.05265**; ADMM-Q —
  **2605.11222**; Loopy — **2609.21450**.
- MR-GPTQ (rotation fusion, negligible overhead, FP4) — arXiv **2509.23202**.
- PALU (Hadamard "no extra overhead", RTX 4090) — ICLR 2025, `proceedings.iclr.cc/.../PALU`.
- Bonsai 2 — `docs.prismml.com/download/formats`; `github.com/PrismML-Eng/llama.cpp`;
  `github.com/PrismML-Eng/mlx`; HF `prism-ml/Ternary-Bonsai-2-27B-*`; open PR
  `ggml-org/llama.cpp#27779`; third-party write-up `local-ai-zone.github.io` deep dive.
- vLLM ecosystem — `vllm-project/compressed-tensors` (README: Transform Support +
  `transform_config`); `vllm-project/llm-compressor` (docs cite vLLM **PR #36799**);
  `amd/Quark` (RotationConfig, rotation+quarot); vLLM `turboquant` (docs v0.22.0);
  `huggingface/transformers` **issue #45203** (PolarQuant) & **issue #31474** (heads/embeddings).

*Compiled 2026-10-13 by prior-work survey (web_search + arXiv API + Semantic Scholar + primary
repo/docs fetches). No GPU, no server, no commits.*
