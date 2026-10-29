# KV-cache fidelity-per-byte: analysis of an external optimization review

An external LLM session (2026-10-08) proposed replacing the shipped KVarN k4v2
KV cache with a ~3.25 bpv trellis/TCQ codec and spending the freed VRAM on
weights. This file records that analysis, its verification status against this
repo and its cited sources, and the parts of it this repo does **not** accept.

## The claim in one line

Q8/Q8 KV is *not* the quality target (KLD 0.006425 vs f16's 0.000037 on this
model family, 174×); the right unit is fidelity-per-byte, and at ~3.3 GiB for
262k the cache can be near-fp16 — freeing ~5 GiB for the weights.

## Verified against this repo

| Claim | Evidence |
|---|---|
| KVarN k4v2_g128 is ~840 B/token/layer, +0.16% ppl (8.223→8.236) | `docs/long-context.md:15`, `:23`, `:34` |
| fp8 default caps at 150k; KVarN pool 420k w/ 4 slots, 262k fits; NIAH correct to 240k | `docs/long-context.md:31`, `:33` |
| 112k single-user: fp8 68.1 tok/s vs KVarN 32.0 (2.13×), decomposed as ~1.98× step time + ~7% acceptance (2.56→2.38 tok/step) | `docs/long-context.md:49-62` |
| 2048-token blocks cost short requests as much as fp8's 800-token blocks; 38 of 64 slots resident | `docs/long-context.md:38` |
| Speculation is exact — sampled distribution unchanged | `docs/long-context.md:61` |
| `DFLASH_TOKENS=15` lookup drafting: 381 tok/s reproducing context | `docs/optimizations.md:311` |
| 16 attention layers in the hybrid DeltaNet stack | `docs/long-context.md:10` |
| vLLM here ships fp8 / int8-head / int4-head / kvarn modes, no per-side (K≠V) mixing | `docs/long-context.md:95` |

## Verified against the cited sources

| Source | Status |
|---|---|
| llama.cpp discussion #23470 (KLD matrix at 65k on Qwen3.8-27B: q8_0/q8_0 0.006425, q8_0/q5_1 0.007628, q8_0/q4_0 0.012340, q4_0/q4_0 0.022060) | ✅ values present on the page; model named |
| HF dataset `spiritbuun/turboquant-tcq-kv-cache` (TCQ 3.25 bpv, ppl 5.802 vs fp16 5.805 on Qwen3.5-27B; FWHT + Viterbi trellis + O(1) window decode) | ✅ claims present **on the dataset card** — see caveats below |
| llama.cpp issue #21591 (boundary first/last layers at higher bits "recovers" quality) | ✅ |
| arXiv 2608.28911 (SemKV, 2.322 code-bits cliff) | ✅ abstract matches |
| KVmix (AAAI), HqeKV (ACL Findings 2026), QTIP (3-bit TCQ 10.34 vs scalar 12.44 on Qwen3-8B) | not independently checked |
| `MIRAI_PREFILL_PLANES=ffn` bit-plane prefill (+18% prefill, KL 0.00028) | ❌ **not in this repo's docs** — from a different project; treat as unverified here |

## Caveats — where the analysis outruns its evidence

1. **The throughput payoff is mis-stated by its own numbers.** It claims a
   ~fp16-fidelity cache gets acceptance back and therefore throughput "near
   fp8's 68 tok/s". Acceptance is only ~7% of the 2.13× tax; the other 1.98× is
   raw step time in the Triton kernels. TCQ needs **new kernels** whose step
   time is unknown (and adds a Viterbi encode on write). Realistic expectation:
   acceptance recovers to ~2.56, step time is a fresh unknown — not a free ride
   back to 68 tok/s.
2. **The TCQ quality claim is a dataset card, not a paper.** Single account,
   single model (Qwen3.5-27B, *not* this model), different ppl scale (5.80 vs
   this repo's 8.09-8.24 — different eval sets, incomparable). Before writing
   any vLLM backend, reproduce the 3.25 bpv ≈ fp16 result offline on
   Qwen3.8-27B (perplexity + KLD at ≥65k, mirroring the #23470 protocol).
3. **Evidence quality is inverted in the tiers.** The strongest measurement
   (KLD matrix, this model at 65k, #23470) supports only the *modest* Tier-1
   asymmetric win; the biggest recommendation (Tier 2 TCQ) rests on the weakest
   source. The plan's own ordering (tune → build → policy → weights) is right,
   but tier 2 should be gated on the offline reproduction above.
4. **The budget table ignores drafter KV at 262k.** B+D ≈ 2.5 GiB presumes the
   drafter (DFlash2) can share 262k context; today the drafter is pinned to
   64k/int8 (`docs/long-context.md:117-124`). A 262k drafter cache is real
   memory the W/KV/B/D table doesn't count.
5. **The weight upgrade is a second project.** "~1.5-2.4 bpw better weights"
   means re-quantizing ~24.4B params (e.g. W4A16 → 5.5-6 bpw TCQ/Q6), with its
   own quality verification — not a config change.

## Action order (amended)

1. **Page decoupling** (2048-token attention block vs DeltaNet's 1.63 MB page)
   — patch code this repo already touches; biggest batch-mode win, no quality
   risk.
2. **Drafter re-quant** to Q2/IQ2 — free memory, exact speculation.
3. **TCQ offline reproduction on Qwen3.8-27B** — perplexity/KLD at 65k+. Only
   if it holds, build it as a parallel `KVQuantMode` (dtype literal + registry +
   KV-cache-spec + backend + kernel), A/B'd against `KVARN`.
4. **Cliff measurement** on this model (SemKV policy layer) — this family is
   unusually KV-robust (q4_0/q4_0 KLD 0.022 here vs collapse on Qwen2.5-7B per
   #23470), so the cliff may sit lower than 2.322 bits.
5. **Weight upgrade** last, verified with NIAH 32k→240k + acceptance + GSM8K,
   not perplexity alone.

## Provenance follow-ups (2026-10-08)

### Mirai (professorpalmer/mirai-s-ada) — verified, and it is NOT a KV codec

The repo serves **Mirai Labs' `Qwen3.8-27B-S-experimental`** via llama.cpp on a
12 GB RTX 4070: the trellis codec (≈2.4 bpw) is applied to the **weights**
(GGUF conversion by alesha-pro, trellis codes copied bit for bit), while the KV
cache is plain **q8_0** reaching 262k via a **tiered cache** (~58k positions in
VRAM, the rest in pinned host RAM) — 34,816 B/token/position. Verified claims:
`GGML_MIRAI_PREFILL_PLANES=ffn` = one int8 plane for the FFN matmuls, +18%
prefill, KL 0.00028, top-token agreement 99.2% (`=2` is the exact path at
−18%); speculation exact, acceptance 77.7%. So Mirai validates *trellis
weights* + *tiered q8_0 KV*, not the TCQ-KV proposal; the analysis cited it
correctly but a reader could come away thinking Mirai runs a compressed KV
cache. It does not — it moves KV to RAM.

### HyperQwen issue/PR scan — nothing on mirai/TCQ/trellis/SemKV

Zero hits across syv-ai/HyperQwen issues+PRs for `mirai`, `tcq`, `trellis`,
`semkv`, `prefill plane`, `q2`/`iq2` drafter. Closest existing work:

- **#262 (MERGED 2026-10-04, in this checkout — see `kvarn/kvarn-fp16-dequant-0.30.0.patch`)**:
  `KVARN_FP16_DEQUANT=1` — fp16 dequant in the fused KVarN decode/verify
  kernels, **2.2× the decode rate at ~94k context (50–52 vs ~23 tok/s)** on a
  3090 @ 250 W, `CTX=huge SPEC=dflash2`, 5 fresh boots, full 268k pool kept.
  Needs #263 (allocator-scratch fix for WSL2 VRAM over-commit) for stable
  fresh-boot speed.
- **MEASURED 2026-10-09, controlled A/B at 112k, this checkout** — #262's 2.2×
  does not reproduce at this context/workload: labd 3-run (512+512 out,
  , fresh boot each) gives fp8 57.0 vs KVarN 28.6 vs
  KVarN+f16d 38.2 tok/s TOTAL → **1.34× from the knob, 1.49× remaining tax**
  (not 2.2×). Decomposition: acceptance did NOT move — tok/step 2.25 → 2.26
  (per-task ±4%, summary 2.24→2.33, qa 2.27→2.18, signs mixed = noise), so
  fp8's 2.53 stays unmatched. The whole knob gain is kernel-side step-rate
  (+33%), which falsifies the doc's ~1.98× step-time share: measured step tax
  after the knob is 1.49×, and the acceptance half (0.89×) is untouched. The
  doc's fp8 baseline also overstates this run's KVarN base (28.6), so the
  2.13×/1.98× decomposition needs a re-measure before any frontier decision
  cites it.- #25: tracking issue for KVarN/DFlash2/CTX=huge. #60, #196: DFlash2/MTP
  acceptance regressions and the draft-vocab acceptance collapse. #102, #69:
  prefix reuse and page padding vs the hybrid block size (the page-decoupling
  thread). #208/#256/#264: KV offload tiers / CPU tier on 2×3090 and eGPU.
  #235: batch-mode `KV=kvarn` capped at MAX_LEN 131072.
  #281/#285: token embeddings into pinned host RAM (`EMBED_UVA`).

**Consequence for this analysis:** its own decomposition says the 2.13×
KVarN tax is ~1.98× step time + ~7% acceptance. The 1.98× half is already
addressed by the merged `KVARN_FP16_DEQUANT` knob (measured 2.2× at 94k,
`CTX=huge`) — the analysis asked for work that shipped four days before it
was written. The remaining gap is only the acceptance half (~7%), which
lowers TCQ's expected payoff further: the honest target is KVarN-with-fp16-
dequant at ~50+ tok/s (94k) plus a ~2.5-7% acceptance recovery, not a
miraculous return to fp8 speeds. Enable the knob and re-run the 112k A/B
before any new codec work.


## Your questions (2026-10-09)

### 1. KVarN + FP16_DEQUANT vs CTX=long fp8 — quality or speed?

**Only speed. Zero quality change by construction, and the A/B above confirms acceptance
doesn't move either.** Mechanism (kvarn/README.md:75-90):  runs the
three fused decode/verify kernels' dequant math (q codes, per-channel/row scales, the
 tile) in fp16 while the dots still accumulate fp32 — same math semantics, same output.
The gain is purely register pressure (halved working-tile width), which is why it only
lifts the long-context split-K decode. With the knob off, the kernels compile to the
same TTGIR/PTX as the port without it.

Measured (2026-10-09, controlled labd 3-run at 112k, fresh boot each, this checkout):
fp8 57.0 / KVarN 28.6 / KVarN+f16d **38.2 tok/s** → **1.34× from the knob, 1.49×
remaining tax**. tok/step 2.25 → 2.26, per-task signs mixed within ±4% = noise.
The doc's ~2.2× at 94k (issue #262) does not reproduce at 112k/this workload.

Quality vs fp8: **there is no KVarN-vs-fp8 KLD A/B measured on this stack.** The
model-specific KLD matrix (#23470, verified) is on q8_0-style codes, not KVarN's 4/2-bit
trellis codes. KVarN's quality case rests on the port's +0.16% perplexity claim, NIAH
correct 4k-240k, and IFEBench unchanged (table above). So: KVarN-with-knob equals plain
KVarN quality — roughly fp8-or-slightly-better, not fp16. TCQ's quality claim
(~fp16 at 3.25 bpv) remains the untested part of the pasted analysis.

### 2. Q2/IQ2 drafter — concrete numbers?

**None on this stack, and the pasted analysis's free claim conflates two things.**
Speculation

## Your questions (2026-10-09)

### 1. KVarN + FP16_DEQUANT vs CTX=long fp8 — quality or speed?

**Only speed. Zero quality change by construction, and the A/B above confirms acceptance
doesn't move either.** Mechanism (kvarn/README.md:75-90): `KVARN_FP16_DEQUANT=1` runs the
three fused decode/verify kernels' dequant math (q codes, per-channel/row scales, the
`p` tile) in fp16 while the dots still accumulate fp32 — same math semantics, same output.
The gain is purely register pressure (halved working-tile width), which is why it only
lifts the long-context split-K decode. With the knob off, the kernels compile to the
same TTGIR/PTX as the port without it.

Measured (2026-10-09, controlled labd 3-run at 112k, fresh boot each, this checkout):
fp8 57.0 / KVarN 28.6 / KVarN+f16d **38.2 tok/s** → **1.34× from the knob, 1.49×
remaining tax**. tok/step 2.25 → 2.26, per-task signs mixed within ±4% = noise.
The doc's ~2.2× at 94k (issue #262) does not reproduce at 112k/this workload.

Quality vs fp8: **there is no KVarN-vs-fp8 KLD A/B measured on this stack.** The
model-specific KLD matrix (#23470, verified) is on q8_0-style codes, not KVarN's 4/2-bit
trellis codes. KVarN's quality case rests on the port's +0.16% perplexity claim, NIAH
correct 4k-240k, and IFEBench unchanged (table above). So: KVarN-with-knob equals plain
KVarN quality — roughly fp8-or-slightly-better, not fp16. TCQ's quality claim
(~fp16 at 3.25 bpv) remains the untested part of the pasted analysis.

### 2. Q2/IQ2 drafter — concrete numbers?

**None on this stack, and the pasted analysis's "free" claim conflates two things.**
"Speculation is distributionally exact" means the *target's* output is unchanged when a
draft token is accepted — it does **not** mean the acceptance *rate* is neutral. The
drafter's own next-token distribution feeds which tokens get proposed; quantizing it to
IQ2 degrades that distribution exactly like HyperQwen #60/#196 degraded it via the draft
vocab (measured acceptance collapse). So the Q2 drafter is an untested hypothesis with a
specific local precedent for failure, not a free knob. Verify before build: run the 112k
labd A/B with an IQ2 drafter checkpoint and log tok/step (2.25-2.53 today) as the metric.

### 3. EMBED_UVA=1 in the user's qwen.sh

Shipped in this checkout (#281/#285): token embeddings moved out of VRAM into pinned
host RAM via UVA. Weight-side VRAM freed — the same slot the budget equation spends on
a higher-bpw weight quant. Complementary to everything here: no interaction with KVarN
page alignment (it's the embedding table, not the cache), already part of the user's
setup, and the INT8-export guard in launcher_common.sh (53557bc) is unrelated.

## Internet scan (2026-10-09): TCQ ~fp16 claim + quantized drafter

### The ~fp16-at-3.25-bpv claim: no independent confirmation, one strong contraindication

The claim still exists **only on the HF dataset card** (`spiritbuun/turboquant-tcq-kv-cache`):
no paper, no leaderboard, no third-party reproduction. But llama.cpp **merged** an
independent implementation of the same trellis/polar-quant family — Google TurboQuant
(feature request #20977 → merged CPU PR #21088/#21089, `tbq3_0`/`tbq4_0`; TurboQuant
Metal/CUDA PRs #21818/#21817/#21062 closed as AI-policy violations). Its measured
sweep (PR #21088 body, Qwen3.5-4B, wikitext-2, ctx=256, CPU, FA on):

| Cache | bpv | PPL (f16 13.8387) | KLD (f16 0) | Same-top-p |
|---|---|---|---|---|
| q8_0 | 8.06 | 13.8348 | 0.00320 | 97.8% |
| q4_0 | 4.06 | 13.8400 | 0.00912 | 93.9% |
| **tbq4_0** | 4.06 | 13.8323 | 0.00960 | 94.1% |
| **tbq3_0** | 3.06 | **14.3198 (+0.48, +3.5%)** | **0.02647** | 91.7% |

Two consequences:
1. **At ~3 bpv the merged implementation lands nowhere near fp16** — +3.5% ppl and
   KLD 2.9× worse than plain q4_0, on the *easiest* eval (ctx=256; #23470's long-context
   degradation only gets worse from there). The dataset card's 5.802-vs-5.805 (~fp16 at
   3.25 bpv) is contradicted by the only replicable third-party datapoint at the same
   bitrate family. Caveats: different model (4B vs 27B), TBQ ≠ the card's exact TCQ
   pipeline (card: FWHT + Viterbi + O(1) window; TBQ3_0 in the PR omits the outlier
   path — "TBQP left for follow-up"), single short-context eval. But the burden is
   now on the card: reproduce **its** result offline before any Tier-2 build — which
   is exactly what the action order already gates on.
2. **Dequant cost is real**: tbq3/tbq4 gen t/s 8.3 vs q8_0 15.7 on CPU (−47%) —
   independent support for caveat 1 (TCQ step time is a fresh unknown; the HF card
   also measured 34 tok/s vs llama.cpp q8_0's 43, CPU).
   Third-party reference impls if we run the reproduction: tonbistudio/turboquant-pytorch,
   TheTom/turboquant_plus (+FWHT/Hadamard rotation speed-up thread, Clifford/"RotorQuant"
   variants). Aaryan-Kapoor's tq3_0 branch: 4.4× compression, gen t/s 11.4 vs q4_0 12.5
   (−9%) — quality table not given.

### Quantized drafter degradation: literature exists, 2-bit specifically does not

- **ML-SpecQD (arXiv 2503.13565)** — the direct paper on quantized draft models:
  MXFP4 (~4-bit block-FP, direct-cast of BF16) drafts work plug-and-play, ≤2× speedup,
  full acceptance ratio measured vs fp16 drafts. Supports "4-bit drafter ≈ fine".
- **SPEQ (arXiv 2510.18525)** — quantized draft built from target weights
  (exponent remapping + parameter sharing): again 4-bit-family, acceptance preserved.
- **No paper or issue found proposing a 2-bit/IQ2 drafter.** arXiv search
  ("quantized draft" + "speculative decoding", 8 newest) returns only those two;
  llama.cpp issue search for iq2/q2_0 draft models returns nothing about acceptance.
  So the Q2/IQ2 drafter question has **zero internet evidence in either direction** —
  the only precedent remains local: HyperQwen #60/#196's draft-vocab acceptance
  collapse. Unverified hypothesis with a specific local failure mode; the 112k labd
  A/B with tok/step (2.25–2.53) stays the gate.

### Bottom line

- TCQ ~fp16 claim: internet-consistent evidence now **leans negative** (merged TurboQuant
  at 3 bpv ≠ fp16); reproduction before build remains mandatory, and the merged PR
  supplies a working reference implementation to reproduce against.
- Q2/IQ2 drafter: no internet evidence at all; 4-bit quantized drafts are literature-
  validated (ML-SpecQD, SPEQ), 2-bit is unexplored territory — treat the doc's "free
  memory, exact speculation" line as untested, gated on the same labd A/B.
