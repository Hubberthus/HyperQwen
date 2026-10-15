# Rotation Re-Quant — Step 1: Layer-Patch Perplexity Gate (plan)

> **Project class: Spike (rotation follow-up, phase 1).** Output is a go/no-go verdict with
> measured numbers, not shipped code.
> Executes the proposal in `docs/superpowers/plans/2026-10-13-salted-quantization-rotation.md`.

**Question:** Does block-Hadamard rotation + uniform INT4 group-128 re-quant of a few body
layers produce *output* (perplexity) at least as good as the AutoRound-INT4 layers it
replaces? I.e. does the measured 8.3 % weight-MSE advantage (down_proj L63) survive contact
with an output-tied metric — the check the salted spike flagged as missing (its revival path #3)?

**Acceptance floor (user requirement, 2026-10-13):** the salted run's |w|-tail plots (original
vs dequant, 9 tensors: L0/L27/L63 × gate/down/q + lm_head) show the current checkpoint
**annihilating the original's low-magnitude weight plateau** — dequant density is zero below
~1e-3, the erased mass reappears as a hump, and the heavy tail is preserved exactly. The patch
copy must keep **at least** this low-amplitude fidelity and beat it where it can. Quantified per
patched tensor (zero counts + low-band |w| mass retention: original vs current dequant vs rot
dequant) and enforced in the Task 3 gate.

## Roles of the artifacts (read first)

- `models/Qwen3.8-27B-W4A16-AutoRound/` — **THE CURRENT CHECKPOINT.** What this box serves
  (7 shards, ~14 GB; INT4 group-128 linears, INT8 heads, marlin-packed 8 int4/int32). It is
  the comparison baseline AND the source of every unpatched layer in the test copy.
- `models-fp16/Qwen3.8-27B/` — **the BF16 originals** (55.6 GB, 18 shards). The
  pre-quantization source the W4A16 checkpoint was made from. Role: **fold source only** —
  rotation is applied to full-precision weights *before* quantization (a quantized checkpoint
  cannot be rotated), and ground truth for MSE. Never a comparison baseline: it does not fit
  the 24 GB 3090, and comparing against it would measure generic quantization damage, not the
  rotation delta.
- `models/Qwen3.8-27B-W4A16-rot-patch/` (NEW) — **the patch copy.** Byte-identical to the
  current checkpoint except 4–8 selected layers' INT4 linears, which are replaced by
  rotation-folded INT4. The test subject. Same footprint → same VRAM fit.

The baseline must be W4A16 because the experiment is a same-footprint, same-format A/B:
"rot-INT4 layers vs the AutoRound-INT4 layers we serve today".

## Cited baseline numbers (do not re-derive)

- down_proj L63: rot-INT4 weight MSE 1.5803e-6 vs current checkpoint 1.7232e-6 (−8.3 %);
  group-128 absmax spread 14.5× → 5.0×; max|w| 0.50 → 0.161
  (`analysis/salted/rotation/rotation_sim.json`)
- Rotation config that produced those numbers: block-diagonal orthonormal Hadamard, 128-blocks,
  on the input dim (`analysis/salted/rotation/rotation_sim.py`)
- lm_head stays INT8 in both checkpoints — out of scope, so any output delta attributes to
  the patched body layers
- in_proj_qkv geometry (linear-attn layers): packed [10240, 640] I32 = 8 int4/int32,
  scale [10240, 40] → dequant [10240, 5120] group-128

## Tech stack

`venv/` (Python 3.12, torch, numpy, safetensors, compressed-tensors, huggingface_hub).
Task 0 CPU; Tasks 1–2 GPU (3090, 250 W cap — settled, do not raise). Both checkpoints fit VRAM
(footprint-identical).

## Global constraints

- Never modify the current checkpoint; the patch copy is a new directory.
- lm_head, embed_tokens, MTP modules, all norms: byte-identical in the patch copy.
- Patched tensors are written in the checkpoint's EXISTING marlin layout (8 int4/int32, same
  interleaving as the tensors they replace); the serving kernel must read them with zero fork
  changes.
- Dequant pin on every patched tensor: manual tiled dequant vs the venv's `compressed_tensors`
  = 0.0 max diff (the spike's pin discipline).
- Identical server config for both checkpoints — same launch script, same bool knobs; only
  the model path differs.
- AGENTS.md bench discipline: cold prompts unique per SCRIPT RUN (prefix-cache salt); a
  SIGTERM'd vLLM lingers — verify the NEW boot's pid + cmdline + pool line before trusting
  numbers.
- No serving-path changes: `patches/`, launchers, and `kvarn/` stay untouched.
- Every number cites its computation.

---

### Task 0: target survey + library-parity gate (CPU)
**Produces:** `analysis/rotation/survey.json` + layer selection + parity verdict
- [x] Step 1: survey ALL 1200 INT4 linears: per-tensor group-128 absmax spread (max/median of
  the group-absmax table) + global max|w|, from the BF16 originals (reuse
  `rotation_sim.py` machinery; stream, ≤4 GB tiles).
- [x] Step 2: select 4–8 layers. Mandatory: L63 (measured 14.5×), ≥1 linear-attention layer
  (e.g. L0), ≥1 full-attention layer (e.g. L27). Fill remaining slots with the highest-spread
  layers from Step 1. Record selection + rationale in `survey.json`.
- [x] Step 3: parity check — run compressed-tensors' built-in transform_config/Hadamard
  pre-quant path on ONE selected tensor; its dequant must match B's hand-rolled fold
  (`rotation_sim.py`) at 0.0 (or within one int4 step). If the library path diverges, use the
  hand-rolled fold for the patch and record why.
- **Gate:** selection + parity pass before Task 1. If the survey shows L63-class structure
  (spread ≥ ~10×) is RARE across the 1200, stop and report that as a (negative) result — do
  not spend GPU time on a patch built from atypical layers.

**Result (2026-10-13): GATE PASS.** 400 INT4 linears surveyed (the checkpoint's actual INT4
set; 24.3 B elements, 62 s wall). Selection: **[0, 27, 38, 45, 56, 58, 60, 63]** (anchors L0
linear-attn, L27/L63 full-attn; fills L38/L45/L56/L58 linear-attn out_proj + L60 down_proj).
Parity: **PASS** — compressed-tensors 0.17.0 `TransformConfig` (hadamard, head_dim=128,
location='weight_input') == hand-rolled float64 fold to float64-ULP level; dequant max diff
6.06e-3 = 0.30× one INT4 step (`analysis/rotation/parity.json`).
Metric cross-check (parent verification): the child's primary "spread" pooled column-group
absmax over all rows; the benchmark metric is "spread_per_row" (per-row group table — L63
reproduces 14.5248 exactly under it). Selection re-ranked under the correct metric and HOLDS:
all fills ≥ 18× per-row; L0 (33.33×) and L60 (33.57×) are the top-2 per-row layers. Selection
unchanged; rationale in `survey.json` now cites both metrics.

### Task 1: build the patch copy
**Produces:** `models/Qwen3.8-27B-W4A16-rot-patch/`
- [x] Step 1: copy the current checkpoint (cp -a); verify total size matches.
- [x] Step 2: for each selected layer's INT4 linears: load the BF16 original → block-128
  orthonormal Hadamard on the input dim → symmetric group-128 uniform INT4 (per-group absmax
  scale, round-to-nearest) → repack in the checkpoint's marlin layout → replace the
  weight_packed + weight_scale tensors (weight_shape unchanged).
- [x] Step 3: dequant pin 0.0 on EVERY patched tensor (venv compressed_tensors).
- [x] Step 4: low-amplitude tail fidelity, per patched tensor (CPU): zero counts + low-band |w|
  mass retention, original vs current-checkpoint dequant vs rot dequant →
  `analysis/rotation/tail_fidelity.json`.
- [x] Step 5: byte-identity check of every UNPATCHED file (hash compare against the current
  checkpoint).
- [x] Step 6: header/config validity — safetensors headers consistent (dtypes, shapes,
  index.json). NOTE: the naive patch kept quantization_config.json unchanged, which makes it
  UNSERVABLE (missing transform_config) — see Task 1b below.

**Result (2026-10-13): BUILD CLEAN, FLOOR FAILED.** 50 tensors patched (6/linear-attn layer ×
6 + 7/full-attn × 2; in_proj_a/b are not INT4 in the checkpoint, correctly excluded).
Round-trip pin 3/3 byte-identical; dequant pin 0.0 on 50/50; 23 unpatched files sha256-identical
(parent spot-verified 2/2 + total size exact 23,132,037,266 B = source); headers/quant_config
unchanged. Parent independently recomputed the L63 down_proj tail row — exact match with
`tail_fidelity.json`. **Tail floor: satisfied on 4/50 tensors** (L27.k, L63.k/o/v — attention
projections); 46/50 lose low-band mass vs the current checkpoint, 36/50 have more exact zeros.
The plan's Step-4 mechanism claim (finer step → more plateau survives) is FALSIFIED by the
measurement: rotation + NAIVE uniform rounding trades low-band L1 mass for heavy-tail L2 MSE —
no contradiction with the 8.3 % MSE win, which is L2 and tail-dominated. The current
checkpoint's low-band fidelity is an AutoRound optimization property; the rot fold paired
rotation with naive rounding. FLOOR-FAIL branch per the decision table: investigate before
Task 2. Candidate fix (Task 1b): rotation + optimized rounder (AutoRound/GPTQ on the rotated
weights) — keeps outlier suppression, may recover low-band mass.

### Task 1b: rotation + AutoRound re-quant (in flight, dispatched 2026-10-14)
**Produces:** `models/Qwen3.8-27B-W4A16-rot-patch-AR/` + `tail_fidelity_autoround.json` +
`mse_table.json`
**Why:** Task 1's floor failed 4/50 because naive uniform rounding gives up low-band L1 mass;
the current checkpoint's low-band fidelity is an AutoRound property. Re-quantize the SAME 50
folded tensors with AutoRound (per-layer, offline; 24 GB card cannot hold the 27B BF16 model —
capture inputs on the INT4 baseline, optimize one layer at a time).
- [x] Capture raw per-linear inputs x on the INT4 baseline (wikitext-2, ~256×2048 tokens).
- [x] Fold (block-128 Hadamard, float64) + per-layer AutoRound with error signal x' = x·H
      (runtime applies H inside the wrapped method; H symmetric involution).
- [x] Export/pins: round-trip byte-identity, dequant pin 0.0 ×50, repack_nibbles marlin layout.
- [x] Splice into `models/Qwen3.8-27B-W4A16-rot-patch-AR/` + **add transform_config** (below).
- [x] Serving smoke A/B (baseline vs patch, same prompts, argmax + logit-diff check).
- [x] Re-measure floor (same t definition) + per-tensor MSE (current vs naive vs rot-AR).

**Result (2026-10-14): BUILD + SERVING PASS, FLOOR STILL 4/50 — AR hypothesis FALSIFIED.**
- Optimizer: hand-rolled per-layer SignRound (auto_round 0.16.0 `SignSGD`, 200 iters/tensor,
  LinearLR 1→0, best-batch tracking, f64-final-codes convention bit-matched to the checkpoint).
- Pins: round-trip 3/3 byte-identical; dequant pin 0.0 ×50; 21 unpatched files sha256-identical;
  total 23,132,042,754 B = baseline + 5,488 B config growth (MATCH, `export_ar.json`).
- transform_config: 38 rotated targets (12 `in_proj_qkv/z` tensors re-quantized PLAIN and dropped
  from targets — the fused 4-partition `in_proj_qkvz` breaks `_validate_tfm_schemes`; fork patch
  parked per user decision). At load: 26 `CompressedTensorsLinearTransformMethod` wrappers = all
  26 fused parent modules of the 38 rotated tensors (Hadamard structural diff 0.0; numeric rel
  5.8e-4 = fp16). `all_50_targets_covered:false` in the pin report is a stale 50-target
  expectation — coverage is complete for the rotated set.
- Serving smoke: **PASS.** AR vs baseline argmax: 5/8 prompts identical, "2+2" one digit of
  drift ("4, 3 + 3"), 2 coherent paraphrases; logprob deltas ~1e-2; negative control (no
  transform) garbage — smoke has teeth.
- **Floor: 4/50, unchanged from naive** (L27.k, L63.k/o/v). AR beats current on zeros 10/50,
  band retention 4/50 — AutoRound rounding does NOT recover the low-band plateau.
- MSE (corrected effective-weight metric, `mse_table_ar.json` v2): rotated-38 vs folded target:
  naive-rot 1.753e-6, rot-AR 1.866e-6 (**AR 6.4 % WORSE than naive RTN**); per-tensor vs current
  (L63 down: cur 1.72e-6 / naive 1.50e-6 / AR 1.53e-6 — the fold's win over current is ~12 %,
  from the rotation, not the rounder). Plain in_proj: option A (AR plain) 3.43e-6 vs option B
  (keep current) 2.94e-6 → **keep current** (no regression, no transform needed).
- **Task 1b verdict:** the "optimized rounder recovers the low-band floor" hypothesis is
  FALSIFIED — the low-band collapse is a property of rotation+group-absmax scale redistribution,
  not of rounding. The serving contribution is real (transform_config wiring makes the patch
  servable, near-baseline output), so Task 2 ppl can test the rotation direction as a whole;
  but weight-space signals now say the direction rests entirely on whether the ~12 % fold MSE
  win on the 38 rotated tensors shows up at output level, against a floor that fails 46/50.
- Cleanup: tmp_negctrl deleted (21.5 G); captures dirs emptied; `opt_results/` (23 G npz codes)
  RETAINED as reproducibility record — flag: delete if disk needed. GPU free (407 MiB), no
  lingering processes.

**KEY DISCOVERY (2026-10-14, parent):** this vLLM fork HAS runtime transform support —
`vllm/.../compressed_tensors/transform/linear.py` (`CompressedTensorsLinearTransformMethod`
wraps the quant method, applies input/output transforms around the gemm) and
`transform/module.py` (`HadamardTransform`). An ONLINE `input`-location hadamard applies H to
the input activations at runtime; its inverse folded into the weight is exactly
W_stored = W·H — the builder's fold (H symmetric involution ⇒ y = (x·H)·Q(W·H)^T ≈ x·W^T).
**Therefore the patch's quantization_config.json MUST gain a transform_config** (everything
else unchanged), else the patch serves wrong math:
```json
"transform_config": {"config_groups": {"rot": {"type": "hadamard", "head_dim": 128,
  "apply": [{"targets": ["<the 50 module base names>"], "location": "input",
  "inverse": false}]}}}
```
Schema: `compressed_tensors/transform/transform_config.py` + `transform_scheme.py`.
TP must stay 1 (online transforms reject TP>1). New mandatory pins:
(1) after load, all 50 linears' quant_method are `CompressedTensorsLinearTransformMethod`;
(2) runtime transform matrix == `build_patch.py` `sylvester(128)` (max abs diff 0.0);
(3) serving smoke A/B passes (negative control: the naive Task 1 patch, no transform_config,
serves garbage — proves the smoke has teeth).
**Consequence:** the naive Task 1 patch (`models/Qwen3.8-27B-W4A16-rot-patch/`) is UNSERVABLE
as built — its floor numbers are still valid weight-space measurements, but it must not be
served. Task 2's smoke step would have caught this; fixed here instead.

### Task 2: perplexity measurement (GPU)
**Produces:** per-run ppl table, `analysis/rotation/ppl.json`
- [ ] Step 1: confirm what `bench/quality_battery.py` measures (corpus, token count, metric);
  use it if it yields per-checkpoint perplexity/logprob. Otherwise a minimal eval script:
  wikitext-2 test split, 2048-token sliding windows at 256 stride, ≥100k tokens total (the
  corpus must be large enough that ppl noise is well below any plausible delta).
- [ ] Step 2: run BOTH checkpoints sequentially, identical flags: current W4A16 ×2 runs,
  rot-patch ×2 runs (duplicates establish measurement noise). Record ppl + wall per run.
- [ ] Step 3: post-load serving smoke on the patch copy (a few tokens through the real server)
  before the full runs — a layout mismatch would show as garbage output, not a subtle ppl.

### Task 3: gate + report
**Produces:** `docs/superpowers/plans/2026-10-13-rotation-step1-findings.md` + ledger update
- Decision rule:
  - **PASS:** patch ppl ≤ current ppl + noise **AND** per-tensor low-amplitude tail fidelity ≥
    current checkpoint (zero counts + low-band mass retention, Task 1 Step 4) → recommend Step 2
    (full 3.23 B-element re-quant). Report by how much the tail beat the floor.
  - **FLOOR-FAIL:** ppl fine but any patched tensor loses low-amplitude mass vs the current
    checkpoint → investigate before any Step 2 (a rotation that erases more of the plateau than
    AutoRound did is not the rotation we want).
  - **FAIL:** patch ppl > current ppl + noise → rotation direction closed at output level;
    document the boundary of what rotation buys at W4A16.
  - **AMBIGUOUS** (within noise): one re-run pair; if still ambiguous, expand to 8 layers and/or
    a larger corpus, then adjudicate.
- [ ] Step 1: findings doc — per-run table, noise estimate, verdict, next-step recommendation.
- [ ] Step 2: SDD ledger updated; rotation doc status line updated.

## Review focus (what this experiment can get wrong if nobody checks)

1. **Cross-run confounds** — prefix cache / server state between the two checkpoints: unique
   salted prompts per run, cold server each time, pid verified before trusting.
2. **Attribution** — only the selected layers may differ; the byte-identity hash (Task 1
   Step 4) is the guard. lm_head/norms/MTP identical by construction.
3. **Underpowered corpus** — if the duplicate-run noise is comparable to the expected delta,
   the corpus is too small; expand before adjudicating (the rule is in Task 3, not a judgment
   call at the end).
4. **Layout mismatch** — a wrong marlin repack makes the kernel read garbage; the dequant pin
   + post-load smoke (Task 2 Step 3) are the guards.
5. **One-tensor generalization** — the 8.3 % is one tensor; the Task 0 survey is what licenses
   extending it to the patch set.
6. **Disk hygiene** — the patch copy (~14 GB) goes to `models/`-adjacent top-level naming only
   if it is clearly a sibling artifact; never inside `models/Qwen3.8-27B-W4A16-AutoRound/`;
   gitignored like the other model dirs (check `.gitignore` covers the new path).

## Non-goals

lm_head / embed_tokens / MTP (out of scope for Step 1), W4A4 activation quant (OptRot
interaction risk), full re-quant (that is Step 2), any power-limit change.

## Estimate

Task 0: a few CPU-hours (1200-tensor survey). Task 1: ~1 h. Task 2: 2–4 GPU-h (four server
boots + loads). Task 3: a couple of hours. Total: within a day.
