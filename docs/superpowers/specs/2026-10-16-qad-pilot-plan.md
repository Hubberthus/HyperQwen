# QAD Pilot Plan — LoRA-based Quantization-Aware Distillation (2026-10-16)

**Status:** PLANNED, not executed. Saved for future application.

## Motivation
All static probes (1–7) failed to find exploitable weight structure (spatial,
temporal, sensitivity, block-VQ, low-rank). QAD inverts the problem: instead of
finding a codebook that matches frozen weights, train the weights/adapters to
fit a simpler codebook. The only path left that can go below 4 bits without the
measured quality collapse.

## Goal
Recover quantization quality loss on MLP layers by training LoRA adapters
against the BF16 teacher, validating whether "training the noise away" works
where every static probe failed.

## Hardware (RTX 3090, 24 GB, 250 W cap)
| Component | VRAM |
|---|---|
| Student (W4A16 quantized, 27B) | ~14 GB |
| LoRA adapters (r=16, 3 layer pairs) | <0.1 GB |
| Optimizer states (LoRA only) | <0.3 GB |
| Activations (batch 1 x 2048 tokens) | ~2-4 GB |
| Total | ~17-19 GB — fits, ~5 GB headroom |

The teacher (BF16, 55.6 GB) cannot sit in VRAM. Design consequence: pre-computed
targets.

## Phases

### Phase 1 — Teacher target generation (2-4 h, one-time, CPU-heavy)
Run the BF16 model (`models-fp16/Qwen3.8-27B`) once over a fixed calibration set
(256 sequences x 2048 tokens from wikitext-2, same corpus as the ppl gate).
Capture per-layer hidden states to disk as safetensors. Never touch the teacher
again during training. (Rejected alternative: serving the teacher via vLLM with
CPU offload — slower per token, competes for RAM.)

### Phase 2 — LoRA training (~1 h)
- Freeze the quantized base (`models/Qwen3.8-27B-W4A16-AutoRound`).
- Layers: same MLP pairs as all probes (gate/down at L0 and L63).
- LoRA r=16 on the target layers' projections.
- Loss: **saliency-weighted** MSE(student hidden states, teacher hidden states)
  at layer outputs — the weight is the per-group sensitivity map from Probe 2
  (`sensitivity_probe.json`), injecting the user's "what to protect" idea into
  the objective itself. Near-free addition; a weak proxy as a loss prior is low-risk.
- **Training modes (declared ablation):**
  - *Joint* (default): all adapters trained in one pass against static teacher targets.
  - *Chained* (ablation arm, tests the user's sequential-dependence idea):
    layer-ordered training — layer N's adapter trains on hidden states already
    flowing through adapted layers 0..N-1. Each adapter predicts the correction
    for its layer given the corrected upstream. Error compounding is the measured risk.
- ~500-1000 updates, batch 1-2.
- Reconstruction check: LoRA merged back into the packed format must reproduce
  trained weights exactly (same pin discipline as every probe).

### Phase 3 — Output-tied ppl gate (1-2 h, GPU)
NON-NEGOTIABLE (the rotation-round lesson): the QAD-tuned checkpoint must beat
the current baseline on the same wikitext-2 ppl harness
(baseline 6.818, noise band ~6e-6). Hidden-state MSE improvement alone is NOT
a pass — the proxy-vs-output gap is what killed rotation.

## Follow-on rounds (contingent on the base pilot passing its ppl gate)
- **QAD + structured sparsification** (the "throwaway" idea, trained): regularize
  the bottom-K% low-sensitivity groups toward zero during adapter training, so QAD
  absorbs what the discarded groups contributed. Converts Probe 3's failed
  post-hoc selection into a trained compression.
- **QAD + trainable rotation** (the codec idea, and a rotation-round revival):
  jointly train the Hadamard fold with the adapters — the rotation failed because
  the fold had no adaptation path; QAD provides it. Highest cost/risk; gate the
  same way.

## Risks
- Hidden-state MSE may not transfer to ppl. If MSE improves but ppl does not,
  that is itself a decisive finding about QAD's applicability here — stop.
- Teacher target generation is the bottleneck (55.6 GB model on a 24 GB card).
- VRAM headroom (~5 GB) is thin; batch size stays at 1-2.

## Timeline
~4-7 h total across all three phases.
