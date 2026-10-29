# QAD pilot session findings — ppl-path dequant bugs (2026-10-29)

## TL;DR
Phase 3 (ppl gate) is blocked: the in-process eval path produces ppl ~1e6-1e18
(baseline 6.818 through vLLM). Two 8-bit tensors in the checkpoint
(embed_tokens, lm_head) are quantized ASYMMETRIC with zero-point 0, while every
in-process consumer applied the 4-bit symmetric rule (offset 2^(nb-1)). Both
fixed and verified against the FP16 teacher. One divergence remains open.

## Verified facts (measured)

1. **Checkpoint schemes** (models/Qwen3.8-27B-W4A16-AutoRound):
   - 400 packed linears: 4-bit, group 128, symmetric (group_0, symmetric=true).
     pin_check max_abs_diff=0 on L0/L63 gate+down.
   - lm_head: 8-bit, group 128, ASYMMETRIC zp=0. Loader can't match it
     (`Could not match re:.*lm_head$`) -> loads with weight=None +
     instance-level packed forward shadow (`load_student.<locals>.fwd`).
   - embed_tokens: 8-bit, ASYMMETRIC zp=0 (inferred from packed shape 4
     codes/int32). Materialized dense by patch_ct with the WRONG symmetric
     offset.

2. **Dequant validation vs FP16 teacher** (models-fp16/Qwen3.8-27B):
   - embed unsigned: corr=1.0000, mean_abs=9.2e-5. Symmetric offset: corr=0.6547.
   - lm_head unsigned: corr=1.0000 (64 rows), std 0.01374 vs teacher 0.01375.
   - Symmetric offset on lm_head biases every row by ~-0.04 (=-125 codes x
     scale) -> logits mean -95.5 -> nll 42.5, ppl 2.9e18.

3. **Fixes applied** (this session):
   - `phase1_teacher.py patch_ct`: apply offset only for symmetric 4-bit
     schemes; 8-bit tensors dequantized unsigned.
   - `pilot.py main()`: dequantize lm_head manually (unsigned) + pop the
     instance forward shadow so the dense weight is used.
   - `pilot.py`: merge-check fix (dequant via unpack_from_int32 instead of
     `m.base.weight`), base_null zeroing, --skip-joint / --skip-chained resume
     flags, periodic adapter checkpoints (every 100 steps + best-loss).

4. **Status after fixes**: base_null ppl improved 2.9e18 -> 2.17e6 (nll 42.5 ->
   14.6). Still ~6 orders from the 6.818 baseline. Embed fix clearly helped;
   something else still differs from the vLLM serving path.

5. **Forward divergence (open)**: the training forward (manual layer_fwd with
   explicit cos/sin from pos_emb.safetensors) diverges from the model's own
   `tm(input_ids)` forward:
   - tm() path: L63 output mse vs teacher targets = 48 (close).
   - manual path: L63 mse = 3,327,856 (catastrophic); L0 identical (90.1) on
     both paths -> divergence starts after layer 0. Suspect the captured
     cos/sin or the position_ids semantics in layer_fwd.
   - **Consequence: ALL phase-2 training (joint + chained arms) ran against the
     divergent forward.** The adapters compensate for it; they are NOT valid
     quality artifacts for the real model. The L0 mse0 plateau at ~36k is
     explained by the broken embed during training.

## Next steps (pick up here)
1. Diff pathA (tm(input_ids)) vs pathB (manual layer_fwd) per layer to isolate
   the positional divergence (probe_fw.py is the starting point).
2. Fix forward_capture to use the standard tm() forward (or corrected
   positions).
3. Retrain phase 2 (~5.7 h joint + ~15 min chained at 20 s/step joint pace).
4. Evaluate phase 3 through the vLLM-served harness (rotation round's
   analysis/rotation/ppl_eval.py + bench/harness.py), NOT the in-process
   ppl_run — the in-process path still has an unexplained gap vs vLLM.
   Requires merging adapters into servable checkpoints.

## Artifacts
- probe_*: probe_ppl.py, probe_lmhead.py, probe_lmhead2-6.py, probe_embed.py,
  probe_embed2.py, probe_fw.py (analysis/qad/)
- run logs: pilot5.log (joint 1000 steps, loss 1.26M->31.4k, 20428 s),
  pilot6.log (chained s1+s2 complete), pilot7-10.log (ppl-gate attempts)
- GUI: pilot_gui_http.py, port 8765, reads a named log file arg
