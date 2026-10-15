#!/usr/bin/env python
"""plain_opt.py — plain (unrotated) AutoRound re-quantization of the 12 linear-attn
in_proj_qkv / in_proj_z tensors.

The vLLM fork's online-transform path cannot serve input transforms for the
in_proj_qkvz fused module: it creates 4 output partitions (flat qkvz layout) while
packed_modules_mapping yields only 2 part names (in_proj_qkv, in_proj_z), so
_validate_tfm_schemes raises KeyError on partition 2. Serving those tensors H-folded
without a runtime H would be silently wrong, so instead we UN-fold: quantize the BF16
originals directly (no Hadamard) with the captured raw inputs as the error signal,
and drop them from transform_config targets. The remaining 38 tensors keep the
rotation fold + runtime transform.

Same optimizer machinery as build_patch_ar.py: auto_round 0.16.0 SignRound applied
per-layer by hand (SignSGD on rounding offsets, lr=1/iters, LinearLR 1->0, iters=200,
best-batch tracking), scales fixed at absmax/8 per (row,128-group) fp16.
Loss = || x (Q(W) - W)^T ||^2 with raw captured x (no H).
"""
import json, os, sys, time, shutil, random, zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp
import build_patch_ar as ar
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

REPO = bp.REPO
SRC = bp.SRC
ANALYSIS = bp.ANALYSIS
CAPTURES = ANALYSIS / "captures_plain"
OPT = ANALYSIS / "opt_results"

ITERS = int(os.environ.get("AR_ITERS", "200"))
BATCH_SEQS = int(os.environ.get("AR_BATCH_SEQS", "4"))
SEQ_LEN = 2048
GPUMEM = float(os.environ.get("AR_GPUMEM", "0.85"))
LAYERS = [int(x) for x in os.environ.get("AR_PLAIN_LAYERS", "0,38,45,56,58,60").split(",")]
LOG = []

def log(m):
    print(m, flush=True)
    LOG.append(str(m))

def flush():
    (ANALYSIS / "plain_opt.log").write_text("\n".join(LOG) + "\n")

def main():
    t0 = time.time()
    targets = ar.build_targets()
    plain = {}
    for L in LAYERS:
        rows = [(b, t) for b, t in targets[L] if t in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")]
        assert len(rows) == 2
        plain[L] = rows
    assert sum(len(v) for v in plain.values()) == 12

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    try:
        from vllm import TokensPrompt
    except Exception:
        from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(str(SRC))
    seqs = ar.load_calibration_ids(tok)
    llm = LLM(model=str(SRC), dtype="bfloat16", max_model_len=SEQ_LEN + 64,
              enforce_eager=True, disable_log_stats=True,
              gpu_memory_utilization=GPUMEM, enable_prefix_caching=False, seed=0)
    model = ar.find_model_object(llm)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    prompts = [TokensPrompt(prompt_token_ids=s) for s in seqs]

    for L in LAYERS:
        tl = time.time()
        mods = {k: v for k, v in ar.discover_capture_modules(model, L).items()
                if k == "linear_attn.in_proj_qkvz"}
        assert list(mods) == ["linear_attn.in_proj_qkvz"]
        cdir = CAPTURES / ("L%d" % L)
        if cdir.exists():
            shutil.rmtree(cdir)
        hooks = ar.CaptureHooks(mods, cdir)
        llm.generate(prompts, sp, use_tqdm=False)
        hooks.finish()
        n_tok = hooks.counts["linear_attn.in_proj_qkvz"]
        assert n_tok % SEQ_LEN == 0
        log("L%d: captured %d rows x 5120 (plain mode)" % (L, n_tok))

        fused = "linear_attn.in_proj_qkvz"
        xb = np.memmap(cdir / (fused + ".bin"), dtype=np.float16, mode="r",
                       shape=(n_tok, 5120))
        nblocks = n_tok // SEQ_LEN
        eval_hi = min(BATCH_SEQS, nblocks) * SEQ_LEN
        eval_idx0 = torch.arange(eval_hi)
        ldir = OPT / ("L%d" % L)
        ldir.mkdir(parents=True, exist_ok=True)
        for base, tsuf in plain[L]:
            tt = time.time()
            W = bp.load_fp16(base)
            out_d, in_d = W.shape
            G = in_d // bp.GROUP
            W64 = W.to(torch.float64).numpy()
            del W
            amax = np.abs(W64.reshape(out_d, G, bp.GROUP)).max(axis=2)
            s64 = np.where(amax > 0, amax / 8.0, 1.0)
            s16 = s64.astype(np.float16)
            s32 = s16.astype(np.float32)

            w = torch.from_numpy(W64.astype(np.float32)).cuda()
            s = torch.from_numpy(np.repeat(s32, bp.GROUP, axis=1)).cuda()
            del s32
            v = torch.zeros_like(w, requires_grad=True)
            opt = SignSGD([v], lr=1.0 / ITERS, weight_decay=0)
            sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0,
                                                      total_iters=ITERS)
            rng = random.Random(2000 + L * 100 + zlib.crc32(tsuf.encode()) % 100)

            def qdv(vv, idx):
                x = torch.from_numpy(np.asarray(xb[idx.numpy()])).cuda()
                z = w / s + vv
                r = torch.round(z)
                r_ste = r.detach() - z.detach() + z
                q = torch.clamp(r_ste, -8.0, 7.0)
                dq = q * s - w
                err = x.to(torch.bfloat16) @ dq.to(torch.bfloat16).t()
                return err.float().pow(2).mean()

            with torch.no_grad():
                loss0 = qdv(v, eval_idx0).item()
            best = (loss0, v.detach().clone())
            for it in range(ITERS):
                seq0 = rng.randrange(max(1, nblocks - BATCH_SEQS))
                idx = torch.arange(seq0 * SEQ_LEN, (seq0 + BATCH_SEQS) * SEQ_LEN)
                loss = qdv(v, idx)
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                lv = loss.item()
                if lv < best[0]:
                    best = (lv, v.detach().clone())
            with torch.no_grad():
                loss_best = qdv(best[1], eval_idx0).item()
            # final codes: CPU float64 divide by fp16-rounded scale (checkpoint convention)
            v_np = best[1].cpu().numpy().astype(np.float64).reshape(out_d, G, bp.GROUP)
            sq64 = s16.astype(np.float64)[:, :, None]
            codes = np.clip(np.round(W64.reshape(out_d, G, bp.GROUP) / sq64 + v_np),
                            -8, 7).astype(np.int64).reshape(out_d, in_d)
            codes_rtn = np.clip(np.round(W64.reshape(out_d, G, bp.GROUP) / sq64),
                                -8, 7).astype(np.int64).reshape(out_d, in_d)
            n_moved = int((codes != codes_rtn).sum())
            del W64, w, s, v, opt, sched
            torch.cuda.empty_cache()
            np.savez(ldir / (base + ".npz"), codes=codes, scale=s16,
                     shape=np.array([out_d, in_d]), loss0=loss0,
                     loss_final=None, loss_best_evalbatch=loss_best,
                     best_batch_loss=best[0], traj_head=[], traj_tail=[],
                     n_codes_moved=n_moved, plain=np.array(1))
            log("  L%d %-22s [%dx%d] loss0=%.6g best_batch=%.6g moved=%d eval %.6g -> %.6g (%+.2f%%) (%.1fs)"
                % (L, tsuf, out_d, in_d, loss0, best[0], n_moved, loss0, loss_best,
                   100 * (1 - loss_best / max(loss0, 1e-30)), time.time() - tt))
        shutil.rmtree(cdir)
        flush()
        log("L%d: plain done (%.1fs)" % (L, time.time() - tl))
    flush()
    log("PLAIN OPT DONE in %.1f min" % ((time.time() - t0) / 60))

if __name__ == "__main__":
    main()
