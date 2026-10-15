#!/usr/bin/env python
"""build_patch_ar.py — AutoRound re-quantization of the 50 rotation-folded tensors.

Builds models/Qwen3.8-27B-W4A16-rot-patch-AR/: a byte-identical copy of the
W4A16-AutoRound baseline in which the 50 selected INT4 linears are replaced by
block-128 Hadamard-folded, input-aware (SignRound-style) optimized group-128
symmetric INT4, and whose quantization_config.json gains a runtime transform_config
so vLLM applies H to activations at serving time (y = (x H) (Q(W H))^T ~= x W^T).

Rounding optimizer (STATED FOR THE RECORD):
  auto_round 0.16.0's SignRound algorithm applied per-layer by hand, using the
  package's own primitives: SignSGD (sign-gradient updates) on the rounding-offset
  parameter v, round-STE quantization Q(v) = clamp(round(WH/s + v), -8, 7) * s,
  lr = 1/iters = 0.005 (the package's 4-bit auto-lr heuristic), LinearLR 1.0 -> 0.0
  over iters=200, best-batch-MSE parameter tracking (collect_best_params analogue),
  MSELoss on || x' (Q(WH,v) - WH)^T ||^2 with x' = x @ H sampled from the captured
  calibration activations (random contiguous 2048-row blocks, 8192 rows/iter).
  Scales are FIXED at s = absmax(WH)/8 per (row, 128-group), positive, float16 —
  the build_patch.py checkpoint convention (identical information content to
  auto_round's +/-absmax/8 sign rule; keeps byte-level comparability with the
  naive-fold reference). Minmax scale tuning is disabled for the same reason.
  The end-to-end autoround pipeline was NOT used because it (a) re-quantizes all
  1200 INT4 linears from the BF16 originals (ignoring our pre-folded W H), and
  (b) emits its own packing format rather than the checkpoint's marlin layout.

Input capture: vLLM 0.31.0 fork offline LLM (in-process, enforce_eager, TP=1) of
the READ-ONLY baseline; forward-pre-hooks on the fused target modules
(in_proj_qkvz covers in_proj_qkv + in_proj_z; qkv_proj covers q/k/v; gate_up_proj
covers gate/up; out_proj/o_proj/down_proj direct). Calibration data: wikitext-2
test (bench/quality-data/wikitext), all non-empty rows tokenized and concatenated
= 296,960 tokens (the task's "~256 x 2048" is not attainable from this corpus).
Captures are stored per layer as float16 raw .bin under analysis/rotation/captures/
(~25-29 GB per layer, peak < 40 GB), consumed immediately by the optimizer for that
layer, then deleted. float16 (not float32) is used to honor the 40 GB disk cap with
the full corpus; optimizer math is float32.

Subcommands:
  run      capture + optimize per layer (resumable; skips layers with done markers)
  export   splice results into the patch copy + pins + hashes + config update
Stages are CPU/GPU-long; run with output redirected to a log.

Env knobs:  AR_ITERS (200)  AR_BATCH_SEQS (4)  AR_LAYERS (comma list subset)
            AR_LIMIT_SEQS (0 = all 145)  AR_GPUMEM (0.85)
"""
import json
import os
import sys
import time
import shutil
import zlib
import random
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp  # noqa: E402  (verified fold/pack/dequant primitives)
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD  # noqa: E402

REPO = bp.REPO
SRC = bp.SRC
FP16 = bp.FP16
ANALYSIS = bp.ANALYSIS
AR_DIR = REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-AR"
CAPTURES = ANALYSIS / "captures"
OPT = ANALYSIS / "opt_results"

ITERS = int(os.environ.get("AR_ITERS", "200"))
BATCH_SEQS = int(os.environ.get("AR_BATCH_SEQS", "4"))
SEQ_LEN = 2048
GPUMEM = float(os.environ.get("AR_GPUMEM", "0.85"))
ONLY_LAYERS = [int(x) for x in os.environ.get("AR_LAYERS", "").split(",") if x != ""]
LIMIT_SEQS = int(os.environ.get("AR_LIMIT_SEQS", "0"))
H128 = bp.sylvester(128)          # float64 [128,128], H @ H.T == I (involution)

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(str(msg))

# ------------------------------------------------------------------ target table
LIN_LAYERS = [0, 38, 45, 56, 58, 60]
FULL_LAYERS = [27, 63]
LIN_TARGETS = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
FULL_TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

# target suffix -> (capture fused module suffix, row-offset-provider)
FUSED_OF = {
    "linear_attn.in_proj_qkv": "linear_attn.in_proj_qkvz",
    "linear_attn.in_proj_z":   "linear_attn.in_proj_qkvz",
    "linear_attn.out_proj":    "linear_attn.out_proj",
    "self_attn.q_proj":        "self_attn.qkv_proj",
    "self_attn.k_proj":        "self_attn.qkv_proj",
    "self_attn.v_proj":        "self_attn.qkv_proj",
    "self_attn.o_proj":        "self_attn.o_proj",
    "mlp.gate_proj":           "mlp.gate_up_proj",
    "mlp.up_proj":             "mlp.gate_up_proj",
    "mlp.down_proj":           "mlp.down_proj",
}

def build_targets():
    """-> {layer: [(base_name, target_suffix), ...]} for the 50 tensors."""
    _, patch = bp.build_patch_set()
    out = {}
    for L in LIN_LAYERS + FULL_LAYERS:
        tgts = LIN_TARGETS if L in LIN_LAYERS else FULL_TARGETS
        rows = []
        for t in tgts:
            base = "model.language_model.layers.%d.%s" % (L, t)
            assert base in patch, "base %s not in patch set" % base
            rows.append((base, t))
        out[L] = rows
    n = sum(len(v) for v in out.values())
    assert n == 50, "expected 50 targets, got %d" % n
    return out

# ------------------------------------------------------------------ round-trip pin
def roundtrip_pin():
    pin_bases = [
        "model.language_model.layers.1.mlp.down_proj",
        "model.language_model.layers.5.mlp.down_proj",
        "model.language_model.layers.10.mlp.down_proj",
    ]
    res = {}
    ok = True
    for b in pin_bases:
        inv = json.loads(bp.INVENTORY.read_text())
        t = next(x for x in inv["tensors"] if x["name"] == b + ".weight_packed")
        packed = bp.load_src_tensor(t["file"], b + ".weight_packed")
        scale = bp.load_src_tensor(t["file"], b + ".weight_scale")
        shape = tuple(bp.load_src_tensor(t["file"], b + ".weight_shape").tolist())
        nib = bp.unpack_nibbles(packed.to(torch.int64).numpy())
        rep = bp.repack_nibbles(nib).astype(np.int32)
        ident = bool(np.array_equal(rep, packed.numpy()))
        md = bp.manual_dequant(packed, scale, shape)
        cd = bp.ct_dequant(packed, scale, shape).numpy()
        ddiff = float(np.abs(md - cd).max())
        ok &= ident and ddiff == 0.0
        res[b] = {"byte_identical": ident, "manual_vs_ct_max_diff": ddiff, "shape": shape}
        log("  roundtrip %s: byte-identical=%s manual-vs-ct=%.3e -> %s"
            % (b.rsplit(".", 2)[-2] + "." + b.rsplit(".", 1)[-1], ident, ddiff, "PASS" if ok else "FAIL"))
    return ok, res

# ------------------------------------------------------------------ capture (vLLM)
def load_calibration_ids(tok):
    import pyarrow.parquet as pq
    pq_path = next((REPO / "bench" / "quality-data" / "wikitext" / "wikitext-2-raw-v1").glob("test*.parquet"))
    tbl = pq.read_table(pq_path)
    texts = [t for t in tbl.column("text").to_pylist() if t]
    ids = []
    for t in texts:
        ids.extend(tok(t)["input_ids"])
    n = len(ids) // SEQ_LEN
    ids = ids[: n * SEQ_LEN]
    seqs = [ids[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in range(n)]
    if LIMIT_SEQS:
        seqs = seqs[:LIMIT_SEQS]
    log("calibration: %d rows -> %d seqs of %d" % (len(ids) + (len(ids) % SEQ_LEN), len(seqs), SEQ_LEN))
    return seqs

class CaptureHooks:
    """forward-pre-hooks accumulating x rows (fp16 CPU) per capture module, flushed to .bin."""
    def __init__(self, modules: dict, out_dir: Path):
        self.handles = {}
        self.counts = {}
        self.buffers = {}
        self.shapes_seen = {}
        self.modules = modules
        out_dir.mkdir(parents=True, exist_ok=True)
        for short in modules:
            self.handles[short] = open(out_dir / (short + ".bin"), "ab")
            self.counts[short] = 0
            self.buffers[short] = []
            self.shapes_seen[short] = []
        self.hooks = []
        for short, (name, mod) in modules.items():
            def make(short):
                def hook(_m, args):
                    x = args[0]
                    x2 = x.detach().reshape(-1, x.shape[-1]).to(torch.float16).cpu()
                    self.buffers[short].append(x2)
                    self.counts[short] += x2.shape[0]
                    self.shapes_seen[short].append(tuple(x.shape))
                    if sum(b.nbytes for b in self.buffers[short]) > 1_500_000_000:
                        self._flush(short)
                return hook
            self.hooks.append(mod.register_forward_pre_hook(make(short)))

    def _flush(self, short):
        if self.buffers[short]:
            self.handles[short].write(torch.cat(self.buffers[short], 0).numpy().tobytes())
            self.buffers[short] = []

    def finish(self):
        for short in self.modules:
            self._flush(short)
            self.handles[short].close()
        for h in self.hooks:
            h.remove()

def find_model_object(llm):
    """Walk llm.llm_engine's attribute graph for the top-level torch.nn.Module."""
    import torch.nn as nn
    seen, found = set(), []
    def rec(obj, depth):
        if depth > 10 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, nn.Module):
            found.append(obj)
            return  # do not descend into the module tree
        for v in list(vars(obj).values()) if hasattr(obj, "__dict__") else []:
            try:
                rec(v, depth + 1)
            except Exception:
                pass
    rec(llm.llm_engine, 0)
    for m in found:
        names = set(n for n, _ in m.named_modules())
        if len(names) > 200 and any(n.endswith("layers.0.mlp.gate_up_proj") for n in names):
            return m
    raise RuntimeError("model object not found; collected %d modules" % len(found))

def discover_capture_modules(model, layer):
    """-> {short_name: (full_name, module)} for the fused capture modules of one layer."""
    tgts = LIN_TARGETS if layer in LIN_LAYERS else FULL_TARGETS
    want = {FUSED_OF[t] for t in tgts}
    suffixes = {"layers.%d.%s" % (layer, w): w for w in want}
    resolved = {}
    for name, mod in model.named_modules():
        for s, w in suffixes.items():
            if name.endswith(s) and w not in resolved:
                resolved[w] = (name, mod)
    assert set(resolved) == want, "capture modules %s != want %s" % (sorted(resolved), sorted(want))
    return resolved

def stage_run():
    t0 = time.time()
    ok, rt = roundtrip_pin()
    if not ok:
        log("ROUND-TRIP PIN FAILED -> abort")
        (ANALYSIS / "build_patch_ar.log").write_text("\n".join(LOG) + "\n")
        sys.exit(1)
    log("round-trip pin: PASS (3/3)\n")

    targets = build_targets()
    layers = ONLY_LAYERS or sorted(targets)
    log("layers to process: %s (iters=%d batch_seqs=%d gpumem=%.2f)" % (layers, ITERS, BATCH_SEQS, GPUMEM))

    # ---- vLLM in-process load of the READ-ONLY baseline
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    try:
        from vllm import TokensPrompt
    except Exception:
        from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(str(SRC))
    seqs = load_calibration_ids(tok)
    llm = LLM(model=str(SRC), dtype="bfloat16", max_model_len=SEQ_LEN + 64,
              enforce_eager=True, disable_log_stats=True,
              gpu_memory_utilization=GPUMEM, enable_prefix_caching=False,
              seed=0)
    model = find_model_object(llm)
    log("model object: %s (%d submodules)" % (type(model).__name__, len(list(model.named_modules()))))
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    prompts = [TokensPrompt(prompt_token_ids=s) for s in seqs]

    torch.backends.cuda.matmul.allow_tf32 = True
    Ht = torch.from_numpy(H128).float().cuda()

    for L in layers:
        done_marker = OPT / ("L%d.done.json" % L)
        if done_marker.exists():
            log("L%d: already done, skipping" % L)
            continue
        tl = time.time()
        mods = discover_capture_modules(model, L)
        log("L%d: capture modules: %s" % (L, {k: v[0] for k, v in mods.items()}))
        cdir = CAPTURES / ("L%d" % L)
        if cdir.exists():
            shutil.rmtree(cdir)
        hooks = CaptureHooks(mods, cdir)
        tc = time.time()
        llm.generate(prompts, sp, use_tqdm=False)
        hooks.finish()
        counts = dict(hooks.counts)
        shapes = {k: dict(v) for k, v in hooks.shapes_seen.items()}
        log("L%d: captured in %.1fs; rows/module: %s; shapes: %s"
            % (L, time.time() - tc, counts, shapes))
        n_tok_set = set(counts.values())
        assert len(n_tok_set) == 1, "row count mismatch across capture modules: %s" % counts
        n_tok = n_tok_set.pop()
        assert n_tok % SEQ_LEN == 0, "n_tok %d not a multiple of %d" % (n_tok, SEQ_LEN)

        # ---------------- optimize every target in this layer
        OPT.mkdir(parents=True, exist_ok=True)
        ldir = OPT / ("L%d" % L)
        ldir.mkdir(parents=True, exist_ok=True)
        per = {}
        nblocks = n_tok // SEQ_LEN
        eval_hi = min(BATCH_SEQS, nblocks) * SEQ_LEN
        eval_idx0 = torch.arange(eval_hi)
        for base, tsuf in targets[L]:
            tt = time.time()
            fused = FUSED_OF[tsuf]
            W = bp.load_fp16(base)                     # BF16 [out,in] original
            out_d, in_d = W.shape
            G = in_d // bp.GROUP
            W64 = W.to(torch.float64).numpy()
            Wrot64 = bp.block_apply(W64, H128)          # float64 fold, exact
            amax = np.abs(Wrot64.reshape(out_d, G, bp.GROUP)).max(axis=2)
            s64 = np.where(amax > 0, amax / 8.0, 1.0)
            s16 = s64.astype(np.float16)                # [out, G]
            s32 = s16.astype(np.float32)
            del W, W64

            xb_bytes = (cdir / (fused + ".bin")).stat().st_size
            assert xb_bytes == n_tok * in_d * 2, \
                "capture %s: %d bytes != %d rows x %d x 2" % (fused, xb_bytes, n_tok, in_d)
            xb_full = np.memmap(cdir / (fused + ".bin"), dtype=np.float16, mode="r",
                                shape=(n_tok, in_d))

            w = torch.from_numpy(Wrot64.astype(np.float32)).cuda()
            s = torch.from_numpy(np.repeat(s32, bp.GROUP, axis=1)).cuda()   # [out,in]
            v = torch.zeros_like(w, requires_grad=True)
            opt = SignSGD([v], lr=1.0 / ITERS, weight_decay=0)
            sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0,
                                                      total_iters=ITERS)
            rng = random.Random(1000 + L * 100 + zlib.crc32(tsuf.encode()) % 100)

            def qdv(vv, idx):
                x = torch.from_numpy(np.asarray(xb_full[idx.numpy()])).cuda()   # [B,in] fp16
                xb = x.view(x.shape[0], G, bp.GROUP).float() @ Ht               # x' = x @ H
                del x
                z = w / s + vv
                r = torch.round(z)
                r_ste = r.detach() - z.detach() + z                        # STE through round
                q = torch.clamp(r_ste, -8.0, 7.0)
                dq = q * s - w
                err = torch.einsum("bgk,ogk->bo",
                                   xb.to(torch.bfloat16),
                                   dq.to(torch.bfloat16).view(out_d, G, bp.GROUP))
                loss = err.float().pow(2).mean()
                return loss

            with torch.no_grad():
                loss0 = qdv(v, eval_idx0).item()
            best = (loss0, v.detach().clone())
            losses = [loss0]
            for it in range(ITERS):
                seq0 = rng.randrange(max(1, nblocks - BATCH_SEQS))
                idx = torch.arange(seq0 * SEQ_LEN, (seq0 + BATCH_SEQS) * SEQ_LEN)
                loss = qdv(v, idx)
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                lv = loss.item()
                losses.append(lv)
                if lv < best[0]:
                    best = (lv, v.detach().clone())
            with torch.no_grad():
                loss_final = qdv(v, eval_idx0).item()
                loss_best = qdv(best[1], eval_idx0).item()
            # Final codes computed on CPU in float64, exactly matching
            # quantize_rotated's convention (f64 divide by fp16-rounded scale),
            # so v=0 reproduces the naive-fold codes bit-for-bit.
            v_np = best[1].cpu().numpy().astype(np.float64).reshape(out_d, G, bp.GROUP)
            sq64 = s16.astype(np.float64)[:, :, None]
            codes = np.clip(np.round(Wrot64.reshape(out_d, G, bp.GROUP) / sq64 + v_np),
                            -8, 7).astype(np.int64).reshape(out_d, in_d)
            codes_rtn = np.clip(np.round(Wrot64.reshape(out_d, G, bp.GROUP) / sq64),
                                -8, 7).astype(np.int64).reshape(out_d, in_d)
            n_moved = int((codes != codes_rtn).sum())
            assert (n_moved == 0) == np.array_equal(codes, codes_rtn)
            del Wrot64, w, s, v, opt, sched
            torch.cuda.empty_cache()
            np.savez(ldir / (base + ".npz"),
                     codes=codes, scale=s16, shape=np.array([out_d, in_d]),
                     loss0=loss0, loss_final=loss_final, loss_best_evalbatch=loss_best,
                     best_batch_loss=best[0], traj_head=losses[:5], traj_tail=losses[-5:],
                     n_codes_moved=n_moved)
            per[base] = {"loss0": loss0, "loss_final": loss_final,
                         "loss_best_evalbatch": loss_best, "best_batch_loss": best[0],
                         "improve_pct": round(100 * (1 - loss_best / max(loss0, 1e-30)), 2)}
            log("  L%d %-30s [%dx%d] loss0=%.6g best_batch=%.6g moved=%d | eval %.6g -> %.6g (%+.2f%%) (%.1fs)"
                % (L, tsuf, out_d, in_d, loss0, best[0], n_moved, loss0, loss_best,
                   per[base]["improve_pct"], time.time() - tt))
        shutil.rmtree(cdir)                      # delete captures after use (40 GB cap)
        (OPT / ("L%d.meta.json" % L)).write_text(json.dumps(
            {"layer": L, "n_tokens": n_tok, "iters": ITERS, "batch_seqs": BATCH_SEQS,
             "per": per}, indent=1))
        done_marker.write_text(json.dumps({"layer": L, "tensors": len(per),
                                           "wall_s": round(time.time() - tl, 1)}))
        (ANALYSIS / "build_patch_ar.log").write_text("\n".join(LOG) + "\n")
        log("L%d: done (%d tensors, %.1fs)\n" % (L, len(per), time.time() - tl))

    (ANALYSIS / "build_patch_ar.log").write_text("\n".join(LOG) + "\n")
    log("ALL LAYERS DONE in %.1f min" % ((time.time() - t0) / 60))

# ------------------------------------------------------------------ export
def write_st(path, key, arr, hlen, header):
    off = bp.st_abs_offset(hlen, header, key)
    b = np.ascontiguousarray(arr).tobytes()
    with open(path, "r+b") as f:
        f.seek(off)
        n = f.write(b)
    assert n == len(b), "short write %s" % key

def stage_export():
    t0 = time.time()
    assert OPT.exists(), "no opt_results — run stage first"
    targets = build_targets()
    # gather all results
    res = {}
    for L, rows in targets.items():
        ldir = OPT / ("L%d" % L)
        for base, _ in rows:
            d = np.load(ldir / (base + ".npz"))
            res[base] = {"codes": d["codes"], "scale": d["scale"],
                         "shape": tuple(int(x) for x in d["shape"])}
    assert len(res) == 50, "expected 50 results, got %d" % len(res)
    log("loaded %d optimization results" % len(res))

    # ---- copy baseline -> AR dir
    if AR_DIR.exists():
        shutil.rmtree(AR_DIR)
    AR_DIR.mkdir(parents=True)
    src_files = sorted(p for p in SRC.iterdir() if p.is_file())
    for p in src_files:
        shutil.copy2(p, AR_DIR / p.name)
    src_total = sum(p.stat().st_size for p in src_files)
    dst_total = sum((AR_DIR / p.name).stat().st_size for p in src_files)
    assert src_total == dst_total, "copy size mismatch"
    log("copied %d files, %d bytes" % (len(src_files), src_total))

    # ---- splice: in-place byte overwrite of weight_packed + weight_scale
    inv = json.loads(bp.INVENTORY.read_text())
    by_shard = {}
    for base, d in res.items():
        t = next(x for x in inv["tensors"] if x["name"] == base + ".weight_packed")
        by_shard.setdefault(t["file"], []).append(base)
    hdr = {}
    for shard in by_shard:
        hlen, h = bp.read_st_header(AR_DIR / shard)
        hdr[shard] = (hlen, h)
    for base, d in sorted(res.items()):
        t = next(x for x in inv["tensors"] if x["name"] == base + ".weight_packed")
        shard = t["file"]
        hlen, h = hdr[shard]
        shape = tuple(bp.load_src_tensor(shard, base + ".weight_shape").tolist())
        assert shape == d["shape"], "%s shape %s vs opt %s" % (base, shape, d["shape"])
        out_d, in_d = shape
        codes = d["codes"]
        assert codes.shape == (out_d, in_d) and codes.min() >= -8 and codes.max() <= 7
        nib = (codes.astype(np.int64) + 8)
        packed = bp.repack_nibbles(nib).astype(np.int32)
        assert packed.shape == (out_d, in_d // 8)
        s16 = d["scale"]
        assert s16.shape == (out_d, in_d // bp.GROUP) and s16.dtype == np.float16
        write_st(AR_DIR / shard, base + ".weight_packed", packed, hlen, h)
        write_st(AR_DIR / shard, base + ".weight_scale", s16, hlen, h)
    log("spliced %d tensors (packed+scale) in %d shards" % (len(res), len(by_shard)))

    # ---- config: add transform_config (quantization_config.json + config.json inline)
    # Targets are EXACT vLLM module part names (is_match is exact-string on full paths;
    # vLLM module paths are language_model.model.layers.N.<...>, and fused modules
    # resolve to their part names via packed_modules_mapping). The 12 linear-attn
    # in_proj_qkv / in_proj_z tensors are quantized PLAIN (unrotated) and are NOT in
    # transform targets: the fused in_proj_qkvz module has 4 output partitions but the
    # fork's packed_modules_mapping yields only 2 part names, so _validate_tfm_schemes
    # cannot cover all partitions (KeyError on partition 2).
    PLAIN_TSUF = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")
    base_names = []
    for L, rows in targets.items():
        for base, tsuf in rows:
            if tsuf in PLAIN_TSUF:
                continue
            base_names.append("language_model.model.layers.%d.%s" % (L, tsuf))
    assert len(base_names) == 38, "expected 38 transform targets, got %d" % len(base_names)
    tcfg = {"config_groups": {"rot": {"type": "hadamard", "head_dim": 128,
            "apply": [{"targets": base_names, "location": "input", "inverse": False}]}}}
    qc_path = AR_DIR / "quantization_config.json"
    qc = json.loads(qc_path.read_text())
    assert "transform_config" not in qc
    qc["transform_config"] = tcfg
    qc_path.write_text(json.dumps(qc, indent=2))
    cj_path = AR_DIR / "config.json"
    cj = json.loads(cj_path.read_text())
    assert "transform_config" not in cj["quantization_config"]
    cj["quantization_config"]["transform_config"] = tcfg
    cj_path.write_text(json.dumps(cj, indent=2))
    log("transform_config added (50 targets) to quantization_config.json + config.json")

    # ---- dequant pin on all 50 (read back from AR dir)
    all_zero = True
    deq = {}
    for base, d in sorted(res.items()):
        t = next(x for x in inv["tensors"] if x["name"] == base + ".weight_packed")
        shard = t["file"]
        with bp.safe_open(AR_DIR / shard, framework="pt", device="cpu") as f:
            pk = f.get_tensor(base + ".weight_packed")
            sc = f.get_tensor(base + ".weight_scale")
        shape = d["shape"]
        md = bp.manual_dequant(pk, sc, shape)
        cd = bp.ct_dequant(pk, sc, shape).numpy()
        diff = float(np.abs(md - cd).max())
        all_zero &= diff == 0.0
        deq[base] = diff
    log("dequant pin: %s (max diff over 50 = %.3e)" %
        ("PASS" if all_zero else "FAIL", max(deq.values())))
    assert all_zero, "dequant pin failed"

    # ---- byte identity of unpatched files
    patched_shards = set(by_shard)
    patched_files = patched_shards | {"quantization_config.json", "config.json"}
    mism = []
    for p in src_files:
        if p.name in patched_files:
            continue
        if bp.sha256(SRC / p.name) != bp.sha256(AR_DIR / p.name):
            mism.append(p.name)
    assert not mism, "byte-identity mismatches: %s" % mism
    log("byte-identity: all %d unpatched files identical (patched: %s)"
        % (len(src_files) - len(patched_files), sorted(patched_files)))

    # ---- total size (per-file accounting: only the 2 edited config JSONs may grow)
    final_total = sum(p.stat().st_size for p in AR_DIR.iterdir() if p.is_file())
    size_ok, size_detail = True, {}
    for p in src_files:
        s0, s1 = p.stat().st_size, (AR_DIR / p.name).stat().st_size
        if p.name in patched_files:
            # 4 patched shards: in-place overwrite -> byte-length MUST be unchanged;
            # 2 config JSONs: expected to grow (transform_config added)
            if p.name.endswith(".safetensors"):
                size_ok &= (s0 == s1)
                size_detail[p.name] = s1
            else:
                size_ok &= (s1 > s0)
                size_detail[p.name] = "%d -> %d (+%d)" % (s0, s1, s1 - s0)
        else:
            size_ok &= (s0 == s1)
    cfg_growth = sum((AR_DIR / n).stat().st_size - (SRC / n).stat().st_size
                     for n in ("quantization_config.json", "config.json"))
    expected = src_total + cfg_growth
    log("total size: %d (expected baseline %d + config growth %d = %d) -> %s"
        % (final_total, src_total, cfg_growth, expected,
           "MATCH" if final_total == expected else "MISMATCH"))
    log("per-file sizes: %s" % size_detail)
    assert size_ok, "unexpected per-file size change"
    assert final_total == expected, "size mismatch"

    (ANALYSIS / "export_ar.json").write_text(json.dumps({
        "patch_dir": str(AR_DIR), "n_tensors": 50, "total_size": final_total,
        "expected_size": expected, "dequant_pin_all_zero": all_zero,
        "dequant_max_diff": max(deq.values()),
        "byte_identity_mismatches": mism, "patched_files": sorted(patched_files),
        "transform_targets": base_names, "runtime_s": round(time.time() - t0, 1)}, indent=1))
    log("export DONE in %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        stage_run()
    elif cmd == "export":
        stage_export()
    else:
        print("usage: build_patch_ar.py run|export")
        sys.exit(2)
