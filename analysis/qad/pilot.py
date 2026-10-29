#!/usr/bin/env python
"""QAD pilot phase 2+3: LoRA r16 on 4 linears + PPL gate.

Phase 1 (teacher targets) is complete and verified (analysis/qad/teacher_targets/).
This script implements:
  - Gate (a): pin check manual-unpack vs compressed_tensors == 0 (target linears).
  - Phase 2: LoRA r16 alpha32 on L0/L63 gate+down, saliency-weighted MSE loss
    at layer outputs, joint arm and chained arm, ~1000 updates, batch 1-2.
  - Gate (b): LoRA merge reconstruction reproduces trained weights exactly.
  - Phase 3: output-tied PPL gate on wikitext-2 (2048-token windows, 256 stride,
    identical token budget across arms, 2 runs per arm, base null FIRST).
"""
import json, math, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, "/home/hubbe/qwen-serving/analysis/qad")
import phase1_teacher as P1

REPO = Path("/home/hubbe/qwen-serving")
ADIR = REPO / "analysis" / "qad"
TT = ADIR / "teacher_targets"
MODEL = REPO / "models/Qwen3.8-27B-W4A16-AutoRound"
CORPUS = REPO / "bench/quality-data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
PROBE = REPO / "analysis/salted/sensitivity_probe.json"

N_LAYERS = 64
HID = 5120
TRAIN_W = 2048
TRAIN_BATCH = 2
UPDATES = 1000
LR = 1e-4
LORA_R = 16
LORA_ALPHA = 32.0
SEED = 42
TARGET_LAYERS = [0, 63]
PHASE1_BATCH = 16
# PPL params (must match analysis/rotation/ppl_eval.py)
PPL_W, PPL_S = 2048, 256


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- gate (a)
def pin_check(full):
    """Re-verify manual unpack == compressed_tensors reconstruction == 0."""
    tm = full.model.language_model
    results = {}
    for L in TARGET_LAYERS:
        for pname in ["gate_proj", "down_proj"]:
            mod = getattr(tm.layers[L].mlp, pname)
            # manual unpack vs compressed_tensors unpack_from_int32 — exact code
            # equality, same pin discipline as analysis/salted/cond_entropy_probe.py
            # (the old body double-dereferenced weight_packed, assumed 2 codes per
            # int32 (these are INT4: 8 per int32), and referenced mod.weight which
            # does not exist on wrapped Linears)
            sch = mod.quantization_scheme
            num_bits = int(sch.weights.num_bits)
            gs = int(sch.weights.group_size)
            shape = tuple(mod.weight_shape.tolist())
            n_out, n_in = shape
            packed = mod.weight_packed.data
            factor = 32 // num_bits
            assert n_in % factor == 0, (n_in, factor)
            p = packed.cpu().to(torch.int64).numpy()
            manual = np.empty((n_out, n_in), dtype=np.int64)
            mask = (1 << num_bits) - 1
            for i in range(factor):
                manual[:, i::factor] = (p >> (num_bits * i)) & mask
            ours_signed = manual - 2 ** (num_bits - 1)
            from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
            ref = unpack_from_int32(packed, num_bits, shape)
            diff = (torch.from_numpy(ours_signed).to(packed.device, torch.int64)
                    - ref.to(torch.int64)).abs().max().item()
            results[f"L{L}.{pname}"] = {"max_abs_diff": diff,
                                        "codes_checked": int(ref.numel())}
            assert diff == 0, f"pin_check failed at L{L}.{pname}"
            log(f"pin_check L{L}.{pname}: max_abs_diff={diff}")
    return results


# ---------------------------------------------------------------- LoRA
class LoRALinear(torch.nn.Module):
    def __init__(self, base, r=16, alpha=32.0):
        super().__init__()
        self.base = base
        self.in_f = int(base.weight_shape[1])
        self.out_f = int(base.weight_shape[0])
        self.A = torch.nn.Parameter(torch.empty(r, self.in_f, device="cuda", dtype=torch.bfloat16))
        self.B = torch.nn.Parameter(torch.zeros(self.out_f, r, device="cuda", dtype=torch.bfloat16))
        torch.nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def wrap_lora(tm, layers=TARGET_LAYERS):
    handles = {}
    for L in layers:
        mlp = tm.layers[L].mlp
        for pname in ["gate_proj", "down_proj"]:
            orig = getattr(mlp, pname)
            lora = LoRALinear(orig)
            setattr(mlp, pname, lora)
            handles[f"L{L}.{pname}"] = lora
    log("wrapped:", list(handles.keys()))
    return handles


def set_trainable(handles, keys):
    for k, m in handles.items():
        req = k in keys
        m.A.requires_grad_(req)
        m.B.requires_grad_(req)


def save_adapters(handles, outdir, tag):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sd = {}
    for k, m in handles.items():
        sd[f"{k}.A"] = m.A.detach().cpu().to(torch.bfloat16)
        sd[f"{k}.B"] = m.B.detach().cpu().to(torch.bfloat16)
    save_file(sd, str(outdir / f"{tag}.safetensors"))
    log(f"saved adapters -> {outdir}/{tag}.safetensors")


def load_adapters(handles, outdir, tag):
    sd = load_file(str(Path(outdir) / f"{tag}.safetensors"))
    for k, m in handles.items():
        m.A.data.copy_(sd[f"{k}.A"].to("cuda", torch.bfloat16))
        m.B.data.copy_(sd[f"{k}.B"].to("cuda", torch.bfloat16))
    log(f"loaded adapters <- {outdir}/{tag}.safetensors")


def merge_effective_weight(handle):
    """W_eff = W_base_dequant + (B @ A) * scaling  (bf16)."""
    base_w = handle.base.weight_dequant_cached() if hasattr(handle.base, "weight_dequant_cached") else None
    if base_w is None:
        # reconstruct from PackedLinearFn logic
        base_w = handle.base.weight_fn(torch.zeros(1)) if False else None
    if base_w is None:
        raise RuntimeError("no dequant path on base module")
    A = handle.A.detach()
    B = handle.B.detach()
    dW = (B @ A) * handle.scaling
    return base_w + dW


# ---------------------------------------------------------------- sensitivity
def sens_weights():
    d = json.load(open(PROBE))
    t = d["tensors"]
    raw = {}
    for L in TARGET_LAYERS:
        rhos = []
        for pname in ["gate_proj", "down_proj"]:
            k = f"model.language_model.layers.{L}.mlp.{pname}"
            rhos.append(abs(t[k]["spearman_sens_vs_scale"]["rho"]))
        raw[L] = float(np.mean(rhos))
    mx = max(raw.values())
    norm = {L: raw[L] / mx for L in raw}
    log(f"sens weights raw={raw} norm={norm}")
    return norm, raw


# ---------------------------------------------------------------- forward
def layer_fwd(layer, h, cos, sin, pos2):
    out = layer(h, position_embeddings=(cos, sin), attention_mask=None,
                position_ids=pos2, past_key_values=None, use_cache=False)
    if isinstance(out, tuple):
        out = out[0]
    return out


def forward_capture(tm, ids, cos, sin, pos2, capture={0, 63}):
    """Full forward; checkpoint non-target layers to fit VRAM."""
    h = tm.embed_tokens(ids)
    outs = {}
    for i, layer in enumerate(tm.layers):
        if i in capture:
            h = layer_fwd(layer, h, cos, sin, pos2)
            outs[i] = h
        else:
            h = torch.utils.checkpoint.checkpoint(layer_fwd, layer, h, cos, sin, pos2,
                                                  use_reentrant=False)
    return outs


# ---------------------------------------------------------------- data
def load_calib():
    sd = load_file(str(ADIR / "calib_ids.safetensors"))
    ids = sd["ids"]  # [256, 2048]
    assert ids.shape == (256, TRAIN_W), ids.shape
    log(f"calib ids: {tuple(ids.shape)} {ids.dtype}")
    return ids


def load_teacher_targets():
    """teacher h at L0 and L63 -> [256, 2048, 5120] bf16 on CPU."""
    out = {}
    for L in TARGET_LAYERS:
        chunks = []
        for b in range(256 // PHASE1_BATCH):
            f = TT / f"L{L:02d}_b{b:02d}.safetensors"
            sd = load_file(str(f))
            chunks.append(sd["h"])
        out[L] = torch.cat(chunks, dim=0)  # [256, 2048, 5120]
        log(f"teacher targets L{L}: {tuple(out[L].shape)} {out[L].dtype}")
    return out


# ---------------------------------------------------------------- training
def make_pos():
    sd = load_file(str(ADIR / "pos_emb.safetensors"))
    return sd["cos"].cuda(), sd["sin"].cuda()


def pos2_ids(bs, W):
    return torch.arange(W, device="cuda").unsqueeze(0).expand(bs, W)


def opt_for(handles, keys, lr):
    ps = []
    for k, m in handles.items():
        if k in keys:
            ps += [m.A, m.B]
    return torch.optim.AdamW(ps, lr=lr)


def slice_batch(idx):
    return torch.tensor(idx, dtype=torch.long)


def train_joint(tm, ids_all, targets, handles, wts, cos, sin, updates=UPDATES, lr=LR):
    set_trainable(handles, set(handles.keys()))
    opt = opt_for(handles, set(handles.keys()), lr)
    n = ids_all.shape[0]
    rng = np.random.RandomState(SEED)
    order = np.arange(n)
    losses, rec = [], []
    mse_before = {}
    best_loss = float("inf")
    t0 = time.time()
    step = 0
    while step < updates:
        rng.shuffle(order)
        for s0 in range(0, n, TRAIN_BATCH):
            if step >= updates:
                break
            seqs = order[s0:s0 + TRAIN_BATCH]
            ids = ids_all[seqs].cuda()
            outs = forward_capture(tm, ids, cos, sin, pos2_ids(ids.shape[0], TRAIN_W))
            mse0 = F.mse_loss(outs[0].float(), targets[0][seqs].cuda().float())
            mse63 = F.mse_loss(outs[63].float(), targets[63][seqs].cuda().float())
            loss = wts[0] * mse0 + wts[63] * mse63
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            rec.append({"step": step, "loss": loss.item(),
                        "mse0": mse0.item(), "mse63": mse63.item()})
            if step == 0:
                mse_before = {"mse0": mse0.item(), "mse63": mse63.item()}
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_adapters(handles, ADIR / "adapters_joint_ckpt", "joint_best")
            step += 1
            if step % 50 == 0:
                r = rec[-50:]
                log(f"joint step {step} loss {np.mean([x['loss'] for x in r]):.6f} "
                    f"mse0 {np.mean([x['mse0'] for x in r]):.6f} "
                    f"mse63 {np.mean([x['mse63'] for x in r]):.6f} "
                    f"({time.time() - t0:.0f}s)")
            if step % 100 == 0:
                save_adapters(handles, ADIR / "adapters_joint_ckpt", f"joint_step{step}")
    return {"losses": losses, "record": rec, "mse_before": mse_before}


def train_chained(tm, ids_all, targets, handles, wts, cos, sin, updates=UPDATES, lr=LR):
    rec = {"stage1": [], "stage2": []}
    mse_before = {}

    # ---- stage 1: train L0 only (loss only needs layer 0)
    set_trainable(handles, {"L0.gate_proj", "L0.down_proj"})
    opt = opt_for(handles, {"L0.gate_proj", "L0.down_proj"}, lr)
    layer0 = tm.layers[0]
    n = ids_all.shape[0]
    rng = np.random.RandomState(SEED)
    order = np.arange(n)
    step = 0
    best_loss = float("inf")
    t0 = time.time()
    while step < updates:
        rng.shuffle(order)
        for s0 in range(0, n, TRAIN_BATCH):
            if step >= updates:
                break
            seqs = order[s0:s0 + TRAIN_BATCH]
            ids = ids_all[seqs].cuda()
            h = tm.embed_tokens(ids)
            h = layer_fwd(layer0, h, cos, sin, pos2_ids(ids.shape[0], TRAIN_W))
            t0_t = targets[0][seqs].cuda()
            mse0 = F.mse_loss(h.float(), t0_t.float())
            loss = wts[0] * mse0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            rec["stage1"].append({"step": step, "loss": loss.item(), "mse0": mse0.item()})
            if step == 0:
                mse_before["stage1_mse0"] = mse0.item()
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_adapters(handles, ADIR / "adapters_chained_ckpt", "s1_L0_best")
            step += 1
            if step % 100 == 0:
                r = rec["stage1"][-100:]
                log(f"chained s1 step {step} mse0 {np.mean([x['mse0'] for x in r]):.6f} ({time.time()-t0:.0f}s)")
                save_adapters(handles, ADIR / "adapters_chained_ckpt", f"s1_L0_step{step}")
    save_adapters(handles, ADIR / "adapters_chained", "stage1_L0")

    # ---- stage 2: precompute input to L63 (layers 0..62, L0 frozen+active, no_grad)
    log("chained stage 2: precomputing input to L63 (layers 0..62, no_grad)...")
    set_trainable(handles, {"L63.gate_proj", "L63.down_proj"})
    h63_in = torch.empty(256, TRAIN_W, HID, dtype=torch.bfloat16, pin_memory=True)
    with torch.no_grad():
        for s0 in range(0, 256, PHASE1_BATCH):
            ids = ids_all[s0:s0 + PHASE1_BATCH].cuda()
            h = tm.embed_tokens(ids)
            for i in range(63):  # layers 0..62
                h = layer_fwd(tm.layers[i], h, cos, sin, pos2_ids(ids.shape[0], TRAIN_W))
            h63_in[s0:s0 + PHASE1_BATCH] = h.cpu()
            log(f"  precompute batch {s0 // PHASE1_BATCH + 1}/16")
    layer63 = tm.layers[63]
    opt = opt_for(handles, {"L63.gate_proj", "L63.down_proj"}, lr)
    step = 0
    rng2 = np.random.RandomState(SEED + 1)
    order = np.arange(256)
    best_loss = float("inf")
    t0 = time.time()
    while step < updates:
        rng2.shuffle(order)
        for s0 in range(0, 256, TRAIN_BATCH):
            if step >= updates:
                break
            seqs = order[s0:s0 + TRAIN_BATCH]
            h_in = h63_in[seqs].cuda(non_blocking=True)
            h = layer_fwd(layer63, h_in, cos, sin, pos2_ids(h_in.shape[0], TRAIN_W))
            t63_t = targets[63][seqs].cuda()
            mse63 = F.mse_loss(h.float(), t63_t.float())
            loss = wts[63] * mse63
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            rec["stage2"].append({"step": step, "loss": loss.item(), "mse63": mse63.item()})
            if step == 0:
                mse_before["stage2_mse63"] = mse63.item()
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_adapters(handles, ADIR / "adapters_chained_ckpt", "s2_L63_best")
            step += 1
            if step % 100 == 0:
                r = rec["stage2"][-100:]
                log(f"chained s2 step {step} mse63 {np.mean([x['mse63'] for x in r]):.6f} ({time.time()-t0:.0f}s)")
                save_adapters(handles, ADIR / "adapters_chained_ckpt", f"s2_L63_step{step}")
    save_adapters(handles, ADIR / "adapters_chained", "stage2_L63")
    return {"record": rec, "mse_before": mse_before}


# ---------------------------------------------------------------- PPL
def build_windows():
    txt = "".join(pq.read_table(str(CORPUS)).column("text").to_pylist())
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    starts = list(range(0, len(ids) - PPL_W + 1, PPL_S))
    return tok, ids, starts


def ppl_run(full, tok, ids, starts, tag, batch=2):
    """Sum NLL over every predicted token across all windows; return (ppl, meta)."""
    tm = full.model.language_model
    lm_head = None
    for nm, m in full.named_modules():
        if nm.endswith("lm_head"):
            lm_head = m
            break
    if lm_head is None:
        raise RuntimeError("lm_head not found")
    total_nll = 0.0
    total_tok = 0
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, len(starts), batch):
            bs = starts[i0:i0 + batch]
            inp = torch.tensor([ids[s:s + PPL_W] for s in bs], device="cuda")
            h = tm(input_ids=inp, use_cache=False)
            h = h[0] if isinstance(h, tuple) else h.last_hidden_state
            logits = lm_head(h)[:, :-1]  # predict token t+1
            tgt = inp[:, 1:]
            lp = torch.log_softmax(logits.float(), dim=-1)
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            total_nll += nll.sum().item()
            total_tok += tgt.numel()
            if (i0 // batch) % 100 == 0:
                log(f"  ppl[{tag}] {i0}/{len(starts)} windows "
                    f"running_ppl={math.exp(total_nll / max(total_tok,1)):.6f} ({time.time()-t0:.0f}s)")
    ppl = math.exp(total_nll / total_tok)
    log(f"ppl[{tag}] FINAL={ppl:.6f} n={total_tok} wall={time.time()-t0:.0f}s")
    return {"ppl": ppl, "sum_nll": total_nll, "n_tokens": total_tok,
            "wall_s": round(time.time() - t0, 1)}


def main():
    torch.manual_seed(SEED)
    t_all = time.time()
    # load student
    log("loading student...")
    P1.patch_ct()  # packed-Embedding guard for compressed_tensors loader
    full = P1.load_student()
    tm = full.model.language_model
    # lm_head loads with weight=None (the loader can't match its packed form)
    # and its 8-bit scheme is ASYMMETRIC with zero-point 0 — unlike the 4-bit
    # MLP linears (symmetric, offset 2^(nb-1), verified by pin_check). Applying
    # the symmetric offset to lm_head biases every row by ~-0.04 and wrecks the
    # logits (measured: nll 42.5, ppl 2.9e18; unsigned dequant matches the FP16
    # teacher with corr=1.0000). Dequantize it manually, no offset.
    lh = dict(full.named_modules())["lm_head"]
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    _sch = lh.quantization_scheme
    _nb, _gs = _sch.weights.num_bits, _sch.weights.group_size
    _rows, _cols = (int(x) for x in lh.weight_shape.tolist())
    _codes = unpack_from_int32(lh.weight_packed, _nb, (_rows, _cols), packed_dim=1)
    _groups = _cols // _gs
    _w = (_codes.to(torch.float32).view(_rows, _groups, _gs)
          * lh.weight_scale.to(torch.float32).view(_rows, _groups, 1)
          ).reshape(_rows, _cols).to(torch.bfloat16)
    lh.weight = torch.nn.Parameter(_w.to("cuda"), requires_grad=False)
    # load_student binds an instance-level packed forward ('fwd' closure) onto
    # lm_head — it shadows the class forward and ignores .weight. Remove it so
    # the dense weight is actually used.
    lh.__dict__.pop("forward", None)
    log(f"lm_head dequantized (asymmetric zp=0): shape={tuple(_w.shape)} std={_w.std():.5f}")
    cos, sin = make_pos()

    # ---- gate (a)
    pin = pin_check(full)
    pin_ok = all(v["max_abs_diff"] == 0 for v in pin.values())

    # ---- sensitivity weights
    wts, raw_wts = sens_weights()

    # ---- phase 2: joint arm
    ids_all = load_calib()
    targets = load_teacher_targets()
    handles = wrap_lora(tm)
    if "--skip-joint" in sys.argv:
        # resume: joint arm already trained in the interrupted run; load its
        # adapters and recover the training record from the old log.
        joint_rec = {"mse_before": {}, "record": []}
        for ln in open(ADIR / "pilot5.log", errors="replace"):
            m = re.search(r"joint step (\d+) loss ([\d.]+) mse0 ([\d.]+) mse63 ([\d.]+)", ln)
            if m:
                joint_rec["record"].append({"step": int(m[1]), "loss": float(m[2]),
                                            "mse0": float(m[3]), "mse63": float(m[4])})
        load_adapters(handles, ADIR / "adapters_joint", "joint")
        log(f"resume: loaded joint adapters; recovered {len(joint_rec['record'])} joint record rows from pilot5.log")
    else:
        t0 = time.time()
        joint_rec = train_joint(tm, ids_all, targets, handles, wts, cos, sin)
        save_adapters(handles, ADIR / "adapters_joint", "joint")
        log(f"joint arm done in {time.time()-t0:.0f}s")

    # gate (b) check: merged forward must reproduce the wrapped module's
    merge_check = {}
    # output. W_eff = W_base_dequant + (B @ A) * scaling.
    for k, m in handles.items():
        A, B = m.A.detach(), m.B.detach()
        dW = (B @ A) * m.scaling
        x = torch.randn(4, 64, m.in_f, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            y1 = m(x)
            # W_base_dequant: use the module's own dequant path
            # dequantize the packed base weights (same path as PackedLinearFn.forward);
            # the module carries no plain `.weight`, so materialize from the pack.
            from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
            base = m.base
            sch = base.quantization_scheme
            nb, gs = sch.weights.num_bits, sch.weights.group_size
            rows, cols = (int(v) for v in base.weight_shape.tolist())
            codes = unpack_from_int32(base.weight_packed, nb, (rows, cols), packed_dim=1)
            off = 1 << (nb - 1)
            groups = cols // gs
            bw = ((codes.to(torch.float32) - off).view(rows, groups, gs)
                  * base.weight_scale.to(torch.float32).view(rows, groups, 1)
                  ).reshape(rows, cols).to(torch.bfloat16)
            Weff = (bw.float() + dW.float())
            y2 = x.float() @ Weff.transpose(-1, -2)
            diff = (y1.float() - y2).abs().max().item()
            rel = diff / y2.abs().max().item()
        merge_check[k] = {"max_abs_diff": diff, "max_rel_diff": rel}
        log(f"merge_check {k}: max_abs={diff:.4e} rel={rel:.4e}")

    # ---- phase 2: chained arm
    if "--skip-chained" in sys.argv:
        # resume: chained arm already trained in the interrupted run; load its
        # adapters (stage1 L0 + stage2 L63) and recover the record from pilot6.log.
        chained_rec = {"record": {"stage1": [], "stage2": []}, "mse_before": {}}
        for ln in open(ADIR / "pilot6.log", errors="replace"):
            m = re.search(r"chained s1 step (\d+) mse0 ([\d.]+)", ln)
            if m:
                chained_rec["record"]["stage1"].append({"step": int(m[1]), "mse0": float(m[2])})
            m = re.search(r"chained s2 step (\d+) mse63 ([\d.]+)", ln)
            if m:
                chained_rec["record"]["stage2"].append({"step": int(m[1]), "mse63": float(m[2])})
        load_adapters(handles, ADIR / "adapters_chained", "stage1_L0")
        s2 = load_file(str(ADIR / "adapters_chained" / "stage2_L63.safetensors"))
        for k, m in handles.items():
            if k.startswith("L63"):
                m.A.data.copy_(s2[f"{k}.A"].cuda().to(torch.bfloat16))
                m.B.data.copy_(s2[f"{k}.B"].cuda().to(torch.bfloat16))
        log(f"resume: loaded chained adapters; recovered s1={len(chained_rec['record']['stage1'])} s2={len(chained_rec['record']['stage2'])} rows from pilot6.log")
    else:
        t0 = time.time()
        chained_rec = train_chained(tm, ids_all, targets, handles, wts, cos, sin)
        log(f"chained arm done in {time.time()-t0:.0f}s")

    # ---- phase 3: PPL gate
    tok, corpus_ids, starts = build_windows()
    log(f"corpus: {len(corpus_ids)} tokens, {len(starts)} windows")
    ppl_results = {}
    # base null FIRST (2 runs) — zero the adapters first so this measures the
    # true base model; the joint/chained runs reload their adapters afterwards.
    for m in handles.values():
        m.A.data.zero_()
        m.B.data.zero_()
    log("base_null: adapters zeroed (true base model)")
    for r in range(2):
        ppl_results[f"base_null_run{r+1}"] = ppl_run(full, tok, corpus_ids, starts, f"base_null_run{r+1}")
    # joint (2 runs)
    load_adapters(handles, ADIR / "adapters_joint", "joint")
    for r in range(2):
        ppl_results[f"joint_run{r+1}"] = ppl_run(full, tok, corpus_ids, starts, f"joint_run{r+1}")
    # chained (2 runs)
    load_adapters(handles, ADIR / "adapters_chained", "stage1_L0")
    s2 = load_file(str(ADIR / "adapters_chained" / "stage2_L63.safetensors"))
    for k, m in handles.items():
        if k.startswith("L63"):
            m.A.data.copy_(s2[f"{k}.A"].cuda().to(torch.bfloat16))
            m.B.data.copy_(s2[f"{k}.B"].cuda().to(torch.bfloat16))
    for r in range(2):
        ppl_results[f"chained_run{r+1}"] = ppl_run(full, tok, corpus_ids, starts, f"chained_run{r+1}")

    # ---- aggregate
    base = [ppl_results[f"base_null_run{r+1}"]["ppl"] for r in range(2)]
    jt = [ppl_results[f"joint_run{r+1}"]["ppl"] for r in range(2)]
    ch = [ppl_results[f"chained_run{r+1}"]["ppl"] for r in range(2)]
    summary = {
        "base_null": base, "joint": jt, "chained": ch,
        "base_mean": np.mean(base), "joint_mean": np.mean(jt), "chained_mean": np.mean(ch),
        "joint_delta": np.mean(jt) - np.mean(base),
        "chained_delta": np.mean(ch) - np.mean(base),
        "joint_noise_band": abs(jt[0] - jt[1]),
        "chained_noise_band": abs(ch[0] - ch[1]),
    }
    log("PPL summary:", json.dumps(summary, indent=2))

    out = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"updates": UPDATES, "lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                   "batch": TRAIN_BATCH, "seed": SEED},
        "sens_weights_raw": raw_wts, "sens_weights_norm": wts,
        "gate_a_pin_check": pin, "gate_a_pass": pin_ok,
        "gate_b_merge_check": merge_check,
        "joint_training": {"mse_before": joint_rec["mse_before"],
                           "mse_final": joint_rec["record"][-1] if joint_rec["record"] else None,
                           "n_updates": len(joint_rec["losses"])},
        "chained_training": {"mse_before": chained_rec["mse_before"],
                             "stage1_final": chained_rec["record"]["stage1"][-1] if chained_rec["record"]["stage1"] else None,
                             "stage2_final": chained_rec["record"]["stage2"][-1] if chained_rec["record"]["stage2"] else None},
        "ppl": ppl_results,
        "summary": summary,
        "total_wall_s": round(time.time() - t_all, 1),
    }
    with open(ADIR / "pilot_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log("wrote", ADIR / "pilot_results.json")


if __name__ == "__main__":
    main()
