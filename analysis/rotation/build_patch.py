#!/usr/bin/env python
"""build_patch.py — rotation-folded INT4 patch for Qwen3.8-27B W4A16 serving checkpoint.

Produces:
  models/Qwen3.8-27B-W4A16-rot-patch/    byte-identical copy of the W4A16-AutoRound
                                          source, with the INT4 linears of 8 selected
                                          layers replaced by block-128 Hadamard
                                          pre-quant, group-128 symmetric INT4,
                                          marlin 8-int4/int32 packing.
  analysis/rotation/tail_fidelity.json
  analysis/rotation/build_patch.log      (stdout redirect at run time)

Selected layers come from analysis/rotation/survey.json['selection']['layers'].
Every INT4 base tensor under model.language_model.layers.<L>.* for those layers is
patched (50 tensors: 6/linear-attn layer, 7/full-attn layer). in_proj_a / in_proj_b
are FP16 (not INT4) and are left untouched, as are lm_head, embed_tokens, mtp.*,
norms, conv1d, A_log/dt_bias, and anything outside the 8 layers.

Procedure (task steps 0-6):
  0. ROUND-TRIP PIN on 3 unpatched INT4 tensors: unpack->repack byte-identical,
     and manual dequant == compressed-tensors dequant (max diff 0.0). Abort on fail.
  1. cp -a source -> patch dir; verify total byte size matches.
  2. per patched tensor: BF16 original (row-tiled) -> float64 block-128 Hadamard on
     the input (last) dim -> per-(row,128-group) absmax scale (=absmax/8, stored as
     float16) -> symmetric INT4 round-to-nearest (codes in [-8,8)) -> marlin repack
     (8 int4/int32, low nibble first) -> IN-PLACE byte overwrite of weight_packed +
     weight_scale at the safetensors data offsets. Same shape+dtype => the file
     header and every unpatched tensor stay byte-identical.
  3. DEQUANT PIN: read the new packed+scale back from the patch file; compressed-
     tensors dequant vs manual dequant, max diff must be 0.0 (per tensor).
  4. TAIL FIDELITY: per patched tensor, three series (BF16 original / current
     checkpoint dequant / rot-patch dequant): exact-zero count and low-band mass
     retention at t = 0.1 * median(per-row 128-group absmax of the ORIGINAL).
     -> analysis/rotation/tail_fidelity.json
  5. BYTE-IDENTITY: sha256 every non-rewritten file in the patch dir; compare to
     source; all must match (only shards holding patched tensors may differ).
  6. HEADER/CONFIG: quantization_config.json unchanged; modified shards' safetensors
     headers keep the same key set / dtypes / shapes as source.

Env knobs:
  ROT_LIMIT=<n>      process only the first n patched tensors (smoke test).
  ROT_SKIP_HASH=1    skip the full per-file sha256 sweep (smoke test speedup).

CPU-only. The source checkpoint is treated as read-only and never modified.
"""
import json
import os
import struct
import sys
import shutil
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# pinned library dequant (independent decoder for the pins)
sys.path.insert(0, "/home/hubbe/qwen-serving/venv/lib/python3.12/site-packages")
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import dequantize as ct_dequantize
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

REPO = Path("/home/hubbe/qwen-serving")
SRC = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
DST = REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch"
FP16 = REPO / "models-fp16" / "Qwen3.8-27B"
ANALYSIS = REPO / "analysis" / "rotation"
INVENTORY = REPO / "analysis" / "salted" / "inventory.json"

BITS = 4
GROUP = 128
TILE_BYTES = 150_000_000  # bound the float64 working set per row-tile

ROT_LIMIT = int(os.environ.get("ROT_LIMIT", "0"))
ROT_SKIP_HASH = os.environ.get("ROT_SKIP_HASH", "0") == "1"

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(str(msg))

# ------------------------------------------------------------------ hadamard
def sylvester(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

def block_apply(W, H):
    """W @ block_diag(H) via broadcast matmul (no dense block-diagonal)."""
    r, c = W.shape
    b = H.shape[0]
    return (W.reshape(r, c // b, b) @ H).reshape(r, c)

# ------------------------------------------------------------------ marlin pack
def unpack_nibbles(packed_i64, bits=BITS):
    """int32-packed -> [out, in] int64 nibbles in [0, 2**bits). Low nibble first."""
    r, c32 = packed_i64.shape
    f = 32 // bits
    mask = (1 << bits) - 1
    nib = np.empty((r, c32 * f), dtype=np.int64)
    for i in range(f):
        nib[:, i::f] = (packed_i64 >> (bits * i)) & mask
    return nib

def repack_nibbles(nib, bits=BITS):
    """[out, in] int64 nibbles -> int32-packed (8/int32, low nibble first). Inverse of unpack."""
    r, c = nib.shape
    f = 32 // bits
    mask = (1 << bits) - 1
    out = np.zeros((r, c // f), dtype=np.int64)
    for i in range(f):
        out |= (nib[:, i::f] & mask) << (bits * i)
    return out

# ------------------------------------------------------------------ dequant
def _to_np_int64(x):
    return x.to(torch.int64).numpy() if isinstance(x, torch.Tensor) else x.astype(np.int64)

def _to_np_f32(x):
    return x.to(torch.float32).numpy() if isinstance(x, torch.Tensor) else x.astype(np.float32)

def manual_dequant(packed, scale, shape, bits=BITS):
    """packed [out, in//8] int32, scale [out, in//128] f16, shape [out, in] -> f32 [out, in]."""
    up = unpack_nibbles(_to_np_int64(packed), bits)
    sc = _to_np_f32(scale)
    gs = shape[1] // sc.shape[1]
    sc = np.repeat(sc, gs, axis=1)
    return (up - 2 ** (bits - 1)).astype(np.float32) * sc

def ct_dequant(packed, scale, shape, bits=BITS):
    gs = shape[1] // scale.shape[1]
    return ct_dequantize(
        x_q=unpack_from_int32(packed, bits, shape),
        scale=scale.to(torch.float32),
        zero_point=None,
        args=QuantizationArgs(strategy=QuantizationStrategy.GROUP, group_size=gs),
    )

# ------------------------------------------------------------------ safetensors low-level
def read_st_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
    return hlen, header

def st_abs_offset(hlen, header, key):
    return 8 + hlen + header[key]["data_offsets"][0]

def st_write_at(path, hlen, header, key, arr):
    off = st_abs_offset(hlen, header, key)
    b = np.ascontiguousarray(arr).tobytes()
    with open(path, "r+b") as f:
        f.seek(off)
        n = f.write(b)
    assert n == len(b), "short write for %s" % key
    return off, len(b)

# ------------------------------------------------------------------ quantize
def quantize_rotated(W_bf16, H, tile_bytes=TILE_BYTES):
    """W_bf16 [out, in] (bf16) -> (packed int32 [out, in//8], scale f16 [out, in//128],
    amax_orig f64 [out, in//128]).
    float64 block-128 Hadamard on the input dim; scale = absmax(rotated)/8; symmetric
    INT4 round-to-nearest, codes in [-8, 8). amax_orig = per-(row,128-group) absmax of
    the ORIGINAL (unrotated) weight, used for the tail-fidelity threshold t."""
    out, inp = W_bf16.shape
    G = inp // GROUP
    packed = np.empty((out, inp // 8), dtype=np.int32)
    scale16 = np.empty((out, G), dtype=np.float16)
    amax_orig = np.empty((out, G), dtype=np.float64)
    rows = max(1, int(tile_bytes // (inp * 8)))
    for r0 in range(0, out, rows):
        r1 = min(r0 + rows, out)
        W = W_bf16[r0:r1].to(torch.float64).numpy()
        Wo = W.reshape(r1 - r0, G, GROUP)
        amax_orig[r0:r1] = np.abs(Wo).max(axis=2)
        Wrot = block_apply(W, H).reshape(r1 - r0, G, GROUP)
        amax_rot = np.abs(Wrot).max(axis=2)
        s64 = np.where(amax_rot > 0, amax_rot / 8.0, 1.0)
        s16 = s64.astype(np.float16)
        sq = s16.astype(np.float64)[:, :, None]
        codes = np.clip(np.round(Wrot / sq), -8, 7).astype(np.int64)
        packed[r0:r1] = repack_nibbles((codes + 8).reshape(r1 - r0, inp)).astype(np.int32)
        scale16[r0:r1] = s16
    return packed, scale16, amax_orig

# ------------------------------------------------------------------ loaders
_FP16_IDX = None
def _fp16_idx():
    global _FP16_IDX
    if _FP16_IDX is None:
        _FP16_IDX = json.loads((FP16 / "model.safetensors.index.json").read_text())["weight_map"]
    return _FP16_IDX

def load_src_tensor(shard, key):
    with safe_open(SRC / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)

def load_fp16(base):
    name = base + ".weight"
    shard = _fp16_idx()[name]
    with safe_open(FP16 / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ------------------------------------------------------------------ patch set
def build_patch_set():
    inv = json.loads(INVENTORY.read_text())
    int4 = [t for t in inv["tensors"] if t["group"] == "INT4-linear"]
    sel = json.loads((ANALYSIS / "survey.json").read_text())["selection"]["layers"]
    pref = ["model.language_model.layers.%d." % L for L in sel]
    def base(t):
        lw = t["logical_weight"]
        return lw[:-len(".weight")] if lw.endswith(".weight") else lw
    patch = {}
    for t in int4:
        b = base(t)
        if any(b.startswith(p) for p in pref) and b not in patch:
            patch[b] = t
    return sel, patch

# ================================================================== main
def main():
    t0 = time.time()
    H = sylvester(128)
    hnorm = float(np.abs(H @ H.T - np.eye(128)).max())
    assert hnorm < 1e-9, "H128 not orthonormal: %g" % hnorm
    log("H128 built; orthonormal (max|H H^T - I| = %.3e)" % hnorm)

    sel, patch = build_patch_set()
    log("selected layers: %s" % sel)
    log("patch set: %d INT4 base tensors" % len(patch))
    if ROT_LIMIT > 0:
        patch = {k: patch[k] for k in list(patch)[ :ROT_LIMIT]}
        log("ROT_LIMIT=%d -> processing %d tensors (smoke test)" % (ROT_LIMIT, len(patch)))

    # ================= STEP 0: round-trip pin =================
    log("\n=== STEP 0: round-trip pin (unpatched INT4) ===")
    pin_bases = [
        "model.language_model.layers.1.mlp.down_proj",
        "model.language_model.layers.5.mlp.down_proj",
        "model.language_model.layers.10.mlp.down_proj",
    ]
    inv = json.loads(INVENTORY.read_text())
    def _base(lw):
        return lw[:-6] if lw.endswith(".weight") else lw
    grp = {_base(t["logical_weight"]): t["group"] for t in inv["tensors"]}
    for b in pin_bases:
        assert grp.get(b) == "INT4-linear", "pin base %s not INT4 (got %s)" % (b, grp.get(b))
        assert b not in patch, "pin base %s is in the patch set!" % b
    roundtrip = {}
    roundtrip_ok = True
    for b in pin_bases:
        t = next(x for x in inv["tensors"] if x["name"] == b + ".weight_packed")
        shard = t["file"]
        packed = load_src_tensor(shard, b + ".weight_packed")
        scale = load_src_tensor(shard, b + ".weight_scale")
        shape = tuple(load_src_tensor(shard, b + ".weight_shape").tolist())
        nib = unpack_nibbles(packed.to(torch.int64).numpy())
        rep = repack_nibbles(nib).astype(np.int32)
        ident = bool(np.array_equal(rep, packed.numpy()))
        md = manual_dequant(packed, scale, shape)
        cd = ct_dequant(packed, scale, shape).numpy()
        ddiff = float(np.abs(md - cd).max())
        ok = ident and ddiff == 0.0
        roundtrip_ok &= ok
        roundtrip[b] = {"byte_identical": ident, "manual_vs_ct_max_diff": ddiff, "shape": shape}
        log("  %s: byte-identical=%s  manual-vs-ct=%.3e  shape=%s  -> %s"
            % (b.replace("model.language_model.layers.", "L"), ident, ddiff, shape, "PASS" if ok else "FAIL"))
    if not roundtrip_ok:
        log("ROUND-TRIP PIN FAILED -> aborting (layout understanding is wrong)")
        (ANALYSIS / "build_patch.log").write_text("\n".join(LOG) + "\n")
        sys.exit(1)
    log("round-trip pin: PASS (all %d)" % len(pin_bases))

    # ================= STEP 1: cp -a =================
    log("\n=== STEP 1: copy source -> patch dir ===")
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    src_files = sorted(p for p in SRC.iterdir() if p.is_file())
    for p in src_files:
        shutil.copy2(p, DST / p.name)
    src_total = sum(p.stat().st_size for p in src_files)
    dst_total = sum((DST / p.name).stat().st_size for p in src_files)
    log("copied %d files; source total %d bytes, patch total %d bytes -> %s"
        % (len(src_files), src_total, dst_total, "MATCH" if src_total == dst_total else "MISMATCH"))
    assert src_total == dst_total, "copy size mismatch"

    # validate offset math: header-derived offset must match safe_open bytes
    hlen0, hdr0 = read_st_header(DST / "model-00001-of-00007.safetensors")
    k0 = "model.language_model.layers.0.mlp.gate_proj.weight_packed"
    off = st_abs_offset(hlen0, hdr0, k0)
    with open(DST / "model-00001-of-00007.safetensors", "rb") as f:
        f.seek(off); probe = f.read(16)
    ref = load_src_tensor("model-00001-of-00007.safetensors", k0).numpy().tobytes()[:16]
    assert probe == ref, "safetensors offset math WRONG (probe != safe_open bytes)"
    log("offset-math self-check: PASS (16-byte probe at header offset matches safe_open)")

    # ================= STEP 2: patch =================
    log("\n=== STEP 2: rotate + quantize + in-place byte overwrite ===")
    by_shard = {}
    for b, t in patch.items():
        by_shard.setdefault(t["file"], []).append(b)
    hdr = {shard: read_st_header(DST / shard) for shard in by_shard}
    for shard, bases in by_shard.items():  # sanity: no offset overlap per shard
        hlen, h = hdr[shard]
        offs = []
        for b in bases:
            for suf in (".weight_packed", ".weight_scale"):
                o0, o1 = h[b + suf]["data_offsets"]
                offs.append((st_abs_offset(hlen, h, b + suf),
                              st_abs_offset(hlen, h, b + suf) + (o1 - o0), b + suf))
        offs.sort()
        for a, c in zip(offs, offs[1:]):
            assert a[1] <= c[0], "offset overlap in %s" % shard

    patch_results = {}
    for shard in sorted(by_shard):
        hlen, h = hdr[shard]
        for b in sorted(by_shard[shard]):
            tb = time.time()
            shape = tuple(load_src_tensor(shard, b + ".weight_shape").tolist())
            assert shape[1] % GROUP == 0 and shape[1] % 8 == 0, "%s bad in-dim %s" % (b, shape)
            src_packed = load_src_tensor(shard, b + ".weight_packed")
            src_scale = load_src_tensor(shard, b + ".weight_scale")
            W = load_fp16(b)
            assert tuple(W.shape) == shape, "%s shape mismatch %s vs %s" % (b, tuple(W.shape), shape)
            assert str(W.dtype) == "torch.bfloat16", "%s not bf16: %s" % (b, W.dtype)
            packed, scale16, amax_orig = quantize_rotated(W, H)
            n1 = st_write_at(DST / shard, hlen, h, b + ".weight_packed", packed)[1]
            n2 = st_write_at(DST / shard, hlen, h, b + ".weight_scale", scale16)[1]
            del W
            patch_results[b] = {"shard": shard, "shape": shape, "amax_orig": amax_orig,
                                "src_packed": src_packed, "src_scale": src_scale}
            log("  [%s] %-58s shape=%s  (%.1fs)"
                % (shard[6:9], b.replace("model.language_model.layers.", "L"), shape, time.time() - tb))
    log("patched %d tensors across %d shards" % (len(patch_results), len(by_shard)))

    # ================= STEP 3: dequant pin =================
    log("\n=== STEP 3: dequant pin (new packed+scale read back from patch file) ===")
    dequant_pin = {}
    all_zero = True
    for b in sorted(patch_results):
        pr = patch_results[b]; shard = pr["shard"]; shape = pr["shape"]
        hlen, h = hdr[shard]
        with safe_open(DST / shard, framework="pt", device="cpu") as f:
            np_ = f.get_tensor(b + ".weight_packed"); ns = f.get_tensor(b + ".weight_scale")
        md = manual_dequant(np_, ns, shape)
        cd = ct_dequant(np_, ns, shape).numpy()
        d = float(np.abs(md - cd).max())
        all_zero &= (d == 0.0)
        dequant_pin[b] = d
        log("  %-58s max|manual - ct| = %.3e  %s"
            % (b.replace("model.language_model.layers.", "L"), d, "PASS" if d == 0.0 else "FAIL"))
    log("dequant pin: %s (%d tensors)" % ("PASS (all 0.0)" if all_zero else "FAIL", len(dequant_pin)))
    assert all_zero, "dequant pin failed"

    # ================= STEP 4: tail fidelity =================
    log("\n=== STEP 4: tail fidelity (zeros + low-band mass retention) ===")
    tf = {"t_def": "t = 0.1 * median(per-row 128-group absmax of the BF16 ORIGINAL); same t for all 3 series per tensor",
          "band_def": "band_mass = sum(|w| for |w| <= t); retention = band_mass / band_mass[orig]",
          "tensors": {}, "summary": {}}
    for b in sorted(patch_results):
        pr = patch_results[b]; shard = pr["shard"]; shape = pr["shape"]; out, inp = shape
        G = inp // GROUP
        t = 0.1 * float(np.median(pr["amax_orig"]))
        W = load_fp16(b)
        Wo = W.to(torch.float32).numpy()
        cur = manual_dequant(pr["src_packed"], pr["src_scale"], shape)
        with safe_open(DST / shard, framework="pt", device="cpu") as f:
            np_ = f.get_tensor(b + ".weight_packed"); ns = f.get_tensor(b + ".weight_scale")
        rot = manual_dequant(np_, ns, shape)
        def metrics(x):
            ax = np.abs(x)
            return int((x == 0.0).sum()), float(ax[ax <= t].sum())
        zo, bo = metrics(Wo)
        zc, bc = metrics(cur)
        zr, br = metrics(rot)
        tf["tensors"][b] = {
            "t": t,
            "orig": {"zeros": zo, "band_mass": bo},
            "current": {"zeros": zc, "band_mass": bc, "retention": (bc / bo) if bo else None},
            "rot": {"zeros": zr, "band_mass": br, "retention": (br / bo) if bo else None},
        }
        log("  %-58s t=%.5g  zeros o/c/r=%d/%d/%d  retain c/r=%.4f/%.4f"
            % (b.replace("model.language_model.layers.", "L"), t, zo, zc, zr,
               (bc / bo if bo else -1), (br / bo if bo else -1)))
        del W, Wo, cur, rot
    beats_zeros = beats_band = floor = 0
    for b, d in tf["tensors"].items():
        cz, rz = d["current"]["zeros"], d["rot"]["zeros"]
        cr = d["current"]["retention"] or 0.0
        rr = d["rot"]["retention"] or 0.0
        if rz < cz: beats_zeros += 1
        if rr > cr: beats_band += 1
        if rz <= cz and rr >= cr: floor += 1
    tf["summary"] = {"n_tensors": len(tf["tensors"]),
                     "rot_beats_current_on_zeros": beats_zeros,
                     "rot_beats_current_on_band_retention": beats_band,
                     "floor_satisfied_no_regression_both": floor}
    (ANALYSIS / "tail_fidelity.json").write_text(json.dumps(tf, indent=1))
    log("tail_fidelity.json written: %d tensors; beats zeros=%d, beats band=%d, floor(both)=%d"
        % (len(tf["tensors"]), beats_zeros, beats_band, floor))

    # ================= STEP 5: byte-identity =================
    log("\n=== STEP 5: byte-identity (sha256) ===")
    rewritten_shards = set(by_shard.keys())
    checked = mism = 0
    mismatch_list = []
    if ROT_SKIP_HASH:
        log("  (ROT_SKIP_HASH=1: full per-file sweep skipped; rewritten shards = %s)"
            % sorted(rewritten_shards))
    else:
        for p in src_files:
            sh = sha256(SRC / p.name); dh = sha256(DST / p.name)
            if p.name in rewritten_shards:
                status = "REWRITTEN (expected diff)" if sh != dh else "identical"
            else:
                checked += 1
                if sh != dh:
                    mism += 1; mismatch_list.append(p.name)
                status = "identical" if sh == dh else "MISMATCH"
            log("  %-45s %s" % (p.name, status))
        log("byte-identity: %d non-rewritten files checked, %d mismatches; %d rewritten shards (allowed to differ)"
            % (checked, mism, len(rewritten_shards)))
        if mismatch_list:
            log("  MISMATCHES: %s" % mismatch_list)
        assert mism == 0, "byte-identity failed: %s" % mismatch_list

    # ================= STEP 6: header/config validity =================
    log("\n=== STEP 6: header/config validity ===")
    qc_same = sha256(SRC / "quantization_config.json") == sha256(DST / "quantization_config.json")
    log("quantization_config.json unchanged: %s" % qc_same)
    hdr_ok = True
    for shard in sorted(rewritten_shards):
        _, sh_h = read_st_header(SRC / shard)
        _, dh_h = read_st_header(DST / shard)
        same_keys = set(sh_h) == set(dh_h)
        same_ds = all(sh_h[k]["dtype"] == dh_h[k]["dtype"] and sh_h[k]["shape"] == dh_h[k]["shape"]
                      for k in sh_h if k != "__metadata__")
        hdr_ok &= same_keys and same_ds
        log("  %s: keys_same=%s dtype_shape_same=%s header_bytes_identical=%s"
            % (shard, same_keys, same_ds, sh_h == dh_h))
    assert qc_same and hdr_ok, "header/config validity failed"

    log("\n=== DONE in %.1fs ===" % (time.time() - t0))

    # ---------------------------------------------------------------- final JSON
    per_layer = {}
    for b in patch_results:
        L = b.split(".layers.")[1].split(".")[0]
        per_layer.setdefault(L, []).append(b.rsplit(".", 1)[-1])
    final = {
        "patch_dir": str(DST),
        "total_size_bytes": dst_total,
        "source_total_bytes": src_total,
        "size_match": src_total == dst_total,
        "n_tensors": len(patch_results),
        "selected_layers": sel,
        "per_layer": per_layer,
        "roundtrip_pin": {"ok": roundtrip_ok, "per_tensor": roundtrip},
        "dequant_pin": {"all_zero": all_zero, "per_tensor_max_diff": dequant_pin},
        "byte_identity": {"checked": checked, "mismatches": mism, "mismatch_list": mismatch_list,
                           "rewritten_shards": sorted(rewritten_shards), "skip_hash": ROT_SKIP_HASH},
        "header_config": {"quant_config_unchanged": qc_same, "headers_consistent": hdr_ok},
        "tail_fidelity_summary": tf["summary"],
        "tail_fidelity_table": {
            b: {"t": d["t"],
                "zeros": {"current": d["current"]["zeros"], "rot": d["rot"]["zeros"]},
                "band_retention": {"current": d["current"]["retention"], "rot": d["rot"]["retention"]}}
            for b, d in tf["tensors"].items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    print("\nFINAL_JSON:")
    print(json.dumps(final, indent=1))

if __name__ == "__main__":
    main()
