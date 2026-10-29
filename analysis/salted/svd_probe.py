"""Probe 7: SVD / low-rank structure of dequantized weights.

Question: does tensor or 32x32-block energy concentrate in a low-rank
subspace enough that "fp16 rank-r factors + 4-bit residual codes" beats
plain 4-bit storage?

Arms (per tensor):
  A. Per-TENSOR SVD on dequantized W (float32): energy captured at rank r
     in {64,128,256,512,1024} (relative Frobenius^2), vs a global
     permutation null of the same tensor (destroys all structure, preserves
     the value distribution exactly).
  B. Per-BLOCK SVD: mean spectrum of sampled 32x32 blocks of the
     dequantized tensor (~2000 blocks/tensor), mean energy at rank
     r in {4,8,16}, vs a within-block permutation null.
  C. Implied bpw for "rank-r fp16 factors + 4-bit residual":
     r*(rows+cols)*16/(rows*cols) + 4*(1 - E_r), also with the null's E_r.

Tensors: gate_proj + down_proj at model.language_model.layers.{0,63}.
Gate: pin check manual unpack vs compressed_tensors.unpack_from_int32 == 0.
Fixed seed 0.

CAVEAT (see JSON): fp16 factor storage cost is included in the bpw formula;
energy capture is a NECESSARY not SUFFICIENT condition — an output-level
accuracy check would follow only if the margin over the null is clear.

Run: venv/bin/python analysis/salted/svd_probe.py   (budget 25 min CPU)
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "venv" / "lib" / "python3.12" / "site-packages"))
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32  # noqa: E402

CKPT = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
OUT = REPO / "analysis" / "salted" / "svd_probe.json"
SEED = 0
GROUP = 128
TENSOR_RANKS = [64, 128, 256, 512, 1024]
BLOCK_RANKS = [4, 8, 16]
BLOCK = 32
N_BLOCKS = 2000
CHUNK = 4096

BASES = [
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.63.mlp.gate_proj",
    "model.language_model.layers.63.mlp.down_proj",
]

_INDEX = None


def shard_for(key):
    global _INDEX
    if _INDEX is None:
        _INDEX = json.load(open(CKPT / "model.safetensors.index.json"))["weight_map"]
    return CKPT / _INDEX[key]


_FILES = {}


def get_file(key):
    path = shard_for(key)
    if path not in _FILES:
        _FILES[path] = safe_open(path, framework="pt", device="cpu")
    return _FILES[path]


def manual_unpack(packed_np, num_bits=4):
    factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    rows, n32 = packed_np.shape
    out = np.empty((rows, n32 * factor), dtype=np.uint8)
    for i in range(factor):
        out[:, i::factor] = ((packed_np >> (num_bits * i)) & mask).astype(np.uint8)
    return out


def pin_check():
    key = "model.language_model.layers.31.mlp.gate_proj.weight_packed"
    f = get_file(key)
    packed = f.get_tensor(key)
    shape = tuple(f.get_tensor(
        "model.language_model.layers.31.mlp.gate_proj.weight_shape").tolist())
    ref = unpack_from_int32(packed, 4, shape)
    p = packed.to(torch.int64).numpy()
    rows, cols = shape
    up = np.empty((rows, cols), dtype=np.int64)
    for i in range(8):
        up[:, i::8] = (p >> (4 * i)) & 15
    diff = int(np.abs((up - 8) - ref.numpy()).max())
    print(f"  PIN: max|manual-unpack_from_int32| = {diff} over {ref.numel()} codes")
    assert diff == 0
    return diff


def load_dequant(base):
    """Full dequantized W (float32) via chunked unpack; signed codes * scale."""
    pk = f"{base}.weight_packed"
    sk = f"{base}.weight_scale"
    f = get_file(pk)
    rows, cols = tuple(f.get_tensor(f"{base}.weight_shape").tolist())
    groups = cols // GROUP
    W = np.empty((rows, cols), dtype=np.float32)
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        packed = np.asarray(f.get_slice(pk)[start:stop]).astype(np.int64)
        codes = manual_unpack(packed).astype(np.float32) - 8.0
        sc = np.asarray(f.get_slice(sk)[start:stop]).astype(np.float32)  # [n, groups]
        W[start:stop] = codes * np.repeat(sc, GROUP, axis=1)
    return W


def energy_at_ranks(s, ranks):
    e = np.cumsum(s ** 2)
    total = e[-1]
    return {r: float(e[r - 1] / total) for r in ranks}


def block_energy(W, rng):
    """Mean per-block energy capture at ranks {4,8,16} over sampled 32x32 blocks,
    and the within-block-permutation null of the same blocks."""
    rows, cols = W.shape
    energies = {r: [] for r in BLOCK_RANKS}
    nulls = {r: [] for r in BLOCK_RANKS}
    picked = 0
    guard = 0
    while picked < N_BLOCKS and guard < N_BLOCKS * 50:
        guard += 1
        i = int(rng.integers(0, rows - BLOCK + 1))
        j = int(rng.integers(0, cols - BLOCK + 1))
        B = W[i:i + BLOCK, j:j + BLOCK].astype(np.float64)
        s = np.linalg.svd(B, compute_uv=False)
        e = energy_at_ranks(s, BLOCK_RANKS)
        Bp = B.ravel().copy()
        rng.shuffle(Bp)
        Bp = Bp.reshape(BLOCK, BLOCK)
        sp = np.linalg.svd(Bp, compute_uv=False)
        ep = energy_at_ranks(sp, BLOCK_RANKS)
        for r in BLOCK_RANKS:
            energies[r].append(e[r])
            nulls[r].append(ep[r])
        picked += 1
    out = {"n_blocks": picked}
    for r in BLOCK_RANKS:
        out[f"energy_rank{r}"] = float(np.mean(energies[r]))
        out[f"null_energy_rank{r}"] = float(np.mean(nulls[r]))
    return out


def probe(base, results):
    t0 = time.time()
    W = load_dequant(base)
    rows, cols = W.shape
    rng = np.random.default_rng(SEED)
    res = {"shape": [rows, cols], "n_elements": int(W.size)}

    # Arm A: tensor SVD + global permutation null
    s = np.linalg.svd(W, compute_uv=False)
    e_real = energy_at_ranks(s, TENSOR_RANKS)
    Wflat = W.ravel().astype(np.float32).copy()
    rng.shuffle(Wflat)
    Wnull = Wflat.reshape(rows, cols)
    s_null = np.linalg.svd(Wnull, compute_uv=False)
    e_null = energy_at_ranks(s_null, TENSOR_RANKS)
    res["tensor"] = {
        f"energy_rank{r}": e_real[r] for r in TENSOR_RANKS} | {
        f"null_energy_rank{r}": e_null[r] for r in TENSOR_RANKS}
    # singular-value decay: sigma_r / sigma_1
    res["tensor"]["sigma_ratio_1024"] = float(s[1023] / s[0])
    res["tensor"]["null_sigma_ratio_1024"] = float(s_null[1023] / s_null[0])

    # Arm C: implied bpw
    bpw = {}
    for r in TENSOR_RANKS:
        for tag, E in (("real", e_real[r]), ("null", e_null[r])):
            bpw[f"bpw_rank{r}_{tag}"] = r * (rows + cols) * 16 / (rows * cols) + 4 * (1 - E)
    res["bpw"] = bpw

    # Arm B: block SVD + within-block permutation null
    res["block"] = block_energy(W, rng)
    results[base] = res
    print(f"  {base} {rows}x{cols}: E1024={e_real[1024]:.4f} "
          f"(null {e_null[1024]:.4f}) bpw1024={bpw['bpw_rank1024_real']:.3f} "
          f"({time.time() - t0:.0f}s)")


def main():
    pin = pin_check()
    results = {
        "pin_max_diff": pin,
        "seed": SEED,
        "ts": datetime.now(timezone.utc).isoformat(),
        "tensors": {},
        "caveats": [
            "factor storage assumed fp16 (16 bits); fp8/int8 factors would lower the factor term",
            "energy capture is necessary but NOT sufficient: no output/activation-level check performed; "
            "a residual re-quantized at 4 bits adds quantization error on top of the discarded tail",
            "null = entry permutation, preserves the value distribution exactly, destroys all "
            "positional/low-rank structure; margin over null measures structure, not usefulness",
        ],
    }
    for base in BASES:
        probe(base, results["tensors"])

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
