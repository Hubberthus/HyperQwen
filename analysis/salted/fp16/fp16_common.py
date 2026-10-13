"""Shared loader for the original (unquantized) Qwen3.8-27B checkpoint.

models-fp16/Qwen3.8-27B is the plain BF16/FP16 source (fetched by
fetch_originals.py, gitignored). Tensor -> shard mapping comes from the
original's own model.safetensors.index.json — its sharding is NOT assumed
to match the W4A16 checkpoint's.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent.parent  # ~/qwen-serving
CKPT_FP16 = REPO / "models-fp16" / "Qwen3.8-27B"
SALTED = REPO / "analysis" / "salted"
FP16 = SALTED / "fp16"
CHUNK_ROWS = 4096  # same tiling as the baseline scripts

_wmap = None


def weight_map():
    global _wmap
    if _wmap is None:
        idx = json.loads((CKPT_FP16 / "model.safetensors.index.json").read_text())
        _wmap = idx["weight_map"]
    return _wmap


def shard_for(base):
    return weight_map()[base + ".weight"]


def tensor_meta(base):
    """(shape, dtype_str) straight from the safetensors slice header."""
    with safe_open(CKPT_FP16 / shard_for(base), framework="pt", device="cpu") as f:
        sl = f.get_slice(base + ".weight")
        return tuple(sl.get_shape()), str(sl.get_dtype())


def tensor_shape(base):
    return tensor_meta(base)[0]


def load_tensor(base):
    """Full tensor as float32 numpy. Only for tensors where a full copy is
    acceptable (largest sampled: lm_head 1.27B elems = 5.1 GB float32)."""
    with safe_open(CKPT_FP16 / shard_for(base), framework="pt", device="cpu") as f:
        w = f.get_tensor(base + ".weight")
    return w.to(torch.float32).numpy()


def tensor_chunks(base, chunk_rows=CHUNK_ROWS):
    """Yield (float32 chunk [r0:r1], row0) — drop-in replacement for
    codebook_gain.tensor_chunks (which dequantizes the W4A16 checkpoint)."""
    with safe_open(CKPT_FP16 / shard_for(base), framework="pt", device="cpu") as f:
        sl = f.get_slice(base + ".weight")
        rows = sl.get_shape()[0]
        for r0 in range(0, rows, chunk_rows):
            r1 = min(r0 + chunk_rows, rows)
            yield sl[r0:r1].to(torch.float32).numpy(), r0
