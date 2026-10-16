"""Probe: conditional-entropy scan on down_proj L63 (single representative tensor).

Question: is there exploitable *sequence* structure in the quantized codes or group
scales that a previous-value (delta/PNG-style) predictor could compress away?
Salted spike proved position-in-DISTRIBUTION is null; nothing has measured
position-in-SEQUENCE.

Arms (each = conditional entropy in bits/code vs marginal entropy baseline):
  A. code[i] | code[i, j-1]        previous column (within-row delta)
  B. code[i] | code[i-1, j]        previous row (vertical)
  C. code[i] | code above+left     2-D PNG Paeth-style predictor (max context, small)
  D. scale[i, g] | scale[i-1, g]   previous row of the same group scale (scale lattice)
  E. scale[i, g] | scale[i, g-1]   previous group along the input axis
Readout: gap = marginal - conditional, in bits. >0.1 bits/weight = worth a kernel probe.

Baseline pin: manual unpack vs compressed_tensors (must be 0.0) — same pin as Task 2.

Tensor: down_proj L63  [5120, 17408], INT4 symmetric group-128 -> 17408/128 = 136
groups/row, 136 scales [5120, 136]. Codes are signed int4 stored as packed int32.

Run:  venv/bin/python analysis/salted/cond_entropy_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent  # ~/qwen-serving
sys.path.insert(0, str(REPO / "venv" / "lib" / "python3.12" / "site-packages"))

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32  # noqa: E402

CKPT = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
OUT = REPO / "analysis" / "salted" / "cond_entropy_probe.json"

BASE = "model.language_model.layers.63.mlp.down_proj"
NUM_BITS = 4
GROUP = 128


def load_packed(base):
    """Find the shard holding this tensor; return packed, scale, shape."""
    shards = sorted(CKPT.glob("*.safetensors"))
    for shard in shards:
        f = safe_open(CKPT / shard, framework="pt", device="cpu")
        keys = list(f.keys())
        if f"{base}.weight_packed" in keys:
            return (
                f.get_tensor(f"{base}.weight_packed"),
                f.get_tensor(f"{base}.weight_scale"),
                tuple(f.get_tensor(f"{base}.weight_shape").tolist()),
                shard,
            )
    raise FileNotFoundError(f"{base}.weight_packed not found in any shard")


def unpack_codes(packed, shape):
    """int32 [rows, cols/8] -> signed int4 codes [rows, cols] (value + 8, i.e. 0..15)."""
    rows, cols = shape
    factor = 32 // NUM_BITS
    assert cols % factor == 0
    p = packed.to(torch.int64).numpy()
    up = np.empty((rows, cols), dtype=np.int64)
    mask = (1 << NUM_BITS) - 1
    for i in range(factor):
        up[:, i::factor] = (p >> (NUM_BITS * i)) & mask
    return up  # unsigned 0..15 codes; +8 is the symmetric offset, uniform shift -> entropy identical


def pin_check(packed, scale, shape):
    """Manual unpack vs compressed_tensors unpack_from_int32 (exact int equality)."""
    ref = unpack_from_int32(packed, NUM_BITS, shape)
    ours = torch.from_numpy(unpack_codes(packed, shape))
    ours_signed = (ours - 2 ** (NUM_BITS - 1)).to(torch.int8)
    diff = (ours_signed.to(torch.int64) - ref.to(torch.int64)).abs().max().item()
    print(f"  PIN unpack {BASE}: max|manual-ct| = {diff} (codes={ref.numel():,})")
    assert diff == 0, "unpack pin failed — aborting"
    return {"tensor": BASE, "num_bits": NUM_BITS, "max_abs_diff": 0, "codes": int(ref.numel())}


def marginal_entropy(codes2d):
    """Exact marginal entropy of the code alphabet from full-tensor counts."""
    n = codes2d.size
    counts = np.bincount(codes2d.ravel(), minlength=16).astype(np.float64)
    p = counts / n
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def cond_entropy_2d(arr, pred):
    """Conditional entropy H(arr | pred) over pairs (arr, pred) — full tensor, exact counts.
    Alphabet sizes inferred from the arrays (16 for int4 codes, 256 for scale bins)."""
    a = arr.ravel()
    p = pred.ravel()
    n_a = int(arr.max()) + 1
    n_p = int(pred.max()) + 1
    joint = np.bincount((a * n_p + p).ravel(), minlength=n_a * n_p).astype(np.float64)
    joint = joint.reshape(n_a, n_p)
    n = joint.sum()
    nz = joint[joint > 0]
    H_joint = -(nz / n * np.log2(nz / n)).sum()
    row_marg = joint.sum(axis=1)
    nzr = row_marg[row_marg > 0]
    H_pred = -(nzr / n * np.log2(nzr / n)).sum()
    return float(H_joint - H_pred)  # = H(a|p)


def main():
    packed, scale, shape, shard = load_packed(BASE)
    rows, cols = shape
    print(f"Tensor {BASE}: shape {shape}, shard {shard}, scales {tuple(scale.shape)}")
    pin = pin_check(packed, scale, shape)
    codes = unpack_codes(packed, shape)          # [rows, cols] uint 0..15
    scales = scale.to(torch.float64).numpy()     # [rows, groups]

    H_marg = marginal_entropy(codes)
    results = {"tensor": BASE, "shape": list(shape), "group": GROUP,
               "pin": pin, "marginal_code_entropy_bits": H_marg}

    # A: previous column (shift each row right by 1; col 0 has no left neighbor ->
    #    drop first column of both)
    pred = np.roll(codes, 1, axis=1)
    results["A_prev_col"] = cond_entropy_2d(codes[:, 1:], pred[:, 1:])

    # B: previous row (drop first row)
    pred = np.roll(codes, 1, axis=0)
    results["B_prev_row"] = cond_entropy_2d(codes[1:, :], pred[1:, :])

    # C: Paeth-style 2-D predictor: element-wise min/median of left, up, (left+up-leftup)
    left = np.roll(codes, 1, axis=1)
    up = np.roll(codes, 1, axis=0)
    upl = np.roll(np.roll(codes, 1, axis=0), 1, axis=1)
    pred = np.minimum(np.minimum(left, up), np.abs(left + up - upl))
    # valid interior region: rows 1.., cols 1..
    results["C_paeth_2d"] = cond_entropy_2d(codes[1:, 1:], pred[1:, 1:])

    # D: scale[i,g] | scale[i-1,g] — bucket scales into 256 equal-frequency bins first
    s = scales
    qs = np.quantile(s, np.linspace(0, 1, 257))
    sb = np.searchsorted(qs, s, side="right") - 1
    sb = np.clip(sb, 0, 255).astype(np.int64)
    prev = np.roll(sb, 1, axis=0)
    results["D_scale_prev_row"] = cond_entropy_2d(sb[1:, :], prev[1:, :])

    # E: scale[i,g] | scale[i,g-1]
    prev = np.roll(sb, 1, axis=1)
    results["E_scale_prev_group"] = cond_entropy_2d(sb[:, 1:], prev[:, 1:])

    # Summary: gaps
    gaps = {k: H_marg - results[k] for k in ["A_prev_col", "B_prev_row", "C_paeth_2d"]}
    for k, v in results.items():
        if k.startswith(("A_", "B_", "C_")):
            print(f"  {k}: H|pred = {v:.4f} bits  (gap {H_marg - v:+.4f})")
        elif k.startswith(("D_", "E_")):
            print(f"  {k}: H|pred = {v:.4f} bits / scale")
    print(f"  marginal code entropy: {H_marg:.4f} bits of {NUM_BITS}")
    print(f"  code gaps: {json.dumps({k: round(g, 4) for k, g in gaps.items()})}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT}")
    return results


if __name__ == "__main__":
    main()
