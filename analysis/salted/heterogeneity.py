"""Task 2: Position-heterogeneity measurement (salted-quantization spike).

Eta-squared (between-salt-class variance / total variance) on dequantized weights,
for salt defs {linear % N, row % N, col % N, (row+col) % N}, N in {4,8,16,32,64}.

Dequantization pin (review item 1):
  We mirror the venv's compressed-tensors exactly by CALLING it for the pin check:
    - compressed_tensors/compressors/pack_quantized/helpers.py:unpack_from_int32
      (pack_factor = 32//num_bits; bits packed along packed dim; signed offset
       2**(num_bits-1) applied after unpacking)
    - compressed_tensors/quantization/lifecycle/forward.py:dequantize
      (GROUP strategy inferred from scale [out, in/128]; symmetric -> w = x_q * scale
       broadcast per group; zero_point absent for symmetric quantization).
  verify_pin() compares our tiled manual unpack+dequant against that library path
  on a real tensor; numbers below are only valid if it reports 0.0.
  Checkpoint scheme (quantization_config.json): INT4 symmetric group-128 (AutoRound).
  lm_head: INT8 symmetric group-128, pack_factor 4 (32//8).

Axis semantics (review item 2):
  Stored Linear weights are W[out_features, in_features]; row = output channel,
  col = input channel, for every sampled tensor:
  - gate_proj: [17408, 5120] -> row = MLP N-axis (17408), col = K (5120)
  - down_proj: [5120, 17408] -> row = out (5120),      col = MLP N-axis (17408)
  - q_proj:    [12288, 5120] -> row = Q heads,         col = hidden (5120)
  - lm_head:   [248320, 5120]-> row = vocab,           col = hidden (5120)
  "row % N" partitions output channels in all cases; the MLP N-axis sits on
  the ROW for gate_proj but on the COLUMN for down_proj.

Run:  venv/bin/python analysis/salted/heterogeneity.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent  # ~/qwen-serving
sys.path.insert(0, str(REPO / "venv" / "lib" / "python3.12" / "site-packages"))

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import dequantize as ct_dequantize
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

CKPT = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
SALTED = REPO / "analysis" / "salted"

SALT_DEFS = ["linear", "row", "col", "row+col"]
NS = [4, 8, 16, 32, 64]
CHUNK_ROWS = 4096  # 4096 x 17408 tiles, float64 sums -> well under 4 GB CPU RAM


# ---------------------------------------------------------------- dequantization
def load_packed(shard_file, base):
    f = safe_open(CKPT / shard_file, framework="pt", device="cpu")
    return (
        f.get_tensor(f"{base}.weight_packed"),
        f.get_tensor(f"{base}.weight_scale"),
        tuple(f.get_tensor(f"{base}.weight_shape").tolist()),
    )


def dequant_chunk(packed_rows, num_bits, n_cols):
    """Manual unpack of an int32 row tile -> int64 values in [0, 2**bits)."""
    factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    p = packed_rows.to(torch.int64).numpy()
    up = np.empty((p.shape[0], n_cols), dtype=np.int64)
    for i in range(factor):
        up[:, i::factor] = (p >> (num_bits * i)) & mask
    return up


def verify_pin(base, shard_file, num_bits):
    """Manual tiled unpack+dequant vs compressed_tensors library path (must be 0)."""
    packed, scale, shape = load_packed(shard_file, base)
    ref = ct_dequantize(
        x_q=unpack_from_int32(packed, num_bits, shape),
        scale=scale.to(torch.float32),
        zero_point=None,  # symmetric: no zero point stored
        args=QuantizationArgs(strategy=QuantizationStrategy.GROUP,
                              group_size=shape[1] // scale.shape[1]),
    )
    gs = shape[1] // scale.shape[1]
    sc = scale.to(torch.float32).repeat_interleave(gs, dim=1).numpy()
    out = np.empty(shape, dtype=np.float32)
    for r0 in range(0, shape[0], CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, shape[0])
        up = dequant_chunk(packed[r0:r1], num_bits, shape[1])
        out[r0:r1] = (up - 2 ** (num_bits - 1)).astype(np.float32) * sc[r0:r1]
    diff = np.abs(out - ref.numpy()).max()
    print(f"  PIN {base}: max|manual - compressed_tensors| = {diff.item():.3e}"
          f"  (|W|.sum={ref.abs().sum().item():.1f}, elems={out.size})")
    # persisted so the citation exists without a re-run (review Important #2)
    return {"tensor": base, "shard": shard_file, "num_bits": num_bits,
            "max_abs_diff": float(diff), "elems": int(out.size),
            "ref_abs_sum": float(ref.abs().sum())}


# ---------------------------------------------------------------- accumulator
class StreamingEta:
    """Exact across chunks: per-class n/sum/sumsq via float64 bincount.
    Welford-equivalent precision; no full-weight copy, no per-chunk averaging."""

    def __init__(self, salt_def, N, cols):
        self.salt_def, self.N, self.cols = salt_def, N, cols
        self.n = np.zeros(N, dtype=np.int64)
        self.s = np.zeros(N, dtype=np.float64)
        self.ss = np.zeros(N, dtype=np.float64)
        self.total = 0

    def add_chunk(self, w, row0):
        rows = w.shape[0]
        c = np.arange(self.cols, dtype=np.int64)[None, :]
        if self.salt_def == "row":
            cls = np.repeat((np.arange(row0, row0 + rows) % self.N)[:, None], self.cols, 1)
        elif self.salt_def == "col":
            cls = np.broadcast_to(c % self.N, w.shape)
        elif self.salt_def == "row+col":
            cls = ((np.arange(row0, row0 + rows)[:, None] + c) % self.N)
        elif self.salt_def == "linear":
            cls = ((np.arange(row0, row0 + rows)[:, None] * self.cols + c) % self.N)
        else:
            raise ValueError(self.salt_def)
        flat = cls.ravel()
        v = w.ravel().astype(np.float64)
        self.n += np.bincount(flat, minlength=self.N)
        self.s += np.bincount(flat, weights=v, minlength=self.N)
        self.ss += np.bincount(flat, weights=v * v, minlength=self.N)
        self.total += flat.size

    def result(self):
        mu = self.s / self.n
        var = self.ss / self.n - mu * mu
        mu_all = self.s.sum() / self.total
        between = ((mu - mu_all) ** 2 * self.n).sum() / self.total
        total = between + (var * self.n).sum() / self.total
        return {"eta_squared": float(between / total), "total_var": float(total),
                "class_mean": mu.tolist(), "class_std": np.sqrt(var).tolist()}


def run_all_salts(shard_file, base, num_bits):
    """Unpack the tensor once; stream chunks into all 20 (salt, N) accumulators."""
    packed, scale, shape = load_packed(shard_file, base)
    gs = shape[1] // scale.shape[1]
    sc = scale.to(torch.float32).repeat_interleave(gs, dim=1).numpy()
    accs = {(sd, n): StreamingEta(sd, n, shape[1]) for sd in SALT_DEFS for n in NS}
    for r0 in range(0, shape[0], CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, shape[0])
        up = dequant_chunk(packed[r0:r1], num_bits, shape[1])
        w = (up - 2 ** (num_bits - 1)).astype(np.float32) * sc[r0:r1]
        for acc in accs.values():
            acc.add_chunk(w, r0)
    return {f"{sd}%{n}": accs[(sd, n)].result() for sd in SALT_DEFS for n in NS}


# ---------------------------------------------------------------- synthetic tests
def synthetic_test():
    print("== Synthetic unit tests (real projection shapes) ==")
    ok = True
    rng = np.random.default_rng(0)
    for rows, cols in [(15360, 5120), (17408, 5120), (5120, 17408)]:
        # structure periodic in row % 8: every 8th row gets its own mean offset
        w = rng.standard_normal((rows, cols)).astype(np.float32)
        w += ((np.arange(rows) % 8) * 0.5)[:, None].astype(np.float32)
        er = StreamingEta("row", 64, cols); er.add_chunk(w, 0)
        ec = StreamingEta("col", 64, cols); ec.add_chunk(w, 0)
        r, c_ = er.result()["eta_squared"], ec.result()["eta_squared"]
        # rows sharing r%64 also share r%8 (8|64) -> between-class info survives at N=64
        good = r > 0.1 and c_ < 0.01
        ok &= good
        print(f"  row-structured {rows}x{cols}: row%64 eta2={r:.4f}  col%64 eta2={c_:.4f}  -> {'PASS' if good else 'FAIL'}")
        w2 = rng.standard_normal((rows, cols)).astype(np.float32)
        worst, worst_key = 0.0, ""
        for sd in SALT_DEFS:
            for n in NS:
                e = StreamingEta(sd, n, cols); e.add_chunk(w2, 0)
                v = e.result()["eta_squared"]
                if v > worst:
                    worst, worst_key = v, f"{sd}%{n}"
        good = worst < 0.01
        ok &= good
        print(f"  iid Gaussian   {rows}x{cols}: worst eta2 over all 20 defs = {worst:.5f} ({worst_key})  -> {'PASS' if good else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--skip-pin", action="store_true")
    args = ap.parse_args()

    if not args.skip_synthetic and not synthetic_test():
        print("SYNTHETIC TESTS FAILED - aborting real run")
        sys.exit(1)

    pin = None
    if not args.skip_pin:
        print("== Dequant pin check (manual tiled vs compressed-tensors library) ==")
        pin = verify_pin("model.language_model.layers.0.mlp.gate_proj",
                         "model-00001-of-00007.safetensors", 4)
        if pin["max_abs_diff"] != 0.0:
            print("PIN FAILED - dequantization does not match library; aborting")
            sys.exit(1)

    # layers 0, 27, 63; q_proj only at 27/63 (layer 0 is a linear-attn layer)
    samples = [
        ("model.language_model.layers.0.mlp.gate_proj", "model-00001-of-00007.safetensors", 4),
        ("model.language_model.layers.0.mlp.down_proj", "model-00001-of-00007.safetensors", 4),
        ("model.language_model.layers.27.self_attn.q_proj", "model-00002-of-00007.safetensors", 4),
        ("model.language_model.layers.27.mlp.gate_proj", "model-00002-of-00007.safetensors", 4),
        ("model.language_model.layers.27.mlp.down_proj", "model-00002-of-00007.safetensors", 4),
        ("model.language_model.layers.63.self_attn.q_proj", "model-00004-of-00007.safetensors", 4),
        ("model.language_model.layers.63.mlp.gate_proj", "model-00004-of-00007.safetensors", 4),
        ("model.language_model.layers.63.mlp.down_proj", "model-00004-of-00007.safetensors", 4),
        ("lm_head", "model-00007-of-00007.safetensors", 8),
    ]

    results = {}
    print("\n== eta^2 on dequantized weights (between-class var / total var) ==")
    for base, shard, bits in samples:
        tkey = base.split("model.language_model.")[-1] if "layers" in base else base
        res = run_all_salts(shard, base, bits)
        results[tkey] = {"desc": base, "num_bits": bits,
                         "salt": {k: v["eta_squared"] for k, v in res.items()}}
        best = max(res.items(), key=lambda kv: kv[1]["eta_squared"])
        # per-class mean/std table for the best salt per tensor (plan Produces
        # item; review Important #1) — feeds Task 4's interpretation
        results[tkey]["best"] = {
            "salt": best[0],
            "eta_squared": best[1]["eta_squared"],
            "total_var": best[1]["total_var"],
            "class_mean": best[1]["class_mean"],
            "class_std": best[1]["class_std"],
        }
        print(f"{tkey:50s} best={best[0]:9s} eta2={best[1]['eta_squared']:.5f}")
        for sd in SALT_DEFS:
            print(f"    {sd:8s}: " + " ".join(f"N{n}={results[tkey]['salt'][f'{sd}%{n}']:.4f}" for n in NS))
        sys.stdout.flush()

    if pin is not None:
        results["_pin"] = pin
    (SALTED / "heterogeneity.json").write_text(json.dumps(results, indent=1))
    print("\nWrote analysis/salted/heterogeneity.json")


if __name__ == "__main__":
    main()
