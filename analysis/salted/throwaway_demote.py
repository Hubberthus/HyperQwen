"""Probe: throwaway-weight stage 2 — demotion simulation (CPU, in memory only).

Question: if the bottom-K% of group-128 sensitivity groups (per the stage-1
ranking in throwaway_ranking.json) were demoted from INT4 to INT2 symmetric,
how much distortion does that add — raw and sensitivity-weighted — and is it
proportional to K (diffuse) or well below K (favorable headroom)?

Simulation (labeled RTN, NOT AutoRound re-quant):
  For a demoted group, keep the group's EXISTING 4-bit scale s and re-round
  the codes to 2-bit symmetric with that scale:
      c2[i,j] = clip(signed_code[i,j], -2, 2)
      w2[i,j] = c2[i,j] * s_g
  delta = w2 - w4, distortion_raw   = sum delta^2 / sum w4^2
          distortion_sens = sum delta^2 * exposure_j^2 / sum w4^2 * exposure_j^2
  (exposure_j = per-column weight L2-norm proxy, same weaker proxy as
  sensitivity_probe.py — clearly labeled.)
  Note w4/s = signed_code exactly, so RTN-at-existing-scale reduces to
  clipping the integer codes to +-2 — no float division noise involved.
  AutoRound re-quant is unnecessary for a distortion curve; RTN is used.

Metrics per tensor per K in {5, 10, 20} (from throwaway_ranking.json curves):
  raw L2 distortion fraction, sensitivity-weighted distortion fraction,
  bits saved per element = K/100 * (4-2)/4 = K/200.

Gates (both asserted before numbers are trusted):
  1. pin check: manual unpack vs compressed_tensors.unpack_from_int32 == 0
  2. demotion-reconstruction: dequantizing the demoted 2-bit codes
     ((c2_unsigned - 2) * s) must reproduce the simulated weights EXACTLY.

No checkpoint writes; no serving-code changes; everything in memory.
Run:  venv/bin/python analysis/salted/throwaway_demote.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent  # ~/qwen-serving
sys.path.insert(0, str(REPO / "venv" / "lib" / "python3.12" / "site-packages"))

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32  # noqa: E402

CKPT = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
OUT = REPO / "analysis" / "salted" / "throwaway_demote.json"
RANKING = REPO / "analysis" / "salted" / "throwaway_ranking.json"
GROUP = 128
CHUNK = 4096
NUM_BITS = 4
OFFSET = 1 << (NUM_BITS - 1)
TARGETS = [
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.63.mlp.gate_proj",
    "model.language_model.layers.63.mlp.down_proj",
]
K_VALUES = [5, 10, 20]


def find_shard(base):
    shards = sorted(CKPT.glob("*.safetensors"))
    for shard in shards:
        f = safe_open(CKPT / shard, framework="pt", device="cpu")
        if f"{base}.weight_packed" in list(f.keys()):
            shape = tuple(f.get_tensor(f"{base}.weight_shape").tolist())
            return f, f"{base}.weight_packed", f"{base}.weight_scale", shape, shard.name
    raise FileNotFoundError(f"{base}.weight_packed not found")


def manual_unpack(packed_np, num_bits):
    """int32 [rows, n32] -> unsigned codes uint8 [rows, cols] (0 .. 2^b-1)."""
    factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    rows, n32 = packed_np.shape
    out = np.empty((rows, n32 * factor), dtype=np.uint8)
    for i in range(factor):
        out[:, i::factor] = ((packed_np >> (num_bits * i)) & mask).astype(np.uint8)
    return out


def simulate(base, results):
    t0 = time.time()
    f, pk, sk, shape, shard = find_shard(base)
    rows, cols = shape
    assert cols % GROUP == 0
    groups = cols // GROUP
    print(f"\n=== {base} {shape} groups={rows * groups:,} shard={shard}")
    res = {"shape": list(shape), "shard": shard, "n_groups": rows * groups,
           "group_size": GROUP}

    # ---- pass 1: pin check + per-column exposure (sum_i w_ij^2) ----
    pin_max, pin_codes = 0, 0
    exp_acc = np.zeros((groups, GROUP), dtype=np.float64)
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        packed_np = np.asarray(f.get_slice(pk)[start:stop]).astype(np.int32)
        sc = np.asarray(f.get_slice(sk)[start:stop]).astype(np.float32)
        ref = unpack_from_int32(torch.from_numpy(packed_np), NUM_BITS, (stop - start, cols))
        man = manual_unpack(packed_np, NUM_BITS).astype(np.int16) - OFFSET
        d = int(np.abs(man.astype(np.int32) - ref.numpy().astype(np.int32)).max())
        pin_max = max(pin_max, d)
        pin_codes += man.size
        codes3 = manual_unpack(packed_np, NUM_BITS).reshape(-1, groups, GROUP)
        codes_sq = (codes3.astype(np.int32) - OFFSET) ** 2
        sc_sq = (sc ** 2).astype(np.float32)
        exp_acc += (codes_sq.astype(np.float32) * sc_sq[:, :, None]).sum(axis=0, dtype=np.float64)
    assert pin_max == 0, f"unpack pin failed for {base}: max|manual-ct|={pin_max}"
    res["pin_check"] = {"max_abs_diff": 0, "codes_checked": pin_codes}
    exp_sq = exp_acc.astype(np.float32)  # [groups, 128]
    total_w2 = float(exp_acc.sum())      # sum w^2 over tensor
    total_sw2 = float((exp_acc * exp_sq.astype(np.float64)).sum())  # sum w^2 * exp^2

    # ---- pass 2: per-group sensitivity S[i,g]; select bottom-K% groups ----
    S = np.zeros((rows, groups), dtype=np.float32)
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        packed_np = np.asarray(f.get_slice(pk)[start:stop]).astype(np.int32)
        sc = np.asarray(f.get_slice(sk)[start:stop]).astype(np.float32)
        codes3 = manual_unpack(packed_np, NUM_BITS).reshape(-1, groups, GROUP)
        codes_sq = (codes3.astype(np.int32) - OFFSET) ** 2
        sc_sq = (sc ** 2).astype(np.float32)
        S[start:stop] = (codes_sq.astype(np.float32) * exp_sq[None]).sum(
            axis=2, dtype=np.float32) * sc_sq
    Sg = S.reshape(-1)
    order = np.argsort(Sg, kind="stable")  # ascending = smallest sensitivity first
    bottom = {}
    for K in K_VALUES:
        idx = order[: int(round(Sg.size * K / 100))]
        m = np.zeros(Sg.size, dtype=bool)
        m[idx] = True
        bottom[K] = m.reshape(rows, groups)
    res["total_sensitivity"] = float(S.astype(np.float64).sum())

    # ---- pass 3: demotion distortion (RTN at existing scale = code clipping) ----
    # INT2 symmetric signed convention: codes c-2, c in {0..3} -> {-2,-1,0,1};
    # representable set is [-2, 1], asymmetric by one level (2-bit cannot hold
    # {-2..2}); RTN = clip signed code into that set, w2 = c2 * s.
    recon_gate_max = 0
    recon_checked = 0
    rng = np.random.default_rng(0)
    out_k = {}
    for K in K_VALUES:
        mask = bottom[K]
        d2_sum = 0.0
        d2w_sum = 0.0
        for start in range(0, rows, CHUNK):
            stop = min(start + CHUNK, rows)
            packed_np = np.asarray(f.get_slice(pk)[start:stop]).astype(np.int32)
            sc = np.asarray(f.get_slice(sk)[start:stop]).astype(np.float32)
            codes3 = manual_unpack(packed_np, NUM_BITS).reshape(-1, groups, GROUP)
            signed = codes3.astype(np.int16) - OFFSET
            mchunk = mask[start:stop]
            if not mchunk.any():
                continue
            c2 = np.clip(signed, -2, 1)  # RTN into 2-bit symmetric set
            delta = (c2 - signed).astype(np.float32) * sc[:, :, None]
            d2 = (delta ** 2).astype(np.float32)
            d2_sum += float((d2 * mchunk[:, :, None]).sum(dtype=np.float64))
            d2w_sum += float((d2 * exp_sq[None] * mchunk[:, :, None]).sum(dtype=np.float64))
            # gate 2: dequantize 2-bit codes must reproduce simulated weights exactly
            ii, gg = np.nonzero(mchunk)
            if ii.size:
                pick = rng.integers(0, ii.size, size=min(8, ii.size))
                for p in pick:
                    i, g = ii[p], gg[p]
                    c2u = (c2[i, g] + 2).astype(np.uint8)  # unsigned 0..3 codes
                    w2 = (c2u.astype(np.int32) - 2).astype(np.float32) * sc[i, g]
                    ref_w2 = c2[i, g].astype(np.float32) * sc[i, g]
                    e = float(np.abs(w2.astype(np.float64) - ref_w2.astype(np.float64)).max())
                    recon_gate_max = max(recon_gate_max, e)
                    recon_checked += w2.size
        assert recon_gate_max == 0, f"demotion reconstruction failed: {recon_gate_max}"
        out_k[K] = {
            "raw_l2_frac": d2_sum / total_w2,
            "sens_weighted_frac": d2w_sum / total_sw2,
            "bits_saved_per_elem": K / 200,
        }
    res["recon_check"] = {"max_abs_diff": 0, "elements_checked": recon_checked}
    res["k_results"] = out_k
    res["int2_note"] = ("2-bit symmetric signed codes {0..3} -> {-2,-1,0,1}; "
                        "RTN clip uses the group's existing 4-bit scale (no requant).")
    res["runtime_s"] = round(time.time() - t0, 1)
    for K in K_VALUES:
        r = out_k[K]
        print(f"  K={K:>2}%: raw={r['raw_l2_frac']:.5f}  sens={r['sens_weighted_frac']:.5f} "
              f" bits/elem saved={r['bits_saved_per_elem']:.3f}")
    print(f"  pin codes={pin_codes:,} (diff 0); recon checked={recon_checked:,} (diff 0); "
          f"runtime {res['runtime_s']}s")
    results[base] = res
    return res


def main():
    t0 = time.time()
    ranking = json.loads(RANKING.read_text())
    results = {
        "probe": "throwaway-weight stage 2: bottom-K demotion to INT2 (RTN, CPU, memory-only)",
        "date": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CKPT.relative_to(REPO)),
        "definition": (
            "bottom-K% groups by stage-1 group sensitivity (S[i,g], ascending); demote to "
            "INT2 symmetric RTN using the group's existing 4-bit scale: "
            "c2 = clip(signed_code, -2, 1), w2 = c2 * s_g (AutoRound re-quant NOT needed "
            "for a distortion curve). raw = sum delta^2 / sum w^2; sens-weighted = "
            "sum delta^2*exposure^2 / sum w^2*exposure^2; exposure_j = per-column weight "
            "L2 norm (WEAKER proxy, same as sensitivity_probe.py)."),
        "activation_source": "PROXY (weaker): exposure_j = ||W[:,j]||_2, no cached activations.",
        "int2_convention": "2-bit symmetric signed codes {-2,-1,0,1} (unsigned c+2 in 0..3).",
        "k_values": K_VALUES,
        "stage1_ranking_source": str(RANKING.relative_to(REPO)),
        "gates": ["pin unpack == 0", "demotion reconstruction == 0"],
        "tensors": {},
    }
    for base in TARGETS:
        simulate(base, results["tensors"])
    results["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT} (total {results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
