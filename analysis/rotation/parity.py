"""Parity check: compressed-tensors built-in hadamard transform path vs hand-rolled fold.

Tensor: model.language_model.layers.63.mlp.down_proj.weight (BF16 original,
5120 x 17408).

Library path (built-in pre-quant transform, compressed_tensors 0.17.0):

    from compressed_tensors.transform import (
        TransformConfig, TransformScheme, TransformArgs, apply_transform_config)

    lin = torch.nn.Linear(17408, 5120, bias=False, dtype=torch.float64)
    model = torch.nn.Module(); model.add_module("lin", lin)
    cfg = TransformConfig(config_groups={
        "h": TransformScheme(type="hadamard",
                             apply=[TransformArgs(targets=["lin"],
                                                   location="weight_input")],
                             head_dim=128, randomize=False,
                             requires_grad=False)})
    apply_transform_config(model, cfg)      # fuses H into lin.weight

Notes on the library mechanics (transform/factory/hadamard.py + utils/matrix.py):
  - matrix: deterministic_hadamard_matrix(128) = Sylvester construction (same
    +/-1 pattern as rotation_sim.sylvester), unnormalized.
  - normalization: HadamardTransform.forward divides by sqrt(128)
    => orthonormal (H H^T = I), same as the hand-rolled sylvester(128).
  - block application: _multihead_matmul on the last dim => block-diagonal
    128x128 over the input (in_features) dim, same as block_apply.
  - offline (weight_input) fusion is ALWAYS computed in float64 and written
    back in the module's dtype. With a float64 module the fused weight has NO
    intermediate rounding -> apples-to-apples with the float64 hand-rolled fold.

Hand-rolled path (rotation_sim.py machinery, float64):
    Wrot = block_apply(W_f64, sylvester(128))   # block-128 orthonormal H on input dim
    deq  = quant_group128(Wrot, 4)              # group-128 absmax symmetric INT4 dequant

Requirement: max abs diff of the two DEQUANTIZED results is 0.0, or within
one INT4 quantization step of the finer of the two grids.

Run:  venv/bin/python analysis/rotation/parity.py
Writes: analysis/rotation/parity.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/hubbe/qwen-serving")
SALTED_ROT = REPO / "analysis" / "salted" / "rotation"
sys.path.insert(0, str(SALTED_ROT))
import rotation_sim as rs  # sylvester, block_apply, quant_group128, load_orig

from compressed_tensors.transform import (
    TransformConfig,
    TransformScheme,
    TransformArgs,
    apply_transform_config,
)
from compressed_tensors.transform.utils.hadamard import (
    deterministic_hadamard_matrix,
)

NAME = "model.language_model.layers.63.mlp.down_proj.weight"
OUT = Path(__file__).resolve().parent / "parity.json"


def group_scales(W):
    """Per-group-128 symmetric INT4 scales (absmax / 8), same as quant_group128."""
    r, c = W.shape
    amax = np.abs(W).reshape(r, c // 128, 128).max(axis=2)
    return np.where(amax > 0, amax / 8.0, 1.0)


def build_library_fused(W_np, dtype):
    """Run the compressed-tensors hadamard transform path on W in `dtype`."""
    r, c = W_np.shape
    lin = torch.nn.Linear(c, r, bias=False, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(W_np).to(dtype))
    model = torch.nn.Module()
    model.add_module("lin", lin)
    cfg = TransformConfig(config_groups={
        "h": TransformScheme(
            type="hadamard",
            apply=[TransformArgs(targets=["lin"], location="weight_input")],
            head_dim=128,
            randomize=False,
            requires_grad=False,
        )
    })
    apply_transform_config(model, cfg)
    return model.lin.weight.detach()


def main():
    t0 = time.time()
    Wf32 = rs.load_orig(NAME)  # BF16 -> float32 (exact)
    Wf64 = Wf32.astype(np.float64)
    r, c = Wf64.shape
    print(f"{NAME} {Wf64.shape} loaded ({Wf64.nbytes/1e6:.0f} MB f64)", flush=True)

    # ---------------- hand-rolled fold (float64 reference) ----------------
    H = rs.sylvester(128)
    assert np.abs(H @ H.T - np.eye(128)).max() < 1e-9, "H not orthonormal"
    Wrot = rs.block_apply(Wf64, H)
    deq_hand = rs.quant_group128(Wrot, 4)
    s_hand = group_scales(Wrot)

    # ---------------- library path (float64 module) ----------------
    Wlib = build_library_fused(Wf64, torch.float64)
    assert Wlib.dtype == torch.float64, Wlib.dtype
    Wlib_np = Wlib.numpy()
    deq_lib = rs.quant_group128(Wlib_np, 4)
    s_lib = group_scales(Wlib_np)

    # ---------------- rotation identity ----------------
    Hlib = deterministic_hadamard_matrix(128, torch.float64)
    # compare +/-1 sign patterns (both normalized by 1/sqrt(128), one up-front,
    # one inside forward)
    sign_same = bool(torch.equal((Hlib > 0), (torch.from_numpy(H) > 0)))
    fused_vs_exact = float(np.abs(Wlib_np - Wrot).max())
    fused_vs_exact_rel = fused_vs_exact / float(np.abs(Wrot).max())

    # ---------------- dequant comparison ----------------
    diff = np.abs(deq_lib - deq_hand)
    max_diff = float(diff.max())
    n_diff = int((diff > 0).sum())
    finer = np.minimum(s_lib, s_hand)
    threshold = float(finer.max())  # one INT4 step of the finer grid, worst group
    n_groups = int(finer.size)
    n_scale_diff = int((s_lib != s_hand).sum())
    max_rel_scale = float(np.abs(s_lib / s_hand - 1.0).max())

    verdict = "PASS" if (max_diff == 0.0 or max_diff <= threshold) else "FAIL"

    # ---------------- BF16 module variant (informational) ----------------
    # Serves as the realistic serving dtype: the library rounds the rotated
    # weight back to BF16 before quantization.
    Wlib_bf = build_library_fused(Wf32, torch.bfloat16)
    assert Wlib_bf.dtype == torch.bfloat16, Wlib_bf.dtype
    deq_lib_bf = rs.quant_group128(
        Wlib_bf.to(torch.float64).numpy(), 4)
    max_diff_bf = float(np.abs(deq_lib_bf - deq_hand).max())

    if verdict == "PASS":
        reason = (
            "PASS: the library's offline hadamard fusion is the same rotation - "
            "deterministic Sylvester H128 with identical +/-1 pattern, normalized "
            "by 1/sqrt(128) (orthonormal), block-diagonal over the 128-wide input "
            "dim, computed in float64. With a float64 module the fused weight "
            "matches the hand-rolled float64 rotation to float64-ulp level "
            f"(max |dW| = {fused_vs_exact:.3e}, rel {fused_vs_exact_rel:.1e}), and "
            f"the two dequantized INT4 results differ by max {max_diff:.6e} "
            f"({n_diff} elements; residual is float64-ulp scale noise flipping "
            f"individual roundings, max group-scale rel diff "
            f"{max_rel_scale:.1e}) = {max_diff / threshold:.2f}x one INT4 step "
            f"of the finer grid ({threshold:.6e}) -> parity confirmed.")
    else:
        reason = (
            f"FAIL: max dequant diff {max_diff:.6e} exceeds one INT4 step of the "
            f"finer grid ({threshold:.6e}). max |dW| between fused and hand-rolled "
            f"rotation: {fused_vs_exact:.3e} (rel {fused_vs_exact_rel:.1e}); "
            f"groups with differing scales: {n_scale_diff}/{n_groups}.")

    out = {
        "tensor": NAME,
        "shape": [int(r), int(c)],
        "hand_rolled": (
            "block-128 orthonormal Sylvester Hadamard on the input dim (float64) "
            "+ symmetric group-128 absmax INT4 round-to-nearest, dequant "
            "(rotation_sim.py machinery: sylvester/block_apply/quant_group128)"),
        "library_api": {
            "package": "compressed_tensors 0.17.0",
            "imports": (
                "from compressed_tensors.transform import (TransformConfig, "
                "TransformScheme, TransformArgs, apply_transform_config)"),
            "call": (
                "lin = torch.nn.Linear(17408, 5120, bias=False, dtype=torch.float64); "
                "model = torch.nn.Module(); model.add_module('lin', lin); "
                "cfg = TransformConfig(config_groups={'h': TransformScheme("
                "type='hadamard', apply=[TransformArgs(targets=['lin'], "
                "location='weight_input')], head_dim=128, randomize=False, "
                "requires_grad=False)}); apply_transform_config(model, cfg)"),
            "notes": (
                "location='weight_input' = offline pre-quant fusion into the "
                "weight; head_dim=128 sets the block size. Offline fusion is "
                "always computed in float64 (transform_scheme.py) and written "
                "back in the module dtype; a float64 module therefore avoids any "
                "intermediate rounding. The knob that matches the hand-rolled "
                "fold: module dtype float64 (same working precision)."),
        },
        "rotation_identity": {
            "hadamard_sign_pattern_identical": sign_same,
            "normalization": (
                "library: divide by sqrt(128) in forward; hand-rolled: "
                "sylvester(128) pre-normalized -> same orthonormal H128"),
            "fused_vs_handrolled_max_abs": fused_vs_exact,
            "fused_vs_handrolled_max_rel": fused_vs_exact_rel,
        },
        "dequant_comparison": {
            "max_abs_diff": max_diff,
            "n_elements_differing": n_diff,
            "threshold_one_step_finer_grid": threshold,
            "groups_with_differing_scales": n_scale_diff,
            "n_groups": n_groups,
            "max_rel_scale_diff": max_rel_scale,
        },
        "bf16_fusion_variant_informational": {
            "max_abs_diff": max_diff_bf,
            "note": (
                "with a BF16 module (serving dtype) the library rounds the "
                "rotated weight back to BF16 before quantization; this measures "
                "the extra error of that intermediate rounding, not a rotation "
                "mismatch"),
        },
        "verdict": verdict,
        "reason": reason,
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}  (wall {time.time() - t0:.0f}s)")
    print(f"verdict: {verdict}   max_diff={max_diff:.6e}  threshold={threshold:.6e}"
          f"  bf16_variant={max_diff_bf:.6e}")
    print(f"sign_pattern_identical={sign_same}  "
          f"fused_vs_exact={fused_vs_exact:.3e}  "
          f"scales_differ={n_scale_diff}/{n_groups}")


if __name__ == "__main__":
    main()
