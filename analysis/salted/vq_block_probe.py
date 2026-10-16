"""Probe 6 — Vector Quantization feasibility on weight codes (codec-analogy line).

Scalar-pattern probes (spatial delta, cross-layer delta, sensitivity demotion)
all came back NO-GO; this probe tests whether BLOCK structure exists instead of
scalar structure. Block = (row, 4 adjacent 128-wide groups) = 512 int4 codes.

Per tensor (gate_proj + down_proj at model.language_model.layers.0 and .63,
i.e. 4 MLP variants):
  1. pin check: manual bit-shift unpack == compressed_tensors.unpack_from_int32.
  2. k-means codebooks over blocks (k = 256 and 1024, ~50k blocks sampled,
     block vectors = int4 codes as int8, signed). Report residual MSE fit and
     the fraction of mean block energy a codebook explains.
  3. Random baseline: globally shuffle the sampled codes (destroys spatial
     structure, keeps the exact marginal code distribution), rebuild blocks,
     same k-means. Real residual ~= shuffled residual => no block structure.
  4. Residual entropy H(residual bin | codebook index) with 16 equal-frequency
     residual bins (4 bits) vs the 4-bit scalar baseline H(code marginal).

Gate: pin check; fixed seeds; sample sizes reported.

Run: venv/bin/python analysis/salted/vq_block_probe.py   (budget 20 min CPU)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "venv" / "lib" / "python3.12" / "site-packages"))
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402

CKPT = REPO / "models" / "Qwen3.8-27B-W4A16-AutoRound"
OUT = REPO / "analysis" / "salted" / "vq_block_probe.json"
GROUP = 128
BLOCK_GROUPS = 4
BLOCK = GROUP * BLOCK_GROUPS  # 512
SAMPLE_BLOCKS = 50_000
KS = [256, 1024]
RES_BINS = 16
SEED = 0

_INDEX = None
_FILES = {}


def get_file(key):
    global _INDEX
    if _INDEX is None:
        _INDEX = json.load(open(CKPT / "model.safetensors.index.json"))["weight_map"]
    path = CKPT / _INDEX[key]
    if path not in _FILES:
        _FILES[path] = safe_open(path, framework="pt", device="cpu")
    return _FILES[path]


def unpack_codes(base):
    """Manual unpack of packed int32 -> signed int4 codes (-8..7), shape [rows, cols]."""
    packed_key = base + ".weight_packed"
    shape_key = base + ".weight_shape"
    f = get_file(packed_key)
    packed = f.get_tensor(packed_key)
    shape = tuple(f.get_tensor(shape_key).tolist())
    ref = unpack_from_int32(packed, 4, shape)
    rows, cols = shape
    p = packed.to(torch.int64).numpy()
    up = np.empty((rows, cols), dtype=np.int64)
    for i in range(8):
        up[:, i::8] = (p >> (4 * i)) & 15
    diff = int(np.abs((up - 8) - ref.numpy()).max())
    return (up - 8).astype(np.int8), diff, shape


def build_blocks(codes):
    """Non-overlapping blocks: [rows, groups, 512] -> [n_blocks, 512]."""
    rows, cols = codes.shape
    assert cols % GROUP == 0
    g = codes.reshape(rows, cols // GROUP, GROUP)
    assert g.shape[1] % BLOCK_GROUPS == 0
    b = g.reshape(rows, g.shape[1] // BLOCK_GROUPS, BLOCK)
    return b.reshape(-1, BLOCK)


def kmeans_blocks(X, k, rng):
    mbk = MiniBatchKMeans(
        n_clusters=k, random_state=int(rng.randint(0, 2**31 - 1)),
        n_init=3, batch_size=2048, max_iter=100, max_no_improvement=10,
        compute_labels=True)
    mbk.fit(X)
    labels = mbk.predict(X)
    resid = X - mbk.cluster_centers_[labels]
    energy = (resid.astype(np.float64) ** 2).sum(axis=1)
    return mbk.cluster_centers_, labels, energy


def cond_residual_entropy(resid, labels, k, edges):
    """H(res_bin | centroid) with 16 equal-frequency global residual bins."""
    flat = resid.ravel()
    bins = np.searchsorted(edges, flat, side="right") - 1
    bins = np.clip(bins, 0, RES_BINS - 1)
    idx = np.repeat(labels, BLOCK).astype(np.int64)
    joint = np.bincount(bins * k + idx, minlength=RES_BINS * k).astype(np.float64)
    joint = joint.reshape(RES_BINS, k)
    n = joint.sum()
    joint /= n
    H_joint = -(joint[joint > 0] * np.log2(joint[joint > 0])).sum()
    pm = joint.sum(axis=0)
    nzp = pm[pm > 0]
    H_pred = -(nzp / n * np.log2(nzp / n)).sum()
    return float(H_joint - H_pred)


def run_tensor(fam, layer):
    base = f"model.language_model.layers.{layer}.mlp.{fam}"
    t0 = time.time()
    codes, pin_diff, shape = unpack_codes(base)
    blocks = build_blocks(codes)
    rng = np.random.RandomState(SEED)
    n_blocks = blocks.shape[0]
    sel = rng.choice(n_blocks, size=min(SAMPLE_BLOCKS, n_blocks), replace=False)
    X = blocks[sel].astype(np.float32)

    # scalar 4-bit baseline entropy of the code marginal (on the sample)
    flat_codes = X.ravel().astype(np.int64)
    cnt = np.bincount(flat_codes + 8, minlength=16).astype(np.float64)
    p = cnt / cnt.sum()
    H_scalar = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    mean_energy = float((X.astype(np.float64) ** 2).sum(axis=1).mean())

    out = {
        "base": base, "shape": list(shape), "pin_max_diff": pin_diff,
        "total_blocks": int(n_blocks), "sampled_blocks": int(X.shape[0]),
        "mean_block_l2sq": mean_energy, "scalar_code_entropy_bits": H_scalar,
        "ks": {},
    }
    print(f"  {base} shape={shape} pin={pin_diff} blocks={n_blocks} "
          f"sample={X.shape[0]} H_scalar={H_scalar:.4f} ({time.time()-t0:.0f}s)")

    # shuffled baseline: globally permute codes (kills spatial structure,
    # preserves marginal exactly), rebuild the same block geometry
    shuffled = rng.permutation(X.ravel()).reshape(X.shape).astype(np.float32)

    for tag, data in [("real", X), ("shuffled", shuffled)]:
        for k in KS:
            t1 = time.time()
            cent, labels, resid_energy = kmeans_blocks(data, k, rng)
            frac = 1.0 - resid_energy.mean() / mean_energy
            # residual conditional entropy (real data residuals only reported
            # for 'real'; for 'shuffled' it characterises the baseline fit)
            flat_resid = (data - cent[labels]).ravel()
            edges = np.quantile(flat_resid, np.linspace(0, 1, RES_BINS + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            bins = np.clip(np.searchsorted(edges, flat_resid, side="right") - 1, 0, RES_BINS - 1)
            idx = np.repeat(labels, BLOCK)
            joint = np.bincount(bins * k + idx, minlength=RES_BINS * k).astype(np.float64)
            joint = joint.reshape(RES_BINS, k)
            n = joint.sum()
            joint /= n
            Hj = -(joint[joint > 0] * np.log2(joint[joint > 0])).sum()
            pm = joint.sum(axis=0)
            pm = pm[pm > 0]
            Hp = -(pm * np.log2(pm)).sum()
            Hcond = float(Hj - Hp)
            out["ks"][f"{tag}_k{k}"] = {
                "resid_mse": float(resid_energy.mean() / BLOCK),
                "explained_fraction": float(frac),
                "resid_cond_entropy_bits": Hcond,
                "index_bits_per_weight": float(np.log2(k) / BLOCK),
            }
            print(f"    {tag} k={k}: resid_mse={out['ks'][f'{tag}_k{k}']['resid_mse']:.4f} "
                  f"explained={frac:.4f} H(res|idx)={Hcond:.4f} ({time.time()-t1:.0f}s)")
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def main():
    t0 = time.time()
    results = {"probe": "vq_block_probe", "seed": SEED, "block": BLOCK,
               "sample_blocks": SAMPLE_BLOCKS, "res_bins": RES_BINS,
               "tensors": {}}
    for layer in [0, 63]:
        for fam in ["gate_proj", "down_proj"]:
            key = f"L{layer}.{fam.split('_')[0]}"
            results["tensors"][key] = run_tensor(fam, layer)
    results["runtime_s_total"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT} ({results['runtime_s_total']}s)")


if __name__ == "__main__":
    main()
