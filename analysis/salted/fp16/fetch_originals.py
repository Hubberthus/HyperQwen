"""Task 0: fetch + verify the original (unquantized) Qwen3.8-27B checkpoint.

Downloads Qwen/Qwen3.8-27B at a pinned revision into models-fp16/Qwen3.8-27B/
(gitignored; models/ and the serving path are untouched), then verifies:
  - file inventory: shard count, total bytes, dtype from safetensors headers
  - config: architectures / dtype / tie_word_embeddings (the W4A16 config says
    bfloat16 + tie_word_embeddings=false — confirm on the original)
  - the 9 sampled tensors exist with shapes matching the W4A16 weight_shape
  - cross-check vs the dequant baseline: streaming Pearson correlation + MSE
    of original vs dequantized W4A16 for one full-attention layer's q/gate/down
    (expect corr >= 0.99; lower => wrong repo/revision, abort)

Writes analysis/salted/fp16/fetch.json (every check persisted — the citation
must exist without a re-run).

Run:  venv/bin/python analysis/salted/fp16/fetch_originals.py [--skip-download]
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent.parent  # ~/qwen-serving
SALTED = REPO / "analysis" / "salted"
sys.path.insert(0, str(SALTED))
import codebook_gain as cg      # noqa: E402  (SAMPLES: baseline shard/bits per tensor)
import heterogeneity as hz      # noqa: E402  (W4A16 dequant path for the cross-check)

from huggingface_hub import HfApi, RepoFolder, hf_hub_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

REPO_ID = "Qwen/Qwen3.8-27B"
DEST = REPO / "models-fp16" / "Qwen3.8-27B"
OUT = SALTED / "fp16" / "fetch.json"

# the 9 tensors sampled by the baseline sweep (layers 0/27/63 + lm_head)
SAMPLES = [base for base, _, _ in cg.SAMPLES]


def log(msg):
    print(msg, flush=True)


def orig_chunks(base, wmap):
    """Yield (float32 chunk [r0:r1], row0) from the original checkpoint."""
    shard = wmap[base + ".weight"]
    with safe_open(DEST / shard, framework="pt", device="cpu") as f:
        sl = f.get_slice(base + ".weight")
        rows = sl.get_shape()[0]
        for r0 in range(0, rows, hz.CHUNK_ROWS):
            r1 = min(r0 + hz.CHUNK_ROWS, rows)
            yield sl[r0:r1].to(torch.float32).numpy(), r0


def baseline_chunks(base, shard, bits):
    """Yield (dequantized float32 chunk [r0:r1], row0) from the W4A16 ckpt."""
    packed, scale, shape = hz.load_packed(shard, base)
    gs = shape[1] // scale.shape[1]
    sc = scale.to(torch.float32).repeat_interleave(gs, dim=1).numpy()
    off = 2 ** (bits - 1)
    for r0 in range(0, shape[0], hz.CHUNK_ROWS):
        r1 = min(r0 + hz.CHUNK_ROWS, shape[0])
        up = hz.dequant_chunk(packed[r0:r1], bits, shape[1])
        yield (up - off).astype(np.float32) * sc[r0:r1], r0


def cross_check(base, wmap):
    """Streaming Pearson corr + MSE of original vs dequantized W4A16."""
    shard, bits = next((s, b) for bb, s, b in cg.SAMPLES if bb == base)
    n = sx = sy = sxy = sxx = syy = sse = 0.0
    for (wo, _), (wb, _) in zip(orig_chunks(base, wmap), baseline_chunks(base, shard, bits)):
        x = wo.ravel().astype(np.float64)
        y = wb.ravel().astype(np.float64)
        n += x.size
        sx += x.sum(); sy += y.sum(); sxy += (x * y).sum()
        sxx += (x * x).sum(); syy += (y * y).sum(); sse += ((x - y) ** 2).sum()
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    return {
        "corr": float(cov / np.sqrt(vx * vy)),
        "mse": float(sse / n),
        "rel_mse_vs_orig_var": float((sse / n) / (vx / n)),
        "elems": int(n),
    }


def main():
    skip_dl = "--skip-download" in sys.argv
    OUT.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    sha = api.model_info(REPO_ID).sha
    log(f"repo {REPO_ID} @ {sha}")

    files = []
    for p in sorted(api.list_repo_tree(REPO_ID, revision=sha, recursive=True),
                    key=lambda p: p.path):
        if isinstance(p, RepoFolder):
            continue
        size = p.size if p.size is not None else (p.lfs.size if p.lfs else 0)
        files.append({"path": p.path, "size": size or 0})
    total = sum(f["size"] for f in files)
    log(f"{len(files)} files, {total / 1e9:.1f} GB expected")
    for f in files:
        log(f"  {f['path']}  {f['size'] / 1e9:.3f} GB")

    disk_before = shutil.disk_usage(REPO).free
    log(f"disk free before: {disk_before / 1e9:.1f} GB")

    t0 = time.time()
    if not skip_dl:
        for i, f in enumerate(files):
            dst = DEST / f["path"]
            if f["size"] and dst.exists() and dst.stat().st_size == f["size"]:
                log(f"[{i + 1}/{len(files)}] {f['path']} already present, skipping")
                continue
            hf_hub_download(REPO_ID, f["path"], repo_type="model", revision=sha,
                            local_dir=str(DEST))
            log(f"[{i + 1}/{len(files)}] {f['path']} done ({f['size'] / 1e9:.3f} GB)")
    download_wall_s = round(time.time() - t0, 1)
    disk_after = shutil.disk_usage(REPO).free
    log(f"download wall time: {download_wall_s}s; disk free after: {disk_after / 1e9:.1f} GB")

    # ---- verify: inventory + config
    shards = sorted(DEST.glob("*.safetensors"))
    n_bytes = sum(s.stat().st_size for s in shards)
    cfg = json.loads((DEST / "config.json").read_text())
    wmap = json.loads((DEST / "model.safetensors.index.json").read_text())["weight_map"]
    log(f"on disk: {len(shards)} safetensors shards, {n_bytes / 1e9:.1f} GB")
    log(f"config: arch={cfg.get('architectures')} dtype={cfg.get('dtype')} "
        f"tie_word_embeddings={cfg.get('tie_word_embeddings')} "
        f"(text_config: {cfg.get('text_config', {}).get('tie_word_embeddings')}) "
        f"layers={cfg.get('text_config', {}).get('num_hidden_layers')}")

    # ---- verify: sampled tensors exist, dtype + shape match the W4A16 geometry
    tensor_check = {}
    for base, shard, bits in cg.SAMPLES:
        name = base + ".weight"
        assert name in wmap, f"{name} missing from original index"
        shard_o = wmap[name]
        with safe_open(DEST / shard_o, framework="pt", device="cpu") as f:
            sl = f.get_slice(name)
            shape_o = tuple(sl.get_shape())
            dtype = str(sl.get_dtype())
        shape_b = cg.tensor_shape(shard, base)
        tensor_check[base] = {
            "orig_shard": shard_o, "dtype_on_disk": dtype,
            "shape_orig": list(shape_o), "shape_w4a16": list(shape_b),
            "shape_match": shape_o == tuple(shape_b),
        }
        log(f"  {base}: dtype={dtype} shape={shape_o} "
            f"w4a16={tuple(shape_b)} match={shape_o == tuple(shape_b)}")

    # ---- cross-check vs the dequant baseline (full-attention layer 27)
    xcheck = {}
    for base in ("model.language_model.layers.27.self_attn.q_proj",
                 "model.language_model.layers.27.mlp.gate_proj",
                 "model.language_model.layers.27.mlp.down_proj"):
        xcheck[base] = cross_check(base, wmap)
        x = xcheck[base]
        log(f"  xcheck {base}: corr={x['corr']:.6f} mse={x['mse']:.3e} "
            f"rel_mse={x['rel_mse_vs_orig_var']:.3e}")

    shapes_ok = all(v["shape_match"] for v in tensor_check.values())
    corr_ok = all(v["corr"] >= 0.99 for v in xcheck.values())
    verdict = "PASS" if (shapes_ok and corr_ok) else "FAIL"

    result = {
        "repo": REPO_ID, "revision": sha,
        "files": files, "expected_bytes": total,
        "disk_free_before_gb": round(disk_before / 1e9, 1),
        "disk_free_after_gb": round(disk_after / 1e9, 1),
        "download_wall_s": download_wall_s,
        "shards_on_disk": len(shards), "bytes_on_disk": n_bytes,
        "config": {
            "architectures": cfg.get("architectures"), "dtype": cfg.get("dtype"),
            "tie_word_embeddings": cfg.get("tie_word_embeddings"),
            "text_tie_word_embeddings": cfg.get("text_config", {}).get("tie_word_embeddings"),
            "num_hidden_layers": cfg.get("text_config", {}).get("num_hidden_layers"),
        },
        "tensor_check": tensor_check,
        "cross_check": xcheck,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(result, indent=1))
    log(f"verdict: {verdict} (shapes_ok={shapes_ok}, corr_ok={corr_ok})")
    log(f"wrote {OUT}")
    log("FETCH DONE" if verdict == "PASS" else "FETCH FAILED")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
