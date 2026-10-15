"""Task 0 survey: group-128 absmax outlier structure over ALL INT4 linear tensors.

Source: BF16 originals at models-fp16/Qwen3.8-27B (read-only).
Scope: 400 INT4 linear weights = group 'INT4-linear' of
analysis/salted/inventory.json (all under model.language_model.layers.*).

Per tensor (same grouping the serving checkpoint uses: per-group scale,
group size 128 along the input/in_features dim):
  - group_absmax[g] = max |w| over all rows in column group g (128 cols)
  - spread = max(group_absmax) / median(group_absmax)
  - max_abs_w = global max |w|
  - numel

Method: stream each tensor from its safetensors shard in row tiles
(<= ~150 MB BF16 per tile) so RAM stays bounded; 9 spawn workers.
The block-128/absmax machinery mirrors analysis/salted/rotation/rotation_sim.py
(struct_metrics: reshape (r, c//128, 128) -> absmax over the last axis).

Selection (Step 1 patch experiment, 8 layers):
  MANDATORY anchors L0 (linear-attn), L27 (full-attn), L63 (full-attn)
  + top-5 remaining layers by layer-max spread, with a diversity cap of
  3 fill slots per projection family (mlp = gate/up/down; attn = q/k/v/o
  for full layers, in_proj_qkv/in_proj_z/out_proj for linear-attn layers).

Run:  venv/bin/python analysis/rotation/survey.py
Writes: analysis/rotation/survey.json
"""
import json
import re
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path("/home/hubbe/qwen-serving")
CKPT_FP16 = REPO / "models-fp16" / "Qwen3.8-27B"
SALTED = REPO / "analysis" / "salted"
OUTDIR = Path(__file__).resolve().parent

FULL_ATTN = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63}
N_WORKERS = 9
TILE_BUDGET = 150 * 1024 * 1024  # bytes of BF16 per tile
FILL_SLOTS = 5
FAMILY_CAP = 3

_weight_map = None


def _init(weight_map):
    global _weight_map
    _weight_map = weight_map


def survey_tensor(job):
    """Stream one BF16 tensor in row tiles; group-128 absmax over input dim."""
    name, layer, attn = job
    shard = _weight_map[name]
    with safe_open(CKPT_FP16 / shard, framework="pt", device="cpu") as f:
        sl = f.get_slice(name)
        r, c = sl.get_shape()
        g = c // 128
        assert g * 128 == c, f"{name}: in-dim {c} not divisible by 128"
        rows = max(1, min(8192, TILE_BUDGET // (2 * c)))
        gmax = np.zeros(g, np.float64)
        gmax_all = 0.0
        # per-row group absmax table (r x g entries) for the prior-round
        # rotation_sim-comparable metric (its struct_metrics max over axis=2
        # only, then ravel)
        pr = np.empty(r * g, np.float32)
        off = 0
        for r0 in range(0, r, rows):
            r1 = min(r0 + rows, r)
            t = sl[r0:r1, :]
            a = np.abs(t.to(torch.float32).numpy())  # bf16 has no numpy dtype
            at = a.reshape(r1 - r0, g, 128)
            np.maximum(gmax, at.max(axis=(0, 2)).astype(np.float64), out=gmax)
            pr[off:off + (r1 - r0) * g] = at.max(axis=2).ravel()
            off += (r1 - r0) * g
            m = float(a.max())
            if m > gmax_all:
                gmax_all = m
        med = float(np.median(gmax))
        prm = float(np.median(pr))
        return {
            "name": name,
            "layer": layer,
            "attn_type": attn,
            "shape": [int(r), int(c)],
            "numel": int(r * c),
            "group_absmax_p50": med,
            "group_absmax_max": float(gmax.max()),
            "spread": float(gmax.max() / med) if med > 0 else float("inf"),
            "spread_per_row": float(pr.max() / prm) if prm > 0 else float("inf"),
            "max_abs_w": gmax_all,
        }


def main():
    t0 = time.time()

    inv = json.loads((SALTED / "inventory.json").read_text())
    lw = {t["logical_weight"] for t in inv["tensors"]
          if t["group"] == "INT4-linear"}
    # Inventory stores the packed-triplet logical name for quantized linears
    # (the module name, WITHOUT the .weight suffix); BF16 originals use
    # <module>.weight. Normalize to BF16 param names, then assert every name
    # exists in the BF16 index BEFORE dispatching pool work.
    names = sorted(n if n.endswith(".weight") else n + ".weight" for n in lw)
    assert all(n.startswith("model.language_model.layers.") for n in names), \
        "INT4-linear scope drifted outside model.language_model.layers"

    idx = json.loads(
        (CKPT_FP16 / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [n for n in names if n not in idx]
    assert not missing, (
        f"{len(missing)} survey names missing from BF16 index, "
        f"e.g. {missing[:3]}")

    layer_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
    jobs = []
    for n in names:
        m = layer_re.match(n)
        layer = int(m.group(1))
        attn = "full_attention" if layer in FULL_ATTN else "linear_attention"
        jobs.append((n, layer, attn))
    print(f"survey scope: {len(jobs)} INT4 linear tensors", flush=True)

    ctx = get_context("spawn")
    results = []
    with ctx.Pool(N_WORKERS, initializer=_init, initargs=(idx,)) as pool:
        for i, res in enumerate(
                pool.imap_unordered(survey_tensor, jobs, chunksize=4), 1):
            results.append(res)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} tensors  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda r: (r["layer"], r["name"]))
    n_elem = sum(r["numel"] for r in results)
    print(f"surveyed {len(results)} tensors, {n_elem/1e9:.3f} B elements, "
          f"wall {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------- selection
    by_layer = defaultdict(list)
    for r in results:
        by_layer[r["layer"]].append(r)
    assert len(by_layer) == 64, len(by_layer)
    layer_best = {l: max(t["spread"] for t in ts) for l, ts in by_layer.items()}
    layer_top = {l: max(ts, key=lambda t: t["spread"]) for l, ts in by_layer.items()}

    def family_of(proj):
        return "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "attn"

    mandatory = [0, 27, 63]
    cands = sorted([l for l in layer_best if l not in mandatory],
                   key=lambda l: (-layer_best[l], l))
    rank_of = {l: i + 1 for i, l in enumerate(cands)}

    fill, counts = [], {"mlp": 0, "attn": 0}
    for l in cands:
        if len(fill) == FILL_SLOTS:
            break
        gr = family_of(layer_top[l]["name"].split(".")[-2])
        if counts[gr] < FAMILY_CAP:
            fill.append(l)
            counts[gr] += 1
    assert len(fill) == FILL_SLOTS, fill

    selected = sorted(mandatory + fill)
    rationale = {
        "0": ("MANDATORY anchor: linear-attention (Gated DeltaNet) representative; "
              f"layer-max spread {layer_best[0]:.2f}x "
              f"({layer_top[0]['name'].split('.')[-2]})"),
        "27": ("MANDATORY anchor: full-attention representative; "
               f"layer-max spread {layer_best[27]:.2f}x "
               f"({layer_top[27]['name'].split('.')[-2]})"),
        "63": ("MANDATORY anchor: final full-attention layer; "
               f"layer-max spread {layer_best[63]:.2f}x "
               f"({layer_top[63]['name'].split('.')[-2]}) — down_proj already "
               "benchmarked in rotation_sim (unrotated spread 14.5x)"),
    }
    for i, l in enumerate(fill):
        top = layer_top[l]
        proj = top["name"].split(".")[-2]
        rationale[str(l)] = (
            f"fill slot {i+1}: global rank {rank_of[l]} by layer-max spread "
            f"({layer_best[l]:.2f}x; top projection {proj}, "
            f"{family_of(proj)} family); picked under the diversity cap of "
            f"{FAMILY_CAP} fill slots per family (final fill split: "
            f"mlp {counts['mlp']}, attn {counts['attn']})")

    top10 = sorted(((l, s, layer_top[l]["name"].split(".")[-2])
                    for l, s in layer_best.items()), key=lambda x: -x[1])[:10]

    out = {
        "spec": {
            "source": "models-fp16/Qwen3.8-27B (BF16 originals, read-only)",
            "scope": (f"{len(jobs)} INT4 linear tensors = group 'INT4-linear' of "
                      "analysis/salted/inventory.json, all under "
                      "model.language_model.layers.* (48 linear-attn layers x6 "
                      "projections + 16 full-attn layers x7 projections)"),
            "grouping": ("groups of 128 columns along the input (in_features) dim, "
                          "same grouping as the serving checkpoint "
                          "(per-group scale, group size 128)"),
            "spread": ("max(group_absmax)/median(group_absmax) over the per-group "
                       "(128-col) absmax table: one absmax per group over ALL rows "
                       "- the value the per-group quantizer scale is derived from"),
            "spread_per_row": ("same ratio over the per-row group absmax table "
                               "(r x g entries); the metric the prior-round "
                               "rotation_sim benchmark reported (L63 down_proj "
                               "unrotated: 14.52x) - kept for comparability"),
            "workers": N_WORKERS,
            "streaming": f"row tiles <= {TILE_BUDGET // (1024*1024)} MB BF16",
        },
        "tensors": results,
        "top10_layers_by_spread": [
            {"layer": l, "spread": s, "top_projection": p} for l, s, p in top10],
        "selection": {
            "rule": (f"mandatory anchors L0/L27/L63 + top {FILL_SLOTS} remaining "
                     f"layers by layer-max group-128 spread; diversity cap "
                     f"{FAMILY_CAP} fill slots per projection family "
                     "(mlp = gate/up/down; attn = q/k/v/o and "
                     "in_proj_qkv/in_proj_z/out_proj)"),
            "layers": selected,
            "rationale": rationale,
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUTDIR / "survey.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUTDIR / 'survey.json'}  (wall {time.time() - t0:.0f}s)")
    print("selected layers:", selected)
    print("top10:", [(l, round(s, 2)) for l, s, _ in top10])


if __name__ == "__main__":
    main()
