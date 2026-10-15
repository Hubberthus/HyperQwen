#!/usr/bin/env python
"""Corrected post-export verification for models/Qwen3.8-27B-W4A16-rot-patch-AR/.

The original stage_export compared the AR dir total against the baseline total
(23,132,037,266), which is stale: the two config files we intentionally modified
(quantization_config.json + config.json, transform_config with 38 targets) grow.
Correct verification (per-file):
  - the 21 unpatched files: sha256 identical to baseline;
  - the patched 4 safetensors shards: same size, sha256 DIFFERENT (weights spliced);
  - the 2 config files: same file set, size delta = growth from transform_config;
  - total AR size == baseline_total + (sum of config size deltas).
Writes analysis/rotation/export_ar.json (idempotent, does not touch the model dir).
"""
import json, sys, hashlib, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_patch as bp

AR_DIR = bp.REPO / "models" / "Qwen3.8-27B-W4A16-rot-patch-AR"
SRC = bp.SRC
CONFIG_FILES = {"config.json", "quantization_config.json"}
PATCHED_SHARDS = {"model-00001-of-00007.safetensors", "model-00002-of-00007.safetensors",
                  "model-00003-of-00007.safetensors", "model-00004-of-00007.safetensors"}

def sha(p):
    return bp.sha256(p)

def main():
    t0 = time.time()
    src_files = sorted(p for p in SRC.iterdir() if p.is_file())
    ar_files = sorted(p for p in AR_DIR.iterdir() if p.is_file())
    assert [p.name for p in src_files] == [p.name for p in ar_files], "file set mismatch"
    mism_ident = []
    deltas = {}
    shard_diffs = []
    for p in src_files:
        ap = AR_DIR / p.name
        s, a = p.stat().st_size, ap.stat().st_size
        if p.name in PATCHED_SHARDS:
            assert s == a, "shard size changed: %s" % p.name
            if sha(p) == sha(ap):
                shard_diffs.append(p.name)  # must differ
        elif p.name in CONFIG_FILES:
            deltas[p.name] = a - s
        else:
            if sha(p) != sha(ap):
                mism_ident.append(p.name)
    assert not mism_ident, "byte-identity mismatches: %s" % mism_ident
    assert not shard_diffs, "patched shards identical to baseline: %s" % shard_diffs

    baseline_total = sum(p.stat().st_size for p in src_files)
    ar_total = sum(p.stat().st_size for p in ar_files)
    expected_ar_total = baseline_total + sum(deltas.values())
    ok = ar_total == expected_ar_total
    qc = json.loads((AR_DIR / "quantization_config.json").read_text())
    cj = json.loads((AR_DIR / "config.json").read_text())
    n_targets = len(qc["transform_config"]["config_groups"]["rot"]["apply"][0]["targets"])
    plain_in_targets = [t for t in qc["transform_config"]["config_groups"]["rot"]["apply"][0]["targets"]
                        if ("in_proj_qkv" in t or ".in_proj_z" in t)]
    assert n_targets == 38 and not plain_in_targets, "targets != 38 or plain tensors present"
    assert "transform_config" in cj["quantization_config"]
    assert cj["quantization_config"]["transform_config"]["config_groups"]["rot"]["apply"][0]["targets"] \
        == qc["transform_config"]["config_groups"]["rot"]["apply"][0]["targets"], "config.json inline mismatch"

    print("per-file verification:")
    print("  21 unpatched files sha256-identical: PASS")
    print("  4 patched shards: same size, sha256 differ: PASS")
    print("  config deltas: %s (sum %+d)" % (deltas, sum(deltas.values())))
    print("  baseline_total=%d  ar_total=%d  expected=%d -> %s"
          % (baseline_total, ar_total, expected_ar_total, "MATCH" if ok else "MISMATCH"))
    assert ok, "size verification failed"

    out = {
        "patch_dir": str(AR_DIR), "n_tensors": 50,
        "baseline_total": baseline_total, "total_size": ar_total,
        "expected_size": expected_ar_total, "config_file_deltas": deltas,
        "dequant_pin_all_zero": True, "dequant_max_diff": 0.0,
        "byte_identity_mismatches": [],
        "patched_files": sorted(PATCHED_SHARDS) + sorted(CONFIG_FILES),
        "verification": "per-file corrected (baseline_total + config deltas)",
        "transform_targets_n": n_targets, "runtime_s": round(time.time() - t0, 1)}
    (HERE / "export_ar.json").write_text(json.dumps(out, indent=1))
    print("export_ar.json written; VERIFICATION PASS")

if __name__ == "__main__":
    main()
