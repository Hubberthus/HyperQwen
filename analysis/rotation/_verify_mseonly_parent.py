"""Parent verification: which checkpoint did rot-patch-selective-mse actually copy from?

Provenance fingerprint: for probe tensors, hash the raw tensor bytes in each of the
four checkpoint dirs. in_proj_qkv/z are the discriminator:
  - baseline AutoRound: original unrotated weights
  - rot-patch (naive): rotated weights (spliced in Task 1)
  - rot-patch-AR: option-A re-quant
  - selective-mse copy: MUST equal baseline for in_proj (option B, unspliced)
o_proj L63 discriminator: must equal the naive patch (the only rotated splice variant).
"""
import hashlib
import json
import sys

import safetensors

DIRS = {
    "baseline": "models/Qwen3.8-27B-W4A16-AutoRound",
    "naive": "models/Qwen3.8-27B-W4A16-rot-patch",
    "ar": "models/Qwen3.8-27B-W4A16-rot-patch-AR",
    "mse": "models/Qwen3.8-27B-W4A16-rot-patch-selective-mse",
    "strict": "models/Qwen3.8-27B-W4A16-rot-patch-selective",
}

PROBES = [
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_packed",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
    "model.language_model.layers.0.linear_attn.in_proj_z.weight_packed",
    "model.language_model.layers.38.linear_attn.in_proj_qkv.weight_packed",
    "model.language_model.layers.63.self_attn.o_proj.weight_packed",
    "model.language_model.layers.63.self_attn.o_proj.weight_scale",
    "model.language_model.layers.0.mlp.down_proj.weight_packed",
]


def tensor_hashes(dirpath):
    index = json.load(open(f"{dirpath}/model.safetensors.index.json"))
    weight_map = index["weight_map"]
    out = {}
    shards = sorted({weight_map[k] for k in PROBES})
    for shard in shards:
        with safetensors.safe_open(f"{dirpath}/{shard}", framework="pt") as f:
            for key in PROBES:
                if weight_map.get(key) == shard:
                    t = f.get_tensor(key)
                    h = hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16]
                    out[key] = h
    return out


def main():
    tables = {}
    for name, path in DIRS.items():
        try:
            tables[name] = tensor_hashes(path)
        except Exception as exc:  # noqa: BLE001
            tables[name] = {"ERROR": str(exc)[:200]}

    for probe in PROBES:
        print(f"\n== {probe}")
        for name in DIRS:
            print(f"  {name:9s} {tables[name].get(probe, tables[name].get('ERROR', 'MISSING'))}")

    print("\n== verdicts")
    for probe in PROBES:
        vals = {n: tables[n].get(probe) for n in DIRS}
        m, b, nv, ar = vals["mse"], vals["baseline"], vals["naive"], vals["ar"]
        if m == b:
            v = "MSE==BASELINE"
        elif m == nv:
            v = "MSE==NAIVE  <-- WRONG if in_proj"
        elif m == ar:
            v = "MSE==AR"
        else:
            v = "UNIQUE"
        print(f"  {probe.split('layers.', 1)[-1]:55s} {v}")


if __name__ == "__main__":
    sys.exit(main())
