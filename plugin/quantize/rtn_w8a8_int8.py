#!/usr/bin/env python3
"""Automated W8A8 INT8 quantizer (calibration-free RTN) — model-agnostic.

Produces the int8 checkpoint SGLang's `--quantization w8a8_int8` CPU/AMX path
consumes (int8 `weight` [out,in] + per-channel fp32 `weight_scale` [out,1];
activations are quantized dynamically per-token at runtime, so no calibration data
is needed). This is the "auto-quantize" step the cpu-model-wiring skill requires to
inherit the donor's INT8 kernel for an unquantized release.

Quantizes every attention/MLP Linear weight per output channel (symmetric):
  scale[o] = max(|W[o,:]|)/127 ; W_int8[o,:] = round(W[o,:]/scale[o]).clamp(-128,127)
Leaves norms/embeddings/lm_head/rotary in original dtype (lm_head -> ignore list).

Usage:
  python rtn_w8a8_int8.py --model <bf16_dir> --out <int8_dir>
Then serve: sglang ... --quantization w8a8_int8 --model-path <int8_dir>
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Linear weights to quantize (HF unfused names); everything else is copied as-is.
TARGET_SUFFIXES = (
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
    ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
)
PACKED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
          "gate_up_proj": ["gate_proj", "up_proj"]}


def quantize_per_channel(w: torch.Tensor):
    w = w.to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0          # [out,1]
    scale = scale.clamp(min=1e-8)
    q = (w / scale).round().clamp(-128, 127).to(torch.int8)    # [out,in]
    return q, scale.to(torch.float32)


def is_target(name: str) -> bool:
    return name.endswith(TARGET_SUFFIXES) and "lm_head" not in name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) copy non-weight files (tokenizer, generation_config, etc.); config.json patched below
    for fn in os.listdir(args.model):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or fn == "config.json":
            continue
        src = os.path.join(args.model, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, fn))

    # 2) patch config.json with the w8a8_int8 quantization_config
    with open(os.path.join(args.model, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "quant_method": "w8a8_int8",
        "is_dynamic": True,
        "ignore": ["lm_head"],
        "packed_modules_mapping": PACKED,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 3) locate shards
    idx_path = os.path.join(args.model, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = [f for f in os.listdir(args.model) if f.endswith(".safetensors")]
        index = None

    new_weight_map = {}
    n_q = 0
    for shard in shards:
        out_tensors = {}
        with safe_open(os.path.join(args.model, shard), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if is_target(name):
                    q, scale = quantize_per_channel(t)
                    out_tensors[name] = q
                    out_tensors[name + "_scale"] = scale
                    new_weight_map[name] = shard
                    new_weight_map[name + "_scale"] = shard
                    n_q += 1
                else:
                    out_tensors[name] = t
                    new_weight_map[name] = shard
        save_file(out_tensors, os.path.join(args.out, shard), metadata={"format": "pt"})
        print(f"wrote {shard}: {len(out_tensors)} tensors")

    # 4) updated index
    if index is not None:
        index["weight_map"] = new_weight_map
        with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    print(json.dumps({"quantized_linears": n_q, "out": args.out,
                      "scheme": "w8a8_int8 per-channel weight, dynamic per-token act"}))


if __name__ == "__main__":
    main()
