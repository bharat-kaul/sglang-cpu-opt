#!/usr/bin/env python3
"""Generic end-to-end throughput + in-engine compute-efficiency bench (model-agnostic).

Measures prefill and decode tokens/s for a model served on SGLang CPU, and derives
the achieved linear-GEMM TFLOP/s during prefill (where GEMMs dominate) from the HF
config, reporting eff_abs against the streamed-GEMM achievable ceiling. This is the
in-engine complement to peer_roofline.py's raw-kernel check: it proves the WIRED
model reaches the expected compute efficiency, not just the isolated kernel.

Run on-node (SGLang spawns subprocesses, so this must be a real script file):
  SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models \\
  srun ... python throughput_bench.py --model <path> --tp 6 \\
       --batch 8 --input-len 512 --output-len 64 --ceiling-tflops 121.2
"""

import argparse
import json
import random
import time


def linear_flops_per_token(cfg):
    h = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim", h // nh)
    inter = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    n_exp = cfg.get("num_experts") or cfg.get("num_local_experts") or 0
    topk = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token") or 0
    attn = 2 * h * (nh + 2 * nkv) * hd + 2 * (nh * hd) * h  # qkv + o
    if n_exp and topk:
        moe_inter = cfg.get("moe_intermediate_size") or inter
        mlp = topk * (2 * h * (2 * moe_inter) + 2 * moe_inter * h) + 2 * h * n_exp  # active experts + router
    else:
        mlp = 2 * h * (2 * inter) + 2 * inter * h  # dense gate_up + down
    return (attn + mlp) * layers + 2 * h * vocab  # + lm_head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True, help="HF config.json for FLOP accounting")
    p.add_argument("--tp", type=int, default=6)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--ceiling-tflops", type=float, default=0.0)
    p.add_argument("--mem-fraction", type=float, default=0.5)
    p.add_argument("--dtype", default="bfloat16", help="CPU AMX compute dtype")
    p.add_argument("--quantization", default=None, help="e.g. w8a8_int8")
    p.add_argument("--out", default="")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    flop_tok = linear_flops_per_token(cfg)

    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model, device="cpu", tp_size=args.tp, dtype=args.dtype,
        quantization=args.quantization,
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=args.mem_fraction, log_level="warning",
        disable_radix_cache=True,  # else identical prefills reuse cached KV
    )
    vocab = cfg["vocab_size"]
    rng = random.Random(0)

    def fresh_ids():
        # fresh tokens each call so no run reuses another's prefix-cached KV
        return [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)]
                for _ in range(args.batch)]

    def run(max_new, ids):
        sp = {"max_new_tokens": max_new, "temperature": 0.0}
        t0 = time.perf_counter()
        engine.generate(input_ids=ids, sampling_params=sp)
        return time.perf_counter() - t0

    run(1, fresh_ids())  # warmup
    t_pref = run(1, fresh_ids())
    t_full = run(args.output_len, fresh_ids())

    prefill_tokens = args.batch * args.input_len
    decode_tokens = args.batch * args.output_len
    decode_time = max(t_full - t_pref, 1e-6)

    prefill_tok_s = prefill_tokens / t_pref
    decode_tok_s = decode_tokens / decode_time
    prefill_tflops = prefill_tokens * flop_tok / t_pref / 1e12
    eff_abs = round(prefill_tflops / args.ceiling_tflops, 3) if args.ceiling_tflops else None

    engine.shutdown()

    result = {
        "model": args.model, "tp": args.tp, "batch": args.batch,
        "input_len": args.input_len, "output_len": args.output_len,
        "linear_gflops_per_token": round(flop_tok / 1e9, 2),
        "prefill_tok_s": round(prefill_tok_s, 1),
        "decode_tok_s": round(decode_tok_s, 1),
        "prefill_achieved_tflops": round(prefill_tflops, 1),
        "streamed_ceiling_tflops": args.ceiling_tflops or None,
        "prefill_eff_abs": eff_abs,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
