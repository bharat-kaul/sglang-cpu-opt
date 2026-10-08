#!/usr/bin/env python3
"""DSv4 wall-time harness for the progressive ledger (model/platform-agnostic).

Uses the CPU-working sglang Engine path (NOT bench_one_batch, which needs pinned
memory on CPU). Per batch size M it measures steady-state DECODE latency per step
(median of >=reps), excluding the cold prefill step, and prefill latency. Random
DISTINCT tokens each call => decorrelated per-token => MoE routing disperses (no
broadcast mode-collapse) and radix cache is disabled so no prefix KV is reused.

Run on-node:
  SGLANG_USE_CPU_ENGINE=1 DETERMINISTIC_DUMMY=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models \\
  python bench_wall_time.py --model <path> --batch-list 1,8,16,32,64 \\
     --input-len 2048 --output-len 8 --reps 3 --out <json>
"""
import argparse
import json
import random
import statistics
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--batch-list", default="1,8,16,32,64")
    p.add_argument("--input-len", type=int, default=2048)
    p.add_argument("--output-len", type=int, default=8, help=">=2; decode steps = output_len-1")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantization", default=None)
    p.add_argument("--load-format", default="dummy")
    p.add_argument("--mem-fraction", type=float, default=0.6)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(f"{args.model}/config.json") as f:
        cfg = json.load(f)
    vocab = cfg["vocab_size"]
    layers = cfg["num_hidden_layers"]
    batches = [int(x) for x in args.batch_list.split(",")]

    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model, device="cpu", tp_size=args.tp, dtype=args.dtype,
        quantization=args.quantization, load_format=args.load_format,
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=args.mem_fraction, log_level="warning",
        disable_radix_cache=True,  # no prefix-KV reuse across calls
    )
    rng = random.Random(0)

    def fresh_ids(m):
        return [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)] for _ in range(m)]

    def gen(max_new, ids):
        sp = {"max_new_tokens": max_new, "temperature": 0.0}
        t0 = time.perf_counter()
        engine.generate(input_ids=ids, sampling_params=sp)
        return time.perf_counter() - t0

    steps = max(args.output_len - 1, 1)
    results = {}
    for m in batches:
        gen(2, fresh_ids(m))  # warmup (compile/first-touch) — never timed
        dec, pre = [], []
        for _ in range(args.reps):
            t_pref = gen(1, fresh_ids(m))                 # prefill + 1 token
            t_full = gen(args.output_len, fresh_ids(m))   # prefill + output_len tokens
            pre.append(t_pref * 1e3)
            dec.append((t_full - t_pref) / steps * 1e3)   # per-step steady-state decode
        results[str(m)] = {
            "decode_ms": round(statistics.median(dec), 3),
            "prefill_ms": round(statistics.median(pre), 3),
            "decode_ms_reps": [round(x, 3) for x in dec],
        }
        print(f"M={m:3d}  decode={results[str(m)]['decode_ms']:.3f} ms/step  "
              f"prefill={results[str(m)]['prefill_ms']:.3f} ms  (reps={args.reps})")

    out = {"model": args.model, "layers": layers, "input_len": args.input_len,
           "output_len": args.output_len, "reps": args.reps, "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", args.out)
    engine.shutdown()


if __name__ == "__main__":
    main()
