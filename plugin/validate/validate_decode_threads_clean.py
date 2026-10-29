"""CLEAN decode-thread confirm: fast 58-core load, then decode STICKY-capped to T (set once, never restored
-> NO per-forward 58->T->58 resize churn), fresh process per T (no KV-pool-growth drift). Resolves the
resident sweep's two confounds. Threading is controlled by the sbatch env (INTEL_CPU_DSV4_DECODE_THREADS=T
+ _STICKY=1); this driver only loads once and times decode early. One T per process; run T=2/4/8 separately.
"""
import argparse, json, os, random, statistics, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--batches", default="1,32")
    p.add_argument("--input-len", type=int, default=8)
    p.add_argument("--output-len", type=int, default=6)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--mem-fraction", type=float, default=0.80)
    p.add_argument("--out", default="")
    args = p.parse_args()
    cfg = json.load(open(args.config))
    import torch
    import sglang as sgl

    engine = sgl.Engine(model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16",
                        load_format="auto", disable_overlap_schedule=True, trust_remote_code=True,
                        mem_fraction_static=args.mem_fraction, log_level="warning", disable_radix_cache=True)
    vocab = cfg["vocab_size"]
    rng = random.Random(0)
    fresh = lambda B: [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)] for _ in range(B)]

    def run(B, k):
        t0 = time.perf_counter()
        engine.generate(input_ids=fresh(B), sampling_params={"max_new_tokens": k, "temperature": 0.0})
        return time.perf_counter() - t0

    results = {}
    for B in [int(x) for x in args.batches.split(",")]:
        run(B, 2)  # warm (compile/alloc + first sticky set) at this shape
        tpref = run(B, 1)
        dec = [run(B, args.output_len) for _ in range(args.reps)]
        td = statistics.median(dec)
        step_ms = (td - tpref) / args.output_len * 1e3
        results[B] = {"decode_step_ms": round(step_ms, 2),
                      "decode_tok_s": round(B / (step_ms / 1e3), 3),
                      "tpref_s": round(tpref, 3),
                      "samples_s": [round(x, 4) for x in dec]}
        print(f"[CLEAN] B={B}: decode_step={step_ms:.1f} ms  ({results[B]['decode_tok_s']} tok/s)  "
              f"n={args.reps} fresh", flush=True)
    engine.shutdown()
    out = {"cores": os.environ.get("CLEAN_CORES", "?"),
           "decode_threads_env": os.environ.get("INTEL_CPU_DSV4_DECODE_THREADS", "?"),
           "sticky": os.environ.get("INTEL_CPU_DSV4_DECODE_THREADS_STICKY", "0"),
           "torch_threads_launch": torch.get_num_threads(),
           "input_len": args.input_len, "output_len": args.output_len, "reps": args.reps,
           "by_batch": results}
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
