"""Validate the MoE-internal region collapse (INTEL_CPU_DSV4_MOE_COOP): correctness + decode timing.
Greedy generation of a FIXED prompt; saves output token ids + decode step time. Run twice (MOE_COOP=0 vs 1,
set in env before spawn) and diff the token ids -> identical = bit-faithful collapse; compare step time -> win.
"""
import argparse, json, os, random, statistics, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--config", required=True)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=10); p.add_argument("--reps", type=int, default=3)
    p.add_argument("--mem-fraction", type=float, default=0.80); p.add_argument("--out", default="")
    args = p.parse_args()
    cfg = json.load(open(args.config))
    import sglang as sgl
    engine = sgl.Engine(model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16",
                        load_format="auto", disable_overlap_schedule=True, trust_remote_code=True,
                        mem_fraction_static=args.mem_fraction, log_level="warning", disable_radix_cache=True)
    vocab = cfg["vocab_size"]; rng = random.Random(1234)  # FIXED seed -> same prompt across runs
    ids = [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)] for _ in range(args.batch)]

    def gen(k):
        t0 = time.perf_counter()
        out = engine.generate(input_ids=ids, sampling_params={"max_new_tokens": k, "temperature": 0.0})
        return time.perf_counter() - t0, out

    gen(2)  # warm
    _, out = gen(args.output_len)  # capture output text for correctness compare
    tpref, _ = gen(1)
    dec = [gen(args.output_len)[0] for _ in range(args.reps)]
    td = statistics.median(dec)
    step_ms = (td - tpref) / max(1, args.output_len - 1) * 1e3
    engine.shutdown()
    # output text (first few of each request) for cross-run identity check
    texts = [o.get("text", "")[:80] for o in (out if isinstance(out, list) else [out])][:4]
    res = {"moe_coop": os.environ.get("INTEL_CPU_DSV4_MOE_COOP", "0"),
           "decode_step_ms": round(step_ms, 2), "decode_tok_s": round(args.batch / (step_ms / 1e3), 2),
           "out_texts": texts}
    print(json.dumps(res, indent=2), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
