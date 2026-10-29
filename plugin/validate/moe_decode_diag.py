"""MoE decode DIAGNOSTIC (one load, high-information): confirm MXFP4/AMX path, active-group row counts,
and MoE thread scaling at B=1 AND B=32 decode shapes. No change to the model — reads the plugin's
[MOE DIAG] (shapes/dtype/amx/active-experts/rows-per-expert) and [MOE ISO] (apply-time vs threads) logs.

Run at full physical cores with NO decode cap so [MOE ISO] spans 4..58 threads; the live decode timing is
irrelevant here (the ISO sweep sets threads explicitly). Decides: is MoE at its BW floor, or does it want
more/fewer threads -> whether there's a credible MoE lever before authoring anything.
"""
import argparse, json, random, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--config", required=True)
    p.add_argument("--input-len", type=int, default=8); p.add_argument("--output-len", type=int, default=4)
    p.add_argument("--mem-fraction", type=float, default=0.80)
    args = p.parse_args()
    cfg = json.load(open(args.config))
    import sglang as sgl
    engine = sgl.Engine(model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16",
                        load_format="auto", disable_overlap_schedule=True, trust_remote_code=True,
                        mem_fraction_static=args.mem_fraction, log_level="warning", disable_radix_cache=True)
    vocab = cfg["vocab_size"]; rng = random.Random(0)
    fresh = lambda B: [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)] for _ in range(B)]
    for B in (1, 32):
        t0 = time.perf_counter()
        engine.generate(input_ids=fresh(B), sampling_params={"max_new_tokens": args.output_len, "temperature": 0.0})
        print(f"[DIAG] B={B} generate wall={time.perf_counter()-t0:.2f}s "
              f"(see [MOE DIAG]/[MOE ISO] above for shapes/active-experts/thread-scaling)", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
