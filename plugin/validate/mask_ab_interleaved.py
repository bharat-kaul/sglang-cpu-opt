"""Interleaved A/B for the vectorized vs loop coverage mask — drift-robust.

The 3-sequential-legs job (393502) showed the mask region 35x faster (8.7s -> 0.25s) and PROVEN
bit-exact, but its end-to-end prefill tok/s was confounded by thermal/run-order drift (the middle
leg, which did MORE work, was faster than the first; the last leg was slowest). This driver removes
that confound: ONE engine load, then N prefill reps ALTERNATING the mask path per rep
(INTEL_CPU_DSV4_MASK_LOOP toggled in os.environ, which _batched_mla_attend reads per call), so any
drift hits both arms equally. Reports the paired median prefill latency and the realized speedup.

Run via run_dsv4_mask_ab_interleaved.sbatch (needs the sglang venv + plugin env).
"""
import argparse
import json
import os
import random
import statistics
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--reps", type=int, default=12, help="reps per arm (total forwards = 2*reps + warmups)")
    p.add_argument("--mem-fraction", type=float, default=0.70)
    p.add_argument("--out", default="")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16",
        load_format="auto", disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=args.mem_fraction, log_level="warning", disable_radix_cache=True,
    )
    vocab = cfg["vocab_size"]
    rng = random.Random(0)

    def fresh_ids():
        return [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)] for _ in range(args.batch)]

    def one_prefill(use_loop):
        if use_loop:
            os.environ["INTEL_CPU_DSV4_MASK_LOOP"] = "1"
        else:
            os.environ.pop("INTEL_CPU_DSV4_MASK_LOOP", None)
        ids = fresh_ids()
        t0 = time.perf_counter()
        engine.generate(input_ids=ids, sampling_params={"max_new_tokens": 1, "temperature": 0.0})
        return time.perf_counter() - t0

    # warmup BOTH paths (compile/alloc) before timing
    for _ in range(2):
        one_prefill(True)
        one_prefill(False)

    loop_t, vec_t = [], []
    # alternate, and alternate which goes first each pair to cancel any residual ordering bias
    for i in range(args.reps):
        if i % 2 == 0:
            loop_t.append(one_prefill(True)); vec_t.append(one_prefill(False))
        else:
            vec_t.append(one_prefill(False)); loop_t.append(one_prefill(True))

    engine.shutdown()
    toks = args.batch * args.input_len
    lm, vm = statistics.median(loop_t), statistics.median(vec_t)
    res = {
        "reps": args.reps, "batch": args.batch, "input_len": args.input_len,
        "loop_prefill_ms_median": round(lm * 1e3, 1), "vec_prefill_ms_median": round(vm * 1e3, 1),
        "loop_tok_s": round(toks / lm, 1), "vec_tok_s": round(toks / vm, 1),
        "speedup": round(lm / vm, 3), "gain_pct": round(100 * (lm / vm - 1), 1),
        "loop_ms_all": [round(x * 1e3, 1) for x in loop_t],
        "vec_ms_all": [round(x * 1e3, 1) for x in vec_t],
    }
    print("[MASK A/B]", json.dumps({k: v for k, v in res.items() if not k.endswith("_all")}, indent=2))
    print(f"  REALIZED prefill: loop {res['loop_tok_s']} -> vec {res['vec_tok_s']} tok/s "
          f"(+{res['gain_pct']}%, drift-robust interleaved median of {args.reps} paired reps)")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
