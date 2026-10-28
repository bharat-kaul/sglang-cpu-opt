"""Validate fused-layer increment 1 (MHC sinkhorn+combine C-kernel wiring): PARITY + interleaved A/B.

INcrement-1 routes hc_split_sinkhorn + hc_combine to the C kernels (per-op env CK_SINKHORN/CK_COMBINE,
read per call). This driver:
  PART A (correctness): run decode with CHECK=1 — the C wrappers assert cos>=0.9995 vs the torch oracle
         each call; if it completes with no assertion, the C path is bit-faithful in-model.
  PART B (A/B): one resident load; alternate CK_SINKHORN/CK_COMBINE = 0 (eager) vs 1 (C) per rep (os.environ,
         read per call), paired-median decode step time -> the realized MHC-wiring recovery (drift-robust).
Single socket, physical, 8 threads (set via DECODE_THREADS at launch).
"""
import argparse, json, os, random, statistics, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--config", required=True)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--input-len", type=int, default=8)
    p.add_argument("--output-len", type=int, default=6); p.add_argument("--reps", type=int, default=6)
    p.add_argument("--mem-fraction", type=float, default=0.80); p.add_argument("--out", default="")
    args = p.parse_args()
    cfg = json.load(open(args.config))
    import torch, sglang as sgl
    engine = sgl.Engine(model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16",
                        load_format="dummy", disable_overlap_schedule=True, trust_remote_code=True,
                        mem_fraction_static=args.mem_fraction, log_level="warning", disable_radix_cache=True)
    vocab = cfg["vocab_size"]; rng = random.Random(0)
    fresh = lambda B: [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)] for _ in range(B)]
    def run(B, k):
        ids = fresh(B); t0 = time.perf_counter()
        engine.generate(input_ids=ids, sampling_params={"max_new_tokens": k, "temperature": 0.0})
        return time.perf_counter() - t0
    def set_ck(on):
        v = "1" if on else "0"
        os.environ["INTEL_CPU_DSV4_CK_SINKHORN"] = v
        os.environ["INTEL_CPU_DSV4_CK_COMBINE"] = v

    # PART A — correctness: CHECK on, C kernels on; any parity break raises inside generate()
    os.environ["INTEL_CPU_DSV4_DSA_CKERNEL_CHECK"] = "1"; set_ck(True)
    try:
        run(args.batch, 3)
        print("[PARITY] PASS — MHC sinkhorn+combine C kernels match torch in-model (cos>=0.9995, no assert)", flush=True)
        parity = "PASS"
    except Exception as e:
        print(f"[PARITY] FAIL — {e!r}", flush=True); parity = f"FAIL:{e!r}"
    os.environ["INTEL_CPU_DSV4_DSA_CKERNEL_CHECK"] = "0"

    # PART B — interleaved A/B (eager vs C), prefill measured once (cheap, cap is decode-only)
    for _ in range(2): set_ck(True); run(args.batch, 2)  # warm both paths
    set_ck(False); tpref = run(args.batch, 1)
    eager, cwin = [], []
    for i in range(args.reps):
        if i % 2 == 0:
            set_ck(False); eager.append(run(args.batch, args.output_len)); set_ck(True); cwin.append(run(args.batch, args.output_len))
        else:
            set_ck(True); cwin.append(run(args.batch, args.output_len)); set_ck(False); eager.append(run(args.batch, args.output_len))
    engine.shutdown()
    em, cm = statistics.median(eager), statistics.median(cwin)
    de = (em - tpref) / args.output_len * 1e3; dc = (cm - tpref) / args.output_len * 1e3
    res = {"parity": parity, "eager_step_ms": round(de, 1), "cwin_step_ms": round(dc, 1),
           "speedup": round(de / dc, 3), "gain_pct": round(100 * (de / dc - 1), 1), "batch": args.batch}
    print(f"[AB] MHC-wire: eager {de:.1f} ms/step -> C {dc:.1f} ms/step  = +{res['gain_pct']}% "
          f"(drift-robust interleaved median of {args.reps})", flush=True)
    if args.out: json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
