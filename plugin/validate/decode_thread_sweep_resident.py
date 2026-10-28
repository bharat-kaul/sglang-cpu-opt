"""Resident-model decode thread sweep — ONE real-weight load, sweep threads via the control file.

Supersedes the 11-reload sweep (job 393786, which also hit the OMP_PROC_BIND=close/OMP_PLACES=cores
thread-COLLAPSE red herring -> flat 4 tok/s regardless of cores = invalid). Here: load the real model
once, then for each thread count T write T to INTEL_CPU_DSV4_DECODE_THREADS_FILE (the plugin re-reads it
each DECODE forward and calls set_num_threads(T), logging `[DECODE THREADCAP] M=.. threads prev->T` so the
count is CONFIRMED to hold — the runtime-config-tuning mandate). Single socket, physical cores only (pin via
the launcher's numactl --physcpubind=0-57 --membind=0). Decode is the headroom (12-62x below BW); this finds
the decode-thread optimum per batch = the adaptive-in-M policy.

Prefill note: the cap hook is DECODE-only, so prefill runs at the init thread count (the launcher's 58);
prefill-thread tuning, if wanted, needs separate processes (a smaller follow-up).
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
    p.add_argument("--ctl", required=True, help="decode-threads control file (INTEL_CPU_DSV4_DECODE_THREADS_FILE)")
    p.add_argument("--threads", default="8,16,30,45,58", help="comma list of decode thread counts to sweep")
    p.add_argument("--batches", default="1,32")
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=16)
    p.add_argument("--reps", type=int, default=3, help="reps per (T,B); median")
    p.add_argument("--mem-fraction", type=float, default=0.80)
    p.add_argument("--out", default="")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    import torch
    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16", load_format="auto",
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=args.mem_fraction, log_level="warning", disable_radix_cache=True,
    )
    vocab = cfg["vocab_size"]
    rng = random.Random(0)

    def fresh_ids(B):
        return [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)] for _ in range(B)]

    def run(B, max_new):
        ids = fresh_ids(B)
        t0 = time.perf_counter()
        engine.generate(input_ids=ids, sampling_params={"max_new_tokens": max_new, "temperature": 0.0})
        return time.perf_counter() - t0

    def set_threads(T):
        with open(args.ctl, "w") as f:
            f.write(str(T))

    threads = [int(x) for x in args.threads.split(",")]
    batches = [int(x) for x in args.batches.split(",")]

    # launch-time thread count (prefill uses this) for reference
    print(f"[sweep] launch torch.get_num_threads()={torch.get_num_threads()}", flush=True)
    set_threads(threads[-1])
    run(1, 2)  # warmup (compile/alloc) at a mid count

    rows = []
    for B in batches:
        for T in threads:
            set_threads(T)
            run(B, 2)  # warm this T (pool resize + context)
            t_pref = run(B, 1)                      # prefill once (tiny input_len -> cheap; cap is decode-only)
            dec = []
            for _ in range(args.reps):
                t_full = run(B, args.output_len)
                dec.append(max(t_full - t_pref, 1e-6))
            td = statistics.median(dec)
            pf_toks = B * args.input_len / t_pref
            dc_toks = B * args.output_len / td
            rows.append({"threads": T, "batch": B, "prefill_tok_s": round(pf_toks, 1),
                         "decode_tok_s": round(dc_toks, 2), "decode_step_ms": round(td / args.output_len * 1e3, 1)})
            print(f"[sweep] B={B:<3} T={T:<3} decode={dc_toks:7.2f} tok/s  step={td/args.output_len*1e3:7.1f} ms", flush=True)

    engine.shutdown()
    # per-batch optimum
    for B in batches:
        br = [r for r in rows if r["batch"] == B]
        best = max(br, key=lambda r: r["decode_tok_s"])
        print(f"[OPT] B={B}: best decode = {best['decode_tok_s']} tok/s @ {best['threads']} threads "
              f"(single socket, physical only)", flush=True)
    if args.out:
        json.dump({"rows": rows, "node": os.environ.get("SLURMD_NODENAME", "")}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
