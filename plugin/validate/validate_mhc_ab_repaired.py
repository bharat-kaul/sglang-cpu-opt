"""REPAIRED MHC A/B — worker-verified, provenance-gated, real-weight decode at the 8-thread operating point.

Fixes the review finding that the prior driver toggled os.environ AFTER sgl.Engine construction, so the
toggles never crossed into the mp.Process worker and BOTH arms ran identically (the A/B was invalid).

Repairs:
  1. WORKER-SIDE CONTROL FILE — env INTEL_CPU_DSV4_MHC_CK_FILE names a file set BEFORE spawn (so the worker
     inherits the path); the client rewrites the file CONTENTS (0=eager / 1=C) per rep; the worker re-reads
     the file EACH forward (_mhc_ck_on in _dsv4_cpu_infra.py), so the toggle actually crosses the boundary.
  2. PROVENANCE GATE — env INTEL_CPU_DSV4_MHC_PROV_FILE names a branch-counter file the worker writes
     ("c=<n> eager=<m>"). After each rep we confirm the EXPECTED arm's counter advanced and the other did
     NOT (reject fallback-contaminated / arm-didn't-switch reps). A rep that fails provenance is discarded.
  3. REAL WEIGHTS — load_format defaults to the real checkpoint (not dummy); timing is still the metric.
  4. Single socket, physical cores, DECODE_THREADS=8 at launch; B=1 and B=32; drift-robust interleaved,
     paired-median, all samples retained.
"""
import argparse, json, os, random, statistics, tempfile, time


def _read_prov(path):
    try:
        with open(path) as f:
            parts = dict(kv.split("=") for kv in f.read().split())
        return int(parts.get("c", 0)), int(parts.get("eager", 0))
    except Exception:  # noqa: BLE001
        return 0, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--config", required=True)
    p.add_argument("--batches", default="1,32"); p.add_argument("--input-len", type=int, default=8)
    p.add_argument("--output-len", type=int, default=6); p.add_argument("--reps", type=int, default=6)
    p.add_argument("--mem-fraction", type=float, default=0.80)
    p.add_argument("--load-format", default="auto", help="auto=real weights; dummy=random (timing-only)")
    p.add_argument("--out", default="")
    args = p.parse_args()
    cfg = json.load(open(args.config))

    # Control + provenance files MUST be named in env BEFORE Engine spawn so the worker inherits the paths.
    tmp = tempfile.mkdtemp(prefix="mhc_ab_")
    ctl = os.path.join(tmp, "mhc_ck.ctl"); prov = os.path.join(tmp, "mhc_prov.txt")
    open(ctl, "w").write("0"); open(prov, "w").write("c=0 eager=0")
    os.environ["INTEL_CPU_DSV4_MHC_CK_FILE"] = ctl
    os.environ["INTEL_CPU_DSV4_MHC_PROV_FILE"] = prov
    os.environ["INTEL_CPU_DSV4_DSA_CKERNEL_CHECK"] = "1"  # per-call parity assert stays armed (fail-closed)

    import sglang as sgl
    engine = sgl.Engine(model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16",
                        load_format=args.load_format, disable_overlap_schedule=True, trust_remote_code=True,
                        mem_fraction_static=args.mem_fraction, log_level="warning", disable_radix_cache=True)
    vocab = cfg["vocab_size"]; rng = random.Random(0)
    fresh = lambda B: [[rng.randrange(3, vocab - 1) for _ in range(args.input_len)] for _ in range(B)]

    def set_arm(on):
        open(ctl, "w").write("1" if on else "0")

    def run_verified(B, k, expect_on):
        """Run one generate; return (seconds, ok, diag) where ok = provenance confirmed the expected arm."""
        c0, e0 = _read_prov(prov)
        set_arm(expect_on)
        ids = fresh(B); t0 = time.perf_counter()
        engine.generate(input_ids=ids, sampling_params={"max_new_tokens": k, "temperature": 0.0})
        dt = time.perf_counter() - t0
        c1, e1 = _read_prov(prov)
        if expect_on:
            ok = (c1 > c0) and (e1 == e0)   # C advanced, NO eager fallback
        else:
            ok = (e1 > e0) and (c1 == c0)   # eager advanced, C never fired
        diag = f"arm={'C' if expect_on else 'eager'} c:{c0}->{c1} eager:{e0}->{e1}"
        return dt, ok, diag

    results = {}
    for B in [int(x) for x in args.batches.split(",")]:
        # warm both arms (and surface any parity assertion early, fail-closed)
        for on in (False, True):
            _, okw, dw = run_verified(B, 2, on)
            print(f"[WARM] B={B} {dw} {'OK' if okw else 'NOT-CONFIRMED'}", flush=True)
        set_arm(False); tpref, _, _ = run_verified(B, 1, False)
        eager, cwin, bad = [], [], 0
        for i in range(args.reps):
            order = [False, True] if i % 2 == 0 else [True, False]
            for on in order:
                dt, ok, diag = run_verified(B, args.output_len, on)
                if not ok:
                    bad += 1
                    if bad <= 2:
                        print(f"[BAD] B={B} rep{i} {diag}", flush=True)
                    continue
                (cwin if on else eager).append(dt)
        if not eager or not cwin:
            results[B] = {"error": "no provenance-confirmed reps", "bad_reps": bad}
            print(f"[AB] B={B}: FAILED — no provenance-confirmed reps (bad={bad})", flush=True)
            continue
        em, cm = statistics.median(eager), statistics.median(cwin)
        de = (em - tpref) / args.output_len * 1e3; dc = (cm - tpref) / args.output_len * 1e3
        results[B] = {"eager_step_ms": round(de, 2), "cwin_step_ms": round(dc, 2),
                      "gain_pct": round(100 * (de / dc - 1), 2), "n_eager": len(eager),
                      "n_c": len(cwin), "bad_reps": bad,
                      "eager_samples": [round(x, 4) for x in eager],
                      "c_samples": [round(x, 4) for x in cwin]}
        print(f"[AB] B={B}: eager {de:.2f} -> C {dc:.2f} ms/step = +{results[B]['gain_pct']}%  "
              f"(n={len(eager)}/{len(cwin)} confirmed, {bad} discarded)", flush=True)
    engine.shutdown()
    out = {"operating_point": {"threads": os.environ.get("DECODE_THREADS", "?"), "socket": 1,
                               "load_format": args.load_format, "input_len": args.input_len,
                               "output_len": args.output_len, "reps": args.reps},
           "by_batch": results}
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
