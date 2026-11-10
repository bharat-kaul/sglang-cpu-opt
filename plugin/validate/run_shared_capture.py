#!/usr/bin/env python3
"""DSv4 shared-expert matched-input CAPTURE driver (reviewer step 1-2; ONE engine load, fail-closed).

Loads the CPU engine in the PRODUCTION config (accumulator-on + routed-clamp-on) with the shared-expert
capture hook enabled, then feeds each FROZEN target's exact token IDs (one request at a time, so the
HashTopK token-id binding / _match_final_chunk accumulation is never cross-request contaminated). The hook
captures shared_expert_cpu's EXACT operands + output on the decision chunk, PACKED weights preserved, and
RETURNS the genuine native output (bit-exact vs the captured reconstruction) -> generation is non-perturbed
by construction.

Non-perturbation is witnessed two ways: (1) the hook asserts torch.equal(reconstruction, native) in-situ;
(2) this driver saves the generated token IDs so an offline step can confirm they match the PRODUCTION
accuracy run (same config, no capture) for the same q-ids.

Completeness is FAIL-CLOSED: every (case x capture-layer) .pt must exist at the end, else exit nonzero.

  python run_shared_capture.py --frozen frozen_targets.json --out-prefix /path/cap \
      --layers 3,20,42 --max-new 16
(env: INTEL_CPU_DSV4_CHUNK_ACCUM=1 INTEL_CPU_DSV4_MXFP4_CLAMP=1 INTEL_CPU_DSV4_SHARED_CAPTURE=<prefix>
      INTEL_CPU_DSV4_SHARED_CAPTURE_FROZEN=<frozen> INTEL_CPU_DSV4_SHARED_CAPTURE_LAYERS=<layers>)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

_pkg = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
if _pkg:
    __import__(_pkg)


def _tok_sha12(ids):
    return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()[:12]


def _git():
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                        "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out-prefix", required=True, help="capture .pt prefix; MUST equal INTEL_CPU_DSV4_SHARED_CAPTURE")
    ap.add_argument("--gens-out", required=True, help="json of generated tokens (non-perturbation witness)")
    ap.add_argument("--layers", required=True, help="comma list; MUST equal INTEL_CPU_DSV4_SHARED_CAPTURE_LAYERS")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    args = ap.parse_args()

    # Fail-closed: the driver's --out-prefix / --layers / --frozen MUST match the hook's env (else the hook
    # captures to a different place / different layers than we verify -> silent incomplete).
    env_pref = os.environ.get("INTEL_CPU_DSV4_SHARED_CAPTURE")
    env_froz = os.environ.get("INTEL_CPU_DSV4_SHARED_CAPTURE_FROZEN")
    env_lay = os.environ.get("INTEL_CPU_DSV4_SHARED_CAPTURE_LAYERS")
    if env_pref != args.out_prefix:
        sys.exit(f"[cap] FATAL: --out-prefix {args.out_prefix!r} != INTEL_CPU_DSV4_SHARED_CAPTURE {env_pref!r}")
    if env_froz != args.frozen:
        sys.exit(f"[cap] FATAL: --frozen {args.frozen!r} != INTEL_CPU_DSV4_SHARED_CAPTURE_FROZEN {env_froz!r}")
    if (env_lay or "3") != args.layers:
        sys.exit(f"[cap] FATAL: --layers {args.layers!r} != INTEL_CPU_DSV4_SHARED_CAPTURE_LAYERS {env_lay!r}")
    if os.environ.get("INTEL_CPU_DSV4_CHUNK_ACCUM") != "1" or os.environ.get("INTEL_CPU_DSV4_MXFP4_CLAMP") != "1":
        sys.exit("[cap] FATAL: production config requires INTEL_CPU_DSV4_CHUNK_ACCUM=1 and "
                 "INTEL_CPU_DSV4_MXFP4_CLAMP=1 (matched to the 397690 accuracy run)")

    frozen = json.load(open(args.frozen))
    items = frozen["items"]
    model = frozen["model"]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    import sglang as sgl
    extra = {}
    for ev, kw, cast in (("CONTEXT_LEN", "context_length", int), ("MAX_TOTAL_TOKENS", "max_total_tokens", int),
                         ("SWA_RATIO", "swa_full_tokens_ratio", float)):
        if os.environ.get(ev):
            extra[kw] = cast(os.environ[ev])
    engine = sgl.Engine(model_path=model, device="cpu", tp_size=1, trust_remote_code=True,
                        disable_overlap_schedule=True, disable_cuda_graph=True,
                        chunked_prefill_size=16384,   # one-chunk prefill: whole prompt completes in one chunk
                        mem_fraction_static=args.mem_fraction,
                        log_level=os.environ.get("LOG_LEVEL", "warning"), watchdog_timeout=100000, **extra)

    recs = []
    sp = {"temperature": 0.0, "max_new_tokens": args.max_new}
    for it in items:
        ids = it["token_ids"]
        t0 = time.time()
        out = engine.generate(input_ids=[ids], sampling_params=sp)   # ONE request -> clean per-case binding
        r = out[0] if isinstance(out, list) else out
        mi = r.get("meta_info", {}) if isinstance(r, dict) else {}
        gen_ids = mi.get("output_token_ids") or mi.get("output_ids") or []
        recs.append({"q": it["q"], "gold": it.get("gold"), "n_tokens": it.get("n_tokens", len(ids)),
                     "sha12": _tok_sha12(ids), "text": r.get("text", "") if isinstance(r, dict) else "",
                     "gen_token_ids": list(gen_ids)})
        print(f"[cap] q{it['q']} sha={_tok_sha12(ids)} gen={len(gen_ids)}tok {time.time()-t0:.1f}s", flush=True)

    obj = {"model": model, "git_sha": _git(), "layers": layers, "max_new": args.max_new,
           "config": {"chunk_accum": 1, "mxfp4_clamp": 1}, "n": len(recs), "records": recs}
    tmp = args.gens_out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, args.gens_out)
    print(f"[cap] wrote gens witness -> {args.gens_out}", flush=True)
    engine.shutdown()

    # FAIL-CLOSED completeness: every (case x layer) capture file must exist.
    missing = []
    for it in items:
        tag = _tok_sha12(it["token_ids"])
        for lid in layers:
            p = f"{args.out_prefix}.{tag}.L{lid}.pt"
            if not os.path.exists(p):
                missing.append((it["q"], lid, p))
    if missing:
        for q, lid, p in missing:
            print(f"[cap] MISSING capture q{q} L{lid}: {p}", flush=True)
        sys.exit(f"[cap] FATAL: {len(missing)}/{len(items)*len(layers)} captures missing (fail-closed)")
    print(f"[cap] COMPLETE: {len(items)*len(layers)} captures ({len(items)} cases x {len(layers)} layers)",
          flush=True)


if __name__ == "__main__":
    main()
