#!/usr/bin/env python3
"""Option 2: TEACHER-FORCED per-position logit capture for CPU-vs-GPU numerical-fidelity parity.

For each frozen token sequence, run ONE forward with return_logprob so sglang reports, at EVERY input
position i, the model's top-k next-token distribution GIVEN the fixed prefix tokens[0..i]. Because the
prefix is the FIXED frozen sequence (not self-generated), both backends are conditioned on IDENTICAL
context at every position -> this removes the greedy-amplification confound entirely (the reason the
free-running paired comparison was uninterpretable). Save the raw per-position top-k so an offline
comparator can do top-1 agreement + logprob deltas and localize the first divergence.

  python capture_logits.py --frozen frozen.json --device cpu --leg cpu --topk 5 \
      --max-seqs 6 --out /path/logits_cpu.json
"""
import argparse
import json
import os
import time

_pkg = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
if _pkg:
    __import__(_pkg)


def _extract_topk(entry):
    """Robust to sglang version: an input_top_logprobs position is a list of (logprob, token_id, ...)."""
    out = []
    if not entry:
        return out
    for t in entry:
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            out.append([float(t[0]), int(t[1])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--leg", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-seqs", type=int, default=6, help="cap # sequences (each is a full forward)")
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frozen = json.load(open(args.frozen))
    items = frozen["items"][:args.max_seqs]
    model = frozen["model"]

    import sglang as sgl
    extra = {}
    for ev, kw, cast in (("CONTEXT_LEN", "context_length", int), ("MAX_TOTAL_TOKENS", "max_total_tokens", int),
                         ("SWA_RATIO", "swa_full_tokens_ratio", float)):
        if os.environ.get(ev):
            extra[kw] = cast(os.environ[ev])
    engine = sgl.Engine(model_path=model, device=args.device, tp_size=args.tp, trust_remote_code=True,
                        disable_overlap_schedule=True, disable_cuda_graph=True,
                        mem_fraction_static=args.mem_fraction, log_level="warning",
                        watchdog_timeout=100000, **extra)

    def _git():
        try:
            import subprocess
            return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                            "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None

    recs = []
    dumped_keys = False
    for it in items:
        ids = it["token_ids"]
        out = engine.generate(input_ids=[ids], sampling_params={"temperature": 0.0, "max_new_tokens": 1},
                              return_logprob=True, logprob_start_len=0, top_logprobs_num=args.topk)
        r = out[0] if isinstance(out, list) else out
        mi = r.get("meta_info", {}) if isinstance(r, dict) else {}
        if not dumped_keys:
            print("[capture] meta_info keys:", list(mi.keys()), flush=True)
            dumped_keys = True
        itop = mi.get("input_top_logprobs") or []
        # per position: top-k [(logprob, token_id)]; argmax token = position's top-1
        pos = [_extract_topk(e) for e in itop]
        argmax = [p[0][1] if p else None for p in pos]
        recs.append({"q": it["q"], "n_tokens": it["n_tokens"], "argmax": argmax,
                     "topk": pos, "input_ids": ids})
        print(f"[capture] q{it['q']}: captured {len(argmax)} positions", flush=True)

    obj = {"leg": args.leg, "device": args.device, "tp": args.tp, "model": model, "topk": args.topk,
           "git_sha": _git(), "n_seqs": len(recs), "records": recs}
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, args.out)
    print(f"[capture] leg={args.leg} wrote {len(recs)} seqs -> {args.out}", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
