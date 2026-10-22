#!/usr/bin/env python3
"""P1 (GSM8K review): run a FROZEN token-ID set through one backend leg and save enriched, paired
per-question records. Legs share identical inputs (engine.generate(input_ids=...)) so GPU / CPU-eager
/ CPU-C are directly comparable. On CPU, --ck-sweep 0,1 runs BOTH eager and C on ONE resident load
(held-fixed) via the CK control file.

  # CPU leg (both eager+C on one load):
  INTEL_CPU_DSV4_CK_CTL_FILE=ck.ctl python paired_eval.py --frozen frozen.json --device cpu \
      --leg cpu --ck-sweep 0,1 --max-new 512 --out-prefix /path/paired_cpu
  # GPU reference leg:
  python paired_eval.py --frozen frozen.json --device cuda --tp 4 --leg gpu --max-new 512 \
      --out-prefix /path/paired_gpu
"""
import argparse
import json
import os
import time

_pkg = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
if _pkg:
    __import__(_pkg)

from task_gsm8k import INVALID, extract_final_answer


def _atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--leg", required=True, help="tag: gpu | cpu")
    ap.add_argument("--ck-sweep", default="", help="CPU: comma list of CK toggle values (0=eager,1=C) run on ONE load")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    frozen = json.load(open(args.frozen))
    items = frozen["items"]
    model = frozen["model"]

    import sglang as sgl
    extra = {}
    for ev, kw, cast in (("CONTEXT_LEN", "context_length", int), ("MAX_TOTAL_TOKENS", "max_total_tokens", int),
                         ("SWA_RATIO", "swa_full_tokens_ratio", float), ("MAX_RUNNING", "max_running_requests", int)):
        if os.environ.get(ev):
            extra[kw] = cast(os.environ[ev])
    engine = sgl.Engine(model_path=model, device=args.device, tp_size=args.tp, trust_remote_code=True,
                        disable_overlap_schedule=True, disable_cuda_graph=True,
                        mem_fraction_static=args.mem_fraction, log_level="warning",
                        watchdog_timeout=100000, **extra)
    sp = {"temperature": 0.0, "max_new_tokens": args.max_new}

    ck_ctl = os.environ.get("INTEL_CPU_DSV4_CK_CTL_FILE", "")
    ck_vals = [v.strip() for v in args.ck_sweep.split(",") if v.strip()] or [None]

    def _git():
        try:
            import subprocess
            return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                            "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None

    for ckv in ck_vals:
        if ckv is not None and ck_ctl:
            with open(ck_ctl, "w") as f:
                f.write(str(ckv))
        tag = args.leg if ckv is None else f"{args.leg}_ck{ckv}"
        t0 = time.perf_counter()
        # BATCH all items in one generate call -> M=N decode (independent greedy requests give identical
        # per-question output to sequential M=1, but avoid the thread-starved M=1 decode regime).
        outs = engine.generate(input_ids=[it["token_ids"] for it in items], sampling_params=sp)
        recs = []
        for it, out in zip(items, outs):
            r = out if isinstance(out, dict) else out
            txt = r["text"] if isinstance(r, dict) else r
            meta = r.get("meta_info", {}) if isinstance(r, dict) else {}
            pred = extract_final_answer(txt)
            recs.append({"q": it["q"], "gold": it["gold"], "pred": (pred if pred != INVALID else None),
                         "correct": int(pred == it["gold"]), "invalid": int(pred == INVALID),
                         "text": txt, "completion_tokens": meta.get("completion_tokens"),
                         "finish_reason": meta.get("finish_reason")})
        corr = sum(r["correct"] for r in recs)
        out_obj = {"leg": tag, "device": args.device, "tp": args.tp, "model": model,
                   "ck": ckv, "git_sha": _git(), "max_new": args.max_new, "n": len(recs),
                   "correct": corr, "accuracy": round(corr / max(1, len(recs)), 4),
                   "elapsed_s": round(time.perf_counter() - t0, 1), "records": recs}
        path = f"{args.out_prefix}_{tag}.json"
        _atomic(path, out_obj)
        print(f"[paired] leg={tag}: {corr}/{len(recs)} = {out_obj['accuracy']}  -> {path}", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
