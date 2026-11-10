#!/usr/bin/env python3
"""Shared-clamp ANSWER-RECOVERY test (reviewer step 3/5; the deliverable — recovery tied to ANSWERS).

ONE model load, production config (accum=1, routed-clamp=1). The plugin installs the shared-clamp op-swap
(INTEL_CPU_DSV4_SHARED_CLAMP=1); this driver toggles the clamp per-arm via the control file and generates
the frozen cases (17 GPU-right/CPU-wrong + 6 recovered controls) each arm:
  arm OFF (ctl=0): native shared expert -> MUST match the production 397690 answers (matched baseline)
  arm ON  (ctl=1): shared SwiGLU clamped at swiglu_limit
Scores with the authoritative extract_final_answer protocol, reports per-question recovery/regression.

  INTEL_CPU_DSV4_SHARED_CLAMP=1 INTEL_CPU_DSV4_SHARED_CLAMP_CTL=/path/ctl \
  python run_shared_clamp_recovery.py --frozen frozen_targets.json --out-prefix /path/rec --max-new 256
"""
import argparse
import json
import os
import time

_pkg = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
if _pkg:
    __import__(_pkg)

from task_gsm8k import INVALID, extract_final_answer

FAILING = {5, 29, 81, 146, 197, 244, 247, 417, 435, 502, 769, 893, 924, 995, 1067, 1191, 1229}
CONTROLS = {302, 562, 640, 693, 1175, 1271}


def _atomic(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)


def _git():
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                        "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    ap.add_argument("--baseline-gens", default="", help="optional 397690 gens glob for OFF-arm cross-check")
    args = ap.parse_args()

    ctl = os.environ.get("INTEL_CPU_DSV4_SHARED_CLAMP_CTL", "")
    if os.environ.get("INTEL_CPU_DSV4_SHARED_CLAMP") != "1" or not ctl:
        raise SystemExit("FATAL: require INTEL_CPU_DSV4_SHARED_CLAMP=1 and INTEL_CPU_DSV4_SHARED_CLAMP_CTL set")

    frozen = json.load(open(args.frozen))
    items = frozen["items"]
    model = frozen["model"]

    import sglang as sgl
    extra = {}
    for ev, kw, cast in (("CONTEXT_LEN", "context_length", int), ("MAX_TOTAL_TOKENS", "max_total_tokens", int),
                         ("SWA_RATIO", "swa_full_tokens_ratio", float), ("MAX_RUNNING", "max_running_requests", int)):
        if os.environ.get(ev):
            extra[kw] = cast(os.environ[ev])
    engine = sgl.Engine(model_path=model, device="cpu", tp_size=1, trust_remote_code=True,
                        disable_overlap_schedule=True, disable_cuda_graph=True,
                        mem_fraction_static=args.mem_fraction, log_level="warning",
                        watchdog_timeout=100000, **extra)
    sp = {"temperature": 0.0, "max_new_tokens": args.max_new}

    arms = {}
    for arm in ("0", "1"):
        open(ctl, "w").write(arm)
        time.sleep(0.2)
        t0 = time.perf_counter()
        outs = engine.generate(input_ids=[it["token_ids"] for it in items], sampling_params=sp)
        recs = {}
        for it, out in zip(items, outs):
            r = out if isinstance(out, dict) else out
            txt = r["text"] if isinstance(r, dict) else r
            meta = r.get("meta_info", {}) if isinstance(r, dict) else {}
            pred = extract_final_answer(txt)
            recs[int(it["q"])] = {"q": int(it["q"]), "gold": it["gold"],
                                  "pred": (pred if pred != INVALID else None),
                                  "correct": int(pred == it["gold"]),
                                  "text": txt, "finish_reason": meta.get("finish_reason"),
                                  "completion_tokens": meta.get("completion_tokens")}
        corr = sum(v["correct"] for v in recs.values())
        arms[arm] = recs
        _atomic(f"{args.out_prefix}_shared{arm}.json",
                {"arm": ("off" if arm == "0" else "on"), "ctl": arm, "git_sha": _git(),
                 "max_new": args.max_new, "n": len(recs), "correct": corr,
                 "elapsed_s": round(time.perf_counter() - t0, 1), "records": list(recs.values())})
        print(f"[recovery] arm={'off' if arm=='0' else 'on'}: {corr}/{len(recs)} correct "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    engine.shutdown()

    off, on = arms["0"], arms["1"]
    recovered = sorted(q for q in off if not off[q]["correct"] and on[q]["correct"])
    regressed = sorted(q for q in off if off[q]["correct"] and not on[q]["correct"])
    changed_ans = sorted(q for q in off if off[q]["pred"] != on[q]["pred"])
    summary = {
        "n": len(off), "git_sha": _git(), "max_new": args.max_new,
        "off_correct": sum(v["correct"] for v in off.values()),
        "on_correct": sum(v["correct"] for v in on.values()),
        "failing_cohort": {
            "recovered": [q for q in recovered if q in FAILING],
            "regressed": [q for q in regressed if q in FAILING],
            "answer_changed": [q for q in changed_ans if q in FAILING]},
        "control_cohort": {
            "regressed": [q for q in regressed if q in CONTROLS],
            "answer_changed": [q for q in changed_ans if q in CONTROLS]},
        "all_recovered": recovered, "all_regressed": regressed, "all_answer_changed": changed_ans,
    }
    _atomic(f"{args.out_prefix}_summary.json", {"summary": summary,
            "per_q": {str(q): {"gold": off[q]["gold"], "off_pred": off[q]["pred"], "on_pred": on[q]["pred"],
                               "off_correct": off[q]["correct"], "on_correct": on[q]["correct"],
                               "cohort": ("failing" if q in FAILING else "control" if q in CONTROLS else "other")}
                      for q in sorted(off)}})
    print("SHARED_CLAMP_RECOVERY_SUMMARY", json.dumps(summary), flush=True)

    if args.baseline_gens:
        import glob
        base = {}
        for f in glob.glob(args.baseline_gens):
            for g in json.load(open(f)).get("gens", []):
                base[int(g["q"])] = extract_final_answer(g["text"])
        mism = [q for q in off if q in base and base[q] != off[q]["pred"]]
        print(f"[recovery] OFF-arm vs 397690 baseline: {len(off)-len(mism)}/{len(off)} match "
              f"(mismatches: {mism})", flush=True)


if __name__ == "__main__":
    main()
