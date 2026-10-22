#!/usr/bin/env python3
"""Chunked, checkpointing gsm8k runner for SGLang CPU (fallback harness).

Same prompt + parsing as task_gsm8k.py (imported, so results are comparable), but
generates in CHUNKS and writes a PARTIAL json after every chunk. Two reasons:
  1. a slurm wall-kill (or watchdog) still leaves a usable partial accuracy number
     (the single-shot task_gsm8k.py writes only at the very end -> kill = nothing);
  2. each chunk prints a progress line, so the run is observable (the engine at
     log_level=warning is otherwise silent during generation).

Run as a FILE on-node (reuses the plugin override + env):
  python task_gsm8k_chunked.py --model <path> --data <jsonl> \\
      --num-questions 300 --chunk-size 64 --out <json>
"""

import argparse
import json
import os
import time

# Pre-import the external plugin so its import-time CPU arg-resolution guards (e.g. the
# torch.cuda.get_device_capability() guard for CPU-only torch) install BEFORE sgl.Engine's
# resolve_once(). Mirrors _smoke.py; without it GLM crashes in handle_model_specific_adjustments.
_pkg = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
if _pkg:
    __import__(_pkg)

from task_gsm8k import INVALID, answer_value, extract_final_answer, one_example, read_jsonl


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--device", default="cpu", help="cpu | cuda (cuda = GPU oracle)")
    p.add_argument("--num-questions", type=int, default=300)
    p.add_argument("--num-shots", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=64,
                   help="questions per generate() call; a partial json is written after each")
    p.add_argument("--shard-index", type=int, default=0,
                   help="this shard's index in [0, num-shards); for parallel multi-node runs")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split the question set into this many disjoint shards (one per node)")
    p.add_argument("--acc-min", type=float, default=0.0)
    p.add_argument("--mem-fraction", type=float, default=0.5)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantization", default=None)
    p.add_argument("--out", default="")
    p.add_argument("--gens-out", default="",
                   help="also save per-question {q,gold,text} here for free offline re-scoring")
    p.add_argument("--watchdog-timeout", type=float, default=7200.0)
    p.add_argument("--chunked-prefill-size", type=int, default=512)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="disable checkpoint resume (re-run from chunk 0 even if a partial exists)")
    p.add_argument("--question-ids-file", default="",
                   help="JSON list of GLOBAL eval-question indices to run (into the post-shots set); "
                        "overrides --num-shards/--shard-index slicing. For re-sharding a straggler tail "
                        "across many nodes — the ids carry the global q so gens/combine stay consistent.")
    p.set_defaults(resume=True)
    p.add_argument("--perf-probe", action="store_true",
                   help="before gsm8k, run a B-batch prefill+decode on the RESIDENT engine to measure "
                        "real-weight tok/s + (with TIMEIT) the decode op attribution — one load, many answers")
    p.add_argument("--perf-batch", type=int, default=32)
    p.add_argument("--perf-input", type=int, default=128)
    p.add_argument("--perf-output", type=int, default=8)
    p.add_argument("--perf-thread-sweep", default="",
                   help="comma list of decode thread counts to sweep on the RESIDENT model (via the control "
                        "file), e.g. 8,16,32,64,128 — one load, full sweep; 0/empty = use the engine default")
    p.add_argument("--perf-m-sweep", default="",
                   help="comma list of decode batch sizes M to sweep, e.g. 1,8,32 — factorial with the thread "
                        "sweep yields the M-ADAPTIVE cap table; empty = just --perf-batch")
    p.add_argument("--perf-only", action="store_true",
                   help="exit after the perf-probe (skip gsm8k) — for a page-cache-amortized thread sweep")
    p.add_argument("--perf-reps", type=int, default=1,
                   help="measure each (M,threads) config this many times on the resident load (held-fixed) "
                        "and report median + min/max — resolves run-to-run variance, firms attribution")
    p.add_argument("--perf-ck-sweep", default="",
                   help="comma list of DSA C-kernel toggle values to sweep (0=eager, 1=fused C) via the CK "
                        "control file — a HELD-FIXED eager-vs-C decode A/B on ONE resident load (lesson A1)")
    p.add_argument("--gens-schema-v2", action="store_true",
                   help="wrap gens-out as {run_identity, gens:[...]} (self-identifying); default = plain list")
    args = p.parse_args()

    lines = read_jsonl(args.data)
    shots = "".join(one_example(lines[i], True) + "\n\n" for i in range(args.num_shots))
    full_eval = lines[args.num_shots:args.num_shots + args.num_questions]
    # Select this task's questions. Default = disjoint contiguous shard (combine sums correct/done).
    # --question-ids-file = an EXPLICIT global-index set (re-shard a straggler tail across many nodes);
    # q_ids carries each question's GLOBAL index so gens/combine stay consistent regardless of layout.
    if args.question_ids_file:
        with open(args.question_ids_file) as f:
            q_ids = [int(i) for i in json.load(f)]
        eval_lines = [full_eval[i] for i in q_ids]
    else:
        shard_offset = 0
        if args.num_shards > 1:
            n = len(full_eval)
            per = (n + args.num_shards - 1) // args.num_shards
            shard_offset = args.shard_index * per
            full_eval = full_eval[shard_offset:min(shard_offset + per, n)]
        eval_lines = full_eval
        q_ids = list(range(shard_offset, shard_offset + len(eval_lines)))
    prompts = [shots + one_example(x, False) for x in eval_lines]
    labels = [answer_value(x["answer"]) for x in eval_lines]
    total = len(labels)

    if total == 0:  # empty id-shard (over-split reshard) — nothing to do, skip the engine load
        print(f"[reshard] shard {args.shard_index} has 0 questions — skipping engine load.", flush=True)
        if args.out:
            _atomic_write(args.out, {"model": args.model, "shard_index": args.shard_index,
                                     "done": 0, "total": 0, "partial": False, "accuracy": 0.0,
                                     "correct": 0, "invalid": 0, "verdict": "EMPTY"})
        return

    # --- checkpoint resume: if a prior run of THIS shard left a partial, pick up where it
    # stopped so a wall-kill/crash loses no work. Re-score the saved gens to restore counts
    # self-consistently (temp=0 is deterministic, so skipped chunks reproduce exactly, and
    # trusting the gens file avoids any out/gens between-write double-count). ---
    resume_done = resume_correct = resume_invalid = 0
    resume_elapsed = 0.0
    gens = []
    if args.resume and args.gens_out and os.path.exists(args.gens_out):
        try:
            with open(args.gens_out) as f:
                gens = json.load(f)
            if isinstance(gens, dict):      # v2 self-identifying form {run_identity, gens:[...]}
                gens = gens.get("gens", [])
        except Exception as e:
            print(f"[resume] ignoring unreadable gens {args.gens_out}: {e}", flush=True)
            gens = []
        for g in gens:
            pr = extract_final_answer(g["text"])
            resume_correct += int(pr == g["gold"])
            resume_invalid += int(pr == INVALID)
        resume_done = len(gens)
    elif args.resume and args.out and os.path.exists(args.out):
        try:
            with open(args.out) as f:
                ck = json.load(f)
            resume_done = int(ck.get("done", 0))
            resume_correct = int(ck.get("correct", 0))
            resume_invalid = int(ck.get("invalid", 0))
        except Exception as e:
            print(f"[resume] ignoring unreadable checkpoint {args.out}: {e}", flush=True)
    if args.out and os.path.exists(args.out):
        try:
            with open(args.out) as f:
                resume_elapsed = float(json.load(f).get("elapsed_s", 0.0))
        except Exception:
            resume_elapsed = 0.0
    if resume_done:
        print(f"[resume] shard {args.shard_index}: resuming from checkpoint done={resume_done}/{total} "
              f"correct={resume_correct} invalid={resume_invalid} prior_elapsed={resume_elapsed:.0f}s",
              flush=True)
    if total > 0 and resume_done >= total:
        print(f"[resume] shard {args.shard_index} already COMPLETE ({resume_done}/{total}) "
              f"— skipping engine load.", flush=True)
        return

    import sglang as sgl

    # GLM-5.3 (hybrid KDA mamba + DSA MLA) needs CPU-valid cache config the DSv4 harness
    # didn't: no_buffer mamba radix + disable_radix_cache (page_size clash) + a capped
    # context_length (native 1M over-reserves). Read from env (mirrors _smoke.py) so the
    # SAME hard-won prompt/scoring/sharding harness carries GLM unchanged.
    extra_kw = {}
    if os.environ.get("MAMBA_RADIX_STRATEGY"):
        extra_kw["mamba_radix_cache_strategy"] = os.environ["MAMBA_RADIX_STRATEGY"]
    if os.environ.get("DISABLE_RADIX") == "1":
        extra_kw["disable_radix_cache"] = True
    if os.environ.get("CONTEXT_LEN"):
        extra_kw["context_length"] = int(os.environ["CONTEXT_LEN"])
    if os.environ.get("MAX_RUNNING"):
        extra_kw["max_running_requests"] = int(os.environ["MAX_RUNNING"])
    # DSV4-Flash native context is ~1M and it has SWA; on a NUMA/SNC-domain-bound MXFP4 run the KV
    # pool over-reserves and the scheduler is OOM-killed (SIGKILL -9) during init. Cap it like the
    # canonical bench_one_batch (--max-total-tokens / --swa-full-tokens-ratio), read from env.
    if os.environ.get("MAX_TOTAL_TOKENS"):
        extra_kw["max_total_tokens"] = int(os.environ["MAX_TOTAL_TOKENS"])
    if os.environ.get("SWA_RATIO"):
        extra_kw["swa_full_tokens_ratio"] = float(os.environ["SWA_RATIO"])
    if os.environ.get("LOAD_FORMAT"):
        extra_kw["load_format"] = os.environ["LOAD_FORMAT"]  # "dummy" for tiny-faithful harness tests

    engine = sgl.Engine(
        model_path=args.model, device=args.device, tp_size=args.tp, dtype=args.dtype,
        quantization=args.quantization,
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=args.mem_fraction, log_level="warning",
        watchdog_timeout=args.watchdog_timeout,
        chunked_prefill_size=args.chunked_prefill_size,
        **extra_kw,
    )
    sp = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens,
          "stop": ["Question", "Assistant:"]}

    # --- perf-probe: amortize the 149GB load into real-weight perf + decode attribution + an
    # M x THREADS factorial sweep on the RESIDENT model. The in-worker decode-thread hook reads the
    # control FILE (INTEL_CPU_DSV4_DECODE_THREADS_FILE) each forward, so we sweep thread counts on the
    # one resident load (no reload) -> the M-ADAPTIVE cap table (best threads per batch M). ---
    if args.perf_probe:
        import random as _rnd
        import shutil as _sh
        with open(f"{args.model}/config.json") as _cf:
            _vocab = json.load(_cf)["vocab_size"]
        _rng = _rnd.Random(0)
        _S, _G = args.perf_input, args.perf_output
        _steps = max(_G - 1, 1)
        _ctl = os.environ.get("INTEL_CPU_DSV4_DECODE_THREADS_FILE", "")
        _mlist = [int(x) for x in (args.perf_m_sweep or str(args.perf_batch)).split(",") if x.strip()]
        _tlist = [int(x) for x in (args.perf_thread_sweep or "0").split(",") if x.strip()]

        def _fresh(m):
            return [[_rng.randrange(3, _vocab - 1) for _ in range(_S)] for _ in range(m)]

        _do_flush = os.environ.get("INTEL_CPU_DSV4_PERF_FLUSH", "0") == "1"

        def _flush():
            # Reset radix/KV-cache state so each measurement starts clean — repeated independent
            # synthetic batches otherwise accumulate and degrade decode. OFF by default: on the CPU
            # engine flush_cache() blocks for a long time (it drains + rebuilds), so it costs more
            # than it saves here; opt in with INTEL_CPU_DSV4_PERF_FLUSH=1. The robust clean-held-fixed
            # alternative is a FRESH engine per config (reload), not a resident repeated measurement.
            if not _do_flush:
                return
            try:
                engine.flush_cache()
            except Exception:
                pass

        def _timed(m):
            _p1 = {"temperature": 0.0, "max_new_tokens": 1}
            _pG = {"temperature": 0.0, "max_new_tokens": _G}
            _flush(); engine.generate(input_ids=_fresh(m), sampling_params={"temperature": 0.0, "max_new_tokens": 2})  # warms the DECODE path (one step); never timed
            _flush()
            _t0 = time.perf_counter(); engine.generate(input_ids=_fresh(m), sampling_params=_p1)
            _tp = time.perf_counter() - _t0
            _flush()
            _t0 = time.perf_counter(); engine.generate(input_ids=_fresh(m), sampling_params=_pG)
            _tf = time.perf_counter() - _t0
            _d = (_tf - _tp) / _steps
            return (m * _S / _tp), (m / _d), _d * 1e3

        print(f"[PERF-SWEEP] ===== resident M x threads factorial  M={_mlist} threads={_tlist} S={_S} G={_G} reps={args.perf_reps} =====", flush=True)
        _swp_json = os.environ.get("INTEL_CPU_DSV4_SWEEP_JSON", "")
        import statistics as _stats
        _res = {}
        _rows = []
        _cklist = [x.strip() for x in args.perf_ck_sweep.split(",") if x.strip()] or [None]
        _ckctl = os.environ.get("INTEL_CPU_DSV4_CK_CTL_FILE", "")
        for _ckv in _cklist:
            if _ckv is not None and _ckctl:
                with open(_ckctl, "w") as _f:
                    _f.write(str(_ckv))
            for _m in _mlist:
                for _t in _tlist:
                    if _t and _ctl:
                        with open(_ctl, "w") as _f:
                            _f.write(str(_t))
                    _decs = []
                    _prefs = []
                    _dmss = []
                    for _r in range(max(1, args.perf_reps)):
                        _pref, _dec, _dms = _timed(_m)
                        _decs.append(_dec); _prefs.append(_pref); _dmss.append(_dms)
                        if args.perf_reps > 1:
                            print(f"[PERF-SWEEP]   rep={_r} M={_m} threads={_t} ck={_ckv}  decode_tok_s={_dec:8.2f}  ms/step={_dms:9.1f}", flush=True)
                    _dec_med = _stats.median(_decs)
                    _dms_med = _stats.median(_dmss)
                    _pref_med = _stats.median(_prefs)
                    _res[(_m, _t)] = _dec_med
                    _spread = (max(_decs) / min(_decs)) if min(_decs) > 0 else 0.0
                    print(f"[PERF-SWEEP] ck={_ckv} M={_m:4d} threads={_t:4d}  decode_tok_s(med)={_dec_med:8.2f}  "
                          f"ms/step(med)={_dms_med:9.1f}  spread={_spread:4.2f}x  "
                          f"[min={min(_decs):.2f} max={max(_decs):.2f}]  prefill_tok_s(med)={_pref_med:8.1f}", flush=True)
                    # CHECKPOINT each config to a durable json (survives a wall-kill; unbuffered observable)
                    _rows.append({"ck": _ckv, "M": _m, "threads": _t, "reps": len(_decs),
                                  "decode_tok_s": round(_dec_med, 2), "decode_tok_s_all": [round(x, 2) for x in _decs],
                                  "decode_tok_s_min": round(min(_decs), 2), "decode_tok_s_max": round(max(_decs), 2),
                                  "decode_spread_x": round(_spread, 3),
                                  "decode_ms_step": round(_dms_med, 1), "prefill_tok_s": round(_pref_med, 1)})
                    if _swp_json:
                        _atomic_write(_swp_json, {"S": _S, "G": _G, "reps": args.perf_reps, "rows": _rows})
        print("[PERF-SWEEP] ===== M-ADAPTIVE cap (best threads per M) =====", flush=True)
        for _m in _mlist:
            _best = max(((_t, _res[(_m, _t)]) for _t in _tlist), key=lambda x: x[1])
            print(f"[PERF-SWEEP] M={_m:4d} -> best threads={_best[0]:4d} ({_best[1]:.2f} tok/s)", flush=True)
        print("[PERF-SWEEP] ===== done =====", flush=True)
        if args.perf_only:
            engine.shutdown()
            print("[PERF-SWEEP] --perf-only set; exiting before gsm8k.", flush=True)
            return

    correct, invalid, done = resume_correct, resume_invalid, resume_done
    # Run-identity stamp (review P0): bind every saved artifact to exactly what produced it, so a
    # result is self-identifying and resume/compare can reject a mismatched continuation.
    def _git_sha():
        try:
            import subprocess
            return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                            "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None
    _toggles = {k: os.environ.get(k) for k in (
        "INTEL_CPU_DSV4_DSA_CKERNEL", "INTEL_CPU_DSV4_DSA_CKERNEL_CHECK", "INTEL_CPU_DSV4_CK_STRICT",
        "INTEL_CPU_DSV4_MXFP4_MOE", "INTEL_CPU_DSV4_DSA_CPU", "INTEL_CPU_DSV4_DECODE_THREADS",
        "INTEL_CPU_DSV4_CK_CTL_FILE", "CONTEXT_LEN", "SWA_RATIO", "MAX_TOTAL_TOKENS") if os.environ.get(k) is not None}
    run_identity = {
        "model": args.model, "device": args.device, "tp": args.tp, "dtype": args.dtype,
        "quantization": args.quantization, "num_shots": args.num_shots,
        "max_new_tokens": args.max_new_tokens, "chunk_size": args.chunk_size,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "sampling_params": sp, "git_sha": _git_sha(), "env_toggles": _toggles,
        "scorer": "extract_final_answer(task_gsm8k)", "protocol_version": 2,
    }
    t0 = time.perf_counter()
    for start in range(0, total, args.chunk_size):
        if start < resume_done:
            continue  # already checkpointed on a prior run — skip
        chunk_p = prompts[start:start + args.chunk_size]
        chunk_l = labels[start:start + args.chunk_size]
        outs = engine.generate(chunk_p, sp)
        # review P0: require EXACTLY one response per requested question — a short `outs` with `zip`
        # would silently truncate scoring while `done` still advances by the label count.
        if len(outs) != len(chunk_l):
            raise RuntimeError(f"engine returned {len(outs)} outputs for {len(chunk_l)} prompts "
                               f"(chunk at {start}) — refusing to checkpoint a truncated chunk")
        recs = [o if isinstance(o, dict) else {"text": o} for o in outs]
        texts = [r.get("text", "") for r in recs]
        metas = [r.get("meta_info", {}) if isinstance(r, dict) else {} for r in recs]
        preds = [extract_final_answer(t) for t in texts]
        correct += sum(int(pr == l) for pr, l in zip(preds, chunk_l))
        invalid += sum(int(pr == INVALID) for pr in preds)
        done += len(chunk_l)
        dt = resume_elapsed + (time.perf_counter() - t0)
        acc = correct / done
        result = {
            "model": args.model, "num_shots": args.num_shots,
            "shard_index": args.shard_index, "num_shards": args.num_shards,
            "done": done, "total": total, "partial": done < total,
            "accuracy": round(acc, 4), "correct": correct, "invalid": invalid,
            "elapsed_s": round(dt, 1), "acc_min": args.acc_min,
            "verdict": "PARTIAL" if done < total else ("PASS" if acc >= args.acc_min else "REPORT"),
            "run_identity": run_identity,
        }
        print(f"[chunk {start // args.chunk_size}] done={done}/{total} "
              f"acc={acc:.4f} invalid={invalid} elapsed={dt:.0f}s", flush=True)
        if args.out:
            _atomic_write(args.out, result)
        if args.gens_out:
            # Enriched schema (review P0): keep immutable identity + termination metadata so a saved
            # generation is scorable/auditable offline (finish reason, token counts, pred).
            gens.extend({"q": q_ids[start + j], "gold": l, "text": t, "pred": p,
                         "completion_tokens": m.get("completion_tokens"),
                         "prompt_tokens": m.get("prompt_tokens"),
                         "finish_reason": m.get("finish_reason")}
                        for j, (t, m, p, l) in enumerate(zip(texts, metas, preds, chunk_l)))
            _atomic_write(args.gens_out, {"run_identity": run_identity, "gens": gens}
                          if args.gens_schema_v2 else gens)

    engine.shutdown()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
