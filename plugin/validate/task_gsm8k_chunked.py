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
    args = p.parse_args()

    lines = read_jsonl(args.data)
    shots = "".join(one_example(lines[i], True) + "\n\n" for i in range(args.num_shots))
    eval_lines = lines[args.num_shots:args.num_shots + args.num_questions]
    # disjoint contiguous shard for parallel multi-node runs (combine sums correct/done)
    shard_offset = 0
    if args.num_shards > 1:
        n = len(eval_lines)
        per = (n + args.num_shards - 1) // args.num_shards
        shard_offset = args.shard_index * per
        eval_lines = eval_lines[shard_offset:min(shard_offset + per, n)]
    prompts = [shots + one_example(x, False) for x in eval_lines]
    labels = [answer_value(x["answer"]) for x in eval_lines]
    total = len(labels)

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

    correct = invalid = done = 0
    gens = []
    t0 = time.perf_counter()
    for start in range(0, total, args.chunk_size):
        chunk_p = prompts[start:start + args.chunk_size]
        chunk_l = labels[start:start + args.chunk_size]
        outs = engine.generate(chunk_p, sp)
        texts = [o["text"] if isinstance(o, dict) else o for o in outs]
        preds = [extract_final_answer(t) for t in texts]
        correct += sum(int(pr == l) for pr, l in zip(preds, chunk_l))
        invalid += sum(int(pr == INVALID) for pr in preds)
        done += len(chunk_l)
        dt = time.perf_counter() - t0
        acc = correct / done
        result = {
            "model": args.model, "num_shots": args.num_shots,
            "shard_index": args.shard_index, "num_shards": args.num_shards,
            "done": done, "total": total, "partial": done < total,
            "accuracy": round(acc, 4), "correct": correct, "invalid": invalid,
            "elapsed_s": round(dt, 1), "acc_min": args.acc_min,
            "verdict": "PARTIAL" if done < total else ("PASS" if acc >= args.acc_min else "REPORT"),
        }
        print(f"[chunk {start // args.chunk_size}] done={done}/{total} "
              f"acc={acc:.4f} invalid={invalid} elapsed={dt:.0f}s", flush=True)
        if args.out:
            _atomic_write(args.out, result)
        if args.gens_out:
            gens.extend({"q": shard_offset + start + j, "gold": l, "text": t}
                        for j, (t, l) in enumerate(zip(texts, chunk_l)))
            _atomic_write(args.gens_out, gens)

    engine.shutdown()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
