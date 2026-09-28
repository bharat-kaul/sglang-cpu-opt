---
name: accuracy-oracle
description: "Use after cpu-model-wiring to prove the CPU-enabled model is numerically correct. THREE layered checks so a failure localizes: (0) a cheap real-prompt COHERENCE smoke test first ('it ran' on random/dummy inputs is NOT 'it's correct'; catches gross zeroed/no-op/routing/layout bugs and silently-stubbed forward paths), (1) per-layer / logits parity vs a trusted reference (HF or the GPU SGLang path) under tolerance, and (2) end-to-end task accuracy via the in-repo harnesses (gsm8k, mmlu, hellaswag) within N points. Includes a correct-by-construction ISOLATION technique to bisect MoE vs shared-path bugs. Blocks enablement on any regression; a per-layer diff pinpoints the offending op for the wiring step to fix."
---

# Accuracy Oracle

Correctness is the non-negotiable gate. A plausible-but-wrong model is worse than
an unenabled one, so prove parity numerically AND on task, and make failures
localizable.

## Reference
- Primary: the HF `modeling_*` forward in fp32/bf16 on the same inputs.
- Secondary: the GPU SGLang path for the same model (catches SGLang-specific
  wiring differences vs pure HF).
- For OLMo 2 / OLMoE, also cross-check the models' published eval numbers.

## Layer 0 — coherence smoke test (do FIRST; catches gross bugs for ~free)
**"It ran" is NOT "it's correct."** A benchmark on RANDOM/DUMMY inputs (`bench_one_batch` with
`--load-format dummy` or random token ids) exercises shapes and TIMING but can pass while the
model emits garbage — random inputs have no "right" output to check against, and a fast
non-erroring run proves nothing about correctness. Before any parity work, greedily generate a
few tokens from a handful of REAL prompts and READ the text:
- Coherent English/code → proceed to Layer 1.
- One token repeated (`"Didži Didži…"`), all-same-token, or gibberish → a GROSS bug (a zeroed/
  no-op op, wrong expert routing, a scrambled layout). Stop and localize BEFORE measuring parity.

**Watch for silently no-op'ing / stubbed forward paths — the #1 gross-bug source.** A WIP branch
that returns EMPTY/ZERO when a feature isn't wired does not error; it silently produces plausible-
but-wrong output. Real case (this repo): the decode sparse-attention selection was stubbed
("later increment / stash empty for now") → empty indices → the attention gathered nothing →
`out=0` → attention became a no-op → the model emitted a single repeated token. Any
"stub/empty-for-now" branch on the forward path is a correctness landmine: give it a correct
**fallback** (e.g. dense attention over all valid KV when the sparse selection is empty), never a
silent zero. Grep the wired forward for `empty`/`zeros(0`/`return None`/`pass  # TODO`.

## Isolation — localize the bug with a correct-by-construction variant
When output is garbage, bisect by swapping ONE component for a slow-but-obviously-correct
reference that shares the rest of the pipeline: e.g. run the MoE via a plain torch
dequant+matmul reference (same weights) while keeping the real attention/norm/rope. Coherent →
the bug is in the swapped component; still garbage → it's in the SHARED path (attention/rope/
norm). Raise the framework **watchdog timeout** first (`watchdog_timeout`) — a deliberately slow
reference otherwise gets killed as a false "hang" (exit 137 with a watchdog stack).

## Layer 1 — numerical parity (localizes)
1. Feed identical token inputs to reference and CPU model.
2. Capture per-decoder-layer hidden states + final logits.
3. Metrics per capture point: cosine similarity and max-abs / relative error.
4. Tolerance (bf16 CPU vs bf16 ref): cosine ≥ 0.999 on hidden states, top-1 logit
   agreement 100% and top-5 KL small on a fixed prompt set. Tighten if the
   reference is fp32.
5. The FIRST layer whose error exceeds tolerance identifies the mis-wired op —
   hand that op back to `cpu-model-wiring` (usually a prepack/layout or a
   norm/rope placement bug).

## Layer 2 — task accuracy (end-to-end truth)
Run the in-repo harnesses on CPU vs reference:
- `benchmark/gsm8k` (reasoning), `benchmark/mmlu` (knowledge), `benchmark/hellaswag`
  (commonsense).
- Pass if each score is within N points of the reference (default N=1.0 absolute,
  or within run-to-run noise), on a fixed decode config (greedy or fixed seed).

## Procedure
1. Run Layer 1 on a small fixed prompt set; block on the first out-of-tol layer.
2. Once parity holds, run Layer 2 harnesses; block on any score regression.
3. Record both results (per-layer max error + task deltas) for the certificate.

## Gate
PASS only if per-layer parity within tolerance AND every task score within N
points. Any regression blocks — never ship on "looks close". Report the tolerances
used so the verdict is auditable.

## Pitfalls
- **A fast, non-erroring run proves nothing about correctness.** A dummy-weight or random-token
  bench that hits the target latency can still be numerically garbage — always run Layer 0 on
  REAL prompts before believing a perf number. Report perf numbers as "unvalidated" until the
  accuracy gate passes.
- Comparing against a differently-configured reference (rope scaling, dtype,
  sampling) manufactures fake diffs — pin dtype, rope params, and sampling first.
- Passing task accuracy while per-layer parity fails can happen (error cancels) —
  keep BOTH; parity is what proves the KERNELS are right, task is what proves the
  MODEL is right.
- Validate EACH precision the workflow generated as its own config with its own
  tolerance: BF16 against the tight parity bar; INT8 (w8a8) against a looser INT8
  budget (a small task-score drop is expected from per-channel weight + dynamic-
  token quant) — never judge INT8 against the bf16 bar, and never ship INT8
  without its own accuracy pass.
