---
name: accuracy-oracle
description: "Use after cpu-model-wiring to prove the CPU-enabled model is numerically correct. LAYERED checks so a failure localizes: (0) a cheap real-prompt COHERENCE smoke test first ('it ran' on random/dummy inputs is NOT 'it's correct'; catches gross zeroed/no-op/routing/layout bugs and silently-stubbed forward paths), (0.5) a per-kernel LOW-BIT PARITY gate for every dtype bridge (fp4/mxfp4/fp8/int4 weights on a bf16/fp8 kernel) — kernel output vs an independent torch dequant oracle from the same packed bytes, because 'a bf16/fp8 kernel exists' is coverage, not correctness, (1) per-layer / logits parity vs a trusted reference (HF or the GPU SGLang path) under tolerance, and (2) end-to-end task accuracy via the in-repo harnesses (gsm8k, mmlu, hellaswag) within N points. Includes a correct-by-construction ISOLATION technique to bisect MoE vs shared-path bugs. Blocks enablement on any regression; a per-layer diff pinpoints the offending op for the wiring step to fix."
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

## Multi-vector differential capture (when you have 2-3 suspects, resolve them in ONE run)
Single-swap isolation needs one run PER suspect. When the model load dominates (~12 min) the
bottleneck is NUMBER OF RUNS, so instead **capture every candidate op's tensors in ONE pass on both
the reference and the target, then diff offline** — one run dispositions all suspects. This is the
correctness analog of `high-information-runs`.
- **Build an op-level reference bundle.** Run the trusted path (GPU SGLang / HF) once with a hook
  that saves each op boundary's INPUT and OUTPUT (embedding, each attn sub-op q/k/v/scores/out, MoE
  out, norms, logits), keyed by `(layer, op, n_tokens, tp_rank)`. This bundle IS the oracle.
- **Op-level teacher forcing kills cascades.** A wrong op makes every downstream op look wrong. Feed
  each target op the reference's ground-truth INPUT (not the target's own upstream output), so each
  op is judged on clean input. First op whose OUTPUT diverges (given matching input) is THE bug — no
  cascade, no ambiguity.
- **Instrument all suspects at once with a disposition matrix.** Before submitting, list the 2-3
  suspects and the tap that distinguishes each; one run collects all. E.g. q-path bug: tap `q_lora`,
  `wq_b` out (nope+rope), and post-norm+rope out → `qlora` differs ⇒ wq_a/q_norm; `in` differs ⇒
  wq_b; `out_rope` differs ⇒ rope kernel — three suspects, one run.

### Diff pitfalls that cost real runs (do these or the run is wasted)
- **COSINE IS MAGNITUDE-BLIND — always pair it with relative max-abs-err / magnitude ratio.** A
  systematic scale drift (e.g. a CPU fp8 dequant path ~8% hot) passes `cos≈1.0` yet compounds across
  layers into garbage. Real case: "fp8 proj parity cos=1.0" hid the actual bug for a long time.
- **Coarse fingerprints (mean/max/first-4) hide permutations, direction flips, and scale.** Save full
  tensors; compare with cosine AND max-abs-err AND magnitude ratio.
- **Match sequences or the diff is meaningless.** The reference (GPU) does CUDA-graph/warmup passes,
  padding (e.g. 256-tok), and **TP-shards** head-dim tensors across ranks; the target (CPU tp=1) does
  not. Filter warmup by token-count/magnitude gates, key by `n_tokens`, and **rank-tag head-sharded
  tensors** (q, attn-core-out) so rank0=head0 lines up with the target's head 0. Unsharded quantities
  (MLA `wkv` latent, the final all-reduced attn output) compare directly.
- **Validate your tap indexing.** `x[0, -4:]` on a `[T, H, D]`-flattened tensor grabs the LAST head,
  not head 0 (and under TP that's a different head on each side) — a silent wrong-head compare wastes
  a whole run. Print shapes; assert the slice hits the head/dim you mean.
- **A self-check reference shares your blind spot.** A naive-torch twin only catches bugs the twin
  doesn't ALSO make (a from-scratch port checked against another from-scratch port passes while both
  are wrong). The independent GPU/HF oracle is the spec.

## Layer 0.5 — low-bit kernel parity gate (per dtype bridge)
**Every `dtype_bridge` from the dtype audit (`model-op-decomposition` §2b) gets its own numeric
gate — coverage ("a bf16/fp8 kernel exists") is NOT correctness.** When a kernel consumes weights
whose STORED dtype differs from its compute dtype (fp4/mxfp4/fp8/int4 experts on a bf16/fp8 AMX
GEMM), a dequant+repack bridge (scale decode, nibble order, group/block layout, VNNI prepack,
SwiGLU gate/up split) sits in between and fails SILENTLY (plausible garbage, not a crash). Prove
it before trusting end-to-end output:
- **Independent oracle from the SAME packed bytes.** Compare the real kernel output to a torch
  dequant+compute reference derived from the checkpoint's own packed weights + scales — decode the
  low-bit values per the KERNEL's exact convention (read the kernel source, e.g. csrc/cpu/vec.h
  `cvt_mxfp4_e2m1_bf16`: standard OCP e2m1 LUT, low-nibble-first, per-32 e8m0 scale = 2^(byte−127)),
  NOT via a different quant path that may use another scale layout. Assert cosine ≥ 0.99 and small
  relative error (bf16 accumulation vs fp32 reference → tolerate ~1e-2, but a wiring bug blows past
  it by orders of magnitude).
- **Two complementary forms.** (1) A STANDALONE synthetic test (random packed weights → prepack →
  kernel vs oracle) proves the kernel + packing math, is CI-able, and needs no model load — but must
  run on real ISA hardware (AMX). (2) An IN-SITU probe gated by an env flag stashes the first layer's
  raw packed weights before they're replaced by the prepacked ones, then on the first real forward
  compares kernel vs oracle on the REAL checkpoint + REAL activations — this catches checkpoint-
  wiring bugs the synthetic test cannot (e.g. the loader's float32-pow2 scale layout, real SwiGLU
  order). Worked example in this repo: `plugin/validate/test_mxfp4_moe_cpu.py` (standalone) +
  `INTEL_CPU_DSV4_MOE_PARITY=1` (`[MXFP4 PARITY]` in-situ), mirroring the DSA `test_dsa_*` gates.
This gate localizes a bridge bug to the exact kernel; end-to-end coherence (Layer 0) or task
accuracy (Layer 2) would only tell you "something is wrong" somewhere in a 43-layer forward.

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
