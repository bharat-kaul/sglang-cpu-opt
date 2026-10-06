---
name: cpu-optimization-playbook
description: "Master index + decision flow for optimizing an SGLang model op on Intel Xeon CPUs. Routes a target op (GEMM, attention, MLP, norm, MoE) through an ordered, composable set of per-technique skills — establish-achievable → parallelize → tile → vectorize/AMX → prepack/BRGEMM → quantize — each gated by the roofline. Use this first; it decides WHICH technique skills to load and in WHAT order."
---

# CPU Optimization Playbook (orchestrator index)

This is the entry point the agentic workflow loads first for any CPU optimization
task. It does not itself optimize; it **routes** the op through a progressive,
composable library of per-technique skills, loading each only when its trigger
condition is met and gating every step against the measured roofline.

> **Load the SCALE-STRATIFIED SPINE alongside this router.** `performance-scales/multiscale-optimization`
> organizes the whole corpus by scale (macro → meso → micro) and enforces the north-star (wall-time
> DOMINATED by kernel time, each kernel at roofline), the UPFRONT multi-scale donor study (model the
> WHOLE operator incl. the seam before estimating/authoring — never a bare GEMM), and the
> confirm-in-engine rule. This router supplies the per-technique leaves; the spine supplies the
> top-down flow + the anti-overstatement gates that prevent reasoning at the wrong scale.

> **North-star deliverable (both legs).** Enablement is "done" ONLY when the new model
> RUNS end-to-end on CPU, is ACCURACY-parity vs a trusted reference (accuracy-oracle,
> real weights), AND ships a published ROOFLINE-TARGET-vs-MEASURED perf artifact at a
> labeled machine config. Authoring/tuning a kernel is a means; the proof is the running,
> accurate, perf-vs-roofline model in the enablement-certificate. `cpu-serving-integration`
> is the step that turns authored kernels into that running model.

> **⛔ CYCLE-EXIT GATE — a perf-optimization cycle is NOT done until the published artifact PAIR
> exists (auto-emit it as the CLOSING STEP, do not wait to be asked).** The moment the final
> measured per-op profile is in hand, emit BOTH, from that profile, at the labeled machine config:
> (1) the **roofline-target-vs-measured** report+chart (`roofline_vs_measured.py`), AND (2) the
> **time-attribution pivot** companion (`time_attribution_pivot.py`, where the wall-clock goes per
> phase, summing to 100% with an explicit unattributed slice) — then ADD the model's bullet to the
> README "Roofline target vs measured" section (mirroring the existing entries) and publish. Both
> files are the deliverable; one without the other is incomplete. **A baseline→optimized→roofline
> "journey"/summary chart is NOT a substitute** — it answers "how much faster did we get," the
> required pair answers "how close to the ceiling, and where does the wall-time go" (the RoI view
> every published result links). *Anti-pattern that actually happened (GLM-5.3 Flash): the cycle
> ended with only the journey chart, which LOOKED like "charts done," so the DSv4-format roofline+pivot
> pair + the README results-section bullet were silently skipped and had to be back-filled.* Treat
> "generated a chart" as NOT satisfying this gate until the roofline report, the pivot report, and the
> README bullet all exist. If the hot op is recurrence/small-op bound (dispatch-limited, not a dense
> GEMM), the FLOP roofline is the WRONG reachable target — publish it but CAPTION the realistic
> engineering floor (e.g. fused-kernel) so "1% of achievable" is not misread as failure.

> **When the RUN is the bottleneck (long load / slow decode / cluster queue) — load the
> run-efficiency skills (apply to EVERY model, both legs):** `high-information-runs` (make each
> expensive run a multi-angle probe; and ALWAYS launch `scripts/await_job.sh` in the async terminal
> after submitting a long job so completion AUTO-WAKES the agent — never fire-and-forget then wait
> for a human to ask "is it done?"); `perf-proxy` (depth-reduced, full-width, DUMMY-weight perf
> ladder — perf iteration only, never accuracy). For TASK ACCURACY specifically, `accuracy-oracle`
> carries the node-parallel + chunked + checkpointed harness, the generate-once → score-OFFLINE
> decouple, and the certify-vs-trusted-reference rule (a bare score certifies nothing).

> **PRESERVE THE KOSHER BASELINE (do this BEFORE the first optimization).** The parity-checked,
> unoptimized CPU build is the reference for the whole perf phase — it must stay runnable and
> untouched. Three guards, all required: (1) **FREEZE the golden fingerprint** (the per-op tensors +
> logits from the unoptimized real-weight build) — the immutable correctness reference; (2) **TAG the
> baseline commit** (e.g. `git tag <model>-baseline-parity`) so the exact baseline CODE is recoverable,
> not just a floating SHA; (3) **make every optimization ENV-GATED default-OFF** (the repo convention —
> DSV4 ships ~47 `INTEL_CPU_*` gates) so gate-off reproduces the baseline byte-for-byte in the SAME tree,
> and develop the opt work on a **separate branch** merged back only after the full golden-parity +
> no-regression confirm. An optimization becomes the DEFAULT (gate flipped on / merged) ONLY after it
> passes the per-op golden-check AND the full-model re-diff. This is how "incrementally optimize +
> integrate, correct by construction" stays honest: the baseline is always one `git checkout <tag>` or
> one gate-off away, so any regression is bisectable to the single op that caused it.

> **Reference-wiring-FIRST.** Before optimizing, wire the full serving path with
> fallback/reference kernels and make it RUN + CORRECT on a **tiny architecturally-faithful
> config** (real arch switches, tiny dims, dummy weights — runs in seconds, same code paths).
> This front-loads the wiring/infra breaks (TP/NUMA, config, allocator, metadata) that static
> analysis misses and gives a running reference to A/B every later optimization against.
> Correctness backbone first, speed on top. See `cpu-serving-integration`.
>
> **ORDERING (correctness → perf → task).** Per-layer PARITY vs the reference baseline is the
> correctness gate FIRST (vs the GPU/HF oracle, real weights, one short prefill — cheap), then it
> GUARDS every optimization: on the dummy-weight `perf-proxy` the guard is the optimized-vs-reference
> A/B on the same input (valid on dummy — it tests the kernel transform, not accuracy), re-confirmed
> per-layer on the full OPTIMIZED model. TASK accuracy (real weights, full generation) is LAST. See
> `accuracy-oracle` CORRECTNESS ORDERING + the `perf-proxy` ladder.

> **TASK-ACCURACY SIGN-OFF — SURFACE THE METHOD AS A USER DECISION (don't silently pick).** The final
> task-accuracy step has two legitimate routes; PRESENT BOTH to the user and let them choose (cost vs
> external-validation tradeoff): **(A) Reproduce the published card number** — exact card protocol
> (shots, `reasoning_effort`/thinking budget, long generations, sampling). Gold-standard external
> validation, but for a frontier agentic/multimodal model this can be days of CPU compute or out of
> scope for the text path (and the card may not even report a cheap task like gsm8k). **(B) CPU-vs-GPU
> equivalence on a SHORT benchmark** — run the IDENTICAL harness (gsm8k) on the CPU build and the GPU
> reference and show they match (aggregate within <1 pt, high per-question agreement, SYMMETRIC
> disagreements, 0 invalid). Cheap, feasible, and a direct correctness proof of the CPU build vs the
> same-model reference — though it certifies *equivalence to the reference*, not an absolute leaderboard
> number. Recommend (B) when (A) is infeasible, but STATE the tradeoff and let the user decide. See
> `accuracy-oracle` (the CPU-vs-GPU cross-check + the PARITY-vs-CAPABILITY distinction).

> Two legs, one plugin. This playbook is the **new-kernel** leg (write/optimize a
> kernel). `model-enablement-playbook` is the **throughput** leg (wire a new model
> from EXISTING kernels). The enablement `coverage-gate` hands a genuine gap to
> THIS leg via the `cpu-optimizer` agent; when the new kernel passes its roofline
> it registers a capability contract and the model re-enters enablement.

> WRITING vs optimizing. To AUTHOR a net-new kernel (a coverage GAP with no CPU
> implementation, e.g. the DSA indexer), load **`kernel-authoring`** FIRST — it maps
> the op to the nearest donor kernel in the SGLang corpus and the 4-layer adaptation
> (control / inner-op / packing / epilogue), grounded in LIBXSMM/TPP + oneDNN. Then
> the technique skills below TUNE the drafted kernel.

> **One law, recurring at every level — FIT THE WORK TO THE RESOURCE, up front, from a
> MEASURED constant.** Each technique below is the same decision at a different level of the
> hierarchy: match the unit of work to the resource's capacity/shape BEFORE tuning, or overhead
> dominates. Decide these up front from `uarch-perf-probe` constants, not after:
> | Level | Fit the work so… | Resource constant (uPP) | Mismatch failure |
> |---|---|---|---|
> | Threads (grain) | work/thread ≥ sync/spawn floor | thread-scaling knee | inverse scaling (MoE 0.1ms@4 vs 2432ms@60) |
> | Cache (tile) | working set ≤ L2/core, reused panel resident | cache-ladder sizes | L2 spill → DRAM re-streaming |
> | SIMD/AMX (layout) | data in VNNI/tile order, dims divisible | tile shape / ISA | gather/stride stalls, padding waste |
> | Bandwidth (precision) | operand bytes ≤ BW can feed | per-domain BW, ridge | BW-starved compute unit |
> These are two-sided (too small wastes the unit; too big overruns it) — sweep the curve, it can
> be non-monotonic. `overhead-attribution` tells you WHICH level is the current bottleneck.

## The skill library (progressive order)

| # | Technique skill | Load when | Gates on |
|---|-----------------|-----------|----------|
| G | `kernel-feasibility-gate` | BEFORE authoring a net-new kernel | user-reviewed roofline + measured baseline; go/no-go |
| A | `kernel-authoring` | writing a NET-NEW kernel (not just tuning) | draft matches FP32 ref |
| S | `cpu-serving-integration` | novel kernels authored+validated; make the model actually RUN | model runs end-to-end + accuracy + roofline-vs-measured |
| 0 | `establish-achievable-performance` | ALWAYS first, once per hardware profile | sets the ceilings |
| D | `sub-numa-clustering` | after 0, BEFORE the tile/vectorize loop; multi-socket / SNC node | per-domain scaling + capacity fit |
| 1 | `roofline-validation` | after every implementation step | % of achievable |
| 2 | `openmp-parallelization` | op runs on >1 core | scaling efficiency |
| 3 | `cache-blocking-tiling` | working set > L2; streamed operands | L2/L1 residency |
| 4 | `amx-vectorization` | dtype∈{bf16,fp16,int8} and `amx_*` present | FLOP/cyc vs peak |
| 5 | `weight-prepacking-brgemm` | stationary operand (weights) reused across calls | close gap to achievable |
| 6 | `quantization-amx-int8` | accuracy budget allows lower precision | INT8 AMX peak |

Each skill is self-contained (capability contract, trigger, procedure, gate) so
new ops reuse them without modification, and new techniques are added as new
skill folders without touching the others.

## Decision flow

```
0. establish-achievable-performance  -> per-DOMAIN compute_peak + streamed_gemm + mem_bw ceilings
0.5 sub-numa-clustering -> FIX the SNC/NUMA domain to optimize WITHIN (capacity fit, tp=#SNC,
    one rank/domain, cpu+mem co-bind, first-touch). Steps below tune ONE domain, then replicate;
    the roofline ceiling is the EFFECTIVE (tp_used/n_domains)x full node, not the ideal node.
1. classify op + arithmetic intensity (roofline-validation: compute- vs memory-bound)
2. if memory-bound  -> cache-blocking-tiling, then re-roofline
   if compute-bound -> amx-vectorization (right ISA/tiling), then re-roofline
3. parallelize (openmp-parallelization): threads, affinity, NUMA/SNC
4. if weights reused (inference) -> weight-prepacking-brgemm  (usually the biggest
   single win for a lone large GEMM; see that skill's measured evidence)
5. if accuracy budget allows -> quantization-amx-int8
6. roofline-validation after each step; loop on the gap; record learned pattern
```

## Composition rule

Apply techniques in the order above and **re-run `roofline-validation` after each**.
Also **re-check numerical parity against the persistent reference oracle after each
step** (see `kernel-authoring` 1b): a slow-but-correct reference is authored/kept as
the correctness fallback, and any optimized variant that drifts from it is a
regression, not a speedup. Speed is only valid on top of correctness.
Stop when efficiency ≥ target (70% of the *streamed achievable* ceiling, not the
resident compute peak — see `establish-achievable-performance`). Record which
technique closed the gap in the skill's learned-patterns so the next op starts
from the winning combination.

## Worked composition

`cpu-gemm-amx-bf16` is a complete worked example that composes skills 0–5 for a
dense BF16 GEMM. Use it as the template when optimizing a new dense linear layer.
