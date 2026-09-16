---
name: cpu-optimization-playbook
description: "Master index + decision flow for optimizing an SGLang model op on Intel Xeon CPUs. Routes a target op (GEMM, attention, MLP, norm, MoE) through an ordered, composable set of per-technique skills — establish-achievable → parallelize → tile → vectorize/AMX → prepack/BRGEMM → quantize — each gated by the roofline. Use this first; it decides WHICH technique skills to load and in WHAT order."
---

# CPU Optimization Playbook (orchestrator index)

This is the entry point the agentic workflow loads first for any CPU optimization
task. It does not itself optimize; it **routes** the op through a progressive,
composable library of per-technique skills, loading each only when its trigger
condition is met and gating every step against the measured roofline.

> **North-star deliverable (both legs).** Enablement is "done" ONLY when the new model
> RUNS end-to-end on CPU, is ACCURACY-parity vs a trusted reference (accuracy-oracle,
> real weights), AND ships a published ROOFLINE-TARGET-vs-MEASURED perf artifact at a
> labeled machine config. Authoring/tuning a kernel is a means; the proof is the running,
> accurate, perf-vs-roofline model in the enablement-certificate. `cpu-serving-integration`
> is the step that turns authored kernels into that running model.

> **Reference-wiring-FIRST.** Before optimizing, wire the full serving path with
> fallback/reference kernels and make it RUN + CORRECT on a **tiny architecturally-faithful
> config** (real arch switches, tiny dims, dummy weights — runs in seconds, same code paths).
> This front-loads the wiring/infra breaks (TP/NUMA, config, allocator, metadata) that static
> analysis misses and gives a running reference to A/B every later optimization against.
> Correctness backbone first, speed on top. See `cpu-serving-integration`.

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
