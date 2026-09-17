---
name: overhead-attribution
description: "Proactive, EARLY discipline to split a model run's time into optimized-KERNEL vs unoptimized-TORCH compute vs FRAMEWORK orchestration, BEFORE committing kernel/optimization budget. Use when a model runs but is slow, when deciding whether to author/optimize a kernel, or whenever a nominal 'kernel' cost looks far from its roofline floor. Layered method: (1) cheap env-gated boundary timers tagged kernel/torch/framework, (2) op-level profiler (torch.profiler/cProfile) for copies/casts/dispatch, (3) kernel-isolation microbench vs the uPP roofline floor. Prevents the two classic wastes: authoring a fast kernel that doesn't move end-to-end time, and optimizing the DSA while the real cost is a mis-configured 'AMX' kernel or framework churn. Complements model-profile-hotspots (adds the framework-vs-kernel lens) and feeds kernel-feasibility-gate."
---

# Overhead Attribution (framework vs kernel, early)

A model that RUNS but is slow has its time in one of four buckets, and the fix is
different for each:
- **KERNEL** — an optimized (AMX/BRGEMM) kernel doing real compute. Near its
  roofline floor → *not* the lever (don't re-author it). Far from the floor →
  a kernel-config problem (threading, weight prepack, dtype path), *not* a new kernel.
- **TORCH** — unoptimized torch/reference compute (fp32 matmuls, softmax, `.float()`
  churn). The classic co-design/kernel-authoring or route-to-existing-kernel lever.
- **FRAMEWORK** — orchestration: dispatch, python loops, copies, casts, gather/scatter,
  routing/topk, metadata. Fixed by fusion / fewer ops / bf16-end-to-end / vectorizing
  python — *no kernel needed*.
- **INTER-KERNEL DEPENDENCY** — the *glue* between two individually-optimal kernels:
  layout/dtype conversions, re-pack, staging/copies to hand off data, serialization
  (no overlap), missing cross-op fusion. Only visible END-TO-END (each kernel is fine
  in isolation). Fixed by unifying layouts/dtypes across adjacent ops, fusing, or
  pipelining — again *no kernel authoring*.

## Two-phase method (isolate-certify, then attribute the e2e residual)
The decomposition that makes the buckets clean (per the operator-roofline discipline).
**Core reasoning — certification is ELIMINATION:** optimizing each operator against its
*measured* roofline in isolation and certifying it at its floor *removes that operator
from the suspect list*. Once every kernel is eliminated as a source of slowdown, the
end-to-end gap can only be the non-kernel components — so attention is forced onto
framework and inter-kernel dependency. You don't guess what's slow; you eliminate the
kernels one by one until only the glue remains.
1. **Phase A — per-operator, in ISOLATION.** For each operator: operator-level roofline
   → optimize/author → measure vs that roofline **standalone** → **certify** it at (or
   near) its floor. A certified kernel is a *fixed point* (its isolated time is its floor)
   and is thereby ELIMINATED as an e2e suspect.
2. **Phase B — plug all certified kernels into the END-TO-END run.** Because each kernel
   is individually optimal, the e2e residual is *by construction* NOT the kernels:
   `residual = measured_e2e − Σ(isolated_kernel_floors)` = **FRAMEWORK + INTER-KERNEL
   DEPENDENCY**. That residual is the second-order target, attacked with fusion / layout
   unification / threading-config / pipelining — never more kernel authoring.
This is exactly the MoE finding: isolated floor ≈ 1 ms, e2e ≈ 1920 ms → the ~1919 ms
residual is framework/config (thread binding), not the kernel. Isolation-certify first
so the e2e gap is unambiguously attributable.

**Do this EARLY** — before `model-profile-hotspots` commits kernel budget. Attributing
the buckets first tells you the *ceiling* of what any optimization can buy and which
layer to touch. (This session's evidence: the presumed hotspot was wrong 4× in a row;
the "AMX" MoE cost 1.9 s/call — a *kernel* ~1000× above its floor whose isolated GEMM is
0.33 ms, so the gap is integration/config, not the kernel.)

## The layered method (cheapest first)
1. **Boundary timers (cheap, always first).** Wrap the hot op boundaries with an
   env-gated, per-rank, DCE-safe timer and TAG each `kernel|torch|framework|parent`
   (`parent` = contains other timed leaves; excluded from the split). Print a split
   over leaf ops at exit. Deterministic and tp-safe (each rank prints its own) —

## The layered method (cheapest first)
1. **Boundary timers (cheap, always first).** Wrap the hot op boundaries with an
   env-gated, per-rank, DCE-safe timer and TAG each `kernel|torch|framework|parent`
   (`parent` = contains other timed leaves; excluded from the split). Print a split
   over leaf ops at exit. Deterministic and tp-safe (each rank prints its own) —
   unlike a sampler that collapses under many OMP threads. *Worked example:*
   `INTEL_CPU_DSV4_TIMEIT=1` in `plugin/intel_cpu_models/_dsv4_cpu_infra.py`
   (`_timed(name, kind)` / `_tacc(name, t0, kind)` → the "overhead split" block).
2. **Op-level profiler (when boundaries are too coarse).** `torch.profiler` (or
   `cProfile` for tp=1) sorted by self-CPU-time attributes the FRAMEWORK residue —
   `aten::copy_`, `aten::to` (casts), dispatch, contiguous — vs the named kernel ops
   (`sgl_kernel::fused_experts`, `fp8_scaled_mm`). Use it to explain the untimed
   remainder the boundary timers miss (dense proj / norm / lm_head).
3. **Kernel-isolation microbench vs roofline (for any suspicious KERNEL).** If a
   `kernel`-tagged op is far from its floor, run it STANDALONE at the real shape and
   compare to the uPP `machine_constants.json` ceiling (compute peak / per-domain BW).
   Standalone-slow → kernel-config (threads, prepack, dtype dispatch, ISA not
   dispatched — check `ONEDNN_VERBOSE`); standalone-fast → the call context is wrong
   (un-prepacked weights, an extra cast, wrong M). This is what distinguishes "the
   kernel is bad" from "we're calling it badly."

## tiny ≠ real (non-negotiable)
Tiny configs mis-rank: weights are cache-resident and per-op dispatch overhead
dominates, so a torch op can look big and an AMX kernel small (or vice-versa). Always
confirm the split at REAL scale (real weights, real tp) before acting. (Tiny said the
indexer was the top DSA piece; at real scale it was the MLA attention.)

## The decision it drives
| Dominant bucket | Lever | NOT the lever |
|---|---|---|
| torch | author/route a kernel; co-design (fusion, bf16, prepack) | — |
| kernel far from roofline | kernel-config: threads, prepack, dtype path, verify ISA | authoring a *new* kernel |
| kernel at roofline | accept (memory/compute-bound floor) | any further kernel work |
| framework | fusion, fewer ops, bf16-end-to-end, vectorize python | a compute kernel |
| inter-kernel dependency | unify layout/dtype across adjacent ops, fuse, pipeline/overlap | a compute kernel |

## Ties into the workflow
- **Phase A / Phase B split:** the per-operator isolation loop (operator roofline →
  optimize → measure vs roofline → certify) lives in `kernel-feasibility-gate` +
  `kernel-authoring` + `roofline-validation`; this skill owns **Phase B** — attributing
  the end-to-end residual (`measured_e2e − Σ isolated_kernel_floors`) to framework vs
  inter-kernel dependency once the kernels are isolation-certified.
- Refines `model-profile-hotspots` — same "run the model, rank by measured time," but
  adds the framework/kernel/torch/inter-kernel tag so the RoI ranking is honest.
- Feeds `kernel-feasibility-gate` — a `kernel`-tagged op far from roofline must go to
  isolation (step 3) BEFORE authoring; a `framework`/`torch` op routes to fusion/authoring.
- Uses `uarch-perf-probe` (`machine_constants.json`) as the roofline floor in step 3.

## Procedure
1. Boundary-time the hot ops (tagged) → get the kernel/torch/framework split at REAL scale.
2. Profile the untimed remainder (step 2) if it is large.
3. For any suspicious KERNEL, isolate + compare to the uPP floor (step 3).
4. Route each bucket by the decision table; only THEN commit kernel/optimization budget.
