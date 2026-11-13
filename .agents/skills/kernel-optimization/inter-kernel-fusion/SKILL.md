---
name: inter-kernel-fusion
description: "Use during standalone optimization to explore producer-consumer fusion and again during serving integration to measure actual handoff costs: conversions, repacking, materialization and barriers. Complements graph-level fusion-analysis. Fused regions need their own source-conformed correctness and resource model; isolated timings alone do not establish a pair's floor."
---

# Inter-Kernel Fusion (kill the glue between optimal kernels)

Two kernels can each be at their roofline while the *pair* is slow, because the handoff
between them moves bytes and blocks overlap. This is the `inter-kernel dependency` bucket
of `overhead-attribution`. Explore it before standalone closure and revisit the actual
serving seam during integration. The fix may require a newly authored fused C/C++ kernel,
which must pass its own standalone gate under the
[canonical policy](../../../../docs/methodology-playbook.md#canonical-acceptance-policy).

## The four glue costs (and their fix)
1. **Dtype conversion at the seam.** Kernel A emits fp32, kernel B wants bf16 → an `aten::to`
   per element between them. Fix: unify the interface dtype so no cast crosses the boundary
   (bf16 end-to-end where the reference allows).
2. **Layout / re-pack at the seam.** A produces row-major, B needs VNNI/packed → a repack copy.
   Fix: have A emit B's layout directly, or pack once and keep it; align tile/block shapes so
   no reshape is needed.
3. **Staging / materialization.** A writes a full intermediate to DRAM that B immediately reads
   (e.g. scores → softmax → PV each materialized). Fix: FUSE into one kernel with an on-chip
   epilogue (online-softmax, `SiLU·mul` in the store) so the intermediate never hits memory —
   raises arithmetic intensity, the classic BW win.
4. **Serialization.** A must fully finish before B starts on the same thread pool, with a
   barrier between. Fix: pipeline/overlap across tiles or ranks, or fuse so there is no barrier.

## Method (measure the seam, not the kernels)
1. Establish source-faithful baselines for both neighbors. Do not defer all fusion
   exploration until both have been independently declared optimized.
2. Compute the seam cost: `e2e(A→B) − isolated(A) − isolated(B)`. If positive and material,
   investigate glue, cache/residency, contention and changed dispatch conditions.
   The residual alone is not attribution; validate with matched boundary evidence and A/B runs.
3. Apply the matching fix; re-measure the *pair* against `isolated(A)+isolated(B)` (the pair's
   baseline is that sum only under matched serial conditions). Recompute fused FLOPs,
   bytes and resource/dependency costs; shared compute can serialize, while eliminated
   memory traffic can lower the floor below either isolated floor.
4. Re-run `roofline-validation` on the fused region.

## Boundary with the other skills
- `fusion-analysis` finds fusion opportunities from the GRAPH, up front (before kernels exist).
  This skill tests source-derived standalone pairs and the seams exposed by actual serving integration.
- `overhead-attribution` DECIDES the residual is inter-kernel (not framework/config) and points here.
- `kernel-authoring` implements a *fused* kernel when the fix is "make A and B one op" (its
  4-layer skeleton + fused-epilogue composition patterns are the mechanism).
- A fused region has its own `max(compute_time, memory_time)` resource bound and
  operation-specific costs; neither the sum nor max of isolated floors is universally valid.
