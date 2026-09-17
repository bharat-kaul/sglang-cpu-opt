---
name: inter-kernel-fusion
description: "The INTER-KERNEL-DEPENDENCY lever from Phase B: once each operator is certified at its roofline in isolation, the remaining end-to-end gap is the GLUE between kernels — dtype/layout conversions, re-pack, staging copies to hand off data, serialization (no overlap), and pairs that should be one fused op. Use AFTER isolation-certification when the e2e residual (measured_e2e − Σ isolated floors) is large and not framework/config. Removes bytes and passes between kernels rather than speeding any single kernel. Complements fusion-analysis (which finds graph-level fusion up front) by targeting the fusions/handoffs that only show up in the wired end-to-end run."
---

# Inter-Kernel Fusion (kill the glue between optimal kernels)

Two kernels can each be at their roofline while the *pair* is slow, because the handoff
between them moves bytes and blocks overlap. This is the `inter-kernel dependency` bucket
of `overhead-attribution`, and it is the Phase-B target: nothing here authors or re-tunes
a kernel — it removes work *between* them.

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
1. Confirm both neighboring ops are isolation-certified (else fix the kernel first).
2. Compute the seam cost: `e2e(A→B) − isolated(A) − isolated(B)`. If positive and material,
   it is glue — classify it as one of the four above (a profiler shows the `aten::to`/`copy_`/
   contiguous ops; a boundary timer shows the barrier gap).
3. Apply the matching fix; re-measure the *pair* against `isolated(A)+isolated(B)` (the pair's
   floor is the max of the two if perfectly fused/overlapped, not the sum).
4. Re-run `roofline-validation` on the fused region.

## Boundary with the other skills
- `fusion-analysis` finds fusion opportunities from the GRAPH, up front (before kernels exist).
  This skill targets the seams that only surface once real kernels are wired and certified.
- `overhead-attribution` DECIDES the residual is inter-kernel (not framework/config) and points here.
- `kernel-authoring` implements a *fused* kernel when the fix is "make A and B one op" (its
  4-layer skeleton + fused-epilogue composition patterns are the mechanism).
- The floor for a fused region is `max(isolated floors)` under perfect overlap — never chase
  below that.
