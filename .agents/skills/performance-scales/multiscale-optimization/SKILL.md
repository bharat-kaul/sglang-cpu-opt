---
name: multiscale-optimization
description: "ENTRY POINT + orchestrator for the scale-stratified CPU performance tree. Load this FIRST for any new-model CPU perf-enablement. It enforces the north-star (wall-time DOMINATED by model-op/kernel time with framework/wiring glue → 0, and EACH dominant kernel at its roofline), the TOP-DOWN traversal (macro → meso → micro — fix the biggest scale first, Amdahl), the UPFRONT MULTI-SCALE DONOR STUDY gate (read the nearest SGLang CPU donor at all three scales and model the WHOLE operator incl. the seam BEFORE estimating/authoring — never a bare GEMM), and the ANTI-OVERSTATEMENT rule (confirm every microbench/roofline-proxy win IN-ENGINE). Routes to macro-scale / meso-scale / micro-scale skills and their per-technique leaves. Use when a model must be carried to best-possible CPU performance autonomously."
---

# Multi-scale CPU optimization (orchestrator)

The scale-stratified spine over the per-technique skills. The existing skills
(`inter-kernel-fusion`, `weight-prepacking-brgemm`, `amx-vectorization`, …) are the
LEAVES; this tree is the organizing spine + the autonomous flow + the anti-patterns
that were previously scattered and therefore not applied up front.

## ⛔ NORTH-STAR (the only definition of "done")
A CPU model is at best-possible performance when BOTH hold, measured:
1. **Kernel-domination:** `Σ(model-op/kernel time) / wall ≥ ~0.90` — framework/wiring glue
   (dispatch, python loops, casts/copies, gather/scatter, routing, barriers, unattributed)
   is a SMALL remainder, not a dominant slice. If glue dominates, you are NOT kernel-bound
   and micro-tuning a kernel is premature.
2. **Each dominant kernel at its roofline:** every op in the kernel-dominated remainder runs
   at ≥ its achievable-roofline fraction for its regime (streamed-BW or AMX-compute), with the
   gap explained (operating-point vs fusion vs uarch), not hand-waved.
Report BOTH numbers for every published result. Accuracy is a hard gate throughout
(`accuracy-oracle`): perf is UNVALIDATED until per-layer parity + task/CPU-vs-GPU equivalence hold.

## ⛔ TOP-DOWN TRAVERSAL (fix the biggest scale first — Amdahl)
Do NOT start at the micro scale. A micro-optimal kernel buried under 90% glue is worthless.
1. **MACRO first** (`macro-scale`): make the profile KERNEL-DOMINATED — kill framework/dispatch
   overhead, fuse the whole operator (no intermediate materialization), parallelize + bind NUMA,
   and pick the OPERATING POINT (batch M) that slides memory-bound ops into the efficient regime.
2. **MESO next** (`meso-scale`): for each now-dominant kernel, get its DATA MOVEMENT right —
   cache-tiling + BRGEMM-resident accumulator + weight prepacking + operand-byte reduction, to the
   streamed-achievable roofline (not the resident compute peak). Classify each op by AI vs the ridge.
3. **MICRO last** (`micro-scale`): get each inner loop at the AMX/VNNI compute peak — tile op,
   packing, inline precision conversion, fp32 accumulation, and a FUSED epilogue.
Re-profile after each scale; the dominant op (and therefore the next lever) changes as you go.

## ⛔ ITERATIVE FEEDBACK LOOPS (the scales are NOT a one-pass waterfall)
Top-down is the DISCOVERY order; the real method is ITERATIVE with BOTTOM-UP feedback — a finding at
a lower scale re-decides a higher one. Expert loops to internalize and run every iteration:
1. **Re-profile after EVERY change (dominant op shifts scale).** Fixing the macro glue can expose a
   meso-bound kernel; speeding one kernel promotes the next op — which may live at a DIFFERENT scale.
   Never assume the post-fix profile; measure it.
2. **A FLOOR finding at a lower scale feeds UP.** If an op is measured at its meso floor (BW-bound,
   AI < ridge) or its micro floor (inner loop at peak) and still too slow, the lever is NOT more of
   the same scale — it moves UP: change the OPERATING POINT (macro), FUSE it into its neighbor so the
   intermediate never materializes (macro/meso), or ACCEPT it as an attributed floor and move on.
   *Worked example (ours): the M-sweep (a macro operating-point experiment) measured the KDA
   recurrence scaling LINEARLY with M → fed back to the macro decision "batching will NOT help the
   recurrence; whole-operator fusion is its only lever" — which then set the meso/micro target.*
3. **In-engine measurement (GATE 2) CORRECTS the roofline model (GATE 1).** When a real impl misses
   the proxy, do NOT cling to the proxy — update the estimate with the costs the proxy omitted
   (conversion/materialization/layout/dispatch) and RE-RANK the RoI. *Worked example: DSA indexer
   naive C++ = 1.07× vs 6–7× predicted → fed back to "the op MUST be whole-operator-fused" AND to a
   feasibility RE-DECISION (realistic ~1.3× may not beat the authoring/maintenance cost).*
4. **Convergence test (when to STOP iterating).** Loop until the north-star holds (kernel-dominated
   ≥ ~0.90 AND each dominant kernel at its roofline) OR every remaining gap is an ATTRIBUTED, accepted
   floor (BW-bound op at its BW; dispatch-bound tiny op not worth fusing; operating-point already
   chosen). "No single-scale lever left, all gaps attributed" = done — not "ran out of ideas."
The dotted upward arrows in `performance-scales/README.md` are these feedback edges; a pass that only
goes macro→meso→micro once, without re-profiling and feeding floors back up, is the waterfall anti-pattern.

## ⛔ GATE 1 — UPFRONT MULTI-SCALE DONOR STUDY (before estimating OR authoring any op)
Before you model a roofline or write a line of kernel code for a novel op, STUDY THE NEAREST
SGLang CPU DONOR at ALL THREE scales (`kernel-authoring/assets/donor-kernel-map.md`):
- **Macro:** is the donor ONE fused operator (e.g. `moe.cpp` fuses SiLU in the store; `fla.cpp`
  fuses kkt_solve + recompute_w_u to avoid materializing A/h/v_new)? Your op must be too.
- **Meso:** how does the donor tile + prepack + keep the C accumulator resident across K (BRGEMM)?
- **Micro:** which tile op / packing / precision-conversion / fused-epilogue does it use?
Then MODEL THE WHOLE OPERATOR INCLUDING THE SEAM — input dtype conversion, intermediate
materialization to DRAM, layout/transpose repacks, and the epilogue pass — NOT a bare GEMM.
**A roofline estimate that omits these macro-scale costs is a KNOWN-OVERSTATING proxy and is
rejected.** (History: a bare-bf16-bmm proxy predicted 6–7× for the DSA indexer; the real non-fused
kernel delivered 1.07× because it paid input conversion + [N,H,S] materialization + a transposed
slow path + a separate epilogue — all absent from the proxy.)

## ⛔ GATE 2 — ANTI-OVERSTATEMENT (confirm every microbench win IN-ENGINE)
An isolated microbench / "ideal matmul" roofline is NOT an achievable target for a real fused op.
CONFIRM each predicted win inside the real engine before believing it. Settled null results (do NOT
re-spend runs re-testing — the HW/compiler already handles them): mid-forward `set_num_threads`
(no-op), accumulator-ILP banking (OoO hides latency), prefetch-distance (HW prefetcher saturates).
See `micro-scale` NULL-TRAPS and `shared/high-information-runs`.

## Flow
1. Wire with reference kernels + prove parity (`cpu-model-wiring`, `accuracy-oracle`).
2. Profile → kernel-domination ratio + per-op shares (`model-profile-hotspots`, `overhead-attribution`).
3. `macro-scale` until the profile is kernel-dominated.
4. For each dominant kernel: `meso-scale` (data movement) then `micro-scale` (inner loop), gated by
   GATE 1 (donor study + whole-operator model) and GATE 2 (in-engine confirm).
5. Publish the roofline-vs-measured + time-attribution pivot PAIR at the labeled operating point
   (`model-profile-hotspots` cycle-exit gate); re-confirm accuracy.

## Cross-reference (the leaves, by scale)
- MACRO: `inter-kernel-fusion`, `fusion-analysis`, `openmp-parallelization`, `sub-numa-clustering`,
  `overhead-attribution`, `runtime-config-tuning`, `cpu-serving-integration`.
- MESO: `cache-blocking-tiling`, `weight-prepacking-brgemm`, `kernel-authoring`,
  `compute-from-native-precision`, `roofline-validation`, `establish-achievable-performance`.
- MICRO: `amx-vectorization`, `cpu-gemm-amx-bf16`, `quantization-amx-int8`, `uarch-perf-probe`.
- Orchestration this plugs into: `cpu-optimization-playbook` (technique routing) +
  `model-enablement-playbook` (the enablement leg).
