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

## ⛔ ITERATIVE, BOTTLENECK-DRIVEN FEEDBACK LOOPS (not a one-pass waterfall)
Top-down (macro→meso→micro) is ONLY the discovery order for the FIRST pass. The real method is a
**bottleneck-driven control loop**, run EVERY iteration:

> **profile → identify the SINGLE dominant bottleneck → classify WHICH scale's lever addresses it →
> apply ONE lever → RE-PROFILE → repeat.**

The next lever is chosen by what the NEW profile reveals — it may be the **SAME scale again
(within-scale)** or a **different scale (across-scale)**. You do NOT march the scales once; you follow
the bottleneck. Loop until converged (below). The mechanisms you will actually use each pass:

1. **WITHIN-SCALE iteration (usually several passes per scale).** One lever rarely finishes a scale.
   Each pass targets the NEWLY-revealed sub-bottleneck, not a fixed checklist. *Macro example:* kill
   the `@torch.compile` dispatch spike → re-profile → a materialization seam now dominates → fuse it →
   re-profile → a thread-cliff now dominates → cap threads → re-profile. All WITHIN macro, driven by
   what each re-profile exposes, before descending. Same pattern inside meso (tile → prepack →
   precision-bytes) and micro (tile-op → fused epilogue).
2. **ACROSS-SCALE (re-profile shifts the scale).** Fixing one scale PROMOTES the next op, which may
   live at a DIFFERENT scale (fixing macro glue exposes a meso-bound kernel; speeding a kernel
   promotes the next op). Measure the post-fix profile; never assume it.
3. **A FLOOR finding at a lower scale feeds UP.** If an op is at its meso floor (BW-bound, AI < ridge)
   or micro floor (inner loop at peak) and still too slow, the lever is NOT more of the same scale — it
   moves UP: change the OPERATING POINT (macro), FUSE it into its neighbor so the intermediate never
   materializes (macro/meso), or ACCEPT it as an attributed floor and move on. *Worked example (ours):
   the M-sweep (a macro operating-point experiment) measured the KDA recurrence scaling LINEARLY with M
   → fed back to "batching will NOT help the recurrence; whole-operator fusion is its only lever."*
4. **In-engine measurement (GATE 2) CORRECTS the roofline model (GATE 1).** When a real impl misses the
   proxy, update the estimate with the omitted costs (conversion/materialization/layout/dispatch) and
   RE-RANK the RoI — do not cling to the proxy. *Worked example: DSA indexer naive C++ = 1.07× vs 6–7×
   predicted → "must whole-operator-fuse" AND a feasibility RE-DECISION (realistic ~1.3× may not pay).*
5. **Convergence test (when to STOP iterating).** Loop until the north-star holds (kernel-dominated
   ≥ ~0.90 AND each dominant kernel at its roofline) OR every remaining gap is an ATTRIBUTED, accepted
   floor (BW-bound op at its BW; dispatch-bound tiny op not worth fusing; operating-point already
   chosen). "No lever left at ANY scale, all gaps attributed" = done — not "ran out of ideas."

The control loop is bottleneck-first: the profile picks the scale each pass. The solid arrows in
`performance-scales/README.md` are the first-pass discovery order; the dotted arrows (and the
within-scale self-loops) are this feedback. A single macro→meso→micro pass without re-profiling and
re-dispatching on the revealed bottleneck is the waterfall anti-pattern.

## ⛔ ANCHOR DISCOVERY — derive the dominant op ANALYTICALLY, then validate by ONE run
Each iteration's real question is "which op, and which knob?" Answer it on PAPER first (architecture +
baseline code + the machine roofline), THEN confirm with a run. The **anchor** = the controllable
parameter the current binding resource is most sensitive to (M / tp·EP / precision / context-S /
fusion / tiling / layout). M is only one instance — discover the anchor, don't assume it.

**A. Analytical op inventory (no run).** From the arch + baseline code, per token per layer compute
`FLOPs`, `bytes_moved` → `AI = FLOP/byte`; classify the binding resource (AI<ridge → BW-bound;
AI>ridge → compute-bound; many tiny ops → dispatch-bound; model>domain → capacity-bound; decode SLA →
latency-bound); and — the discriminator — **does AI scale with the batchable dim M?** Weight-reused-
across-batch ops → `AI ∝ M` (amortize); per-sequence state / softmax / elementwise → `AI = const` in M
(immune); sparse MoE → effective `M·topk/E`. Weight each by LAYER COUNT.

**B. Set the operating-config anchors** (analytical, then a cheap confirm — see `macro-scale`):
`M* ≈ ½·ridge·b` (GEMM-sweep-confirmed 80–90%-roofline range, bounded by GATE 0 + latency); `tp*` =
smallest feasible tp clearing the capacity floor (divisibility/padding set; EP-first for MoE; CPU tp>1
hurts decode). Order: capacity → tp/EP → per-rank budget → M.

**C. Predict the dominant op = the POST-ANCHOR residual.** Apply the anchors on paper: anchor-movable
ops amortize to ~roofline; anchor-immune ops remain. **Rank by post-anchor residual × layer-count, NOT
the raw M=1 share** — else you chase an op the anchor was about to fix (the M=1 trap). Top residual =
predicted frontier; its own anchor is whatever ITS binding resource is most sensitive to (usually NOT
the operating-config anchor that just amortized the others — e.g. a per-sequence recurrence → fusion).

**D. VALIDATE BY ONE RUN (GATE 2).** One GATE-0-sized profile at the derived (tp/EP, M) config (dummy
weights OK for perf, full depth, target ISA) → confirm kernel-domination ratio + per-op shares match
the prediction. MATCH → optimize the predicted frontier. MISMATCH → the paper model omitted a cost
(materialization / dispatch / sparse-routing / comm) → UPDATE it and re-rank. The run corrects the
paper; never cling to the paper.

**E. Iterate — the anchor shifts.** Optimizing the frontier changes the binding resource → a new
anchor and a new dominant op → repeat A–D. Converge per the control-loop test.

*Worked example (GLM, derived on paper → confirmed by run): KDA gated-delta recurrence AI≈1 and
**constant in M** (per-sequence [H,K,V] state), 34 of 45 layers → predicted dominant at M*; projection
GEMMs AI 2→~64 as M→M* → amortize to ~1%. Validated: measured recur+conv = 74–79% of decode at M=32,
projections ~1%. The paper flagged the frontier (KDA fusion) AND the M=1 trap (don't chase projections)
before any run.*

## ⛔ GATE 0 — CAPACITY / FEASIBILITY PRE-FLIGHT (compute BEFORE submitting ANY expensive run)
Do the memory math UP FRONT, every launch — never submit-to-OOM. A wrong guess burns a ~15–30 min
load-to-SIGKILL round-trip on a scarce node. BEFORE each submit, size the FULL resident footprint vs the
node (per-NUMA/SNC DOMAIN for a single-rank tp=1 engine, not node total) and cap batch/context/depth to fit:
- **Model resident (the big fixed cost).** A low-bit checkpoint DEQUANTS at load: CPU **W8A16 fp8→bf16 ≈ 2×
  the on-disk fp8** (+ an AMX-prepack transient). **Dummy weights allocate the SAME real-shape tensors** —
  dummy is NOT small. (GLM: 306 GB fp8 → ~600 GB bf16 resident.)
- **Activations (scale with batch × tokens × layers) — the usual OOM culprit at batch.** An UNOPTIMIZED
  path MATERIALIZES per-op intermediates (the KDA scan, [N,H,S] scores) → multiply the estimate. Batch-M
  prefill of T tokens = M·T tokens through every layer.
- **State / KV pools.** Linear-attn (mamba/KDA) state = `max_running × per-req-state` — LARGE at batch;
  KV = `context_len × max_tokens × per-token-KV`. CPU `mem_fraction_static` = TOTAL engine budget
  (model+pools), not a GPU-style KV-only fraction — too high reserves a giant pool and OOMs.
- **Rule:** fixed model-resident FIRST, then the REMAINING domain RAM caps `batch × context × depth`.
  If it doesn't fit: cut batch, cut prefill tokens, cut depth, cap context, or pick a bigger-RAM node —
  BEFORE submitting. Cross-ref `accuracy-oracle` ⛔ CAPACITY BUDGET for the full ladder.
*Worked example (do NOT repeat): GLM tree-pilot iter-1 submitted full-45L dummy (~600 GB) + **batch-32** ×
256-tok prefill + unoptimized KDA-scan materialization on a 1.5 TB GNR at mem_frac 0.5 → the batch-32
activations/state blew past the ~900 GB remainder → **SIGKILL at the MoE (8192 tokens)**, a wasted ~30 min.
The 10-second pre-flight (600 GB model ⇒ batch-32 full-depth infeasible) would have said "start at M=1 or
small batch / short prefill" and never submitted it.*

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
