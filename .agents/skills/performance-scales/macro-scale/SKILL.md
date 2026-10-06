---
name: macro-scale-optimization
description: "MACRO scale of the multi-scale CPU performance tree (do this FIRST). Goal: make the wall-time profile DOMINATED by model-op/kernel time by killing framework/wiring/dispatch overhead, fusing the WHOLE operator (no intermediate materialization), parallelizing + binding NUMA/SNC, and choosing the OPERATING POINT (batch M) that slides memory-bound ops into the efficient AMX regime. This is where the biggest wall-time wins live (they dwarf any single-kernel tweak) and where the classic mistakes hide (modeling an op as a bare GEMM, treating batching as a code win). Load after multiscale-optimization; route to inter-kernel-fusion, openmp-parallelization, sub-numa-clustering, overhead-attribution, runtime-config-tuning."
---

# Macro-scale optimization — make the profile KERNEL-DOMINATED

Biggest scale, done first (Amdahl). The question this scale answers: **is the wall-time
spent in model kernels, or in glue?** Until it's kernel-dominated, no micro-tuning matters.

## Diagnostic (start here)
- **Time-attribution pivot** (`model-profile-hotspots`, `overhead-attribution`): split wall into
  `kernel / torch / framework / unattributed`, summing to 100% with an explicit unattributed slice.
- **Kernel-domination ratio** = `Σ(kernel-op time) / wall`. Target ≥ ~0.90. Below that → macro work.
- **Seam cost** for adjacent ops: `seam = e2e(A→B) − isolated(A) − isolated(B)`. Positive & material
  ⇒ a glue tax (cast/copy/contiguous/barrier); fuse it. The fused-region floor = `max(isolated floors)`.

## The four macro levers (ranked by typical wall-time impact)
1. **Eliminate framework/dispatch overhead.** Many tiny ops → dispatch-bound (python/torch per-op
   launch dominates compute). Fixes: fewer/larger ops, bf16-end-to-end (no per-op up/down-cast),
   vectorize python loops, and dispatch-elimination (GPU: CUDA-graph; CPU analog: `torch.compile`/
   inductor warmed at init — the nearest CPU equal, captures less). *Real case: a 288-expert router
   was `@torch.compile(dynamic)` whose one-time CPU inductor compile landed INSIDE the timed prefill
   (27 s); warming it at init → 0.1 s, bit-exact.* Also a per-forward `@torch.compile` on a small op
   can cost seconds on CPU — if a pure-torch op is fast isolated but seconds in-engine AND
   thread-invariant, suspect `@torch.compile`; call `.__wrapped__` or warm it.
2. **WHOLE-OPERATOR fusion (no intermediate materialization).** The op is ONE fused region, not a
   GEMM followed by a separate epilogue that round-trips [N,…] intermediates through DRAM. Donors:
   `moe.cpp` (SiLU·mul fused in the store), `fla.cpp` ("fuse kkt_solve + recompute_w_u to avoid
   materialize A; fuse recompute_w_u + update_v to avoid materialize h and v_new"). **This is THE
   lesson that makes a C++ port worth it — a non-fused op-by-op kernel pays input conversion +
   materialization + layout repack + a serialized epilogue and recovers almost nothing (DSA indexer:
   predicted 6–7×, naive got 1.07×).** See `inter-kernel-fusion`, `fusion-analysis`.
3. **Parallelization + NUMA/SNC binding.** Right thread count (beware thread-CLIFFS — some fused
   kernels scale INVERSELY with threads), `parallel_2d` tiling, and single-rank first-touch on ONE
   SNC domain (or `numactl --interleave` for capture). See `openmp-parallelization`,
   `sub-numa-clustering`, `runtime-config-tuning` (OMP spin-wait fix `OMP_WAIT_POLICY=passive` +
   `KMP_BLOCKTIME=0` was a 63×/10× win — the whole first profile was a busy-wait artifact).
4. **OPERATING POINT (batch M).** This is a CONFIG/serving lever, NOT a kernel rewrite. At M=1 decode,
   weight-streaming / per-token dequant is UN-AMORTIZED — the inherent floor (DSv4 MoE: 26% of BW
   roofline). Batching to M=16/32/64 amortizes the weight read across the batch → slides into the
   efficient AMX regime (DSv4 MoE 26%→75%, 2.9×). Honest framing: "raising M amortizes dequant" is an
   operating-point move, not a code-perf win at fixed M — but it is usually the single biggest decode
   lever and must be chosen BEFORE deciding any kernel is "at roofline" (a kernel's roofline fraction
   is defined at an operating point). See `runtime-config-tuning`.

## Operating-config determination (DERIVE M and tp/EP up front, confirm cheaply)
Don't pick M or tp by trial — derive them, then confirm. Order: **capacity → tp/EP → per-rank budget → M**
(tp sets per-rank shapes + the memory budget that caps M).
- **M (the batch sweet spot).** A weight-GEMM's `AI ≈ 2M/b` (b = bytes/weight), so it crosses the ridge at
  `M* ≈ ½·ridge·b` (EMR ridge≈51 → bf16 M≈25–50, fp8 M≈12–25; the 16–64 band). Confirm with a GEMM sweep on
  the model's dominant shapes → the M where real oneDNN BRGEMM hits **80–90% of the BW roofline**; treat it
  as a RANGE (M drifts in continuous batching). Bound: `M ≤ min(sweet-spot, memory-cap[GATE 0], latency-SLA)`.
  ⚠ M amortizes WEIGHT-bound ops only; per-sequence state / softmax ops are M-immune and become the frontier
  at M* (that's anchor discovery, `multiscale-optimization`).
- **tp / EP (sharding).** `tp* = smallest feasible tp clearing the capacity floor`:
  - **Capacity floor:** `tp_min = ⌈(model_resident + pools) / domain_RAM⌉` (GATE 0, per-NUMA/SNC domain). tp=1
    if it fits the node interleaved.
  - **Divisibility / padding:** feasible set = divisors of the sharded dims (heads/intermediate; **block-quant
    shard must stay a multiple of 128**) or paddable with `(padded−real)/real` < a few % waste.
  - **Comm cost (CPU-specific):** per-layer all-reduce is expensive → **tp>1 HURTS latency-bound decode**
    (measured GNR tp=2 = 2.8× worse than EMR tp=1). CPU tp is a **capacity/prefill lever, NOT a decode
    throughput lever** (reverse of GPU). So **minimize tp** for decode-dominated serving.
  - **EP-first for MoE:** expert-parallel splits experts whole — no padding, no per-layer all-reduce on the
    expert GEMM, sidesteps block-quant divisibility. EP primary for the MoE; TP secondary for dense/attention.
  - Confirm with a tp∈{1,2,4} micro-sweep (comm cost) + interleave-vs-per-domain, on the target node. See
    `sub-numa-clustering`, `runtime-config-tuning`.

## Exit criterion
Framework/glue share below threshold (kernel-domination ≥ ~0.90) at the chosen operating point.
The remaining wall is model kernels → descend to `meso-scale` for each dominant kernel.

## Pitfalls this scale exists to prevent
- Micro-tuning a kernel while 90% of wall is glue (wrong scale).
- Modeling a novel op's roofline as a bare GEMM — omits the macro seam costs → overstates (GATE 1).
- Treating batching as a kernel achievement — it's an operating-point; label it.
