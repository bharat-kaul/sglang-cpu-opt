---
name: openmp-parallelization
description: "Use when a CPU kernel must scale across cores/sockets on Intel Xeon. Covers the WORK-PER-THREAD (grain-size) floor chosen up front, thread count selection, OpenMP affinity/pinning, NUMA/SNC binding, physical-vs-SMT core choice for AMX, and scaling-efficiency validation. Apply after the single-core kernel is correct; gate on measured scaling. Key law: when per-thread work drops below the sync/spawn overhead, more threads makes it SLOWER (can invert catastrophically) -- size the partition so per-thread work exceeds the overhead floor."
---

# OpenMP Parallelization

Turn a correct single-core kernel into one that scales — without NUMA thrash or
oversubscription.

## The grain-size floor (decide work-per-thread UP FRONT, before picking thread count)
Parallel speedup = useful work − (fork/join + barrier + false-sharing + per-thread setup).
When the **per-thread work falls below that overhead**, synchronization DOMINATES and adding
threads makes the op *slower* — sometimes catastrophically (non-monotonic / inverted scaling).
So the FIRST decision is not "how many threads" but "is there enough work per thread":
- Estimate per-thread work = total FLOPs (or bytes) ÷ threads, over the parallel dimension
  actually being split (M tokens, N cols, or experts). If a decode step splits M=1 across 60
  threads, each thread does ~nothing and pays full barrier cost.
- Require `work_per_thread ≥ overhead_floor` (the barrier/spawn cost — a measurable machine
  constant; see `uarch-perf-probe` thread-scaling). If not, **reduce the thread count** (or
  fuse/batch to raise the work) rather than parallelize into the floor.
- The scaling curve can be non-monotonic — sweep it; the optimum is often well below all-cores
  for small-batch / decode / few-expert-active ops. Small work → few threads is FASTER.
- When you do NOT own the kernel (a library op that over-threads small work), fix it at deploy
  time by capping threads around the call — see `runtime-config-tuning`.

## Trigger
Op runs on more than one core (nearly always). Load after the op is functionally
correct and the per-core kernel path is chosen.

## Procedure
1. **Thread count.** Start with one thread per **physical** core of the target
   domain. For AMX, ignore the SMT sibling — the TMUL unit is shared, so 2 threads
   per core do not add FLOPs and hurt cache. Sweep {½, ¾, 1}× physical cores; more
   is not always better (a lone 8Kx8K GEMM peaked at 96 of 128 cores on GNR).
2. **Affinity / pinning.** `OMP_PROC_BIND=close`, `OMP_PLACES=cores` (or
   `KMP_AFFINITY=granularity=fine,compact,1,0`). Never leave threads unpinned.
3. **NUMA / SNC.** The domain GRANULARITY + capacity decision is `sub-numa-clustering`
   (done BEFORE this loop); here you just ENFORCE it. Bind compute and memory to the
   chosen domain: `numactl --cpunodebind=<d> --membind=<d>`. Prefer a single socket for
   a lone op; only span sockets when the problem amortizes cross-socket UPI. Honor the
   profile's SNC layout (`numa_nodes / sockets` = nodes per socket).
4. **First touch.** Initialize/allocate data on the thread/domain that will use it
   (first-touch policy) so pages land in the local NUMA node.
5. **Work partition.** Partition the parallel (e.g. M) dimension into per-core
   blocks large enough to amortize fork/join but small enough to balance load.

## Gate (scaling efficiency)
`scaling_eff = tflops(n_threads) / (n_threads/1 * tflops(1_core_domain))`.
Expect ≥ 0.9 up to a single socket for a compute-bound op. If it collapses when
crossing a socket, you are NUMA-bound → stay single-socket or replicate weights
per socket. Record the winning thread/affinity/NUMA config in the profile.

## Anti-pitfalls
- **Over-threading small work = inverse scaling (the grain-floor failure).** DeepSeek-V4 MoE
  `fused_experts_cpu` on a small token batch measured **0.1 ms @ 4 threads vs 2432 ms @ 60**
  (~24,000× slower) — the kernel was at its floor with few threads; the default 60 buried it in
  sync/padding overhead. A one-shot thread-count micro-sweep picked 4. Always sweep; never
  assume all-cores for small-batch/decode ops.
- Full-node interleaved was WORSE than single-socket for a lone GEMM on GNR — do
  not reflexively use all cores.
- `MKL_THREADING_LAYER`/`OMP` runtimes fighting `numactl` produced erratic scaling
  (84c > 96c < 128c). Pin one runtime, verify with the scaling sweep.
