---
name: openmp-parallelization
description: "Use when a CPU kernel must scale across cores/sockets on Intel Xeon. Covers thread count selection, OpenMP affinity/pinning, NUMA/SNC binding, physical-vs-SMT core choice for AMX, and scaling-efficiency validation. Apply after the single-core kernel is correct; gate on measured scaling."
---

# OpenMP Parallelization

Turn a correct single-core kernel into one that scales — without NUMA thrash or
oversubscription.

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
3. **NUMA / SNC.** Bind compute and memory to the same domain:
   `numactl --cpunodebind=<d> --membind=<d>`. Prefer a single socket for a lone
   op; only span sockets when the problem amortizes cross-socket UPI. Honor the
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
- Full-node interleaved was WORSE than single-socket for a lone GEMM on GNR — do
  not reflexively use all cores.
- `MKL_THREADING_LAYER`/`OMP` runtimes fighting `numactl` produced erratic scaling
  (84c > 96c < 128c). Pin one runtime, verify with the scaling sweep.
