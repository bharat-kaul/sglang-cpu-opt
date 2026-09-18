---
name: sub-numa-clustering
description: "Use RIGHT AFTER establish-achievable-performance and BEFORE the parallelize→tile→vectorize tuning loop, on any multi-socket / SNC (sub-NUMA-clustered) Xeon. Decides the NUMA/SNC DOMAIN you optimize WITHIN: SNC on/off, domain granularity (numa_nodes/sockets), the capacity-per-domain fit that determines whether a model must be sharded across domains, and the CPU serving mapping (tp = number of SNC clusters, one rank per domain, cpu+mem co-bound, first-touch). INCLUDES an UPFRONT OOM CHECK (run before the first launch): per-rank LOAD-TRANSIENT peak footprint (not just resident weights — dtype up-convert like fp4→fp8 doubling + checkpoint-held + AMX-prepack copies) vs ONE domain's RAM, deciding tp-shard vs tp=1-interleave, and how to keep the interleave memory policy from being clobbered by the framework's thread init. Everything downstream — the roofline ceiling, thread-count sweep, cache blocking, AMX vectorization, weight prepack — is measured and tuned within ONE domain and then replicated across domains, so the domain must be fixed FIRST or the tuning runs against a wrong, unstable interleaved/cross-domain footprint. USE FOR: placing a kernel or a whole model onto NUMA/SNC domains; choosing tp for CPU serving; the per-domain-RAM OOM check before a run; diagnosing cross-domain bandwidth thrash / erratic thread scaling / single-rank OOM at one-domain RAM. DO NOT USE FOR: intra-domain thread-count/affinity micro-tuning (that is openmp-parallelization), or GPU placement."
---

# Sub-NUMA Clustering (choose the optimization DOMAIN before tuning within it)

A Granite-Rapids-class Xeon exposes each socket as several **sub-NUMA clusters (SNC)** —
independent memory domains, each with its own local DRAM slice and bandwidth. Optimization
is only meaningful **relative to a domain**: the compute peak, the NUMA-local bandwidth
ceiling, and the working-set/footprint that tiling and vectorization tune against are all
**per-SNC-domain** numbers. So the FIRST decision is *which domain you optimize within*;
every per-kernel technique then operates inside one domain and replicates across domains.

## Where it sits in the sequence — and WHY (before tiling/vectorization)
```
0.  establish-achievable-performance   → per-DOMAIN ceilings (NUMA-local BW, per-SNC compute peak)
0.5 sub-numa-clustering  (THIS)        → fix the domain: SNC granularity, capacity fit,
                                          tp = #SNC, one rank/domain, cpu+mem co-bind, first-touch
1.  roofline-validation                → measured within one domain, vs that domain's ceiling
2.  openmp-parallelization             → thread count / affinity WITHIN the domain
3.  cache-blocking-tiling              → blocks sized to the domain's L2/L1 + NUMA-local BW
4.  amx-vectorization                  → per-core compute (domain-local operands)
5.  weight-prepacking-brgemm / 6. quantization
    → scale out: replicate the tuned per-domain kernel, one rank per SNC domain
```
**Order matters — SNC comes before tiling and vectorization**, because:
- The **ceiling** those steps target is per-domain. Tile/vectorize against a full-node
  interleaved footprint and you tune to the wrong (lower, unstable) number.
- Cross-domain first-touch or `--interleave=all` produces **erratic scaling** (measured on
  GNR: 84c > 96c < 128c) and full-node-interleaved lost to single-socket for a lone GEMM.
  The tuning loop is only stable once the domain is pinned.
- SNC is a **structural/topology + placement** decision; tiling/vectorization are per-kernel
  micro-optimizations *inside* it. Structure precedes micro-optimization.
- Cache blocking (L2/L1) and AMX (per-core TMUL) are themselves domain-agnostic, but the
  memory that **feeds** them is domain-local — so the domain still must be fixed first.

## Two facets
1. **Topology / placement (decide once, up front).** SNC on/off; nodes per socket
   (`numa_nodes / sockets`); how many domains to use; whether a lone op stays in one domain
   (usually yes) or replicates per domain. For a whole model: the **capacity fit** below.
2. **Execution (enforced downstream).** cpu+mem co-binding (`numactl --cpunodebind=<d>
   --membind=<d>`), first-touch allocation, one runtime pinned so it doesn't fight numactl.
   openmp-parallelization owns the intra-domain thread sweep; this skill owns the domain.

## Capacity-per-domain fit (the serving decision)
On CPU serving, each rank is sized to **one domain's** memory, not the whole node:
- SGLang CPU computes per-rank memory as `total_free / n_numa` and **binds each rank to one
  NUMA/SNC domain** (first-touch lands there). A single-rank (`tp=1`) process therefore caps
  at ONE domain's RAM regardless of total node RAM → a model larger than one domain **OOMs**.
- **Rule:** shard across domains with TP. `tp = #SNC domains` (or a divisor of it that also
  divides the attention head count). Per-rank weight footprint ≈ `model_bytes / tp` must fit
  one domain's RAM. Worked example: 806 GB fp8 model, 6 SNC domains × ~248 GB each →
  `num_attention_heads=128` is not divisible by 6, so `tp=4` (~201 GB/rank < 248 GB) with
  NUMA-local bandwidth.
- **Binding:** when `ranks ≤ #NUMA domains`, no explicit bind is needed (one rank/domain
  auto). When `ranks > #NUMA domains`, set `SGLANG_CPU_OMP_THREADS_BIND="c0-cN|..."` (one
  `|`-group per rank) or it asserts.
- **MLA caveat:** framework unaligned-TP head padding assumes GQA and can violate a
  single-latent-KV-head (MLA) invariant — verify config helpers no-op for MLA (see
  cpu-serving-integration).

## UPFRONT OOM CHECK (do this BEFORE the first run — it is cheap and saves hours)
The per-domain fit must be checked against the **load-transient PEAK**, not the resident model
size, and it decides tp vs interleave. Compute up front:
1. **Per-rank LOAD-PEAK footprint**, not just final weights. Load transiently holds MULTIPLE
   full copies: the on-disk checkpoint (held while processing), any dtype up-conversion
   (e.g. fp4→fp8 DOUBLES expert bytes; fp8→bf16), AND the AMX-prepacked copy. Measured on
   GNR/Flash: a ~275 GB fp8 model reached ~304 GB resident but the fp4 checkpoint alone was
   ~162 GB loaded *before* any dequant — instrument RSS per layer (`psutil`) if unsure.
   Rule of thumb: budget **peak ≈ 1.5–2× resident** for a load-time dtype convert + prepack.
2. **Compare to ONE domain's RAM** (`total_node_RAM / n_SNC`, ~248–258 GB on GNR SNC-on).
   If `per_rank_peak > one_domain_RAM` → you WILL OOM at that domain even with TBs free
   node-wide (the OOM-killer fires at ~one node's 256 GB, RSS far below total). Pick a path:
   - **(A) TP-shard (default for prefill / compute-bound):** `tp = #SNC` (or a divisor that
     also divides head count); per-rank peak ≈ `model/tp` must fit one domain. Natural
     per-domain memory + bandwidth. BUT TP adds an all-reduce every layer.
   - **(B) tp=1 + INTERLEAVE across domains (for memory-bound DECODE):** TP does NOT help
     memory-bound decode — each rank still streams its shard (same total bytes/token) and TP
     only *adds* all-reduce/coordination (measured: tp=4 decode ~8× SLOWER than tp=1 on the
     same layers). So for best decode latency keep `tp=1` and spread the model across all
     domains with an interleave memory policy. Requires: report full-node free memory to the
     framework's mem sizer, cap the KV pool (`--max-total-tokens`), and — critically — see
     the interleave-clobber learning below.
3. Reducing the load PEAK (so a tighter fit works): free intermediates per layer + return
   memory to the OS (`gc.collect()` + glibc `malloc_trim(0)`); note the CPU torch/`malloc`
   allocator does NOT return freed memory to the OS on its own.
4. **Scheduler cgroup cap (rule out FIRST — it masquerades as a NUMA OOM):** on Slurm/k8s a
   job with no explicit whole-node memory request gets a **cgroup memory limit** (~`DefMemPerCPU
   × allocated_CPUs`), often close to one domain's RAM. The process is then OOM-killed at that
   TOTAL RSS *regardless of how well pages are spread across domains* — so an `exit=137` at a
   round GB number well below node total is a cgroup cap, NOT NUMA concentration. Verify with
   `numastat -p <pid>` (spread = not NUMA-bound). Fix: request the whole node
   (`--exclusive --mem=0` on Slurm) for any load that approaches one-domain RAM.

## Loading data across domains — HOW the weights get placed (any model)
Overcoming the capacity constraint is a **placement** problem solved at LOAD time: you control
*which domain each weight page lands on*. Placement is decided by the OS NUMA **memory policy
in effect at the moment a page is first written** (first-touch) — NOT by the model code and
NOT by which core computes on it later. There are two mechanisms; pick per the OOM-check path:

- **(A) Per-domain first-touch loaders (the sharded / `tp=#domains` path).** Each rank (or
  loader thread) is cpu+mem bound to ONE domain (`numactl --cpunodebind=d --membind=d`, or
  SGLang's auto one-rank-per-domain) and writes ONLY its own shard of the weights. Because the
  writing thread is domain-local, first-touch lands each shard in that domain's DRAM →
  NUMA-local bandwidth, no policy tricks needed. This is the default and the fastest, but it
  requires the model to be SHARDED so every domain owns a disjoint slice.
- **(B) Interleave policy (the `tp=1` capacity path — one model too big for one domain).** Set
  an **interleave** memory policy (`numactl --interleave=all` or libnuma
  `numa_set_interleave_mask(numa_all_nodes_ptr)`) so the OS round-robins each page across all
  nodes **regardless of which thread touches it**. This is thread-independent, so it spreads
  the weights `1/N` per domain even when the loader is effectively single-threaded (the common
  case for a dequant/prepack loop). It trades some bandwidth (each rank now reads remote pages
  too) for the capacity to hold a model larger than one domain.

**The one rule that makes either work (and the #1 failure mode):** the policy must be ACTIVE
at the instant of allocation/first write. Two things silently defeat it — (1) a single-threaded
load/dequant loop first-touches everything to node 0 unless an interleave policy overrides it;
(2) the serving framework's thread-init (e.g. `sgl_kernel.init_cpu_threads_env`) **resets the
memory policy to node-local** when it pins threads, so any policy you set at process start is
gone before weights load. Therefore: **(re-)assert the placement policy AFTER thread-init and
BEFORE the weight-load loop**, and disable the framework's own membind (`SGLANG_AUTO_NUMA_BIND=0`)
so it doesn't re-pin the rank. **Verify placement, don't assume it** — check per-node RSS
(`numastat -p <pid>`, or sum `/proc/<pid>/numa_maps`) during load: you want ~`total/N` per node
for path B, or the shard sitting on its own node for path A. A node-0 pile-up = policy was
clobbered or never active.

## Procedure
1. Read the hardware profile: sockets, cores/socket, SNC nodes/socket, per-domain RAM &
   NUMA-local bandwidth (from establish-achievable-performance's STREAM/GEMM microbench).
2. **Lone kernel:** pick ONE domain to optimize within; bind cpu+mem to it; first-touch its
   operands. Tune (roofline → threads → tile → vectorize) against that domain's ceiling.
   Scale out only if the op is replicated across domains (e.g. per-rank weights).
3. **Whole model (serving):** apply the capacity-fit rule → choose `tp`; then pick the LOAD
   path (see "Loading data across domains"): sharded per-domain first-touch (A) if it fits per
   rank, else `tp=1` + interleave (B) for a model bigger than one domain / memory-bound decode.
   Set/re-assert the placement policy after thread-init, confirm per-node RSS during load, run.
4. Validate cross-domain behavior: if scaling collapses crossing a domain/socket, you are
   NUMA-bound → stay in-domain and replicate, do not interleave.

## Gate
- **Per-domain scaling efficiency** ≥ 0.9 up to one domain/socket (openmp gate, but measured
  in-domain). A collapse when crossing a domain = NUMA thrash, not a kernel limit.
- **Capacity fit:** per-rank footprint < one domain's RAM (else re-shard: raise tp).
- Record the winning `{SNC on/off, tp, domain, bind}` in the hardware profile so later ops
  and models start from it.

## Learnings (GNR, transferable)
- Full-node interleave is NOT free bandwidth: a lone GEMM was WORSE full-node-interleaved
  than single-socket; NUMA-local sharding (one rank/domain) is how you reach the ~1261 GB/s
  aggregate, not one un-sharded replica. (So interleave is a CAPACITY workaround for a model
  that must be tp=1, not a bandwidth win — expect BW below a NUMA-local shard.)
- `tp=1` on a big model OOMs at one-domain RAM even with TBs free node-wide — the OOM-killer
  fires at ~one node's 256 GB while total RSS is far below node total. Shard, OR interleave.
- **Interleave gets CLOBBERED — you must re-apply it.** `numactl --interleave=all` works in
  isolation (a 288 GB alloc spread exactly 1/6 across 6 nodes), but the framework's thread
  init (`sgl_kernel.init_cpu_threads_env`) resets the memory policy to node-LOCAL when it
  binds threads, so a tp=1 rank re-concentrates on node 0 and OOMs. Fix: RE-APPLY the
  interleave mask AFTER thread init and BEFORE weight load — `libnuma
  numa_set_interleave_mask(numa_all_nodes_ptr)` (verified: RSS then sailed past the one-node
  256 GB wall). Also `SGLANG_AUTO_NUMA_BIND=0` so the framework doesn't `--membind` the rank.
  And note a single-threaded load/dequant loop first-touches node 0, so the interleave POLICY
  (not first-touch) is what spreads it.
- Two runtimes (MKL/OMP) fighting numactl → erratic thread scaling; pin one.
- Fix the domain BEFORE the tuning loop; otherwise every roofline/tile/vectorize number is
  measured against a shifting, cross-domain footprint.
