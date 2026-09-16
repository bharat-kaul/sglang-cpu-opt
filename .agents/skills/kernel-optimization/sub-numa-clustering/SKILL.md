---
name: sub-numa-clustering
description: "Use RIGHT AFTER establish-achievable-performance and BEFORE the parallelize→tile→vectorize tuning loop, on any multi-socket / SNC (sub-NUMA-clustered) Xeon. Decides the NUMA/SNC DOMAIN you optimize WITHIN: SNC on/off, domain granularity (numa_nodes/sockets), the capacity-per-domain fit that determines whether a model must be sharded across domains, and the CPU serving mapping (tp = number of SNC clusters, one rank per domain, cpu+mem co-bound, first-touch). Everything downstream — the roofline ceiling, thread-count sweep, cache blocking, AMX vectorization, weight prepack — is measured and tuned within ONE domain and then replicated across domains, so the domain must be fixed FIRST or the tuning runs against a wrong, unstable interleaved/cross-domain footprint. USE FOR: placing a kernel or a whole model onto NUMA/SNC domains; choosing tp for CPU serving; diagnosing cross-domain bandwidth thrash / erratic thread scaling. DO NOT USE FOR: intra-domain thread-count/affinity micro-tuning (that is openmp-parallelization), or GPU placement."
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

## Procedure
1. Read the hardware profile: sockets, cores/socket, SNC nodes/socket, per-domain RAM &
   NUMA-local bandwidth (from establish-achievable-performance's STREAM/GEMM microbench).
2. **Lone kernel:** pick ONE domain to optimize within; bind cpu+mem to it; first-touch its
   operands. Tune (roofline → threads → tile → vectorize) against that domain's ceiling.
   Scale out only if the op is replicated across domains (e.g. per-rank weights).
3. **Whole model (serving):** apply the capacity-fit rule → choose `tp`; set binding; confirm
   per-rank footprint fits one domain; then run.
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
  aggregate, not one un-sharded replica.
- `tp=1` on a big model OOMs at one-domain RAM even with TBs free node-wide — the memory
  accounting and binding are per-domain. Shard.
- Two runtimes (MKL/OMP) fighting numactl → erratic thread scaling; pin one.
- Fix the domain BEFORE the tuning loop; otherwise every roofline/tile/vectorize number is
  measured against a shifting, cross-domain footprint.
