---
name: model-roofline-analysis
description: "Use after decomposition and fusion analysis to build per-phase, per-op-class rooflines and prioritize the complete standalone optimization queue. Apply max(compute time, memory time) using labeled bounds and calibrated targets. ROI selects work order, never exempts low-share kernels from native C/C++ optimization."
---

# Model Roofline Analysis (top-down, Amdahl-ranked)

The lesson that created this skill: we roofline-analyzed the DSA indexer *kernel*
first and only discovered via Amdahl that it was ~0.5% of a weight-streaming-bound
decode. A model-level roofline run FIRST would have ranked precision/weight-
streaming above that kernel immediately. So: **analyze the whole model before any
kernel.** This tier orders the work; `kernel-feasibility-gate` evaluates designs
for every required kernel.

## Pipeline position
Discovery/analysis → coverage → source-faithful bring-up → profiling → standalone
optimization → reviewed integration, per the
[canonical acceptance policy](../../../../docs/methodology-playbook.md#canonical-acceptance-policy).
Do not require an unavailable full-model profile before implementing a missing functional path.
An analytical roofline is a workload-on-hardware prediction, not a measured model ceiling.
Build and publish it before a model run when possible. Use short targeted microbenchmarks to
refine material uncertain parameters; full-model profiling is validation, not a prerequisite.

## Inputs
Op graph; node achievable ceilings (`establish-achievable-performance`: compute +
mem BW, **per SNC/NUMA domain** — the atomic unit). Those ceilings come from
**uArch Performance Probe (`uarch-perf-probe`) `machine_constants.json`** — compute
peak per dtype, DRAM + **per-SNC-domain BW matrix**, ridge — measured + self-validated
on the target node, not hand-typed priors). Follow the
[matched target contract](../../../../docs/methodology-playbook.md#matched-end-to-end-target-publication).
Distinguish tensor/expert-parallel weight sharding from independent serving replicas:
replicas duplicate weight traffic, while ranks shard or replicate particular tensors.
Calibrate the allocated CPU sets and NUMA placement concurrently. Replicas sharing a
domain cannot each be assigned its isolated bandwidth; disjoint domains still need a
matched aggregate measurement. Do not scale bandwidth by a core/domain fraction without
evidence. Capacity and head/expert divisibility constrain legal TP, not independent
replica count. Count actual work and traffic for each deployment before aggregating. The
workload: prefill vs decode; **batch sweep (Xeon is
a low-batch, memory-bound target — sweep B = 1, 8, 16, 32)**; seq len / context;
speculative M.

## Procedure
1. **Split by PHASE.** Prefill (large M, compute-heavy, activation-streamed) vs
   decode (M=1 or small spec-M, **weight-streaming-bound**). Analyze separately —
   the bottleneck flips between them.
2. **Per-op-class roofline.** For attention (MLA/GQA/sparse), MoE grouped-GEMM,
   dense projections, norm/rope/act, embedding/lm_head, indexer:
   - FLOPs and BYTES (weights + activations + KV). The **decode weight-streaming
     term can dominate**: active bytes / matched allocated bandwidth gives a logical
     streaming time bound, not a complete achievable model floor.
   - AI = FLOPs/bytes → compute-bound (AI≥ridge) or memory-bound (AI<ridge).
   - **Batched-MoE decode (critical on CPU):** dense/attention weights stream ONCE
     and amortize over the batch, but each token routes to a DIFFERENT expert subset,
     so the distinct-expert working set GROWS with batch. The uniform estimate
     `E*(1 - (1 - 1/E)^(k*B))` **OVER-counts** (it is the dummy-random result) — real
     routing can be concentrated. Use existing per-layer/per-forward observations or
     explicit analytical routing scenarios with sensitivity ranges. Historical DSv4 **59**
     at B32 is a scenario, not a portable measured constant. A fitted coverage curve may
     estimate unmeasured points when labeled as such; it cannot certify those points.
     Do not require a real-weight model load to publish such a prediction. Per-step bytes =
     non_expert_bytes + distinct_experts(B)*per_expert_layer_bytes*moe_layers + KV(B).
     For one replica, `tok/s_bound = B / (step_bytes / allocated_BW)`.
     Socket throughput needs all replicas' work and shared-resource schedule, not
     independent full-bandwidth copies of that equation.
     Sweep B=1,8,16,32 — a model with few experts / high top-k saturates early (B=32
     may stream ~half the whole model/step) and batching barely helps; a model with
     many experts / low top-k scales much better. **Name the regime (compute- /
     memory- / LATENCY-bound):** at small B the per-step **dispatch/latency** overhead
     can dominate BOTH compute/step and memory/step, so measured decode sits far below
     the BW roof (DSv4 B=32 = **8% of its BW roof = latency-bound, NOT BW-bound**). The
     `bytes/BW` roofline is then a **large-batch ASYMPTOTE**; the lever to reach it is
     lower fixed overhead through native composition/fusion and suitable threading;
     larger batch is another lever only if the agreed operating point permits it.
3. **Amdahl-rank.** Each op-class's share of the phase time = its (bytes/BW) or
   (FLOPs/compute), whichever binds. Sort. Identify the top contributors and the
   long tail. **Two ceilings — report BOTH:** the single dominant-resource roofline
   (`FLOPs/peak` or `bytes/BW`) is an **IDEALIZED upper bound** that assumes every
   OTHER op-class is free. Build an analytical time budget at matched shape/dtype/cache
   conditions: each op floor is `max(FLOPs/compute_rate, bytes/bandwidth)`.
   Sum only serialized classes; model overlap explicitly. Time efficiency is
   `target_time / measured_time` (equivalently measured throughput / target throughput).
   Report `idealized_time / target_time` as the modeled structural fraction.
   A best-observed library rate remains provisional until justified, not an automatic ceiling.
   Declare rate-transfer, native-dequant, reuse and fusion assumptions. List exclusions;
   do not present an incomplete subtotal as a complete model ceiling or fit the target to
   current runtime. Refine the most sensitive inputs with short standalone probes if needed.
   Keep hardware bounds, analytical estimates, calibrated targets and clean measurements
   separately labeled. Publish prefill/decode targets per platform/configuration; mark
   missing targets pending and omit their percentage attained.
   *On DSv4 prefill only ~40% of wall-time is the modeled GEMM; the
   other ~60% (glue/eager-math/fallback-matmul) is what the idealized `FLOPs/peak`
   pretends is free — so "43% of roofline" overstates the gap and hides where it is.*
4. **Lever per top item.** Memory-bound → **reduce operand precision** (fewer weight
   bytes — usually the #1 decode lever), prefetch, cache-block, fusion. Compute-bound
   → AMX/vectorization, tiling, prepack. (Uses the kernel-level roofline concepts
   from `roofline-validation`.)
5. **ROI priority.** Use `share * (1 - target_time / baseline_time)` for non-overlapped
   work (or `share * (1 - 1/speedup)`). Low ROI lowers priority, not the standalone
   acceptance requirement. Mark each candidate: covered by a donor kernel (retune/config)
   vs needs a new kernel (→ `kernel-feasibility-gate`).
6. **Present the ranked plan to the user** (table: phase, op-class, share%, regime,
   lever, covered/new). This is the optimization roadmap.

## Output — ranked optimization plan
`[{phase, op_class, share_pct, regime, recommended_lever, mapping: donor|new-kernel|
config, roi}]`. Feeds coverage/wiring for config/donor items; feeds
`kernel-feasibility-gate` for every incomplete standalone kernel, ordered by ROI.

## Worked example (DeepSeek-V4 decode, config-grounded, FULL 2-socket GNR node)
GNR full node ≈ 1261 GB/s aggregate (2×630.7, NUMA-local sharded; ~0.85 realistic).
**Flash-v4.1** (~286B, 256 experts top-6, FP8) scales well: analytical B=1 ~81 → B=32 ~260
tok/s aggregate (distinct experts/layer 6→**59** measured real-router = ~23% of experts
streamed at B=32 — *not* the uniform/dummy ~135 = 53%, which is a mode-collapse over-count).
**V4-Pro flash** (~1.4T, 384 experts top-8, FP8) saturates: B=1 ~22 → B=32 ~58 tok/s,
0.55 s/token latency (B=32 streams ~half the 1.4 TB model/step). Decode ranking (both):
(1) weight streaming ≈ dominant → **precision (fp8/int4)** #1 lever; (2) MLA/MoE donor
paths; (3) DSA indexer nn kernel ≈ **0.5%** (M=1) → lower priority, still required. The batch sweep, not a single
B=1 number, is the decision input. **NB — these are IDEALIZED BW ceilings:** the MEASURED
DSv4 decode is **overhead/latency-bound** and sits far below (B=32 = 8% of the BW roof), so
score against the achievable time-budget, and **validate recoverable headroom with a cheap
counterfactual ablation** (timing-only proxy; preserve downstream routing/cache/scheduling
or label the bound unvalidated) when prioritizing optimization designs — see `kernel-feasibility-gate` +
`overhead-attribution`.

## Gate
Show the ranked plan to the user. The analytical and empirical tiers prioritize all
required kernels; the kernel tier confirms each design with a microbenchmark.
Minimal functional bring-up can precede a model profile. Final native optimization
and opportunity-disposition requirements apply regardless of end-to-end share.

**The analytical bound is a hypothesis, not a verdict.** A computed roofline (model- or kernel-level) is a
*prediction* of the ceiling; the measured-vs-predicted gap ("integration efficiency") is only trustworthy once
the empirical tier tests it. Never declare an op "at ceiling" from the analytical number alone — a config sweep
(tiles / grain / effective-threads) must show the bound does not move, and the nominal roofline must be the
*operative* ceiling for that op's regime (not a byte/FLOP roof over a dispatch/selection/cache-resident op).
See `roofline-validation` ("a low % is a tested hypothesis") and `runtime-config-tuning`.
