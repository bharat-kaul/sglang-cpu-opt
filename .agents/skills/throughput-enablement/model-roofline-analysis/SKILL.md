---
name: model-roofline-analysis
description: "TOP tier of the roofline hierarchy — whole-MODEL performance analysis that runs after decomposition + fusion-analysis and BEFORE any kernel-level work. Splits the model by phase (prefill vs decode), builds a per-op-class roofline (FLOPs, weight+activation+KV bytes, arithmetic intensity, compute- vs memory-bound), Amdahl-RANKS each op-class's share of phase time, and emits a ranked high-ROI optimization plan with the right lever per item (memory-bound → precision/prefetch/fusion; compute-bound → AMX/tiling). Only op-classes above the ROI threshold descend to the kernel-level `kernel-feasibility-gate`. Prevents optimizing/authoring kernels that don't move end-to-end time (e.g. a compute kernel in a weight-streaming-bound decode)."
---

# Model Roofline Analysis (top-down, Amdahl-ranked)

The lesson that created this skill: we roofline-analyzed the DSA indexer *kernel*
first and only discovered via Amdahl that it was ~0.5% of a weight-streaming-bound
decode. A model-level roofline run FIRST would have ranked precision/weight-
streaming above that kernel immediately. So: **analyze the whole model before any
kernel.** This tier decides WHAT to optimize; `kernel-feasibility-gate` decides
HOW/whether for the survivors.

## Pipeline position
`model-op-decomposition → fusion-analysis → model-roofline-analysis →
model-profile-hotspots → coverage-gate → (wire | kernel-feasibility-gate →
kernel-authoring)`. This analytical ranking picks the targets and decides the
profiled run is worth doing; `model-profile-hotspots` then MEASURES them and the
measured hotlist gates entry to the kernel tier.

## Inputs
Op graph; node achievable ceilings (`establish-achievable-performance`: compute +
mem BW, **per SNC/NUMA domain** — the atomic unit). Those ceilings come from
**uArch Performance Probe (`uarch-perf-probe`) `machine_constants.json`** — compute
peak per dtype, DRAM + **per-SNC-domain BW matrix**, ridge — measured + self-validated
on the target node (never hand-typed priors; on a new uarch it is the sole source).
Aggregate ceiling =
`domains_used × per_domain_ceiling`, reached ONLY with NUMA-local weight sharding
(TP / expert-parallel, one rank per domain); a single un-sharded replica is capped
at ONE domain's BW. **SNC bake-in (do not assume the ideal full node):** usable
domains = the *achievable* `tp` from `sub-numa-clustering`, bounded by per-domain
CAPACITY fit AND head/expert DIVISIBILITY — which may be < total domains, leaving
domains idle and a sub-full-node ceiling. Use `effective_ceiling =
(tp_used / n_total_domains) × full_node_ceiling`. Worked: 6 SNC domains but
`num_attention_heads=128` forbids tp=6 → tp=4 → decode floor uses only 4/6 of node
BW (≈841 of 1261 GB/s). Feed the EFFECTIVE ceiling into every roofline below. The
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
     term dominates**: `active_bytes / full_node_BW` = ms/token floor.
   - AI = FLOPs/bytes → compute-bound (AI≥ridge) or memory-bound (AI<ridge).
   - **Batched-MoE decode (critical on CPU):** dense/attention weights stream ONCE
     and amortize over the batch, but each token routes to a DIFFERENT expert subset,
     so the distinct-expert working set GROWS with batch. Expected distinct experts
     per layer for B tokens, top-k of E: `E*(1 - (1 - 1/E)^(k*B))`. Per-step bytes =
     non_expert_bytes + distinct_experts(B)*per_expert_layer_bytes*moe_layers + KV(B).
     agg_tok/s = B / (step_bytes / full_node_BW); per-user latency = step time.
     Sweep B=1,8,16,32 — a model with few experts / high top-k saturates early (B=32
     may stream ~half the whole model/step) and batching barely helps; a model with
     many experts / low top-k scales much better. Decode stays memory-bound across
     this range on Xeon (check compute/step ≪ memory/step).
3. **Amdahl-rank.** Each op-class's share of the phase time = its (bytes/BW) or
   (FLOPs/compute), whichever binds. Sort. Identify the top contributors and the
   long tail.
4. **Lever per top item.** Memory-bound → **reduce operand precision** (fewer weight
   bytes — usually the #1 decode lever), prefetch, cache-block, fusion. Compute-bound
   → AMX/vectorization, tiling, prepack. (Uses the kernel-level roofline concepts
   from `roofline-validation`.)
5. **ROI gate.** projected_speedup × Amdahl_share ≥ threshold → candidate; else
   **SKIP** (record). Mark each candidate: covered by a donor kernel (retune/config)
   vs needs a new kernel (→ `kernel-feasibility-gate`).
6. **Present the ranked plan to the user** (table: phase, op-class, share%, regime,
   lever, covered/new). This is the optimization roadmap.

## Output — ranked optimization plan
`[{phase, op_class, share_pct, regime, recommended_lever, mapping: donor|new-kernel|
config, roi}]`. Feeds coverage/wiring for config/donor items; feeds
`kernel-feasibility-gate` ONLY for the high-ROI new/tuned-kernel items.

## Worked example (DeepSeek-V4 decode, config-grounded, FULL 2-socket GNR node)
GNR full node ≈ 1261 GB/s aggregate (2×630.7, NUMA-local sharded; ~0.85 realistic).
**Flash-v4.1** (~286B, 256 experts top-6, FP8) scales well: B=1 ~81 → B=32 ~260 tok/s
aggregate (distinct experts/layer 6→135 = 53% of the model streamed at B=32),
latency ~120 ms. **V4-Pro flash** (~1.4T, 384 experts top-8, FP8) saturates: B=1 ~22
→ B=32 ~58 tok/s, 0.55 s/token latency (B=32 streams ~half the 1.4 TB model/step).
Decode ranking (both): (1) weight streaming ≈ dominant → **precision (fp8/int4)** #1
lever; (2) MLA/MoE donor paths; (3) DSA indexer nn kernel ≈ **0.5%** (M=1) → **SKIP**.
The batch sweep, not a single B=1 number, is the decision input — it exposes that the
1.4T model is bandwidth-saturated and batching buys little, while the 286B model
scales to ~220–260 tok/s.

## Gate
No kernel-level roofline or authoring begins for an op until model-roofline-analysis
has ranked it above the ROI threshold AND `model-profile-hotspots` has confirmed it
with a measured profile, and the user has seen the ranked plan. The analytical model
tier picks the targets; the empirical tier measures them; the kernel tier
(`kernel-feasibility-gate`) confirms each survivor with a microbench.
