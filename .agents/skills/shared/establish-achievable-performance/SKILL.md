---
name: establish-achievable-performance
description: "Use FIRST, before optimizing any CPU kernel. Establishes the achievable performance ceilings on the actual silicon via microbenchmarks: (1) the resident-tile AMX compute PEAK, (2) the realistic STREAMED-GEMM achievable, (3) DRAM bandwidth. Distinguishes peak (all operands in L1/registers) from achievable (operands streamed from cache/DRAM) so the optimization loop gates against the right number. Writes the ceilings into the hardware profile."
---

# Establish Achievable Performance (calibration)

Use the [canonical acceptance policy](../../../../docs/methodology-playbook.md#canonical-acceptance-policy).
Separate idealized/resident bounds, calibrated operation-specific targets and
provisional best-observed baselines. A microkernel rate alone cannot establish
the achievable target for a full streamed kernel.

First apply the [startup infrastructure gate](../../../../docs/methodology-playbook.md#mandatory-startup-infrastructure-gate).
Budget and allocate the calibration probe itself, verify actual node/ISA/NUMA/software identity,
and reuse a historical calibration only when its contract matches. Do not benchmark the login
host as a stand-in for the target, or treat old hardware profiles as current availability.

For multi-replica publication, use the
[matched end-to-end target contract](../../../../docs/methodology-playbook.md#matched-end-to-end-target-publication).
Measure the actual concurrent CPU/NUMA layout, including shared-domain contention; do not multiply
isolated rates by replica count. Keep independent process samples and raw timing intervals.
Ensure the aggregate memory working set exceeds the applicable shared LLC before labeling a
probe DRAM-oriented; document logical bytes versus write-allocate/controller traffic. Warm
library GEMMs and resident-tile probes remain calibration inputs, not packed-model targets.
The model roofline itself is analytical: map source-derived work to these resource rates
with explicit assumptions and sensitivity. No full-model run is required. Refine a material
shape/precision/conversion uncertainty with a short targeted microbenchmark, not a model load.

## The two compute ceilings (do not conflate)

1. **Compute PEAK (resident).** All operands live in tile registers / L1; zero
   memory traffic. Measured by `tools/microbench/amx_peak.c` (back-to-back
   TDPBF16PS into ≥4 accumulators). This is a resident-probe reference bound, not
   a promised streamed rate. On GNR 6980P: ~350 TF/socket (≈100% of the 1024 FLOP/cyc/core AMX peak).
   Use it to (a) verify the FLOP/cycle constant and (b) confirm the AMX all-core
   clock — not as the GEMM target.

2. **Streamed-GEMM ACHIEVABLE.** A real GEMM must stream A and B through the cache
   hierarchy into the TMUL. The achievable ceiling is set by how fast you feed the
   tiles, i.e. by operand reuse and cache blocking (LIBXSMM/TPP docs are explicit:
   "results depend on whether operands are streamed or not"). Measure it with a
   tuned, weight-pre-packed BRGEMM reference (`weight-prepacking-brgemm`) at matching
   shapes/locality/cache conditions. Label an estimate or best-observed library number
   provisional; it is not proof of a ceiling. Justify the acceptance target with the
   applicable resource model and measurement, including packing/conversion costs.

## Memory ceiling

`tools/microbench/stream_triad.c` → achievable DRAM GB/s. Measure it **per SNC/NUMA
domain** (cpu+mem bound to one domain) — that is the atomic unit; interleaved
full-node reads can suffer first-touch/NUMA artifacts. Prefer concurrently measured aggregate
bandwidth for the actual layout; `domains_used × per_domain_BW` is only an explicitly labeled
scaling assumption until tested. Replicas in the same domain share its bandwidth.
This sets the memory-side input to the analytical roofline.

## Procedure (once per hardware profile)

0. **Run uArch Performance Probe (uPP) FIRST — it is the primary calibration engine**
   (`tools/uarch_perf_probe/`, skill `uarch-perf-probe`). One run emits
   `machine_constants.json` covering the streamed-achievable compute peak per dtype, the
   DRAM triad BW, the **per-SNC-domain BW matrix** (regenerates the SNC constants, not
   assumed), the roofline ridge, the gather-stage crossover, and thread scaling — and it
   **self-validates** (frequency hygiene + ISA presence + DCE-guarded timing), so an
   unpinned/mis-configured node is flagged instead of silently corrupting the ceilings. On a
   new uarch (e.g. a GNR follow-on) this is the *sole* source of the constants; on GNR it
   refreshes/verifies the priors and catches drift.
1. Map the uPP outputs into the profile `achievable` block (see step 2). uPP's `compute_peak`
   is the **streamed-achievable** (oneDNN/AMX GEMM) number; the **resident** compute PEAK
   (upper bound) is the complementary native probe `tools/microbench/amx_peak.c` (a uPP
   native-probe extension point). Legacy path: `tools/microbench/run_microbench.sh` /
   `tools/calibrate.py` still work but are subsumed by uPP.
2. Record into the profile `achievable` block:
   - `compute_peak_tflops_per_socket` (resident microkernel — amx_peak native probe)
   - `streamed_gemm_tflops_per_socket` (uPP `compute_peak.bfloat16`; mark estimated if projected)
   - `mem_bw_gbs_per_socket` (uPP `memory.dram_triad`; per-SNC-domain from uPP `numa.bw_matrix`
     — aggregate = domains_used × per_domain)
3. For the idealized roofline, pair the labeled resident/native compute bound with the
   relevant memory bandwidth; `ridge = compute_rate / bandwidth`.
   For acceptance, justify a matched operation-specific target using
   `floor_time = max(FLOPs/compute_rate, bytes/bandwidth)` plus explicitly modeled
   conversion, recurrence and dispatch costs. Do not relabel a best-observed streamed
   rate as the hardware compute peak, or infer the bottleneck from that rate alone.
   Cache blocking makes higher reuse possible; it does not prove resident-peak utilization.

## Why this is step 0

The microkernel simultaneously (a) sets the realistic target, (b) empirically
fixes FLOP/cycle, and (c) measures the AMX all-core clock (which is well below max
turbo). Without it, every downstream "efficiency %" is meaningless.

## Bundled tools
- **uArch Performance Probe (`tools/uarch_perf_probe/`) — PRIMARY.** Full self-validating
  suite → `machine_constants.json` (streamed compute peak, DRAM + per-SNC-domain BW, ridge,
  gather crossover, thread scaling). Run this first; the rest are complements/legacy.
- `tools/microbench/amx_peak.c` — resident-tile AMX BF16 compute peak (the UPPER-BOUND number
  uPP does not yet measure natively; add as a uPP native probe).
- `tools/microbench/stream_triad.c` — STREAM triad memory bandwidth (subsumed by uPP `memory`).
- `tools/calibrate.py` — legacy Day-0 profile filler (subsumed by uPP).
