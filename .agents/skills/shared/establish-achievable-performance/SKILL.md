---
name: establish-achievable-performance
description: "Use FIRST, before optimizing any CPU kernel. Establishes the achievable performance ceilings on the actual silicon via microbenchmarks: (1) the resident-tile AMX compute PEAK, (2) the realistic STREAMED-GEMM achievable, (3) DRAM bandwidth. Distinguishes peak (all operands in L1/registers) from achievable (operands streamed from cache/DRAM) so the optimization loop gates against the right number. Writes the ceilings into the hardware profile."
---

# Establish Achievable Performance (calibration)

The number you chase is NOT the datasheet peak. It is the *achievable* ceiling a
saturating microkernel sustains on this silicon. Critically, there are **two**
compute ceilings and they differ a lot — measuring against the wrong one is how
you get a fake "5.8x gap" (or, worse, ship a kernel that looks fine vs an
unreachable number).

## The two compute ceilings (do not conflate)

1. **Compute PEAK (resident).** All operands live in tile registers / L1; zero
   memory traffic. Measured by `tools/microbench/amx_peak.c` (back-to-back
   TDPBF16PS into ≥4 accumulators). This is an UPPER BOUND no streamed GEMM can
   reach. On GNR 6980P: ~350 TF/socket (≈100% of the 1024 FLOP/cyc/core AMX peak).
   Use it to (a) verify the FLOP/cycle constant and (b) confirm the AMX all-core
   clock — not as the GEMM target.

2. **Streamed-GEMM ACHIEVABLE.** A real GEMM must stream A and B through the cache
   hierarchy into the TMUL. The achievable ceiling is set by how fast you feed the
   tiles, i.e. by operand reuse and cache blocking (LIBXSMM/TPP docs are explicit:
   "results depend on whether operands are streamed or not"). Measure it with a
   tuned, weight-pre-packed BRGEMM reference (`weight-prepacking-brgemm`); if none
   is available, estimate it as the operand-feed-bounded value from the roofline.
   THIS is the gate the optimization loop uses.

## Memory ceiling

`tools/microbench/stream_triad.c` → achievable DRAM GB/s (single socket is the
reliable figure; interleaved full-node reads suffer first-touch/NUMA artifacts).
Sets the memory-bound side of the roofline.

## Procedure (once per hardware profile)

1. Build + run both microbenchmarks on-node (`tools/microbench/run_microbench.sh`)
   or auto-generate the whole profile with `tools/calibrate.py`.
2. Record into the profile `achievable` block:
   - `compute_peak_tflops_per_socket` (resident microkernel)
   - `streamed_gemm_tflops_per_socket` (BRGEMM reference; mark estimated if projected)
   - `mem_bw_gbs_per_socket` (STREAM)
3. All later skills gate against `streamed_gemm_*`, and treat `compute_peak_*` as
   the theoretical bound.

## Why this is step 0

The microkernel simultaneously (a) sets the realistic target, (b) empirically
fixes FLOP/cycle, and (c) measures the AMX all-core clock (which is well below max
turbo). Without it, every downstream "efficiency %" is meaningless.

## Bundled tools
- `tools/microbench/amx_peak.c` — resident-tile AMX BF16 compute peak.
- `tools/microbench/stream_triad.c` — STREAM triad memory bandwidth.
- `tools/calibrate.py` — probes the node + runs both, emits a filled profile (Day-0).
