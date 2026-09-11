---
name: roofline-validation
description: "Use when validating whether a measured CPU kernel (GEMM, attention, MLP) achieves its performance ceiling. Computes AMX/AVX compute peak and memory-bandwidth peak from a hardware profile, the ridge point, arithmetic intensity, and reports achieved % of roofline with a pass/fail verdict. Pairs with cpu-gemm-amx-bf16."
---

# Roofline Validation

Turn a raw throughput number into a defensible verdict. Gate against the
**achievable** ceiling measured by a saturating microkernel, not the paper peak.

## Peak vs achievable (measure the ceiling, don't assume it)
Before judging any kernel, establish the achievable ceilings on the target
silicon with microbenchmarks (run once per hardware profile, cache in its
`achievable` block):
- **Compute**: `tools/microbench/amx_peak.c` — resident-tile TDPBF16PS loop, zero
  memory traffic → achievable AMX compute TFLOP/s (also fixes FLOP/cycle empirically).
- **Memory**: `tools/microbench/stream_triad.c` → achievable DRAM GB/s.
Example (GNR 6980P): achievable = 350.8 TF/socket compute, 630.7 GB/s BW;
paper peak compute is ~351 TF/socket and 844.8 GB/s — so AMX compute is ~100%
achievable while DRAM tops out at ~75% of peak.

## Inputs
- A hardware profile (`hardware-profiles/<part>.yaml`) with its `achievable` block.
- Measured `achieved_tflops` (or GFLOP/s) for the kernel.
- Problem dims `M,N,K` and the actual DRAM bytes moved (or an upper bound).

## Procedure
1. `peak_compute = cores * flops_per_cycle_per_core * freq` (sanity only).
2. `achievable_compute` = microkernel result from the profile (PRIMARY ceiling).
3. `peak_bw`, `achievable_bw` similarly.
4. `ridge = achievable_compute / achievable_bw` (FLOP/byte).
5. `AI = 2*M*N*K / bytes_moved`. If `AI >= ridge` → compute-bound (gate against
   `achievable_compute`), else memory-bound (gate against `AI * achievable_bw`).
6. `efficiency = achieved / applicable_achievable_ceiling`.

## Verdict rules
- `efficiency >= 0.70` on a single socket, large square GEMM → **PASS**.
- `0.70 > efficiency >= 0.5` → **MARGINAL**: revisit blocking/threads/NUMA.
- `efficiency < 0.5` → **FAIL**: structural issue (wrong ISA dispatched, NUMA
  thrash, under-threaded).
- `efficiency > 1.0` → **INVALID CEILING**: the achievable measurement or the
  frequency is wrong; re-run the microkernel before reporting anything.

## Anti-pitfalls
- Never use max turbo for the compute ceiling; AMX throttles the all-core clock.
- Confirm the intended ISA actually ran (e.g. `ONEDNN_VERBOSE=1` showing
  `amx` kernels) before trusting a high number.
- Report the ceiling assumptions (freq, FLOP/cycle) alongside the % so the
  claim is auditable.
