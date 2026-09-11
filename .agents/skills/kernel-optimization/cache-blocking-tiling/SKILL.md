---
name: cache-blocking-tiling
description: "Use when a CPU kernel's working set exceeds L2 and operands must stream (large GEMM, attention, conv). Covers the M/N/K cache-blocking hierarchy, register/accumulator blocking, and keeping the reused panel L2-resident so the compute units stay fed. This is what converts the resident compute peak into achievable streamed throughput."
---

# Cache Blocking / Tiling

The gap between resident compute peak and streamed achievable is a data-movement
problem. Blocking is how you close it: size the loops so the reused operand stays
in cache and the compute unit never starves.

## Trigger
Working set (A, B, C) > L2, or roofline says memory/operand-feed bound. Load after
`amx-vectorization` has chosen the inner tile.

## Blocking hierarchy (fastest → slowest storage)
1. **Registers / AMX tiles:** the C accumulator block (`M_r x N_r`, e.g. 32x32 =
   4 tiles) stays resident across the entire K reduction — never store/reload
   partial sums. This is the register/accumulator-blocking rule.
2. **L1:** the current A sub-panel streams through L1.
3. **L2 (`l2_bytes_per_core`):** hold the reused B panel (`K_b x N_b`) in L2 so it
   is read once from DRAM and reused across all M blocks. Choose `K_b, N_b` s.t.
   `2 * K_b * N_b (bytes, bf16) ≲ 0.5 * l2_bytes_per_core` (leave room for A/C).
4. **L3 / DRAM:** stream the rest.

## Loop order
Outer over M-blocks (parallel across cores), then N-blocks, innermost the K
reduction that accumulates into the resident C tiles. This maximizes B reuse from
L2 and keeps C in registers — the structure the BRGEMM building block encodes
(`weight-prepacking-brgemm`).

## Procedure
1. From the profile read `l2_bytes_per_core`, `tile_m/n/k`.
2. Pick `M_r x N_r` = number of AMX accumulators × tile size (≥4 accumulators).
3. Solve `N_b, K_b` for L2 residency of the B panel; `M_b` for load balance.
4. Emit the blocked loop nest; verify B is read from DRAM ≈ once (perf: LLC/DRAM
   traffic ≈ K*N*2 bytes, not O(M) times that).

## Gate
Re-run `roofline-validation`. Achieved should rise toward `streamed_gemm_*`.
If DRAM traffic ≫ operand size, blocking is wrong (panel not staying in L2).
