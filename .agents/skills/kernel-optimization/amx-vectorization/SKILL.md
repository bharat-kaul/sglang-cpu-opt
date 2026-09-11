---
name: amx-vectorization
description: "Use when a compute-bound CPU kernel can use Intel AMX tiles or AVX-512 for dense BF16/FP16/INT8 math. Covers AMX tile shapes and VNNI operand layout, multiple accumulators to hide TDPBF16PS latency, FP32 accumulation, and graceful AVX-512 fallback. Picks the widest vector/matrix unit the capability contract allows."
---

# AMX / Vectorization

Map the inner math onto the widest matrix/vector unit the hardware advertises.

## Capability contract
| Path | Requires |
|------|----------|
| AMX BF16 tile | `amx_tile` + `amx_bf16` |
| AMX FP16 tile | `amx_tile` + `amx_fp16` (GNR+ only) |
| AMX INT8 tile | `amx_tile` + `amx_int8` (see `quantization-amx-int8`) |
| AVX-512 BF16 fallback | `avx512_bf16` |
Probe via `lscpu` + `ONEDNN_VERBOSE=1` (expect `amx` primitive). Do NOT trust
`torch._C._cpu._is_amx_tile_supported()` (False on GNR while AMX runs).

## AMX inner kernel rules
1. **Tile shape (BF16):** A `TILE_M x TILE_K` (16x32), B `TILE_K x TILE_N` (32x16
   VNNI2), C `TILE_M x TILE_N` FP32. Require `N % TILE_N == 0`, `K % TILE_K == 0`;
   pad otherwise.
2. **FP32 accumulation:** TDPBF16PS accumulates FP32 natively — never accumulate
   in BF16.
3. **≥4 accumulators:** issue back-to-back tile ops into ≥4 independent C tiles to
   hide the multi-cycle TDPBF16PS latency behind its throughput (this is what the
   `amx_peak.c` microkernel does to reach the compute peak).
4. **VNNI operand layout:** B (and packed A) must be in VNNI/tile order for
   contiguous tile loads → see `weight-prepacking-brgemm`.

## Fallback
If AMX absent but `avx512_bf16` present, use AVX-512 dot-product (lower peak);
record the capability gap. If a needed ISA (e.g. AMX-FP16) exists in hardware but
no kernel emits it yet, fall back + flag a scaffolding task — never emit
unrunnable code.

## Gate
Measure FLOP/cyc/core achieved vs the profile's `amx_bf16_flops_per_cycle_per_core`.
On a resident microkernel expect ~100%; on a streamed GEMM this step alone will
not reach peak — combine with `cache-blocking-tiling` + `weight-prepacking-brgemm`.
