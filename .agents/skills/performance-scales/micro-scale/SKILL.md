---
name: micro-scale-optimization
description: "MICRO (micro-architectural) scale of the multi-scale CPU performance tree (do LAST, only on kernel-dominant ops already correct at the meso scale). Goal: get each kernel's INNER LOOP at the AMX/VNNI compute peak — the tile op (dpbf16/dpbusd), VNNI packing, inline precision conversion (fp8→bf16 in the load with the scale folded into the FMA), fp32 accumulation, round-to-nearest store, and the FUSED epilogue (activation/dequant in the store, never a separate pass). Encodes the SETTLED null-result traps (thread-cap, ILP banking, prefetch distance) so runs are not re-spent re-testing what the HW/compiler already handles. Load after meso-scale; route to amx-vectorization, cpu-gemm-amx-bf16, quantization-amx-int8, uarch-perf-probe."
---

# Micro-scale optimization — get the INNER LOOP at the AMX/VNNI peak

Smallest scale, done last. The question: **does this kernel's inner loop run at the
microkernel compute peak, with the epilogue fused into the tile-store?** Only matters once
macro (kernel-dominated) and meso (data movement at roofline) are already handled.

## The inner-loop recipe (donor: gemm_fp8.cpp / gemm_int8.cpp / gemm_int4.cpp)
1. **Tile op**: bf16 → `_mm512_dpbf16_ps`; int8/int4 → `_mm512_dpbusd[s]_epi32`. FP32 accumulation in
   a register-resident tile buffer across the K reduction.
2. **VNNI packing** (`vec_pack.h`): weights pre-transposed to VNNI2 (2 elems/32-bit) so the tile op
   streams contiguous packed operands; pack ONCE (meso), consume packed here.
3. **Inline precision conversion**: fp8 e4m3 → bf16 via `CVT_FP8_TO_BF16_EXT` IN the load (not upcast
   to a separate fp32 buffer first); per-block (K/128) scale FOLDED into the accumulate FMA
   (`_mm512_fmadd_ps(vsum, vscale, vc)`), not a separate scale pass. See `compute-from-native-precision`.
4. **FUSED epilogue** (the micro half of whole-operator fusion): activation + dequant + bias happen
   in the store step — `SiLU(C0)·C1` (moe.cpp), `(Cacc − comp)·scale_a·scale_b` — then convert to out
   dtype with `_mm512_cvtne2ps_pbh` (round-to-nearest-even). NEVER emit the pre-activation intermediate
   to DRAM for a second pass. (This is the micro-scale manifestation of the macro no-materialization rule.)
5. **Rounding / clamp**: int paths use `_mm512_roundscale_ps` TO_NEAREST | NO_EXC; rely on output-dtype
   saturation. See `amx-vectorization`, `cpu-gemm-amx-bf16`, `quantization-amx-int8`.

## Diagnostic
`uarch-perf-probe`: compare the inner loop to the resident microkernel peak (TF/s), check VNNI/AMX
tile occupancy and that the epilogue is fused (no `aten::to`/`copy_`/second parallel_for in the trace).

## ⛔ SETTLED NULL-TRAPS — do NOT re-spend runs testing these (the HW/compiler already handles them)
Four microbench hypotheses that LOOKED promising but were NULL in-engine (confirm-in-engine rule):
1. **Mid-forward `set_num_threads`** — no-op; does not resize the live OMP pool (microbench showed
   12–136×; in-engine zero).
2. **Accumulator-ILP / register banking** — 0–2%; the OoO engine already hides accumulate latency.
3. **Prefetch-distance tuning** — null; the HW prefetcher saturates at any hint distance.
4. **"Ideal matmul" roofline as a target** — a bare-bmm proxy is NOT achievable for a fused op; it
   omits conversion/materialization/layout/epilogue. Always confirm a real fused impl in-engine.
If tempted by one of these, STOP — it's a known dead end. See `shared/high-information-runs`.

## Exit criterion
Inner loop at (or explainably near) the microkernel peak for its precision, epilogue fused, no
intermediate materialization in the trace. The kernel is now at its roofline → return to
`multiscale-optimization` and re-profile (the next dominant op may be at a different scale).

## Pitfalls this scale exists to prevent
- Hand-AMX micro-tuning an op that is actually glue-bound (macro) or data-movement-bound (meso).
- A separate dequant/activation pass that re-materializes the pre-epilogue tensor (kills AI).
- Re-deriving the settled null traps and burning runs on them.
