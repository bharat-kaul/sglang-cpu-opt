---
name: weight-prepacking-brgemm
description: "Use when a stationary operand (weights) is reused across GEMM calls (i.e. inference) on Intel Xeon. Encodes Alexander Heinecke's batch-reduce GEMM (BRGEMM) — the single building block of high-performance CPU deep learning — plus one-time weight pre-packing into VNNI/AMX-tile layout. This is typically the largest single win for a lone large GEMM and is what closes the gap to the streamed-achievable ceiling."
---

# Weight Pre-packing & Batch-Reduce GEMM (BRGEMM)

The missing expert knowledge for closing the streamed-GEMM gap. Source: Georganas,
Kalamkar, …, **Heinecke**, "High-Performance Deep Learning via a Single Building
Block" (arXiv:1906.06440) and Tensor Processing Primitives (arXiv:2104.05755),
realized in LIBXSMM/TPP.

## The batch-reduce GEMM building block
Express a large GEMM as a reduction over a **batch** of small block-GEMMs that all
accumulate into the same C block, kept resident in tile registers:

$$C_{ij} \mathrel{+}= \sum_{r} A_{ir}\,B_{rj}$$

- The C accumulator (`M_r x N_r` AMX tiles) **never leaves registers** across the
  whole K reduction — no store/reload of partial sums.
- Only A and B blocks stream in. The library becomes tuned loops (M, N, batch)
  around this ONE optimized kernel — no per-shape hand-coding.
- This unifies `cache-blocking-tiling` (loop structure), `amx-vectorization`
  (inner tile), and pre-packing (below) into a single reusable primitive.

## Weight pre-packing (the biggest lone-GEMM win)
Stock `torch.matmul`/BLAS re-pack the weight matrix into AMX/VNNI tile order on
**every** call. In inference the weights are constant, so pack **once at load
time** and reuse — mirrors SGLang `amx_process_weight_after_loading` /
`convert_weight_packed` (`dim_is_supported`: `OC%16==0`, `IC%32==0`).

Procedure:
1. At load: convert B (weights) to VNNI2/tile layout; store the packed buffer.
2. Per call: run BRGEMM streaming A against packed B; the inner loop does
   contiguous tile loads (no repack).
3. Verify the pack cost is amortized (excluded from the steady-state timing).

## Measured evidence (GNR 6980P, 8Kx8Kx8K BF16, single socket)
- Resident compute PEAK (`amx_peak.c`): ~350 TF/socket (unreachable by streamed GEMM).
- Stock `torch.matmul` (repacks each call): ~60.6 TF.
- MKL `cblas_gemm_bf16bf16f32`, per-call pack: ~42.4 TF.
- MKL with **weights pre-packed once** (`cblas_gemm_bf16bf16f32_pack` +
  `..._compute`, `tools/mkl_gemm_bf16_packed.c`): **54.8 TF** (best, 96 threads).
- So pre-packing recovered **+29% over per-call MKL** (42.4 -> 54.8) — real, but
  NOT the dominant factor: even the best library path (~55-61 TF) reaches only
  ~16-17% of the resident peak.
Conclusion: the apparent "5.8x gap" is mostly measuring against the RESIDENT peak,
which no streamed library GEMM approaches. Pre-packing is necessary but not
sufficient. Establishing the true streamed-achievable ceiling requires a
cache-blocked BRGEMM (LIBXSMM/TPP), not installed here — see Next steps.

## Next steps to actually close the gap
1. Build LIBXSMM (or TPP-PyTorch-extension) and run its JIT BRGEMM for 8Kx8Kx8K
   BF16; that number is the real `streamed_gemm` ceiling.
2. If it too lands far below the resident peak, the streamed ceiling IS the ~15-25%
   regime for a lone square GEMM and the profile should record that honestly
   (the resident peak stays a theoretical bound only).
3. Cross-check the AMX all-core frequency under a *streamed* GEMM (vs the resident
   microkernel's 2.68 GHz) — memory traffic may pull the clock down further.

## Investigated (2026-09-09): is the 60 TF already BRGEMM?
YES. oneDNN dispatched `brg_matmul:avx10_1_512_amx` — `brg` = batch-reduce GEMM.
So torch.matmul's 60.6 TF is ALREADY a BRGEMM result; BRGEMM is the baseline, not
an untried lever. The amx_peak.c microkernel (4-accumulator TDPBF16PS, C resident)
is itself a BRGEMM kernel and hits ~350 TF cache-resident (~100% of AMX peak).
Conclusion for a lone streamed 8Kx8Kx8K GEMM: the ~60 TF wall is data movement,
not kernel quality; a different BRGEMM library (LIBXSMM) is expected to land in the
same ~55-90 TF band, NOT near 350. LIBXSMM/TPP wins show up in end-to-end DL
(fused epilogues, persisted blocked layouts, small/odd shapes, batched throughput),
not in one isolated large square GEMM. A standalone LIBXSMM number was not obtained
(sample driver does a per-thread scalar reference = unusably slow; a custom tiled
shared-B BRGEMM needs careful VNNI/layout work — deferred to a batch job).

## Gate
Set the profile's `achievable.streamed_gemm_tflops_per_socket` to the pre-packed
BRGEMM result and re-run `roofline-validation` at ≥70% of THAT. If still short,
return to `cache-blocking-tiling` (panel not L2-resident) or `openmp-parallelization`.

## Reference implementation options
- LIBXSMM (JIT BRGEMM microkernel) or the TPP-PyTorch-extension (both Heinecke et al.).
- MKL packed API for a quick prepacking proof point (`tools/mkl_gemm_bf16_packed.c`).
