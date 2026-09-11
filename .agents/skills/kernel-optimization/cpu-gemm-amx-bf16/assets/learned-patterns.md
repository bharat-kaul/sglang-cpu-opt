# Learned patterns — CPU GEMM AMX/BF16
# Appended by the cpu-optimizer workflow after each validated run. Newest first.

## 2026-09-09 — LIBXSMM/BRGEMM investigation conclusion
- torch.matmul's 60.6 TF is ALREADY a BRGEMM (oneDNN `brg_matmul:avx10_1_512_amx`).
  BRGEMM is the baseline here, not an untried optimization.
- amx_peak.c IS a BRGEMM microkernel; 350 TF cache-resident = ~100% AMX peak.
- For a lone streamed 8Kx8Kx8K GEMM the ~60 TF wall is data movement, not kernel
  quality; a different BRGEMM lib (LIBXSMM) is expected in the same ~55-90 TF band,
  not near 350. LIBXSMM/TPP wins are in end-to-end DL (fusion, persisted blocked
  layouts, small/odd shapes, batched throughput), not one isolated square GEMM.
- Standalone LIBXSMM number NOT obtained: sample gemm_kernel driver runs a
  per-thread scalar reference (unusably slow); custom tiled shared-B BRGEMM needs
  careful VNNI/column-major/address-BR work — deferred to an sbatch batch job.
- DECISION: accept this grounded conclusion for now (option 2).

## 2026-09-09 — BRGEMM / two-ceiling insight (Heinecke arXiv:1906.06440)
- The resident microkernel's 350 TF/socket is the COMPUTE PEAK (all operands in
  L1/registers), NOT the achievable ceiling for a STREAMED GEMM. LIBXSMM docs:
  perf "depends on whether operands are streamed or not". Gating a streamed 8K
  GEMM against 350 TF manufactured much of the '5.8x gap'.
- Missing expert building block: batch-reduce GEMM (BRGEMM), C += sum_r A_r*B_r
  with the C accumulator resident in tile registers across the whole K reduction;
  the library is just tuned loops around this one kernel.
- PRE-PACK RESULT (mkl_gemm_bf16_packed.c, measured): pack weights once = 54.8 TF
  vs per-call MKL 42.4 TF = +29%. But torch 60.6 TF still higher, and ALL library
  paths sit at only ~16-17% of the 350 TF resident peak. => pre-packing is
  necessary but NOT the dominant factor; the resident peak is simply not the
  streamed ceiling.
- OPEN: build LIBXSMM/TPP to establish the true cache-blocked BRGEMM streamed
  ceiling; until then profile records streamed_gemm=60.6 TF as a provisional floor
  (estimated), gated separately from the 350 TF theoretical bound.
- ACTION: skills reorganized into a progressive library (establish-achievable,
  openmp, cache-blocking-tiling, amx-vectorization, weight-prepacking-brgemm,
  quantization-amx-int8) routed by cpu-optimization-playbook; gate against
  streamed_gemm, not compute_peak.

## 2026-09-09 — Day-0 parameterization proven (calibrate.py)
- tools/calibrate.py auto-generated a full profile on pcl-gnrap01 by probing
  lscpu/numactl + running the microbenchmarks: sockets/cores/NUMA/ISA/turbo filled,
  achievable ceilings measured (single-core 3.66 TF, socket 350.2 TF, node 695.7 TF,
  BW 631.9 GB/s). Only [SPEC] fields (mem MT/s, channels, tile shape) need datasheet.
- Microkernel implied ~939 FLOP/cyc/core @ max turbo -> confirms the 1024 ISA constant.
- Retarget path is now: (A) run calibrate.py on new silicon, or (B) hand-fill
  _TEMPLATE.yaml with estimated:true for pre-silicon. roofline.py --use-profile-achievable
  reads the ceiling from the profile, so the profile is the single parameterized source.
- FIX: L2 cache field must divide lscpu total by instance count to be per-core.

## 2026-09-09 — ACHIEVABLE ceiling established via microbenchmarks (Xeon 6980P)
- amx_peak.c (resident-tile TDPBF16PS, zero mem traffic): single-core 3.67 TF
  (=> ~1024 FLOP/cyc @ ~3.58 GHz single-core turbo, CONFIRMS the constant);
  single-socket 350.8 TF (128c @ ~2.68 GHz = ~100% of AMX peak); full-node 694.3 TF.
- stream_triad.c: 630.7 GB/s single socket (74.6% of 844.8 peak). Full-node
  interleaved read was anomalously low (214 GB/s) — first-touch/NUMA artifact;
  use single-socket BW as the reliable achievable figure.
- LESSON: gate optimization against ACHIEVABLE (350 TF/socket), not paper peak.
  The microkernel both sets the target AND empirically fixes FLOP/cyc + AMX clock,
  so it is now step 0 (Calibrate) of the workflow.

## 2026-09-09 — headroom chase for 8Kx8Kx8K BF16 vs 350 TF/socket achievable
- Stock torch.matmul (oneDNN brgemm avx10_1_512_amx): best 60.6 TF = 17.3% of achievable.
- MKL cblas_gemm_bf16bf16f32: best 42.4 TF = 12.1% of achievable, erratic scaling
  (84c 42, 96c 29, 128c 39) under GNU threading + numactl — WORSE than torch.
- CONCLUSION: both off-the-shelf library GEMMs FAIL the 70% gate for a single large
  GEMM on GNR; the ~5.8x gap is an implementation problem, not a roofline artifact
  (silicon proven to sustain 350 TF). Correctness on both paths passed.
- NEXT (open the loop): custom brgemm scaled from amx_peak.c with (a) B pre-packed
  to VNNI tile layout once, (b) K/N cache-blocked to keep B panel L2-resident, (c)
  M blocked across a single socket's cores, (d) >=4 tile accumulators. Compare vs 350 TF.

## 2026-09-09 — 8Kx8Kx8K BF16, Intel Xeon 6980P (Granite Rapids), stock torch.matmul
- AMX dispatch CONFIRMED via oneDNN: `brg_matmul:avx10_1_512_amx` (bf16 src/wei/dst).
- Measured AMX all-core frequency: 2.58 GHz (NOT the 3.9 GHz max turbo).
- Correctness: mean_rel_err 1.74% vs FP32 (within 2% BF16 gate), no NaN.
- Best achieved: 60.6 TFLOP/s single-socket (96 threads) = 17.9% of the 338 TFLOP/s
  AMX ceiling → roofline verdict FAIL. Full-node 256T was WORSE (7.5%): NUMA/over-
  threading hurts a lone 8K GEMM.
- Scaling single-socket: 42T→27.5, 84T→50.2, 96T→59.3 TFLOP/s (~94% scaling to 96T),
  so the bottleneck is per-core AMX utilization (~254 FLOP/cyc/core achieved), not
  just NUMA.
- GOTCHA: `torch._C._cpu._is_amx_tile_supported()` returns False on GNR even though
  AMX runs. Probe capability via lscpu + ONEDNN_VERBOSE instead.

### Optimization headroom to pursue next (open the loop again)
1. Avoid per-call output reorder: accumulate/emit FP32 output; keep weights pre-packed
   in AMX/VNNI tile layout (convert once, reuse) rather than re-packing each matmul.
2. Bind a lone GEMM to one socket (or one SNC domain scaled out), never interleave a
   small GEMM across both sockets.
3. Cache-block K/N so the B panel stays L2-resident (2 MiB/core); use >=4 tile
   accumulators to hide TDPBF16PS latency.
4. Evaluate a hand-tuned brgemm (libxsmm / direct oneDNN with fixed layouts) vs the
   generic torch.matmul path — stock ATen leaves ~4-5x on the table here.
