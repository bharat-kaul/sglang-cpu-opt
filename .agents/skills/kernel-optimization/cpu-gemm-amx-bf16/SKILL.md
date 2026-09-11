---
name: cpu-gemm-amx-bf16
description: "Use when optimizing a dense matrix multiply (GEMM / linear / nn.Linear / torch.matmul) for Intel Xeon CPUs with AMX in BF16 precision (Granite Rapids and successors). USE FOR: making a MxKxN BF16 GEMM hit the AMX compute roofline; choosing tiling, thread count, NUMA/affinity, and weight layout; validating correctness and % of peak. DO NOT USE FOR: GPU GEMMs, FP32-only paths, or memory-bound elementwise ops. Retarget to a new Xeon by editing assets/hardware-profiles/<part>.yaml only."
---

# CPU GEMM Optimization — AMX / BF16

Playbook an HPC performance engineer follows to drive a dense BF16 GEMM to the
AMX compute roofline on Intel Xeon. The **methodology below is microarchitecture
invariant**; every hardware-specific number lives in a *hardware profile*
(`assets/hardware-profiles/<part>.yaml`). To retarget Granite Rapids → Diamond
Rapids, an expert edits/adds a profile file — no change to this playbook.

> This skill is the **worked composition** of the per-technique skill library for
> the dense-GEMM case. It applies, in order: `establish-achievable-performance`,
> `amx-vectorization`, `cache-blocking-tiling`, `weight-prepacking-brgemm`,
> `openmp-parallelization`, and `roofline-validation`. Use `cpu-optimization-playbook`
> to pick this composition for a new dense linear layer.

## Capability contract (check first, fail gracefully)

Every technique names a required capability. Before applying, confirm the target
profile's `capabilities` block advertises it AND the runtime reports it. Probe
capability with `lscpu` flags (`amx_bf16`, `amx_tile`) and confirm the kernel
library will use it via `ONEDNN_VERBOSE=1` (expect an `amx` primitive, e.g.
`brg_matmul:avx10_1_512_amx`). Do NOT rely on
`torch._C._cpu._is_amx_tile_supported()` — it returns `False` on Granite Rapids
even while oneDNN dispatches AMX BF16 kernels. Otherwise fall back to the
next-best path and record the gap. Do NOT emit code the ISA/toolchain cannot run.

| Technique | Requires |
|-----------|----------|
| AMX BF16 tile GEMM | `amx_tile`, `amx_bf16` |
| BF16 weight pre-pack (VNNI tile layout) | `amx_bf16` + kernel/library packing support |
| AVX-512 BF16 fallback | `avx512_bf16` |
| FP32 reference | always |

## Optimization sequence (ordered — calibrate → memory layout → tile → parallel → clock)

0. **Calibrate the ACHIEVABLE ceiling first (do this before optimizing anything).**
   Peak (paper) FLOP/s is not the target — the *achievable* ceiling a saturating
   microkernel sustains on the actual silicon is. Run the bundled microbenchmarks
   once per hardware profile and store the results in the profile's `achievable`
   block:
   - `tools/microbench/amx_peak.c` — resident-tile TDPBF16PS loop (4 accumulators,
     zero memory traffic) → per-core and per-socket **achievable compute TFLOP/s**.
     This also empirically fixes the FLOP/cycle constant instead of assuming it.
   - `tools/microbench/stream_triad.c` — STREAM triad → **achievable DRAM GB/s**.
   All later gates compare against these measured numbers, not the paper peak.
   On Granite Rapids 6980P the microkernel yields ~350 TF/socket (~=100% of AMX
   peak) and STREAM ~631 GB/s/socket; treat 350 TF/socket as the GEMM target.
1. **Classify & roofline-gate.** Compute arithmetic intensity `AI = 2*M*N*K /
   bytes_moved`. Compare to the profile ridge point (`peak_bf16_flops /
   peak_mem_bw`). If `AI > ridge` the kernel is **compute-bound** → target the
   AMX FLOP ceiling. If below, stop and switch to the memory-optimization skill.
   (A large square BF16 GEMM such as 8Kx8Kx8K is firmly compute-bound.)
2. **Precision & accumulation.** Inputs BF16, accumulate in FP32 (AMX TDPBF16PS
   accumulates FP32 natively). Never accumulate in BF16.
3. **Tile to the AMX unit.** Register tiles are fixed by the ISA:
   `TILE_M x TILE_K` (A) and `TILE_K x TILE_N` (B) → `TILE_M x TILE_N` FP32 acc.
   For BF16: `TILE_M=16, TILE_N=16, TILE_K=32`. Require `N % TILE_N == 0`,
   `K % TILE_K == 0`; pad otherwise. Use multiple tile accumulators (>=4) to hide
   the TDPBF16PS latency behind its throughput.
4. **Cache block.** Block K and N so the reused B panel fits L2
   (`l2_bytes_per_core`) and the A/C working set streams. Keep the innermost
   accumulation resident in tile registers; block M across cores.
5. **Weight layout / pre-pack.** Pre-pack the stationary operand (weights) into
   VNNI/AMX tile order once at load time so the inner loop does contiguous tile
   loads (mirrors SGLang `amx_process_weight_after_loading` /
   `convert_weight_packed`, `dim_is_supported`: `OC%16==0`, `IC%32==0`).
6. **Parallelize.** One GEMM: partition the M (and/or N) dimension across cores
   with OpenMP; `OMP_NUM_THREADS = cores_per_domain`. Pin threads
   (`KMP_AFFINITY=granularity=fine,compact` or `OMP_PROC_BIND=close`,
   `OMP_PLACES=cores`). One thread per **physical** core (ignore the SMT sibling
   for AMX — the TMUL unit is shared).
7. **NUMA.** Bind compute and memory to the same domain(s)
   (`numactl --cpunodebind=<d> --membind=<d>`). Prefer a single socket for a lone
   GEMM to avoid cross-socket UPI traffic; only span sockets when the problem is
   large enough to amortize it. Honor the profile's SNC layout.
8. **Clock reality.** AMX runs at a lower all-core frequency than the SSE/AVX2
   turbo. Roofline peak must use the **measured AMX all-core frequency**, not the
   advertised max turbo. Measure it; do not assume.

## Correctness gate (mandatory, before any perf claim)

- Compute an FP32 reference `C_ref = A_fp32 @ B_fp32`.
- Require `mean_rel_err < 2e-2` and no NaN/Inf (BF16 has ~8 mantissa bits; ~1e-2
  relative error is expected and acceptable). A regression here blocks the result.

## Performance gate (roofline)

- Gate against the **achievable** ceiling from step 0, not the paper peak.
  `efficiency = achieved_tflops / achievable_ceiling_tflops`.
- Target: **>= 70%** of the *achievable* AMX ceiling for a large square GEMM on a
  single socket. Below target -> return to step 4/6/7 (blocking, threads, NUMA)
  with the gap as feedback. `efficiency > 100%` means the measured ceiling or
  frequency is wrong — re-run the microkernel, do not report.
- Known baseline (GNR 6980P, 8Kx8Kx8K): stock `torch.matmul` ~60 TF and MKL
  `cblas_gemm_bf16bf16f32` ~42 TF both land at only ~12-17% of the 350 TF/socket
  achievable ceiling — i.e. off-the-shelf libraries FAIL this gate for a lone
  large GEMM. Closing it needs weight pre-packing + explicit cache blocking
  (steps 4-5) or a custom brgemm microkernel scaled from `amx_peak.c`.

## Retargeting to a new Xeon (Day-0 intent)

The hardware profile is the single parameterized source of truth; the playbook
above never changes. Two ways to produce a profile for a new part:

**Mode A — AUTO (hardware in hand, preferred).** Run the calibrator on the node;
it fills every `[AUTO]` field (topology, ISA caps, caches, turbo) and runs the
microbenchmarks to populate the measured `achievable` ceilings:
```
srun --partition=<newpart> --cpus-per-task=<n> python tools/calibrate.py \
     --out .agents/skills/kernel-optimization/cpu-gemm-amx-bf16/assets/hardware-profiles/<part>.yaml
```
You then only supply the `[SPEC]` datasheet fields (memory MT/s, channels, tile
shape). The roofline and thread/NUMA plan recompute automatically; gate with
`roofline.py --profile <part>.yaml --use-profile-achievable`.

**Mode B — MANUAL (pre-silicon / next-gen bring-up).** Copy `_TEMPLATE.yaml`,
fill `[SPEC]` from the datasheet, and set the `achievable` block to your best
projection with `estimated: true` so the verdict labels the ceiling as
unvalidated. Re-run Mode A the moment hardware is available to replace the
projection with measured numbers.

If the new part adds an ISA (e.g. AMX-FP16/AMX-complex) that no kernel emits yet,
the capability contract makes the workflow fall back and flag a scaffolding task
rather than emit unrunnable code.

## Bundled tools

- `../../../../tools/calibrate.py` — Day-0 profile generator (probes node + runs microbench).
- `../../../../tools/microbench/amx_peak.c` — AMX BF16 compute-peak microkernel (achievable ceiling).
- `../../../../tools/microbench/stream_triad.c` — memory-bandwidth microbenchmark (achievable BW).
- `../../../../tools/gemm_amx_bf16.py` — BF16 AMX GEMM benchmark + FP32 correctness check.
- `../../../../tools/mkl_gemm_bf16.c` — tuned MKL AMX BF16 GEMM (headroom-chase baseline).
- `../../../../tools/roofline.py` — reads a profile, gates achieved vs the achievable ceiling.
- `../../../../tools/slurm_gemm.sh` — launches the sweep on a Slurm GNR node with the
  affinity/NUMA knobs above and ONEDNN_VERBOSE AMX-dispatch verification.
