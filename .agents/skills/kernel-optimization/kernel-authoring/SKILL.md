---
name: kernel-authoring
description: "Generative skill for the NEW-KERNEL leg (Thesis 2): WRITE a new CPU/AMX (or ARM) kernel by adapting an existing donor kernel, rather than inventing from scratch. Distilled from the SGLang CPU kernel corpus (x86 AMX + aarch64 NEON), LIBXSMM/TPP (the batch-reduce-GEMM origin), and oneDNN (SGLang's M>4 brgemm backend). Encodes the one building block (BRGEMM), the 4-layer kernel skeleton, the arch-invariant-vs-arch-specific split, op→donor→adapt recipes, composition patterns (online softmax, gather-GEMM, fused epilogue, MoE grouping), and the validation gates. Load this FIRST when a coverage-gate GAP needs a net-new kernel."
---

# Kernel Authoring (generative — write a new kernel from a donor)

The optimize/validate skills (`amx-vectorization`, `cache-blocking-tiling`,
`weight-prepacking-brgemm`, `quantization-amx-int8`, `roofline-validation`) tune a
kernel that already exists. This skill covers the missing half: **authoring a new
kernel**. The rule is: never invent from a blank file — **adapt the nearest donor
kernel** in the SGLang CPU corpus (see `assets/donor-kernel-map.md`) and change only
the layer that must change.

## The one building block
Every dense-math CPU kernel is a **batch-reduce GEMM (BRGEMM)**: a register-resident
C accumulator with A and B *streamed* past it. Source of truth: LIBXSMM/TPP
(Heinecke et al., "High-Performance Deep Learning via a Single Building Block",
SC'19; TPP arXiv:2104.05755) — GEMM + element-wise TPPs compose into full operators
**with no platform-specific code**. In SGLang: `M>4` routes to
`at::native::cpublas::brgemm()` (oneDNN, dispatches `avx512_core_amx`); `M≤4` and
gather paths use the in-tree `tinygemm_kernel_*` (`_mm512_dpbf16_ps` with ≥16 FP32
accumulators). GEMM, MoE grouped-GEMM, attention QK/AV, and the DSA indexer are all
BRGEMM variants.

## The 4-layer skeleton (author each layer explicitly)
1. **Arch-INVARIANT control** (reuse verbatim across x86/ARM): the M/N/K blocking
   loop nest; keep C register-resident, stream A/B; per-channel weight + per-token
   activation quant; **FP32/INT32 accumulation**; epilogue scale; post-op fuse;
   `parallel_2d` over M×N blocks. Confirmed identical between `gemm_int8.cpp` (x86)
   and `aarch64/gemm_int8.cpp` (ARM).
2. **Arch-SPECIFIC inner microkernel** (the ONLY part re-authored per ISA): x86 AMX
   `TILE 16×16×32`, `dpbf16_ps`/`dpbusd_epi32`; ARM `vdotq_s32` (4-lane) /
   `vmmlaq_s32` (2×2 i8mm). Keep it behind an `op::` interface (ARM pattern) so the
   control layer is ISA-agnostic.
3. **Packing** (arch-specific layout, invariant concept): bf16 → VNNI2 `[K/2,N,2]`;
   int8 → VNNI4 `[K/4,N,4]` **plus a 4-byte/col INT32 compensation** for the
   u8×s8 trick; constraints `OC%16==0`, `IC%32==0`. Pack once at load, amortized.
4. **Epilogue**: `Cf32 = (Cacc − compensation) * scale_a[m] * scale_b[n] (+bias)` →
   convert to out dtype (`cvtne2ps_pbh`). Fuse activation here (SiLU·mul for MoE).

## op → donor → adapt (full table in `assets/donor-kernel-map.md`)
| New op class | Donor to adapt | Change only |
|---|---|---|
| dense GEMM (new dtype/shape) | `gemm.cpp` / `gemm_int8.cpp` | packing + inner op + epilogue |
| reduced precision | `gemm_{int8,fp8,int4}.cpp` | conversion intrinsic + scale granularity + compensation |
| attention variant | `flash_attn.cpp`, `extend.cpp`, `decode.cpp` | mask/softmax/KV-layout; reuse brgemm for QK & AV |
| sparse/indexed GEMM | `decode.cpp` `index_gemm_kernel_{nn,nt}` | gather + (for AMX) stage-then-brgemm |
| MoE | `moe.cpp` | routing/gather; reuse grouped brgemm |
| low-AI (norm/rope/act) | `norm.cpp`/`rope.cpp`/`activation.cpp` | the element-wise body; keep two-pass FP32 |

## Composition patterns (bespoke code that wraps BRGEMM)
- **Online softmax** (`flash_attn.h`): running max `m_i`, rescale `v' *= exp(m_old−m_i)`,
  accumulate `P@V` with brgemm `add_C=true`; O(1) state per row → no seqlen² matrix.
- **Gather-GEMM** (`decode.cpp`): AMX has **no gather**; for the AMX path you must
  **stage gathered rows into a contiguous VNNI temp, then BRGEMM** (this is the
  authoring move for the missing M=16 indexer). AVX-512 tinygemm gathers per-column.
  - **When it pays off (measure, don't assume):** staging costs an extra `K·N` copy.
    Prefer stage-then-BRGEMM only when `M ≥ TILE_M` (16) AND the staged copy amortizes
    over the tile-op count; at tiny M or tiny `K·N` the tuned AVX-512 tinygemm can win.
    Benchmark both against the achievable ceiling before committing.
  - **Fuse the upcast into the gather:** if B is fp8/int8, apply the dtype conversion
    (with `B_scale`) DURING the staging copy — never a second pass over `K·N`.
  - **Zero-pad invariant:** when padding `K→TILE_K` / `N→TILE_N`, the padded A columns
    and staged B rows MUST be zeroed, or the padded lanes corrupt the dot product.
- **Fused activation epilogue** (`moe.cpp`): emit `SiLU(C0)*C1` in the store step to
  avoid materializing the intermediate — raises effective AI.
- **MoE grouping** (`moe.cpp` `moe_align_block_size`): thread-local expert counts →
  padded sorted token list → one grouped BRGEMM per expert block.

## Two non-negotiable co-design rules (both externally corroborated)
1. **Gate against the STREAMED-achievable ceiling, not the resident peak.** LIBXSMM
   docs state it directly: results "depend on whether operands are streamed or not;
   all-in-L1 can favor an implementation that is worse for the real workload." Use
   `establish-achievable-performance` + `roofline-validation`.
2. **Verify the intended ISA actually dispatched** (`ONEDNN_VERBOSE`, kernel-throughput
   magnitude) — never trust capability flags (`_is_amx_tile_supported` false-negatives
   on GNR).

## Concrete constants (GNR, from the corpus)
TILE_M/N/K = 16/16/32; BLOCK_M/N = 32; BLOCK_K = 128; `can_use_brgemm = M>4`;
≥16 FP32 accumulators hide `dpbf16_ps` latency; int8 uses `dpbusd_epi32` + per-col
compensation; grain size ≈ 1024 for low-AI ops; first-touch NUMA + one TP rank/SNC.

## Reduced-precision cheatsheet
Precision is a **data-movement lever, not just a compute lever**: low-precision
STORAGE cuts operand bytes (helps memory-bound ops AND feed-limited compute-bound
ops — store fp8/int4, upcast in the microkernel), while low-precision COMPUTE
(int8 AMX) also raises the compute ceiling ~2x. Always count the up-convert/dequant
cost (it can become the bottleneck) and gate accuracy per precision.
- **INT8 (W8A8)**: `dpbusd_epi32`; W per-channel, A dynamic per-token; u8×s8 trick →
  subtract `128*Σcol(B)` compensation in epilogue.
- **FP8 (e4m3)**: no native SIMD → `CVT_FP8_TO_BF16_EXT` upcast, per-block (K/128)
  scale folded via `fmadd` in the accumulation loop.
- **INT4**: 2 weights/byte, unpack nibbles → int8, per-group scale + optional
  asymmetric zero-point compensation.

## Procedure
0. **GATE:** `kernel-feasibility-gate` MUST have passed — a user-reviewed roofline +
   a measured baseline on the target node + go/no-go. Do NOT author otherwise.
1. Read the coverage-gate GAP + the novel op's math (from the model/reference kernel).
   A NEW-FUSED-KERNEL task may instead arrive from `fusion-analysis` (an identified
   fusion with no donor) — treat the fused op as the target, same 4-layer approach.
1b. **Establish a slow-but-correct REFERENCE implementation FIRST — the persistent
   numerical oracle.** Never optimize without a correct fallback to diff against. If a
   donor/external kernel exists, its output is the oracle. If the op is genuinely novel
   (no donor — e.g. a new DSA kernel), AUTHOR a naive torch/FP32 reference before the
   fast path and KEEP it: every optimized variant is checked against it on random
   inputs at every step, and it ships alongside the kernel as the correctness gate
   (and, in make-it-work, it is also what lets the model run end-to-end before the AMX
   kernel exists). A fast kernel that drifts from the reference is a regression.
2. Pick the donor (table above / asset). Read it fully; identify the 4 layers.
3. Re-author ONLY the layer(s) that differ; reuse control/packing/epilogue.
4. Establish the node's achievable ceiling for this op class first.
5. Implement; verify numerical parity vs an FP32 reference on random inputs.
6. Verify ISA dispatch; run `roofline-validation`; compare peer-relative to the donor
   at a matching shape. Loop on the gap.
7. Register the new kernel's capability contract so `coverage-gate` marks the op covered.

## Gate
A new kernel ships only when: numerically matches the FP32 reference within tolerance,
the intended ISA verifiably dispatched, and efficiency ≥ target % of the *achievable*
(streamed) ceiling AND ≥ 0.90 of a peer kernel of the same class/shape. Record the
winning constants (block sizes, accumulator count) as a learned pattern.

## Worked target (Thesis-2 flagship) — with MEASURED feasibility finding
DSA lightning-indexer nn GEMM at M=16 (`decode.cpp:13` TODO). The `kernel-feasibility-gate`
microbench (GNR 6980P **and** EMR 8592+, single core) found the op is **issue/conversion-
bound, NOT memory-bound** (IPC ~3.0–3.8, LLC-miss <2%, ~11–14% of FP32 peak; the
fp8→bf16→fp32 up-convert dominates). This OVERRIDES the naive plan:
- Single-pass over B gives **no** benefit (64 KB gathered B stays L2/L3-resident, so the
  4× re-stream is free; and M=16 with 128 fp32 accumulators SPILLS registers → slightly
  *slower*). Do NOT naively stage-then-brgemm here.
- The real lever is **cutting the up-convert cost**: compute with `dpbf16_ps` (bf16 in,
  fp32 accum) instead of up-converting to fp32 + `fmadd_ps` — removes the fp32 up-convert
  and doubles MAC/instruction. Cheap, no staging, no spill.
- Full AMX tiling is justified ONLY if the Amdahl share warrants it AND the staging +
  tile-config overhead at this tiny shape is measured to pay off.
Lesson: the roofline *bracketed* it as maybe-memory-bound; the microbench *proved* it
issue-bound and killed the wrong design before a line was written — exactly why the
feasibility gate is mandatory. See `assets/donor-kernel-map.md` for the donor layers.

## References
- LIBXSMM / TPP (Heinecke, Pabst, Henry et al.) — BRGEMM as the single building block;
  JIT per-shape specialization; build-once-deploy-everywhere across AMX/AVX-512/NEON/SVE/RVV.
- oneDNN `brgemm` ukernel — SGLang's `M>4` backend; tile-config + post-ops fusion.
- SGLang CPU corpus — the grounded donor implementations (`assets/donor-kernel-map.md`).
