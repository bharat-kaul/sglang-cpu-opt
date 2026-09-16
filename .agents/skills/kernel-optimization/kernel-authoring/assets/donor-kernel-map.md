# Donor Kernel Map — SGLang CPU corpus (grounded reference for kernel-authoring)

Extracted from a full read of `sglang/python/sglang/kernels/aot/csrc/cpu/` (x86 AMX +
aarch64 NEON). Use to pick the donor kernel and know its 4 layers before adapting.
Paths are relative to that dir. Line numbers are approximate anchors.

## Shared primitives & constants
- `gemm.h`: `TILE_M=TILE_N=16`, `TILE_K=32`; `block_size_m()=block_size_n()=32`;
  `BLOCK_K=128`; `can_use_brgemm<bf16>(M) = M>4`; `get_row_size<int8>(K)=K+4` (compensation).
- `vec.h`: dtype conversions (`CVT_BF16_TO_FP32`, `CVT_FP8_TO_BF16_EXT` w/ `kFP8_BIAS=0x3b800000`,
  MXFP4 LUT), `load_float_vec2`, `vec_reduce_sum/max`, `quantize_row_int8`, transposes
  (`transpose_16x16_32bit`, `transpose_2x32_16bit`).
- `vec_pack.h`: VNNI packing (`pack_vnni_Nx32`, `pack_vnni_Kx32`) — dense `[N,K]`→`[K/2,N,2]`.
- `common.h`: `parallel_2d`, `data_index_init/step` (flatten 2D iter), `GRAIN_SIZE=1024`.
- Capability gate: `vec.h:3` `#if __AVX512F__ && __AVX512BF16__ && __AMX_BF16__ → CPU_CAPABILITY_AVX512`.
  No scalar fallback in tinygemm (hard-fails); FP32 FMA path for tiny N.

## Dense GEMM — donor: `gemm.cpp` (+ `gemm.h`)
- Microkernel `tinygemm_kernel_nn` (~gemm.cpp:245-270): FP32 accumulators
  `__m512 vc[ROWS*COLS]` (≥16 independent → hide `dpbf16_ps` ~4-cyc latency); broadcast
  A scalar, load B from VNNI, `_mm512_dpbf16_ps`, store via `cvtne2ps_pbh`.
- Loop nest (~275-312): `for mb: for nb: for k in K/2: unrolled (row,col) dpbf16`.
- Prepack (~24-51, vec_pack.h:48-70): `transpose_16x16_32bit` → VNNI; `OC%16`, `IC%32`.
- Parallel (~596-602): `parallel_2d(MB,NB,...)`.
- Working set/block: C 4KiB (registers), A 8KiB, B 8KiB → ~20KiB (L1/L2 resident).

## Reduced precision — donors: `gemm_int8.cpp`, `gemm_fp8.cpp`, `gemm_int4.cpp`
- INT8: `_mm512_dpbusd_epi32` (u8×s8→i32); A dynamic per-token (`quantize_row_int8`,
  round + +128 offset); W static per-channel; **compensation** `Bcomp=128*Σcol(B)` stored
  at `B + block_size_n()*K` (gemm.cpp:8-30), subtracted in epilogue
  `(vc - Bcomp)*As[m]*Bs[n]` (gemm_int8.cpp:176-194). VNNI4 int8 layout.
- FP8 (e4m3): upcast `CVT_FP8_TO_BF16_EXT` (ternary-logic, bias 1/256) → `dpbf16_ps`;
  per-block (K/128) scale via `fmadd` in accum loop (gemm_fp8.cpp:283-324). A stays bf16.
- INT4: 2 weights/byte, `load_uint4_as_int8` unpack, per-group scale + optional
  asymmetric zero-point; `dpbusds`/`dpbusd`; group compensation `Σ(W-ZP)` (gemm_int4.cpp:570-596).
- ~All share the block loop; only conversion + scale granularity + compensation differ.

## Attention — donors: `flash_attn.cpp/.h`, `extend.cpp`, `decode.cpp`
- KV layout: VNNI-packed K `[hd/32, seq, 32]`, V `[seq/16, hd_v, 16]` (flash_attn.cpp:30-70).
- Online softmax (flash_attn.h:80-260): running max `m_i`, `m_delta=exp(m_old−m_i)`,
  `s_delta=exp(s−m_i)`, `s' = s'*m_delta + Σs_delta`, `v' = v'*m_delta + P@V`;
  `_mm512_reduce_max_ps`/`fexp_u20`.
- QK & AV both via `brgemm` (flash_attn.cpp:113-130): QK `add_C=false`; AV `add_C=true`;
  pad n to `div_up(n,TILE_K)*TILE_K`.
- Decode (M=1/small) → tinygemm; extend/prefill (M large) → brgemm; extend has 2 stages
  (cached prefix + new tokens with causal mask, extend.cpp:201-310); GQA `head_kv=head/groups`.

## Sparse/indexed GEMM (DSA) — donor: `decode.cpp` (THESIS-2 TARGET)
- `index_gemm_kernel_nt` (Q@Kᵀ, ~986): A `[M,K]`, B VNNI KV gathered by `indices[n]`,
  C fp32 scores; patterns: M==1 → 1-8-8 (BLOCK_N=8); M>1 → BLOCK_M=4, BLOCK_N=6.
- `index_gemm_kernel_nn` (P@V, ~831): A fp32 weights, B rows gathered by `indices[k]`,
  C **pre-scaled per row** then accumulated; M==1 → BLOCK_N=8*16=128; M>1 → 4×96.
- **Missing (decode.cpp:13 TODO): AMX kernel for `index_gemm_kernel_nn` at M=16.**
  Authoring move: AMX has no gather → **stage** `B[indices[k],:]` into contiguous VNNI
  temp `[K,N]`, pre-scale C by row, pad K→TILE_K, then `brgemm(M=16,N,K, add_C=true)`.
  Donor inner = the AVX-512 tinygemm here; donor BRGEMM = `gemm.cpp`.

## MoE — donor: `moe.cpp` (+ `moe.h`)
- `moe_align_block_size` (~30-100): thread-local expert counts → cumsum → pad each expert
  to BLOCK_M → `sorted_ids` (tokens grouped by expert) + `expert_ids`/`offsets`.
- `fused_experts_kernel_impl` (~200-410): parallel_2d over (MB,NB); per block pick
  `expert_id`; W1 split upper/lower for **fused `SiLU(C0)*C1`** in store; W2 grouped GEMM;
  scatter by topk weights. W1 `[E,2N,K]`, W2 `[E,K,N]`, VNNI-packed.
- Shared expert: all tokens, no top-k (`shared_expert_*_kernel_impl`).

## Low-AI / fused ops — donors: `norm.cpp`, `rope.cpp`, `activation.cpp`, `qkv_proj.cpp`, `kvcache.cpp`
- RMSNorm (norm.cpp:150-248): two-pass, FP32 `dpbf16_ps` sum of squares, then scale·weight
  fused; per-row `parallel_for`.
- Activation+mul (activation.cpp:6-60): `out=act(x[:,d])*x[:,d+dim]` fused (2 ops/2 loads/1 store).
- RoPE (rope.cpp:200-270): interleaved permute, FP32 cos/sin, per-(seq,head), grain=1024/64.
- KV write (kvcache.cpp:16-60): per-batch page memcpy; fp8/int8 stored as bytes, dequant on read.
- qkv_proj.cpp: staged (Q/KV proj → RMSNorm → q_b → bmm → RoPE) — fusion only at streaming
  boundaries; GEMM stays separate (too complex to epilogue-fuse a position-indexed op).

## ARM (aarch64) — donors: `aarch64/gemm_int8.cpp`, `aarch64/moe.cpp`, `aarch64/op.h`
- Inner op: `vdotq_s32` (4-lane sdot) or `vmmlaq_s32` (2×2 i8mm); `op.h` templates
  `sdot_matmul<R=4,C=8>`, `i8mm_matmul<R=4,C=8>`; 128-bit int32x4 accumulators.
- **No packing** — reads B column-major in place; **no compensation** (native s8×s8).
- Same control layer: L2-aware N-slicing (`slice=64 if M*K>L2 else 8`), per-channel int8
  quant (identical `quantize_row_int8`), FP32 accumulate-from-int32 + per-elem scale.
- **Portability lesson**: arch-INVARIANT = blocking loops, packing *concept*, quant scheme,
  FP32 accumulation, epilogue scale, embarrassingly-parallel blocks. Arch-SPECIFIC = the
  inner matrix/vector instruction, the exact pack layout (VNNI vs in-place), compensation
  (needed for x86 u8×s8, not ARM s8×s8), vector width (512 vs 128), abstraction style
  (ARM `op.h` namespace vs x86 `#if CPU_CAPABILITY_AVX512` inline).

## Authoring checklist (per new kernel)
- [ ] Identify donor + its 4 layers (control / inner-op / packing / epilogue).
- [ ] Reuse control+packing+epilogue; re-author only inner op (and packing if new dtype).
- [ ] FP32/INT32 accumulation; correct scale granularity + compensation.
- [ ] Divisibility (OC%16, IC%32) or a padding path.
- [ ] `parallel_2d` (GEMM) or `parallel_for`+grain (low-AI); first-touch NUMA.
- [ ] Verify vs FP32 ref; verify ISA dispatch; roofline vs achievable; peer-relative.
