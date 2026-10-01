---
name: compute-from-native-precision
description: "Use WHENEVER a model ships weights in ANY reduced or packed precision — fp4/MXFP4/NVFP4/nf4/int4/int8/fp8, or AWQ/GPTQ/bitsandbytes-style packed blocks. The rule: compute DIRECTLY from the packed/native representation, fusing dequant into the GEMM epilogue — NEVER materialize a widened (bf16/fp32) copy of the weights, neither as a load-time re-encode to reach a familiar kernel nor as a per-forward temporary. Precision-agnostic front for the HW-specific quantization-amx-int8 and weight-prepacking-brgemm skills. Fires at enablement wiring AND kernel authoring."
---

# Compute from native precision — never dequantize to a wide temp

**THE RULE (precision-agnostic):** if the checkpoint ships in a reduced/packed
precision, keep it packed and feed a kernel that **dequantizes inside the matmul**
(fused epilogue, per-tile). Never produce a separate widened copy of the weights —
not at load time to satisfy an existing bf16/fp32 kernel, and not per-forward as a
scratch temp. The matmul reads the *same values* either way; widening buys nothing
and costs on three axes.

This is the generic form of "multiply straight from the packed MXFP4 nibbles."
MXFP4 is just today's instance; the rule holds for int4/int8/fp8/nf4/AWQ/GPTQ.

## Why (three independent costs of widening)
1. **Memory traffic.** Widening N-bit → bf16/fp32 multiplies the bytes streamed per
   token. CPU **decode is weight-streaming-bound**, so more weight bytes = directly
   slower. (Operand precision is the #1 decode lever — see `model-roofline-analysis`.)
   Evidence: dequantizing a 2.78T MXFP4 model's experts would write ~194 GB/token of
   pure format conversion *before a single multiply-accumulate* (Kimi-K3 census).
2. **Allocation / materialization overhead.** A per-forward widened temp is
   alloc + first-touch bound; the "cost" shows up smeared across downstream ops and is
   easily mistaken for real compute (see `overhead-attribution`). Our own CPU DSV4
   bring-up dequantized FP4→FP8 per layer — a textbook instance of this anti-pattern.
3. **Capacity.** Widening doubles/quadruples the resident footprint, which can tip a
   model that fit ONE NUMA/SNC domain into the tp>1 / interleave / mbind swamp — a
   self-inflicted overflow (see `sub-numa-clustering`, `quantization-amx-int8` capacity).

## Decision procedure
1. **Identify** the checkpoint's native quant enum + group size + scale layout
   (e.g. MXFP4 e2m1 nibble × e8m0 per-32 scale; int4 W4A16; fp8 per-channel).
2. **Grep the kernel library for a fused low-bit compute path** before writing anything
   (`W4A16`/`W8A16`/`CPUQuantMethod::<FMT>`, a `fused_experts` / `*_linear` with a packed
   path). It usually already exists for GPU/NPU and just needs wiring on the target (CPU).
3. **Pick the compute mode by HW, keep weights packed either way:**
   - HW HAS a native low-precision matmul (e.g. INT8 AMX) → use it; this also raises the
     COMPUTE ceiling (`quantization-amx-int8`).
   - HW has NO native low-precision matmul (e.g. 4-bit on most CPUs) → keep weights packed
     and **dequant-in-kernel to the compute dtype inside the GEMM** (fused per-tile). The
     cheap dequant hides behind the memory stall; a separate pass does not.
4. **Prepack ONCE at load**, not per forward; pack the scales alongside the weights.
5. **NEVER up-convert the whole checkpoint** to reach a familiar kernel. If the instinct is
   "dequant it to bf16/fp8 so kernel X accepts it," STOP and wire the native path instead.

## Validation (keyed on LOSSLESS vs LOSSY, not on compute precision)
Up-converting for compute does NOT recover bits dropped at downconvert — accuracy is set by
the *operand* values entering the matmul.
- **Lossless** (native format kept, or an exact re-encoding: fp4 e2m1 × pow2 e8m0 → bf16 is
  bit-exact) → zero accuracy loss; just CONFIRM it, no eval needed.
- **Lossy** (you quantized a higher-precision checkpoint DOWN to fit) → MUST run
  `accuracy-oracle`; the loss is at the downconvert and no compute precision undoes it.
- Two axes still cost accuracy even with lossless weights and need a check: **activation
  precision** (WxA16 vs WxA8) and any **scale/group re-encoding** done to match a kernel.

## Anti-patterns (each one is this skill being violated)
- A separate dequant pass that materializes bf16/fp32 weights before the matmul.
- A per-forward widened scratch temp of the packed weights/activations.
- Up-converting the shipped checkpoint at load to reuse an existing wide-dtype kernel.

## Cross-refs
- `quantization-amx-int8` — HW specifics (INT8 AMX tiles, W4A16 fused dequant, capacity lever).
- `weight-prepacking-brgemm` — load-time pack layout (VNNI, scale pack) for the fused path.
- `model-roofline-analysis` — operand precision as the top decode (memory-bound) lever.
- `overhead-attribution` — how a widening temp masquerades as per-op compute cost.
