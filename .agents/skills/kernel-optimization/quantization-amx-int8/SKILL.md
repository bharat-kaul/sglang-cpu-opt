---
name: quantization-amx-int8
description: "Use when the accuracy budget allows lower precision to raise the compute ceiling on Intel Xeon. Covers INT8 AMX (2x the BF16 tile throughput), weight-only vs full INT8, per-channel scales, and the correctness gate. Apply last, after the BF16 path is tuned, when you need to move the compute ceiling itself."
---

# Quantization — INT8 AMX

BF16 tuning maximizes efficiency at a fixed ceiling; quantization **raises the
ceiling**. AMX INT8 (`amx_int8`) does ~2x the ops/cycle of AMX BF16
(`amx_int8_ops_per_cycle_per_core` ≈ 2048 vs 1024).

## Trigger
Compute-bound op already tuned in BF16, and the model's accuracy budget tolerates
INT8 (validate!). Requires `amx_int8`.

## Options (increasing risk/reward)
1. **Weight-only INT8** (activations BF16): halves weight bandwidth, modest compute
   gain; safest accuracy. Mirrors SGLang `GPTQLinearIntelAMXMethod` /
   `IPEXAWQLinearMethod`.
2. **Full INT8 (W8A8):** both operands INT8 → uses INT8 AMX tiles at ~2x BF16
   compute peak; needs activation quantization + scales.
3. Per-channel (weights) / per-token (activations) scales to preserve accuracy;
   accumulate in INT32, dequantize to FP32.

## Tile shape
INT8 AMX tile K is 64 (vs 32 for BF16): A `16x64`, B `64x16` VNNI4, C `16x16`
INT32. Pack constraints shift accordingly (`IC % 64 == 0`).

## Correctness gate (stricter than BF16)
- Compare end-to-end task accuracy (gsm8k/mmlu), not just per-op error.
- Gate: task-accuracy drop within the model's stated budget (e.g. ≤1%). A
  per-op MSE check is necessary but NOT sufficient — quantization error is
  distributional. Block on accuracy regression.

## Performance gate
Re-establish the achievable ceiling for INT8 (`amx_peak.c` with INT8 tiles) and
gate the INT8 GEMM at ≥70% of the INT8 streamed-achievable, same as BF16.

## Order
Apply LAST in the playbook — after parallelization/tiling/BRGEMM — because it
changes the correctness contract and should be evaluated against a fully-tuned
BF16 baseline.

## Storage dtype ≠ compute dtype (sub-8-bit checkpoints, e.g. fp4/nvfp4/mxfp4)
GNR AMX computes in **bf16 and int8 ONLY — there is no native 4-bit (fp4/int4) matmul**.
A checkpoint's weight dtype (fp4 experts in DeepSeek-V4-Pro; nvfp4/mxfp4/int4 GGUF) is a
**storage/quantization format**, not a compute path on this ISA. So a sub-8-bit checkpoint
is an ENABLEMENT GAP, not a ready kernel: the GEMM MUST upconvert the weights to a
supported compute dtype. The framework's CPU MoE/GEMM kernel will reject the packed
sub-8-bit layout (DSV4: `fused_experts_cpu` asserts `packed_w1.size(2)==packed_K`; fp4's
2-per-byte packing gives K/2 → `3588 vs 7168`). Route it to a supported path:
- **Accuracy-parity FIRST target = bf16 compute.** Dequant `fp4_level × block_scale → bf16`
  adds NO new quantization (bf16 represents the few fp4 levels exactly) → true parity vs the
  fp4 reference. w8a8-int8 adds ACTIVATION quant error → parity risk; do it later.
- **Memory: upconvert is 4× (fp4→bf16), 2× (fp4→int8).** Check it FITS. DSV4-Pro experts
  ≈735 GB fp4 → ≈2.9 TB bf16 > a 1.5 TB node. Two bf16-compute paths:
  1. **Dequant-at-load → bf16 in RAM:** uses the EXISTING bf16 `fused_experts_cpu` (unquant
     path), zero new kernel — but only where 4× fits (tiny-faithful, smaller/flash models).
  2. **fp4 in RAM, dequant-per-tile IN-KERNEL → bf16 AMX:** memory stays 1× (fp4), same
     numerics as (1) — a NEW kernel to author (sub-8-bit-weight, bf16-compute grouped GEMM).
     This is the real large-model path.
- **Sequence:** bf16 parity (reference) → int8 → int4-storage/int8-compute, each diffed vs the
  bf16 result. Same as the playbook's correctness-first rule; int4-storage on GNR is ALSO
  dequant-to-int8/bf16 (no int4 compute).
