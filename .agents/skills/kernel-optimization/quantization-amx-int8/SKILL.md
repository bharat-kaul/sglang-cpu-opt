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
