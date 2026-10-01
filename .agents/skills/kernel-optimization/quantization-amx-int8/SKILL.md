---
name: quantization-amx-int8
description: "Use when the accuracy budget allows lower precision to raise the compute ceiling on Intel Xeon, OR to SHRINK the resident footprint so a model fits one NUMA/SNC domain (a CAPACITY lever, applied FIRST — see the capacity section). Covers INT8 AMX (2x the BF16 tile throughput), weight-only vs full INT8, native low-bit STORAGE + in-kernel dequant (fp4/MXFP4/int4 W4A16), per-channel scales, the never-up-convert-a-low-bit-checkpoint rule, and the correctness gate. For the compute-ceiling use, apply last after the BF16 path is tuned; for the capacity/fit and memory-bound-decode uses, apply first."
---

# Quantization — INT8 AMX

BF16 tuning maximizes efficiency at a fixed ceiling; quantization **raises the
ceiling**. AMX INT8 (`amx_int8`) does ~2x the ops/cycle of AMX BF16
(`amx_int8_ops_per_cycle_per_core` ≈ 2048 vs 1024).

## Two INDEPENDENT benefits (they decouple — pick by regime)
Low precision helps in two separate ways; a target may have one without the other:
1. **Raises the COMPUTE ceiling** — only when the HW has the low-precision matmul
   (INT8 AMX). Relevant to COMPUTE-bound ops (prefill GEMM).
2. **Reduces STREAMED BYTES → raises the MEMORY-bound ceiling** — works **even when the
   HW has NO native low-precision compute**: store the weights low-precision and
   **dequant-to-bf16 in-kernel**. On CPU **decode is weight-streaming-bound**, so fewer
   weight bytes = directly faster, and this is usually the BIGGER win. This is why fp4 /
   int4 STORAGE is valid on GNR (no 4-bit matmul) purely as a bandwidth/memory optimization
   — the cheap in-kernel dequant is hidden behind the memory stall. (See
   `model-roofline-analysis`: "reduce operand precision" is the #1 decode lever.)

So choose by regime: compute-bound → need real low-precision AMX tiles (benefit 1);
memory-bound decode → low-precision STORAGE + dequant-to-bf16 suffices (benefit 2),
no low-precision compute required.

## Third benefit: CAPACITY — make a too-big model FIT one NUMA domain (apply FIRST)
Low-precision STORAGE also **shrinks the resident footprint**, which is often the difference
between a model that fits ONE SNC domain (→ clean `tp=1`) and one that overflows it (→ the
whole tp>1 / interleave / mbind swamp; see `sub-numa-clustering`). When a model is only ~1.1–2×
over one domain's RAM, precision is almost always the fix — and unlike benefits 1–2 (tuning
levers), this one is a **prerequisite to running at all**, so apply it FIRST, before any
placement/sharding decision.

**THE RULE: never up-convert a low-bit checkpoint to reach a familiar kernel.** If the
checkpoint ships fp4 / MXFP4 / int4 and the instinct is to dequant it to fp8/bf16 so an
existing kernel accepts it, STOP — that *doubles or quadruples* the footprint and can
manufacture the domain overflow yourself. (This is the precision-agnostic
`compute-from-native-precision` rule; this skill is its INT8/AMX HW instance.) Instead:
1. Grep the kernel library for the checkpoint's **native** quant enum (`MXFP4`, `INT4_W4A8`,
   `NVFP4`, …) and for a matching fused-dequant compute op (e.g. a CPU `fused_experts_cpu`
   with a `CPUQuantMethod::MXFP4` W4A16 path). The fast W4A16 path usually already exists in
   the serving stack for GPU/NPU and just needs wiring on CPU.
2. Keep the weights in their native low-bit form; feed the low-bit kernel (dequant is fused in
   the GEMM — no bf16 materialization, which is what makes a naive per-token Python dequant
   ~100× too slow). Mirror the framework's own low-bit method (load-time prepack + scale pack
   + the fused apply) rather than authoring a kernel.
3. **Accuracy is keyed on whether the SHRINK is LOSSLESS — not on the compute precision.**
   Accuracy is fixed by the *operand* precision (the weight values entering the matmul);
   upconvert-to-bf16-for-compute does NOT recover bits thrown away at downconvert. So:
   - **Lossless shrink = zero accuracy loss, no eval needed** (just confirm it): the resident
     low-bit is an *exact re-encoding* of the source — either the checkpoint is **native** in
     that format (keeping it discards nothing), or the format represents every value exactly
     (fp4 e2m1 × power-of-2 e8m0 → bf16 is bit-exact; fp8 captures fp4 levels exactly). Here a
     W4A16 path and the higher-precision path compute on IDENTICAL weight values.
   - **Lossy shrink = MUST validate** (`accuracy-oracle`): if you shrink by *quantizing a
     higher-precision checkpoint down* (bf16/fp8 → int4/fp4 to fit), the loss is incurred at
     the downconvert and no compute precision undoes it.
   - Even with lossless weights, two axes still cost accuracy and need a check: **activation
     precision** (W4A16 vs W4A8) and any **scale/group re-encoding** to match a kernel (a
     de-replication is exact; a real group-size change or fp4→int4 scale requant is lossy).

Real case (DeepSeek-V4-Flash, GNR): the routed experts ship **native MXFP4** (~137 GB). A
Plan-A dequant to fp8 inflated them to ~275 GB, which exceeds one 258 GB SNC domain and forced
days of tp>1 / interleave / page-migration dead-ends. Keeping them MXFP4 and routing to the
existing CPU MXFP4 W4A16 kernel fit one domain → `tp=1` → the entire placement problem
evaporated. Lesson: **treat a capacity overflow as a footprint/precision problem first, a
placement problem second** — and question any step that INFLATES a low-bit checkpoint.

## Trigger
Either regime, once the model's accuracy budget tolerates lower precision (validate!):
- **Compute-bound** op already tuned in BF16 → full INT8 (W8A8) to raise the compute
  ceiling. Requires `amx_int8`.
- **Memory-bound** op (esp. decode weight-streaming) → low-precision **STORAGE** to cut
  streamed bytes, dequant-to-bf16 in-kernel. Does NOT require low-precision compute
  support — valid on GNR for fp4/int4 weights purely as a bandwidth/memory win.

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
     This is the real large-model path. **Note: this is ALSO the PERFORMANT decode path**,
     not just a capacity fix — decode is weight-streaming-bound, so keeping weights fp4
     (1/4 the bytes of bf16) directly raises the memory-bound throughput ceiling; the
     in-kernel dequant is cheap relative to the bytes saved (benefit 2 above).
- **Sequence:** bf16 parity (reference) → int8 → int4-storage/int8-compute, each diffed vs the
  bf16 result. Same as the playbook's correctness-first rule; int4-storage on GNR is ALSO
  dequant-to-int8/bf16 (no int4 compute).
- **The dequant bridge is UNVERIFIED until a numeric parity gate passes.** Whether you dequant at
  load (path 1) or per-tile in-kernel (path 2), the fp4→bf16 decode (scale layout, nibble order,
  group/block size, VNNI prepack, SwiGLU gate/up split) fails SILENTLY when wrong — plausible
  garbage, not a crash. Do NOT treat "the bf16 kernel exists and it ran" as correct. Emit the
  per-kernel parity gate from `accuracy-oracle` §low-bit parity gate: kernel output vs an
  independent torch dequant oracle from the SAME packed bytes, decoded per the KERNEL's exact
  convention (read the kernel source, not a different quant path). Worked example:
  `plugin/validate/test_mxfp4_moe_cpu.py` + the `INTEL_CPU_DSV4_MOE_PARITY` in-situ probe.
