---
name: quantization-amx-int8
description: "Use when the accuracy budget allows lower precision to raise the compute ceiling on Intel Xeon, OR to SHRINK the resident footprint so a model fits one NUMA/SNC domain (a CAPACITY lever, applied FIRST — see the capacity section). Covers INT8 AMX (2x the BF16 tile throughput), weight-only vs full INT8, native low-bit STORAGE + in-kernel dequant (fp4/MXFP4/int4 W4A16), per-channel scales, the never-up-convert-a-low-bit-checkpoint rule, and the correctness gate. For the compute-ceiling use, apply last after the BF16 path is tuned; for the capacity/fit and memory-bound-decode uses, apply first."
---

# Quantization — INT8 AMX

BF16 tuning maximizes efficiency at a fixed ceiling; quantization **raises the
ceiling**. AMX INT8 (`amx_int8`) does ~2x the ops/cycle of AMX BF16
(`amx_int8_ops_per_cycle_per_core` ≈ 2048 vs 1024).

Follow the [canonical precision and target policy](../../../../docs/methodology-playbook.md#canonical-acceptance-policy).
Start from the original model's numerical contract, not an assumed BF16 baseline.
Explore reduced-precision movement for weights, KV, activations and intermediates;
measure conversion/scale/packing costs rather than assuming they are hidden.

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
3. **Separate storage loss from compute conformance.**
   - **Lossless storage:** verify exact decoding over the real format/scale domain.
     Equal weight values do not waive validation of the consuming kernel.
   - **Lossy shrink:** obtain an explicit accuracy budget and validate numerical/task
     outcomes before promotion. Up-conversion cannot restore discarded bits.
   - In both cases validate activation precision, scales/groups, accumulation and published
     intermediate-rounding boundaries. Higher precision is not automatically source-conformed.

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
Measure the actual INT8 path and justify its matched target separately from the resident
probe. Apply the agreed threshold plus the complete standalone opportunity ledger.

## Order
Preserve native packed storage from bring-up. Evaluate movement precision throughout
optimization; additional lossy compute/storage variants require their own approval and
comparison against a tuned source-conformed baseline, not necessarily BF16.

## Storage dtype ≠ compute dtype (sub-8-bit checkpoints, e.g. fp4/nvfp4/mxfp4)
Probe the target's supported compute ISA; do not assume native fp4/int4 matmul.
A checkpoint's weight dtype (fp4 experts in DeepSeek-V4-Pro; nvfp4/mxfp4/int4 GGUF) is a
**storage/quantization format**, not a compute path on this ISA. So a sub-8-bit checkpoint
requires a compatible packed-input kernel: if absent, it is an ENABLEMENT GAP. The GEMM decodes tiles to a
supported compute dtype. The framework's CPU MoE/GEMM kernel will reject the packed
sub-8-bit layout (DSV4: `fused_experts_cpu` asserts `packed_w1.size(2)==packed_K`; fp4's
2-per-byte packing gives K/2 → `3588 vs 7168`). Route it to a supported path:
- **Accuracy-parity FIRST target = the published numerical contract.** A BF16 compute
  path is eligible only when its decode, accumulation and rounding conform to that contract.
  Validate scaled values, not just the unscaled fp4 levels. Extra activation quantization
  requires separate approval and validation.
- **Memory: upconvert is 4× (fp4→bf16), 2× (fp4→int8).** Check it FITS. DSV4-Pro experts
  ≈735 GB fp4 → ≈2.9 TB bf16 > a 1.5 TB node. Distinguish reference from production:
  1. **Reference-only widened fixture:** allowed for an explicitly labeled, capacity-budgeted
     oracle comparison; never a production fallback, even when it fits.
  2. **Production: fp4 in RAM, per-tile decode IN-KERNEL to supported compute dtype.**
     Retain packed weights plus scales; reuse a compatible kernel or author the missing path.
     Prove source conformance rather than assuming identical arithmetic to the widened fixture.
     This is also a candidate performance lever,
     not just a capacity fix — decode is weight-streaming-bound, so keeping weights fp4
     (1/4 the bytes of bf16) directly raises the memory-bound throughput ceiling; the
     in-kernel dequant is cheap relative to the bytes saved (benefit 2 above).
- **Sequence:** source-conformed native-storage baseline → approved additional precision
  variants, each validated independently. Do not replace the original contract with a
  higher-precision baseline by default.
- **The dequant bridge is UNVERIFIED until a numeric parity gate passes.** Whether you dequant at
  load for an oracle fixture or per-tile in production, the fp4→bf16 decode (scale layout, nibble order,
  group/block size, VNNI prepack, SwiGLU gate/up split) fails SILENTLY when wrong — plausible
  garbage, not a crash. Do NOT treat "the bf16 kernel exists and it ran" as correct. Emit the
  per-kernel parity gate from `accuracy-oracle` §low-bit parity gate: kernel output vs an
  independent torch dequant oracle from the SAME packed bytes, decoded per the PUBLISHED FORMAT's exact
  convention (read the kernel source, not a different quant path). Worked example:
  `plugin/validate/test_mxfp4_moe_cpu.py` + the `INTEL_CPU_DSV4_MOE_PARITY` in-situ probe.
