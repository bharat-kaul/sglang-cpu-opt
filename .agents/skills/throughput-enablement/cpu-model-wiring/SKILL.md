---
name: cpu-model-wiring
description: "Use after coverage-gate passes to implement the CPU-enabled model in the EXTERNAL plugin (never fork sglang/). Encodes the concrete SGLang wiring surface: attach PackWeightMethod to linear/embedding weights (VNNI prepack), route attention through the intel_amx backend via use_intel_amx_backend fast paths, run FusedMoE through the CPU AMX path with _amx_process_weight_after_loading, optionally enable W8A8-int8, and register the model class via SGLANG_EXTERNAL_MODEL_PACKAGE. Produces a clean, PR-shaped diff a human can upstream."
---

# CPU Model Wiring (plugin, no fork)

Turn a COVERED op graph into a running CPU model by connecting each op to its
existing kernel. All code lands in the Intel plugin package and integrates via
`SGLANG_EXTERNAL_MODEL_PACKAGE` + the attention-backend registry. You never edit
files under `sglang/` — this keeps the demo upstream-clean and lets a human cut a
community PR from the same diff.

## The wiring surface (per op kind)
1. **Linear weights (qkv, o, gate/up/down, lm_head).** Attach
   `PackWeightMethod` (`sglang.srt.layers.amx_utils`) so weights convert to
   VNNI2/tile order ONCE at load (`convert_weight_packed`; requires `OC%16==0`,
   `IC%32==0` — pad per the fallback if not). Per-call GEMM then streams A against
   packed B (BRGEMM) with no repack.
2. **Attention.** Gate the CPU fast path on `use_intel_amx_backend(self)`
   (`sglang.srt.utils`) and route through the `intel_amx` attention backend for
   GQA/MHA; MLA models use the `forward_mla_fused_rope_cpu` mixin pattern.
3. **MoE.** Run experts through `FusedMoE`
   (`sglang.srt.layers.moe.fused_moe_triton`) and call
   `_amx_process_weight_after_loading(module, ["w1","w2"])` (and shared-expert
   weights) so expert weights are prepacked for the CPU grouped GEMM.
4. **Norm / RoPE / activation.** Reuse `RMSNorm`, `rotary_embedding`,
   `SiluAndMul`/`GeluAndMul` as-is — no wiring beyond correct placement (QK-norm
   applies RMSNorm to q,k before attention).
5. **Precision variants (inherit ALL the donor supports).** Precision is a donor
   capability: the CPU dense/MoE kernels expose BOTH an AMX BF16 path and an AMX
   INT8 (`w8a8_int8`) path (~2x compute peak, `amx_int8_ops_per_cycle_per_core`).
   Generate EVERY precision the donor supports, each as its own config + certificate:
   - **BF16** — no checkpoint change; primary, accuracy-tight.
   - **INT8 (w8a8)** — needs an int8 checkpoint (int8 weights + per-channel
     `weight_scale`); `--quantization w8a8_int8` does NOT quantize a plain bf16
     checkpoint online. If the release is unquantized, first run an automated,
     calibration-free pass (per-channel weight int8 + dynamic per-token activation
     = RTN) to emit a compressed-tensors int8 checkpoint, THEN serve with
     `--quantization w8a8_int8`. Validate at the INT8 budget (looser) and gate perf
     against the INT8 ceiling. Skip only if no accuracy budget allows it.
6. **Registration.** Expose the model class through the external package so the
   registry discovers it without touching core; keep the class a thin subclass of
   the upstream model that only overrides load-time prepack + CPU forward paths.

## Launch contract (single GNR node)
```
SGLANG_USE_CPU_ENGINE=1 sglang serve \
  --model-path <NEW_MODEL> --device cpu --tp <SNC_COUNT> \
  --trust-remote-code --disable-overlap-schedule
```
`--tp` = number of sub-NUMA clusters (one TP rank per SNC); bind cores with
`SGLANG_CPU_OMP_THREADS_BIND`. Confirm the AMX all-core layout matches the profile.

## Procedure
1. Subclass the upstream model in the plugin; override only load-time prepack hooks
   and the CPU forward fast paths from the wiring surface.
2. Register via the external package; load with `--device cpu`.
3. Run a single forward pass on a short prompt; confirm no shape/layout errors and
   that AMX (not AVX-512 fallback) dispatched (`ONEDNN_VERBOSE=1`).
4. Hand off to `accuracy-oracle`.

## Gate
Model loads on CPU, completes a forward pass, and the intended AMX kernels
dispatch. A pass here is functional only — accuracy and performance are proven by
the next two gates. Keep the diff minimal and upstream-shaped (thin subclass, no
core edits) so `enablement-certificate` can attach it for PR review.

## Pitfalls
- Forgetting the prepack hook → correct output but stock repack-every-call GEMM →
  the perf gate will fail peer-relative even though accuracy passes. This is the
  most common wiring bug; check prepack first when peer-relative underperforms.
- Editing `sglang/` "just to get it running" breaks the no-fork guarantee and the
  PR story — always subclass in the plugin.
- Divisibility pad must match what the packing kernel expects, or AMX silently
  falls back to AVX-512.
