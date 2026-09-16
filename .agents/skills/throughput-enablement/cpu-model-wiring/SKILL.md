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
7. **Runtime infra (the layer the op graph does NOT contain).** Compute wiring is
   not enough — the model also needs its runtime substrate on CPU, and these are
   frequent GAPs that only appear at bring-up (see the `coverage-gate` infra list):
   - **KV-cache / memory pool**: per-arch pools assume a specific packed layout +
     `store_dtype` (e.g. DSV4 = 584 B/token uint8 fp8-nope+bf16-rope+scales). You
     cannot `--kv-cache-dtype` your way out; wire a CPU path for the arch's native
     pool + accessors, or provide a CPU-valid (pool, dtype) pair.
   - **Attention-backend selection / compat guards**: an arch may force a backend
     (`dsv4`) that a generic guard rejects on CPU ("fp8 KV ⇒ intel_amx only").
     Wire/relax the guard in the PLUGIN (monkeypatch), never in `sglang/`, so the
     arch's backend + its pool are a supported CPU pair.
   - **Quant plumbing / device gates / config schema**: MLA may assert
     `weight_scale_inv` (needs fp8 block-quant, not bf16); `is_cpu()` requires
     `SGLANG_USE_CPU_ENGINE=1`; the framework-native config schema may differ from
     the model's dataclass (build config.json from the framework schema).

## Triage every gap into one of three buckets (cost differs by ~100x)
At each bring-up break, classify before acting — most gaps are cheap:
- **Routing gap** (cheapest): a CPU/torch path already exists but a predicate
  (`support_triton`, `is_cpu`, a platform guard, an env flag like
  `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`) didn't select it. Flip the predicate. Most
  infra + many DSA-adjacent breaks are this.
- **Mechanical port** (cheap): a Triton bookkeeping/index kernel with NO fallback,
  but pure index math — replicate in torch (e.g. paged `alloc_extend`, compressed-attn
  metadata). A per-item loop is fine for make-it-work.
- **Authoring gap** (the real cost): no CPU-viable reference — a genuine compute
  kernel, often a CUDA-JIT/accelerator kernel plus its data structures (e.g. the DSV4
  KV compressor = CUDA-JIT plan byte-layout + softmax-pool compute + state pool,
  prefill+decode × ratio 4/128). A handful of these dominate the enablement cost;
  route them through `kernel-feasibility-gate` → `kernel-authoring`.
VERIFY a claimed fallback actually runs on CPU: a "torch fallback" may target ANOTHER
accelerator (HIP/XPU) and call that accelerator's Triton/sgl_kernel (e.g.
`CompressorHip` uses `fused_softmax_pool_triton`) — not CPU-portable. Read it before routing.

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
3. **Walk the make-it-work bring-up ladder** on a TINY same-arch config (all ops
   present, tiny dims — iterate fast, no big node needed until AMX timing):
   instantiate → load weights (dummy is fine) → build KV/memory pool → select
   attention backend → run one prefill+decode. Fix each break and RE-RUN — infra
   gaps hide behind each other, so the next only appears after the current is fixed.
   Every fix lands in the plugin (thin subclass or targeted monkeypatch), never in
   `sglang/`. Record the ladder (each break → fix) as bring-up provenance.
4. Scale to the real config; confirm no shape/layout errors and that AMX (not AVX-512
   fallback) dispatched (`ONEDNN_VERBOSE=1`).
5. Hand off to `accuracy-oracle`.

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
- **Gaps hide behind gaps.** A clean instantiate does not mean it runs — the KV
  pool, backend guard, or quant plumbing break only after load. Budget make-it-work
  as an iterative ladder, and treat each newly-revealed infra gap as expected, not
  as scope creep. Use a tiny same-arch config so each iteration is seconds, not
  minutes on a scarce big-memory node.
- **A "fallback" is not automatically a CPU fallback.** HIP/XPU/NPU paths are
  non-CUDA but still call that accelerator's Triton/custom ops. Confirm the fallback
  is pure torch (or a CPU sgl_kernel) before routing CPU to it.
- **With dummy weights, a shape-correct stub can unblock profiling.** For an
  authoring-gap op you haven't ported yet, a stub that returns correctly-shaped
  tensors lets the WHOLE model run so `model-profile-hotspots` can rank where the
  real kernel effort belongs — then author the RoI-ranked ops for real (numeric
  correctness is validated later, with real weights).
