---
name: kernel-capability-registry
description: "The contract layer between an op graph and the existing CPU/AMX kernels. Use after model-op-decomposition to load, for every supported op, its kernel provider, capability requirements (dtype, VNNI layout, divisibility, attention/mask kind), the already-enabled DONOR model that exercises it, and that donor's measured roofline efficiency. Coverage-gate matches the op graph against this registry; peer-relative-roofline uses the donor efficiencies as the performance bar. Kept in assets/registry.yaml so new kernels register without code changes."
---

# Kernel Capability Registry

The single source of truth for "what op → which existing kernel, under what
contract, proven on which model, at what efficiency." This is the keystone that
makes both coverage and performance claims deterministic and auditable.

## Structure (`assets/registry.yaml`)
Each entry is one op kind:
```
- op: dense_gemm            # canonical op name from model-op-decomposition
  kernel: amx_brgemm_bf16   # existing CPU kernel that provides it
  provides:                 # capability contract the op MUST satisfy
    dtype: [bf16, fp16, int8]
    layout: vnni2_prepacked  # via PackWeightMethod / convert_weight_packed
    divisibility: {OC: 16, IC: 32}
  sglang_hook: sglang.srt.layers.amx_utils.PackWeightMethod
  donor: llama              # already-enabled model that runs this kernel
  donor_efficiency: 0.72    # donor's measured % of achievable ceiling (this node)
```

## What each field is for
- `provides` — the capability contract. An op is COVERED only if its signature (from
  decomposition) satisfies every field. A mismatch (e.g. `IC%32 != 0`, sliding
  mask a kernel lacks) is a GAP, not "close enough".
- `sglang_hook` — the exact plugin wiring point (`cpu-model-wiring` uses it).
- `donor` + `donor_efficiency` — the peer baseline. Because the new model runs the
  IDENTICAL kernel, `peer-relative-roofline` expects it to reach `donor_efficiency`
  (± tolerance) on matching shapes; a shortfall means a wiring bug, not a kernel limit.

## Seed coverage (verified against the SGLang tree)
| op | kernel | donor model | hook |
|----|--------|-------------|------|
| dense_gemm (qkv, o, gate/up/down) | AMX BRGEMM bf16 + prepack | llama, qwen2 | `amx_utils.PackWeightMethod`, `convert_weight_packed` |
| gqa_attention | intel_amx attention backend | llama, qwen2 | `utils.use_intel_amx_backend` |
| mla_attention | fused MLA rope CPU | deepseek_v2 | `deepseek_common/.../forward_mla_fused_rope_cpu` |
| moe_grouped_gemm | FusedMoE CPU path | qwen2_moe, deepseek_v2 | `_amx_process_weight_after_loading`, `moe.fused_moe_triton.FusedMoE` |
| rms_norm / qk_norm | CPU RMSNorm | llama, olmo2 | `layers.layernorm.RMSNorm` |
| rope (+scaling) | CPU rotary | llama, qwen2 | `layers.rotary_embedding` |
| swiglu / geglu | SiluAndMul / GeluAndMul | llama | `layers.activation` |
| w8a8_int8 gemm | AMX INT8 + per-token quant | (quant variant) | `hardware_backend/cpu/quantization`, `--quantization w8a8_int8` |
| embed / lm_head | VocabParallelEmbedding + PackWeight | all | `vocab_parallel_embedding` |

## KNOWN GAPS (route to cpu-optimizer, do NOT mark covered)
- `dsa_sparse_attention` / lightning indexer / compressor (DeepSeek V4/Flash):
  CUDA sm120 / NPU / HIP only; **no CPU kernel** — the CPU AOT source itself flags
  the missing `index_gemm_kernel_nn` (M=16) AMX kernel. This is the leg-2 flagship.
- Any attention-sink / novel-mask variant a donor does not already run on CPU.
- fp8 per-token-group quant on CPU (dsv4 wo_a path) — no AMX kernel yet.

## Procedure
1. Load `assets/registry.yaml`; ensure `donor_efficiency` for this node is filled
   (from the donor's `roofline-validation` run — else run it once per node).
2. Hand the registry to `coverage-gate` for matching.
3. After a successful enablement, append the new model as an additional donor for
   each op it exercised, with ITS measured efficiency (learned-patterns).

## Gate
The registry is valid only if every donor efficiency was measured on the CURRENT
hardware profile. Stale or cross-node efficiencies invalidate the peer bar — re-run
the donor roofline before trusting a peer-relative verdict.
