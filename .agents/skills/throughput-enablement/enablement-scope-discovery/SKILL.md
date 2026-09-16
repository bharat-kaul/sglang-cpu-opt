---
name: enablement-scope-discovery
description: "Use RIGHT AFTER model-roofline-analysis and BEFORE the first empirical bring-up/measurement (model-profile-hotspots). Upgrades the curated op-graph (model-op-decomposition) into a COMPLETE, code-path-grounded enablement scope: a dependency-closure walk of the model's ACTUAL forward + attention backend + KV/memory-pool + allocator code that follows every device dispatch (imports, is_cuda/is_hip/is_xpu guards, env flags, JIT loaders) down to the concrete kernel that runs on the target device, recursing into sub-ops. Enumerates EVERY novel kernel family across ALL subsystems (not just attention) PLUS the runtime infra/substrate, classifies each leaf (CPU-ready / routing / mechanical-port / authoring), and emits the honest scope + cost BEFORE bring-up so the effort isn't discovered reactively. Prevents the failure where a static op-scan misses a second kernel family (e.g. DSV4's MHC in the norm path) and the whole infra layer (KV pool, allocator, backend guards)."
---

# Enablement Scope Discovery (dependency-closure, pre-bring-up)

The lesson that created this skill: for DeepSeek-V4 the op-scan enumerated the DSA
attention family but MISSED (a) a second novel family — MHC hash-clustering in the
**norm** path — and (b) the entire runtime substrate (KV pool layout, paged
allocator, attention-backend guards). Both only surfaced during bring-up, one break
at a time. A model has as many gaps as it has accelerator-only code paths, not as
many as the op graph lists. **Scan the actual code to the sub-op leaf, across all
subsystems and the substrate, before the first run.**

## Pipeline position
`model-op-decomposition → fusion-analysis → model-roofline-analysis →
enablement-scope-discovery → model-profile-hotspots → coverage-gate → …`. This is the
static closure that makes the scope honest; it feeds `coverage-gate` and is done
BEFORE any measurement (you cannot measure what will not yet run).

## Inputs
The op graph (`model-op-decomposition`); the SGLang model class; the arch's
attention backend class; the KV/memory-pool + `kv_cache_configurator` for the arch;
the allocator; grep access to the whole repo.

## Procedure
1. **Seed from the op graph**, then WALK THE ACTUAL CODE, not a curated checklist.
   Open the model class forward, the decoder layer, the attention backend's
   `init_forward_metadata_*` + forward, and the memory-pool/configurator for this arch.
2. **Dependency-closure to the leaf.** For each op AND each runtime step, follow the
   dispatch to the concrete kernel that runs on the TARGET device: chase `import`s,
   `is_cuda()/is_hip()/is_xpu()/is_cpu()` branches, env-flag gates, platform guards,
   and JIT/TileLang/DeepGEMM loaders. **Recurse into sub-ops** — a torch OUTER op may
   call accelerator-only INNER kernels (e.g. a torch `hc_pre` that calls a TileLang
   `hc_split_sinkhorn`; a torch compress that needs a CUDA-JIT plan). Stop only at a
   leaf that is CPU-ready or has no CPU path.
3. **Enumerate ALL subsystems, not just attention.** Novel families hide outside the
   headline op:
   - attention (+ sparse/indexer/**compressor**), norm/layernorm (+ **clustering/hash
     e.g. MHC**), routing, MoE, activation, embedding/head, positional.
   - AND the RUNTIME SUBSTRATE (invisible to op-decomposition): KV-cache/memory-pool
     layout + store dtype, paged **allocator kernels** (alloc_extend/decode, write-
     cache-indices), attention-**backend selection + compat guards**, quant plumbing
     (weight_scale_inv, block quant, KV quant method), forward-batch metadata prep
     (positions, compressed-attn metadata), device predicates (`SGLANG_USE_CPU_ENGINE`),
     config schema (framework-native vs dataclass).
4. **Grep the arch's code paths for accelerator markers** to catch what reading misses:
   `triton|tilelang|deep_gemm|flash_mla|\.cuh|sgl_kernel|nvcc|get_device_properties|
   pin_memory=True|cuda_graph`, and device guards that lack an `is_cpu` branch. Every
   hit on a path this arch executes is a scope item.
5. **Classify each leaf** (drives the cost): **CPU-ready** (no work) / **routing**
   (a CPU/torch path exists behind a predicate/flag — flip it) / **mechanical-port**
   (Triton index/bookkeeping kernel, no fallback, pure math → torch) / **authoring**
   (accelerator-only compute, no CPU-viable reference → real kernel; author a torch
   reference first). See `cpu-model-wiring` for the taxonomy.
6. **Emit the scope manifest + cost.** Group by family/subsystem; list every leaf with
   its classification and the CPU lever; sum the cost (routing≈cheap, port≈medium,
   authoring≈dominant). This is the scope you commit to and show the user BEFORE
   bring-up — not a number that grows one break at a time.

## Output — enablement scope manifest
`{kernel_families: [{name, subsystem, leaves:[{op, classification, cpu_lever}]}],
infra: [{item, classification, cpu_lever}], estimate: {routing_n, port_n, authoring_n}}`.
Feeds `coverage-gate` (the authoring/port leaves are the real GAPs) and sets
expectations for `cpu-model-wiring`'s bring-up ladder.

## Gate
The manifest must be grounded in code paths (file:line per leaf), cover every
subsystem AND the substrate, and recurse to leaves — not stop at outer ops. A leaf
you cannot classify is `UNKNOWN` (treated as authoring until proven otherwise). Only
after this closure is the enablement effort honestly scoped; bring-up then CONFIRMS
it rather than discovering it.

## "Can we swap the backend instead of porting?" — check before assuming
A tempting shortcut is to route a novel arch's attention to an existing CPU backend
(e.g. `intel_amx` MLA) to avoid porting its kernel set. It only works for ops the
BACKEND dispatches. It does NOT avoid kernels the MODEL FORWARD calls directly.
Verify three things first: (1) are the target kernels invoked in the model class
forward or in the attention backend? (grep the model .py) — model-forward calls
(`fused_q_norm_rope`, MHC `hc_pre`) are unavoidable by a backend swap; (2) is the
backend hardcoded for the arch? (`arg_groups/model_overrides/<arch>.py`); (3) does
the model forward even have `use_intel_amx_backend` CPU branches (deepseek_v2 does;
deepseek_v4 does NOT)? If the model forward lacks CPU branches, the swap requires
re-authoring the forward — usually MORE work than porting the individual kernels.
DeepSeek-V4 spike outcome: NOT VIABLE for exactly these reasons.

## Worked example (DeepSeek-V4, what this would have surfaced up front)
- **DSA family** (attention): compressor (CUDA-JIT plan + softmax-pool + state pool),
  indexer (fp8 paged MQA logits — torch path behind a flag = routing), sparse-prefill,
  index_gemm(M16). **MHC family** (norm): mhc_pre / hc_split_sinkhorn / hc_combine /
  mhc_post — TileLang-only (authoring; torch outer frag only).
- **Infra**: DSV4 fp8 KV pool + guard (routing/patch), paged allocator alloc_extend/
  decode + write_cache_indices + compute_position (mechanical-port / routing via
  support_triton), FlashMLA metadata (routing → None), deep_gemm indexer metadata
  (routing via env), device gate `SGLANG_USE_CPU_ENGINE`, config schema (transformers
  -native), fp8 block-quant requirement.
Producing THIS list before the first run turns a reactive break-cascade into a planned
work list — the whole point of the skill.
