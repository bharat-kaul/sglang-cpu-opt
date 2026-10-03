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
See `sub-numa-clustering` for the capacity-fit + tp rule (per-rank footprint must fit
one domain's RAM; tp bounded by head/expert divisibility).

### Single-rank FULL-NODE run (correctness capture of a model bigger than one SNC domain)
For a correctness CAPTURE (not a perf run) of a model that **overflows one SNC/NUMA domain**, you may
NOT be able to shard: `tp` is bounded by head/expert divisibility (e.g. 64 heads ⇒ tp∈{1,2,4}, never 6),
so a model too big for one domain at the largest legal tp has no sharded fit. Run it as a **single rank
spanning the whole node** instead: `--tp 1` + `SGLANG_CPU_OMP_THREADS_BIND=0-<ncores-1>` (ALL physical
cores across ALL domains). Why this works and plain `numactl --interleave=all` does NOT:
- sglang's CPU memory accounting (`get_available_gpu_memory`, `srt/utils/common.py`) **divides free RAM
  by `n_numa_node`** — so a tp=1 rank is budgeted only `total/n_numa` (e.g. 1.5 TB ÷ 6 ≈ 250 GB) and a
  bigger model is refused / OOM-killed. And `init_threads_binding` **pins the rank's OMP threads to ONE
  domain's cores** by default (`SGLANG_CPU_OMP_THREADS_BIND="all"`), forcing first-touch onto node 0 —
  which **overrides an external `numactl`**. Setting `SGLANG_CPU_OMP_THREADS_BIND` to custom cores makes
  `get_cpu_memory_capacity` stop dividing (returns `None` = full RAM) AND the threads span the listed
  cores, so allocation spreads across all domains and the whole model fits the node. Cross-domain BW is
  irrelevant for a capture. Tell-tale you hit the default-pin trap: loader logs `avail mem ≈ total/n_numa`.
  [GLM-5.3 Flash: 306 GB fp8 real model loaded + ran coherently on a 1.5 TB / 6-SNC node only with
  `SGLANG_CPU_OMP_THREADS_BIND=0-255`; every prior `mem_fraction`/`numactl` attempt OOM-killed at 250 GB.]

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
- **Arg-RESOLUTION device probes crash on CPU BEFORE any model module imports.** A model
  whose arch name is in an sglang family list (e.g. the DSA family — `is_deepseek_dsa()`
  true for DeepSeek-3.2 AND GLM-5) can route `resolve_once()` into a CUDA/ROCm branch that
  calls `torch.cuda.get_device_capability()` / `get_device_properties()` — which RAISES on a
  CPU-only torch build. This fires during `Engine.__init__ → _launch_subprocesses →
  resolve_once`, i.e. BEFORE the external model package's model modules (and their
  `install()` hooks) import — so a fix placed in the model subclass is TOO LATE. Fix pattern:
  install the CPU device-probe guard at **plugin-PACKAGE import** (`intel_cpu_models/__init__.py`)
  and ALSO pre-import the plugin in the launcher before `Engine(...)`; shim the probe to a sane
  default (e.g. capability (9,0)) only when `not torch.cuda.is_available()` (inert on GPU). Note a
  sibling arch may NOT be in the list (DeepSeek-V4 is not → never hit this), so "the last DSA model
  worked on CPU" does not mean the next one will — check the arch's membership in every such list.
- Editing `sglang/` "just to get it running" breaks the no-fork guarantee and the
  PR story — always subclass in the plugin.
- Divisibility pad must match what the packing kernel expects, or AMX silently
  falls back to AVX-512.
- **Gaps hide behind gaps.** A clean instantiate does not mean it runs — the KV
  pool, backend guard, or quant plumbing break only after load. Budget make-it-work
  as an iterative ladder, and treat each newly-revealed infra gap as expected, not
  as scope creep. Use a tiny same-arch config so each iteration is seconds, not
  minutes on a scarce big-memory node.
- **DECODE breaks hide behind PREFILL.** Prefill and decode take DIFFERENT code paths
  (extend vs absorb/decode, different attention backend entrypoints). A full prefill pass
  (`MAX_NEW=1`) can succeed while decode has its own ladder of breaks. Separate the two:
  drive prefill to green first (it's the correctness-fingerprint path), then exercise decode
  with `MAX_NEW>=2`. The per-layer PARITY fingerprint should capture BOTH passes (prefill `pf`,
  first-decode `dc0`) keyed by token-count so one diff proves both. (GLM-5.3: prefill fully green
  while decode had ~4 more breaks; verified both via pf + dc0 fingerprint parity.)
- **NoPE MLA on CPU (qk_rope_head_dim=0 → `rotary_emb` is None) is a real sglang gap.** Neither
  stock CPU MLA path works out-of-the-box: (a) the fast `MLA_FUSED_ROPE_CPU` path's kernel
  DIVIDES BY qk_rope_head_dim → SIGFPE (exit -8) at decode, AND its `PackWeightMethod`
  transpose+VNNI-packs `w_kc`/`w_vc`; (b) the GENERIC absorb path IS NoPE-native (skips rope when
  `rotary_emb is None`) and CPU-viable, but its plain `torch.bmm` needs the LOGICAL `w_kc`
  `[heads,qk_nope,kv_lora]`, not the packed bytes. FIX: for NoPE MLA, KEEP `w_kc`/`w_vc` logical
  (no-op the MLA's `PackWeightMethod` in `init_mla_fused_rope_cpu_forward` when `rotary_emb is None`)
  AND route the dispatcher (`_dispatch_mla_subtype`) to the generic `AttnForwardMethod.MLA`. Models
  WITH rope keep the fused path. Dead-ends (don't repeat): rerouting to generic WITHOUT un-packing
  (bmm shape/layout mismatch, same class as the DSV4 `wo_a` VNNI-vs-einsum bug), and a dummy
  `cos_sin_cache` (fixes the deref but the kernel still SIGFPEs on the div-by-zero). PERF TODO: the
  logical-`w_kc` bmm is not AMX-accelerated (same trade as `wo_a`).
- **A weight AMX-packed (transpose+VNNI) for path A, consumed by plain torch in path B, is scrambled.**
  `_amx_process_weight_after_loading` both transposes AND VNNI-packs; transposing back does NOT recover
  the logical weight. When you reroute an op to a different forward path, ensure its weights are in the
  layout THAT path expects (keep them logical if it uses plain bmm/einsum). Grep for weights consumed
  outside `.apply` (bespoke bmm/einsum) when a rerouted path gives cos≈0 or a bmm shape mismatch.
- **A "fallback" is not automatically a CPU fallback.** HIP/XPU/NPU paths are
  non-CUDA but still call that accelerator's Triton/custom ops. Confirm the fallback
  is pure torch (or a CPU sgl_kernel) before routing CPU to it.
- **With dummy weights, a shape-correct stub can unblock profiling.** For an
  authoring-gap op you haven't ported yet, a stub that returns correctly-shaped
  tensors lets the WHOLE model run so `model-profile-hotspots` can rank where the
  real kernel effort belongs — then author the RoI-ranked ops for real (numeric
  correctness is validated later, with real weights).
