# Automated CPU Model Enablement — Executive Summary

**Business problem.** New AI models release faster than we can enable performant CPU
inference in frameworks like SGLang (month+/model). Hypothesis: because new models
mostly recombine a small set of already-hand-optimized kernels, an agentic workflow
with parameterized skill files can take a model config, map its ops to existing
kernels, wire it into the framework, and prove correctness + performance — collapsing
enablement from a month to hours, delivered as an Intel-maintained plugin (no fork).

**Approach — two theses, one plugin.**
- **Thesis 1 (throughput / all-known-kernels):** models whose ops are fully covered by
  existing AMX kernels are enabled by *wiring + validation* only.
- **Thesis 2 (new-kernel leg):** genuinely novel ops get an AI-written, roofline-tuned
  kernel, human-gated. The workflow's coverage-gate decides which leg a model takes.

**Capability inheritance (the central principle).** A newly enabled model automatically
inherits *every capability the donor kernels expose* — not just their compute, but their
**precision variants (BF16 and INT8/w8a8), ISA path (AMX), weight prepacking / VNNI layout,
the intel_amx attention backend, and the FusedMoE grouped-GEMM path**. The workflow generates
a config for each capability the donor supports (e.g. *both* BF16 and INT8), and whenever the
donor kernels gain a capability or get faster, every enabled model — including these —
inherits that gain with **no re-enablement**.

---

## Thesis 1 — PROVEN (3 GREEN certificates, Intel Xeon 6980P / Granite Rapids, single socket, BF16 AMX)

| Model | Precision | Accuracy (vs HF) | gsm8k | Throughput prefill / decode (tok/s) |
|-------|-----------|------------------|-------|--------------------------------------|
| OLMo-2-7B (dense) | BF16 | next-token parity **1.0** | **0.60** | **1321 / 91** |
| OLMo-2-7B (dense) | **INT8** (auto-quantized) | parity **1.0** | **0.61** | **2013 / 135  (~1.5×)** |
| OLMoE-1B-7B (MoE) | BF16 | parity **1.0** | 0.19* | **7629 / 251** |

*OLMoE is a ~1B-active base model; 0.19 is its expected level (0 parse errors). *All models
were NOT previously CPU-enabled in SGLang; each was enabled same-day by the generic workflow,
whose only per-model output is one thin generated class.*

**Peer-relative performance proof (the core result).** OLMo-2-7B's dense GEMMs, run on the
*identical* AMX kernel that powers Llama-3-8B, reached the donor's efficiency at every op:
`eff_rel` = qkv 1.26, o 1.10, gate_up 0.97, down 0.94 — all ≥ 0.90 (PASS). Correctness is
proven twice (per-token parity + task accuracy); performance is proven relative to a shipped
model on the same silicon and kernel. **Precision matrix:** BF16 and INT8 were both produced
automatically from the same donor kernels (INT8 via a calibration-free RTN quantizer),
INT8 giving ~1.5× throughput at preserved accuracy.

## Thesis 2 — SCOPED & READY (DeepSeek Flash v4.1, DeepSeek-V4 / DSA)

Coverage-gate analysis (same-day) found the model is **mostly covered** by the DeepSeek-V2 CPU
kernels (MLA core, dense GEMM, MoE, norm, rope, top-k). The **only** new work is one bounded
kernel family — DeepSeek Sparse Attention: 1 PARTIAL (the AMX lightning-indexer decode GEMM at
M=16, currently AVX-512 only) + 3 GAP (KV compressor, fp8 group-quant/Hadamard, sparse-prefill).
The analyzer independently rediscovered SGLang's own in-tree `decode.cpp:13` TODO, and emitted
4 scaffolding tasks for the new-kernel leg. This confirms V4.1 belongs to Thesis 2, not Thesis 1.

---

## Why this is the best achievable performance for Thesis 1 (with rising headroom)

Peer-relative validation shows OLMo-2-7B **matches or beats the donor model** on the identical
kernel (eff_rel 0.94–1.26). That means the enablement extracts essentially **100% of what the
donor kernel delivers** — there is no wiring loss (no missing prepack, no AVX-512 fallback, no
NUMA thrash), which is exactly what the two-sided (absolute + peer-relative) gate guarantees.

Any remaining headroom lives in the **shared kernel itself**, not the enablement: on this
silicon a streamed BF16 GEMM tops out near ~60 TF/socket versus the ~350 TF resident peak — a
data-movement wall (cache blocking / BRGEMM), not a model-wiring gap. Because every enabled
model *reuses that shared donor kernel*, improving it once (e.g. better BRGEMM blocking,
LIBXSMM/TPP, INT8) **automatically lifts every enabled model — including these — with no
re-enablement.** So Thesis 1 performance is, by construction, bounded by the donor kernel and
rises in lockstep with it; we have proven we sit at that bound today.

## How to share this with someone else

Everything is self-contained under `sglang-cpu-opt/` — portable, no SGLang fork:
- `.agents/` — the agentic workflow: `agents/` (model-enablement, cpu-optimizer) and grouped
  `skills/` (`throughput-enablement/`, `shared/`, `kernel-optimization/`; see `skills/README.md`).
- `plugin/` — generated model classes (`intel_cpu_models/`), validators (`validate/`), the
  auto-quantizer (`quantize/`), machine-checkable **certificates** (`certificates/*.yaml`), and
  the DeepSeek coverage analysis (`coverage/*.yaml`).
- `tools/` — node calibration + microbenchmarks.

Share options:
1. **Git:** commit `sglang-cpu-opt/` and push, or `git bundle create sgl-cpu.bundle HEAD` for a
   single-file transfer.
2. **Tarball (results + workflow only):**
   `tar czf sgl-cpu-enablement.tgz sglang-cpu-opt/.agents sglang-cpu-opt/plugin/certificates \
    sglang-cpu-opt/plugin/coverage sglang-cpu-opt/plugin/validate/results`
3. The **certificates** (`plugin/certificates/*.yaml`) are the shareable proof — each is
   self-describing with the numbers plus the exact reproduce commands. The **skills** are plain
   `SKILL.md` files that drop into any repo alongside the two agent definitions.
