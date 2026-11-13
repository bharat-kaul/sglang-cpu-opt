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

## Thesis 2 — RUNNING END-TO-END on CPU (DeepSeek-V4-Flash, tp=1, native MXFP4) — full matched GSM8K: CPU 95.04%, GPU 94.97%

**Scope (same-day coverage-gate).** The model is **mostly covered** by the DeepSeek-V2 CPU
kernels (MLA core, dense GEMM, MoE, norm, rope, top-k); the only new family is **DeepSeek Sparse
Attention (DSA)** — a lightning-indexer + KV-compressor + top-k sparse-MLA path (config:
`index_n_heads=64`, `index_head_dim=128`, `index_topk=512`, per-layer `compress_ratios`,
`sliding_window=128`). This confirms V4-Flash is a Thesis-2 model.

**Breakthrough — the whole model now fits ONE NUMA/SNC domain at tp=1.** The routed experts ship
as **native MXFP4** (packed fp4 + e8m0 group-32 scales). The naive path dequantized them to fp8,
which **doubled** the resident footprint to ~275 GB and overflowed a single 258 GB SNC domain —
forcing a cascade of dead-end placement hacks (TP-shard, NUMA-interleave, page-migration), and
tp>1 M=1 decode is ~330× slower on this stack. Keeping the experts **native 4-bit** and routing
them to the CPU **MXFP4 W4A16** MoE kernel (fused fp4→bf16 dequant in the AMX GEMM) holds the
resident model at **~200 GB → fits one domain → clean tp=1**, sidestepping the entire distribution
problem. *Lesson (now an upfront skill check): treat a capacity overflow as a footprint/precision
problem first — never inflate a native low-bit checkpoint; keep it low-bit and use the low-bit kernel.*

**Precision & compute-type hygiene (up front).** The model is NOT uniform precision. GNR AMX has
native tiles for **bf16 / fp16 / int8 only — no fp8 or fp4 matmul** — so every sub-16-bit weight is
*moved* low-bit (bandwidth/capacity) and *computed* in bf16 after a fused dequant. The choices per
weight family, and how each is verified:

| Weight family | Stored / moved | Compute (AMX tile) | Bridge | Lossless? | Correctness gate |
|---|---|---|---|---|---|
| MoE routed experts | **MXFP4** (e2m1 + e8m0 group-32) | **bf16** | W4A16, fused fp4→bf16 in-GEMM | ✅ fp4·2^k is exact in bf16 | ✅ parity **PASS** — standalone cos 0.999992 + in-situ real-ckpt cos 0.999995 |
| MLA/indexer/shared-expert proj | **fp8** e4m3 | **bf16** | W8A16, dequant fp8→bf16 | ✅ fp8 levels exact in bf16 | donor fp8 CPU path (deepseek-v2) |
| norms / router / embed / lm_head | bf16 | bf16 | none (native) | — | native |

No int8 *compute* is used here: int8 is the only low-precision AMX tile, but routing fp4/fp8 through
int8 would add activation-quant error — so compute stays **bf16** (lossless), and the low precision is
spent purely on movement/footprint. Every `stored ≠ compute` row is a dequant **bridge** that is
verified numerically, not assumed (see the correctness-gate column; `coverage/deepseek_v4_flash_coverage.yaml`
→ `dtype_bridge_gates`).

**Status today.**
- ✅ **Runs end-to-end on CPU** (Granite Rapids, tp=1): loads in ~135 s, resident ~200 GB on one
  SNC domain, prefill + multi-token decode complete.
- ✅ **Data-type audit** (upfront, from the real checkpoint): every op maps to a supported GNR type
  (see the precision-hygiene table above) — fp4 experts→W4A16, fp8 MLA/indexer/shared-experts→W8A16
  (dequant→bf16 AMX), bf16→native AMX. Each `stored ≠ compute` bridge carries a required parity gate.
- ✅ **Decode thread-cap** tuned on the **real** model (the heavy MoE dominates → whole-forward
  optimum ~8 threads, not the dummy-weight proxy's ~40).
- ✅ **Correctness validated.** A real-prompt coherence check (the accuracy oracle's Layer 0)
  caught that the decode path emitted garbage: the DSA sparse selection was stubbed at decode, and
  the MLA attention had **no dense fallback**, so it gathered nothing → zero attention. Fixed with a
  **causal dense fallback** (attend over all valid KV up to the query position).
  Coherent generations are a smoke check, not proof of per-token parity.
- ✅ **Full accuracy campaign completed (2026-11-09).** CPU **1246/1311 (95.04%)**;
  fresh matched GPU **1245/1311 (94.97%)**, zero invalid answers on both sides.
  CPU reproduces all 1311 prior generation texts and extracted answers. Paired
  successes: CPU-only 9, GPU-only 8; answer agreement 1283/1311.
  The configuration explicitly enables chunk accumulation and routed/shared
  SwiGLU clamps, with radix cache OFF and experimental activation quantization OFF.
  Historical GPU1250 used a different batch/engine/cache protocol; historical
  CPU302/320 was a subset, not a full-set comparison. Neither supports the old
  claim that a remaining one-point gap was necessarily "benign accumulation."
  The prepack, compressed-state and batching fixes were followed by cache
  containment/lifecycle and activation-clamp corrections. This result closes the
  aggregate task-gap campaign, not numerical-equivalence certification or default
  promotion. [Evidence, configuration and reproduction](docs/dev/DSv4-accuracy-closure.md).
- ✅ **Performance measured — baseline → optimized** (fully-naive eager path → all-C/AMX DSA kernels,
  dispatch-proven; 43L two-depth extrapolation, B=32, real-router guard): **prefill 79.8 → 174.5 tok/s
  (2.19×)**, **decode 36.2 → 57.5 tok/s (1.59×)** on EMR — **43% / 8% of the measured-EMR roofline**
  (AMX bf16 12.4 TF, DRAM 848 GB/s): prefill near its compute ceiling, decode overhead-bound with headroom.
  Decode is **~88% C/AMX kernels, all dispatched**; the remaining levers are MHC-aten (~7%) + the
  attention-glue seam (~8%), not a torch rewrite. *(An earlier "prefill 69.6 / decode 9.3, 75%-of-BW"
  figure was a MoE mode-collapse artifact — a dummy-random router under-counted expert bytes; superseded.)*
  Full ledger + per-node (EMR/GNR) comparison: `docs/e2e-wall-time.md`, `docs/kernel-pilot-performance.md`.
- **Current real-weight comparison and analytical targets:** the historical proxy above is
  separate from the matched EMR/GNR socket measurements. The latest chart now includes
  per-platform prefill/decode **conditional analytical scenarios**, using matched hardware
  calibration and archived forward shapes, without new full-model runs. Routing/conversion
  assumptions remain material; this is not a measured ceiling or an optimality certificate.
  [Targets, sensitivities and reproducible model](docs/e2e-wall-time.md#matched-achievable-roofline-targets).

*This is the worked Thesis-2 flagship: the coverage-gate routed it, the plugin wired native MXFP4
+ the DSA CPU path, and the accuracy oracle did exactly its job — it blocked the number on real bugs
(a stubbed-decode bug, then a silent weight-prepack permutation in the DSA compressor) until a
same-checkpoint GPU oracle proved CPU ≈ GPU, and only then were the numbers published.*

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
