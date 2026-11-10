# sglang-cpu-opt — Automated CPU Model Enablement for SGLang

**North Star:** Deliver an AI Systems Performance Engineering Agentic Workflow along with skill files to automate day-0 support for new AI inference models with accuracy and performance.

**How:** An **agentic workflow** (parameterized skill files + agents) that takes a newly
released model and enables **performant, accurate CPU inference** in SGLang on Intel
Xeon (AMX) — as an Intel-maintained **plugin, no fork** — by reusing existing
hand-optimized kernels or creating new kernels (having distilled knowhow into skill files) and proving both correctness and performance.

## TL;DR — status at a glance
A reusable **spine-and-leaf playbook** (a phase-gated methodology **spine** + per-technique optimization **leaf** skills) drives every enablement. Two proven paths:
- **Thesis 1 — reuse known kernels:** OLMo-2-7B (BF16 + auto-INT8) and OLMoE enabled; **accuracy parity 1.0**; peer-relative within donor-kernel efficiency (eff_rel 0.94–1.26). → [details](#thesis-1--throughput-all-known-kernels--proven)
- **Thesis 2 — author new kernels:** **DeepSeek-V4-Flash** — novel DSA + MHC kernels authored & validated; **full matched GSM8K evaluation: CPU 1246/1311 (95.04%), GPU 1245/1311 (94.97%)**, with zero invalid answers. No aggregate CPU deficit in this run; not a claim of numerical equivalence or statistically significant superiority. **Baseline→optimized 2.19× prefill / 1.59× decode** remains a separate performance measurement. → [accuracy result and reproduction](docs/dev/DSv4-accuracy-closure.md)
- **Dig deeper:** 📘 [methodology playbook](docs/methodology-playbook.md) · 📗 [kernel-optimization playbook](docs/standalone-kernel-optimization-playbook.md) · 🧩 [kernel performance journey](docs/kernel-pilot-performance.md) · ⏱️ [end-to-end wall time](docs/e2e-wall-time.md)

> **Start here:** [THESIS_SUMMARY.md](THESIS_SUMMARY.md) — the one-page summary.
>
> **Correctness & hygiene, up front:** every enablement clears a fixed gate set *before* any number is published — (1) **dtype hygiene** — each weight family's stored dtype is audited and mapped to a supported compute dtype (low-bit **moved**, bf16 **computed**), and every `stored ≠ compute` dequant bridge is **parity-checked, not assumed**; (2) **coverage-gate** — no novel op is silently dense-approximated; (3) **accuracy oracle** — per-op kernel parity → real-prompt coherence → per-layer parity → task accuracy; (4) **reference oracle (Thesis 2, wherever a GPU + the model are available)** — a GPU reference forward captures per-(layer,op) tensors/logits so CPU bring-up is diffed against ground truth; the first divergent layer localizes integration bugs that per-op checks can't; (5) **honest roofline** — target vs measured at one labeled machine config. Unproven numbers are labeled UNVALIDATED, never implied.
>
> **Bird's-eye view of the agentic workflow & skills** — the staged, measure-first methodology behind the North Star (HW characterization → analysis → roofline → coverage → reuse/author → validate → certify), with all **32 skills** mapped to each stage.

![Staged agentic workflow for Day-0 CPU model enablement: 32 skills mapped from HW characterization through roofline, kernel reuse and authoring, validation and certification](docs/agentic-workflow-slide.png)

> Interactive version: [docs/agentic-workflow-slide.html](docs/agentic-workflow-slide.html) (open in a browser).

## Two theses, one plugin

Two enablement paths through the same agentic workflow: **Thesis 1** reuses existing AMX kernels
(wiring + validation only); **Thesis 2** authors the missing kernels for a genuinely new architecture.

### Thesis 1 — throughput, all-known-kernels — PROVEN
Results on Intel Xeon 6980P / Granite Rapids, **single socket**:

> **Machine config is matched:** roofline **and** measured below are BOTH **single socket**
> (TP=1, 128c NUMA 0-2, batch 8). We always compare roofline vs measured at the *same* config
> (never single-socket roofline against dual-socket measured).

| Model | Precision | Accuracy (vs HF) | gsm8k | Throughput prefill / decode (tok/s) | Roofline vs measured (1-socket) |
|-------|-----------|------------------|-------|-------------------------------------|---------------------------------|
| OLMo-2-7B (dense) | BF16 | next-token parity 1.0 | 0.60 | 1321 / 91 | GEMM 60.6→18.2 TF (30% of streamed ceiling) · [report](plugin/validate/results/olmo2_7b_bf16_roofline.md) |
| OLMo-2-7B (dense) | INT8 (auto-quantized) | parity 1.0 | 0.61 | 2013 / 135 (~1.5×) | int8 ceiling ~121 TF/socket · report pending |
| OLMoE-1B-7B (MoE) | BF16 | parity 1.0 | 0.19* | 7629 / 251 | MoE decode weight-streaming bound · report pending |

*1B-active base model. Machine-checkable proof: [plugin/certificates/](plugin/certificates).

**Peer-relative performance** (OLMo-2-7B dense GEMMs vs the Llama-3-8B donor on the
identical AMX kernel): `eff_rel` 0.94–1.26 — all PASS. Both BF16 and INT8 were produced
automatically from the same donor kernels (capability inheritance).

### Thesis 2 — new-kernel leg

When the coverage-gate finds a genuinely new op, we **author the kernel**. The method is a
**spine-and-leaf playbook**: a shared **spine** every model walks — *HW characterization → decompose →
coverage-gate → roofline → reuse-or-author → validate → certify* — that calls into per-technique **leaf**
skills (parallelize, tile, vectorize/AMX, prepack/BRGEMM, quantize, fuse) whenever a kernel must be driven
toward its roofline. Two pages document it in full:

- 📘 **[Methodology playbook](docs/methodology-playbook.md)** — *the spine*: the Phase 0→3 waterfall + gate
  ladder, the per-kernel performance-gap ledger, and §5 correctness-bug localization & serving-completeness
  (boundary-capture + reference-replay, the weight-prepack audit, the three serving axes).
- 📗 **[Kernel-optimization playbook](docs/standalone-kernel-optimization-playbook.md)** — *the leaf*: the
  staged *how* for driving one kernel to its roofline (establish-achievable → parallelize → tile →
  vectorize/AMX → prepack/BRGEMM → quantize → fuse), each step gated by measurement.

#### What the playbook enforces (reusable across models)
- **Coverage-gate routing** — decompose the model, match every op to existing CPU kernels, and route
  *only* the genuinely novel ops to the author leg (here: DSA + MHC); never dense-approximate a gap.
- **Up-front precision / data-type hygiene** — census each weight family's *stored* dtype from the
  **real checkpoint** and map it to a HW-supported *compute* dtype (low-bit **moved**, bf16
  **computed**); every `stored ≠ compute` dequant bridge is **parity-checked, not assumed**.
- **Reference-first kernel authoring** — write each novel kernel against an in-tree numeric oracle and
  gate on parity *before* it enters the serving path. No unverified kernel ships.
- **Autonomous debugging loop (skill-driven) — test cheaply, learn the most per run, fail fast.** The
  agent drives bring-up and bug-localization itself through a set of workflow skills:
  **cheap iteration** on a *tiny arch-faithful* config (seconds, not minutes on a scarce big-memory
  node) plus a **deterministic-dummy CPU↔GPU parity** harness that localizes correctness **per-layer
  without a real-weight load** (bit-identical dummy weights across the CPU engine and the GPU
  container); **high-information runs** that disposition *several* hypotheses in one expensive run (a
  pre-submit disposition matrix + multi-vector capture, never a one yes/no run); **async-heartbeat
  auto-wake** so the loop runs *submit → wait → auto-wake → fix → resubmit* with no human polling; and
  **sneak-preview + early-abort** on any long run — chunk-checkpoint a usable partial, read it against
  an expected floor, and `scancel`+fix the moment it is clearly off instead of paying the full wall.
  See [`.agents/skills`](.agents/skills) — `shared/high-information-runs`,
  `throughput-enablement/accuracy-oracle`, `throughput-enablement/cpu-model-wiring`.
- **Performance methodology — cheapest, highest-signal first (the order matters):**
  1. **Rule out systemic-config pathology FIRST.** A single mis-setting (thread cliff, OpenMP
     spin-wait, NUMA bind) inflates *every* op roughly uniformly and dwarfs any per-op tuning — sweep
     it before ranking hotspots. (Here it was worth **63× prefill / 10× decode** on its own.)
  2. **Truncated model + dummy weights.** A depth-reduced, full-**WIDTH** proxy reproduces every
     per-op bottleneck queue-free in ~1/N the time; nearly all fix-iteration lives here.
  3. **Full model + dummy weights.** The authoritative perf-vs-roofline run at full shapes/footprint on
     the target node — *before* paying any real-weight load — catching depth-aggregate + capacity
     effects the proxy can't.
  4. **Full model + real weights.** Paid **only** for accuracy + a final no-regression confirm.
- **Accuracy oracle gates before any perf claim** — real-prompt coherence → per-token parity → task
  accuracy; a failing gate blocks the number (it did here — see below).
- **Boundary-capture + reference-replay localization** — when a per-layer cosine vs a *same-checkpoint*
  GPU oracle flags a divergent layer, dump that op's operands and replay an **independent** reference on the
  **identical** operands to split *arithmetic bug* from *wrong operands*, then an **operand-identity check**
  (live weight vs raw checkpoint) names the line. This drove a ~15-pt DSV4 gap to one weight; method in
  [playbook §5](docs/methodology-playbook.md).

**We applied this playbook end-to-end to DeepSeek-V4-Flash** — the flagship below. (A second,
**structurally different** architecture, a clean mostly-autonomous run of the *same* playbook to demonstrate
**velocity**, is being **re-run in a dedicated session** and will be added here when complete.)

#### 1 · DeepSeek-V4-Flash — flagship (novel kernels authored + validated)
**DeepSeek-V4-Flash** (284B total / 13B active MoE; 43 layers, 64-head MLA, 256 experts top-6) carries
**two genuinely novel op families** — DeepSeek Sparse Attention (lightning-indexer + KV-compressor +
top-k sparse-MLA) and MHC hash-clustering — while everything else is covered by the DeepSeek-V2 CPU donors.

> **Deep dives for this model:** 🧩 **[Kernel performance journey](docs/kernel-pilot-performance.md)** (per-kernel roofline → baseline → C/C++, + GNR-vs-EMR) · ⏱️ **[End-to-end wall time](docs/e2e-wall-time.md)** (baseline→optimized tok/s + roofline calc) · ✅ **[Full accuracy evaluation](docs/dev/DSv4-accuracy-closure.md)** (CPU 95.04%, matched GPU 94.97%) · 🔎 **[Prefill/decode review brief](docs/dsv4-prefill-decode-review-brief.md)** (code map + perf data + tried & ruled out) · 📐 **[Per-GEMM regime analysis](docs/dsv4-gemm-regime-analysis.md)** (compute- vs BW- vs latency-bound).

- **Scope** — scope-discovery surfaced the *true* scope (DSA + MHC **plus** a runtime infra layer, not
  the "4 DSA kernels" a static op-scan implied) → [coverage/scope](plugin/coverage/deepseek_v4_flash_coverage.yaml).
- **Novel kernels authored + parity-validated on CPU** — DSA compressor softmax-pool, lightning-indexer,
  sparse-prefill attention ([compressor](plugin/intel_cpu_models/dsa_compressor_cpu.py),
  [indexer](plugin/intel_cpu_models/dsa_indexer_cpu.py),
  [sparse-attn](plugin/intel_cpu_models/dsa_sparse_attention_cpu.py)) + the MLA-core/MHC/infra layer
  ([_dsv4_cpu_infra.py](plugin/intel_cpu_models/_dsv4_cpu_infra.py)); the **composed** DSA attention
  reduces **exactly to dense at top-k=all** (err 1.8e-7), proving the sparse math
  ([dsa_attention_cpu.py](plugin/intel_cpu_models/dsa_attention_cpu.py)). A `index_gemm(M16)`
  feasibility gate returned a data-driven DEFER — no wrong kernel built.
- **Runs end-to-end at tp=1 by keeping the MXFP4 nibbles** — the routed experts stay **native 4-bit**
  on the CPU **MXFP4 W4A16** MoE kernel (fused fp4→bf16), so the model holds **~200 GB → fits ONE
  NUMA/SNC domain → clean tp=1** (loads ~135 s) → [run](plugin/validate/run_deepseekv4_flash_mxfp4_tp1.sbatch).
- **Correctness PASSES** — the accuracy oracle caught a real decode bug (DSA selection stubbed at
  decode → MLA gathered nothing → zero attention); fixed with a causal **dense fallback**. Per-token
  parity now passes (coherent, correct generations); the MXFP4 bridge is parity-checked standalone
  (cos 0.999992) and in-situ on the real checkpoint (cos 0.999995).
- **Full accuracy evaluation completed (2026-11-09): CPU 1246/1311 versus fresh GPU 1245/1311.**
  The evaluated CPU configuration enables chunk accumulation and routed/shared SwiGLU clamps,
  disables unsupported prefix-cache reuse, and leaves experimental activation quantization OFF.
  CPU reproduces all 1311 generation texts and answers from its prior 1246 run.
  Earlier prepack, compressed-state and batching repairs remain part of the serving path.
  [Result, exact flags, limitations and reproduce commands](docs/dev/DSv4-accuracy-closure.md).
- **Performance measured** — **baseline → optimized** (fully-naive eager path → all-C/AMX DSA kernels,
  dispatch-proven; 43L two-depth extrapolation, B=32, real-router guard): **prefill 79.8 → 174.5 tok/s
  (2.19×), decode 36.2 → 57.5 tok/s (1.59×)** on EMR, at **43% / 8% of the measured-EMR roofline** (prefill
  near its compute ceiling; decode overhead-bound with headroom).

**Key learnings (each distilled into a reusable skill — full set in the [playbook](docs/methodology-playbook.md)):**
1. **Compute from the native precision — keep the MXFP4 nibbles, never up-convert.** Feeding packed fp4
   straight to the W4A16 kernel is both a *capacity* win (~200 GB vs ~275 GB → fits one domain → tp=1) and
   *lossless* (fp4·2^k is exact in bf16). Up-converting fp4→fp8 doubled the footprint and forced a tp>1 swamp
   (~330× slower M=1 decode).
2. **Validate the measurement before optimizing.** The first per-op profile was an artifact — idle OpenMP
   threads busy-wait and thrash each forward; `OMP_WAIT_POLICY=passive` was worth **63× prefill / 10× decode**
   in that regime (but is a *regime-specific* lever — it regresses B=32 throughput, where spin wins).
3. **Audit every weight for prepack-vs-consumption mismatch (the ~15-pt correctness bug).** On a port that
   blanket VNNI/AMX-packs linears, a weight consumed by a plain matmul must be kept **row-major** — a packed
   weight + plain matmul is a *silent permutation*. The DSA `wkv_gate` hit this; signature = *live weight vs
   raw checkpoint orthogonal but per-row norms preserved*.
4. **Localize correctness bugs with boundary-capture + reference-replay, and validate three serving axes.**
   Per-layer cosine vs a same-checkpoint GPU oracle → first divergent layer → operand-identity check names the
   line; then prefill ≠ serving-correct (decode persistence + cross-request batching are separate gates). Full
   method: [playbook §5](docs/methodology-playbook.md).

**Remaining (scoped):** wire the authored DSA kernels into an **incremental-sparse decode**
(O(context²)→O(context·topk)) + the paged flash-MLA *serving* runtime — backend plumbing + tuning,
not novel-kernel authoring (the DSA math is authored + proven in isolation).

#### DeepSeek-V4-Flash — full matched accuracy evaluation

The final custom-protocol GSM8K run evaluates the same **1311 post-shot questions**
on both backends: **CPU 1246 (95.04%), GPU 1245 (94.97%)**. CPU-only successes:
9; GPU-only successes: 8; extracted-answer agreement: 1283/1311.
This closes the aggregate accuracy-gap campaign, not every per-question numerical
difference. It is not a published-score reproduction or a repeatability certificate.

The historical GPU1250 run used batch64 and a different engine/cache lifecycle;
the fresh comparison uses identical shard boundaries, batch32, prefill512,
radix cache OFF, prompts, stops and scoring. CPU is TP1, GPU TP4. The earlier
302/320 CPU subset must not be compared as if it were the same evaluation as
the 1311-question GPU run. [Full provenance](docs/dev/DSv4-accuracy-closure.md).

**(A) Numerical diagnostics and task accuracy answer different questions.**
Historical component checks reported MXFP4 bridge cosine 0.999992 standalone /
0.999995 in situ, and sparse-to-dense error 1.8e-7 at top-k = all. These are
bounded operator checks, not proof of bit identity or all-token parity.
Per-layer captures helped localize the earlier prepack bug; later matched-prefix
traces still show CPU/GPU differences and reference execution sensitivity.
Coherent generations and smooth layer-cosine curves cannot establish that a
residual difference is harmless. The [chronological review](docs/dev/DSv4-shared-clamp-disposition-REVIEW.md)
separates operator evidence, selected-case experiments and full-task results.

**(B) Accuracy recovery required multiple corrections: attention layout/state, cache lifecycle, and routed/shared SwiGLU clamping.**

![DSV4-Flash final matched full-set CPU/GPU accuracy](docs/plots/dsv4_accuracy_full_20261109.svg)

| Row | gsm8k | Note |
|---|---|---|
| Published Flash-**Base** (DeepSeek) | **≈0.908** | reference point — a *different* (base, FP8) checkpoint |
| Historical GPU reference on Flash-Base | **0.923** | 1210/1311, [`gpu_base_gsm8k.json`](plugin/validate/results/gpu_base_gsm8k.json); a custom-protocol reference, not exact published-protocol reproduction |
| CPU baseline (pre-fix) | **0.796** | the number the "eval harness" story tried to explain away |
| + `wkv_gate` row-major fix (prefill) | **0.865** | compressed attention now correct in prefill (layer-2 cosine 0.9997) |
| + decode persist + request-aware mask (historical subset) | 0.9437 | 302/320; not comparable as a full-set result |
| Historical GPU, post-trained Flash | 0.9535 | 1250/1311; older execution protocol |
| **Latest CPU, full set** | **0.9504** | **1246/1311**, chunk accumulation + routed/shared **SwiGLU clamps ON**, experimental quantization OFF |
| **Fresh matched GPU, full set** | **0.9497** | **1245/1311**, TP4; [machine-readable result](plugin/validate/results/dsv4_latest_fullacc_20261109.json) |

- **The decisive move was the correctly-bound oracle.** The published 90.8 is Flash-**Base** (FP8), a
  *different* checkpoint — which we **evaluated on GPU** (**0.923** under our custom protocol); our
  post-trained Flash has no published gsm8k number, so "below 90.8" was never a
  valid signal. Running the **GPU on the same checkpoint + same harness** gave **0.9535** → the real target was
  ~0.95, and CPU at 0.80 was genuinely ~15 pts low. *Comparator-provenance dissolved the false "below published"
  panic; the correctly-bound GPU oracle then revealed the gap was real after all.*
- **An early bug: a silent weight-layout permutation.** The DSA compressor's `wkv_gate` projection was
  AMX-VNNI-**prepacked** (as the CPU path packs every linear) but **consumed by a plain matmul**
  (`compute_kv_score → linear_bf16_fp32`) → scrambled projection → garbage compressed KV → wrong attention on
  every compressed layer (37 of 43). This is the **same bug class** the plugin had already fixed for `wo_a`;
  `wkv_gate` was simply never audited.
- **Localized to one line via boundary-capture + reference-replay** (method now in the playbook §5): per-layer
  attn-output cosine vs the GPU oracle broke **exactly at the first compressed layer** (pure-SWA layers 0/1 =
  0.9995); a boundary capture + independent replay on identical operands showed the attention *arithmetic* was
  correct (self-consistency cos≈1.0) so the compressed-KV *values* were wrong; the operand-identity check — live
  `wkv_gate.weight` vs the raw checkpoint = **cos 0.0007 with per-row norms preserved** — named it a prepack
  permutation.
- **The initial attention fixes** (playbook §5.4): (1) keep `wkv_gate` row-major → **0.865**;
  (2) persist per-request compressed KV so **decode** attends long-range context (a prefill-only handoff ran
  decode SWA-only); (3) make the coverage mask **request-aware** (a position-only mask leaks across a batched
  prefill) → **0.9437 on the historical 320-question subset**. Subsequent full-set
  work addressed cache lifecycle, chunk accumulation and the missing SwiGLU clamps detailed below.
  The earlier attribution of the remaining gap to "benign accumulation" was not
  established by that subset or by layer cosine alone. The latest matched full-set
  result above supersedes that conclusion.
- **The later SwiGLU correction: apply the configured clamp in BOTH routed and shared experts.**
  The CPU expert paths had omitted the clamp used by the GPU/reference path.
  The native CPU correction computes `SiLU(min(gate, limit)) * clamp(up, -limit, limit)`
  with this checkpoint's `swiglu_limit=10`, preserving packed-weight GEMMs.
  It is implemented in the [routed-expert loader](plugin/intel_cpu_models/_dsv4_moe_clamp.py)
  and [shared-expert loader](plugin/intel_cpu_models/_dsv4_shared_clamp.py), using the
  [clamped activation kernel](plugin/kernels/dsv4_moe_clamp.h).
  In the recorded full-set comparison, routed-clamp/shared-OFF CPU scored **1240/1311**;
  routed+shared clamps scored **1246/1311** (**13 recoveries, 7 regressions; net +6**).
  The latest full CPU run reproduced all 1311 texts and answers from that 1246 run.
  See the [experiment record](docs/dev/DSv4-shared-clamp-disposition-REVIEW.md).
  **These fixes are on `main`, but remain opt-in:** the reported **95.04%** CPU result uses
  `INTEL_CPU_DSV4_MXFP4_CLAMP=1` and `INTEL_CPU_DSV4_SHARED_CLAMP=1`, along with the
  [complete evaluated configuration](docs/dev/DSv4-accuracy-closure.md).

**Reproducible, not hand-waved:** generations are saved (`--gens-out`) and scored offline
([`score_gsm8k.py`](plugin/validate/score_gsm8k.py) grid); the full run is node-parallel + chunked
([`run_gsm8k_dsv4_sharded.sbatch`](plugin/validate/run_gsm8k_dsv4_sharded.sbatch) +
[`combine_gsm8k_shards.py`](plugin/validate/combine_gsm8k_shards.py)); the standard harness is
[`run_gsm8k_lmeval.sbatch`](plugin/validate/run_gsm8k_lmeval.sbatch). Method codified in
[`accuracy-oracle`](.agents/skills/throughput-enablement/accuracy-oracle/SKILL.md).

**Precision & compute-type hygiene (checked up front, before any kernel is chosen).** GNR AMX has
native matmul tiles for **bf16 / fp16 / int8 only — no fp8 or fp4**. So model discovery audits every
weight family's *stored* dtype (from the real checkpoint, not just `quantization_config`) and maps it
to a *compute* dtype the hardware actually supports: sub-16-bit weights are **moved** low-bit (a
bandwidth/capacity win) and **computed in bf16** after a fused dequant. Every `stored ≠ compute` step
is a dequant **bridge** verified numerically, not assumed:

| Weight family (DeepSeek-V4-Flash) | Stored / moved | Compute (AMX) | Bridge | Lossless? | Correctness gate |
|---|---|---|---|---|---|
| MoE routed experts | **MXFP4** (e2m1 + e8m0 group-32) | **bf16** | W4A16, fused fp4→bf16 in-GEMM | ✅ fp4·2^k exact in bf16 | ✅ [parity](plugin/validate/test_mxfp4_moe_cpu.py) PASS · standalone cos 0.999992 + in-situ real-ckpt cos 0.999995 |
| MLA / indexer / shared-expert proj | **fp8** e4m3 | **bf16** | W8A16, dequant fp8→bf16 | ✅ fp8 levels exact in bf16 | donor fp8 CPU path |
| norms / router / embed / lm_head | bf16 | bf16 | none (native) | — | native |

int8 is the only low-precision AMX *compute* tile; fp4/fp8 route through **bf16** compute (not int8)
to stay lossless — the low precision is spent purely on movement/footprint. Keeping the experts
**native 4-bit** (instead of up-converting fp4→fp8, which doubles the footprint) is what lets the
whole model fit **one NUMA domain at tp=1**. Bridges + status: [dtype_bridge_gates](plugin/coverage/deepseek_v4_flash_coverage.yaml).

## Roofline target vs measured (published with every result)
Every published result carries the **roofline achievable target** alongside the **measured**
number, plus a **per-op breakdown** of which kernels fall short of their roofline (measured is
usually below target — the gap is the remaining optimization RoI). Target is published up front;
measured fills in when the kernels run. **Both legs share the same per-kernel roofline-vs-measured
view** — in the **reuse leg (Thesis 1)** measured = the *reused donor* kernel; in the **new-kernel
leg (Thesis 2)** measured = *our authored* kernel — and **Thesis 2 additionally carries a whole-model
end-to-end wall-time roofline-vs-measured chart** (the full enablement, not just one op).
- **Thesis 1 — donor-kernel optimization** (OLMo-2-7B, BF16, GNR): [report](plugin/validate/results/olmo2_7b_donor_roofline.md)
  · ![chart](plugin/validate/results/olmo2_7b_donor_roofline.png) — reused donor GEMMs run at 0.94–1.26× the donor's own efficiency (wiring preserved) yet only **42–60% of the AMX ceiling** → headroom. Backend is oneDNN/LIBXSMM BRGEMM (best-in-class inner loop), so the gap is **composition/memory-traffic**, not tile-loop quality.
- **Thesis 2 — new-kernel authoring** · DeepSeek-V4-Flash (authored DSA kernels; MXFP4 W4A16 MoE; EMR).
  **(a) Per-kernel roofline vs baseline vs C/C++** — the authored DSA kernels against their own roofline
  (same view as Thesis 1):

  ![DeepSeek-V4-Flash per-kernel roofline vs baseline vs C/C++ (DSA kernels, M=32 streaming)](docs/plots/operating_point_v6.png)

  Full per-kernel ledger, M-sweep, and GNR-vs-EMR: **[kernel performance journey](docs/kernel-pilot-performance.md)**.

  **(b) End-to-end wall time — roofline vs baseline vs optimized** (the whole-model view, unique to the full
  enablement): baseline → optimized **2.19× prefill / 1.59× decode** (43L, B=32, real-router guard), at
  **43% / 8%** of the roofline — but **both rooflines are IDEALIZED**: the prefill 409 counts only the
  **dense-GEMM (linear-layer) FLOPs at AMX peak** (attention/DSA/MHC/norms/glue assumed free), and the decode 678
  assumes pure **BW-bound** streaming. Decode is **BW-bound by nature but latency-bound by implementation** —
  measured 57.5 is **12× below the BW floor** (M=1 is 62× below), a per-step overhead/threading gap that **is the
  opportunity** (thread right-sizing + BRGEMM expert-grouping + dispatch reduction — the Kimi-K3 single-stream
  levers), not a ceiling. **The prefill *achievable* is being re-derived by counterfactual** after a clean
  interleaved A/B showed our region-timer attribution over-predicted a lever (the mask) that gave **0%**
  end-to-end — a measurement-reliability lesson, see **[end-to-end wall time](docs/e2e-wall-time.md)**:

  ![DeepSeek-V4-Flash end-to-end wall time — baseline vs optimized vs idealized roofline (EMR; GNR pending)](docs/plots/dsv4_node_walltime_v6.png)

  Full ledger + roofline calculation + the decode reassessment: **[end-to-end wall time](docs/e2e-wall-time.md)**.
  *(An earlier "75% of DRAM-BW roofline" figure was a MoE
  mode-collapse artifact — a dummy-random router under-counted expert bytes; the real-router `W_active`=40 GB
  number supersedes it.)*
- Regenerate from a `model-profile-hotspots` run: `python plugin/validate/roofline_vs_measured.py
  --in <profile.json> --out-prefix plugin/validate/results/<name>`.

## Repository map
- [`docs/`](docs) — the deep-dive pages linked from Thesis 2 above:
  [methodology](docs/methodology-playbook.md) + [kernel-optimization](docs/standalone-kernel-optimization-playbook.md)
  playbooks, [kernel-journey](docs/kernel-pilot-performance.md) + [end-to-end wall-time](docs/e2e-wall-time.md)
  performance, and [`docs/archive/`](docs/archive) (point-in-time page snapshots; git history is the full archive).
- [`.agents/`](.agents) — the workflow: `agents/` (model-enablement, cpu-optimizer) and
  grouped `skills/` — see [`.agents/skills/README.md`](.agents/skills/README.md), which
  marks the **throughput-thesis** skills.
- [`plugin/`](plugin) — generated model classes (`intel_cpu_models/`), generic validators
  (`validate/`), the auto-quantizer (`quantize/`), **certificates** (`certificates/*.yaml`),
  and the DeepSeek coverage analysis (`coverage/*.yaml`).
- [`tools/`](tools) — node calibration + microbenchmarks.

## Reproduce
Each certificate carries its exact `reproduce:` commands. SGLang CPU install:
`sglang/docs/docs/hardware-platforms/cpu_server.mdx`. The validators are generic —
point `--model` / `--config` at any model. Weights are not stored here (downloaded separately).
