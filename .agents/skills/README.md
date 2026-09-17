# Skills — organized by workflow leg

Two legs share one plugin. Skills are grouped so it is obvious which belong to the
**throughput thesis** (enable a new model from EXISTING kernels) versus the
**new-kernel leg** (write/optimize a kernel). `shared/` skills are used by both.

## `throughput-enablement/` — THE THROUGHPUT THESIS
Enable a new all-known-kernels model on CPU, no fork. Load in this order:
1. `model-enablement-playbook` — orchestrator index (load first)
2. `model-op-decomposition` — config → normalized op graph
2b. `fusion-analysis` — graph-level fusion opportunities (BW/AI lever); routes each to a
   donor fused-kernel or a new-fused-kernel task; cross-checks external impls/claims
   (vLLM, TRT-LLM, FlashInfer, authors' report, blogs) — code = oracle, claim = hint
2c. `model-roofline-analysis` — TOP roofline tier: whole-model, per-phase (prefill/decode),
   Amdahl-ranked high-ROI plan; gates entry to the kernel-level roofline (feasibility-gate)
2c2. `enablement-scope-discovery` — dependency-closure scan of the ACTUAL forward + backend +
   KV/memory-pool code to the sub-op leaf; enumerates ALL novel kernel families (across every
   subsystem, not just attention) + the runtime substrate; classifies routing/port/authoring;
   honest scope BEFORE the first bring-up (catches second families like DSV4 MHC + the infra layer)
2d. `model-profile-hotspots` — EMPIRICAL tier: run the model (random weights ok), per-kernel
   measured time vs kernel roofline floor, rank by RoI = share*(1-efficiency); measured
   hotlist wins over the analytical ranking and commits kernel budget
3. `kernel-capability-registry` — op → existing kernel contracts (+ donors, precisions)
4. `coverage-gate` — 100% covered? else route the gap to the new-kernel leg
5. `cpu-model-wiring` — wire in the plugin (prepack, intel_amx, FusedMoE, BF16 + INT8)
6. `accuracy-oracle` — per-layer parity + task (gsm8k/mmlu), per precision
7. `peer-relative-roofline` — perf vs the donor running the identical kernel
8. `enablement-certificate` — machine-checkable pass/fail + PR-ready diff

Driven by the `model-enablement` agent.

## `shared/` — used by BOTH legs
- `establish-achievable-performance` — calibrate the node's achievable ceilings (step 0)
- `roofline-validation` — turn a throughput number into a pass/fail verdict
- `overhead-attribution` — split a slow run into KERNEL vs unoptimized-TORCH vs FRAMEWORK
  time EARLY (tagged env-gated boundary timers → op profiler → kernel-isolation vs the uPP
  roofline floor), so you fix the right layer and don't author a kernel that can't move
  end-to-end time. Refines `model-profile-hotspots`; feeds `kernel-feasibility-gate`.

The throughput thesis needs `throughput-enablement/` **plus** `shared/`.

## `kernel-optimization/` — NEW-KERNEL LEG (write/optimize a kernel)
For a coverage GAP (e.g. DeepSeek-V4 DSA indexer). Driven by the `cpu-optimizer` agent.
- `cpu-optimization-playbook` — orchestrator index for this leg
- `kernel-feasibility-gate` — MANDATORY go/no-go BEFORE authoring: user-reviewed roofline
  walk-through + baseline microbenchmark + Amdahl; prevents building the wrong kernel
- `kernel-authoring` — GENERATIVE: write a new kernel by adapting a donor (SGLang corpus
  map + LIBXSMM/TPP + oneDNN); load first when authoring, before the tuning skills
- `cpu-serving-integration` — MAKE IT RUN: wire authored kernels into the model's serving
  runtime (attention backend forward + paged KV-cache pack/unpack + metadata); ends at the
  proof deliverable (model runs + accuracy + roofline-vs-measured)
- `openmp-parallelization`, `cache-blocking-tiling`, `amx-vectorization`,
  `weight-prepacking-brgemm`, `quantization-amx-int8`
- `cpu-gemm-amx-bf16` — worked example composing the above

## Cross-cutting / org-reusable plug-ins (used by ANY workflow)
- `uarch-perf-probe` — **uArch Performance Probe (uPP):** standalone microbenchmark suite that
  CHARACTERIZES a CPU microarchitecture and emits `machine_constants.json` (compute peak per
  dtype, cache/DRAM BW ladder, NUMA/SNC BW matrix, roofline ridge, gather-stage crossover,
  thread scaling) with self-validation (freq-hygiene + ISA-dispatch + DCE-guarded timing).
  Converts "trust the priors" → "regenerate the priors on the real silicon" for an unseen uarch
  (e.g. a GNR follow-on), and refreshes/verifies them on a known part. Feeds
  `establish-achievable-performance`, `model-roofline-analysis`, and the `kernel-authoring`
  knobs. Runnable suite: `tools/uarch_perf_probe/`. No model/serving dependency — any agentic
  workflow can call it; this repo's DeepSeek-V4 workflow is the demo consumer.
