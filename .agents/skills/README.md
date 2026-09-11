# Skills — organized by workflow leg

Two legs share one plugin. Skills are grouped so it is obvious which belong to the
**throughput thesis** (enable a new model from EXISTING kernels) versus the
**new-kernel leg** (write/optimize a kernel). `shared/` skills are used by both.

## `throughput-enablement/` — THE THROUGHPUT THESIS
Enable a new all-known-kernels model on CPU, no fork. Load in this order:
1. `model-enablement-playbook` — orchestrator index (load first)
2. `model-op-decomposition` — config → normalized op graph
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

The throughput thesis needs `throughput-enablement/` **plus** `shared/`.

## `kernel-optimization/` — NEW-KERNEL LEG (write/optimize a kernel)
For a coverage GAP (e.g. DeepSeek-V4 DSA indexer). Driven by the `cpu-optimizer` agent.
- `cpu-optimization-playbook` — orchestrator index for this leg
- `openmp-parallelization`, `cache-blocking-tiling`, `amx-vectorization`,
  `weight-prepacking-brgemm`, `quantization-amx-int8`
- `cpu-gemm-amx-bf16` — worked example composing the above
