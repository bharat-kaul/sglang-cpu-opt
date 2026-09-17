---
name: uarch-perf-probe
description: "uArch Performance Probe (uPP) — a standalone, workload-agnostic microbenchmark suite that CHARACTERIZES a CPU microarchitecture and emits a machine_constants.json of trusted, measured constants (compute peak per dtype, cache/DRAM bandwidth ladder, NUMA/SNC bandwidth matrix, roofline ridge point, gather-stage crossover, thread-scaling). USE THIS when: bringing up optimization on a NEW or unseen uarch with no priors (e.g. a GNR follow-on); refreshing/verifying priors on a KNOWN part to catch drift or config regressions; needing the constants that feed kernel-authoring knobs (BLOCK_N, TILE, accumulators, one-rank-per-domain, stage-vs-tinygemm M) or the roofline ceiling; establishing a per-node achievable-performance baseline. It is a PLUG-IN: any agentic workflow across the org can call the runner and consume the JSON; it has NO dependency on any model or serving stack. DO NOT use it to optimize a specific kernel (that's kernel-authoring) or to profile a running workload (that's per-op timers/profilers) — uPP characterizes the MACHINE, not the workload."
---

# uArch Performance Probe (uPP)

**One line:** the durable asset for cross-uarch optimization is not the constants, it is
the *self-validating harness that regenerates them on real silicon*. uPP is that harness.

It converts the workflow from **"trust the GNR priors"** to **"regenerate the priors on the
actual part"** — which ports to an unseen uarch with zero priors and also catches drift on
a known part. Constants are trusted because they were *measured here*, not inherited.

## When to invoke
- **New/unseen uarch, no priors** (GNR follow-on, a different vendor part): run the full suite
  first; its `machine_constants.json` becomes the sole source of the kernel-authoring constants.
- **Known part (even GNR):** run it to *verify* the corpus priors still hold and to catch a
  bad machine config (turbo/governor drift, a NUMA/SNC mode change, a missing ISA) before it
  quietly corrupts an optimization run. uPP flags these; the priors alone cannot.
- **Before authoring a kernel:** it supplies the `kernel-feasibility-gate` baseline (achievable
  peak + ridge) and the `establish-achievable-performance` per-domain numbers.

## What it emits (each probe → a CONSTANT → a kernel KNOB)
| Probe | Constant | Feeds |
|---|---|---|
| `compute_peak` | achieved GF/s per dtype (bf16/int8/fp32) | roofline compute asymptote; is int8's 2× real here? |
| `memory` | cache-ladder BW vs footprint + DRAM triad BW | L1/L2/L3 knees → BLOCK_K, BLOCK_M/N; roofline BW asymptote |
| `numa` | per-(cpu,mem)-domain BW matrix; local/remote | domain count, per-domain BW → one-TP-rank-per-domain + first-touch; SNC-baked roofline |
| `roofline_ridge` | flops/byte ridge = peak/BW per dtype | the memory-bound-vs-compute-bound decision |
| `gather_crossover` | smallest M where gather amortizes | stage-then-BRGEMM vs tinygemm (`prefer_amx_stage_M_ge`) |
| `threading` | compute + BW scaling vs threads | cores-to-saturate-BW, grain, split-K viability |

Output: `machine_constants.json` (+ `.md`) with `meta` (node/ISA/freq-hygiene), each probe's
raw sweep, and a `derived_kernel_knobs` block that other skills read directly.

## Self-validation gates (a wrong constant is worse than no constant)
1. **Frequency hygiene** — records governor + turbo state; if not pinned (`performance`,
   turbo-off) it emits a `⚠` and marks `clean=false`. A characterization run on an unpinned
   node is not authoritative.
2. **ISA presence** — records `amx_bf16/amx_int8/avx512_*`; a "compute peak" without the
   expected ISA present means a silent fallback → the number is not the ceiling you think.
3. **Measurement discipline** — every timing uses warmup + best-of-N + a dead-code-elimination
   guard (see `hygiene.timeit`), so a compiler cannot elide the timed work.
4. **Per-probe isolation** — one probe failing never aborts the suite; it records `status`.
Corollary of the corpus rule *"verify the ISA actually dispatched"* applied to the
characterization layer itself.

## Invoke
```bash
# full run (pin frequency first for authoritative numbers)
python tools/uarch_perf_probe/runner.py --out machine_constants.json
# fast subset (skips the subprocess NUMA matrix) for a login/dev node
python tools/uarch_perf_probe/runner.py --quick
# on a batch node
sbatch tools/uarch_perf_probe/run_uarch_probe.sbatch
```
Consume: point kernel-authoring / roofline steps at `machine_constants.json`; read
`derived_kernel_knobs` for the ready-to-use knobs.

## How it feeds the rest of the workflow
- `establish-achievable-performance` / `model-roofline-analysis`: use `compute_peak`,
  `memory`, `numa` for the ceiling (SNC baked in from the measured per-domain BW).
- `kernel-authoring`: use `derived_kernel_knobs` (`prefer_amx_stage_M_ge`,
  `cores_to_saturate_bw`, `ridge_flops_per_byte`, `per_domain_bw_gbps`) to instantiate the
  arch-invariant 4-layer skeleton with measured constants instead of assumed ones.

## Native probes (extension point)
The current probes are torch/oneDNN-level (they measure *achievable* peaks through the same
BRGEMM/AMX stack kernels use — the right number for authoring). Cycle-accurate raw-intrinsic
probes (isolated `dpbf16ps`/tile-op latency, accumulator-depth sweep, prefetch-distance sweep)
are a documented C++ extension: add a probe under `probes.py` that emits the same
constant→knob contract, with the same dispatch + hygiene self-checks. A *categorically* new
uarch feature (new tile shape, new memory tier, new clustering) needs a newly-authored probe —
uPP characterizes known axes completely and should be extended, doc-informed, for new ones.

## Org-reuse contract
- **Standalone:** `tools/uarch_perf_probe/` depends only on torch + stdlib + optional
  `numactl`/`lscpu`. No model/serving import. Copy the folder or add it to `PYTHONPATH`.
- **Stable output schema:** consumers rely on `machine_constants.json` keys, not internals.
- **Demo example:** this repo's DeepSeek-V4 CPU workflow is a reference consumer — its roofline
  charts and kernel-authoring gate read the same constants uPP emits.

## Procedure
1. Pin frequency (governor=performance, turbo off) if you control the node; else note it.
2. Run the full suite on the target; confirm `meta.freq_hygiene.clean` and the expected ISA.
3. Review `derived_kernel_knobs`; sanity-check against any datasheet asymptotes.
4. Commit `machine_constants.json` next to the workflow that consumes it (provenance).
5. On a known part, diff against the last run to catch drift.
