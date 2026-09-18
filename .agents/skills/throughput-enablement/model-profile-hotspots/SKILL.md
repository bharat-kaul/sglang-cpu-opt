---
name: model-profile-hotspots
description: "EMPIRICAL roofline tier — runs the ACTUAL model on the target node (random/dummy weights are fine; timing not accuracy) and collects a per-kernel wall-time profile, then compares each kernel's MEASURED time against its own kernel-level roofline floor to compute recoverable headroom. Ranks kernels by RoI = measured_share × (1 − measured/roofline_floor): the recoverable fraction of end-to-end time. Runs AFTER the analytical `model-roofline-analysis` and BEFORE `kernel-feasibility-gate`. Catches the cases where analysis is wrong (a kernel far below its roofline for a reason the analytical model didn't predict — e.g. issue/conversion-bound, register spill, bad blocking) and prevents authoring kernels that are already near their ceiling or too small a share to matter."
---

# Model Profile Hotspots (measured, RoI-ranked)

The lesson that created this skill: the DSA indexer analytical roofline said
"memory/gather-bound," but the microbench showed it was issue/conversion-bound and
already near its *achievable* ceiling — no headroom. Analysis alone would have sent
us to author the wrong kernel. **Measure the real run, compare each kernel to its own
roofline, and rank by recoverable time.** This tier turns the analytical ranking into
a measured, defensible hotlist.

## Pipeline position
`model-op-decomposition → fusion-analysis → model-roofline-analysis →
model-profile-hotspots → coverage-gate → (wire | kernel-feasibility-gate →
kernel-authoring)`. This is the bridge between the analytical model tier (WHAT should
dominate) and the kernel tier (HOW/whether): it confirms with real numbers WHERE time
actually goes and which kernels have room to improve.

## Why both roofline tiers exist
- `model-roofline-analysis` = **analytical, top-down**. Cheap, needs no run, gives the
  expected ranking and levers. Can be wrong.
- `model-profile-hotspots` = **empirical, measured**. Needs a run, gives ground truth
  for time distribution and per-kernel efficiency. Discrepancies vs the analytical
  ranking are exactly the surprises worth investigating.
Use the analytical tier to decide the run is worth doing and to interpret the profile;
use this tier to commit budget to specific kernels.

## Inputs
The ranked plan from `model-roofline-analysis`; the wired (or partially wired) model
runnable on the target node; per-op-class roofline floors (FLOPs/bytes → time on the
node's achievable ceilings, from `establish-achievable-performance` +
`roofline-validation`); the workload point(s) to profile (prefill and decode
separately; representative batch / seq-len / spec-M).

## Step 0 (do FIRST): rule out a SYSTEMIC-config pathology before per-op ranking
A single systemic runtime-config mis-setting inflates **every** op roughly uniformly — so if
you rank per-op hotspots first you optimize on inflated numbers and chase the wrong ops. **The
tell is uniform slowness: the whole run is far below roofline and no single op stands out after
normalizing.** Before any per-op RoI work, do a cheap systemic sweep (see `runtime-config-tuning`):
thread count/cap (esp. the **decode M=1 thread cliff** — capping the decode forward to
`domain_cores − 2` gave a **~200× decode win** here; the full bound starves the framework thread),
NUMA/membind, prepack, ISA dispatch. One decode-thread-cap sweep reranked the entire hotlist and
dwarfed every per-op micro-optimization we had tried first (gather vectorization, einsum reforms).
Only once the systemic config is fixed is the per-op profile trustworthy. **Regret from this repo:
we spent multiple fix cycles on per-op micro-opts before the systemic thread sweep that made them
irrelevant — do the systemic sweep FIRST.**

## Procedure
1. **Run the real model, random weights OK.** Accuracy is irrelevant here — only
   timing. Launch with dummy/random weights so no checkpoint is needed
   (SGLang `--load-format dummy`). **If the full model is slow to load/run, do this on a
   `perf-proxy` FIRST** — a depth-reduced (few-layer) but FULL-WIDTH, real-config copy that
   reproduces every per-op bottleneck queue-free in ~1/N the time; iterate fixes there and
   confirm ONCE on the full model. (Shrink DEPTH, never WIDTH — a narrow config mis-ranks.)
   Warm up, then profile a steady-state window.
   Profile **prefill and decode separately** (the hot kernels differ per phase).
2. **Collect a per-kernel wall-time profile** on the node:
   - **Run `overhead-attribution` FIRST and tag every hot op `kernel|torch|framework`**
     — the tp-safe, per-rank, DCE-guarded boundary timers are the primary mechanism
     (worked example: `INTEL_CPU_DSV4_TIMEIT=1`, which prints the kernel/torch/framework
     split); a sampler collapses under tp×OMP threads. The profiler below explains the
     untimed FRAMEWORK residue (copies/casts/dispatch).
   - Torch/SGLang profiler trace (per-op self time), and/or `ONEDNN_VERBOSE=1` for
     per-primitive oneDNN timings (the AMX/BRGEMM GEMMs), and/or `perf record`.
   - **Rank by WALL time, not `cProfile` self-time.** cProfile adds ~µs of per-Python-call
     overhead, so it massively OVER-WEIGHTS Python-call-heavy glue (list-comps, per-token
     loops, dict gathers): here a Python gather loop showed **46 s of cProfile "self"** that
     was ~0 in wall-time (the vectorized fix changed decode by nothing). Use the env-gated
     wall timers (`INTEL_CPU_DSV4_TIMEIT`) to rank; use cProfile only to *confirm* C-kernel
     (AMX) costs, never to rank Python-glue ops.
   - Aggregate self-time by kernel and attribute to op-classes (attention, MoE
     grouped-GEMM, dense proj, norm/rope/act, embedding/lm_head, indexer).
3. **Measured Amdahl share.** For each kernel/op-class: `measured_share =
   self_time / phase_time`. This REPLACES the analytical share — reconcile against
   `model-roofline-analysis`; flag any op-class whose measured share differs
   materially from the analytical prediction (a modeling gap to explain).
4. **Per-kernel efficiency vs roofline.** For each hot kernel compute its roofline
   floor (min over compute-bound and memory-bound times on the node's achievable
   ceilings for the ACTUAL dtype/ISA). `efficiency = roofline_floor / measured_time`
   (∈(0,1]). High efficiency ⇒ near ceiling ⇒ little to gain even if it's a big share.
5. **RoI = recoverable end-to-end fraction.**
   `roi = measured_share × (1 − efficiency)` = the fraction of phase time you could
   recover if this kernel hit its roofline. Rank kernels by `roi` descending.
6. **Classify each hot kernel:**
   - Route by the `overhead-attribution` bucket FIRST: **torch** (unoptimized) → author/route
     a kernel or co-design; **kernel far below its floor** → kernel-isolation (config: threads,
     prepack, dtype path, verify ISA) NOT a new kernel; **kernel at floor** → accept; **framework**
     → fusion / fewer ops / bf16-end-to-end, no kernel. Then within that:
   - big share **and** low efficiency → **top RoI**, descend to `kernel-feasibility-gate`.
   - big share **and** high efficiency → already near ceiling; the only lever left is
     a *different roofline* (e.g. lower precision to cut the memory-bound floor) — note it.
   - small share → **SKIP** regardless of efficiency (Amdahl caps the payoff).
7. **Present the measured hotlist to the user** (table: phase, kernel/op-class,
   measured_share%, measured time, roofline floor, efficiency, roi, covered/new,
   action). This, not the analytical table alone, is what commits kernel budget.

## Output — measured RoI-ranked hotlist
`[{phase, kernel, op_class, measured_share_pct, measured_ms, roofline_floor_ms,
efficiency, roi, mapping: donor|new-kernel|config, action: author|tune|skip}]`.
Only `author`/`tune` items with meaningful `roi` descend to `kernel-feasibility-gate`.

**Publish it.** Emit the same data as a roofline-TARGET-vs-MEASURED artifact (model
level + per op) via `plugin/validate/roofline_vs_measured.py` — a table + bar chart
that every published result links (see `enablement-certificate` §9). Publish the
target up front; measured (usually below) fills in, and the per-op gap ranked by
shortfall×share is the headline "which kernels underperform vs roofline" view.
**Roofline and measured MUST be the SAME machine config** (socket count, TP, batch,
precision) and labeled as such (single socket vs full 2-socket node) — profile at the
same point the roofline was computed for, never mix socket counts.

## Gate
No kernel enters `kernel-feasibility-gate` unless this tier measured it as a high-RoI
hotspot (meaningful share AND headroom below its roofline). If the profile contradicts
the analytical `model-roofline-analysis` ranking, the measured result wins — and the
discrepancy is recorded (it usually points at a real effect the analytical model
missed, like the DSA issue/conversion bound).

## Notes / gotchas
- Random weights change values, not shapes or dtypes — timing of dense/quant kernels
  is representative. Data-dependent control flow (MoE routing, sparse/topk selection)
  can differ from real weights; for those, note it and, if it matters, profile with a
  realistic routing distribution rather than pure-random logits.
- Profile on the TARGET node (same ISA/clock/BW) — see `establish-achievable-performance`.
- Disable caches that fabricate speedups (e.g. radix/KV reuse across identical prompts)
  so the decode profile reflects real per-token work.
- `efficiency` uses the *achievable* ceiling, not the marketing peak — otherwise every
  kernel looks like it has headroom.
- **Shape-correct stubs unblock the profile.** If make-it-work is blocked on an
  un-ported authoring-gap kernel, stub it to return correctly-shaped tensors so the
  full model runs and every OTHER op gets measured. The stubbed op shows as ~0 time
  (flag it as "not yet implemented", not "free"); everything else is real signal that
  ranks where to spend kernel effort first.
- **A stub/bypass only works if a NON-TARGET path exists to fall back to.** Before
  planning "bypass feature X to profile the rest", verify the code has a dense/fallback
  path when X is disabled. A backend or module that IS the novel feature has nothing to
  fall back to — e.g. DeepSeek-V4's dsv4 attention backend has NO dense MLA path; DSA
  sparse attention is its forward, so skipping the compressor just starves the sparse
  core. In that case there is no profile-first shortcut: the feature must be authored to
  run at all. Check for the fallback path during `enablement-scope-discovery`.
