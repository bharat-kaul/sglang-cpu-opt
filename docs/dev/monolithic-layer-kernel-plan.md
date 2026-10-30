# Monolithic per-layer fused decode kernel — scoped build plan (branch: dsv4-fused-layer-decode)

**Goal.** Recover the measured decode **serial/fork-join overhead** by collapsing the per-layer op sequence into
**one cooperative C++ parallel region per layer** (no per-op fork-join barrier, no OMP park-wake between ops).

## ⚡ MEASURED JUSTIFICATION (perf profiling, jobs 394960/394972 — supersedes the old region-timer "29%")
A `perf` self-time + LBR call-graph profile of **real B=32 warm decode** (32 threads, symbol-bearing build) shows:
- **~60% of the decode step = OpenMP barrier spin** (`gomp_barrier_wait_end` 33.8% + `gomp_team_barrier_wait_end`
  29.3%), reached from `start_thread`/`clone3` **directly** = worker threads **idle at the team barrier** waiting
  for the main thread to launch the next parallel region.
- **Only ~9–13% = actual MoE kernel compute** (`tinygemm`/`parallel_2d`); the MoE's OWN region barrier is ~4%.
- **99% is the in-process forward** (`sglang::schedul`); framework detok/scheduler = 0.03%, **Python dispatch =
  0.87%** → the 60% is **C++ fork-join across hundreds of small per-op parallel regions**, NOT Python, NOT a
  single op, NOT the MoE kernel.
- **Prize:** decode = **7.8× above its MoE weight-streaming floor** (716 ms vs 92 ms @ 329 GB/s). The serial 60%
  is the gap. Removing it (workers get work during the serial sections) → realistic **~2–4× on decode**. This is
  the single biggest remaining decode lever.
- **Why warm didn't already get it:** warm (spin) made each wait *cheaper* (no park/wake syscall) = the 2.2×; the
  structural **waits remain** (60% of the now-faster step). Only region-collapse removes the waits themselves.
- **Reconciles the old region-timer "MoE 72% / glue 29%"** (L.7): most of that "MoE wall" is idle-barrier spin
  during the call's serial setup + surrounding per-op dispatch, which a region timer cannot see.

**PROFILING-REFINED PRIORITY (changes the plan below):** the win is the **layer-level region collapse** (one
cooperative region per layer). The old step-1 "remove Python dispatch" is now known **low-value** (0.87%). The
hard+valuable part is making the layer ops cooperate in ONE region (ATen serializes inside a naive nested region
— so the ops must be reimplemented region-cooperatively; this is the multi-week core).

## What is established (do not re-litigate)
- **The overhead is OVERHEAD, not compute** — now at ~60% (measured by perf, supersedes the 29% region-timer
  estimate). Monolithic per-layer region is the right fix. CONFIRMED by profiling.
- **Scope = "level-1" monolithic:** the big kernels (MoE/attn) stay their own calls; we collapse the GLUE +
  **NOT** reimplement MoE/attention — we fuse the elementwise glue + Python dispatch *around* the existing
  kernels (inter-kernel-fusion fused prologue/epilogue). This is the tractable, 29%-capturing level.
- **Small per-op swaps are ~0% e2e** (increment-1 MHC-wire: correct but +0.1%). Only the structural
  region-collapse moves decode. (L.11)

## The non-negotiable safety net (build FIRST)
**Per-layer bit-faithful parity harness** — reuse `INTEL_CPU_DSV4_FULLCAP` (captures each layer's output
hidden + attn/MoE outputs to `FULLCAP_FILE`) + `plugin/validate/diff_fullcap.py`:
1. Freeze a prompt; run the **reference** path → `fullcap_ref.pt`.
2. Run the **fused** path (env-gated) → `fullcap_fused.pt`.
3. `diff_fullcap.py` → per-layer cos + max-rel-err; **gate: cos ≥ 0.9995 every layer, no divergence cliff**
   (the §5 boundary-capture method; same gate that localized the `wkv_gate` bug).
Every increment below passes this gate before any perf number is quoted, and ships behind an env flag with an
**eager fallback** so main is never at risk.

## Incremental build (each: parity-gate → interleaved-A/B → keep only if correct AND e2e-positive)
1. **C++ per-layer loop driver (dispatch removal).** Replace the Python `for layer in layers` + per-op torch
   dispatch with a single C++ entry that calls the layer ops in sequence. Captures the Python-dispatch slice
   of the 29% without touching kernel math. *Risk: low (pure control-flow port). Parity: trivially exact.*
2. **Fused elementwise prologue/epilogue.** Fold `input_layernorm`, `post_attention_layernorm`, the residual
   adds, and rope application into the prologue/epilogue of the adjacent kernel calls (one region each),
   removing their standalone parallel regions + the intermediate materialization. *Risk: med. Parity: cos≈1.*
3. **Fused MHC boundary.** Collapse `hc_post + hc_pre + norms` (currently eager fp32: upcast + rmsnorm +
   `F.linear(hc_fn)` + sinkhorn + combine) into one C++ region, reference = the eager path (oracle). This is
   the biggest single eager block (~12%). *Risk: med-high (fp32→bf16 precision care). Parity-gated + fallback.*
4. **Single shared parallel region per layer.** Wrap steps 1–3's ops in ONE `#pragma omp parallel` with
   in-region barriers (cheap) instead of per-op fork-join. The big kernels (MoE/attn) stay their own calls but
   the GLUE around them is one region. *Risk: med. Measures the full 29% recovery.*

## Measurement (per increment)
- Correctness: the FULLCAP per-layer diff (gate above) + a greedy per-token parity on a real prompt.
- Perf: drift-robust **interleaved A/B** (one resident load, toggle the fused path per rep, paired median) —
  region-timers only nominate; the A/B proves (L.7). Single socket, physical, the decode thread optimum.
- Final: a sharded gsm8k run to confirm task accuracy is unchanged before merge to the pilot branch.

## Status
- [x] Analysis, de-risk, scoping (this doc), branch, parity infra identified.
- [x] Increment-0 (MHC sinkhorn/combine C-wire): correct, +0.1% (kept as dispatch hygiene; not an e2e lever).
- [ ] Parity harness wired as a one-command gate.
- [ ] Increments 1–4 (above), each parity-gated + A/B-measured.
- [ ] Sharded gsm8k accuracy confirm → merge.

*This is a multi-stage authoring effort; the structural region-collapse (steps 1–4) is where the 29% lives.*
