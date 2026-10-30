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

## ⚡ THESIS CONFIRMED + CEILING MEASURED (POC job 395025)
A counterfactual microbench — 43 layers × N small ops, **(A) per-op fork-join** vs **(B) one cooperative
`omp parallel` region per layer** (warm, 32/48 threads) — directly measures the recoverable overhead:
| ops/layer | A per-op (ms) | B one-region (ms) | region-collapse speedup |
|---|---|---|---|
| 12 | 5.11 | 1.57 | **3.26×** |
| 24 (~DSV4) | 10.04 | 2.43 | **4.14×** |
| 40 | 16.45 | 3.79 | **4.34×** |
Region-collapse recovers **~4×** of the fork-join overhead portion. Applied to decode (716 ms = ~286 ms real
parallel work + ~430 ms fork-join overhead): monolith → **~390 ms = ~1.8× e2e**, ceiling (near-full overhead
removal) **~286 ms = ~2.5× e2e** on decode — on top of the banked warm config. This is the measured prize and it
justifies the multi-week build.

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

## Increment 4 result (job 395049) — wire/parity/measure loop PROVEN; standalone fusion is neutral/negative
- **In-model parity PASS** (COOP_MHC=1 + CHECK=1, no assert, no fallback): the fused coop_hc_combine_rmsnorm is
  bit-faithful on the live decode path. The wire -> in-model-parity -> measure loop (with eager fallback) works.
- **A/B:** B=1 184.8->183.1 ms (~flat); B=32 704.9->**751.5 ms (~6.6% SLOWER)**.
- **Why (two real lessons):**
  1. **No fork-join saved yet** — without the layer driver each hc_pre runs STANDALONE, so coop_parallel_for
     still forks its own region. The saving only appears INSIDE one layer region (the driver, not yet built).
  2. **The demonstrator kernels are naive scalar fp32**, vs the AVX-512-vectorized originals they replace
     (hc_combine einsum + production rmsnorm_cpu). Per-element slower -> shows at B=32; plus .contiguous() copies.
- **Implication for the build:** the monolith win requires BOTH (a) the C++ per-layer DRIVER (one region/layer,
  to actually remove the fork-joins) AND (b) the cooperative kernels to be AS FAST as the originals (AVX-512,
  no extra copies). Individual gated fusions are correctness bricks, not wins (confirms L.11). Next: either
  vectorize the coop kernels to parity-speed first, or build the driver and accept temporary per-kernel slowness
  until enough ops are in one region that the removed fork-joins dominate. Keep COOP_MHC OFF by default.

## Vectorization + C++ driver validated (jobs 395073) — mechanism proven on REAL kernel work
- **Coop kernel vectorized (AVX-512):** coop_hc_combine_rmsnorm_f32 is bit-faithful (cos=0.9999967) AND 8x
  faster than the torch reference (30.9 vs 251 us/call @ B=32). Fixes the inc-4 regression; wired kernel is now
  parity-speed.
- **C++ layer-driver demonstrator** — real combine+rmsnorm as K ops/layer x 43 layers, per-op fork-join (A) vs
  one cooperative region/layer (B), warm 32 threads:
  | M (batch) | K=12 | K=24 |
  |---|---|---|
  | 1  | 3.66x | 4.38x |
  | 32 | 2.51x | 1.61x |
  | 64 | 2.29x | 2.23x |
  The driver delivers 1.6-4.4x on the glue, largest at small M (decode). Validates the mechanism on REAL work
  (vs the toy POC). Speedup shrinks as work/row grows (fork-join a smaller fraction), consistent with the model.
- **REMAINING (the sustained core):** convert the actual DSV4 layer ops (MLA attention, DSA compressor/indexer/
  topk/attend, MHC sinkhorn, MoE gate/experts) from at::parallel_for to region-cooperative (coop_parallel_for)
  form, and a C++ per-layer driver that opens ONE region and sequences them, wired into the model forward +
  per-layer parity gate. The mechanism is proven; this is the multi-week op-by-op conversion + wiring.

## Region-aware primitives landed — the high-leverage enablers (no e2e yet; mechanism for the driver)
- **parallel_2d → region-aware** (common.h): nested => cooperate on the driver's team (no new fork); standalone
  => fork as before. Makes EVERY parallel_2d caller (MoE fused_experts + more) cooperative with ZERO per-kernel
  rewrites. Verified BIT-IDENTICAL MoE output vs the unmodified build (max diff 0.0) — provably safe.
- **coop_at_parallel_for** (coop.h): region-aware drop-in for at::parallel_for -> converting the ~21 files that
  use it is now a MECHANICAL find-replace. This is the safe "parallel wave" recipe: each file independent, only
  coop.h + the registration file are coordination points.
- **STATUS:** mechanism ~complete (coop_parallel_for, coop_at_parallel_for, region-aware parallel_2d, coop_region,
  coop_barrier). Remaining = (1) mechanical find-replace at::parallel_for -> coop_at_parallel_for in the layer's
  hot kernels (MoE stages 1.5/3 silu+reduce, attention, DSA, norms) + a per-kernel SCRATCH/race audit under the
  shared team; (2) the C++ per-layer DRIVER that opens one coop_region and sequences the (now-cooperative)
  kernels; (3) wire into the forward + per-layer parity gate + e2e A/B. No e2e movement until a contiguous
  critical mass is in one region (L.11). The hard de-risking (does the mechanism work + stay bit-exact) is DONE.

## First POSITIVE in-model e2e signal (job 395074) — vectorized MHC coop kernels wired (both boundary ops)
With coop_hc_post + coop_hc_combine_rmsnorm (now AVX-512, 8x/56x faster than the torch glue) wired under
COOP_MHC=1, decode A/B (within-job, back-to-back, 4 reps median, warm 32thr):
  B=1 : 184.9 -> 184.7 ms  (flat; MHC a smaller fraction at B=1)
  B=32: 704.5 -> **685.6 ms = +2.7% faster**
This FLIPS the inc-4 scalar result (-6.6%) -> vectorization was the fix, as predicted. Mechanism: the ~60%
barrier spin is workers waiting on the MAIN-THREAD serial glue; replacing slow torch glue (hc_post 1696us,
combine+norm 251us) with fast kernels shortens that serial stretch -> less barrier wait, even WITHOUT the
region-collapse driver. No fallbacks (coop kernels ran in-model). CAVEATS: +2.7% is modest and near the ~5%
cross-job noise floor (the WITHIN-job back-to-back delta is the trustworthy part; baseline reproduced 704.9
across jobs); only 2 of ~24 ops wired; the CHECK-leg in-model parity log was inconclusive (harness, not a
fallback) — isolated parity is bit-exact (cos>=0.9999) + inc-4 in-model CHECK passed on identical math; re-confirm
cleanly before default-on. COOP_MHC stays OFF by default. Signal: faster-glue + region-collapse are ADDITIVE
levers on the same barrier; this banks the faster-glue portion.

## OVERALL e2e wall-time banked (jobs 395077/395078) — naive -> fully-optimized, single EMR socket
decode_step (ms) and prefill (s), naive (cold/8thr/no-coop) vs optimized (warm/M-adaptive-threads/coop-MHC):
  B=1 : decode 612 -> 173 ms = 3.54x   | prefill 1.71 -> 0.38 s = 4.5x
  B=32: decode 3224 -> 725 ms = 4.45x  | prefill 6.75 -> 2.02 s = 3.3x
  B=64: decode 4290 -> 1167 ms = 3.68x | prefill 10.31 -> 3.36 s = 3.1x
=> ~3.5-4.5x decode AND ~3-4.5x prefill, banked and validated, from CONFIG (warm pool + M-adaptive threads) +
the coop-MHC glue fusion. Prefill ALSO improved (the warm/thread config helps both phases; prefill was never the
monolith target). This is the total wall-time win from the whole campaign. The remaining monolith ceiling
(~1.8-2.5x MORE on decode) requires the Python->C++ layer-forward port (multi-week); mechanism fully de-risked.
