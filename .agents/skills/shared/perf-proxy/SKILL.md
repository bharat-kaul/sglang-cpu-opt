---
name: perf-proxy
description: "Overcome the long-run structural bottleneck: build a DEPTH-reduced, FULL-WIDTH, real-config proxy of the target model and do performance bottleneck-hunting + fix iteration on it (queue-free, ~1/N the time) BEFORE any full-model run. Use EARLY — before model-profile-hotspots and the optimization loop — whenever the real model takes many minutes to load and runs slowly. Accuracy is ignored (dummy weights). The proxy shrinks num_hidden_layers to the minimum set that covers all op families, but keeps every per-layer WIDTH (hidden, experts, intermediate, heads) and the RUNTIME config (tp, threads, NUMA, dtype) REAL — because per-op bottlenecks (framework/kernel, grain-size) are set by width+config, not depth. Progress up a DUMMY-weight ladder — truncated-dummy (iterate) → FULL-dummy (authoritative perf-vs-roofline on the target node, before any real-weight load) → full-weight (ONLY for accuracy + a final no-perf-regression confirm). Dummy is valid for perf (real shapes/dtypes/byte-streaming, random values); never pay the full-weight load to answer a PERF question."
---

# Perf Proxy (depth-reduced, full-width — fast, faithful bottleneck iteration)

The structural tax on optimization is the RUN: real weights load (many min) + slow decode +
serial cluster queue. But **performance bottlenecks live in the per-op regime, which is set by
per-layer WIDTH (hidden, experts, intermediate, heads) and the RUNTIME config (tp, threads,
NUMA, dtype) — NOT by the number of layers.** Layers only *multiply* per-layer cost. So a model
with the real width but only a few layers reproduces every per-op bottleneck faithfully while
loading and running ~`(full_layers / proxy_layers)`× faster — and small enough to skip the queue.

## The recipe
1. Copy the real config; set `num_hidden_layers` to the **minimum that covers all op families**
   (attention variants, MoE vs dense-MLP, any per-layer switches). Keep EVERY width dim real.
2. `--load-format dummy` (accuracy irrelevant — random weights, same code paths + shapes).
3. Run with the **real runtime config** you want to characterize (tp, thread count, NUMA bind,
   dtype). For pathologies that depend on per-rank threads, match the real tp; for pure
   bottleneck *discovery*, any faithful-width config exposes them.
4. Iterate fixes here (queue-free, minutes). Extrapolate a per-layer-dominated cost to the full
   model by `× (full_layers / proxy_layers)`; CONFIRM on the full model at the end.

## CRITICAL: shrink DEPTH, not WIDTH (the opposite of the correctness tiny config)
`cpu-serving-integration`'s *tiny architecturally-faithful* config shrinks WIDTH for **wiring
correctness** (runs in seconds, exercises the arch switches). That config **MISLEADS
performance** — small width changes the per-op regime (cache-fit, different thread optimum,
dispatch-bound). This repo learned it the hard way: a `hidden=512, 8-expert` tiny config gave
the WRONG perf conclusion ~4× in a row. The perf proxy keeps width REAL and shrinks only depth.

## CRITICAL: run on the TARGET ISA host (AMX) when at all possible
The proxy is DEPTH-reduced, so it is small and fast **on any host** — which tempts you onto a
queue-free login/dev node. But if that node lacks the target ISA (e.g. no AMX), every
kernel-bound op (dense GEMM, MoE, attention matmul) runs a **slow scalar/vector fallback** and
its cost is **non-representative**. Two concrete failures this causes:
- the **absolute** decode/prefill number is dominated by fallback junk (measured: the 4-layer
  Flash proxy on a no-AMX login node spent ~50% of decode in fallback dense/MoE kernels that are
  cheap AMX weight-streams on the real node);
- you go **blind to M=1 kernel pathologies on the real node** — the exact bug class already found
  here (`fused_experts` inverse thread-scaling, ~1000× above floor). You cannot find a dense-GEMM
  M=1 threading pathology on a host that never runs the AMX kernel.

Rule: **run the proxy on a node with the target ISA.** The proxy is small, so even a queued
target node (e.g. an idle GNR partition) usually costs only minutes — check `sinfo`/`squeue`
first; an idle target node beats a mis-representative login node every time. Only fall back to a
non-ISA host for **device-independent** iteration (torch/Python framework overhead: DSA/MHC
fallbacks, cast churn, Python-loop gather, metadata rebuild) — and then NEVER trust its
kernel-op ranking or its absolute tok/s; treat only the torch/Python-overhead ops as real.

## Validated (this repo)
DeepSeek-V4-Flash 4-layer proxy on the login node (no queue):
- reproduced the **real** MoE shape `w13=(256, 4096, 4096)` (tiny had `(8,512,512)`),
- reproduced the **inverse thread-scaling** pathology at real shape (`300 ms@8thr` vs
  `3853 ms@60thr` vs `8124 ms@120thr`),
- decode `30.5 s/token × (43/4) ≈ 328 s ≈` the real full-model `317 s/token` — **predictive**,
- ran in **~8.5 min, queue-free**, vs ~65 min queued for the full 43-layer model.

## What it does NOT capture (confirm on the full model at the end)
- **Depth-aggregate effects:** total weight-streaming bandwidth pressure (the memory wall is
  depth×width), cross-layer cache eviction, inter-layer pipeline/overlap, KV-cache growth.
- **Absolute end-to-end tok/s** — extrapolate `× depth-ratio` but VALIDATE once on full depth.
- **Accuracy** — by design (dummy weights); accuracy uses the real-weight full model separately.
- **Tensor-parallel / multi-rank UPSIDE** — the proxy validates TP *viability* (does the plugin run at
  tp>1 without hang/crash — the biggest risk), *divisibility* (sglang asserts `dim % tp == 0`; a 2-min
  fail-fast rules out bad tp values — e.g. DSV4's 2^15 dims reject tp=6), and *correctness* (tp=1 vs
  tpN tokens), ALL cheaply on the free farm. But it CANNOT show the TP *speedup*: on a depth-reduced
  model the weight-stream is negligible, so the per-layer all-reduce comm tax DOMINATES and tp>1 reads
  SLOWER than tp=1 (measured: 4-layer tp=2 decode 65 vs tp=1 85). TP's benefit is capacity (fit) +
  full-depth weight-stream sharding = depth-aggregate; only the full-DUMMY rung (on the target node)
  shows it. So use the proxy to DE-RISK TP (viability/divisibility/correctness) before the one scarce-
  node run, but never to decide TP upside.

## The validation ladder — DUMMY for perf, FULL WEIGHTS only for accuracy (run in this order)
Three dummy rungs, cheapest first (plus a config-sweep rung 0 that precedes them all); each
DE-RISKS the next. Dummy weights are VALID for perf (shapes, dtypes, byte-streaming, and kernel
code paths are all real — only the VALUES are random) and INVALID for accuracy. So run the ENTIRE
perf/roofline campaign on dummy and pay the real-weight load only at the very end, for accuracy +
a final confirm. **Never pay the full-weight load to answer a PERF question.**
0. **Systemic-config sweep FIRST — before ANY per-op ranking, on any rung.** A single systemic
   mis-setting (thread count/cap + the decode M=1 thread cliff, **OpenMP spin-wait**
   `OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0`, NUMA/membind, prepack, ISA dispatch) inflates *every*
   op roughly uniformly, so ranking hotspots first optimizes inflated numbers and chases the wrong
   ops. Sweep it before trusting a profile — here it was worth **63× prefill / 10× decode** on its
   own, bigger than every per-op tweak combined. See `runtime-config-tuning` + `model-profile-hotspots`
   Step 0. (This rung is cheap and runs on the truncated proxy too — do it there first.)
1. **Truncated depth + DUMMY** (the proxy above). Bottleneck-hunting, per-op profiling, config/
   thread tuning, opt A/B. Queue-free, ~1/N time — most iteration lives here.
2. **FULL depth + DUMMY** (run BEFORE any real-weight load). Real full-model shapes and full
   resident footprint, still no checkpoint read. Purpose:
   - the AUTHORITATIVE perf-vs-roofline number, on the TARGET perf node (e.g. GNR, higher BW),
     at realistic serving batch — the real mission metric, minus accuracy;
   - confirm the truncated→full extrapolation actually holds — catches the depth-aggregate
     effects the proxy cannot (total BW/memory-wall pressure, full resident fit vs ONE NUMA
     domain, allocator/NUMA at full size, KV growth);
   - DE-RISK the expensive final run: if full-dummy exposes a perf or capacity problem (overflows
     one domain, decode falls off roofline at full depth), fix it BEFORE paying the load.
   Load stays cheap (dummy skips the checkpoint read); only the resident allocation is full-size.
3. **FULL depth + FULL weights** (final — once rung 2 perf is satisfactory and the accuracy gate
   is ready). The ONLY rung needing the real checkpoint, and only for: (a) ACCURACY (task / per-
   layer parity — `accuracy-oracle`); (b) a final confirm that real values don't change perf
   (they must not — identical shapes/dtypes/code paths; only load time differs, so a perf delta
   here is a BUG). Expensive load → done LAST, ideally once.

## Where it sits in the workflow (EARLY)
Run the proxy BEFORE the expensive full-model empirical tier:
`… → model-roofline-analysis → **perf-proxy (build + iterate here)** → model-profile-hotspots
(on the proxy) → fixes → **full-dummy perf-vs-roofline (rung 2, target node)** → full-weight
accuracy + perf-confirm (rung 3)`.
- Pairs with `high-information-runs`: the proxy makes each run CHEAP; high-information-runs makes
  each run ANSWER MORE → fewer × cheaper runs.
- The overhead/isolation probes (`overhead-attribution`, kernel-isolation) run ON the proxy.
- Final `enablement-certificate` perf number + accuracy come from the FULL model.
