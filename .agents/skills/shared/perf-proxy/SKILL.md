---
name: perf-proxy
description: "Overcome the long-run structural bottleneck: build a DEPTH-reduced, FULL-WIDTH, real-config proxy of the target model and do performance bottleneck-hunting + fix iteration on it (queue-free, ~1/N the time) BEFORE any full-model run. Use EARLY — before model-profile-hotspots and the optimization loop — whenever the real model takes many minutes to load and runs slowly. Accuracy is ignored (dummy weights). The proxy shrinks num_hidden_layers to the minimum set that covers all op families, but keeps every per-layer WIDTH (hidden, experts, intermediate, heads) and the RUNTIME config (tp, threads, NUMA, dtype) REAL — because per-op bottlenecks (framework/kernel, grain-size) are set by width+config, not depth. Confirm depth-aggregate effects + absolute tok/s on the full model at the end."
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

## Where it sits in the workflow (EARLY)
Run the proxy BEFORE the expensive full-model empirical tier:
`… → model-roofline-analysis → **perf-proxy (build + iterate here)** → model-profile-hotspots
(on the proxy) → fixes → confirm ONCE on the full model`.
- Pairs with `high-information-runs`: the proxy makes each run CHEAP; high-information-runs makes
  each run ANSWER MORE → fewer × cheaper runs.
- The overhead/isolation probes (`overhead-attribution`, kernel-isolation) run ON the proxy.
- Final `enablement-certificate` perf number + accuracy come from the FULL model.
