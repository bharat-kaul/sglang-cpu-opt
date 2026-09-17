---
name: runtime-config-tuning
description: "The FRAMEWORK/CONFIG lever, separate from kernel authoring: make already-optimal kernels actually run at their roofline by fixing thread count + affinity, NUMA/SNC binding, OMP/env, TP-rank-to-domain mapping, weight prepack, and dtype/ISA dispatch — the settings that decide whether a kernel gets its cores and bandwidth. Use when overhead-attribution tags an op 'kernel far below its roofline floor' (isolated-fast but slow end-to-end), when scaling to tp>1, or when a whole run is uniformly slow (a config problem inflates everything). Cheap, reversible, and usually the highest-leverage first fix — no code. Reads uPP machine_constants.json for the core/BW budget; feeds roofline-validation."
---

# Runtime Config Tuning (make optimal kernels reach their roofline)

A certified-optimal kernel still misses its roofline if the runtime starves it of
cores or bandwidth. This is the **framework/config** bucket from `overhead-attribution`:
the kernel is fine, the *deployment* is wrong. It is the first thing to fix because it
is cheap, reversible, requires no code, and a single mis-setting inflates the WHOLE run.

**Origin lesson (this repo):** DeepSeek-V4-Flash MoE `fused_experts` measured ~1920 ms/call
while its isolated GEMM floor was ~0.33 ms and its fp8 weight-stream floor ~0.67 ms —
~1000–2000× above roofline. The kernel was not the problem; the tp=4 run set no
`OMP_NUM_THREADS` / `SGLANG_CPU_OMP_THREADS_BIND`, so each rank was thread-starved and
contended. No kernel work could have fixed that.

## The knobs (in leverage order), each vs a uPP-measured budget
1. **Thread count + affinity.** Each TP rank must get a disjoint, NUMA-local core set.
   Set `OMP_NUM_THREADS` = cores-per-domain and bind (`SGLANG_CPU_OMP_THREADS_BIND`,
   `numactl --cpunodebind`/`--physcpubind`, `GOMP/KMP_AFFINITY`). Check against uPP
   `threading.cores_to_saturate_bw` and `meta.cpu_count / n_domains`. Symptom of failure:
   throughput flat vs threads, or all ranks on the same cores.
2. **NUMA / SNC binding + first-touch.** One TP rank per SNC domain, membind local; weights
   first-touched on the owning domain. Remote access costs the uPP `remote_bw_penalty`
   (~0.6 on GNR). Aggregate BW = `domains_used × per_domain_bw` only if sharded NUMA-local.
3. **TP-rank ↔ domain mapping.** #ranks should divide the domain count and shard heads/experts
   evenly; a bad map leaves domains idle (see `sub-numa-clustering`).
4. **Weight prepack / layout.** Confirm the kernel's VNNI/packed layout is actually applied
   (`is_vnni=True` reaching a *packed* weight), else it falls off the fast path.
5. **Dtype / ISA dispatch.** Verify the intended ISA fired (`ONEDNN_VERBOSE=1`, kernel
   throughput magnitude) — capability flags false-negative on GNR. A silent fp8→scalar or
   AVX-512 fallback looks like a "slow kernel".
6. **Batching / grouping.** Grouped-MoE and gather kernels should process only the routed
   experts/rows, not the full set — confirm the op isn't streaming everything.

## Procedure
1. Get the op's isolated roofline floor (kernel-isolation) and the uPP core/BW budget.
2. Read the ACTUAL runtime state — threads, affinity, membind, packed-ness, dispatched ISA
   (a one-time diagnostic log of `torch.get_num_threads()` + weight shape/dtype at the first
   call is enough; worked example: the `[MOE DIAG]` line under `INTEL_CPU_DSV4_TIMEIT=1`).
3. Change ONE knob at a time; re-measure vs the floor; keep what moves it.
4. Re-run `roofline-validation`; if the op now sits at its floor, the config was the lever.

## Boundary with the other skills
- `overhead-attribution` DECIDES this is the lever (kernel-tagged, far from floor, isolated-fast).
- `sub-numa-clustering` owns the SNC math (domains, effective tp, capacity/divisibility).
- `openmp-parallelization` tunes intra-kernel threading of code you OWN; this skill tunes the
  runtime around kernels you may NOT own (framework/library), which is the common day-0 case.
- Never reach for kernel authoring while a config knob is still wrong — it wastes budget and
  hides the real cause.
