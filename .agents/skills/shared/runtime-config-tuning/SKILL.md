---
name: runtime-config-tuning
description: "The FRAMEWORK/CONFIG lever, separate from kernel authoring: make already-optimal kernels actually run at their roofline by fixing thread count + affinity, OpenMP wait policy (idle-thread spin vs park — OMP_WAIT_POLICY/KMP_BLOCKTIME, a frequent root cause of a whole forward being uniformly slow or the 2nd forward collapsing), NUMA/SNC binding, OMP/env, TP-rank-to-domain mapping, weight prepack, and dtype/ISA dispatch — the settings that decide whether a kernel gets its cores and bandwidth. Use when overhead-attribution tags an op 'kernel far below its roofline floor' (isolated-fast but slow end-to-end), when scaling to tp>1, or when a whole run is uniformly slow (a config problem inflates everything). Cheap, reversible, and usually the highest-leverage first fix — no code. Reads uPP machine_constants.json for the core/BW budget; feeds roofline-validation."
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

**Origin lesson 2 — the DECODE (M=1) thread cliff (this repo):** at decode the batch is 1
token, so every GEMM/op is trivially small and the cost is the parallel-for *barrier*, not
compute. Running the whole decode forward at the framework's default thread count (the full
NUMA-domain core count, e.g. 42) was catastrophic; a decode-wide cap swept sharply:

| threads | 8 | 16 | 24 | 32 | **40 (bound−2)** | 42 (bound) |
|---|---|---|---|---|---|---|
| decode s/tok | 0.563 | 0.429 | 0.306 | 0.182 | **0.062** | **12.8** |

Monotonic improvement up to `bound−2`, then a **~200× cliff at the full bound**: using *all*
domain cores leaves none for the main/framework/OS thread, so the barrier stalls on
descheduled worker threads. Two rules fall out:
- **Leave headroom.** Cap threads at ~`domain_cores − 2` for the M=1 decode phase; never use
  the full bound. Prefill (large M, compute-bound) is separate — it wants all cores.
- **Thread settings are PHASE-dependent.** The optimum for decode (barrier-bound, wants
  headroom) ≠ prefill (compute-bound, wants all cores). Cap per-phase (hook the decode
  forward), don't set one global thread count.
- **Decode throughput is a BATCHING win first, kernel second — but the THREAD-COUNT optimum does
  NOT simply scale with batch (this repo, measured in-model).** The batching win is real and large:
  per-token MoE cost fell 1.39 ms → 0.32 ms (4.4×) and BW efficiency 26% → 75% from M=1 → M=64 in the
  ISOLATED MoE kernel, and aggregate decode throughput rose ~10× (8→81 tok/s) B=1→32 through the real
  engine — purely from continuous batching, NO kernel change. **So profile decode at a REALISTIC
  serving batch, not M=1** (M=1 is the single-stream regime non-batching engines like Kimi-K3 are stuck
  in; a continuous-batching server escapes it). **BUT do not scale the thread cap off the isolated
  kernel's preference.** The isolated MoE wants MORE threads as M grows (M≥16 → 32); the WHOLE decode
  forward does not — measured in-model, a static cap of ~8 threads matched or BEAT a batch-aware cap
  that scaled to 32 at every batch (B=16: 63 vs 57 tok/s; B=32: 85 vs 81). Reason: the MoE is only
  ~37% of the decode forward; the other ~63% (attention/norms/MHC/dense) is barrier-bound at small
  batch and prefers FEW threads, and it dominates the composite thread choice. **Tune the decode thread
  count on the WHOLE forward in-model (it collapses to ~8 here), not on the isolated hot kernel** —
  the recurring "isolated microbench misleads" trap. Weight-streaming decode is BW-bound at every tier
  (K3 NVMe-streamed, this repo RAM-resident 4-bit); the throughput lever is amortizing the stream over
  the batch, not the kernel's thread count.
- **Confirmed dead-end: giving the hot kernel MORE threads than the rest of the (capped) forward does
  NOT help when the kernel is FRAMEWORK-bound in-model.** Tried raise-only MoE-decode-threads (MoE@32/64
  while the decode forward stays at 8) under passive wait — no change (B=32 84.9→85.5, noise). Reason:
  in-model the decode MoE sits ~4.5× above its isolated-kernel time (expert_apply 6.3 ms vs 1.39 ms at
  M=1) — it is dominated by the per-call dispatch/gather FRAMEWORK overhead, not the kernel's bandwidth,
  so the kernel's thread preference is moot in-model. Lesson: before chasing a kernel's thread/BW optimum,
  check the in-model-vs-isolated gap (`overhead-attribution`) — if the op is framework-bound, NO thread
  knob moves it; attack the dispatch/gather glue instead.
- **Isolated microbenches MISLEAD here.** An isolated M=1 GEMM on an idle node is *fastest at
  the full thread count* (0.017 ms @ 42) — the exact opposite of in-model, because idle has no
  framework threads to contend. You MUST measure in-model wall time (a per-op timer like
  `INTEL_CPU_DSV4_TIMEIT`), not an isolated kernel bench, to see the contention cliff.
- **Don't self-tune by sweeping across live decode steps — it's confounded.** Probing a
  different thread count on each of the first few decode steps mixes two confounds: each probe
  sits at a *different context length*, and resizing the thread pool upward charges the resize
  cost to the higher-thread probe. In this repo that in-run sweep picked threads=8 and rated the
  true optimum (40) as the *worst* — the exact inverse of the clean fixed-cap sweep. Use a
  **deterministic rule** (`domain_cores − headroom`) taken from an **offline fixed-cap sweep**
  (one fixed value per run, compare steady-state medians); never a live in-run search.
- **A thread cap tuned on a PROXY / dummy weights does NOT transfer to the real model.** The
  `bound−2` decode optimum above was found on a 4-layer perf-proxy with *dummy* MoE weights (a
  trivial MoE), so attention dominated and wanted many threads. On the REAL model the MoE is a
  heavy 256-expert `fused_experts` with its OWN steep inverse thread scaling (~0.1 ms @ 8 vs
  ~2 s @ 40), and it DOMINATES the decode forward — so the real-model whole-forward optimum
  collapsed to ~8 threads, ~5× lower than the proxy's 40. **Re-tune the thread count on the REAL
  weights** (real MoE compute), never ship the proxy's value. The dominant op sets the cap.
- **One cap tuned to the DOMINANT op — do NOT nest a second per-op cap inside it.** It is
  tempting to keep a high whole-forward cap (good for attention) and drop threads only around
  the MoE. In this repo, composing a per-op MoE `set_num_threads` INSIDE the decode-wide cap
  regressed decode ~200× (the per-op resize churns the parallel-for pool / re-inits the barrier
  every layer). Use a SINGLE cap set to the dominant op's optimum for the whole forward.
- **tp>1: you CANNOT cap threads per-forward — it DEADLOCKS.** `set_num_threads()` inside the
  forward desyncs the sgl_kernel shared-memory spin barriers (MoE dispatch / all-reduce) that
  were sized to the thread count at init — verified hang (even with gloo all-reduce; and a
  "set once, no restore" variant hung too because unequal-core domains give ranks different
  counts). For tp>1, leave headroom at LAUNCH via `SGLANG_CPU_OMP_THREADS_BIND` (bind each rank
  to its domain cores MINUS ~2) so the count is fixed once and the barriers size correctly. Note
  tp>1 M=1 decode is ~330× slower than tp=1 here regardless (barrier cliff per rank) — prefer
  tp=1 for memory-bound decode (see `sub-numa-clustering`).
- **TP HURTS a cap-needing decode when the model fits one domain — tp=1+cap beats tp>1 (measured).**
  Because the decode cap can't apply at tp>1 (deadlock), a barrier-bound M=1 decode is stuck at the full
  per-rank bind count. DSV4 full-43 dummy: GNR tp=2 (per-socket, 128 thr/rank, cap self-skipped) decode
  = 3.3 tok/s vs EMR tp=1+cap=8 = 9.3 (**2.8× WORSE**); prefill +11% (compute shards). So TP is a
  CAPACITY / prefill lever — for decode on a model that FITS one NUMA/SNC domain, tp=1 with the cap wins;
  extra domains buy nothing for decode. Shard (tp>1) only when the model does NOT fit one domain, and
  accept the decode-thread penalty then. (Cheap full-dummy GNR run killed the GNR-tp-for-decode idea
  before any full-weight bring-up — the perf-proxy ladder working.)

**Origin lesson 3 — OpenMP IDLE-THREAD SPIN-WAIT is the deeper cause; fix it FIRST (this repo).**
Much of "the decode thread cliff" and a separate "the 2nd forward is ~300× slower than the 1st"
pathology trace to ONE root cause: by default idle OpenMP workers **busy-wait** between parallel
regions (`OMP_WAIT_POLICY=active`; Intel/LLVM OMP `KMP_BLOCKTIME`=200 ms). With many threads and
sequential/nested regions (a transformer forward is thousands of small regions), the idle pool spins
on every core and **thrashes the next region** — so a forward that follows prior forwards collapses.
Measured (DSV4, 4-layer proxy, 64 threads): benchmark prefill **167 s** vs warmup 0.57 s (293×), and
decode **1.5 s/step**. Setting `OMP_WAIT_POLICY=passive` + `KMP_BLOCKTIME=0` (park idle threads) at
the SAME 64 threads: prefill **2.66 s (63×)**, decode **0.13 s/step (10×)**, reproducible. This is
scheduling-only → **token-identical** (no accuracy gate needed, just confirm ids).
- **Diagnosis signature:** a *uniform* ∝-work slowdown of a WHOLE forward (every op inflated by a
  similar factor), present on the 2nd+ forward but not the 1st, that SURVIVES process pinning
  (`numactl --physcpubind`) and disappears only at 1 thread → it's spin-wait, not op cost, not NUMA,
  not allocator, not thread *count*. (A standalone warm microbench of any single op looks fine; the
  pathology is contention BETWEEN regions, so it only shows in-model across multiple forwards.)
- **This REFRAMES Origin lesson 2:** the decode "cap threads at bound−2" and "never use the full
  bound" guidance is largely a WORKAROUND for spin-wait (fewer spinning threads = less thrash). With
  passive wait, uncapped decode at the full bound is fine (0.21 s, no cliff); the cap then buys only a
  minor M=1 win (0.21→0.13 s). **Apply the spin-wait fix FIRST (set once at launch, keep full
  threads); treat per-phase thread-capping as a small secondary lever, not the primary fix** — and it
  avoids the per-forward `set_num_threads` churn that the tp>1 / don't-nest rules warn against.
- **Beware the red herring:** `OMP_PROC_BIND=close`/`OMP_PLACES=cores`/`GOMP_CPU_AFFINITY` can
  *collapse* a framework that sets its own thread count (SGLang `init_cpu_threads_env`) down to 1
  thread — which "fixes" the cliff trivially by removing parallelism. Verify the thread COUNT held
  (log `torch.get_num_threads()` per forward) before crediting a pinning env. Prior art converges
  here: Kimi-K3 on the same Xeon uses `OMP_PROC_BIND=close OMP_PLACES=cores` + one rank/socket, set
  ONCE at launch, never retuned per-forward.

## The knobs (in leverage order), each vs a uPP-measured budget
0. **OpenMP wait policy (set FIRST, at launch).** `OMP_WAIT_POLICY=passive` + `KMP_BLOCKTIME=0`
   so idle workers park instead of spinning. Highest-leverage, zero-code, token-identical; prevents
   the inter-region spin-contention that inflates every multi-forward run (Origin lesson 3). Must be
   set BEFORE the OpenMP pool initializes (launch env / sbatch, not mid-process).
1. **Thread count + affinity.** Each TP rank must get a disjoint, NUMA-local core set.
   Set `OMP_NUM_THREADS` = cores-per-domain and bind (`SGLANG_CPU_OMP_THREADS_BIND`,
   `numactl --cpunodebind`/`--physcpubind`, `GOMP/KMP_AFFINITY`). Check against uPP
   `threading.cores_to_saturate_bw` and `meta.cpu_count / n_domains`. Symptom of failure:
   throughput flat vs threads, or all ranks on the same cores.
2. **NUMA / SNC binding + first-touch.** One TP rank per SNC domain, membind local; weights
   first-touched on the owning domain. Remote access costs the uPP `remote_bw_penalty`
   (~0.6 on GNR). Aggregate BW = `domains_used × per_domain_bw` only if sharded NUMA-local.
3. **TP-rank ↔ domain mapping.** #ranks should divide the domain count and shard heads/experts
   evenly; a bad map leaves domains idle (see `sub-numa-clustering`). **sglang asserts `dim % tp == 0`
   (QKV/MoE projection sizes, heads) — so valid tp is constrained by the MODEL, not the hardware domain
   count.** A model with 2^k dims (e.g. DSV4 proj=32768, 64 heads) accepts only power-of-2 tp {2,4,8,..},
   NEVER tp=6 — so a 6-SNC-domain node can't do one-rank-per-domain for it; fall back to tp=2 (per socket).
   Fail-fast cheaply on the perf-proxy before committing a scarce-node run.
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
