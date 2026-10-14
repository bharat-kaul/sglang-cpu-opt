# DSv4-Flash CPU — end-to-end wall time: baseline & progressive improvement

Live ledger of the **North-Star metric** — end-to-end tok/s for DeepSeek-V4-Flash on CPU — as each authored
kernel is integrated into the serving path. This is the **Phase 2** deliverable of the
[methodology playbook](methodology-playbook.md#phase-2--north-star-end-to-end-wall-time); the per-kernel
standalone results live on the [kernel-pilot performance page](kernel-pilot-performance.md).

![DSv4-Flash CPU end-to-end wall time — progressive improvement](plots/e2e_walltime_v1.png)

Solid bar = measured & integrated (fingerprint-gated); dashed = pending Phase-2 integration.

## Operating point & headline
- **Input** S = 1024, **output** G = 8/step (G=256 applied analytically), full depth **43 layers**.
- **North-Star:** B=32 aggregate decode tok/s (throughput) **and** B=1 single-stream decode tok/s (latency floor).
  The baseline below is the **B=1 latency floor**; the B=32 throughput point is added once decode-batching is wired
  (the torch-reference baseline OOM/watchdog-kills at B≥8, so only B=1 is measurable pre-optimization).
- **Node:** EMR Xeon 8592+ (64c/socket), `--constraint=ddr5600 --exclusive`, OMP64 close/cores, `numactl -N0 -m0`.

## Baseline (measured) — the denominator every optimization is scored against

| metric | full-depth (43L) | source |
|---|---:|---|
| **decode** | **1253 ms/step = 0.80 tok/s** (B=1) | two-depth extrapolation, job 385387 |
| **prefill (TTFT)** | **67.8 s** (S=1024) | two-depth extrapolation, job 385387 |

This is the **unoptimized** path: torch-reference DSA with the materialized O(S²) attention and **no authored
kernels wired**. It is deliberately the slow starting point — exactly what Phase-2 integration replaces.

**Why "dummy weights" are not fast:** `--load-format dummy` removes the checkpoint read and the fp4/fp8→bf16
dtype bridge, so *model load* is instant — but a GEMM with random weights does the **identical FLOPs**, so
*compute* is unchanged. The slowness is the unoptimized torch attention, not the weights.

**⚠ But the dummy *router* can mislead MoE wall-time (mode collapse).** DSv4 routes each token to a small
subset of its **256 experts**, and the batched-decode cost is set by the **distinct-expert working set** —
which *grows with batch* (`distinct_experts(B)·per_expert_bytes`). A random dummy router routes **degenerately**
(collapses the whole batch onto the same few experts), so a naive dummy proxy would read too few experts and
report a **falsely fast** MoE decode (and a single-expert path would hide the grouped-GEMM gather/scatter
routing bugs). This proxy therefore **keeps the real router/gate weights** (tiny — loaded even with dummy
*expert* weights), drives **varied tokens**, and **gates on the measured distinct-expert-per-batch histogram
matching the real `E(distinct | B)`** (not 1, not all-N) before any MoE wall-time is trusted. *(Methodology:
the [perf-proxy MoE mode-collapse rule](methodology-playbook.md#phase-2--north-star-end-to-end-wall-time).)*

## Calibration (two-depth linear model)
Wall time is linear in depth because every DSv4 layer is the same MoE+MLA+DSA archetype, so we extrapolate
from two shallow proxies instead of paying full depth each iteration:

$$t(L) = t_{fixed} + L\cdot t_{layer},\quad t_{layer}=\frac{t_8-t_4}{4},\quad t_{fixed}=t_4-4\,t_{layer}$$

| regime | proxy4 (4L) | proxy8 (8L) | $t_{fixed}$ | $t_{layer}$ | → 43L |
|---|---:|---:|---:|---:|---:|
| decode (ms/step) | 198.3 | 306.5 | 90.2 | 27.04 | **1253** |
| prefill (ms) | 7216 | 13425 | 1007 | 1552 | **67 754** |

Per-layer cost dominates (decode 93%, prefill 98.5%), so $t_{fixed}$ is small and the extrapolation is
well-conditioned. One full-depth dummy run will **validate** it before any headline number is final.

## Progressive-improvement ledger
Each optimization is integrated **one at a time**, **env-gated default-OFF**, and accepted only when its
**dummy-weight output fingerprint is unchanged** (a perf transform, not a numeric change). A measured
**no-change or regression is a valid, published stage** — the *why* is recorded, same as the standalone phase.

| # | stage | status | decode tok/s (B=1, 43L) | prefill TTFT | notes |
|---|---|---|---:|---:|---|
| 0 | **baseline** (torch-ref DSA, no opt kernels) | ✅ measured | **0.80** | **67.8 s** | job 385387; the denominator |
| 1 | + indexer logits | ⏳ pending | — | — | oneDNN donor + fused epilogue; **bf16-KV** traffic lever here |
| 2 | + indexer top-k | ⏳ pending | — | — | pure-authored selection |
| 3 | + compressor | ⏳ pending | — | — | pure-authored, hand-`exp` (D1) |
| 4 | + sparse attend | ⏳ pending | — | — | oneDNN donor + fused softmax (bf16 rejected standalone) |
| 5 | + MHC sinkhorn | ⏳ pending | — | — | fused recurrence |
| 6 | + MHC hash/combine | ⏳ pending | — | — | accumulate-once |
| 7 | + Lane-A ceiling verify | ⏳ pending | — | — | MLA proj / MoE experts / lm_head at roofline |
| 8 | + system/dispatch overhead | ⏳ pending | — | — | bind-once, fewer/bigger parallel regions, allocator/NUMA |

Full data: [wall_time_progress.json](../plugin/validate/results/wall_time_progress.json).

## Method & gates (per stage)
1. **Freeze the kosher baseline** — tag the commit, golden dummy-weight fingerprint; every optimization is
   env-gated default-OFF so gate-off reproduces the baseline byte-for-byte.
2. **Integrate one op**, measure proxy4 (+ periodic proxy8 recalibration), extrapolate to 43L per regime.
3. **MoE routing fidelity** (this is an MoE model): keep the **real router/gate weights** + drive **varied
   tokens**, and gate the proxy's **distinct-expert-per-batch histogram** against the real `E(distinct | B)`
   (not 1, not all-N) before trusting any MoE wall-time — a collapsed dummy router under-counts expert traffic.
4. **Gate:** the dummy-weight output fingerprint stays invariant (overhead/precision-equivalent transforms only).
5. **Publish the delta** here — including a no-change/regression, with the root cause.
6. After all ops: **one full-depth dummy run validates** the extrapolation and exposes residual depth-accumulating
   overhead to bisect; then the real-weight **accuracy** sign-off (Phase 3).

## Provenance
- Baseline + calibration: EMR job **385387** (`BATCHES=1` proxy4+proxy8, S=1024, reps=3, tp=1, bf16, dummy
  weights, mem-fraction 0.6), via `plugin/validate/bench_wall_time.py` / `run_dsv4_step0_calib.sbatch`.
- Chart: `plugin/validate/make_e2e_walltime_plot.py` from `results/wall_time_progress.json` (dependency-free SVG).
- The B≥8 baseline points are unmeasurable pre-optimization (torch-reference materialized attention OOM/
  watchdog-kills); they populate as the optimized DSA kernels replace that path.
