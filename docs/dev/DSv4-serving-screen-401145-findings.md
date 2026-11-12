# DSv4 serving optimization campaign: findings and dispositions

**Latest disposition, 2026-11-11:** C4 is retained as an experimental result,
not integrated. Section 10 records full task accuracy, mixed end-to-end
performance and the crossover screen. Earlier sections preserve the decisions
made at each stage; their "next step" language is historical, not a current
work queue. No globally optimal-kernel or complete-native-coverage claim follows.
The [batch-regime analysis](DSv4-batch-regime-analysis.md) distinguishes request
batch, operator geometry and dispatch boundaries.
Earlier [standalone findings](https://github.com/bharat-kaul/sglang-cpu-opt/blob/dsv4-standalone-meta-opt/docs/dev/dsv4-standalone-meta-findings.md)
and the [native inventory](https://github.com/bharat-kaul/sglang-cpu-opt/blob/dsv4-standalone-meta-opt/docs/dev/dsv4-native-kernel-inventory.md)
are on their separate experimental branch; their baseline is not this
accuracy-qualified serving baseline.

Analyzed 2026-11-09 (Pacific). Baseline main `34fe50f`, fixed32 warm CPU threads,
TP1, both routed/shared clamps ON. **Job401145 was attribution only: no optimized
kernel, measured candidate speedup, default promotion or new task-accuracy claim.**

Sections 1-5 retain job401145's historical disposition. The bounded follow-up,
job401206, is recorded in section 6; it adds actual sparse operands and decode
boundaries without repeating the earlier full-request benchmark.

## Result in one paragraph

The strongest supported lead is **the actual FP32 serving sparse-attention core
during long-context prefill**, not indexer KV replication. Corrected main
compression pooling and surrounding work are the next visible area. Indexer
replication and compressor projection casts are small in the parity-passing
prefill/B32 records. A large uninstrumented portion remains inside decoder layers;
we cannot call that portion framework overhead or declare its kernels optimal.

The job was already FAILED at 23:32:00 before the requested cancellation was
attempted. Its output-parity guard stopped it at the second B32/S3072 warmup:
2/32 generated token sequences differed. Three other workload points completed
their repetitions with output agreement. Keep those qualified partial results;
do not describe the overall job as passing or successfully cancelled.

## 1. Evidence validity and estimators

- Node `pcl-sprh16.sc.intel.com`, worker PID3839623, job401145; one resident engine.
- Frozen source:
  `/scratch/bkaul/dsv4-serving-screen/main-20261110T061822Z/source`.
- Manifest SHA256:
  `ff4a027a7c60c1d60edf6c9790fc8c3e514f1a6c8868e3888b2709b1b67bd5ae`.
- Raw evidence: `/scratch/bkaul/dsv4-serving-screen/job-401145/results`.
- Posthoc reconciled output:
  `/scratch/bkaul/dsv4-serving-screen/job-401145/partial-analysis.json`.
- [Aggregation procedure](../../plugin/validate/analyze_serving_screen_401145.py)
  rechecks frozen files, stamped installed sources/library, prompt hash, canonical
  run identity, unique worker sequences, complete phase/token coverage, all43
  layer observations, failures, thread counts, raw response IDs and saved
  summaries. It verifies exclusive scope totals do not exceed forward time.
- Frozen/stamped source checks passed at analysis time. Private extension hashes
  are collected **posthoc**; they are not a retrospective loaded-library trace.
- Actual phase/token/sequence-length schedules match across compared requests.
  That is not proof of identical internal KV contents or identical consumed
  prefixes after generation diverges.

For the first three points, table entries are medians over **three paired
requests in the same process**, excluding warmups. Each share is first computed
as that request's summed region time divided by its summed model-forward phase
time. Per-forward time is the phase total divided by its forward count, then the
median over requests. Min/max and individual request records are retained in JSON.
They are not independent-process medians and not single-kernel invocation times.
No overlapping parent/child costs are summed to manufacture a larger opportunity.

The perturbation estimator below is the median of the three **paired** on/off
ratios, not the original driver's ratio of independently computed medians.
Diagnostic-only is conservative: any pair differing by >5%, coordinate mismatch
or output-parity failure qualifies. Low aggregate perturbation alone does not
prove every leaf timer is unbiased.

| Point | Phase | Actual tokens/forward | Off ms/forward | On ms/forward | Paired on/off median (min-max) | Evidence status |
|---|---|---:|---:|---:|---|---|
| B1/S128 | Prefill | 128 | 2248.7 | 2281.8 | 1.015 (1.013-1.017) | Output agreement, low observed perturbation |
| B1/S128 | Decode | 1 | 208.1 | 225.6 | 1.083 (1.082-1.090) | Diagnostic-only: probe overhead |
| B1/S3072 | Prefill | 512, six forwards/request | 5031.8 | 5045.3 | 1.003 (0.985-1.023) | Output agreement, low observed perturbation |
| B1/S3072 | Decode | 1 | 221.4 | 239.2 | 1.066 (1.039-1.098) | Diagnostic-only: probe overhead |
| B32/S128 | Prefill | 256, sixteen forwards/request | 2225.4 | 2277.8 | 1.024 (1.019-1.025) | Output agreement, low observed perturbation |
| B32/S128 | Decode | 32 | 862.4 | 890.7 | 1.027 (0.980-1.034) | Output agreement, low observed perturbation |
| B32/S3072 | Prefill | 512, 192 forwards/request | 5303.9 | 5512.8 | 1.039 (one warm pair) | Output-parity failure; diagnostic only |
| B32/S3072 | Decode | 32 | 1127.3 | 1112.4 | 0.987 (one warm pair) | Output-parity failure; diagnostic only |

This explains the runtime: the last point processed 98304 input tokens per
request, not one forward. Its off warmup spent 1018.3s in192 prefill forwards and
7.9s in seven decode forwards. Repeating entire requests was excessive for this
initial attribution task. Future isolation should replay representative calls,
amortizing state preparation without silently changing that state.

## 2. Measured component shares

Percent of **instrumented model-forward phase time**, not full client latency.
Pool below means the whole corrected `_compress_prefill_core` scope, including
pooling, norm, RoPE and roundtrip work, **not just the softmax arithmetic**.
Projection combines main and indexer `compute_kv_score` calls.

| Point/phase | Sparse core | Whole sparse pipeline | Main pool scope | Whole projection | Projection casts | KV replication |
|---|---:|---:|---:|---:|---:|---:|
| B1/S128 prefill | 6.02% | 7.99% | 5.90% | 2.53% | 0.88% | 0.049% |
| B1/S3072 prefill | **26.03%** | 28.08% | **10.50%** | 1.93% | 0.26% | 0.210% |
| B32/S128 prefill | **14.12%** | 17.46% | **12.06%** | 2.52% | 0.29% | 0.048% |
| B32/S128 decode | 5.35% | 9.41% | 2.16% | 3.70% | 0.91% | 0.059% |
| B1/S128 decode, diagnostic | 1.52% | 8.45% | 0.41% | 10.95% | 5.70% | No calls |
| B1/S3072 decode, diagnostic | 4.08% | 11.86% | 0.39% | 10.66% | 5.59% | No calls |
| B32/S3072 prefill, parity-failed warm pair | 37.21% | 40.73% | 9.69% | 1.72% | 0.20% | 0.050% |
| B32/S3072 decode, parity-failed warm pair | 18.37% | 28.53% | 1.79% | 2.93% | 0.74% | 0.051% |

For B1/S3072 prefill, the sparse-core share ranges25.94-26.17% over the three
instrumented requests. For B32/S128 prefill it ranges13.46-14.23%.
The timing spread and the low paired perturbation make these useful
prioritization observations, not proof of removable time.

### A. Sparse serving: advance to matched-input feasibility

At B1/S3072 prefill, a hypothetical 2x improvement to the measured26.03% core
would yield about **1.150x phase speedup** if all other costs held fixed.
At B32/S128 prefill, the same hypothetical2x gives only **1.076x**.
These are conditional Amdahl calculations, **not delivered gains**.

The core dominates the sparse pipeline during prefill. The pipeline-minus-core
residual is approximately2.04% at B1/S3072 and3.34% at B32/S128
(differences of displayed median shares, not a separately measured median).
Pure preparation cleanup alone is therefore not the strongest prefill lever.

Actual long-prefill q is `[512,64,512]`, not `[B,64,512]`. The implementation is
the FP32 shared-KV masked online kernel. Its contract includes masks and sinks;
the faster historical BF16/indexed donor is not a drop-in equivalent.

Next action: capture a small set of real q/KV/mask/sink inputs and replay the
existing core and eligible alternatives with **the same numerical boundary**.
Include preparation and selection order in the comparison. Profile mask scan
versus visible-key arithmetic and exp recurrence before choosing a rewrite.
This result reopens a *specific serving kernel* investigation; it does not
contradict negative experiments on a different standalone entry.

### B. Main compressed-KV production: advance, but split the scope first

The complete `forward_core_compressor` parent occupies19.82% of B1/S3072
prefill and20.80% of B32/S128 prefill. Its corrected pool scope is10.50%/12.06%;
the parent's exclusive residual is7.69%/6.72%.

That residual includes uninstrumented operations inside the compressor wrapper;
it is **not proven pure Python overhead**. The parent also overlaps main
projection, so do not add it to the whole-projection column.

Halving only the pool scope would give about1.055x/1.064x respectively, below
the >10% materiality threshold for the whole phase. A broader source-faithful
compression/hand-off improvement could matter, but requires finer attribution.
At B32/S128 decode the compressor parent's exclusive residual is8.88%, whereas
the pool scope is2.16%: continuation/history/publication deserves attention,
not an assumption that the standalone pooling arithmetic is the bottleneck.

Next action: isolate one representative complete-window call, including overlap,
norm, RoPE, quantization and history bookkeeping. Reuse the already available
native donor only where its contract matches. Preserve accepted history and
rounding semantics.

### C. Indexer KV replication: deprioritize for these measured points

Replication is at most0.21% in the parity-passing prefill/B32-decode points.
Even ideal deletion of that observed share gives only about1.0021x phase
speedup. This does not justify a separate shared-KV kernel project for these
operating points.

Actual indexer shapes explain why the API-domain analytical worst case was
insufficient:

- B1/S3072 prefill: q rows512, shared compressed keys128, expanded KV
  `[512,128,128]`, FP32 (32 MiB logical buffer).
- B32/S128 prefill: q rows256, keys64; expanded `[256,64,128]` (8 MiB).
- B32 decode: q rows32, keys8; expanded `[32,8,128]` (128 KiB).
- B1 decode: no observed native indexer logits/replication call.

Large logical buffer size alone does not establish a large wall-time share.
Do not extrapolate this disposition to B64, different histories or larger actual
S without checking their corresponding shapes/costs.

The duplicate native top-k call remains a concrete cleanup candidate, but the
whole top-k wrapper is0.018% at B1/S3072 prefill and0.076% at B32/S128 decode.
It is not the material opportunity sought here.

### D. Compressor projection conversion: secondary, B1 remains uncertain

The projection occupies about2-3% of parity-passing prefill and3.70% of B32
decode; casts alone are0.26-0.91% at those long-prefill/B32 points. The large
analytical conversion volume did not translate into a dominant measured cost.

B1 decode reports a larger projection share, but instrumentation itself increases
forward time by roughly7-8% at the paired median. The projection wrapper includes
dispatch-observer and tensor-metadata overhead. In B1/S3072, projection share
ranges7.02-10.68% and cast share2.91-5.62% across requests. Treat those as
diagnostic leads, not a basis for a promised >10% gain.

Next action only if B1 decode becomes the priority: replay one projection at
actual shapes with a lightweight timer, separately measuring conversion and
GEMM. No widened-weight cache or precision substitution is approved.

## 3. The largest remaining blind spot

Decoder `layer.total` **exclusive** time accounts for:

- 77.61% at B1/S128 prefill.
- 49.78% at B1/S3072 prefill.
- 58.99% at B32/S128 prefill.
- 72.12% at B32/S128 decode.

This is what the selected probes did **not** subdivide: dense projections,
routed/shared MoE, MHC, norms and other work. It includes optimized native
compute, eager composition and probe bookkeeping. It cannot be renamed
"framework overhead" or used to assert that the three initially suspected
costs exhaust the opportunity.

For decode, the strongest next *measurement* is a cheap boundary split of that
residual around routed MoE, shared expert, dense projections and MHC on a small
number of real forwards. This differs from launching another full prompt
repetition campaign.

## 4. Parity failure and history limitations

B32/S3072 warm on/off output IDs differ in two rows (zero-based row24 and27).
First differing token positions are6 and7 (the seventh/eighth generated tokens).
All32 rows agree through the first six output tokens. The complete request's
parity check nevertheless fails; no successful case-summary file was written
for the on arm, while its raw responses and worker events were retained.

This does not prove a new model accuracy regression, nor prove that the probe
changed arithmetic. There is only one off/on warm pair, no same-condition
off/off repeat at this workload, and equal metadata is not equal hidden/KV
state. The prior task-accuracy closure is not reopened by this diagnostic.
Resolve repeatability with a bounded matched-state control before promoting a
change based on this workload.

The B32 long-context timings above are retained as **diagnostic observations
only**; they are excluded from the materiality decision and speedup bounds.
Even the prefill share is not promoted by the fact that divergence appeared
later in generation: earlier internal tensor/state equality was not checked.

Cache/history is also a scaling confound in the successful points. At B32/S128,
the native sparse call sees roughly4804-5860 assembled KV rows across
prefill/decode, larger than a simple fresh `32*128` estimate. At B32/S3072
decode the recorded KV shapes include32768 and57344 rows. The engine is resident
across cases; these observations are not an independently reset B-scaling study.
The metadata records shapes but **not mask contents/density**. Therefore the
analytical quadratic-mask-scan hypothesis remains unproven by these timings.

## 5. Disposition and next experiment boundary

| Item | Disposition from current evidence |
|---|---|
| Long-prefill FP32 serving sparse core | Highest measured lead among instrumented regions; matched-input isolation next |
| Main compression / history / handoff | Second visible lead; split full producer rather than optimize pilot pooling blindly |
| Indexer replication | Deprioritize at observed shapes; too small to deliver material phase gain alone |
| Projection casts | Deprioritize for measured prefill/B32; B1 requires lower-overhead isolation |
| Duplicate top-k | Small cleanup, not headline optimization |
| General dense/shared/routed/MHC decode | Still unranked; large uninstrumented residual, needs a few boundary-timed forwards |
| B32/S3072 pair | Failed parity; diagnostic only |
| B64 and batch-transition coverage | Still open per the [analytical map](DSv4-batch-regime-analysis.md) |

No more full-request repetitions are needed to make these provisional decisions.
Do not launch a new model load merely to reproduce the same broad screen.
Choose a representative per-forward state or captured operator operands; validate
that replay preserves the consumed state, then use short paired measurements.
Independent-process replication belongs to candidate acceptance after a lever
shows promise, not to repeating long prompt preparation to find the first hotspot.

Reproduce the fixed-job aggregation (choose an unused output path):

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/bkaul/venvs/sglang-cpu/bin/python \
  plugin/validate/analyze_serving_screen_401145.py \
  --output /tmp/dsv4-401145-analysis.json
```

## 6. Bounded follow-up 401206: actual operands and decode boundaries

Completed 2026-11-10 Pacific on `pcl-sprh12.sc.intel.com`, same EMR hardware
class, fixed32 warm threads, CPU0-31/NUMA0 and approved math flags. Exactly one
request each at B1/S3072, B32/S128 and B64/S128, five output tokens: four decode
forwards include the next C4 block completion. No warmup requests, repeated
generation pairs, B32/B64 long-context cases or task evaluation.

### Evidence and qualifications

- Frozen source:
  `/scratch/bkaul/dsv4-serving-screen/meaningful-20261110-bDqKQY/source`.
- Manifest SHA256:
  `9d6064f04ed37b844ede8e39da6d32f498507f7916d91aaa40baef4ca124780e`.
- Results: `/scratch/bkaul/dsv4-serving-screen/job-401206/results`.
  `complete.json` records capture-run completion; `replay.json` records the
  subsequent checked replay and boundary aggregation.
- Replay JSON SHA256:
  `d996e1b560ad2e95b89636d7a57ebda5e421bdcfb669d185db0d54a45daf0c17`.
- Capture worker PID1971307, replay PID1972820.
  Loaded sparse replay library SHA256:
  `3a5ffa267cf282289c9daab92b345b10218cf130e5908ee422041998260c428f`.
- Exit file0 and final `SERVING_SCREEN_JOB_COMPLETE job=401206` marker present;
  source manifests checked before/after. Slurm accounting storage is disabled,
  so the waiter's queue disappearance is not an authoritative sacct success.
  Completion here is established by the launcher exit, final marker and outputs.
- 66 actual forwards:54 prefill chunks preparing state,12 decode steps.
  Eighteen captures: layers0/2/3 (SWA/C4/C128), final prefill and first decode,
  at each point. Captures contain actual FP32 q/KV/sink, bool mask and FP32 output.
- First prefill at each point and every capture-containing forward are excluded
  from boundary attribution. The remaining prefill counts are4/14/30; decode
  counts are3 each. Shares below are **sums of scope times / sums of forward
  times**, not medians or independent-process estimates.
- The lightweight mode removes the earlier TorchDispatchMode and adds nested
  boundaries, but has **no matched off/on perturbation check**. Attribution is
  diagnostic. No generated-token repeatability or fresh accuracy claim.
- Request times were178.780s /44.054s /94.464s. The first model forward took
  150.674s, including lazy native builds; it is excluded. Loading and building
  are not kernel inference latency. This still needs one-time state preparation,
  but does not repeat it for statistical benchmarking.
- Thirteen focused tests passed, including full-driver rejection of missing
  instruments and full-replay-entry rejection of nonfinite, wrong-coordinate,
  missing, hash-mismatched and wrong-output evidence. Real replay checks all18
  captured outputs, not just synthetic fixtures.

### A. A measured, shape-specific sparse-call improvement -- not a general win

The replay removes KV rows invisible to **every** query using an ascending mask
union, preserving each query's original visible-key order. Original replay,
prepared compaction and preparation-inclusive compaction all matched the captured
native output **bytes** at all18 coordinates. Nothing changed in serving.

Below, time is the median of three alternating trials in one replay process on
reused buffers. The compact arm includes union selection, KV/mask gathers and the
unchanged native kernel. Preparation-independent timing is retained separately
in JSON and is not used to credit a win.

| Point | Phase, layer | Original ms (min-max) | Compact including preparation ms (min-max) | Call latency reduction |
|---|---|---:|---:|---:|
| B1/S3072 | Prefill, C4 L2 |78.384 (77.934-78.484)|79.707 (77.998-84.921)|-1.7% |
| B32/S128 | Prefill, SWA L0 |6.909 (6.833-6.909)|5.956 (5.934-6.000)|13.8% |
| B32/S128 | Prefill, C4 L2 |8.174 (8.124-8.290)|7.276 (7.262-7.390)|11.0% |
| B32/S128 | Prefill, C128 L3 |6.995 (6.968-7.228)|6.037 (5.994-6.050)|13.7% |
| B64/S128 | Prefill, SWA L0 |8.197 (8.148-8.214)|6.074 (6.026-6.084)|25.9% |
| B64/S128 | Prefill, C4 L2 |9.396 (9.286-9.424)|7.339 (7.327-7.415)|21.9% |
| B64/S128 | Prefill, C128 L3 |8.199 (8.161-8.241)|5.994 (5.986-6.079)|26.9% |

This is a new **11-27% latency reduction on six captured short-prefill calls**,
not proof that an entire phase accelerates by that amount. These final prefill
chunks have M256: only two short requests are processed while the assembled KV
pool includes other requests/history. At B64 L2, the pool has8756 rows but its
union-visible subset has320; at B32 L2 it is4544 versus320. Earlier chunks have
different pool histories, so the last chunk cannot establish their average gain.

Compaction loses at all nine captured decode coordinates once preparation is
included: B32 latency increases33-51%; B64 increases57-65%; B1 increases99-151%.
It also loses at the three long-prefill coordinates (about2-9%).
**Do not install an unconditional compaction path.** A cheap prefill-only
construction/selection policy is a possible narrow follow-up, not a promoted
optimization. Sparse core occupies only12.2%/14.3% of the measured short-prefill
phases, so these call gains are not evidence of a >10% full-phase improvement.

For long-prefill C4, the original call takes78.384ms; the all-masked same-shape
control takes1.949ms (2.49%). Together with the neutral/loss compaction result,
this prioritizes **visible-key dot/online-exp/value work**, rather than removing
dead mask rows, at that captured point. The control changes branches and work;
it is not an exact scan-cost subtraction or proof of a compute roofline.

### B. Decode is substantially attributed, not proven optimal

| Scope (% of instrumented decode time) | B1/S3072 | B32/S128 | B64/S128 |
|---|---:|---:|---:|
| Routed MoE |13.19|36.97|40.04|
| Shared MoE |2.80|4.66|2.92|
| Dense FP8 method |8.57|6.07|5.06|
| MHC pre |8.73|2.85|2.04|
| MHC post |4.28|2.14|1.37|
| Sparse core |3.85|4.73|7.13|
| Complete main compressor parent |11.43|17.64|21.42|
| Attention parent exclusive remainder |24.04|13.66|11.30|
| Decoder-layer exclusive remainder |2.88|0.77|0.50|

Parent scopes overlap descendants (including some dense work): **do not sum this
table as disjoint buckets**. The report retains exact exclusive accounting.
The former50-72% layer remainder is now largely assigned to named scopes.
Attention's exclusive remainder still contains uninstrumented arithmetic,
conversions and orchestration -- it is **not proven Python/framework overhead**.
The routed MoE share is substantial, but does not establish recoverable headroom
without a matched packed-kernel/weight-traffic baseline.

### C. Compression has periodic decode cost, not one smooth per-batch curve

| Point | Input sequence length | Whole forward ms | Main compressor ms | Pool/norm/RoPE/roundtrip scope ms |
|---|---:|---:|---:|---:|
| B32/S128 |130|918.15|95.50|0|
| B32/S128 |131|926.34|101.91|0|
| B32/S128 |132, C4 completion|1130.71|327.46|159.09|
| B64/S128 |130|1421.90|176.74|0|
| B64/S128 |131|1377.03|172.96|0|
| B64/S128 |132, C4 completion|1881.32|652.90|317.49|

These are individual steps, not replicas of one input. At the C4-completion
step the current accumulator calls the corrected producer672 times at B32 and
1344 times at B64 (21 C4 layers times batch). Source
`_accumulate_request_blocks` processes requests/finished windows separately and
calls `_compress_prefill_core` before appending each history row.

The observed step spike coincides with that producer/accumulator work. This
supports a focused investigation of **batching completed windows and reducing
carry/history operations while preserving overlap, FP32/BF16 boundaries and
FP8 roundtrip**. It does not show how much a rewrite would remove. Publish-only
handoff is small (about1% of decode); it is not the main target.

There is one completion among three retained decode steps here, whereas steady
C4 cadence is one per four steps. Do not extrapolate the table's averaged shares
as stationary throughput. No next-C128 completion was measured.

### Follow-up disposition

1. **Long-prefill C4 sparse:** investigate the existing FP32 GEMM/fused-softmax
   donor against these captured operands and actual mask/sink semantics. No
   automatic BF16 substitution; mask-union compaction did not help this point.
2. **Compression:** completed-window batching/history is now a concrete
   source-grounded opportunity, particularly the B32/B64 periodic decode spike.
3. **Short-prefill sparse compaction:** retain the measured call-level win as a
   narrow candidate, reject unconditional use and do not claim a serving gain.
4. **Decode routed MoE:** large measured share, unknown remaining headroom.
   Attention's11-24% exclusive residual still requires a finer split if pursued.

Captured operands can be reused without another model load. Replay procedure:
[`replay_serving_sparse.py`](../../plugin/validate/replay_serving_sparse.py);
run design:
[`serving-screen.md`](../../plugin/validate/serving-screen.md).

No main/default changes, production kernel changes or accuracy promotion.

## 7. Compression batching feasibility 401622: narrow win, modest decode upside

Completed 2026-11-10 Pacific. This bounded screen took about a minute of allocated
work, with **no model load or generation**. It tested the proposed mechanism
instead of collecting another broad attribution trace.

### Contract and evidence

- Job401622, `pcl-sprh01.sc.intel.com`, PID3349292, fixed32 warm threads,
  CPU0-31/NUMA0, Torch2.12.0+cpu.
- Snapshot `/scratch/bkaul/dsv4-compression-batch/source-20261111-4pfEnb/source`;
  manifest SHA256
  `4a94a8af2cc500021bf9c5655ed774c41608c619fc8b3b2056e3a743759be980`.
- Result `/scratch/bkaul/dsv4-compression-batch/job-401622/results/complete.json`;
  SHA256 `9b92748b179ee83dc493c7ed3a3df4f120cbe545852e5ec3482946635923f01e`.
  The result binds source/library hashes, seed, shapes, process, runtime and all
  trial samples. Launcher exit0, final marker and before/after source checks
  passed; Slurm accounting remains unavailable.
- Inputs are **synthetic signed projection rows and shape-faithful carry/history**,
  not actual captured neural activations. B1/32/64, C4/C128, histories128/3072,
  ordinary/completion steps:24 coordinates.
- Experimental scheduling batches ready windows through the **unchanged**
  production `_compress_prefill_core`. Its request-wise pooling remains a loop;
  norm/RoPE/FP8 work shares one batched call. No new native kernel or serving
  change was made.
- Timing includes carry append, window selection, batching preparation, pooling,
  norm, RoPE, FP8 roundtrip, history append and production history publication.
  Projection, restoring matched state and validation are outside timing.
- Outputs **and full carry/history/handoff state** matched bytes at all24
  measured coordinates, on every comparison. Three offline tests also cover
  heterogeneous/empty/partial/multi-window requests, two seeds, FP8 on/off, slot
  reuse, duplicate slots and a state-corruption negative control through the
  benchmark entry point. This is not a real-weight accuracy certificate.

### Measured completion-step costs

Each row is one layer's complete accumulation/publication work across the batch.
Numbers are median(min-max) milliseconds from three alternating trials in one
process, after one warm call per arm. Reused-buffer timing, not a DRAM roofline.

| Batch | History | Ratio | Scalar ms | Batched ms | Latency reduction |
|---|---:|---:|---:|---:|---:|
| B1 |3072|4|0.449 (0.443-0.583)|0.473 (0.459-0.491)|-5.4%|
| B32 |128|4|9.257 (9.195-9.379)|5.814 (5.769-5.916)|37.2%|
| B32 |3072|4|10.431 (10.396-11.612)|6.851 (6.820-8.178)|34.3%|
| B64 |128|4|18.125 (17.916-18.418)|10.835 (9.638-10.925)|40.2%|
| B64 |3072|4|23.697 (23.494-25.618)|16.665 (16.350-16.776)|29.7%|
| B32 |128|128|14.021 (13.154-14.127)|17.095 (15.381-17.161)|-21.9%|
| B64 |128|128|26.936 (26.159-27.112)|30.479 (29.941-31.447)|-13.2%|
| B64 |3072|128|27.210 (26.703-27.475)|32.045 (31.588-32.782)|-17.8%|

**Disposition:** retain a C4/B32-B64 completion-only scheduling candidate.
Reject unconditional batching: B1 and C128 completion medians regress.
C128/B32/history3072 is noisy and only4.4% slower by median, so classify that
specific coordinate neutral/inconclusive, not a strong regression. Ordinary
steps have small changes retained in the JSON; no claim that they improved.

### Translation into potential decode upside

Use the actual cadence, not the completion-step percentage:

`saved time per decode step = sum(layers_r * ((r-1)/r * ordinary_saving + 1/r * completion_saving))`.

The unconditional prototype's synthetic extrapolation is16.4ms at B32/history128
and40.7ms at B64/history128, including ordinary-step differences and C128 losses.
Those are **conditional sums across synthetic layers**, not measured full-model
improvements.

A more defensible narrow policy would leave ordinary steps, B1 and C128 unchanged.
For C4 completion only, `21 * completion_saving / 4` gives:

| Shape | Conditional saved time per decode step |
|---|---:|
| B32/history128 |18.1ms|
| B64/history128 |38.3ms|

Against the prior roughly1s/1.5s decode operating points, that is **around2-3%
potential time reduction**, not >10%. This cross-job comparison is an
order-of-magnitude planning estimate: inputs, node instance, cache behavior,
layer variation and runtime interference are not matched. No numerical claim
of an achieved end-to-end gain follows from it.

**Stop criterion reached for this mechanism:** do not spend a real-weight load
or broad evaluation promoting this modest scheduling-only candidate now. The
30-40% local C4 win is real in this synthetic screen, but the cadence dilutes its
model impact. This does not rule out a more substantial fused/batched pooling
implementation; that is a different mechanism, not demonstrated here.
The larger measured long-prefill sparse core remains a higher-upside candidate
for direct replay on the existing captured tensors.

Reproduce with
[`run_compression_batch.sbatch`](../../plugin/validate/run_compression_batch.sbatch)
and a frozen source snapshot. Implementation and aggregation are in
[`bench_compression_batch.py`](../../plugin/validate/bench_compression_batch.py).
No main/default changes and no production batching path enabled.

## 8. Actual-input long-prefill sparse replay 401625: a material C4 candidate

Completed 2026-11-10 Pacific. **No model reload, new native kernel build, BF16
substitution or serving change.** Existing FP32 alternatives were evaluated on
the actual final B1/S3072 prefill chunk from job401206: M512, H64, D512 at
layers0/2/3 (SWA/C4/C128). This identifies a material candidate, not an accepted
numerical change or achieved model-level gain.

### Evidence validity

- Job401625, node `pcl-sprh01.sc.intel.com`, PID3349960, fixed32 warm threads,
  CPU0-31/NUMA0, Torch2.12.0+cpu.
- Snapshot `/scratch/bkaul/dsv4-sparse-donor/source-20261111-XxUHv1/source`;
  manifest SHA256
  `14f58e7185dea8a678db1efc936cd70fe9297e9f6e43359455ada4c0452ac8e2`.
- Result `/scratch/bkaul/dsv4-sparse-donor/job-401625/results/complete.json`;
  SHA256 `f98780230e14affdb2698adf7f31bcea0e92274433806f17dcdfa93f218216c6`.
- Reused original native binary SHA256
  `3a5ffa267cf282289c9daab92b345b10218cf130e5908ee422041998260c428f`.
  Capture hashes are pinned independently in the replay driver. Original native
  replay reproduced captured output bytes at all three coordinates; each arm's
  repeated outputs were byte-identical to its own warm output.
- Exit0, final job marker, complete JSON and before/after source checks passed.
  Slurm accounting storage remains disabled; no sacct success claim.
- Initial job401624 stopped before measurement because the harness incorrectly
  expected the captured master thread affinity to be0-31. The actual original
  worker and capture record master affinity`[0]`, with32 threads. The identity
  expectation was corrected to the actual record, not relaxed. Five tests now
  include the positive actual-capture entry and negative hash rejection.
- Shared-GEMM transcription was tested against the **actual frozen native
  branch's `_oracle` closure**, extracted from its source, including masks/sinks.
  Additional tests distinguish missing masks/sinks, cover zero/single/multiple
  visible keys and reject invalid or mismatched replay evidence.

### Measured paths

All operands and outputs remain FP32. Masks are bool. Times below include the
preparation specific to each alternative:

- **Online native:** current serving masked kernel.
- **Shared GEMM:** existing native branch's FP32 shared-KV oracle, using two
  matrix multiplications, masked softmax and per-head sink.
- **Shared GEMM/chunk32:** same calculation in32-query chunks, with output
  concatenation included.
- **Gathered donor:** queries grouped by equal visible-key count, ascending key
  selection, gather at most16 queries per group, then the existing native FP32
  BMM/fused-softmax donor. No padded logits. Both libm and hand-exp variants.

Upstream common FP32 preparation is unchanged and outside every arm. Selection,
gathers, masks, softmax and output assembly needed by an arm are inside timing.
This is three alternating trials in one process after one warm call per arm,
on reused input buffers -- not a streaming roofline or independent-process
performance certificate.

| Layer type | Online native ms | Shared GEMM ms | Shared GEMM/chunk32 ms | Gathered donor ms | Gathered hand-exp donor ms |
|---|---:|---:|---:|---:|---:|
| SWA L0 |12.131|19.294|25.698|15.631|15.405|
| **C4 L2** |**73.789**|**37.073**|**33.798**|54.022|52.997|
| C128 L3 |14.081|20.538|26.717|17.117|16.319|

Values are medians. At C4, native min-max is73.763-74.130ms; shared GEMM is
36.090-38.689ms; chunk32 is33.718-34.254ms; gathered hand-exp is52.488-54.301ms.
All samples and ranges for all rows remain in JSON.

**C4 winner: 2.183x call speedup, 54.2% lower latency**, using existing FP32
shared GEMMs with32-query chunks. Unchunked shared GEMM also wins1.990x.
The existing gathered hand-exp donor wins only1.392x after adaptation costs.
All four alternatives lose on the captured SWA and C128 calls: **reject an
unconditional attention replacement**.

### Numerical result is separate from speed

At C4, shared GEMM/chunk32 matches the frozen shared-GEMM oracle's output bytes.
It does **not** match the accuracy-approved online kernel's output bytes:

- Maximum absolute difference: `7.3909759521484375e-6`.
- Relative L2 difference: `6.73666755227963e-7`.
- Element agreement after BF16 rounding: `99.9642014503479%`.

Chunked and unchunked shared GEMM match each other at all three captured
coordinates. These small differences do not establish downstream logit/token
equivalence or authorize a tolerance. The candidate remains **numerically
unapproved for serving**. No accuracy/default promotion.

### Grounded conditional phase upside

The clean long-prefill events1-4 from job401206 attribute:

| Native sparse family | Share of instrumented prefill forward time |
|---|---:|
| SWA |0.4568%|
| C4 |19.3990%|
| C128 |4.9511%|

Use the **C4 share**, not the whole24.81% sparse share, since the other families
regress. If the measured2.183x call speedup held across the C4 calls:

`time_saved = 0.1939904826 * (1 - 1/2.1832175104) = 10.51%`

`phase_speedup = 1/(1 - time_saved) = 1.1175x`.

That is approximately **10.5% less prefill time / 11.7% higher prefill throughput**,
conditionally. The unchunked shared-GEMM path implies9.65% less phase time;
the gathered hand-exp donor only5.47%. These are planning estimates, **not
measured full-phase improvements**. They combine earlier-chunk attribution
on a different node instance with final-chunk replay of one C4 layer; density,
history and layer-dependent numerics still need coverage. They establish no
decode improvement and no whole-request gain without a prefill/decode mix.

### Relation to the published standalone sparse results

The earlier [section3.4](../kernel-pilot-performance.md#34-sparse-attend--onednn-donor--fused-softmax--the-compute-bound-one)
measured a gathered `[N,K,D]` donor path. The current serving path instead uses
shared `[K,D]`, per-query masks and an online kernel. Its performance conclusion
cannot be transferred across those entry points. This result is consistent with
the earlier advice to use FP32 GEMM where appropriate, **not** a renewed BF16
claim. It also shows why the published donor must be measured with real serving
gather/selection costs rather than importing its standalone speedup.

**Disposition:** advance the C4 long-prefill shared-GEMM candidate to bounded
numerical and matched-forward verification; retain current SWA/C128/decode paths.
Do not yet hard-code a layer or shape threshold from one sample, claim a
best-possible kernel, or run a broad task evaluation. Existing optimized native
GEMMs establish the opportunity; whether a native masked/fused wrapper improves
it further is unmeasured.

Driver:
[`bench_long_prefill_sparse.py`](../../plugin/validate/bench_long_prefill_sparse.py).
Protocol:
[`serving-screen.md`](../../plugin/validate/serving-screen.md).

## 9. Integrated C4 trial 401627: measurable saving, numerical promotion blocked

The bounded trial completed with exit0 and both analysis/job completion markers.
It used one resident load, baseline A / candidate / baseline B, the same
B1/S3072/G2 prompt and fixed32 EMR configuration. Only candidate C4 prefill
dispatch changed. **The saving survives model-forward integration, but this
formulation is not numerically equivalent and is not promoted.**

### Provenance and controls

- Job401627, node `pcl-sprh01.sc.intel.com`, worker PID3350482.
- Snapshot `/scratch/bkaul/dsv4-serving-screen/c4-trial-20261111-BRryag/source`;
  source-manifest SHA256
  `f97a1509442ba97e1107de91459c4c5bebd0227d5080d28202a206535c49850a`.
- Analysis `/scratch/bkaul/dsv4-serving-screen/job-401627/results/trial-analysis.json`;
  SHA256 `60d521348e3e1276e293541715212f17e8eb95e06514c1022ab2aa73bac6e35e`.
- Native-binary manifest SHA256
  `9aacea6d2c3c060fa276423864eab2bb413807dc6de9e19f77fdf5f061f8cf65`.
- 21 forwards: six512-token prefill chunks and one decode forward per arm.
  Exactly126 candidate calls (21 C4 layers x six prefill chunks); SWA/C128
  and decode dispatch remained native online.
- Scheduler flush and plugin-history reset succeeded. Allocator/position/input
  schedules, sparse shapes and layer0 input anchors matched. No candidate mask
  changed in this sample.
- Baseline A/B **captured last-row hidden tensors at all43 layers, full logits,
  and generated token IDs matched byte-for-byte**. This is sampled hidden-state
  repeatability, not comparison of every hidden token.
- The analyzer checked event, response and tensor identities/hashes; those hashes
  were checked again when publishing this disposition. Nineteen targeted tests
  passed before submission, including invalid evidence and failure propagation.

### Measured integration result

The primary estimator is the **sum of five matched prefill-forward times per
arm**, excluding the first chunk from every arm because baseline A compiled
native extensions there. These are one sequential sandwich, not independent
replicas or a median. Symmetric tracing/hash/clone costs remain inside timing;
tensor serialization is outside.

| Timed scope, five chunks | Baseline A | Candidate | Baseline B |
|---|---:|---:|---:|
| Whole model forward | 25.8243s | 23.4551s | 25.4918s |
| C4 sparse boundaries | 5.4582s | 3.2195s | 5.4598s |

Candidate forward time is **9.17% lower than A and7.99% lower than B**
(1.1010x and1.0868x). Baseline B is1.29% faster than A.
The C4 boundaries fall about41.0% (1.695x), saving2.239-2.240s; the whole
forward saves2.037-2.369s. Thus the local sparse saving largely survives the
rest of the model, rather than disappearing into preparation/framework cost.

Supplementary full-prefill comparison: candidate's six warm chunks total
27.8290s versus trailing baseline B's29.9473s, **7.07% lower /1.0761x**.
There is only one fully warm baseline for that comparison: baseline A's first
chunk includes compilation. Do not mix its174.1055s full-prefill total into a
speedup claim. Neither comparison is client latency, replicated throughput,
or a result over B1-64.

The earlier replay's2.183x sparse speedup and10.51% conditional phase saving
were not achieved across this integrated scope: the measured sparse ratio is
1.695x and primary forward saving7.99-9.17%. The replay was one layer at the
final chunk; integration spans all21 C4 layers and five chunk histories.
This corrects the projection's scope without declaring either measurement invalid.

### Numerical consequence: small isolated error did not stay small downstream

All three arms generated `[4086, 304]`, and candidate/baseline argmax matched
at every captured logit boundary. That is **not task-accuracy acceptance**.

| Boundary | Candidate logit max absolute difference | Relative L2 difference |
|---|---:|---:|
| Final prefill, S3072 | 0.765625 | 2.485% |
| Following decode, S3073 | 0.859375 | 2.360% |

Across all seven forwards, logit relative L2 differences range2.360-5.026%,
with maximum absolute difference1.65625. Final-layer last-row hidden relative
L2 differs3.141% at final prefill and4.315% at decode; some intermediate-layer
samples differ more. Masks remained unchanged, so changed sparse selection
is not required for this downstream divergence. The run does not isolate
which later quantization/routing/reduction boundaries amplify the initial error.

In particular, unchanged decode **dispatch** does not imply unchanged decode
**state**: it consumes history produced during the changed prefill. The decode
logit difference is observed; no decode speedup or accuracy claim is made.

### Disposition and claim-evidence ledger

| Claim | Evidence/estimator | Falsifier or unresolved limitation | Verdict |
|---|---|---|---|
| C4 composition can lower model-forward time here | Job401627, five-chunk sums against both controls | Baseline instability, broader shapes, independent replicas | Scoped diagnostic saving |
| Integrated saving exceeds10% reliably | Observed time reduction7.99-9.17%; full warm prefill7.07% | Already not supported by this run | Not established |
| Tiny sparse output error is harmless end-to-end | Job401625 isolated error versus job401627 downstream logits | Observed percent-level logit divergence | Inference rejected |
| Matching two tokens proves accuracy | One prompt, G2 | Task-level numerical acceptance absent | Rejected |
| Candidate is ready as native/default implementation | Python composition of existing native GEMMs; no approved numerical gate | Arithmetic acceptance and native implementation unresolved | Blocked |

Stop promotion of this formulation. Do not launch a broad accuracy campaign or
change defaults from this result. A future native optimization must either
preserve the approved numerical behavior or obtain an explicitly reviewed
accuracy budget; faster GEMMs alone do not satisfy the original requirement.
The diagnostic is complete and no additional job is pending.

## 10. Authorized full C4 accuracy and matched performance wave (401687)

Completed 2026-11-11. This subsequent user-authorized experiment supersedes
section 9's decision not to launch an accuracy campaign; it does not change
production defaults. All six resident workers exited zero, with complete
coverage of 1,311 paired questions across 16 logical shards.

| Run | Correct / 1,311 | Invalid |
|---|---:|---:|
| Current fixed32 native baseline | 1,245 (94.97%) | 0 |
| C4 candidate | 1,252 (95.50%) | 0 |
| Historical accuracy-closure CPU | 1,246 (95.04%) | 0 |
| Historical accuracy-closure GPU | 1,245 (94.97%) | 0 |

Candidate versus the fresh baseline: 14 recoveries, 7 regressions, net +7.
The historical corpus was re-audited and compared by identical question IDs;
its CPU runtime used fixed8/sticky/passive rather than this wave's
fixed32/nonsticky/active configuration. The declared task criterion is met:
**NO_OBSERVED_LOSS**, not statistical superiority, bit-equivalence or automatic
production promotion.

### Performance: a long-prefill gain, but short-prefill batch regressions

The same six resident processes each ran a warm baseline/candidate pair and
one measured pair at each point, with order varied across workers. All probes
generated five tokens with EOS ignored (four decode steps). Values below are
the **median of six paired percentage time changes**, with min/max, not the
ratio of the two marginal median times. Negative means less time.

| Point | Prefill model-forward time change | Decode model-forward time change | Complete request wall-time change |
|---|---:|---:|---:|
| B1 / S3072 | -7.82% [-8.76, -7.11] | +0.41% [-0.69, +5.50] | -7.60% [-8.35, -6.90] |
| B32 / S128 | +18.92% [+18.06, +23.78] | +0.24% [-2.96, +2.67] | +17.13% [+16.32, +21.53] |
| B64 / S128 | +53.03% [+45.16, +55.92] | +0.28% [-3.81, +3.40] | +48.66% [+41.91, +51.31] |

For orientation, marginal median request times (baseline -> candidate) were
30.622 -> 28.191 seconds, 35.741 -> 41.786 seconds, and 75.553 -> 111.265 seconds,
respectively. These marginal medians are not the paired-percentage estimator.
Phase timings are instrumented model-forward sums, not TTFT/ITL. The four-step
decode window does not cover a full C128 cycle or establish steady-state decode.

**Disposition:** the tested C4 formulation meets the declared task-accuracy
criterion and saves time at the measured long-prefill point, but is not a
general replacement: both short-prefill batched points regress in every
process pair. These three points do not establish a general dispatch threshold;
the subsequent crossover screen and final no-integration decision are below.
The formulation remains Python composition of native GEMMs, not a newly
authored fused C++ kernel.

Evidence root: `/scratch/bkaul/dsv4-c4-accuracy/wave-20261110-wJzAC6`.
The persisted `summary.json` was reproduced through the frozen driver's full
final validator after checking all six explicit exit files.

- Summary SHA256: `7a252d26b68a56ff3155a19fd81c2d2270f607e4d3c19b2fb4d6ed67151e93bc`.
- Source manifest SHA256: `d71a92e508c4954028597b3b7123de5e8d0246c6047b7b7e4d6d015b0cb34e0c`.
- Plan SHA256: `2294706f76fd5f61371f0951fd868e493fbca9d959aa0ab2f84dca0f369e7086`.
- Aggregation: frozen `source/plugin/validate/c4_accuracy_wave.py`,
  `summarize(root, final=True)`. Persisted JSON normalizes Python schedule
  tuples to lists; JSON-normalized recomputation matches the saved artifact.

### Archived crossover screen (401721): no production integration

The user initially authorized cheap numerical/crossover checks, a native port,
conditional dispatch and a small paired task screen. After the crossover screen,
the user chose to retain the measurements without proceeding with integration.
No native port, production selector or small paired task screen was implemented.
The existing production implementation and defaults remain unchanged.

The three end-to-end points above confound batch and prompt length. A batch-only
rule is therefore unsupported. On the pinned C4 capture, `N=512`, `H=64`,
`D=512`, shared `K=1536`; each query sees 768--896 keys (mean 831.75, density
0.54150390625). The shared GEMM computes masked-out scores too. Its two GEMMs
cost approximately `4*N*H*D*K` FLOPs; the existing online path costs approximately
`5*H*D*sum(visible_keys)` plus exponentials, recurrence and mask scans. These
are FP32 paths, not AMX BF16. FLOP counts alone do not establish the bottleneck.

[screen_c4_crossover.py](../../plugin/validate/screen_c4_crossover.py) screens:

- Query counts 1, 2, 4, 8, 16, 32, 64, 128, 256, 512.
- Shared key counts 128, 512, 1024, 1536, 3072.
- Visible fractions 1, 1/2, 1/4, 1/8, 1/16, with block and strided masks
  (the identical all-visible pattern is measured once): 450 coordinates.
- Three separate fixed32 EMR processes, three order-varied trials per arm,
  both buffer-reuse and LLC-evicted timing. Eviction reads/modifies a buffer
  larger than twice the reported local LLC.
- Exact baseline replay of the original capture before synthetic geometry.
  Synthetic cases repeat/slice actual q/KV values; they are not serving batch
  measurements. Independent FP64 checks sample first/middle/last query rows,
  and every repeated output is checked against its arm's initial bytes.
  The `1e-4` relative-L2 corruption screen is not production arithmetic acceptance
  or a substitute for source/GPU precision semantics.
- Per-process FP32 GEMM and streaming read-modify-write calibration proxies.
  These are not ISA peaks or a universal DRAM ceiling.

The following was the screen's original decision plan, not authorization to
continue after the final disposition below.

| Outcome | Original proposed disposition |
|---|---|
| Replay, finite-output, coordinate, or repeatability failure | Reject; retain partial diagnostics, no completion record |
| GEMM wins only at sufficient mask density/work | Qualify that region for a native port; remeasure the native implementation before choosing a selector |
| Small or sparse shapes regress | Keep existing online path; do not generalize from batch 1 |
| Timing changes materially across processes or cache modes | Investigate before claiming a dispatch boundary |
| Numerical screen succeeds | Proceed to native-specific boundary checks, not task-accuracy certification |

Frozen root: `/scratch/bkaul/dsv4-c4-crossover/run-20261111-IkZodq`.
Source manifest SHA256:
`a195a5a67105eac5b31970c4dddbb66abafe5fad6dba66503ea0c0c1f8c6c0a2`.
Job401721 has an explicit exit file and per-process completion records; queue
disappearance is not success. No weights are loaded. The GNR swap investigation
remains paused.

**Completed:** all three processes emitted 450 synthetic coordinates plus the
real capture, with explicit job exit zero. The reconciler checks coordinate,
process, source/build/runtime and timing identities before aggregation.
Twelve targeted crossover/numerical/replay tests pass, including missing
coordinates, relabelled shapes, duplicate process identities, invalid outputs,
non-repeatability and evaluated-builder drift.

The captured C4 result is **1.789x** LLC-evicted speedup, range
**1.772--1.955x**, using the median/min/max of three process ratios of arm
medians (three alternating trials per arm). The marginal median times are
76.789 ms online and 42.926 ms Python GEMM. Reused-buffer speedup is 1.792x,
range 1.768--1.944x. This corroborates an isolated win, but does not reproduce
the earlier single-process 2.183x magnitude; keep the run identities and
estimators separate.

Mask density matters, but density alone is also insufficient. Among the tested
half-visible geometries, the **minimum process speedup across both mask patterns
and both cache modes** was:

| Queries N | K512 | K1024 | K1536 | K3072 |
|---|---:|---:|---:|---:|
| 32 | 1.533x | 1.965x | 1.999x | 2.810x |
| 64 | 1.680x | 2.113x | 2.013x | 2.749x |
| 128 | 1.666x | 2.183x | 2.190x | 2.827x |
| 256 | **0.919x** | 1.564x | 1.819x | 2.768x |
| 512 | **0.913x** | 1.582x | 1.735x | 2.825x |

Thus even `N>=32, density>=1/2, K>=512` would admit observed regressions.
The tested `K>=1024` points are a more conservative native-port starting region,
not evidence for every intervening shape or a finalized production selector.
At 1/16 visibility every tested coordinate's median lost; at 1/4 visibility
results are mixed. With integration declined, retain the existing production
path for every region.

Worst sampled relative-L2 versus gathered FP64, over all three processes and
all screened inputs, was `1.179e-6` online and `3.073e-7` Python GEMM.
This checks sampled mathematical accuracy, not all-output/GPU equivalence or
task accuracy of an as-yet-unwritten native port.

**Feasibility:** the real capture needs approximately 103.08 GFLOPs for the
two dense FP32 GEMMs versus 69.77 GFLOPs of main online arithmetic (excluding
exponentials/recurrence bookkeeping). The three 4096-square FP32 GEMM calibration
medians are 4.935, 5.059 and 5.077 TF/s. Using their median gives an optimistic
20.37 ms compute-only proxy for the dense work, not an achievable whole-kernel
target. The input/output minimum footprint is about 131.75 MiB; the chunk32
score buffer is about 12 MiB before a sink column. Reuse, softmax and allocation
traffic prevent treating this footprint as actual DRAM bytes. The optimization
direction is better FP32 GEMM reuse plus fused preparation/output assembly,
not lower-precision AMX or an unmeasured memory-bottleneck assertion.

**Final disposition (2026-11-11): record findings; do not integrate.** Retain
the accuracy, end-to-end performance and crossover evidence, including losses.
The demonstrated long-prefill benefit does not justify the native-port,
dispatch and validation work for the current priorities. This is a
geometry-dependent optimization, not proof that only batch 1 can benefit:
batch and prompt length were confounded in the end-to-end measurements, and
operator query counts are not request batch sizes.

No native port, conditional production dispatch, or additional small/full
accuracy run will proceed under this plan. No production defaults change.
Reconsider only with renewed user direction and a workload that makes the
long-prefill benefit worth the integration cost; the experimental candidate's
accuracy result would not certify a future native implementation.

Summary: `run-20261111-IkZodq/summary.json`, SHA256
`297fddee1489b9446b07ca313696d531f288b0a1714f7e1fc1ccaca4ec031cdd`.
The frozen post-run reconciler is
`run-20261111-IkZodq/analysis/aggregate_crossover.py`; invoke its
`summarize(root, "401721", source_manifest_sha256)` to reproduce the summary.
The summary stamps this reconciler's own hash separately from the benchmark
source manifest. Raw per-process samples remain alongside the summary.
