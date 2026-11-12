# Provisional serving opportunity screen

**Campaign disposition (2026-11-11):** the protocols below are retained for
reproduction, not instructions to launch another experiment. The completed
C4 accuracy/performance and crossover results are in
[section 10 of the findings](../../docs/dev/DSv4-serving-screen-401145-findings.md#10-authorized-full-c4-accuracy-and-matched-performance-wave-401687).
The user chose to record them without native integration, conditional dispatch
or further accuracy screens. GNR measurement is a separate capacity/performance
task, not a reopening of C4 integration.

This is instrumentation, not an optimization or an accuracy certificate.
Job401145's [partial findings and dispositions](../../docs/dev/DSv4-serving-screen-401145-findings.md)
retain three completed points and exclude the parity-failed B32/long-context
pair from performance conclusions. Do not rerun this full-request protocol as
the default attribution method; use representative forwards/operator replay.
The [B1-64 analytical coverage map](../../docs/dev/DSv4-batch-regime-analysis.md)
defines which shapes/dispatch boundaries need additional measurements. This
B1/B32 screen is not a substitute for that coverage; B64 at context3072 also
requires a different token-capacity configuration or a matched shorter-context panel.

## Bounded follow-up: meaningful iterations, not another benchmark

Pass `meaningful` as the launcher's second argument. It makes **one request**
each at B1/S3072, B32/S128 and B64/S128, with five output tokens (four decode
forwards), on one resident load. No warmup requests, repeated pairs, task
evaluation, or B32/B64 long-context generation. B denotes requests, not prefill
tokens per forward. The B64 point extends coverage, not an optimality claim.

The lightweight observer disables the projection TorchDispatchMode and adds
boundaries around routed/shared MoE, dense methods, MHC, norms, attention and
logits, plus compressed-history accumulation/publication. It retains nested
exclusive accounting. Capture the **last prefill** and **first decode** calls at
layers 0/2/3 (SWA/C4/C128), preserving actual q/KV/mask/sink and native outputs.
Cloning perturbs those whole forwards: exclude them from attribution, along with
the first prefill at each point (possible JIT/allocator transients). Serialization
is outside the forward timer. Remaining decode steps also encounter the next C4
completion, but not the next C128 completion.

After shutdown, replay those calls without loading weights. Check original replay
against the captured FP32 output bytes. A mask-union compaction diagnostic removes
only keys invisible to **every** query, preserving visible-key order; test output
bytes and measure both prepared and preparation-inclusive calls. An all-masked
control retains the scan/allocation/dispatch shape but removes visible arithmetic;
its changed branch behavior means it is not an exact scan-cost subtraction.

| Question | Evidence | Disposition |
|---|---|---|
| Sparse scan versus visible arithmetic | Actual per-query visibility, union density, original replay, all-masked diagnostic | Large masked-only cost warrants index-scan investigation; otherwise inspect visible-key dot/exp/value recurrence. |
| Dead pool rows are removable cheaply | Original vs union-compacted replay, including mask selection and gathers, exact output bytes | Advance only a material preparation-inclusive result; prepared-only wins are not serving gains. |
| Decode remainder is MoE/dense/MHC or glue | Nested boundary times across all 43 layers | Rank measured scopes; do not label untimed remainder framework overhead. |
| Compression cost is history or producer | Accumulator, pool, overlap, FP8 roundtrip and publish scopes | Rank exclusive costs before any producer rewrite. |
| Capture/replay remains faithful | Input/output hashes, expected case/layer/forward coordinates, original native byte replay | Any mismatch fails the run; no success marker or promotion. |

These are **diagnostic attribution** samples: no matched off/on perturbation test,
generation-repeatability certificate, clean-reset batch-scaling comparison or
independent-process performance claim. Sparse replay uses three inexpensive
alternating trials on reused buffers, not three full-model requests. Runtime
flags and accuracy-approved computation remain unchanged.

### Compression batching decision screen (no model load)

`run_compression_batch.sbatch` runs `bench_compression_batch.py` on a frozen
snapshot at the same fixed32 EMR configuration. It uses **synthetic signed
projection rows and shape-faithful carry/history**, not captured model tensors.
The experimental adapter is confined to validation code; serving is unchanged.
It batches ready windows through the existing `_compress_prefill_core`, preserving
request boundaries via its `ext` lengths. Pooling still loops by request, but
norm/RoPE/FP8 work shares one batched call. This tests scheduling overhead before
committing to a new native producer.

Each arm includes carry append, window selection, batching preparation, the actual
producer, history update and production history publication. Restoring identical
initial state and checking output/carry/history/handoff bytes are outside timing.
Test B1/32/64, C4/C128, histories128/3072 and ordinary/completion steps; three
alternating reused-buffer trials follow one warm call per arm. This is not a
DRAM-roofline experiment or end-to-end benchmark.

| Result | Decision |
|---|---|
| Output, carry, history or handoff differs | Hard failure; no completed result or serving change |
| Batching is neutral/loss after preparation | Reject this batching mechanism; no model rerun for it |
| Completion wins but ordinary steps regress | Account for both at 1/4 and 1/128 cadence before proceeding |
| Meaningful cadence-weighted synthetic saving | Candidate for matched real-input/whole-forward verification; not a model gain |

The conditional extrapolation uses21 C4 and20 C128 layers, including ordinary
steps at `(r-1)/r` and completion at `1/r`. It cannot model variation among real
layers, growing history, cache contention or interference with other inference
operators. The measured completed-window spike alone must not become an
every-token speedup claim.

### Long-prefill sparse donor screen (actual captures, no model load)

`run_long_prefill_sparse.sbatch` evaluates the final B1/S3072 prefill chunk
(M512) at layers0/2/3 using the actual job401206 q/KV/mask/sink/output captures.
Hashes and exact worker coordinates are pinned, and original native replay must
match captured output bytes. Reuse the hash-bound original native binary on the
same EMR ISA class; no build or serving change is necessary.

Arms: current online native kernel, the native branch's existing FP32 shared-KV
GEMM oracle, that oracle in32-query chunks, and the published gathered FP32
GEMM/fused-softmax donors with libm/hand-exp variants. The gathered adapter groups
queries with equal visible-key counts, preserves ascending key order and limits
each gather to16 queries. No padding logits or BF16 operand conversions. All
selection/gather/mask/softmax/output assembly is included in timing.

Three alternating reused-buffer trials follow one warm call per arm. Numerical
metrics compare each output both to the captured online result and the shared
GEMM oracle. Nonfinite, wrong-shaped, nonrepeatable outputs or an original replay
mismatch fail; **small numerical errors do not authorize a new acceptance
tolerance**. Any non-byte-identical candidate remains unapproved for serving.
Hand-exp is separately labeled; FP32 does not imply identical reduction order.

| Result | Disposition |
|---|---|
| GEMM paths lose including preparation | Reject these adapters; do not infer a better native masked donor is impossible |
| Chunked shared GEMM wins | Investigate materialization/dispatch economics before authoring; preserve masks/sinks |
| Gathered donor wins only with preparation excluded | Not a serving win (this screen includes preparation) |
| Material candidate win with numerical differences | Report speed and errors separately; preserve correctness/promotion gate |
| Small local win | Apply measured phase share before spending a model load |

These are three layer-type representatives at one final chunk, not all layers,
chunks or contexts. Conditional phase upside is not an end-to-end measurement.

### Bounded integrated C4 trial (one load)

Pass `c4-trial` as the launcher's second argument. Run the same B1/S3072/G2
prompt in baseline A, candidate, baseline B order. Before each arm require a
successful scheduler flush; on its first position-zero prefill clear the plugin
KV, compressed-history, carry, handoff and selection maps. Keep deterministic
Hadamard and compiled-kernel caches warm. No full benchmark or task evaluation.

Only the candidate arm's actual ratio4 prefill calls use the FP32 shared-GEMM
composition in32-query chunks. SWA, C128 and decode retain the native online
dispatch. This is diagnostic Python composition of existing native GEMMs,
not the requested final C++ serving implementation or an approved math change.

Record all43 sparse calls and layer last-row outputs at each forward, plus full
next-token logits when present. Match chunk/position/input/cache-slot schedules,
layer0 q/KV anchors, shapes and baseline masks; report candidate mask changes as
downstream numerical consequences. Require baseline A/B sampled hidden/logit
bytes and output tokens to repeat. Check greedy token IDs against captured logits.
The six512-token prefill chunks and one decode forward per arm are a bounded
schedule, not independent performance replicas.

Exclude the first prefill chunk from **every** timing arm, because baseline A
can compile native extensions there. Sum the remaining five whole-forward
times and C4 sparse boundaries; compare candidate with each baseline separately.
Input hashing and last-row cloning inside the timer are symmetric but not free.
Serialization and numerical checks are outside the timer. No clean full-request
latency, stationary decode throughput, or >10% certified gain claim follows.

| Observed outcome | Disposition |
|---|---|
| Invalid identity, coverage, coordinates, nonfinite output or illegal dispatch | Fail closed; do not write an accepted analysis |
| Baseline sampled outputs do not repeat | Retain diagnostic failure and withhold timing ratios |
| Kernel saving disappears in whole-forward timing | Reject this composition as demonstrated model-level upside |
| Candidate is faster than both controls, stable control timings | Retain scoped evidence for a native implementation decision; not replicated certification |
| Candidate changes hidden/logit values, masks or tokens | Publish the differences; no invented accuracy tolerance and no default promotion |
| Decode dispatch is unchanged | Check its outputs, but do not claim matched decode state or speedup after changed prefill |

`analyze_sparse_trial.py` writes `trial-analysis.json` only after structural and
finite-value validation. A valid diagnostic remains promotion-blocked even if
tokens match. Baseline-repeatability failure writes an explicit blocked report
with null speedup ratios and exits nonzero. The launcher propagates failures.

### C4 numerical decision: matched CPU/GPU operator checks first

`check_c4_numerics.py` reuses the hash-pinned job401206 final-prefill captures,
without loading weights. CPU and single-H200 jobs run independently. CPU replays
the approved native output bytes and the exact trial formulation, then evaluates
a separately expressed FP64 gathered-key reference with an explicit sink
denominator. FP64 tests mathematical rounding error, not the published GPU
blockwise precision contract.

The GPU job uses the exact image ID retained by the accuracy closure, runs its
real `flash_mla_sparse_fwd` and the model authors' real TileLang `sparse_attn`,
and saves outputs. The image's dsv4 backend defaults to sparse prefill via
`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=True`; its `auto` prefill setting does not
enable Q8KV8. Each CPU mask becomes ascending visible indices with -1 padding
and explicit lengths. Q/KV must survive BF16 conversion exactly; the learned
sink remains FP32. Both 16-head groups (TP4-local geometry, no distributed
execution) and64-head calls are checked twice for output repeatability.

Report CPU FP32 versus GPU BF16, and both rounded to the common BF16 boundary,
separately. Never call FP32-to-BF16 differences an operator-contract bug without
accounting for the GPU's blockwise probability rounding and output cast.

| Result | Disposition |
|---|---|
| Bad evidence, changed operands, invalid GPU output or native replay failure | Fail closed; do not aggregate |
| Candidate disagrees with masking/sink mathematical semantics | Fix or reject before any model accuracy run |
| Candidate has small mathematical error and comparable GPU distance | Operator evidence supports task evaluation, not automatic acceptance |
| CPU candidate and baseline are nearly indistinguishable versus GPU, but model logits differ | The remaining question is task accuracy; stop local precision diagnostics |
| GPU head-grouping changes outputs | Retain actual TP4-local reference separately; do not pool geometries |

This cannot establish B1-64 coverage or task accuracy. The completed full GSM8K
CPU/GPU campaign is the existing task comparator, not a generic numerical
tolerance for future variants. A subsequent model test needs an explicit
accuracy disposition rule; no threshold is inferred from this replay's errors.

Job401642 failed before GPU computation: CUDA initialization failed inside its
H200 container, and subsequent allocated host-level `cuInit` also returned3.
An allocated single-GPU H100 NVL node returned0. The fallback GPU replay uses
the same pinned image and SM90 kernel family on that H100, not an H200 output
reproduction. This partition has one GPU per node and no GRES; allocate an
exclusive node and verify the sole GPU UUID rather than selecting another user's
device. CPU job401641 remains usable independently; it completed successfully.

Source: the frozen CPU accuracy campaign at
`/scratch/bkaul/dsv4_latest_fullacc_20261107/source`, manifest SHA256
`ab7753eae9f726c55708becf29d9b2815ff29457fa4098f2249fc0f39bdd1c29`.
The current isolated branch is `dsv4-perf-main`, based on published main commit
`34fe50f1aca4dc9d1ff17111a38ff5f323d7b106`. Its 18 model-runtime files and 10
kernel/header files match the accuracy snapshot. Both routed and shared clamps
and CHUNK_ACCUM are enabled. Default promotion remains separate from the
[task-accuracy closure](../../docs/dev/DSv4-accuracy-closure.md).

## What this measures

### Matched C4 accuracy and bounded performance wave

`c4_accuracy_wave.py` and `run_c4_accuracy_wave.sbatch` run six exclusive EMR
workers, with two concurrent model loaders admitted by shared atomic directory
slots. Each worker loads once and handles two or three of sixteen logical GSM8K
shards. The full 1311-question, eight-shot protocol is evaluated in both arms:
approved online sparse attention versus FP32 shared-GEMM C4-prefill only.
SWA, C128 and decode retain the native path. Mixed chunking is disabled.
Scheduler caches and plugin inference histories reset before every arm.
Both arms use fixed32 active-spin runtime, not the historical fixed8 runtime.

Each resident worker also executes B1/S3072, B32/S128 and B64/S128 with exactly
five generated tokens and EOS ignored. Both arms are warmed separately at each
point, then one measured pair is collected; order alternates across workers.
This yields six independent process pairs per point without additional model
loads. Reports retain each pair and median/min/max of its time reduction.
Prefill and decode times are instrumented model-forward sums, not serving TTFT
or inter-token latency; client wall time includes framework overhead. Four decode
steps sample C4 completion but not a full C128 cycle. These bounded probes do not
establish long steady-state decode throughput or B1-64 universal optimality.
Variable-length accuracy generations are not pooled into controlled speedups.

Frozen source, dataset, prompts, native library, runtime environment and array
identity bind the evidence. Sparse coverage, candidate gating, generated work,
paired first-input/cache anchors and performance forward schedules are checked.
Malformed evidence is fatal; warmups never enter performance aggregates.

| Result | Disposition |
|---|---|
| All 1311 pairs, no observed aggregate loss, zero invalid answers | Task screen supports further consideration; not statistical noninferiority or automatic promotion |
| Valid score loss, invalid answers, or measured performance loss | Retain and report loss; do not promote on speed alone |
| Failed worker or malformed evidence | Stop the wave; do not publish successful completion |
| At least 160 pairs with a deficit of 5 percentage points, or over 5% invalid after 160 answers | Catastrophic early stop, not the acceptance tolerance |

Freeze the tested source and its manifest, then run `prepare --root ROOT`.
Submit the frozen launcher with `sbatch --parsable --array=0-5 ... ROOT`,
save its numeric array ID in `ROOT/array-job.txt`, and immediately run the
driver's `watch --root ROOT --job ARRAY_ID` plus the completion waiter.
`summary --root ROOT` requires all six explicit successful exits, complete
question coverage and complete performance pairs. Queue disappearance is not
success. Defaults and historical accuracy artifacts remain unchanged.

One real-weight resident load, batches 1/32, input lengths 128/3072, eight output
tokens, chunked prefill 512. Prompts are recorded exact-length token prefixes of
GSM8K examples, not random-vocabulary inputs; this is not a task-accuracy run.
Warm both instrumentation arms at each point, then three repeated on/off pairs
with alternating order. Worker control is a file read before each model forward.
The original compute and output are unchanged; there is no BF16 attention swap.

The runtime deliberately differs from the parallel accuracy run: warm finite
spin, a fixed 32-thread pool and sticky=0, versus passive fixed-eight sticky=1.
This tests the existing optimized runtime rather than advertising recovery of
already-known cold-runtime overhead. Kernel/math flags match the frozen accuracy
launch. Pin to EMR CPU 0-31 / NUMA 0 on an exclusive allocation. This is a bounded
screening configuration, not a claim that 32 is optimal at every batch/phase.
Validate 32 threads before/after forwards and inside every observed decoder layer.
Abort after a returned forward exceeds 180 seconds; the engine watchdog is 300
seconds so a stuck first request cannot consume the entire allocation unnoticed.

## Rejected first screen: job 401095

The original screen used 58 prefill threads with warm spin and adaptive decode.
It completed only the two warmups and one measured off/on pair at B1/S128/G8
before cancellation. The measured prefill forward totals were 1777.166s off and
1777.575s on; seven decode forwards totaled 1.316s off and 1.435s on.
These are individual request observations, not replicated medians. Decode
instrumentation changed this pair by about 9%, so its detailed split is diagnostic.
No other workload point completed; no final result was produced.

The prior decode operating-point document already warned that 58 warm threads
thrash during prefill. Reusing its adaptive decode recommendation while restoring
58 threads at every prefill was a harness error, not evidence that native kernels
need rewriting. A read-only process snapshot found 65 threads pinned to CPU0,
174 allowed on 0-57, and 23 pinned individually to CPUs1-23; only the torch
libgomp runtime was mapped. This supports investigating pool/affinity interactions,
but does not by itself establish the cause. The new fixed-32 setup also removes
per-phase pool resizing, so any recovery cannot isolate thread count from resizing.

Retain raw evidence at `/scratch/bkaul/dsv4-serving-screen/job-401095/results`.
Do not count recovery from this rejected setup as a new kernel/fusion gain.

| Hypothesis | Instruments | Disposition |
|---|---|---|
| Sparse serving implementation/preparation dominates | Whole batched pipeline, gather, mask construction, assembly, common preparation, FP32 native preparation, actual native wrapper, shapes/dtypes/strides | Compare core vs preparation share. Existing standalone BF16 result is not imported as a serving speedup; first reconcile source precision/mask/sink contracts. |
| Indexer replicates shared keys unnecessarily | Expansion wall time and input shape/bytes, total indexer, native logits/top-k wrappers | If absent at a phase/shape, say absent. If ideal removal cannot reach materiality, stop for that point. Do not equate allocation bytes with DRAM traffic. |
| Compressor projection casts consume material time | Whole compute_kv_score, ATen dispatch observer recording casts, transpose and mm independently; ratio and indexer/main tag | Native GEMM versus preparation split selects whether to investigate packing/conversion. Preserve FP32 output and accepted intermediate boundaries. |
| Instrumentation itself distorts cost | Uninstrumented and instrumented per-phase forward totals on the same recorded inputs, actual generated-token equality, same resident process | More than 5% median shift marks attribution diagnostic-only. Never call it a measured removable cost without qualification. |

Nested scopes record inclusive and exclusive nanoseconds; do not sum overlapping
parents. Reports retain layer IDs, actual token counts and sequence lengths,
worker PID/sequence, source hashes and generated tokens. Reported ideal-removal
and hypothetical 2x component ratios are conditional Amdahl calculations, not
implemented speedups. Three request pairs are not independent process replicas.
Per-forward JSONL survives worker shutdown; partial point results survive timeout.
Failures in instrumented operations remain fatal even if a legacy fallback catches
the original exception. Existing dispatch marks are labeled marks, not proof of
successful execution.

## Tests and launch

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  /scratch/bkaul/venvs/sglang-cpu/bin/python -m unittest discover \
  -s plugin/validate -p 'test_serving_*.py' -v
bash -n plugin/validate/run_serving_opportunities.sbatch
# Freeze the verified source before submitting; the launcher takes that directory.
# Its parent must contain source-sha256.txt with paths relative to the source root.
# Its parent must also contain source-base.txt holding the pinned main commit.
sbatch plugin/validate/run_serving_opportunities.sbatch /path/to/frozen/source
```

Tests cover normalized-AST identity with the frozen infrastructure, unchanged
tensor bytes through the installed probe entry point in a fake model, nested
accounting, malformed controls, swallowed exceptions, absent/duplicate/invalid
events and a full synthetic driver with success/rejection side effects. A real
SGLang import smoke test checks installation separately, without loading weights.
Native generated-token equality remains a runtime check, not established by mocks.
Worker evidence must also match the expected job and total actual token work:
`B*S` prefill and `B*(G-1)` decode tokens, with consistent sequence coordinates.
The launcher verifies the frozen source manifest before and after execution.

Use the existing blocking Slurm waiter, then require job `exit_code=0`,
`results/complete.json`, and the `SERVING_SCREEN_JOB_COMPLETE` log marker.
The waiter can report disappearance without successful completion on this cluster.
Do not treat that as a benchmark pass.

Do not co-locate with the accuracy jobs or mutate their source, extension caches
or controls. The launcher owns its result/build directories. Inspect accuracy
loader readiness and queue capacity before submission; no need to probe its server.
