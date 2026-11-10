# Shared-clamp disposition review: recover the production failure, then localize it

> **Chronological record.** Early recommendations below were superseded by later
> matched-batch experiments and the full evaluation at the end of this document.
> Current disposition: [accuracy campaign closed, CPU1246 / GPU1245](DSv4-accuracy-closure.md).

Date: 2026-11-05.
Reviewed: [worker disposition](./DSv4-shared-clamp-disposition-for-reviewer.md)
at `12c04d1`, implementation at `9739f86`, and raw artifacts from jobs 397780 and
397796. No new model run was launched for this review.

## Verdict

**Do not accept "shared clamp ruled out as the gap-closer."**

The experiment establishes a CPU source-semantics correction and a zero-net
answer delta in a different execution configuration. It does not establish the
effect on the production residual failures.

More importantly, it exposed a potentially useful end-to-end lead:
**10 of the 17 historical CPU failures become correct with the shared clamp OFF
in the new harness.** Investigate that execution-path change before selecting
another kernel to modify. This is not proof of a better global configuration,
but it is a much stronger next experiment than an arbitrary attention/router pivot.

Keep shared clamp default off. Do not launch another full accuracy run yet.
The next task is production-failure reproduction and a controlled end-to-end
comparison, not a general validation framework.

## What the evidence does and does not show

### 1. The answer experiment lost most of its intended failures

The saved summary reports OFF=16/23 and ON=16/23, with q1191 recovered and q502
regressed. I independently reconciled the records:

- Historical failures already correct with shared clamp OFF:
  **5, 146, 417, 435, 502, 769, 893, 924, 995, 1067**.
- Thus only **7 of the original 17** remain wrong in the new OFF arm.
- The reported 8/23 answer agreement with production is not a minor caveat;
  it prevents using this experiment to rule out an intervention on the original
  failing trajectories.
- q995, the unrecovered original skip target, is already correct in the new OFF
  arm. That makes it a useful probe for the execution difference.

The zero-net result remains a valid observation about the saved outputs. Calling
it production-neutral or attributing the baseline changes to "borderline tokens"
goes beyond the evidence.

### 2. Prefill execution changed, not just text versus frozen IDs

| Surface | Production 397690 | New diagnostic |
|---|---|---|
| Prefill | Explicit 512-token chunking | Capture explicitly sets 16384 for whole-prompt prefill; recovery does not pass a chunk size |
| Requests per call | Original contiguous shard groups, up to 32 | Capture sends one request at a time; recovery submits the selected 23 together |
| Input API | Text | Frozen token IDs |
| Stops | `Question`, `Assistant:` | Neither supplied in recovery |
| Static memory fraction | Harness default 0.5 | Diagnostic default 0.80 |

Sources: [production launcher](../../plugin/validate/run_gsm8k_dsv4_clamp_fullacc.sbatch),
[production harness](../../plugin/validate/task_gsm8k_chunked.py),
[capture driver](../../plugin/validate/run_shared_capture.py),
[recovery driver](../../plugin/validate/run_shared_clamp_recovery.py).

The recovery's resolved prefill size was not saved. The inspected installed
runtime has a 4096 fallback when GPU memory is unavailable, but that source
default is not a recorded runtime value for this job. Do not silently equate
the recovery configuration with either production or the capture configuration.

Keeping `CHUNK_ACCUM=1` does not make whole-prompt prefill exercise the production
cross-chunk path. Likewise, submitting 23 requests does not prove that every
kernel executes at M=23.

**Prefill chunking and request scheduling are now the highest-priority
unseparated hypotheses, not established root causes.**

### 3. Missing stops are real, but rescoring does not explain this score difference

All 46 saved recovery completions report the 256-token length limit. I trimmed
each at the first production stop string and rescored with
`task_gsm8k.extract_final_answer`: both arms remain 16/23, with no extracted-answer
changes. Missing stops therefore do not explain these scores through extraction
alone. They can still change batch membership and later decode scheduling; restore
the original stops for a controlled run.

### 4. Native GPU shared-output parity was not measured

The [replay](../../plugin/validate/replay_shared_clamp.py) compares CPU donor OFF,
CPU donor ON, and captured CPU output. The raw report contains 69 unique
case/layer pairs and reports OFF exactly equal to the captured CPU output on
all 69. That is useful donor-fidelity evidence.

It does not include the planned native GPU clamp-on/clamp-disabled comparison.
Use "source-semantics correction; native GPU output parity unmeasured," rather
than "proven operator-parity fix."

The statement about only L42 changing also needs its scope: the decision row
changes at L42 in the three sampled layers. L20 changes on other rows in 22/23
captures. Neither this sample nor the initial prompt row covers all layers and
later decoding decisions.

## Recommended takeover: one focused end-to-end experiment

### A. Reproduce a real production batch, not a reconstructed failing-only cohort

Start with production shard 12's first group: **q984 through q1015**, containing
q995. It has no earlier user batch to reconstruct. Historical CPU score is 29/32,
with q989, q995, and q1008 wrong; compare every question against its saved CPU and
authoritative GPU answer, not only q995.

Use the existing production harness and its `--question-ids-file` support, with
`--num-questions 1319`, original text prompts, original order, chunk-size 32,
prefill 512, original stops, dtype and memory settings. Retain accumulator and
routed clamp ON. Set shared clamp OFF. Do not write another answer-generation
harness.

First require reproduction of the target's production failure. If it does not
reproduce, compare the earliest generated-token divergence and resolved runtime
settings; use a repeated OFF control to distinguish reproducibility from a
controlled intervention. Do not proceed as if a different baseline answers the
original question.

### B. Once reproduced, run a small 2-by-2 comparison on that same batch

| Arm | Prefill regime | Shared clamp | Question answered |
|---|---|---|---|
| A | Production 512 | OFF | Does the historical failure reproduce? |
| B | Production 512 | ON | Does shared clamp help the actual production trajectory? |
| C | Whole-request prefill | OFF | Does removing cross-chunk execution recover the failure? |
| D | Whole-request prefill | ON | Is there a chunking/clamp interaction? |

Use an explicit supported whole-request setting and record actual chunk
boundaries; a larger global token budget alone need not make every request
unchunked when several requests share a batch.

Preserve the same request group and generation/scoring protocol across arms.
Use fresh process/request state per arm, or an already-established equivalent
reset; do not invent a new reset mechanism for this experiment. Run independent
arms concurrently if resources permit. Score all 32 answers and record losses
as well as recoveries.

Add only lightweight, existing captures for the target: actual prompt/generated
token IDs, prefill chunk boundaries and batch membership, and selected layer
outputs/decision logits. These should support locating the first meaningful
divergence without another model load. No generic op registry or new validator
framework is required.

### C. Disposition the result immediately

| Observation | Next action |
|---|---|
| B recovers q995 without unacceptable collateral losses | Shared clamp remains a viable production intervention; expand the bounded test before a full run. |
| C recovers q995 and B does not | Prioritize chunk-boundary state/accumulation and chunk-dependent numerical dispatch. Compare the same absolute token/prefix across chunked and whole-request paths; localize the first differing boundary. |
| D alone recovers | Investigate the interaction; do not attribute it to either lever alone. |
| A cannot reproduce the failure | Isolate baseline execution differences first, including actual tokenization, scheduler grouping and runtime settings. Do not select another kernel based on this non-reproduction. |
| A reproduces, but none of B/C/D helps | Next isolate request grouping at fixed chunking; then use a native GPU replay on the reproduced failure's common prefix to choose router/expert versus attention/MHC work. |
| An op-level intervention improves outputs but not answers | Record the local result and move on; do not promote it or spend a full run on it alone. |

Use q502's original first group, **q492 through q523**, as a follow-up replication
of a promising signal. Do not change the primary batch to selected failures
halfway through the experiment.

## How this leads to a fix rather than another diagnostic loop

1. Preserve an actual wrong answer as the reproducible starting point.
2. Find the smallest execution change that changes that answer toward the reference.
3. Compare identical inputs/state at the implicated boundary, including the actual
   native GPU path when claiming CPU/GPU parity.
4. Correct the identified implementation difference, then verify generated answers
   and collateral cases under production execution.
5. Only then run the full matched evaluation to measure aggregate recovery.

A chunking-dependent answer change does not by itself prove a lifecycle bug:
legitimate batch-dependent rounding can also alter greedy trajectories. The
same-prefix comparison distinguishes state/content errors from numerical
differences and measures decision margins. Neither "harmless drift" nor "gap
closed" should be assumed.

The deliverable is a reproduced production failure plus a causal intervention
and its answer-level result. **No new validation-framework project, no blind
kernel pivot, and no full-accuracy run before that evidence.**

## Evidence locations and limits

- `/scratch/bkaul/dsv4_shared_cap/shared_rec_397796_{shared0,shared1,summary}.json`
- `/scratch/bkaul/dsv4_shared_cap/shared_clamp_replay_397780.json`
- `/scratch/bkaul/dsv4_shared_cap/cap_gens_397780.json`
- `/scratch/bkaul/dsv4_shared_rec_397796.log`
- `/scratch/bkaul/dsv4_clamp_gens_397690_{6,12}.json`
- `/scratch/bkaul/dsv4_clamp_stamp_397690_0.txt`
- `/scratch/bkaul/dsv4_shared_cap/rec_stamp_397796.txt`
- `/scratch/bkaul/gpu_flash_gens.json` (authoritative historical GPU comparator)

Checks performed: source/provenance comparison, raw cohort/replay reconciliation,
and production-stop rescoring of 46 outputs. The saved capture witness has empty
generated-token-ID lists on all 23 records, so it cannot establish token-level
non-perturbation versus a separate production run. The in-situ operator equality
is narrower evidence. The recovery log confirms radix caching was forced off;
this review does not attribute the result to cross-arm radix-cache reuse.

## Execution update (2026-11-05)

User authorized the takeover. Baseline job **397815** completed on **pcl-sprh04**,
using q984..1015, original text prompts/stops, prefill 512, accumulator and routed
clamp ON, shared clamp OFF. The historical node sprh06 was taken by another job;
sprh04 has matching reported topology/features. Node identity is recorded rather
than claiming a same-node repeat.

Isolated source checkout: `/data/nfs_home/bkaul/dsv4-residual-execution`.
Evidence root: `/scratch/bkaul/dsv4_residual_execution_20261105`.
The source archive and per-file hashes are saved there. Three focused tests passed
for production-harness invocation and the read-only forward trace; Python syntax
and shell syntax checks passed. Pylance timed out, so these are runtime/syntax
checks, not a Pylance diagnostic pass.

The existing production runner now optionally records resolved runtime settings
and actual output token IDs. A gated trace records actual forward inputs,
positions, request-pool slots, chunk metadata, and raw model top-two logits.
Those raw logits are not labelled sampled decisions on incomplete prompt chunks.
No model numerics were changed for the baseline.

**Baseline reproduced:** 29/32 correct; **32/32 extracted answers and complete
output texts exactly match the historical batch**. q995 remains -120 versus
gold/GPU 480. Resolved runtime confirms prefill 512, BF16, memory fraction 0.5,
radix cache disabled. All archived source hashes were rechecked before launching
the variants. This is text/answer reproduction, not a claim about historical
token IDs (those were not saved in the old run).

The remaining arms are now running, each in a fresh process:

| Job | Node | Prefill | Shared clamp |
|---|---|---|---|
| 397837 | pcl-sprh04 | 512 | ON |
| 397838 | pcl-sprh08 | whole-request requested (`-1`) | OFF |
| 397836 | pcl-sprh09 | whole-request requested (`-1`) | ON |

Each has an attached blocking `sbatch --wait`; a heartbeat monitor covers the
three jobs.

### First variant results

| Arm | Job | Correct / 32 | Changes from reproduced baseline |
|---|---|---|---|
| 512, shared OFF | 397815 | 29 | Baseline |
| 512, shared ON | 397837 | 30 | q995 recovers; all other 31 extracted answers preserved |
| Whole request, shared OFF | 397838 | 30 | q989 and q995 recover; q1009 regresses |
| Whole request, shared ON | 397836 | 31 | q989 and q995 recover; no new incorrect answers |

**q995 now answers 480 in both whole-request arms**, versus -120 in the reproduced
production baseline. Runtime reports prefill -1; each whole-request trace has
four EXTEND calls with zero prefix lengths throughout. Actual consumed q995
prompt IDs exactly match the same 1127-token sequence in all three completed
arms. These are bounded-set results, not full-accuracy improvement.

The first baseline-versus-whole/OFF generated-token divergence is at zero-based
index 8, after an identical generated prefix. The baseline's raw top logits are
token 1660 at 30.625 and token 344 at 30.375; whole/OFF reverses their order
(344 at 30.625, 1660 at 29.625). Thus the execution change alters the decision on
a common prefix, not merely answer extraction. This does not yet distinguish
cross-chunk state handling from batch/shape-dependent numerical computation.

Whole/ON first differs from baseline at index 1 (tokens 16163 versus 21040).

**The production-prefill/shared-ON arm also recovers q995 to 480.** It retains
prefill 512 and changes no other extracted answer in this batch. Thus shared
clamp was not ruled out by the earlier non-reproducing recovery harness: it
helps this real production failure without requiring a prefill configuration
change. This is one bounded intervention result, not aggregate-gap closure or
a proof that chunking is defective.

All four arms were independently rescored from saved text with the production
extractor. Results are persisted in
`/scratch/bkaul/dsv4_residual_execution_20261105/factorial_summary.json`.

The planned replication completed on q502's original 32-question group
(q492..523), using the same production harness and prefill 512:

- **397854**, shared OFF, pcl-sprh04: 30/32; all 32 extracted answers and
  complete texts exactly reproduce the historical production batch.
- **397853**, shared ON, pcl-sprh08: **31/32**; q502 changes from 522.4 to
  the correct **525**. All other 31 extracted answers are preserved.

Both used fresh processes and unchanged archived model/harness sources. Evidence is under
`/scratch/bkaul/dsv4_residual_execution_20261105/q502_replication`.

Across the two disjoint original batches, the production shared-clamp
intervention improves **59/64 to 61/64**: two recoveries (q995, q502), no new
incorrect answers. This is positive bounded evidence, not a full-set estimate.
The earlier diagnostic q502 "regression" did not carry over to its reproduced
production trajectory.

### Full production-protocol evaluation launched

The bounded results justify the next agreed step: full accuracy. Array
**398117**, all **16 shards running**, evaluates the same **1311** question IDs
in their original shard and batch order. It retains **prefill 512**, original
text prompts/stops, BF16, memory fraction 0.5, accumulator and routed clamp ON,
and enables only the shared-clamp intervention. Whole-request prefill is
**not** being promoted or substituted.

The existing launcher gained array-specific output-directory and shard-identity
handling; model and Python harness sources are unchanged from the successful
bounded runs. Three focused tests and shell syntax validation passed. Sources
and hashes are archived at `/scratch/bkaul/dsv4_residual_fullacc_20261105`.

Comparison targets: primary production 397690 (**1240/1311**), accumulator-only
396143 (**1241/1311**), authoritative GPU (**1250/1311**). Final disposition
requires all 16 shards, all 1311 unique IDs, independent rescoring, and paired
recoveries/regressions. Default-off remains the production setting pending
the result.

A blocking completion waiter and automatic partial-score monitor are attached.
The latter aborts a grossly regressed run below 85% after at least 256 scored
questions (or its existing excessive-invalid-output check); this operational
floor is not an accuracy acceptance threshold. No full-run score is available
yet.

### Full result: array 398117

The final completion log was written at **22:25:27 Pacific on November 5**.
The automatic follow-through was missed; the completion-monitor shell sessions
were no longer available when checked at 23:35. Completion here is supported by
all 16 explicit launcher completion markers and complete saved outputs, not
the absence of the job from the scheduler.

| Configuration | Correct / 1311 | Accuracy |
|---|---|---|
| Accumulator-only historical CPU | 1241 | 94.66% |
| Routed-clamp CPU, shared OFF (397690) | 1240 | 94.58% |
| Routed + shared clamps, production prefill 512 (398117) | **1246** | **95.04%** |
| Historical GPU reference | 1250 | 95.35% |

All 16 shards completed; the combined set has exactly 1311 unique expected
question IDs, with zero invalid answers. Independently rescoring every saved
generation reproduces the shard counts. Recorded runtime confirms prefill 512,
BF16, memory fraction 0.5, and radix caching disabled; generation identities
record accumulator and both clamps ON and the original sampling/stopping rules.

Against routed-clamp CPU 397690:

- **13 recoveries:** 5, 139, 201, 244, 247, 435, 502, 603, 769, 886, 924, 995, 1280.
- **7 regressions:** 193, 298, 386, 904, 1175, 1231, 1239.
- **Net +6**, or +5 versus accumulator-only CPU.
- Both bounded targets retain their correct answers: q502=525 and q995=480.

The aggregate score deficit to GPU decreases from 10 questions to **4**.
This is **not per-question parity or full closure**: GPU-right/CPU-wrong cases
number **14**, while CPU-right/GPU-wrong cases number **10**. Extracted-answer
agreement with GPU is **1279/1311**, lower than routed-only CPU's previously
measured 1283/1311 despite the improved task score.

Remaining GPU-right/CPU-wrong cases:
**29, 81, 146, 197, 298, 386, 417, 893, 904, 1067, 1175, 1191, 1229, 1239**.

Disposition: the shared correction produces a measured full-set accuracy gain
under the production protocol, contradicting the earlier non-reproducing
harness's broad negative disposition. It also has real regressions. Keep the
gate/default unchanged pending review; next diagnosis should use the reproduced
production trajectories of these residual cases, particularly new GPU-right
regressions, rather than assume the remaining four-point count deficit is four
specific failures or harmless numerical drift.

Machine-readable results:
`/scratch/bkaul/dsv4_residual_fullacc_20261105/combined.json` and
`/scratch/bkaul/dsv4_residual_fullacc_20261105/paired_summary.json`.
This is one full evaluation, not a replicated accuracy/determinism certificate.

### Follow-up authorized: diagnose new production regressions

The 1246 result and evaluated source archives remain frozen; shared clamp stays
default off. A separate checkout, `/data/nfs_home/bkaul/dsv4-shared-regression`,
adds only diagnostic capture/replay code. No new numerical correction has been
made.

Original production batches for **q1175** and **q1239** are being replayed with
prefill 512 and shared clamp OFF/ON:

| Target | Original batch | Shared OFF job | Shared ON job |
|---|---|---|---|
| q1175 | q1148..1179 | 398194 | 398192 |
| q1239 | q1230..1261 | 398193 | 398195 |

All four CPU jobs started. Capture records native shared inputs, routed inputs,
combined outputs and packed weights across all 43 layers at the last prompt row
and the proposed first divergent generated decision (index 22 for q1175, 33 for
q1239). Request identity is checked against actual consumed tokens and positions.
The proposed common prefix is inferred using historical text and saved ON token
IDs; **fresh OFF/ON generated IDs must confirm it before comparison**.

After generation, native CPU donor OFF/ON replays operate on those same operands,
including direct shared and combined boundaries. The active arm must reproduce
the captured native combined output. GPU preparation requires complete replay
coverage and **all 32 production answer texts to reproduce in each arm**, not
just a successfully written capture.

Native GPU job **398197** is queued with scheduler `afterok` dependencies on all
four CPU jobs. It requests two H200s and starts automatically only after CPU
completion. Its preflight rejects non-reproducing CPU evidence before loading
the GPU model. GPU replay uses the native shared MLP on each exact CPU full-batch
input, with configured limit 10, unchanged repeat, and a finite large-limit
control (1e30) that retains the clamp-enabled code branch while leaving the
observed values unclamped. Results include TP-reduced outputs and projection
operands where exposed by the native path. They are operator diagnostics,
**not substituted GPU task-accuracy measurements**.

Evidence root: `/scratch/bkaul/dsv4_shared_regression_20261106`.
CPU/GPU source snapshots and hashes, per-case plans, scheduler watchlists and
completion waiters are recorded there or attached to the session.
Two focused tests passed for request binding/native-output preservation and
invalid comparison rejection; Python/shell syntax checks passed. Actual native
GPU execution has not completed and is not claimed validated yet.

Decision after replay: change model numerics only if the comparison identifies
a concrete implementation discrepancy. Then test q1175/q1239 answers and retain
q995/q502 recoveries before another full evaluation. If shared output agreement
does not explain the regressions, use the same captured common-prefix/layer
evidence to choose the next boundary; do not expand into a general validation
framework or label the remaining gap harmless.

GPU launch update: **398197 failed before operator replay**, because its default
sliding-window KV ratio allocated only 256 slots, below the runtime's 384-token
minimum. This was a diagnostic launch configuration error, not a measured
CPU/GPU discrepancy. The failed GPU source is archived separately.

Retry **398227** is running with explicit `swa_full_tokens_ratio=0.5` (requested
2048 SWA slots at the unchanged 4096-token total budget). CPU captures and model
numerics are unchanged. Completion waiting and a persistent retry heartbeat are
attached. No native GPU comparison result is available yet.

### Native GPU replay completed: 398227

The retry completed in **14m12s**. All **344** planned comparisons were written
(two questions x two CPU trajectories x two decision positions x 43 layers).
The comparison checks the captured CPU input against the GPU input, source/run
identity, native CPU combined-output reproduction, GPU unchanged-repeat
equality, and agreement of TP-reduced outputs across both ranks. These are
operator observations, not a new GPU accuracy evaluation.

The CPU runs also reproduced all 32 full generation texts in each of their
four historical batches. Fresh output IDs confirm first divergence at index
22 for q1175 (OFF token 24623 versus ON 1893) and index 33 for q1239 (OFF 23
versus ON 553). The answer regressions reproduce: 11 to 7 and 7 to 8.

**The shared clamp is not showing a broad mismatch to the GPU clamp.** On the
344 target rows, enabling it makes CPU shared output closer to the configured
GPU output in 121 comparisons, unchanged in 201, and farther in 22. This counts
paired target-row L2 distances, not correct answers.

| Question / decision | CPU ON vs GPU ON | CPU OFF vs GPU ON | CPU OFF vs GPU inactive |
|---|---:|---:|---:|
| q1175 / first divergence (22) | 2.798% | 3.049% | 2.773% |
| q1239 / first divergence (33) | 3.049% | 3.082% | 3.036% |

Entries are median target-row relative L2 across 86 comparisons (43 layers,
both CPU trajectories), **not repeated-run error bounds**. Similar residual
differences remain when neither backend clamps. This does not prove the
differences harmless, identify their source, or establish that the clamp
implementation is correct for every input. It does argue against blindly
reverting the shared clamp as a general GPU-parity correction.

The recorded GPU shared weights are FP8; native projection outputs and down
inputs are BF16, with TP2-sharded shared projections. The CPU uses packed FP8
weights. Stage arithmetic/activation quantization and TP reduction remain
possible sources of the residual operator difference.

**Next action, already running (398240):** use the saved first-divergence
operands for a bounded CPU-only stage comparison. It compares gate projections,
activation boundaries, and down projections on the *same saved GPU down input*.
It also measures its standalone-GEMM reconstruction against the captured native
CPU shared output, so reconstruction error cannot silently masquerade as a
production defect. No model reload, full accuracy run, or numerical change is
needed for this step.

Evidence: `/scratch/bkaul/dsv4_shared_regression_20261106/gpu_398227/comparison.json`.
Analysis source:
`/data/nfs_home/bkaul/dsv4-shared-regression/plugin/validate/analyze_shared_regression_stages.py`.
The full-set result remains **1246/1311**, and the shared-clamp default stays OFF.

Stage-run operational correction: **398240 timed out after 111/172 units** at
its 20-minute limit. Its metrics were only held in memory, so they cannot be
used as completed evidence. That all-or-nothing output was a mistake in this
diagnostic. Array **398242** now splits the four independent CPU trajectories
across four nodes, with 43 units per task, 30-minute limits, and per-unit JSONL
checkpoints. Sources are frozen in `stage_source_retry`; the attached array
waiter and heartbeat watch all four tasks. Early saved results show nonzero
standalone-reconstruction error, which is explicitly retained rather than
calling this decomposition bit-identical to the fused production kernel.

### Stage result: the difference is already present before the clamp

Array **398242 completed** all four tasks, preserving **172 first-divergence
coordinates** (43 layers x two trajectories x two questions), each with clamp
ON and inactive controls. Final coordinate/source/run checks passed.

| Boundary, shared clamp ON | q1175 | q1239 |
|---|---:|---:|
| Gate/up projection, CPU vs GPU on identical hidden inputs | 1.533% | 1.532% |
| Post-activation, respective backend projection inputs | 2.155% | 2.118% |
| Down projection, CPU vs GPU on identical saved GPU activation inputs | 1.999% | 2.128% |
| Native complete shared output, CPU vs GPU | 2.798% | 3.049% |
| Standalone CPU reconstruction vs native fused CPU output | 0.063% | 0.068% |

Numbers are medians of target-row relative L2 across 86 layer/trajectory
comparisons per question, not accuracy percentages or replicated error bounds.
The reconstruction is **not bit-identical** to the native CPU kernel; its
largest target-row discrepancy is 0.310%. Its error is reported separately,
not subtracted or assumed negligible for greedy token selection.

On the **exact saved GPU gate/up outputs**, the FP32 expression
`silu(min(gate, 10)) * clamp(up, -10, 10)`, stored in BF16, reproduces the
saved GPU activation tensor with `torch.equal` in **172/172** comparisons.
The corresponding inactive-control expression also matches in **172/172**.
This validates this formula on these recorded operands, **not** bit equality
of the CPU donor's vectorized activation on arbitrary inputs.

The gate projection difference precedes the clamp. A roughly 2% difference
also survives at the down projection when the activation input is made
identical. Similar values occur with the clamp inactive. Therefore, the
shared-output discrepancy cannot be assigned solely to the clamp expression.
The next numerical boundary to investigate is **FP8 linear arithmetic,
activation quantization, and TP partition/reduction**, not another speculative
clamp rewrite. Source in the cached GPU image routes block-FP8 linear operations
through a W8A8 runner; the CPU native kernel consumes BF16 activations. Exact
runner selection and each contribution still need isolation.

**End-to-end disposition:** this narrows the operator diagnosis but does not
prove that these differences cause q1175/q1239 to select the wrong answer.
No additional numerical fix or full accuracy run is justified yet. The next
causal test should isolate the shared-linear precision/TP difference, then
apply only that isolated intervention to the two reproduced regression batches
and q995/q502 recovery controls. Require a change in the first-divergence
token margin and final answers, not merely a smaller tensor norm. Do not
trade away CPU precision simply to imitate GPU tokens without this evidence.

The complete measurement summary and input hashes are in
`/scratch/bkaul/dsv4_shared_regression_20261106/gpu_398227/stage_summary.json`;
the four per-trajectory files and incremental checkpoints are beside it.
The aggregation procedure is saved at
`/data/nfs_home/bkaul/.copilot/session-state/e91384d5-db40-48a0-8408-da05c9035424/files/summarize_shared_stages.py`.
Evaluated analysis sources remain in `stage_source_retry`. Two existing
capture/comparison tests and Python/shell syntax checks passed; Pylance was
unavailable. Full-set accuracy remains **1246/1311**, with no default changes.

### Authorized precision/TP isolation

The next native GPU replay retains the same 344 frozen inputs and adds
read-only FP8 GEMM taps at the 172 first-divergence coordinates. It saves the
**actual quantized activations and scales consumed by native GEMMs**, rank-local
outputs, and native FP8 weights/scales once per projection/layer/rank. No
dequantized weight copy or alternative GPU arithmetic is introduced. Existing
unchanged-repeat, native-input, and TP-reduced-output comparisons still run.

| Question | Instrument / control | Disposition |
|---|---|---|
| Activation quantization contribution | CPU packed GEMMs on original BF16 versus actual GPU quantized-and-rescaled inputs | Quantization implicated only if this reduces the matched-stage difference |
| TP contribution | Full-K CPU result versus separately rounded partial-K results plus BF16 sum | Split/reduction implicated only if the paired stage difference shrinks |
| Joint effect | Both controls together | Detect interaction rather than credit either control alone |
| Different effective weights | Native GPU FP8 weights/scales versus corresponding CPU packed weights/scales | Reject a quantization-only interpretation if weight values differ |
| Remaining GEMM difference | Both controls still differ from native GPU | Do not force a clamp fix or claim parity |

These initial four controls are **stagewise matched-input experiments**, not
four complete counterfactual MLP trajectories: a changed gate output would
produce a new down input requiring fresh quantization. Any end-to-end diagnostic
must handle that correctly and preserve the native disabled path. It must then
test the original q1175/q1239 batches and q995/q502 controls, including token
margins and final answers. No full-set rerun is authorized by tensor similarity
alone. Three focused tap/capture tests and Python/shell syntax checks passed
before this launch.

Launched **GPU 398565** (TP2 H200), with the four CPU factorial tasks in
**array 398568** automatically dependent on GPU success. Both have attached
completion waiters and persistent heartbeats. The dependent array is cancelled
if the GPU prerequisite fails. Frozen source directories are
`gpu_precision_source` and `precision_cpu_source` under the existing evidence
root. Four focused tests and a scheduled native packed-FP8 full/split GEMM
preflight passed.

The CPU control uses the actual GPU quantized values/scales, then stores their
rescaled values in BF16 as required by the CPU kernel. Any additional BF16
rounding is measured separately; it is not called an exact native GPU FP8
execution. Effective weight equality and reproduction of the preceding GPU
398227 output are required before crediting a factorial result. Per-coordinate
results are checkpointed. No end-to-end precision intervention has been
launched yet; it depends on these measurements.

### Precision factorial result: activation quantization is the dominant operator difference

GPU **398565** and all four tasks of CPU array **398568** completed.
All 172 generated-decision coordinates passed effective-weight equality and
unchanged native GPU output checks against 398227.

| Question / projection | Original CPU | Quantization only | TP only | Both |
|---|---:|---:|---:|---:|
| q1175 gate/up | 1.533% | 0.180% | 1.533% | 0.180% |
| q1175 down | 1.999% | 0.311% | 2.013% | 0.268% |
| q1239 gate/up | 1.532% | 0.180% | 1.532% | 0.180% |
| q1239 down | 2.128% | 0.316% | 2.133% | 0.271% |

These are median target-row relative L2 values across the same 86 layer/
trajectory samples per question. Quantization accounts for most of the measured
shared-linear discrepancy; TP-only does not explain it. This is still not an
answer-recovery result.

**Next intervention:** a default-off, native CPU shared-kernel diagnostic
quantizes and rescales BF16 activations in groups of 128 before both existing
packed-FP8 GEMMs. It retains the native GEMMs, shared clamp, and routed-output
addition. It is an activation-precision experiment, not a promoted accuracy fix.
Its disabled branch must reproduce the captured native outputs before any
answer test.

An important qualification: CUDA uses fast division in the native quantizer.
The CPU reciprocal has a few different FP8 rounding-boundary decisions (for
example, 12/131072 gate elements at q1175/L20), despite exactly matching scales.
The BF16 rescaling also introduces rounding relative to native GPU FP8 GEMM.
Therefore this intervention is **not exact GPU emulation**. Both discrepancies
are measured, not hidden. A one-layer native disabled-path/build preflight
passed, including rejection of nonfinite inputs; complete captured-operand
validation is required next. If it substantially reduces shared-output
differences, the original answer batches and recovery controls are the deciding
test. No full accuracy rerun or default change is justified yet.

The factorial heartbeat missed aggregate completion because it replaced an
already-observed terminal task state with UNKNOWN after Slurm expired that task
record. The blocking `sbatch --wait` succeeded, all four final files contain 43
records, and all four completion markers are present. This was a monitor
bookkeeping failure, not a failed experiment; subsequent monitoring must retain
observed terminal states.

Native candidate validation is now running as **array 398765**: 86 captured
coordinates per CPU trajectory, both disabled clamp arms at direct/combined
boundaries, enabled shared-output comparisons, and native GPU quantizer probes.
The numerical donor source is frozen in `quant_candidate_source`; no source
changes are being made to that evaluated artifact.

The answer-test checkout is prepared at
`/data/nfs_home/bkaul/dsv4-shared-quant-e2e`, with the exact donor source used by
validation and a read-only selected-logit trace. Evidence/plan:
`/scratch/bkaul/dsv4_shared_precision_e2e_20261106`. Its four original 32-request
batches cover q1175, q1239, q995, and q502 (128 questions total). It is **not
launched yet**. The baseline shared-ON first-divergence margins for the two
regressions are both 0.125 logits: q1175 favors token 1893 over 24623, and q1239
favors 553 over 23. Candidate margins will only be compared on a matched consumed
prefix; an earlier divergence cannot be treated as the same decision. Eight
focused harness/capture tests passed after adding the read-only token trace.

### Native validation passed; original answer batches launched

Array **398765 completed all 344 captured coordinates**. The disabled diagnostic
reproduced both clamp arms at the direct and combined boundaries. With the
diagnostic enabled, median native shared-output differences became:

| Target / position | Baseline vs GPU | Diagnostic vs GPU |
|---|---:|---:|
| q1175 / prompt decision | 3.056% | 1.035% |
| q1175 / generated decision 22 | 2.798% | 0.945% |
| q1239 / prompt decision | 3.177% | 1.037% |
| q1239 / generated decision 33 | 3.049% | 0.889% |

Across the captured native-quantizer probes, scales matched exactly, while
**2660/56,360,960 FP8 codes differed** because CPU/CUDA reciprocal rounding is
not identical. These observations support an end-to-end precision diagnostic,
not an exact-parity or accuracy claim.

The original four 32-request answer batches are now running in array **398801**:

| Task | Job | Target | Role |
|---|---|---|---|
| 0 | 398802 | q1175 | Regression |
| 1 | 398803 | q1239 | Regression |
| 2 | 398804 | q995 | Recovery control |
| 3 | 398801 | q502 | Recovery control |

The intervention is shared activation quantization only: original prefill 512,
batch order, prompts, stops, max-new-tokens 256, accumulator and both clamps ON
are retained. No TP modification is included. Starts are staggered by 45
seconds; blocking completion and persistent score/status monitoring are active.
As an operational gross-regression guard, a completed batch below 24/32 correct
or with an invalid answer stops the remaining array. This is not a promotion
threshold.

The [native donor](../../../dsv4-shared-quant-e2e/plugin/intel_cpu_models/_dsv4_shared_clamp.py)
and [activation diagnostic](../../../dsv4-shared-quant-e2e/plugin/kernels/dsv4_shared_quant_diagnostic.h)
are in the frozen answer-test checkout. The evidence root contains its
`source_overlay.tar.gz`, source hashes, exact question IDs, and baseline file
hashes. No answer-test outcome is available yet. Full-set accuracy remains
1246/1311 and the new diagnostic remains default OFF.

### End-to-end result: partial recovery, not a general correction

Array **398801 completed all four batches**, with 128 expected unique question
IDs and zero invalid answers. Independent rescoring of the saved text agrees
with all batch scores; source hashes, baseline hashes, runtime prefill,
generation settings, intervention flags, and completion markers were checked.

| Target batch | Baseline correct | Diagnostic correct | Target answer |
|---|---:|---:|---|
| q1175 | 30/32 | 29/32 | Still 7; gold is 11 |
| q1239 | 30/32 | 32/32 | **8 to 7, recovered** |
| q995 control | 30/32 | 30/32 | 480 retained; target text/token IDs unchanged |
| q502 control | 31/32 | 31/32 | 525 retained; reasoning text changed |
| **Total** | **121/128** | **122/128** | **Net +1** |

- Recoveries: **q1231** (250 to 50), **q1239** (8 to 7).
- Regression: **q1168** (2 to 0; gold 2).
- Both requested recovery controls remain correct.
- GPU extracted-answer agreement is **125/128 in both arms**. q1231's new
  correct answer disagrees with the historical GPU's incorrect 250; q1168 is
  wrong in the diagnostic but still differs from the GPU's incorrect 16.

**The actual token divergences are earlier than the originally captured clamp
decisions.** Comparing the old decision-22/33 logits would compare different
consumed prefixes and is not valid:

| Target | First baseline/diagnostic difference | Baseline old-token minus new-token logit | Diagnostic margin on same consumed prefix |
|---|---:|---:|---:|
| q1175 | Generated token 0: 1166 to 26037 | 0.000 | -0.125 |
| q1239 | Generated token 9: 14 to 294 | +0.125 | -0.125 |

These are raw BF16-logit observations from the actual forward traces, with
fresh generated IDs and reconstructed request histories establishing the
common prefix. q1239's intervention changes the trajectory and recovers the
answer. q1175 changes wording immediately but remains wrong. This does not
isolate a single responsible layer or establish that matching GPU quantization
is universally preferable.

**Disposition and next recommendation**

1. The experiment establishes that shared activation precision contributes to
   an actual answer regression, not just tensor differences. It does **not**
   establish a general CPU integration fix: one requested regression remains,
   one collateral regression appears, and aggregate GPU answer agreement is
   unchanged.
2. Keep the diagnostic default OFF and the **1246/1311 evaluated configuration
   frozen**. Do not extrapolate 122/128 to a new full-set score or launch another
   full evaluation from this selected, single-run result.
3. Stop further blind shared-clamp/precision sweeps. The next investigation
   should follow **q1175 on a matched CPU/GPU consumed prefix**, capturing
   layer/subsystem boundaries to locate the first consequential difference
   outside (or upstream of) shared-linear arithmetic. Include q1239 as a
   recovered control and q1168 as a collateral-regression control. Judge any
   resulting change by the original answer batches, not tensor norms alone.

Machine-readable result:
`/scratch/bkaul/dsv4_shared_precision_e2e_20261106/paired_summary.json`.
The persisted aggregation/re-scoring procedure is
`/data/nfs_home/bkaul/.copilot/session-state/e91384d5-db40-48a0-8408-da05c9035424/files/analyze_shared_quant_e2e.py`.
This is one targeted evaluation, not replicated accuracy certification.
No additional model jobs are running for this experiment.

## q1175 follow-up: native trajectory and matched-prefix tracing

The next experiment keeps the 1246 configuration unchanged: accumulator, routed
clamp and shared clamp ON; shared-quantization diagnostic OFF. Two original
32-question batches contain q1175, q1239 and q1168. Their 64 prompt token
sequences were checked against the original CPU execution traces.

CPU runs the complete natural answer batches and must reproduce **all original
texts and output IDs**, not just the three target answers. GPU runs the same
natural batches, then separately replays the first 41 decisions while forcing
the recorded CPU tokens **after** native model computation and logit capture.
The forced arm is a conditional numerical diagnostic, not an accuracy run.
It preserves incremental KV updates rather than re-prefilling a concatenated
prompt plus answer. GPU uses TP2 and eager execution; it is not a replacement
for the historical TP4 full-accuracy reference.

Captured targets retain all MHC streams and native dtypes at all 43 layers:
layer input/output, MHC pre/post, normalized attention input/output, router
gate output and selected experts, combined MoE input/output, final MHC head,
normalization and raw logits. Records contain actual consumed token IDs,
positions, request slots, flattened-token and logit-row coordinates. Model
hooks are read-only in the natural arm; forced-token substitution is GPU-only.
Every decision is checkpointed independently.

| Observation | Interpretation | Next action |
|---|---|---|
| CPU batch does not reproduce original complete outputs | Diagnostic is not production-faithful | Stop attribution and identify the runtime/capture difference |
| Natural GPU does not recover q1175 | Current TP2/eager reference does not reproduce the historical answer advantage | Keep that limitation explicit; do not call conditional differences a recovery |
| Exact consumed prefixes differ | Layer differences include trajectory differences | Compare only genuinely matching coordinates, using the forced arm |
| A boundary amplifies disagreement on matching prefixes | Localizes a subsystem, but its operands may already differ | Replay that native operation on identical operands before changing code |
| Routing IDs first diverge | Discrete routed-expert selection is a candidate amplifier | Inspect gate margins and matched-operand routing, not another shared-precision sweep |
| Layer states agree but head/logits differ | Head normalization/projection or near-tie decision is implicated | Isolate the head using the captured native inputs |
| Operator discrepancy is confirmed | Candidate implementation issue, not yet answer recovery | Make a gated correction and rerun the original answer batches and controls |

This is a bounded residual-gap investigation, not a new acceptance framework.
The full-set result remains **CPU 1246/1311 versus GPU 1250/1311**.

**Execution note.** Initial jobs 399280 (GPU) and 399281 (CPU) failed in the
diagnostic plumbing, not in a model-correctness assertion. GPU eager-runner
warm-up uses synthetic request slots; those calls must be excluded before
request-history tracking. CPU completed its first generation batch but a local
variable collision in result bookkeeping prevented saving the returned answers.
Its 82 q1175/q1168 boundary files survive, but they are **not** a completed
full-batch reproduction result.

Both failure paths now have entry-path regression tests. Raw generation
responses are checkpointed before interpretation, and the CPU batches are
split across two independent farm jobs to avoid serial generation cost.
Retry source is frozen separately under
`/scratch/bkaul/dsv4_q1175_trace_20261107/retry1/`.
Jobs: **399426** (CPU q1175/q1168), **399427** (CPU q1239), **399428** (GPU).
Completion waiters and the on-disk watchlist are active. No new accuracy claim
or numerical correction follows from the initial partial captures.

### Completed trace: early attention disagreement, but GPU oracle not reproduced

All three retry jobs completed. Independent analysis reproduced **64/64 original
CPU texts and token-ID lists**. CPU scores remain 30/32 in each batch. The trace
contains 123 CPU target decisions (three questions, decisions 0 through 40)
and both natural and forced GPU trajectories. GPU rank-0/rank-1 canonical
attention outputs, MoE outputs, routing and logits agreed at all 123 compared
forced-prefix coordinates.

| Target | Gold | Reproduced CPU | Current native GPU, TP2/eager | Historical GPU, TP4 |
|---|---:|---:|---:|---:|
| q1175 | 11 | 7 | **6** | 11 |
| q1239 | 7 | 8 | **7** | 7 |
| q1168 | 2 | 2 | 20, length-limited extraction | 16, length-limited extraction |

The current GPU natural batches score 29/32 and 32/32, **61/64 versus CPU
60/64**. This selected comparison does not update either full-set score.
In particular, the new GPU q1175 reasoning explicitly concludes six apples:
it is not a correct reference trajectory for diagnosing why CPU misses eleven.

**What the boundaries actually localize**

- At q1175's prompt decision, CPU/GPU layer-0 normalized attention inputs
  differ by **0.0186% relative L2**, but attention outputs differ by **3.162%**.
  At decision 9, the saved layer-0 attention inputs are equal while attention
  output relative L2 is **2.826%**. Thus disagreement is already present before
  the first MoE/shared expert. The corresponding prompt-decision attention
  output differences are 3.105% for q1239 and 3.183% for q1168.
- These are **attention-module** comparisons, including projections and the
  accumulated KV history, not identical-Q/K/V kernel comparisons. A difference
  here does not by itself establish an attention-kernel implementation bug.
- Routing subsequently amplifies disagreement. At q1175's prompt decision,
  the first differing expert set is layer 8; layer 11's MoE input/output
  relative errors are about 8.05%/27.55%. These are downstream observations,
  not evidence that layer 11 is the original defect.

**The GPU repeat control also matters**

At q1175 decision 0, natural and forced GPU runs have the same consumed target
prompt and the same recorded final-prefill token/position batch. Nevertheless,
their raw logits differ by **5.521% relative L2** (maximum absolute difference
1.71875), and argmax changes from token 1166 to 26037. This occurs before any
generated q1175 token has been forced.

The first differing **compared attention/MoE target-row** boundary is layer-1 attention:
its input is equal, its output differs by 0.244%; layer 0 agrees. For q1168,
the first such difference is layer-2 attention (1.708%). q1239's prompt-decision
GPU logits reproduce exactly. Request slots differ between the runs. Earlier
prefill scheduling and the complete KV operands were not captured, so this is
**not proof of a GPU cache bug or intrinsic kernel nondeterminism**. It is proof
that this run has not established a stable, known-correct q1175 reference.

**Disposition**

No new numerical correction is justified by this trace, and neither clamp nor
quantization defaults change. Do not start another CPU full-accuracy run.
The next work should be:

1. **Restore the correct GPU reference first.** The authoritative saved run
   used TP4 and 64-question batches, not this TP2/eager 32-question diagnostic.
   Recover its effective cache/graph settings and original batch context;
   require q1175=11 and repeatable logits before using it as a rescue oracle.
   The historical generation identity records TP4, batch64 and harness commit
   `dc62aa04e8ba73906f0d093420289a04b2b7bbbb`; it does not stamp every resolved
   runtime setting.
2. **Follow attention, not another shared-expert rewrite.** On the restored
   reference, retain complete prefill scheduling and inspect native projection
   inputs/outputs plus KV operands at the first differing attention boundary.
   Separate projection precision, KV/history and attention arithmetic using
   matched operands. The current saved target-row states alone cannot make
   that separation.
3. Only a concrete discrepancy should lead to a gated code correction. Its
   acceptance remains recovery of q1175 in the original CPU batch without
   breaking q1239/q1168 or the prior recovery controls.

The important result is a narrower investigation: **an early attention-side
difference and an unreproduced GPU control**, not another purported shared-clamp
fix. The aggregate four-answer full-set gap remains real as a comparison of
the recorded runs, but it is not yet isolated to a CPU-specific implementation
defect.

Evidence and reproduction:

- Root: `/scratch/bkaul/dsv4_q1175_trace_20261107/retry1/`
- Independent disposition: `summary.json`; all per-decision measurements:
  `boundary_metrics.jsonl`.
- Raw responses, resolved settings, identities and complete captures remain in
  `cpu_399426/`, `cpu_399427/` and `gpu_399428/`.
- Frozen analysis command:

  ```sh
  ROOT=/scratch/bkaul/dsv4_q1175_trace_20261107/retry1
  PYTHONPATH="$ROOT/source/plugin/validate" \
    /scratch/bkaul/venvs/sglang-cpu/bin/python \
    "$ROOT/analysis_source/analyze_prefix_trace.py" \
    "$ROOT" 399426,399427 399428
  ```

Fourteen focused tests passed, including explicit synthetic warm-up exclusion,
full driver success/failure, raw-response preservation and rejection of an
incorrect capture coordinate. The real CPU/GPU runs and analysis above are the
evidence for the model observations; those tests alone are not.
All trace jobs and their waiters have finished.

## Full end-to-end closure: fresh matched CPU/GPU evaluation

The user redirected the investigation to full accuracy rather than additional
single-question debugging. This evaluation is now complete.

| Backend | Correct / evaluated | Accuracy | Invalid |
|---|---:|---:|---:|
| Latest CPU accuracy-fix configuration | **1246 / 1311** | **95.04%** | 0 |
| Fresh native GPU, TP4 | **1245 / 1311** | **94.97%** | 0 |

CPU is ahead by **one answer (0.076 percentage points)** in this run.
This does not establish a statistically meaningful CPU advantage or numerical
equivalence. It does establish that the previously reported aggregate deficit
against the historical GPU result is absent in this fresh matched evaluation.

Independent closure checks verified all 16 shards per backend, exactly the
1311 expected question IDs with no duplicates, saved completion markers,
source hashes, data identity, generation settings and matching resolved
cache/context/scheduling settings. Saved texts were independently rescored.

CPU uses accumulator, routed clamp and shared clamp ON, shared-quantization
diagnostic OFF, and no diagnostic tracing. Its **1311 extracted answers and
1311 complete generation texts exactly reproduce the earlier 1246 CPU run**.
No new numerical correction was introduced for this evaluation.

Both backends use eight shots, greedy decoding, 256 maximum new tokens,
the existing Question/Assistant stop strings and scorer, 16 identical logical
shards, 32-question batches, prefill512 and radix cache OFF. CPU is TP1;
GPU is TP4. The GPU uses a fresh engine for each logical shard, as does CPU.
GPU engines ran sequentially; CPU generation ran across EMR nodes.

Paired outcomes:

- CPU correct / GPU wrong: **9** questions:
  7, 57, 139, 149, 228, 247, 769, 822, 924.
- GPU correct / CPU wrong: **8** questions:
  94, 146, 298, 386, 904, 1175, 1191, 1231.
- Extracted-answer agreement: **1283/1311**.

Historical GPU1250 remains a valid recorded result under its older execution
protocol, not the matched baseline for this run. Fresh GPU1245 comprises
16 regressions and 11 recoveries against that historical corpus. No causal
attribution for those changes is claimed.

Execution: CPU array401042 plus replacement array401068 for not-yet-started
shards; GPU401058. CPU loading initially admitted two engines, increased to
four at the user's request without changing started engines, question slices
or frozen model source. All 16 CPU shards and the GPU evaluation completed.

**Disposition: close this accuracy campaign.** No additional investigation,
full rerun, or default promotion is implied. Report the latest CPU number as
1246/1311 and the fresh matched GPU number as 1245/1311. These are one full
custom-protocol GSM8K evaluation, not a published-benchmark reproduction or
a repeated-run correctness certificate.

Evidence:

- `/scratch/bkaul/dsv4_latest_fullacc_20261107/paired_summary.json`
- Per-shard texts/token IDs, scores, runtime settings and completion markers:
  `/scratch/bkaul/dsv4_latest_fullacc_20261107/{cpu,gpu}/shard_*/`
- Frozen source manifest and protocol:
  `/scratch/bkaul/dsv4_latest_fullacc_20261107/source_hashes.json` and `plan.json`.
- Reproduction:

  ```sh
  /scratch/bkaul/venvs/sglang-cpu/bin/python plugin/validate/close_latest_accuracy.py \
    /scratch/bkaul/dsv4_latest_fullacc_20261107 \
    --data /scratch/bkaul/gsm8k_test.jsonl --out /tmp/dsv4-rechecked-summary.json
  ```

The command above runs from the repository root and requires the retained
external evidence. See [the concise closure and integration record](DSv4-accuracy-closure.md)
for flags, provenance and the spine/leaf lessons encoded from this campaign.
