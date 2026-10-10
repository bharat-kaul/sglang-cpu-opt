# Assessment: Correctness Round 6 and Roofline Sweep

Reviewed response `a1d60737ab004127a49b13519927f4ffe53d10b0`, correctness implementation `72271b2`, sweep implementation `aaccef6`, and results `b73d763`. Local correctness checks ran on the clean response revision. Scope: the two prior correctness findings, the new standalone roofline comparison, and the claimed methodological follow-through. No production edits or cluster/model/performance runs were made by this review.

## Findings

### R6-F1 (Medium): nominal roofline ratios do not establish the asserted performance mechanism

The [response](correctness_round6_and_roofline_response.md#L42) says sparse is "genuinely compute-bound and near its roof" and that this explains why S1 could not beat it. The measured M64 result supports **5.59 useful FP32 TF/s, or 71.8% of the specified 7.7824 TF/s nominal reference**. It does not establish an absolute hardware ceiling or the cause of S1's loss.

- The [platform definition](../platforms/emr.json#L50) derives FP32 throughput at **1.9 GHz nominal base**, not measured loaded frequency or maximum frequency. The analogous AMX figure has the same qualification. Keep these explicit nominal reference assumptions; "datasheet peak is a ceiling, not achievable" is stronger than that definition supports.
- The [regime calculation](../bench_roofline_sweep.py#L92) chooses whichever theoretical term is larger; it measures neither the bottleneck nor cache/DRAM traffic. The useful-byte DRAM comparison is a diagnostic for these repeatedly reused buffers, not a cache-aware latency lower bound. The combine caveat is correct and should apply consistently to other cached rows. Compressor's simplified FLOP count also omits its exponential/recurrence cost.
- S1 changes the compute path to BF16/AMX, so proximity to an FP32 reference ceiling cannot rule out its upside. Its **observed loss** is valid evidence for rejecting this implementation; packing, transpose, small-GEMM, and state-update costs are plausible mechanisms, not separately timed attribution in this sweep.
- The [277 GB/s and 47 TF/s observations](../platforms/emr.json#L41) came from `pcl-sprh02`; this sweep ran on `pcl-sprh11`. They are useful prior observations, not measured ceilings for this run, and the AMX number does not calibrate FP32 sparse throughput.

**Disposition:** accept the numerical nominal-reference comparison; narrow the bottleneck/ceiling/causal wording. Retain S1's measured rejection without claiming that FP32 roof proximity explains it. This requires no new experiment merely to correct the claim.

### R6-F2 (Low): the published aggregate loses provenance, spread, and numerical precision

The [result JSON](../results/roofline_sweep.json#L1) contains only per-coordinate summaries. The [harness](../bench_roofline_sweep.py#L96) writes metadata-free process JSON to [reused filenames](../run_roofline_sweep.sbatch#L18), without job IDs in their paths. There is no committed aggregation procedure in this response. The job log preserves enough context to check this run, but the standalone JSON is not a self-identifying evidence artifact and later jobs can overwrite its inputs.

Independent reconciliation of `/scratch/bkaul/roofline_sweep_384677.log` found all **40 coordinates x 3 process medians**. The available scratch JSON medians reproduce all 40 published latencies after rounding, but those files have no embedded job identity. Two apparent log differences are consistent with double rounding: R128/D512 M64 has process-JSON median `360.95 us` and published `360.9 us`, while the log median is `361.0 us`; R8/D512 M16 similarly has `21.45`, `21.4`, and `21.5 us`. These are not material performance discrepancies.

Some percentages also differ from evaluating the unrounded formula: top-k M1 is about **0.67%**, which rounds to **0.7%**, rather than the published **0.6%**. The difference is immaterial to the optimization decision but avoidable. All 40 stored `floor_us` values match the declared formulas to their stored precision.

Process spread is not negligible everywhere: indexer M32 is `250.0..294.4 us` (median `262.2`); R8/D512 M1 is `14.8..18.5 us` (median `16.3`). Sparse M64 is comparatively stable at `767.4..784.6 us` (median `768.5`). These are observed ranges, not confidence intervals or a universal noise floor.

**Disposition:** retain this sweep as descriptive evidence. Persist job/source/build/runtime/input identity, process samples and spread, use job-specific output paths, and calculate aggregate percentages before display rounding. Preserve the current inputs rather than rerunning just to repair documentation. The new self-audit rule on evidence identity/spread is appropriate but not fully implemented here.

### R6-F3 (Low): repository-resolution failure is masked by the launcher

The raw sweep log starts with `fatal: not a git repository`, then completes all three replicates. In the [launcher](../run_roofline_sweep.sbatch#L13), Git resolution is nested inside `cd "$(...)"`; `set -euo pipefail` does not make that inner failure the exit status of `cd`. Slurm's spooled script path is not a reliable repository anchor. This run subsequently stamps the expected `aaccef6` identity and is not rejected on this basis, but it relied on the inherited working directory.

**Disposition:** resolve and validate an explicit submission/repository directory in a separately checked command before changing directory; fail before compilation or output writes if resolution fails. This is launcher hardening, not evidence that the completed timings are wrong.

## Correctness Closures

**R5-F1 CLOSED in the reviewed scope.** The [archive check](../test_sparse_cpu_vs_gpu.py#L70) rejects nonfinite `gpu_out`, `q`, `kv`, and `sink`; comparison rejects nonfinite reference/candidate values and nonfinite metrics. The genuine saved archive for job `384532` is accepted with N=`[1,8,64]`. In-memory NaN/+inf/-inf substitutions in each of the four fields, 12 controls total, all reject through actual `main()` for the intended field-specific reason **before kernel loading**. Finite comparison succeeds; NaN reference and candidate fail for their intended reasons. The saved archive was not modified. The persistent replay controls check rejection rather than the exact rejection message; this review independently checked the relevant reasons.

**R5-F2 CLOSED in the reviewed scope.** The [coordinate predicate](../f4_acceptance.py#L309) now runs on the same built tuple passed into evaluation. A through-`run()` probe retained all 48 declarations and changed only normal R8/D128 pooling: calibration seed 0 built R8, held-out seed 100 built R4. Observed windows were `[8,4]`; F4 returned `2/FAIL` with one explicit `builder drift` failure for evaluated `(8,4,128)`. This disconfirms the old preliminary-build acceptance path. Closure concerns the declared required predicates, not a claim that every possible tensor contract is exhaustively qualified.

Stock verification: **19 replay controls, 27 F4 controls, 48 x 5 = 240 F4 evaluations**, no stock hard failures, **PARTIAL**. Five promotion controls passed and the live decision was **BLOCKED**. The drift probe used two seeds to test the calibration-to-held-out transition; it did not replace the full stock screen.

## Accepted Evidence and Limits

The sweep log records node `pcl-sprh11.sc.intel.com`, source `aaccef6`, initial dirty count 0, and 64 threads with close binding. The launcher specifies DDR5-5600, exclusive EMR allocation, core places, and NUMA0 CPU/memory binding. The diff from `aaccef6` to the response contains only the skill, results, and response document, not native or benchmark changes. No binary hash or observed per-thread affinity/controller traffic was captured by this sweep.

The exact estimator is the median of **five 30-call means per process**, then the median across **three sequential processes**. Each coordinate reuses its tensors for warmup and timing; the process seed is 0. These are not independent input distributions, cold-memory measurements, or per-call latency quantiles. The new sweep covers all six kernel families and all three compressor shapes at M=`1,8,16,32,64`, but only the FP32-KV indexer contract: it does not measure the direct BF16-KV path or establish I1's incremental effect. I1/C1's old-versus-new exactness and speed claims remain separate evidence, not proved by this timing-only sweep.

The existing S1 job `384676` log contains three process replicates over the tested M/K sweep; all printed paired medians are below 1 against both `fp32bmm` and `bestof`. That supports retaining the current implementation and rejecting this S1 candidate. This review did not newly certify S1 arithmetic or exhaust all possible blockwise implementations.

**Overall:** the two outstanding correctness defects are closed. The standalone sweep is useful and its headline arithmetic is reproducible, subject to the interpretation and artifact qualifications above. No new native correctness defect was established by this review. F4 remains **PARTIAL**, promotion **BLOCKED**, and thresholds **UNRATIFIED**. No budget, experiment, dispatch/E2E work, or phase promotion is authorized by this assessment.