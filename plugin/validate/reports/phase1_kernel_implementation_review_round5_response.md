# Response to the Round-5 Kernel Review (assessment of the round-4 response)

Responding to [phase1_kernel_implementation_review_round4_assessment.md](plugin/validate/reports/phase1_kernel_implementation_review_round4_assessment.md) (published `6080d61`). Response commit `0376948`. No native kernel source changed; the five non-sparse kernels are untouched since round 2. Verified locally: full `f4_acceptance.py` run = STATUS **PARTIAL**; `f4_acceptance.py --selftest` = **SELFTEST OK**; `test_sparse_cpu_vs_gpu.py --selftest` = **REPLAY-SELFTEST OK** (12/12); `promotion_gate.py --selftest` = **OK**, live decision = **BLOCKED**.

**Summary:** I accept all three findings. The reviewer confirmed A1–A3 closed and A4/A5 partial; R4-F1/F2/F3 are the remaining partial edges and are now fixed in code, each with a dedicated control. The two process gaps the review named — no persistent replay self-test, and residual claim-drift in the manifest comment — are also closed. The budget sketch is expanded to the reviewer's full four-point experimental design. Status remains PARTIAL and promotion remains BLOCKED; nothing here requests sign-off.

## Fixes

### R4-F1 (Medium) — replay bound the wrong (optional) job field
The oracle producer saves `slurm_job_id` ([test_sparse_gpu_oracle.py](plugin/validate/test_sparse_gpu_oracle.py#L80)); the replay check read `prov.get("job")`, so a correct requested identity failed and a spoofed `job` alias could override the real run.
- `validate_archive()` now binds to the producer's **canonical `slurm_job_id`** ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L47)) and **rejects a conflicting legacy `job` alias** rather than letting it override the canonical identity ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L50)).
- Controls (replay selftest): correct requested `slurm_job_id` accepted, incorrect rejected, conflicting alias rejected ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L148)).

### R4-F2 (High) — record coordinates were checked for presence, not consistency
Changing only a record's `N` to 64 while the tensors stayed batch-1 was accepted — M1 execution reported as M64.
- `validate_archive()` now checks coordinate **consistency** before any comparison: `N == q.shape[0] == kv.shape[0] == gpu_out.shape[0]`, the full `q[N,H,D]` / `kv[N,K,D]` / `gpu_out[N,H,D]` / `sink[H]` shape relationships, and a finite positive `scale` ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L63)).
- Controls: inconsistent-`N`, inconsistent-`H/D`, and non-positive-`scale` records are each rejected.

### R4-F3 (High) — required distribution / compressor-shape coverage was removable
Path/M presence alone let heavy-tailed indexer M64 and the compressor R8/D128 shape be dropped silently.
- `run()` now enforces required **joint coordinates** via `_required_joint_coords()` ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L479)): the full signed M-sweep under **both** distributions, and all three costed compressor shapes under both distributions. Each required coordinate must be present **and its built input tensor must match the declared shape** ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L534)) — presence is not treated as a correct coordinate.
- Controls through `run()`: removing the heavy-tailed indexer M64 coordinate, or the compressor R8/D128 shape, each hard-fails ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L845)).

### Persistent replay self-test (review's process point)
A4 previously had no standing control. `test_sparse_cpu_vs_gpu.py` now has a `--selftest` ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L130)) with 12 controls — genuine-archive accept; correct / incorrect / conflicting job identity; wrong / missing / unvalidated source hash; empty inventory; missing coordinate; inconsistent `N`; inconsistent `H/D`; non-positive scale. The kernel compile is made **lazy** so the validator is testable standalone without an AMX node or the saved IO.

### Claim-drift corrected at source
The sparse manifest comment no longer reads "judged vs the GPU-ratified BF16 oracle and **accepted within the kernel's own noise floor**" — it now reads "**SCREENED** vs the BF16 blockwise replica, within the **PROPOSED, UNRATIFIED** screen threshold only (**not acceptance authority**)" ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L238)).

## Verified closures vs the round-5 findings

| Finding | Status |
|---|---|
| R4-F1 canonical job field + alias conflict | **Fixed** + 3 identity controls |
| R4-F2 record coordinate consistency | **Fixed** + 3 consistency controls |
| R4-F3 required joint coordinates (dist/shape) | **Fixed** (built-tensor-verified) + 2 removal controls |
| A4 persistent replay self-test | **Added** (12 controls, standalone) |
| manifest claim-drift ("GPU-ratified / own noise floor") | **Corrected at source** |

## Budget proposal — reviewer guidance adopted

Per the recommendation I keep **option (a)** (a bounded pre-sign-off qualification experiment, only after explicit human approval of its plan) and do **not** use option (b) to unblock. [kernel_opt_queue.json](plugin/validate/results/kernel_opt_queue.json) `_downstream_budget_methodology` is expanded from the ε-sketch to the full four-point design:

1. **Workload + boundaries:** named deployment workload, captured input/caller boundaries, model revision, dtypes, TP setup, reference/candidate identities, and the required M/K/mask/layer/state regimes with explicit proxy limitations.
2. **Perturbation construction:** norm/scaling, sign/direction, and correlations across channels/tokens/layers/steps; **replay measured candidate residual tensors** where possible plus justified stress directions (one generic magnitude cannot characterize systematic error).
3. **Metrics + acceptance:** downstream/task metrics defined **independently** of the candidate's observed error; calibration **proposes** an envelope, **frozen** before untouched validation; state how single-op errors **combine** when multiple approximate kernels are active (dummy propagation does not bound real-weight sensitivity).
4. **Plan:** bounded compute/data plan, all open hypotheses, required captures, and a disposition matrix **before** any expensive launch; persist expanded GPU-oracle IO for candidate replay; retain run identity; keep policy / coverage / performance-floor / full-model gates separate.

It explicitly records that the ~1.95e-3 sparse residual is **not a general upper bound** — the current F4 calibration already reports **3.90625e-3** for fp32-bmm and best-of at M16 against the matched-boundary blockwise reference — so this is evidence not to ratify 1.95e-3 or 4e-3, and the three discrepancy types (candidate-vs-replica, replica-vs-GPU, composed-ordering) must not be conflated into one ε. Approval to collect evidence is not approval of the resulting threshold or promotion.

## Remaining PENDING (unchanged; honest)

- Persist expanded GPU-oracle IO for C++ replay (B4 / K∈{128,160,640} / sentinels — currently only N∈{1,8,64} records are saved) and the broader non-sparse all-M and sparse K/head/layout/state coverage documented in round 3.
- Full launcher timing/runtime identity binding (thread counts/placement, binary-library hashes, input hashes).
- Comparator-drift caveats on the performance table (not attributable to new optimization).
- The sparse serving error budget itself and the phase-dependency decision remain the pending human-ratification items; this review did not authorize the experiment, ratify a budget, or amend the dependency.

This response grants itself no waiver, budget, or phase promotion.
