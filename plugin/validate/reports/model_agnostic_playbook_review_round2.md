# Model-Agnostic Playbook Review: Response Assessment

**Date:** 2026-10-09

**Reviewer:** GitHub Copilot

**Reviewed revision:** `2b979d902e1a6f797334dae8a7bf4250b2e22d09`

**Response:** [model_agnostic_playbook_review_response.md](model_agnostic_playbook_review_response.md)

**Original recommendations:** [model_agnostic_playbook_review.md](model_agnostic_playbook_review.md), reviewer commit `9f5f36f`.

## Decision

**Accept as partial remediation with verified improvements, not closure of the playbook review.** The reason-checked F4 negative tests are a concrete improvement, and the additional sparse GPU cases provide useful evidence. However, contradictory skill instructions remain, and the shared qualification and promotion mechanisms are still pending.

Deferring a broad adapter framework and a two-model portability demonstration is reasonable until a concrete second-model requirement exists. Model-agnostic portability must then remain explicitly unverified. That deferral does not justify postponing the smaller shared-gate work needed by the current DSv4 campaign.

## Verified Improvements

- **F4 self-test:** executed the current `--selftest` with the specified CPU interpreter, one Torch thread and `MAX_JOBS=2`. All **13 checks passed**, including the control showing that an unrelated `NameError` is not accepted as a successful shape-rejection test. The test code now loads the indexer and sparse modules required by the formerly ineffective cases.
- **Coverage inventory:** the harness now requires declared `(op, kind)` entries and the indexer M=1/8/16/32/64 set, an improvement over op-name-only checks.
- **Numerical guidance:** kernel authoring now explicitly distinguishes a bring-up FP32 diagnostic from a source-conformed acceptance reference. The proxy skill adds relevant applicability qualifications.
- **GPU evidence:** the raw job384532 log records the expected published-kernel SHA-256 check, the actual job identity, and execution of independent-batch K128/160/640 and sentinel cases. These support the reported observations for those cases, not an unrestricted serving-conformance certificate.
- **Performance evidence:** job384531 contains three process repetitions with within-process timing summaries. This is useful evidence, but the response's parity interpretation is stronger than those observations establish.

These checks were performed during the response assessment and are recorded here for sharing. No new cluster jobs or performance runs were launched for the review.

## Remaining Findings

### 1. High: Contradictory Numerical and Proxy Instructions Remain

The new conditional reference paragraph is appropriate, but [kernel authoring's shipping gate](../../../.agents/skills/kernel-optimization/kernel-authoring/SKILL.md#L137) still requires matching an FP32 reference. Its procedure also continues to prescribe FP32-reference validation. The [optimization playbook](../../../.agents/skills/kernel-optimization/cpu-optimization-playbook/SKILL.md#L118) retains final-only downconversion guidance, and [perf-proxy](../../../.agents/skills/shared/perf-proxy/SKILL.md#L122) still says real-weight values must not change performance and treats a difference as a bug.

**Consequence:** adding a qualifying paragraph without replacing conflicting normative rules preserves the original failure mechanism: an agent can follow one instruction and violate another.

**Required:** replace the conflicting rules at their decision points. Source-stage semantics govern numerical acceptance; proxy validity is conditional and must be evidenced. Keep mathematical diagnostics separate from acceptance references. No new framework is required for this correction.

### 2. High: The Core Enforcement Work Remains Pending

The response acknowledges that discovery, one prerequisite graph, dependency-based evidence invalidation, and evidence-derived claims are not completed. Those mechanisms address failures already observed in DSv4; they are not speculative requirements introduced solely for another model family.

The new [campaign mandate block](../../../.agents/skills/kernel-optimization/cpu-optimization-playbook/SKILL.md#L38) describes the following sections as mechanized enforcement. Written mandates guide execution; they are not themselves executable promotion checks.

**Required:** implement a small shared promotion contract linking tested scope, candidate/reference/build identities, applicable policy, unresolved obligations, and approval. Connect existing validators to it. `PARTIAL`, stale evidence, missing prerequisites, and a completed-but-negative review must not permit promotion. Preserve the existing human approval and DSv4 campaign requirements.

### 3. Medium: Parity and Numerical-Floor Interpretations Remain Unsupported

The [response's implementation summary](model_agnostic_playbook_review_response.md#L25) labels indexer M8 approximately0.98x as parity and sparse discrepancy as a BF16 op-order floor.

Job384531 records indexer M8 process-median speedups of **0.98x, 1.05x, and 0.96x**. These observations show variability, not established equivalence or a universal no-regression floor. A predefined same-contract comparison policy and an appropriate uncertainty assessment are required before calling the result a tie.

Job384532 records GPU-versus-blockwise-replica maximum discrepancies around **1.953e-3** for several tested cases. That supports an observed discrepancy, not proof that it is unavoidable, intrinsic noise, or an approved acceptance bound. No new numerical bound is ratified by this review.

**Required:** report observed values and scope without assigning an unproved cause. Keep numerical conformance, approximation budgets, repeatability, and hardware noise distinct. Generate these labels from validated records when the shared evidence interface is introduced.

### 4. Medium: Coverage Enforcement Is Stronger, but Still Partial

[F4's coverage declaration](../f4_acceptance.py#L429) now checks `(op, kind)` and indexer M coordinates. This does not yet represent the complete independently specified shape/layout/state/dispatch domain described in the original recommendations. Declared M metadata also needs binding to the actual evaluated tensors and path.

**Required:** extend the existing inventory mechanism incrementally to the claimed domain. A result should identify exactly which synthetic, captured-caller, or full-model scope it qualifies. Keep expected coverage independent of observed results; deleting a required case must not silently narrow a supposedly complete gate.

## Position on Deferrals

| Work item | Recommendation |
|---|---|
| Broad model/framework/hardware adapter framework | Defer until concrete demand justifies the abstractions |
| Two-model portability demonstration | Defer with portability explicitly unverified; use it when a second family is queued |
| Remove conflicting normative rules | Complete now; this is a local correction |
| Reliable skill discovery and one prerequisite graph | Complete as current-workflow reliability work |
| Shared evidence identity, stale-result handling, and promotion checks | Implement a bounded first slice now using existing tools |
| Exact coverage and workflow regression tests | Extend incrementally alongside the shared gate |

The original recommendation was staged reuse of existing infrastructure, not a big-bang rebuild. The portability demonstration was a test of generality, not a prerequisite for constructing a large framework. Avoid conflating those two proposals when applying YAGNI.

## Minimum Next Delivery

1. Remove contradictory reference, precision and proxy rules; verify that the actual runtime discovers the intended skill entry points.
2. Define one executable promotion record covering scope, dependency identities, applicable policy, approval and stale status. Keep the agents as consumers of that decision rather than independent interpreters of success text.
3. Connect the existing F4, measurement, reconciliation and reporting paths to that record. Preserve historical observations; invalidate only qualifications affected by changed dependencies.
4. Add workflow regression tests for a missing required case, changed candidate/reference, incorrect comparator, infrastructure failure inside a negative test, and attempted promotion with PARTIAL or unapproved evidence. Assert the intended rejection reason through the real entry path.
5. Publish a precise implemented/pending table. Do not describe prose additions as mechanized enforcement or the current work as a completed model-agnostic redesign.

## Evidence and Limits

Raw logs inspected: `/scratch/bkaul/replicated_all_384531.log` and `/scratch/bkaul/sparse_gpu_oracle_384532.log`.

Executed local check: `OMP_NUM_THREADS=1 MAX_JOBS=2 /scratch/bkaul/venvs/sglang-cpu/bin/python -B plugin/validate/f4_acceptance.py --selftest`, exit0,13 passing checks.

This assessment checks the response's architectural disposition and selected supporting evidence. It is not a new all-kernel correctness review, a statistical performance certification, or full-model serving qualification. No production code, skills, acceptance policy or author records were changed. Only this reviewer document is authored for publication.