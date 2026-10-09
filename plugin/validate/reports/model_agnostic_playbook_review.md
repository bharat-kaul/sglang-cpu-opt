# Model-Agnostic Enablement Playbook: Review and Recommendations

**Date:** 2026-10-09

**Reviewer:** GitHub Copilot

**Scope:** Workflow architecture, skill discovery, correctness and performance evidence, and portability across model families. Recommendations only; no implementation or new model qualification is included.

## Executive Recommendation

Retain the existing skills and validation tools, but organize them around **one contract-driven workflow with executable prerequisites, explicit evidence states, and model/framework/hardware adapters**.

The repeated DSv4 review failures are not explained solely by missing instructions. Many principles already exist, but instructions conflict, checks enforce different scopes, and reports can claim more than the evidence establishes. Some failures also reflect inconsistent execution of existing rules. More lesson paragraphs alone will not reliably prevent recurrence.

The objective is not to promise bug-free enablement. It is to detect missing, stale, contradictory, or insufficient evidence before certification, route failures to bounded repairs, and keep policy decisions under explicit human authority.

## Review Basis

The architecture review began at `ddd44d9`. Before packaging this report, the reviewed skill files and F4 source were checked against current HEAD `1e3096d77ca72eb3933c3c11b3003e8e9f325e09`; those files were unchanged. Other implementation and measurement work has continued independently.

Three read-only reviews examined orchestration, evidence handling, and portability. The parent checked the controlling skill text, representative executable gates, and the VS Code discovery implementation. Historical failures are documented in [phase1_kernel_implementation_review.md](phase1_kernel_implementation_review.md) and [phase1_kernel_implementation_review_round2.md](phase1_kernel_implementation_review_round2.md). They explain the workflow risks; this report does not reopen every historical kernel finding or certify subsequent repairs.

## Findings

### 1. High: Numerical Guidance Has Competing Authorities

[Kernel authoring](../../../.agents/skills/kernel-optimization/kernel-authoring/SKILL.md#L110) prescribes a naive FP32 reference and validation against it. The [optimization playbook](../../../.agents/skills/kernel-optimization/cpu-optimization-playbook/SKILL.md#L89) broadly prescribes FP32 accumulation with final-only downconversion. In contrast, [self-audit](../../../.agents/skills/shared/adversarial-self-audit/SKILL.md#L82) requires fidelity to every published numerical stage.

The historical indexer and sparse failures demonstrate why this matters: higher precision, different intermediate rounding, and a different normalization order can change selections or outputs. Agreement between a candidate and an adapter sharing the same assumption does not establish source conformance.

**Recommendation:** make the independently evidenced source contract authoritative before donor selection or optimization. Generic precision advice must be conditional. Keep an FP32 mathematical reference as a diagnostic unless its conformance has been established. Storage changes must account for every consumer and persistent-state update, not merely one consuming GEMM.

### 2. High: Multiple Entry Points Prescribe Incompatible Phase Orders

[Enablement](../../../.agents/skills/throughput-enablement/model-enablement-playbook/SKILL.md#L61) places model profiling before coverage and wiring, while also requiring gates in order. The [optimization sequence](../../../.agents/skills/kernel-optimization/cpu-optimization-playbook/SKILL.md#L54) places a systemic-configuration pass in Phase 2, although other skills require it before trusting profiles.

**Recommendation:** define a single prerequisite graph, distinguishing diagnostic execution from promotion of an optimized implementation. A runnable reference path and validated instrumentation may be necessary before optimization decisions; that does not authorize deployment or bypass the existing DSv4 Phase-1 requirements. Agents should route through this graph rather than maintain separate procedural copies.

### 3. High: Executable Gates Do Not Share a Qualification Contract

There is meaningful validation infrastructure already. [Perf ingestion](../parse_perf_sweep.py#L38) validates samples and coordinates; [reconciliation](../dsv4_roofline_p2.py#L446) binds inventories and destinations. However, [F4 inventory checking](../f4_acceptance.py#L431) operates at op-name granularity rather than the full required case/dispatch domain. Its [negative-test helper](../f4_acceptance.py#L549) treats any exception as successful rejection, so an unrelated setup error can satisfy a test intended to prove a particular invariant.

Qualification and publication also need shared dependency identities: candidate, reference adapter, build, workload, environment, and policy. Rendering consistency alone does not establish that a measurement still applies to changed code.

**Recommendation:** reuse the existing exact-binding and validation patterns through a common evidence interface. Specify expected coverage independently of observed results. Negative tests must assert the intended rejection reason and prove the target check executed, with valid controls alongside failures.

### 4. High: Model-Specific Observations Became Universal Rules

[Perf-proxy](../../../.agents/skills/shared/perf-proxy/SKILL.md#L8) overstates depth reduction and dummy-weight performance equivalence. Routing distributions, sparsity, state evolution, heterogeneous stages, and cache residency can legitimately alter execution. The [decomposition contract](../../../.agents/skills/throughput-enablement/model-op-decomposition/SKILL.md#L92) describes a decoder step, not arbitrary model execution. The [certificate](../../../.agents/skills/throughput-enablement/enablement-certificate/SKILL.md#L66) applies fixed efficiency requirements to every hot op, despite their differing performance regimes.

**Recommendation:** qualify proxy applicability and performance-model applicability explicitly. Keep DSv4's all-C/C++ requirement as a campaign policy, not a universal enablement invariant. Models should declare their own workload axes, state transitions, task metrics, and reference authority. KV-cache sweeps, token equality, GSM8K, AMX, and Slurm are not universal requirements.

### 5. High: Default Discovery Does Not Reliably Expose the Workflow

The [skill organization](../../../.agents/skills/README.md#L25) nests most skills beneath workflow-group directories. The [VS Code locator at the installed revision](https://github.com/microsoft/vscode/blob/2a59476c9bfcb90b3ddc372c36762471b7dfad1c/src/vs/workbench/contrib/chat/common/promptSyntax/utils/promptFilesLocator.ts) checks immediate child directories for skill entry files rather than recursively descending those groups.

Explicit registration or manual reads can reach the skills; this is not a claim that they were never loaded. It is a reproducibility problem for a new workspace or user relying on defaults.

**Recommendation:** use supported discoverable entry locations or explicit group-root registration. Add a startup discovery check that lists the resolved orchestrator and required skills. Keep agents as thin routers, descriptions concise, and detailed case histories in referenced material.

## Proposed Architecture

| Layer | Owns |
|---|---|
| Core workflow | Prerequisites, evidence validation, state transitions, repair routing, certification |
| Model adapter | Execution stages, tensor/state contracts, dynamic branches, legal workload axes, representative inputs, proxy eligibility |
| Framework adapter | Loading, dispatch tracing, capture hooks, integration boundaries, state lifecycle |
| Hardware adapter | Supported arithmetic/layouts, donor capabilities, topology, measured ceilings |
| Task and execution adapters | Quality policy, performance objective, workload distribution, and local/cluster execution |

The shared sequence is:

**Declare objective and scope -> establish reference/contracts -> runnable bring-up and capture -> qualify correctness and measurement -> optimize -> integrate -> certify the tested scope.**

Model-specific policy determines the promotion prerequisites. Diagnostic execution is not equivalent to candidate acceptance or permission to advance phases.

## Make the Workflow Proactive

1. **Establish contracts before implementation.** Record storage, compute and accumulator types, intermediate rounding, normalization/order, masks, output boundaries, layouts, and state semantics. Independently conform the reference adapter. Record unknowns as obligations, not inferred facts.
2. **Bind evidence to dependencies.** Use versioned records with candidate/reference/build/input/environment/policy identities. A dependency change marks affected qualifications `STALE`; retain historical measurements. Distinguish `PASS`, `PARTIAL`, `FAIL`, `BLOCKED`, and `STALE`. Exit zero and review completion alone cannot promote a candidate.
3. **Check exact coverage.** Required cases should cover applicable shapes, phases, layouts, dtypes, state transitions, seeds, and observed dispatch branches. Keep synthetic-fragment, captured-caller and full-model qualification separate. Do not generate the expected inventory from whatever tests happened to run.
4. **Test the workflow itself.** Turn the historical failure classes into executable regression fixtures through real ingestion, evaluator, wrapper, and publication entry points. Require typed rejection reasons and valid controls; unrelated exceptions are infrastructure failures, not successful fault detection.
5. **Route failures to bounded repairs.** Return the owning component, failed invariant, affected evidence, and smallest recheck. Automate safe repairs and derived-report regeneration where practical. Stop or escalate when evidence or authority is missing; do not repeatedly rerun without a discriminating hypothesis.
6. **Separate policy from executor judgment.** The executor proposes, the reviewer assesses evidence, and the designated authority ratifies policy changes. Never resolve a failure by silently widening tolerances, dropping cases, changing comparators, or relabelling uncertainty as parity.
7. **Bound optimization work.** Maintain an opportunity register containing hypothesis, expected contribution, falsifying experiment, result, and disposition. Prioritize high-impact opportunities; measured losses and justified deferrals are valid. Do not demand proof that every imaginable optimization has been exhausted.
8. **Generate claims from validated records.** Reports and certificates should derive reference identity, metric meaning, tested scope, estimator, and approval status from evidence. Avoid independent strings that can drift from the computation. Performance evidence must retain raw trials, comparison order, variability, and same-contract comparator identity.

## Acceptance Tests for the Redesign

These are proposed tests, not tests executed during this architecture review.

| Fault or scenario | Required outcome |
|---|---|
| Candidate and local oracle both adopt incorrect intermediate rounding | Independent source-conformance gate rejects them despite mutual agreement |
| One required dispatch/layout/state case is removed, but every op name remains | Coverage is incomplete; full-scope qualification is blocked |
| A negative test raises `NameError` before reaching its target check | Self-test fails as infrastructure error, not expected contract rejection |
| A source, adapter, policy, or build changes after qualification | Affected evidence becomes stale; unaffected evidence remains reusable |
| A different-arithmetic comparator or duplicate trial is substituted | Corresponding performance claim is rejected |
| A child process fails but subsequent logging succeeds | The wrapper still fails |
| A threshold is proposed but not approved, or evidence is only PARTIAL | Certification remains blocked; authorized diagnostics may continue |
| A recurrence kernel is evaluated with an inapplicable GEMM threshold | Policy applicability is rejected rather than inventing a performance failure |
| A new workspace loads the project with default configuration | Required workflow entry points are discoverable, or missing registration is reported explicitly |

## Implementation Order

**Stage 1: Resolve routing and contradictory authority.** Fix discovery; establish one prerequisite graph; replace unconditional numerical/proxy rules with contract-scoped guidance. Move DSv4 histories and scheduler recipes into referenced campaign/adapter material. Preserve explicit current user constraints.

**Stage 2: Connect the executable evidence path.** Define the shared contract and evidence interface, then adapt existing ingestion, F4, reconciliation, reporting and certification tools. Add dependency invalidation, exact case coverage, and typed rejection tests. Do not create another standalone validator for each lesson.

**Stage 3: Demonstrate portability.** Exercise the unchanged core on two dissimilar small, pinned models, such as image classification and conditional diffusion. Supply adapters rather than edit core logic. Require preprocessing and classification coverage for the former; conditioning, denoiser, timestep/scheduler state and decoder coverage for the latter. Deliberately omit a component, mismatch a layout, and propose an invalid proxy; each must block or narrow qualification.

## Success Criteria and Limits

Success means missing, stale, contradictory or insufficient evidence is detected before human certification; failures identify actionable repairs; high-impact optimization opportunities have explicit dispositions; and a different model family requires adapters rather than new core rules.

The existing donor-first approach, dependency-closure discovery, captured-shape discipline, parser/reconciliation checks, and separation of screening from ratified acceptance are foundations to preserve. The redesign should connect and enforce them, not replace them wholesale.

No new performance runs, model qualification, tolerance ratification, or workflow implementation were performed for this report. Independent human review remains necessary for reference authority, policy, and deployment decisions.