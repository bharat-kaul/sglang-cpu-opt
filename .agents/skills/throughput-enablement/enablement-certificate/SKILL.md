---
name: enablement-certificate
description: "Use at every model-enablement phase exit and final certification. Enforces a reviewed plan, source-faithful evidence replay, bound adversarial audits and external approvals through enablement_gate.py. Covers native kernels, accuracy, dispatch, absolute and peer performance, determinism, publication and plugin-only diff. GREEN applies only to the reviewed domain, not universal optimality."
---

# Enablement Certificate

The deliverable. Turns "the agent says it works" into an evidence-bound, reproducible,
machine-checkable artifact for human review. If a required check fails or evidence is
missing, the certificate is BLOCKED — no partial credit.

Apply the [canonical acceptance policy](../../../../docs/methodology-playbook.md#canonical-acceptance-policy).
The executable entrypoint is [enablement_gate.py](../../../../plugin/validate/enablement_gate.py).
Historical hand-filled GREEN YAML records are legacy evidence, not certificates accepted by this gate.
Do not relabel them: produce a new, source-bound evidence bundle and replay the required checks.
The campaign-specific `promotion_gate.py` and raw-GEMM `peer_roofline.py` reports retain their
narrow scopes; neither is a substitute for this phase-prefix/model certificate.

## Executable phase exits and final certificate

Use an independently reviewed plan, starting from the deliberately incomplete
[plan template](assets/gate-plan-template.json). Nulls/empty coverage cannot pass.
The plan pins the gate implementation and source-faithful replay adapters. Model-specific adapters
must exist and be reviewed before acceptance; this gate does not implement a new model's oracle.

```bash
python plugin/validate/enablement_gate.py \
  --repo /path/to/clean-source \
  --plan /path/to/approved-plan.json --plan-sha256 APPROVED_PLAN_SHA256 \
  --evidence /path/to/bundle/evidence.json \
  --through standalone \
  --approvals /path/to/approvals.json --approvals-sha256 EXTERNALLY_APPROVED_SHA256 \
  --output /path/to/reports/standalone.json
```

Use `--through discovery|bringup|standalone|integration|final`. Each invocation replays the
entire required prefix; a previous PASS file is not sufficient. `final` emits `GREEN`; earlier
exits emit only `PHASE_PASS`. Failure exits 2, prints `BLOCKED` to stderr and leaves the output
unchanged. Store output outside both the source and evidence trees. Replay is bounded to 600 seconds
per adapter invocation; expensive captures/model runs happen beforehand, not inside the gate.

| Phase | Required checks |
|---|---|
| discovery | scope, source_contract, infrastructure, calibration, capacity |
| bringup | reference, correctness |
| standalone | native, correctness, performance, optimization, fusion, precision, determinism |
| integration | dispatch, serving, correctness, performance |
| final | accuracy, performance, peer, determinism, publication, diff |

All phases require a recorded, evidence-bound adversarial self-audit and executable negative controls
for every check. Standalone and final additionally require externally approved human records;
integration rechecks the standalone approval. The auditor cannot manufacture these approvals.

### Local failure, restart and approval handling

Automation should invoke [run_enablement_phase.py](../../../../plugin/validate/run_enablement_phase.py),
which wraps this SAME gate, rather than retrying a bare command in a loop:

```bash
python plugin/validate/run_enablement_phase.py \
  --repo /path/to/clean-source \
  --plan /approved/plan.json --plan-sha256 APPROVED_PLAN_SHA256 \
  --evidence /bundle/evidence.json --through standalone \
  --approvals /approved/approvals.json --approvals-sha256 EXTERNAL_APPROVAL_SHA256 \
  --run-dir /runs/MODEL-IDENTITY/standalone --max-attempts 3
```

Use one canonical run directory per phase identity. It must be outside source/evidence trees.
The runner persists a journal and unique `attempt-NNNN/{stdout.log,stderr.log,report.json}` files.
Only an exit-zero gate report bound to the expected plan/evidence/approval is committed as accepted.
Accepted phases are terminal: repeated invocation cannot relaunch or overwrite them. Use a new
phase directory for the next phase, whose gate replays the whole prefix.
If the caller is interrupted after acceptance was committed but before receiving success, the
accepted journal remains terminal; an uncertain caller response is not permission to relaunch.

| Situation | Required behavior |
|---|---|
| Missing, rejected or stale approval | `WAITING_APPROVAL`, exit 3, no adapter launches and no attempt spent |
| Human supplies the correct renewed approval/hash | `--resume` rechecks the external record; the executor cannot self-approve |
| Adapter crash, timeout or invalid evidence | `BLOCKED`, exit 2, one failed attempt recorded; no automatic retry |
| Interrupted runner | No acceptance from partial output; terminate owned children on handled signals |
| Restart after failure/interruption | Explicit `--resume --retry` authorizes exactly one attempt, within the original budget |
| Retry budget exhausted | Refuse before launching any child; changing the budget in-place is rejected |
| Another owner/descendant remains live | Refuse immediately; never delete the lease to force a retry |
| Source/plan/evidence/runner identity changes | Refuse reuse of that run directory; create a reviewed new identity and renew affected approvals |
| Truncated/missing journal with leftover work | Refuse automatic reconstruction/reset; preserve evidence for operator investigation |

Every actual attempt replays the entire prefix. Immutable, hash-verified raw captures may be reused;
cached PASS labels, partial reports and reports left by a killed parent are never adopted.
There is no retry that skips a failed correctness check or raises a tolerance.

**Process ownership:** the Linux/POSIX file lease is inherited by the gate and its replay adapters.
Thus SIGKILL of the runner does not allow a second runner to duplicate a still-active check.
SIGTERM/SIGINT and adapter timeout clean up the adapter process group; interruption during a
negative control is failure, not a successful rejection test. If the gate itself is killed, an
orphan adapter may need operator cleanup before the lease releases. Adapters must stay local,
bounded and synchronous: no detached services, remote jobs or deliberate dropping of the lease.
Never edit/reset a live journal or create a second directory to bypass a live owner.

**Demonstrated boundary:** [test_phase_recovery.py](../../../../plugin/validate/test_phase_recovery.py)
uses real local subprocesses, SIGTERM/SIGKILL, timeouts, a spawned descendant, duplicate runners,
explicit restarts, stale prior-phase artifacts and missing/rejected/renewed approvals. Its raw
observations are synthetic, not model measurements. This is process-failure recovery on a local
POSIX filesystem, not proof against storage/server power loss, remote daemons or Slurm job races.
Step 5 must exercise Slurm job-ID reconciliation, cancellation/preemption, queue/accounting delays
and the ambiguous submit-before-journal window on the actual launch adapter before enabling retries
there. An ambiguous remote submission must BLOCK, never blindly resubmit.

### Trust boundary (do not turn a hash into self-approval)

`--plan-sha256` comes from the operator's independently reviewed plan. It must not be computed from
an unreviewed candidate plan and silently treated as authorization. Likewise, obtain the approval
file digest from the human approval channel, not from the executor's own assertion. Reviewer text
is attribution, not authentication; the gate verifies the externally supplied digest, not a signature.
Credentials are never part of the bundle.

The plan reviewer must verify complete source-derived inventory, source-conformed adapters,
calibrated targets and original checkpoint dtype/QAT semantics. Hashes bind bytes; they do not prove
an adapter is mathematically correct or a claimed hardware identity is true. Adapters must verify
these from raw source/capture/run manifests, not echo the requested identity or a producer's PASS flag.
The synthetic test adapter exercises enforcement only and is not a production conformance oracle.

### Version 1 input contracts

All inputs are strict JSON: unknown/missing fields and duplicate keys reject. Version 1 fixes the
phase/check inventory above; changes require review of the gate, tests and plan together.

- **Identity:** exactly `model`, `source_commit` (40 hex), `config_sha256`,
  `checkpoint_revision`, `reference_revision`, `hardware_sha256`, `runtime_sha256`,
  `dtype`, `node`. The source worktree must be clean at that commit before and after replay.
  Adapters verify actual configuration/checkpoint/hardware/runtime identity, including thread/NUMA
  settings in the hashed runtime manifest. Reference revisions are independently pinned.
- **Plan:** the template's fields. `batches` selects declared request batches within **1..64**,
  including both 1 and 64 (default sweep 1,2,4,8,16,32,64). This does not confuse prefill token M
  with request batch. `ops` is the complete compute/fused-op inventory. Each `kernel_cases` entry is
  exactly `{op, batch, shape, dispatch, distribution}`; string fields name source-bound coordinates.
  Every op x declared batch is required; include all relevant shapes/branches/distributions as
  additional joint cases. Adapters validate the actual tensor dimensions/contents, not just labels.
- **Each check specification:** exactly `adapter: {path, sha256}`, `inputs: [role, ...]`,
  `cases: [...]`, `limits: {metric: number, ...}`, `timeout_seconds`.
  All standalone checks use the complete `kernel_cases` domain. Integration/final performance,
  peer, dispatch and serving checks cover every declared batch. Other scopes (full task,
  all-layer parity, publication) are independently reviewed in the plan, not inferred by the gate.
  Adapter paths are relative to the source repo; artifact paths are relative to the evidence file.
  Both must resolve inside their roots. Input roles cannot alias the same file.
- **Timing checks:** performance/peer also require `baseline_identity`; same node, hardware,
  runtime and dtype as the candidate. Performance adds positive `target_ms` in case-list order,
  derived from the approved calibration and validated by its adapter. Required raw roles are
  `candidate, baseline, calibration` for performance; `candidate, baseline` for peer;
  `candidate, reference` for correctness. Other roles are explicitly declared per check.
- **Startup infrastructure:** discovery also requires distinct `inventory` and `execution_plan`
  inputs following the [startup gate](../../../../docs/methodology-playbook.md#mandatory-startup-infrastructure-gate).
  The reviewed adapter validates CPU/GPU/scheduler/storage/software observations, per-node/rank
  budgets, execution placement, concurrency and launch-time freshness. It must report zero
  `missing`, `stale`, `overcommitted` counts. Missing/provisional site adapters block acceptance;
  the generic gate does not collect live inventory or submit jobs. Historical freshness is bound
  to the original submission/allocation, not invalidated merely by later certificate replay.
- **Limits:** exact metrics are defined in `METRICS` in the gate. All mismatch/missing/violation/
  non-native/unresolved/untested/broken/core-change/lint-error counts must be zero; capacity
  headroom >=1 byte; determinism repeats >=3. Performance requires unrounded `eff_abs >=0.70`
  and same-contract `speedup >=1`; peer requires `eff_rel >=0.90`. Accuracy uses `score_drop_pp`
  <=1 percentage point, or a stricter approved budget. Plans can tighten, not weaken these defaults.
  A target exceeded by measurement requires target/model review, not an automatic acceptance.
  An alternative policy/donorless route needs explicit review and implementation; version 1
  deliberately blocks rather than accepting a waiver string.

**Evidence envelope:** exactly `{schema_version: 1, identity, phases: {...}}`, with exactly the
requested phase prefix. Each phase has `checks` (exact required check names) and `audit`.
Each check has:

```json
{
  "inputs": {"candidate": {"path": "candidate.json", "sha256": "64_HEX_DIGEST"}},
  "negative_controls": [
    {
      "name": "missing-coordinate",
      "inputs": {"candidate": {"path": "missing-case.json", "sha256": "64_HEX_DIGEST"}},
      "expected_error": "missing observation coordinates"
    }
  ]
}
```

This illustrates the shape only: include every role required by the check. Controls must have valid
artifact hashes, different inputs from the positive case and a specific expected rejection reason.
The same adapter and numerical/coverage checker are replayed; crashes/timeouts do not count as
successful negative controls. All positive inputs must pass. Hashes are rechecked after replay
across the whole prefix; failure never replaces an accepted report.

**Adapter protocol:** reviewed Python executable, reads one JSON request on stdin:
`{identity, phase, check, cases, inputs}`; `inputs` maps roles to verified absolute paths.
It validates raw artifacts/production replays and prints one JSON result:
`{identity, phase, check, observations: [...]}`. Exit 0 is necessary but not sufficient.
Each non-timing observation is `{case, metrics}` with exact expected metrics; violations are
reported numerically for the outer gate to reject. Each timing observation is `{case, samples}`.
A sample is `{candidate: {identity, process_id, ms}, baseline: {identity, process_id, ms}}`.
Both sides need >=3 distinct process identities, finite positive times and matching expected identity.
The gate retains samples and derives median ratios itself:
`eff_abs=target_ms/median(candidate_ms)`, `speedup` or `eff_rel=median(baseline_ms)/median(candidate_ms)`.
No rounded input ratio, non-finite metric, duplicate/missing coordinate or unexpected result field passes.
An adapter must not launch jobs, modify inputs/source or report preflight operands instead of the
actual evaluated tensors. Put raw process spread, run-order and calibration provenance in its inputs.

**Audit:** exactly `{checks_sha256, reviewer, goal_alignment, findings}`. `checks_sha256` is computed
by `checks_digest(plan_hash, evidence, phase)` over the identity and all prefix check manifests.
`goal_alignment` explains correctness and best demonstrated Xeon performance at request batch <=64,
including B1 latency and the declared throughput points. Each finding is
`{issue, status: "RESOLVED", resolution, evidence: {path, sha256}}`; any unresolved finding blocks.
Do not hide open opportunities by omitting them. Mechanical validation cannot judge the quality of
an audit; reviewers still inspect its reasoning and supporting evidence.

**Human approval file:** exactly `{schema_version: 1, approvals: {...}}`, containing exactly the
required reviewed phases in the requested prefix. Each record is
`{decision: "APPROVED", reviewer, evidence_sha256}`. Bind the digest using
`approval_digest(plan_hash, evidence, phase)`, which includes the prefix's checks and audits.
The external reviewer supplies the trusted digest of this approval file to the caller.
Any changed source, plan, evidence or audit invalidates the corresponding binding.

## Sections (all required)
1. **Coverage** — the `coverage-gate` per-op table: every op → matched kernel +
   donor, verdict ALL COVERED (fallbacks listed with their efficiency haircut).
   Every compute kernel must be C/C++ (authored or native library), not a temporary
   eager reference. Attach each kernel's measured optimization/opportunity ledger,
   including fusion, supported domain and unresolved work. Low-share kernels are not exempt.
2. **Accuracy** — `accuracy-oracle`: per-layer max-abs / cosine vs reference (with
   tolerances) + task scores (gsm8k / mmlu / hellaswag) vs reference, deltas within
   N points.
3. **Dispatch** — proof the intended AMX kernels ran (`ONEDNN_VERBOSE` excerpt /
   profile), no silent AVX-512 fallback where AMX was expected.
4. **Performance (absolute)** — `roofline-validation` per hot op: eff_abs vs
   achievable ceiling, with freq + FLOP/cycle assumptions.
5. **Performance (peer-relative)** — `peer-relative-roofline`: per op eff_rel vs the
   donor running the identical kernel; end-to-end normalized prefill+decode vs peer.
6. **Determinism** — repeated runs bit-stable / within noise.
7. **Provenance** — kernels + skills + fallbacks used, model + config hash, node
   profile, plugin commit, reproduce commands.
8. **Diff** — the plugin-only patch (thin subclass, zero `sglang/` edits),
   lint/format-clean to SGLang conventions, ready to open as a PR.
9. **Roofline target vs measured (PUBLISHED artifact PAIR — auto-emit at cycle close).** Every
   published result carries, as the CLOSING STEP of the perf cycle, TWO linked artifacts: (a) the
   roofline-achievable TARGET next to the MEASURED number (model level AND per hot op) via
   `plugin/validate/roofline_vs_measured.py`, AND (b) the time-attribution PIVOT companion
   (`plugin/validate/time_attribution_pivot.py`, where the wall-clock goes per phase, summing to
   100% with an explicit unattributed slice) — PLUS the model's bullet in the README "Roofline
   target vs measured" section. All three (roofline report, pivot report, README bullet) are
   required; a baseline→optimized "journey"/summary chart is NOT a substitute (it answers a
   different question and must not be mistaken for "charts done"). Publish the target UP FRONT
   (measured usually falls below it); the per-op efficiency-gap chart, ranked by recoverable
   fraction (shortfall × share), shows exactly which kernels underperform their roofline and are
   the next optimization RoI. For a recurrence/small-op-bound hot op, caption the realistic
   (fused-kernel) floor so a low FLOP-roofline efficiency is not misread as failure. Link both
   reports/charts from the certificate and the repo results page.
   **Same machine config, always labeled:** roofline and measured MUST be at the
   identical config (socket count, TP, batch, precision) — never a single-socket
   roofline against a dual-socket measured. State "single socket" vs "full node (2
   socket)" explicitly in the `node` field and the results table.

## Reviewer-facing hygiene blurb (put it where the boss/user looks FIRST)
A certificate proves rigor to someone who opens it — but leadership and users mostly see the
LANDING surfaces (README top, THESIS_SUMMARY, the workflow slide, a results page). Each of those
MUST carry a SHORT "what's checked, up front" blurb so a reader sees the right things were done
WITHOUT digging into the certificate. Keep it to a few lines enumerating the gates in place:
- **dtype hygiene** — stored→compute dtype audited per weight family; every `stored ≠ compute`
  dequant bridge parity-checked (not assumed).
- **coverage** — no novel op silently dense-approximated.
- **accuracy** — real-prompt coherence → per-layer parity → task accuracy.
- **honest roofline** — target vs measured at one labeled machine config.
Rules: (1) state it as a claim BACKED BY A LINK to the evidence (certificate / parity test /
coverage yaml), never a bare assertion; (2) any number not yet proven is labeled **UNVALIDATED /
PENDING**, never implied as done — an honest "in progress" blurb beats an overclaim a reviewer can
puncture (this is exactly what pre-empts the "did you actually verify X?" question). This blurb is
the human-facing dual of the machine-checkable certificate; publish both.

## Pass rule
GREEN only from a successful `enablement_gate.py --through final` invocation:
coverage complete, accuracy within approved tolerances at both operator/layer and task levels,
native dispatch confirmed, standalone and integration performance accepted, declared peer comparisons
≥ 0.90, determinism accepted, publication checked, and no core edits/lint failures.
Any miss → BLOCKED with the failing check; earlier phase success is only PHASE_PASS.
These thresholds use justified matching targets and are not sufficient alone:
every standalone kernel also needs canonical completion, native dispatch, source-conformed
precision, and explicit opportunity dispositions. Version 1 blocks missing peer evidence;
an alternative route requires a separately reviewed policy/implementation, not a waiver string.

## Throughput-thesis metric (the business proof)
Record per model:
- `wall_clock_hours` — config in to green certificate out.
- `automation_pct` — steps completed with no human intervention.
- `human_review_minutes` — expert time to accept the certificate + diff.
- `throughput_vs_generic` — serving tok/s vs the pre-optimization generic-CPU run.
Plot these against the historical month+ baseline — that delta IS the thesis.

## Procedure
1. Obtain the reviewed plan and its independently supplied digest. Implement/review the necessary
   source-faithful adapters; collect raw evidence, never synthesize missing numbers.
2. At each phase exit, record the adversarial audit, resolve findings and obtain required human
   approvals bound to that prefix. Run the executable with `--through <phase>`.
3. At final acceptance, run with `--through final`; publish its generated JSON, the pinned
   plan/evidence bundle, reproduce commands and plugin-only diff. Never hand-fill a GREEN field.
4. On BLOCKED, return to the failed check without replacing the previous accepted report.
   On GREEN, hand the reviewed deliverable to the human expert for the upstream PR.
5. Only after acceptance, append the model's provenance-bound efficiencies to the donor registry
   so the next enablement starts warmer.

## Gate
A certificate is valid only for its approved domain and pinned identities, after successful
full-prefix replay and human approval. Preserve raw observations, negative controls and audits.
Predictions are not measurements; cross-node results are not matched comparisons.
Source semantics and audit completeness still require reviewed adapters and expert scrutiny.
