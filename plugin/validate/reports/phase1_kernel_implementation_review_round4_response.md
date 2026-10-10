# Response to the Round-4 Kernel Review (assessment of the round-3 response)

Responding to [phase1_kernel_implementation_review_round3_response_assessment.md](plugin/validate/reports/phase1_kernel_implementation_review_round3_response_assessment.md) (published `90aa669`). Response commit `29c6a79`. No native kernel source changed; the five non-sparse kernels are untouched since round 2. Verified locally: full `f4_acceptance.py` run = STATUS **PARTIAL**, `--selftest` = **SELFTEST OK** (all positive/negative controls pass), `promotion_gate.py --selftest` = **OK / live BLOCKED**.

**Summary:** I accept every finding. All five (A1 High, A2 High, A3 Medium, A4 High, A5 Medium) are fixed in code with a dedicated typed control for each, and the residual claim-drift the review flagged is corrected at its source. One item is advanced from "deferred" to a **concrete proposal for your ratification** (the sparse serving error budget + the phase-dependency authorization), described at the end. Status remains PARTIAL and promotion remains BLOCKED; these fixes harden the gate, they do not request sign-off.

## Fixes

### A1 (High) — selection check rejected valid `-inf` masks
The blanket finiteness test was a false-positive regression: the pinned `Indexer.forward` adds `-inf` to causal/padding positions before top-k, so `-inf` is a **supported domain**, not malformed evidence.
- Added `_invalid_evidence()` ([plugin/validate/f4_acceptance.py](plugin/validate/f4_acceptance.py#L291)): rejects **NaN and +inf only**, allows `-inf`. Applied to the selection path ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L370)).
- **Positive control** (case 18): a valid `-inf` masked selection must **not** be rejected — [f4_acceptance.py](plugin/validate/f4_acceptance.py#L757). **Negative control retained** (case 19): NaN evidence is still rejected — [f4_acceptance.py](plugin/validate/f4_acceptance.py#L768). Both pass.

### A2 (High) — composed screen did not use the published comparison contract
Two mismatches, both fixed in the composed branch ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L313)):
- **Operand matching:** the candidate now receives the **same bf16-rounded** q/gathered-KV the reference consumes (`qb = q.bfloat16().float()`, gather `.bfloat16().float()`), so the screen measures the kernel's arithmetic, not input quantization.
- **Published ordering:** the authoritative reference now uses `torch.topk(..., sorted=True)` ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L329)) — the published default — and builds its 64-entry blocks from that order, while the **candidate deliberately keeps its own selection order**. The ordering consequence on the blockwise online-softmax is therefore **measured** (not hidden under a `sorted=False` reference that only matched membership). The original-FP32-input path is retained as a **separately-labelled diagnostic** (`metric_fp32in_diag`), so input-quantization and arithmetic effects stay distinguishable.

This is screening (non-gating under the unratified policy); the point addressed is the comparison contract, not the PARTIAL verdict.

### A3 (Medium) — composed-repeat and reference-dtype bypassed the contract
- `_same_contract()` is now module-level and checks **dtype + shape + device** ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L286)); the composed-repeat check uses it ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L327)) instead of a bare `torch.equal` (which crossed fp32→fp64).
- The **reference** is now validated like the candidate: the tuple3 path checks reference dtype/device per element ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L436)); the generic-continuous path checks reference dtype/device ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L453)) so an equal-valued fp64 reference is no longer accepted.
- **Controls:** fp64-reference rejection (case 20, [f4_acceptance.py](plugin/validate/f4_acceptance.py#L775)) and composed-repeat dtype-flip rejection (case 21, [f4_acceptance.py](plugin/validate/f4_acceptance.py#L782)). Both pass.

### A4 (High) — replay's `validated` flag was not expected-source enforcement
[plugin/validate/test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py) now binds evidence to the expected published source rather than trusting a producer flag:
- Compares `kernel_py_sha256` to `EXPECTED_KERNEL_SHA` ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L21), [L48](plugin/validate/test_sparse_cpu_vs_gpu.py#L48)), **env-overridable** (`ORACLE_KERNEL_SHA`) rather than permanently hardcoded; optional caller-supplied `ORACLE_JOB` run identity.
- Rejects an **empty** record inventory and any record missing required coordinates (`N, scale, sink, gpu_out, q, kv`) before any comparison ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L59)). (`weights_only=True` + allowlist from round 3 retained.)

### A5 (Medium) — cases were removable without an inventory failure
- Shipped sparse paths extended to the **full M-sweep {1,8,16,32,64}** ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L246)).
- `run()` now enforces **declared coordinates**, not mere presence: required indexer paths incl. bf16-KV ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L498)), masked-compressor `(op,kind)` added to `_REQUIRED_COVERAGE`, and per-path sparse M-sweep completeness ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L502)).
- **Controls through `run()`** (case 22): independent removal of the bf16-KV coordinate, the masked-compressor coordinate, and the sparse M16 coordinate each hard-fails `run()` ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L800)). All pass.

### Claim-drift corrected at source
Removed the unproven **"op-order floor"** language — an observed residual is not a proven minimum attainable error — in the sparse screen comment, the `ORACLE_CONFORMANCE_BY_OP["sparse"]` metadata, and the budget serialization, each now reading "**observed replica-vs-GPU discrepancy, not a proven minimum / not characterized hardware noise**". Corrected the stale **"globally-normalized"** budget note to **blockwise (source-faithful)**.

## Verified closures vs the round-4 findings

| Finding | Status |
|---|---|
| A1 valid-mask rejection | **Fixed** + positive/negative controls |
| A2 composed operand-matching + published ordering | **Fixed**; FP32-input kept as labelled diagnostic |
| A3 composed-repeat + reference dtype/device | **Fixed** + two controls |
| A4 replay expected-source + inventory | **Fixed** (sha bind, empty/under-specified reject) |
| A5 coordinate inventory + sparse M-sweep | **Fixed** + three inventory-through-run controls |
| claim-drift ("floor", "globally-normalized") | **Corrected at source** |

## Proposed pending item (for your ratification)

The one substantive item I now put forward as a **concrete proposal** rather than a deferral is the **sparse continuous serving budget + the phase-dependency authorization**. The derivation is specified in [kernel_opt_queue.json](plugin/validate/results/kernel_opt_queue.json) `_downstream_budget_methodology`:

1. Take the kernel's measured output residual ε (sparse ~1.95e-3 vs the source-faithful blockwise replica).
2. Inject an ε-scaled perturbation at the kernel's output boundary in the model proxy.
3. **Propose** ε on a **calibration** split only; **confirm** it generalizes on a **reserved, untouched validation** split — ε is never chosen on held-out data.
4. Keep **dummy-weight structural** propagation separate from **real-weight sensitivity + task** validation.

**Phase-dependency cycle (surfaced, not silently resolved):** Phase-1 sign-off nominally needs ratified budgets, but their derivation is a Phase-2 full-model experiment. I request one of two explicit human decisions: **(a)** authorize a bounded pre-sign-off qualification experiment to derive the budget, or **(b)** amend the phase dependency so Phase-1 closes on screening + selection gates with the budget tracked as a Phase-2 obligation. Until either is ratified, sparse continuous stays **screening-only**, `f4_acceptance.py` stays **PARTIAL**, and `promotion_gate.py` stays **BLOCKED**.

## Remaining PENDING (unchanged; honest)

- R3-F3: an independently-justified sparse serving error budget (the proposal above) and the expanded GPU/caller qualification (persist the B4 / K∈{128,160,640} / sentinel oracle io for C++ replay — currently only N∈{1,8,64} records are saved).
- Full launcher timing/runtime identity binding (thread counts/placement, binary-library hashes, input hashes).
- Comparator-drift caveats on the performance table remain as documented in round 3 (not attributable to new optimization).

This response grants itself no waiver, budget, or phase promotion.
