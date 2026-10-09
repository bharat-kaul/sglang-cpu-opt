# Response to Playbook Review Round-2 (Response Assessment)

**Responding to:** [model_agnostic_playbook_review_round2.md](model_agnostic_playbook_review_round2.md) (reviewer commit `cca2832`).
**From:** executor. This responds to the four remaining findings + the Minimum Next Delivery. I **agree with the assessment**; two findings were my own recurring lessons (fixed the *instance* not the *class*; slipped back into unproven causal labels). Corrections are at the decision points, not new caveats.

## Disposition of remaining findings

| # | Finding | Disposition | Action (this round) |
|---|---------|-------------|---------------------|
| 1 | Contradictory numerical/proxy rules remain | **Agree — fixed at the decision points** | `kernel-authoring` step 5 **and the ship Gate** now require the **SOURCE-CONFORMED** reference (FP32 math is a bring-up diagnostic only); `perf-proxy` real-vs-dummy "is a BUG" is now **CONDITIONAL** (bug only for per-layer-dominated weight-value-independent cost; VERIFY for routing/sparsity/state/cache); the playbook "downconvert only the final store" rule now **defers to source-stage rounding** (match intermediate stage boundaries when the published op rounds there). |
| 2 | Core enforcement is prose, not an executable check | **Agree — bounded first slice delivered** | New [promotion_gate.py](promotion_gate.py): the single EXECUTABLE promotion decision. Fail-closed; BLOCKS on PARTIAL/FAIL F4, UNRATIFIED policy, PENDING obligations, dirty tree, or missing/stale human approval. Binds evidence identity (git sha, reference revision). Connects to F4 (`run()`), the queue, and the policy. Selftest proves it blocks PARTIAL/FAIL/unratified/stale-approval. **Live decision today: BLOCKED.** |
| 3 | Parity / numerical-floor labels unsupported | **Agree — corrected** | Removed "parity" and "bf16 op-order floor". Indexer M8 now reported as its three process-medians **0.98/1.05/0.96×** (observed variability straddling 1.0, not a tie). Sparse 1.95e-3 reported as an **observed max-abs discrepancy** vs the replica, cause not characterized as intrinsic/hardware noise, not an acceptance bound. |
| 4 | Coverage enforcement still partial | **Agree — partial; incremental** | (op,kind)+indexer-M set is in place; full shape/layout/state/dispatch domain + binding declared-M to the evaluated tensor is **PENDING**, to extend alongside the shared gate. Scope of each result is labelled (synthetic microbench vs captured vs full-model). |

## Correcting my own overclaim

The round-1 response described the mandate block as "mechanized enforcement." That was wrong — **a written mandate guides execution; it is not an executable check.** The executable enforcement is `promotion_gate.py` (new) + the F4 harness; the mandate block is policy text those gates enforce. I've stopped calling prose "mechanized."

## Precise implemented vs pending

| Item | State | Evidence |
|---|---|---|
| Typed negative-test rejection reasons (F4) | **IMPLEMENTED** | `f4_acceptance.py --selftest` (14 typed checks); found + fixed real latent bugs |
| Independent (op,kind)+indexer-M coverage inventory | **IMPLEMENTED (partial domain)** | `f4_acceptance.py` run() inventory checks |
| Conflicting numerical/proxy rules removed at decision points | **IMPLEMENTED** | kernel-authoring step5+Gate, perf-proxy, playbook precision rule |
| Executable promotion gate (fail-closed, identity-bound) | **IMPLEMENTED (first slice)** | `promotion_gate.py` (+ `--selftest`); live = BLOCKED |
| Claim-strength corrections (no parity/floor) | **IMPLEMENTED** | this doc + phase1_kernel_review.md |
| Source-faithful 64-block sparse replica + coverage | **IMPLEMENTED** | `sparse_ref.py`; job 384532 (≤1.95e-3, batches/K-unions/sentinels) |
| Replicated same-contract no-regression (all kernels) | **IMPLEMENTED** | job 384531 (order-varied, identity-bound) |
| Full shape/layout/state/dispatch coverage + declared-M↔tensor binding | **PENDING** | incremental extension of the inventory |
| Dependency-based evidence invalidation (STALE propagation) | **PARTIAL** | promotion_gate flags dirty/stale-approval; full per-record STALE PENDING |
| Evidence-derived claims (reports generated from records) | **PENDING** | principle #8; to land with the shared evidence interface |
| Skill discovery verification + one prerequisite graph | **PENDING** | to verify the runtime locator; mandate block is interim |
| Downstream error budgets (ratified) | **PENDING** | Phase-2 full-model propagation; reviewer-ratified |
| Broad adapter framework + two-model portability demo | **DEFERRED** | until a second model family is queued (portability explicitly UNVERIFIED) |

Accepting the reviewer's correction on YAGNI: the portability *demo* is a test of generality, not a prerequisite for a framework — I am not conflating them. The local rule-corrections, the bounded promotion gate, and incremental coverage are done/underway **now**; only the broad framework + portability demo are deferred.

## Not ratified here

No tolerance, numerical bound, or policy is ratified by this response. `promotion_gate.py` returns **BLOCKED** until a human ratifies the acceptance policy and records an approval bound to the evidence sha. Reference conformance, policy, and deployment remain the human reviewer's authority.
