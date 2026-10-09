# Audit trail — review-gated gate: kernel-impl-and-provenance

- executor: Claude Opus 4.8
- reviewer: GPT Astra 6
- started: 2026-10-09

## Round 1 — verdict: FAIL (CHANGES REQUIRED)

**Review report (reviewer: GPT Astra 6)** -> `plugin/validate/reports/dsv4_flash_implementation_review.md`
- 9 findings (F1-F9), several HIGH-severity real bugs, with executed C++/Python counterexamples.

**Executor response (standalone, hand to reviewer)** -> `plugin/validate/reports/kernel-impl-and-provenance_response_round1.md`

**Collateral + Addressing (executor: Claude Opus 4.8)** — per finding:

| # | Finding | Disposition | What changed (consequence, verified) |
|---|---------|-------------|--------------------------------------|
| F1 | Compressor NaN on valid overlap mask | **CLOSED** | online-softmax guards masked (-inf) positions (corr/e=0) + all-masked→0; reviewer's repro now returns finite all-ones (0 NaNs @ D=128/512) |
| F2 | Indexer dispatch changes numerics / top-k | **CLOSED** | integration entry uses ONE matmul contract (tiled, fp32 scores) for all M; M=1 vs M=8/16 now 0.0 diff, 512/512 selection; also collapsed M=1 latency 1.023ms→0.100ms |
| F3 | Sparse timings joined from WRONG variant | **CLOSED** | `parse_perf_sweep.py` ingests by NAMED column + explicit variant + validates finite/positive; sparse now the SCALAR column [0.068,0.46,0.894,1.793,3.64]; scalar ~3.6x M=1 advantage restored |
| F4 | Re-verification not fail-closed; false inequality | **PARTIAL** | false `0.999963 < 0.9999 FAIL` inequality corrected (it PASSES); machine-evaluated per-op acceptance policy + nonzero bench/job exits = **OPEN** (surfaced) |
| F5 | Provenance/reconciliation overstate | **CLOSED (core)** | `reconcile_kernels` now requires exact 6-kernel coverage + nonempty mappings (reviewer's dropped-entry + empty-mapping mutations now REJECTED, tested); stale sparse oracle citation fixed; `inference/kernel.py` named as the deep primitive source |
| F6 | Measured join costs different contracts | **CLOSED** | top-k output = INT64 (262,144 B, added to `_EXPECT`); indexer compute = BF16 (AMX) with FP32 public storage |
| F7 | BW-wall / no-ROI conclusion unsupported | **RETRACTED** | removed "not closable by kernel work" / "DRAM wall" claims from join + impl_review; recorded ranked OPEN ROI hypotheses; full Amdahl ledger = **OPEN** (surfaced) |
| F8 | Playbook M-sweep contradiction + routing | **PARTIAL** | M-sweep rule fixed ("evidence to interpret, not a verdict"); entry-skill (.agents) routing of the new gates = **OPEN** (surfaced) |
| F9 | C++ entry points lack shape guards | **CLOSED** | TORCH_CHECK device/dtype/rank/dimension/divisibility/scalar guards on all 6 kernels |

Perf RE-MEASURED on the fixed kernels (SLURM 384414, validated parser). All 37 generator self-tests pass.

**Addressing**: committed; the next round (GPT Astra 6 re-review) will verify the closures and the OPEN items.

## Outcome: SURFACED to user

6 findings CLOSED (F1, F2, F3, F5, F6, F9) + F7 retracted + F8 M-sweep fixed. **OPEN items SURFACED for a user call** before certification:
- F4: a declared machine-evaluated per-op acceptance policy (finiteness + abs/rel error + discrete decisions) and fail-on-error bench/job exits — a design choice (tolerances) needing your input. **KEPT OPEN in the executor todo.**
- F7: a call-count-weighted Amdahl ROI ledger (needs in-engine attribution / profiling) before any exhaustion claim.
- F8: routing the new kernel-provenance/reconciliation gates through the `.agents` entry skills (the "old scheme" deferred set).
- Donor-dispatch proof for sparse_attend (concrete donor entry + capability + dispatch evidence).
Nothing is certified; correctness is microbench scope, E2E verification PENDING.

## Round 2 — verdict: PENDING (GPT Astra 6 re-review TRIGGERED)

**Trigger (executor: Claude Opus 4.8)** — base commit db5c5fd. Round-2 reviewer collateral emitted →
`tools/review_loop/runs/kernel-impl-and-provenance/reviewer_round2.prompt.txt`. GPT Astra 6 is asked to
INDEPENDENTLY re-verify every round-1 closure (F1, F2, F3, F5, F6, F9) by counterexample (re-injecting the
old behavior must fail), confirm the retractions/corrections (F4 inequality, F7 overclaim, F8 M-sweep rule)
are gone-not-relabeled, and assess the OPEN items' plans.

**Pass-scope**: round 2 passes only if no round-1 closure is unsound. The four STILL-OPEN items are
explicitly OUT of pass-scope:
- **F4 — KEPT OPEN (executor todo):** machine-evaluated per-op acceptance policy + fail-closed bench/job
  exits; **tolerances await a user decision**. Reviewer to state the minimum acceptance-policy contract.
- F7 Amdahl ROI ledger (profiling); F8 `.agents` entry-skill routing; sparse donor-dispatch proof.

**Expected reviewer output** → `plugin/validate/reports/kernel-impl-and-provenance_review_round2.md` +
one `<<<REVIEW-VERDICT {...} REVIEW-VERDICT>>>` block (PASS / FAIL / SURFACE). Awaiting GPT Astra 6.
