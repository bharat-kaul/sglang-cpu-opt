# DSv4-Flash kernel implementation + provenance review — executor response, round 3

- **gate**: kernel-impl-and-provenance
- **executor (responding)**: Claude Opus 4.8
- **reviewer (addressed)**: GPT Astra 6
- **reviewer report**: [kernel-impl-and-provenance_review_round3.md](kernel-impl-and-provenance_review_round3.md) (round 3, verdict FAIL, commit `48a3312`, findings R3-F3/F5/P1)
- **date**: 2026-10-09

Round 3 was correct again: the round-2 closures were real improvements but incomplete at the integrity
layer. The reviewer's executed probes (parser mutations, reconcile mutations, artifact regeneration diff)
exposed three deeper gaps, now closed and verified. Nothing is certified — microbench scope, E2E PENDING.

## Per-finding disposition

| # | Finding (reviewer) | Disposition | What changed — consequence, verified |
|---|--------------------|-------------|--------------------------------------|
| R3-F3 | Duplicate samples overwrite evidence; invalid value overwritten before validation; extra-M silently ignored | **CLOSED** | `parse_perf_sweep.py` now (a) matches BENCH markers at LINE START only, so shell `set -x` echo traces are not mistaken for real runs; (b) validates each value for finiteness/positivity the instant it is read, before any store, so an invalid sample cannot be overwritten; (c) RAISES on a duplicate `(bench,M,rep)` sample instead of letting a dict overwrite it; (d) RAISES on any unexpected M coordinate (exact coverage). `--selftest` adds duplicate / NaN-before-overwrite / extra-M=128 / echo-trace-ignored cases — all PASS (9 total). Real 384414 medians unchanged. |
| R3-F5 | Nonempty gap list ≠ complete accounting; 4 mutations pass; duplicate declaration loses identity | **CLOSED** | added an INDEPENDENT in-code stable gap-ID inventory (`_KERNEL_GAP_CONTRACT`) that cannot be deleted alongside the document's evidence; each reconciliation entry's `gap_dispositions` must carry `gap_id`s that EXACTLY and UNIQUELY cover it; duplicate `kernels[]` declarations are rejected BEFORE set conversion. The reviewer's exact 4 mutations — keep-only-first, unrelated disposition, delete-gaps+dispositions, duplicate-declaration — are all REJECTED (tested). Generator self-test rc=0 (43 PASS). |
| R3-P1 | Published report retains old identity conflation + stale wording despite corrected generator | **CLOSED** | `reports/dsv4_roofline_vs_measured_emr.txt` regenerated (0 stale `@e3fdcb9` conflation labels; now 6 HISTORICAL-correctness + 6 CURRENT-timing labels); the join is refactored to a deterministic `render()` with a `--verify <path>` regeneration guard (exits non-zero if the saved artifact drifts from the generator); the kernel/benchmark BUILD/SOURCE DIGEST of job 384414 is recorded as explicitly UNRESOLVED (launcher records node/time/config, not a source hash) and is NOT retroactively assigned to HEAD; residual "best-of dispatcher" (indexer source) and combine "minimal traffic achieved" overclaims corrected to the single-tiled-path fact and a hypothesis. |

## Reviewer-confirmed-good (carried): R2-F9, R2-F9b, R2-F3 subcases, R2-F5 subcases, R2-P1 source fix, R2-P2 comment.

The reviewer also noted (and I have preserved, not weakened) that `_KERNEL_COST_CONTRACT` binds row NAMES,
not typed operands; the independent OPS precision/shape self-tests (FP32→BF16 pool mutation + zero-pool-bytes
rejection) remain the operand/precision guard. The reconcile docstring now states this scope explicitly
rather than implying name-matching proves captured-input alignment.

## Still OPEN (explicitly OUT of pass-scope)

- **F4 — KEPT OPEN (executor todo):** machine-evaluated per-op acceptance policy. The reviewer reaffirmed the
  minimum contract (reference/precision, input domain, finite-output, abs/rel bounds, discrete-selection/tie
  criteria) and that the **finiteness + bench/job exit-status propagation can proceed without the tolerances**
  — only the numerical bounds are a **user design decision** (still awaited).
- **F7** full call-count-weighted Amdahl ROI ledger (needs in-engine attribution / profiling); optimization
  exhaustion is NOT established (the round-2 priority queue stands).
- **F8** `.agents` entry-skill routing; **sparse donor-dispatch proof** (concrete donor entry + capability +
  dispatch evidence).

## Artifacts to re-verify against (reference-first)

- Parser + self-test: `plugin/validate/parse_perf_sweep.py` (`--selftest`), `results/perf_sweep.json`
- Reconcile + self-test: `plugin/validate/dsv4_roofline_p2.py` (`--selftest`, `reconcile_kernels`, `_KERNEL_GAP_CONTRACT`, `_KERNEL_COST_CONTRACT`)
- Join + regeneration guard: `plugin/validate/dsv4_roofline_vs_measured.py` (`--verify <path>`), regenerated `reports/dsv4_roofline_vs_measured_emr.txt`
- Records: `plugin/validate/results/{impl_review,kernel_provenance}.json`
- Kernels: `plugin/kernels/dsa_pilot/indexer_logits.cpp`
- Audit trail: `tools/review_loop/runs/kernel-impl-and-provenance/{audit_trail.md,ledger.json}`
- Round-4 reviewer package: `tools/review_loop/runs/kernel-impl-and-provenance/reviewer_round4.prompt.txt`

## Outcome

**SURFACE** — three round-3 findings closed + verified; four items remain OPEN pending a user decision
(F4 tolerances) or profiling (F7). Next: GPT Astra 6 round-4 re-review of the round-3 closures. Rounds are
driven manually (the automated loop is disabled).
