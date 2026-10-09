# DSv4-Flash kernel implementation + provenance review — executor response, round 4

- **gate**: kernel-impl-and-provenance
- **executor (responding)**: Claude Opus 4.8
- **reviewer (addressed)**: GPT Astra 6
- **reviewer report**: [kernel-impl-and-provenance_review_round4.md](kernel-impl-and-provenance_review_round4.md) (round 4, verdict FAIL, commit `8874f23`; R3-P1 CLOSED; findings R4-F5, R4-F3)
- **date**: 2026-10-09

Round 4 closed R3-P1 and found two deeper integrity gaps in the destination binding and the parser schema.
Both are now closed and verified through the full gate. Nothing is certified — microbench scope, E2E PENDING.

## Per-finding disposition

| # | Finding (reviewer) | Disposition | What changed — consequence, verified |
|---|--------------------|-------------|--------------------------------------|
| R4-F5 | Valid gap IDs accept empty / `caller:` / unrelated / wrong-existing destinations; the full generator self-test still returns True through the file-read path | **CLOSED** | `_KERNEL_GAP_CONTRACT` is now a `gap_id → REQUIRED-DISPOSITION` map (not just an id set): each entry's disposition must EQUAL its one required destination (kind + exact target). The generic resolver is hardened to reject an empty `unmodeled:` sub (which previously matched every tracker item), an empty/implicit `caller:` reference, and nonexistent modeled rows. The reviewer's exact 4 repoints of compressor `wkv_wgate_r4` (`caller:`, `caller:NO_SUCH_CALLER`, `unmodeled:`, `modeled:MHC hc_fn (16384->24, FP32)`) are all REJECTED (tested); generator self-test rc=0 (47 PASS). |
| R4-F3 | Duplicate header name silently selects the last column; a malformed-then-retried REP block disappears from evidence; truncated/unparseable rows are skipped; negative M ignored by `isdigit()` | **CLOSED** | parser now (a) RAISES on a DUPLICATE header column name (ambiguous selection); (b) treats a SECOND real `>>> BENCH=… REP=…` block for the same `(bench,rep)` as a duplicate run and RAISES — independent of whether the first block yielded parseable samples; (c) inside a recognized table, RAISES (does not skip) on an unexpected/negative M coordinate, a truncated row missing the named column, an unparseable value, or a non-finite/non-positive time. `--selftest` +4 cases (duplicate-header / unparseable-value / truncated-row / negative-M) — all PASS (13 total). Real 384414 medians unchanged. |

## Reviewer-confirmed-good and carried: R3-P1 (report regeneration + identity separation + `--verify` drift check), and all prior round-1/2/3 closures.

The reviewer's note that row/gap identifiers do not replace captured operand contracts, and that R=8 pool /
full window+compressed sparse coverage and FP32-vs-BF16 boundary distinctions remain, is acknowledged — those
are perf-phase coverage items (and are written into the F4 policy draft below), not provenance-gate bypasses.

## F4 — DRAFT policy emitted for your ratification + reviewer suggestion (you asked to share these)

Two documents to hand the reviewer:
- **DRAFT policy** → `plugin/validate/results/acceptance_policy.json` (PROPOSED / UNRATIFIED). Three layers:
  (0) tolerance-independent hard gates (finiteness, shape/dtype, input-domain guards, thread-count
  determinism, fail-closed bench+job exits); (1) numerical acceptance vs the AUTHORITATIVE oracle (BF16
  primitive for AMX paths, FP32 oracle for scalar; cosine + relerr on normal + heavy-tailed; bit-exact where
  achievable); (2) discrete-selection acceptance (top-k set-match with a tie band — the gate for the indexer).
  Per-op oracle selection and proposed starting tolerances are included. The numerical tolerances are a USER
  design decision; the structure/oracles/layer-0 gates are implementable before the numbers are set.
- **Reviewer ask** → `tools/review_loop/runs/kernel-impl-and-provenance/reviewer_F4.prompt.txt` (7-point design
  review: oracle selection, metric adequacy, tie rule, bit-exact classification, fail-closed enforcement,
  shape coverage, proposed numbers). The reviewer SUGGESTS; you RATIFY.

## Still OPEN (explicitly OUT of pass-scope)

- **F4** acceptance policy — DRAFT proposed; **tolerances await your decision** + reviewer suggestions. The
  layer-0 hard gates + bench/job exit propagation can be implemented now, independent of the tolerances.
- **F7** full call-count-weighted Amdahl ROI ledger (profiling); optimization exhaustion is NOT established.
- **F8** `.agents` entry-skill routing; **sparse donor-dispatch proof**.

## Artifacts to re-verify against

- Parser + self-test: `plugin/validate/parse_perf_sweep.py` (`--selftest`, 13 cases), `results/perf_sweep.json`
- Reconcile + self-test: `plugin/validate/dsv4_roofline_p2.py` (`--selftest`, `reconcile_kernels`, `_KERNEL_GAP_CONTRACT`)
- Join + regeneration guard: `plugin/validate/dsv4_roofline_vs_measured.py` (`--verify`), `reports/dsv4_roofline_vs_measured_emr.txt`
- Records: `plugin/validate/results/{kernel_provenance,impl_review,acceptance_policy}.json`
- Audit trail: `tools/review_loop/runs/kernel-impl-and-provenance/{audit_trail.md,ledger.json}`
- Round-5 reviewer package: `tools/review_loop/runs/kernel-impl-and-provenance/reviewer_round5.prompt.txt`

## Outcome

**SURFACE** — both round-4 findings closed + verified; the F4 draft + reviewer ask are emitted for your
ratification; four items remain OPEN. Next: GPT Astra 6 round-5 re-review of the round-4 closures (and,
separately, the F4 design review). Rounds are driven manually (the automated loop is disabled).
