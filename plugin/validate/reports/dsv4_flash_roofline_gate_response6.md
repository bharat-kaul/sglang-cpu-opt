# DSv4 Roofline — Author Response to Sixth Review (H1 certification claim)

Responds to [dsv4_flash_roofline_gate_review6.md](dsv4_flash_roofline_gate_review6.md). The reviewer is
correct: the prior status was a keyword-derived `<metric> certified @ microbench` label constructed
UNCONDITIONALLY — it would "certify" an absent or `FAIL` record. Fixed via the reviewer's minimal
reporting-only closure (no new hardware runs).

Run: `python plugin/validate/dsv4_roofline_p2.py --selftest` (33 checks, all PASS, fail-closed).

## H1 — Certification inferred from free text — FIXED

- `observation_evidence()` / `_evidence_from_record()` no longer infer a metric or a positive match. The
  recorded correctness field is reported **VERBATIM** under a single fixed attribution:
  `Historical author-reported microbench result; test scope UNVERIFIED; current-target/E2E verification PENDING`.
  The word "certified" is gone from the evidence entirely.
- Outcome is never asserted: an absent result renders `unrecorded`; a reported failure (e.g.
  `FAIL: numerical mismatch`, `cos 0.0; FAIL`) is retained verbatim with **no** positive-match status added;
  an unresolved revision stays `UNRESOLVED(pending)`.
- Applied consistently to **both** canonical reports (main AUDITABLE OBSERVATIONS block and the join),
  with the attribution printed in each.
- **Regression checks added** exercising the real reader+evidence function on the reviewer's three
  counterexamples (missing correctness, `FAIL: numerical mismatch`, `cos 0.0; FAIL`): each asserts the field
  is preserved literally, the revision stays unresolved, and **no** certification string is produced. The
  missing-record fail-closed check is preserved. Total 33 self-tests.
- Accepted items unchanged: record-derived revisions, withheld absolute latency, UNVERIFIED speedup, G1/G3
  numerical + tracker fixes, and all declared ideal formulas.

## Playbook lesson corrected
The measurement rule now states: report the recorded correctness field VERBATIM under the fixed UNVERIFIED
attribution; do NOT infer a positive match/metric/certification from free text; absent → `unrecorded`, a
failure is retained as a failure; regression-test that missing/failing/failed-cosine evidence yields no
certification.
