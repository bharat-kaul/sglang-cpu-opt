# DSv4 Roofline — Author Response to Fifth Review (H1 certification label)

Responds to [dsv4_flash_roofline_gate_review5.md](dsv4_flash_roofline_gate_review5.md). The remaining H1
gap — reports printing **CERTIFIED correctness** for loaded kept passes — is closed by scoping the
correctness claim to the microbench and marking end-to-end verification pending. No new hardware runs.

Run: `python plugin/validate/dsv4_roofline_p2.py --selftest` (29 checks, all PASS, fail-closed).

## H1 — Correctness was labeled CERTIFIED despite unresolved revisions / mismatched shapes — FIXED

A recorded cosine/set-match is **not context-independent**: it is scoped to the tested
implementation/shape/dtype/layout/reference/tolerance. The reports no longer call it CERTIFIED.

- `observation_evidence()` now returns the raw record correctness plus a status
  `"<metric> certified @ microbench shape; E2E verification PENDING (current target)"`
  (`metric` = cosine or set-match, derived from the record). The misleading `certified_correctness`
  key is gone.
- Both canonical reports render it accordingly:
  - main report: `correctness: <record string>  [<metric> certified @ microbench shape; E2E verification PENDING (current target)]`
  - join: column header `correctness (microbench; E2E PENDING)`, with a footer stating the microbench
    match is not a current-target certificate.
- What is retained vs withheld, explicitly: the **microbench cosine/set-match IS a real auditable fact**
  (kept, read from the record); **E2E / current-target validity is a separate claim that stays PENDING**;
  kernel revision is derived from the kept pass (`UNRESOLVED(pending)` where applicable); absolute latency
  and speedup remain withheld/UNVERIFIED.
- Self-test adds a check asserting every observed row is labeled `E2E verification PENDING` and never a bare
  CERTIFIED (29 checks total). The accepted H1 record-reading/withholding mechanics and the G1/G3 numerical
  + tracker fixes are unchanged.

## Playbook lesson corrected
The roofline-accounting "measurement" rule previously said correctness was "context-independent" — the
reviewer's correction is now encoded: a recorded cosine/set-match is scoped to its tested contract and must
be reported as a microbench result with E2E verification pending, never a current-target certificate.
