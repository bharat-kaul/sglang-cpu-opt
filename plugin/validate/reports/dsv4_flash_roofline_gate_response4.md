# DSv4 Roofline — Author Response to Fourth Review (H1)

Responds to [dsv4_flash_roofline_gate_review4.md](dsv4_flash_roofline_gate_review4.md). The single
remaining blocker (H1) is fixed by READING and VALIDATING every observation against its external result
record, rather than hand-typing certification strings. The accepted G1/G3 numerical + tracker fixes are
unchanged.

Run: `python plugin/validate/dsv4_roofline_p2.py --selftest` (all PASS, fail-closed).

## H1 — Published certification claims were not validated against their records — FIXED

Chosen approach: validate against the records (and withhold what the structured records do not support).

- `observed()` now takes only a **record filename** — no hand-typed `commit` or `certifies` string.
  `load_record()` opens `results/op_passes/<record>` (fail-closed on missing/malformed/no-kept-pass) and
  `observation_evidence()` DERIVES the evidence from the record's **kept** pass:
  - **kernel revision** is read from the kept pass (separate from any narrative). The top-k row therefore
    no longer publishes `06faec0` (a *discarded* pass); it reads `pending` and renders
    `UNRESOLVED(pending)`. A false result-to-revision association is now structurally impossible.
  - **correctness** (context-independent) is read from the kept pass and is the only CERTIFIED quantity.
  - **speedup** is always rendered **author-reported / UNVERIFIED** — the records' structured ratios are
    at a superseded context and the corrected ratios live only in prose, so a narrative is never promoted
    to a certificate.
- Both canonical artifacts now derive this evidence from the records: the main report's AUDITABLE
  OBSERVATIONS block and the join (`dsv4_roofline_vs_measured.py`, which imports `load_record`) print
  `record`, `kernel-rev`, and `CERTIFIED correctness`, with speedup/absolute-latency withheld.
- The self-test now **opens every record** (fail-closed) and asserts: each observed record loads and has a
  kept pass; the top-k revision is not `06faec0`; speedup is never certified; and a missing record FAILS
  (the record is opened, not assumed). Re-running the reviewer's "make every `op_passes` open fail"
  reproduction now fails the self-test and blocks emission.

## Non-blocking note corrected
- The self-test now reports its checks live (the stale "26" count is dropped). `--selftest` prints each
  check and the aggregate PASS/FAIL.

Previously accepted G1 (FP32 sparse-attention ceiling, indexer FP32 public boundary, withheld absolute
latencies) and G3 (tracker enum/identity/coverage, fail-closed) are unchanged. Explicit partial-model
exclusions remain limitations, not a complete-inventory claim. The reusable-skill occupancy edit remains a
tracked playbook follow-up.
