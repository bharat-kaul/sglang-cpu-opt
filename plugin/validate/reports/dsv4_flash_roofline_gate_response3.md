# DSv4 Roofline — Author Response to Third Review (G1–G3)

Responds to [dsv4_flash_roofline_gate_review3.md](dsv4_flash_roofline_gate_review3.md).
All three blockers fixed; verified R1/R3 + F-series corrections preserved. Self-test now 26 checks,
all PASS, fail-closed.

Run: `python plugin/validate/dsv4_roofline_p2.py --selftest`.

## Disposition

### G1 — Sparse-attention join applied the BF16 ceiling to FP32 work — FIXED
- The join now carries an EXPLICIT compute dtype per row and selects the compute resource from it
  (`CPEAK`: bf16 → AMX 124.5 TF, fp32 → AVX-512 7.78 TF). The sparse-attention row (FP32 KV/query/
  output per `bench_sparse_attend.py`) is `fp32` → ideal **275.9 µs**, compute-bound — matching the
  reviewer's 275.941053 µs counterexample (was 117 µs / bandwidth under the wrong AMX ceiling).
- A fail-closed assertion requires the sparse-attention row's compute dtype to be `fp32`.

### G2 — Indexer observation did not match its claimed benchmark contract — FIXED
- Re-pointed the indexer row to **`bench_idx_logits.py`**, which actually exercises M=32 / ctx=1024
  with per-request keys. Counted its **FP32 public boundary** — `q[M,H,D] + kv[M,S,D] + w[M,H] +
  logits[M,S]` = **17,965,056 bytes**, FP32 compute — not the BF16 9,052,160 boundary. The BF16
  projected-query boundary is kept only as a labeled PROSPECTIVE target, not attached to the
  observation.
- The absolute 0.536 ms is **not present in any auditable raw record** (the op_passes records hold
  correctness + speedup ratios, not absolute node latency), so it is **WITHHELD** rather than
  invented. This is applied to the whole class: `topk`, `sinkhorn`, `combine`, and the indexer are
  now `observed` rows that render no absolute latency.
- Each observation links an **auditable result record** (`results/op_passes/*.json` + kernel commit)
  and states what it certifies (correctness + speedup-vs-torch @M=32). The canonical main report is
  now self-contained: it prints an **AUDITABLE OBSERVATIONS** block with each record@commit +
  certified quantity (the previous footer referred to notes that were never printed).

### G3 — Tracker validation accepted invalid/incomplete content — FIXED
- Disposition is now a **bounded enum** checked by exact match (not `startswith`); qualifiers move to
  a separate `qualifier` field (`wo_a` updated). `MODELED_BOGUS` is rejected.
- Required fields enforced: every item needs a non-empty unique `item` identity; `MODELED`/`MEASURED`
  need a `source`; `EXPLICITLY-UNMODELED` needs `reason` **and** `plan`. Duplicate identities rejected.
- **Coverage** is enforced: a set of known exclusion categories must remain present, so removing all
  `EXPLICITLY-UNMODELED` items (or any required category) now fails — previously such removals were
  silently accepted.
- The self-test exercises the **real** `validate_tracker()` with the reviewer's exact invalid inputs
  (`MODELED_BOGUS`; an item with only `{"disposition":"MODELED"}`; all unmodeled items removed;
  duplicate identity) and asserts each is rejected — not a re-implementation on a BOGUS string.

The deferred reusable-skill occupancy-formula edit remains a tracked follow-up for the playbook phase.
