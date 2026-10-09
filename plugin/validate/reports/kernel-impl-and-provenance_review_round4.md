# Kernel Implementation and Provenance: Round 4

2026-10-09. Reviewer: GitHub Copilot; no separate Astra execution is claimed. Reviewed commit: `e3ea6c43ef6f352d1a2f49dd4e9ecc4456324e4d`.

**FAIL: R3-F5 and R3-F3 are improved but not fully closed. R3-P1 is closed.** Reviewed the [round-3 response](kernel-impl-and-provenance_response_round3.md) against the [previous report](kernel-impl-and-provenance_review_round3.md). The four explicitly open items remain outside this partial verdict.

## Findings

### R4-F5 High: Gap IDs Are Checked, but Their Dispositions Can Still Be Empty or Wrong

[reconcile_kernels](../dsv4_roofline_p2.py#L515) accepts an empty `unmodeled:` destination because an empty string matches every tracker item. At [line 517](../dsv4_roofline_p2.py#L517), any `caller:` prefix bypasses destination validation. Keeping all required gap IDs, each of these changes to the compressor's `wkv_wgate_r4` disposition independently returns `(True, [])`:

- `caller:`
- `caller:NO_SUCH_CALLER`
- `unmodeled:`
- `modeled:MHC hc_fn (16384->24, FP32)`

The last target exists but is the wrong operation: an MHC projection does not account for compressor wkv/wgate. Exact ID coverage cannot establish that each gap is actually accounted for. The stable-ID change prevents deleting evidence, but not replacing its destination with an empty or unrelated one.

**Whole-gate consequence:** serving the modified provenance document with `caller:` through the normal file-read path still makes the full generator self-test return True with all **43** checks passing. This is not merely an isolated-helper bypass.

**Required closure:** bind each gap ID to its permitted disposition kind and destination, with nonempty exact tracker IDs and explicit caller references. Reject missing and unrelated targets, not just missing IDs. Grouped gaps need an explicit coverage relationship. Test these cases through the complete emission gate. This is the unresolved destination requirement from round 3, not a request for a new orchestration framework.

### R4-F3 Medium: Malformed Timing Tables Can Still Be Accepted as Valid Evidence

[parse_perf_sweep.py](../parse_perf_sweep.py#L60) constructs the header dictionary without rejecting duplicate names. A sparse header `M sc_ms sc_ms` with rows such as `1 nan 1.0` silently selects the last `sc_ms`; the first declared scalar latency is never validated. A complete synthetic sweep with such sparse headers is accepted.

At [line 67](../parse_perf_sweep.py#L67) and [line 71](../parse_perf_sweep.py#L71), truncated or unparseable timing rows are skipped before sample identity is recorded. A real sparse REP=1 block containing full-width rows `M-value 9.999 BROKEN`, followed by a valid REP=1 block and replicas 2/3, is accepted. The malformed attempt has disappeared from the evidence. A truncated numeric row before the retry is also accepted. Negative M rows are ignored by the `isdigit()` filter rather than rejected as unexpected coordinates.

These are schema/ingestion failures, not numerical-tolerance questions and not evidence that the existing job's medians are wrong. **Required closure:** validate a unique, unambiguous header and the grammar of rows belonging to that table; distinguish known trace/noise lines from malformed samples. Reject malformed numeric coordinates and failed timing conversions inside a recognized table. Define duplicate run-marker handling independently of whether its first attempt yielded parseable samples. Do not fix this solely by adding another literal test string.

## Verified Improvements

| Claimed fix | Independent result |
|---|---|
| Duplicate numeric sample, NaN-before-retry, extra M=128 | All rejected; round-3 exact parser counterexamples are fixed |
| Shell echo traces and bare integer noise | Correctly ignored without changing the result |
| Gap coverage and declarations | Prior four mutations rejected; missing, duplicate and unknown gap IDs rejected |
| Target lookup | Nonexistent modeled row and nonempty nonexistent tracker target rejected; empty/caller/wrong-existing targets remain open |
| Independent pool precision/byte checks | Full self-test rejects FP32-to-BF16 precision and zero-byte mutations |
| Published artifact | Fresh render exactly matches saved file; six HISTORICAL and six CURRENT labels; simulated drift exits 2 |

The **nine** committed parser self-tests and **43** generator self-tests pass on the unmodified records. Independent raw-log extraction by header name again matches all six sweeps in [perf_sweep.json](../results/perf_sweep.json), including sparse scalar `[0.068, 0.460, 0.894, 1.793, 3.640]` ms. Historical correctness records and timing values are unchanged by this response.

The report now explicitly declares job 384414's kernel/benchmark build digest **UNRESOLVED** and does not assign it to current HEAD. The old revision-conflation labels, inverted compute-floor wording, obsolete indexer best-of-dispatch description and combine minimum-traffic claim are corrected. R3-P1 is therefore closed, including the negative regeneration check.

## Scope and Remaining Work

No executable kernel logic changed in this response; the indexer diff is comments and export description only. Previous sparse-guard and zero-selection results are carried forward, not represented as newly rerun C++ tests. The pinned upstream contract remains `60d8d70770c6776ff598c94bb586a859a38244f1`; this round does not re-establish full numerical conformance or capture provenance.

The shape/precision distinctions remain: R=128/D=512 pool timing does not characterize both R=8 pools; FP32 public benchmark traffic is not the main BF16 boundary; sparse K=512 pre-gathered timing is not the complete 128/640/160 window-plus-compressed path. Row/gap identifiers do not replace captured operand contracts.

Optimization exhaustion is not claimed and is not established. No new performance experiments or full call-count-weighted ledger were supplied. The prior optimization queue remains open. F4 numerical acceptance, F7 full ROI ledger, F8 entry-skill routing and sparse donor proof remain outside this partial pass-scope. Finiteness and benchmark/job exit propagation can proceed independently of a user decision on numerical bounds.

All new probes were in memory; no production files or evidence records were changed. No cluster job, performance sweep, GPU parity or full-model run was launched. Only this report is added.

<<<REVIEW-VERDICT
{"status":"FAIL","findings":["R4-F5: valid gap IDs accept empty, nonexistent or unrelated destinations; full self-test still passes","R4-F3: duplicate timing headers and malformed sample retries bypass fail-closed ingestion"],"surface":true,"note":"Round-3 exact counterexamples now fail as intended; published-report regeneration and identity separation are closed. Destination binding and parser schema integrity remain incomplete. Acceptance policy, ROI ledger, skill routing and donor proof remain explicitly open."}
REVIEW-VERDICT>>>