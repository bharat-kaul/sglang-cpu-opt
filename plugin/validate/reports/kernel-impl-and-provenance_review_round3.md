# Kernel Implementation and Provenance: Round 3

2026-10-09. Reviewer: GitHub Copilot; no separate Astra execution is claimed. Reviewed commit: `ab7b6185deb611c46714b452131f7e78d1d5fcc6`.

**FAIL: three claimed closures remain incomplete.** Reviewed the [round-2 response](kernel-impl-and-provenance_response_round2.md) against the [previous findings](kernel-impl-and-provenance_review_round2.md). Acceptance policy, full ROI ledger, entry-skill routing and sparse donor proof remain explicitly outside the partial pass-scope. Their continued absence is not the reason for this verdict.

## Findings

### R3-F3 High: Duplicate Samples Overwrite Evidence Instead of Failing

[parse_perf_sweep.py](../parse_perf_sweep.py#L69) assigns `data[bench][M][rep] = value` before checking replica IDs. A dictionary guarantees unique stored keys, not unique input samples. Executed independent synthetic sweeps with all six benchmarks and all five M values:

- Replica values 1, 2, 3 produce median 2. Appending another sparse REP=2 block with value 99 is accepted and changes every sparse median to 3.
- A sparse REP=1 block containing NaN followed by a complete valid sweep is accepted: the invalid evidence is overwritten before finiteness validation.
- Extra M=128 rows are silently ignored, contradicting the response's exact-M-coverage claim. Missing expected coordinates do fail.

The old single-replica, infinity, NaN-without-overwrite and missing-M cases now fail correctly; reordered headers select the correct column. All current real medians remain correct. This finding concerns ingestion integrity, not a new numerical tolerance requirement.

**Required:** reject repeated real samples before assignment and validate each encountered value immediately. Anchor real benchmark markers at line start so shell `set -x` echo traces are not mistaken for duplicate runs. Enforce exact coordinates, or explicitly document deliberate extra-coordinate filtering. Add duplicate and overwritten-invalid tests; the existing five self-tests all pass without detecting these cases.

### R3-F5 High: Nonempty Gap Lists Are Still Not Complete Gap Accounting

[reconcile_kernels](../dsv4_roofline_p2.py#L474) checks whether a gap list is nonempty, not whether every declared gap is covered. Each of these mutations still returns `(True, [])`:

- Keep only the compressor's first gap disposition, dropping the r128/indexer projection dispositions and the combined postprocessing/state disposition.
- Replace all compressor dispositions with `[{"gap":"unrelated","disposition":"caller:"}]`.
- Delete both compressor's declared gaps and its dispositions.
- Duplicate the compressor declaration in `kernels[]`; the set construction at [line 452](../dsv4_roofline_p2.py#L452) loses duplicate identity, and the subsequent dictionary can overwrite declarations.

The independent on-disk kernel inventory and exact cost-row-name sets are useful improvements. All four previous mutations now fail; so do row supersets/subsets, an unknown seventh entry and a simulated on-disk rename. But the record's claim that every fragment gap is dispositioned still does not follow from these checks.

**Required:** stable gap IDs with complete, unique coverage, explicit grouped coverage where needed, and nonempty resolvable destinations. Protect the required gap inventory from being deleted alongside its evidence; reject duplicate declarations before set/dictionary conversion. No need for a general orchestration framework.

**Shape-check boundary:** `_KERNEL_COST_CONTRACT` binds names, not typed operands. An isolated precision mutation passes reconciliation, but the surrounding full self-test correctly rejects both an FP32-to-BF16 pool mutation and zero pool bytes. Those are **not** full-gate bypasses. Preserve those independent checks; do not describe filename-to-row matching alone as proof of benchmark/captured-input alignment.

### R3-P1 Medium: Published Output Still Contains the Old Identity Conflation

The generator now prints separate CURRENT timing and HISTORICAL correctness sections. However, the tracked [published report](dsv4_roofline_vs_measured_emr.txt#L5), still referenced by `impl_review.json`, was not regenerated. Its row header still attaches `indexer_logits.json@e3fdcb9` to the timing table. It also retains the old M=1 regression and inverted compute-floor wording at [line 13](dsv4_roofline_vs_measured_emr.txt#L13).

An in-memory generation differs from the saved file: fresh output has six historical-correctness labels and six current-timing labels; the saved output has zero of either. **Required:** regenerate and commit the referenced artifact, then enforce a regeneration comparison. The source-level labeling fix is accepted; the published-artifact closure is not.

The current structured timing record correctly names job 384414/pcl-sprh09, and historical `op_passes` files are unchanged. The sparse donor is now consistently a candidate in the updated provenance record. Exact source/build identity of job 384414 remains unresolved: the raw run header and launcher record node/time/configuration, not a kernel/benchmark source digest. Separating old correctness revisions avoids false binding but does not create a missing timing-build identity. Mark this limitation explicitly; do not retroactively assign current HEAD to that run.

## Verified Closures

| Item | Independent result | Disposition |
|---|---|---|
| R2-F3 named columns and finite stored samples | Rejects single replica, infinity, NaN and missing M; accepts reordered header with the correct scalar column | Fixed subcases; duplicate handling remains open |
| R2-F5 kernel and row inventory | Rejects all four prior mutations, row supersets/subsets, unknown kernel and simulated inventory rename | Fixed subcases; gap completeness remains open |
| R2-F9 sparse guards | Both entries reject excess/short KV batches, wrong sink length and meta-device tensors | Closed for the claimed guards |
| R2-F9b zero selection | `[3,16], k=0` and `[2,0], k=0` match Torch shape and int64 dtype; positive k=1/16/512 set checks pass | Closed |
| R2-P1 source labels and records | Six distinct historical/current sections; current structured job identity corrected; historical records preserved | Source fix accepted; saved output stale |
| R2-P2 precision comments | Exported fused-vs-tiled difference reproduced; tiled integration identical at M=1/7/8/16 | Comment fix accepted; numerical policy remains open |

At H=64/D=512/K=512, both sparse variants return finite outputs on the tested random input. Maximum error against the FP32 stage oracle is `4.32134e-7` for scalar and `0.00246298` for AMX. These observations are not acceptance certificates. On the prior seed-1 FP4-grid input, indexer fused-vs-tiled difference remains `0.044673919677734375`; replaying the published BF16 stage expression gives `0.09235763549804688` maximum difference and 511/512 selected-index intersection. These synthetic CPU stage checks do not establish GPU or model parity, and do not newly block the declared partial scope.

## Provenance, Shapes and Optimization

Independent header-based extraction of job 384414 again matches every stored sweep: indexer `[0.100,0.279,0.366,0.540,0.908]`, top-k `[0.017,0.028,0.032,0.039,0.014]`, pool `[0.374,0.490,0.489,0.499,0.571]`, Sinkhorn `[0.002,0.014,0.014,0.014,0.015]`, combine `[0.002,0.012,0.012,0.012,0.014]`, sparse scalar `[0.068,0.460,0.894,1.793,3.640]` ms at M=1/8/16/32/64. The corrected indexer BF16 compute resource/FP32 public bytes and top-k int64 output accounting remain intact.

No new captured-shape manifest or measurement coverage is supplied. Pool performance still covers R=128/D=512, not the two R=8 contracts. FP32 benchmark boundaries must not be treated as the main model's BF16 traffic. Sparse K=512 pre-gathered timing does not validate the model's complete window-plus-compressed 128/640/160 path or the assumed donor implementation. These are acknowledged scope differences, not interchangeable coordinates.

**Optimization exhaustion is still not established.** This commit supplies guard/validator/reporting changes, not new performance experiments or the full call-count ledger. The round-2 priorities remain: indexer conversion/packing/reduction attribution, both R=8 pool shapes, small-S top-k simplification, small-op launch grain/fusion, and a capability-proven sparse path. The conditional call-count sums in the previous report remain provisional, not model latency.

The indexer header now retracts exhaustion and correctly distinguishes BF16-rounded fused scores, but nearby header/export text still describes the obsolete best-of dispatcher. The current join also still says combine's [plateau is minimal traffic achieved](../dsv4_roofline_vs_measured.py#L68). Minimal useful-byte accounting does not prove minimum executed traffic or exhausted ROI. Keep that as a hypothesis until discriminating measurements support it. These are residual reporting caveats; the FAIL above does not depend on finishing F7.

F4's reference, precision and tie/selection tolerances still need a declared policy. Finiteness checks and benchmark/job exit propagation can be implemented independently of those numerical decisions; the existing sweep launcher still prints exits and continues. F8 entry-skill routing and donor-dispatch proof remain explicitly pending. None is silently promoted to a new partial-scope blocker here.

## Review Scope

Executed the five parser self-tests, all 39 generator self-tests, independent in-memory mutations and raw-log reconstruction, current C++ extensions at one thread, and artifact regeneration comparison. Inventory rename was simulated without changing files. Used the pinned published model contract from the previous round at revision `60d8d70770c6776ff598c94bb586a859a38244f1`; no upstream revision, hardware ceiling or production integration change is inferred. No performance sweep, cluster job, GPU parity or full-model run was launched. Only this review report is added; executor code, evidence records and loop state are untouched.

<<<REVIEW-VERDICT
{"status":"FAIL","findings":["R3-F3: duplicate samples overwrite evidence and can hide nonfinite values","R3-F5: partial or unrelated gap dispositions and duplicate declarations pass reconciliation","R3-P1: published report retains old identity labels despite corrected generator"],"surface":true,"note":"Sparse guards, zero selection, named-column parsing and exact row inventory improve and their scoped repros pass. Three claimed closures remain incomplete. Numerical acceptance, full ROI ledger, skill routing and donor proof remain explicitly open; optimization exhaustion is not established."}
REVIEW-VERDICT>>>