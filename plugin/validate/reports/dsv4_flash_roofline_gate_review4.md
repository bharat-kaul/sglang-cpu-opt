# DSv4 Flash Roofline Gate: Fourth Review

Date: 2026-10-08.

Reviewed revision: `33eb7c14908dc1fbf9891847335d062f73e91d9e`.

Publication scope: the branch advanced to `e9115a18175547fa5ef0383400860fc73a898ec5` during review. That commit changes only playbook-generation documentation; the audited roofline code and observation records are unchanged. The new documentation was not separately audited here.

Reviews the [third author response](dsv4_flash_roofline_gate_response3.md) against the [third review](dsv4_flash_roofline_gate_review3.md).

## Verdict

**Not approved for publication: one observation-provenance blocker remains. The prior numerical G1 and tracker G3 blockers are resolved. Implementation review remains deferred.**

The previously requested numerical corrections tested in this review pass. The reports now withhold unsourced absolute latencies, select the correct FP32 resource for sparse attention, use the stated FP32 indexer public boundary, and reject the earlier invalid tracker inputs. Both canonical reports reproduce exactly.

The remaining issue is narrower: the new AUDITABLE OBSERVATIONS section calls historical summaries certified correctness/speedup evidence, but contains a demonstrably wrong top-k revision and several unresolved revisions. The generators do not load or validate those records. Withholding absolute latency was appropriate; it does not validate the replacement relative-performance claims.

No new accounting defect was found in the revised formulas exercised here. This is not certification of a complete whole-model cost inventory: previously explicit exclusions remain. Keep the nominal-peak methodology and the already accepted corrections.

## Verified Closure

| Previous issue | Executed result | Disposition |
|---|---|---|
| G1: Sparse-attention compute resource | Explicit FP32; 2,147,483,648 FLOPs; nominal target **275.941053 us**, compute branch | Resolved |
| G2: Indexer source/boundary | Correct benchmark identified; FP32 per-request public boundary **17,965,056 bytes**, nominal FP32 target **69.793684 us** | Contract correction verified |
| G2: Unsupported absolute latencies | All observed rows return no numeric absolute latency at every displayed M; join removes unsourced absolute values and off-roof ratios | Resolved |
| G2: Main report hides observation notes | AUDITABLE OBSERVATIONS block now prints record, revision, and claim | Rendering resolved; evidence accuracy remains open below |
| G3: Prefix-invalid disposition | Actual loader rejects `MODELED_BOGUS` | Resolved |
| G3: Missing item identity/source | Actual loader rejects the previous incomplete input | Resolved |
| G3: Removed exclusion coverage | Actual loader rejects removal of all EXPLICITLY-UNMODELED categories | Resolved |
| G3: Duplicate identity | Actual loader rejects duplicate items | Resolved |
| G3: Missing/malformed tracker | Self-test fails, preserving fail-closed normal publication | Preserved |

The existing reference checks also preserve the main-layer counts, attention positions, shared-expert FP8 storage, compressor projection capacities, MHC dimensions, pooling contract, and distinct compute ceilings.

## Remaining Finding

### H1. High: Published Certification Claims Are Not Validated Against Their Records

The [main top-k observation](../dsv4_roofline_p2.py#L251) and [join row](../dsv4_roofline_vs_measured.py#L45) publish:

> indexer_topk.json @ 06faec0: set-match 1.0; 7.62x vs torch.topk at M=32, context=1024.

The linked [top-k record](../results/op_passes/indexer_topk.json) does not associate that result with that revision:

| Record entry | Kernel revision | M=32 ratio | Status |
|---|---|---:|---|
| Pass 1, row-parallel nth_element | `06faec0` | 1.63x | Discarded |
| Pass 2, chunked implementation | `pending` | 9.55x | Kept, earlier context |
| Best verdict, corrected context=1024 | `pending` | 7.62x | Re-measurement summary |

Thus the report combines the corrected-context result with an earlier discarded implementation's revision. This is not merely missing absolute timing. It is a false result-to-revision association within the supplied evidence.

Also verified: `git show 06faec0:plugin/validate/results/op_passes/indexer_topk.json` fails because the record did not exist at that commit. A kernel revision can legitimately predate its result record, but then **kernel revision and result-record revision must be separate fields**, and both must identify the evidence actually being cited. Here the record's own kernel association already contradicts the report.

Four other joined records have the literal revision `pending`: compressor, sparse attention, MHC Sinkhorn, and MHC combine. They may be retained as historical reported results, but are not revision-pinned certifications. The sparse record also retains an explicit unverified-shape warning alongside a later corrected-shape narrative; selecting a narrative does not independently validate the run. These observations do not establish that the underlying implementations are wrong; implementation validation has not been performed.

The [observed helper](../dsv4_roofline_p2.py#L137) accepts `record`, `commit`, and `certifies` as strings. Its [self-test](../dsv4_roofline_p2.py#L380) checks only a `.json` suffix and nonempty fields for one observation. The [renderer](../dsv4_roofline_p2.py#L468) prints the supplied claim. Neither it nor the join reads the result records to establish that the claim is supported.

An in-memory test made every attempted `op_passes` file open fail. The self-test still passed and the main report still emitted the 7.62x certification; there were **zero attempts to open a result record**. The records currently exist, but their existence and contents have no effect on these publication claims.

**Required closure: choose one of the following without inventing evidence.**

1. **Publish verified associations:** load structured records; identify the particular result entry, benchmark coordinate, kernel revision, and record revision; compare the published quantity against that entry; validate the association before calling it certified. Add a test that rejects the `06faec0`/7.62x mismatch and prevents missing/unresolved records from being presented as certified.
2. **Withhold certification:** keep the corrected analytical targets, but remove correctness/speedup certification claims from this report, or explicitly label the summaries **historical, author-reported, unverified**, with unresolved revisions visible. Do not use them to claim implementation readiness or validated speedup. A correct record link can be retained as background without promoting it to a certificate.

Option 2 is sufficient to remove this publication blocker; no fresh hardware run or implementation audit is required just to report the current evidence honestly. It follows the same acceptable withholding policy already used for absolute latency.

## Scope and Nonblocking Notes

- The response says 26 self-tests; execution produces **25 passing checks**. Correct the count when updating the response, but this is not a separate blocker.
- Previous explicit omissions, including state/postprocessing, vector work, runtime capacity, and fallback execution costs, remain limitations of a partial model. They are not silently promoted to a complete total or runtime-fit proof by this review.
- Deferred reusable-skill occupancy alignment remains a playbook follow-up, not a new roofline blocker.
- This review does not independently certify the correctness assertions or speedup ratios in any op-pass summary. A cosine summary alone is not an end-to-end correctness certificate.

## Validation Performed

1. Executed the current conformance/contract self-test: **25 PASS**.
2. Regenerated [the main report](dsv4_roofline_emr.txt) and [canonical join](dsv4_roofline_vs_measured_emr.txt) in memory: exact matches.
3. Retested G3 counterexamples through the actual loader with in-memory file substitution: invalid enum, missing fields, dropped coverage, duplicate identity, missing file, and malformed JSON are rejected.
4. Checked that all observed latency cells are withheld and that the revised sparse/indexer target numbers match their declared contracts.
5. Read the six linked structured result summaries, compared top-k result/revision fields, and checked the advertised historical path with Git.
6. Tested record unavailability without changing files; the generator still passed and emitted its hard-coded observation claims.

No optimized implementation was reviewed or executed. No parity tests, new performance measurements, model loads, or cluster jobs were run. Production code, response documents, and previous reports are unchanged.

## Next Gate

Resolve H1 by validating the result associations or withholding the certification claims, regenerate the canonical reports, and recheck publication. The numerical and tracker fixes accepted above should remain closed unless subsequent changes affect them. Implementation review can proceed only after that roofline/reporting gate is accepted.