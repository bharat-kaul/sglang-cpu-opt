# DSv4 Flash Roofline Gate: Fifth Review

Date: 2026-10-08.

Reviewed revision: `6af22cb5aad6fec8b6b94da313588971064aac97`.

Reviews the [fourth author response](dsv4_flash_roofline_gate_response4.md) against the [fourth review](dsv4_flash_roofline_gate_review4.md). An existing uncommitted playbook edit was left untouched and is outside this review.

Publication scope: that playbook work was committed during review as `6d4aefcd2326bec23180151eb95542f1384daf59`. The intervening commit changes only playbook-generation documentation; audited roofline code and records remain unchanged.

## Verdict

**The accepted numerical accounting remains closed within its declared partial scope. Publication is still blocked by H1: unsupported correctness certification. Implementation review remains deferred.**

The latest fix genuinely reads result records, derives the selected pass's revision, marks pending revisions unresolved, withholds absolute latencies, labels speedups unverified, and fails the main self-test when observation records cannot be opened. Those parts of H1 are resolved.

However, both canonical reports now print **CERTIFIED correctness** for every loaded kept pass, including unresolved revisions and evidence with an explicit shape-provenance warning. The response and code justify this by calling correctness context-independent. A recorded numerical comparison is not context-independent: it applies to the tested implementation, inputs, shapes, dtypes, layout, reference, and tolerance. Loading a summary does not extend that evidence to another contract or certify a current implementation.

This is the same publication concern from the previous review, not a new numerical roofline requirement. No kernel tests or fresh hardware runs are required to remove the unsupported certification label.

## Verified Fixes

- All **28** current self-tests pass on the real records.
- The main report and canonical join regenerate exactly from their generators.
- The incorrect top-k `06faec0` association is gone. Evidence now reports `UNRESOLVED(pending)` from the kept pass.
- Speedup summaries are explicitly author-reported/unverified; unsupported absolute latencies remain withheld at every displayed M.
- The previous counterexample that makes all `op_passes` opens fail now causes the main self-test to fail. Record access is real, not inferred from a filename suffix.
- Previously accepted shapes, counts, byte accounting, precision ceilings, and tracker checks remain passing. Explicit model exclusions are retained; this is not a complete whole-model cost or runtime-fit certificate.

## H1: Remaining Certification Gap

### High: A Kept-Pass String Is Promoted to a Certificate Without Scope Validation

[load_record](../dsv4_roofline_p2.py#L151) selects the last pass with `kept=True` and copies its `correctness` text. [observation_evidence](../dsv4_roofline_p2.py#L168) places that text in `certified_correctness` regardless of its unresolved revision or the record's shape provenance. The [join renderer](../dsv4_roofline_vs_measured.py#L103) similarly places the copied string under CERTIFIED correctness.

Concrete evidence from existing records:

| Published row | Evidence available | Unsupported promotion |
|---|---|---|
| Top-k | Kept pass has revision `pending`; summary says set-match 1.0 | Rendered as certified correctness with no identified tested revision |
| MHC Sinkhorn/combine | Kept-pass cosine summaries and revision `pending` | Same unsupported certification |
| Sparse attention | Record explicitly warns that its microbench used H=8, D=128, Dv=128 rather than H=64, D=512; kept revision is `pending` | Cosine summary appears as certified next to the join's H=64, K=512, D=512 target |

The [sparse-attention record](../results/op_passes/sparse_attend.json#L7) also contains a later narrative about corrected dimensions. That narrative does not provide a revision- and shape-matched structured correctness result for the selected kept pass. This review does not conclude that the implementation fails at either shape; it concludes that the displayed certification is not established by the supplied evidence.

Two in-memory probes further show that the evidence function does not validate a correctness result:

| Injected kept-pass record | Returned `certified_correctness` |
|---|---|
| `{"kept": true, "commit": "pending"}` | `unrecorded` |
| Same record with `"correctness": "FAIL: numerical mismatch"` | `FAIL: numerical mismatch` |

Neither is rejected or downgraded. These are synthetic validator probes, not claims that an actual kernel failed a numerical test. They demonstrate that the CERTIFIED heading is generated independently of a verified pass status and its scope.

Even an authentic cosine result near 1.0 only supports the recorded comparison. It does not establish all shapes, exact selection agreement, current-revision behavior, or end-to-end correctness. A mathematical proof covering a family could establish broader correctness, but no such proof is supplied or checked by this reader.

## Minimal Closure

The shortest sufficient correction is to use the withholding option already offered in the fourth review:

1. In both canonical reports, label copied correctness summaries **historical, author-reported, UNVERIFIED for the current target**. Remove CERTIFIED and the claim that correctness is context-independent.
2. Preserve unresolved revisions and relevant shape warnings, or explicitly state that the summaries' scope has not been validated. Do not turn a missing result or a reported failure into a passing claim.
3. Update the self-test to assert that unresolved, shape-mismatched, and missing-correctness records cannot produce a certification label. Keep the newly working missing-record failure checks.
4. Regenerate the artifacts. Keep the accepted ideal target numbers unchanged.

That reporting change is sufficient for H1 closure without proving the implementations correct in this phase. Alternatively, a real certification path must validate an identified implementation revision, matching test coordinates/reference, a defined pass criterion, and an actual result. It cannot be implemented by checking whether a revision looks hexadecimal or whether a summary string is present.

## Validation and Scope

Executed the generator's 28 checks, compared regenerated artifacts exactly, exercised real record-open failures, inspected derived evidence for existing records, and ran the two in-memory correctness-status probes above. No files were changed by those probes.

No new numerical defect was found in the previously accepted accounting checks. The analysis remains deliberately partial where exclusions are declared. No optimized implementation was inspected or executed, and no parity tests, model loads, performance measurements, or cluster jobs were launched.

Only this review document is added. Production code, the author response, earlier reports, and the user's playbook work are preserved.

**Next gate:** remove or substantiate the unsupported correctness certification, then recheck publication. Implementation correctness remains a separate, later review; this roofline report must not certify it in advance.