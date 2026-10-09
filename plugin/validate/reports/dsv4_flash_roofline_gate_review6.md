# DSv4 Flash Roofline Gate: Sixth Review

Date: 2026-10-08.

Reviewed revision: `b970518231e76dc4e6dcc3dd28ca4e50218ed52a`.

Reviews [response5](dsv4_flash_roofline_gate_response5.md) against the [fifth review](dsv4_flash_roofline_gate_review5.md).

## Finding: H1 Remains Open

**High: the generator still certifies a match without checking that a match was recorded or passed.**

[observation_evidence](../dsv4_roofline_p2.py#L168) now separates microbench results from current-target verification, which is a meaningful correction. However, it unconditionally constructs `<metric> certified @ microbench shape; E2E verification PENDING (current target)`. The metric is chosen by substring search in free text; neither a passing outcome nor the claimed test scope is validated.

The narrower scope and lowercase spelling do not fix this. Certification of a microbench match still needs evidence of that match. The following in-memory probes exercise the real reader and evidence function, using valid JSON with a kept pass and revision `pending`:

| Correctness field in the injected pass | Returned correctness | Returned status prefix |
|---|---|---|
| Absent | `unrecorded` | `match certified @ microbench shape` |
| `FAIL: numerical mismatch` | `FAIL: numerical mismatch` | `match certified @ microbench shape` |
| `cos 0.0; FAIL` | `cos 0.0; FAIL` | `cosine certified @ microbench shape` |

All three statuses also include the E2E-pending disclaimer. These are synthetic evidence-validation probes, not findings that an actual kernel failed. They demonstrate that the generated positive match claim is independent of the result, not merely missing an end-to-end qualification.

On the real top-k record, the generator likewise prints `set-match certified @ microbench shape` alongside `UNRESOLVED(pending)`. Reading the kept-pass summary verifies what the file reports; it does not independently establish the numerical experiment or identify its tested implementation. Historical summaries remain useful evidence when attributed as such.

The [new self-test](../dsv4_roofline_p2.py#L435) checks for `E2E verification PENDING`, but does not reject certification on missing or failing correctness. Its description says "never CERTIFIED", while the required status itself still asserts certification. Thus all 29 tests can pass with the counterexamples above.

The [join report](../dsv4_roofline_vs_measured.py#L103) no longer prints a certification heading and explicitly limits the claim to microbench scope. That improvement is accepted. It does not repair the main generator's unconditional certification.

## Accepted Changes

- The context-independent correctness claim is removed from the evidence function.
- Current-target/E2E verification is explicitly pending in both canonical reports.
- The misleading `certified_correctness` key is removed.
- Record-derived revisions, unresolved labels, withheld absolute latency, and unverified speedup handling remain intact.
- Missing observation records still cause the main self-test to fail.
- All **29** existing self-tests pass; both canonical reports regenerate exactly.
- No analytical formulas changed in this response. Previously accepted numerical accounting and declared exclusions remain accepted within their partial scope.

## Minimal Closure

Do not infer a positive result or certificate from free text. Report the field literally under an attribution such as:

`Historical author-reported microbench result; test scope UNVERIFIED; current-target/E2E verification PENDING`.

This permits retaining a recorded cosine or set-match without claiming independent verification. For an absent result, say `unrecorded`; for a reported failure, retain the failure without adding a positive match status. An unresolved revision must remain unresolved. Apply consistent attribution to both reports.

Add regression checks for missing correctness, explicit failure, and failed cosine, as well as unresolved and shape-mismatched evidence. None may generate a certification claim. Preserve the working missing-record check and accepted ideal numbers, then regenerate the artifacts.

No new hardware runs, full-model tests, or implementation proof are required for this reporting-only closure. Alternatively, retaining a genuine certification path requires validated experiment evidence, not a keyword-derived label.

## Verdict and Scope

**Numerical accounting remains accepted within its declared partial scope. H1 still blocks publication of the roofline package as gate-passed; implementation review remains deferred.**

Validation: ran the 29 checks, confirmed fail-closed behavior when record opens fail, compared both regenerated reports exactly, inspected real top-k evidence, and reproduced all three counterexamples above without changing source or records.

No kernels were inspected or executed. No parity runs, model loads, benchmarks, or cluster jobs were launched. Only this review document is added; production code, author responses, artifacts, and playbook work are unchanged.