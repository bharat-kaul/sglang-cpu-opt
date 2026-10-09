# DSv4 Flash Roofline Gate: Seventh Review

Date: 2026-10-08.

Reviewed revision: `636469ef3b04f2e5e04f1061b88718a812dc6af9`.

Reviews [response6](dsv4_flash_roofline_gate_response6.md) against the [sixth review](dsv4_flash_roofline_gate_review6.md).

## Verdict

**PASS within the declared partial analytical scope. H1 is closed. No remaining blocking finding was identified in this re-review.**

The response implements the reporting-only closure requested in the sixth review. Historical correctness summaries are no longer promoted to numerical-match certificates. The previously accepted numerical accounting is unchanged, and the canonical reports reproduce exactly.

The roofline gate no longer prevents proceeding to implementation review. This report does not itself perform or pass that separate review.

## H1 Closure Verified

[The evidence builder](../dsv4_roofline_p2.py#L174) copies the recorded correctness field under the fixed attribution:

`Historical author-reported microbench result; test scope UNVERIFIED; current-target/E2E verification PENDING`.

It no longer derives a metric or passing outcome from keywords. [The observation path](../dsv4_roofline_p2.py#L184) still reads the actual record; an absent correctness field becomes `unrecorded`, and pending revisions remain `UNRESOLVED(pending)`. Both the main report and [join renderer](../dsv4_roofline_vs_measured.py#L103) use the unverified attribution.

The prior counterexamples now behave correctly through the real reader, evidence builder, and both renderers:

| Injected correctness | Reported result | Added positive certification |
|---|---|---|
| Absent | `unrecorded` | None |
| `FAIL: numerical mismatch` | `FAIL: numerical mismatch` | None |
| `cos 0.0; FAIL` | `cos 0.0; FAIL` | None |

All three retain unresolved revision status and the unverified attribution. These are synthetic reporting tests, not numerical failures of actual kernels.

The real sparse-attention record's shape-warning-bearing evidence also remains explicitly test-scope-unverified. The reader does not reproduce its full warning, but the global scope disclaimer satisfies the previously offered reporting-only closure: the historical result is not asserted to validate the current target.

## Validation

- All **33** existing self-tests pass, including the three new missing/failing-correctness regressions through the record reader and evidence builder.
- Independent probes exercise the same counterexamples through `observation_evidence()`, the main renderer, and the join renderer. All preserve the result without generating a certificate.
- Blocking observation-record opens makes the main self-test return `False`; fail-closed behavior is retained.
- Both canonical artifacts regenerate exactly: [main report](dsv4_roofline_emr.txt) and [join report](dsv4_roofline_vs_measured_emr.txt).
- The remaining word `certified` in the main artifact is in the self-test description `never certified`, not a positive claim.
- The response changes evidence reporting, tests, and documentation, not analytical formulas. Previously accepted accounting checks and explicit exclusions remain intact.

The initial broad missing-file probe also blocked temporary test fixtures; it was narrowed to observation records. An initial whole-output keyword assertion also matched the negative phrase `never certified`; inspecting its context resolved that assertion. Neither is an unresolved product defect or an artifact mismatch.

## Limits and Next Gate

Acceptance is for the declared partial ideal roofline and its evidence presentation. It is not a complete whole-model runtime or memory-fit certificate, a sustainable-performance measurement, or proof that nominal headroom is recoverable. Declared unmodeled costs remain exclusions.

Historical microbench correctness, current-target validity, implementation revisions, absolute latency, and speedups have not acquired stronger validation through this reporting fix. Correctness summaries remain author-reported; current-target/E2E verification remains pending; unsupported latency and speedup claims remain withheld or unverified.

Implementation review may now begin as a separate phase, using identified revisions and shape-grounded evidence. No kernels were inspected or executed during this re-review, and no parity runs, model loads, performance measurements, or cluster jobs were launched.

Only this review document is added. Source, author responses, generated artifacts, and playbook work are unchanged.