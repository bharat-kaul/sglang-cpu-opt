---
name: adversarial-self-audit
description: "Use BEFORE submitting ANY gate/validator/record/review-response for review (both legs). Encodes the pre-submit discipline that stops the multi-round FAIL spiral: fix the vulnerability CLASS not the shown instance, validate by FALSIFICATION not confirmation, apply provenance to your OWN claims, and test adversarial cases through the FULL gate. Run it as a hard pre-submit self-audit so the reviewer finds nothing you could have found yourself."
---

# Adversarial Self-Audit (pre-submit gate for any reviewed artifact)

Every gate, validator, parser, reconciler, correctness/acceptance record, and
review-response you produce is adversarially reviewed. The failure mode this skill
prevents: **you patch the exact counterexample the reviewer gave, claim closure,
and the reviewer finds the NEXT instance of the SAME class** — turning a 1-2 round
job into 5+. Internalize the adversary BEFORE submitting; do not outsource
skepticism to the reviewer.

> **Anti-pattern that actually happened (DSv4 kernel-impl gate, 5+ FAIL rounds).**
> Parser: positional column → duplicate-overwrite → duplicate header → `+1`/`1.0`
> coordinate — FOUR rounds, ONE defect ("the validator trusts its input's
> structure"), patched one spelling at a time. Reconcile: names → coverage →
> completeness → destination-binding — ONE defect ("the spec isn't bound
> field-by-field to an independent source"). Plus claims written from intent, not
> fact: "all 6 kernels guarded" (one wasn't), "validated named columns" (still
> positional), a cited "kernel.py indexer primitive" that **does not exist**.

## The five rules (apply every one, every submit)

1. **FIX THE CLASS, NOT THE INSTANCE.** When a counterexample appears, name the
   vulnerability *category* and close the whole category in ONE move — a
   fail-closed **whitelist grammar** (accept the exact declared schema, reject
   everything else) or **exact field-by-field binding to an independent spec** —
   then enumerate every sibling instance and test them. Never ship a fix that only
   makes the single shown input pass.

2. **VALIDATE BY FALSIFICATION, NOT CONFIRMATION.** A test that encodes "the last
   counterexample now fails" is confirmation of a known case and shares your blind
   spots — **green ≠ correct**. Before submitting, RED-TEAM your own artifact: list
   the attack class and try to BREAK it with inputs you were NOT given (sibling
   spellings; empty / duplicate / wrong-but-plausible; boundary, sentinel −inf,
   zero, near-zero, mixed-case, whitespace). Add a **discriminating regression
   control**: confirm the test detects the OLD behavior, not just agreement with the
   new code.

3. **TEST THROUGH THE FULL GATE, not an isolated helper.** "All self-tests pass"
   is false comfort if a mutated *document/log* still passes the real entry path.
   Serve adversarial cases end-to-end (the real `main()` / full self-test / real
   input file) and assert the real side effects (zero output writes on reject,
   nonzero job/child→wrapper exit, record unchanged).

4. **PROVENANCE APPLIES TO YOUR OWN CLAIMS.** Before writing any
   "closed / all / validated / per the primitive / N kernels" claim, OPEN the file,
   grep the symbol, run the check, and confirm the claim is literally true NOW.
   Cite every external reference from the actual source inventory (the published
   module's real symbol list), never from plausibility or symmetry with a sibling.
   A claim you did not just verify is a finding waiting to happen.

5. **PRE-APPLY THE REVIEWER'S DIRECTION.** Reviews converge predictably:
   instance→class, label→consequence, nonempty→complete, present→correct-destination,
   intent→source, helper→full-gate. Walk that whole staircase yourself in the
   self-audit instead of being walked down it one step per round.

## Pre-submit checklist (run before handing any artifact to review)

- [ ] For each change: the **invariant** it must guarantee is written, and enforced
      by a whitelist / exact-binding — not a patch for the shown input.
- [ ] Sibling-instance list enumerated and tested (unseen spellings/mutations).
- [ ] Adversarial cases run **through the real entry path**, side effects asserted.
- [ ] A regression control proves the test catches the OLD behavior.
- [ ] Every scope/citation/"closed" claim re-verified against the artifact's
      actual state and the real external source.
- [ ] Negative result is **fail-closed** (reject/raise/nonzero-exit), never a
      silent skip or a label flip.

## Scope

Framework- and HW-agnostic; the method is the skill. Applies to both legs and to
every reviewed deliverable — correctness/acceptance gates (`accuracy-oracle`,
`coverage-gate`, `kernel-feasibility-gate`, `enablement-certificate`), provenance/
reconciliation records, ingestion/parsers, and the review-response itself. Pair
with `high-information-runs` (design the probe) and the external-reference
provenance rule (cite an authority a reviewer can open).
