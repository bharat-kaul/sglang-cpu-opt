---
name: adversarial-self-audit
description: "Use BEFORE submitting ANY gate/validator/record/review-response/perf-or-correctness claim for review (both legs). Encodes the pre-submit discipline that stops the multi-round FAIL spiral — TWO clusters: (A) structural (fix the CLASS not the instance; falsify not confirm; provenance of your OWN claims; test through the full gate) and (B) measurement/conformance (conform to the AUTHORITATIVE source boundary not your reconstruction; test the ADVERSARIAL domain not the convenient one; a Claim-Evidence Ledger with banned inflated words; replicated-median + run-identity provenance; measurement-physics sanity). Run it as a hard pre-submit self-audit so the reviewer finds nothing you could have found yourself."
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

## Second cluster — MEASUREMENT, CONFORMANCE & CLAIM-STRENGTH (what the next review found)

> **Anti-pattern that actually happened (DSv4 Phase-1 kernel review, 8 findings).** We validated the
> indexer against our OWN oracle that reduced in **fp32**; the PUBLISHED op reduces in **bf16** with
> **signed** weights — our oracle shared our blind spot and we only tested nonnegative weights, so real
> selection differences were invisible until the reviewer replayed the published stage. We named a
> screening number a **"noise floor"** (it was observed error + margin vs a **global** replica while the
> real kernel is **64-block**, and we compared **mismatched** operands). We claimed a **"≥1.0× floor at
> every M"** while a datum sat at **0.96×**, called a single sequential mean a **"median"**, and said
> **"bit-identical"** from a **cosine**. A GPU wrapper piped through `tail` and printed **exit 0 on a
> crash**. None of these were parser bugs — they were claims stronger than the evidence, validated against
> our own artifacts in the domain we chose.

**Why review keeps finding things (the root).** The reviewer makes three moves we skip. (1) They go to the
**independent authoritative source** — the published op, the real kernel, the raw log — while we check our
own reconstruction (oracle/replica/benchmark), which shares our blind spots. (2) They probe the
**adversarial domain** — signed not nonneg, every shape/branch, the gate itself — while we test the
convenient one. (3) They reconcile **claim-vs-evidence word by word** while we write claims from intent. A
clean review = make all three moves YOURSELF first. The audit question is not "does it pass my check"
(confirmation) but "what would the reviewer's source-check, domain-check, and word-check break"
(falsification).

6. **CONFORM TO THE AUTHORITATIVE SOURCE BOUNDARY, NOT YOUR RECONSTRUCTION.** An oracle/replica you built
   shares your blind spots. Before trusting it, OPEN the published op (or run the real kernel) and match
   EACH stage's **dtype / accumulation / normalization-block structure / output precision** — not a
   plausible paraphrase. Where a replica stands in, LABEL it a replica and record exactly how it DIFFERS
   (global vs blockwise, operand rounding); keep the gate **PROPOSED** until a source-faithful comparison
   exists. Always compare **matched operands and the same output boundary** (a bf16-vs-fp32 operand
   mismatch makes the diff an artifact).

7. **TEST THE ADVERSARIAL DOMAIN, NOT THE CONVENIENT ONE.** The test domain must COVER the claim's domain.
   Enumerate the REAL caller domain and hit the hard corner: **signed** (not nonneg), the real
   shapes/K-values, non-contiguous, masked/sentinel, empty/degenerate (N=0, k=0), and **every dispatch
   branch** (each M regime; each fast-path specialization must re-enter the same validation). A benchmark
   at M is NOT an acceptance gate at M — coverage must equal the claim.

8. **RECONCILE EVERY CLAIM TO ITS EVIDENCE; BAN INFLATED WORDS.** Carry a **Claim-Evidence Ledger**: each
   claim → exact job/number → **estimator** (median-of-N, never a single sequential mean) → domain tested →
   the measurement that would **falsify** it → verdict. If the language exceeds the evidence, weaken the
   language. **Banned until proven:** "bit-exact/identical" (needs `torch.equal`, not cosine); "noise
   floor"/"ratified" (needs independent justification + a source-faithful reference); "every M"/"all"/
   "held"/"floor" (needs every case tested — one 0.96 datum kills "≥1.0 floor"); "only path"/"exhausted"/
   "resolved"/"proven"/"irreducible" (needs a falsification attempt).

9. **ESTIMATOR & PROVENANCE HONESTY.** Label the estimator exactly; replicate **≥3** and report median +
   spread for any floor / no-regression claim; bind each run to **source/build/thread-affinity/input**
   identity; a harness must PROPAGATE failure (`pipefail`; test it FAILS on a known-bad run) and bind its
   outputs to run identity; investigate anomalies (comparator instability across jobs) before crediting a
   ratio to the intended cause.

10. **MEASUREMENT-PHYSICS SANITY.** A bandwidth above nominal DRAM = **cache-resident**, not saturation;
    utilization VARIES across M — never report one point as uniform; roofline = `max(bytes/BW, flops/peak)`
    at the op's ACTUAL dtype; a warm-cache microbench is not DRAM traffic.

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
- [ ] **Source boundary:** oracle/reference conforms to the AUTHORITATIVE published op / real kernel at
      each stage (dtype / accumulation / block structure / output precision); any replica's differences are
      recorded and the gate is PROPOSED until source-faithful; operands/output boundary are matched.
- [ ] **Domain ⊇ claim:** tested signed/real-shapes/sentinel/empty/every-dispatch-branch; coverage equals
      the claim's domain (a benchmark at M is not an acceptance gate at M).
- [ ] **Claim-Evidence Ledger** attached: every claim cites its job/number + estimator + the datum that
      would falsify it; no banned word without proof; ≥3-trial median+spread for any floor/no-regression
      claim; runs bound to source/build/affinity identity; harness tested to FAIL on a known-bad run.
- [ ] **Physics sane:** no bandwidth above DRAM called "saturation"; no single point reported as uniform
      utilization; roofline uses the op's actual dtype.

## Scope

Framework- and HW-agnostic; the method is the skill. Applies to both legs and to
every reviewed deliverable — correctness/acceptance gates (`accuracy-oracle`,
`coverage-gate`, `kernel-feasibility-gate`, `enablement-certificate`), provenance/
reconciliation records, ingestion/parsers, and the review-response itself. Pair
with `high-information-runs` (design the probe) and the external-reference
provenance rule (cite an authority a reviewer can open).
