---
name: adversarial-self-audit
description: "Use BEFORE submitting ANY gate/validator/record/review-response/perf-or-correctness claim for review (both legs). Encodes the pre-submit discipline that stops the multi-round FAIL spiral — THREE clusters: (A) structural (fix the CLASS not the instance; falsify not confirm; provenance of your OWN claims; test through the full gate); (B) measurement/conformance (conform to the AUTHORITATIVE source boundary not your reconstruction; test the ADVERSARIAL domain; a Claim-Evidence Ledger with banned inflated words; replicated-median + run-identity provenance; measurement-physics sanity); and (C) evidence validity & disposition (validate BOTH sides and reject invalid-not-just-imprecise evidence; bind checks to the EVALUATED artifact not a preliminary sample; bind evidence to expected identity + coordinate consistency not a producer flag; closure != promotion and a measured loss/neutral is a valid disposition). Run it as a hard pre-submit self-audit so the reviewer finds nothing you could have found yourself."
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
   ratio to the intended cause. **The evidence ARTIFACT must be self-identifying**: embed job/source/node/
   config in the published JSON (not reused filenames a later job overwrites), persist the **process
   samples + spread** (not just the aggregate), compute derived values (percentages) **before** display
   rounding, and commit the **aggregation procedure** so the published numbers are reproducible. A
   launcher that resolves its repo/dir **inside** `cd "$(...)"` hides the inner failure from `pipefail` —
   resolve+validate in a separately-checked step and FATAL-exit before any compile/output.

10. **MEASUREMENT-PHYSICS SANITY.** A bandwidth above nominal DRAM = **cache-resident**, not saturation
    (apply the cache caveat to EVERY reused-buffer row, not just the obvious one); utilization VARIES across
    M — never report one point as uniform; roofline = `max(bytes/BW, flops/peak)` at the op's ACTUAL dtype,
    but the peak is a **nominal reference** (base-clock datasheet, a DIFFERENT node's observation is a prior
    not a ceiling) — report "% of nominal reference", never "% of an achievable ceiling"; the larger-term
    **regime tag is a diagnostic, not a measured bottleneck**, and a simplified FLOP count (omitting
    exp/recurrence) must say so; a warm-cache microbench is not DRAM traffic; roof-proximity in one dtype
    does NOT explain a different-dtype candidate's loss (retain the measured loss without the causal claim).

## Third cluster — EVIDENCE VALIDITY, EVALUATED-ARTIFACT BINDING & IDENTITY (what the later rounds found)

> **Anti-pattern that actually happened (DSv4 rounds 4-6).** Each round the reviewer took the SAME rule one
> notch deeper: a replay bound evidence to a producer's own `validated=TRUE` flag (not the expected source
> **sha**/canonical **job** field); it checked record **key presence** but not **value consistency** (an
> M1 record **relabelled N=64** was accepted → M1 run reported as M64); it then checked shapes but not
> tensor **contents** (a **NaN** oracle output passed and printed NaN metrics as if valid); and a
> required-coordinate check validated a **preflight build** while a stateful builder handed `run_case` a
> **different** tuple. Every one was a sibling of "the validator trusts its input."

11. **VALIDATE EVIDENCE ON BOTH SIDES, AND REJECT INVALID — NOT JUST IMPRECISE.** The reference/oracle is
    evidence too: validate its **arity, shape==candidate, dtype, device, finiteness** before comparing, and
    reject **invalid** inputs (NaN/+inf tensor contents, a non-finite reference OR candidate output,
    non-finite cos/mae) as a **hard failure** — invalid evidence is not an approximation exceeding a
    tolerance, and a gate must never print NaN metrics as if valid. Distinguish a **legal domain value**
    (e.g. a published `-inf` causal mask) from an **invalid** one (NaN/+inf) with an explicit,
    field-specific policy; add a **valid positive control** alongside each invalid-input negative.

12. **BIND VALIDATION TO THE ACTUAL EVALUATED ARTIFACT, NOT A PRELIMINARY SAMPLE.** A preflight build, a
    first call, or a separate sample does not establish the shape/dtype of the tuple actually passed to the
    candidate and reference — a stateful/dynamic builder drifts. Build **once per evaluation**, validate
    **that** tuple, use it for **both** sides, on **every** seed; keep a declaration-presence check separate
    from execution evidence, and add a **builder-drift** negative control that fails for the coordinate
    mismatch itself.

13. **BIND EVIDENCE TO EXPECTED IDENTITY & COORDINATE CONSISTENCY, NOT A PRODUCER FLAG OR MERE PRESENCE.**
    A producer's own "validated" boolean cannot bind evidence to the consumer's EXPECTED source — compare
    the expected **content hash** and the producer's **canonical** run-identity field (reject a conflicting
    legacy **alias**, don't let it override). Validate **coordinate consistency** (declared `N` == the
    actual tensor batch of every tensor; full shape relationships; finite positive scalars) not just key
    presence, and enforce required coverage as the **joint coordinate** (op×path×distribution×M×shape),
    un-removable and proven by a through-`run()` removal control — presence of a label is not evidence the
    shape was evaluated. **A consumer that COMBINES/aggregates evidence must VALIDATE the inputs it
    combines**, not merely record their metadata: enforce the declared replicate count, **distinct** process
    identity (the producer must emit one), **compatible** run/source/runtime contracts, complete coordinates,
    and finite-positive values — fail-closed with **no output written** on reject, and route historical/
    unstamped inputs through an **explicit, separately-qualified legacy path** (not the same status as
    validated ones). And **verify the POSITIVE case too**: an over-strict check breaks valid inputs (a `-d
    .git` test FATAL-s on a git **worktree** where `.git` is a file; `rev-parse` success already validates —
    don't validate a value and then discard it for an unchecked fallback).

14. **CLOSURE ≠ PROMOTION; A MEASURED LOSS/NEUTRAL IS A VALID DISPOSITION.** Keep **arithmetic acceptance
    separate from speed**: prove a "bit-exact" change by building OLD and NEW and diffing outputs (not
    cosine), screen a non-bit-exact variant separately, and never let a performance result flip an
    UNRATIFIED correctness threshold or a BLOCKED promotion gate. **Attribute before you change** (a
    neutral measurement means the compiler/library already does it — revert, don't ship an unmeasured
    micro-opt); classify the **regime** (BW / compute / dispatch) first; a precision or blockwise lever only
    pays when the op is in the matching regime (vector-exp can be slower than scalar; many small packed
    GEMMs can lose to two large library GEMMs). Record a rejection/neutral with its number and mechanism —
    it is a legitimate completion, not a failure to be hidden.

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
- [ ] **Evidence validity (both sides):** reference AND candidate validated for arity/shape/dtype/device/
      finiteness; invalid (NaN/+inf) rejected as a hard failure (not a tolerance miss); legal domain values
      (published −inf masks) distinguished with an explicit policy + a valid positive control.
- [ ] **Evaluated-artifact binding:** the coordinate/shape check runs on the SAME tuple passed to candidate
      and reference, every seed (not a preflight sample); a builder-drift control fails for the mismatch.
- [ ] **Identity & coordinate consistency:** evidence bound to the expected content hash + canonical run-id
      (conflicting alias rejected), declared N == tensor batch, required coverage is the joint coordinate
      and un-removable (through-`run()` control).
- [ ] **Closure ≠ promotion:** arithmetic acceptance kept separate from speed; bit-exact proven by diff not
      cosine; neutral/loss dispositions recorded with number+mechanism; no UNRATIFIED threshold or BLOCKED
      gate flipped by a perf result.

## Scope

Framework- and HW-agnostic; the method is the skill. Applies to both legs and to
every reviewed deliverable — correctness/acceptance gates (`accuracy-oracle`,
`coverage-gate`, `kernel-feasibility-gate`, `enablement-certificate`), provenance/
reconciliation records, ingestion/parsers, and the review-response itself. Pair
with `high-information-runs` (design the probe) and the external-reference
provenance rule (cite an authority a reviewer can open).
