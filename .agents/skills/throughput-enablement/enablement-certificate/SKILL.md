---
name: enablement-certificate
description: "Final gate of model enablement. Packages every gate's evidence into one machine-checkable pass/fail Enablement Certificate (op coverage, per-layer + task accuracy, ISA dispatch, absolute AND peer-relative roofline, end-to-end throughput vs peer, determinism, provenance) plus a clean plugin-only diff and the reproduce commands. Green certificate = correctness + completeness + performance proven; a human reviews it and upstreams a community-compliant PR. Also records the throughput-thesis metric: wall-clock and human-review time per model."
---

# Enablement Certificate

The deliverable. Turns "the agent says it works" into a signed, reproducible,
machine-checkable artifact a human can trust and upstream. If any field is red or
missing evidence, the certificate FAILS — no partial credit.

## Sections (all required)
1. **Coverage** — the `coverage-gate` per-op table: every op → matched kernel +
   donor, verdict ALL COVERED (fallbacks listed with their efficiency haircut).
2. **Accuracy** — `accuracy-oracle`: per-layer max-abs / cosine vs reference (with
   tolerances) + task scores (gsm8k / mmlu / hellaswag) vs reference, deltas within
   N points.
3. **Dispatch** — proof the intended AMX kernels ran (`ONEDNN_VERBOSE` excerpt /
   profile), no silent AVX-512 fallback where AMX was expected.
4. **Performance (absolute)** — `roofline-validation` per hot op: eff_abs vs
   achievable ceiling, with freq + FLOP/cycle assumptions.
5. **Performance (peer-relative)** — `peer-relative-roofline`: per op eff_rel vs the
   donor running the identical kernel; end-to-end normalized prefill+decode vs peer.
6. **Determinism** — repeated runs bit-stable / within noise.
7. **Provenance** — kernels + skills + fallbacks used, model + config hash, node
   profile, plugin commit, reproduce commands.
8. **Diff** — the plugin-only patch (thin subclass, zero `sglang/` edits),
   lint/format-clean to SGLang conventions, ready to open as a PR.
9. **Roofline target vs measured (PUBLISHED artifact PAIR — auto-emit at cycle close).** Every
   published result carries, as the CLOSING STEP of the perf cycle, TWO linked artifacts: (a) the
   roofline-achievable TARGET next to the MEASURED number (model level AND per hot op) via
   `plugin/validate/roofline_vs_measured.py`, AND (b) the time-attribution PIVOT companion
   (`plugin/validate/time_attribution_pivot.py`, where the wall-clock goes per phase, summing to
   100% with an explicit unattributed slice) — PLUS the model's bullet in the README "Roofline
   target vs measured" section. All three (roofline report, pivot report, README bullet) are
   required; a baseline→optimized "journey"/summary chart is NOT a substitute (it answers a
   different question and must not be mistaken for "charts done"). Publish the target UP FRONT
   (measured usually falls below it); the per-op efficiency-gap chart, ranked by recoverable
   fraction (shortfall × share), shows exactly which kernels underperform their roofline and are
   the next optimization RoI. For a recurrence/small-op-bound hot op, caption the realistic
   (fused-kernel) floor so a low FLOP-roofline efficiency is not misread as failure. Link both
   reports/charts from the certificate and the repo results page.
   **Same machine config, always labeled:** roofline and measured MUST be at the
   identical config (socket count, TP, batch, precision) — never a single-socket
   roofline against a dual-socket measured. State "single socket" vs "full node (2
   socket)" explicitly in the `node` field and the results table.

## Reviewer-facing hygiene blurb (put it where the boss/user looks FIRST)
A certificate proves rigor to someone who opens it — but leadership and users mostly see the
LANDING surfaces (README top, THESIS_SUMMARY, the workflow slide, a results page). Each of those
MUST carry a SHORT "what's checked, up front" blurb so a reader sees the right things were done
WITHOUT digging into the certificate. Keep it to a few lines enumerating the gates in place:
- **dtype hygiene** — stored→compute dtype audited per weight family; every `stored ≠ compute`
  dequant bridge parity-checked (not assumed).
- **coverage** — no novel op silently dense-approximated.
- **accuracy** — real-prompt coherence → per-layer parity → task accuracy.
- **honest roofline** — target vs measured at one labeled machine config.
Rules: (1) state it as a claim BACKED BY A LINK to the evidence (certificate / parity test /
coverage yaml), never a bare assertion; (2) any number not yet proven is labeled **UNVALIDATED /
PENDING**, never implied as done — an honest "in progress" blurb beats an overclaim a reviewer can
puncture (this is exactly what pre-empts the "did you actually verify X?" question). This blurb is
the human-facing dual of the machine-checkable certificate; publish both.

## Pass rule
GREEN only if: coverage ALL COVERED, accuracy within tol on BOTH layers, dispatch
confirmed, every hot op passes BOTH eff_abs ≥ 0.70 AND eff_rel ≥ 0.90, end-to-end
≥ 0.90 of peer, determinism holds, and the diff touches no core files. Any miss →
RED with the specific failing field and the gate to return to.

## Throughput-thesis metric (the business proof)
Record per model:
- `wall_clock_hours` — config in to green certificate out.
- `automation_pct` — steps completed with no human intervention.
- `human_review_minutes` — expert time to accept the certificate + diff.
- `throughput_vs_generic` — serving tok/s vs the pre-optimization generic-CPU run.
Plot these against the historical month+ baseline — that delta IS the thesis.

## Procedure
1. Collect each gate's recorded evidence; refuse to synthesize missing numbers.
2. Emit the certificate (structured, e.g. `certificates/<model>.yaml`) + the diff.
3. If GREEN, hand to the human expert for PR; if RED, return to the named gate.
4. Append the model + its measured efficiencies to the registry as a new donor
   (learned-patterns) so the next enablement starts warmer.

## Gate
A certificate is valid only if every number traces to a recorded gate run on the
current node. No estimated, cross-node, or hand-entered values — provenance is what
makes it reviewable and what lets the boss trust the hours-to-enable claim.
