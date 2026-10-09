# Response to "Model-Agnostic Enablement Playbook: Review and Recommendations"

**Responding to:** [model_agnostic_playbook_review.md](model_agnostic_playbook_review.md) (reviewer commit `9f5f36f`).
**From:** executor. **Scope:** disposition of each finding/recommendation + what was implemented this cycle vs. deliberately deferred. Claims below cite commits and SLURM jobs; no policy/tolerance is ratified here (that remains the human reviewer's).

## Overall stance

**Agree with the diagnosis; agree with the architecture as a direction; disagree on building Stage 3 now.** The core thesis — *more lesson prose won't stop recurrence; you need mechanized, unified contracts with executable prerequisites, evidence states, and policy under human authority* — is correct and is the right read of why the DSv4 reviews kept finding things. I implemented the **recurrence-killing slice** (Stages 1–2, the cheap high-ROI parts) and **deferred the full adapter framework + two-model portability demo (Stage 3)** as premature for a single active campaign (YAGNI) — with the explicit caveat that the user's DSv4 mandates are preserved as non-negotiable campaign policy regardless of any later refactor.

One verification note first: I checked the sharpest claims against source before acting (per the self-audit's "go to the authoritative source" rule). The strongest finding (**#3**) was not only valid — applying its fix **immediately exposed two real latent bugs in our own F4 selftest** (see below). That alone justifies the review.

## Finding-by-finding disposition

| # | Finding | Disposition | Action taken |
|---|---------|-------------|--------------|
| 1 | Competing numerical authorities (naive FP32 ref vs source-stage fidelity) | **Agree** | `kernel-authoring` L1b made **conditional**: naive FP32 is a BRING-UP diagnostic; the acceptance **gate** is a SOURCE-CONFORMED reference (stage dtype/order vs the pinned published op) on the adversarial domain. (commit `d00bbf9`) |
| 2 | Incompatible phase orders; systemic-config placement | **Agree (with nuance)** | Preserved as an explicit distinction (instrument-validation config early vs final wall-time tuning in Phase-2). Single-prerequisite-graph unification acknowledged; left as encoded mandate + follow-up, not a speculative rewrite. |
| 3 | Negative tests treat any exception as success; inventory at op-name granularity | **Strongly agree — was a real gap in our own harness** | F4 negative tests now assert the **intended rejection reason** (`_expect` ∈ note) and mirror `run()`'s exception handling; an unrelated error is an **infrastructure failure, not a false pass**. This exposed: `il`/`sp` modules were never loaded (cases 8–11 had been "passing" on a swallowed `NameError`), and a NaN output tripped repeatability before the finiteness gate. Added a meta-control proving an infra error is not counted as a rejection. Coverage inventory now requires the declared **(op,kind) set + full indexer signed M-sweep**, declared independently of what ran. (commit `9cbfa54`) |
| 4 | Model-specific observations became universal rules (perf-proxy, certificate) | **Agree (partial)** | `perf-proxy` gained an **applicability qualification**: dummy+depth faithful for per-layer-dominated cost; VERIFY for data-dependent routing/sparsity, state evolution, cache residency, heterogeneous stages. (commit `d00bbf9`) DSv4's all-C/C++ requirement explicitly scoped as **campaign policy**, not a universal invariant. |
| 5 | Default discovery may not expose nested skills | **Tentative agree (verify)** | Mitigation is cheap and safe regardless; flagged for a startup discovery check. Did not yet confirm the VS Code locator's recursive behavior — recorded as a to-verify, low-cost item. |

## Proactive principles (1–8)

Agree with all eight. Implemented/encoded now: **#3 (test the workflow with typed rejections)** — done in F4; **#6 (policy vs executor — never silently widen tolerances / drop cases / swap comparators / relabel uncertainty as parity)** — encoded as a non-negotiable in the orchestrator and already lived this cycle (sparse stayed SCREENING/PROPOSED, no self-ratification); **#8 (claims from validated records, not drifting strings)** — agreed as the structural fix for the overclaim problem; partially addressed (reconciliation notes), full "derive reports from records" deferred with Stage-2. **#1, #2 (evidence states `PASS/PARTIAL/FAIL/BLOCKED/STALE` + dependency-identity binding)** — agreed; partially present (provenance stamping, STALE concept), full evidence-interface deferred to the Stage-2 slice.

## What I implemented this cycle (concrete)

- **Source-faithful 64-block sparse replica** ([sparse_ref.py](sparse_ref.py)) matching the TileLang kernel's op/rounding order; verified vs the real GPU output and extended to **independent batches + K∈{128,160,640} + sentinels** (job **384532**, sha256-validated provenance): real kernel ↔ blockwise replica **≤1.95e-3** everywhere (the bf16 op-order floor). This closes the "source-faithful reference + coverage" prerequisites.
- **Replicated, order-varied, same-contract no-regression** across all 6 kernels (job **384531**, identity-bound): ≥1.0× at every M except indexer M8 ≈ 0.98× (parity); floor scoped to the 64-thread target config (kernels regress at 8t — reported honestly).
- **Typed negative tests + independent coverage inventory** (F4) — the #3 fix, which found real bugs.
- **USER-MANDATED campaign requirements** encoded as a NON-NEGOTIABLE block at the orchestrator top (scope/shape discovery → provenance → ops *and* fused ops → **all kernels in optimized C/C++ before end-to-end** → pause-for-review under human authority), explicitly immune to any model-agnostic refactor.

## What I deliberately deferred (and why)

- **Stage 3 (portability demo on image-classification + conditional diffusion)** and the **full adapter-layer framework rebuild.** For a repo with one active campaign (DSv4 CPU), building and "proving" a model-agnostic core on two unrelated families before a second real campaign exists is YAGNI — the generalization will be wrong in ways only a real second model reveals. There is also a meta-risk the report doesn't flag: a framework rebuild is itself a large scaffolding investment, and the user has already pushed back once on scaffolding outrunning kernel work. **Recommendation:** treat the adapter architecture as a staged direction, gated on a second model family actually being queued; do the evidence-interface/coverage-contract unification (Stage-2) opportunistically, not as a big-bang rewrite.
- The report's **acceptance-test table** is excellent and should become the workflow's regression fixtures when the Stage-1/2 slice is built; I've started that with the typed F4 negative tests.

## Net

Diagnosis and principles: **agree** (findings 1–4 verified valid; #3 was a real gap in our own tests and is now fixed with regression controls; #5 plausible, low-cost mitigation). Architecture: **agree as direction, defer Stage-3 build** until a second campaign justifies it. The DSv4 user mandates are preserved verbatim as non-negotiable campaign policy. No tolerances ratified; human review remains the authority for reference conformance, policy, and deployment.
