# DSv4-Flash kernel review — executor response, round 5 (closures) + F4 policy revision

- **gate**: kernel-impl-and-provenance
- **executor (responding)**: Claude Opus 4.8
- **reviewer (addressed)**: GPT Astra 6
- **reports**: [round-5 closures](kernel-impl-and-provenance_review_round5.md) (commit `5b05106`, FAIL: R5-F3) and [F4 policy design review](kernel-impl-and-provenance_F4_policy_review.md) (FAIL: F4-D1..D4)
- **date**: 2026-10-09

Two parallel reviews. The round-5 closure audit confirmed **R4-F5 closed (11/11)** and the **R4-F3 original
counterexamples closed**, leaving one residual (R5-F3). The F4 design review found two objective errors in my
DRAFT policy (D1, D2) plus clarifications (D3, D4). All addressed; nothing certified (microbench scope, E2E
PENDING).

## Round-5 parser finding

| # | Finding | Disposition | What changed — consequence, verified |
|---|---------|-------------|--------------------------------------|
| R5-F3 | `^-?\d+$` coordinate recognizer treats `+1` and `1.0` as prose, so a full-width table row with those spellings and a NaN latency is silently skipped and ingestion succeeds | **CLOSED** | the parser now treats ANY numeric-looking first token inside a recognized table as a data row and validates it as an integer coordinate: `+1`→M=1 (then the NaN / duplicate-M is caught), `1.0`/`1e0`→REJECTED as a malformed coordinate, `-1`/`128`→unexpected coordinate. Verified on the REAL `/scratch/bkaul/dsa_perf_sweep_384414.log`: injecting `+1`/`1.0`/`-1`/`01` rows with a NaN `sc_ms` right after the sparse REP=1 header is REJECTED in every case (zero output writes); the unmodified medians are unchanged. `--selftest` +2 cases (15 total). |

## F4 policy design review — DRAFT revised to v2

| # | Finding (reviewer) | Disposition | What changed in `acceptance_policy.json` |
|---|--------------------|-------------|------------------------------------------|
| F4-D1 | High — the DRAFT named a "published kernel.py BF16 indexer primitive"; **no such primitive exists** (scoring is in `Indexer.forward`) | **CLOSED (v2)** | indexer oracle = published `model.py Indexer.forward` scoring stage (BF16 einsum + ReLU·weights + reduction + caller mask/offset); the FP32 epilogue replay is demoted to a diagnostic proxy |
| F4-D2 | High — the DRAFT's Sinkhorn row/col-sum ≤ 1e-5 invariant **rejects the reference** (the finite 20-iter eps-regularized algorithm doesn't converge that tight; counterexample E1 ≈ 2400× over) | **CLOSED (v2)** | removed the doubly-stochastic bound; convergence residuals are reported as diagnostics only; the algorithm eps 1e-6 is kept distinct from any acceptance tolerance; the gate is cosine + abs/rel on pre/post/comb vs the conformed recurrence |
| F4-D3 | Medium — determinism conflated with oracle bit-exactness (E2: deterministic FP32 reductions can differ) | **CLOSED (v2)** | split into three separately-recorded claims (repeat-run reproducibility / same-path cross-thread equality / equality-to-a-pinned-oracle); exact-to-oracle is a per-named-frozen-path USER choice; compiler/runtime/backend identity recorded |
| F4-D4 | Medium — tie/metric/enforcement/precedence underspecified | **CLOSED (v2)** | tie cutoff + gathered scores derived from the REFERENCE (never candidate logits), with a uniqueness guard and explicit −inf sentinel handling; relerr gets a named denominator + near-zero absolute allowance; zero-vector cosine undefined → absolute check; layer-1-vs-selection precedence stated; non-finite metrics rejected |

Also incorporated from the 7-area review: per-op oracle corrections (pool = FP32 softmax-pool after
overlap/APE prep, not the full compressor; combine = `hc_pre` multiply+sum+cast, einsum equivalent not a
bit-exact oracle; sinkhorn = conform the SGLang helper to the published recurrence; top-k = `torch.topk`
default-sorted, order-independent membership scope; sparse = BF16 primitive for serving equivalence + inline
FP32 as a scalar diagnostic that lacks the BF16 unnormalized-exp cast); **domain-specific finiteness** in
layer 0 (legal −inf padding/mask positions); a **machine-enumerated shape/case manifest** (the 128/640/160
lengths are context-4096-specific; ratio-4 ⇒ an 8-token pool, not ratio-8; runtime-captured tensors +
full-model parity remain required gates); fail-closed enforcement specifics (child exit must propagate to
the job exit; a layer-0-only run is labeled PARTIAL); and the ratifiable decisions enumerated **by key**.

## Still OPEN (explicitly OUT of pass-scope)

- **F4** — policy v2 proposed; the **numerical tolerances are a user decision**, now enumerated by key in
  `acceptance_policy.json → user_decisions_required_by_key`. The layer-0 gates + exit propagation + oracle/
  metric structure + the manifest can be implemented without the numbers.
- **F7** Amdahl ROI ledger; **F8** `.agents` entry-skill routing; **sparse donor-dispatch proof**.

## Artifacts

- Parser + self-test: `plugin/validate/parse_perf_sweep.py` (`--selftest`, 15 cases), `results/perf_sweep.json`
- Policy v2: `plugin/validate/results/acceptance_policy.json`
- F4 re-review package: `tools/review_loop/runs/kernel-impl-and-provenance/reviewer_F4_round2.prompt.txt`
- Round-6 reviewer package: `tools/review_loop/runs/kernel-impl-and-provenance/reviewer_round6.prompt.txt`
- Audit trail: `tools/review_loop/runs/kernel-impl-and-provenance/{audit_trail.md,ledger.json}`

## Outcome

**SURFACE** — R5-F3 closed; the F4 draft is revised to v2 addressing D1-D4. Next: GPT Astra 6 round-6
(re-verify R5-F3) and F4 round-2 (re-review the revised policy). Rounds are driven manually (automated loop
disabled). The F4 numerical tolerances still await your ratification.
