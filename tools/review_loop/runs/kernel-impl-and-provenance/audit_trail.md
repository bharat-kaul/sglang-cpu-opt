# Audit trail — review-gated gate: kernel-impl-and-provenance

- executor: Claude Opus 4.8
- reviewer: GPT Astra 6
- started: 2026-10-09

## Round 1 — verdict: FAIL (CHANGES REQUIRED)

**Review report (reviewer: GPT Astra 6)** -> `plugin/validate/reports/dsv4_flash_implementation_review.md`
- 9 findings (F1-F9), several HIGH-severity real bugs, with executed C++/Python counterexamples.

**Executor response (standalone, hand to reviewer)** -> `plugin/validate/reports/kernel-impl-and-provenance_response_round1.md`

**Collateral + Addressing (executor: Claude Opus 4.8)** — per finding:

| # | Finding | Disposition | What changed (consequence, verified) |
|---|---------|-------------|--------------------------------------|
| F1 | Compressor NaN on valid overlap mask | **CLOSED** | online-softmax guards masked (-inf) positions (corr/e=0) + all-masked→0; reviewer's repro now returns finite all-ones (0 NaNs @ D=128/512) |
| F2 | Indexer dispatch changes numerics / top-k | **CLOSED** | integration entry uses ONE matmul contract (tiled, fp32 scores) for all M; M=1 vs M=8/16 now 0.0 diff, 512/512 selection; also collapsed M=1 latency 1.023ms→0.100ms |
| F3 | Sparse timings joined from WRONG variant | **CLOSED** | `parse_perf_sweep.py` ingests by NAMED column + explicit variant + validates finite/positive; sparse now the SCALAR column [0.068,0.46,0.894,1.793,3.64]; scalar ~3.6x M=1 advantage restored |
| F4 | Re-verification not fail-closed; false inequality | **PARTIAL** | false `0.999963 < 0.9999 FAIL` inequality corrected (it PASSES); machine-evaluated per-op acceptance policy + nonzero bench/job exits = **OPEN** (surfaced) |
| F5 | Provenance/reconciliation overstate | **CLOSED (core)** | `reconcile_kernels` now requires exact 6-kernel coverage + nonempty mappings (reviewer's dropped-entry + empty-mapping mutations now REJECTED, tested); stale sparse oracle citation fixed; `inference/kernel.py` named as the deep primitive source |
| F6 | Measured join costs different contracts | **CLOSED** | top-k output = INT64 (262,144 B, added to `_EXPECT`); indexer compute = BF16 (AMX) with FP32 public storage |
| F7 | BW-wall / no-ROI conclusion unsupported | **RETRACTED** | removed "not closable by kernel work" / "DRAM wall" claims from join + impl_review; recorded ranked OPEN ROI hypotheses; full Amdahl ledger = **OPEN** (surfaced) |
| F8 | Playbook M-sweep contradiction + routing | **PARTIAL** | M-sweep rule fixed ("evidence to interpret, not a verdict"); entry-skill (.agents) routing of the new gates = **OPEN** (surfaced) |
| F9 | C++ entry points lack shape guards | **CLOSED** | TORCH_CHECK device/dtype/rank/dimension/divisibility/scalar guards on all 6 kernels |

Perf RE-MEASURED on the fixed kernels (SLURM 384414, validated parser). All 37 generator self-tests pass.

**Addressing**: committed; the next round (GPT Astra 6 re-review) will verify the closures and the OPEN items.

## Outcome: SURFACED to user

6 findings CLOSED (F1, F2, F3, F5, F6, F9) + F7 retracted + F8 M-sweep fixed. **OPEN items SURFACED for a user call** before certification:
- F4: a declared machine-evaluated per-op acceptance policy (finiteness + abs/rel error + discrete decisions) and fail-on-error bench/job exits — a design choice (tolerances) needing your input. **KEPT OPEN in the executor todo.**
- F7: a call-count-weighted Amdahl ROI ledger (needs in-engine attribution / profiling) before any exhaustion claim.
- F8: routing the new kernel-provenance/reconciliation gates through the `.agents` entry skills (the "old scheme" deferred set).
- Donor-dispatch proof for sparse_attend (concrete donor entry + capability + dispatch evidence).
Nothing is certified; correctness is microbench scope, E2E verification PENDING.

## Round 2 — verdict: PENDING (GPT Astra 6 re-review TRIGGERED)

**Trigger (executor: Claude Opus 4.8)** — base commit db5c5fd. Round-2 reviewer collateral emitted →
`tools/review_loop/runs/kernel-impl-and-provenance/reviewer_round2.prompt.txt`. GPT Astra 6 is asked to
INDEPENDENTLY re-verify every round-1 closure (F1, F2, F3, F5, F6, F9) by counterexample (re-injecting the
old behavior must fail), confirm the retractions/corrections (F4 inequality, F7 overclaim, F8 M-sweep rule)
are gone-not-relabeled, and assess the OPEN items' plans.

**Pass-scope**: round 2 passes only if no round-1 closure is unsound. The four STILL-OPEN items are
explicitly OUT of pass-scope:
- **F4 — KEPT OPEN (executor todo):** machine-evaluated per-op acceptance policy + fail-closed bench/job
  exits; **tolerances await a user decision**. Reviewer to state the minimum acceptance-policy contract.
- F7 Amdahl ROI ledger (profiling); F8 `.agents` entry-skill routing; sparse donor-dispatch proof.

**Expected reviewer output** → `plugin/validate/reports/kernel-impl-and-provenance_review_round2.md` +
one `<<<REVIEW-VERDICT {...} REVIEW-VERDICT>>>` block (PASS / FAIL / SURFACE). Awaiting GPT Astra 6.

## Round 2 — verdict: FAIL (GPT Astra 6 re-review returned; commit `33c0b57`)

**Review report (reviewer: GPT Astra 6)** → `plugin/validate/reports/kernel-impl-and-provenance_review_round2.md`
- Reviewer ran real C++/parser/reconcile probes. Confirmed-good: F1, F2 (scoped), F3 medians, F6 bytes,
  F7/F8. Six findings showed round-1 closures were not adversarial enough.

**Executor response (standalone)** → `plugin/validate/reports/kernel-impl-and-provenance_response_round2.md`
(hand this to the reviewer alongside `reviewer_round3.prompt.txt`)

**Collateral + Addressing (executor: Claude Opus 4.8)** — per finding:

| # | Finding | Disposition | What changed (consequence, verified) |
|---|---------|-------------|--------------------------------------|
| R2-F3 | Ingestion still positional; accepts 1 rep as "3", admits +inf, reorder picks wrong variant | **CLOSED** | `parse_perf_sweep.py` rewritten: NAMED header column (`cpp_ms`/`sc_ms`), exact replica set {1,2,3}, `math.isfinite`, exact M coverage; `--selftest` runs 5 cases (missing-rep / +inf / reorder / drop-M / happy) — all PASS; real 384414 medians unchanged |
| R2-F5 | Reconciliation derives required set from the same mutable doc; 4 mutations pass | **CLOSED** | `reconcile_kernels` now binds to an INDEPENDENT on-disk inventory + a per-kernel machine-readable cost-row contract + exact/unique coverage + per-gap nonempty; the reviewer's exact 4 mutations (remove-from-both / gut-gaps / duplicate / wrong-row) are REJECTED (tested); generator self-test rc=0 (39 PASS) |
| R2-F9 | sparse_attend guards incomplete (excess-batch + meta-device accepted) | **CLOSED** | both sparse entries add CPU-device + q/kv batch-equality guards; probe q[1,2,32]/kv[2,8,32] and meta-device now REJECTED (verified); valid case still works |
| R2-F9b | top-k guard rejects valid zero selection | **CLOSED** | `indexer_topk` allows `k>=0`; `k==0` returns `[N,0]` int64 == `torch.topk(·,0)` (verified) |
| R2-P1 | Current timing joined to historical implementation identities | **CLOSED** | join separates CURRENT timing (384414/pcl-sprh09, printed once) from HISTORICAL correctness record (relabeled); `impl_review.json` perf field → 384414/pcl-sprh09 + reverified-commit note; `kernel_provenance.json` sparse overclaims corrected (scalar DOES win M=1 ~3.55×; donor = CANDIDATE, not proven production) |
| R2-P2 | Batch consistency ≠ published-reference numerics; false "shared precision" comment | **PARTIAL** | false "both variants share score precision" comment corrected (fused BF16-bmm ≠ tiled; verified max diff 0.55); stale "EXHAUSTED/not closable" header verdict removed; F6 BW>compute-floor wording fixed. The authoritative per-op acceptance policy (F4) remains **OPEN** (user tolerances) |

All kernel probes compiled + verified; generator self-test rc=0; parser self-test OK.

## Outcome: SURFACED to user (round 3 re-review ready)

Six round-2 findings addressed + verified (R2-F3, R2-F5, R2-F9, R2-F9b, R2-P1 closed; R2-P2 comment/prose
closed, its acceptance-policy remainder folds into F4). **Still OPEN (unchanged):** F4 acceptance policy
(**tolerances await a user decision**), F7 Amdahl ROI ledger, F8 `.agents` entry-skill routing, sparse
donor-dispatch proof. Round-3 reviewer package emitted → `reviewer_round3.prompt.txt`. Nothing certified;
microbench scope, E2E verification PENDING.

## Round 3 — verdict: FAIL (GPT Astra 6 re-review returned; commit `48a3312`)

**Review report** → `plugin/validate/reports/kernel-impl-and-provenance_review_round3.md`
- Reviewer ran the 5 parser + 39 generator self-tests, independent in-memory mutations, raw-log
  reconstruction, and artifact regeneration comparison. Confirmed-good: R2-F9, R2-F9b, the R2-F3/R2-F5
  subcases, the R2-P1 source-label fix, the R2-P2 comment fix. Three closures were still incomplete.

**Executor response (standalone)** → `plugin/validate/reports/kernel-impl-and-provenance_response_round3.md`
(hand this to the reviewer alongside `reviewer_round4.prompt.txt`)

**Collateral + Addressing (executor: Claude Opus 4.8)** — per finding:

| # | Finding | Disposition | What changed (consequence, verified) |
|---|---------|-------------|--------------------------------------|
| R3-F3 | Duplicate samples overwrite evidence; invalid value can be overwritten before validation; extra-M silently ignored | **CLOSED** | parser now matches BENCH markers at LINE START only (shell `set -x` echo traces ignored), validates each value the instant it is read (invalid cannot be overwritten), RAISES on a duplicate `(bench,M,rep)` sample, and RAISES on any unexpected M. `--selftest` +4 cases (duplicate / NaN-before-overwrite / extra-M=128 / echo-trace-ignored) — all PASS; real 384414 medians unchanged |
| R3-F5 | Nonempty gap list ≠ complete coverage; 4 mutations pass; duplicate declaration loses identity | **CLOSED** | added an INDEPENDENT in-code stable gap-ID inventory (`_KERNEL_GAP_CONTRACT`); each entry's `gap_dispositions` must carry `gap_id`s that EXACTLY + UNIQUELY cover it; duplicate `kernels[]` declarations rejected before set conversion. The reviewer's 4 mutations (partial / unrelated / delete-gaps+dispositions / duplicate-declaration) are REJECTED (tested); generator self-test rc=0 (43 PASS) |
| R3-P1 | Published report retains old identity conflation + stale wording | **CLOSED** | `reports/dsv4_roofline_vs_measured_emr.txt` regenerated (0 stale `@e3fdcb9` labels; 6 historical + 6 current labels); join refactored to `render()` + a `--verify` regeneration guard; build/source digest of job 384414 recorded as explicitly UNRESOLVED (not assigned to HEAD); residual "best-of dispatcher" and combine "minimal traffic achieved" overclaims corrected |

All parser (9) + generator (43) self-tests pass; changed kernels compile; integration==tiled 0.0.

## Outcome: SURFACED to user (round 4 re-review ready)

Three round-3 findings addressed + verified. **Still OPEN (unchanged):** F4 acceptance policy (**tolerances
await a user decision**), F7 Amdahl ROI ledger, F8 `.agents` entry-skill routing, sparse donor-dispatch
proof. Round-4 reviewer package emitted → `reviewer_round4.prompt.txt`. Nothing certified; microbench scope,
E2E verification PENDING.

## Round 4 — verdict: FAIL (GPT Astra 6 re-review returned; commit `8874f23`)

**Review report** → `plugin/validate/reports/kernel-impl-and-provenance_review_round4.md`
- **R3-P1 CLOSED by reviewer** (report regeneration + identity separation + `--verify` drift check verified).
  Two findings remained: destination binding and parser schema integrity.

**Executor response (standalone)** → `plugin/validate/reports/kernel-impl-and-provenance_response_round4.md`
(hand this to the reviewer alongside `reviewer_round5.prompt.txt`)

**Collateral + Addressing (executor: Claude Opus 4.8)** — per finding:

| # | Finding | Disposition | What changed (consequence, verified) |
|---|---------|-------------|--------------------------------------|
| R4-F5 | Valid gap IDs accept empty / `caller:` / unrelated / wrong-existing destinations; full self-test still passes | **CLOSED** | `_KERNEL_GAP_CONTRACT` is now a gap_id→REQUIRED-DISPOSITION map (not just an id set); each entry's disposition must EQUAL its one required destination; the generic resolver now rejects empty `unmodeled:`/`caller:` and nonexistent targets. The reviewer's 4 repoints (`caller:`, `caller:NO_SUCH_CALLER`, `unmodeled:`, `modeled:hc_fn`) are all REJECTED (tested through the full gate); generator self-test rc=0 (47 PASS) |
| R4-F3 | Duplicate header name silently selects last; malformed-sample retry disappears; truncated/unparseable rows skipped; negative M ignored | **CLOSED** | parser now rejects a DUPLICATE header column (ambiguous); treats a SECOND real block for the same `(bench,rep)` as a duplicate run REGARDLESS of whether the first parsed; and inside a recognized table RAISES on an unexpected/negative M, a truncated row, an unparseable value, or a non-finite/non-positive time (no silent skip). `--selftest` +4 cases (dup-header / unparseable / truncated / negative-M) — all PASS (13 total); real 384414 medians unchanged |

Parser (13) + generator (47) self-tests pass; report `--verify` rc=0.

## F4 design-review package emitted (user asked to share with the reviewer)

- **DRAFT policy** → `plugin/validate/results/acceptance_policy.json` (PROPOSED / UNRATIFIED): three layers
  (tolerance-independent hard gates; numerical vs the AUTHORITATIVE oracle; discrete-selection with a tie
  band), per-op oracle selection, and proposed starting tolerances. Numerical tolerances are a USER decision.
- **Reviewer ask** → `tools/review_loop/runs/kernel-impl-and-provenance/reviewer_F4.prompt.txt`: 7-point
  design review (oracle selection, metric adequacy, tie rule, bit-exact classification, fail-closed
  enforcement, shape coverage, proposed numbers). Reviewer SUGGESTS; user RATIFIES.

## Outcome: SURFACED to user (round 5 re-review ready + F4 draft to share)

Both round-4 findings addressed + verified. **Still OPEN (unchanged):** F4 acceptance policy (DRAFT proposed;
**tolerances await a user decision** + reviewer suggestions), F7 Amdahl ROI ledger, F8 `.agents` entry-skill
routing, sparse donor-dispatch proof. Round-5 reviewer package emitted → `reviewer_round5.prompt.txt`.
Nothing certified; microbench scope, E2E verification PENDING.
