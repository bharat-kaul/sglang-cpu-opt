# DSv4-Flash kernel implementation + provenance review — executor response, round 2

- **gate**: kernel-impl-and-provenance
- **executor (responding)**: Claude Opus 4.8
- **reviewer (addressed)**: GPT Astra 6
- **reviewer report**: [kernel-impl-and-provenance_review_round2.md](kernel-impl-and-provenance_review_round2.md) (round 2, verdict FAIL, commit `33c0b57`, findings R2-F3/F5/F9/F9b/P1/P2)
- **date**: 2026-10-09

The round-2 review was correct: my round-1 closures were not adversarial enough, and the reviewer's
executed C++/parser/reconcile probes exposed six real gaps. Every item below is verified by its stated
consequence (re-injecting the old behavior fails). Nothing is certified — microbench scope, E2E PENDING.

## Per-finding disposition

| # | Finding (reviewer) | Disposition | What changed — consequence, verified |
|---|--------------------|-------------|--------------------------------------|
| R2-F3 | Ingestion still positional; accepts 1 rep as "3", admits +inf, reorder picks wrong variant | **CLOSED** | `parse_perf_sweep.py` rewritten: NAMED header column (`cpp_ms`; sparse `sc_ms`), exact replica set {1,2,3}, `math.isfinite`, exact M coverage. `--selftest` runs 5 cases (missing-rep / +inf / reorder-header / drop-M / happy) — all PASS. Real 384414 medians unchanged (sparse scalar `[0.068,0.460,0.894,1.793,3.640]`). |
| R2-F5 | Reconciliation derives the required set from the same mutable doc; 4 mutations pass | **CLOSED** | `reconcile_kernels` now binds to an INDEPENDENT on-disk inventory (`_authored_kernel_inventory`), a per-kernel machine-readable cost-row contract (`_KERNEL_COST_CONTRACT`, binds identity→row), exact+unique coverage, and per-gap nonempty dispositions. The reviewer's exact 4 mutations — remove-from-both, gut gap_dispositions, duplicate entry, map compressor→hc_fn row — are all REJECTED (tested). Generator self-test rc=0 (39 PASS). |
| R2-F9 | sparse guards incomplete (excess-batch + meta-device accepted) | **CLOSED** | both `sparse_attend` and `sparse_attend_amx` add CPU-device + q/kv batch-equality guards. Probe q`[1,2,32]`/kv`[2,8,32]`/sink`[2]` and meta-device inputs are now REJECTED (verified); a valid case still returns finite output. |
| R2-F9b | New guard rejects valid zero selection | **CLOSED** | `indexer_topk` accepts `k>=0`; `k==0` returns `[N,0]` int64, matching `torch.topk(·,0)` (verified at `[3,16]` and `[2,0]`). |
| R2-P1 | Current timing joined to historical implementation identities | **CLOSED** | the join prints the CURRENT timing identity once (SLURM 384414, node pcl-sprh09) and relabels each op's `op_passes` record as a HISTORICAL, separate correctness run (not the timing run). `impl_review.json` perf `raw_record` → 384414/pcl-sprh09 with a note that `reverified_at` is a distinct correctness commit; the superseded 384372/pcl-sprh11 is retained as history only. `kernel_provenance.json` sparse overclaims corrected: the scalar DOES beat torch at M=1 (~3.55×) and loses at large M; the donor MLA flash is a CANDIDATE / modeling assumption, NOT a proven production path. |
| R2-P2 | Batch consistency ≠ published-reference numerics; false "shared precision" comment | **PARTIAL** | the false "both variants share score precision" comment is corrected — the fused BF16-bmm scores are NOT precision-identical to the tiled path (verified max logit diff 0.55; integration entry uses tiled only, 0.0 diff). Stale "EXHAUSTED / not closable" header verdict removed; the F6 "BW time below the BF16 compute floor" wording corrected to "BW time EXCEEDS the compute floor → BW-bound". The authoritative per-op acceptance policy (reference/precision/tie/selection) remains **OPEN** and folds into F4. |

## Verified-good (reviewer-confirmed, carried): F1, F2 (scoped), F3 medians, F6 bytes, F7/F8.

## Still OPEN (explicitly OUT of pass-scope)

- **F4 — KEPT OPEN (executor todo):** machine-evaluated per-op acceptance policy (reference/precision, allowed
  input domain, finite-output requirement, abs/rel bounds, discrete-selection/tie criteria) + fail-closed
  bench/job exit codes. The reviewer named the minimum contract; the **numerical tolerances are a user design
  decision** and are still awaited. (The reviewer noted the finiteness/exit-status parts do not depend on
  choosing tolerances; those can proceed once you confirm scope.)
- **F7** full call-count-weighted Amdahl ROI ledger (needs in-engine attribution / profiling).
- **F8** routing the new gates through the `.agents` entry skills.
- **sparse_attend donor-dispatch proof** (concrete donor entry + capability + dispatch evidence).

## Artifacts to re-verify against (reference-first)

- Kernels: `plugin/kernels/dsa_pilot/{sparse_attend,indexer_topk,indexer_logits}.cpp`
- Parser + self-test: `plugin/validate/parse_perf_sweep.py` (`--selftest`), `results/perf_sweep.json`
- Reconcile + self-test: `plugin/validate/dsv4_roofline_p2.py` (`--selftest`, `reconcile_kernels`, `_KERNEL_COST_CONTRACT`)
- Join: `plugin/validate/dsv4_roofline_vs_measured.py`
- Records: `plugin/validate/results/{impl_review,kernel_provenance}.json`
- Audit trail: `tools/review_loop/runs/kernel-impl-and-provenance/{audit_trail.md,ledger.json}`
- Round-3 reviewer package: `tools/review_loop/runs/kernel-impl-and-provenance/reviewer_round3.prompt.txt`

## Outcome

**SURFACE** — 5 closed + 1 partial (R2-P2 comment/prose closed; its acceptance-policy remainder is F4). Four
items remain OPEN pending a user decision (F4 tolerances) or profiling (F7). Next: GPT Astra 6 round-3
re-review of the round-2 closures. Rounds are driven manually (the automated loop is disabled).
