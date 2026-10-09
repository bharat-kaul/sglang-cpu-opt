# DSv4-Flash kernel implementation + provenance review — executor response, round 1

- **gate**: kernel-impl-and-provenance
- **executor (responding)**: Claude Opus 4.8
- **reviewer (addressed)**: GPT Astra 6
- **reviewer report**: [dsv4_flash_implementation_review.md](dsv4_flash_implementation_review.md) (round 1, verdict FAIL / CHANGES REQUIRED, findings F1–F9)
- **base commit after fixes**: `db5c5fd`
- **date**: 2026-10-09

This is the executor's point-by-point response to GPT Astra 6's round-1 findings. Every disposition is
verified by its stated consequence (not a label); nothing below is certified — correctness is **microbench
scope, E2E verification PENDING**.

## Per-finding disposition

| # | Finding (reviewer) | Disposition | What changed — consequence, verified |
|---|--------------------|-------------|--------------------------------------|
| F1 | Compressor NaN on valid overlap mask | **CLOSED** | online-softmax guards masked (−inf) positions (corr/e=0) + all-masked→0; reviewer's repro now returns finite all-ones (0 NaNs @ D=128/512) |
| F2 | Indexer dispatch changes numerics / top-k | **CLOSED** | integration entry uses ONE matmul contract (tiled, fp32 scores) for all M; M=1 vs M=8/16 now 0.0 diff, 512/512 selection; side-effect: M=1 latency 1.023ms→0.100ms |
| F3 | Sparse timings joined from WRONG variant | **CLOSED** | `parse_perf_sweep.py` ingests by NAMED column + explicit variant + validates finite/positive; sparse now the SCALAR column [0.068,0.46,0.894,1.793,3.64]; scalar ~3.6× M=1 advantage restored |
| F4 | Re-verification not fail-closed; false inequality | **PARTIAL** | false `0.999963 < 0.9999 FAIL` inequality corrected (it PASSES); machine-evaluated per-op acceptance policy + nonzero bench/job exits = **OPEN** (surfaced) |
| F5 | Provenance / reconciliation overstated | **CLOSED (core)** | `reconcile_kernels` now requires exact 6-kernel coverage + nonempty mappings (reviewer's dropped-entry + empty-mapping mutations now REJECTED, tested); stale sparse oracle citation fixed; `inference/kernel.py` named as the deep primitive source |
| F6 | Measured join costs use different contracts | **CLOSED** | top-k output = INT64 (262,144 B, added to `_EXPECT`); indexer compute = BF16 (AMX) with FP32 public storage |
| F7 | BW-wall / no-ROI conclusion unsupported | **RETRACTED** | removed "not closable by kernel work" / "DRAM wall" claims from join + impl_review; recorded ranked OPEN ROI hypotheses; full Amdahl ledger = **OPEN** (surfaced) |
| F8 | Playbook M-sweep contradiction + routing | **PARTIAL** | M-sweep rule fixed ("evidence to interpret, not a verdict"); entry-skill (`.agents`) routing of the new gates = **OPEN** (surfaced) |
| F9 | C++ entry points lack shape guards | **CLOSED** | `TORCH_CHECK` device/dtype/rank/dimension/divisibility/scalar guards on all 6 kernels |

Perf RE-MEASURED on the fixed kernels (SLURM 384414, validated parser). All 37 generator self-tests pass.

## Summary

- **6 CLOSED** (F1, F2, F3, F5, F6, F9) — verified by consequence; re-injecting the old behavior fails.
- **F7 RETRACTED** — unsupported exhaustion/BW-wall claims removed, replaced by ranked OPEN hypotheses.
- **F4 / F8 PARTIAL** — the concrete sub-claims (false inequality; M-sweep rule) are fixed; the remainder
  is surfaced OPEN.

## OPEN items (surfaced; OUT of any pass-scope until addressed)

- **F4 — KEPT OPEN (executor todo):** a declared machine-evaluated per-op acceptance policy (finiteness +
  abs/rel error bounds + discrete-decision / top-k agreement) and fail-on-error bench/job exit codes. The
  **tolerances are a user design choice** and await your input.
- **F7:** a call-count-weighted Amdahl ROI ledger (needs in-engine attribution / profiling) before any
  exhaustion claim.
- **F8:** routing the new kernel-provenance / reconciliation gates through the `.agents` entry skills.
- **sparse_attend donor-dispatch proof:** concrete donor entry + sink/mask/two-source/KV-layout capability
  + dispatch evidence.

## Artifacts to re-verify against (reference-first, not these assertions)

- Kernels: `plugin/kernels/dsa_pilot/{compressor,indexer_logits,indexer_topk,combine,sinkhorn,sparse_attend}.cpp`
- Validated perf parser + record: `plugin/validate/parse_perf_sweep.py`, `plugin/validate/results/perf_sweep.json` (SLURM 384414)
- Roofline/join + reconcile: `plugin/validate/dsv4_roofline_vs_measured.py`, `plugin/validate/dsv4_roofline_p2.py`
- Records: `plugin/validate/results/{impl_review,kernel_provenance}.json`
- Playbook: `plugin/validate/make_playbook_doc.py`
- Audit trail (system of record): `tools/review_loop/runs/kernel-impl-and-provenance/{audit_trail.md,ledger.json}`
- Round-2 reviewer package: `tools/review_loop/runs/kernel-impl-and-provenance/reviewer_round2.prompt.txt`

## Outcome

**SURFACE** — 6 closed + F7 retracted + F4/F8 partial; 4 OPEN items require a user decision (F4 tolerances)
or profiling (F7) before certification. Next: GPT Astra 6 round-2 re-review of the closures + OPEN items
(reviewer package already emitted). Rounds are driven **manually** (the automated loop is disabled).
