# Response to the R7 Roofline-Response Assessment (`450837e`) — R8-F1

Addressing [roofline_round7_response_assessment.md](plugin/validate/reports/roofline_round7_response_assessment.md) (`450837e`). The assessment confirmed **R7-F2 CLOSED** (all four launchers normalize root/subdir/outside correctly, worktree-style layout exercised) and **R7-F1 original counterexamples CLOSED** (11 through-`main()` negatives, zero output on reject), with one remaining portion: the identity contract. Fixed in `781f881`; no native/F4/replay/promotion changes.

## R8-F1 (Medium) — typed provenance schema + reference-peak binding
The aggregator compared provenance with `.get()`, so **all-missing fields compared equal as `None`** (unidentifiable inputs still marked VALIDATED), and `node`/`torch`/`peak_nominal` were not in the contract. Fixed in [aggregate_roofline.py](plugin/validate/aggregate_roofline.py):
- Each new-format `_provenance` is validated against a **required typed schema** — `slurm_job_id`, `head`, `threads`, `omp`, `bind`, `process_index`, `node`, `torch` — each **present and non-empty** — **before** any cross-record comparison. All-`None` no longer passes.
- The **complete compatibility contract now includes `node` and `torch`** and must agree across the single-node replicated sweep; a cross-node run must be a **separately-qualified experiment**, not silently pooled.
- The stamped **`peak_nominal` is bound to the aggregator's own BW/AMX/FP32 constants** — rejected if they differ, because the floor would otherwise be computed against a reference the producer did not measure against.

**Controls** (aggregator `--selftest`, now 19, all pass): missing required field (`slurm_job_id`/`head`/`node`/`torch`/`process_index`), empty identity, different node, different torch, and a differing reference-peak each reject for the intended reason with **no output written**; the positive control is a **complete producer-shaped** record. The historical `--legacy` path is unchanged, and [results/roofline_sweep.json](plugin/validate/results/roofline_sweep.json) stays `LEGACY_RECONSTRUCTED` (sparse M64 71.81%).

## Verified closures (per the assessment)
- **R7-F2 CLOSED** — 12 launcher combinations (4 × root/subdir/outside) behave as intended; worktree `.git`-file layout exercised.
- **R7-F1** original counterexamples CLOSED; the identity-contract remainder is now closed by R8-F1.
- **R6-F1 / R5-F1 / R5-F2** closures and the historical arithmetic (all 40 coordinates, top-k M1 0.67%) remain unchanged; the published artifact is explicitly `LEGACY_RECONSTRUCTED` with the run identity under `reconstructed_from_log`, not producer-stamped.

## Scope / status
No native kernel source, replay validator, F4 harness, or promotion policy changed; no cluster rerun (the `--legacy` path reproduces the preserved inputs). The sweep remains standalone microbench vs a **nominal-reference** roofline, FP32-KV indexer contract (does not isolate I1's incremental effect). F4 **PARTIAL**, promotion **BLOCKED**, thresholds **UNRATIFIED**; no budget, experiment, dispatch/E2E, or phase promotion requested.
