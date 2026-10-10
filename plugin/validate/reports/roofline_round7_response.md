# Response to the R7 Roofline-Response Assessment (`4ef6fbb`)

Addressing [roofline_round6_response_assessment.md](plugin/validate/reports/roofline_round6_response_assessment.md) (`4ef6fbb`). The assessment confirmed R6-F1 **CLOSED**, R6-F2/F3 **repaired for the historical case** with two remaining gaps (R7-F1 prospective-aggregation validation, R7-F2 root normalization). Both fixed in `7cc4353`; correctness closures R5-F1/R5-F2 untouched. F4 **PARTIAL**, promotion **BLOCKED**.

## R7-F1 (Medium) — the aggregator now validates the evidence it combines
[aggregate_roofline.py](plugin/validate/aggregate_roofline.py) is fail-closed and writes **no output** on any violation:
- rejects a **replicate-count** mismatch (declared count ≠ number of inputs);
- rejects **incompatible** run/source/runtime contracts across processes (`slurm_job_id`, `head`, `threads`, `omp`, `bind` must agree);
- rejects **duplicate / non-distinct `process_index`** — and the producer ([bench_roofline_sweep.py](plugin/validate/bench_roofline_sweep.py)) now **emits `process_index`** (`ROOFLINE_PROC` from the sbatch loop) so genuine replicates are distinguishable from duplicate inputs;
- rejects **incomplete** (kernel, M) coordinates and **non-finite / non-positive** latencies (no more NaN samples/median/percentage);
- historical **unstamped** scratch files are accepted **only** under an explicit, separately-qualified `--legacy` path (`status = LEGACY_RECONSTRUCTED`, provenance **not** producer-stamped); a stamped input under `--legacy`, or an unstamped input without it, is rejected.

A committed **9-control `--selftest`** proves each rejection and the valid 3-process + legacy-import **positive controls** (AGGREGATE-SELFTEST OK). The published [results/roofline_sweep.json](plugin/validate/results/roofline_sweep.json) was **regenerated via the committed `--legacy` path** (`status = LEGACY_RECONSTRUCTED`); the log-reconstructed run identity is retained but explicitly labelled a reconstruction (`reconstructed_from_log`), not producer-stamped. Arithmetic is unchanged (sparse M64 71.81%, top-k M1 0.67%, combine M64 97.54%).

## R7-F2 (Low) — fallback now resolves the canonical root
The fallback no longer cd's to the submit directory. It captures the canonical **root** via `git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel` (a submit **subdirectory** normalizes to the root) and cd's to **that** ([run_roofline_sweep.sbatch](plugin/validate/run_roofline_sweep.sbatch#L13)); applied to all four launchers.

While verifying, I caught a bug my **own R6-F3 hardening** would have introduced: the `-d "$REPO/.git"` check is **false on a git worktree checkout** (`.git` is a file here, not a directory) and would have FATAL-ed on a valid repo. Dropped it — `rev-parse --show-toplevel` success already validates the repo. Verified across all four launchers: **root** resolves root, **subdirectory** fallback resolves to the root, **outside a repo** FATAL-exits before any work.

## Verified closures (per the assessment)
- **R6-F1 CLOSED** — nominal-reference ratios, diagnostic regime tags, cache caveat, simplified FLOPs, different-node priors, and S1's measured-loss-vs-unmeasured-cause distinction are all qualified.
- **R6-F2** historical arithmetic/spread reproduced at all 40 coordinates; prospective aggregation is now validated (R7-F1).
- **R6-F3** invalid-root rejection stands; root normalization completed (R7-F2).

## Correction accepted
The reviewer is right that "19 replay controls through real `main()`" conflated the **19 persistent selftest controls** with the reviewer's separate main-entry fault probes. Precisely: the replay **selftest** has 19 controls (validator-level); the reviewer independently exercised `main()` with in-memory substitutions. I will keep those two counts distinct in future wording.

## Scope / status
No native kernel source, replay validator, F4 harness, or promotion policy changed. No cluster rerun (inputs preserved; `--legacy` reproduces them). The sweep remains standalone microbench vs a **nominal-reference** roofline, FP32-KV indexer contract (does not isolate I1's incremental effect). F4 **PARTIAL**, promotion **BLOCKED**, thresholds **UNRATIFIED**; no budget, experiment, dispatch/E2E, or phase promotion requested.
