# Response to the Round-6 Assessment (`6562b3a`) — Roofline Interpretation & Artifact Hygiene

Addressing [correctness_round6_and_roofline_assessment.md](plugin/validate/reports/correctness_round6_and_roofline_assessment.md) (`6562b3a`). The assessment confirmed both correctness defects **CLOSED** (R5-F1, R5-F2) and raised three findings on the standalone roofline sweep — all "no new experiment needed." Fixed in `53f1e57`. No native kernel source changed; no rerun (inputs preserved per the reviewer). F4 stays **PARTIAL**, promotion **BLOCKED**, thresholds **UNRATIFIED**.

## R6-F1 (Medium) — narrow the bottleneck/ceiling/causal wording
Accepted; the numbers stand, the claims are narrowed:
- **"achieved%" is now explicitly a fraction of the NOMINAL reference** (1.9 GHz base-clock peak), **not an absolute hardware ceiling**. The "datasheet peak is a ceiling, not achievable" phrasing is removed from the bench and the results schema.
- **regime** is relabelled a "larger-theoretical-term **diagnostic**, NOT a measured bottleneck" (it measures neither the real bottleneck nor cache/DRAM traffic); the cache-resident caveat now applies to every cached row, and the compressor FLOP count is noted as **simplified (omits the exp/recurrence cost)**.
- **S1's rejection is retained on its measured loss alone.** I removed the claim that FP32-roof proximity explains it — S1 changes the compute path to BF16/AMX, so proximity to an FP32 reference cannot rule out its upside; packing/transpose/small-GEMM/state-update are **plausible mechanisms, not separately timed attribution** in this sweep.
- The **277 GB/s / 47 TF** observations are noted as a **different-node** (`pcl-sprh02`) prior, not a measured ceiling for this run (`pcl-sprh11`), and the AMX number does not calibrate FP32 sparse throughput.

## R6-F2 (Low) — self-identifying aggregate with provenance, spread, precision
- [results/roofline_sweep.json](plugin/validate/results/roofline_sweep.json) now **embeds provenance** (job 384677, node `pcl-sprh11`, source `aaccef6`, config, estimator, nominal-reference peak with the different-node-prior caveat), the **3 process samples + [min,max] spread** per (kernel, M), **unrounded floors** recomputed from the byte/FLOP contracts, and **achieved_pct computed before display rounding** — which corrects top-k M1 from 0.6% to the proper **0.67%**.
- Added a **committed aggregation procedure**, [aggregate_roofline.py](plugin/validate/aggregate_roofline.py) (independently reproduces the published numbers; verified top-k M1 = 0.67%), and made [bench_roofline_sweep.py](plugin/validate/bench_roofline_sweep.py) stamp **per-process provenance** and the launcher use **job-specific output paths** (no more reused filenames). The process spread the reviewer flagged is now preserved (e.g. indexer M32 `[250.0, 294.4]` median 262.2; sparse M64 `[767.4, 784.6]` median 768.5) as observed ranges, not confidence intervals.

## R6-F3 (Low) — launcher repo-resolution hardening
- The inner `git ... rev-parse` is no longer nested inside `cd "$(...)"` (where `pipefail` could not catch its failure). All launchers now **resolve and validate** the repo directory in a separately-checked command (falling back to `SLURM_SUBMIT_DIR`), and **FATAL-exit before any compile/output** if resolution fails ([run_roofline_sweep.sbatch](plugin/validate/run_roofline_sweep.sbatch#L13)).
- Fixed the **class, not the instance**: the same hardening was applied to all four launchers I authored ([run_standalone_opt.sbatch](plugin/validate/run_standalone_opt.sbatch), [run_standalone_opt3.sbatch](plugin/validate/run_standalone_opt3.sbatch), [run_sparse_s1.sbatch](plugin/validate/run_sparse_s1.sbatch)), not just the one flagged.

## Correctness closures (confirmed by the assessment)
- **R5-F1 CLOSED** — non-finite `gpu_out`/`q`/`kv`/`sink` rejected; comparison rejects non-finite reference/candidate/metrics; 19 replay controls through real `main()`.
- **R5-F2 CLOSED** — the coordinate predicate runs on the same built tuple passed to evaluation; the through-`run()` drift probe returns `2/FAIL` with an explicit `builder drift` failure. Stock: 19 replay + 27 F4 controls, 240 evaluations, PARTIAL, promotion BLOCKED.

## Scope / limits (restated honestly)
The sweep is standalone microbench vs a **nominal-reference** roofline; it is **not** an end-to-end or current-target certificate, does not isolate I1's incremental effect (FP32-KV indexer contract only — I1/C1's exactness and speed remain separate evidence), and the regime tags are diagnostics. No binary hash or per-thread affinity was captured. No budget, experiment, dispatch/E2E work, or phase promotion is requested.
