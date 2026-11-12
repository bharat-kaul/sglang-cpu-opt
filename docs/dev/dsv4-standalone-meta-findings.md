# Standalone DSv4 follow-up: findings and rejected prototypes

This is an archived experiment record, not a completion certificate or serving promotion.
The campaign starts from public revision `76cbcff6e0bcd81d0320262e5e9f39e723144089`.
No accuracy-worktree or serving-default changes are part of these experiments.

**Disposition at publication (2026-11-11):** retain the gated second prototype
and reproduction tools for research; no further Sinkhorn experiments or serving
integration are scheduled. The top-k scope harness is retained as unsubmitted
exploration, not benchmark evidence. Later work moved to the accuracy-qualified
[serving campaign](https://github.com/bharat-kaul/sglang-cpu-opt/tree/dsv4-perf-main).

The follow-up [full native/serving inventory](dsv4-native-kernel-inventory.md)
incorporates the separate fusion campaign, including superseded hypotheses and
the distinction between native coverage debt and measured performance upside.

## Baseline and implementation inconsistencies

- Job **398241** reproduced the six-kernel timing protocol with three processes.
  At M32, Sinkhorn was 52.50 us versus the published 26.5 us. The source at the
  results-refresh commit `709c0f4` used parallel grain 0; the current source uses
  grain 512 following the serving-overhead fix `2518d0e`. Thus the historical
  table and current default do not describe the same dispatch policy.
- Matched job **398566** confirmed a dispatch contribution: M32 serial 42.179 us
  versus grain-1 23.976 us. This is not a new optimization. The user explicitly
  chose to preserve the serving-safe policy and investigate faster serial math.
- Top-k still lost to its historical Torch comparator at M2 in job 398241:
  33.19 versus 24.64 us. Job 398566 found an existing serial-dispatch opportunity
  (32.944 to 13.124 us), but broader S/K boundaries still need measurement.
- The newer BF16 sparse implementation is prior work, not a campaign gain.
  Job **398250** reproduced its matched-allocation cold comparison: FP32 687.032 us,
  BF16 314.510 us, ratio 2.184x. Historical mixed-precision checks are not a new
  source-conformance or model-accuracy approval.
- The uPP library's nominal DRAM triad uses three 64 MiB buffers, totaling
  192 MiB, which fits the measured EMR socket's 320 MiB LLC. Job **398805**
  instead used the existing 6 GiB triad in `plugin/validate/uarch_probe.py`.
  Its 280.44 GB/s useful-byte result and 7.14 TF/s library FP32 GEMM result
  are turbo-enabled observations, not fixed-frequency or recurrence ceilings.

## Sinkhorn: measured opportunity, first prototype not accepted

Job **398645** found that 19 additional recurrence rounds cost 27.213 us at M32,
out of 42.872 us total. This intervention changes the workload; it is not a
valid optimized result. The one-round result still includes real initialization.
Job 398805 sampled the scalar hc=4 normalization path. Wider vector loops exist
in the binary, but their presence alone does not establish shape-specific use.
Sample shares include startup and other threads, not removable wall-time shares.

The first prototype put one 4x4 matrix in one AVX-512 register, retaining addition
order and full-precision division. Job **399279**, three process medians, cold us:

| M | Immutable serial donor | Prototype | Existing grain-1 | Existing grain-4 |
|---:|---:|---:|---:|---:|
| 32 | 43.123 | 36.565 | 25.591 | 26.811 |
| 64 | 76.217 | 61.026 | 27.477 | 26.664 |

The prototype improved serial execution but lost to the existing parallel paths.
It did **not** meet the campaign's comparative target. Worse, inserting its branch
inside the generic loop also slowed gate-off execution: M32 45.550 us. It is not
eligible for promotion.

All three processes completed 201 native byte-equality cases each. These include
empty/strided inputs, two seeds, sharp/zero inputs, iteration/epsilon variants,
generic hc fallback and grain boundaries. This establishes only the tested
old-native versus candidate relation, not universal or model-level correctness.

Raw records and exact first-prototype source snapshots are retained on the
measurement filesystem under
`/scratch/bkaul/dsv4-standalone-meta-opt/sink-simd-job-399279/`.
Each timing cell is the median of 50 calls per process; the table uses the
median of those three process medians. Source, binary and input hashes are in
each replica record; the initial producer is preserved in `sources/`.

## Interleaving result and remaining gap

The original generic function body is restored. Experimental builds alone
(`SINK_SIMD_EXPERIMENT`) expose `mhc_sinkhorn_simd4`; ordinary builds and callers
continue to use `mhc_sinkhorn`. The temporary first-prototype `SINK_SIMD4`
environment switch is no longer used.

The second experiment reused donor initialization verbatim, then interleaved 1, 4
or 8 independent matrices through the recurrence on the same thread. The final
combination output doubles as a small staging buffer; no extra full tensor is
allocated.

Job **399421**, median of three process medians, cold microseconds:

| M | Original serial | Best tested prior dispatch | Eight-stream candidate |
|---:|---:|---:|---:|
| 1 | 12.051 | 12.051 | 14.194 |
| 2 | 13.131 | 13.131 | 14.966 |
| 4 | 14.787 | 14.787 | 16.490 |
| 8 | 20.111 | 20.111 | 18.736 |
| 16 | 27.116 | 25.313 | 20.709 |
| 32 | 44.014 | 26.367 | 26.757 |
| 64 | 76.457 | 27.667 | 36.809 |

M16 is a material, shape-specific result: **1.222x** versus the best tested old
dispatch. Candidate process medians span 20.507--20.747 us; old grain-1 spans
24.862--25.327 us. This does not establish a universal improvement: M1/M2 regress,
M32 is effectively tied under the campaign's <5% convention, and M64 loses.
All three processes again completed the 201 native byte-equality cases. Each
timing cell has 56 samples with balanced randomized arm position.
Exact sources and raw records are retained under
`/scratch/bkaul/dsv4-standalone-meta-opt/sink-simd-job-399421/`.

The original function body was verified textually identical to the pinned donor.
Gate-off cold process medians no longer show the first prototype's >5% regression
across the tested M sweep. This is measured behavior, not a proof over all inputs
or runtime configurations.

## Constant-shape initialization: rejected as below the materiality bar

Job **399443** tested a checked, pinned-source-generated hc=4 initialization clone
alongside the same eight-stream recurrence. Scalar exp and all math were unchanged.
Three processes again passed 201 native byte-equality cases; timing used 64
balanced samples per arm/cache condition.

| M | Eight-stream, generic init (us) | Specialized init (us) | Incremental speedup |
|---:|---:|---:|---:|
| 16 | 19.9455 | 19.7050 | 1.012x |
| 32 | 26.0760 | 25.0600 | 1.041x |
| 64 | 36.2985 | 34.8045 | 1.043x |

These cold improvements are below 5%, the user's noise convention. The added
generation/linking complexity is not justified; the specialization was removed,
returning to the measured second-prototype source. Its exact source snapshots,
generated donor and raw samples remain under
`/scratch/bkaul/dsv4-standalone-meta-opt/sink-simd-job-399443/`.

The M16 interleaving result repeated: generic-init eight-stream 19.9455 us versus
best old dispatch 24.8860 us in this job. The historical published M16 number is
23.3 us; comparing that historical number to a new job is contextual, not a
controlled causal comparison. No unconditional default promotion is justified:
small-M regressions and the M64 loss versus parallel dispatch remain open.
