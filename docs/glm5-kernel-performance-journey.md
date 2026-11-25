# GLM-5.3-Flash CPU kernel performance journey

**Emerald Rapids (EMR), batch M = 1–64.** Fusing activation and FP8 quantization cut that isolated kernel's latency by **9.92–83.02×**. The more informative caller-level result is KDA prefill: at M=64 the parallel subchunk reduced fresh prefill from **195.84 to 121.33 ms (1.61×)**, but the same substitution *increased* latency at M=8 and 16. That is why the journey records regressions and tests dispatch rules separately.

**How to read the plots.** A point is the matching CPU reference latency divided by candidate latency on the same node; **1× is parity**, and below 1× is a loss. Ratios use replicated process medians. Cold means a cache flush before each timed call, not measured DRAM traffic. The figures show measured batch sizes only; connecting lines do not qualify intermediate sizes. Axes use log scales and the vertical range varies by panel. These are standalone or named caller measurements on synthetic inputs, **not model speedups, GPU accuracy results or percentages of an accepted roofline**.

[Activation fusion](#activation-and-fp8-quantization) · [FP8 linear](#native-fp8-linear) · [Decode and norm](#kda-decode-and-gated-rmsnorm) · [Prefill](#kda-prefill-from-subchunk-to-caller) · [Eleven-family wave](#eleven-family-native-candidate-wave) · [Roofline comparison](#roofline-comparison) · [Data and reproduction](#data-and-reproduction)

## Activation and FP8 quantization

Jobs **406471 → 406472 → 406473**: fuse clamp, SiLU, multiply, BF16 rounding and block-128 FP8 quantization; compare scalar and vector conversion with serial and parallel loops; then measure a fixed adaptive dispatch at both widths for every integer M. The final curve is a measured policy, not the fastest point cherry-picked from each curve.

![Activation fusion speedups by width, variant and batch, cache-flushed](plots/glm5/activation_cold_v1.png)
<details><summary>Warm-cache measurements</summary>

![Activation fusion, warm](plots/glm5/activation_warm_v1.png)
</details>

## Native FP8 linear

Jobs **406480 → 406483**: keep checkpoint weights packed, compare native checks alone against bounded vector-tile conversion, then vary row and column tile sizes. The `block64` candidate measured **1.89–4.02×** at the four MLP shapes and seven M anchors. The panel labels retain the output dtype and the two non-batch dimensions from the benchmark. The projections and vocabulary head outside this subset are not covered by these timings.

![FP8 linear candidate speedups on four measured MLP shapes, cache-flushed](plots/glm5/linear_cold_v1.png)
<details><summary>Warm-cache measurements</summary>

![FP8 linear, warm](plots/glm5/linear_warm_v1.png)
</details>

## KDA decode and gated RMSNorm

- **406495 → 406497 → 406498, decode:** move convolution, gates and recurrence into native loops while retaining the reference reduction order. Compare serial and parallel loops, then contiguous-state copies against the original commit. Serial copies helped M=1 but lost at large M. The fixed policy used serial at M=1, parallel otherwise, and the native commit at M=1 and M≥32. Across both convolution-state layouts it delivered **6.55–8.70× cold**. That ratio is for the *combined decoder*, not for each constituent kernel.
- **406496 → 406498 → 406518, norm:** compare native serial and parallel preparation, then fuse sigmoid, ordered affine/gate multiplies and BF16 store. The fused candidate beat the frozen reference, but lost to the prior native implementation at **seven cold M values**. Both curves are shown against the same reference. No unconditional replacement was selected.

![Decode and gated RMSNorm journeys with state layout and commit variants, cache-flushed](plots/glm5/decode_norm_cold_v1.png)
<details><summary>Warm-cache measurements</summary>

![Decode and gated RMSNorm, warm](plots/glm5/decode_norm_warm_v1.png)
</details>

## KDA prefill: from subchunk to caller

**407379:** keep the FP32 batched-matmul donor, move the subchunk control flow to C++, and use AVX-512 for TF32 operand rounding. The parallel workers had to inherit inference mode before the full-prefill benchmark could run. The serial path lost at all 18 full-prefill coordinates; parallel won at 14. At M=64, fresh/prefix/ragged speedups were **1.61× / 1.59× / 1.05×**. The `intra_*` panels isolate the subchunk; `fresh`, `prefix` and `ragged` include the rest of the KDA prefill step and its state writes.

![Serial and parallel prefill speedups at isolated and complete KDA boundaries, cache-flushed](plots/glm5/prefill_cold_v1.png)
<details><summary>Warm-cache measurements</summary>

![Prefill isolated and complete boundaries, warm](plots/glm5/prefill_warm_v1.png)
</details>

**407549, held-out policy:** use parallel native code for fused subchunks with M≥32; retain the reference otherwise. Fresh and prefix requests at M=33, 40, 48, 56 and 63 improved **1.17–1.61× cold**. The smaller sizes and ragged requests exercise the fallback; ratios near 1 there do not represent a new kernel win. No serving default changed.

![Held-out whole-prefill dispatch, cache-flushed](plots/glm5/holdout_cold_v1.png)
<details><summary>Warm-cache measurements</summary>

![Held-out whole-prefill dispatch, warm](plots/glm5/holdout_warm_v1.png)
</details>

## Eleven-family native candidate wave

Build **407572**, EMR array **407573**. Each candidate matched the tested valid synthetic outputs and layouts at M=1–64. Performance was sampled at seven anchors, not 64. Some candidates perform fewer checks than the enclosing caller: use these results to select caller experiments, not to bypass validation.

| Plot | Optimization and measured boundary | What the measurement does not establish |
|---|---|---|
| `router` | FP32 projection and top-8; omit the redundant mask for a single expert group | Whole MoE layer latency |
| `combine` | Fuse eight ordered routed contributions, the shared contribution and BF16 store | Grouping, gather or scatter cost; **0.30× at M=8** rules out unconditional dispatch |
| `mixed_norm` | Native four-channel residual mix and normalization; retain unrounded variance | Full mHC pre/post and Sinkhorn latency |
| `absorb` | Native wrapper around BF16 MLA absorption using `bmm` | Expansion or a consistent gain over the existing donor |
| `rotate` | Row-local Hadamard butterflies followed by BF16 rounding and FP8 quantization | Full indexer; losses at M=4 and 8 |
| `compress` | Four-slot gated contraction and rotation/quantization | Mask handling and actual cache writes |
| `tail` | Copy selected history and append the causal tail | Scoring, top-k or physical-page remapping |
| `pack` | Produce the packed 528-byte latent row with group scales | Random reads or masked writes |
| `attend` | Batched score/softmax/value math on already-gathered keys, context 257 | Long-context selection and cache read; little change against the original donor |
| `contract` | Reduce four BF16 residual streams into one output pass | Embedding, final normalization or vocabulary projection |
| `allocate` | Compute page-64 token indices directly | Allocator state transitions and whole runtime latency |

![All eleven native candidate subgraphs versus their CPU references, cache-flushed](plots/glm5/family_candidates_cold_v1.png)
<details><summary>Warm-cache measurements</summary>

![Eleven native candidate subgraphs, warm](plots/glm5/family_candidates_warm_v1.png)
</details>

## From isolated kernels to their callers

Build **407601**, EMR array **407602**. Each caller was checked against its CPU reference for M=1–64; seven batch sizes were timed in three processes. The cache tests also checked that invalid active writes left prior bytes intact and that reads rejected invalid selected rows, not unused rows.

| Caller | Cold speedup range | Interpretation |
|---|---:|---|
| MLA value expansion | 0.95–1.07× | Wrapping the existing batched GEMM did not consistently improve it. |
| Masked latent-KV write | 1.81–6.86× | The pack gain survives active-row validation, packing and committing to the cache. |
| Selected latent-KV read | 1.92–12.60× | Gather and convert only the selected packed rows, validating scales and BF16 outputs. |
| Cached selected attention | 1.99–2.10× | Replacing the cache read alone roughly halved the caller time. The workload cycles through lengths 2047, 2048, 2049, 2175 and padding; it is not the no-prefix BF16 path. |
| Pooled compression and cache write | 0.95–1.07× | The rotation change alone is approximately neutral once mask handling and writes are included. |
| Indexer projection and quantization | 0.93–1.87× | Mixed by batch size; the full projection caller reached 1.87× at M=64. |

![Caller-level speedups including cache state and validation, cache-flushed](plots/glm5/caller_comparisons_cold_v1.png)
<details><summary>Warm-cache measurements</summary>

![Caller-level speedups, warm](plots/glm5/caller_comparisons_warm_v1.png)
</details>

## Roofline comparison

**Unlike the speedup plots, this chart shows absolute latency; lower is better.** Each panel compares the frozen CPU reference and native candidate from array **407573** with a provisional model at the seven measured batch sizes. The solid lines use cache-flushed timings (median of three process medians). These are the eleven isolated operations, not complete families or serving calls.

![Reference and native cold latency in microseconds against a provisional compute/byte model for eleven EMR subgraphs](plots/glm5/family_roofline_cold_v1.png)

**What the dashed line means.** Calibration job **406478** measured large-array triad bandwidth of **184.04 GB/s** and cache-warm library GEMM rates of **3.29 TFLOP/s FP32** and **17.59 TFLOP/s BF16** on EMR. For router, absorption and gathered attention, model time is `max(unique logical bytes / triad bandwidth, GEMM FLOPs / matching dtype rate)`. Count two FLOPs per multiply-add; attention includes both contractions and counts the fixture's shared KV input once. For the other eight panels, the line is **only** useful input/output bytes divided by triad bandwidth, not a compute ceiling. The model excludes temporary traffic, conversion, selection, validation and dispatch. Logical bytes are not DRAM-counter measurements.

**What to investigate first.** Expert combine at M=8 takes **679 µs** native versus **205 µs** baseline, despite a byte model of just **3.56 µs**. At M=64 the same candidate takes **470 µs** versus **11,102 µs** baseline. Test serial versus threaded combine around M=8 before changing dispatch. For MLA absorption at M=64, native takes **6.39 ms**, baseline **6.77 ms**, and the model gives **0.125 ms**. Time the identical strided BF16 batched GEMM without the wrapper, then add validation back; this separates a shape/layout limitation from wrapper cost.

**Qualification still pending.** The calibration was a separate run, not a shape-matched measurement on every timing node. Do not read the gap as recoverable speedup. For credible efficiency percentages we need matched-node, matched-thread streamed and resident measurements: exact-layout GEMM for the contractions; copy/gather, conversion and dispatch measurements for the other paths. Warm timings need a separate cache-resident reference, not the triad line.

[Absolute timings, work counts and model times (CSV)](plots/glm5/roofline_cold.csv) · [Work formulas, source hashes and renderer](../plugin/validate/plot_glm5_roof.py). Run `python3 -B plugin/validate/plot_glm5_roof.py` to render from the CSV; pass the campaign reference-probes directory to regenerate it from the pinned jobs.

## All 64 batch sizes and follow-up callers

EMR array **407619** completed all eleven isolated C++ candidates at **every integer M=1–64**. Each point below compares cache-flushed reference and candidate timings on the same node (three processes, three samples per process). Five candidates had a faster median at all 64 sizes: pooled compression excluding writes, causal tail append, latent pack, four-stream contraction and allocation-index calculation. This is a pointwise median comparison, not confidence intervals or qualification of five complete families. The other six had at least one slower median.

![Eleven subgraphs measured at every integer batch size](plots/glm5/dense_sweep_cold_v1.png)

[Speedups, including losses](plots/glm5/dense_sweep.csv). The roofline section above remains provisional. The new same-fixture component controls (matrix multiply or first-input copy) are not complete-operation ceilings.

**Caller checks.** Job **407631** exercised new no-prefix BF16 attention and pooled scoring, selection and physical-token remapping. Cold median paired speedups were **1.08×** for the tested extend caller and **1.22× / 1.23×** for short/long decode. Job **407632** measured **1.39–1.53×** for four runtime preparation transitions, and **1.16–1.90×** for final contraction/norm/head on tested *nonempty* selected-row sets. Embedding expansion was mixed (**0.73–3.04×**), so universal replacement is not supported. These are synthetic caller tests, not whole-model results.

**Full-width expert and mHC retry.** Job **407630** exposed an empty-batch layout mismatch in mHC pre: an in-place multiply retained the wrong stride. Restoring the reference’s allocating multiply fixed the layout without relaxing the test. In **407634**, three EMR processes matched the synthetic CPU reference’s output bytes and layouts for mHC pre/post at every M=0–64, and for the full-width MoE and joined mHC-pre → MoE → mHC-post caller at M=0, 1, 2, 4, 8, 16, 32 and 64. The two full-width callers were **1.36–1.42× faster cold** across the seven nonempty anchors. This is not an all-M timing result or real-checkpoint accuracy evidence.

## Full-width expert sweep and small-batch fixes

EMR jobs **407638–407641** split the 64 nonempty batch sizes across four nodes. The joined mHC-pre → MoE → mHC-post caller matched reference bytes and layouts and was **1.37–1.47× faster cold at every M**. MoE alone was **1.33–1.46×**, and mHC post **4.87–27.64×**. Isolated mHC pre had a slower median at **8/64** sizes (minimum **0.985×**); the joined result does not justify replacing standalone pre unconditionally. Paired measurements on synthetic full-width W8A8 weights; median of three process medians per path, three cold samples per process. These are caller results, not tokens/s.

![Full-width MoE and mHC callers at all batch sizes](plots/glm5/full_width_callers_cold_v1.png)

[Process medians and job IDs](plots/glm5/full_width_sweep.csv) · [Chart renderer](../plugin/validate/plot_glm5_caller_sweep.py)

**Small-batch dispatch.** Job **407642** kept expert combine serial through M=8 and index rotation serial through 256 input rows. Both beat their matched reference at all 64 sizes (**1.63–26.59×** and **1.29–3.95×** respectively). Compression, which also calls rotation, had one cold loss (**0.992×**); an isolated rotation win does not certify compression. Job **407644** used a serial AVX-512 embedding path through M=16: at M=8 it took **59–65 μs** cold. At vocabulary width 154,880 the head matched output bytes for **2,144** batch/selected-count combinations; timings on boundary cases do not cover every combination.

**Measured EMR calibration, job 407645.** On 16 physical cores in NUMA domain 0, triad reached **184.02 GB/s**. Register-resident BF16/FP32 compute reached **45.57/3.746 TFLOP/s**, versus **12.78/2.991 TFLOP/s** with streamed weights in dense linear. Shape matters: cold M=64 MLA absorption BMM measured **0.884 TFLOP/s** and router logits **0.230 TFLOP/s**. Those last two are rates for the matching contractions, not ceilings for validation, routing or the whole caller. The earlier roofline chart retains its original calibration and caveats.

## Ragged indexer extend and mixed decode

EMR job **407646** compared the opt-in native ragged gather, scoring and token remap with the CPU reference. Output and committed cache/ring bytes matched on ten synthetic caller fixtures, including padding, zero complete pools and tied cutoff scores. The table shows cold speedup as the median of three **paired process ratios**, with their range. It excludes the upstream projection chain.

| Caller fixture | Median cold speedup | Process range |
|---|---:|---:|
| Fresh ragged extend, 64 rows | 1.25× | 1.22–1.29× |
| Extend at cutoff | 1.06× | 1.04–1.06× |
| Mixed decode, 64 rows | 1.58× | 1.57–1.62× |
| Extend, no complete pools | 1.02× | 0.84–1.13× |
| Extend, no active rows | 1.05× | 0.91–1.05× |

The degenerate extend cases did not consistently beat the reference. This is not an all-context speedup or a change to serving dispatch.

**Including projection and quantization.** Job **407647** used frozen layer-3 checkpoint weights and identical synthetic inputs for both paths. Across ten fixtures, projection outputs, selected indices and committed cache/ring bytes matched the CPU reference. Cold speedup for the complete indexer caller was **1.17×, 1.18× and 1.27×** at decode M=7, 8 and 9, and **1.60×** at M=33. Active ragged extend measured **1.06–1.61×**. Singleton decode and extend with no completed pools were approximately neutral. All-padding decode was slower in each process (**0.955–0.992×**, median **0.965×**). These are paired process ratios, not full-layer or serving measurements.

**All-padding decode follow-up.** In EMR job **407659**, the revised opt-in indexer reduced all-padding M=8 cold latency from **5.99 to 4.36 ms** versus the CPU reference. Paired speedup was **1.37×** (three processes: **1.36–1.38×**). With one live row, median speedup ranged from essentially neutral to **2.22×** across the tested fixtures. Active M=8 and M=33 retained **1.17×** and **1.61×**. Projection outputs, selected indices and committed cache/ring bytes matched on ten synthetic decode fixtures. Invalid-shape, dtype and nonfinite-input tests still rejected invalid calls without state changes. This resolves the reported M=8 regression, not every possible batch/context case.

**Standalone mHC pre holdout.** The initial fused candidate in **407664** regressed at M=8 warm and cold. For **407897** the rule was fixed before timing: use the archived native implementation at M=8 and the fused implementation elsewhere. On fresh inputs, no measured median was slower than the frozen CPU reference across **64 batch sizes × two cache conditions**. This closes the measured standalone regression, not the entire family.

## MLA absorption and expansion

Job **407924** retained the strided BF16 batched GEMM and changed the validation around it. The C++ path scans dense BF16 storage for nonfinite exponent bits without allocating an intermediate mask; non-dense views still use the stride-aware check. Compared with the frozen CPU reference, absorption was **1.10–6.19× faster cold**, and expansion **1.41–6.00× faster cold**. Both improved at every M=1–64 across the three-process median timings and matched the reference output bytes and layouts. Tests also checked that nonfinite inputs or results were rejected and unused storage outside a view was ignored. These figures measure the projections including validation, not the complete MLA layer.

![MLA projection speedups across all batch sizes](plots/glm5/mla_checks_cold_v1.png)

[Exact cold speedups (CSV)](plots/glm5/mla_checks.csv)

## Packed FP8 projections: all seven shapes

EMR job **407926** compared the existing optimized kernel (406483) to the frozen native reference (405982) on representative packed checkpoint weights for **all seven FP8 text-graph projection shapes**. Outputs matched in bytes and dtype at every M=1–64. Across 1,792 comparisons (seven shapes × 64 batch sizes × warm/cold × two timed boundaries), no median paired speedup fell below 1×. Cold speedups were **1.40–6.61×** for matrix multiplication alone and **1.27–5.55×** including activation quantization. This closes the FP8 projection **shape sweep**, not serving or the overall kernel phase.

![Cold speedup for packed FP8 projection shapes](plots/glm5/fp8_shapes_cold_v1.png)

[Per-coordinate timings and paired process speedup ranges](plots/glm5/fp8_shapes.csv)

## Native KDA prefill with state updates

In EMR job **407927**, the opt-in native caller matched the frozen CPU caller byte-for-byte on outputs and both convolution and temporal state across **72 synthetic cases**. These include fresh M=1–64, prefixes, padded ragged inputs, and a 64→65 chunk-boundary diagnostic. Four invalid-input tests left both state pools unchanged.

The comparison below is against the **previous selective implementation** (407549), not just the original baseline. With caches flushed, fresh prefill improved **1.32–1.85×** across M=1–64. The two padded ragged cases improved **1.08–1.13×**. Each point is the median of three paired process ratios, with three cold calls per path per process. This measures the complete tested prefill caller, not model throughput.

![Fresh KDA prefill with native state updates versus the previous selective caller](plots/glm5/kda_extend_native_cold_v1.png)

[All 72 cases, including prefixes and ragged inputs (CSV)](plots/glm5/kda_extend_native.csv)

## Complete cached latent attention

Jobs **407929–407931** tested every M=1–64 at context lengths **2047, 2048 and 2049**. The C++ caller validates the selected tokens, gathers packed FP8 keys with the reference BF16 rounding, and shares one FP32 panel between the score and value contractions. Outputs matched reference bytes at all 192 coordinates. Cold speedups were **1.19–1.28× over the earlier optimized cache-read path** and **2.39–2.68× over the frozen CPU reference** (median of three paired process ratios at each coordinate). Additional mixed-context fixtures through length 4099, padding, invalid selection and cache-write ordering were tested; the quoted speedup ranges apply only to the 192 cutoff coordinates.

![Cached attention speedups over both earlier callers around the selection cutoff](plots/glm5/cached_latent_cold_v1.png)

[All 192 paired comparisons (CSV)](plots/glm5/cached_latent.csv). These are synthetic caller timings, not serving throughput.

## Pooled compression including masked cache writes

The earlier rotation-only change did not consistently speed up the full write. Jobs **407933–407934** instead fused the four-slot contraction, rotation, quantization and cache commit in C++. For both scale modes and every M=1–64, the tested masks produced reference-identical bytes; rejected batches did not partially update the cache.

In **407934**, cold mixed-mask writes were **3.17–9.37× faster than the CPU reference** and **2.81–8.86× faster than the earlier native rotation path**. Isolated compression improved **2.54–5.93×** over the 407642 implementation. In the six complete decode-ring cases, the gain over native rotation was **1.07–2.14×**. These are medians of three paired process ratios. One process measured **0.65×** against native rotation for the rounded M=32 return-only case; it was not a median regression.

![Mixed-mask pooled writes at both scale settings](plots/glm5/pooled_commit_cold_v1.png)

[Per-fixture median and process range](plots/glm5/pooled_commit.csv)

## Qualification status through job 407934

Of the **11 families in the original parallel wave**, eight had no measured regression on their named reference comparisons. Three still had losses: standalone MoE routing at seven batch sizes, and two families sharing the projected-indexer caller on sparse or degenerate inputs. Across the **full 17-family inventory**, the corresponding count was **14 with no measured loss, three with a loss**.

These are benchmark outcomes, not completion percentages. Three families had matched component calibration, but none had an achievable roofline matched to its full scope. **Zero families were formally qualified** at this cutoff. The measurements above remain valid for their stated inputs; coverage outside those inputs and whole-caller efficiency still require evidence. Serving defaults are unchanged.

## Matched whole-caller scheduling trials

EMR jobs **407941–407943** compared two executable schedules on the same node at each of **576 cold coordinates**. Narrower FP8 projection tiles beat the existing schedule at **119/448** shape/batch coordinates, but never on MLA output or dense gate/up. Coarser head scheduling beat the existing fresh KDA extend path at only **2/64** sizes. For cached attention at S=2048, dense position lookup won at **63/64**, with a median paired speedup across M of **1.014×**. These are demonstrated alternatives for specific callers, **not optimal-latency or hardware ceilings**. The attention comparison used one cold call per arm per process; projection and KDA used three. None was promoted.

![Alternative schedules versus current native schedules, cold](plots/glm5/schedule_trials_cold_v1.png)

[Per-coordinate paired speedups and median latencies (CSV)](plots/glm5/schedule_trials.csv)

**Standalone router:** job **407939** removed temporary logits-validation masks but lost against the frozen reference at M=14, 25 and 45. Job **407944** also fused the selected-weight gather and index conversion: it beat the reference at **60/64** sizes, losing at M=7, 10, 16 and 17 (worst **0.937×**). The losses moved, so neither run justifies a fixed batch blacklist or an unconditional dispatch change.

## Sparse indexer holdouts

Job **407945** retested the two projected-indexer cases flagged in the qualification review. With fixed C++ validation and masking, extend with no completed pool was **1.13× faster cold** than the frozen caller (paired process range **1.09–1.15×**); decode with only the first row active was **1.10× faster** (**1.05–1.13×**). All three processes favored the candidate in both cases. Active decode gained **1.25× at M=8** and **2.06× at M=33**. Outputs, selection indices and committed cache/ring bytes matched on ten synthetic fixtures.

In first-live decode, the query-weight multiply alone took about **0.50 ms reference versus 0.51 ms candidate**. The speedup was in the complete caller, not a faster matrix multiply. These results resolve the two named holdouts, not a whole-family roofline or a serving change.

## Router decision: retain the existing native kernel

EMR jobs **407951 and 407952** used layer-4 checkpoint weights and replayed activations. At every M=1–64, the alternative matched reference expert IDs, weights and layouts, including tie and correction-bias controls. It did **not** demonstrate a consistent speed advantage: **14 batch-size/cache-condition combinations** were slower than the existing native kernel in both jobs. Against the frozen CPU reference, no loss repeated across jobs, but the holdout minimum was **0.986× warm and 0.989× cold**. Even repeated timings of the identical native kernel varied. Retain the existing kernel; there is no evidence for a batch-size exception list or a change to serving dispatch. This resolves the implementation choice, not whole-family roofline qualification.

## Data and reproduction

**Qualification update through 407952:** 16 of 17 families have no loss on their current named source comparisons (14 previously clean, plus two resolved sparse-indexer holds). The remaining family retains its existing native router; the alternative was not promoted. No family yet has both complete caller coverage and a matched whole-family roofline. The 576 comparisons of feasible schedules across three families do not establish ceilings. Priority caller gaps are prefixed KDA, fresh no-prefix attention and sparse physical remapping.

[Exact plotted ratios (CSV)](plots/glm5/measurements.csv) · [Source artifact names and SHA-256 hashes](plots/glm5/sources.json) · [Evidence exporter](../plugin/validate/export_glm5_journey.py) · [Chart renderer](../plugin/validate/plot_glm5_journey.py)

To render from the portable CSV, run `python -B plugin/validate/plot_glm5_journey.py` from the repository root with `rsvg-convert` available. To refresh the data from the cluster, run `python -B plugin/validate/export_glm5_journey.py <campaign-reference-probes-directory>` first. For new measurements, add the completed paired evidence to the exporter, update the interpretation here, version any changed PNG filenames and run `python plugin/validate/check_doc_images.py`. Baseline-only profiles, failed jobs and queued caller trials do not produce speedup curves.
