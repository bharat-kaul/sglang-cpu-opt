# Phase-1 Kernel Implementation Review: Round 3

Reviewer: GitHub Copilot. Implementation reviewed at `cca28324d9eaef0867b95d0d6a2545a05e8ff42c`, with the concurrent delta through `453da98d48fb5286e26c543377adfc6414c72093` checked before publication. Review source continuity is established by `git diff cca2832..453da98`.

Scope: all six authored kernels, M=1/8/16/32/64, the three compressor shapes, correctness enforcement, existing performance evidence, nominal roofline interpretation, and remaining optimization opportunities. This follows [round 2](phase1_kernel_implementation_review_round2.md) and assesses the current [author checkpoint](phase1_kernel_review.md) and [optimization queue](../results/kernel_opt_queue.json). No production, policy, author-record, or automation edits were made for this review. No new cluster, GPU, full-model, or performance runs were launched.

## Verdict

**The implementations show real, scoped progress; Phase 1 is still PARTIAL, not ready for unconditional correctness or performance sign-off.** The original indexer arithmetic and dispatch defects remain closed. All five unchanged non-sparse implementations passed the scoped native comparisons below. Sparse now has a much better blockwise reference, and current C++ best-of replay against saved GPU outputs is repeatable at every requested M.

Remaining blockers are concrete: invalid reference evidence can survive F4, the required case inventory is incomplete, sparse composition is not tested against the published consumer semantics, numerical budgets remain unratified, and the performance table is not the advertised three-process aggregate. No new defect in the five unchanged non-sparse native sources was found. Absence of a defect in these tests is not a full caller or model certificate.

The concurrent `453da98` commit corrects the indexer M8 statistical-parity claim and adds a separate promotion gate. It does not change the six kernels, F4, sparse reference, or timing harness. Its broader playbook/promotion behavior is outside this kernel review; findings below do not claim that no executable gates exist.

## Findings

### R3-F1 - High: F4 still accepts malformed reference evidence

The [tuple branch](../f4_acceptance.py#L402) checks candidate arity but then zips the candidate with an unchecked reference. Full `run()` fault injections returned `0/PARTIAL` for an empty Sinkhorn reference and for three valid reference outputs followed by an extra infinite tensor. A scalar combine reference also broadcast silently. The [standalone selection branch](../f4_acceptance.py#L346) accepted all-NaN input/reference logits with distinct candidate indices.

These are evidence-validation defects, not observations that ordinary native outputs are wrong. Reference type, arity, per-output shape/dtype, and allowed finite/mask domain must be validated before comparisons. Existing reason-checked self-tests catch several earlier defects but do not cover these controls.

The [repeat checks](../f4_acceptance.py#L333) also accepted equal-valued FP32-first/FP64-second outputs for combine, Sinkhorn, and composition. `torch.equal` can report equality across dtype; each invocation needs its own output-contract check. Current [output contracts](../f4_acceptance.py#L272) do not encode device/layout requirements. Keep these metadata failures separate from numerical screening.

### R3-F2 - High: coverage and composition do not enforce the requested qualification scope

The [inventory gate](../f4_acceptance.py#L434) requires op/kind presence and indexer M labels, not the full required coordinates. Removing all BF16-KV cases, the masked compressor case, or heavy-tailed indexer M64 individually still returned `0/PARTIAL`. Removing an entire op or all indexer M8 cases correctly failed. This is partial enforcement, not an absent gate.

The persistent [manifest](../f4_acceptance.py#L190) has BF16-KV at M1/32, top-k at M1/8/32, compressor at M8, and Sinkhorn/combine at M1/8. All four sparse functions are directly screened only at M8/H64/K512 with Gaussian inputs. Thus the production best-of M1 branch is absent from direct sparse F4 coverage. Required path, M, shape, distribution, mask/layout, and relevant caller-state coordinates need independently declared identities; benchmark rows and this review's temporary probes cannot substitute for that gate.

The [composed check](../f4_acceptance.py#L298) uses scalar FP32 attention and a global FP32 reference, gathering candidate/reference selections in their respective orders. It therefore tests FP32 mathematical composition, not the published 64-block BF16 consumer. Top-k membership equality does not ensure identical block grouping or intermediate BF16 rounding. Direct source-boundary screening does not close this composed-path gap.

### R3-F3 - High: sparse approximation remains unqualified for serving

The principal boundaries in [the blockwise replica](../sparse_ref.py#L33) now match the pinned source for valid gathered rows: BF16 operands; FP32 score accumulation; scaling before the running maximum; 64-entry blocks; BF16 casting of each block's unnormalized exponentials before the value GEMM; FP32 running rescaling; sink after the loop; BF16 output. The published source uses `T.exp`, not explicit exp2/log2 scaling. Generated GPU lowering and identical reduction/FMA order were not established.

The [native implementations](../../kernels/dsa_pilot/sparse_attend.cpp#L13) still implement different arithmetic: FP32 scalar/BMM paths omit blockwise BF16 exponential rounding; the experimental AMX path also rounds GEMM outputs and normalized weights. GPU equality is not mandatory without such a policy, but these differences require an independently justified error budget and scoped qualification. Neither closeness nor the failure of naive BF16 BMM proves that the AMX design is faithful or exhausted.

Job 384532 checks GPU versus replica, not C++ on the expanded cases. The [oracle script](../test_sparse_gpu_oracle.py#L75) saves only its original three records before expanded coverage. B4/K128, K160, K640 and K512-with-64-trailing-sentinels results are printed but their inputs/outputs are not persisted. Removing 64 trailing sentinels preserves valid block boundaries; it does not test interspersed sentinels, all-invalid rows, offsets, or cache/state behavior.

This review adds a bounded C++ replay below, but only for the saved shared-pool K512 cases. The uniform observed maximum error is a sample result, not an intrinsic hardware-noise floor, production tolerance, or full-model acceptance certificate.

### R3-F4 - Medium: headline timing aggregation and comparison claims need correction

The [headline table](phase1_kernel_review.md#L51) reproduces process repeat 1 of job 384531, not an aggregate of all three repeats. The complete reconstruction appears below. At indexer M8 the process speedup medians are 0.98/1.05/0.96, with median 0.98 and overall within-trial range 0.95-1.07. The concurrent commit correctly stops calling this statistical parity. Neither a hard >=1 floor nor equivalence has been established; any noninferiority margin or waiver needs explicit authorization.

The [timing references](../bench_replicated.py#L53) must be described precisely. Indexer uses the corrected BF16 arithmetic stages, but Torch returns BF16 storage while C++ returns FP32. Sparse uses the explicitly diagnostic global-FP32 fallback, not the published blockwise BF16 contract. Combine uses FP32 einsum, not necessarily the published multiply/sum/final-cast arithmetic boundaries. These comparisons remain useful local fallback measurements; they are not complete same-boundary serving qualification.

Comparator drift is material. Current indexer M16 Torch medians are 310.3/922.7/261.9 us versus C++ 205.4/219.1/237.4 us. The older job 384526 used different FP32-stage arithmetic and nonnegative weights. Sparse M64 Torch was about 804 us in job 384476 and is now 2690.8/2858.9/2730.1 us, while C++ changed from about 749 us to a current median 729.1 us. Do not attribute the larger ratio to a new native optimization. Five native sources are unchanged since round 2; sparse changed only its finite-scale guard.

### R3-F5 - Medium: timing and replay identities remain incomplete

Job 384531 records node, clean source revision `1e3096d77ca72eb3933c3c11b3003e8e9f325e09`, Torch 2.12.0+cpu, and OMP settings. The [launcher](../run_replicated_all.sbatch#L14) prints only six affinity CPUs from a separate process before `numactl`. Effective Torch/MKL thread counts, worker placement, loaded-binary/library/compiler identities, input hashes, clocks, and residency are not recorded. Source continuity through the current reviewed revision is supported; complete binary/runtime binding is not.

The [CPU replay script](../test_sparse_cpu_vs_gpu.py#L32) now explicitly uses unrestricted `weights_only=False`, accepts legacy records, merely prints provenance, and compares unrounded FP32 output against BF16 GPU output. For a tensor archive this unnecessarily permits pickle execution if the file is replaced or untrusted. Enforce the expected provenance and output boundary with a restricted load. This review did so without modifying the script, using `weights_only=True` and an explicit allowlist for the installed `TorchVersion` metadata type.

The timing harness has no accuracy, finiteness, repeatability, or no-regression assertions. It silently omits Sinkhorn if its import fails, although all six kernels are present in this log. Separate correctness validation is reasonable only when the exact tested source/build/case scope is bound to the timing evidence.

### R3-F6 - Medium: acceptance sequencing and current-record reconciliation remain unresolved

The [budget proposal](../results/kernel_opt_queue.json#L7) selects the largest epsilon preserving dummy-model argmax/KL on a set called held-out. Using that set to choose epsilon makes it tuning data. Dummy-weight sensitivity, unspecified perturbation direction/correlation, and observed output error alone do not establish a production budget. Reserve untouched validation data and distinguish dummy structural validation from real-weight sensitivity/task validation.

There is also a dependency cycle: Phase-1 sign-off requires budgets, while their proposed derivation is deferred to Phase 2. Explicitly authorize a bounded qualification experiment before sign-off, or explicitly amend the phase dependency. Do not silently promote a proxy or ratify a tolerance.

Historical tables should remain historical. However, the queue's reconciliation cites failed job 384529 for replicated evidence, current tables are mixed with older ratios, and sparse GLOBAL/pending language survives after the blockwise reference was implemented. F4 also [emits a GLOBAL-reference label](../f4_acceptance.py#L380) while executing the blockwise adapter. Existing exact/irreducible/at-limit/done claims need clear scope and status, not an implicit overwrite of baseline evidence. Deferring a low-ROI experiment is valid; claiming the design space is exhausted is different.

## Correctness Evidence

### Five non-sparse kernels

The five sources are unchanged from `1124abf`. The read-only native review performed 86 scoped comparisons, counting Sinkhorn per output, plus 20 full-F4 interface/coverage probes. Native repeatability checks passed. Tests used the prescribed interpreter `/scratch/bkaul/venvs/sglang-cpu/bin/python`, one thread and compiler flags `-O3 -fopenmp -march=native`, except four lightweight top-k branch cases at four threads. No timings from these login-node tests are used as performance evidence.

| Kernel | Current scoped checks | Result |
|---|---|---|
| Indexer logits | 17 evaluations: dispatch/tiled/BF16-KV at all five M; fresh signed seed-1 FP4-grid inputs; strided S1031 at M1/8 | Exact against BF16-stage oracle; induced selection preserved |
| Top-k | 14 evaluations: all five M, ties, k0, k=S, clamping, empty sequence; masked/strided serial/chunk/row cases | Selection checks passed |
| Compressor | 30 comparisons: all five M x (R,D)=(128,512),(8,512),(8,128) x masked/unmasked | Maximum absolute error 8.35e-7 |
| Sinkhorn | 15 output comparisons at all five M against existing SGLang oracle | Maximum absolute error 2.39e-7 |
| Combine | 10 comparisons at all five M with strided H4096/H1031 inputs | Maximum absolute error 0 against existing local oracle |

Indexer grid replay used fresh operands, not a claimed reconstruction of the exact historical counterexample. Combine's zero local-oracle error does not establish bit-exactness to the published multiply/sum/cast path. Continuous comparisons are screening under unratified budgets. All-masked pooling is not qualified by masked-but-valid tests. These are synthetic operands, not captured tensors from a real full-model forward; shape/caller provenance remains a separate gate.

The previously verified 13 reason-checked F4 self-tests were not repeated in this kernel-review pass. Original malformed indexer input controls were not repeated because their implementation is unchanged; their prior scoped closure is retained.

### Sparse replay against saved GPU outputs

The parent reviewer loaded the current archive with restricted deserialization and required job `384532`, validated-source-hash metadata, exact kernel SHA-256 `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`, and exactly the N1/8/64 record inventory. This binds the declared saved provenance, not a separately signed/content-hashed archive. Q/KV were BF16-rounded on both sides; the best-of C++ output was additionally BF16-rounded for the primary comparison. Inputs are synthetic GPU-oracle cases, not real-model activation captures.

| M | GPU-output origin | FP32 C++ output vs GPU max abs | BF16 C++ output vs GPU max abs | BF16 C++ vs blockwise replica max abs | Replica vs GPU max abs |
|---:|---|---:|---:|---:|---:|
| 1 | Saved N1 | 0.001340299845 | 0.001953125 | 0.001953125 | 0.0009765625 |
| 8 | Saved N8 | 0.001284569502 | 0.001953125 | 0.001953125 | 0.001953125 |
| 16 | First 16 rows of saved N64 | 0.001221060753 | 0.001953125 | 0.001953125 | 0.0009765625 |
| 32 | First 32 rows of saved N64 | 0.001229643822 | 0.001953125 | 0.001953125 | 0.0009765625 |
| 64 | Saved N64 | 0.001458525658 | 0.001953125 | 0.001953125 | 0.001953125 |

All five candidate outputs were finite and exactly repeatable at one thread. All 12 checks across scalar/BMM/AMX/best-of entry points and NaN/+inf/-inf scale rejected with the expected finite-scale reason. Finite nonpositive scale remains an intentional default-scale sentinel. This closes the specific nonfinite-scale defect, not every sparse input-domain question. The experimental AMX numerical path was not replayed across all M here.

Independent source inspection used HF revision `60d8d70770c6776ff598c94bb586a859a38244f1`, [inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py). The independently fetched source hash matches the archive's validated hash.

Expanded GPU-versus-replica results in job 384532: B4/K128 and K160 max error 0.001953; K640 0.0009766; K512 with 64 trailing sentinels 0.001953. They are not expanded C++ results and cannot be replayed from the saved three-record archive.

## Reconstructed Performance

Source: `/scratch/bkaul/replicated_all_384531.log`, process blocks at lines 5-34, 38-67, and 71-100. Node: `pcl-sprh08.sc.intel.com`; OMP 64/close. These are existing EMR measurements, not final target-GNR results.

Each trial value is a mean of 30 calls. Each process reports the median of five trial means; the table below takes the median of the three process summaries. Speedup is the median of the three process-level medians of paired reference/C++ ratios, NOT the ratio of the aggregated latencies or a pooled 15-trial median. Brackets span the three process summaries. Reconstruction uses printed, rounded values.

The [timing implementation](../bench_replicated.py#L32) makes one initial warm call per implementation and another before every timed block: six warm calls plus 150 timed calls per implementation/coordinate/process. Trial order alternates C/R, R/C, C/R, R/C, C/R; it is unbalanced 3:2 and repeats identically across processes. Raw trial pairs are not persisted. Sequential same-node repeats with seed 0 and reused buffers do not sample independent input distributions or cold-state performance.

| Kernel | M | C++ us median [min,max] | Torch us median | Speedup median [min,max] |
|---|---:|---:|---:|---:|
| Indexer logits | 1 | 102.1 [97.2,103.7] | 1169.7 | 11.52 [11.29,11.98] |
| Indexer logits | 8 | 202.5 [192.8,203.0] | 199.0 | 0.98 [0.96,1.05] |
| Indexer logits | 16 | 219.1 [205.4,237.4] | 310.3 | 1.49 [1.10,4.08] |
| Indexer logits | 32 | 268.2 [266.6,273.1] | 838.9 | 3.06 [2.63,3.20] |
| Indexer logits | 64 | 427.7 [404.2,430.3] | 715.3 | 1.75 [1.66,1.77] |
| Sparse best-of | 1 | 73.7 [72.1,74.3] | 258.5 | 3.54 [3.26,3.58] |
| Sparse best-of | 8 | 167.2 [144.2,169.5] | 375.9 | 2.24 [2.22,2.29] |
| Sparse best-of | 16 | 256.3 [234.6,258.2] | 457.2 | 1.78 [1.74,1.82] |
| Sparse best-of | 32 | 392.1 [391.3,402.2] | 555.4 | 1.42 [1.35,1.42] |
| Sparse best-of | 64 | 729.1 [726.9,732.7] | 2730.1 | 3.72 [3.56,3.85] |
| Compressor R128/D512 | 1 | 26.2 [25.7,26.3] | 216.7 | 8.26 [8.09,8.45] |
| Compressor R128/D512 | 8 | 77.8 [76.8,77.9] | 212.1 | 2.75 [2.71,2.91] |
| Compressor R128/D512 | 16 | 138.2 [137.1,139.6] | 298.2 | 2.16 [2.11,2.19] |
| Compressor R128/D512 | 32 | 261.8 [261.3,262.8] | 398.4 | 1.47 [1.46,1.48] |
| Compressor R128/D512 | 64 | 562.1 [508.0,562.8] | 1563.9 | 2.79 [2.64,2.95] |
| Indexer top-k | 1 | 2.6 [2.6,2.8] | 5.5 | 2.08 [2.04,2.10] |
| Indexer top-k | 8 | 11.9 [11.9,12.3] | 55.3 | 4.67 [4.60,4.69] |
| Indexer top-k | 16 | 12.1 [11.8,13.4] | 167.8 | 13.26 [11.76,14.27] |
| Indexer top-k | 32 | 12.9 [12.6,13.3] | 325.4 | 25.79 [24.88,25.86] |
| Indexer top-k | 64 | 13.6 [13.6,13.8] | 398.2 | 28.80 [28.53,29.23] |
| Sinkhorn | 1 | 2.4 [2.4,2.4] | 225.7 | 94.41 [93.68,96.07] |
| Sinkhorn | 8 | 14.0 [13.5,14.3] | 281.8 | 19.36 [17.73,20.19] |
| Sinkhorn | 16 | 14.2 [13.5,14.8] | 291.7 | 19.73 [18.76,21.48] |
| Sinkhorn | 32 | 14.0 [13.6,15.1] | 319.6 | 21.05 [20.99,23.56] |
| Sinkhorn | 64 | 15.5 [15.3,15.5] | 362.6 | 21.59 [21.51,23.01] |
| Combine | 1 | 2.3 [2.3,2.3] | 11.7 | 5.12 [4.76,5.18] |
| Combine | 8 | 11.5 [11.3,11.7] | 27.1 | 2.32 [2.25,2.39] |
| Combine | 16 | 11.5 [11.4,11.6] | 26.9 | 2.34 [2.30,2.35] |
| Combine | 32 | 11.8 [11.5,11.8] | 26.4 | 2.24 [2.24,2.36] |
| Combine | 64 | 13.6 [13.4,13.7] | 28.3 | 2.07 [2.03,2.12] |

Job 384531 has no R8 compressor timings. Historical job 384474 C++ vectors, ordered M1/8/16/32/64, are R8/D512 = 16/20/25/28/42 us and R8/D128 = 13/14/14/16/20 us. They are retained historical evidence, not current replicated qualification. New timing also lacks BF16-KV, TP-head variants, sparse union K, mask/layout/state variants, and meaningful chunked top-k selection: S1024/k512 exercises serial/row paths in this sweep.

## Roofline Interpretation

Nominal [EMR constants](../platforms/emr.json): 358.4 GB/s, BF16 AMX 124.5184 TF/s, FP32 AVX512 7.7824 TF/s. Actual benchmark I/O is FP32 except top-k INT64 output. The following are useful-byte/compute diagnostics, not measured achievable latency or a promised optimization factor. Exponentials, divisions, selection, recurrence, packing, cache traffic, and dispatch need their own measured ceilings.

| Kernel | Useful boundary bytes/call | Ideal us at M1/8/16/32/64 |
|---|---|---|
| Indexer logits | 561408M | 1.566 / 12.531 / 25.063 / 50.126 / 100.251 |
| Sparse FP32 | 1310720M+256 | 8.623 / 68.985 / 137.971 / 275.941 / 551.882 |
| Compressor R128/D512 | 526336M+262144 | 2.200 / 12.480 / 24.229 / 47.726 / 94.720 |
| Compressor R8/D512 | 34816M+16384 | 0.143 / 0.823 / 1.600 / 3.154 / 6.263 |
| Compressor R8/D128 | 8704M+4096 | 0.036 / 0.206 / 0.400 / 0.789 / 1.566 |
| Top-k, byte-only | 8192M | 0.023 / 0.183 / 0.366 / 0.731 / 1.463 |
| Sinkhorn, byte-only | 192M+108 | 0.00084 / 0.00459 / 0.00887 / 0.01744 / 0.03459 |
| Combine | 81936M | 0.229 / 1.829 / 3.658 / 7.316 / 14.631 |

Indexer GEMM leading work is 16777216M FLOPs at BF16 peak, plus FP32 epilogue work. Sparse's two FP32 GEMMs total 67108864M leading FLOPs. The resulting sparse useful-compute utilization is 11.7/41.3/53.8/70.4/75.7%, not uniform saturation across M. Combine M64 implies 385.6 GB/s useful throughput, above nominal DRAM BW: cached/reused operands mean this is not evidence of DRAM saturation. Tiny Sinkhorn/top-k byte floors do not describe their achievable instruction/dispatch ceiling.

Actual traffic exceeds unique boundary bytes: indexer packing/conversion and its M1 256 KiB score intermediate; sparse repeated KV reads and M64 8 MiB scores; pooling's shared APE/running state; top-k scratch. The level of the cache hierarchy serving those accesses matters. Direct BF16-KV changes indexer boundary bytes to 299264M, but is absent from this replicated run. Do not mix that hypothetical ceiling with FP32-I/O measurements.

## Remaining Optimization Work

First stabilize the instrument: preserve raw paired blocks and full identities; establish/check threads and affinity before warmup; balance starting order across processes; distinguish reused-buffer from rotating-buffer conditions; investigate indexer M16 and sparse M64 comparator drift. No new dispatch decision should be justified by the unstable ratio alone.

| Kernel | Implemented | Remaining discriminating work | Priority rationale |
|---|---|---|---|
| Indexer | BF16-stage epilogue, parallel query packing, tile-local KV conversion, M1 whole-S path, BF16-KV input | Same-binary conditional unused Abuf allocation; conformed M1 whole-S vs tiled; packing/conversion/epilogue attribution; BF16-KV timing | Largest conditional measured non-sparse cost; M8 floor unresolved |
| Sparse | Scalar M1; FP32 BMM plus fused softmax for larger M; experimental naive BF16 BMM | Source-boundary-correct 64-block BF16-input/FP32-output library GEMM; actual K/head/mask/caller qualification before speed claims | Numerical contract first; rejected naive BF16 BMM does not reject the donor design |
| Compressor | N x D tiling with stable online pooling | All three shapes under replicated timing; vector-exp or stable multipass A/B preserving masks | Two exp calls per recurrence step remain; R8 evidence is stale |
| Top-k | Serial M1 and guarded row/chunk dispatch | Captured S/k dispatch cases; scratch reuse | Small conditional total cost supports bounded deferral, not exhaustion |
| Sinkhorn | Fused per-row recurrence | Serial-threshold measurement; preserve epsilon/iteration semantics | Dispatch dominates tiny M; dense GEMM peak is not its ceiling |
| Combine | 1024-element accumulate-once tiles | Published multiply/sum/cast parity; serial-threshold measurement; only then consider fusion | Cache/dispatch behavior, not proven DRAM saturation |

Using the existing [conditional call-count model](../dsv4_roofline_p2.py#L300), standalone median latency x calls gives the following ms per batched decode step, ordered M1/8/16/32/64: indexer x21 = 2.144/4.253/4.601/5.632/8.982; top-k x21 = 0.055/0.250/0.254/0.271/0.286; Sinkhorn x86 = 0.206/1.204/1.221/1.204/1.333; combine x86 = 0.198/0.989/0.989/1.015/1.170; R128 pooling x20/128 = 0.004/0.012/0.022/0.041/0.088. R8 pools each have 21/4 conditional calls but no current replicated timings.

These are prioritization estimates, not measured engine cost, realizable savings, or model-wide shares. K512 pre-gathered sparse attention is not the full varying-K serving path (including K128/640/160 unions at context4096), so multiplying it across all layers would misstate the aggregate. MHC threshold/fusion work may reasonably be deferred given combined conditional cost 0.404-2.503 ms/step; document that bounded decision instead of saying irreducible.

## Closure and Exit Conditions

| Prior issue | Current disposition |
|---|---|
| Original P1-F1 indexer stages/selection | Scoped closure retained and reinforced by current all-M comparisons |
| Original P1-F2 indexer guards/empty batch | Prior scoped closure retained; unchanged implementation |
| R2-F1 F4 hard checks/inventory | Partially repaired; R3-F1/F2 remain |
| R2-F2 sparse reference/boundaries | Principal direct-reference boundaries repaired; serving composition, coverage, and budget remain |
| R2-F3 performance evidence | Replicated run is useful; aggregate/contract/identity/no-regression issues remain |
| R2-F4 GPU provenance/replay | Wrapper/source-hash improvements demonstrated; restricted, bound production replay remains open |
| R2-F5 nonfinite scale | Closed for NaN/+inf/-inf across all four public paths by 12 reason-checked probes |
| R2-F6 reconciliation/claim strength | Improved again at 453da98, still partial |

Before sign-off: close reference/repeat-output validation holes; enforce the exact required case inventory; qualify the actual sparse composition/caller boundary; resolve budget authorization and phase sequencing; correct current timing aggregates and evidence identities; establish or explicitly resolve the M8 no-regression requirement. Preserve historical baseline sweeps. Full-model deterministic-dummy correctness, real-weight/dtype/task validation, captured shape provenance, and target-hardware performance remain authoritative later gates, not consequences of synthetic local tests.