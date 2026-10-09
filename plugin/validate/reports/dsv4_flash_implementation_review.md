# DSv4 Flash: Implementation, Provenance, ROI and Playbook Review

Date: 2026-10-09. Reviewed revision: `4b52728a02c62862bc2a259b4b8cbcd88755405b`.

**Verdict: CHANGES REQUIRED. Implementation correctness and the claim that worthwhile optimization opportunities are exhausted do not pass.** The [seventh roofline review](dsv4_flash_roofline_gate_review7.md) accepted a partial ideal model with unverified observations. The subsequently added measured join has new measurement and contract defects; that acceptance does not certify these additions. Previously closed main-inventory accounting issues are not reopened wholesale.

## Findings

### F1 High: Compressor Emits NaNs for Valid Overlap Padding

[compressor.cpp](../../kernels/dsa_pilot/compressor.cpp) initializes the running maximum to `-INFINITY`, then evaluates `exp(m - mnew)` and `exp(x - mnew)` unconditionally. When the first score is `-inf`, both involve `-inf - -inf`; NaNs poison the running sum and accumulator even when subsequent positions are valid.

This is a published-model case, not merely malformed input. At the first ratio-4 overlap window, `Compressor.overlap_transform(score, -inf)` supplies four leading masked positions followed by four valid positions. Excluding construction of the overlap buffer from the kernel does not exclude consuming its valid output.

**Executed C++ reproduction:** `kv=[0,0,0,0,1,1,1,1]`, `score=[-inf,-inf,-inf,-inf,0,0,0,0]`, broadcast over channels; zero APE. At both `[N,R,D]=[1,8,128]` and `[1,8,512]`, Torch returns finite all-ones output; the actual extension returns 128 and 512 NaNs respectively. Normal/heavy-tailed finite tests do not cover masking.

**Required:** handle leading masked entries without an indeterminate rescale, define the all-masked contract, and test first-overlap, partial-window and decode-boundary states against the published compressor. This blocks the blanket FAITHFUL/PASS disposition for the pool stage.

### F2 High: Indexer Dispatch Changes Numerics and Top-k Selection

[indexer_logits.cpp](../../kernels/dsa_pilot/indexer_logits.cpp) silently casts FP32 public inputs to BF16. Its small-batch `bmm` path additionally stores scores in BF16, while its tiled `brgemm` path retains FP32 scores before the weighted reduction. The dispatcher therefore changes numerical behavior when the same query is batched differently.

**Executed C++ evidence:** using FP4-representable query/key levels, signed BF16-exact weights, H=64/D=128/S=1024, and seed 1, replicating the same request from M=1 to M=8 gives maximum logit difference `0.044673919677734375`, FP64-measured cosine `0.9999982451604809`, and only **511/512** selected-index agreement. Against the local FP32 oracle, maximum error is about `0.044674` for M=1 versus `1.91e-6` for M=8. Thus a very high cosine does not establish selection equivalence, even on the model's quantization grid.

A separate FP32 public-contract probe with two nearly equal query heads and weights +1/-1 returns zero instead of approximately -0.127945 in both paths because the input conversion erases the difference. This second probe is not claimed to be a captured QAT activation; it demonstrates the undocumented approximation in the advertised FP32 boundary.

[bench_idx_logits.py](../bench_idx_logits.py) samples weights using `torch.rand`, whereas the published `weights_proj` is a linear projection with no nonnegativity constraint. It also checks logits independently from top-k. The separate top-k benchmark does not exercise indexer-induced selection changes.

**Required:** define the authoritative input, accumulation and output precision contract; reconcile it with upstream QAT/rounding; test signed/cancelling inputs and composed logits-to-top-k, including the M=7/8 dispatch boundary. Preserve one numerical contract across variants, or explicitly validate any allowed approximation against the authoritative target. These synthetic counterexamples demonstrate a contract risk, not a measured full-model accuracy loss.

### F3 High: Sparse-attention Timings Are Joined from the Wrong Variant

[perf_sweep.json](../results/perf_sweep.json) labels the sparse row `scalar (kept)` but copies the **Torch reference** column from `/scratch/bkaul/dsa_perf_sweep_384372.log`. Independent extraction by header name gives:

| M | Published / actual Torch ref (ms) | Actual scalar median (ms) | Actual AMX median (ms) |
|---|---:|---:|---:|
| 1 | 0.245 | 0.069 | 3.542 |
| 8 | 0.324 | 0.463 | 0.421 |
| 16 | 0.395 | 0.899 | 0.513 |
| 32 | 0.555 | 1.786 | 0.650 |
| 64 | 0.836 | 3.651 | 1.956 |

All five other operations' stored medians match their raw `cpp_ms` columns. This is a localized but substantive error: the published sparse off-ceiling values describe a different implementation. It also hides the scalar variant's approximately 3.55x M=1 advantage over Torch; the claim that neither authored variant wins at any real-shape point is too broad.

**Required:** derive measurements from named fields with an explicit variant key, retain all replicas and reference times, and validate the derived JSON against the raw record before rendering. Preserve a durable raw artifact/hash and the run's source/build identity; a mutable scratch pathname and free-text run description are insufficient for reproducible publication.

### F4 High: Correctness Re-verification Does Not Fail Closed

The six [benchmarks](../bench_idx_logits.py) print cosine/set-match but do not assert acceptance or fail on nonfinite output. The [batch runner](../run_dsa_perf_sweep.sbatch) uses `set -x`, logs each exit status, continues, and ends with a successful `echo`; a failed subprocess does not make the job fail.

**Executed checker probe:** replaced the indexer extension with an all-zero result and disabled only timing loops in an in-memory AST copy of the benchmark. It printed cosine `0.000000` at every M and completed normally. No source file was changed by the probe.

[impl_review.json](../results/impl_review.json) also says `0.999963 (< 0.9999) FAIL`. The inequality is false. Every displayed AMX cosine in the raw sweep exceeds its stated 0.9999 threshold. If a stricter tolerance was intended, it must be declared and evaluated, not inferred after the result. Conversely, passing that cosine alone would not establish acceptable elementwise error or end-to-end correctness. Some raw FP32 cosine estimates even exceed one, reinforcing the need for stable metric computation and complementary checks.

The summary's claim of realistic-distribution coverage for all kernels is not supported by these scripts: indexer/top-k/sparse/Sinkhorn/combine use fixed-seed normal/uniform data; only the separate compressor re-verification is described as heavy-tailed, without its reproducible harness in the supplied collateral.

**Required:** a machine-evaluated per-op policy covering finiteness, absolute/relative error and discrete decisions where applicable; numerical metrics rather than rounded prose; nonzero benchmark and job exits on failure; tests that inject wrong results. Keep microbench and current-target gates separate.

### F5 High: Provenance and Reconciliation Overstate Their Guarantees

[reconcile_kernels](../dsv4_roofline_p2.py#L399) checks names and substrings, not complete semantic coverage. **Executed mutations both return `(True, [])`:** remove the compressor reconciliation entry; or set every entry's `cost_rows=[]` and `gap_dispositions=[]`. Observation coverage is checked by searching filenames anywhere in serialized provenance, rather than by an exact kernel-to-row relation. Operand shapes, storage/compute dtypes and complete gap coverage are not compared. The 35 passing self-tests therefore do not prove the advertised reconciliation.

The [provenance record](../results/kernel_provenance.json) also cites `intel_cpu_models/dsa_sparse_attention_cpu` as the sparse oracle. [That module](../../intel_cpu_models/dsa_sparse_attention_cpu.py) implements the older per-head K/V interface **without an attention sink**. The actual newer oracle is inline in [bench_sparse_attend.py](../bench_sparse_attend.py). Its full-FP32 arithmetic is also not identical to the published BF16 primitive's rounding sequence.

The supposedly opaque upstream primitives are available in pinned `inference/kernel.py`: Sinkhorn's epsilon placement can be inspected directly, and sparse attention exposes its BF16 operands/intermediate probability cast, masked indices and sink normalization. A call-site plus an unpinned/local fallback is not the deepest available provenance chain.

Revision resolution is useful but not equivalent to identifying a historical run. Five recorded kernel file blobs match HEAD; the indexer record's `e3fdcb9` differs only by subsequently added comments, so that difference is **not** an executable regression. However, old result coordinates and sparse shape warnings remain in the records after assigning resolved revisions, while provenance still describes several revisions as pending. These files need one coherent experiment identity, not stronger claims inferred from a hexadecimal string.

**Required:** exact coverage of all six kernel IDs and their required nonempty mappings; a disposition for every declared fragment gap; typed operand/precision/cadence contracts; actual oracle symbols and pinned source links; captured module-boundary evidence. Reject missing mappings, not just nonexistent row names. Historical results must retain their original coordinates and be superseded by separately identified runs.

### F6 High: The Measured Join Still Costs Different Contracts

Two concrete mismatches remain in [dsv4_roofline_vs_measured.py](../dsv4_roofline_vs_measured.py):

- **Top-k output width:** it charges int32 indices, but [indexer_topk.cpp](../../kernels/dsa_pilot/indexer_topk.cpp) returns `torch::kLong`. Executed at M=32/S=1024/k=512, output alone is 131,072 bytes; input plus output is **262,144**, not 196,608 bytes. A later `.int()` in the published model does not change this benchmark's output contract. There is no top-k entry in `_EXPECT` to catch this.
- **Indexer compute resource:** it selects the FP32 AVX-512 peak for the whole contraction although the measured kernel converts operands to BF16 and calls BF16 `bmm`/`brgemm`, followed by FP32 reduction. FP32 public storage and accumulation do not make the matrix arithmetic FP32. At M=32, the current FP32 ideal is about 69.79 us; the same useful-boundary traffic floor is about 50.13 us. The FP32 compute-bound label therefore also conflicts with its BW-bound plateau narrative.

**Required:** distinguish public storage, internal compute and intermediate precision. Model mixed stages explicitly or clearly define the ideal mathematical target separately from the executed implementation. Count conversion/packing/scratch work separately when explaining measured latency; do not mistake minimum useful bytes for measured DRAM traffic.

### F7 High: The Bandwidth-wall / No-more-ROI Conclusion Is Unsupported

The [join](../dsv4_roofline_vs_measured.py) and [implementation summary](../results/impl_review.json) declare the remaining gap unclosable by kernel work. Their own data does not establish that cause:

- At M=64, useful-boundary throughput is about **39.0 GB/s for indexer** and **64.4 GB/s for compressor**, versus the cited 214-277 GB/s reference range. These are useful-byte rates, not measured memory-controller traffic, but they do not demonstrate saturation.
- Compressor latency is nearly flat from M=8 to M=32 (0.472-0.476 ms) while modeled bytes grow nearly fourfold. More per-row parallel work can produce falling off-ceiling ratios without approaching a DRAM wall. Its kernel parallelizes only over N and executes two exponentials per channel per window position. Exponential throughput, recurrence and insufficient task parallelism are plausible alternatives requiring discrimination.
- The benches repeatedly reuse the same tensors; warm cache residency is not characterized. A DRAM roofline cannot by itself diagnose that execution regime.
- The M=1 indexer takes 1.023 ms against a 0.101 ms median Torch reference. This is an existing approximately 10.1x achievable-gap signal, not a speculative roofline promise. Its historical record already warns about this regression.
- With S=1024 and k=512, the top-k chunk cap is `S/k=2`, so M=1 has at most two first-stage tasks, not 64. Each chunk retains all its elements before the second-stage selection. The extra copies and parallel region are an obvious small-S candidate; the current claim that small M saturates the cores is false.

**Required:** retain these as open, ranked hypotheses. Use same-work best-path comparisons, thread/task and cache-regime checks, conversion/packing/GEMM/epilogue attribution, and replicated A/Bs before declaring a plateau. Existing nominal targets remain legitimate ideal diagnostics; this finding does not require replacing them with a new hardware calibration or treating every residual ratio as achievable speedup.

### F8 High: Playbook Guidance Is Both Incomplete in Routing and Internally Contradictory

The generated [playbook source](../make_playbook_doc.py) has valuable source-provenance, shape-capture, fragment-disposition, negative-test and measurement rules. But its new M-sweep instruction says to **expect** off-ceiling convergence to the BW wall and treats convergence as evidence, while a later instruction prohibits causal conclusions from distance alone. The first rule teaches the unsupported inference in F7. A sweep is evidence to interpret, not an expected verdict to confirm.

The actual entry-point skills, [model-enablement-playbook](../../../.agents/skills/throughput-enablement/model-enablement-playbook/SKILL.md), [cpu-optimization-playbook](../../../.agents/skills/kernel-optimization/cpu-optimization-playbook/SKILL.md), and [kernel-authoring](../../../.agents/skills/kernel-optimization/kernel-authoring/SKILL.md), do not route through the newly named kernel-provenance/reconciliation gates. The kernel-authoring procedure still emphasizes random-input FP32 parity, without explicit masked-state, signed-weight or discrete-composition checks. Updating only the document generator does not reliably update what the agent loads.

This is **not** a claim that all gates are prose-only: the main generator really executes its self-tests before emission and already rejects several invalid tracker cases. The defects are the missing coverage/contract assertions, permissive benchmark/job exits, and lack of validated raw-data ingestion. Improve those specific gates rather than adding another layer of unchecked PASS fields.

### F9 Medium: Public C++ Entry Points Lack Required Shape Guards

The raw-pointer kernels assume relationships they do not check. Examples: indexer does not enforce equal q/kv batch and head dimensions or weight `[N,H]`; compressor does not enforce score/APE shapes; combine does not require `hc>0`, divisibility or pre `[M,hc]`; Sinkhorn does not check scale/base lengths. Invalid calls can cause out-of-bounds access or division by zero instead of a controlled error. Top-k also assumes a two-dimensional FP32 CPU tensor.

**Required before integration:** explicit CPU/device, dtype-policy, rank and related-dimension checks, valid scalar ranges, and safe tail/empty-input behavior. No invalid-pointer crash probes were run during this review.

## Per-kernel Disposition

| Kernel | Provenance / implementation assessment | Decision |
|---|---|---|
| Indexer logits | Scoring formula recognizable; precision contract differs across variants; composed selection counterexample | Block correctness sign-off; fix/validate precision before retuning dispatch |
| Indexer top-k | Valid selection approach for ordinary finite inputs; int64 output; tie/mask policy needs explicit testing | Keep candidate; correct traffic accounting and test small-S policy |
| Compressor | Pool formula recognizable; valid leading-mask case produces NaNs | Block overlap use; repair and replay module-boundary states |
| Sinkhorn | Epsilon ordering, pre/post transforms and iterations agree with pinned primitive on inspection | Keep candidate; enforce tolerances, range/shape guards and reproducible evidence |
| Combine | Weighted-sum equation and fused read-once approach agree with published hc_pre | Keep candidate; add contract guards and asserted parity |
| Sparse attend | Scalar MQA+sink equation is recognizable; cited oracle is stale, AMX rounding differs, measured column is wrong | Preserve as surfaced candidate, not a certified production path; correct evidence before route decision |

The donor destination is not established by the supplied benchmark. The current [_torch_flash_mla_with_kvcache](../../intel_cpu_models/_dsv4_cpu_infra.py#L1589) has a batched, masked Torch/BF16 path; naming a donor MLA flash in a cost row does not prove dispatch to that donor. The earlier partial roofline already excludes fallback execution costs. Preserve that distinction and require a concrete donor entry point, sink/mask/two-source/KV-layout capability check and dispatch evidence before calling the handoff complete.

## Remaining ROI Candidates

These are priorities for investigation, not promised speedups. Deployment value still needs workload frequency, invocation counts and in-engine attribution.

| Priority | Candidate | Evidence / cheapest useful check |
|---|---|---|
| First | Remove M=1 indexer regression | Existing same-run Torch reference is about 10x faster; after F2, compare numerically equivalent paths and decompose cast/pack/dispatch/GEMM/reduction |
| First | Compressor channel/task partitioning and stable vectorized softmax alternatives | N-only parallelism, two exp calls per element, valid-mask failure; compare channel tiling and a stable multi-pass implementation at all three pool contracts |
| Next | Simplify top-k when chunks retain the whole input | At S=2k, two-stage selection adds copies/regions without pruning; compare direct row selection across M, preserving tie semantics |
| Next | Remove redundant indexer conversion/packing where real inputs permit | Both variants cast inputs; tiled code packs each query serially each call. Measure these stages before considering fusion/prepacking, and do not assume dynamic queries can be packed once at model load |
| Conditional | Fused/indexed donor attention rather than scalar per-head streaming | The scalar rereads KV per head; establish actual compatible donor dispatch first. Preserve the measured M=1 scalar advantage instead of deleting it with a blanket regression label |
| Lower / measure first | Sinkhorn/combine serial thresholds or adjacent MHC fusion | Already small absolute costs and credible fused implementations; only proceed if call-count-weighted savings exceed engineering cost/noise |
| Required for whole-model ranking | R=8 main/indexer pools, omitted surrounding stages and active donor/fallback costs | Perf sweep measures only R=128/D=512 pool, while main/indexer ratio-4 pools each execute 21/4 amortized calls per step versus 20/128 for ratio-128. Do not transfer a single-shape plateau to those different contracts |

A claim that no other ROI-worthy optimization exists requires a scoped Amdahl ledger: measured time and frequency, correctness-qualified best alternative, plausible savings bound, experiment cost, and explicit open exclusions. Six microbench medians alone cannot establish whole-model exhaustion. Keep the accepted partial analytical inventory, but do not promote its excluded costs to zero or its hypothetical donor paths to deployed performance.

## Gates Needed Before the Next Review

1. **Before kernel authoring:** pin the actual primitive implementation and oracle; capture module-boundary tensors/metadata now, full serving-path tensors at integration. Include masks, state transitions, signed values, quantized grids, layouts and phase/TP variants.
2. **Before accepting a variant:** assert finite outputs and agreed numerical/discrete criteria; test caller-to-kernel composition and batch-dispatch boundaries. A deliberately broken result must fail the checker and the job.
3. **Before a performance run:** require passing contract tests and record exact source/bench/dependency/build identity, selected variant, affinity, inputs and cache/warmup policy. Keep the user's existing hardware-selection and replicated-run rules.
4. **Before rendering:** parse named raw metrics with replica coordinates; reject missing, duplicate, wrong-variant or nonpositive timings; validate all kernel/mapping/gap coverage and actual runtime tensor byte widths and compute resources.
5. **Before saying plateau/no ROI:** test competing bottleneck hypotheses, compare the same work to its observed fast path, and publish the call-count-weighted ROI ledger. An M-sweep cannot waive this gate.
6. **Before integration sign-off:** prove the chosen donor's exact capability/dispatch, then run the established full-model correctness ladder. This review does not demand an expensive full-model run to fix local reporting or masked-softmax defects.

Route these checkpoints from the actual orchestrator skills, with explicit artifact names and executable commands. Retain the useful existing reviewer-negative-test discipline, but expand tests from isolated counterexamples to missing coverage, precision transitions and composed behavior.

## Evidence and Review Limits

Opened both upstream files at HF revision `60d8d70770c6776ff598c94bb586a859a38244f1`: [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py) and [kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py). Inspected all six C++ kernels, their benchmarks/oracles, records, the raw 384372 log, cost/provenance validation, the playbook generator and entry skills.

Executed the generator's **35 passing checks**, named-column median reconstruction, dropped-coverage reconciliation probes, kernel-blob comparisons, single-thread C++ compressor/top-k/indexer probes, and the zero-output benchmark-checker probe. Correctness probes used `/scratch/bkaul/venvs/sglang-cpu/bin/python`, current source and the benchmark compiler flags, with one Torch/OpenMP thread. No new latency measurements, model loads, cluster jobs or GPU oracle executions were performed. Existing raw measurements are observations on pcl-sprh11, not new reviewer measurements.

The FP4-grid reproduction uses `torch.manual_seed(1)`, levels `[-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6]`, random integer indices into those levels for q `[1,64,128]` then kv `[1,1024,128]`, and weights `(torch.randn(1,64)/sqrt(8192)).bfloat16().float()`. Compare the dispatcher's M=1 output with the first output after repeating all three inputs eight times, then compare their top-512 sets. This is synthetic contract evidence, not a captured model forward.

Only this review document is added. Production code, benchmarks, results, historical reports and playbook files are unchanged. **Next disposition:** repair F1/F2, correct and validate the measured evidence/contracts, then reassess ROI. Do not certify implementations or close the optimization queue on the current collateral.