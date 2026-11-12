# DSv4 native-kernel and serving-path inventory

Date: 2026-11-07. **Static inventory, not a runtime coverage or performance certificate.**

**Publication note (2026-11-11):** this preserves the dated source observations,
not a claim about current main. Subsequent accuracy closure and the C4 serving
campaign are recorded on the
[performance branch](https://github.com/bharat-kaul/sglang-cpu-opt/tree/dsv4-perf-main).
Cross-worktree sources below are shared-filesystem evidence, not published
copies; source-key links point to the provenance table rather than pretending
an uncommitted source has a GitHub permalink.

## Result and scope

The six pilot kernels are only a subset of the forward. This inventory contains
**33 compute/data-movement operation groups and 8 serving-substrate surfaces**.
These are explicit work-accounting units, **not 41 missing C++ kernels**. Multiple
groups can share a donor or be fused; a single group can contain several tensor
operations.

Within the 33 groups:

- **12 L: native library/core paths** already exist, sometimes with eager
  conversion/layout preparation. This does not certify their ISA, optimality,
  or absence of surrounding overhead.
- **7 G: native alternatives exist but are gated/partial/mismatched** against the
  serving contract. These include the six pilot operations and MHC post.
- **14 E: eager compositions** remain in the inspected CPU path. No matching
  whole-operation native dispatch was established for these groups.

The scope is the real Flash model's ordinary TP=1 CPU prefill and decode forward:
embedding, all 43 decoder layers, final MHC head, normalization and LM-head
projection. Main-layer compression ratios are 0 on 2 layers, 4 on 21, and 128 on
20; the config has one extra ratio entry for the additional prediction layer,
which is not counted here. There are 256 routed experts, top-6 selection,
one shared expert, three hash-routed layers, hc=4, and sqrtsoftplus routing.

Runtime substrate is counted separately because Python orchestration is not
itself a missing math kernel. Speculative/MTP, dspark capture, GPU, context/pipeline
parallelism, LoRA, alternative quantizations and multi-node execution are not
certified by this inventory. TP communication is recorded as a conditional surface.
Sampling/tokenization, HTTP and scheduling policy are outside the model-forward
boundary; logits processing is included.

**No new kernel/wiring implementation, model run, benchmark, default change or
shared-library rebuild was performed for this inventory.** Earlier Sinkhorn work
and the unvalidated, unsubmitted top-k scope harness remain separate experiments.

## Sources and revision boundaries

Do not treat these worktrees as a single already-integrated implementation:

| Key | Source | Identity and role |
|---|---|---|
| A | `dsv4-tree-pilot/plugin/intel_cpu_models/_dsv4_cpu_infra.py` | Accuracy worktree HEAD `9320cd1c46379843759378696d8b3018241caf53`, with existing uncommitted changes. Current serving-source observation, not the optimization baseline. |
| K | `dsv4-tree-pilot/plugin/intel_cpu_models/_dsa_ckernels.py` | Same accuracy worktree; actual wrapper contracts/defaults. |
| O | [Optimization worktree](../../) | HEAD `76cbcff6e0bcd81d0320262e5e9f39e723144089`; pilot experiments and newer standalone BF16 sparse kernel. |
| F | `dsv4-fusion-measure/docs/dev/DSv4-fusion-campaign-review.md` | Clean worktree HEAD `08b46f0`; prework and separate cooperative wiring. Not automatically present in O or A. |
| U | Installed SGLang source under `/scratch/bkaul/venvs/sglang-cpu/lib/python3.12/site-packages/sglang/` | Inspected upstream dispatch and math; source existence is not proof of the loaded binary in a historical run. |
| C | `sglang/python/sglang/kernels/aot/csrc/cpu/` | Clean source HEAD `2ca405fac0994db407088f647422ffbf035a4735`; existing cooperative assets and shared-expert barrier repair. |

Worktree-relative paths in this table resolve under `/data/nfs_home/bkaul/`.
Fusion evidence names below resolve within that F worktree.
Additional source keys: `F-infra` is F's
`plugin/intel_cpu_models/_dsv4_cpu_infra.py`; `coop` is C's `coop_region.cpp`.
`compressor`, `indexer` and `clamp` are A's sibling files
`dsa_compressor_cpu.py`, `dsa_indexer_cpu.py` and `_dsv4_moe_clamp.py`.

Primary SHA256 anchors:

```text
A _dsv4_cpu_infra.py
fcd58e77349c0d890ab1d00ffd5065560d6737e0b7585b0d487df296993bf491
K _dsa_ckernels.py
fbbd17c1d06711d561790641f06a0dd654c6a5c40603278c9b1364abeebd256c
U srt/models/deepseek_v4.py
7f2e6f75707c44311cfa3c398f6f708c7a8d5af17c3d55ec99f6af2996d07437
/scratch/bkaul/models/DeepSeek-V4-Flash/config.json
b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818
F docs/dev/DSv4-fusion-campaign-review.md
dd8196a4f47d8e4de277e61f91ced591b12419a626ca36f958e3ed995de67f7b
```

Line references below identify the inspected versions. A is evolving in another
session; recheck identity before implementation. This inventory does not roll its
accuracy changes into O.

## 1. Compute and data-movement inventory

L means a reusable native primitive/core, not a requirement to rewrite library
math by hand. E means a multi-operation eager composition, not that ATen lacks C++
implementations internally. G means implementation existence and default serving
coverage must not be conflated.

| ID | Operation group | Class | Concrete CPU path and remaining disposition | Source |
|---|---|---|---|---|
| C01 | Token embedding | L | Native ATen embedding through the embedding quant method; preserve the donor. | U `srt/layers/quantization/unquant.py:361`; U `srt/layers/vocab_parallel_embedding.py:542` |
| C02 | Expand input into hc streams | L | `unsqueeze().repeat()` uses a native allocation/copy primitive. Account for bytes; not a new arithmetic-kernel task. | U `srt/models/deepseek_v4.py:3047` |
| C03 | Standard RMSNorm modules, including final norm | L | `rmsnorm_cpu` / `fused_add_rmsnorm_cpu` on the CPU-capability branch; native Torch composition otherwise. Does not cover the custom norm sequences below. | U `srt/layers/layernorm.py:844` |
| C04 | Attention/indexer dense projections | L | Packed FP8 `fp8_scaled_mm_cpu` or BF16 `weight_packed_linear`, depending on layer method. Includes q/kv/query-low-rank and indexer weight/query projections, not the special compressor projection. Verify each layer's prepack/dispatch, not just AMX availability. | U `srt/layers/quantization/fp8.py:1061`; U `srt/layers/quantization/unquant.py:403`; A:4759 |
| C05 | Per-head query self-norm + RoPE | E | FP32 upcast, reductions, sqrt/divide, position lookup, rotation, stack/clone/copy. The function name says fused; CPU implementation is not one fused native operation. | [A:1717][A] |
| C06 | Key norm + RoPE | E | Weighted RMSNorm and interleaved trailing-dimension RoPE using eager operations. Producer boundary must retain the accepted rounding schedule. | [A:1841][A] |
| C07 | KV quantization, packing and paged write | E | Per-64-channel max/log2/exp2/scales, FP8 conversion, BF16 rope part; `SetKAndS.execute = SetKAndS.torch`. Dense shadow is also maintained. Native generic KV writers are not automatically compatible with this layout. | [A:1766][A], [A:1789][A], [A:1817][A] |
| C08 | Main/indexer compressor projection | L | `compute_kv_score -> linear_bf16_fp32 -> torch.mm(x.float(), weight.float().t())` on the ordinary CPU fallback. Native FP32 GEMM, but per-call conversion and row-major weight contract remain. Not the pilot softmax-pool kernel. | U `srt/layers/attention/dsv4/compressor.py:444`; U `kernels/ops/attention/dsv4/gemm.py:115`; [A:969][A] |
| C09 | Corrected main C4/C128 pooling | E | `_compress_prefill_core`: APE addition, C4 previous-window overlap, softmax and weighted reduction. Does **not** call the pilot compressor wrapper. Applies to completed windows in prefill and the enabled continuation/accumulator paths. | [A:4132][A], [A:4242][A], [A:4510][A] |
| C10 | Main compressed-key norm, RoPE and FP8 roundtrip | E | Same producer performs RMSNorm, positional rotation and quantize/dequantize. FP32-intermediate versus BF16-intermediate is accuracy-gated; cannot remove these casts as a presumed no-op. | [A:4160][A] |
| C11 | Indexer query RoPE + Hadamard | E | Projection is C04; trailing RoPE then FP32 matrix multiply by cached Walsh-Hadamard matrix. This route explicitly skips FP8 query quantization; not a source-faithfulness certificate. | [A:4759][A], [A:2056][A] |
| C12 | Indexer compressor pooling | G | This separate compressor **does** call `dsa_compressor_cpu.compress_softmax_pool`, whose native arm is the pilot `compressor_softmax_pool_fp32intrin`; off unless enabled. Do not equate its coverage with C09. | [A:4773][A]; [compressor wrapper:45][compressor] |
| C13 | Indexer compressed-key normalization | E | Concatenate pools, convert, `_rmsnorm_torch`; separate from pooling. Its stage contract must be reconciled with the trusted indexer, not inferred from the main compressor. | [A:4794][A], [A:2045][A] |
| C14 | Indexer score GEMM + ReLU/head weighting/reduction | G | Pilot native fused epilogue available. Serving currently materializes shared KV as contiguous `[N,S,D]` before invoking it; eager einsum remains the off/fallback arm. Signed weights and intermediate rounding matter. | [A:4807][A]; [indexer wrapper:36][indexer]; [K:166][K] |
| C15 | Indexer causal mask and selected-position handoff | E | `key_start <= positions`, masked fill, scatter and repeat-interleave. Request/position correctness is part of the serving boundary. | [A:4832][A], [A:4880][A] |
| C16 | DSA indexer top-k | G | Pilot `nth_element` implementation, but wrapper calls it **twice**. Top-k-only override is also constrained by the outer indexer-native branch. Existing M2 serial speedup is prior dispatch-rule application, not a new algorithm. | [K:172][K]; [A:4836][A]; [indexer wrapper:47][indexer] |
| C17 | Attention KV gather, two-source assembly and masks | E | Valid-slot selection/index-select, SWA/coverage/request masks, compressed-handoff consumption, concatenation and conversions. Mask vectorization already exists; do not reimplement it as new work. | [A:1942][A], [A:2074][A], [A:2122][A] |
| C18 | Sparse attention QK/softmax/sink/PV | G | Serving native arm is FP32 `sparse_attend_masked(shared KV, mask, sink)`; requires equal K/V dimensions and a sink. Off/fallback path is multiple eager GEMMs/softmax operations. Neither historical `bestof_hx` nor newer indexed BF16 standalone implementation is this serving entry. | [A:2303][A]; [K:227][K]; [sparse kernel:389][sparse] |
| C19 | Attention output inverse RoPE | E | `_cpu_fused_rope_inplace(..., inverse=True)`, eager interleaved rotation. Must preserve the inverse/sign and position boundary before the grouped projection. | [A:1746][A]; U `srt/models/deepseek_v4.py` MQALayer.forward |
| C20 | Grouped absorbed output projection `wo_a` | L | Row-major BF16 einsum/library contraction; deliberately bypasses packed-linear method. Keep layout correction. Its comment saying not AMX-accelerated is not executed-ISA evidence; profile the actual library dispatch before proposing replacement. | [A:969][A]; U `srt/models/deepseek_v4.py:405`, `:1847` |
| C21 | Final attention output projection `wo_b` | L | Standard quantized/packed linear donor, plus conditional TP reduce. No new standalone replacement justified by this inventory. | U `srt/models/deepseek_v4.py:1861`; C04 providers |
| C22 | MHC pre upcast/RMS statistic/mix scaling | E | FP32 flatten, square/mean/rsqrt and multiplication around a separate linear. Earlier fused no-temp variant was measured slower and removed. | [A:445][A]; U `srt/models/deepseek_v4.py:1981` |
| C23 | MHC mix projection | L | `F.linear` library GEMM on FP32 flattened streams. Native math already exists; separate from Sinkhorn and the surrounding reductions. | [A:485][A]; U `srt/models/deepseek_v4.py:1981` |
| C24 | MHC Sinkhorn | G | Pilot C++ versus Torch reference via worker control/master flag. Experimental SIMD result is limited to tested shapes, not a serving default. Strict exceptions are currently swallowed by a nested catch. | [A:4005][A]; [K:199][K]; [standalone findings](dsv4-standalone-meta-findings.md) |
| C25 | MHC pre weighted combine / optional following norm | G | Pilot combine available; separate coop combine+RMSNorm donor already exists. The cooperative integration sits inside the TIMEIT+TIMEIT_HCPRE wrapper, so COOP_MHC alone does not install it. Preserve intermediate dtype semantics. | [A:445][A], [A:498][A], [A:4027][A]; [coop donor:138][coop] |
| C26 | MHC post/residual mixing | G | Default FP32 einsum plus post-scale/add; existing `coop_hc_post_cpu` is gated. Older BF16 BMM substitution was rejected for in-model performance/drift. | [A:4066][A]; [coop donor:222][coop] |
| C27 | Final MHC head | E | Routed to `hc_head_torch`: upcast/statistic, linear, sigmoid/scale and weighted stream sum. No proven whole-operation native coverage here. | [A:4107][A]; U `srt/models/deepseek_v4.py:494`, `:2884` |
| C28 | Expert router projection | L | MoEGate linear/native donor; CPU FP32 packed-layout guard exists. Route score formation is a separate operation. | [A:2577][A]; U `srt/models/deepseek_v2.py:1178` |
| C29 | Normal expert routing score/top-k/renormalize | E | Real config is **sqrtsoftplus**, ungrouped. CPU JIT branch is disabled; `fused_topk_cpu` has native softmax/sigmoid only, otherwise calls the Torch composition. Therefore generic native top-k availability does not establish DSv4 router coverage. | U `srt/models/deepseek_v2.py:680`; U `srt/layers/moe/topk.py:817`, `:885`, `:2408` |
| C30 | Hash expert routing | E | First three hash layers use Torch score formation, tid2eid lookup/gather and normalization. Plugin explicitly disables CUDA fused hash-top-k. Not the DSA pilot top-k. | [A:2647][A]; U `srt/layers/moe/hash_topk.py:134`, `:213` |
| C31 | Routed expert group/pack, gate-up GEMM, SwiGLU, down GEMM and reduction | L | Native packed MXFP4 donor when MXFP4_MOE is selected; native unpack stays inside compute. Small occupancy uses vector tiny-GEMM, not universally AMX. Accuracy clamp selects a separate private native build. Preserve that accepted baseline; do not widen weights or substitute the unclamped coop driver. | [A:2931][A], [A:3120][A]; [clamp loader][clamp]; U `kernels/aot/csrc/cpu/gemm_fp8.cpp:428` |
| C32 | Shared expert and routed/shared merge | L | `shared_expert_cpu` already computes shared gate-up/activation/down and combines routed output. D1 cooperative routed+shared driver exists separately, with no demonstrated end-to-end speedup. | U `srt/models/deepseek_v2.py:1195`; [fusion wiring:2714][F-infra]; C `moe.cpp:1394` |
| C33 | LM-head projection and logits processing | L | Native quant-method linear or ATen matmul, followed by framework slicing/gather/logprob work as requested. Preserve donor; logits postprocessing is not a new fused-attention kernel. | U `srt/models/deepseek_v4.py:3299`; U `srt/layers/logits_processor.py:769`, `:868`, `:1121` |

## 2. Serving substrate inventory

| ID | Surface | Current implementation / obligation | Source |
|---|---|---|---|
| R01 | Registration, CPU backend/pool configuration and allocator selection | External model overrides upstream; install patches at import. CPU FP8 pool guard bypass and device predicates are part of enablement. DSA_CPU=1 is required for route-2 wiring; bare flag-off is not the supported complete DSv4 route. | [model entry](../../plugin/intel_cpu_models/deepseek_v4_cpu.py); [A:535][A], [A:772][A], [A:4866][A] |
| R02 | Weight loading, packing and capacity | Native packed MXFP4 flag; scales/layout handling; row-major exceptions for wo_a and compressor. Legacy FP4 widening branches still exist when alternative flags are selected. No capacity-heavy load was attempted. | [A:969][A], [A:2931][A] |
| R03 | Paged allocation and request-token indices | Torch/cumsum/arange plus Python per-request loops for alloc_extend/decode; CPU fallback for last-loc and cache-index writes. These are not among the six pilot kernels. | [A:5092][A], [A:5131][A], [A:5166][A] |
| R04 | Positions/compressed metadata and scheduling | Torch compressed metadata builder; FlashMLA GPU scheduling metadata disabled; support_triton references repointed to CPU paths. CPU paged compressor plan replaced by route-2 state. No-op plan does not mean compression compute is absent. | [A:4969][A], [A:5029][A], [A:4887][A] |
| R05 | Dense KV shadow, packed cache and validity | Separate dictionaries/tensors, growth/copy/index-select and paged writes. Identity, request ownership and slot reuse are correctness obligations before eliminating a copy or cache. | [A:1841][A], [A:1913][A], [A:1942][A] |
| R06 | Compression history/carry and handoffs | Per-layer/request dictionaries, append/cat, absolute window progression, overlap carry, publication and consume/pop. CHUNK_ACCUM/DECODE_CONT and radix containment are accuracy-work controls, not optimization knobs to toggle casually. | [A:625][A], [A:4242][A], [A:4510][A], [A:4581][A] |
| R07 | Thread policy, affinity, NUMA and collectives | Warm-pool and adaptive thread prework exists. TP=1 scope differs from TP>1, where post-init resizing may be unsafe; optional gloo rerouting is separate. Kernel T56 optimum is not the validated whole-model B32 T32 optimum. | F `docs/dsv4-decode-operating-point.md`; [A:1279][A], [A:1414][A] |
| R08 | Dispatch evidence, fallback/error handling and worker controls | Current counters cover selected ops, not this whole inventory. Some increment before successful execution; nested MHC catches swallow strict rethrows. Control-file unreadability can select fallback silently. Must repair before any no-eager/default-promotion claim. | [K:49][K], [K:94][K], [K:181][K]; [A:3975][A], [A:4005][A] |

The example accuracy launcher, `dsv4-tree-pilot/plugin/validate/run_gsm8k_dsv4_sharded_base.sbatch`,
explicitly enables DSA_CPU, MXFP4_MOE and DSA_CKERNEL. These are launch-profile
choices, **not package defaults**. Even that profile leaves C09/C10/C29/C30 and
other eager groups intact. Merely exporting CKERNEL=1 is not complete coverage.

## 3. Fusion prework ingested: preserve outcomes, distinguish evidence

Read the requested [campaign review][F], its
`docs/dev/DSv4-decode-perf-lever-evidence.md`,
`docs/dev/DSv4-whole-layer-port-feasibility.md`,
`docs/dev/DSv4-per-layer-driver-port-design.md`,
`docs/dev/b32-moe-apply-bandwidth-task-scope.md`,
and `docs/dsv4-decode-operating-point.md`.
Also inspected the Scope-A result/harness and the cooperative source/wiring.
Historical experiment numbers below are inherited evidence, not rerun results.

| Prior work | Disposition for this campaign |
|---|---|
| Warm pool + adaptive threads | Existing baseline, not a new gain. Do not benchmark a new kernel against a cold/incorrectly threaded baseline and credit config recovery to fusion. |
| MHC native dispatch / Scope-A | The review's attribution needs correction; see inconsistency I1 below. Small reported improvement is not demonstrated cooperative fusion over the warm optimum. |
| MoE internal region collapse | Reported -3.5%; keep rejected/off. No repetition merely because a cooperative region sounds cheaper. |
| D1 routed+shared cooperative driver | Existing donor, not missing code. Shared-expert barrier fix exists at C `moe_fp8.cpp:338`. Reported cosine agreement is not bitwise proof; full-model performance was not measured. Current wiring excludes hash layers and clamp-enabled layers. |
| Whole-layer region D2-D5 | Parked: paged store, data-dependent selection and state handoffs are boundaries in the existing design. Crossing them requires a much larger serving-runtime port. No demonstrated 1.8-2.5x win justifies it. |
| MoE bandwidth accounting | Keep distinct-expert bytes and real captured routing. Reject raw-stream-only ceiling and uniform-routing proxy. Stream+unpack is a comparator, not proof all its gap is removable. |
| T48 to T56 at B32 | Approximately 123 to 143 GB/s reported for the isolated apply. **Not a delivered 1.15x model decode gain**; end-to-end validation remains explicitly pending. |
| Expert-major decomposition | Reported 2.8-3.3x worse; rejected. Do not repeat removal of N-splitting as a fresh idea. |
| Sparse BF16 gather changes | Existing standalone work already tried explicit SIMD pointer setup, packing alternatives, hardware gather and no-prefetch. Read [its record](../../plugin/kernels/dsa_pilot/sparse_attention_kernel/README.md) before proposing another gather variant. |

The prework is a reason to stop repeating **barrier-collapse** experiments, not
proof that every source-faithful data-movement fusion or missing dispatch has
zero upside.

## 4. Inconsistencies and acceptance blockers

**I1 - Scope-A mechanism and operating point do not match the headline.**
The [review][F] labels job 397710 as combine+post-norm cooperative fusion on top
of a config-optimized baseline. Its
saved result, `plugin/validate/results/dsv4_mhc_ab_repaired_397710.json`,
states threads=8. Its
launcher, `plugin/validate/run_dsv4_mhc_ab_repaired.sbatch`,
sets OMP_WAIT_POLICY=passive and DECODE_THREADS=8; its
driver, `plugin/validate/validate_mhc_ab_repaired.py:46`,
switches **MHC_CK_FILE**, not COOP_MHC. That control selects native
Sinkhorn/combine, not combine+norm region fusion. Historical commit 86a3632 has
the same control; ef204b1 adds the result. Unstamped inherited environment cannot
rescue attribution: the experiment does not toggle the cooperative mechanism.
The 4.51%/2.42% values remain reported native-dispatch observations, not the
claimed warm-baseline fusion result.

**I2 - The same prework still contains superseded recommendations.**
The driver design retains 1.8-2.5x and an unchecked D1 status despite the later
campaign stopping D2-D5. Whole-layer feasibility still says to build Region 1.
The bandwidth scope retains "more threads do not help" and a 2.3x recommendation,
while later real-routing/thread sweeps revise both. Use the later dispositions,
not the earlier TODOs, as the experiment ledger.

**I3 - "Fusion closed", "half structural" and universal floors overstate scope.**
An M1 self-CPU profile, a negative MoE-internal experiment and the absence of a
D1 end-to-end run do not exhaust all batches, contexts or memory-traffic fusions.
The source does not establish a quantitative half-structural decomposition.
Do not turn those assertions into accepted ceilings or reopen work without new
evidence; preserve them as bounded hypotheses.

**I4 - Native inventory labels are not executable coverage.**
The generated [model manifest](../../plugin/intel_cpu_models/deepseek_v4_cpu.py)
still calls several now-authored DSA operations gaps while declaring generic norm,
RoPE/top-k coverage too broadly. C05/C06/C29/C30 show why operator-family labels
cannot substitute for the actual DSv4 branch.

**I5 - Sparse standalone and serving boundaries differ.**
The newer BF16 result explicitly prepares selected rows outside timing. Serving
still gathers/builds masks and calls another FP32 kernel. A speed ratio from one
cannot be assigned to the other without matched preparation, masks, sinks, output
dtype and numerical boundaries.

**I6 - Strict/provenance checks do not prove zero fallback.**
MHC's inner catch swallows the strict rethrow; some marks count attempted rather
than completed native calls. MHC control uses the sinkhorn setting for both
Sinkhorn and combine, bypassing an independent combine override. Existing
Scope-A branch counters also increment before actual kernel completion.
The duplicate top-k call is real work hidden behind one mark.

**I7 - Correctness changes invalidate casual baseline substitutions.**
Row-major weight fixes, compressor intermediate-precision controls, carry/history,
and MXFP4 activation clamp alter what must be timed and compared. The old
unclamped D1 path is not a replacement for the accuracy-selected clamp path.
Nothing in this inventory resolves the separate CPU/GPU accuracy investigation.

## 5. Bounded follow-up, not a new speculative fusion campaign

| Priority | Candidate / work item | What is established now | Stop or advance rule |
|---|---|---|---|
| 0 | Reliable dispatch coverage and baseline identity | Concrete wrapper/control defects; partial registry; divergent branches. | Repair actual-success counters and failure propagation, then observe every expected group by phase. This is coverage work, not a promised speedup. |
| 1 | Remove duplicate native DSA top-k | Two identical invocations in K:172, one returned result. | One surgical change with output/call-count regression tests and serving-site timing. Do not generalize the standalone M2 serial knob before S/K qualification. |
| 2 | Shared-KV indexer interface | Serving creates FP32 `[N,S,D]` from shared `[S,D]` solely to satisfy the pilot interface. | Profile preparation-inclusive cost. Materialized buffer is `4*N*S*D` bytes, replicated storage above shared form `4*(N-1)*S*D`; these are logical allocation bytes, not measured DRAM traffic. Use a donor accepting shared KV only if the removed cost can matter. |
| 3 | Serving-compatible sparse donor | Faster prior standalone native implementation exists, but contract and preparation differ. | First reconcile mask/sink/precision/output and cache inputs against the accuracy-approved boundary. One matched replay including preparation; no blind symbol swap and no automatic new kernel. |
| 4 | Corrected compressor / norm-RoPE-KV / router composition | Missing whole-operation native coverage is explicit (C05-C13, C29-C30). Generic donors may cover pieces, not the full contract. | Attribute complete producer costs on real prefill/decode/context shapes first. Preserve rounding, overlap and request state. Attempt only a bounded donor adaptation justified by measured cost. |
| 5 | wo_a and compressor projection conversion/layout | Both intentionally avoid incorrect VNNI consumption; core library GEMMs already exist. | Inspect executed ISA and conversion-inclusive latency before changing precision or packing. A comment saying "not AMX" is insufficient. |
| Park | Further Sinkhorn variants, whole-layer coop, expert-major MoE | Narrow Sinkhorn win already recorded; negative/unsupported fusion outcomes retained. | No further experiment without new discriminating evidence. |
| Separate confirmation | MoE isolated T56 setting | Existing isolated signal, not whole-model gain. | Validate under the accepted accuracy/runtime configuration before changing the end-to-end thread map. |

Coverage completion and performance optimization are different goals: E groups
remain coverage debt even if their measured speedup potential is small. They may
need a native implementation to meet the user's requirement, but cannot be sold
as material performance wins without measurement.

For any future expensive run, use one resident load with worker-confirmed controls
and a predeclared multi-angle disposition matrix. Report native path and fallback
counts, layer/phase/shape, full preparation costs, and input/output contracts.
Materiality remains >10%; <5% is not grounds to retain complexity. No default
promotion until correctness, actual execution and the relevant performance
comparison are separately accepted.

## 6. What this closes and what remains

The **static inventory deliverable** closes here: the model-forward groups and
serving substrate are explicitly enumerated, native/eager boundaries mapped,
prework reconciled, and no new performance upside assumed. This is not a claim
that every optional framework feature or every ATen-internal instruction has
been audited.

Still open: runtime execution coverage at the accuracy-approved configuration;
source-conformance gates where noted; selecting/reusing exact native donors for
the eager groups; implementing reviewed wiring; and measured optimization
dispositions. **Zero additional kernels are claimed complete or optimal by this
inventory.**

[A]: #sources-and-revision-boundaries
[K]: #sources-and-revision-boundaries
[F]: #sources-and-revision-boundaries
[F-infra]: #sources-and-revision-boundaries
[coop]: #sources-and-revision-boundaries
[compressor]: #sources-and-revision-boundaries
[indexer]: #sources-and-revision-boundaries
[sparse]: ../../plugin/kernels/dsa_pilot/sparse_attend.cpp
[clamp]: #sources-and-revision-boundaries
