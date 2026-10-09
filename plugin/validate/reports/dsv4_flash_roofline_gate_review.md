# DSv4 Flash Review: Roofline Correctness Gate

Date: 2026-10-08.

Reviewed repository revision: `35ef45585d987c63cb5676e15814131f52dbe6f8`.

## Verdict

**FAIL: the current roofline is not quantitatively correct. Implementation review is deferred, as requested, until this gate passes.**

The eight built-in invariants pass and the checked-in report body regenerates exactly. Nevertheless, independent checks against the published model and checkpoint metadata expose incorrect attention work, main-layer counts, compression cadence, shared-expert precision, and MHC dimensions. These errors affect the optimization targets and their interpretation, not just presentation.

Keep the agreed nominal machine-peak methodology. Sustainable-ceiling calibration and full serving integration are not prerequisites for fixing this analytical model. Conversely, passing self-consistency tests or labeling an item MODELED/MEASURED does not establish that it represents DSv4 Flash.

This report supersedes preliminary observations from this review session where they differ. In particular, the verified main-layer compression counts are **2 uncompressed, 21 ratio-4, and 20 ratio-128**, not 3/20/20. Router score GEMMs do run on hash-routed layers in the published reference; counting 43 router GEMMs is not itself a defect.

## Evidence and Scope

- Prior reviews: [original verdict](dsv4_roofline_emr_verdict.md) and [re-evaluation](dsv4_roofline_emr_reevaluation.md).
- Audited accounting: [generator](../dsv4_roofline_p2.py), [generated report](dsv4_roofline_emr.txt), [roofline/measurement join](../dsv4_roofline_vs_measured.py), [platform profile](../platforms/emr.json), and [open-item tracker](../results/roofline_open_items.json).
- Workflow requirements: [playbook generation source](../make_playbook_doc.py#L311) and [model roofline skill](../../../.agents/skills/throughput-enablement/model-roofline-analysis/SKILL.md).
- Independent reference pinned to Hugging Face revision `60d8d70770c6776ff598c94bb586a859a38244f1`: [checkpoint config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json), [inference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py), [inference config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json), and [checkpoint index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/model.safetensors.index.json).
- Checkpoint dtype/shape evidence was read from the safetensors header of `model-00002-of-00046.safetensors` at that same revision, using HTTP range requests. No tensor payload was loaded.

Reference-forward inspection here establishes the workload being modeled. It is not an audit of the authored CPU kernels. No new kernel benchmarks, parity tests, model runs, or cluster jobs were launched for this roofline gate. Existing measured numbers are not independently certified by this report.

## Findings

### R1. High: Attention Graph and Main-Layer Counts Are Wrong

The [inventory](../dsv4_roofline_p2.py#L204) charges full-context attention at `S=4096` on all 43 layers, then an additional top-512 attention operation on 20 layers. The published [Attention.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L479) instead concatenates window and compressed indices and performs one sparse attention operation over those sources.

The checkpoint config contains **44** compression entries. The reference constructs 43 main blocks and then an MTP block with the next index; see [Transformer construction](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L785). Only the first 43 entries belong to this main-model decode inventory. Their counts are 2/21/20; the trailing ratio-0 entry belongs to MTP. Consequently, `N_IDX=20`, `N_COMP=40`, and `N_SWA=3` in the [generator constants](../dsv4_roofline_p2.py#L61) are incorrect for the pinned model.

At a decode boundary with 4096 available tokens, a full 128-token window, and the stated ideal independent-request traffic model:

| Main-layer kind | Layers | Attention positions per request per layer |
|---|---:|---:|
| Ratio 0 | 2 | 128 |
| Ratio 4 | 21 | 128 + min(512, 4096/4) = 640 |
| Ratio 128 | 20 | 128 + 4096/128 = 160 |
| Total across main layers | 43 | **16,896** |

The current inventory charges **186,368** positions across its two attention entries. Its QK/AV FLOP count at M=1 is **24,427,626,496**, versus **2,214,592,512** under the graph above, about **11.03x** too large. Using the generator's own BF16 KV/query/output byte assumptions and 358.4 GB/s, the M=32 attention subtotal is **17.77664 ms**, versus **2.048 ms** for the corrected single-call graph.

These are controlled accounting counterexamples, not measured latencies or a corrected whole-model prediction. Boundary availability, masks, sinks, and other operations still need explicit treatment. The ratio-128 compressed source is currently absent as a distinct workload, while full-context attention substitutes work the reference does not execute.

**Close when:** derive layer variants from the pinned config sliced to the selected main/MTP scope; test expected counts; derive per-layer window plus compressed availability from the reference; count query/output traffic once for a fused attention call. Test pre-boundary and boundary decode positions rather than assuming every context equals raw sequence length.

### R2. High: Compression Is Neither a Complete Subgraph Nor a Valid Per-Step Cost

The [compressor inventory row](../dsv4_roofline_p2.py#L218) applies one 0.5 ms pooling time to every compressed layer on every step. The [join](../dsv4_roofline_vs_measured.py#L29) identifies its measured pooling shape as `M=32, R=128, D=512`. This cannot describe both compression variants or every batch size.

In the published [Compressor](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L274), key and gate projections execute for each incoming token, but pooling occurs only when `(start_pos + 1) % ratio == 0`. Ratio-4 compression uses overlapping windows, pooling 8 positions at dimension D, not 128. The indexer has its own compressor at D=128 in addition to the main attention compressor at D=512.

For the 43 main layers, synchronized steady decode has main-pool calls only on the relevant boundaries. Amortized over 128 steps, the main pool has **21/4 + 20/128 = 5.40625** calls per step, with distinct shapes, and the indexer pool has **21/4 = 5.25**. These are not interchangeable with 40 identical calls every step. Asynchronous request positions require per-request boundary accounting instead.

The main compressor's `wkv` and `wgate` projections are absent from `OPS`, not just the indexer chain acknowledged by the tracker. With the reference's checkpoint-BF16 storage convention, these matrices alone occupy **520,093,696 bytes** across the main compressors; indexer compressor matrices add **88,080,384 bytes**. These are matrix-only counts, before norms, positional parameters, state, or CPU FP32 expansion. The reference constructor and comments distinguish checkpoint BF16 storage from its FP32 convenience representation.

**Close when:** inventory main and indexer compression separately; distinguish every-token projections/state writes from boundary pooling/postprocessing; choose boundary-specific or explicitly amortized decode reporting; include omitted matrices and their capacity. Mark remaining omissions in the emitted report itself, not only a separate tracker.

### R3. High: Shared Experts Are FP8, Not the Modeled MXFP4

The [shared-expert model](../dsv4_roofline_p2.py#L166) now correctly counts three matrices but applies `FP4=0.53125` bytes/weight. The pinned [MoE constructor](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L617) applies explicit FP4 only to routed experts. The shared expert uses the FP8 default selected by the inference config.

Checkpoint header verification for layer 0 confirms:

| Tensor | Dtype | Shape |
|---|---|---|
| `layers.0.ffn.shared_experts.w1.weight` | F8_E4M3 | [2048, 4096] |
| `layers.0.ffn.shared_experts.w3.weight` | F8_E4M3 | [2048, 4096] |
| `layers.0.ffn.shared_experts.w2.weight` | F8_E4M3 | [4096, 2048] |
| Corresponding scales | F8_E8M0 | [16, 32], [16, 32], [32, 16] |

Across 43 layers, the current shared-expert matrix capacity is **574,881,792 bytes**. FP8 plus its block scales gives **1,082,196,480 bytes**. Keeping all other shared-expert assumptions unchanged, the M=1 time becomes **3.021487 ms**, not **1.605989 ms**; M=32 becomes **3.082424 ms**, not **1.666926 ms**.

**Close when:** bind precision per tensor family to checkpoint metadata. If FP4 shared experts are a proposed requantization target, label that as a separate hypothetical model requiring numerical validation, not the unchanged Flash checkpoint's storage. Routed-expert FP4 metadata accounting is a valid correction and should be retained.

### R4. High: MHC Projection Shape and Precision Are Conflated

The [MHC hc_fn row](../dsv4_roofline_p2.py#L221) is labeled `16384->24` but calls `wgemm` with `NH*HD = 32768` as its input dimension. The reference uses `hc_mult*hidden_size = 4*4096 = 16384`; [Block construction](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L663) is explicit.

Across 86 calls at M=1, this doubles matrix FLOPs from **67,633,152** to **135,266,304**. The checkpoint header independently confirms `layers.0.hc_attn_fn` is **F32 [24, 16384]**, not BF16. The head mixing parameters are also explicitly FP32 in the reference constructor.

Importantly, the doubled K and halved bytes/weight accidentally cancel for hc_fn's fixed weight-byte count: the existing 135,266,304-byte total matches the correctly shaped FP32 matrices. That does **not** validate the row: FLOPs, activation traffic, compute precision, and the claimed BF16 representation remain wrong or unsupported. Merely halving K while retaining BF16 would introduce a weight-capacity underestimate relative to the checkpoint.

The [hc_head formula](../dsv4_roofline_p2.py#L191) also labels RMS normalization, affine/sigmoid gating, and weighted reduction as MODELED, while counting only the projection plus `HC*H` arithmetic and using BF16 weights. Small omitted vector work can be explicitly excluded from an ideal target, but cannot be silently included by the row's name.

**Close when:** separate stored dtype, working dtype, intended compute dtype, and dimensions; derive MHC dimensions from HC and hidden size; account for or explicitly exclude normalization/gating/reduction work. A BF16 conversion is a prospective numerical change unless supported by evidence for the chosen workload. Do not substitute an AMX BF16 compute ceiling as though it were native FP32 execution.

### R5. High: Fixed Measured Placeholders and Incomplete Join Traffic Do Not Establish Roofline Distance

The [measured helper](../dsv4_roofline_p2.py#L116) returns the same latency at M=1 and M=64. Every latency row tested has this property. The tracker calls compressor/top-k/Sinkhorn/combine values measured at grounded shapes, while norm/RoPE/embed/hash values are described as approximate small-op floors. No shape-indexed provenance is attached to these constants in the model.

Measurement at one operating point is valid evidence only for that point and its setup. Relabeling a constant MEASURED does not justify a batch sweep, calling it an ideal floor, or transferring it across the compression variants in R2. Nor does zero FLOPs/bytes establish the actual bottleneck.

The separate [join inventory](../dsv4_roofline_vs_measured.py#L23) has additional analytical inconsistencies:

- Indexer traffic omits the `M*64*4` head-weight read that the main fused-indexer model explicitly counts. Its query is BF16 here but FP4-sized in the main model, without an explicit stored-versus-simulated-precision bridge. FP4 simulation in the reference is not evidence of a packed FP4 query tensor.
- Top-k counts only input logits, omitting selected-index output traffic. Sinkhorn counts only `M*24*4` input bytes and no outputs. Those outputs feed downstream operations; standalone rows cannot silently assume them fused away.
- The compressor expression uses `128*512*4` as its extra term rather than a batch-dependent pooled output size. For the stated pooling input `[M,128,512]`, output is `[M,512]`; any positional-score read must be separately identified. The expression is not an auditable complete boundary inventory.
- MHC combine counts `M*4*4096` operations, only one multiply per input element, without the reduction additions, and omits the combination-weight read. This understates arithmetic intensity even where the chosen BW branch remains unchanged.

Some omitted terms are small and need not reverse a performance decision. They nevertheless make exact reported off-roof ratios unsuitable as correctness evidence. Using nominal DRAM and AMX references is allowed, but a gap alone does not prove overhead dominance or validate a donor-routing choice. Cache-resident versus DRAM-streaming benchmarks also require distinct traffic assumptions.

**Close when:** keep shape-specific observations separate from ideal targets; attach source result, node, revision, dtype, batch, context, timing scope, and summary statistic; use N/A outside measured coverage absent a justified interpolation model. Derive each joined row's inputs/outputs/metadata and arithmetic from its declared boundary. Reconcile the join with the main inventory, making genuinely different workload/precision assumptions explicit. Verification of the measurements and implementations themselves remains deferred.

### R6. Medium: Hardware Provenance Still Contradicts Itself

The [platform profile](../platforms/emr.json) has a useful new measured-reference note for `pcl-sprh02`, DDR5-5600, 277 GB/s and 47 TFLOP/s. However, `host`, scan metadata, and `measured_how` still describe `pcl-spr10`; `machine_peak.note` still names 214.4 GB/s and 40.3 TFLOP/s as a reference FLOOR and calls the gap optimization headroom. These contradict the agreed interpretation and updated fields.

`amx_bf16_ghz_allcore` still presents 1.9 GHz as all-core AMX frequency even though it is a chosen base-clock reference. Its written derivation is **124.5184 TFLOP/s**, not 124.6; the small rounding discrepancy is secondary to the provenance problem.

**Close when:** separate node-specific profiles or remove stale inherited claims; mark topology/capacity provenance explicitly; label the compute assumption nominal, not measured all-core frequency. Keep machine-peak targets and measured observations distinct. No new calibration is required to make these corrections.

### R7. Medium: The Publication Gate Checks Self-Consistency, Not Workload Correctness

The [selftest](../dsv4_roofline_p2.py#L247) verifies useful properties: batch scaling, GEMM traffic, a no-crossing case, three shared-expert matrices, and top-k occupancy. But it never compares the inventory with the reference graph, actual layer indexing, tensor-family dtypes, compression boundaries, or source-backed measured entries. Its capacity assertion checks only that a sum is positive. All eight checks passed despite R1-R5.

The [tracker](../results/roofline_open_items.json) explicitly discloses the indexer compressor sub-chain, runtime state/capacity, and shared-pool fallback as unmodeled. That transparency is useful, but disclosure is not closure. The main compressor omission is not adequately represented, and the report generator neither consumes the tracker nor enumerates its unresolved quantities. Listed-weight fit is explicitly limited and must not be promoted to full runtime feasibility; embedding weight capacity and other unlisted parameters are also outside that sum.

The [model roofline skill](../../../.agents/skills/throughput-enablement/model-roofline-analysis/SKILL.md#L53) still teaches `E*(1-(1-1/E)^(k*B))`, the with-replacement occupancy formula corrected in the generator. It also prescribes achievable ceilings while this exercise explicitly allows nominal targets. The former is a mathematical regression risk; the latter needs an explicit method-selection distinction, not an instruction to abandon the agreed nominal analysis.

**Close when:** add independent config/reference/header-backed tests and enforce tracker coverage at report generation. Align the reusable skill with the corrected distinct-top-k expectation and explicitly separate nominal analytical targets from measured achievable-performance analysis. Keep unresolved costs visible and exclude them from claims of complete totals or rankings.

## Disposition of the Previous Ten Corrections

| Prior correction | Current disposition | Evidence |
|---|---|---|
| 1. Separate hardware profiles | Still open, partly improved | Updated observations coexist with stale node/floor claims; R6 |
| 2. Correct output projections | Verified resolved for the declared analytical shapes/storage | Eight grouped 4096x1024 wo_a matrices with BF16 storage; wo_b K=8192. No donor execution certification implied |
| 3. Correct indexer projections and missing work | Partly resolved, still open | Query K=1024 and weights_proj explicit; compressor chain still incomplete; R1/R2 |
| 4. Shape-dependent ridge crossings | Verified resolved for the declared formulas | Uses full denominator and returns no crossing when nonpositive |
| 5. Attention ownership and fused traffic | Local formula fixed, graph still open | KV scales with independent requests and score materialization removed; R1 |
| 6. Contexts and invocation counts | Still open | Compressed index length improved; main/MTP indexing, attention graph, and compression cadence fail; R1/R2 |
| 7. Consistent fused-indexer accounting | Partly resolved, insufficient precision/boundary evidence | Main row adds query/head weights and removes score DRAM; cross-artifact byte/dtype disagreement remains; R5 |
| 8. Three-matrix shared expert | Matrix count resolved, quantitative row still wrong | Shared-expert FP8 storage modeled as FP4; R3 |
| 9. Occupancy and quantized storage | Partly resolved | Distinct-top-k occupancy and routed MXFP4 scales fixed; family-specific dtypes and reusable skill remain inconsistent; R3/R4/R7 |
| 10. Latencies, capacity, MHC, consistent totals | Still open, partly improved | One inventory and explicit limited capacity are improvements; graph omissions, dtype errors, fixed observations, and incomplete join persist; R2/R4/R5/R7 |

## Reproduction and Verification

Executed successfully on the review host without model weights or accelerator execution:

1. `python3 plugin/validate/dsv4_roofline_p2.py --selftest`: all eight checks pass.
2. Regenerated the report in memory and compared it with the checked-in text after removing its two-line introductory header: exact equality.
3. Imported the actual generator and checked MHC FLOPs, fixed-row batch invariance, attention subtotal, shared-expert bytes, and nominal AMX derivation.
4. Parsed the pinned checkpoint/inference configs and inspected the pinned reference graph, including main/MTP indexing and compression-boundary conditions.
5. Read safetensors metadata through bounded HTTP range requests: verified layer-0 shared-expert FP8 weights/scales and FP32 MHC shape. Other layers were not exhaustively header-audited; the uniform-family extrapolation follows the published constructors/config.

The following small accounting reproduction uses the pinned config and the actual local generator. It is deliberately independent of kernel implementations:

```python
import collections
import importlib.util
import json
import urllib.request

revision = "60d8d70770c6776ff598c94bb586a859a38244f1"
url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/{revision}/config.json"
with urllib.request.urlopen(url, timeout=30) as response:
    config = json.load(response)
ratios = config["compress_ratios"][:config["num_hidden_layers"]]
assert collections.Counter(ratios) == {0: 2, 4: 21, 128: 20}
positions = sum(
    128 + (min(512, 4096 // ratio) if ratio == 4
           else 4096 // ratio if ratio else 0)
    for ratio in ratios
)
assert positions == 16896

spec = importlib.util.spec_from_file_location(
    "roofline", "plugin/validate/dsv4_roofline_p2.py"
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
projection = next(op for op in model.OPS if op.name.startswith("MHC hc_fn"))
expected_flops = 2 * model.HC * model.H * 24 * (2 * model.L)
print("MHC actual/expected FLOPs:", projection.flop(1), expected_flops)
print("Main layer counts actual/expected:",
      (model.N_SWA, model.N_IDX, model.N_COMP), (2, 21, 41))
print("Attention positions actual/expected:",
      model.L * model.S + model.N_IDX * model.IDX_TOPK, positions)
```

## Gate to Resume Implementation Review

1. Correct R1-R4 against the pinned reference and checkpoint metadata; make current versus hypothetical precision choices explicit.
2. Replace unsupported latency sweeps and repair the analytical join in R5. Source-backed observations may remain without being rebenchmarked at this stage, but must not be certified as new measurements.
3. Resolve provenance and gate inconsistencies in R6-R7; test shapes, counts, dtypes, boundaries, and tracker coverage independently of the generator formulas.
4. Regenerate the artifacts, reconcile costs and capacity from the corrected inventory, and publish an explicit list of still-unmodeled quantities. Do not claim a complete whole-model ceiling, runtime fit, or ROI ranking while required contributions remain unaccounted for.
5. Re-review the roofline. Only after it passes should the authored implementations, correctness references, benchmark timing, measured speedups, and serving-path behavior be audited.

**No implementation-correctness or measured-performance approval is given by this report.**