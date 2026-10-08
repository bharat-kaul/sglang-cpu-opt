# DSv4 EMR Roofline Verdict and Required Corrections

Date: 2026-10-08. Reviewed [roofline report](dsv4_roofline_emr.txt) and [generator](../dsv4_roofline_p2.py) at checkout `a7b0ac1e7914be19a06da0e11fa066ad7131a3bf`, including the report update in `1f5b2ea11c027afe883bdd2006a564dabdd5bbd8`.

Publication scope: the branch advanced to `01b6fc3` before this report was added. Its intervening generator changes affect comments, not the audited accounting formulas. Additional pilot kernels and benchmark results landed in that interval and were not evaluated here; references to new-kernel development describe the audit's methodology, not a claim that no pilot implementations now exist.

## Verdict

The roofline framework and machine-peak-driven optimization methodology are appropriate. However, the published report does **not hold quantitatively as written**: incorrect shapes, operation counts, and traffic accounting invalidate several costs and classifications. Correct those first; changing bandwidth or AMX constants alone will not repair the model.

An ideal roofline is an optimization target, not a prediction of achievable performance. It is sufficient to specify the workload and intended fusion, derive useful FLOPs and justified memory traffic, set nominal hardware targets, optimize, and report the remaining gap when progress plateaus:

$$
t_{\mathrm{ideal}} = \max\left(\frac{B}{BW_{\mathrm{peak}}}, \frac{F}{P_{\mathrm{peak}}}\right).
$$

The target need not be demonstrated attainable. Using AMX at 1.9 GHz is acceptable as an explicitly chosen nominal reference; it is not a measured all-core AMX clock or an unconditional hardware maximum. Conversion, vector work, cache bandwidth, and tile utilization can be investigated when they help explain a plateau rather than being fully modeled upfront. Sustainable-ceiling calibration is **not a prerequisite** for this exercise.

"Below the nominal ridge" means the bandwidth branch dominates this ideal model. It does not prove bandwidth currently limits the implementation: dispatch, conversion, or other work can dominate initially. A measured plateau is a practical stopping point, not proof of optimality.

## EMR Measurement Context

Performance tests ran on **pcl-sprh02**, not the login node, through exclusive SLURM jobs **384146** and **384154** in partition `emr` with constraint `ddr5600`. Both completed with exit code `0:0`. Computation used physical cores 0-63 and memory on NUMA node 0; worker affinity was verified. The machine is a two-socket Xeon Platinum 8592+ with 64 cores and 320 MiB LLC per socket.

The existing STREAM 5.10 source was used unchanged, compiled with GCC 14.3.1, OpenMP, AVX-512, 268435456 double elements per array (2 GiB per array), and 20 iterations. All six STREAM runs passed numerical validation. BF16 GEMMs used an isolated Python 3.12.13 / PyTorch 2.14.1+cpu environment, preallocated outputs, three warmups, and seven timed samples. oneDNN 3.12.0 confirmed `brg_matmul:avx10_1_512_amx` execution with 64 threads.

| Measurement | Observed result across the two jobs |
|---|---:|
| STREAM Triad, 8 threads | 108.7-108.8 GB/s |
| STREAM Triad, 32 threads | 211.3-213.4 GB/s |
| STREAM Triad, 64 threads | 276.7-279.3 GB/s |
| Square BF16 GEMM, best observed | 47.0 TFLOP/s |
| Best sustained GEMM median, dimension 12288 | 46.4-46.5 TFLOP/s |

These measurements are reference observations, not replacement optimization targets or guaranteed operator performance floors. STREAM reports useful-byte bandwidth, not measured memory-controller traffic. Actual AMX-load frequency and DRAM-controller traffic were not measured. The live `intel_pstate/no_turbo` setting was `0`, despite the scheduler's `notrb` tag.

## Ten Required Corrections

### 1. Separate Hardware Profiles

In [the platform profile](../platforms/emr.json) and the generator's reporting functions, distinguish DDR5-4400 `pcl-spr10`, the source of the original measurements, from DDR5-5600 `pcl-sprh02`. Their nominal eight-channel payload ceilings are 281.6 GB/s and 358.4 GB/s respectively. Record host, topology, memory configuration, and measurement provenance instead of mixing observations across configurations.

Retain the chosen machine-peak basis. Label `64 * 1024 * 1.9e9 = 124.5184e12 FLOP/s` as a nominal base-clock AMX reference. Rename "reference FLOOR" to "measured reference" and avoid asserting the entire observed gap is recoverable headroom. A second sustainable-ceiling model is optional, not required.

### 2. Correct Output Projections in `OPS`

Model `wo_a` as eight grouped `4096 x 1024` matrices, producing 8192 outputs per token. Its total weight element count matches the flattened calculation, but its output traffic, grouping, and execution shape do not. Add a `groups=1` parameter to `wgemm`, using `groups*K*N` weight elements and `M*groups*(K+N)` activation elements.

Change `wo_b` input dimension from `OLORA` to **`OG * OLORA`**, or 8192 at TP=1. This increases its weight and FLOP accounting eightfold. For the current CPU implementation, `wo_a` is row-major **BF16**, as selected by `_install_wo_a_logical_unpack` and `_FP8_WO_A_GEMM=False` in [the CPU implementation](../../intel_cpu_models/_dsv4_cpu_infra.py). A prospective FP8 version can be modeled separately as a target, not mislabeled as the current implementation.

### 3. Correct Indexer Projections and Include Missing Work

The CPU indexer calls `c4_indexer.wq_b(q_lora)`. With `QLORA=1024`, change the query projection from `4096 -> 8192` to **`1024 -> 8192`**. Confirm the dimensions against the selected model configuration.

Include `weights_proj(x)`, compressor key/score projections, normalization, RoPE, and Hadamard work. A single `4096 -> 128` indexer `wk` entry does not cover this chain. Describe the intended new CPU kernel rather than assuming a complete optimized indexer already exists.

### 4. Replace the Weight-Only `cross_M` Approximation

Compute the crossing from the same traffic model used by the AI table:

```python
def ridge_crossing(flops_per_token, fixed_bytes, bytes_per_token, ridge):
    denominator = flops_per_token - ridge * bytes_per_token
    return None if denominator <= 0 else ridge * fixed_bytes / denominator
```

For a grouped GEMM, use `flops_per_token = 2*groups*K*N`, `fixed_bytes = groups*K*N*bpw` plus scales/padding, and `bytes_per_token = groups*(K+N)*AB`. Display "no crossing under this traffic model" when the result is `None`.

Under the report's stated traffic model, indexer `wk` saturates at AI about 124 and the router at about 241: neither reaches the nominal ridge of 348. The stated `wqkv_a` shape crosses near M=252, not M=174. Recalculate when fusion, dtype, or cache assumptions change.

### 5. Fix Attention Ownership and Fusion Accounting

For independent requests, the ideal latent K-equals-V attention model needs **`M * effective_context * HD * kv_bpw`** KV bytes, not a single cache shared implicitly across the batch. Count queries and outputs explicitly; use one KV pass only as an optimistic reuse assumption and account for scales and additional traffic where needed.

Remove full `M * NH * effective_context` score-tensor DRAM traffic from the ideal fused case. Retain actual materialization traffic in an unfused model. Make prefix sharing explicit: a common prefix P can give `P + M*(S-P)` unique tokens only where reuse at the modeled memory hierarchy is justified.

Model the current masked shared-pool fallback separately. `_batched_mla_attend` computes against a gathered KV union before masking, so **executed FLOPs can exceed useful per-request FLOPs**. Its compute-bound behavior would not validate a differently defined independent-request ideal model.

### 6. Fix Contexts and Invocation Counts

Pass compressed index length separately from raw sequence length. Ratio-4 indexing at raw context 4096 has approximately **1024 completed compressed positions**, subject to availability and masking, not 4096 index keys. Use `min(index_topk, available_compressed_positions)` for selection and include the layer's sliding-window/tail contribution.

Do not automatically sum full-context attention across 43 layers and an additional sparse-attention pass across 40 layers. Describe the actual or intended combination of sliding-window and compressed sources. Validate compression ratios and per-layer invocation counts against the model; distinguish per-step compression updates from completed-window and amortized work.

### 7. Fix `indexer_fused`

For independent requests and an ideal fused GEMM/ReLU/head-weight/reduction target, use:

```python
matmul_flops = 2 * M * IDX_NH * index_context * IDX_HD
vector_ops = 3 * M * IDX_NH * index_context
dram_bytes = M * (
    index_context * IDX_HD * key_bpw
    + IDX_NH * IDX_HD * query_bpw
    + IDX_NH * 4
    + index_context * 4
)
```

This includes per-request keys, queries, FP32 head weights, and reduced FP32 outputs. Remove the full intermediate-score traffic term when genuinely fused. Track vector epilogue work separately from AMX FLOPs. Extend accounting if top-k selection is fused too.

### 8. Replace the Shared-Expert Single-GEMM Entry

Replace the shared-expert call to `wgemm` with a dedicated three-matrix cost:

```python
matmul_flops = 6 * M * H * MOE_I
weight_bytes = 3 * H * MOE_I * fp4_effective_bpw
activation_bytes = 2 * M * H * AB
dram_bytes = weight_bytes + activation_bytes + actual_spill_bytes
```

The activation expression is the ideal input-once/output-once fused case; charge intermediates where they cross the modeled memory hierarchy. Account for SiLU/multiplication separately and multiply by the verified call count. Under the original 41-layer, FP4-payload-only assumptions, weights alone require **1.439 ms** at 358.4 GB/s, not the published **0.481 ms**. Fusion does not remove two of the three matrices.

### 9. Correct Expert Occupancy and Quantized Storage

For uniform independent requests selecting `TOPK` distinct experts, use **`E * (1 - (1 - TOPK/E)**M)`** in `moe_experts`. The existing with-replacement approximation predicts fewer than six experts for one top-6 request. Nonuniform routing requires explicit probabilities or measured routing distributions.

Include quantization scales and padding in both traffic and capacity. MXFP4 with one one-byte scale per 32 weights occupies **0.53125 bytes/weight**, not 0.5, before other layout overhead. Use actual FP8 scale-block dimensions too. Low-bit storage does not imply native low-bit AMX arithmetic; keep conversion/unpacking distinct from the nominal BF16 compute reference.

Report latency and token throughput, not just whether AI crosses the ridge. Batching can improve expert throughput without changing the nominal bandwidth-bound classification; EP alone does not increase tokens per expert at fixed global batch.

### 10. Replace Invented Latency Floors and Reconcile Totals

In `latency`, `Op.row`, and the output formatting, replace unexplained fixed norm/top-k/compressor/Sinkhorn times with **explicitly unmodeled states**, justified analytical latency bounds, or shape-specific measured models. Do not set FLOPs and bytes to zero and thereby declare latency the binding resource. Reference PyTorch timings are not theoretical floors for new kernels.

Use a consistent tensor/operator inventory in `OPS` and `p0()`. Include missing MHC combine/post/head work, verify the "hash (3 layers)" invocation count rather than multiplying it by 43, and print per-call cost, calls per step, and per-step cost. Do not double-count absorbed operations in fused totals. A `fused=True` label must change the modeled work/traffic or be marked as a proposal.

Use bytes internally and distinguish GB from GiB. Include quantization metadata, packing, dequantized/shadow weights, KV state, workspace, and loading peaks before labeling a quantity total resident capacity. Weight fit establishes feasibility, not that TP=1/EP=1 is performance-optimal. Keep current-implementation and ideal-target tables distinct.

## Therefore Track

For a fixed implementation's work and traffic accounting, increasing achieved bandwidth or FLOP/s is a useful progress metric. Across changes to the algorithm or fusion, it can be misleading. Therefore track:

- **Latency and useful throughput** as the primary outcomes.
- Useful FLOPs, executed FLOPs where relevant, and traffic estimates separately.
- Achieved bandwidth/FLOP/s and distance from the appropriate nominal roof as diagnostic metrics.

Recompute AI and the target when an optimization changes traffic or work. Compare operating points separately when changing batch size or parallelism.

For example, moving 100 MB in 1 ms gives 100 GB/s. After fusion, moving 40 MB in 0.5 ms gives 80 GB/s: the kernel is **twice as fast despite lower achieved bandwidth**. Eliminating redundant arithmetic can similarly reduce executed FLOP/s while improving performance. Require measured progress in latency or useful throughput, not monotonically increasing executed bandwidth or FLOP/s across different algorithms.

## Kernel Development and Validation Scope

Applicable existing DeepSeek-V2/V3 CPU kernels, including MLA components, can be benchmarked as donors at the required shapes and layouts and adapted where necessary. For new CPU kernels such as the indexer and compressor, derive the ideal target, implement standalone kernels, validate against a correctness reference, and optimize those implementations. **Full SGLang integration is not a prerequisite** for analytical rooflines or standalone kernel development; it later checks dispatch, composition, data movement, and end-to-end effects.

The measurements above establish machine reference observations and generic BF16 behavior, not the performance of adapted MLA donors or new indexer/compressor kernels. This limits performance claims about those implementations, not the validity of using nominal machine-peak targets. Acceptance checks should cover correct shapes and counts, independent-batch traffic scaling, explicit sharing, three-matrix experts, quantization metadata, and consistent fused accounting.

This report records required corrections and the agreed optimization methodology. It does not modify the generator, platform profile, or kernels.