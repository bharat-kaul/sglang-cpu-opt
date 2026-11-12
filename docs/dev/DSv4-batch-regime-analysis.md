# DSv4 B1-64: per-kernel analytical coverage

Date: 2026-11-09 (Pacific). Baseline: main `34fe50f1aca4dc9d1ff17111a38ff5f323d7b106`.
Working branch: `dsv4-perf-main`. **Source-derived analysis, not new timing evidence
or an optimality certificate.** Covers the six pilot kernels and the 33 operation
groups / eight substrate surfaces in the earlier
[inventory](https://github.com/bharat-kaul/sglang-cpu-opt/blob/dsv4-standalone-meta-opt/docs/dev/dsv4-native-kernel-inventory.md).

## 1. Answer: representative points, not 64 blind experiments

B1 and B32 can establish a useful scaling hypothesis when the implementation,
operand sharing, cache regime and useful work per row remain fixed. They cannot
alone establish that an implementation or its speedup is uniform through B64.
The source lets us identify the exceptions before running more experiments:

| Family | What the source establishes | Representative evidence needed |
|---|---|---|
| Sinkhorn, fixed hc=4/20 iterations | Same serial recurrence for decode rows 1-64 | Endpoints plus an interior check; retain M16 because the experimental candidate had a shape-specific result |
| Combine / row norms / fixed-width pointwise work | Work and useful bytes linear in rows; row parallelism ramps up to available threads | Low-row point, saturation neighborhood, B64; no need to test every integer |
| Dense packed GEMMs and shared expert | Explicit small-M / BRGEMM dispatch at M4/M5, 32-row blocking | M1,4,5,32,33,64; generic BLAS paths need their own dispatch observation |
| Indexer logits | Integer tile geometry changes with N and thread count | Cover small-tile and large-tile regimes; inspect N31/32 and N63/64 at the fixed 32-thread operating point |
| Top-k | N1 serial; otherwise row/chunk dispatch depends jointly on N, S, k, threads | N1/2, plus both sides of any *reachable* chunking boundary; realistic k/S is essential |
| Compressor pool | Rows are completed windows, not request batch; channel tiling changes | Per-call window counts and R/D variants, including empty and window-completion steps |
| Serving sparse attention | Already N*64 tasks at N1; masked scan length is the global assembled KV length | B endpoints at matched histories, recording global K and visible keys; not just a fixed-K pilot |
| Routed MXFP4 experts | Dispatch's average occupancy stays 1 for decode B1-64; routing/working-set changes remain | Real distinct-expert and occupancy distributions at B1,8,32,64 |

This is **regime coverage**, not a requirement to execute the union of every row
above in a full-model job. Cheap native microbenchmarks test branch boundaries;
full-model runs establish that the representative kernel shapes actually occur
and that their cost matters.

## 2. Boundaries, symbols and roofline accounting

Use the [accuracy-fix configuration](DSv4-accuracy-closure.md): TP1, both clamps ON,
chunk accumulation ON, diagnostic variants OFF, radix OFF. No numerical contract
or default is changed here. The performance screen uses fixed 32 warm threads
instead of the accuracy wave's passive/sticky-eight configuration; compare numbers
only within a stamped runtime configuration.

Model config read from `/scratch/bkaul/models/DeepSeek-V4-Flash/config.json`:

- Hidden width `d=4096`; 43 layers; hc=4; Sinkhorn iterations=20.
- Attention `H=64`, latent/key/value `D=512`, RoPE width=64, sliding window=128.
- Indexer `Hi=64`, `Di=128`, top-k limit=512.
- Main compression: 2 uncompressed, 21 ratio-4, 20 ratio-128 layers.
- Query rank=1024; output groups=8; output rank/group=1024.
- Routed experts `E=256`, top-6, intermediate width=2048; one shared expert.
- Vocabulary=129280; first three layers use hash routing.

`B` is live requests. `M` is actual tokens in a forward: ordinarily `M=B` for
decode, but `M=sum(extend_lengths)` for prefill. Chunked prefill512 means a
B64 prompt batch is many forwards, **not** one GEMM with M=64 or M=64*prompt_length.
`N` in pilot APIs is an operation-local row count. `S` is the indexer key count;
`Kpool` is the assembled attention KV row count; `V_i` is visible keys for query i.
`P=32` denotes the current screen's thread count, not the silicon's core count.

For conventional GEMM use `F=2*M*K*Nout` and
`t_floor=max(F/P_compute, Q/BW)`, with compute precision and the relevant
cache/DRAM level explicit. For exp, divide recurrences, selection and small
dispatch-bound calls, add their issue/dependency/launch limits: a FLOP/DRAM
roofline alone is not the operative ceiling.

All byte formulas below are **logical tensor traffic/footprints**, not measured
DRAM transfers. Cache reuse, write allocation, extra passes and packing can change
physical traffic. A useful starting model is `t(M)=a + work(M)/rate(M)`, with
`rate(M)` piecewise by the source's dispatch and cache boundaries. Two points can
fit `a` and a slope but cannot test that model; a third independent point can.

No current-node, fixed-32-thread achievable ceiling has been measured here.
Historical 280.44 GB/s / 7.14 TF/s observations from job398805 are not transplanted
into absolute forecasts. The analysis below therefore establishes scaling and
candidate mechanisms, not percentages of peak or a quantitative whole-model ROI
ranking. Actual attribution must supply the latter.

## 3. Six pilot kernels: detailed justification

### 3.1 MHC Sinkhorn

[Source](../../plugin/kernels/dsa_pilot/sinkhorn.cpp#L17):
input `[M,1,24]` FP32, outputs 4 pre + 4 post + 16 mix values per row.
Ignoring one-time casts, useful input/output footprint is `192*M + 108` bytes
(shared scale/base included). It stays about 12 KiB at M64.

There are 24 exponentials per row in initialization and, for 20 iterations,
640 matrix normalization divisions per row, plus sigmoid reciprocals and scalar
reductions. The 4x4 recurrence state is tiny. **Recurrence/divide throughput and
call overhead, not DRAM bandwidth, are the relevant hypotheses.**

Default grain is 512: all decode M1-64 stays on one serial task. Arithmetic work
is linear in M with no B16/B32/B64 default dispatch switch. Consequently B1/32/64
is a defensible coarse test of an affine serial curve. An interleaved/SIMD candidate
introduces its own occupancy/tail behavior; baseline linearity does not guarantee
constant candidate speedup. Preserve the prior M16 result and small-M regressions.

**Required controls:** M1,16,32,64 for that existing candidate; empty/strided and
numerical controls remain correctness obligations, not extra performance points.
M512/513 belongs to a separate prefill grain-boundary test only if those call
sizes are in scope. It must not be used to replace the serving-safe decode policy.

### 3.2 MHC weighted combine

[Source](../../plugin/kernels/dsa_pilot/combine.cpp#L14):
`[M,4*4096]` FP32 and `[M,4]` -> `[M,4096]` FP32.
Accumulation performs approximately `2*4*M*4096` FLOPs. Minimum useful tensor
traffic is `4*M*(4*4096 + 4096 + 4) = 81936*M` bytes:
0.0781 / 2.5005 / 5.0010 MiB at M1/32/64; arithmetic intensity about 0.4 FLOP/byte.

The 1024-float accumulator tile is fixed and divides hidden width exactly.
Parallelism is **over M rows**, not over the hidden-dimension tiles. Thus M1
uses one row task; M32 can fill a 32-thread pool; M64 adds another row per worker.
Low-M scaling is not the same as saturated throughput. A warm reused 5 MiB tensor
is not a DRAM-streaming experiment.

**Sampling:** M1,8,32,64, adding M31/33 only if saturation behavior is anomalous.
Include the [wrapper's FP32 conversion and output cast](../../plugin/intel_cpu_models/_dsa_ckernels.py#L220)
when assessing fusion, separately from prepared-input core timing. A core that
already streams once can still benefit from eliminating adjacent passes.

### 3.3 Indexer logits

[Source](../../plugin/kernels/dsa_pilot/indexer_logits.cpp#L73):
`q[N,64,128]`, `kv[N,S,128]`, `weights[N,64]` -> logits `[N,S]`.
Dominant contraction work is `16384*N*S` FLOPs plus head epilogue and conversions.
Replicated FP32 KV occupies `512*N*S` bytes; shared KV would occupy `512*S`.
Packing q costs O(N*64*128); both KV conversion and the epilogue cost O(N*S).

The default integration entry is tiled BRGEMM, not the separate M1/bmm exports.
Its actual epilogue rounds score, product and final sum to BF16 boundaries while
storing FP32 values. Preserve signed weights. Some earlier source comments describe
unrounded scores; the executable epilogue is authoritative
([rounding stages](../../plugin/kernels/dsa_pilot/indexer_logits.cpp#L178)).

For P threads:

```text
tpt = 2 if N >= 64 else 1
tiles_per_row_target = ceil(tpt * P / N)
Sb = min(1024, max(16, round_up(ceil(S / tiles_per_row_target), 16)))
tasks = N * ceil(S / Sb)
```

For fixed S1024 and P32, these are **derived**, not measured:

| N | Sb | Tasks | FP32 score scratch per task |
|---:|---:|---:|---:|
| 1 | 32 | 32 | 8 KiB |
| 2 | 64 | 32 | 16 KiB |
| 4 | 128 | 32 | 32 KiB |
| 8 | 256 | 32 | 64 KiB |
| 16 | 512 | 32 | 128 KiB |
| 31 | 512 | 62 | 128 KiB |
| 32 | 1024 | 32 | 256 KiB |
| 63 | 1024 | 63 | 256 KiB |
| 64 | 1024 | 64 | 256 KiB |

The N64 `tpt` branch **does not change Sb at P32/S1024**. A textual branch is
not automatically a timing cliff. Conversely N31/32 does change tile geometry.
Do not infer exact cache residence from scratch alone: include KV, packing,
other workers and the node's actual cache topology.

**Sampling:** N1,8,16,31,32,64 at representative S; inspect residuals before
expanding. Recompute the table if P or S changes. Test the actual short-S serving
calls too; a fixed S1024 pilot does not cover them.

Serving currently constructs contiguous `[N,S,128]` from shared `[S,128]`.
The avoidable replicated *storage* is `512*(N-1)*S` bytes, so this opportunity
is zero at N1 and increases with N. At N64/S1024 the buffer is 32 MiB versus
0.5 MiB shared. Eliminating it does not eliminate N-fold dot-product arithmetic.
With chunked prefill, N and S are determined by current token/window counts,
not directly by B. If S itself grows with N, replication becomes quadratic;
the screen must identify the actual N/S pairs before assigning benefit.

### 3.4 DSA top-k (not expert-router top-k)

[Source](../../plugin/kernels/dsa_pilot/indexer_topk.cpp#L99):
FP32 logits `[N,S]` -> int64 indices `[N,min(k,S)]`.
Expected selection work is O(N*S), plus index initialization and candidate merges;
comparisons, data order and dispatch dominate a simplistic FLOP roofline.

Exact default branch conditions:

- k=0: allocation/empty return.
- N=1: serial.
- N>=P: row-parallel.
- Otherwise, `C=floor(P/N)`; use chunked selection only if `2*k <= floor(S/C)`.

At S1024/k512/P32, effective multi-chunk selection does not occur: C>=2 fails
the guard; C=1 degenerates to row-parallel. The important low-batch boundary is
N1/2, not a vague assumption of smooth scaling. At S4096/k512/P32, multi-chunk
selection becomes possible at N7 (C4), persists through N16 (C2), then C1
degenerates to row-parallel. These S values are API-domain examples, not claims
that current chunked serving produces them.

**Sampling:** N1,2,8,32,64 for the first shape; add N6/7 and N16/17 only if the
second shape is exercised. Include k=S, masked/tied scores and different orderings
in correctness/workload checks; arbitrary tie identity is not proven by top-k
set comments.

The [serving wrapper](../../plugin/intel_cpu_models/_dsa_ckernels.py#L172)
executes the native selector twice. Removing one call is a concrete work-removal
candidate across nonempty shapes, but "half the selector calls" is not "2x the
model." Preserve observed output/tie behavior and measure wrapper cost separately.

### 3.5 Compressor softmax-pool

[AVX-512 serving export](../../plugin/kernels/dsa_pilot/compressor.cpp#L354):
`kv,score[Nw,R,Dc]`, `ape[R,Dc]` -> `[Nw,Dc]`.
Useful tensor footprint is `8*Nw*R*Dc + 4*R*Dc + 4*Nw*Dc` bytes.
Online recurrence performs approximately Nw*R*Dc exponentials (one per channel
update); a plain GEMM FLOP peak does not describe this kernel.

The intrinsic path partitions channels using
`dpt=ceil(2*P/Nw)`, `DT=round_up(max(16,ceil(Dc/dpt)),16)`,
`tasks=Nw*ceil(Dc/DT)`. For P32/Dc128:

| Nw | DT | Tasks |
|---:|---:|---:|
| 1 | 16 | 8 |
| 4 | 16 | 32 |
| 8 | 16 | 64 |
| 16 | 32 | 64 |
| 31 | 48 | 93 |
| 32 | 64 | 64 |
| 63 | 64 | 126 |
| 64 | 128 | 64 |

**Important inconsistency:** the scalar fallback has a `Nw*R*Dc < 512*1024`
serial-grain rule. The AVX-512 intrinsic core called by the wrapper does **not**;
it uses grain0. The fallback's policy cannot justify the intrinsic path's curve.

In the current indexer caller, pooling is invoked per request with
`Nw=floor(extend_length/4)`, R4, Dc128. B32 therefore need not mean Nw32; a
512-token chunk can produce Nw128. Empty/incomplete windows can skip the call.
Main corrected compression instead uses eager pooling, with ratio4 overlap
equivalent to R8/D512 and ratio128 to R128/D512; those are not this native dispatch.

**Sampling:** observed Nw1,4,8,32,63,64,128 where reachable; maintain separate R/D
families if comparing a future native main-pool implementation. Decode tests need
window-completing and non-completing positions, not just another B.

### 3.6 Sparse attention: two different kernels

**Historical pilot:** `[N,64,512]` q and `[N,K,512]` selected KV.
[bestof_hx](../../plugin/kernels/dsa_pilot/sparse_attend.cpp#L155) selects scalar
online attention for N1 and FP32 bmm + vector-exp for N>=2.
Two GEMMs cost approximately `4*N*64*K*512` FLOPs, excluding softmax.
At K512, q+KV+output footprint is about 1.25 MiB per N before score scratch.
The GEMM's inner matrix row dimension is H64, not request batch B.
**Sampling:** N1,2,8,32,64, fixed K plus any observed K/cache boundary.

**Actual serving:** [sparse_attend_masked](../../plugin/kernels/dsa_pilot/sparse_attend.cpp#L389)
accepts shared `[Kpool,512]` KV and `[N,Kpool]` mask. It always uses the same
FP32 online recurrence, parallel over N*64 heads. There are already 64 tasks at
N1 on a P32 pool; it is not initially starved for row parallelism like combine.
Per visible key/head, the dot product plus rescaled accumulator update costs
approximately 5*512 FLOPs, plus two scalar exponentials and normalization work.
The kernel checks every mask entry per head, including false entries:

```text
mask checks = N * 64 * Kpool
useful arithmetic ~ 5 * 64 * 512 * sum_i(V_i)
shared KV footprint = 4 * Kpool * 512
q + output footprint = 8 * N * 64 * 512
mask allocation = N * Kpool bytes
```

At N64, q+output alone is 16 MiB. As independent request histories accumulate,
Kpool can grow with B while each query sees only its own keys. Then mask scanning
can grow as B squared even when visible arithmetic grows linearly. This is a
conditional complexity result, not a measurement that scanning dominates.
Record Kpool, visible count, context, request ownership and cache lifecycle.
Do not replace V_i with 512: the corrected compressed-handoff mask is built
from request identity and causal coverage, not simply the standalone selected-K
tensor. See [assembly/masks](../../plugin/intel_cpu_models/_dsv4_cpu_infra.py#L2124).

**Sampling:** B1,8,32,64 at matched per-request contexts, including ragged histories.
Standalone fixed-K results cannot prove the serving curve. The newer BF16
indexed donor remains a separate, mixed-precision prior result; its speed is not
assigned to this FP32 masked contract.

## 4. Other high-value deductions

### Compressor projection: fixed conversion cost can favor low batch

The installed `Compressor` builds a BF16 weight `[2*coff*D,4096]`, with coff2
for C4 and coff1 for C128. Ordinary CPU `linear_bf16_fp32` evaluates
`mm(x.float(), weight.float().t())`, returning FP32.

| Projection | Nout | FP32 weight conversion destination per call |
|---|---:|---:|
| Main C4 | 2048 | 32 MiB |
| Main C128 | 1024 | 16 MiB |
| Indexer C4 | 512 | 8 MiB |

Under one projection call per relevant layer per forward, this is
`21*32 + 20*16 + 21*8 = 1160 MiB` of conversion destinations, independent of B.
This is not 1160 MiB of measured DRAM traffic or persistent allocation.
BF16 reads plus FP32 writes for the weight conversion alone are `6*K*Nout`
bytes; GEMM subsequently reads the widened operand. Compute is `2*M*K*Nout`,
so a conversion-only accounting ratio is M/3 FLOP/byte. It grows from 0.33 at
M1 to 21.33 at M64; it is **not the full GEMM arithmetic intensity**.

Thus B1 can overstate conversion's share at B64 even when total conversion
bytes stay fixed. A native BF16-input / FP32-output route may remove those casts,
but output dtype alone does not guarantee equal accumulation/numerics. Do not
cache widened model weights as a shortcut. **Sample M1,8,32,64; separate casts,
GEMM, allocation and preparation, with accepted precision boundaries intact.**

### Routed versus shared MoE: different dispatch axes

Installed native `moe_fp8.cpp` sets
`avg_M=max(1,floor(M*topk/E))`, then uses BRGEMM only when avg_M>4 for packed
uint8 MXFP4. For decode M=B, topk6, E256, avg_M is **1 at every B1-64**.
The first crossing is M214, relevant to prefill rather than this decode interval.
The clamp loader patches the activation, not this dispatch.

This corrects an overly general argument that routed decode necessarily crosses
the M4/M5 AMX boundary when request batch does. It does not on this source path.
Nevertheless routing affects the number of active experts, per-expert occupancy,
32-row padded blocks and locality. A weight-stream model must use actual routing:

```text
F_routed ~ 6 * M * topk * 4096 * 2048
Q_weights >= distinct_experts(M) * bytes_per_expert
per-expert packed weight payload = 3 * 4096 * 2048 / 2 = 12 MiB
```

Scales and repeated block reads add to that payload. It is not necessarily read
exactly once per active expert. Do not substitute the uniform-routing occupancy
formula or fit a universal curve from a single historical B32 routing count.

The **shared** FP8 expert uses `can_use_brgemm(M)` directly, hence M4/M5 is a
real dispatch boundary. Packed dense BF16/FP8 projections have the same threshold
in their native donor; plain FP32/BF16 BLAS/einsum paths are not automatically
governed by that helper. Both paths use 32-row blocking.

**Sampling:** routed B1,8,32,64 with per-layer routing histograms and hashes;
shared/dense M1,4,5,32,33,64. Hold the clamp configuration fixed.

## 5. Complete operation-group accounting

Each row gives a scaling justification and the observation that prevents an
unsupported extrapolation. These are operation groups, not 33 independent
hand-written kernels. Core details above take precedence over shorthand here.

| ID | Operation | Shape/work law over the B range | Representative test / falsifier |
|---|---|---|---|
| C01 | Embedding | Gather M rows of width4096; O(M*d) bytes, token-dependent reuse | B1/32/64 with distinct and repeated IDs; cache reuse invalidates one universal bandwidth rate |
| C02 | hc expansion | Copy M*d to M*4*d BF16; linear bytes | B1/32/64; allocation/thread overhead may dominate small M |
| C03 | RMSNorm modules | Row reductions, O(M*width), native row-parallel | B1/8/32/64 at each actual width; distinguish residual-fused form |
| C04 | Attention/indexer dense projections | Fixed K/Nout weights; 2*M*K*Nout; weight cost amortizes | Packed donor M4/5 and 32/33; observed dtype/prepack required |
| C05 | Query norm + RoPE | M*64 rows of D512; FP32 reductions and trailing64 rotations | B1/32/64, multiple positions; already many head rows at B1 |
| C06 | Key norm + RoPE | M rows of D512; eager passes, O(M*512) | B1/8/32/64; unlike query norm, only M independent key rows |
| C07 | Quantize/pack/write KV | O(M*512), per64-channel scale reductions plus indexed stores | B1/32/64; page/slot boundaries and repeated slot history are independent axes |
| C08 | Compressor projection | Fixed 8/16/32 MiB FP32 weight conversions plus GEMM | M1/8/32/64, each projection family; cast share changes with M |
| C09 | Main pooling | Sum completed windows, overlap R8 or R128, D512 | Per-call window counts and positions just before/at completion; B alone insufficient |
| C10 | Compressed norm/RoPE/roundtrip | O(completed_windows*512), staged rounding | Same window coordinates as C09; do not erase BF16/FP8 boundaries |
| C11 | Index query RoPE/Hadamard | RoPE O(M*64*64); dense Hadamard matmul about 2*M*64*128^2 | B1/8/32/64; native fast transform would be a new numerical/reduction-order proposal |
| C12 | Indexer pooling | Nw=floor(Lext/4) per call, R4/D128; tiled exp recurrence | Window-count table in section3.5; empty decode is absence, not zero-cost optimization |
| C13 | Indexer key norm | O(S*128), concat/cast/reduce | Actual S from C12, not B; changing S family invalidates fixed-B inference |
| C14 | Indexer logits | 16384*N*S FLOPs; replication O(N*S*128) | Section3.3 tile regimes and actual shared-KV expansion sizes |
| C15 | Indexer masks/handoff | Dense causal mask O(N*S), selected-position scatter/repeat | N/S joint coordinates, ties/masked rows and per-request positions |
| C16 | DSA top-k | Expected O(N*S), row/chunk/serial branches; duplicate native call | Section3.4; test work removal at wrapper boundary |
| C17 | KV gather/assembly/masks | O(Kpool*512) gather plus O(N*Kpool) masks/copies | B1/8/32/64 at same context and lifecycle; global Kpool may grow with B |
| C18 | Sparse attention | Visible compute O(64*512*sum(V_i)); mask scan O(N*64*Kpool) | Section3.6; record mask density, not just q shape |
| C19 | Inverse output RoPE | O(M*64*64) rotations plus views/copies | B1/32/64, inverse sign and positions preserved |
| C20 | Grouped wo_a | 8 groups, K4096->R1024; about 2*M*8*4096*1024 FLOPs; BF16 weights64 MiB | M1/4/5/32/64; plain einsum's executed library path must be observed, not inferred from packed donor |
| C21 | wo_b | M*8192 -> M*4096, standard packed projection | Packed M4/5,32/33; TP reduction absent in this TP1 scope |
| C22 | MHC pre statistics | Upcast and reductions over M*16384, scale 24 mixes/row | B1/32/64; separate statistic passes from C23 GEMM |
| C23 | MHC mix projection | FP32 K16384->Nout24, 786432*M FLOPs; weights1.5 MiB | M1/8/32/64; skinny library GEMM may be issue-bound, not AMX roofline |
| C24 | Sinkhorn | Serial fixed20 recurrence for M1-64 | Section3.1; M16 candidate result cannot be averaged away |
| C25 | Combine / following norm | Linear useful traffic; M-row parallel; wrapper casts | Section3.2; fusion must include boundary casts and accepted rounding |
| C26 | MHC post | FP32 `[M,4,4]` mixing `[M,4,4096]`, then scale/add/cast | B1/8/32/64; library small-bmm and eager launches versus existing gated donor |
| C27 | Final hc head | Upcast/statistic, K16384->4 projection, sigmoid/weighted sum | B1/32/64; once per forward, not 43 times |
| C28 | Router projection | M*4096 -> M*256, about 2*M*4096*256 FLOPs | M1/4/5/32/64; establish its FP32/native dispatch, not presumed low-bit |
| C29 | Normal routing | sqrtsoftplus/score adjustment/top6/normalize on `[M,256]` | B1/8/32/64 with real logits; existing softmax/sigmoid fused top-k does not cover this scoring function |
| C30 | Hash routing | First3 layers, lookup/gather plus scores/normalization | Same B anchors, token-ID diversity; separate from C29 and DSA selection |
| C31 | Routed MXFP4 experts | avg_M=1 throughout decode B1-64; active experts and padding grow | Real route histograms at B1/8/32/64; M214 only for prefill branch coverage |
| C32 | Shared expert/merge | One fixed FP8 expert; GEMM amortization + activation/merge O(M) | M1/4/5/32/33/64; keep shared clamp ON and routed output fixed |
| C33 | LM head / logits | For generation, selected hidden rows -> vocab129280; rows often B, not all prefill M | Actual logits-row count and dtype; B1/4/5/32/64; logits output alone32 MiB at B64 FP32 (rounded) |

The C33 FP32 output is exactly 31.5625 MiB at B64; return-logprob and other
generation options alter the postprocessing scope. Eager compositions already
call native ATen primitives; "not a fused C++ op" does not mean scalar Python
arithmetic or an automatic performance loss.

### Serving substrate: independent axes, not missing arithmetic kernels

| ID | Surface | Analytical disposition |
|---|---|---|
| R01 | Registration / backend selection | Constant setup plus dispatch; verify selected implementation at every tested joint coordinate |
| R02 | Loading / packing / capacity | Mostly one-time; exclude from steady kernel timing, retain startup and footprint separately; packed weights stay packed |
| R03 | Allocator / request-token indices | Request count, token count and page crossings; B endpoints do not cover fragmented/free-list state |
| R04 | Positions / compressed metadata | O(requests+tokens+windows) plus eager launches; cover prefill/decode and compression remainders |
| R05 | Dense shadow / packed cache validity | Growth and gathers depend on live slots/history, not merely B; measure copies and preserve ownership |
| R06 | Compression carry / handoff | Work spikes at ratio4/128 completion; fresh versus continued histories are separate coordinates |
| R07 | Threads / NUMA / collectives | Fix P/affinity/wait policy for curve attribution; retune serving policy separately; TP>1 excluded |
| R08 | Dispatch evidence / fallback controls | Verify successful native return, not attempted-call counters; absence of a call is not a speedup |

## 6. Minimal campaign design and falsification rules

1. **Start with the existing screen, not more model loads.** Job401145 uses
   B1/32, input128/3072, G8, fixed32 warm threads, and three repeated request
   pairs. It is attribution and probe-perturbation evidence, not the B1-64
   completion gate or three independent process replicas.
2. **Use native microbenchmarks for source boundaries.** Core common points
   M1,8,32,64; add 2 for top-k/pilot sparse, 4/5 for packed dense/shared,
   16 for Sinkhorn candidate, 31/32 for indexer and 63/64 for compressor.
   Prefill uses recorded M/Nw/S, including Nw128 and M214 if relevant.
   These sets are a starting design; measured unexplained curvature triggers
   local refinement, not a blind entire grid.
3. **Add a B64 serving anchor with a valid capacity configuration.**
   B64*(3072+8)=197120 tokens exceeds the current 131072-token pool.
   Requesting that workload unchanged can induce scheduling/retraction and
   does not establish a steady B64 kernel shape. Use a matched, shorter-context
   B1/32/64 panel (for example context1024), or increase the pool only after
   memory-capacity validation. Never silently compare B64 at a different context.
4. **Probe compression boundaries deliberately.** At context128/G8, ratio4
   completes windows during decode; ratio128 does not complete its next window.
   Neither does context3072/G8. Add positions immediately before/at a ratio128
   boundary to test that branch; do not use absence in this screen to close it.
5. **Use actual operands and three independent process replicas for accepted
   performance comparisons.** Retain per-process samples, median/spread,
   shape/dtype/strides, source/build hashes, threads, cache policy, masks and
   routing. Check baseline and candidate through the same full wrapper.
6. **Interpolation is a tested model, not an optimality claim.** Within a
   source-constant regime, fit the declared work-law model and hold out an
   interior point. A systematic residual above 5%, an unexpected dispatch,
   cache-size crossover, or differing workload identity defeats the simple
   interpolation; explain/refine before reporting a range-wide result.
   A >10% *measured* improvement is material; <5% remains the campaign's
   practical noise convention, not a statistically established noise floor.
7. **Separate local speed from whole-model opportunity.** For measured share s
   and component speedup r, ideal whole-phase speedup is
   `1/(1-s+s/r)`. Ideal deletion exceeds1.10x only when s>1/11 (~9.09%);
   a 2x component requires s>2/11 (~18.18%). Cost buckets overlap and must not
   be added. Interactions can invalidate the bound's fixed-other-cost assumption.

## 7. Evidence ledger and inconsistencies

| Claim | Evidence / estimator | What would falsify or limit it | Disposition |
|---|---|---|---|
| Sinkhorn serial through decode64 | Current source grain512, arithmetic enumeration M1-64; no timing estimator | Override flag / different binary / larger actual M | Source-established under declared defaults |
| Indexer effective tiles vary with N | Integer formulas and appendix table, current source | Different P/S or another export selected | Source-established geometry, performance hypothesis |
| Intrinsic compressor lacks scalar serial guard | Compare scalar lines52-55 with intrinsic lines299-305 and wrapper export | Non-AVX512 build uses fallback | Do not transfer scalar policy to AVX512 |
| Routed decode stays avg_M1 | Installed donor E from weight count, topk6, exact integer enumeration | Expert sharding, changed donor or different actual M | Source-established at TP1/B1-64, not proof of executed ISA |
| Shared/dense M4/5 boundary | Installed gemm.h and respective donor call sites; clamp loaders inspected | Different quant method or BLAS path | Apply only to those donors |
| Sparse scan can be quadratic in B | Shared Kpool loop and request-assembled histories | Kpool bounded independently of B, or attention called per request | Conditional scaling, no measured dominance claim |
| Compressor FP32 conversion destinations1160 MiB | Model counts and producer widths, arithmetic sum | Skipped/changed calls, cached/conforming alternate GEMM path | Logical volume, not measured DRAM |
| Broad standalone >10% improvement absent from campaign evidence | Existing experiment record; no new timings here | A complete qualified candidate sweep | Not an exhaustion/optimality claim |
| Sinkhorn M16 ~18% lower latency | Prior job399421, median of three process medians, narrow shape | Other M or serving overhead | Prior experimental result, not a range-wide gain |
| Job401095 cannot establish the model curve | Only B1/S128 warmups + one measured pair; 58-thread prefill pathology | No complete matrix/replicas | Rejected baseline; retained diagnostic only |

Additional reconciliation:

- Indexer comments claiming unrounded FP32 scores are stale relative to explicit
  BF16 rounding in the executed epilogue. Do not model a different precision path.
- The standalone sparse N1/N2 dispatcher is not the serving masked kernel.
- The earlier warm decode recommendations did not justify restoring 58 warm
  threads for every prefill. Recovery from that harness error is not a new gain.
- B1/32 were selected for a serving screen, not measured coverage of every kernel.
  Their representativeness must be justified per family as above.
- No fixed32 achievable roofline, complete timing matrix, universal no-regression
  result, or default promotion is established by this analytical report.

## 8. Source provenance and reproduction

### Historical review reconciliation

The initial 2026-11-06 review inspected public `d3bbc96` and the separately dirty
accuracy worktree at `9320cd1`. These are historical findings, not a fresh claim
that each defect survives in current main. The
[native inventory](https://github.com/bharat-kaul/sglang-cpu-opt/blob/dsv4-standalone-meta-opt/docs/dev/dsv4-native-kernel-inventory.md)
retains source identities and the later fusion-prework reconciliation.

| Historical inconsistency | Retained lesson / later disposition |
|---|---|
| Six named native kernels presented as complete fused serving coverage | Audit full operations and callers; eager norm/RoPE, corrected compression and KV preparation were outside that six-name claim. |
| Duplicate native top-k call hidden behind one counter; MHC strict exceptions swallowed | Count successful calls and inject failures through actual wrappers. These are qualification defects, not established causes of an ordinary accuracy gap. |
| Standalone sparse donor and serving masked-online path described interchangeably | Preserve distinct signatures and preparation costs; the subsequent C4 campaign measured the serving interface instead. |
| Indexer replication and compressor casts looked large from source | Logical allocation bytes are not measured DRAM or model headroom; serving attribution deprioritized some of these leads. |
| Mask-A/B result called neutral/overlapped after its worker-control bug invalidated it | Invalid evidence cannot establish either a speedup or a null result. |
| Low-row MXFP4 called AMX; old passive/eight-thread advice treated as universal | Use actual shape-dependent ISA and stamped runtime. Later warm-thread measurements are a different operating point. |
| Roofline denominators mixed domains/sockets, nominal/measured peaks and cache/DRAM | Preserve node, ISA, threads, working-set size and estimator for each number. A 192 MiB triad is not a DRAM calibration on a 320 MiB LLC. |
| Regime table used BF16/FP8/FP4 crossover 190/95/48 with ridge about 380 and AI coefficients 1/2/4 | Those stated equations instead give about 380/190/95; actual operator/ISA analysis is still required. |
| Published all-M floor contradicted by top-k M2; Sinkhorn timing used old dispatch grain | Reproduce the current source and retain the counterexample; no whole-range optimality follows. |
| CPU/GPU subset percentages compared on different question sets; residual declared benign too early | Superseded by the matched full-set accuracy closure, not numerical equivalence. Partial monitoring must use identical IDs. |
| Cooperative fusion/whole-layer gains inferred from spin samples and small demonstrators | Samples are not critical-path elapsed savings. The inventory corrects the Scope-A mechanism; whole-layer gains remained unmeasured. |
| Raw-stream MoE headroom repeated after unpack-adjusted estimates revised it | Preserve the stricter denominator, actual routing and later rejected/neutral experiments; do not re-propose existing fused dequant or vector work as missing. |
| Early GAP/TODO metadata and broad fused/bit-exact labels outlived the implementation stage | Version the inventory, distinguish planned versus executed fusion, and require byte equality for bit-identity claims. |

### Reproduction anchors

Repository sources are pinned by main34fe50f. The local infrastructure has only
the existing instrumentation changes; normalized-AST equality to the frozen
accuracy source was tested. Native pilot files and dispatch wrapper are unchanged.
Installed CPU sources (not automatically identical to every historical binary):

| Source under installed `sglang/` | Lines / role | SHA256 |
|---|---|---|
| `kernels/aot/csrc/cpu/gemm.h` | 12: block32; 23-43: dispatch thresholds | `1abe4fd1185873bba5485de79d4db8eb4e99ad36f3f92b3b7f1e32145b56285f` |
| `kernels/aot/csrc/cpu/moe_fp8.cpp` | 58: routed avg_M; 284: shared M dispatch | `aa1649f138f19010c61ecbffd7bcc45a29c510d7450b3ec96c1033b418a95035` |
| `srt/models/deepseek_v4.py` | 405: grouped wo_a; 494: final hc; 1981: pre statistics/mix | `7f2e6f75707c44311cfa3c398f6f708c7a8d5af17c3d55ec99f6af2996d07437` |
| `srt/layers/attention/dsv4/compressor.py` | 355-366: overlap/projection width; 457: projection call | `ec75b5f042100fae04c45d569765101912e083362e491a8375e0127909a00d9f` |
| `kernels/ops/attention/dsv4/gemm.py` | 115: CPU FP32 casts/mm | `7d08628d9f298c404afe3201552b8c39df5a1f8718db2c1f3dc36b3a2659d428` |

Installed root:
`/scratch/bkaul/venvs/sglang-cpu/lib/python3.12/site-packages/sglang`.
Config SHA256:
`b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818`.
Clamp patches preserve the inspected dispatch:
[routed loader](../../plugin/intel_cpu_models/_dsv4_moe_clamp.py#L60),
[shared loader](../../plugin/intel_cpu_models/_dsv4_shared_clamp.py#L65).

The following standard-library calculation reproduces geometry and checks the
specific boundaries discussed above. It is not a timing test or native-kernel
correctness test:

```python
def ceildiv(a, b):
    return (a + b - 1) // b

def indexer(n, s=1024, p=32):
    tpt = 2 if n >= 64 else 1
    tpn = max(1, ceildiv(tpt * p, n))
    sb = min(1024, max(16, ceildiv(ceildiv(s, tpn), 16) * 16))
    return sb, n * ceildiv(s, sb)

def compressor(n, d=128, p=32):
    dpt = max(1, ceildiv(2 * p, n))
    dt = ceildiv(max(16, ceildiv(d, dpt)), 16) * 16
    return dt, n * ceildiv(d, dt)

def topk(n, s, k, p=32):
    k = min(k, s)
    if k == 0:
        return "empty"
    if n == 1:
        return "serial"
    if n >= p:
        return "row"
    c = max(1, p // n)
    if 2 * k <= s // c and c > 1:
        return "chunk"
    return "row"

rows = [
    (n, *indexer(n), *compressor(n), topk(n, 1024, 512),
     max(1, n * 6 // 256))
    for n in range(1, 65)
]
for row in rows:
    if row[0] in (1, 2, 4, 8, 16, 31, 32, 63, 64):
        print(row)
assert {row[-1] for row in rows} == {1}
assert next(m for m in range(1, 1000) if max(1, m * 6 // 256) > 4) == 214
assert indexer(31) == (512, 62) and indexer(32) == (1024, 32)
assert indexer(64) == (1024, 64)
assert compressor(63) == (64, 126) and compressor(64) == (128, 64)
assert all(topk(n, 1024, 512) != "chunk" for n in range(1, 65))
assert topk(6, 4096, 512) == "row" and topk(7, 4096, 512) == "chunk"
assert topk(16, 4096, 512) == "chunk" and topk(17, 4096, 512) == "row"
assert (21 * 4096 * 2048 * 4 + 20 * 4096 * 1024 * 4
        + 21 * 4096 * 512 * 4) // 2**20 == 1160
assert 64 * (3072 + 8) > 131072
assert 64 * 129280 * 4 / 2**20 == 31.5625
```
