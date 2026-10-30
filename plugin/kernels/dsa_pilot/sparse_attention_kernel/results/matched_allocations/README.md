# Sparse attention with matched allocation and timing policy

Slurm job **394982**, node **pcl-sprh02.sc.intel.com**, one EMR 8592+ socket,
**64 physical cores (CPU 0-63), NUMA memory 0, no SMT**.

## What is identical

- Both allocate, gather and prepare their inputs **before** timing.
- Both receive contiguous Q[N,H,512], already-selected KV[N,K,512], and FP32 sink.
- Each calls `to(native_dtype).contiguous()` internally, matching the original
  FP32 input-normalization method. These are no-ops for the prepared inputs.
- Both allocate a fresh output tensor using the **same PyTorch CPU allocator**
  inside every timed function call, and release that output before the timer stops.
- Both are measured by the same Python timing helper and pybind interface,
  with randomized paired order and identical CPU/memory binding.
- Cold-cache samples each evict 768 MiB outside the timer.

The FP32 function is copied verbatim from pinned commit
`7305559f0eb6ac9c2be127b8efb2e5bff34712ca`: no `bmm_out`, no preallocated output,
and no modified math. BF16 uses a new allocating wrapper around the unchanged core.

## What remains different by design

Native dtype determines bytes, not allocation policy: FP32 output is 4 MiB;
BF16 output is 2 MiB. FP32 allocates a 4 MiB score tensor inside its first
batched GEMM. The BF16 algorithm does not materialize that full tensor; it
uses worker-local packing and online softmax scratch. No dummy allocation
is added to either algorithm to conceal that difference.
BF16 receives an identity-index tensor prepared outside timing solely for
the existing indexed C ABI; actual source-cache selection is already done.
Its sequential staging/packing remains timed, as does FP32's library workspace.
A warmed allocator may reuse freed pages; fresh allocation does not mean
a fresh OS mapping. Internal BLAS workspace behavior is library-controlled.

## Results

B=32, M=1, H=64, D=512, source S=1024, selected K=512.
Median of three process medians, 51 samples per path/cache condition.

| Condition | Original FP32 (us) | BF16, matched policy (us) | Speedup |
|---|---:|---:|---:|
| cold | 710.949 | 330.163 | 2.15x |
| hot | 349.466 | 133.616 | 2.62x |

| Condition | FP32 process medians (us) | BF16 process medians (us) |
|---|---|---|
| cold | 693.743, 732.310, 710.949 | 330.163, 337.515, 307.083 |
| hot | 349.466, 347.980, 364.226 | 133.616, 132.166, 138.294 |

## Allocation audit and accuracy

Untimed PyTorch memory profiling observed:

| Path | Input dtype-conversion copies | Recorded positive allocations |
|---|---:|---|
| fp32 | 0 | aten::bmm: 4,194,304 bytes, aten::bmm: 4,194,304 bytes |
| bf16 | 0 | aten::empty: 2,097,152 bytes |

All three processes test two normal seeds, sharper inputs, partial blocks,
multiple queries/96 heads, and one key. Both outputs are freshly allocated
and repeatable; the BF16 wrapper is bit-identical to the existing indexed
BF16 core on the same selected values.
Normal-input FP32-reference cosine: **0.9999978614**, relative L2:
**0.00206812**, maximum absolute error: **0.00131011**.
FP32 bit mismatches: **1,048,556 / 1,048,576**.
Accuracy follows the agreed mixed-precision tolerance, not FP32 bit equality.

**Conclusion:** the implementation-level speedup survives matching the
original FP32 allocation/timing policy. This is not an attribution of the
gain to precision alone or AMX alone. No new bandwidth-saturation or
whole-model throughput claim is made.

Evidence: [summary.json](summary.json), [build.json](build.json),
[original FP32 source](fp32_original.cpp),
[replicate 1](replicate-1.json), [replicate 2](replicate-2.json),
[replicate 3](replicate-3.json).
