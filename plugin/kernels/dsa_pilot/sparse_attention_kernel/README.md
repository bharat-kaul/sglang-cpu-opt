# Sparse attention on Emerald Rapids

Standalone C ABI / C++ implementation, not enabled in the serving model.
**Faster than the FP32 baseline; mixed-precision accuracy passes; DRAM-bandwidth
saturation has NOT been achieved.** The remaining cost is mostly gathered data
movement / packing, not AMX arithmetic.

## Measured result

2026-10-29, Slurm `emr`, constraint `ddr5600`, node `pcl-sprh05`,
Xeon Platinum 8592+, **one socket, physical CPUs 0-63, memory node 0, no SMT**.
GCC 14.3.1, PyTorch 2.14.1+cpu for reference validation only.
Final measurements: job 394622; sanitizer/determinism: job 394624.
Medians of three independent process medians:

| Same-process paired comparison | FP32 baseline | AMX kernel | Speedup |
|---|---:|---:|---:|
| Hot inputs | 371.966 us | 124.188 us | 3.00x |
| Cold inputs and scratch | 578.224 us | 289.082 us | 2.00x |

The baseline already receives gathered FP32 KV; the AMX timing **includes gather
and packing** from the BF16 source cache. Thus this comparison favors the baseline.
Outputs are allocated by the baseline but preallocated for the C ABI. Allocation
is therefore not identical; the standalone numbers below also exclude allocations.
No dtype conversion is charged to either pre-gathered comparison.
The PyTorch comparison process has 127 OS threads from its runtime pools, all
restricted to physical CPUs 0-63; these are not SMT siblings. The pure C++ driver
uses only its 64-thread OpenMP team and avoids those extra framework pools.

| Pure C++ driver, no PyTorch | Median |
|---|---:|
| Hot inputs | 118.086 us |
| Cold inputs and scratch | 261.612 us |
| Rotating inputs larger than LLC | 312.244 us |

All three beat the supplied **465 us** historical baseline. The cold standalone
result is also below the historical **276 us FP32 compute floor**, but that floor
does **not** apply to BF16 AMX.

- Shape: `B=32, M=1, H=64, D=512, source S=1024, selected K=512`.
- Cold testing writes 768 MiB outside the timed interval, also evicting scratch.
- Streaming rotates 32 distinct address sets: 1 GiB source KV plus Q/output.
  Selected KV alone is 512 MiB, exceeding the socket's 320 MiB LLC. Values are
  repeated across sets, but addresses are distinct; every call gathers and packs.
- Streaming-only IMC counters: **87.72 GB/s**, **26.75 MB/call** actual traffic.
  Counters exclude initialization, cache eviction, and instrumented profiling.
  They are system-wide on an exclusive node, not exact per-process accounting.
- Local STREAM triad: **294.3 GB/s** (job 394601). Useful input/output/index traffic
  is **21,037,312 bytes**, giving an optimistic **71.48 us** memory floor.
  The measured streaming result reaches **22.9% of that useful-byte roofline**.
  STREAM useful bytes and IMC physical traffic are different measurements.
- Work: **2,147,483,648 GEMM FLOPs**. Resident AMX calibration is 142.3 TFLOP/s:
  a 15.09 us compute lower bound, **not** a streamed-GEMM achievable ceiling.
- Frequency was not fixed. Hot/cold/streaming conditions are distinct; do not
  present the hot result as DRAM-bandwidth performance.

[Machine-readable summary](results/summary.json) |
[Roofline report](results/roofline.md) / [chart](results/roofline.svg) |
[Time-attribution report](results/pivot.md) / [chart](results/pivot.svg)

The reports concern **this standalone operator**, not whole-model throughput.
The attribution uses the slowest worker's phase times from one actual median-wall
streaming profile; synchronization, imbalance and instrumentation remain explicitly
unattributed. In that profile, gather/packing/init costs 199.63 us of 297.49 us.
The generic pivot renderer rounds seconds to three decimals; raw microsecond
values are in [native-3.json](results/native-3.json).

## Numerical contract

Reference algorithm:
[DeepSeek-V4-Pro TileLang sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py#L277).
The CPU FP32 oracle is the **untouched** `sparse_attend_fp32bmm_hx` from
`dsv4-tree-pilot`, commit `7305559f0eb6ac9c2be127b8efb2e5bff34712ca`,
path `plugin/kernels/dsa_pilot/sparse_attend.cpp`.
The local tag `sparse-attn-emr-fp32-baseline` preserves that commit.

Q/KV, **unnormalized** softmax probabilities, and output are BF16; scores,
online max/sum, rescaling, and output accumulation are FP32. Blocks contain
64 selected keys. The sink adds denominator mass but no value.

As explicitly agreed, accuracy means mixed-precision parity with measured error,
**not bitwise equality to FP32**. Identical BF16 random inputs are widened exactly
to FP32 for the original reference. Seed 42, normal distribution:

| Comparison | Cosine | Relative L2 | Maximum absolute error | BF16 bit mismatches |
|---|---:|---:|---:|---:|
| Original FP32 oracle | 0.9999978614 | 0.00206812 | 0.00131011 | 374,663 / 1,048,576 |
| Independent 64-block mixed-precision oracle | 0.9999999955 | 0.00009522 | 0.001953125 | 1,534 / 1,048,576 |

Against the original FP32 values, **1,048,556 / 1,048,576 FP32 bit patterns differ**.
The mixed-precision oracle is a separate PyTorch implementation of the TileLang
rounding locations, not a GPU execution of TileLang. Different reduction order,
AMX arithmetic, and the AVX-512 exponential explain residual differences.
There is **no claim of GPU bitwise parity or full-model accuracy**.

Validation covers two normal seeds, sharper inputs (`randn * 4`), partial blocks,
multiple queries, 32/64/96 heads, masked first block, all-masked selections,
duplicates, empty selections, one key, a dominating sink, and invalid index/head
rejection. Acceptance gates in [bench.py](bench.py):

- FP32 oracle: relative L2 < 0.006 and relative maximum error < 0.012.
- Mixed-precision oracle: relative L2 < 0.003 and relative maximum error < 0.008.
- All outputs finite; repeated calls exactly equal.
- Normal-output SHA-256 identical at 1 and 64 threads and under UBSan trap mode:
  `c481c6273fead5a70c907c8047831f2dd8874059083fd3a149fe76fea811b03c`.

All-masked/empty rows return zero, including an initially all-masked block.
This intentionally avoids the undefined `-inf - -inf` behavior in the source.
Very large sinks are handled without exponent overflow.

## Implementation and API

[sparse_attn.h](sparse_attn.h) is C-compatible; [sparse_attn.cpp](sparse_attn.cpp)
has no Torch, BLAS, or oneDNN dependency.

1. Divide `(batch, query, head-group)` into 32-head tasks: 64 tasks at the target shape.
2. Pack the small stationary Q once. Compute **key-major scores**, avoiding a large
   KV transpose for each QK product.
3. Fuse row gathering with AVX-512 VNNI value packing.
4. Use 32x32 AMX microkernels with four FP32 accumulator tiles.
5. Apply FP32 vector softmax/rescaling, transpose/cast probabilities with AVX-512,
   then accumulate the BF16 probability/value product in FP32.

The fast path has no heap allocation. Scratch is about 237 KiB per worker.
Initialize Linux AMX permission with `sparse_attn_emr_init()` before execution and
check every return code. This is an **EMR-specific** entry point, not a portable
dispatcher or silent fallback. The caller owns memory, tensor strides (contiguous
only), shape validation against buffer sizes, and OpenMP/NUMA configuration.
Only `D=512` and positive `H` divisible by 32 are supported. No serving dispatch
was modified; the FP32 path remains unchanged.

## Reproduce

Compile on the login node. `-march=sapphirerapids` selects the AMX/AVX-512 ISA
shared with EMR; do **not** use login-host `-march=native`.

```bash
mkdir -p /tmp/sparse-attn-emr-build
g++ -O3 -std=c++17 -fopenmp -march=sapphirerapids \
  -mprefer-vector-width=512 -fno-math-errno -ffp-contract=off \
  plugin/kernels/dsa_pilot/sparse_attention_kernel/sparse_attn.cpp \
  plugin/kernels/dsa_pilot/sparse_attention_kernel/bench.cpp \
  -o /tmp/sparse-attn-emr-build/bench
```

For multi-node execution use a **shared** build directory, not login-local `/tmp`.
The following full harness builds the native binary, shared library, untouched
FP32 extension, and UBSan trap-mode library. It requires an existing CPU PyTorch
environment with its C++ extension prerequisites.

```bash
git fetch origin dsv4-tree-pilot
PYTHON=/scratch/savancha/emr-roofline-venv-20261008/bin/python
BUILD="$HOME/.cache/sparse-attn-emr-build"
MAX_JOBS=2 "$PYTHON" plugin/kernels/dsa_pilot/sparse_attention_kernel/bench.py \
  --build "$BUILD" --build-only

srun -p emr --constraint=ddr5600 -N 1 -n 1 -c 64 \
  --hint=nomultithread --exclusive -t 00:10:00 \
  env SPARSE_IMC=1 bash plugin/kernels/dsa_pilot/sparse_attention_kernel/run.sh \
  "$BUILD" "$PWD/plugin/kernels/dsa_pilot/sparse_attention_kernel/results" "$PYTHON"

python plugin/kernels/dsa_pilot/sparse_attention_kernel/publish.py
```

The script pins CPUs/memory, records provenance, runs all accuracy checks, three
independent timing processes, and optional IMC counters. `SPARSE_IMC=1` requires
permission for system-wide `perf` counters; failures are explicit. FIFO controls
must be node-local, not on NFS. Rerunning overwrites the result files; preserve
published results if desired. Refresh [calibration.json](results/calibration.json)
when changing hardware or memory configuration. UBSan uses traps because the login
node's GCC UBSan shared runtime is missing.

Losing experiments were removed: whole-next-block L2 prefetch, eight-row staging,
and transparent huge-page promotion did not improve sustained streaming. A
32-thread trial improved some cold timings, but all published main results use
the requested **64 physical cores**. Memory-bandwidth saturation remains future
work; it is not concealed by a hot-cache speedup.
