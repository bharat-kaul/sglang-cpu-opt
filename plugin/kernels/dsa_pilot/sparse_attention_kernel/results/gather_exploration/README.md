# AVX-512 gather and packing exploration

Job **395103**, **pcl-sprh16.sc.intel.com**, EMR Xeon 8592+, one socket,
**64 physical cores / CPUs 0-63 / NUMA memory 0 / no SMT**.

## Conclusion

**Keep the current kernel.** More explicit pointer SIMD yields a sub-1%
change here; alternative unpack/shuffle packing is effectively tied.
Hardware gather plus transpose is slower, and removing prefetch hurts cold inputs.
These are results for this workload and the tested alternatives, not a claim
that every possible permutation/gather strategy has been exhausted.

## What the current code actually does

Lines 144-148 select pointers, not BF16 payloads. Per 64-key block they read
256 bytes of indices and select 64 row addresses. Each selected row then
supplies 1024 contiguous bytes (512 BF16 values), for 64 KiB of KV payload.
The following loop already uses 512-bit loads and two
`_mm512_permutex2var_epi16` operations per pair of row segments to produce VNNI.
Permutes rearrange values already in registers; they cannot fetch arbitrary
source rows without loads or gathers.

Disassembly of the baseline staging function confirms that GCC already
auto-vectorizes the pointer setup: masked `vmovdqu32`, `vpmovsxdq`,
`vpsllq $0xa`, masked `vpaddq`, and vector stores. Source-level scalar
syntax does not imply scalar machine execution.

## Variants

- **baseline:** unchanged production source, plus a test-only staging export.
- **vector_pointers:** explicitly build eight 64-bit row addresses using
  AVX-512 masked loads, sign-extension, shifts, adds and mask blending.
- **unpack_shuffle:** replace two word permutes with two lane-local
  word unpacks and two qword permutes. Loads and prefetch remain unchanged.
- **hardware_gather:** gather BF16 pairs across 16 selected rows using
  `vpgatherdd`, transpose 16x16 dwords, then VNNI-pack. No prefetch.
  This is an experiment restricted to the tested S<=1024 domain.
- **no_prefetch:** original loads/packing with software prefetch removed
  as a control to distinguish gathering from prefetch effects.

Hardware gather must fetch the same data but scatters a vector's lanes across
rows and then reconstructs row-major AMX operands. That extra work is not
needed when contiguous row loads already feed the current packer.
The measurements support rejecting this particular design; no PMU claim
about a unique bottleneck is made.

## Timing results

B=32, M=1, H=64, D=512, S=1024, K=512. Three independent processes,
41 samples per variant/layout/cache condition, randomized order.
Inputs are prepared outside timing. Every variant allocates and releases
its output inside the same Python/ctypes call. No allocation advantage
is given to a candidate. These are BF16-versus-BF16 experiments, not a new
comparison with the FP32 baseline.

**Indexed** uses random selected rows of the source KV cache. **Pre-gathered**
uses already-selected contiguous rows plus identity indices, preserving the
allocation boundary of the previous matched-allocation experiment.
The wrapper here is identical across variants but differs from the previous
pybind benchmark; do not compare absolute times across those experiments.
Cold tests evict 768 MiB before each call; hot tests reuse cached buffers.
Both eviction and input preparation are outside the timer. Frequency is not locked.

### indexed cold

| Variant | Median (us) | Latency change vs baseline | Process medians (us) |
|---|---:|---:|---|
| baseline | 292.106 | +0.00% | 292.106, 298.035, 277.911 |
| vector_pointers | 290.224 | -0.64% | 290.224, 295.973, 274.992 |
| unpack_shuffle | 289.132 | -1.02% | 289.132, 296.900, 276.368 |
| hardware_gather | 347.058 | +18.81% | 347.058, 362.435, 339.250 |
| no_prefetch | 311.364 | +6.59% | 311.364, 325.113, 303.230 |

### indexed hot

| Variant | Median (us) | Latency change vs baseline | Process medians (us) |
|---|---:|---:|---|
| baseline | 127.699 | +0.00% | 128.419, 127.699, 117.469 |
| vector_pointers | 126.791 | -0.71% | 127.726, 126.791, 116.700 |
| unpack_shuffle | 127.069 | -0.49% | 128.345, 127.069, 117.196 |
| hardware_gather | 139.025 | +8.87% | 139.444, 139.025, 130.318 |
| no_prefetch | 125.469 | -1.75% | 125.469, 125.673, 115.012 |

### pregathered cold

| Variant | Median (us) | Latency change vs baseline | Process medians (us) |
|---|---:|---:|---|
| baseline | 282.130 | +0.00% | 284.353, 282.130, 268.805 |
| vector_pointers | 280.652 | -0.52% | 280.652, 281.709, 266.968 |
| unpack_shuffle | 282.304 | +0.06% | 282.304, 284.442, 264.005 |
| hardware_gather | 323.666 | +14.72% | 323.666, 331.544, 316.588 |
| no_prefetch | 298.141 | +5.68% | 298.141, 304.687, 287.817 |

### pregathered hot

| Variant | Median (us) | Latency change vs baseline | Process medians (us) |
|---|---:|---:|---|
| baseline | 127.360 | +0.00% | 128.174, 127.360, 117.251 |
| vector_pointers | 126.173 | -0.93% | 127.496, 126.173, 116.358 |
| unpack_shuffle | 127.008 | -0.28% | 128.450, 127.008, 117.374 |
| hardware_gather | 140.205 | +10.09% | 140.205, 140.565, 130.298 |
| no_prefetch | 124.420 | -2.31% | 125.271, 124.420, 114.613 |

Negative latency change means faster. Sub-1% differences are not presented
as a proven production gain: process variation and code layout remain
confounders. The shuffle variant also changes sign across some cases.
The no-prefetch variant improves hot-cache timing slightly but regresses
the cold-cache path, so it is not a general replacement.

## Correctness and audit trail

Every variant passed **36 raw-bit staging cases** per process, comparing
both gathered and VNNI-packed buffers with an independent indexing/layout
oracle. Inputs include arbitrary 16-bit patterns (including BF16 NaNs and
subnormals), duplicate indices, masks and K=0/1/7/8/15/16/63/64/65/73.
All variants also matched the baseline output **bit-for-bit** on eight
attention cases: two normal seeds, sharper inputs, a multi-query tail,
masked first block, all-masked, one key and empty selections.
UBSan trap-mode results are recorded separately in [ubsan.json](ubsan.json).
This preserves existing BF16 behavior; it is not FP32/GPU bitwise parity.

Raw samples, paired order, source/binary hashes, instruction counts,
CPU affinity and diagnostic phase traces are saved in
[replicate-1.json](replicate-1.json), [replicate-2.json](replicate-2.json),
and [replicate-3.json](replicate-3.json). Phase traces are instrumented
preallocated-output diagnostics, not the allocating headline times.
No new bandwidth roofline or full-model claim is made.

## Reproduce

Compile on the login node; run only in an EMR Slurm allocation:
```bash
PYTHON=/scratch/savancha/emr-roofline-venv-20261008/bin/python
KERNEL=plugin/kernels/dsa_pilot/sparse_attention_kernel
BUILD="$HOME/.cache/sparse-gather-exploration"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$KERNEL/explore_gather.py" --build "$BUILD" --build-only
srun -p emr --constraint=ddr5600 -N 1 -n 1 -c 64 \
  --hint=nomultithread --exclusive -t 00:10:00 \
  bash "$KERNEL/run_gather_exploration.sh" "$BUILD" \
  "$PWD/$KERNEL/results/gather_exploration" "$PYTHON"
python "$KERNEL/report_gather_exploration.py"
```

The build emits separate generated C++ files, shared libraries and disassemblies
in BUILD. Production `sparse_attn.cpp` is never edited. Compiler flags are
`-O3 -fopenmp -march=sapphirerapids -mprefer-vector-width=512 -std=c++17
-fno-math-errno -ffp-contract=off -Werror=aggressive-loop-optimizations`.
