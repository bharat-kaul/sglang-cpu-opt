# DSv4-Flash DSA — C/C++ Kernel Pilot (Intel Xeon / EMR)

Performance for the six authored CPU kernels of the DeepSeek-V4-Flash DSA path at the serving operating
point **(M=32 decode batch, input S=1024, output G=256)**: for each kernel we show the **roofline floor**,
the **baseline** (the torch op it replaces), and the **C/C++ speedup** — with oneDNN/MKL use called out.
All latencies are **streaming (cold-cache, LLC-flushed per call)** — the reliable methodology: warm-cache
reuse lets an L3-resident working set be re-read from cache and hides the real DRAM ceiling. From
[operating_point.json](../plugin/validate/results/operating_point.json) (EMR, **3-replicate cross-run median**,
jobs 385065–67, the **promoted
variant** per kernel + **512-bit build flags**, via [bench_operating_point.py](../plugin/validate/bench_operating_point.py));
regenerate plots with [make_kernel_pilot_plots.py](../plugin/validate/make_kernel_pilot_plots.py).

> **Method:** these results are the Phase-1 output of the [methodology](methodology-playbook.md); the staged
> *how* each kernel was optimized is the [Standalone Kernel Optimization Playbook](standalone-kernel-optimization-playbook.md).

> **Caveats.** (1) Latencies are **streaming/cold**; the warm case (where the preceding GEMM leaves inputs hot
> in cache) is faster and is quoted per kernel below. (2) The roofline peak is the **nominal datasheet
> reference** (BW 358.4 GB/s, AMX bf16 124.5 TF, FP32 7.78 TF @ 1.9 GHz base) — **not a measured achievable
> ceiling**, so "% of roofline" means "% of nominal reference." (3) `indexer logits` and `sparse attend`
> (**†**) reuse a **oneDNN/MKL GEMM donor** and author the fused epilogue; the other four are **pure-authored**
> (no library GEMM). (4) Standalone microbench, **3-replicate cross-run median** (jobs 385065–67). Correctness
> is tracked separately (F4) and remains **PARTIAL** (promotion **BLOCKED**); the hand-`exp` kernels retain **both** the exact-libm path (bit-exact
> oracle) and the Cephes path (~1 ULP, measured) for end-to-end accuracy A/B.

## 1. Per-kernel at the operating point (M=32, S=1024, streaming): roofline vs baseline vs C/C++

![Per-kernel roofline vs baseline vs C/C++ at M=32 (streaming)](plots/operating_point_v3.png)

| kernel | implementation | baseline (torch) µs | C/C++ µs | **speedup** | roofline floor µs | % of roofline |
|---|---|---:|---:|---:|---:|---:|
| indexer logits † | oneDNN BRGEMM + authored epilogue (512-bit) | 876 | 309 | **2.8×** | 50.1 | 16% |
| indexer top-k | pure-authored (`nth_element` selection) | 420 | 41 | **10.3×** | 0.73 | 2% |
| compressor | pure-authored (hand-inlined Cephes `exp`) | 328 | 60 | **5.4×** | 47.7 | 79% |
| sparse attend † | oneDNN `at::bmm` + hand-`exp` softmax/sink | 686 | 460 | **1.5×** | 275.9 | 60% |
| MHC sinkhorn | pure-authored (fused recurrence) | 640 | 25 | **25.2×** | 0.017 | 0% |
| MHC combine | pure-authored (accumulate-once) | 67 | 26 | **2.5×** | 7.3 | 28% |

The two **oneDNN-donor** kernels (†) are GEMM-bound; the authored gain is the fused epilogue, plus — for
sparse — the **hand-inlined Cephes `exp`** in the softmax. The four **pure-authored** kernels get the large
gains: sinkhorn 27.8× and top-k 12.5× (torch pays ~40 tiny dispatched ops / a full sort that the fused kernel
collapses), and **compressor 5.3×** (a hand-inlined AVX-512 `exp` replaced a per-lane `libm` call). After
optimization, **compressor is DRAM-bandwidth-bound (74% of the BW roof)** and **sparse is compute-bound (61%
of the FP32 roof)** — both near their real hardware ceiling, so their speedups are bounded by the machine, not
by more kernel work.

## 2. % of nominal roofline across the M-sweep (1→64): arithmetic intensity (streaming)

![% of nominal roofline vs M (streaming)](plots/roofline_msweep_v3.png)

| kernel | M=1 | M=8 | M=16 | M=32 | M=64 |
|---|---:|---:|---:|---:|---:|
| indexer logits † | 1% | 4% | 9% | 16% | 25% |
| indexer top-k | 0% | 0% | 1% | 2% | 4% |
| compressor | 9% | 36% | 56% | 79% | 103%* |
| sparse attend † | 8% | 40% | 46% | 60% | 68% |
| MHC sinkhorn | 0% | 0% | 0% | 0% | 0% |
| MHC combine | 2% | 8% | 14% | 28% | 50% |

<sub>\* compressor M=64 >100% = still partly cache-resident at that size (the nominal DRAM roof under-counts
when the working set fits L3).</sub>

As the decode batch M grows, arithmetic intensity rises and the BW/compute-bound kernels climb toward their
roofline (streaming, at M=64): **compressor → 83%**, **sparse → 68%**, **combine → 49%**, **indexer logits →
24%**. `indexer top-k` (~4%) and `sinkhorn` (~0%) stay low because they are **selection/dispatch-bound** — a
byte-or-FLOP roofline is the wrong ceiling for them, so the low % is a diagnostic, not a recoverable gap. This
is the "how close to the metal, and does more work help?" view.


## 3. Kernel deep-dives — roofline / baseline / C/C++ for every kernel

For each of the six kernels: **(a)** the roofline floor (byte/FLOP contract vs nominal peak, at M=32 unless
noted), **(b)** the baseline torch op it replaces and why it is slow, and **(c)** the authored C/C++ and the
optimizations. All numbers are M=32, S=1024, **streaming**, from
[operating_point.json](../plugin/validate/results/operating_point.json) (3-replicate median jobs 385065–67, promoted variants + 512-bit
flags); floor = `max(bytes/BW, FLOPs/peak)` vs the **nominal** peak (not an achievable ceiling).

| kernel | bound | floor µs | C/C++ µs | % roof | baseline µs | speedup |
|---|---|---:|---:|---:|---:|---:|
| 3.1 indexer logits † | BW (AMX) | 50.1 | 309 | 16% | 876 | 2.8× |
| 3.2 indexer top-k | selection | 0.73 | 41 | 2% | 420 | 10.3× |
| 3.3 compressor | BW (FP32) | 47.7 | 60 | 79% | 328 | 5.4× |
| 3.4 sparse attend † | **compute** (FP32) | 275.9 | 460 | 60% | 686 | 1.5× |
| 3.5 MHC sinkhorn | latency/dispatch | 0.017 | 25 | ~0% | 640 | 25.2× |
| 3.6 MHC combine | BW (FP32) | 7.3 | 26 | 28% | 67 | 2.5× |

### Optimization levers per kernel (vs roofline)

Which [playbook](standalone-kernel-optimization-playbook.md) stages each kernel actually used, and the roofline
result. "At ceiling" kernels used few levers *by design* — the regime forbids more.

| kernel | donor (S2) | parallel/tile (S3) | **fusion (S4)** | **precision (S5)** | vectorize/AMX (S6) | transcendental (S6) | result |
|---|---|---|---|---|---|---|---|
| indexer logits | oneDNN BRGEMM | S-tiled + M=1 path | in-cache epilogue (no `[N,H,S]` DRAM) | bf16 GEMM; **bf16-KV free win** | 512-bit, AMX | — | 16% of AMX roof (BW) |
| indexer top-k | — | row / chunked parallel | — (selection) | — | — | — | selection-bound (2%) |
| compressor | — | N×D-tile (fills M=1) | **streaming online-softmax** (no `w[N,R,D]`) | fp32 (bf16 **rejected** — conformance + non-win) | 512-bit | **hand-inlined Cephes `exp`** (~1 ULP) | **79% of DRAM roof** |
| sparse attend | oneDNN `at::bmm` | parallel over N·H | **in-place fused softmax/sink** (no 4 temps) | fp32-accumulate (**bf16-AMX tested ×3, REJECTED: measured loss**) | 512-bit | **hand-inlined Cephes `exp`** | 60% of FP32 roof (compute) |
| MHC sinkhorn | — | per-row parallel | **20-iter fused recurrence** (L1-resident) | fp32 | — | — | dispatch-bound (~0%) |
| MHC combine | — | parallel over M | **accumulate-once** (no `[m,hc,H]` temp) | fp32 | 512-bit SIMD | — | 28% of DRAM roof (BW) |

**Fusion** was applied to *every* kernel that had an avoidable intermediate — it is the single most common and
highest-value structural lever here (compressor's streaming softmax, sparse's in-place softmax, combine's
accumulate-once, sinkhorn's fused recurrence, indexer's in-cache epilogue all eliminate a DRAM round-trip).
**Precision** is a *conformance-gated* lever, not a free dial: it is a free win only when the kernel's own
compute already rounds the operand (indexer bf16-KV), and a measured non-win where the op isn't read-BW-bound
or the reference is fp32 (compressor bf16). The roofline column is the honest ceiling check: after fusion +
the hand-`exp`, compressor is at the **DRAM** wall and sparse at the **compute** wall — no lever left.

### Roofline-vs-measured gap: why, opportunity, and owning phase (per kernel)

The systematic view the summary table implies: for **every** kernel, the root cause of the gap between the
nominal roofline and measured, the concrete opportunity (and expected effect), and **which playbook phase owns
the fix**. "Wrong ceiling" = the byte/FLOP roofline is not the operative bound (selection / dispatch /
recurrence), so the low % is diagnostic, **not** a recoverable gap. This is the per-kernel instance of the
model-agnostic **performance-gap ledger** defined in the [methodology playbook](methodology-playbook.md#4-per-kernel-performance-gap-ledger-model-agnostic).

| kernel | % roof | gap is | why perf is lost (root cause) | opportunity (expected effect) | owning phase |
|---|---:|---|---|---|---|
| **indexer logits** | 16% | **recoverable (BW)** | the GEMM is only ~4.4 µs of 309 µs; the gap is `kv`-read traffic (16.8 of 18 MB) + fp32→bf16 convert/pack + small-M dispatch — **not** the contraction | **bf16 KV cache** halves the dominant read (floor 50→~25 µs, ~1.5×, amortizes across all KV ops); then a fused single-pass streaming kernel **if** attribution shows achievable-BW, not fixed overhead | **Phase 2 (E2E integration)** for bf16-KV; **Phase 1 attribution** (`INDEXER_BREAKDOWN`) first |
| **indexer top-k** | 2% | **wrong ceiling** | selection is `O(S)` compares + thread dispatch, not bytes/FLOPs; a byte roofline cannot be approached | at regime ceiling (`nth_element` + chunked-parallel already fill cores); none material | **Closed** — regime ceiling (reopen only if E2E attribution flags it) |
| **compressor** | 79% | **near ceiling (BW+exp)** | after the D1 hand-`exp`, residual = irreducible fp32 read traffic + the `exp` compute itself | small; bf16 **rejected** (non-conformant to fp32 ref + measured non-win) | **Closed** — at the DRAM+exp wall (revisit bf16 only if the reference dtype changes) |
| **sparse attend** | 60% | **compute — bf16 lever tested & REJECTED** | fp32 AVX-512 peak (7.78 TF) is the ceiling; both bmms run fp32 on the `at::bmm` donor | **bf16-AMX attempted in 3 configs (`amx`, `blockbf16`, `bf16bmm_hx`) — all measured SLOWER than the fp32 donor** (`bf16bmm_hx` 0.02–0.88× at M=1–64, EMR job 385343): the *score* bmm does win in bf16 (1.5–2× at M≥16) but the *value* bmm doesn't, and the fp32↔bf16 conversion casts + M=1 dispatch eat the gain. Tolerance-clean (cos 0.99999 normal) but **perf-losing** → fp32 donor at 60% roof is the practical ceiling | **Phase 1 — done (measured loss, published)**; residual untested idea = score-gemm-only bf16 hybrid; FP8 = stretch |
| **MHC sinkhorn** | ~0% | **wrong ceiling** | 20-iter serial recurrence + dispatch over tiny L1-resident data; FLOP roofline is meaningless here | none meaningful (already fused, L1-resident) | **Closed** — regime ceiling |
| **MHC combine** | 28% | **near ceiling (cache)** | working set is **cache-resident**, so the nominal DRAM roof under-counts; residual = small fixed dispatch | limited; bf16 possible but marginal (not read-BW-bound) | **Closed** — at the cache-resident ceiling |

**Reading it:** after measurement, **only one** kernel carries a recoverable gap with a named, still-open lever —
**indexer logits** (memory traffic → **Phase 2**). **Sparse attend's** bf16-AMX lever was the obvious compute-side
candidate, but it was **attempted in 3 configs and measured slower than the fp32 donor at every M** (job 385343) —
a valid, published disposition, not a deferral; the fp32 path at 60% roof is the practical ceiling (one untested
residual: a score-gemm-only bf16 hybrid). The remaining four kernels are at their **regime ceiling** (the roofline
is either met or the wrong ceiling), so their low/again-high % is honest, not owed work. The one open recoverable
row is mirrored into the actionable backlog: integration-class → the
[Phase-2 carry-forward levers](#phase-2-carry-forward-levers-integration-class-from-standalone-analysis) table.

#### What didn't work, and why (rejected / negative-result optimizations)

Negative results are **first-class** here: every attempted lever, its measured outcome, and the **root cause**,
so the same dead end is not re-walked and no "we should try bf16" is re-proposed without the answer. (Full data:
[standalone_opt.json](../plugin/validate/results/standalone_opt.json); sparse bf16: EMR job 385343.)

| kernel | optimization tried | measured | why it lost (root cause) | kept as |
|---|---|---|---|---|
| sparse attend | bf16 `at::bmm` + materialized softmax (`amx`) | slower + non-conformant | the materialized 4-temp softmax swamps the bf16 GEMM gain | experimental |
| sparse attend | 64-block brgemm flash (`blockbf16`) | **0.08–0.65×** (1.6–12.8× slower) | AMX pack `N≤64` cap → per-block transpose + 2 packs + 2 small brgemms ×`K/64` + rescale overwhelm 2 large MKL bmms | experimental, screened |
| sparse attend | big bf16 bmms + fused softmax (`bf16bmm_hx`) | **0.02–0.88×** (slower at every M) | *score* bmm wins in bf16 but *value* bmm doesn't; fp32↔bf16 casts + M=1 dispatch eat the gain | experimental (job 385343) |
| compressor | ATen vectorized `exp` (`C2`) | **0.62–1.01×** (slower) | `Vectorized::exp` loses to scalar `std::exp` (blendv overhead; the vector polynomial is no win here) | experimental, screened |
| compressor | two-pass stable pooling (`C3`) | ~1.0–1.14× (R128 large-M only) | doesn't decisively beat the **bit-exact** C1; marginal *and* not bit-exact | experimental, retain C1 |
| compressor | bf16 `exp` (`D3`) | non-win (70.7 vs 61.9 µs, slower cold) | fp32 inputs still stream from DRAM, so bf16 rounding adds work without cutting the dominant read; also non-conformant | rejected |
| indexer logits | hoist invariant weight-rounding (`I3`) | ~1.00× all M | the compiler already hoists it; epilogue weight-rounding is not a material standalone cost | reverted |

**The pattern:** the recurring loser is **bf16/AMX precision** on ops where the operand **conversion cast** (or a
non-benefiting *second* GEMM, or small-M dispatch) costs more than the lower-precision compute saves — a
**measured, published reason**, not an assumption. The one precision win that *does* hold (indexer bf16-KV) is
exactly the case where the kernel **already rounds the operand**, so bf16 is free *and* cuts the dominant read.

---


### 3.1 Indexer logits † (oneDNN donor + fused epilogue)

`logits[n,s] = Σ_h relu( q[n,h,:] · kv[n,s,:] ) · weight[n,h]` — a `[H,D]×[D,S]` score GEMM per request
(H=64, D=128, S=1024) followed by a relu·weight·sum reduction over the H heads. **† oneDNN donor.**

**Roofline (M=32).** FLOPs = `2·M·H·S·D + 3·M·H·S` = 2·32·64·1024·128 + … ≈ **543 MFLOP**;
bytes = q`[M,H,D]` + kv`[M,S,D]` + weight + logits = 1.05 + 16.78 + … ≈ **17.97 MB** (the kv read dominates).
AI ≈ 30 FLOP/byte ≪ AMX ridge 347 → **BW-bound**. Floor = max(17.97 MB/358.4 GB/s = **50.1 µs**, 543 MFLOP/124.5 TF = 4.4 µs) = **50.1 µs**. C/C++ 309 µs = **16%** (streaming); building with the 512-bit flags (`zmm=0→322`) lifted it ~1.17×.

**Baseline (torch)** — [bench_replicated.py `_idx_ref`](../plugin/validate/bench_replicated.py#L53):
```python
s = torch.einsum("nhd,nsd->nhs", q.bfloat16(), kv.bfloat16())
return (s.relu_() * w.bfloat16().unsqueeze(-1)).sum(1)
```
The `einsum` materializes the full `[N,H,S]` score tensor in DRAM, then relu, multiply, and sum each make
another full `[N,H,S]` pass — the `[N,H,S]` scores round-trip to memory ~3–4× (558.3 µs).

**C/C++** — [indexer_logits.cpp `indexer_logits_tiled`](../plugin/kernels/dsa_pilot/indexer_logits.cpp):
1. **Score GEMM on the oneDNN cpublas BRGEMM donor** (bf16 AMX, VNNI-packed `q^T` reused across all S-tiles).
2. **Tiled so scores never leave cache** — S is processed in `Sb`-row tiles; each tile's `[Sb,H]` scores land
   in an L1/L2 stack buffer and the **relu·weight·sum epilogue runs in-cache**, removing the `[N,H,S]` DRAM
   round-trip that made a naive port BW-bound. kv is streamed exactly once.
3. **Adaptive tiling** (`~4·threads` tasks, `Sb` multiple of 16) keeps brgemm M large (AMX-efficient) while
   still filling cores at small N; a dedicated M=1 path does one GEMM over all S to avoid tiny-tile dispatch.
4. **bf16-faithful stage rounding** in the epilogue matches the published model's bf16 boundaries bit-for-bit.

**Net:** the GEMM is the donor's; the authored win is the **in-cache fused epilogue** (no score round-trip),
and building with the 512-bit flags lifted it **343 → 293 µs** streaming. Still 17% of the nominal BW roof —
useful byte-throughput is below reference BW, so ROI is **open**, not a wall.

---

### 3.2 Indexer top-k (pure-authored selection)

Select the top-k=512 KV indices per row of `logits[N,S=1024]`. Sparse attention only needs the **set** of
top-k (gather is permutation-invariant), so this is a **selection**, not a sort.

**Roofline (M=32).** FLOPs = 0 (comparisons, not float ops). bytes = logits`[M,S]` + indices`[M,k]` int64 =
0.13 + 0.13 = **0.26 MB** → floor = 0.26 MB/358.4 GB/s = **0.73 µs**. C/C++ 41 µs = **2%** (streaming) — but a
byte/FLOP roofline is the **wrong ceiling** here: the cost is `O(S)` comparison work + thread dispatch, not
memory. The 2% is a diagnostic, not a recoverable gap.

**Baseline (torch)** — `torch.topk(logits, 512, dim=1, sorted=False).indices`. torch.topk does a **full
per-row sort/partition** (and over-returns ordering we discard), 420 µs streaming.

**C/C++** — [indexer_topk.cpp](../plugin/kernels/dsa_pilot/indexer_topk.cpp):
1. **`std::nth_element` (O(S) selection)** instead of a sort — no ordering computed, just the partition.
2. **Best-of dispatch:** large N → one row per thread (saturates the machine); **small N → chunked 2-pass**
   (split each row into C chunks, exact local top-k per chunk in parallel, merge the C·k candidates) so M=1/8
   use **all** cores, not 1/8 of them — this closed the small-M gap that used to route to torch.topk.
3. **Guards**: N=1 runs serially (no thread-team spinup, which regressed at M=1); chunking is used only when
   `k ≪ chunk-length` (else the merge is pure overhead).

**Net:** ~10.3× streaming (24× warm) by replacing a sort with a saturating parallel selection. Pure-authored, no library.

---

### 3.3 Compressor (pure-authored online-softmax pool)

Reduce a window of `R`=128 raw KV tokens to one compressed token via a **per-channel** softmax-weighted pool:
`out[n,d] = Σ_r softmax_r( score[n,r,d] + ape[r,d] ) · kv[n,r,d]`. Each of D=512 channels softmaxes over R
independently. Shapes `kv, score=[N,R,D]`, `ape=[R,D]` → `out=[N,D]`.

**Roofline (M=32).** bytes = kv + score + ape + out = 8.39 + 8.39 + 0.26 + 0.07 = **17.10 MB**;
FLOPs ≈ `3·M·R·D` ≈ 6.3 MFLOP. Naively AI ≈ 0.37 FLOP/byte ≪ FP32 ridge 21.7 would say BW-bound, DRAM
floor = 17.10 MB/358.4 GB/s = **47.7 µs**. **This label was wrong** (see the retry below): at decode M the
17 MB working set is **cache-resident** (EMR has 320 MB L3), so the DRAM roofline is not the ceiling, and the
real bottleneck was the **scalar `exp`**, not bandwidth. The shipped scalar path was 176.1 µs; the retried
hand-exp path is **36.9 µs** (below, reading at ~460 GB/s from cache).

**Baseline (torch)** — [dsa_compressor_cpu.py](../plugin/intel_cpu_models/dsa_compressor_cpu.py):
```python
w = (score.float() + ape.float().unsqueeze(0)).softmax(dim=1)   # materializes w[N,R,D]
return (w * kv.float()).sum(dim=1)
```
`score+ape`, the softmax (max/exp/sum/div), `w*kv`, and `.sum` each make a full `[N,R,D]` pass and the softmax
**materializes the weights `w[N,R,D]` in DRAM** → the ~17 MB useful traffic is multiplied ~5–7× (409 µs).

**C/C++** — [compressor.cpp](../plugin/kernels/dsa_pilot/compressor.cpp), inner loop:
```cpp
for (r = 0; r < R; ++r) {                 // ONE streaming pass over the window
  #pragma omp simd                        // (3) AVX-512 over the D channels
  for (j = 0; j < dd; ++j) {
    float x    = srow[j] + arow[j];
    bool  up   = x > m[j];
    float mnew = up ? x : m[j];
    float arg  = surv_masked ? -INF : (up ? (m[j]-x) : (x-m[j]));
    float t    = std::exp(arg);           // (4) C1: ONE exp (other factor is exp(0)=1)
    float corr = up ? t : 1.f, e = up ? 1.f : t;
    l[j]   = l[j]*corr + e;               // running denom
    acc[j] = acc[j]*corr + e*krow[j];     // running weighted accumulator
    m[j]   = mnew;                        // running max
  }
}
for (j) orow[j] = l[j] > 0 ? acc[j]/l[j] : 0;   // single divide at the end
```
1. **Streaming online ("flash") softmax — the big win.** Softmax+pool fused into **one pass over R** with
   running `(m,l,acc)`; `w[N,R,D]` is **never materialized** and each input is read **once**.
2. **N×D-tile parallelism** fills all cores even at M=1 (N-only parallelism regressed there).
3. **AVX-512 SIMD over D** (tile width rounded to 16 lanes).
4. **C1 one-exp** — the non-surviving softmax factor is exactly `exp(0)=1`, hardcoded; **bit-exact**, ~halves
   the exp work at R=128.
5. Masked `−∞` positions contribute nothing; FP32 accumulate; single final divide.

**Net (shipped C1):** win is #1 (kill the DRAM weights temporary, ~6 passes → 1). But C1 still sat at
176.1 µs — attributed at the time to “grain/exp throughput.”

#### 3.3.1 Retry (libxsmm/RAJAPerf `hpc-kernel-optimization` SKILL): the real bottleneck was the `exp`

Applying the SKILL's Stage-0/1 (build flags + classify by inspecting the built object) exposed two things the
“BW-bound” label hid:

- **The build never emitted 512-bit code.** `objdump` of the shipped `-O3 -fopenmp -march=native` build shows
  `zmm=0` (GCC defaults to 256-bit), and `at::vec` falls back to 8-lane AVX2 in a JIT extension unless
  `-DCPU_CAPABILITY_AVX512` is set — which is why an earlier vectorized-exp variant looked slow (mis-built).
- **The op is `exp`-compute-bound and cache-resident, not BW-bound.** The scalar per-lane `libm` exp was the
  serial bottleneck; the 17 MB working set lives in L3.

Measured on EMR (job 385022, interleaved median, M=32, **warm cache**, rel-err vs the fp32 torch oracle):

| variant | µs @ M=32 | vs shipped C1 | rel-err | notes |
|---|---:|---:|---:|---|
| C1 scalar (shipped) | 176–186 | 1× | 4.0e-7 | per-lane `std::exp` |
| C2 `at::vec` Sleef exp, 512-bit | 80.7 | 2.2× | 4.8e-7 | non-inlined exp call |
| **D1 hand-written AVX-512 Cephes exp (fp32)** | **36.9** | **4.8×** | **2.8e-7** | **shipped throughput path** |
| D2 reduced degree-4 exp (fp32) | 32.5 | 5.4× | 2.1e-5 | marginal, less accurate |
| D3 bf16 operands + fp32 exp island | 38.3 | 4.6× | **6.4e-3** | **no speed gain, large error** |

**D1 — [`compressor_softmax_pool_fp32intrin`](../plugin/kernels/dsa_pilot/compressor.cpp)** — hand-inlines an
AVX-512 Cephes `exp` (range-reduce + degree-6 FMA poly + `2^n` bit-scale) instead of the `at::vec`/Sleef
call. It is **fp32 throughout**, **as accurate as the scalar oracle** (2.8e-7), and needs **no special build
flags** (pure `AVX512F` intrinsics). Removing the non-inlined exp call boundary is the whole 2.2× over C2.

#### 3.3.1a Streaming (cold-cache) — the reliable number

The warm table above reuses one buffer, so a 17 MB working set that fits in L3 is re-read from cache (that is
why D1 "reads at 460 GB/s", above DRAM peak). For a **reliable** standalone number we flush ≥2× the LLC
before every call so inputs stream from DRAM (job 385039, M=32):

| variant | warm µs | **stream µs** | stream GB/s | % of nominal DRAM roof |
|---|---:|---:|---:|---:|
| C1 scalar (shipped) | 187 | 283.8 | 60 | 17% |
| C2 Sleef-512 | 80 | 90.8 | 188 | 53% |
| **D1 hand-exp** | 36.7 | **61.9** | 276 | **77%** |
| D3 bf16 | 38 | 70.7 | 242 | 68% |

So **D1's reliable M=32 latency is ~62 µs** (the warm 37 µs was ~1.7× cache-flattered). Cold, D1 runs at
**77% of the nominal DRAM roofline (85% at M=64)** — it is now genuinely **DRAM-bandwidth-bound**, the honest
post-optimization ceiling: the hand-exp pushed the kernel to the memory wall, so further `exp` work cannot
help. The warm 37 µs is still the right model for the *fused pipeline*, where the compressor consumes `kv`/
`score` the preceding GEMM just wrote (hot in cache). **D3 bf16 is slower cold too** (70.7 vs 61.9) — a
non-win on bandwidth as well, since the fp32 inputs still stream from DRAM and the bf16 rounding only adds
work. (Streaming is now the default standalone-kernel methodology; see
[bench_compressor_vectorize.py](../plugin/validate/bench_compressor_vectorize.py) `tmed_stream`.)

#### 3.3.2 Why bf16 does **not** help here (and the corpus agrees)

The proposal “run in bf16, upconvert for `exp`, downconvert after” is the standard attention pattern — and it
is in the corpus, but for **sparse-attention / indexer**, not the compressor: `inference/kernel.py::sparse_attn_kernel`
keeps q/kv/o in bf16 yet runs every softmax internal (`acc_s`, `scores_max`, `sum_exp`, `T.exp`) in **fp32**.
For the compressor the reference does the opposite *on purpose*: `inference/model.py::Compressor.forward` has
`x = x.float()  # compression need fp32` and keeps the entire pool in fp32, downcasting to bf16 only after the
pool at the norm. There is **no GPU compressor kernel** at all. So we did not miss a bf16 compressor — there
isn't one.

The measurement confirms the mechanism: since `exp` must be fp32, **bf16 gives zero `exp`-throughput gain** —
D3 is the *same speed* as the fp32 D1 (38.3 vs 36.9 µs) but its error jumps to **6.4e-3** (4 orders worse).
bf16 would only save read bytes, and this op is cache-resident + `exp`-bound, not read-BW-bound. So D1 (fp32,
hand-exp) is the right answer; bf16 is a conformance regression with no upside.

> **Note.** The §1 table/plots now use the promoted D1 at **streaming** (3-replicate median, ~60 µs); this §3.3
> narrative keeps the detailed warm/stream retry from jobs 385022/385039 (62 µs, within single-run variance).
> **Correctness:** D1 passes the F4 hard gates — output contract,
> finiteness, run-to-run determinism, and the overlap `−∞` mask case — and screens at 2.4–7.2e-7 (identical
> to the scalar oracle); an all-masked channel is guarded to return 0, matching C1 (EMR job 385033). Both
> the scalar-`libm` path (C1, bit-exact oracle) and the Cephes path (D1, ~1 ULP, measured — §4) are retained
> as an end-to-end accuracy A/B pair — the exact path is never deleted.

---

### 3.4 Sparse attend † (oneDNN donor + fused softmax) — the compute-bound one

MLA sparse attention: q`[N,H,D]` attends over the gathered latent kv`[N,K,D]` (MQA: one latent shared across
H=64 heads, D=K=512) with a per-head softmax **sink**. Two GEMMs (`q@kv^T` score, `w@kv` value) + softmax.
**† oneDNN donor.**

**Roofline (M=32).** FLOPs = `4·M·H·K·D` = 4·32·64·512·512 ≈ **2.15 GFLOP** (two `H×K×D` GEMMs);
bytes = kv`[K,D]` + q + out ≈ **41.9 MB**. AI ≈ 51 FLOP/byte **>** FP32 ridge 21.7 → **compute-bound** (the
only one). Floor = max(2.15 GFLOP/7.78 TF = **275.9 µs**, 41.9 MB/358.4 GB/s = 117 µs) = **275.9 µs**.
C/C++ 460 µs = **60%** of the FP32 roof (streaming), baseline 686 µs → **1.5×** (small *by design* — it is
already near the compute ceiling).

**Baseline (torch)** — [`_sparse_ref` → `ora_sparse_fp32`](../plugin/validate/bench_replicated.py#L58): the
fp32 einsum sparse-attn fallback — two matmuls plus a softmax that allocates four extra `[N,H,K]` temporaries
(`m`, `e`, `denom`, `w`).

**C/C++** — [sparse_attend.cpp `sparse_attend_bestof`](../plugin/kernels/dsa_pilot/sparse_attend.cpp):
1. **Both GEMMs on the oneDNN/MKL `at::bmm` donor** (fp32, near ~64% of the fp32 roof where a scalar loop
   would lose badly) — this is the right call in a compute-bound regime.
2. **Fused in-place softmax + per-head sink in ONE parallel pass** over `[N,H,K]` (max/exp/sum/div + the sink
   logit), avoiding the baseline's four full temporaries; bit-conformant with the fp32 oracle.
3. **Best-of dispatch:** N=1 → a scalar **online-softmax** path (no bmm launch overhead, bit-exact, ~3.4×);
   N≥2 → the donor-GEMM + fused-softmax path.
4. **Hand-inlined Cephes `exp` in the softmax** ([`sparse_attend_bestof_hx`](../plugin/kernels/dsa_pilot/sparse_attend.cpp)):
   the same compressor lever — replacing the per-lane `std::exp` with the inlined `exp512_ps` — lifts it
   **541 → 460 µs** streaming (1.18×). The `exp` was a meaningful share even though the op is GEMM-bound.
   ~1 ULP, F4-screened (screens identically to the libm path, §4).

**Net:** compute-bound, so the lever is the library GEMM; the authored contribution is the **fused softmax/
sink epilogue** (now with a hand-inlined `exp`) over the donor. 60% of the fp32 roof is the realistic
near-ceiling for this op.

**BF16-AMX lever — attempted in-phase, REJECTED on measured loss (EMR job 385343).** The compute-bound regime
makes bf16-AMX (124.5 TF, ~16× the fp32 peak) the obvious candidate, and the GPU reference *is* bf16. Three
configs were authored and measured: `sparse_attend_amx` (bf16 `at::bmm` + materialized softmax), `blockbf16`
(64-block brgemm flash), and `bf16bmm_hx` (big bf16 bmms + fused fp32 softmax). **All are slower than the fp32
donor at every M** — `bf16bmm_hx` ran 0.02–0.88× of `fp32bmm_hx` (≈15% slower at the M=32 operating point, far
worse at M=1/64). The instrument check isolates why: the *score* bmm genuinely wins in bf16 (1.5–2× at M≥16),
but the *value* bmm does not, and the fp32↔bf16 conversion casts + small-M dispatch eat the GEMM savings.
Tolerance is clean (cos 0.99999 normal / 0.9996 sharp), so it is **tolerance-acceptable but performance-losing**
— a valid, published disposition. The fp32 donor stays the shipped path; the one untested residual is a
score-gemm-only bf16 hybrid, and FP8 is a later stretch.

---

### 3.5 MHC sinkhorn (pure-authored fused recurrence)

Per row, build a tiny `hc×hc` (4×4) comb matrix and run 20 Sinkhorn normalization iters (alternating row/col
normalize), plus `pre=sigmoid+eps`, `post=2·sigmoid`. Inputs are tiny (`mixes[B,S,24]`).

**Roofline (M=32).** FLOPs ≈ `M·hc·hc·iters·5` ≈ 51 KFLOP; bytes ≈ **6.1 KB**. Floor = 6.1 KB/358.4 GB/s =
**0.017 µs**. C/C++ 25 µs = **~0%** (streaming) — the roofline is the **wrong ceiling**: the op is a tiny
per-row serial recurrence of divides/exps, so it is **latency/dispatch-bound**, not byte- or FLOP-bound. The
~0% is a diagnostic that this kernel will never be roofline-limited.

**Baseline (torch)** — [`_hc_split_sinkhorn_torch`](../plugin/validate/bench_replicated.py#L77) (sglang MHC oracle): ~40 tiny
global tensor ops (each of the 20 iters does a row-normalize and a col-normalize as separate ops over the
whole `[B,S,hc,hc]` tensor), each paying full dispatch + a memory pass — 640 µs streaming.

**C/C++** — [sinkhorn.cpp](../plugin/kernels/dsa_pilot/sinkhorn.cpp):
1. **Fuse all 20 iters per row** with the `hc×hc` matrix kept **register/L1-resident** — no per-iter global
   tensor pass, no 40 kernel launches; just one parallel loop over rows.
2. **eps placement matches the oracle exactly** (initial row-norm adds eps to the result, later iters add eps
   to the denominator) so parity holds.

**Net:** ~25× streaming purely by collapsing ~40 dispatched torch ops into one fused per-row recurrence. Pure-authored.

---

### 3.6 MHC combine (pure-authored accumulate-once)

`y[m,h] = Σ_k pre[m,k] · x_flat[m, k·H + h]` — a weighted sum over hc=4 clusters (`einsum("mk,mkh->mh")`),
H=4096.

**Roofline (M=32).** FLOPs ≈ `M·H·7` ≈ 0.92 MFLOP; bytes = x_flat`[M,hc·H]` + pre + y ≈ **2.62 MB**. AI ≈
0.35 FLOP/byte → **BW-bound**. Floor = 2.62 MB/358.4 GB/s = **7.3 µs**. C/C++ 26 µs = **28%** (streaming),
baseline 67 µs → **2.5×**. (Warm it is cache-resident at ~63%; cold it streams from DRAM at 28% of the roof.)

**Baseline (torch)** — [`_combine_ref`](../plugin/validate/bench_replicated.py#L67):
```python
return torch.einsum("mk,mkh->mh", pre.float(), x_flat.reshape(m, hc, h).float())
```
The einsum broadcasts `pre` against the `[m,hc,H]` view, creating a full `[m,hc,H]` intermediate before the
reduction (67 µs streaming).

**C/C++** — [combine.cpp](../plugin/kernels/dsa_pilot/combine.cpp):
1. **Accumulate-once tiling** — per H-tile, accumulate over the hc clusters into an L1 buffer, **read x once
   and write y once**. The naive per-k `y += p·x` pass re-reads and re-writes the whole y row hc times; this
   does it in a single pass with **no `[m,hc,H]` broadcast temporary**.
2. **AVX-512 SIMD over H**, parallel over M.

**Net:** 2.5×, limited because it is already close to the BW roof — the win is killing the broadcast temporary
and the hc-fold redundant y traffic.


## 4. Re-optimization wave — streaming journeys for every kernel

Applying the compressor learnings (512-bit build flags, `objdump` ISA check, hand-inlined Cephes `exp` where
exp-bound, and **streaming / cold-cache timing**) to all six kernels as a parallel wave, consolidated and
**3-replicate cross-run median** (EMR jobs 385065–67). Latencies are **streaming** at M=32 — the reliable number.

| kernel | torch | shipped C/C++ | re-optimized | lever | regime |
|---|---:|---:|---:|---|---|
| compressor | 328 | 284 | **60** | hand-exp Cephes **4.7×** | now DRAM-bound (79% roof) |
| sparse_bestof † | 686 | 541 | **460** | hand-exp Cephes **1.18×** | GEMM-bound (oneDNN donor) |
| indexer_logits † | 876 | 343 | **309** | 512-bit flags **1.11×** | GEMM-bound (oneDNN donor) |
| indexer_topk | 420 | 41 | 41 | none | selection-bound (at ceiling) |
| sinkhorn | 640 | 25 | 25 | none | dispatch-bound (at ceiling) |
| combine | 67 | 26 | 26 | none | cache-resident / BW (at ceiling) |

† oneDNN GEMM donor. `shipped` is the pre-wave C/C++ variant (compressor scalar C1, sparse `bestof`, indexer
256-bit build, single-run reference); `re-optimized` is the promoted variant (compressor `fp32intrin`, sparse
`bestof_hx`, indexer 512-bit), 3-replicate median. The torch and C/C++ columns are now 3-run medians; the
robust signal is the shipped→re-optimized delta (both C/C++).

**Findings.** Two new levers. The **512-bit flag fix** (`-mprefer-vector-width=512 -DCPU_CAPABILITY_AVX512` —
GCC defaults to 256-bit and `at::vec` falls back to AVX2 in a JIT build; `objdump` showed `zmm=0 → 322` for
indexer) lifts the oneDNN-donor indexer 1.16×, and the **hand-inlined Cephes `exp`** transfers to sparse's
softmax (1.19× via `sparse_attend_bestof_hx`). The other three kernels are already at their regime ceiling —
selection (`nth_element`), tiny dispatch, and cache-resident FMA have no `exp` to inline and no bandwidth to
recover, so the flags move them within noise. Only the compressor had the large **~4.7× (streaming; 4.8× warm)**,
because it was uniquely *exp-bound and unvectorized*; this is the honest result of measuring rather than
assuming every kernel has the same headroom.

**`exp` accuracy (industry-standard ULP bound).** The inlined `exp512_ps` was measured against a correctly-
rounded fp64 reference over the softmax range: **max 1.000 ULP, 0% of inputs exceed 1 ULP** (EMR job 385051) —
it meets the ~1 ULP faithfully-rounded bar (0.5 ULP / correctly-rounded is impractical for a hot vector kernel
and not met by the reference stack either: GPU `ex2.approx` ≈ 2 ULP). Each hand-`exp` kernel retains **both**
the exact-`libm` path (bit-exact oracle) and the Cephes path for end-to-end accuracy A/B; the Cephes path is
F4-screened (passes the contract/finiteness/determinism/mask gates) until the E2E task metric confirms it.


## Provenance
- Kernel latency + speedup: EMR node, **3-replicate cross-run median (jobs 385065–67)** of the promoted variant
  per kernel + 512-bit flags, **streaming** (LLC-flushed per call); baseline = the torch op each kernel replaces
  (torch refs are the noisiest column — they allocate temporaries — hence the 3-run median). Compressor warm/
  stream retry detail: jobs 385022/385039.
- Roofline floor: analytic `max(bytes/BW, FLOPs/peak)` from each kernel's byte/FLOP contract vs the nominal datasheet peak.
- `exp` ULP: `tools/exp_ulp_test.cpp` vs fp64 (job 385051). Correctness: F4 acceptance gate (PARTIAL, screening-only;
  compressor D1 + sparse `bestof_hx` gated-clean, jobs 385033/385053); GPU oracle (H200, job 384502) for the sparse reference dtype.
- Methodology: **streaming (cold-cache) timing** is the standing standalone-kernel method; **`objdump %zmm`** ISA
  check and the **512-bit build flags** (`-mprefer-vector-width=512 -DCPU_CAPABILITY_AVX512`) are standard.

## Next: end-to-end (Phase-2)
Progressive end-to-end wall-time on the depth-reduced dummy-weight proxy (proxy4 + proxy8 → `t(L)=t_fixed+L·t_layer`
extrapolation to 43 layers), at the chosen operating point (throughput headline B=32 @ S=1024, plus the B=1
single-stream latency floor), integrating each kernel one at a time. The E2E tok/s curves will be added here.

### Phase-2 carry-forward levers (integration-class, from standalone analysis)
Some wins are **not** standalone-kernel optimizations — the kernel already supports them; they are realized only
in the **serving runtime** (KV-cache dtype/layout, operand format, batching). They are logged here so the
standalone-phase analysis is **carried into** E2E optimization instead of being lost, each with the mechanism,
what it amortizes across, the expected effect, and the gate.

| lever | class | mechanism | amortizes across | expected effect | gate | status |
|---|---|---|---|---|---|---|
| **bf16 KV cache** | memory traffic | store KV as bf16; the GEMMs already round KV→bf16 to compute, so it is **numerically identical** to today — it just halves the dominant read | **every KV-reading op** (indexer logits, sparse-attend gather, MLA attend) | indexer floor 50→~25 µs; standalone measured **1.1–1.5×** (under-counts E2E: shared cut + decode M=1 is most BW-bound) | dummy-weight **fingerprint-invariant** (no numeric change) | **owed** — kernel path exists (`kv_bf16`); needs serving-path cache dtype |
| batch the decode (B≥2) | arithmetic intensity | raise M off the M=1 cliff (operating point already headlines B=32) | all per-token ops | moves BW-bound ops up their roofline | fingerprint-invariant per request | tracked via the operating point |
| indexer convert/pack attribution | fixed overhead | `INDEXER_BREAKDOWN` stage timers → decide if the 16%-of-roof gap is achievable-BW vs fixed convert/pack/dispatch | indexer logits | gates whether a custom single-pass kernel is worth authoring | n/a (measurement) | **pending** — run before any custom indexer kernel |

**Rule:** each lever stays **default-OFF + env-gated**; it is promoted only when its wall-time delta is measured at
the operating point and its dummy-weight fingerprint is unchanged (traffic/layout transform, not a numeric change).
Memory-traffic / dtype / layout findings from the standalone phase are logged here at discovery, not rediscovered later.
