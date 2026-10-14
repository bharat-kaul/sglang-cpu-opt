# DSv4-Flash DSA — C/C++ Kernel Pilot (Intel Xeon / EMR)

Performance for the six authored CPU kernels of the DeepSeek-V4-Flash DSA path at the serving operating
point **(M=32 decode batch, input S=1024, output G=256)**: for each kernel we show the **roofline floor**,
the **baseline** (the torch op it replaces), and the **C/C++ speedup** — with oneDNN/MKL use called out.
All latencies are **streaming (cold-cache, LLC-flushed per call)** — the reliable methodology: warm-cache
reuse lets an L3-resident working set be re-read from cache and hides the real DRAM ceiling. From
[operating_point.json](../plugin/validate/results/operating_point.json) (EMR, **3-replicate cross-run median**,
jobs 385457–59, the **promoted
variant** per kernel + **512-bit build flags**, via [bench_operating_point.py](../plugin/validate/bench_operating_point.py));
regenerate plots with [make_kernel_pilot_plots.py](../plugin/validate/make_kernel_pilot_plots.py).

> **Method:** these results are the Phase-1 output of the [methodology](methodology-playbook.md); the staged
> *how* each kernel was optimized is the [Standalone Kernel Optimization Playbook](standalone-kernel-optimization-playbook.md).

> **Caveats.** (1) Latencies are **streaming/cold**; the warm case (where the preceding GEMM leaves inputs hot
> in cache) is faster and is quoted per kernel below. (2) The roofline peak is the **nominal datasheet
> reference** (BW 358.4 GB/s, AMX bf16 124.5 TF, FP32 7.78 TF @ 1.9 GHz base) — **not a measured achievable
> ceiling**, so "% of roofline" means "% of nominal reference." (3) `indexer logits` and `sparse attend`
> (**†**) reuse a **oneDNN/MKL GEMM donor** and author the fused epilogue; the other four are **pure-authored**
> (no library GEMM). (4) Standalone microbench, **3-replicate cross-run median** (jobs 385457–59). Correctness
> is tracked separately (F4) and remains **PARTIAL** (promotion **BLOCKED**); the hand-`exp` kernels retain **both** the exact-libm path (bit-exact
> oracle) and the Cephes path (~1 ULP, measured) for end-to-end accuracy A/B.

## 1. Per-kernel at the operating point (M=32, S=1024, streaming): roofline vs baseline vs C/C++

![Per-kernel roofline vs baseline vs C/C++ at M=32 (streaming)](plots/operating_point_v5.png)

| kernel | implementation | baseline (torch) µs | C/C++ µs | **speedup** | roofline floor µs | % of roofline |
|---|---|---:|---:|---:|---:|---:|
| indexer logits † | oneDNN BRGEMM + authored epilogue (fused q-prep, adaptive tiling) | 909 | 212 | **4.3×** | 50.1 ‡ | 24% ‡ |
| indexer top-k | pure-authored (`nth_element` selection) | 511 | 40 | **12.8×** | 0.73 | 2% |
| compressor | pure-authored (hand-inlined Cephes `exp`) | 328 | 64 | **5.2×** | 47.7 | 75% |
| sparse attend † | oneDNN `at::bmm` + hand-`exp` softmax/sink | 740 | 465 | **1.6×** | 275.9 | 59% |
| MHC sinkhorn | pure-authored (fused recurrence) | 616 | 26 | **23.3×** | 0.017 | 0% |
| MHC combine | pure-authored (accumulate-once) | 62 | 27 | **2.3×** | 7.3 | 27% |

The two **oneDNN-donor** kernels (†) are GEMM-bound; the authored gain is the fused epilogue, plus — for
sparse — the **hand-inlined Cephes `exp`** in the softmax. The four **pure-authored** kernels get the large
gains: sinkhorn 27.8× and top-k 12.5× (torch pays ~40 tiny dispatched ops / a full sort that the fused kernel
collapses), and **compressor 5.3×** (a hand-inlined AVX-512 `exp` replaced a per-lane `libm` call). After
optimization, **compressor is DRAM-bandwidth-bound (75% of the BW roof)** and **sparse is compute-bound (59%
of the FP32 roof)** — both near their real hardware ceiling, so their speedups are bounded by the machine, not
by more kernel work.

**‡ `indexer logits` — the nominal 50.1 µs floor is the *wrong ceiling*.** The byte roofline assumes the op
streams KV from DRAM, but attribution (job 385397, M=32) shows it is **GEMM/dispatch-bound, not DRAM-bound**:
the KV read is only **35 µs (11%) and already runs at ~483 GB/s** (≥ nominal BW, L3-assisted), while the
**brgemm is ~100 µs (32%)** — a small-GEMM regime far below AMX peak. The **operative floor is ≈ 110 µs**
(irreducible brgemm ~100 µs + one q convert/pack + a parallel-region barrier), so the measured **212 µs is
~52% of the achievable ceiling**, not 24%. The nominal 50.1 µs is kept in the tables only as the uniform
cross-kernel byte/FLOP contract.

## 2. % of nominal roofline across the M-sweep (1→64): arithmetic intensity (streaming)

![% of nominal roofline vs M (streaming)](plots/roofline_msweep_v5.png)

| kernel | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| indexer logits † | 1% | 2% | 4% | 8% | 15% | 24% | 31% |
| indexer top-k | 0% | 0% | 0% | 0% | 1% | 2% | 4% |
| compressor | 8% | 13% | 22% | 34% | 53% | 75% | 94% |
| sparse attend † | 8% | 14% | 26% | 36% | 48% | 59% | 66% |
| MHC sinkhorn | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| MHC combine | 2% | 2% | 4% | 8% | 15% | 27% | 53% |

As the decode batch M grows, arithmetic intensity rises and the BW/compute-bound kernels climb toward their
roofline (streaming, at M=64): **compressor → 94%**, **sparse → 66%**, **combine → 53%**, **indexer logits →
31%**. `indexer top-k` (~4%) and `sinkhorn` (~0%) stay low because they are **selection/dispatch-bound** — a
byte-or-FLOP roofline is the wrong ceiling for them, so the low % is a diagnostic, not a recoverable gap.
**`indexer logits` is the same story**: it is GEMM/dispatch-bound, so its true (operative) floor is ≈ 110 µs —
the 24–31% here is against the *nominal DRAM* roof (the wrong ceiling); against the operative floor it is
already **~52% at M=32** (see §1 ‡ and §3.1). This
is the "how close to the metal, and does more work help?" view.

> **How to read "% of roof" here — it is a *per-M* roofline, not one fixed peak held across the sweep.**
> The hardware peaks are **constant** (nominal DRAM `BW` and the ISA/dtype compute `peak` — AMX for indexer,
> FP32 for the rest). What is **re-evaluated at every M** is the roofline *floor*, because bytes and FLOPs
> both scale with M:
> $$\text{floor}(M)=\max\!\Big(\tfrac{\text{bytes}(M)}{BW},\ \tfrac{\text{FLOPs}(M)}{\text{peak}}\Big),\qquad \%\text{roof}(M)=\frac{\text{floor}(M)}{\text{measured}_\text{stream}(M)}.$$
> So each cell is measured against the roofline evaluated at **that M's own arithmetic intensity**, and the
> binding arm can **flip** (memory-bound at small M → compute-bound past the ridge). This is *why* a kernel's %
> climbs with M (e.g. compressor 8→94%) even though the hardware ceiling never moves — the kernel is
> saturating more of a floor that itself rises toward the fixed peak as M grows.


## 3. Kernel deep-dives — roofline / baseline / C/C++ for every kernel

For each of the six kernels: **(a)** the roofline floor (byte/FLOP contract vs nominal peak, at M=32 unless
noted), **(b)** the baseline torch op it replaces and why it is slow, and **(c)** the authored C/C++ and the
optimizations. All numbers are M=32, S=1024, **streaming**, from
[operating_point.json](../plugin/validate/results/operating_point.json) (3-replicate median jobs 385457–59, promoted variants + 512-bit
flags); floor = `max(bytes/BW, FLOPs/peak)` vs the **nominal** peak (not an achievable ceiling). For **indexer
logits** the nominal floor is the *wrong ceiling* — it is GEMM/dispatch-bound (KV read only 11%), so its
**operative floor is ≈ 110 µs** (measured brgemm ~100 µs + irreducible q-pack) and 212 µs ≈ **52% of achievable**,
not the 24% shown (‡).

| kernel | bound | floor µs | C/C++ µs | % roof | baseline µs | speedup |
|---|---|---:|---:|---:|---:|---:|
| 3.1 indexer logits † | GEMM/dispatch | 50.1 ‡ | 212 | 24% ‡ | 909 | 4.3× |
| 3.2 indexer top-k | selection | 0.73 | 40 | 2% | 511 | 12.8× |
| 3.3 compressor | BW (FP32) | 47.7 | 64 | 75% | 328 | 5.2× |
| 3.4 sparse attend † | **compute** (FP32) | 275.9 | 465 | 59% | 740 | 1.6× |
| 3.5 MHC sinkhorn | latency/dispatch | 0.017 | 26 | ~0% | 616 | 23.3× |
| 3.6 MHC combine | BW (FP32) | 7.3 | 27 | 27% | 62 | 2.3× |

### Optimization levers per kernel (vs roofline)

Which [playbook](standalone-kernel-optimization-playbook.md) stages each kernel actually used, and the roofline
result. "At ceiling" kernels used few levers *by design* — the regime forbids more.

| kernel | donor (S2) | parallel/tile (S3) | **fusion (S4)** | **precision (S5)** | vectorize/AMX (S6) | transcendental (S6) | result |
|---|---|---|---|---|---|---|---|
| indexer logits | oneDNN BRGEMM | S-tiled + M=1 path | in-cache epilogue (no `[N,H,S]` DRAM) | bf16 GEMM; **bf16-KV free win** | 512-bit, AMX | — | 24% of AMX roof (BW/dispatch) |
| indexer top-k | — | row / chunked parallel | — (selection) | — | — | — | selection-bound (2%) |
| compressor | — | N×D-tile (fills M=1) | **streaming online-softmax** (no `w[N,R,D]`) | fp32 (bf16 **rejected** — conformance + non-win) | 512-bit | **hand-inlined Cephes `exp`** (~1 ULP) | **75% of DRAM roof** |
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
| **indexer logits** | **24% nom / ~52% op** | **GEMM+dispatch-bound** (not DRAM) | measured (job 385397, M=32): KV read **35 µs (11%) @ ~483 GB/s** (≥ BW), brgemm **~100 µs (32%)**, convert/pack/alloc/dispatch **~62%** → **operative floor ≈ 110 µs** (brgemm + irreducible q-pack), *not* the 50.1 µs nominal DRAM roof | **DONE (Phase 1):** fused q-prep + adaptive tiling (TPT 4→1/2) = **1.46–1.99× bit-exact** (309→212 µs). **OPEN — ISA-adaptive AVX-512:** fp32 `at::bmm` score wins **2.6×@M1 → tie@M32 → AMX@M64** (job 385664; AMX underfills the K=128/N=64 shape), biggest at decode M=1 — but fp32 is a **precision change** (top-k set-match poor on random data), and the conformance-safe bf16 single-bmm measured *slower*, so the shipped bf16 tiled brgemm stays the bit-exact path. Also: merge 2 parallel regions; **bf16-KV** (~46 µs) | **Phase 1 done**; ISA-AVX → **Phase 3** (fp32 accuracy) / `dpbf16`; bf16-KV → **Phase 2** |
| **indexer top-k** | 2% | **wrong ceiling — confirmed by measurement** | measured (job 385511, M=32): `nth_element` selection **18 µs (46%)** + read/dispatch **21 µs**; **serial=521 µs → parallel 13×** ⇒ selection-*compute*-bound (not dispatch), a byte roofline cannot be approached | **core-fill tested (job 385516): `TOPK_CHUNKS` within-row chunking measured SLOWER at every M** (0.46–0.93× at k=512 — merge cost > core-fill; set-match preserved). Shipped row-parallel is optimal; small-k already takes the chunked path via the guard | **Closed — measured ceiling** (reopen only if real k≪S shifts it) |
| **compressor** | 75% | **near ceiling (BW+exp)** | after the D1 hand-`exp`, residual = irreducible fp32 read traffic + the `exp` compute itself | small; bf16 **rejected** (non-conformant to fp32 ref + measured non-win) | **Closed** — at the DRAM+exp wall (revisit bf16 only if the reference dtype changes) |
| **sparse attend** | 59% | **compute — all bf16 levers tested & REJECTED** | measured (job 385511, M=32): the two `at::bmm` GEMMs are **88%** of the op (score 48% + value 40%) at **66% of fp32 peak** (donor-limited); softmax 12% | **bf16 attempted in 4 configs — all measured SLOWER**: `amx`/`blockbf16`/`bf16bmm_hx` (job 385343) and now the **score-only hybrid `sparse_attend_scorehybrid` (job 385516): 0.91× at M=32, bf16-KV-provided *worse* (value needs an fp32 upcast), cos 1.0** — the score bmm wins in bf16 but the value bmm doesn't and the casts eat it. **fp32 donor at 59% is the confirmed ceiling**; a hand fp32 GEMM is unlikely to beat oneDNN (route around, publish); FP8 = stretch | **Phase 1 — done (measured loss, published)** |
| **MHC sinkhorn** | ~0% | **wrong ceiling (dispatch/alloc) — recovered at low M** | measured (job 385511, M=32): the 20-iter recurrence is only **1.6 µs (6%)**; **94% is fixed dispatch + 3 output allocs** (flat vs M) — the FLOP roofline is meaningless | **RECOVERED (job 385516): adaptive serial dispatch — serial 1.15–2.09× faster at rows≤8** (parallel team spinup dominates the tiny 4×4 work), parallel wins rows≥16; **bit-exact**, shipped. Helps the B=1/low-batch decode floor | **Shipped (adaptive)**; M≥16 unchanged |
| **MHC combine** | 28% | **wrong ceiling (operative BW) — confirmed** | measured (job 385511, M=32): the kernel reads at **84.6 GB/s ≈ the bare `torch.sum` proxy (85 GB/s)** — the 2 MB working set is size-limited so operative BW ≪ nominal 358, and the kernel already matches the memory system's delivery | none — no config beats a raw read (confirmed at the operative-BW ceiling); bf16 marginal (not read-BW-bound) | **Closed — measured ceiling** |

**Reading it:** every row is now a **measured** disposition, not an asserted one — a bound is a tested
hypothesis, so each "ceiling" was probed with its named knob before closing (jobs 385511 attribution + 385516
recovery). **indexer logits** keeps an open recoverable gap (memory traffic → **Phase 2**). **sparse attend's**
bf16 lever was the obvious compute-side candidate; it was attempted in **four** configs and measured slower
at every M — including the score-only hybrid (job 385516) — so the fp32 donor at 59% is the confirmed ceiling.
**topk** and **combine** were likewise *confirmed* at ceiling by testing the knob (within-row chunking lost;
the kernel already reads at the bare-`torch.sum` bandwidth). **sinkhorn** is the payoff of refusing to assume:
the "recurrence-bound" label was wrong (recurrence is 6%, 94% is dispatch), and an adaptive serial dispatch
**recovered a 1.15–2.09× bit-exact win at low batch** — which the "at-ceiling" label would have buried. The one
open recoverable row is mirrored into the actionable backlog: integration-class → the
[Phase-2 carry-forward levers](#phase-2-carry-forward-levers-integration-class-from-standalone-analysis) table.

#### What didn't work, and why (rejected / negative-result optimizations)

Negative results are **first-class** here: every attempted lever, its measured outcome, and the **root cause**,
so the same dead end is not re-walked and no "we should try bf16" is re-proposed without the answer. (Full data:
[standalone_opt.json](../plugin/validate/results/standalone_opt.json); sparse bf16: EMR jobs 385343/385516.)

| kernel | optimization tried | measured | why it lost (root cause) | kept as |
|---|---|---|---|---|
| sparse attend | bf16 `at::bmm` + materialized softmax (`amx`) | slower + non-conformant | the materialized 4-temp softmax swamps the bf16 GEMM gain | experimental |
| sparse attend | 64-block brgemm flash (`blockbf16`) | **0.08–0.65×** (1.6–12.8× slower) | AMX pack `N≤64` cap → per-block transpose + 2 packs + 2 small brgemms ×`K/64` + rescale overwhelm 2 large MKL bmms | experimental, screened |
| sparse attend | big bf16 bmms + fused softmax (`bf16bmm_hx`) | **0.02–0.88×** (slower at every M) | *score* bmm wins in bf16 but *value* bmm doesn't; fp32↔bf16 casts + M=1 dispatch eat the gain | experimental (job 385343) |
| sparse attend | **score-only bf16 hybrid** (`scorehybrid`: bf16 score, fp32 value) | **0.91× M=32** (0.01–0.91×; bf16-KV *worse*) | the attribution-motivated residual — but the kv→bf16 cast for the score (and the fp32 upcast for the value when KV arrives bf16) exceed the ~114 µs score-GEMM saving; cos 1.0 | experimental (job 385516) |
| indexer top-k | force within-row chunking (`TOPK_CHUNKS`, core-fill) | **0.46–0.93×** (slower every M) | at k=512 each chunk selects ~all; the candidate-merge cost exceeds the core-fill benefit (set-match preserved) | diagnostic (job 385516) |
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

**Roofline (M=32).** FLOPs = `2·M·H·S·D + 3·M·H·S` ≈ **543 MFLOP**; bytes = q`[M,H,D]` + kv`[M,S,D]` +
weight + logits = 1.05 + 16.78 + … ≈ **17.97 MB** (kv-dominated). The *nominal* roofline is
`max(17.97 MB/358.4 GB/s = **50.1 µs**, 543 MFLOP/124.5 TF = 4.4 µs) = **50.1 µs** — **but that is the wrong
ceiling.** Attribution (job 385397, M=32) shows the op is **not DRAM-bound**: the KV read is only **35 µs (11%)
at ~483 GB/s** (≥ nominal BW, L3-assisted), whereas the **brgemm itself is ~100 µs (32%)** — a small-GEMM/
dispatch regime, far below AMX peak, not a byte stream. The **operative floor is therefore ≈ 110 µs**
(irreducible brgemm ~100 µs + one q convert/pack + a parallel-region barrier), *not* 50 µs. Measured
**212 µs ≈ 52% of that operative ceiling** (and only 24% of the misleading nominal DRAM roof, which the summary
tables keep solely as the uniform cross-kernel contract).

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

**Net:** the GEMM is the donor's; the authored wins are the **in-cache fused epilogue** (no score round-trip),
the 512-bit build (**343 → 293 µs**), and the **fused q-prep + adaptive tiling** (**293 → 212 µs**). At 212 µs
the op sits at **~52% of its ≈110 µs operative (GEMM+dispatch) floor**; the residual gap is reducible overhead
(merge the two parallel regions, further pack fusion), so ROI is **open**, not a wall.

**Attribution redirected the lever, then a fused q-prep paid off (jobs 385397/385402).** Before authoring, we
*attributed* the 309 µs (Stage 1.5): the **KV DRAM read is only ~11% and already runs at ~483 GB/s (≥ nominal
BW)** — *not* the bottleneck, despite the bytes being KV-dominated; the real ~62% was **per-call convert +
VNNI-pack + alloc + dispatch**. So the lever is the overhead, not the read. The fix: **fuse the q
convert+transpose+VNNI-pack into one pass** over a reused thread-local buffer, removing the full `q.to(bf16)`
tensor and the per-n `transpose().contiguous()` allocation. Result (3-rep, streaming): **bit-exact** (worst
diff 0.0 at every M) and **~1.27× at M=32 (≈312 → ≈245 µs)**, 1.24–1.30× across M=8–64 — lifting the kernel to
**~20% of the nominal BW roof** (3.6× vs the torch baseline). *This is the worked example behind the
[Stage 1.5 rule](methodology-playbook.md#4-per-kernel-performance-gap-ledger-model-agnostic): the roofline of
one component (KV bytes) over-predicted; measuring the whole op redirected the lever to the real cost.*

**Then: dispatch-bound, not BW-bound — fewer tiles (job 385422).** Decomposing the *fused* kernel showed the
remaining ~245 µs is a **~170 µs fixed, M-independent cost** (warm M=8→176, M=32→191 µs: 4× the data, +8%
time), *not* the KV read (~15%, at BW) and *not* `cpublas::pack` (~2%, measured — which stopped a wasted
inline-pack rewrite). The fixed cost is the **tiling structure**: the adaptive tiler pinned the task count at
`~4×threads ≈ 256` tiles *regardless of M*, so ~256 parallel tasks each pay a per-task tax (scheduling +
scratch alloc + a `brgemm` call). A tile-count sweep (bit-exact across tile counts) confirmed it — cutting
tiles 512→64 drops time ~2× monotonically — so the tiler now targets **1 tile/thread for M<64** (2× at M=64,
where 1×'s tiles overflow L2): **bit-exact**, and strictly best-case per M. **Cumulative vs the original:
M=32 ≈312 → ≈207 µs (1.51×), M=8 ≈285 → ≈143 µs (1.99×).** The honest ceiling here is the **operative floor
≈ 110 µs** (dispatch + L3 + the ~100 µs `brgemm`), not the 50 µs DRAM floor (the working set is L3-resident),
so "% of the BW roof" is the wrong frame for this op.

**The regime shifts as you optimize (jobs 385422/385427).** With the dispatch floor stripped, the KV read/
convert becomes a *larger* share: `bf16-KV` would save ~15 µs at the old tiling but **~46 µs at the new** — i.e.
the op is moving *toward* read-bound, so the Phase-2 **bf16-KV** lever grows in value as the dispatch floor
drops. Remaining recoverable levers, in order: merge the two parallel regions (one barrier) · **bf16-KV**
(~46 µs, Phase-2 integration) · cold-alloc (~1.2× warm→stream tax, mostly engine-amortized).

---

### 3.2 Indexer top-k (pure-authored selection)

Select the top-k=512 KV indices per row of `logits[N,S=1024]`. Sparse attention only needs the **set** of
top-k (gather is permutation-invariant), so this is a **selection**, not a sort.

**Roofline (M=32).** FLOPs = 0 (comparisons, not float ops). bytes = logits`[M,S]` + indices`[M,k]` int64 =
0.13 + 0.13 = **0.26 MB** → floor = 0.26 MB/358.4 GB/s = **0.73 µs**. C/C++ 40 µs = **2%** (streaming) — a
byte/FLOP roofline is the **wrong ceiling**, now **confirmed by measurement** (job 385511, M=32): `nth_element`
selection is **18 µs (46%)**, read+dispatch **21 µs (54%)**, and forcing a single thread costs **521 µs** — so
parallelism already delivers **13×**. It is selection-*compute*-bound (real, parallelizable work), **not**
dispatch-bound; the 2% is a diagnostic, not a recoverable gap.

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
The **core-fill knob was tested, not assumed** (job 385516): forcing within-row chunking (`TOPK_CHUNKS`) to use
all 64 cores at M=32 measured **0.46–0.93× (slower)** at k=512 — each chunk selects ≈all so the candidate-merge
exceeds the core-fill benefit (set-match preserved). The shipped row-parallel dispatch is the measured optimum
at this k; small-k already takes the chunked path via the guard — **measured ceiling, not asserted.**

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
C/C++ 465 µs = **59%** of the FP32 roof (streaming), baseline 740 µs → **1.6×** (small *by design* — it is
already near the compute ceiling). **Attribution (job 385511, M=32) localizes the gap in the donor GEMMs:**
score GEMM **228 µs (48%)** + value GEMM **189 µs (40%)** = **88%** of the op at **66% of the fp32 peak**
(donor-limited), softmax only **56 µs (12%)** — so the 41% below the roof is the `at::bmm` donor's small-batch
GEMM efficiency, not recoverable composition overhead.

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
— a valid, published disposition. The one residual — a **score-GEMM-only bf16 hybrid** — is now **also tested**
(`sparse_attend_scorehybrid`, job 385516): keep the value GEMM fp32 and run only the bf16-able score GEMM in
AMX. It **loses at every M** (0.91× at M=32, pathological at M=1–2; **bf16-KV-provided is *worse*** because the
value GEMM then needs an fp32 upcast of the KV), cos 1.0 — the kv→bf16 cast exceeds the ~114 µs score saving.
That is the **4th** bf16 config to measure a loss, so the **fp32 donor at 59% is the confirmed ceiling** (a hand
fp32 GEMM is unlikely to beat oneDNN — route around, publish); FP8 is a later stretch.

---

### 3.5 MHC sinkhorn (pure-authored fused recurrence)

Per row, build a tiny `hc×hc` (4×4) comb matrix and run 20 Sinkhorn normalization iters (alternating row/col
normalize), plus `pre=sigmoid+eps`, `post=2·sigmoid`. Inputs are tiny (`mixes[B,S,24]`).

**Roofline (M=32).** FLOPs ≈ `M·hc·hc·iters·5` ≈ 51 KFLOP; bytes ≈ **6.1 KB**. Floor = 6.1 KB/358.4 GB/s =
**0.017 µs**. C/C++ 25 µs = **~0%** (streaming) — the roofline is the **wrong ceiling**, and measurement
(job 385511) **corrects the mechanism**: the 20-iter recurrence is only **1.6 µs (6%)**; **94% is fixed
dispatch + 3 output-tensor allocs** (the latency is flat vs M), so it is dispatch/alloc-bound, *not*
recurrence-bound. The ~0% of the FLOP roof is a diagnostic; this kernel will never be roofline-limited.

**Baseline (torch)** — [`_hc_split_sinkhorn_torch`](../plugin/validate/bench_replicated.py#L77) (sglang MHC oracle): ~40 tiny
global tensor ops (each of the 20 iters does a row-normalize and a col-normalize as separate ops over the
whole `[B,S,hc,hc]` tensor), each paying full dispatch + a memory pass — 640 µs streaming.

**C/C++** — [sinkhorn.cpp](../plugin/kernels/dsa_pilot/sinkhorn.cpp):
1. **Fuse all 20 iters per row** with the `hc×hc` matrix kept **register/L1-resident** — no per-iter global
   tensor pass, no 40 kernel launches; just one parallel loop over rows.
2. **eps placement matches the oracle exactly** (initial row-norm adds eps to the result, later iters add eps
   to the denominator) so parity holds.
3. **Adaptive serial/parallel dispatch** (job 385516): rows are independent so serial == parallel **bit-for-bit**,
   and serial is **1.15–2.09× faster at rows≤8** (the parallel team spinup dominates the tiny 4×4 work); parallel
   wins at rows≥16. Ships the per-M optimum (serial ≤ 12 rows, else parallel) — a bit-exact win at the
   **B=1/low-batch decode** floor, the regime the "at-ceiling" label would have buried.

**Net:** ~25× streaming by collapsing ~40 dispatched torch ops into one fused per-row recurrence (M≥16), plus
the adaptive low-M dispatch above. Pure-authored.

---

### 3.6 MHC combine (pure-authored accumulate-once)

`y[m,h] = Σ_k pre[m,k] · x_flat[m, k·H + h]` — a weighted sum over hc=4 clusters (`einsum("mk,mkh->mh")`),
H=4096.

**Roofline (M=32).** FLOPs ≈ `M·H·7` ≈ 0.92 MFLOP; bytes = x_flat`[M,hc·H]` + pre + y ≈ **2.62 MB**. AI ≈
0.35 FLOP/byte → **BW-bound**. Floor = 2.62 MB/358.4 GB/s = **7.3 µs**. C/C++ 26 µs = **28%** (streaming),
baseline 67 µs → **2.5×**. **Measurement (job 385511) shows this is the *operative-BW* ceiling, not a kernel
shortfall:** the kernel reads at **84.6 GB/s ≈ the bare `torch.sum` proxy (85 GB/s)** — the 2.62 MB working set
is size-limited at M=32, so the operative BW is far below nominal 358 and the kernel already matches what the
memory system delivers (no config beats a raw read; confirmed at ceiling, not asserted).

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

![Optimization journey per kernel — baseline → successive levers → roofline ceiling](plots/kernel_journeys_v2.png)

Every authored kernel's **optimization journey** against its **roofline ceiling** — the latency after each named
lever (torch baseline → shipped, M=32 streaming), with each kernel's bars descending toward its **operative-floor
line** (purple dashed) and labelled with the cumulative speed-up (×) and **% of roofline** (or *at ceiling*).
Maintained as the levers land, so the per-kernel record shows *how* the speed-up was built **and how close it got
to the achievable floor**. Data: [kernel_journeys.json](../plugin/validate/results/kernel_journeys.json)
(regenerate: `make_kernel_journeys.py`).

#### New opportunities from the DSA *dataflow* view (upside vs roofline)

DeepSeek Sparse Attention is a **4-kernel pipeline** — compressor → indexer_logits → top-k → sparse-attend
(externally verified against the DeepSeek-V3.2 spec + our coverage report) — so the kernels must be read **in
context**, not as islands. Placing them in the graph **re-prioritizes** the backlog:

| opportunity | kernel | upside vs roofline | priority | status |
|---|---|---|---|---|
| **ISA-adaptive AVX-512** | indexer_logits | **2.73×@M1** (160→59 µs) ≈ **85% of the nominal DRAM roof** at the decode floor; tie@M32 | **HIGH** — the indexer is the **long-context bottleneck** (scales with seq-length S) | measured; precision-gated → Phase-3 / `dpbf16` |
| **logits→top-k streaming fusion** | indexer + top-k | ~0 at S=1024 (logits tensor 128 KB) but **grows with S** (the length-S logits DRAM round-trip) — a **long-context** lever that standalone S=1024 hides | MED (context-dependent) | untested — scope at the real deployment S |
| **pipeline-dtype re-test** | sparse attend | LOW — the bf16-KV sim already showed casts aren't the whole wall (softmax + AMX-underutil remain, 7.5× off the bf16 roof); and sparse is **bounded by top-k** (constant cost, *not* the bottleneck) | **LOW** — deprioritize; the bf16 prize needs a flash rewrite, not a dtype swap | one confirming measurement, then close |
| **fp8 indexer-query** | (missing op) | small direct perf (query is tiny); it is the architecture's *intended* precision | LOW (coverage) | GAP — op not yet on CPU |

**Graph-view headline:** the **indexer** scales with context and is the real DSA bottleneck → its AVX-512
adaptive path is the top lever; **sparse** is bounded by top-k and already near its practical ceiling →
deprioritize. A standalone per-kernel % would have ranked these differently — the architecture view corrects it.

Applying the compressor learnings (512-bit build flags, `objdump` ISA check, hand-inlined Cephes `exp` where
exp-bound, and **streaming / cold-cache timing**) to all six kernels as a parallel wave, consolidated and
**3-replicate cross-run median** (EMR jobs 385457–59). Latencies are **streaming** at M=32 — the reliable number.

| kernel | torch | shipped C/C++ | re-optimized | lever | regime |
|---|---:|---:|---:|---|---|
| compressor | 328 | 284 | **64** | hand-exp Cephes **4.4×** | now DRAM-bound (75% roof) |
| sparse_bestof † | 740 | 541 | **465** | hand-exp Cephes **1.18×** | GEMM-bound (oneDNN donor) |
| indexer_logits † | 909 | 343 | **212** | fused q-prep + adaptive tiling **1.6×** | dispatch+L3-bound |
| indexer_topk | 511 | 41 | 40 | none | selection-bound (measured ceiling) |
| sinkhorn | 616 | 25 | 26 | adaptive serial (low-M) | dispatch-bound; recovered 1.15–2.09× at rows≤8 |
| combine | 62 | 26 | 27 | none | operative-BW (measured ceiling) |

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
- Kernel latency + speedup: EMR node, **3-replicate cross-run median (jobs 385457–59)** of the promoted variant
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
single-stream latency floor), integrating each kernel one at a time. **Live baseline + progression chart:
[end-to-end wall-time page](e2e-wall-time.md)** (baseline measured: B=1 decode 0.80 tok/s, prefill 67.8 s TTFT).

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
