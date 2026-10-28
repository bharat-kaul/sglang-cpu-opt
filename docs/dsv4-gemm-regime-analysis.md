# DSv4-Flash — per-GEMM roofline-regime analysis (Xeon operating range M ≤ 64)

**Lens (the goal).** Every GEMM must be either **compute-bound** (tiled so the stationary operand is reused from
L2/registers, approaching the resident AMX peak) **or** **bandwidth-bound** (stream weights from DRAM once at
full BW when reuse is impossible). Anything *below* the memory slope — stalling on launch/barrier/dispatch — is
**latency-bound**, a defect to fix, not a ceiling. **Optimization must be adaptive in `M`** (tokens sharing a
weight), because `M` decides the regime.

**Machine constants (EMR, per socket = the tp=1 serving unit).** Resident AMX bf16 peak ≈ **125 TF/socket**
(64c × 1024 bf16 FLOP/cyc × ~1.9 GHz); DRAM BW ≈ **329 GB/s/socket** (~657 node, interleaved); L2 = 2 MB/core.
**Ridge = resident_peak / BW ≈ 380 FLOP/byte** (invariant across sockets).

**The reuse law.** Weight-stationary GEMM: `AI = 2·M·N·K / weight_bytes = M · (2/elsize)` — AI **is** the reuse
factor `M`, scaled by pack density. Compute-bound iff `AI > 380`. **Crossover `M` (where AI = ridge):**

| weight dtype | AI(M) | **M needed to be compute-bound** |
|---|---|---:|
| bf16 | M | **190** |
| fp8 | 2M | **95** |
| fp4 (MXFP4) | 4M | **48** (but routed experts never reach it — see below) |

---

## ⚑ Headline: in the Xeon operating range (M ≤ 64), **every GEMM is bandwidth-bound**

The compute-bound regime needs `M ≈ 48–190`. The Xeon target is **low-batch** (prefill a short prompt / chunked;
decode B ≤ 64), so **M is almost always below crossover → the whole model runs on the memory slope.** The
compute-bound, L2-tiled regime only applies to large-batch prefill, which is *not* the Xeon operating point.
**So the optimization goal across the Xeon range is to REACH the DRAM-BW ceiling — but reaching it REQUIRES
maximal effective AMX use, not less of it.** BW-bound sets the *ceiling*: `t = max(FLOPs/P_compute, Bytes/BW)`.
You only *achieve* `Bytes/BW` when the compute term hides under the memory stream, which demands all of:
- **(a) fast AMX compute (C/C++ with AMX / AVX-512 intrinsics)** so `FLOPs/P_compute ≪ Bytes/BW`. *Proof — one
  routed expert @M=32:* the weight is a **38 µs** DRAM stream; AMX clears the math in **13 µs** (hides → BW-bound
  holds), but **vanilla torch takes 805 µs = 21× ABOVE the BW floor** → a torch fallback turns a BW-bound op into
  a *compute-bound-on-a-slow-path* op. "It's BW-bound anyway" is NOT a license to skip AMX.
- **(b) tiling + double-buffering** so the streamed weight tile is consumed by AMX *while the next tile streams*
  (overlap compute with the DRAM read) and the M activations are reused against the resident tile — without it,
  stream-then-compute serializes to ~2× the BW floor.
- **(c) fusion** to cut the `Bytes` term itself — keep intermediates in registers/L2, never materialize — which
  lowers the binding memory time and raises effective AI toward the ridge.
- **(d) kill per-call launch/barrier/dispatch overhead** (the current 12–62× gap).

The failure mode is reading "BW-bound" as "don't bother tiling / fall back to torch" — that abandons the compute
headroom that lets the op hide under the stream. **BW-bound is the ceiling; maximal AMX + tiling + fusion is how
you reach it.**

### Prefill, low-batch (total tokens M)

| GEMM | pack | AI @ M=8 | AI @ M=32 | AI @ M=64 | regime (M ≤ 64) |
|---|---|---:|---:|---:|---|
| dense proj | bf16 | 8 | 32 | 64 | **BW-bound** (crossover M=190) |
| dense proj | fp8 | 16 | 64 | 128 | **BW-bound** (crossover M=95) |
| MoE shared expert | fp4 | 32 | 128 | **256** | **BW-bound**, but **closest** to ridge at M=64 (67%) |
| MoE routed expert | fp4 | 4 | 4 | 8 | **BW-bound (deep)** — top-6/256 → ~1–2 tok/expert |

### Decode (batch B; M_eff = B for dense/shared, ≈ tokens-per-active-expert for routed)

| GEMM | pack | AI @ B=8 | AI @ B=32 | AI @ B=64 | regime |
|---|---|---:|---:|---:|---|
| dense proj | bf16 | 8 | 32 | 64 | **BW-bound** |
| MoE shared expert | fp4 | 32 | 128 | 256 | **BW-bound** (closest at B=64) |
| MoE routed expert | fp4 | ~5 | ~13 | ~15 | **BW-bound (deep)** — ~1–4 tok/active-expert even at B=64 |

**Key structural fact:** the **MoE routed experts are BW-bound at *every* M in the range** (M_eff ≈ 1–4, AI ≤ 15),
because top-6-of-256 routing fragments the batch — batching the *request* does not batch the *expert*. Only the
dense + shared-expert GEMMs accumulate M_eff = B, and even they stay BW-bound through B = 64. **The routed-expert
path is the dominant decode cost and is irreducibly BW-bound — its only ceiling is DRAM BW.**

---

## Are the kernels achieving the BW ceiling? (no — this is the headroom)

All GEMMs use the right building blocks (dense `amx_brgemm`/VNNI; MoE `fused_experts_cpu` grouped AMX; DSA
`cpublas brgemm`). But "BW-bound regime" ≠ "achieving BW":

- **Decode measures 12× (B=32) to 62× (M=1) BELOW the DRAM BW floor** → the weights are **not** streaming at BW;
  the step stalls on ~258 tiny per-expert GEMM launches/step (6 experts × 43 layers) + OpenMP barriers +
  `fused_experts_cpu` inverse-thread-scaling (8 thr = 0.1 ms vs 120 thr = 5175 ms). **Latency-bound, not
  BW-bound.** This is the single biggest gap in the model.
- **Prefill low-M** is the same shape of problem (skinny BW-bound GEMMs), just less extreme because M is larger.

---

## Adaptive-in-M optimization strategy (the actionable output)

The regime is **BW-bound throughout M ≤ 64**, but — per the headline — *reaching* the BW ceiling requires maximal
AMX the whole way (fast compute to hide under the stream, tiling to overlap, fusion to cut traffic). Every row
below is **C/C++ with AMX/AVX-512 intrinsics**, never a torch fallback; what *adapts* with M is the streaming
strategy and thread count, **not** whether to use AMX:

| M regime | regime | strategy (all AMX/AVX-512 C/C++) | threads | precedent in repo |
|---|---|---|---|---|
| **M = 1** (pure latency) | BW-bound, launch-dominated | AMX/AVX-512 GEMV: stream weight once contiguously + **SW prefetch**, consume with AMX as it arrives, **minimal threads** (barrier spin-up > the work) | **few (≤4–8)** | sinkhorn adaptive-serial (1.15–2.09×), indexer `_m1` variant, decode-cap=8 |
| **M = 2…32** (skinny GEMM) | BW-bound | **BRGEMM** tile: load weight tile once, **double-buffer** (AMX consumes tile N while tile N+1 streams), **reuse across the M tokens** from registers; **grouped/batch-reduce GEMM** over active experts so weights stream **back-to-back** (one stream, not 258 launches); **fuse** the dequant/epilogue so no intermediate is materialized | **scale up with M** | BRGEMM (`amx_brgemm`), grouped `fused_experts_cpu` |
| **M = 32…64** (upper range) | BW-bound, approaching fp4 ridge | as above + **L2 cache-blocking** of the weight panel (shared-expert/dense fp4 at M→64, AI→256) so it is reused from L2, prepping the compute-bound crossover | **all cores (saturate BW ≈ 64 cores)** | — |
| **M ≥ 95 (fp4) / 190 (bf16)** — *outside Xeon range* | compute-bound | full AMX L2-tiling, feed TMUL from L2 (resident peak) | all cores | the large-batch-prefill path |

**Two things that must be adaptive — neither is "drop AMX":**
1. **Thread count scales with M.** At M = 1 the work is tiny, so 64–128 threads is *slower* (barrier/spin-up ≫
   compute — the measured inverse scaling); at M = 64 you want all ~64 cores to saturate BW. Cheapest first win,
   config/dispatch only.
2. **Streaming layout scales with M** (GEMV → double-buffered BRGEMM → L2-blocked), but the **compute primitive
   is always AMX/AVX-512** — a vanilla-torch path at any M puts the op 20×+ above its BW floor (headline (a)).

**The highest-leverage kernel change:** convert the routed-expert decode path from **~258 tiny per-expert
launches** into **one grouped batch-reduce GEMM** over the active experts, so their fp4 weights stream
back-to-back at DRAM BW instead of each paying launch + barrier latency — the direct "latency-bound → BW-bound"
fix for the phase that dominates CPU serving.

*(All numbers spec/analytical or prior-measured; the decode "12–62× below BW" is from job 393516 + the overhead
attribution. Confirm each lever with a drift-robust interleaved counterfactual per the playbook reliability law
before authoring. An M=4096 large-batch-prefill compute-bound case exists only as the out-of-range contrast.)*
