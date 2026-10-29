> **⚠ CORRECTED after independent review (see [review](dsv4_decode_warmpool_moe_independent_review.md) +
> [response](dsv4_decode_warmpool_moe_review_response.md)).** Three fixes applied below: (1) decode-step rates
> were under-normalized (divided by `output_len` not `output_len−1`) — corrected rates are ~20% higher, the
> **relative 2.2× is unchanged**; (2) the MoE decode runs the **non-BRGEMM vector path, NOT AMX** (`avg_M =
> B·6/256 ≤ 1.5 < 5` for all B≤64) — the "underfilled AMX tiles / 6–10× above floor" explanation is WITHDRAWN
> pending a proper rebuild; (3) warm has a real **prefill regression** (~5×, sticky residue + spin) so it is NOT
> "strictly better" — promote only after phase-separated measurement. The pivot (MoE first, defer monolith)
> stands.

# DSv4 decode — warm-pool config win, re-attribution, and the pivot from monolith to MoE

**Status:** for independent review. **Branch:** `dsv4-fused-layer-decode`. **Node:** EMR single socket (pcl-sprXX),
physical cores only, no SMT. **Operating point:** tp=1, bf16 activations, MXFP4 experts, decode cap 8 threads
(sticky), B=1 and B=32, input-len 8, real weights.

This document asks the reviewer to **falsify** a strategy change: we were heading toward a monolithic per-layer
fused kernel to recover a measured ~29% decode "glue". A cheap config probe then recovered **2.2×** of decode
for free, and a re-attribution at that new baseline shrank the monolith's target to **≤12%** while surfacing the
MoE kernel as the real remaining lever (72% of the warm decode step). We propose **shelving the monolith** and
**pivoting to the MoE kernel**. Please attack the measurements and the logic below.

---

## 1. One-paragraph thesis

At the cold baseline (OMP threads park immediately between ops: `OMP_WAIT_POLICY=passive`, `KMP_BLOCKTIME=0`),
decode is dominated by inter-op thread **park/wake** latency — every op is its own `at::parallel_for`, and across
43 layers × every token the wake cost compounds. Simply keeping the pool hot (`active` + `KMP_BLOCKTIME=infinite`)
gives a **clean, low-variance 2.19–2.48× decode speedup** with identical numerics. Re-attributing decode at that
warm baseline shows the "glue" the monolith was meant to remove is now **12%**, the eager-torch ops collapsed from
26% to 8.5%, and **`moe.expert_apply` is now 72%** of the (much smaller) step and sits well above its weight-stream
roofline floor. Therefore the highest-value max-performance lever is the **MoE kernel**, not layer fusion.

---

## 2. How we got here (evidence chain, each job worker-valid)

All A/Bs use env set **before** worker spawn, or a worker-side control file re-read per forward — i.e. they cross
the `sgl.Engine` `mp.Process` boundary (the class bug that invalidated earlier A/Bs; see the review-response doc).

| # | Job(s) | What | Result |
|---|---|---|---|
| 1 | 394133 / 394148 | Repaired, provenance-gated MHC A/B (worker-verified) | Framework proven (0 bad reps); MHC C-kernel wiring **+2.26% B=32** (small, as expected) |
| 2 | 394490 | MoE diagnostic (active experts, kernel path) | MXFP4/AMX path confirmed; decode routing is **~1–6 rows/expert** (tiny GEMVs). `[MOE ISO]` microbench numbers were **non-representative** (8t MoE ×43 > whole step) — discarded for extrapolation |
| 3 | 394491 | Resident sub-8 thread sweep | **CONFOUNDED** (per-forward 58→T→58 resize churn + KV-pool drift aligned with the 2→4→6→8 order). "2 threads fastest" was an artifact |
| 4 | 394542–4 | **Clean** decode-thread confirm (sticky churn-free, fresh per T) | B=32: T2 4274 / T4 3129 / **T8 2660** ms — more threads **better**, 8 optimal; matches trusted 394148 (2726). B=1 flat 4–8. No free sub-8 win |
| 5 | 394560/1 | **Warm-pool A/B** (cold vs hot threads), 8t sticky | **B=1 502→202 ms (2.48×)**, **B=32 2690→1230 ms (2.19×)**, near-zero sample variance |
| 6 | 394565 | **Warm-vs-cold re-attribution** (TIMEIT, 8t sticky) | residual 28%→**12%**, torch-eager 26%→8.5%, kernel 44%→**78%**, **MoE 72%** of warm step |

---

## 3. The two load-bearing measurements (please scrutinize)

### 3a. Warm-pool A/B (jobs 394560 cold / 394561 warm)
Fresh process each, 8-thread **sticky** decode cap (set once, no per-forward resize), real weights, single socket,
env set pre-spawn.

| | cold (passive / blocktime 0) | warm (active / blocktime ∞) | speedup |
|---|---|---|---|
| B=1  | 502 ms/step (1.99 tok/s), samples 4.65–4.79 s | **202 ms/step (4.95 tok/s)**, samples 1.81–1.82 s | **2.48×** |
| B=32 | 2690 ms/step (11.9 tok/s), samples 22.8–23.2 s | **1230 ms/step (26.0 tok/s)**, samples 11.2–11.6 s | **2.19×** |

Validity anchors: the **cold arm reproduces the trusted baseline** (B=32 cold 2690 ≈ 394148 2726 ≈ clean-T8 2660),
and warm samples have near-zero variance. The only change is the OMP wait policy / blocktime.

### 3b. Re-attribution at warm (job 394565)
Same TIMEIT per-op decode attribution at 8t sticky, run cold then warm (fresh `one_batch` each), parents
`moe`/`dsa.mla.attend` excluded to avoid double-count; residual = `layer.total − Σ(timed leaves)`.

```
cold  wall 71.0s = kernel 31.3 (44%) + torch 18.7 (26%) + fw 1.1 (1.6%) + residual 19.9 (28%)
warm  wall 32.0s = kernel 24.9 (78%) + torch  2.7 (8.5%)+ fw 0.4 (1.4%) + residual  3.9 (12%)
per-op cold->warm: mhc.hc_post 4.99->0.33, dsa.indexer 5.36->0.62, mhc.hc_pre 4.79->0.55, moe 26.6->22.9
```
Reading: park/wake was buried in the residual AND the eager-torch ops AND partly the kernels. Warm removes it
nearly everywhere **except the MoE** (−14% only), which is genuinely kernel-bound and now dominates at **72%**.

---

## 4. The strategic claim we want falsified

**Claim A — the warm config is a real, near-free ~2.2× decode win.** It is a threading policy only (numerics
identical). *Not yet validated:* (i) accuracy/stability under warm; (ii) prefill effect; (iii) tp>1 safety — the TP
shm spin-barriers; an earlier spin probe HUNG, but that was at high thread counts and 8t ran clean here.
*Falsify by:* an accuracy run (e.g. GSM8K) warm vs cold; a tp>1 warm run; confirming prefill is not regressed.

**Claim B — the monolith's incremental prize is ≤12%, so shelve it.** At warm, the inter-op residual (dispatch +
staging copies — the only thing a persistent-region layer kernel can remove that isn't already a single kernel) is
3.9s / 32s = 12%. The monolith **cannot** touch the 72% MoE (already one kernel). *Falsify by:* arguing the residual
is larger than measured (e.g. the TIMEIT leaf timers under-count copies), or that a persistent region would also
let the MoE scale past 8 threads cooperatively (a second-order benefit we did not measure).

**Claim C — the MoE kernel is the new #1 lever and has real headroom.** `moe.expert_apply` = 72% of warm decode,
~44 ms/apply at B=32, while the weight-stream floor (~1.1 GB of 4-bit expert weights for ~90 active experts at the
achievable 8-thread BW) is ~4–8 ms → **~6–10× above floor**. So it is **not** BW-bound; candidates are AMX-tile
underutilization (1–6 rows/expert vs a 16-row tile), MXFP4 dequant cost, or per-expert launch overhead. *Falsify by:*
showing the roofline bytes are wrong (e.g. only active rows stream, or caching across tokens reduces traffic), or
that dequant is the true floor and 44 ms is near it.

---

## 5. Known caveats / threats to validity (self-audit)

- **Short workload.** Attribution + A/B used input-len 8, output-len 6–12. Decode park/wake is per-step and
  per-op so it generalizes, but long-context decode adds KV traffic we did not measure.
- **`[MOE ISO]` microbench retired.** Its absolute numbers were non-representative (exceeded the whole step); the
  e2e clean sweep is the authority. Do not cite ISO for MoE thread scaling.
- **Thread-count narrative was corrected (L.13).** Earlier "decode prefers fewer threads / more cores hurt" rested
  on the confounded resident sweep + the ISO microbench. The clean confirm says B=32 wants **8** (more is better up
  to 8); B=1 is flat. Adaptive-threading prize at cold ≈ 0 — but may be non-zero **at warm** where the MoE dominates.
- **Warm = spin.** On a dedicated socket it is strictly better; in a shared/continuous-serving deployment the idle
  spin wastes power/cores between requests. The 2.2× is a max-performance (dedicated) number.
- **Cold residual 28% ≈ the earlier 29% layer-scaling glue** — consistent, raises confidence in the method.

---

## 6. Recommended next moves (pending review)

1. **Bank Claim A:** warm-vs-cold accuracy + prefill + tp>1 safety checks; if clean, make warm the default decode
   config and re-baseline the published operating point (decode ≈ 2.2× faster).
2. **Pursue Claim C (the MoE):** roofline/profile `fused_experts_cpu(MXFP4)` at warm (BW vs dequant-compute vs
   launch overhead), re-sweep threads **at warm** (does the now-dominant MoE want > 8 threads?), and only then decide
   a concrete MoE kernel lever (bigger row grouping / tile packing, dequant fusion, or thread/grouping changes).
3. **Monolith:** shelve, with the 12% residual recorded as a future low-priority lever (revisit only if the MoE is
   driven to its floor and 12% becomes the top remaining item).

**North star unchanged:** maximum correct decode throughput on one EMR socket. The data says the road there now
runs through the MoE kernel and the warm thread pool — not a monolithic layer fusion.
