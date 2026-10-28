# DSv4-Flash — per-GEMM roofline-regime analysis (compute-bound vs BW-bound)

**Lens (the goal).** Every GEMM must be either **compute-bound** (tiled so the stationary operand is reused from
L2/registers, approaching the resident AMX peak) **or** **bandwidth-bound** (when tiling/reuse is impossible,
streaming weights from DRAM once at full BW). A kernel that is *neither* — sitting below the memory slope — is
**latency/launch/dispatch-bound**, which is a defect to fix, not an achieved ceiling.

**Machine constants (EMR, per-socket = tp=1 serving unit).** Resident AMX bf16 peak ≈ **125 TF/socket**
(64 cores × 1024 bf16 FLOP/cycle × ~1.9 GHz; `TDPBF16PS` = 16×16×32 tile = 16,384 FLOP / 16-cycle recip-tput);
per-socket DRAM BW ≈ **329 GB/s**; L2 = **2 MB/core**. The two scale together across sockets, so the ridge is
invariant: **ridge = resident_peak / BW ≈ 380 FLOP/byte** (bf16). *(These are spec-derived; the resident peak
still needs a pinned `amx_peak.c` measurement — the current docs mistakenly used a DRAM-streamed 12.4 TF as the
compute ceiling; see the playbook "compute-bound is a machine-spec verdict" rule.)*

**The reuse law.** For a weight-stationary GEMM (activations/output small vs weights), arithmetic intensity
`AI = 2·M·N·K / weight_bytes = M · (2/elsize)` — i.e. **AI = the reuse factor M, scaled by the pack density**
(bf16 →M, fp8 →2M, fp4 →4M). So **`M` (tokens sharing the weight) is the single knob** that decides the regime:
`AI > 380` ⇒ compute-bound; `AI < 380` ⇒ BW-bound.

---

## Prefill (B=32, S=128 → M = 4096 tokens)

| GEMM | pack | M_eff | AI | regime (vs ridge 380) | backend | tiled? |
|---|---|---:|---:|---|---|---|
| dense proj q/kv/o | bf16 | 4096 | 4096 | **compute-bound** (11× over ridge) | `amx_brgemm` | ✅ BRGEMM, VNNI-prepacked |
| dense proj | fp8 | 4096 | 8192 | **compute-bound** | `amx_brgemm` fp8 | ✅ |
| **MoE routed expert** | fp4 | **~96** | **384** | **marginal — right AT the ridge** | `fused_experts_cpu` | ✅ grouped AMX |
| MoE shared expert | fp4 | 4096 | 16384 | **compute-bound** (deep) | `fused_experts_cpu` | ✅ |
| DSA sparse-attend (QK/PV) | bf16 | H=64 | tile-bound | compute-bound per 64-block | `sparse_attend.cpp` | ✅ cpublas brgemm, 64-block flash |

**Findings (prefill):**
1. **Dense + shared-expert GEMMs are deeply compute-bound and correctly tiled** (BRGEMM, weights VNNI-prepacked,
   weight ≫ L2 so cache-blocking is mandatory and present). Their ceiling is the **resident ~125 TF/socket** — so
   the right scoring is %-of-resident, not %-of-streamed. If they are not near resident peak, the suspects are
   **thread-collapse** (running 128 threads where bf16 GEMM peaks ~32 — measured 30 TF @32thr vs 12–15 TF @128thr)
   and block/panel sizing, **not** the op being memory-bound.
2. **MoE routed experts sit RIGHT AT the ridge (AI≈384 vs 380)** — only *marginally* compute-bound, because
   top-6-of-256 routing fragments the 4096-token batch to **~96 tokens/expert**. This is the one prefill GEMM
   whose regime is fragile: a little less batch or a little more pack-width and it tips BW-bound. **Lever: raise
   tokens-per-expert** (bigger batch, or expert-grouping/sorting so each expert's tokens are contiguous), which
   moves it decisively compute-bound.
3. Prefill overall measures ~5% of the resident compute ceiling — **not** because the GEMMs are bad, but because
   (a) only ~40% of prefill wall is GEMM (the rest is attention/DSA/MHC/glue the compute roofline treats as free)
   and (b) the GEMMs themselves are likely thread-collapsed. Both are recoverable; neither is a BW wall.

---

## Decode (M = 1 per sequence; batched B = 32)

| GEMM | pack | M_eff | AI | regime | achieving its ceiling? |
|---|---|---:|---:|---|---|
| dense proj | bf16 | 1 | 1 | **BW-bound** (correct) | ❌ 12–62× below BW floor |
| dense proj | fp8 | 1 | 2 | **BW-bound** | ❌ |
| **MoE routed expert** | fp4 | **1** (even at B=32!) | 4 | **BW-bound** | ❌ |
| MoE shared expert | fp4 | B | 4B | BW-bound (M=1) → mild AI at batch | ❌ |

**Findings (decode):**
4. **Every decode GEMM is BW-bound *by nature*** (AI 1–4 ≪ 380) — this is the **correct** regime; there is no
   tiling that creates reuse for a single token. So the target ceiling IS the DRAM BW floor (678 tok/s @ M=32).
5. **The critical MoE finding: batching does NOT help routed experts.** At B=32, 32 tokens spread over ~59
   *distinct* experts → **~0.5–1 token per expert** → M_eff stays 1, AI stays 4, still BW-bound. (Only the dense
   + shared-expert GEMMs get real batch reuse.) So the routed-expert decode path can *only* be made fast by
   **actually hitting DRAM BW** — streaming each active expert's fp4 weights once at full bandwidth.
6. **But the kernels are 12–62× BELOW the BW floor** — so they are **not even BW-bound; they are
   latency/launch-bound.** The decode step issues ~6 experts × 43 layers ≈ **258 tiny per-expert GEMM launches**,
   each paying kernel-launch + OpenMP barrier + the `fused_experts_cpu` **inverse-thread-scaling** pathology
   (8 thr = 0.1 ms, 120 thr = 5175 ms). The weights are *not* streaming at BW; the cores are spinning on barriers
   and dispatch. **This is the single biggest gap in the whole model** and it is a *fix*, not a ceiling.

---

## Verdict against the goal

| regime | op | status |
|---|---|---|
| **compute-bound, tiled, near resident** | prefill dense + shared-expert GEMMs | ✅ right building block (BRGEMM); **verify %-of-resident + fix thread-collapse** |
| **marginally compute-bound** | prefill MoE routed expert (AI≈ridge) | ⚠ batch-fragmentation-limited → **raise tokens/expert** |
| **BW-bound by nature, achieving BW** | — | ❌ nothing in decode reaches its BW floor yet |
| **latency/launch-bound (defect)** | **all decode GEMMs** (12–62× below BW) | ❌ **the headroom** — kill per-step launch/barrier overhead |

**The two concrete levers this lens exposes (both measurable by counterfactual before building):**
- **Prefill:** thread-right-size the compute-bound GEMMs to ~1 NUMA domain (escape the 128-thread collapse) and
  confirm %-of-**resident**; raise MoE tokens-per-expert (grouping/sort) to move it off the ridge.
- **Decode:** convert the ~258 tiny per-expert launches into **one grouped BRGEMM weight-stream** (batch-reduce
  GEMM over the active experts) + dispatch reduction + per-op thread right-sizing, so the routed-expert path
  finally **streams fp4 weights at DRAM BW** instead of stalling on launch/barrier latency. This is the direct
  path from "latency-bound" to "BW-bound" for the phase that dominates CPU serving.

*(Every measured number here is spec/analytical or prior-measured; the "achieving its ceiling?" column for decode
is from the decode reassessment, job 393516, and the overhead attribution. Confirm each lever with a drift-robust
interleaved counterfactual per the playbook reliability law before authoring.)*
