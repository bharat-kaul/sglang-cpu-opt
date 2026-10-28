# Review request — will per-layer fusion actually recover the ~29% DSv4 CPU decode "glue"?

**Audience:** independent CPU-perf reviewer. **Ask:** pressure-test the claim below *before* we spend
multi-week, accuracy-critical effort authoring a monolithic per-layer fused kernel. We want you to try to
**falsify** the "29% is recoverable by per-layer fusion" thesis, not confirm it.

**Branch:** `dsv4-fused-layer-decode` · **Plan:** `docs/dev/monolithic-layer-kernel-plan.md` ·
**Context:** `docs/dsv4-gemm-regime-analysis.md`, `docs/dsv4-prefill-decode-review-brief.md`.

---

## 1. The claim (one sentence)
At the Xeon single-socket, low-batch decode operating point, **~29% of the DSv4 decode step is per-layer
"glue"** (dispatch + OMP park/wake between a layer's ops), and collapsing each layer into **one C++ region**
(fused glue + Python-loop removal, *around* the existing MoE/attention kernels) will recover most of it.

## 2. Evidence chain (all measured, single socket EMR, physical cores, 8 threads, dummy+real-router)
| # | finding | value | source |
|---|---|---|---|
| a | decode is **12–62× above the single-socket BW floor** → latency/dispatch-bound, NOT BW-bound | B=32: 20× (122 ms floor vs ~2.4 s) | job 393887/393902 |
| b | **more cores make decode WORSE** (barrier/park-wake thrash) | B=32: 8thr 13.4 → 58thr 2.7 tok/s | job 393887 |
| c | decode step is **99% per-layer, 1% per-step-fixed** (lm_head/embed/sample) | `step_ms = 48 + 75.8·N`, near-perfect linear | job 393977 |
| d | **per-layer glue = 29% of layer time**, stable across N∈{4,8,16,32} | 29/29/29/29% | job 393977 |
| e | the 29% glue is **overhead, not compute** | 17.4 s glue vs ~0.1 s real elementwise (norms/residual/rope) | magnitude arg (job 394053 passive leg) |
| f | per-op TIMEIT decode split (B=32) | kernel 43% / eager-torch 22% / framework-glue 8% / **inter-op residual 27%** | job 393902 |
| g | **small per-op kernel swaps are ~0% e2e** (the 29% is NOT accessible incrementally) | MHC sinkhorn+combine C-wire: parity PASS, **+0.1%** | job 394034 |

## 3. The de-risk we did (and the one we could NOT cleanly do)
- Confirmed (e) the glue is overhead by magnitude, not compute → a region that removes inter-op dispatch +
  park/wake should recover it.
- We tried a **spin-vs-passive** probe to split "park/wake" from "dispatch" — but the spin leg **hung on the
  documented spin-wait inter-region thrash pathology** (Origin lesson 3), so spin is not even a usable config
  lever and the split is unmeasured. We fell back to the magnitude argument.

## 4. ⚠ The critical uncertainty we want the reviewer to attack
**Does "level-1" fusion (keep MoE/attention as separate kernel calls, fuse only the glue around them) actually
remove the inter-op park/wake — or only the Python dispatch + the small elementwise-op regions?**

The 29% = `layer.total − Σ(timed ops)`. The timed ops (MoE, attention, dense, DSA, MHC) each run their **own**
`at::parallel_for` (enter/exit their own OMP parallel region). So **between** two big kernels there is still a
region boundary → a park (end of kernel A) + wake (start of kernel B). **If a large part of the 29% is that
*inter-big-kernel* park/wake, level-1 fusion will NOT remove it** — only a true single-region kernel that runs
MoE+attention cooperatively inside one `omp parallel` (level-2, = reimplementing those kernels, the
multi-week high-risk path) would. We have **not decomposed the 29% into:**
- (i) Python per-layer loop + per-op torch/custom-op dispatch overhead  → level-1 removes,
- (ii) standalone elementwise-op regions (norms/residual/rope fork-joins) → level-1 removes,
- (iii) park/wake at the big-kernel (MoE/attn) region boundaries         → level-1 does **NOT** remove.

**If (iii) dominates, the level-1 kernel buys far less than 29% and the honest target is level-2 (much larger).**
This is the single most important thing to settle before authoring. Suggested cheap decomposition the reviewer
may endorse/improve: add TIMEIT around the inter-kernel seams + the standalone norm/residual ops, or a
microbench of N back-to-back `at::parallel_for` no-op regions at 8 threads/passive to price one park/wake, ×
(#big-kernel-boundaries/layer × 43).

## 5. Amdahl context (even full success ≠ reaching BW)
Decode at 8 threads is **~20× above the BW floor**. The 29% glue is one piece; the **other ~71% is the timed
kernels, themselves far above their isolated floors** (e.g. `moe.expert_apply` in-model ≫ isolated, the
inverse-thread-scaling pathology). So **recovering 29% → ~1.4× → still ~14× above BW.** Per-layer fusion is
necessary but **not sufficient** to make decode BW-bound; the big-kernel in-model inflation (thread/grain) is a
separate, larger lever. **Question for the reviewer: is per-layer fusion even the right *first* target, vs
attacking the MoE in-model inflation (the 33% single biggest op)?**

## 6. Alternatives considered / ruled out (challenge these too)
- **Full megakernel (fuse lm_head/embed too):** per-step-fixed is only 1% (job 393977) → not worth it.
- **Incremental per-op fusions:** ~0% e2e (finding g) → rejected as the mechanism.
- **Spin wait / thread-count tuning:** spin thrashes (hung); thread count only picks the least-bad point on a
  bad curve (finding b). Config alone does not reach BW.
- **torch.compile(layer):** no sglang CPU flag; would graph-break on the custom DSA/MoE ops (fuses only the
  elementwise glue) — arguably a cheap *probe* of (ii) worth running before the hand kernel. Reviewer view?

## 7. Specific questions for the reviewer
1. Is the **level-1-vs-level-2** concern (§4) real — i.e., will inter-big-kernel park/wake survive level-1, and
   if so, is level-1's realistic recovery closer to ~10% than ~29%?
2. What is the **cheapest reliable decomposition** of the 29% into (i)/(ii)/(iii) before authoring?
3. Given §5 (Amdahl), is per-layer fusion the right **first** lever, or should the **MoE in-model inflation**
   (33%, inverse-thread-scaling) go first?
4. Any CPU mechanism we're missing to remove inter-op park/wake **without** reimplementing MoE/attn (persistent
   thread pool / task graph / `at::launch` / a single outer region that the kernels detect and reuse)?
5. Is the accuracy risk/validation plan (per-layer FULLCAP cos≥0.9995 + sharded gsm8k) sufficient, given the
   fp32→bf16 MHC-boundary fusion?

## 8. Reproduce
- Layer-scaling (c,d,e): `plugin/validate/run_dsv4_rung1_layerscaling.sbatch` → `results/dsv4_rung1_layerscaling_1socket.csv`.
- Thread sweep (b): `plugin/validate/run_dsv4_decthr_resident.sbatch`.
- Per-op attribution (f): `plugin/validate/run_dsv4_lever2_attrib.sbatch` → `results/dsv4_decode_attribution_1socket.json`.
- MHC-wire A/B (g): `plugin/validate/run_dsv4_inc1_validate.sbatch` → `results/dsv4_inc1_mhc_result.json`.
- Glue probe (e): `plugin/validate/run_dsv4_glue_spin_vs_passive.sbatch`.
