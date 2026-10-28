# DSv4-Flash CPU — prefill & decode optimization brief (for external review)

**Purpose.** A self-contained, code-and-data-grounded brief so a reviewer can spot *additional* prefill/decode
optimization opportunities on the DeepSeek-V4-Flash CPU (SGLang, Intel Xeon AMX) enablement. It states the
operating points, the measured roofline position, **what has been tried and ruled out (with the measured
reason)**, and the open candidates — so the reviewer does not re-suggest a dead end.

**One reliability rule governs every number here** (learned the hard way, see §6): a per-op *region-timer* or
`self_cpu` share only **nominates** a candidate; a lever is only real if a **drift-robust interleaved
counterfactual A/B** (one engine load, arms alternated per rep, paired median) moves **end-to-end wall**. Numbers
below are tagged **[E2E-A/B]** (trustworthy end-to-end counterfactual), **[ISO]** (isolated microbench),
or **[REGION]** (region-timer attribution — *nominating only, not proof*).

---

## 1. Operating points & headline numbers (EMR, 1 socket, tp=1, real MXFP4 weights)

| phase | regime | idealized roofline | measured | position |
|---|---|---|---|---|
| **Prefill (TTFT)** | **compute-bound** (AMX) | 409 tok/s (`AMX_peak 12.4TF / 30.2 GFLOP·tok⁻¹`) | ~**174.5** (43L extrap) / ~**202** (throughput_bench) | ~43% of idealized; true achievable **unknown** (see §6) |
| **Decode (TPOT)** | **BW-bound by nature, LATENCY-bound by impl** | 678 tok/s @ M=32 (`M·BW / W_active`, BW 848 GB/s, W_active 40 GB) | **57.5** @ M=32 (throughput cfg); **~1.5** @ M=1 | **12× below BW floor @ M=32, 62× @ M=1** |

- Baseline→optimized (fully-naive eager → all-C/AMX DSA kernels): **2.19× prefill / 1.59× decode** [E2E-A/B].
- Accuracy of this optimized path is certified: gsm8k **0.9437 ≈ GPU 0.9535** (same checkpoint).
- **W_active grows with batch** (MoE): distinct experts 59 @ B=32 → ~104 @ B=64, so the decode roofline is
  **sub-linear** in batch (678 @ M=32 → ~840 @ M=64, +24% not +100%).

Data: [`dsv4_node_walltime.json`](../plugin/validate/results/dsv4_node_walltime.json),
[`operating_point.json`](../plugin/validate/results/operating_point.json),
full ledger in [`e2e-wall-time.md`](e2e-wall-time.md).

---

## 2. Code map (where each op lives)

Enablement is an external plugin — **no `sglang/` fork**. Everything is monkeypatched in from
[`plugin/intel_cpu_models/`](../plugin/intel_cpu_models).

| op / stage | Python wiring (CPU path) | authored C++ kernel |
|---|---|---|
| **DSA lightning-indexer** (q@ckᵀ·relu··w·sum·mask·topk) | [`dsa_indexer_cpu.py`](../plugin/intel_cpu_models/dsa_indexer_cpu.py) | [`indexer_logits.cpp`](../plugin/kernels/dsa_pilot/indexer_logits.cpp) + [`indexer_topk.cpp`](../plugin/kernels/dsa_pilot/indexer_topk.cpp) |
| **DSA KV-compressor** (compress_softmax_pool + rmsnorm) | [`dsa_compressor_cpu.py`](../plugin/intel_cpu_models/dsa_compressor_cpu.py) | [`compressor.cpp`](../plugin/kernels/dsa_pilot/compressor.cpp) |
| **DSA sparse attention** (gather·mask·QK·online-softmax·sink·PV) | [`dsa_sparse_attention_cpu.py`](../plugin/intel_cpu_models/dsa_sparse_attention_cpu.py), [`dsa_attention_cpu.py`](../plugin/intel_cpu_models/dsa_attention_cpu.py) | [`sparse_attend.cpp`](../plugin/kernels/dsa_pilot/sparse_attend.cpp) |
| **MHC hyper-connections** (hc_pre/hc_post: upcast·rmsnorm·gemm·sinkhorn·combine) | `_dsv4_cpu_infra.py` `hc_pre`/`hc_post` (~L445) | [`sinkhorn.cpp`](../plugin/kernels/dsa_pilot/sinkhorn.cpp) + [`combine.cpp`](../plugin/kernels/dsa_pilot/combine.cpp) |
| **MLA attention core** (QK / softmax / PV, mask build, KV gather) | `_dsv4_cpu_infra.py` `_batched_mla_attend` (~L1870), `_build_coverage_mask_vectorized` (~L1821) | (uses `sparse_attend.cpp`) |
| **MoE routed experts** (MXFP4 W4A16) | `_dsv4_cpu_infra.py` MoE wiring (~L2291) | `sgl_kernel::fused_experts_cpu` (upstream AMX) |
| **dense proj / fp8** | `dense.fp8_linear` path | `sgl_kernel::fp8_scaled_mm_cpu` (upstream) |
| **q/k norm+rope+KV-write** | `_cpu_fused_q_norm_rope` / `_cpu_fused_k_norm_rope_flashmla` (~L1505–1629) | fused torch-ref port |
| **ISA dispatch kernels** (C/C++) | [`_dsa_ckernels.py`](../plugin/intel_cpu_models/_dsa_ckernels.py) | the `dsa_pilot/*.cpp` set |
| **dispatch-provenance + TIMEIT instrumentation** | `_dsv4_cpu_infra.py` (~L29–340) | — |

Config flags of interest (all default-safe): `INTEL_CPU_DSV4_DSA_CKERNEL` (fused DSA C path),
`INTEL_CPU_DSV4_MXFP4_MOE`, `INTEL_CPU_DSV4_DECODE_THREADS`, `INTEL_CPU_DSV4_MASK_LOOP`/`_MASK_ASSERT`,
`INTEL_CPU_DSV4_TIMEIT[_PHASE]`, `INTEL_CPU_DSV4_REAL_ROUTER`, `INTEL_CPU_DSV4_ATTN_FUSED`.

---

## 3. Per-kernel roofline position (M=32 streaming, isolated) [ISO]

| kernel | C++ µs | % of own roofline | vs torch |
|---|---:|---:|---:|
| indexer_logits | 211.5 | at-ceiling* | 4.3× |
| indexer_topk | 39.8 | at-ceiling* | 12.8× |
| compressor_R128 | 63.6 | at-ceiling* | 5.2× |
| sparse_bestof (attend) | 465.2 | at-ceiling* | 1.6× |
| sinkhorn | 26.5 | cache-resident (nominal roof wrong) | 23.3× |
| combine | 27.4 | at-ceiling* | 2.3× |

\*`%-of-nominal-roof` reads >100% because these are cache-resident at M=32 (the nominal DRAM roof under-counts);
the point is **each authored kernel beats its torch baseline and is not the bottleneck in isolation**. Source:
[`operating_point.json`](../plugin/validate/results/operating_point.json), chart in
[`kernel-pilot-performance.md`](kernel-pilot-performance.md).

---

## 4. Fusion / seam analysis — DONE and MEASURED

Full decode graph + 4-class seam inventory: [`d2-kernel-graph-loop-analysis.md`](d2-kernel-graph-loop-analysis.md).
The major fusions are **already authored and wired** (the DSA kernels *are* the fusions):

- **indexer_fused**: einsum→relu→·w→sum→mask→topk = **5 dispatches + scores/logits staging → 1 kernel**.
- **sparse_attend_fused**: gather·mask·QK·**online-softmax**·sink·PV → **1 kernel** (never materializes scores/p).
- **compressor**: softmax-pool + **rmsnorm epilogue** fused.
- **q/k norm+rope+fp8-pack+paged-write**: fused.
- **MoE**: `fused_experts_cpu` (AMX MXFP4) — already fused.

**Measured end-to-end benefit of the fused DSA path** (held-fixed **ON vs OFF** A/B, job 386039) [E2E-A/B]:
**prefill 880→1003 tok/s (+14%), decode 91.2→94.2 (+3%)** at B=32 (4-layer proxy). Modest because the fused DSA
ops are a *small share* of total time; **MoE + dense GEMMs dominate** and were already C/fused. Data:
[`dsv4_p2integ_385956_ck{0,1}_b{1,32}.json`](../plugin/validate/results).

---

## 5. What we TRIED and RULED OUT (do not re-suggest without new evidence)

| # | idea | result | why ruled out | evidence |
|---|---|---|---|---|
| 1 | **Vectorize the attention mask** (per-token Python loop → vectorized, bit-exact) | region 8.7s→0.25s (35×) **[ISO]** but **0% end-to-end** **[E2E-A/B]** | the region time is **overlapped/hidden** (chunked-prefill pipelining), not on the critical path — a region-timer share is not a critical-path attribution | job 393518, [`dsv4_mask_ab_393518.json`](../plugin/validate/results/dsv4_mask_ab_393518.json) |
| 2 | **Mask ablation "+7.3%"** (force cheap dense mask) | discarded | **confounded** — the cheap mask changed attention semantics (dense vs sparse → different PV work); delta ≠ mask-construction cost | job 393314 |
| 3 | **`OMP_WAIT_POLICY=passive` for throughput** | **regresses** B=32 (prefill 2.18×→1.59×, decode 1.4×→0.3×) | passive parks busy worker threads; it is an *idle-thread-thrash* fix (B=1 latency), not a throughput lever | runtime-config-tuning |
| 4 | **Decode thread-cap=8 at M=32** | **6.5× slower** (8.78 vs 57.5 tok/s) | thread-cap is an **M=1-only** latency lever; M=32 wants all cores | job 393516 |
| 5 | **More threads on MoE at M=1** | **worse** (all-threads 1.16 vs 4-thread 1.54 tok/s); isolated 8thr=0.1ms vs 120thr=5175ms | **inverse thread scaling** — tiny per-expert GEMM parallelized over too many cores = barrier/overhead blowup | [`deepseek_v4_flash_overhead.json`](../plugin/validate/results/deepseek_v4_flash_overhead.json) |
| 6 | **Up-convert MXFP4 experts → fp8** | abandoned | doubles footprint → forces tp>1 → ~330× slower M=1 decode; and fp4·2^k is already exact in bf16 (no accuracy reason) | compute-from-native-precision |
| 7 | **Fused no-temp einsum rmsnorm (hc_pre)** | **non-win** (fused 0.115s vs default 0.043s) | the default BLAS path already beats the hand-fused einsum on the clean passive path | `_dsv4_cpu_infra.py` L446 |
| 8 | **"~63 decode achievable" / "~275 prefill achievable"** | **withdrawn** | both derived from region-timer shares (ruled unreliable by #1); "dispatched to C" was wrongly read as "at kernel roofline" | §6 |
| 9 | **Sequential-leg A/B for perf deltas** | **unreliable** | thermal/run-order drift > effect and inverts with order; must use held-fixed/interleaved | jobs 393502 vs 393518 |

**Shipped wins (kept):** all-C/AMX DSA fusion (+14% prefill/+3% decode) [E2E-A/B]; sinkhorn **adaptive serial
dispatch** (1.15–2.09× at low batch, bit-exact); MXFP4-native experts (fits 1 NUMA domain → tp=1); the
`wkv_gate` row-major correctness fix (the accuracy story).

---

## 6. The open question + the honest gaps

**See the companion [per-GEMM roofline-regime analysis](dsv4-gemm-regime-analysis.md)** — each GEMM classified compute-bound (tiled, L2-reuse) vs BW-bound (no reuse) vs latency-bound (defect), by arithmetic intensity vs the resident ridge.

**Decode is the headroom.** It is BW-bound by nature but runs **12–62× above the BW floor** — a per-step
**latency/overhead** problem, NOT a physics ceiling, NOT a kernel-quality problem (the kernels are at-ceiling in
isolation, §3). The gap is ~258 tiny per-expert/per-layer GEMM launches/step + barriers + dispatch, not weight
streaming. **Open candidates (each to be proven by an interleaved counterfactual, per the reliability rule):**

1. **BRGEMM / grouped-expert GEMM** — fuse the ~6 experts × 43 layers into one **batch-reduce GEMM** so weights
   stream continuously instead of hundreds of latency-bound launches. *(Highest expected leverage; un-done.)*
2. **Dispatch reduction / graph-capture** of the decode step (cut the per-launch Python/dispatch latency).
3. **Thread right-sizing per op** (M=1 wants ≤4 threads; the MoE inverse-scaling pathology).
4. **Weight prefetch / software pipelining** (overlap next-expert load with current compute).
5. **MHC ↔ attention/residual** and **norm/rope ↔ GEMM-epilogue** fusions (un-analyzed cross-op seams).

**Prefill** is compute-bound and near a healthy fraction of its *achievable* (not idealized) ceiling; its glue is
largely overlapped (#1), so the true prefill-achievable is **unknown pending counterfactual attribution** — we do
NOT currently have a credible prefill lever.

**Honest gaps for the reviewer to probe:**
- No **prefill-specific** fusion/graph analysis (the §4 analysis is decode-only).
- The decode **per-seam collapse** (`e2e − Σ isolated`) is planned but **not yet measured**, so individual seams
  aren't ranked by payoff.
- The decode op-share breakdown is **region-timer-based** → treat as candidate-nominating only; the critical-path
  split at M=1/M=32 needs counterfactual confirmation.
- GNR cross-node numbers are **pending** (a valid per-NUMA-domain aggregate run; prior attempts OOM'd / were
  mis-bound).

---

### Reproduce
- Per-kernel: `operating_point.json` + `kernel-pilot-performance.md`.
- E2E ledger + rooflines: `e2e-wall-time.md`.
- Reliability A/Bs: `run_dsv4_mask_ab_interleaved.sbatch` (mask), `run_dsv4_decode_reassess.sbatch` (decode sweep),
  `run_dsv4_perf_matrix.sbatch` (baseline→optimized).
- Methodology + the reliability law: [`methodology-playbook.md`](methodology-playbook.md) §2 (Phase 2).
