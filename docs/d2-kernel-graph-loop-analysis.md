# D2 — decode kernel GRAPH-LOOP + FUSION analysis (all-C/C++, oracle-validated)

**Framing correction (per the directive):** the target is **not** "wire one existing op at a time." It is
**(1) the whole decode compute chain in C/C++** (today the serving forward is an *eager-torch scaffold*),
**(2) a graph/loop analysis across the multiple kernels** that fuses the chain and kills the seam glue, and
**(3) every node validated against its oracle.** Porting ops one-by-one to C++ leaves the dominant cost —
the **glue between ops** (casts, materialized intermediates, Python loops, serialized dispatch) — in place.
This doc is the graph-loop analysis that decides the fusion boundaries and the all-C/C++ kernel set.

Supersedes the "wire-existing-op" lens in `d2-serving-integration-scope.md` §3/§5. Baseline: real-weight M=32
decode ~0.5–0.8 tok/s, ~41–46 s/step (~1 s/layer × 43 of eager torch), memory/dispatch-bound (`d2_precommit_probes.json`).

## 1. The decode chain graph (per layer, ×43)

Nodes = compute kernels; edges = intermediates (dtype / shape = the seam). `C✓`=C++ wired, `C·unused`=C++
authored but bypassed, `torch`=eager scaffold.

```
x[N,H] bf16
 ├─► MHC hc_pre:  upcast(fp32) → rmsnorm → gemm → hc_split_sinkhorn[C✓ _ck.mhc_sinkhorn] → hc_combine[C✓ _ck.mhc_combine]
 │                 seams: upcast fp32 cast; gemm out → sinkhorn in; combine → residual        (infra :466–502, :2933–2963)
 │
 ├─► INDEXER (_cpu_forward_c4_indexer, infra :3050):   ALL eager torch; C++ indexer kernels BYPASSED
 │     wq_b(q_lora) ─[q fp?]→ rope(trailing64) ─→ q.float()·Hadamard ─[q fp32]→                (materialize q[N,nh,128])
 │     weights_proj(x) ─→ .float()*scale ─[w fp32 N,nh]→                                        (materialize w)
 │     compressor.compute_kv_score → compress_softmax_pool[C✓] ─→ rmsnorm ─→ .float() ─[ck fp32 S,chd]→
 │     einsum("nhd,sd->nhs")[torch] ─[scores fp32 N,nh,S]→ relu·w·sum[torch] ─[logits N,S]→
 │     masked_fill[torch] ─→ topk[torch] ─[topk N,ktop]→  ⟵⟵ indexer_logits.cpp + indexer_topk.cpp  = C·unused
 │
 ├─► COMPRESSOR core (_cpu_forward_core_compressor, infra :3006):
 │     compute_kv_score → compress_softmax_pool[C✓] → rmsnorm ─[compressed S,hd]→ stash          (seam: .to(x.dtype), rmsnorm)
 │
 ├─► MLA ATTEND (_batched_mla_attend, infra :1563):   eager torch; sparse_attend.cpp BYPASSED
 │     gather K_main→bf16 ─→ Python per-token mask loop `for t in range(T)` [serialize] ─→
 │     qb.to(bf16) · Kb.t() [QK matmul] ─[scores fp32 T,H,nk]→ masked_fill ─→ softmax(+sink, multi-pass) ─→
 │     p.to(bf16) · Vb [PV matmul] ─[out fp32 T,H,hdv]→                           ⟵⟵ sparse_attend.cpp = C·unused
 │
 └─► MoE:  fused_experts_cpu  [C✓ AMX, MXFP4]                                                   (infra :2291)
```

## 2. Seam inventory — the glue to kill (inter-kernel-fusion's 4 classes)

| # | Glue class | Where (evidence) | Fix |
|---|---|---|---|
| 1 | **Dtype cast at seam** | `q.float()`·Hadamard, `w.float()`, `ck.float()`, `scores.float()`, `qb.to(bf16)`, `p.to(bf16)`, hc_pre fp32 upcast | unify bf16 end-to-end across the chain; fuse dequant/upcast into the producing kernel's epilogue |
| 2 | **Layout / repack** | `.contiguous()`/`.reshape` around QK/PV, Hadamard matmul, `compute_kv_score` split | emit consumer layout directly; pack KV once (VNNI) and keep |
| 3 | **Staging to DRAM** | `scores[N,nh,S]`, `logits[N,S]`, `q[N,nh,128]`, `ck[S,chd]`, softmax `amax/exp/sum` passes, `scores[T,H,nk]` | fuse: indexer logits+relu+weight+sum+topk = ONE kernel (never materialize scores/logits); attend QK+softmax+sink+PV = ONE online-softmax kernel (never materialize scores/p) |
| 4 | **Serialization** | Python `for t in range(T)` mask loops (attend), `for L in ext` segment loop (indexer/compressor), einsum→relu→sum→mask→topk as 5 dispatches, the 43-layer Python loop | push per-token/per-segment loops into C++; fuse the dispatch sequence into the single kernel; evaluate per-layer loop residual |

## 3. All-C/C++ kernel set + fusion boundaries (the plan)

Each row = one **fused** C/C++ kernel to land in the serving forward, its donor (authored kernel already on
disk), and its **oracle** for validation.

| Fused C/C++ kernel | Replaces (eager chain) | Donor on disk | Oracle |
|---|---|---|---|
| **indexer_fused** = q@ckᵀ · relu · per-head-weight · sum · causal-mask · top-k | einsum→relu→`*w`→sum→masked_fill→topk (5 dispatches + scores/logits staging) | `dsa_pilot/indexer_logits.cpp` (+ `_tiled`/`_m1` variants) + `indexer_topk.cpp` | `dsa_attention.py` indexer ref; current eager path (A/B same input) |
| **sparse_attend_fused** = gather · mask · QK · online-softmax(+per-head sink) · PV | `_batched_mla_attend` (Python mask loops + QK matmul + multi-pass softmax + PV) | `dsa_pilot/sparse_attend.cpp` (`_amx`/`_blockbf16`/online-softmax variants) | `dsa_attention.py` absorbed-MLA ref (v==k + sink); eager `_batched_mla_attend` A/B |
| **compressor** (keep, verify fused rmsnorm epilogue) | compress_softmax_pool + rmsnorm + cast | `dsa_pilot/compressor.cpp` [C✓ wired] | `compress_softmax_pool_reference` |
| **mhc** (keep sinkhorn+combine; fold hc_pre gemm+rmsnorm epilogue) | hc_pre upcast→rmsnorm→gemm→sinkhorn→combine | `dsa_pilot/sinkhorn.cpp`,`combine.cpp` [C✓ wired] | `_hc_split_sinkhorn_torch` ref |
| **MoE** | — | `fused_experts_cpu` [C✓ AMX] | existing |

**Loop analysis:** the inner per-token (`for t in range(T)`) and per-segment (`for L in ext`) Python loops must
move INTO the fused C++ kernels (they are the serialization tax, not real compute). The outer 43-layer loop stays
in Python for now; measure its residual after the inner fusions (likely small vs the per-layer kernel time).

## 4. Validation (oracle-gated, per inter-kernel-fusion method)
1. Each fused kernel vs its **reference oracle** (unit parity) — the `..._reference`/`dsa_attention.py` paths.
2. **Seam re-measure:** `e2e(fused) − Σ isolated(kernel)` → confirm the glue (casts/staging/loops) is gone; the
   fused region's floor is `max(isolated floors)` under overlap, not the sum.
3. **End-to-end A/B** vs the eager scaffold on the **tiny-faithful** config (all arch switches, dummy weights) —
   a numerical regression harness; then **real-weight accuracy** vs the oracle (gsm8k parity to the 0.797 baseline).
4. `roofline-validation` on each fused region (compute vs the memory-bound decode floor).

## 5. First executable steps (bounded — the fused C++ kernels already exist)

Precise wiring status (why the eager scaffold persists despite the kernels existing):
- **indexer_fused**: `indexer_logits.cpp`+`indexer_topk.cpp` exist AND have a dispatch wrapper
  (`dsa_indexer_cpu` → `_ck.indexer_logits/topk`, oracle-checked). **But** the serving
  `_cpu_forward_c4_indexer` (infra :3050) **reimplements the math in eager torch** (`einsum`→`relu`→`sum`→
  `topk`) and never calls the wrapper → even with `INTEL_CPU_DSV4_DSA_CKERNEL=1` the serving indexer stays eager.
  **Fix = replace the eager block with the `dsa_indexer_cpu` call** (compute_q/weights feed it).
- **sparse_attend_fused**: `sparse_attend.cpp` exists (fuses QK+online-softmax+sink+PV, `_amx`/`_blockbf16`
  variants) **but has NO `_dsa_ckernels` wrapper and no `enabled("sparse_attend")`**; `_batched_mla_attend`
  (infra :1563) is fully eager. **Fix = add the wrapper (+parity check) then wire `_batched_mla_attend` to it.**
- **compressor / MHC**: already C-wired via `_ck` (`DSA_CKERNEL=1`).

Steps:
1. **Wire indexer** — swap the eager `einsum→relu→sum→topk` in `_cpu_forward_c4_indexer` for `dsa_indexer_cpu`
   (built fused C++), gated behind a flag; keep the eager path as the A/B oracle.
2. **Add `sparse_attend` wrapper** to `_dsa_ckernels` (mirror `indexer_logits`: build + `_parity` cos-check vs
   the eager `_batched_mla_attend` / `dsa_attention.py` ref) and **wire `_batched_mla_attend`** to it.
3. **Run with `INTEL_CPU_DSV4_DSA_CKERNEL=1 ...CHECK=1`** on **tiny-faithful** (mirror `make_glm5_tiny.py`) —
   the built-in per-call parity asserts C==torch oracle; then **real-weight** decode re-measure + gsm8k parity
   vs the 0.797 baseline.
4. Measure the **seam collapse** (`e2e − Σ isolated`) and `roofline-validation` each fused region.
