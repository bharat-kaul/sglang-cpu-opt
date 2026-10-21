# D2 — CPU serving-integration scope: route DSV4 decode through the optimized AMX kernels

**Goal:** replace the eager-torch *correctness-first scaffold* (the cause of ~41–46 s/step real-weight M=32
decode) with optimized AMX CPU kernels, so decode moves toward its roofline.

> **⚠️ Scope corrected by the D2 pre-commit probes (jobs 390564/390570, `results/d2_precommit_probes.json`).**
> The original framing — "the kernels largely already exist; this is wiring, not authoring" — is **wrong**.
> On-node enumeration of the 77 registered CPU `sgl_kernel::` ops shows:
> - **FlashMLA is CUDA-only.** `sgl_kernel.flash_mla`'s functions all `raise _IMPORT_ERROR` because the backing
>   `flashmla_ops` CUDA extension fails to import on CPU — there is **no working CPU MLA serving kernel to wire**
>   (which is exactly why the plugin fabricates `_torch_flash_mla_with_kvcache`).
> - **The CPU attention ops that DO exist** (`decode_attention_cpu`, `extend_attention_cpu`,
>   `flash_attn_varlen_func`) are **MHA/GQA-oriented** and do not match the MLA-absorbed form (`v==k` single
>   latent, per-head sink, top-k gather) — so they must be **adapted**, not drop-in wired.
> - **The DSA compute kernels** (`plugin/kernels/dsa_pilot/{sparse_attend,indexer_logits,indexer_topk,compressor,sinkhorn,combine}.cpp`)
>   were **authored standalone but are NOT built/registered** into the serving runtime.
>
> **Net:** D2 is a **build + adapt + multi-week serving-integration** effort (the cpu-serving-integration
> "chain integration" surface), **not** light wiring. The MoE is the only piece already fully AMX-wired.
> Weigh this against Amdahl (attention ≈ 20% ⇒ ~1.25× if only it is done), a **memory-bound** decode
> (perf-stat: IPC ~1.2, cache-miss ~54%, LLC-miss ~47%), and the fact that the model already **runs + is
> accurate (gsm8k 0.797)**.

Pipeline position (cpu-serving-integration): the DSV4 CPU path is at **stage 1 (structural bypass — runs +
accurate but eager-torch)**; this scope is **stage 2 (build/adapt + wire the optimized kernels)**.

## 1. Current state — the eager-torch scaffold ("Route 2", `_install_dsa_cpu_wire`)

The GPU serving kernels are replaced on CPU by eager-torch reimplementations over a **dense KV stash**
(`_KV_BUF` / `_stash_kv_gather`), deliberately avoiding the paged state-pool:

| GPU serving kernel | CPU scaffold (eager torch) | decode share | file:anchor |
|---|---|---:|---|
| `flash_mla_with_kvcache` (MLA attend) | `_torch_flash_mla_with_kvcache` → `_batched_mla_attend` | **20%** | `_dsv4_cpu_infra.py:1496,1626` |
| paged compressor plan | `_cpu_forward_core_compressor` | 3% | `:2939` |
| `forward_c4_indexer` (DSA index logits) | `_cpu_forward_c4_indexer` (torch einsum) | 6% | `:2983` |
| hc_pre / MHC | eager torch `hc_pre` + `hc_split_sinkhorn` + `hc_combine` | 13% | `:455` |
| MoE experts | **already AMX** `torch.ops.sgl_kernel.fused_experts_cpu` (MXFP4) | 29% (thread-cfg) | `:2291` |

Result: runs end-to-end + accurate (**gsm8k 0.797**), but the DSA/MHC/attention path is ~42% eager torch,
dispatch-bound across 43 layers → ~46 s/step. Threads give only ~1.35× (confirmed not the lever).

## 2. How the optimized kernels are reached (the mechanism)

The `sgl_kernel` python module **imports fine** on CPU (77 registered ops) — the earlier note that
"`import sgl_kernel` fails" was wrong; what fails is the **`flashmla_ops` CUDA extension** inside
`sgl_kernel.flash_mla`. The AMX CPU kernels are **registered torch operators** reached via
`torch.ops.sgl_kernel.<op>` (e.g. `fused_experts_cpu`, `convert_weight_packed`, `decode_attention_cpu`).
Registration is lazy (worker-side), so enumerate on-node — which the D2 probe did authoritatively:

| CPU attention/relevant op | present? | usable for DSV4 MLA-sparse as-is? |
|---|---|---|
| `decode_attention_cpu`, `extend_attention_cpu`, `flash_attn_varlen_func` | ✅ registered | ❌ MHA/GQA form — needs absorbed/sink/top-k **adaptation** |
| `qkv_proj_with_rope[_fused_weight]`, `fused_qkvzba_split_reshape_cat_cpu` | ✅ registered | ✅ projection/rope building blocks (wireable) |
| `topk_softmax_cpu`, `copy_all_layer_kv_cache_cpu` | ✅ registered | ✅ routing / KV-copy building blocks |
| `flash_mla*` (dense + sparse MLA serving) | ❌ CUDA-only | — must adapt a CPU donor or build the authored kernel |
| `index_gemm`, `sparse`, `indexer`, `compress` | ❌ not registered | authored on disk (`dsa_pilot/*.cpp`) but **unbuilt/unwired** |

Donor compute sources (x86 AMX) live in sglang's `kernels/aot/csrc/cpu/` (`flash_attn.cpp`, `gemm*.cpp`,
`moe.cpp`) and the plugin's authored `plugin/kernels/dsa_pilot/`.

## 3. The wiring surface (what to change), attention path FIRST

### Key constraint (why the scaffold exists)
The stock **`intel_amx` attention backend is MHA/GQA-oriented and REJECTS DSV4's MLA** (`_dsv4_cpu_infra.py:5`:
"assumes fp8 KV ⇒ MHA + intel_amx backend, and therefore REJECTS DSV4's…"). So DSV4 MLA-absorbed sparse
attention cannot simply route to the stock backend — it needs the MLA-absorbed form (`value==key`, per-head
sink, gather-by-topk) wired to the AMX primitives.

### First target — the MLA decode attention (`_batched_mla_attend`, 20%)
- **Form (authoritative):** MLA-absorbed — `out = softmax(q·kᵀ·scale ⊕ per_head_sink)·k`, `head_dim_v==D`,
  1 latent KV head (MQA), keys gathered by top-k indices. Oracle = the eager `_batched_mla_attend` itself
  (already parity-validated end-to-end) + the arch pure-torch ref.
- **Donor:** `flash_attn.cpp/.h` online-softmax (QK & AV via brgemm, `add_C` for AV) adapted to MLA (v==k,
  sink as a virtual key column dropped after softmax) + the DSA gather (`decode.cpp` stage-then-brgemm, or the
  authored `indexer`/`sparse_attend` kernels in `plugin/kernels/dsa_pilot/`).
- **Wiring point:** `_batched_mla_attend` (or `_torch_flash_mla_with_kvcache`) → call the AMX kernel; keep the
  eager path behind a flag as the reference oracle (A/B-diff same input = regression harness).

### Then — indexer / compressor / MHC (the rest of the 42%)
- `_cpu_forward_c4_indexer` → the authored indexer kernel already exists (`plugin/kernels/dsa_pilot/indexer_logits.cpp`,
  `_dsa_ckernels.indexer_logits`) + `decode.cpp index_gemm`; wire it to replace the torch einsum.
- `_cpu_forward_core_compressor` → the authored `compressor.cpp`.
- MHC `hc_pre/hc_split_sinkhorn/hc_combine` → the authored `sinkhorn.cpp` / `combine.cpp`.
  (These authored kernels exist and were standalone-validated; the gap is wiring them into the serving forward,
  which the held-fixed A/B earlier showed is neutral-at-decode *in the dummy proxy* — re-measure on real.)

## 4. Iteration vehicle — tiny-faithful (not the 149 GB model)

Build a **tiny but architecturally-faithful** config: copy EVERY arch switch from the real config (MLA, DSA on,
MHC/hash on, MXFP4 block-quant, head_dim/rope/compress-ratios, `num_key_value_heads==1`) but shrink the
dimensions (2 layers, tiny hidden/experts), `--load-format dummy`. Runs in **seconds on a login node**,
exercises the **same wiring/infra code paths**, and the optimized kernels sit behind a flag so the eager path
stays runnable as the A/B oracle. (Attention correctness is weight-independent → tiny-faithful validates it;
accuracy still needs a real-weight run at the end.)

## 5. First concrete step (on-node, cheap)
1. ~~**Enumerate the available CPU attention ops**~~ — **DONE** (D2 pre-commit probe,
   `results/d2_precommit_probes.json`): resolved the wire-vs-author fork — see the §2 table and the ⚠️ box.
   No CPU MLA/sparse serving op exists; the DSA compute kernels are authored-but-unbuilt; the generic CPU
   attention ops need absorbed-form adaptation. **So each attention piece is adapt/build+wire, not drop-in.**
2. **Decision gate BEFORE building:** given the above + Amdahl (attention ≈20% ⇒ ~1.25×), a memory-bound
   decode, and an already-accurate running model (gsm8k 0.797), confirm the effort is worth the multi-week
   build/adapt/integration surface, or re-scope to the highest-ROI single piece first.
3. If proceeding: **stand up the tiny-faithful config** + the eager-vs-optimized A/B flag harness, then
   **build+wire one piece** (attention first — adapt a CPU donor or build `dsa_pilot/sparse_attend.cpp`),
   A/B-diff vs the eager oracle on tiny-faithful, and re-measure the real-weight decode headline.

> **Measurement caveat (from the probes):** the synthetic resident-engine perf-probe accumulates radix/KV
> state and degrades decode after the FIRST measurement (390564 rep1 ran >29 min vs rep0 ~140 s); `flush_cache()`
> blocks pathologically on the CPU engine. A clean held-fixed sweep must measure each config in a **fresh
> engine (reload)**. The real-weight M=32 decode baseline is **~0.5–0.8 tok/s (~41–46 s/step)**, reproducible;
> the earlier 1.59 tok/s was an outlier.

## Gate (cpu-serving-integration PROOF)
Done = runs end-to-end on CPU + accuracy-parity vs the eager/reference oracle (real weights) + published
roofline-target-vs-measured at a labeled config. The eager scaffold is the correctness oracle for every step.
