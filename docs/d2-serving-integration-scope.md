# D2 — CPU serving-integration scope: route DSV4 decode through the optimized AMX kernels

**Goal:** replace the eager-torch *correctness-first scaffold* (the cause of ~46 s/step real-weight M=32
decode) with the optimized AMX CPU kernels, so decode moves toward its roofline. The kernels largely already
exist (sgl_kernel CPU corpus); this is a **wiring/integration** effort, not a from-scratch authoring one.

Pipeline position (cpu-serving-integration): the DSV4 CPU path is at **stage 1 (structural bypass — runs +
accurate but eager-torch)**; this scope is **stage 2 (wire the optimized kernels)**.

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

Not the `sgl_kernel` python module (`import sgl_kernel` fails) — the AMX CPU kernels are **registered torch
operators**: `torch.ops.sgl_kernel.<op>` (e.g. `fused_experts_cpu`, `convert_weight_packed`). Registration is
LAZY — it happens when the engine worker loads the kernel extension, so enumerate on-node, not on the login
shell. The donor compute kernels live in sglang's `kernels/aot/csrc/cpu/` (x86 AMX): `flash_attn.cpp`,
`decode.cpp` (DSA `index_gemm_kernel_nt`=Q@Kᵀ, `index_gemm_kernel_nn`=P@V), `gemm*.cpp`, `moe.cpp`.

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
1. **Enumerate the available CPU attention ops:** in the engine worker (or a tiny-faithful load), print
   `torch._C._dispatch_get_all_op_names()` filtered to `sgl_kernel::` — confirm which of
   {flash/extend/decode attention, `index_gemm`} are built for CPU. This decides *wire-existing* vs
   *author-adapt-from-donor* for each piece.
2. **Stand up the tiny-faithful config** + the eager-vs-optimized A/B flag harness.
3. **Wire the MLA decode attention** (first target) and A/B-diff vs the eager oracle on tiny-faithful; then
   re-measure the real-weight decode headline.

## Gate (cpu-serving-integration PROOF)
Done = runs end-to-end on CPU + accuracy-parity vs the eager/reference oracle (real weights) + published
roofline-target-vs-measured at a labeled config. The eager scaffold is the correctness oracle for every step.
