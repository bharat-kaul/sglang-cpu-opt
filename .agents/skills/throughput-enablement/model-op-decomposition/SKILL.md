---
name: model-op-decomposition
description: "Use FIRST for a new model enablement, right after node calibration. Turns a model config (HF config.json + the SGLang model class) into a normalized op graph: layer inventory, per-op dtypes and shapes, an UPFRONT data-type audit mapping each op's ACTUAL stored checkpoint dtype to the best target-platform compute dtype (flagging missing-native-compute dequant paths, low-bit-checkpoint INFLATION traps, and suboptimal fp32/torch ops), attention kind (MHA/GQA/MLA/sliding-window/sparse), positional scheme (RoPE variant / NoPE), norm type + placement (pre/post, QK-norm), MLP/activation (SwiGLU/GeGLU), and MoE topology (num experts, top-k, shared expert, grouping). This graph is the input to coverage-gate; getting it complete and correct is what makes coverage trustworthy."
---

# Model Op Decomposition

Produce the normalized op graph the coverage gate reasons over. A missed op here
becomes an undetected gap later, so enumerate exhaustively from the model source,
not from assumptions about the family.

## Inputs
- HF `config.json` (or the SGLang `configs/<model>.py`) — dims, heads, KV heads,
  layers, intermediate size, num experts / top-k, rope params, sliding window,
  vocab.
- The SGLang model class (`python/sglang/srt/models/<model>.py`) — the ground
  truth for which layer objects actually run (`RadixAttention`, `RMSNorm`,
  `RotaryEmbedding`, `SiluAndMul`, `FusedMoE`, linear parallel layers).
- A reference forward (HF `modeling_*` or the GPU SGLang path) for shape capture.

## Procedure
1. **Layer inventory.** Walk the decoder block once; list every op instance with
   its class and constructor args. Note repetition (× num_layers) and any
   per-layer variation (e.g. interleaved sliding-window layers, dense-vs-MoE
   layers, first-k-dense-then-MoE).
2. **Per-op signature.** For each op record: dtype (bf16/fp16/int8/fp8), the
   operand shapes at the target seq/batch, and the divisibility of GEMM dims
   (OC, IC) — coverage checks `OC%16==0`, `IC%32==0` for AMX packing.
2b. **DATA-TYPE AUDIT vs the TARGET PLATFORM (do this UPFRONT, from the real
   checkpoint — it decides footprint, kernel, and perf before any run).** Census
   the *actual* stored dtype of EVERY weight from the checkpoint headers (e.g.
   safetensors `__metadata__`/dtype per tensor), grouped by op role — do NOT trust
   the top-level `quantization_config` (it often describes only the dense/attention
   blocks; MoE experts, indexer, shared experts, embeddings, norms frequently
   differ). **FETCH METADATA-ONLY FIRST — do NOT wait for the full (often ~TB)
   download.** The audit needs only `config.json` + the safetensors INDEX
   (`model.safetensors.index.json`), optionally one shard's leading JSON HEADER — a few
   KB that carry every tensor's dtype + shape. Pull those while the full weight download
   runs in the BACKGROUND (`model-enablement-playbook` §"Parallelize the long pole"), so
   the whole audit + scope/coverage completes before any tensor bytes land, and the
   small-file fetch also confirms repo access/gating early. Real case (GLM-5.3 Flash):
   metadata-only fetch finished the dtype audit — fp8 e4m3 128×128 block-quant
   experts/proj + bf16 rest → one W8A16 bridge — with zero weight bytes downloaded. Then
   map each op's stored dtype → the **best compute dtype the target HW actually
   supports** and flag three things:
   - **Missing native compute → dequant path.** GNR AMX has bf16/int8/fp16 only —
     no native fp8 or fp4 matmul. So fp8 weights → W8A16 (dequant→bf16 AMX), fp4/
     MXFP4 → W4A16 (fused dequant→bf16). Confirm a kernel exists for each (grep the
     kernel lib for the native quant enum: `MXFP4`, `INT4_W4A8`, `FP8_W8A16`).
   - **INFLATION traps.** If any pipeline step UP-converts a low-bit checkpoint to a
     bigger dtype to reach a familiar kernel (fp4→fp8, fp8→bf16), FLAG IT — it can
     multiply the resident footprint and manufacture a one-domain overflow (see
     `sub-numa-clustering` §UPFRONT OOM CHECK and `quantization-amx-int8`: never
     inflate a native low-bit checkpoint; keep it low-bit + use the low-bit kernel).
   - **Suboptimal-dtype ops.** Flag ops running as generic fp32/torch where a bf16/
     int8 AMX kernel exists (e.g. custom attention/indexer sub-ops); those leave
     the AMX tiles idle. Route them through the packed AMX GEMM.
   - **CORRECTNESS OBLIGATION (auto-emitted, do not skip).** Whenever stored_dtype ≠
     target_compute_dtype there is a DEQUANT/REPACK BRIDGE (fp4→bf16, fp8→bf16, int4
     unpack, VNNI prepack, group/block scale decode) sitting between the checkpoint
     and the donor kernel. "A bf16/fp8 kernel exists" (coverage) is NOT the same as
     "the bridge is numerically correct" — the scale layout, nibble order, SwiGLU
     gate/up split, and pack dispatch are all easy to get silently wrong (they
     produce plausible-looking garbage, not a crash). So for EACH such op family the
     audit MUST emit a required numeric **parity gate**: compare the real kernel's
     output to an INDEPENDENT torch dequant+compute oracle built from the SAME packed
     bytes (see `accuracy-oracle` §low-bit parity gate). A converted-dtype op is only
     "done" once this gate passes — end-to-end coherence alone does not localize a
     bridge bug to the kernel.
   Emit a table `{op, stored_dtype, target_compute_dtype, kernel, inflation?,
   dtype_bridge?, parity_gate, note}`.
   This audit is the input to coverage-gate (a missing low-bit kernel is a gap; a
   present one with a dtype_bridge carries a correctness obligation, NOT an
   unconditional "covered"), to accuracy-oracle (every dtype_bridge → one parity
   gate), and to the capacity/OOM check (resident footprint = sum of stored bytes,
   NOT the inflated ones).
3. **Attention classification.** MHA / GQA (num_kv_heads) / MLA (kv-lora,
   absorbed) / sliding-window (window size, interleave pattern) / sparse
   (indexer/selector). Record mask kind and head_dim. This is the single most
   common source of a hidden gap.
4. **Positional scheme.** RoPE (theta, scaling: linear/dynamic/yarn/llama3),
   partial-rotary, or NoPE. QK-norm present? (OLMo2/OLMoE apply RMSNorm to q,k.)
5. **MLP + activation.** SwiGLU (`SiluAndMul`) / GeGLU (`GeluAndMul`) / plain;
   gate_up fused?
6. **MoE topology (if any).** num_experts, top_k, renormalize, shared expert(s),
   grouped/segmented routing, expert dtype/quant. Maps to the FusedMoE CPU path.
7. **Embedding + head + final norm.** VocabParallelEmbedding, tied weights, logits
   softcap.
8. Emit the graph as a compact structured record (one row per distinct op kind
   with count, dtype, shape family, and the classification fields above).

## Output contract
A list of `{op_kind, count, dtype, shape_family, attention_kind?, pos_scheme?,
norm_kind?, act?, moe?}` entries covering 100% of the compute in a decoder step,
PLUS the data-type audit table `{op, stored_dtype, target_compute_dtype, kernel,
inflation?, note}` (step 2b). Anything you cannot classify is marked `UNKNOWN` —
never guessed — so coverage treats it as a gap.

## Gate
Re-derive the graph for an already-enabled reference model of the same family and
diff against its known structure; the decomposer must reproduce it exactly before
you trust it on the new model.

## Pitfalls
- Per-layer heterogeneity (sliding-window interleave, first-dense-then-MoE) is
  easy to miss — walk EACH layer index, do not assume a uniform stack.
- "Same family" ≠ "same ops": a new RoPE scaling or a new attention-sink term is a
  distinct op the registry must have a contract for.
- Fused ops in the SGLang class (fused qkv, fused gate_up) must be recorded as the
  fused kernel, not the mathematical sub-ops, so coverage matches the real kernel.
- This graph is the COMPUTE op list only. It does NOT capture the runtime substrate
  (KV pool, allocator, backend guards) or novel families in non-attention subsystems
  (norm-path hash-clustering, etc.). Hand off to `enablement-scope-discovery` for the
  dependency-closure that surfaces those BEFORE bring-up.
