---
name: model-op-decomposition
description: "Use FIRST for a new model enablement, right after node calibration. Turns a model config (HF config.json + the SGLang model class) into a normalized op graph: layer inventory, per-op dtypes and shapes, attention kind (MHA/GQA/MLA/sliding-window/sparse), positional scheme (RoPE variant / NoPE), norm type + placement (pre/post, QK-norm), MLP/activation (SwiGLU/GeGLU), and MoE topology (num experts, top-k, shared expert, grouping). This graph is the input to coverage-gate; getting it complete and correct is what makes coverage trustworthy."
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
norm_kind?, act?, moe?}` entries covering 100% of the compute in a decoder step.
Anything you cannot classify is marked `UNKNOWN` — never guessed — so coverage
treats it as a gap.

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
