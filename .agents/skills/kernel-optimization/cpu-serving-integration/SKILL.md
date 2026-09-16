---
name: cpu-serving-integration
description: "Use in the new-kernel leg AFTER the novel compute kernels are authored + parity-validated (kernel-authoring) and BEFORE the enablement-certificate. Captures how to wire authored CPU kernels into the model's SERVING RUNTIME so it actually runs end-to-end: the attention BACKEND forward, the paged KV-cache (exact packed byte layout + pack/unpack), the forward-batch metadata, and absorbed-attention specifics. Distinguishes the (smaller) novel-compute authoring from the (larger) serving-runtime port, and ends at the workflow's north-star PROOF: model runs + accuracy vs a trusted reference + roofline-target-vs-measured published. Learnings distilled from the DeepSeek-V4 DSA/MLA CPU bring-up."
---

# CPU Serving Integration (make the model actually run)

Authoring the novel kernels (kernel-authoring) proves the math; it does NOT make the
model serve. A modern arch's attention runs through a **backend forward** that calls a
serving kernel (e.g. `flash_mla_with_kvcache` / `flash_mla_sparse_fwd`) over a **paged
KV cache**, driven by per-query indices + metadata. Getting a running model means
porting THAT layer to CPU — a larger, distinct effort from the compute kernels it calls.

## Pipeline position
`kernel-authoring (novel compute, parity-validated) → cpu-serving-integration →
accuracy-oracle → roofline-vs-measured → enablement-certificate`. This is the "make it
serve" phase; its gate is the workflow's proof deliverable.

## The serving-runtime surface (what to port)
1. **Attention backend forward.** The arch's backend (`<arch>_backend.py`) `forward`
   calls the CUDA serving kernel over the paged cache. Reimplement it in torch/CPU,
   REUSING the already-authored compute kernels. Know the attention form:
   - **MLA-absorbed** (DeepSeek): value == key (one latent), so
     `out = softmax(q·kᵀ·scale ⊕ per_head_sink)·k` — the per-head attention *sink* is a
     virtual key column dropped after softmax. `head_dim_v == head_dim`.
   - GQA/MHA: standard, k and v distinct.
2. **Paged KV cache pack/unpack.** The cache is a **packed byte buffer**, not plain
   tensors — you MUST reproduce the exact byte layout to write AND read it. DSV4 example:
   584 B/token = 448 fp8 nope + 64 bf16 rope (128 B) + 7 ue8m0 tile-scales + 1 pad, with
   nope/rope in the token region and scales in a separate per-page region (see
   `index_buf_accessor._set_k_and_s_torch` — it has an in-tree torch writer; write the
   matching **unpack**). Per-64-tile fp8 uses pow2 (ue8m0) scales: `val * 2^(scale-127)`.
3. **Metadata prep.** positions, compressed-attn metadata, page indices, topk lengths —
   several are Triton with torch fallbacks (route via `enablement-scope-discovery`).
4. **Gather.** Per query, gather its attended tokens via the page indices (+ compressed
   "extra" indices), unpack to bf16, attend.

## Procedure
1. Read the backend `forward` to get the interface (q/k/v shapes, compress_ratio, sink,
   which caches + index tensors) and the serving kernel's contract.
2. Get the AUTHORITATIVE attention math from the arch's **pure-torch test reference**
   (`test/kits/.../<arch>_attention.py`) — it is the oracle for the absorbed form + sink.
3. Port `forward` to torch: store KV (packed) → read caches → unpack → per-query gather →
   attention (reusing authored kernels) → return. Iterate on a TINY same-arch config.
4. Validate: (a) runs end-to-end; (b) `accuracy-oracle` — per-layer/logits parity + task
   vs a trusted reference (HF / GPU path) on REAL weights; (c) `model-profile-hotspots` →
   `roofline_vs_measured.py` for the perf proof (same machine config, labeled).

## Learnings (DeepSeek-V4, transferable)
- **A novel-feature backend has no dense fallback** — DSA sparse attention IS the dsv4
  forward; you cannot bypass it to a dense path. Verify a fallback exists before
  planning any bypass (also in `model-profile-hotspots`).
- **Compute authoring ≪ serving port.** The novel DSA math (compressor/indexer/sparse)
  was a handful of small parity-validated kernels; the paged flash-MLA serving forward
  (unpack + absorbed attention + sparse gather + plan/state-pool) is the larger effort.
  Scope them separately in `enablement-scope-discovery`.
- **The paged cache is a packed byte layout** with a separate scale region — reproduce it
  exactly (there is usually an in-tree torch accessor for the write; mirror it for read).
- **v==k + sink** is the MLA-absorbed shortcut that makes the torch attention simple.

## Gate — the workflow's PROOF deliverable
Enablement is only "done" when the model **RUNS end-to-end on CPU**, is **accuracy-parity
vs a trusted reference** (accuracy-oracle, real weights), AND has a **published
roofline-target-vs-measured** perf artifact at a labeled machine config. Running without
accuracy, or accuracy without the perf-vs-roofline proof, is not done — the certificate
requires all three (enablement-certificate). That triple IS the thesis proof.
