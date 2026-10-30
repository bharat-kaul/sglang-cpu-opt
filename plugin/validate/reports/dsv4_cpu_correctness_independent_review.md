# DSv4 Flash CPU correctness: independent review and handoff

Date: 2026-10-23  
Reviewed repository HEAD: `dc62aa04e8ba73906f0d093420289a04b2b7bbbb`

## Latest follow-up: after the direct compressed-KV handoff (2026-10-23)

**Recommendation: do not spend the next model run A/B-ing block-start versus
an invented block-representative RoPE position.** The authors' reference
explicitly uses block start, including for C4 overlap:
`freqs_cis[:cutoff:ratio]` during initial prefill and
`freqs_cis[start_pos + 1 - ratio]` when a decode block completes.
Check the actual frequency tensors and selected rows, not speculative positions.

The updated investigator brief reports that the old missing compressed source
was observed in a live run, and that the new direct handoff is consumed
(`parts=2`, `extra_K=(291,512)`). This supersedes the old disconnected-stash
description for that new path. However, a worse output cosine after adding the
branch does not establish that compressed-key values are the only remaining
problem. Q, membership, scale, sink and output processing still require
same-input checks at layer 2. Layers 0/1 do not certify those layer-2 operands.

### The offline producer claim needs correction

[`test_compressor_offline.py`](../test_compressor_offline.py) does not call
the production compressor. It compares its own `corrected_compress_prefill`
against its own `reference_compress_prefill`, sharing helpers and essentially
the same implementation. It bypasses the fused projection/channel split,
loaded-APE inversion, production RMSNorm, FP8 roundtrip and handoff. The test
also skips lengths below one complete block.

I executed the actual production function definitions in isolation on the
same synthetic operands as that test oracle (C4, length 12, D=512, RoPE=64,
seed 71). FP8 simulation was deliberately disabled to compare the pre-FP8
boundary:

| Comparison | Result |
|---|---|
| Test replica versus test reference | `torch.equal=True` |
| Actual producer versus test reference | `torch.equal=False`, maxabs `0.015625`, cosine approximately `0.99999982` |

There is a concrete dtype-boundary difference: production `_rmsnorm_torch`
returns the norm weight's FP32 dtype, while the test/reference casts back to
the BF16 activation dtype before RoPE. This small discrepancy is **not evidence
that it causes the large model divergence**. It demonstrates why the reported
zero-error test cannot certify the production path.

### Disposition of the four proposed conventions

1. **RoPE position:** block start is specified. Verify actual CPU/GPU frequency
   rows, including YaRN configuration and the common layer-2 Q/window/compressed
   coordinate system. Base 160000 alone does not identify the entire table.
2. **Half ordering:** installed SGLang loads `cat([wkv, wgate])`; the reference
   uses previous-block first-half channels and current-block second-half
   channels. The revised split agrees with this source contract. Verify the
   actual loaded projection and APE against checkpoint values on identical x.
3. **Representation/absorb:** the reference concatenates window and compressor
   outputs directly before sparse attention. There is no additional
   compressed-only absorb transform in that forward. Do not invent a rescaling.
4. **Sink:** one per-head sink participates in the joint denominator for
   window plus compressed keys; it is not duplicated per source.

Also, 291 of 419 keys is a **count fraction**, not a measured softmax mass.
It neither predicts cosine 0.196 nor proves that the compressed source dominates.
Measure per-head window/compressed/sink mass and logit ranges.

### Next step: one layer-2 boundary capture, then offline replay

Capture matched request/token/global-head coordinates and:

- Actual compressor input x and fused projection; loaded APE/norm identity.
- Pooled output, post-norm output with dtype, post-RoPE output, post-FP8 output.
- Actual Q, SWA keys, compressed keys and source-labelled membership consumed
  by attention; frequency rows, scale and sink.
- Attention-core output before and after inverse-RoPE, and module output.

Compare production against an independent reference on **identical operands**,
first validating the replay on GPU operands/output. The first failing boundary
determines the fix: projection/weights, pool/APE, norm/rotation/quantization,
handoff/mask, attention arithmetic, or output processing. Preserve aligned Q;
the earlier rank-local last-head capture was not sufficient.

Cheap first action: change the offline test to invoke the actual producer,
rather than another transcription. No expensive model run is needed for that.

The current direct handoff is still an investigative prefill path, not a
general serving repair: decode skips production, cross-request identity is
absent from its visibility mask, and it does not enforce top-k selection when
completed compressed blocks exceed 512. These do not by themselves explain
the single-request, 291-block observation, but they block claiming GSM8K
generation or general serving correctness.

The checks in this follow-up are source review and an isolated numerical
probe, not a new model run or proof of recovered accuracy.

## Earlier follow-up: compressed-attention localization

**This section supersedes the investigation priority below.** The repaired
captures make the compressed-attention integration the next priority, rather
than more clipping experiments. The earlier clipping contract finding remains
valid; it has not been established as the cause of the accuracy deficit.

### Verified new evidence and qualifications

I verified that the CPU/GPU `.attnout` artifacts contain identical input IDs,
matching all 1,167 tokens of frozen sequence 0. Recomputed output cosines and
norm ratios in FP64 after checking equal shapes and finite values:

| Layer | CPU/GPU cosine | CPU/GPU norm ratio |
|---|---:|---:|
| 0 | 0.999488 | 0.999644 |
| 1 | 0.999494 | 0.992583 |
| 2 | 0.196443 | 0.734443 |
| 3 | 0.502771 | 1.226028 |

These are **last-token attention-module output vectors**, not every token or
same-input sub-op comparisons. They strongly prioritize the first compressed
layer, but do not globally certify every preceding subsystem.

The clipping diagnostic records only the first encountered routed expert in
the first diagnostic call: `_SWIGLU_DIAG["done"]` and `["stats"]` suppress later
measurements. Shared experts were not clamped by the described A/B. Thus
"0% on the measured sample; no material improvement in this routed-only A/B"
is supported; "clipping never fires anywhere" is not. Do not let this
qualification delay investigating the much stronger layer-2 signal.

### Priority 1: compressed KV producer is disconnected from attention

In `_dsv4_cpu_infra.py`:

- `_cpu_forward_core_compressor` writes `_COMPRESSED_STASH[(layer_id, ratio)]`.
- Repository search and AST inspection found assignments, but no consumers of
  `_COMPRESSED_STASH`. `_TOPK_STASH` likewise has writes and no readers.
- `_batched_mla_attend` and its loop alternative instead gather keys from
  `_KV_BUF` / `_KV_VALID`, keyed by the actual cache tensor's `data_ptr()`.
- The CPU compressor override neither writes the backend's extra-key cache nor
  populates that cache's CPU attention stash.
- The normal backend passes the SWA cache as `k_cache` and the compressed cache
  as `extra_k_cache`. The CPU extra-cache gather can return `None`, silently
  omitting that contribution.
- The CPU indexer override does not publish selected physical compressed-slot
  IDs into the backend's `c4_sparse_page_indices`; it publishes a different,
  raw-token mask through `_SEL_HOLDER`.

This is a missing producer-to-consumer connection in the reviewed current
worktree, not a rounding hypothesis. The saved captures do not bind this
worktree's exact source content, so the findings are not proof that these
same definitions executed in the historical capture job.
The next live probe should count actual compressed rows written and read under
the **same layer/cache identity**.

Executed isolated reproduction using unchanged function definitions from the
worktree, without an Engine:

- Running the compressor produced two compressed rows but populated zero
  `_KV_BUF` destinations.
- With zero query, sink zero, SWA value `[1, 0]` and requested compressed value
  `[0, 3]`, the attention function returned `[0.5, 0]` when the extra stash was
  absent: exactly SWA-only behavior.
- Explicitly writing the compressed cache changed the result to
  `[1/3, 1]`, as required by the joint softmax over two keys plus sink.

This demonstrates the isolated omission mechanism, not measured full-model
accuracy restoration.

### Priority 2: C4 compressor channel layout and overlap are wrong

The installed `Compressor` has `coff=2` for ratio 4. Its fused projection output
is `[T, 4D]`, with weights loaded as `cat([wkv, wgate])`:

```text
[KV_previous_half | KV_current_half | score_previous_half | score_current_half]
```

Both CPU compressor implementations slice `[:D]` as KV and `[D:2D]` as score.
They therefore use the **second KV half as scores**, ignoring the actual score
channels. The single-window pooling and `ape.view(-1, D)[:ratio]` also do not
implement C4 overlap.

Executed falsification probe: change only the actual score channels of an
`[8, 4D]` projection tensor. The CPU core compressor output is unchanged.

The authors' C4 compressor pools the previous block's first half together with
the current block's second half, over up to eight positions. In the first block,
the missing previous block has zero values and negative-infinite scores.
SGLang also rearranges APE at load time; respect the loaded layout rather than
assuming the raw checkpoint layout.

The CPU core override further stops at RMSNorm: it lacks the compressed-key
RoPE, quantization and destination-cache write performed by the reference.
For decode it explicitly stashes an empty tensor rather than maintaining
incremental compression state. A one-line score-slice correction is therefore
insufficient.

### Targeted answers: which sub-op and which subtleties?

**Order:** compressed-cache wiring and actual key membership first; C4
compression layout/overlap next; compressed-key RoPE/quantization; only then
indexer score fidelity. Include inverse-RoPE and output projection in the
boundary capture, but do not start by changing the already shared frequency
table generator.

For the single 1,167-token captured prompt, the last query has 291 completed
ratio-4 blocks, fewer than top-k 512. Correct selection therefore includes every
eligible compressed block. Ranking precision alone cannot remove valid blocks
in that case; bad membership, coordinates, validity masks or cache population
still can.

Reference details to enforce:

- Attention sees **128 recent uncompressed keys plus 291 compressed keys** at
  the last query, plus one zero-value sink in the softmax denominator. These
  are separate representations, not deduplicated raw-token positions.
- Compressed block `j` becomes visible at query position `t` only when
  `(j + 1) * ratio <= t + 1`. The CPU's `key_start <= position` predicate is
  too early for incomplete blocks. Reject invalid selected indices after
  top-k as well as masking logits before selection.
- Compressed RoPE is applied after pooling and normalization, at the block
  start position. Q uses its query position; inverse-RoPE applies to the final
  attention output at the query position.
- Indexer query and index keys use matching RoPE, normalized Hadamard rotation,
  and the reference's quantization contract. The CPU query skips FP4
  simulation; its index keys omit additional reference transforms.
- The indexer weight factor is `head_dim**-0.5 * n_heads**-0.5`; the current
  CPU path reads the installed `weight_scale`, so this factor is not the
  strongest lead.
- Do not expand compressed IDs into a raw-token bitmap and apply it by list
  position to SWA cache indices. Preserve source, request, logical block and
  physical slot identity.
- Preserve compression state across chunked prefill and decode, including
  incomplete blocks and the overlap carry.

### Review of the subsequent FORCE_DENSE confirmation

**The new result supports the investigation priority, but is not an independent
confirmation that compressed-KV content is the cause, nor an exclusion of all
indexer wiring defects.**

The actual `_batched_mla_attend` implementation uses `force_dense` to change
the **main/SWA** mask. It still uses the supplied `extra_indices` and
`extra_topk` for the compressed cache. The loop path likewise retains explicit
compressed-cache selection. Neither path synthesizes missing compressed keys.
Thus this switch is not "all eligible compressed keys, bypassing the indexer."

I executed the unchanged attention function with controlled inputs:

| Fixture | Result |
|---|---|
| Compressed cache absent, force off versus on | Both `[0.5, 0]`, verified with `torch.equal` |
| Compressed cache populated, force on, select extra slot 0 | `[1/3, 1]` |
| Same inputs and force on, select extra slot 1 | `[1/3, 3]` |

The positive control demonstrates that compressed selection remains active
under `force_dense`. The negative control demonstrates that dense/sparse
equality is also expected when the compressed branch is simply absent.
These are isolated-function tests, not a replay of the reported model job.

The updated response reports the same GPU-relative cosine for dense and sparse.
Equal rounded cosines are not evidence that the two CPU tensors are identical.
Use distinct capture files, exact input/run/config identity, `torch.equal` and
maximum absolute difference to substantiate that claim.

The Q tap also flattens tokens and heads and records the last row. A CPU
last-global-head vector and a GPU rank-local last-head vector need not represent
the same head. Preserve `(token, global_head, tp_rank)` identity, or gather all
head shards in the verified order. Q is still a useful boundary once aligned;
the current mismatched tap cannot clear the Q/RoPE path.

Recommended wording for the other brief: "The FORCE_DENSE switch did not
improve the reported layer-2 cosine. Source inspection and isolated probes
prioritize compressed-cache wiring and C4 compression defects; end-to-end
causal attribution remains pending."

### Next decisive probe

On the existing frozen prompt, capture layer 2's input, Q, SWA cache rows,
compressed rows after each transform, selected logical/physical IDs, actual
gather masks, attention output before/after inverse-RoPE, and projection output.
First assert that the final query consumes 128 SWA and 291 compressed entries.
Fail on missing cache writes rather than silently dropping an entire source.

Validate short boundaries offline at lengths 3/4/5 and 7/8/9 for C4 overlap,
127/128/129 for SWA and C128, and equivalent monolithic/chunked/decode execution.
Then replay the actual attention operation on identical operands.

The request gate still accepts a same-length nonconstant fallback after exact
ID matching fails. Remove that fallback and preserve rank-specific captures.
The checked artifacts do match exact IDs; the fallback is a remaining
instrumentation weakness, not a reason to discard those particular artifacts.

## Read this first

**Two concrete findings change the investigation:**

1. The CPU routed and shared expert paths omit the checkpoint's required
   **SwiGLU clipping at 10.0**. The BF16 alternative and in-situ MoE oracle share
   this omission. Their agreement does not exonerate MoE.
2. The saved GPU attention capture contains **256 repetitions of checkpoint
   token 0**, not the frozen diagnostic prompt. The CPU capture starts with the
   real prompt token, 10375. The cross-device embedding mismatch is explained by
   different captured inputs, not merely TP or MHC layout.

The clipping omission is a demonstrated computation-contract defect. **Its
contribution to the real accuracy gap has not yet been measured.** Do not claim
that fixing it restores accuracy until a matched evaluation establishes that.

This review made no implementation changes and launched no model jobs. It used
saved evidence, checkpoint embedding slices, source inspection, and small
installed-native-operator probes.

## 1. Matched-question accuracy confirms the gap

Re-scored saved CPU and GPU generations with the existing extractor and checked
matching gold labels:

| Same question IDs 40-319 | Correct | Accuracy |
|---|---:|---:|
| GPU post-trained Flash | 270/280 | 96.43% |
| CPU Flash, job 391610 | 223/280 | 79.64% |

Paired outcomes: 49 GPU-correct/CPU-wrong, 2 CPU-correct/GPU-wrong, 221 both
correct, 8 both wrong.

Including shard 0's saved partial results gives 312 matched questions:
GPU 301/312, CPU 246/312.

The earlier CPU job 390591 scores 216/280 on IDs 40-319, versus 223/280 for
391610. This is a matched historical improvement of seven answers, not proof
that a specific fix caused it.

Evidence:

- [GPU summary](../results/gpu_flash_gsm8k.json)
- [GPU generations](/scratch/bkaul/gpu_flash_gens.json)
- CPU generation files:
  `/scratch/bkaul/dsv4_d2acc_391610_gens0.json` through `gens7.json`
- Historical CPU generation files:
  `/scratch/bkaul/dsv4_d2acc_390591_gens0.json` through `gens7.json`

These observations supersede the earlier concern that a Base-versus-post-trained
checkpoint mismatch might explain the entire discrepancy.

## 2. Demonstrated defect: missing DeepSeek SwiGLU clipping

### Authoritative contract

[Checkpoint config](/scratch/bkaul/models/DeepSeek-V4-Flash/config.json) declares
`swiglu_limit: 10.0`.

The authors' `Expert.forward` in
[reference model.py](/scratch/bkaul/models/DeepSeek-V4-Flash/inference/model.py)
computes:

```python
gate = self.w1(x).float()
up = self.w3(x).float()
gate = gate.clamp(max=10)
up = up.clamp(min=-10, max=10)
activation = F.silu(gate) * up
```

This applies to routed experts and the shared expert. Gate has an upper clamp
only; up has both lower and upper clamps.

### CPU paths that omit it

In [the CPU plugin](../../intel_cpu_models/_dsv4_cpu_infra.py):

- `_patched_apply` invokes `fused_experts_cpu` for MXFP4 with
  `rc.gemm1_alpha` and `rc.gemm1_clamp_limit`, not `rc.swiglu_limit`.
- `_torch_fp4_moe_apply`, the BF16 diagnostic alternative, uses unbounded
  `F.silu(g) * u`.
- `_mxfp4_insitu_parity`, the local oracle, also uses unbounded `F.silu(g) * u`.

In installed
[DeepseekV2MoE.forward_cpu](/scratch/bkaul/venvs/sglang-cpu/lib/python3.12/site-packages/sglang/srt/models/deepseek_v2.py),
the shared expert is sent directly to `torch.ops.sgl_kernel.shared_expert_cpu`.
That operator has no clipping parameter and bypasses the ordinary shared
expert forward's clipping logic.

The installed
[MoeRunnerConfig construction](/scratch/bkaul/venvs/sglang-cpu/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/layer.py)
keeps `swiglu_limit` separate from `gemm1_clamp_limit`; these are not aliases.

### Native-operator probes executed

The probes used the installed CPU binary, one thread, small exactly
representable weights, and below-limit positive controls.

| Operator family | Gate/up = 4: native / reference | Gate/up = 32: native / reference |
|---|---:|---:|
| Routed MXFP4 | 15.6875 / 15.6875 | **1024 / 100** |
| Shared FP8 W8A16 | 15.6875 / 15.6875 | **1024 / 100** |
| Shared BF16 | 15.6875 / 15.6875 | **1024 / 100** |

The output projection selects the affected activation. In the second column
clipping is inactive; in the third the reference clips both projected values
to 10.

**Inference:** the CPU family implements the unclipped function for these
inputs. The existing high-cosine MoE check compares two implementations of that
same unclipped function. It is not a source-faithful validation of DeepSeek's
expert computation.

### Do not apply the wrong fix

Do not simply populate the existing CPU `alpha/limit` arguments.
The installed
[clamp_sigmoid_and_mul_stub](/scratch/bkaul/venvs/sglang-cpu/lib/python3.12/site-packages/sglang/kernels/aot/csrc/cpu/moe.h)
uses an interleaved gate/up layout and computes a factor of **`up + 1`**.
DeepSeek requires non-interleaved gate/up halves and **no `+1`**.

Fix both routed and shared surfaces, preserve the native packed-weight compute
path, and test the precise DeepSeek formula rather than reusing an incompatible
activation mode.

### What remains unknown

- How frequently real gate/up values cross the clipping bounds, by layer,
  routed versus shared expert, and prefill versus decode.
- The corresponding FFN output change on identical real operands.
- How much of the GSM8K deficit this omission explains.

Those are the next measurements, not reasons to continue treating MoE as
exonerated.

## 3. Demonstrated debugger defect: wrong GPU input captured

Compared `layer_in` against checkpoint `embed.weight` using small safetensor
row slices:

| Capture | Result |
|---|---|
| [CPU attention capture](../results/attnio_cpu.pt) | Shape `[1167, 4, 4096]`; first embedding exactly equals token 10375, the first frozen input token |
| [GPU attention capture](../results/attnio_gpu.pt) | Shape `[256, 4, 4096]`; **every token and all four MHC lanes exactly equal token 0's embedding** |

The GPU data is consistent with a startup/dummy forward. It is not the frozen
request. This observation does not identify which startup mechanism invoked
the forward, but establishes that the captured inputs are different.

[Capture code](../../dsv4_layer_capture/__init__.py) records the first model or
layer-0 invocation without binding it to expected input IDs.

Other capture limitations observed:

- Both attention dumps lack the intended `kv` and `woa_in` tensors.
- All GPU ranks use the same destination and `.tmp` pathname.
- Captures lack sufficient request, token-position, chunk and rank identity.
- The new MoE capture is also first-forward-bound and can repeat the dummy-input
  failure.

**Required repair before another expensive diagnostic run:**

1. Arm the capture for an expected input-token sequence and positions, not a
   forward-call ordinal.
2. Reject dummy/warmup or unrelated forwards without consuming the capture.
3. Write rank-specific files with global head/expert-coordinate metadata.
4. Require all declared tensors; missing boundaries must fail the capture.
5. Save immutable tensor snapshots, original dtype/shape/stride, phase, prefix
   lengths, chunk coordinates, request identity and source/build identity.
6. Verify the embedding anchor against expected checkpoint rows before trusting
   downstream differences.

CPU TP1 and GPU TP4 need not be forced equal for every comparison. Replicated
boundaries can be compared directly; sharded ones require explicit reconstruction
in global coordinates.

## 4. Routing is not exonerated, but compare more than expert sets

Executed matched-input screening of installed CPU selector functions:

- `fused_topk_cpu`, using signed synthetic logits, correction bias and
  `sqrtsoftplus`: exact selected sets versus the reference; maximum normalized
  weight difference approximately `2.98e-8`.
- `HashTopK._forward_torch`: exact IDs and normalized weights versus the
  reference on the same logits and token-to-expert table.

These are small synthetic checks, not validation of real runtime inputs,
weights or tables.

The installed DSV4 MoE constructor sets `use_grouped_topk=False`. The brief's
grouped-top-k description is not the effective DSV4 configuration inspected.

The decisive routing check must capture:

- input token IDs and positions;
- router inputs/logits;
- global expert IDs **paired with their weights**;
- correction bias and effective configuration;
- routed output before/after scaling;
- shared output and final sum.

Different expert sets on already-divergent hidden states do not prove a selector
bug. Feed the **same logits** to both selectors. Exact hash IDs can be checked
directly against `tid2eid[input_ids]`; hash weights still depend on router logits.

The current [MoE comparator](../compare_moe_topk.py) intersects layer sets,
truncates to the smaller token count, removes negative IDs, ignores weights, and
can print "IDENTICAL" at a 0.99 threshold. Repair these acceptance weaknesses
before using it to clear routing. Require complete expected layers/tokens,
coordinate alignment, legal IDs and valid ID/weight pairs. Account explicitly
for shared slots or physical-to-logical remapping.

## 5. Recommended next run: high-information capture, then offline replay

### Inputs

Use a declared tiny nonzero-token sequence plus one frozen GSM8K prompt. The tiny
input helps localize early divergence and eliminates dummy-input confusion;
the realistic prompt exercises compressor, cache and normal activation behavior.

### Capture together

| Instrument | Purpose |
|---|---|
| Per-block inputs/outputs and attention/FFN boundaries | Find the first comparable divergent boundary |
| Routed/shared gate and up ranges, clipping counts | Determine whether missing clipping is active on real data |
| Expert-ID/weight pairs and router inputs | Distinguish selection from upstream divergence |
| Routed/shared outputs, scaling and sum | Diagnose FFN integration beyond kernel-only checks |
| Actual Q, gathered K/V, masks, sink and attention output where needed | Separate attention math from wrong operands/cache membership |

Keep captures read-only. A clipping-enabled variant changes downstream
activations and must be separately labeled; it cannot be mixed into baseline
evidence without qualification.

### Same-input replay disposition matrix

| Observation | Next action |
|---|---|
| Inputs differ before the operation | Investigate the preceding boundary |
| Inputs match and outputs differ | Inspect this operation's dispatch, arithmetic and layout |
| CPU disagrees with a GPU-validated reference on identical inputs | Localized implementation defect |
| Operation outputs match but the next boundary differs | Inspect residual handling, mutation, scaling, layout and reductions |
| Clipping changes real FFN outputs materially | Carry a clipping-correct variant into paired model validation |
| Clipping rarely activates or barely changes FFN output | Retain the contract fix, but continue localization rather than crediting it with the accuracy gap |

For attention, validate any independent reference on captured **GPU operands and
output** before treating it as an oracle for CPU operands. CPU-internal
self-consistency alone repeats the previous blind spot.

## 6. Additional methodological cautions

- The six frozen prompts share **1,064 tokens**. Repeated first divergence at
  position 1 is mostly the same prefix observation, not six independent
  localizations. Report shared-prefix and question/answer-tail results separately.
- A strict identity-checked replay of the saved CPU-eager and GPU-0.5.20 logits
  gives 5,598/6,788 top-1 matches, approximately 82.47%. This confirms the saved
  final-logit disagreement, not its internal origin.
- GPU/GPU 97.8% agreement is an observed cross-build comparison, not a universal
  numerical noise floor or proof of equivalence.
- Small changes in top-1 agreement cannot clear an operation that is known to
  implement the wrong formula.
- Once capture works, test request invariance: alone versus co-batched,
  monolithic versus chunked prefill, and fresh versus reused cache. Differences
  beyond calibrated arithmetic effects expose state/indexing faults.

## Action order and acceptance

1. **Restore DeepSeek clipping in routed and shared experts.** Add controls
   below/at/above 10, large negative gate, negative up, and checks that reject
   accidental `up + 1` or interleaved-layout semantics. Validate both prefill and
   decode dispatch branches.
2. **Repair request-bound, rank-safe capture.** The embedding anchor must pass;
   dummy inputs, missing tensors and incomplete rank coverage must not produce
   an accepted comparison.
3. **Measure clipping activity and same-input FFN deltas on real data.** Do not
   call the full bug fixed based only on synthetic probes.
4. **Follow the first valid divergent boundary** to routing, attention, MHC or
   another integration surface.
5. **Validate on matched question IDs against the GPU reference**, retaining
   paired outcomes and immutable run identity. Separate local operator
   correctness from model accuracy restoration.

## Scope and provenance

The review inspected the current worktree, including the existing uncommitted
capture changes, and installed SGLang CPU sources under
`/scratch/bkaul/venvs/sglang-cpu/`. Native probes used that environment's
`sgl_kernel` operators with `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`.

Local scratch paths in this report are evidence pointers for the shared
environment, not publicly hosted artifacts. No new inference evidence or
measured accuracy restoration is claimed.
