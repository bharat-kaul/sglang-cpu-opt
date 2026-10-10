# Assessment of the Kernel Round-3 Response

Reviewer: GitHub Copilot. Reviewed response `ebada06..b45226fe9794805a7cdd976737ad2fe2a54aed2e` against [round 3](phase1_kernel_implementation_review_round3.md), published as `10bd990`. The worktree was clean at review start and before report creation. The response changes F4, the CPU replay script, the author checkpoint, and the queue; no native kernel source changed.

**Verdict: accept the demonstrated fixes, but do not close round 3 or promote Phase 1.** The response fixes several concrete reference-validation failures, adds sparse M1 cases, corrects all headline timing aggregates, restricts replay deserialization, and improves budget sequencing. It also introduces a valid-mask rejection and leaves important composed-path and provenance checks incomplete. Overall status remains PARTIAL.

## Findings

### A1 - High: the new selection finiteness check rejects valid masks

The new [selection check](../f4_acceptance.py#L351) rejects every nonfinite logit, including legal negative-infinity mask entries. This is a regression from rejecting invalid NaN evidence to rejecting a supported caller domain.

Reproduction: logits `[[3, 2, -inf, -inf]]`, k=2. The native top-k returns `[[0, 1]]`; the existing membership checker reports zero non-tie mismatches. `run_case()` nevertheless returns `(True, 'non-finite reference logits')`.

The pinned [published Indexer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L425) adds negative infinity to causal-mask positions before top-k, then converts invalid selected positions to sentinels. This is not a hypothetical malformed-input case. Distinguish invalid NaN/+inf from explicitly allowed masked -inf; retain the NaN negative control and add a valid masked positive control. The all-masked/sentinel policy must be explicit rather than inferred from a blanket finiteness test.

### A2 - High: composed screening still does not use the published comparison contract

Replacing the global reference with the blockwise reference is useful, but two mismatches remain in the [composed branch](../f4_acceptance.py#L298).

- The candidate still receives original FP32 Q/KV from the builder; only the reference operands are BF16-rounded. With seed 0, a tap at the actual candidate call measured maximum rounding gaps of 0.015459060668945312 for Q and 0.015562057495117188 for gathered KV. The candidate output is now correctly rounded for comparison, but the operands are not matched. The result mixes input quantization with candidate arithmetic error.
- The reference uses `torch.topk(..., sorted=False)` before constructing 64-entry blocks. The published indexer calls `.topk(...)` with the default sorted output. Membership equivalence was sufficient for a mathematical global softmax; it does not establish blockwise BF16 equivalence. On the existing seed-0 composed builder, sorted and unsorted reference selections have identical membership but different order. With matched BF16 operands and the same blockwise reference, their outputs differ in 40,946 elements, maximum absolute error **0.001953125**.

Use the published ordering for the authoritative reference and deliberately preserve candidate ordering to measure its consequences. Match input boundaries for the direct comparison; an original-FP32-input comparison may remain as a separately labelled diagnostic. This does not require bit-exact candidate ordering or GPU equality absent a ratified policy. It requires measuring the approximation against the intended consumer, with ordering, operand rounding, and arithmetic effects distinguishable.

The seed-0 composed case currently reports `screen_pass=False` but no hard failure. That is intentional under the unratified screening policy, not itself a gate bypass. The issue is the comparison contract, not that a continuous screen can return PARTIAL.

### A3 - Medium: repeat dtype validation still bypasses the composed branch

The new ordinary-path `_same_contract` correctly checks dtype and shape before equality. The earlier [composed return path](../f4_acceptance.py#L309) never reaches it and still uses only `torch.equal(cand, cand2)`.

A wrapper returning the actual native FP32 output on the first invocation and the equal-valued FP64 output on the second passed the full `run()` with all 39 cases and calibration/validation seeds 0/100: return 0, STATUS PARTIAL, no hard failure. This reproduces the composed member of the original R3-F1 finding. Validate both composed outputs' metadata independently, as for ordinary cases.

Reference dtype validation also remains incomplete: replacing the combine reference with its equal-valued FP64 result passes `run_case()` with `screen_pass=True`, max error 0. Common contracts still omit device/layout. The new tuple-arity and scalar-reference fixes are real but are not a complete reference/output contract validator.

### A4 - High: replay's validated flag is not expected-source enforcement

The [replay loader](../test_sparse_cpu_vs_gpu.py#L37) now uses `weights_only=True` with a `TorchVersion` allowlist and rejects missing top-level provenance/records. Those are improvements. It does not compare `kernel_py_sha256` to the expected published hash: it only tests the truthiness of `kernel_py_sha256_validated`.

Through the actual `main()` entry point, in-memory archive substitutions produced:

| Archive condition | Observed result |
|---|---|
| Current job 384532 archive | Accepted; 12 matched-output rows |
| Source hash replaced by 64 zeroes; validated=True | Accepted; 12 matched-output rows |
| Provenance contains only validated=True, with no hash or job | Accepted; 12 matched-output rows |
| Current provenance but empty records | Accepted; zero comparisons, normal completion |
| Current source hash but validated=False | Rejected with the expected UNVALIDATED reason |

The first safe load of the real archive separately checked job 384532 and expected SHA-256 `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`. Substitutions did not modify the archive. Every replay load invocation requested `weights_only=True`; this is not a deserialization regression.

Require the expected source identity and a nonempty, explicitly scoped record inventory; validate required metadata and record coordinates before comparison. A caller-supplied expected job/run identity is preferable to permanently hardcoding one job. A producer's validation flag alone cannot bind evidence to the consumer's expected source, and an empty archive cannot qualify any case.

### A5 - Medium: additional cases are still removable without an inventory failure

All four sparse paths now have direct M1 and M8 cases, closing the specific absence of the best-of M1 case. However, [inventory enforcement](../f4_acceptance.py#L460) still checks op/kind presence plus indexer M labels, not the required path/shape/distribution coordinates.

Full `run()` probes with seeds 0/100 returned 0/PARTIAL after each independent removal:

| Removed coverage | Remaining cases | Result |
|---|---:|---|
| All four sparse M1 cases | 35 | No hard failure |
| All BF16-KV cases | 37 | No hard failure |
| Masked compressor case | 38 | No hard failure |

The required M16/32/64 sparse coverage and the other missing coordinates documented in round 3 are not added by this response. Keep declaration completeness and inventory enforcement distinct. New cases improve the former but do not fix the latter.

## Verified Closures and Remaining Scope

| Round-3 finding | Assessment of response |
|---|---|
| R3-F1 reference/repeat validation | Wrong reference tuple arity, scalar-reference broadcasting, NaN selection evidence, and ordinary repeat dtype flips now have working rejection controls. Composed repeat metadata and reference dtype remain open; valid masked selection regressed (A1/A3). |
| R3-F2 coverage/composition | Sparse M1 added for every path; blockwise reference substituted. Full inventory, matched operands, and published reference ordering remain open (A2/A5). |
| R3-F3 sparse serving qualification | No native change or new expanded GPU/caller qualification. Budgets and serving conformance remain unratified. |
| R3-F4 performance aggregation | **Headline table correction verified:** all 30 ratios match medians of the three process-level paired-speedup summaries at displayed precision. M8 is correctly not called a tie. Comparator drift, arithmetic/storage contracts, missing timing coordinates, and the unresolved floor remain unchanged. |
| R3-F5 replay/timing identity | Restricted loading and matched BF16 output comparison are verified. Expected source/inventory enforcement remains incomplete (A4); timing identity is unchanged. |
| R3-F6 budgets/reconciliation | Correct job citation 384531; calibration-only proposal and untouched validation split; dummy structural versus real-weight sensitivity distinction; phase-dependency cycle explicitly surfaced. These are accepted procedural corrections, not budget ratification or authorization to cross the phase gate. |

The corrected [headline wording](phase1_kernel_review.md#L53) still calls the run same-contract and identity-bound without resolving the specific round-3 caveats. Precisely, speedup is the median across processes of each process's median of five paired ratios; each individual timing is a 30-call mean. It is not a ratio of median latencies. No new timing was needed to verify the corrected numbers.

Some claim drift remains outside the changed display label: [F4's conformance metadata](../f4_acceptance.py#L88) still calls observed residuals an op-order floor, and its budget serialization retains a globally-normalized-reference description. Observed discrepancies do not prove a minimum attainable error. The updated [queue methodology](../results/kernel_opt_queue.json#L7) also remains a proposal requiring specified perturbation structure, downstream/task criteria, and human approval. This review grants no waiver, budget, or phase promotion.

## Verification and Limits

- Unmodified full F4 run: **39 cases x 5 seeds = 195 evaluations**, return 0/PARTIAL, no hard failures. A command using unsupported `--self-test` selected this normal-run path; the selftest function was then invoked directly. The supported flag is `--selftest`.
- Existing expanded selftest: **17 checks passed**, return 0. The response's count of 18 was not reproduced. This includes positive and infrastructure-reason controls, not 17 independent native correctness certificates.
- Four full-run fault/inventory probes with real native candidates and seeds 0/100, plus the targeted masked-selection, composed input-boundary, reference-dtype, and sorted-versus-unsorted blockwise probes described above.
- Actual replay entry point exercised with the safe-loaded current archive and four in-memory provenance/inventory variants. Current best-of BF16-output maximum error vs saved GPU is 0.001953125 at N1/8/64. The script's FP32 cosine calculation printed 1.000004 at N1, an additional reason not to interpret its cosine column as exactness.
- All 30 corrected headline ratios independently checked against `/scratch/bkaul/replicated_all_384531.log` at displayed precision.

Runtime checks used `/scratch/bkaul/venvs/sglang-cpu/bin/python -B`, OMP_NUM_THREADS=1, MAX_JOBS=2, and existing native extension caches/compiler settings. No calibration output, production source, saved archive, policy, author record, or performance record was changed. No cluster/GPU jobs, model loads, new benchmarks, or native optimization experiments were run. Prior all-M native correctness evidence and performance opportunity assessments remain scoped as in round 3; this response review does not repeat or broaden them.

Next acceptance action: repair A1, make composed and replay contract checks match their stated claims, and enforce the required case inventory. Then rerun the narrow positive/negative controls. Continue to hold promotion pending the already documented budget, caller/full-model, and performance gates.