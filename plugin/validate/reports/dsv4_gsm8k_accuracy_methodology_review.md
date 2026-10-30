# DSv4 Flash GSM8K: accuracy methodology review and action plan

Reviewed: 2026-10-21  
Primary evidence: job 390591, compared with job 390511  
Scope: saved generations, scoring, sampling, sharding, checkpointing, launcher,
published-score comparability, and available numerical-parity evidence.

## Executive decision

**245/320 = 76.56% is reproducible, but it is not an apples-to-apples comparison
with the published 90.8%. CPU numerical equivalence is also not established.**

The strongest observed harness concern is the 256-token generation cap, alongside
a checkpoint/prompt comparability problem. Parsing causes demonstrated false
negatives, but the extraction alternatives tested do not recover the large gap.
Ordinary small-sample variance is not a persuasive explanation for the whole gap.

Do not characterize this result as either a published-score reproduction or a
CPU correctness certificate. Retain it as a historical, custom-protocol accuracy
measurement.

## Evidence and analysis

### 1. Published comparison uses a different checkpoint variant

The [official model card at revision
60d8d70770c6776ff598c94bb586a859a38244f1](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/README.md)
reports **90.8% GSM8K, eight-shot exact match, for DeepSeek-V4-Flash-Base**.
That checkpoint uses FP8 mixed precision. The post-trained
DeepSeek-V4-Flash uses FP4 + FP8 mixed precision; the same table does not give it
a corresponding GSM8K score.

The evaluated records name `/scratch/bkaul/models/DeepSeek-V4-Flash`, whose
[local configuration](/scratch/bkaul/models/DeepSeek-V4-Flash/config.json)
declares FP4 experts. Local config and tokenizer download metadata identify the
model-card revision above. This is not a hash-based verification of every weight
file, and the artifacts do not identify a separate "4.1" revision.

The harness submits raw `Question: ... Answer:` continuations. The post-trained
model documentation specifies a dedicated message encoder and reasoning modes;
the harness does not use that encoder. A prompt-mode effect is plausible but has
not been measured by this review.

### 2. The generation budget is a substantial concern

Saved outputs from job 390591 were re-tokenized with the local tokenizer:

| Saved-output length | Questions | Correct | Incorrect |
|---|---:|---:|---:|
| Exactly 256 tokens | 124 | 71 | 53 |
| Below 256 tokens | 196 | 174 | 22 |
| Total | 320 | 245 | 75 |

Thus 38.75% of outputs sit exactly at the configured cap, including 53 of the
75 errors. Of those 53 errors, 26 have no recognized answer marker.

This is evidence of likely length censoring, not proof that longer generation
would recover those answers. Some long outputs already contain wrong answers
or repetitive reasoning. The records do not retain finish reasons or generated
token IDs, so re-tokenized text length is a diagnostic rather than authoritative
termination metadata.

Reconstructed prompts contain 1,089-1,199 tokens. Prompt plus the 256-token
budget stays below 1,500, so the configured 4,096-token context limit is not the
apparent constraint.

### 3. Parsing has false negatives, but tested alternatives do not close the gap

Offline replay on the same 320 saved outputs:

| Extraction policy | Correct | Accuracy |
|---|---:|---:|
| Current chunked-runner extractor | 245 | 76.56% |
| Current extractor also accepting `answer is:` | 246 | 76.88% |
| Last recognized marker, otherwise current fallback | 236 | 73.75% |
| Last integer anywhere | 191 | 59.69% |
| Upstream lm-eval flexible-regex/normalization replay | 189 | 59.06% |

The last row is an extraction replay, **not a full lm-eval run**. It used the
upstream regex selection and exact-match ignore patterns against original gold
answer text. The current upstream sources inspected were
[gsm8k.yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml)
and [RegexFilter](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/filters/extraction.py).
Those links are moving references, not a pinned publisher protocol.

Concrete examples, using global post-demonstration question IDs:

- **q45:** the response concludes `The answer is: $40.` The marker regex rejects
  the colon after "is"; the two-paragraph fallback returns 4. See
  [shard 1 generations](/scratch/bkaul/dsv4_d2acc_390591_gens1.json).
- **q311:** a `Final Answer` section states 15 electric poles. No recognized
  marker is found, and the fallback returns 3. See
  [shard 7 generations](/scratch/bkaul/dsv4_d2acc_390591_gens7.json).
- Adversarial helper probes also show that the integer fallback reads `12.5`
  as 5 and first-marker selection ignores a later explicit correction.

Many correct outputs ramble after an answer, explaining why last-number
extraction performs worse. The one-answer colon recovery is not an exhaustive
estimate of all parsing losses. Do not optimize extraction on test answers and
then report the resulting score as an untouched evaluation.

The reported one invalid answer also understates incomplete-answer frequency:
an intermediate number selected by the fallback is treated as valid.

### 4. Sampling and aggregation

All 320 saved question IDs are unique and contiguous, 0-319. Their gold labels
match dataset rows 8-327, zero-based. Rows 0-7 are used as demonstrations.
Every shard's saved generations reproduce its recorded correct count, and
running the real combiner reproduces the committed combined JSON exactly.

This is a contiguous test-set prefix, not a random sample. Under an illustrative
binomial model, 245/320 has a 95% Wilson interval of approximately
**71.6%-80.9%**. Reaching at least 90.8% on these questions requires 46 additional
correct answers. The 26/40-35/40 shard range does not make the pooled estimate
equally uncertain.

Prefix-selection bias remains unmeasured; the binomial interval does not account
for it. Test-split demonstrations do not overlap scored questions here, but
they differ from train-split few-shot selection and leave at most 1,311 test
questions available under the current construction.

### 5. Numerical regression is unresolved

[Job 390511](../results/dsv4_gsm8k_accuracy_390511.json) scores 255/320 (79.69%);
[job 390591](../results/dsv4_d2acc_390591_COMBINED.json) scores 245/320 (76.56%).
Pairing saved generations by question ID gives:

| Transition | Questions |
|---|---:|
| Baseline correct to D2 incorrect | 24 |
| Baseline incorrect to D2 correct | 14 |
| Net change | -10 answers / -3.125 percentage points |

The exact paired McNemar p-value is approximately 0.143. This is neither
convincing evidence of a directional regression nor proof of equivalence.
Historical run identity is insufficient to attribute the difference exclusively
to the C kernels; a launcher that does not explicitly disable inherited toggles
also does not establish a controlled kernels-off baseline.

[Shard 0's log](/scratch/bkaul/dsv4_d2acc_390591_0.log) records three indexer parity
failures, with cosine values 0.999381, 0.998085, and 0.995515 against a 0.9995
threshold. Each triggers eager fallback.
[The provenance record](/scratch/bkaul/dsv4_d2acc_prov_390591_sh0.json)
records three eager indexer calls.

This was not an unconditional all-C parity pass. However, the logged failures
have one score column, so they do not themselves demonstrate changed top-k
selection or explain the accuracy loss. The check compares against CPU eager
code, not an independent GPU reference.

## Acceptance-gate defects

These defects were observed in the harness; the saved-data reconciliation does
not indicate that they corrupted this particular 245/320 result.

| Surface | Finding | Consequence / required change |
|---|---|---|
| [Single-shot runner](../task_gsm8k.py) vs [chunked runner](../task_gsm8k_chunked.py) | Single-shot stops on blank lines and uses last-integer extraction; chunked does not | Unify the protocol or explicitly version and label different protocols |
| [Combiner](../combine_gsm8k_shards.py) | No required unique-shard coverage or compatible-run validation | Reject duplicates, missing required shards, incompatible contracts, invalid counts, and empty inputs |
| [Resume](../task_gsm8k_chunked.py) | Saved records are not bound to current questions or generation/model configuration | Validate identity and exact completed-ID coverage before resuming |
| [Generation accounting](../task_gsm8k_chunked.py) | `zip` can truncate outputs while `done` increases by requested labels | Require exactly one response per requested question before checkpointing |
| [Wave launcher](../run_dsv4_d2_emr_acc_wave.sbatch) | Prints Python status without propagating failure | Preserve nonzero child exit status through the pipeline and scheduler wrapper |
| [Wave acceptance threshold](../run_dsv4_d2_emr_acc_wave.sbatch) | `--acc-min 0.30` | Label this as report/smoke status, not reference accuracy or numerical parity |
| [Saved generation schema](../task_gsm8k_chunked.py) | Only question ID, gold, and text are retained | Persist immutable identity, token IDs/counts, finish reason, and scorer/protocol version |

Executed failure controls:

- Real combiner accepted two identical shard-0 records as 58/80, exit 0.
- Real combiner accepted an empty glob, exit 0.
- Reproducing the launcher's pipeline/echo structure printed `exit=1` for a
  failing producer while the wrapper exited 0.

The current upstream lm-eval GSM8K configuration uses train-split demonstrations,
five shots by default, raw GSM8K answer formatting, and different stop/extraction
rules. Neither that default nor this custom harness establishes the publisher's
exact eight-shot evaluation protocol.

## Prioritized action plan

### P0: Make future evidence interpretable

- [ ] Separate historical accuracy reporting, numerical equivalence, and
  published-score reproduction into distinct statuses.
- [ ] Pin model/checkpoint, tokenizer, dataset, prompt IDs and token IDs, scorer,
  source/build, runtime settings, and all kernel toggles in each artifact.
- [ ] Preserve finish reasons and generated token IDs/counts.
- [ ] Make correctness-check failures fatal in validation runs, or explicitly
  mark fallback runs as ineligible for an all-C parity claim.
- [ ] Fix wrapper failure propagation and aggregation/resume validation.

**Acceptance:** positive complete inputs pass; duplicate/missing/incompatible
inputs fail without publishing an accepted result; known child failures produce
nonzero scheduler-job status; a saved result identifies exactly what was run.

### P1: Separate implementation fidelity from harness quality

Run a paired comparison of **trusted GPU, CPU eager, and CPU C** using the same
Flash checkpoint, exact input token IDs, question IDs, generation budget, and
scorer. Explicitly set rather than inherit all relevant toggles.

Include both successes and failures, particularly short wrong answers, cap-length
outputs, parser misses, and baseline-to-D2 disagreements. Preserve paired
outcomes; inspect teacher-forced logits or first-divergence evidence where
generation disagreement needs localization. Greedy text need not be identical
across numerically different backends.

**Acceptance:** a predefined equivalence margin and paired confidence analysis,
with no unresolved validation failures hidden by fallback. A non-significant
difference alone is not equivalence.

### P1: Measure prompt and generation-budget effects separately

On a frozen diagnostic set, compare current raw prompting against the official
message encoding with an explicit reasoning mode. Compare the current budget
with larger budgets, changing one factor at a time or using a declared factorial
design. Save full outputs and termination metadata.

**Acceptance:** report paired accuracy changes, length-termination frequency,
parse failures, and residual errors. Do not count mere occurrence of the gold
number anywhere in a response as a correct final answer.

### P2: Freeze and run the final evaluation

Use fixed train-split demonstrations and all 1,319 GSM8K test questions. Freeze
prompting, generation, and extraction before this run. Keep diagnostic tuning
results separate from the final evaluation.

**Acceptance:** complete unique test coverage, compatible immutable run identity,
reproducible per-question scoring, and separately reported reference agreement
and task accuracy.

### Separate track: Reproduce the published 90.8%

Use **Flash-Base**, not the post-trained Flash checkpoint, and establish the
publisher's actual prompt, examples, decoding budget, scorer, and aggregation
protocol. Do not assume that a current lm-eval default is the published setup.

**Acceptance:** checkpoint and evaluation protocol comparability established
before interpreting any difference from 90.8%.

## Review provenance and limitations

- Review began at repository revision `7b2432f9d3f65af16c12db802020d82a6c7bfd21`.
  The reviewed harness and Python dispatch files remained unchanged through
  `f654421`, although other work was committed concurrently.
- Analysis used all eight saved generation files for each of jobs 390591 and
  390511, local dataset/tokenizer files, committed result summaries, and logs.
- Primary raw evidence:
  `/scratch/bkaul/dsv4_d2acc_390591_gens{0..7}.json` and
  `/scratch/bkaul/dsv4_gsmwave_390511_gens{0..7}.json`.
  These scratch artifacts are local references, not repository-hosted evidence;
  archive them with a future action/validation record for durable reproduction.
- No model inference, new GPU reference run, or kernel rebuild was performed.
- Reported token lengths are re-tokenized saved text, not recorded generation
  token counts. Historical source/checkpoint/build identity is incomplete.
- No harness or kernel code was changed by this review. This report records
  findings and proposed acceptance criteria, not completed remediation.
