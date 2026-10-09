# Kernel Implementation and Provenance: Round 5

2026-10-09. Independent reviewer: **GitHub Copilot**; no GPT Astra 6 execution is claimed. Pinned HEAD: `19ea82a4bbb9059ea70873fb1610f6d8ed8bc533` (verified before review; initially clean worktree).

**Partial-scope verdict: FAIL.** R4-F5 is closed. The original R4-F3 counterexamples are fixed, but its malformed-table-row rejection remains incomplete. R3-P1 remains closed. This verdict does not assess F4 policy or require any of the explicitly excluded work.

## Finding

### R5-F3 Medium: Malformed Numeric Coordinates Still Hide Invalid Timing Samples

The coordinate recognizer at [parse_perf_sweep.py:50](../parse_perf_sweep.py#L50) accepts only `^-?\d+$`. The unconditional skip at [parse_perf_sweep.py:72](../parse_perf_sweep.py#L72) therefore treats `+1` and `1.0` as prose even inside a recognized timing table. Neither the invalid latency nor the repeated coordinate reaches validation. This is the remaining malformed-numeric-row class from R4-F3, not a numerical-tolerance or job-exit-policy issue.

**Grounded counterexample:** read the actual `/scratch/bkaul/dsa_perf_sweep_384414.log`; in the first real sparse REP=1 table, insert either of the following full-width rows immediately before its existing M=1 row, leaving the entire real log otherwise intact:

```text
M cos_sc cos_amx ref_ms sc_ms amx_ms sc_x amx_x
+1 1.000000 0.999991 0.235 nan 3.602 3.44x 0.07x
```

The second tested variant changes only `+1` to `1.0`. The header above identifies the existing table, not an additional injected header. The original next row is `1 1.000000 0.999991 0.235 0.068 3.602 3.44x 0.07x`.

**Full entry-path result:** call `parse_perf_sweep.main()` with argv `['parse_perf_sweep.py', '/scratch/bkaul/dsa_perf_sweep_384414.log', '384414']`. Patch `builtins.open` so that the normal raw-log read returns the mutated text and the normal destination write returns a tracked `StringIO`; all other reads use the real opener, and other writes are forbidden. Both variants return normally and attempt exactly one output write at [parse_perf_sweep.py:133](../parse_perf_sweep.py#L133). The emitted JSON equals the existing parsed record, including its claim of validated finite positive samples.

```text
REAL_LOG_MUTATION '+1 1.000000 0.999991 0.235 nan 3.602 3.44x 0.07x' ACCEPT mock_output_writes=1
REAL_LOG_MUTATION '1.0 1.000000 0.999991 0.235 nan 3.602 3.44x 0.07x' ACCEPT mock_output_writes=1
REAL_LOG_MUTATION '-1 1.000000 0.999991 0.235 nan 3.602 3.44x 0.07x' REJECT unexpected M coordinate -1; mock_output_writes=0
REAL_LOG_MUTATION '01 1.000000 0.999991 0.235 nan 3.602 3.44x 0.07x' REJECT non-finite/non-positive time nan; mock_output_writes=0
```

This is evidence loss, not a claim that the unmodified job has bad medians: an extra invalid sample disappears and ingestion succeeds. If these coordinate spellings are unsupported, reject the malformed numeric-looking row; if supported, normalize it and validate the latency and duplicate identity. Either policy should reject this input. Preserve explicit handling of genuine shell traces/prose. The required regression is rejection through `main()` with **zero destination writes**, not a particular error string.

## Verified Closures

| Check | Independent result |
|---|---|
| R4-F5 original repoints of compressor `wkv_wgate_r4`: `caller:`, `caller:NO_SUCH_CALLER`, `unmodeled:`, `modeled:MHC hc_fn (16384->24, FP32)` | All four rejected by the full generator `selftest()` through normal provenance file reads; result `False`, not just an isolated-helper failure. |
| R4-F5 nearby variants | Seven more rejected: required destinations attached to swapped r4/r128 gap IDs; wrong disposition kind with the right target; trailing whitespace; kind-case change; target-case change; plausible r128 row; plausible indexer row. Total **11/11 rejected**. |
| R4-F3 duplicate `M sc_ms sc_ms` header with first latency NaN | `FAIL ingest: ... duplicate header column(s) ['sc_ms']`; no output write. |
| R4-F3 broken REP=1 followed by valid retry | `FAIL ingest: ... unparseable 'sc_ms' value 'BROKEN'`; no output write. Earlier rejection is sufficient. |
| R4-F3 truncated row followed by retry; negative M | Rejected as missing `sc_ms` / unexpected M=-1; no output writes. |
| R4-F3 first attempt has no samples or no header, then valid REP=1 | Both rejected as duplicate run markers, independent of parse success. |
| Required `SC_MS` instead of `sc_ms`; NaN/inf/zero/negative named latency | All rejected through parser `main()`; no output writes. |
| Leading-zero duplicate M / REP | Duplicate M=`01` and REP=`01` retries rejected after integer normalization. |
| R3-P1 retained | Fresh render equals saved report; `main --verify` succeeds; mocked saved-text drift exits **2**. Six HISTORICAL correctness labels and six CURRENT timing labels remain separate. |

The binding now occurs at [dsv4_roofline_p2.py:522](../dsv4_roofline_p2.py#L522), against the independent [required-destination map](../dsv4_roofline_p2.py#L417); each full mutation test observed **two normal provenance reads**. The genuine record passes. No destination bypass was found within the requested scope. This establishes the declared ID-to-destination coverage, not captured operand alignment or complete numerical conformance.

**External cross-check, not just map agreement:** fetched the published [model source](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py) and [config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) at revision `60d8d70770c6776ff598c94bb586a859a38244f1`. `Compressor.__init__` defines `coff=1+(ratio==4)` and two projections, each `dim -> coff*head_dim`. With dim=4096, main D=512 and indexer D=128, their combined widths are respectively 2048, 1024 and 512, agreeing with the three distinct [cost rows](../dsv4_roofline_p2.py#L307). `Indexer` defines wq_b as 1024 -> 64*128 and weights_proj as 4096 -> 64, then handles top-k masking/offset in its caller. `Block.hc_pre` defines the separate 4*4096 -> (2+4)*4 = 16384 -> 24 projection. Thus the tested MHC/r128/indexer substitutions genuinely misidentify the r4 projection; they are not cosmetic label variations. Compressor postprocessing/state and indexer rotation/quantization remain declared gaps, not newly certified coverage.

**Permissiveness not counted as a finding:** a sole valid M=`01` row normalizes correctly; extra trailing tokens leave the named latency unchanged; an extra uppercase `SC_MS` column does not shadow the schema's lowercase `sc_ms`. These cases preserve the selected timing. They are distinct from silently dropping an entire malformed row whose named timing is NaN.

## Measurement and Test Evidence

All probes ran as in-memory heredocs under `PYTHONDONTWRITEBYTECODE=1 /scratch/bkaul/venvs/sglang-cpu/bin/python -B -`. No test harness file was written. The generator's temporary evidence files were also replaced with in-memory streams; normal production reads remained real except the deliberately injected provenance record. Parser `main()` was **never run with an unmocked output write**.

```text
GENERATOR_BASELINE True PASS_COUNT 47 normal_record_reads 2
PARSER_SELFTEST exit=0 PASS_COUNT 13
ALL_DESTINATION_MUTATIONS_REJECTED 11/11
REAL_PARSER_MAIN output equals stored JSON; output write mocked
REPORT_RENDER matches stored; HISTORICAL=6 CURRENT=6
REPORT_DRIFT exit=2 STALE REPORT
```

An independent line-by-line extraction, not `parse_ops()`, read 18 real bench/replica markers and selected `cpp_ms` by header (`sc_ms` for sparse). All **30 medians** match [../results/perf_sweep.json](../results/perf_sweep.json), as does the complete JSON emitted by mocked `main()` on the original log:

| Bench | M=1, 8, 16, 32, 64 medians (ms) |
|---|---|
| Indexer logits | 0.100, 0.279, 0.366, 0.540, 0.908 |
| Top-k | 0.017, 0.028, 0.032, 0.039, 0.014 |
| Compressor | 0.374, 0.490, 0.489, 0.499, 0.571 |
| Sinkhorn | 0.002, 0.014, 0.014, 0.014, 0.015 |
| Combine | 0.002, 0.012, 0.012, 0.012, 0.014 |
| Sparse scalar | 0.068, 0.460, 0.894, 1.793, 3.640 |

Input/output SHA-256 values were checked unchanged before/after parser probes:

```text
raw log:   74569cf5416d65411e97cbde25c9f406483769139d527fbb297f6d09f1edf3fc
perf JSON: ba892887a64bc2fe7ec4a80274521e45aed722d7000c8f3cb51a06a031964392
```

## Scope

Only this report is authored. No code, production records, other reports, ledger/state, commits or pushes are changed by this review. No cluster jobs, performance runs, compilation or expensive live correctness tests were performed. F4 policy/tolerance ratification, F7 full Amdahl ledger, F8 routing, donor-dispatch proof and optimization exhaustion are excluded. Existing measurement build-digest and current-target/E2E limitations remain; this report does not upgrade historical microbench evidence into certification.

<<<REVIEW-VERDICT
{"status":"FAIL","findings":["R5-F3 (Medium): malformed numeric-coordinate rows can hide NaN timing samples and still reach parser main output"],"surface":false,"note":"R4-F5 closed; all original R4-F3 counterexamples rejected, but malformed-row rejection remains incomplete. All 30 raw-log medians and R3-P1 regeneration checks pass. F4 policy, F7 ledger, F8 routing, donor proof and live correctness are excluded."}
REVIEW-VERDICT>>>