# Kernel Implementation and Provenance: Round 6

2026-10-09. Independent reviewer: **GitHub Copilot**, not GPT Astra 6.
Exact reviewed HEAD: `75db694e4e551b0b55eef596fd3bd2ae17c22bf5`; initially clean worktree.

## Findings

**No in-scope blocker. R5-F3 CLOSED; partial-scope PASS.** The original hidden-NaN counterexamples now fail through the normal parser `main()` file-read path before any destination write. This is only the closure scope of [reviewer_round6.prompt.txt](../../../tools/review_loop/runs/kernel-impl-and-provenance/reviewer_round6.prompt.txt), not approval of the separate F4 policy. Read and independently checked the claims in [kernel-impl-and-provenance_response_round5.md](kernel-impl-and-provenance_response_round5.md) against [kernel-impl-and-provenance_review_round5.md](kernel-impl-and-provenance_review_round5.md).

The recognizer in [parse_perf_sweep.py](../parse_perf_sweep.py#L50) admits signed decimal/exponent tokens. The subsequent integer conversion, coordinate-membership, latency and duplicate checks reject invalid rows before serialization. The independent grammar authority is the actual raw log plus the benchmark emitter: [bench_sparse_attend.py](../bench_sparse_attend.py#L39) iterates integer M values and prints them with `M:>4`, with latency under the named `sc_ms` header. No model-semantic claim is reopened here.

## Observed Checks

Executed an in-memory heredoc with `/scratch/bkaul/venvs/sglang-cpu/bin/python -B -` (Python 3.12.13). Imported the current parser and called its real `main()` with argv containing `/scratch/bkaul/dsa_perf_sweep_384414.log` and job `384414`; no replacement of `parse_ops()`. The baseline read the real file. Mutations intercepted that same read with `StringIO`. All destination opens were intercepted with tracked `StringIO` streams; unexpected write destinations and low-level opens were forbidden. **No unmocked parser-main output write occurred.**

For each required spelling, copied the real first sparse REP=1 row, changed only M and `sc_ms`, and inserted it immediately after its real header, retaining the original row. Tested both `NaN` and `BROKEN` with three whitespace forms: single spaces, padded double spaces, and padded tabs with CRLF.

| Coordinate | Observed rejection | Cases / destination writes |
|---|---|---|
| `+1`, `01`, `+01`, bare `1` | Non-finite time / unparseable `sc_ms` | 24/24 rejected; 0 writes |
| `1.0`, `1e0`, `.5` | Malformed M coordinate, not an integer | 18/18 rejected; 0 writes |
| `-1` | Unexpected M coordinate | 6/6 rejected; 0 writes |

- **48/48 required cases rejected.** Representative grounded row: `+1 1.000000 0.999991 0.235 NaN 3.602 3.44x 0.07x`; observed `FAIL ingest: bench_sparse_attend.py M=1 rep=1: non-finite/non-positive time nan`, zero destination opens.
- **8/8 valid-latency extra rows rejected:** the same eight coordinate spellings with original `sc_ms=0.068`, exercising duplicate or malformed/unexpected-coordinate handling; zero writes.
- **8/8 nearby cases rejected:** `+0`, `-0`, `+128`, `-01`, `1.`, `+.5`, `1E+0`, `+1e-0`, each with `BROKEN`; zero writes. The first four demonstrate that successful `int()` conversion cannot evade coordinate membership.
- **3/3 valid aliases accepted:** replacing, rather than duplicating, the original M=1 row with `+1`, `01`, or `+01` yields identical JSON; one mocked destination open each.
- **2/2 noise placements accepted:** blank lines, shell echo of the benchmark marker, `+ for REP in 1 2 3`, the real exit-status line, and `Completed 3 reps; no additional samples.` inserted inside the table or appended after the log leave JSON unchanged. Existing real shell traces also pass.
- **Current self-tests: 15 PASS, exit 0**, through `main --selftest`, zero destination writes.
- **Discriminating regression control:** loaded the actual parser from round-5 reviewed SHA `19ea82a4bbb9059ea70873fb1610f6d8ed8bc533` into memory. Both original `+1` and `1.0` NaN injections return normally with one mocked output open; current code rejects both. The test detects the old behavior, not merely agreement with the new implementation.

**Nonblocking grammar boundary:** a full-width `0_1 ... NaN ...` row is skipped and the remaining valid log emits unchanged JSON into the mock. Python `int('0_1')` accepts it, but underscore-separated literals are neither the declared decimal/exponent grammar nor output of this benchmark's integer formatting. This is not evidence of an in-scope benchmark-row bypass and does not reopen R5-F3. It does limit any broader claim that every Python integer literal or arbitrary corrupted string is recognized. Genuine prose must not be rejected merely because it contains numbers; no universal free-text grammar is certified.

## Raw Measurement Cross-Check

Independent line-by-line extraction did not call parser helpers or use its `SPEC`: selected `cpp_ms` by raw header name for five benches and `sc_ms` for sparse; checked 18 unique bench/replica blocks, 90 positive finite samples and exact replica sets `{1,2,3}`; took the middle sorted observation for each coordinate. All **30 medians** equal [../results/perf_sweep.json](../results/perf_sweep.json). Baseline mocked `main()` also emits the identical complete JSON, not only equal medians.

| Bench | M=1, 8, 16, 32, 64 median ms |
|---|---|
| Indexer logits | 0.100, 0.279, 0.366, 0.540, 0.908 |
| Top-k | 0.017, 0.028, 0.032, 0.039, 0.014 |
| Compressor | 0.374, 0.490, 0.489, 0.499, 0.571 |
| Sinkhorn | 0.002, 0.014, 0.014, 0.014, 0.015 |
| Combine | 0.002, 0.012, 0.012, 0.012, 0.014 |
| Sparse scalar | 0.068, 0.460, 0.894, 1.793, 3.640 |

All 332 tracked files plus the raw input retained their SHA-256 hashes across probes. Raw log SHA-256: `74569cf5416d65411e97cbde25c9f406483769139d527fbb297f6d09f1edf3fc`. Parsed record SHA-256: `ba892887a64bc2fe7ec4a80274521e45aed722d7000c8f3cb51a06a031964392`.

## Limits

Only this report is authored. No fixes, policy/ledger changes, other report edits, nested agents, commits, pushes, C++ rebuilds, cluster jobs or expensive runs. F4 policy/tolerance ratification, F7 Amdahl ledger, F8 entry routing and sparse donor-dispatch proof remain explicitly out of pass-scope. Earlier closures are context, not newly recertified here. This review establishes parser ingestion integrity at the tested grammar, not numerical correctness, captured-shape validity, E2E certification, build provenance or optimization exhaustion. The other agent's independent F4 report is not edited.

<<<REVIEW-VERDICT
{"status":"PASS","findings":[],"surface":false,"note":"R5-F3 closed: 48 required and 16 nearby/duplicate main-path cases reject with zero destination writes; 15 self-tests pass and all 30 raw medians match. Partial scope only; F4, F7, F8 and donor-dispatch proof remain excluded."}
REVIEW-VERDICT>>>