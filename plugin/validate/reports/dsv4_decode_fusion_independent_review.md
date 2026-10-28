# Independent review: DSv4 decode fusion and EMR socket bandwidth efficiency

## Goal and decision

The goal is to maximize correct, sustained DSv4 decode performance on one EMR
socket and move achieved bandwidth efficiency toward a defensible,
workload-specific roofline. Per-layer fusion is a candidate mechanism, not the
objective. The right first change is whichever demonstrably removes the largest
cost at acceptable implementation and accuracy risk.

**Decision:** approve a short measurement-and-prototype phase, not yet the
multi-week monolithic-layer implementation as specified. The reported 29% is
an unattributed layer-local residual, not a measured recoverable fusion benefit.
Do not interpret this as accepting current decode performance or abandoning
fusion. It is a recommendation to find the highest-leverage mechanism before
committing to a particular architecture.

Reviewed on 2026-10-28 at pushed commit
`6f9ffc2986f48f422dcfc8f694c31181d97f4cba`, branch
`dsv4-fused-layer-decode`:

- [Review request](dsv4_decode_fusion_review_request.md)
- [Build plan](../../../docs/dev/monolithic-layer-kernel-plan.md)

Evidence below consists of source inspection, arithmetic reconstruction and
small executable probes. No new full-model performance measurement was run.

## 1. What the 29% establishes

The [layer-scaling CSV](../results/dsv4_rung1_layerscaling_1socket.csv)
reproduces:

```text
decode_step_ms = 47.92 + 75.83 * layers
residual / layer_total = 29.0–29.3%
```

This supports layer-local scaling. It does not establish that the residual is
dispatch, park/wake, or recoverable through a C++ driver.

The [layer timer](../../intel_cpu_models/_dsv4_cpu_infra.py) wraps
`Layer.forward`; the outer Python layer-iteration overhead is outside that
timer. Removing that loop cannot be credited against the residual inside it.

The [attribution data](../results/dsv4_decode_attribution_1socket.json)
already includes `hc_pre + hc_post` as timed work: 12.4% of layer time.
MHC fusion targets that timed slice as well as any genuine seams; it cannot
simply be counted again as part of the residual.

A lower-bound arithmetic estimate for elementwise operations does not exclude
copies, allocations, metadata, cache operations, uninstrumented computation or
scheduling stalls. Replace the build plan's "CONFIRMED recoverable" statements
with a hypothesis.

## 2. C++ control flow is not a shared parallel team

Keeping existing native calls preserves their internal parallel regions.
Moving calls into C++ may remove Python overhead, but calling the same ATen
operations normally retains ATen dispatch, allocation and threading costs.
Actual elementwise fusion can remove intermediate tensors and parallel regions.
Those are distinct mechanisms and should be measured separately.

Synchronous timers around native kernels generally include their fork/join and
wake-up delays. A wake-up at entry to kernel B is not automatically charged to
the untimed gap preceding B. The claimed residual decomposition needs a timeline.

The installed ATen `Parallel-inl.h` checks `!at::in_parallel_region()` before
parallelizing. A single outer `omp parallel` is not a transparent reuse mechanism:

- invoking the whole native operation from one worker may serialize its work;
- invoking it from every worker may duplicate work or race on outputs;
- separately threaded libraries may introduce nested teams or oversubscription.

Persistent cooperative execution needs explicit work partitioning, barriers,
scratch ownership and compatible kernel entry points. Existing inner math can
potentially be reused; rewriting every kernel is not necessarily required.
Nevertheless, build-plan increment 4 is an architecture experiment, not a
medium-risk wrapper guaranteed to recover 29%.

## 3. The incremental-fusion exclusion is unsupported

[validate_mhc_wire_increment1.py](../validate_mhc_wire_increment1.py)
constructs `sgl.Engine` before setting the sinkhorn/combine and parity-check
environment variables. These changes occur in the client; the model runs in
an already-started worker process.

Consequently, the reported +0.1% result does not establish that the worker
compared different implementations. The parity PASS likewise does not prove
that worker-side checking was enabled. This is the same process-boundary
control error previously found in the mask A/B.

The driver also estimates decode time by subtracting a separate prefill and
dividing by requested output length, rather than measuring actual subsequent
decode steps. It does not retain individual timing samples in its result.

Withdraw "incremental changes cannot help." Even a valid neutral result for
sinkhorn/combine would exclude only that intervention at its measured workload.

## 4. Correctness infrastructure is a blocking prerequisite

The real entry point of [diff_fullcap.py](../diff_fullcap.py) was tested:

| Adversarial input | Observed result |
|---|---|
| No shared capture keys | `PARITY OK`, exit 0 |
| NaN candidate tensor | `PARITY OK`, exit 0 |
| Opposite-valued tensors | Prints divergence, still exits 0 |

The tool also compares only intersecting keys and strips rank suffixes without
collision validation. It is a diagnostic printer, not a fail-closed gate.

The current [FULLCAP implementation](../../intel_cpu_models/_dsv4_cpu_infra.py)
recognizes prefill through `2 <= T <= 16` and decode through `T == 1`. It does not
cover the B=32 decode target through those paths. Its `dc_counter // 64` step
label is not appropriate for a 43-layer model. Capture identity is not bound to
exact requests and explicit forward steps.

Required gate:

- Nonempty, complete expected op/layer/step coverage.
- Exact request/token/rank coordinates and no duplicate-key overwrite.
- Shape, dtype and finiteness validation on both sides.
- Nonzero exit on every failed check.
- Absolute/relative error requirements in addition to cosine.
- Cache/state checks across multiple decode steps and compression boundaries.
- For a pure control-flow port, exact tensor/state equality as the initial target.

Preserve existing FP32/BF16 rounding boundaries during fusion. A precision
change is a separate intervention, not a prerequisite for eliminating dispatch.
Finish with matched real-weight task evaluation after local and full-model
parity gates; do not use a task score to excuse invalid intermediate evidence.

## 5. Roadmap toward socket bandwidth efficiency

### A. Establish trustworthy operating points and attainable bounds

Carry B=1 latency and the intended throughput batches separately, with stated
context and generation lengths. Record actual steady-state batch and output
counts. Use one-socket physical-core affinity and verify worker affinity,
memory placement, thread counts and the loaded OpenMP runtime.

Measure bandwidth with the same socket/core/memory configuration used by the
candidate. A full-socket bandwidth figure is an aspirational comparison for an
8-thread configuration, not proof that those eight threads can attain it.

Maintain both:

- **Useful-byte rate:** independently accounted active packed weight and scale
  bytes per step divided by measured step time.
- **Measured memory traffic:** socket memory-controller reads/writes where
  available, with remote traffic distinguished.

Useful-byte rate and physical traffic are not interchangeable. Extra traffic
can make memory busy without delivering useful model throughput.

Include compute at the actually dispatched ISA, native low-bit conversion,
KV traffic and repeated reads in the performance model. Use at least
`max(bytes / bandwidth, work / compute_rate)` for an applicable kernel bound;
do not classify the gap to an ideal weight-only bound entirely as overhead.

### B. Fix experimental controls and reconcile the residual

Use worker-side controls with acknowledgment and branch counters, or a direct
worker-local replay. The repository's
[control-file mechanism](../../intel_cpu_models/_dsa_ckernels.py) is prior art,
but prove the intended per-op branch changed. Reject fallback-contaminated runs.

Collect a nested, identity-bound timeline of one real decode layer:
native calls, norms/RoPE/MHC, copies, allocation, cache/metadata operations and
thread activity. Retain call counts and nesting rather than subtracting
arbitrarily labelled inclusive totals. Measure instrumentation perturbation.

Scheduling traces can distinguish running, waiting and descheduling. A no-op
parallel-region benchmark is a useful calibration, not a reliable substitute
for real-kernel boundary measurement.

### C. Investigate MoE inflation alongside one concrete glue prototype

MoE is approximately 33% of the cited layer profile. Verify the actual MXFP4
binary path and measure it on captured real expert occupancies, packed weights,
thread counts and NUMA placement.

Specific questions:

- Does vector/BRGEMM dispatch use each active group's actual row count, or a
  diluted average over all experts?
- How much time is alignment/routing, allocation, conversion, compute and
  parallel-region synchronization?
- Is scratch reused, and are packed panels reused where the contract permits?
- Do more active workers improve useful bandwidth or add synchronization cost?

Do not transfer the old FP8/GNR 0.1 ms versus 5175 ms example to current
MXFP4/EMR as a measured floor. Keep packed storage; avoid whole-model widening.

In parallel with that analysis, compare a bounded sequence on identical operands:

| Variant | Mechanism isolated |
|---|---|
| Original sequence | Baseline |
| C++ driver, same operators and math | Python/control overhead |
| One genuinely fused glue segment | Intermediate traffic and standalone regions |

Benchmark real-shaped operands and state, then confirm promising variants in
the actual worker. A compile experiment on a pure glue segment is reasonable:
custom ops do not inherently cause graph breaks when registration supports
tracing; mutation, Python control and state interfaces require inspection.

### D. Escalate to persistent-team execution only with measured justification

If a representative kernel-pair prototype demonstrates that cooperative
execution removes significant boundary cost while preserving parallelism and
outputs, design persistent-team adapters. Explicitly define worker ownership,
barriers, exception handling, scratch lifetime and library threading behavior.

`at::launch` is not itself a guarantee of reduced intra-op synchronization.
Do not put unchanged internally parallel kernels under an outer team and assume
they will cooperate.

## 6. Decision thresholds and expected benefit

These are Amdahl calculations, not achieved speedups:

| Hypothetical improvement | Overall speedup |
|---|---:|
| Remove all of a 29% slice | 1.41x |
| Remove half of that slice | 1.17x |
| Double an operation occupying 33% | 1.20x |

The choice should be driven by demonstrated gain versus implementation risk,
not by a commitment to monolithic fusion. Use replicated, order-balanced
measurements with matched inputs/state, worker-side arm identity, actual
decode-step timing and reported spread. Confirm results without profiling.

## Final recommendation

Approve measurement repairs, a fail-closed parity gate, real-operand MoE/layer
replay and a small fusion prototype. Do not yet approve the claim that level-1
fusion captures most of 29%, or that incremental approaches are exhausted.

**The end goal remains substantially faster correct decode and higher useful
socket bandwidth efficiency.** Fixing measurement is not the endpoint; it is
the shortest reliable way to choose between native-kernel dispatch, data
movement, threading and cooperative fusion as the next performance investment.
