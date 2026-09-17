---
name: high-information-runs
description: "Experiment-design discipline for when the RUN is the bottleneck (long model load + slow real-scale decode + serial cluster queue). Loop wall-time = num_runs x run_cost; you can't cut run_cost (real weights/scale), so cut NUM_RUNS: make each expensive run a MULTI-ANGLE probe that resolves several hypotheses at once, with a disposition matrix planned BEFORE submitting. Use when iterating on a model that takes many minutes to load and runs slowly, when cluster queue waits dominate, or whenever you catch yourself submitting one run to answer one yes/no question. Front-load every cheap in-situ probe onto one load; pre-screen the cheap-to-kill hypotheses off-line first."
---

# High-Information Runs (maximize dispositions per expensive run)

When the run is the bottleneck, **loop wall-time = num_runs × run_cost**. `run_cost` is
mostly fixed (real weights load + slow decode + queue wait), so the only lever is
`num_runs`. Cut it by designing each expensive run to **disposition several hypotheses at
once**, not one. The default trap — one ~35-min run per yes/no question — makes the loop
serial and slow; a single well-instrumented run can answer five.

## The method
1. **Separate the expensive part from the cheap part.** Expensive = model load + a few
   slow steps. Cheap = anything measurable ONCE the model is resident: in-situ config
   sweeps, boundary sub-timers, ISA-dispatch checks, shape/dtype logs, in-process A/B of
   variants. Attach ALL cheap probes to ONE run — **one load, many answers**.
2. **Enumerate the open hypotheses first, then instrument ALL of them in one run.** Before
   submitting, list every open question; if the same load can answer five, don't submit a
   run that answers one.
3. **In-situ A/B and factorial sweeps.** Use the loaded model to time N configs in-process
   (as the MoE thread-sweep timed 4/8/16/60 threads in one call) and to A/B a fix vs
   baseline — resolving "which wins" WITHOUT a second run. Generalize a single-op probe to
   a per-op sweep so one run dispositions a *systemic* question across all ops.
4. **Write the DISPOSITION MATRIX before submitting.** A table: each measurable outcome →
   what it rules in/out → the next action. If a branch can't be dispositioned by the run as
   designed, ADD the instrument that would. This forces the run to be decisive, not merely
   data-gathering (which costs a second run just to interpret).
5. **Pre-screen cheaply; carry only the survivors — together.** Kill cheap-to-kill
   hypotheses on tiny / login-node / microbench first (free). The expensive run carries only
   the hypotheses that NEED real scale, and carries them ALL at once.
6. **Instrument the NEXT question now.** From the analytical ranking, anticipate the likely
   next bottleneck and add its probe, so the run that resolves the current one already holds
   the data for the next — collapsing two runs into one.
7. **Env-gated, inert-by-default probes.** One build must run in many modes without rebuilds,
   and instrumentation must never pollute a clean measurement (guarded timers/sweeps).

## Anti-patterns
- One-question-per-run serial iteration (the default) — each expensive run a single yes/no.
- Re-loading the model to flip one flag that could have been swept in-process.
- Gathering data with no disposition plan → a follow-up run just to interpret it.
- Carrying an already-cheaply-killable hypothesis into the expensive run.

## Worked example (this repo)
The decode-slowness diagnosis. Serial version: run 1 "is MoE the cost?", run 2 "is it
threads?", run 3 "which thread count?", run 4 "does it hit other ops?", run 5 "what's in the
unattributed 45%?" — five ~35-min runs. Batched version: ONE run carries
overhead-attribution tags (all ops → the share split), a per-op in-situ thread sweep
(systemic vs MoE-only), the MoE-fix A/B (autotune pick + effect), and dense/norm/lm_head
sub-timers (the 45%) — dispositioning the whole "decode" question in a single load.

## Disposition-matrix template
| Instrument in the run | Outcome | Rules in / out | Next action |
|---|---|---|---|
| overhead split (kernel/torch/framework) | which bucket dominates | picks the lever class | route to the bucket's skill |
| per-op thread sweep | which ops scale inversely | MoE-only vs systemic | point cap vs decode-wide thread-fit |
| fix A/B in-process | Δ vs baseline | confirms/denies the fix | ship or discard, no extra run |
| sub-timers on the unattributed | where the residual is | dense vs norm vs lm_head | next kernel/config target |

## Ties in
- `model-profile-hotspots`: this is HOW to run that empirical tier when runs are expensive.
- `overhead-attribution`: its boundary timers + in-situ ISO are the cheap probes to batch.
- `uarch-perf-probe`: the off-line pre-screen (login node / microbench) before the expensive run.
