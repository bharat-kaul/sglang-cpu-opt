---
name: high-information-runs
description: "Experiment-design discipline for when the RUN is the bottleneck (long model load + slow real-scale decode + serial cluster queue). Loop wall-time = num_runs x run_cost; you can't cut run_cost (real weights/scale), so cut NUM_RUNS: make each expensive run a MULTI-ANGLE probe that resolves several hypotheses at once, with a disposition matrix planned BEFORE submitting. Use when iterating on a model that takes many minutes to load and runs slowly, when cluster queue waits dominate, or whenever you catch yourself submitting one run to answer one yes/no question. Front-load every cheap in-situ probe onto one load; pre-screen the cheap-to-kill hypotheses off-line first."
---

# High-Information Runs (maximize dispositions per expensive run)

## ⛔ ASYNC-HEARTBEAT — never fire-and-forget a job then wait for a human to ask "is it done?"
A recurring velocity leak: submit an sbatch (or a long download/decode), then STOP and idle until the
operator prompts a status check. The submit→detach→human-prompt round-trip serializes the whole loop on
human attention. Kill it by making completion AUTO-WAKE the agent:
1. **Immediately after submitting any long job, launch a BLOCKING WAITER in the agent's ASYNC terminal**
   (`scripts/await_job.sh <jobid> [log] [hb_secs]`, or `sbatch --wait <script>`). The async terminal
   notifies the agent on completion → it wakes and dispositions the result with NO human prompt. The
   waiter also emits periodic HEARTBEAT lines (elapsed + last log line) for liveness and a DONE summary.
2. **Keep a WATCHLIST LEDGER** (one row per in-flight job: id, purpose, node, log, expected PASS signal,
   disposition-per-outcome); update at submit and at completion.
3. **Re-orient from the watchlist FIRST on every wake** — what's pending, did it land — before new work.
4. **Parallel jobs → one waiter each** (or one over the array); each completion wakes the agent. Don't
   serialize behind a single manual check.
This is the control-flow dual of the disposition matrix: the matrix says what each outcome means; the
async-heartbeat guarantees the agent is actually there to read the outcome the instant it exists.

When the run is the bottleneck, **loop wall-time = num_runs × run_cost**. `run_cost` is
mostly fixed (real weights load + slow decode + queue wait), so the only lever is
`num_runs`. Cut it by designing each expensive run to **disposition several hypotheses at
once**, not one. The default trap — one ~35-min run per yes/no question — makes the loop
serial and slow; a single well-instrumented run can answer five.

## ⛔ MANDATORY PRE-SUBMIT GATE (default behavior — do NOT wait to be told)
Before launching ANY expensive run, satisfy this checklist. If you catch yourself submitting a run to
answer one yes/no question, STOP and widen it.
1. **List EVERY open hypothesis now** (all suspects, not just the top one).
2. **Instrument ALL of them in this one run** — multi-vector capture of every candidate op boundary on
   reference + target, keyed by `(layer, op, n_tokens, tp_rank)`.
3. **Write the disposition matrix** (outcome → rules in/out → next action). If a hypothesis can't be
   dispositioned as designed, ADD the instrument or don't submit.
4. **Fold in every "check next" caveat.** If you write "if X looks clean, next check Y", instrument Y in
   the SAME run. Deferring a follow-up to a later run is THE anti-pattern this skill exists to kill.
5. **Parallelize by default:** independent hypotheses → a WAVE across idle nodes; CPU + GPU reference →
   launch CONCURRENTLY (different partitions); one decisive run → the fastest idle single node. Check
   `sinfo`/`squeue` first. Only serialize jobs that share mutable global state.
6. **Independence due-diligence BEFORE co-running/parallelizing** (a hidden dependency wastes the run):
   - **No shared mutable state:** each hypothesis's instrument/variant must not alter state the others
     read — env flags, weights, KV/caches, global counters, output-file keys, RNG, thread count.
   - **Capture vs variant:** a read-only CAPTURE (tap) can always be co-run. A VARIANT that changes the
     forward (e.g. force-bf16 A/B, force-dense) CONTAMINATES any other measurement of the "real" path in
     that run — allow at most ONE forward-altering variant per run, and only co-run captures that are
     INVARIANT to that change (e.g. a shadow comparing kernel-vs-reference on the same input).
   - **No result-dependency:** if designing hypothesis B's test needs A's outcome, they are SEQUENTIAL,
     not parallel — don't fake-parallelize; carry B once A is known (or instrument B's superset now).
   - **Distinct output keys:** rank/layer/op/token/file keys so co-captured signals don't collide.
   - If any fails → ISOLATE (separate runs/nodes) or SEQUENCE. Parallelism only pays when the branches
     are genuinely independent.

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

## Match the node to the run SHAPE (infra awareness cuts wall-time as much as instrumentation)
`num_runs × run_cost` also depends on WHERE you run. **MANDATORY PRE-LAUNCH GATE — execute every single
launch, no reflex `--partition=` default:**
1. **Classify the task:** correctness / weight-dump / capture / bisection = ISA-PORTABLE (any idle node,
   prefer the farm); perf/roofline FINAL = target HW only; GPU oracle = GPU box. A dump/capture is NEVER
   a reason to take the scarce fast single node.
2. **Check availability across ALL candidate partitions in one shot** before deciding:
   `sinfo -p emr,gnrap -h -o "%P %t %D"` + `squeue -p gnrap -h` (is the single fast node busy/queued?).
3. **Pick by (task-class × availability × wave-count)**, then launch. Catching yourself send a portable
   capture to the scarce fast node — or waiting in its queue behind others — is the bug.
4. **GPU-centric rule (when the plan needs BOTH CPU and GPU runs):** they sit on different partitions, so
   ALWAYS co-run them in parallel (CPU job + GPU job at once), never CPU-then-GPU. Before firing the GPU
   side: (a) SIZE it to the FEWEST GPUs that fit the need (a capture/oracle only needs enough to LOAD;
   `tp` must divide the head count; fit by memory/GB-per-GPU — don't grab the whole box out of habit);
   (b) check the free GPU count and request only that many (by UUID); (c) fewer GPUs/job ⇒ more concurrent
   GPU jobs fit ⇒ parallel GPU waves; (d) serialize only jobs sharing mutable global state (e.g. a shared
   CDI spec). Net: assess need → size GPUs minimally → check free → fire CPU + GPU (and GPU waves) at once.
Route by run shape:
- **Independent hypotheses (bisection, a variant sweep that can't be merged into one load) → fan out a
  WAVE across the many-node partition** (e.g. EMR/SPR with 7-8 idle nodes): N runs finish in ~1 run of
  wall-time instead of N. This is the parallel dual of a high-information run — when you CAN'T collapse
  to one run, collapse the wall-time instead.
- **A single decisive run → the FASTEST single node** (e.g. GNR), not the parallel farm. Don't default
  to the farm out of habit for a one-shot; but if the fast node is `alloc`/busy, a job already RUNNING
  on a slower node beats a PENDING job on the fast one — queue wait is part of run_cost.
- **The reference oracle lives on its own node class** (e.g. GPU box). Request only what fits (partial
  GPUs / `--tp N` by UUID); don't block on the whole node. Serialize jobs that share mutable global
  state (e.g. concurrent `nvidia-ctk cdi generate` clobber a shared spec → "unresolvable CDI devices").
- **Portability gates the choice.** CORRECTNESS is ISA-portable → run it anywhere idle (the farm).
  PERF is NOT → final numbers must be on the target HW, so a perf run can't be parallelized onto the
  farm; a correctness bisection can.
- **Pre-flight cheap probes on the login node / a tiny alloc** (shape/dtype/offline math) so the
  scheduled nodes only carry what needs real scale.

## Anti-patterns
- One-question-per-run serial iteration (the default) — each expensive run a single yes/no.
- Re-loading the model to flip one flag that could have been swept in-process.
- Gathering data with no disposition plan → a follow-up run just to interpret it.
- Carrying an already-cheaply-killable hypothesis into the expensive run.
- Defaulting to the parallel farm for a single decisive run (an idle FASTER single node wins), or
  blocking on a busy fast node when an idle slower node is already free (queue wait is run_cost).
- Running independent bisection hypotheses SERIALLY when idle nodes could run them as a parallel wave.

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
