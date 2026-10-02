# GLM-5.3 Flash — enablement kickoff (fresh-session prompt)

> Paste the block below into a NEW chat session opened on this workspace, or just tell the
> new session: **"Read `.agents/tasks/glm-5.3-flash-kickoff.md` and execute it."**
> This is a deliberate clean test of whether the playbook is self-sufficient.

---

TASK: Enable GLM-5.3 Flash for performant, ACCURATE CPU inference in SGLang on Intel
Xeon, delivered as the external plugin (NO fork of sglang/). This is a deliberate CLEAN
TEST of the agentic playbook we built while enabling DeepSeek-V4-Flash: the playbook must
carry this end-to-end with MINIMAL guidance from me.

SUCCESS CRITERION — AUTONOMY + VELOCITY:
- Drive the whole thing yourself. I should be able to leave and return to a DONE result.
- Terse output: a short final report only. Interrupt me with AT MOST ONE batched ask, and
  only for a TRUE blocker (HuggingFace gated-model access/token, or a genuine novel-op
  design fork that needs a human decision). Otherwise, no check-ins.
- This is the velocity proof: INSTRUMENT the enablement cost from the start (wall-clock,
  number of fix-cycles, and which known gotchas the playbook auto-avoided). The delta vs
  the DeepSeek-V4-Flash effort is the result. Fair baseline note: the DSv4-Flash time
  INCLUDED building the playbook; this GLM run is the clean "playbook-applied" run.

USE THE PLAYBOOK (do not reinvent the method):
- Read FIRST: .agents/skills/README.md, then the model-enablement-playbook and
  cpu-optimization-playbook skills; load the other skills as their triggers fire.
- Also read user memory /memories/debugging-discipline.md (carries the hard-won lessons:
  spin-wait/canonical-CPU-config, compute-from-native-precision, don't-decompose-fused-op,
  validate-the-measurement-first).
- Apply the full ladder the playbook prescribes:
  coverage-gate + enablement-scope-discovery (find the TRUE novel ops vs donor-covered)
  -> up-front dtype hygiene (census STORED dtype from the real checkpoint, map to a
     HW-supported COMPUTE dtype, keep native low-bit nibbles — never up-convert —
     parity-check EVERY stored!=compute bridge)
  -> reference-first, parity-gated authoring of any genuine GAP op (vs in-tree oracles)
  -> accuracy-oracle ladder: per-op kernel parity -> real-prompt coherence -> per-LAYER
     parity (GPU/reference forward if available) -> task accuracy (gsm8k)
  -> perf ladder, in ORDER: (0) rule out systemic-config pathology FIRST
     (OMP_WAIT_POLICY=passive, KMP_BLOCKTIME=0, thread cap, NUMA) -> (1) truncated-depth
     full-width DUMMY proxy (fast iterate) -> (2) full-model DUMMY (authoritative
     perf-vs-roofline) -> (3) full weights (accuracy + no-regression confirm).
     tp=1 + decode-thread-cap is the decode default; batching is the decode lever.

CAPTURE NEW LEARNINGS (a PRIMARY output — the skills are the product):
- Expect this model to surface gotchas the playbook didn't cover. Whenever you hit a new
  insight, pitfall, or decision rule, DISTILL it into the right place in .agents/skills/
  (update an existing skill, or add one) — concise, grounded in the measured evidence that
  produced it. If it's a cross-session/standing lesson, also update /memories/.
- Treat "what did GLM teach us that DSv4-Flash didn't" as an explicit deliverable, not an
  afterthought. The point of the clean test is to both USE and IMPROVE the playbook.

MODEL SOURCE: pull GLM-5.3 Flash config + weights from Hugging Face (as we did for
DSv4-Flash) into /scratch/bkaul/models/<name>. It is NOT on the cluster yet. If the repo
is gated, that's the one thing to ask me for (HF token / license acceptance).

ENVIRONMENT (site-specific bootstrap):
- venv: source /scratch/bkaul/venvs/sglang-cpu/bin/activate (torch CPU; links libgomp).
- Plugin runtime env (at LAUNCH, before import): SGLANG_USE_CPU_ENGINE=1,
  SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models,
  PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin, OMP_WAIT_POLICY=passive,
  KMP_BLOCKTIME=0 (libgomp reads OMP at load -> MUST be in the launch env, not in-process).
- Plugin code: plugin/intel_cpu_models/ ; validators + sbatch harnesses: plugin/validate/
  (already parameterized — reuse them: throughput, timeit, gsm8k, parity).
- Cluster: EMR partition `emr` (FREE farm, 24 nodes ~1TB, 2 sockets x 64c, 226 GB/s/domain)
  for all iteration; GNR partition `gnrap` (1 scarce node, 6 SNC domains) only for a final
  perf-node run if needed. De-risk on the free farm before any scarce-node run.

GOVERNANCE (same as DSv4-Flash):
- Correctness gates BEFORE any perf claim; never dense-approximate a novel op.
- Parallel DISCOVERY of opts is fine; INTEGRATION serial, one-by-one, each validated vs an
  immutable reference before locking in. Env-gate each opt for A/B + rollback.
- NEVER edit plugin source while jobs are pending (jobs read the live tree at launch).
- Honest roofline: always compare measured vs target at the SAME labeled machine config;
  label anything unvalidated as UNVALIDATED.
- Checkpoint progress to /memories/session/ as you go (survives compaction).

DELIVERABLE + GITHUB INTEGRITY:
- GLM-5.3 Flash that RUNS end-to-end on CPU, is accuracy-parity vs a trusted reference, and
  ships a roofline-target-vs-measured artifact — plus the enablement certificate and the
  velocity delta vs DSv4-Flash.
- Update the GitHub Thesis 2 story to ADD GLM-5.3 Flash: README.md (the
  "Thesis 2 — new-kernel leg" section — add GLM alongside DeepSeek-V4-Flash, same
  approach-vs-demonstrated structure), THESIS_SUMMARY.md, and a
  plugin/validate/results/glm_5_3_flash_roofline.* artifact (reuse roofline_vs_measured.py).
- ENSURE CONSISTENCY/INTEGRITY across ALL docs before pushing: no stale or contradictory
  claims, every link resolves, numbers match across README / THESIS_SUMMARY / results /
  certificate, measured-vs-target always labeled at the same config, and the skill count /
  cross-references stay correct. Do a final read-through pass; then commit + push.
- Finish with the short report: velocity delta, accuracy, roofline-vs-measured, and the new
  learnings captured into the playbook.
