# DSv4 shared-expert SwiGLU-clamp — disposition for reviewer

> **Historical, superseded disposition.** The subsequent
> [review and matched-batch follow-up](DSv4-shared-clamp-disposition-REVIEW.md)
> challenged this experiment's baseline fidelity. See the
> [final full accuracy closure](DSv4-accuracy-closure.md) for the current result.

**Date:** 2026-11-05  ·  **Worktree:** `/data/nfs_home/bkaul/dsv4-clamp-accuracy-worker` (branch
`dsv4-mxfp4-clamp-accuracy`, detached at `9739f86`)  ·  **Model:** DeepSeek-V4-Flash (fp8, 43 layers, 1 shared
expert, `swiglu_limit=10.0`).

## TL;DR disposition
The shared-expert CPU kernel omits the configured `swiglu_limit` clamp (plain SiLU) while GPU clamps. We
built the correction (approach **A**, native-kernel donor, behind a flag) and ran it end-to-end on the
remaining failures. Result: **the clamp is a real operator-parity fix but is NET-NEUTRAL on answers and does
NOT close the residual gap.** Recommended disposition per the agreed decision tree
("operator discrepancy real, answers don't net-recover") → **keep it default-OFF, record it as a parity fix,
and move to the next boundary.** A full sharded re-run is **not warranted** (net-zero recovery on the bounded
set + a baseline-fidelity caveat, below).

---

## What was tested (three gated stages, each fail-closed)

### 1. Matched-input capture (operands from the live production config)
Captured the EXACT `shared_expert_cpu` operands (packed fp8 weights preserved) on the production config
(`INTEL_CPU_DSV4_CHUNK_ACCUM=1` + `INTEL_CPU_DSV4_MXFP4_CLAMP=1`) for the authoritative target set — the **17
GPU-right / CPU-wrong** cases (incl. q995) + **6 recovered controls** — at layers 3/20/42.
- Job **397780**, 69 captures (23 cases × 3 layers), fail-closed completeness passed.
- **Non-perturbation proven bit-exact in-situ** on all 69 (`torch.equal(reconstruction, native)`); the hook
  returns the genuine native output → capture-on ≡ capture-off by construction.
- Authoritative comparator fixed during this work: GPU = **`gpu_flash_gens.json` = 1250** (tp4 H200, git
  `dc62aa0`). (An earlier `tp2fair_395349`=1247 file was the WRONG comparator and would have frozen a wrong
  14-case set; against `gpu_flash` everything reconciles: q995 GPU-right, **17** GPU-right/CPU-wrong, 18→17,
  controls `302,562,640,693,1175,1271`.)

### 2. Operator gate (CPU-native, does the clamp even bind?)
Donor-build a clamped `shared_expert_cpu` (changes ONLY the SwiGLU activation; native GEMMs/dtypes/scaling/add
untouched), replay on the 69 captured operand sets. Artifact: `shared_clamp_replay_397780.json`.
- **Clamp-OFF reproduces production BIT-EXACT on all 69** (reviewer requirement "inactive clamp == production,
  incl. the routed/shared add" — met).
- **Clamp binds only at the final layer:**

  | layer | decision-row changed | max abs Δ (combined) |
  |------|----------------------|----------------------|
  | 3    | 0 / 23               | 0.0                  |
  | 20   | 0 / 23               | 0.0                  |
  | **42** | **23 / 23**        | **2.0**              |

  (L42 direct-shared Δ reaches ~80 on non-decision rows; combined decision-row Δ ≤ 2.0.) So the lead is LIVE
  at the decision row — justified running the answer test.

### 3. Answer-recovery (the deliverable — recovery tied to generated ANSWERS)
Approach **A** wired as a one-line op-swap (`torch.ops.sgl_kernel.shared_expert_cpu` → clamped donor for ALL
MoE layers; `forward_cpu` / hash routing untouched), gated `INTEL_CPU_DSV4_SHARED_CLAMP=1`, per-arm control
file, **no eager fallback** (enabled-but-unbuildable raises). One model load, two arms over the 23 cases,
authoritative `extract_final_answer` scoring. Job **397796**, artifacts `shared_rec_397796_*.json`.

```
OFF (native) = 16/23     ON (clamp) = 16/23     NET RECOVERY = 0
failing cohort:  recovered=[1191]   regressed=[502]   (controls: no change)
only 4 answers changed at all:
  q1191 gold=2   off=2.666(✗) on=2(✓)      RECOVERED
  q502  gold=525 off=525(✓)  on=454.5(✗)   REGRESSED
  q29   gold=2   off=30(✗)   on=13(✗)       changed, still wrong
  q244  gold=73  off=77(✗)   on=82(✗)       changed, still wrong
```

---

## ⚠️ Baseline-fidelity caveat (read before drawing production conclusions)
The OFF arm reproduces the production **397690** answers on only **8/23** cases (15 mismatches). The recovery
harness feeds **frozen token-IDs, batched M=23**; production **397690** fed **text, sharded, chunk-32**. The
batched/frozen path flips borderline greedy tokens on its own, so **this 23-case harness is NOT a faithful
proxy for the production failing set**. The off-vs-on delta is internally valid (both arms share the identical
feeding), but a production answer-level claim would require running the clamp through the real sharded
protocol. Given net-zero recovery on the bounded set, that expensive run is not recommended.

This feeding-path discrepancy is itself a worthwhile thread for your own debug (see open items).

---

## Decision-tree disposition
- Operator discrepancy real (L42 decision row) ✓ ·  Net answer recovery = 0 ·  Controls intact.
- → **Record the correction, keep default-OFF, classify as operator-parity (not accuracy-recovering), move to
  the next boundary.** Do NOT launch a full sharded re-run on this lever.

---

## Code / harness inventory (so you can launch your own debug sessions)

All paths under the worktree `/data/nfs_home/bkaul/dsv4-clamp-accuracy-worker` unless noted. Commits:
`98ca26a` (capture blocker repairs), `5b3bdbb` (capture harness), `9739f86` (approach-A donor + recovery
harness). CPU venv: `/scratch/bkaul/venvs/sglang-cpu`; model `/scratch/bkaul/models/DeepSeek-V4-Flash`.

**Capture (stage 1)**
- `plugin/dsv4_layer_capture/shared_expert.py` — matched-input capture hook (token-ID bound, fail-closed,
  bit-exact non-perturbation proof). Env: `INTEL_CPU_DSV4_SHARED_CAPTURE=<prefix>`,
  `INTEL_CPU_DSV4_SHARED_CAPTURE_FROZEN=<frozen.json>`, `INTEL_CPU_DSV4_SHARED_CAPTURE_LAYERS=3,20,42`.
- `plugin/validate/run_shared_capture.py` + `.sbatch` — one-load driver + launcher (fail-closed completeness).
- `plugin/validate/freeze_gsm8k_tokenids.py` — builds `frozen_targets.json` from `--q-ids` (8-shot).

**Approach A — the clamp correction (stage 2 & 3)**
- `plugin/intel_cpu_models/_dsv4_shared_clamp.py` — private donor (clamped `shared_expert_cpu`: patches only
  the SHARED fp8 SwiGLU activation; renames the kernel; links against installed `common_ops`) + `load()` +
  `install_op_swap()`. Env: `INTEL_CPU_DSV4_SHARED_CLAMP=1`, `INTEL_CPU_DSV4_SHARED_CLAMP_CTL=<file>`
  (`0`=native, `1`=clamp). Build dir: `TORCH_EXTENSIONS_DIR=/scratch/bkaul/dsv4-shared-clamp-ext-stable`.
- `plugin/kernels/dsv4_shared_clamp.h` — runtime clamp-limit global (`<0` off, `>0` clamp).
- `plugin/kernels/dsv4_moe_clamp.h` + `plugin/intel_cpu_models/_dsv4_moe_clamp.py` — REUSED clamp epilogue
  (`dsv4_silu_clamp_mul`, GPU semantics: gate `min(,limit)`, up `clamp(±limit)`, `silu(gate)*up`).
- `plugin/intel_cpu_models/_dsv4_cpu_infra.py` — installs `install_op_swap()` when the flag is set.

**Operator gate (stage 2)**
- `plugin/validate/replay_shared_clamp.py` — clamp off/on replay on captures; off must be bit-exact vs
  production, reports decision-row deltas. Run on an AMX node (builds donor once).

**Answer-recovery (stage 3)**
- `plugin/validate/run_shared_clamp_recovery.py` + `.sbatch` — one load, both arms, recovery table +
  OFF-vs-397690 cross-check.

**Artifacts (`/scratch/bkaul/dsv4_shared_cap/`)**
- `frozen_targets.json` (23 cases) · `cap_397780.<sha12>.L{3,20,42}.pt` (69 captures) ·
  `cap_gens_397780.json` · `shared_clamp_replay_397780.json` (operator gate) ·
  `shared_rec_397796_{shared0,shared1,summary}.json` (recovery) · `*stamp*` provenance.
- Baselines: `/scratch/bkaul/gpu_flash_gens.json` (GPU 1250, authoritative) ·
  `/scratch/bkaul/dsv4_clamp_gens_397690_*.json` (production CPU 1240) ·
  `/scratch/bkaul/dsv4_ca_B_accum_gens_396143_*.json` (B-accum 1241).
- SLURM jobs: capture **397780**, recovery **397796** (partition `emr`).

**Reproduce (sketch)**
```bash
# frozen targets (17 failing + 6 controls)
python plugin/validate/freeze_gsm8k_tokenids.py --model <model> --data /scratch/bkaul/gsm8k_test.jsonl \
   --num-shots 8 --q-ids 5,29,81,146,197,244,247,417,435,502,769,893,924,995,1067,1191,1229,302,562,640,693,1175,1271 \
   --out /scratch/bkaul/dsv4_shared_cap/frozen_targets.json
# operator gate (AMX node): replay clamp off/on on captures
python plugin/validate/replay_shared_clamp.py --caps "/scratch/bkaul/dsv4_shared_cap/cap_397780.*.L*.pt" \
   --frozen /scratch/bkaul/dsv4_shared_cap/frozen_targets.json --out <replay.json>
# answer-recovery (one load, both arms): sbatch -p emr plugin/validate/run_shared_clamp_recovery.sbatch
```

---

## Open items / suggested next boundaries (for your debug sessions)
1. **Harness-vs-production feeding gap** (OFF reproduces 397690 on 8/23). The batched-M=23 / frozen-token-ID
   path is not production-faithful; worth isolating whether it's batching (M=23 vs chunk-32) or tokenization
   (frozen IDs vs text). Until resolved, bounded answer tests on this harness under-represent production.
2. **Next parity boundary.** The shared clamp is now ruled out as the gap-closer. Candidate next: the DSA/MLA
   attention numerical path, or the gate/topk routing precision — pick via the same capture→operator-gate→
   answer-recovery ladder (the harness above generalizes: swap the captured op + the donor).
3. **Keep the clamp available, default-off** (`INTEL_CPU_DSV4_SHARED_CLAMP=1`) as a correctness option; it is
   bit-exact to production when off and a true parity fix when on, just not accuracy-moving on GSM8K.
