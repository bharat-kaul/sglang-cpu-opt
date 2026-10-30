# DSV4-Flash CPU decode — operating point & validated runtime config

**Scope:** single EMR socket, tp=1, physical cores (no SMT), real weights. This page documents the
**validated decode runtime configuration** that gives **~4–5× decode throughput over the naive baseline**
through configuration alone (no kernel changes), and the evidence behind it. Quick start:
`source plugin/intel_cpu_models/recommended_decode_env.sh`.

## The two levers (both pure configuration)

### 1. Warm thread pool — ~2.2× on its own
On CPU every op is its own `at::parallel_for` (fork/join). With threads parking immediately between ops
(`OMP_WAIT_POLICY=passive`), decode pays inter-op **wake latency** across 43 layers × every op × every token.
Keeping the pool hot recovers it.

**Important runtime fact:** torch 2.12 loads **libgomp** (GNU OpenMP), so **`GOMP_SPINCOUNT` is the knob**;
`KMP_BLOCKTIME` is inert on this stack. A **finite** spincount (`1e6`) captures the full win **and** lets
threads park on long idle — the multi-tenant-safe setting, and it avoids the full-socket busy-spin thrash.

| B=1 decode-step @32 threads | cold (passive) | sc=1e6 | sc=1e8 | infinite |
|---|---:|---:|---:|---:|
| ms/step | 594 | 187 | 184 | 184 |

### 2. M-adaptive decode threads — batch-dependent optimum
The decode thread optimum **rises with batch** (more rows → more row-parallel work). 8 threads was a *cold*
artifact (park/wake masked the benefit of more cores); warm, the optimum is 24–48. **58 threads thrashes**
under warm (busy-spin oversubscription) — ceiling is ~48.

Decode-step ms (warm, sticky, real weights), by threads × batch:

| threads | B=1 | B=8 | B=16 | B=32 | B=64 |
|---|---:|---:|---:|---:|---:|
| 16 | 184 | 416 | 565 | 879 | 1477 |
| 24 | **182** | **363** | **504** | 726 | 1187 |
| 32 | 182 | 405 | 541 | **688** | 1102 |
| 48 | 210 | 426 | 533 | 742 | **1020** |

Recommended map (`INTEL_CPU_DSV4_DECODE_THREADS="1:24,16:24,32:32,48:48"`): B≤16 → 24, B=32 → 32, B≥48 → 48.

## Net decode throughput (validated config vs naive cold@8 baseline)

| B | cold@8 (tok/s) | warm + M-adaptive (tok/s) | speedup |
|---|---:|---:|---:|
| 1 | 1.63 | 5.49 | 3.4× |
| 8 | 4.64 | 22.0 | 4.7× |
| 32 | 9.87 | 46.5 | 4.7× |
| 64 | 14.76 | 62.7 | 4.25× |

(Full-model real-weight decode, input-len 8, single socket. Decode-step normalized by `output_len − 1`
since token 1 comes from prefill/extend — see the measurement note below.)

## Validation gates (both PASS)

- **Accuracy parity** — gsm8k, same 200-question subset, differing *only* in threading env:
  cold@8 **0.750** vs warm@32 **0.755** → parity within noise. The config is numerically safe.
- **Finite-spin == infinite** across B=1/8/32 (sc1e6 within ~3% of infinite). The deployable setting.

## Caveats / scope (honest)

- **tp=1 single-socket only.** The per-forward thread cap is applied only at `world_size==1`; at tp>1 the
  SGLang TP shm spin-barriers are sized at init and a post-init `set_num_threads` deadlocks. Out of scope here.
- **Warm = spin.** On a dedicated socket it is strictly better; on a shared/continuous-serving node the finite
  spincount (not infinite) is the right choice so idle threads eventually park.
- **Thread ceiling ~48.** 58-thread warm thrashes during compute/prefill-heavy phases.
- **STICKY is a serving choice.** `STICKY=1` avoids per-forward resize in steady decode; `STICKY=0` lets the
  next prefill reclaim full cores (recommended for mixed prefill/decode serving).
- **Measurement note (corrected):** earlier decode-step numbers divided the `(output_len) − (1)` token delta by
  `output_len` instead of `output_len − 1`; the tables above use the corrected `output_len − 1`. The relative
  speedups are unchanged; absolute ms are ~20% higher than the first (buggy) report.

## Reproduce

```bash
source plugin/intel_cpu_models/recommended_decode_env.sh
# thread/batch sweep (sticky, clean): plugin/validate/run_dsv4_moe_campaign.sbatch
#   sbatch --export=ALL,WARM=1,DTHR=32,STICKY=1,BATCHES=1_8_32,OUTLEN=8,REPS=3 run_dsv4_moe_campaign.sbatch
# accuracy parity: run_dsv4_gsm8k_warmcheck.sbatch (TAG=ref vs TAG=warm)
```

Evidence: campaign jobs 394560–394668. Full analysis +
[review](../plugin/validate/reports/dsv4_decode_warmpool_moe_independent_review.md) ·
[response](../plugin/validate/reports/dsv4_decode_warmpool_moe_review_response.md).
