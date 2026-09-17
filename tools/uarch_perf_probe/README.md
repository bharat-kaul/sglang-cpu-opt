# uArch Performance Probe (uPP)

Standalone, workload-agnostic microbenchmark suite that **characterizes a CPU
microarchitecture** and emits a `machine_constants.json` of measured constants that feed
kernel-authoring and roofline workflows.

The durable asset for optimizing on an unseen uarch (e.g. a GNR follow-on) is not a table of
constants — it is *the self-validating harness that regenerates them on the real silicon*.
uPP is that harness. It also refreshes/verifies priors on a known part to catch config drift.

> No dependency on any model or serving stack. Any agentic workflow can call the runner and
> consume the JSON. See the `uarch-perf-probe` skill for the agent-facing contract.

## Quickstart
```bash
pip install torch            # only hard dependency; numactl/lscpu optional (NUMA probe)
python tools/uarch_perf_probe/runner.py --out machine_constants.json     # full
python tools/uarch_perf_probe/runner.py --quick                          # skip NUMA matrix
python tools/uarch_perf_probe/runner.py --probes compute_peak,memory     # subset
sbatch tools/uarch_perf_probe/run_uarch_probe.sbatch                     # batch node
```
For **authoritative** numbers, pin the frequency first (governor=performance, turbo off); uPP
flags an unpinned node with `⚠` and `meta.freq_hygiene.clean=false`.

## Probes → constants → knobs
| Probe | Emits | Kernel knob it informs |
|---|---|---|
| `compute_peak` | achieved GF/s per dtype (bf16/int8/fp32) | roofline compute asymptote; is int8 2× real here |
| `memory` | cache-ladder BW vs footprint + DRAM triad | L1/L2/L3 knees → BLOCK_K/M/N; roofline BW asymptote |
| `numa` | per-(cpu,mem)-domain BW matrix, local/remote | domains, per-domain BW → 1 rank/domain, SNC roofline |
| `roofline_ridge` | flops/byte ridge = peak/BW | memory-bound vs compute-bound decision |
| `gather_crossover` | smallest M where gather amortizes | stage-then-BRGEMM vs tinygemm |
| `threading` | compute + BW scaling vs threads | cores-to-saturate-BW, grain, split-K |

## Output
`machine_constants.json` + `.md`:
- `meta`: node, ISA flags, freq-hygiene, NUMA node count, torch/cores.
- one block per probe with the raw sweep + `status`.
- `derived_kernel_knobs`: ready-to-use values (`prefer_amx_stage_M_ge`, `cores_to_saturate_bw`,
  `ridge_flops_per_byte`, `per_domain_bw_gbps`, `n_tp_ranks_hint`, `remote_bw_penalty`).

## Files
```
tools/uarch_perf_probe/
  runner.py     # CLI: orchestrate -> machine_constants.{json,md} + derived knobs
  probes.py     # the probe library (one function per probe)
  hygiene.py    # timing (warmup/best-of-N/DCE guard) + ISA/freq/NUMA capture
  run_uarch_probe.sbatch
```

## Extending (native probes / new uarch)
The probes are torch/oneDNN-level (they measure *achievable* peaks through the same AMX/BRGEMM
stack kernels dispatch to). Add cycle-accurate raw-intrinsic probes, or a probe for a
categorically-new uarch feature (new tile shape, memory tier, clustering), as a new function in
`probes.py` that returns the same `{status, ...}` contract and carries its own dispatch +
hygiene self-checks. Keep the JSON schema stable for consumers.

## Demo consumer
This repo's DeepSeek-V4 CPU enablement workflow is the reference consumer: its roofline charts
and kernel-authoring feasibility gate read the same constants uPP emits.

## Sample output (GNR, demo)
[`samples/gnr_pcl-gnrap01_demo.md`](samples/gnr_pcl-gnrap01_demo.md) — a run on the GNR node:
uPP independently regenerated the SNC constants from measurement (**6 domains**, per-domain BW
~226 GB/s, remote/local BW 0.61) that the workflow had baked into its roofline from priors —
i.e. the tool rediscovers the priors on real silicon. Caveats visible in that sample (honest,
and both are flagged by the run): frequency was **not pinned** (turbo on → `clean=false`), the
**int8 peak looks low** (`torch._int_mm` likely not dispatching AMX-int8 — a native-probe /
dispatch-check refinement), and the **smallest cache-ladder points are dispatch-bound noise**
(need more iters for tiny buffers). Pin frequency + add the int8 dispatch check for an
authoritative run.
