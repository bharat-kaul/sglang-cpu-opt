# Performance Scales — a scale-stratified reference for autonomous CPU model perf

A **parallel tree** over the per-technique skills, organizing the SGLang-CPU performance corpus by
the three scales at which lessons actually live, so a NEW model can be carried to best-possible
performance **autonomously** without re-deriving them the expensive way.

## Why this tree exists
The lessons were already in the corpus — but scattered across ~25 per-technique skills and loaded
REACTIVELY, so they were not applied UP FRONT. Result: macro-scale truths (whole-operator fusion,
seam cost, operating point) were missed while reasoning at the micro/meso scale → a roofline proxy
predicted 6–7× for a kernel that delivered 1.07×. This tree makes the multi-scale knowledge an
up-front, enforced flow.

## North-star (the only definition of "done")
1. **Kernel-domination:** `Σ(model-op/kernel time) / wall ≥ ~0.90` — framework/wiring glue → a small
   remainder. 2. **Each dominant kernel at its roofline** for its regime, gap explained. Accuracy is a
hard gate throughout (`accuracy-oracle`).

## The scales + the feedback loops (solid = first-pass discovery; dotted = bottleneck-driven feedback)
```mermaid
flowchart TB
  NS["★ NORTH-STAR: wall = Σ kernel-time (glue→0) AND each kernel at its roofline"] --> LOOP
  LOOP["↻ EVERY ITERATION (bottleneck-driven): profile → pick the SINGLE dominant bottleneck → classify its scale → apply ONE lever → re-profile. Next lever may be the SAME scale (within) or another (across)."] --> MACRO
  subgraph MACRO["MACRO — make the profile KERNEL-DOMINATED  ↻ iterate within"]
    M1["whole-operator fusion (no DRAM materialization)"]
    M2["dispatch elimination (fewer/larger ops, bf16 e2e, warm torch.compile)"]
    M3["operating point (batch M): amortizes WEIGHT-bound ops; per-sequence STATE ops do NOT"]
    M4["parallelism + NUMA/SNC; thread-cliffs; OMP spin-wait"]
    M5["measure the SEAM, not the kernel"]
  end
  MACRO -->|kernel-dominated| MESO
  subgraph MESO["MESO — each kernel's DATA MOVEMENT at the streamed roofline  ↻ iterate within"]
    S1["two-ceiling: streamed ≠ resident peak; BRGEMM is the baseline"]
    S2["AI vs RIDGE (measured BW): >ridge compute-bound (grows w/M); <ridge BW-bound (op-point only)"]
    S3["weight prepack → persisted VNNI/AMX layout"]
    S4["low-precision STORAGE = data-movement lever; contiguous-layout invariant"]
  end
  MESO -->|data movement at roofline| MICRO
  subgraph MICRO["MICRO — each INNER LOOP at the AMX/VNNI peak  ↻ iterate within"]
    U1["tile op (dpbf16/dpbusd) + fp32 accumulate"]
    U2["VNNI pack; inline fp8→bf16 in load; scale folded in FMA"]
    U3["FUSED epilogue (activation/dequant in the store)"]
    U4["SETTLED NULL-TRAPS: thread-cap / ILP / prefetch — do NOT re-test"]
  end
  MICRO -.->|re-profile → re-dispatch on the revealed bottleneck| LOOP
  MESO  -.->|op BW-bound / at floor → feed UP (operating point or fuse)| LOOP
  MICRO -.->|inner loop can't beat floor → fuse into neighbor| LOOP
  GATES["CROSS-CUTTING GATES: (0) CAPACITY/FEASIBILITY pre-flight — size memory vs node BEFORE every launch, cap batch/context/depth to fit · (1) upfront multi-scale DONOR study, model WHOLE operator incl. seam · (2) confirm microbench wins IN-ENGINE · (3) correctness parity + CPU↔GPU equivalence"]
  GATES -.-> MACRO
  GATES -.-> MESO
  GATES -.-> MICRO
```
Iteration is **bottleneck-driven and happens BOTH within a scale and across scales**: every pass
re-profiles and the revealed dominant bottleneck picks the next lever (same scale = within-scale
iteration; a floor feeding up or the dominant op shifting = across-scale). See
`multiscale-optimization` § ITERATIVE, BOTTLENECK-DRIVEN FEEDBACK LOOPS. A single macro→meso→micro
pass without re-profiling + re-dispatching is the waterfall anti-pattern.

## The tree (traverse TOP-DOWN — fix the biggest scale first)
- **[multiscale-optimization](multiscale-optimization/SKILL.md)** — ENTRY POINT: north-star,
  top-down traversal, GATE 1 (upfront multi-scale donor study + whole-operator/seam model), GATE 2
  (confirm microbench wins in-engine).
- **[macro-scale](macro-scale/SKILL.md)** — make the profile KERNEL-DOMINATED: kill framework/dispatch
  overhead, fuse the WHOLE operator (no materialization), parallelize + NUMA, pick the OPERATING POINT
  (batch M). *Biggest wall-time wins; done first.*
- **[meso-scale](meso-scale/SKILL.md)** — each dominant kernel's DATA MOVEMENT at the streamed-roofline:
  BRGEMM resident-C, cache-tiling, weight prepacking, operand-byte reduction; AI-vs-ridge classifies
  compute-bound (kernel-worthy) vs BW-bound (operating-point-only).
- **[micro-scale](micro-scale/SKILL.md)** — each INNER LOOP at the AMX/VNNI peak: tile op, VNNI packing,
  inline precision conversion, FUSED epilogue; the settled null-traps.

## Scale → leaf-skill map (the per-technique skills remain the detail)
| scale | leaf skills |
|------|-------------|
| macro | `inter-kernel-fusion`, `fusion-analysis`, `openmp-parallelization`, `sub-numa-clustering`, `overhead-attribution`, `runtime-config-tuning`, `cpu-serving-integration` |
| meso | `cache-blocking-tiling`, `weight-prepacking-brgemm`, `kernel-authoring`, `compute-from-native-precision`, `roofline-validation`, `establish-achievable-performance` |
| micro | `amx-vectorization`, `cpu-gemm-amx-bf16`, `quantization-amx-int8`, `uarch-perf-probe` |

Plugs into `cpu-optimization-playbook` (technique routing), `model-enablement-playbook` (enablement
leg), and `model-profile-hotspots` (the roofline-vs-measured + pivot cycle-exit artifact).
