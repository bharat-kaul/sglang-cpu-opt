# DSV4-Flash CPU decode — RECOMMENDED runtime config (validated campaign, 2026-10-29).
# Source before launching the CPU engine:  source plugin/intel_cpu_models/recommended_decode_env.sh
# Scope: single EMR socket, tp=1, physical cores (no SMT). Gives ~4-5x decode throughput over the naive
# (park-immediately, 8-thread) baseline, accuracy-preserving (gsm8k parity 0.750 vs 0.755), validated by the
# warm-pool + thread-sweep + finite-spin + accuracy campaign (jobs 394560-394668). See
# docs/dsv4-decode-operating-point.md for the evidence, caveats, and the corrected rate tables.

# 1) WARM THREAD POOL (the ~2.2x lever). torch 2.12 loads libgomp, so GOMP_SPINCOUNT is the real knob
#    (KMP_BLOCKTIME is inert). A FINITE spincount captures the full win AND lets threads park on long idle
#    (multi-tenant-safe; avoids the full-socket busy-spin thrash). Do NOT use OMP_PROC_BIND/OMP_PLACES
#    (collapses SGLang to 1 thread on this stack).
export OMP_WAIT_POLICY=active
export GOMP_SPINCOUNT=${GOMP_SPINCOUNT:-1000000}      # 1e6 == infinite within noise, gentler on shared nodes

# 2) M-ADAPTIVE DECODE THREADS (the batch-dependent optimum). Bigger batch = more row-parallel work = more
#    threads; ceiling ~48 (58 THRASHES under warm). Map is "M:T,..." -> pick T for the largest M<=batch.
#    Measured optima: B<=8 ~24, B<=16 ~24, B<=32 ~32, B>=48 ~48.
export INTEL_CPU_DSV4_DECODE_THREADS="1:24,16:24,32:32,48:48"
# STICKY=1 avoids a per-forward thread resize during steady decode; leave 0 for mixed prefill/decode serving
# so prefill reclaims full cores (sticky leaves decode's reduced count for the next prefill).
export INTEL_CPU_DSV4_DECODE_THREADS_STICKY=${INTEL_CPU_DSV4_DECODE_THREADS_STICKY:-0}

# 3) Pin to one socket's PHYSICAL cores at launch (example for a 60-core socket-0; adjust per node):
#    numactl --physcpubind=0-57 --membind=0 python -m sglang... (58 = 60 physical - 2 headroom for OS/main)
