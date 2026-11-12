#!/bin/bash
#SBATCH --job-name=dsv4_meta_sink_profile
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodelist=pcl-sprh01
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/dsv4_meta_sink_profile_%j.log
set -euo pipefail
ROOT=/data/nfs_home/bkaul/dsv4-standalone-meta-opt
PYTHON=/scratch/bkaul/venvs/sglang-cpu/bin/python
RESULTS="/scratch/bkaul/dsv4-standalone-meta-opt/sink-profile-job-${SLURM_JOB_ID}"
export RESULTS
test ! -e "$RESULTS" || { echo "FATAL: results already exist"; exit 1; }
mkdir -p "$RESULTS"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH OMP_WAIT_POLICY GOMP_SPINCOUNT KMP_BLOCKTIME
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|INDEXER_*|TOPK_*|SINK_*|SPARSE_*|ONEDNN_*|DNNL_*) unset "$key" ;;
    esac
done
export OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=threads PROBE_THREADS=64
sha256sum plugin/validate/run_standalone_meta_sink_profile.sbatch \
    plugin/validate/uarch_probe.py tools/uarch_perf_probe/{probes,hygiene}.py \
    > "$RESULTS/calibration-sources.sha256"
numactl --hardware > "$RESULTS/numa.txt"
numactl --physcpubind=0-63 --membind=0 "$PYTHON" -u - <<'PY'
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, "tools")
sys.path.insert(0, "plugin/validate")
from uarch_perf_probe import hygiene, probes
from uarch_probe import measure_bw

fp32 = probes.probe_compute_peak(dtypes=("float32",))
bw = measure_bw()
rate = fp32["peak"]["float32"]["gflops"]
if not all(math.isfinite(x) and x > 0 for x in (rate, bw)):
    raise RuntimeError("invalid calibration")
result = {
    "job": os.environ["SLURM_JOB_ID"], "pid": os.getpid(), "meta": hygiene.env_report(),
    "compute_peak": fp32, "stream_triad_useful_GBs": bw,
    "triad_working_set_bytes": 3 * 512 * 1024 * 1024 * 4,
    "fp32_gemm_ridge_flops_per_byte": rate / bw,
    "scope": "Best-of library GEMM and 6GiB triad probes, not a serial-division/recurrence ceiling. Frequency hygiene is reported, not changed.",
    "memory_probe": "uarch_probe.measure_bw; NOT uPP's 192MiB aggregate triad, which fits this socket's 320MiB LLC",
}
Path(os.environ["RESULTS"], "machine_constants.json").write_text(json.dumps(result, indent=2) + "\n")
print("CALIBRATION_COMPLETE", rate, bw, flush=True)
PY
numactl --physcpubind=0-63 --membind=0 perf record -e cycles:u -F 499 \
    -o "$RESULTS/perf.data" -- "$PYTHON" -u plugin/validate/standalone_meta_sink_profile.py \
    --baseline-dir /scratch/bkaul/dsv4-standalone-meta-opt/sink-iterations-job-398645 \
    --output "$RESULTS/profile.identity.json"
perf report --stdio --no-children --sort symbol -i "$RESULTS/perf.data" \
    > "$RESULTS/perf-report.txt"
perf annotate --stdio --percent-type local-period -i "$RESULTS/perf.data" \
    > "$RESULTS/perf-annotate.txt"
echo "SINK_PROFILE_COMPLETE job=$SLURM_JOB_ID results=$RESULTS"
