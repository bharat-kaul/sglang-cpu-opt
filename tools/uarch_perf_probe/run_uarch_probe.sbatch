#!/bin/bash
# uArch Performance Probe — full characterization on a batch node.
# Emits machine_constants.{json,md} for the node it lands on. For authoritative numbers the
# node should have a pinned frequency (governor=performance, turbo off); uPP flags otherwise.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=uarch_probe
#SBATCH --output=/scratch/%u/uarch_probe_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
cd /data/nfs_home/bkaul/sglang-cpu-opt
OUT=/scratch/$USER/machine_constants_${SLURM_JOB_ID}.json
# Full suite incl. the NUMA/SNC bandwidth matrix (spawns numactl subprocesses per domain pair).
python tools/uarch_perf_probe/runner.py --out "$OUT"
echo "[exit=$?] wrote $OUT (+ .md). Copy next to the consuming workflow for provenance."
