#!/bin/bash
#SBATCH --partition=emr
#SBATCH --constraint=ddr5600
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=dsv4_cpu0522
#SBATCH --output=/scratch/bkaul/dsv4_cpu0522_%j.log
export XDG_RUNTIME_DIR=/scratch/bkaul/xdg_${SLURM_JOB_ID}; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
PROOT=/scratch/bkaul/podman/root; PRUN=/scratch/bkaul/podman/runroot
podman --root $PROOT --runroot $PRUN --events-backend=none run --rm \
  -v /scratch:/scratch --ipc=host --network=host \
  -e SGLANG_USE_CPU_ENGINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  docker.io/lmsysorg/sglang:latest python /scratch/bkaul/probe_cpu_0522.py 2>&1 | tail -20
echo "[exit=${PIPESTATUS[0]}]"
