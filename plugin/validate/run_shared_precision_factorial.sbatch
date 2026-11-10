#!/bin/bash
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --job-name=shared_precision
#SBATCH --output=/scratch/bkaul/shared_precision_%A_%a.log
set -euo pipefail
ROOT="${1:?evidence root required}"
SOURCE="${2:?frozen script directory required}"
GPU_JOB="${3:?native GPU capture job required}"
ANALYSIS="${4:-replay_shared_precision_factorial.py}"
CPU_JOBS=(398194 398192 398193 398195)
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0
exec /scratch/bkaul/venvs/sglang-cpu/bin/python -u \
    "$SOURCE/$ANALYSIS" "$ROOT" "$GPU_JOB" \
    --cpu-job "${CPU_JOBS[$SLURM_ARRAY_TASK_ID]}"
