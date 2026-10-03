#!/bin/bash
# GLM-5.3 Flash CPU DUMMY-weight per-layer FINGERPRINT (tiny arch-faithful config + DETERMINISTIC
# dummy weights, prefill-only). Pairs with run_glm5_gpu_dummy_fullcap.sbatch: SAME tiny config +
# SAME numpy name-seeded dummy init -> bit-identical weights -> the two captures diff apples-to-
# apples (diff_fullcap.py) to prove the CPU wiring + kernel math matches the GPU reference.
# ISA-portable correctness run -> emr/gnrap, whichever frees first.
#SBATCH --partition=emr,gnrap
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH --job-name=glm5_cpu_dummy_fc
#SBATCH --output=/scratch/bkaul/glm5_cpu_dummy_fullcap_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0
cd /data/nfs_home/bkaul
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/make_glm5_tiny.py
export MODEL_PATH=/scratch/bkaul/models/glm5-tiny
export LOAD_FORMAT=dummy
export DETERMINISTIC_DUMMY=1
export INTEL_CPU_GLM_FULLCAP=1
export FULLCAP_FILE=/scratch/bkaul/glm5_cpu_dummy_fullcap_${SLURM_JOB_ID}.pt
export MEM_FRAC=0.30
export LOG_LEVEL=info
export MAMBA_RADIX_STRATEGY=no_buffer
export DISABLE_RADIX=1
# PREFILL-ONLY: the per-layer PREFILL fingerprint is the correctness gate; decode NoPE-MLA deferred.
export MAX_NEW=1
export SMOKE_PROMPTS="The capital of France is"
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/_smoke.py
echo "[fullcap] $FULLCAP_FILE"; ls -la "$FULLCAP_FILE" 2>/dev/null
python3 -c "import torch;d=torch.load('$FULLCAP_FILE',map_location='cpu');print(len(d),'tensors');[print(k,tuple(v.shape)) for k,v in list(d.items())[:24]]" 2>/dev/null || echo "no fingerprint saved"
echo "[exit=$?]"
