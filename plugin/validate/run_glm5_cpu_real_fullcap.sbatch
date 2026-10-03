#!/bin/bash
# GLM-5.3 Flash CPU REAL-WEIGHT all-layer prefill FINGERPRINT (fp8 checkpoint -> fp8->bf16 W8A16
# bridge). Captures pf.L0-44 + hcpost + logits on ONE short prompt, to diff vs the ARCHIVED GPU
# real-weight fingerprint (fingerprints/glm5_gpu_fullcap_REAL_381095.pt, all 45 layers, tp4) ->
# the authoritative ALL-LAYER correctness gate AND the first exercise of the fp8 bridge. Big-mem
# node: 306GB fp8 -> ~600GB after the bf16 bridge + activations.
# Big-mem node REQUIRED: 306GB fp8 -> ~600GB after the bf16 bridge; needs the 1.5TB gnrap node.
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --job-name=glm5_cpu_real_fc
#SBATCH --output=/scratch/bkaul/glm5_cpu_real_fullcap_%j.log
set -x
source /scratch/$USER/venvs/sglang-cpu/bin/activate
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models
export PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0
cd /data/nfs_home/bkaul
export MODEL_PATH=/scratch/bkaul/models/GLM-5.3-Flash
# REAL weights (no LOAD_FORMAT=dummy) -> exercises the fp8->bf16 W8A16 bridge.
export INTEL_CPU_GLM_FULLCAP=1
export FULLCAP_FILE=/scratch/bkaul/glm5_cpu_real_fullcap_${SLURM_JOB_ID}.pt
# CPU mem_fraction_static = TOTAL engine budget (model + KV/mamba pools), NOT a GPU-style KV-only
# fraction. The ~600GB bridged model needs a HIGH fraction of the 1.5TB node; 0.15 left the mamba
# cache negative. MAX_RUNNING=1 (one capture prompt) keeps the per-req mamba/KV caches tiny.
export MEM_FRAC=${MEM_FRAC:-0.9}
export MAX_RUNNING=${MAX_RUNNING:-1}
export LOG_LEVEL=info
export MAMBA_RADIX_STRATEGY=no_buffer
export DISABLE_RADIX=1
# PREFILL-ONLY: the archived GPU fingerprint is prefill (pf.L0-44); match it. Same first prompt.
export MAX_NEW=1
export SMOKE_PROMPTS="The capital of France is"
python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/_smoke.py
echo "[fullcap] $FULLCAP_FILE"; ls -la "$FULLCAP_FILE" 2>/dev/null
python3 -c "import torch;d=torch.load('$FULLCAP_FILE',map_location='cpu');ks=list(d.keys());print(len(d),'tensors; pf layers:',sorted(int(k.split('.L')[1].split('.r')[0]) for k in ks if k.startswith('pf.L') and 'hcpost' not in k)[-3:])" 2>/dev/null || echo "no fingerprint saved"
echo "[exit=$?]"
