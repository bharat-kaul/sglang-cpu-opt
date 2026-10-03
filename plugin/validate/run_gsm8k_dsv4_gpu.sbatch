#!/bin/bash
# GPU ORACLE for the gsm8k task cert: run OUR harness (same clean-CoT prompt, same extraction)
# on GPU via the stock sglang CUDA image, so CPU-vs-GPU under the IDENTICAL harness certifies the
# CPU implementation (greedy -> expect near-identical). DSV4-Flash ~149GB -> tp=2 (2x H200 141GB).
# NO plugin (GPU uses stock sglang deepseek_v4); PYTHONPATH only for the task_gsm8k import.
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=dsv4_gsm8k_gpu
#SBATCH --output=/scratch/bkaul/dsv4_gsm8k_gpu_%j.log
set -x
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR" /tmp/podman-runroot; chmod 700 "$XDG_RUNTIME_DIR"
export TMPDIR=/scratch/bkaul/tmp; mkdir -p "$TMPDIR"
nvidia-ctk cdi generate --format=yaml > "$HOME/.config/cdi/nvidia.yaml" 2>/dev/null
IMG=docker.io/lmsysorg/sglang:latest
nvidia-smi -L
# Reuse the shared pre-pulled image in the storage.conf graphroot (/scratch). The subuid blocker
# (no /etc/subuid range) is bypassed by overlay ignore_chown_errors=true in ~/.config/containers/
# storage.conf; --userns=host maps the container to the invoking user at run time.
podman --cgroup-manager=cgroupfs \
  run --rm --device nvidia.com/gpu=all --userns=host \
  --ipc=host --network=host \
  -v /scratch:/scratch -v /data/nfs_home/bkaul:/data/nfs_home/bkaul \
  -e PYTHONPATH=/data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_CUMEM_ENABLE=0 \
  "$IMG" \
  python /data/nfs_home/bkaul/sglang-cpu-opt/plugin/validate/task_gsm8k_chunked.py \
    --model /scratch/bkaul/models/DeepSeek-V4-Flash \
    --data /scratch/bkaul/gsm8k_test.jsonl \
    --device cuda --tp 2 \
    --num-questions ${NQ:-64} --chunk-size ${CHUNK:-16} --num-shots 8 \
    --max-new-tokens ${OLEN:-512} --watchdog-timeout 3000 --chunked-prefill-size 512 \
    --out /scratch/bkaul/dsv4_gsm8k_gpu_${SLURM_JOB_ID}.json \
    --gens-out /scratch/bkaul/dsv4_gsm8k_gpu_gens_${SLURM_JOB_ID}.json 2>&1 | tail -60
echo "[exit=$?]"
