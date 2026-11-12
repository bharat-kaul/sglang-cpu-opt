#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --job-name=dsv4_c4_numerics
#SBATCH --output=/scratch/bkaul/dsv4_c4_numerics_%j.log
set -euo pipefail
ROOT=$(realpath "${1:?frozen source required}")
DEVICE="${2:?cpu or cuda}"
case "$DEVICE" in cpu|cuda) ;; *) echo "Invalid device" >&2; exit 2 ;; esac
OUT="/scratch/bkaul/dsv4-c4-numerics/job-${SLURM_JOB_ID}"
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code"' EXIT
(cd "$ROOT" && sha256sum --quiet -c ../source-sha256.txt)
for key in $(compgen -e); do
    case "$key" in
        INTEL_CPU_*|DSV4_*|SGLANG_*|OMP_*|GOMP_*|KMP_*|SPARSE_*|ONEDNN_*|DNNL_*) unset "$key" ;;
    esac
done
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [ "$DEVICE" = cpu ]; then
    export PYTHONPATH="$ROOT/plugin" OMP_NUM_THREADS=32 OMP_WAIT_POLICY=active GOMP_SPINCOUNT=1000000
    numactl --physcpubind=0-31 --membind=0 \
        /scratch/bkaul/venvs/sglang-cpu/bin/python -u "$ROOT/plugin/validate/check_c4_numerics.py" \
        --device cpu --output "$OUT/results"
else
    mkdir "$OUT/cdi" "$OUT/run" "$OUT/tmp"
    export XDG_RUNTIME_DIR="$OUT/run" TMPDIR="$OUT/tmp"
    chmod 700 "$XDG_RUNTIME_DIR"
    nvidia-ctk cdi generate --format=yaml --output="$OUT/cdi/nvidia.yaml"
    if [ "${SLURM_JOB_PARTITION:?}" = h100 ]; then
        # This partition has one GPU per node and no Slurm GRES; request an exclusive node.
        UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    else
        IFS=',' read -r -a ALLOCATED <<< "${CUDA_VISIBLE_DEVICES:?No allocated GPU}"
        test "${#ALLOCATED[@]}" = 1
        UUID=$(nvidia-smi --id="${ALLOCATED[0]}" --query-gpu=uuid --format=csv,noheader)
    fi
    [[ "$UUID" =~ ^GPU-[0-9a-f-]+$ ]]
    IMAGE=docker.io/lmsysorg/sglang:v0.5.20-cu130
    GPU_IMAGE_ID=$(podman --root /scratch/bkaul/podman/root --runroot /scratch/bkaul/podman/runroot \
        --events-backend=file --cgroup-manager=cgroupfs image inspect "$IMAGE" --format '{{.Id}}')
    test "$GPU_IMAGE_ID" = 7fafb457002ab4d5a2aa6796d7267f086401065d022347b2f49ae366b6c2687a
    podman --root /scratch/bkaul/podman/root --runroot /scratch/bkaul/podman/runroot \
        --cdi-spec-dir "$OUT/cdi" --events-backend=file --cgroup-manager=cgroupfs \
        run --pull=never --rm --network=none --device "nvidia.com/gpu=$UUID" --ipc=host -v /scratch:/scratch \
        -e PYTHONPATH="$ROOT/plugin" -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
        -e OMP_NUM_THREADS=4 -e GPU_IMAGE_ID="$GPU_IMAGE_ID" -e SLURM_JOB_ID="$SLURM_JOB_ID" \
        -e CUDA_VISIBLE_DEVICES="$UUID" \
        -e TILELANG_CACHE_DIR="$OUT/tilelang" -e TRITON_CACHE_DIR="$OUT/triton" \
        -e TORCHINDUCTOR_CACHE_DIR="$OUT/inductor" -e CUDA_CACHE_PATH="$OUT/cuda" \
        "$GPU_IMAGE_ID" python -u "$ROOT/plugin/validate/check_c4_numerics.py" \
        --device cuda --output "$OUT/results"
fi
(cd "$ROOT" && sha256sum --quiet -c ../source-sha256.txt)
test -s "$OUT/results/complete.json"
echo "C4_NUMERICAL_JOB_COMPLETE job=$SLURM_JOB_ID device=$DEVICE"
