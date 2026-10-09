#!/bin/bash
# F4 acceptance gate — fail-closed job wrapper. Exercises the real harness through the exact job wrapper so a
# child nonzero exit PROPAGATES to a nonzero job exit (the policy's layer-0 bench->job propagation gate).
# Runnable as a plain bash script OR via sbatch. Exit 0 = PARTIAL (no hard-gate failure; thresholds UNRATIFIED,
# NOT a full PASS); nonzero = a hard-gate failure or an infrastructure error (blocks).
#SBATCH --job-name=f4_accept
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/f4_accept_%j.log
set -euo pipefail                                   # any child failure aborts with nonzero exit (propagation)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."                                    # repo root
# shellcheck disable=SC1090
source "/scratch/${USER}/venvs/sglang-cpu/bin/activate"

echo ">>> F4 failure-injection self-test (every injected fault MUST be caught)"
python plugin/validate/f4_acceptance.py --selftest  # exits nonzero if any fault slips -> set -e aborts

echo ">>> F4 acceptance manifest + calibration (PARTIAL ok; FAIL -> nonzero -> job fails)"
python plugin/validate/f4_acceptance.py --calibrate plugin/validate/results/f4_calibration.json

echo ">>> F4 gate wrapper OK (status PARTIAL; thresholds UNRATIFIED — NOT a full PASS)"
