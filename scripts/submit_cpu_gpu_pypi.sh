#!/usr/bin/env bash
set -euo pipefail

# Submit 3-stage chain:
#   1) CPU pipeline PBS
#   2) GPU NN PBS (afterok CPU)
#   3) PyPI publish PBS (afterok GPU)
#
# Usage:
#   bash scripts/submit_cpu_gpu_pypi.sh
#   bash scripts/submit_cpu_gpu_pypi.sh scripts/run_pipeline_pbs.sh scripts/run_nn_gpu_pbs.sh scripts/run_publish_pypi_pbs.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU_SCRIPT="${1:-${ROOT}/scripts/run_pipeline_pbs.sh}"
GPU_SCRIPT="${2:-${ROOT}/scripts/run_nn_gpu_pbs.sh}"
PYPI_SCRIPT="${3:-${ROOT}/scripts/run_publish_pypi_pbs.sh}"

for f in "$CPU_SCRIPT" "$GPU_SCRIPT" "$PYPI_SCRIPT"; do
  if [[ ! -f "$f" ]]; then
    echo "PBS script not found: $f" >&2
    exit 1
  fi
done

echo "Submitting CPU job: $CPU_SCRIPT"
CPU_JOB_ID="$(qsub "$CPU_SCRIPT")"
echo "CPU job id: $CPU_JOB_ID"

echo "Submitting GPU job after CPU success: $GPU_SCRIPT"
GPU_JOB_ID="$(qsub -W depend=afterok:${CPU_JOB_ID} "$GPU_SCRIPT")"
echo "GPU job id: $GPU_JOB_ID"

echo "Submitting PyPI job after GPU success: $PYPI_SCRIPT"
PYPI_JOB_ID="$(qsub -W depend=afterok:${GPU_JOB_ID} "$PYPI_SCRIPT")"
echo "PyPI job id: $PYPI_JOB_ID"

echo "Done. Chain:"
echo "  CPU  -> $CPU_JOB_ID"
echo "  GPU  -> $GPU_JOB_ID (afterok CPU)"
echo "  PyPI -> $PYPI_JOB_ID (afterok GPU)"

