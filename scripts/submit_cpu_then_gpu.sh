#!/usr/bin/env bash
set -euo pipefail

# Submit CPU PBS job first, then submit GPU PBS job with dependency:
# GPU starts only when CPU job exits successfully (afterok).
#
# Usage:
#   bash scripts/submit_cpu_then_gpu.sh
#   bash scripts/submit_cpu_then_gpu.sh scripts/run_pipeline_pbs.sh scripts/run_nn_gpu_pbs.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU_SCRIPT="${1:-${ROOT}/scripts/run_pipeline_pbs.sh}"
GPU_SCRIPT="${2:-${ROOT}/scripts/run_nn_gpu_pbs.sh}"

if [[ ! -f "$CPU_SCRIPT" ]]; then
  echo "CPU PBS script not found: $CPU_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$GPU_SCRIPT" ]]; then
  echo "GPU PBS script not found: $GPU_SCRIPT" >&2
  exit 1
fi

echo "Submitting CPU job: $CPU_SCRIPT"
CPU_JOB_ID="$(qsub "$CPU_SCRIPT")"
echo "CPU job id: $CPU_JOB_ID"

echo "Submitting GPU job after CPU success: $GPU_SCRIPT"
GPU_JOB_ID="$(qsub -W depend=afterok:${CPU_JOB_ID} "$GPU_SCRIPT")"
echo "GPU job id: $GPU_JOB_ID"

echo "Done. GPU will start only if CPU job completes successfully."

