#!/bin/bash
#PBS -P jh2
#PBS -q gpuvolta
#PBS -l walltime=04:00:00
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=200GB
#PBS -l wd
#PBS -N comp_nn_train
#PBS -j oe
#PBS -m bea
#PBS -l storage=scratch/jh2+gdata/jh2+scratch/mk27

set -euo pipefail

# ========= Fill paths =========
PROJECT_ROOT="${PROJECT_ROOT:-/g/data/jh2/jt4478/make_LC_copy}"
VENV="${VENV:-}"
GALAXY="${GALAXY:-ngc628-c}"
OUTNAME="${OUTNAME:-test}"
NFRAME="${NFRAME:-5}"
REFF_LIST="${REFF_LIST:-1,2,3,4,5,6,7,8,9,10}"
NCL="${NCL:-500}"
ML_OUT_DIR="${ML_OUT_DIR:-${PROJECT_ROOT}/nn_sweep_out/${GALAXY}}"
DET_PATH="${DET_PATH:-${PROJECT_ROOT}/det_3d_${GALAXY}.npy}"
NPZ_PATH="${NPZ_PATH:-${PROJECT_ROOT}/allprop_${GALAXY}.npz}"
# ==============================

module load python3/3.11.0
[[ -n "${VENV}" && -f "${VENV}" ]] && source "${VENV}"

cd "${PROJECT_ROOT}"

# Build ML inputs from finished pipeline outputs.
python scripts/build_ml_inputs.py \
  --main-dir "${PROJECT_ROOT}" \
  --galaxy "${GALAXY}" \
  --outname "${OUTNAME}" \
  --nframe "${NFRAME}" \
  --reff-list ${REFF_LIST//,/ } \
  --out-det "${DET_PATH}" \
  --out-npz "${NPZ_PATH}"

# Train NN on GPU.
python scripts/perform_ml_to_learn_completeness.py \
  --det-path "${DET_PATH}" \
  --npz-path "${NPZ_PATH}" \
  --clusters-per-frame "${NCL}" \
  --nframes "${NFRAME}" \
  --nreff "$(awk -F',' '{print NF}' <<< "${REFF_LIST}")" \
  --out-dir "${ML_OUT_DIR}" \
  --save-best

