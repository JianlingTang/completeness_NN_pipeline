#!/bin/bash
#PBS -P jh2
#PBS -q normalbw
#PBS -l walltime=00:30:00
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N comp_pypi_publish
#PBS -j oe
#PBS -m bea
#PBS -l storage=scratch/jh2+gdata/jh2+scratch/mk27

set -euo pipefail

# ========= Fill paths =========
PROJECT_ROOT="${PROJECT_ROOT:-/g/data/jh2/jt4478/make_LC_copy}"
VENV="${VENV:-}"
# Export this securely in your shell before qsub:
#   export PYPI_API_TOKEN='pypi-...'
PYPI_API_TOKEN="${PYPI_API_TOKEN:-}"
# ==============================

if [[ -z "${PYPI_API_TOKEN}" ]]; then
  echo "PYPI_API_TOKEN is empty. Export it before submitting this job." >&2
  exit 1
fi

module load python3/3.11.0
[[ -n "${VENV}" && -f "${VENV}" ]] && source "${VENV}"

cd "${PROJECT_ROOT}"
python -m pip install --upgrade build twine
rm -rf dist build
python -m build

export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="${PYPI_API_TOKEN}"
python -m twine upload --skip-existing dist/*

