#!/usr/bin/env bash
# Smoke test: venv, (optional) pip install -e ., path checks (+ optional LEGUS download), run_pipeline --check-only.
#
# Usage (Gadi login — has outbound internet for pip):
#   bash scripts/hpc_login_smoke_test.sh
#
# Gadi compute nodes usually have NO internet → pip cannot reach PyPI (errno 101).
# Prepare .venv on the login node first (run this script once without SKIP_PIP_INSTALL),
# then in PBS use:
#   export SKIP_PIP_INSTALL=1
#   bash scripts/hpc_login_smoke_test.sh
#
# Override any variable by exporting before the script, e.g.:
#   export GALAXY=ngc628-c CHECK_PHOTOMETRY=1
#   bash scripts/hpc_login_smoke_test.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============ Gadi / jt4478 — edit defaults here ============
: "${PROJECT_ROOT:=/scratch/jh2/jt4478/comp_pipeline_restructure}"
: "${COMP_SLUG_LIB_DIR:=/g/data/jh2/jt4478/cluster_slug}"
: "${COMP_PSF_PATH:=/g/data/jh2/jt4478/PSF_all}"
# Directory that contains the BAOlab executable `bl` (use .../bin if that is where `bl` lives)
: "${COMP_BAO_PATH:=/g/data/jh2/jt4478/baolab-0.94.1g}"
: "${COMP_FITS_PATH:=${PROJECT_ROOT}}"
: "${COMP_MAIN_DIR:=${PROJECT_ROOT}}"
: "${GALAXY:=ngc1566}"
# 0 = no photometry/IRAF checks. 1 = check_pipeline_paths --run-photometry + run_pipeline --check-only with photometry
: "${CHECK_PHOTOMETRY:=0}"
# 1 = skip "pip install" (use on PBS compute after venv was built on login)
: "${SKIP_PIP_INSTALL:=0}"
# ============================================================

export PROJECT_ROOT
export COMP_SLUG_LIB_DIR COMP_PSF_PATH COMP_BAO_PATH COMP_FITS_PATH COMP_MAIN_DIR
export GALAXY

cd "$PROJECT_ROOT"
echo "==> PROJECT_ROOT=$PROJECT_ROOT"
echo "==> COMP_FITS_PATH=$COMP_FITS_PATH  GALAXY=$GALAXY  CHECK_PHOTOMETRY=$CHECK_PHOTOMETRY"
echo ""

# Optional: Gadi Python module (uncomment)
# module purge
# module load python3/3.11.0

# macOS tarball junk — safe no-op on Linux
find . -maxdepth 4 -name '.DS_Store' -delete 2>/dev/null || true
find . -maxdepth 4 -name '._*' -delete 2>/dev/null || true

if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "ERROR: pyproject.toml not found under PROJECT_ROOT=$PROJECT_ROOT" >&2
  exit 1
fi

if [[ "$SKIP_PIP_INSTALL" == "1" ]]; then
  if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    echo "ERROR: SKIP_PIP_INSTALL=1 but .venv missing. On Gadi login run once without SKIP_PIP_INSTALL:" >&2
    echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e ." >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.venv/bin/activate"
  echo "==> python: $(command -v python) ($(python -V 2>&1)) [SKIP_PIP_INSTALL=1, no pip]"
else
  if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    echo "==> Creating .venv ..."
    python3 -m venv "$PROJECT_ROOT/.venv"
  fi
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.venv/bin/activate"
  echo "==> python: $(command -v python) ($(python -V 2>&1))"
  echo "==> pip install -e . ..."
  pip install -q -U pip
  pip install -q -e .
fi

echo ""
echo "==> check_pipeline_paths.py --galaxy $GALAXY --setup-if-missing ..."
if [[ "$CHECK_PHOTOMETRY" == "1" ]]; then
  python scripts/check_pipeline_paths.py \
    --main-dir "$COMP_MAIN_DIR" \
    --fits-path "$COMP_FITS_PATH" \
    --psf-path "$COMP_PSF_PATH" \
    --bao-path "$COMP_BAO_PATH" \
    --slug-lib-dir "$COMP_SLUG_LIB_DIR" \
    --galaxy "$GALAXY" \
    --setup-if-missing \
    --run-photometry
else
  python scripts/check_pipeline_paths.py \
    --main-dir "$COMP_MAIN_DIR" \
    --fits-path "$COMP_FITS_PATH" \
    --psf-path "$COMP_PSF_PATH" \
    --bao-path "$COMP_BAO_PATH" \
    --slug-lib-dir "$COMP_SLUG_LIB_DIR" \
    --galaxy "$GALAXY" \
    --setup-if-missing
fi

echo ""
echo "==> run_pipeline.py --check-only --galaxy $GALAXY ..."
if [[ "$CHECK_PHOTOMETRY" == "1" ]]; then
  python scripts/run_pipeline.py --galaxy "$GALAXY" --check-only
else
  python scripts/run_pipeline.py --galaxy "$GALAXY" --check-only --no_photometry
fi

echo ""
echo "OK: login smoke test finished. Next: qsub a real job or run run_pipeline without --check-only."
