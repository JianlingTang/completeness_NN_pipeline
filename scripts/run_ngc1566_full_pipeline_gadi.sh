#!/usr/bin/env bash
# Full Phase A + B on Gadi (default photometry ON → completeness diagnostics + catalogue counts).
# Params: reff=5, nframe=5, ncl=400, galaxy=ngc1566 (override with env).
#
# Prerequisites (login node once):
#   python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .
#
# PBS: use SKIP_PIP_INSTALL=1 (no PyPI on compute). Request enough mem/time; photometry uses IRAF.
#
#   export SKIP_PIP_INSTALL=1
#   export IRAF=/path/to/your/iraf
#   qsub your_job.pbs   # job body: bash scripts/run_ngc1566_full_pipeline_gadi.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${PROJECT_ROOT:=/scratch/jh2/jt4478/comp_pipeline_restructure}"
: "${COMP_SLUG_LIB_DIR:=/g/data/jh2/jt4478/cluster_slug}"
: "${COMP_PSF_PATH:=/g/data/jh2/jt4478/PSF_all}"
: "${COMP_BAO_PATH:=/g/data/jh2/jt4478/baolab-0.94.1g}"
: "${COMP_FITS_PATH:=${PROJECT_ROOT}}"
: "${COMP_MAIN_DIR:=${PROJECT_ROOT}}"
: "${GALAXY:=ngc1566}"
: "${OUTNAME:=run_reff5_nframe5_ncl400}"
: "${NFRAME:=5}"
: "${NCL:=400}"
: "${REFF_LIST:=5}"
: "${SKIP_PIP_INSTALL:=1}"
: "${RUN_THREE_PANEL_PLOTS:=1}"

export PROJECT_ROOT
export COMP_SLUG_LIB_DIR COMP_PSF_PATH COMP_BAO_PATH COMP_FITS_PATH COMP_MAIN_DIR
export IRAF="${IRAF:-}"

cd "$PROJECT_ROOT"

if [[ "$SKIP_PIP_INSTALL" == "1" ]]; then
  if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    echo "ERROR: .venv missing. Build on login: python3 -m venv .venv && pip install -e ." >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    python3 -m venv "$PROJECT_ROOT/.venv"
  fi
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.venv/bin/activate"
  pip install -q -U pip
  pip install -q -e .
fi

# Photometry loads PyRAF/IRAF inside cluster_pipeline/photometry/aperture_photometry.py at runtime.
# No pyraf preflight here — ensure IRAF is set in the job environment if needed (e.g. export IRAF=/path/to/iraf).
if [[ -z "${IRAF:-}" ]]; then
  echo "WARNING: IRAF is unset; aperture photometry may fail until you export IRAF (see docs/INSTALL_IRAF.md)." >&2
fi

echo "==> run_pipeline: galaxy=$GALAXY outname=$OUTNAME nframe=$NFRAME ncl=$NCL reff_list=$REFF_LIST"
echo "==> forcing photometry ON + post-CI labels only (no white-match fallback)"

python scripts/run_pipeline.py \
  --galaxy "$GALAXY" \
  --nframe "$NFRAME" \
  --ncl "$NCL" \
  --reff_list "$REFF_LIST" \
  --outname "$OUTNAME" \
  --parallel

if [[ "$RUN_THREE_PANEL_PLOTS" == "1" ]]; then
  echo "==> plot_three_panel_white_synthetic_recovered.py for frames 0..$((NFRAME - 1)) ..."
  for ((f = 0; f < NFRAME; f++)); do
    python scripts/plot_three_panel_white_synthetic_recovered.py \
      --galaxy "$GALAXY" \
      --outname "$OUTNAME" \
      --reff "$(echo "$REFF_LIST" | cut -d, -f1)" \
      --frame "$f" \
      || echo "WARN: three-panel plot failed for frame $f (optional)" >&2
  done
fi

echo "OK: full pipeline finished. Diagnostics under $PROJECT_ROOT/$GALAXY/white/diagnostics/"
