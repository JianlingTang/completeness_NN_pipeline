#!/usr/bin/env bash
# Quick smoke test for IRAF + PyRAF on the login node (same venv logic as PBS + run_ngc1566_full_pipeline_gadi.sh).
#
# Usage (Gadi):
#   export IRAF=/g/data/jh2/jt4478/iraf-2.17   # or your IRAF root
#   bash scripts/test_photometry_login_node.sh
#
# Optional env (match your PBS job):
#   PROJECT_ROOT=/scratch/jh2/jt4478/comp_pipeline_restructure
#   SKIP_PIP_INSTALL=1   # default 1: use existing .venv only (no pip on compute)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SKIP_PIP_INSTALL:=1}"

export PROJECT_ROOT
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

export IRAF="${IRAF:-}"
if [[ -z "${IRAF:-}" ]]; then
  echo "WARNING: IRAF is unset; aperture photometry may fail until you export IRAF (see docs/INSTALL_IRAF.md)." >&2
fi

exec python scripts/test_photometry_iraf_smoke.py "$@"
