#!/usr/bin/env bash
# Run comp_pipeline_restructure with slugpy (read_cluster from SLUG2).
# Uses this project's .venv (has pyraf for 5-filter photometry). Set IRAF for full photometry.
# Requires: SLUG_DIR so slugpy finds lib/filters; PYTHONPATH so Python finds slugpy + cluster_pipeline.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLUG2=/Users/janett/Documents/cluster_demographics/slug2
# Use comp_pipeline_restructure's venv (has pyraf); fallback to cluster_demographics venv
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  VENV_PYTHON="$ROOT/.venv/bin/python"
else
  VENV_PYTHON=/Users/janett/Documents/cluster_demographics/.venv/bin/python
fi

export SLUG_DIR="$SLUG2"
# slug2 first (slugpy), then project root (cluster_pipeline)
export PYTHONPATH="$SLUG2:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# IRAF for aperture photometry (optional; if unset, photometry is skipped)
export IRAF="${IRAF:-/usr/local/lib/iraf}"

cd "$ROOT"
"$VENV_PYTHON" scripts/run_pipeline.py "$@"
