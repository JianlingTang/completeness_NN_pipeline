#!/bin/bash
# Run full pipeline (injection + detection + matching + photometry + catalogue)
# with IRAF available for aperture photometry. Clusters are sampled directly from SLUG
# (no user-supplied coords). Requires IRAF to be installed.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default IRAF path (Apple Silicon pkg installer puts it here)
export IRAF="${IRAF:-/usr/local/lib/iraf}"
if [[ ! -d "$IRAF" ]]; then
  echo "IRAF not found at $IRAF"
  echo ""
  echo "Install IRAF once (requires your password):"
  echo "  1. Remove quarantine:  xattr -c $ROOT/.deps/iraf_download/iraf-2.18.1-1-arm64.pkg"
  echo "  2. Open installer:      open $ROOT/.deps/iraf_download/iraf-2.18.1-1-arm64.pkg"
  echo "  3. Click through and enter your password when prompted."
  echo ""
  echo "Then run this script again."
  exit 1
fi

echo "Using IRAF at $IRAF"
PYTHON="${ROOT}/.venv/bin/python"
NCL=500
NFRAME=2

# IRAF phot prompts for coords/output; feed newlines so it accepts defaults (non-interactive)
# 500 clusters × 2 frames = 1000 prompts
( for _ in $(seq 1 $((NCL * NFRAME))); do echo; done ) | "$PYTHON" scripts/run_small_test.py --ncl "$NCL" --nframe "$NFRAME" --run_photometry
