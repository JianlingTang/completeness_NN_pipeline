#!/usr/bin/env python3
"""
Run pipeline stages 1–3 (injection, detection, matching) then plot
completeness vs magnitude diagnostics.

Usage:
  python scripts/run_stage123_and_plot_diagnostics.py [--galaxy GALAXY] [--outname NAME] [--save FIG.png]

Requires: numpy, matplotlib. Pipeline deps: cluster_pipeline (scipy, etc.).
"""
import argparse
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run stage 1–3 and plot completeness vs magnitude")
    parser.add_argument("--galaxy", type=str, default="ngc628-c_white-R17v100", help="Galaxy ID")
    parser.add_argument("--outname", type=str, default="pipeline", help="Pipeline run name")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path")
    parser.add_argument("--no-pipeline", action="store_true", help="Skip pipeline run; only plot from existing diagnostics")
    args = parser.parse_args()

    from cluster_pipeline.config import get_config
    from cluster_pipeline.pipeline import plot_completeness_diagnostics, run_galaxy_pipeline

    config = get_config()
    galaxy_id = args.galaxy
    outname = args.outname

    if not args.no_pipeline:
        print("Running pipeline max_stage=3 (injection, detection, matching)...")
        run_galaxy_pipeline(
            galaxy_id,
            config=config,
            outname=outname,
            max_stage=3,
            keep_frames=False,
        )
        print("Pipeline done. Writing diagnostics...")

    print("Plotting completeness vs magnitude...")
    ax = plot_completeness_diagnostics(galaxy_id, config, outname=outname)

    if args.save:
        ax.get_figure().savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved {args.save}")
    else:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            print("Set --save FIG.png to save the figure (no display).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
