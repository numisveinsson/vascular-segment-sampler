#!/usr/bin/env python3
"""Summarize and plot mesh_smooth_study CSV (standalone; no VTK required).

Writes per-configuration figures and a `summary.csv` of mean/std/median metrics across cases.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mesh_smooth_study.plots import generate_study_plots, write_summary_csv


def main():
    p = argparse.ArgumentParser(
        description="Summarize and plot mesh smoothing study CSV results."
    )
    p.add_argument("--csv", required=True, help="Path to CSV from run_study.py")
    p.add_argument("--out_dir", required=True, help="Directory for figures and summary.csv")
    p.add_argument(
        "--summary_csv",
        default=None,
        help="Path for the aggregated summary CSV (default: <out_dir>/summary.csv)",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    p.add_argument(
        "--improve_over_none",
        action="store_true",
        help="Plot per-case improvement relative to the 'none' (no-smoothing) baseline "
        "instead of raw metrics (oriented so positive = better).",
    )
    args = p.parse_args()

    default_summary_name = (
        "summary_improve_over_none.csv" if args.improve_over_none else "summary.csv"
    )
    summary_csv = args.summary_csv or os.path.join(args.out_dir, default_summary_name)
    summary_path = write_summary_csv(
        args.csv, summary_csv, improve_over_none=args.improve_over_none
    )
    if summary_path is not None:
        print(summary_path)

    paths = generate_study_plots(
        args.csv,
        args.out_dir,
        dpi=args.dpi,
        fig_format=args.format,
        improve_over_none=args.improve_over_none,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
