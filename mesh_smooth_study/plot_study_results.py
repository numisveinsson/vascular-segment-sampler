#!/usr/bin/env python3
"""Plot metrics from mesh_smooth_study CSV (standalone; no VTK required)."""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mesh_smooth_study.plots import generate_study_plots


def main():
    p = argparse.ArgumentParser(description="Plot mesh smoothing study CSV results.")
    p.add_argument("--csv", required=True, help="Path to CSV from run_study.py")
    p.add_argument("--out_dir", required=True, help="Directory for figure files")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    args = p.parse_args()
    paths = generate_study_plots(args.csv, args.out_dir, dpi=args.dpi, fig_format=args.format)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
