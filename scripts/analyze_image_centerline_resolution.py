"""Analyze image resolution vs. centerline radii for a dataset.

Given a directory that contains an ``images`` subfolder (``.mha``) and a
``centerlines`` subfolder (``.vtp``), where matching cases share the same
file stem in both subfolders, this script computes, per case:

  * in-plane and out-of-plane image resolution (spacing)
  * radius statistics from the centerline ``MaximumInscribedSphereRadius``
    point-data array (min / max / mean / median / std / percentiles)
  * the ratio between the minimum centerline radius and the maximum image
    spacing value (a "resolvability" indicator -- values below 1 mean the
    thinnest vessel is smaller than the coarsest voxel dimension)

Results are written to a CSV and a set of summary plots.

Example
-------
    python scripts/analyze_image_centerline_resolution.py \
        --data-dir /path/to/dataset \
        --output-dir /path/to/dataset/resolution_analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

try:
    import SimpleITK as sitk
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "SimpleITK is required. Install with `pip install SimpleITK`."
    ) from exc

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy as v2n
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'vtk' package is required. Install with `pip install vtk`."
    ) from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RADIUS_ARRAY_NAME = "MaximumInscribedSphereRadius"


@dataclass
class CaseResult:
    """Per-case metrics."""

    name: str
    spacing_x: float
    spacing_y: float
    spacing_z: float
    in_plane_spacing: float  # coarser of the two in-plane (axial) dimensions
    out_of_plane_spacing: float  # slice thickness (through-plane)
    max_spacing: float
    min_spacing: float
    n_radii: int
    radius_min: float
    radius_max: float
    radius_mean: float
    radius_median: float
    radius_std: float
    radius_p05: float
    radius_p95: float
    min_radius_to_max_spacing_ratio: float


def read_centerline_radii(vtp_path: str, array_name: str = RADIUS_ARRAY_NAME) -> np.ndarray:
    """Read the radius point-data array from a ``.vtp`` centerline file."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    poly = reader.GetOutput()

    point_data = poly.GetPointData()
    arr = point_data.GetArray(array_name)
    if arr is None:
        available = [
            point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())
        ]
        raise KeyError(
            f"Array '{array_name}' not found in {vtp_path}. "
            f"Available point-data arrays: {available}"
        )
    radii = v2n(arr).astype(float).ravel()
    radii = radii[np.isfinite(radii)]
    return radii


def get_image_spacing(mha_path: str) -> np.ndarray:
    """Return the (x, y, z) voxel spacing of an image without loading pixels."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(mha_path)
    reader.ReadImageInformation()
    return np.array(reader.GetSpacing(), dtype=float)


def analyze_case(name: str, img_path: str, cent_path: str, array_name: str) -> CaseResult:
    spacing = get_image_spacing(img_path)
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])

    # In-plane = the two axial dimensions (x, y); out-of-plane = slice thickness (z).
    in_plane = max(sx, sy)
    out_of_plane = sz
    max_spacing = float(np.max(spacing))
    min_spacing = float(np.min(spacing))

    radii = read_centerline_radii(cent_path, array_name)
    if radii.size == 0:
        raise ValueError(f"No finite radii found in {cent_path}")

    r_min = float(np.min(radii))
    ratio = r_min / max_spacing if max_spacing > 0 else float("nan")

    return CaseResult(
        name=name,
        spacing_x=sx,
        spacing_y=sy,
        spacing_z=sz,
        in_plane_spacing=in_plane,
        out_of_plane_spacing=out_of_plane,
        max_spacing=max_spacing,
        min_spacing=min_spacing,
        n_radii=int(radii.size),
        radius_min=r_min,
        radius_max=float(np.max(radii)),
        radius_mean=float(np.mean(radii)),
        radius_median=float(np.median(radii)),
        radius_std=float(np.std(radii)),
        radius_p05=float(np.percentile(radii, 5)),
        radius_p95=float(np.percentile(radii, 95)),
        min_radius_to_max_spacing_ratio=float(ratio),
    )


def match_cases(
    images_dir: str,
    centerlines_dir: str,
    image_ext: str,
    centerline_ext: str,
) -> List[tuple]:
    """Match images and centerlines that share the same file stem."""
    images = {
        os.path.splitext(f)[0]: os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith(image_ext)
    }
    centerlines = {
        os.path.splitext(f)[0]: os.path.join(centerlines_dir, f)
        for f in os.listdir(centerlines_dir)
        if f.endswith(centerline_ext)
    }

    common = sorted(set(images) & set(centerlines))
    missing_cent = sorted(set(images) - set(centerlines))
    missing_img = sorted(set(centerlines) - set(images))

    if missing_cent:
        print(f"[warn] {len(missing_cent)} image(s) without centerline: {missing_cent}")
    if missing_img:
        print(f"[warn] {len(missing_img)} centerline(s) without image: {missing_img}")

    return [(name, images[name], centerlines[name]) for name in common]


def write_csv(results: List[CaseResult], csv_path: str) -> None:
    import csv

    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"[ok] wrote {csv_path}")


def make_plots(
    results: List[CaseResult],
    radii_by_case: Dict[str, np.ndarray],
    out_dir: str,
    dpi: int = 1000,
) -> None:
    names = [r.name for r in results]
    x = np.arange(len(names))
    in_plane = np.array([r.in_plane_spacing for r in results])
    out_plane = np.array([r.out_of_plane_spacing for r in results])
    max_spacing = np.array([r.max_spacing for r in results])
    r_min = np.array([r.radius_min for r in results])
    r_mean = np.array([r.radius_mean for r in results])
    r_std = np.array([r.radius_std for r in results])
    ratio = np.array([r.min_radius_to_max_spacing_ratio for r in results])

    # Tick label rotation/size depends on case count.
    rot = 90 if len(names) > 15 else 45
    fsz = max(4, min(8, 400 / max(1, len(names))))

    def _set_xticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=rot, ha="right", fontsize=fsz)

    # 1) In-plane vs out-of-plane resolution per case (grouped bars).
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(names)), 5))
    w = 0.4
    ax.bar(x - w / 2, in_plane, w, label="in-plane (max of x,y)", color="#4c72b0")
    ax.bar(x + w / 2, out_plane, w, label="out-of-plane (z)", color="#dd8452")
    ax.set_ylabel("spacing")
    ax.set_title("Image resolution: in-plane vs out-of-plane")
    ax.legend()
    _set_xticks(ax)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "resolution_in_vs_out_of_plane.png"), dpi=dpi)
    plt.close(fig)

    # 1b) In-plane vs out-of-plane scatter (anisotropy view).
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(in_plane, out_plane, c="#4c72b0", alpha=0.7, edgecolors="k", linewidths=0.3)
    lim = max(in_plane.max(), out_plane.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="isotropic (y=x)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("in-plane spacing (max of x,y)")
    ax.set_ylabel("out-of-plane spacing (z)")
    ax.set_title("Image anisotropy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "resolution_anisotropy_scatter.png"), dpi=dpi)
    plt.close(fig)

    # 2a) Radii statistics: boxplot of full radius distribution per case.
    data = [radii_by_case[n] for n in names]
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(names)), 5))
    ax.boxplot(data, showfliers=False)
    ax.set_ylabel(f"{RADIUS_ARRAY_NAME}")
    ax.set_title("Centerline radius distribution per case")
    ax.set_xticks(x + 1)
    ax.set_xticklabels(names, rotation=rot, ha="right", fontsize=fsz)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radii_boxplot.png"), dpi=dpi)
    plt.close(fig)

    # 2b) Radii statistics: mean +/- std with min/max markers.
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(names)), 5))
    ax.errorbar(x, r_mean, yerr=r_std, fmt="o", color="#4c72b0", label="mean +/- std", capsize=3)
    ax.scatter(x, [r.radius_min for r in results], marker="v", color="#55a868", label="min")
    ax.scatter(x, [r.radius_max for r in results], marker="^", color="#c44e52", label="max")
    ax.set_ylabel("radius")
    ax.set_title("Centerline radius statistics per case")
    ax.legend()
    _set_xticks(ax)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radii_statistics.png"), dpi=dpi)
    plt.close(fig)

    # 3) Ratio of min radius to max spacing per case.
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(names)), 5))
    colors = ["#c44e52" if v < 1 else "#55a868" for v in ratio]
    ax.bar(x, ratio, color=colors)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.7, label="ratio = 1 (radius = voxel)")
    ax.set_ylabel("min radius / max spacing")
    ax.set_title("Resolvability: min centerline radius vs max image spacing")
    ax.legend()
    _set_xticks(ax)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "min_radius_to_max_spacing_ratio.png"), dpi=dpi)
    plt.close(fig)

    # 3b) Min radius vs max spacing scatter.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(max_spacing, r_min, c="#4c72b0", alpha=0.7, edgecolors="k", linewidths=0.3)
    lim = max(max_spacing.max(), r_min.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="min radius = max spacing")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("max image spacing")
    ax.set_ylabel("min centerline radius")
    ax.set_title("Min radius vs max spacing")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "min_radius_vs_max_spacing_scatter.png"), dpi=dpi)
    plt.close(fig)

    print(f"[ok] wrote plots to {out_dir}")


def print_summary(results: List[CaseResult]) -> None:
    if not results:
        print("No cases analyzed.")
        return

    def col(attr):
        return np.array([getattr(r, attr) for r in results], dtype=float)

    print("\n===== Dataset summary ({} cases) =====".format(len(results)))
    for label, attr in [
        ("in-plane spacing", "in_plane_spacing"),
        ("out-of-plane spacing", "out_of_plane_spacing"),
        ("max spacing", "max_spacing"),
        ("radius min", "radius_min"),
        ("radius mean", "radius_mean"),
        ("min radius / max spacing", "min_radius_to_max_spacing_ratio"),
    ]:
        vals = col(attr)
        print(
            f"  {label:28s}: mean={np.mean(vals):.4f}  "
            f"min={np.min(vals):.4f}  max={np.max(vals):.4f}"
        )

    under = [r.name for r in results if r.min_radius_to_max_spacing_ratio < 1.0]
    if under:
        print(
            f"\n  [!] {len(under)} case(s) with min radius < max spacing "
            f"(under-resolved): {under}"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Directory containing images/ and centerlines/ subfolders")
    p.add_argument("--images-subdir", default="images", help="Name of the images subfolder (default: images)")
    p.add_argument("--centerlines-subdir", default="centerlines", help="Name of the centerlines subfolder (default: centerlines)")
    p.add_argument("--image-ext", default=".mha", help="Image file extension (default: .mha)")
    p.add_argument("--centerline-ext", default=".vtp", help="Centerline file extension (default: .vtp)")
    p.add_argument("--radius-array", default=RADIUS_ARRAY_NAME, help=f"Radius point-data array name (default: {RADIUS_ARRAY_NAME})")
    p.add_argument("--output-dir", default=None, help="Where to write CSV and plots (default: <data-dir>/resolution_analysis)")
    p.add_argument("--dpi", type=int, default=1000, help="Resolution (DPI) of saved plots (default: 1000)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    images_dir = os.path.join(args.data_dir, args.images_subdir)
    centerlines_dir = os.path.join(args.data_dir, args.centerlines_subdir)
    for d in (images_dir, centerlines_dir):
        if not os.path.isdir(d):
            print(f"[error] not a directory: {d}", file=sys.stderr)
            return 1

    out_dir = args.output_dir or os.path.join(args.data_dir, "resolution_analysis")
    os.makedirs(out_dir, exist_ok=True)

    cases = match_cases(images_dir, centerlines_dir, args.image_ext, args.centerline_ext)
    if not cases:
        print("[error] no matching image/centerline pairs found", file=sys.stderr)
        return 1
    print(f"[info] found {len(cases)} matching case(s)")

    results: List[CaseResult] = []
    radii_by_case: Dict[str, np.ndarray] = {}
    for name, img_path, cent_path in cases:
        try:
            res = analyze_case(name, img_path, cent_path, args.radius_array)
            results.append(res)
            radii_by_case[name] = read_centerline_radii(cent_path, args.radius_array)
            print(
                f"  {name}: spacing=({res.spacing_x:.3f},{res.spacing_y:.3f},{res.spacing_z:.3f}) "
                f"r_min={res.radius_min:.3f} ratio={res.min_radius_to_max_spacing_ratio:.3f}"
            )
        except Exception as exc:  # keep going on individual failures
            print(f"  [skip] {name}: {exc}", file=sys.stderr)

    if not results:
        print("[error] no cases analyzed successfully", file=sys.stderr)
        return 1

    write_csv(results, os.path.join(out_dir, "resolution_radii_metrics.csv"))
    make_plots(results, radii_by_case, out_dir, dpi=args.dpi)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
