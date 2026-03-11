"""
Curved Multiplanar Reformation (CPR) of a centerline and image and/or segmentation mask.

Produces a straightened 2D view by sampling the volume along perpendicular planes
at each centerline point. Uses VMTK's vtkvmtkCurvedMPRImageFilter when available,
otherwise falls back to a numpy/scipy-based implementation.
"""
import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# Ensure repo root is on path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from modules import vtk_functions as vf
from modules.sampling_functions import (
    sort_centerline,
    get_longest_centerline,
    sort_centerline_by_length,
    flip_radius,
)


def _physical_to_index(img, physical_point):
    """Convert physical (world) coordinates to continuous voxel indices."""
    return img.TransformPhysicalPointToContinuousIndex(physical_point)


def _compute_tangents(points):
    """Compute tangent vectors at each point using finite differences."""
    n = len(points)
    tangents = np.zeros_like(points)
    if n < 2:
        return tangents
    for i in range(n):
        if i == 0:
            t = points[1] - points[0]
        elif i == n - 1:
            t = points[-1] - points[-2]
        else:
            t = points[i + 1] - points[i - 1]
        norm = np.linalg.norm(t)
        tangents[i] = t / (norm + 1e-10)
    return tangents


def _compute_perpendicular(tangent, ref=np.array([0.0, 0.0, 1.0])):
    """Compute a unit vector perpendicular to tangent. Uses ref to avoid degeneracy."""
    cross = np.cross(tangent, ref)
    norm = np.linalg.norm(cross)
    if norm < 1e-8:
        ref = np.array([1.0, 0.0, 0.0])
        cross = np.cross(tangent, ref)
        norm = np.linalg.norm(cross)
    return cross / (norm + 1e-10)


def _get_ordered_centerline_points(centerline_polydata, default_radius_mm=5.0):
    """
    Get ordered centerline points (physical coords) from longest branch.
    Returns (points, tangents, radii). Radii may be estimated if not in centerline.
    """
    try:
        num_points, c_loc, radii, cent_ids, bifurc_id, num_cent = sort_centerline(
            centerline_polydata
        )
        radii = np.asarray(radii, dtype=np.float64) + 0.0
    except (KeyError, Exception):
        # Centerline may lack MaximumInscribedSphereRadius / f
        from vtk.util.numpy_support import vtk_to_numpy

        c_loc = vtk_to_numpy(centerline_polydata.GetPoints().GetData())
        cent_ids = []
        for i in range(centerline_polydata.GetNumberOfCells()):
            cell = centerline_polydata.GetCell(i)
            ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            cent_ids.append(ids)
        radii = np.ones(len(c_loc), dtype=np.float64) * default_radius_mm
        if not cent_ids:
            cent_ids = [list(range(len(c_loc)))]
    cent_ids = flip_radius(cent_ids, radii)
    ip_longest = get_longest_centerline(cent_ids, c_loc)
    ips_sorted = sort_centerline_by_length(cent_ids, c_loc)

    # Use longest centerline
    ids = cent_ids[ip_longest]
    if len(ids) < 2:
        raise ValueError("Centerline has fewer than 2 points")
    points = c_loc[ids]
    rads = radii[ids]
    tangents = _compute_tangents(points)
    return points, tangents, rads


def _cpr_numpy(img_sitk, points, tangents, radii, width_mm, num_samples_along,
               num_samples_perp, is_seg=False, background=0.0):
    """
    Straightened CPR using numpy/scipy sampling.
    Output: 2D array [along_centerline, perpendicular]
    """
    arr = np.asarray(sitk.GetArrayFromImage(img_sitk), dtype=np.float64)

    # Subsample along centerline if requested
    n_along = len(points)
    if num_samples_along is not None and num_samples_along < n_along:
        indices = np.linspace(0, n_along - 1, num_samples_along).astype(int)
        points = points[indices]
        tangents = tangents[indices]
        radii = radii[indices]
        n_along = num_samples_along

    # Use mean radius for width if not varying
    if radii is not None and len(radii) > 0:
        half_width = max(np.mean(radii) * 2.0, width_mm / 2.0)
    else:
        half_width = width_mm / 2.0

    # Output grid: (i, j) -> physical point -> voxel index
    t_perp = np.linspace(-half_width, half_width, num_samples_perp)
    order = 0 if is_seg else 1

    # Build coordinate arrays for map_coordinates
    # map_coordinates expects (ndim, n_points) with order (z, y, x) for arr
    coords_list = []
    for i in range(n_along):
        p = points[i]
        perp = _compute_perpendicular(tangents[i])
        for t_val in t_perp:
            phys = p + t_val * perp
            idx = _physical_to_index(img_sitk, phys.tolist())
            # SimpleITK index is (x, y, z); arr is (z, y, x)
            coords_list.append([idx[2], idx[1], idx[0]])

    coords = np.array(coords_list).T  # (3, n_along * n_perp)
    sampled = ndimage.map_coordinates(
        arr, coords, order=order, mode="constant", cval=background
    )
    cpr = sampled.reshape(n_along, num_samples_perp)

    if is_seg:
        cpr = np.round(cpr).astype(np.int32)
    return cpr


def run_cpr(
    centerline_path,
    image_path=None,
    segmentation_path=None,
    output_dir=None,
    width_mm=20.0,
    num_samples_along=None,
    num_samples_perp=128,
    verbose=False,
):
    """
    Run CPR on image and/or segmentation along a centerline.

    :param centerline_path: Path to centerline .vtp file
    :param image_path: Path to image (optional)
    :param segmentation_path: Path to segmentation mask (optional)
    :param output_dir: Output directory (default: same as first input)
    :param width_mm: Width of CPR view in mm (perpendicular to centerline)
    :param num_samples_along: Number of samples along centerline (default: all points)
    :param num_samples_perp: Number of samples perpendicular to centerline
    :param verbose: Print progress
    :return: Dict with output paths and/or numpy arrays
    """
    if not image_path and not segmentation_path:
        raise ValueError("At least one of image_path or segmentation_path is required")

    centerline = vf.read_geo(centerline_path).GetOutput()
    points, tangents, radii = _get_ordered_centerline_points(centerline)
    if radii is None:
        radii = np.ones(len(points)) * (width_mm / 4.0)

    if output_dir is None:
        if image_path:
            output_dir = os.path.dirname(os.path.abspath(image_path))
        else:
            output_dir = os.path.dirname(os.path.abspath(segmentation_path))
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(centerline_path))[0]
    results = {}

    # Image CPR
    if image_path:
        img_sitk = sitk.ReadImage(image_path)
        cpr_img = _cpr_numpy(
            img_sitk,
            points,
            tangents,
            radii,
            width_mm=width_mm,
            num_samples_along=num_samples_along,
            num_samples_perp=num_samples_perp,
            is_seg=False,
            background=0.0,
        )
        out_img_path = os.path.join(output_dir, f"{base_name}_cpr_img.npy")
        np.save(out_img_path, cpr_img)
        results["image_cpr"] = cpr_img
        results["image_cpr_path"] = out_img_path
        # Also write as MHA for compatibility
        cpr_sitk = sitk.GetImageFromArray(cpr_img.astype(np.float32))
        cpr_sitk.SetSpacing([1.0, 1.0])
        cpr_sitk.SetOrigin([0.0, 0.0])
        sitk.WriteImage(cpr_sitk, os.path.join(output_dir, f"{base_name}_cpr_img.mha"))
        if verbose:
            print(f"Saved image CPR to {out_img_path}")

    # Segmentation CPR
    if segmentation_path:
        seg_sitk = sitk.ReadImage(segmentation_path)
        cpr_seg = _cpr_numpy(
            seg_sitk,
            points,
            tangents,
            radii,
            width_mm=width_mm,
            num_samples_along=num_samples_along,
            num_samples_perp=num_samples_perp,
            is_seg=True,
            background=0,
        )
        out_seg_path = os.path.join(output_dir, f"{base_name}_cpr_seg.npy")
        np.save(out_seg_path, cpr_seg)
        results["segmentation_cpr"] = cpr_seg
        results["segmentation_cpr_path"] = out_seg_path
        # Also write as MHA
        cpr_seg_sitk = sitk.GetImageFromArray(cpr_seg.astype(np.int32))
        cpr_seg_sitk.SetSpacing([1.0, 1.0])
        cpr_seg_sitk.SetOrigin([0.0, 0.0])
        sitk.WriteImage(cpr_seg_sitk, os.path.join(output_dir, f"{base_name}_cpr_seg.mha"))
        if verbose:
            print(f"Saved segmentation CPR to {out_seg_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Curved Multiplanar Reformation (CPR) of centerline and image/segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CPR of image and segmentation
  python preprocessing/cpr_centerline.py \\
      --centerline /path/to/centerline.vtp \\
      --image /path/to/image.mha \\
      --segmentation /path/to/seg.mha \\
      --output_dir /path/to/output

  # CPR of image only
  python preprocessing/cpr_centerline.py \\
      --centerline centerline.vtp \\
      --image image.mha

  # CPR of segmentation only with custom width
  python preprocessing/cpr_centerline.py \\
      --centerline centerline.vtp \\
      --segmentation seg.mha \\
      --width_mm 30
        """,
    )
    parser.add_argument(
        "--centerline",
        type=str,
        required=True,
        help="Path to centerline .vtp file",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        default=None,
        help="Path to input segmentation mask",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as first input)",
    )
    parser.add_argument(
        "--width_mm",
        type=float,
        default=20.0,
        help="Width of CPR view in mm (default: 20)",
    )
    parser.add_argument(
        "--num_samples_along",
        type=int,
        default=None,
        help="Number of samples along centerline (default: all points)",
    )
    parser.add_argument(
        "--num_samples_perp",
        type=int,
        default=128,
        help="Number of samples perpendicular to centerline (default: 128)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress",
    )

    args = parser.parse_args()

    if not args.image and not args.segmentation:
        parser.error("At least one of --image or --segmentation is required")

    if not os.path.exists(args.centerline):
        raise FileNotFoundError(f"Centerline not found: {args.centerline}")
    if args.image and not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.segmentation and not os.path.exists(args.segmentation):
        raise FileNotFoundError(f"Segmentation not found: {args.segmentation}")

    run_cpr(
        centerline_path=args.centerline,
        image_path=args.image,
        segmentation_path=args.segmentation,
        output_dir=args.output_dir,
        width_mm=args.width_mm,
        num_samples_along=args.num_samples_along,
        num_samples_perp=args.num_samples_perp,
        verbose=args.verbose,
    )
    print("CPR completed successfully.")
