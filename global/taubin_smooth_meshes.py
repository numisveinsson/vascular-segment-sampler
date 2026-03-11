"""
Apply Taubin (windowed sinc) smoothing to .vtp mesh files in a directory.
"""

import sys
import os

# Add project root to path so "from modules import ..." works
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules import vtk_functions as vf


def taubin_smooth_mesh(poly, iteration=25, boundary=False, feature=False, smoothing_factor=0.0):
    """
    Apply Taubin smoothing to a VTK PolyData mesh.
    Uses vtkWindowedSincPolyDataFilter (implements Taubin's λ/μ algorithm).
    """
    return vf.smooth_polydata(
        poly,
        iteration=iteration,
        boundary=boundary,
        feature=feature,
        smoothingFactor=smoothing_factor,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply Taubin smoothing to .vtp mesh files in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python taubin_smooth_meshes.py --meshes_dir /path/to/meshes --out_dir /path/to/output
  python taubin_smooth_meshes.py --meshes_dir ./data/surfaces/ --out_dir ./data/surfaces_smoothed/ --iterations 50
        """,
    )
    parser.add_argument(
        "--meshes_dir",
        "--meshes-dir",
        type=str,
        required=True,
        help="Directory containing .vtp mesh files",
    )
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for smoothed meshes",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Number of smoothing iterations (default: 25)",
    )
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.0,
        help="Smoothing factor affecting pass band (default: 0.0)",
    )
    parser.add_argument(
        "--boundary",
        action="store_true",
        default=False,
        help="Enable boundary smoothing",
    )
    parser.add_argument(
        "--feature",
        action="store_true",
        default=False,
        help="Enable feature edge smoothing",
    )

    args = parser.parse_args()

    meshes_dir = args.meshes_dir
    out_dir = args.out_dir

    if not os.path.exists(meshes_dir):
        raise ValueError(f"Meshes directory not found: {meshes_dir}")

    os.makedirs(out_dir, exist_ok=True)

    from modules.logger import get_logger

    logger = get_logger(__name__)

    mesh_files = [f for f in os.listdir(meshes_dir) if f.endswith(".vtp")]
    mesh_files.sort()

    if not mesh_files:
        logger.error(f"No .vtp files found in {meshes_dir}")
        sys.exit(1)

    logger.info(f"Processing {len(mesh_files)} mesh files from {meshes_dir}")

    for mesh_file in mesh_files:
        input_path = os.path.join(meshes_dir, mesh_file)
        output_path = os.path.join(out_dir, mesh_file)

        try:
            reader = vf.read_geo(input_path)
            poly = reader.GetOutput()

            smoothed = taubin_smooth_mesh(
                poly,
                iteration=args.iterations,
                boundary=args.boundary,
                feature=args.feature,
                smoothing_factor=args.smoothing_factor,
            )

            vf.write_geo(output_path, smoothed)
            logger.info(f"Done: {mesh_file} -> {output_path}")
        except Exception as e:
            logger.error(f"Failed to process {mesh_file}: {e}")
            raise

    logger.info("All done.")
