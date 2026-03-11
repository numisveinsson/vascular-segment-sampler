"""
Apply Taubin (λ-μ) smoothing to .vtp mesh files in a directory.
Uses cotangent Laplacian via taubin_smooth_polydata (numpy-based, no torch).
"""

import sys
import os

# Add project root to path so "from modules import ..." works
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules import vtk_functions as vf


def taubin_smooth_mesh(poly, it=50, mu1=0.5, mu2=0.51):
    """
    Apply Taubin λ-μ smoothing to a VTK PolyData mesh.
    Uses cotangent Laplacian (alternates smoothing and inflation to reduce shrinkage).
    """
    return vf.taubin_smooth_polydata(poly, it=it, mu1=mu1, mu2=mu2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply Taubin (λ-μ) smoothing to .vtp mesh files using cotangent Laplacian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python taubin_smooth_meshes_cotlaplacian.py --meshes_dir /path/to/meshes --out_dir /path/to/output
  python taubin_smooth_meshes_cotlaplacian.py --meshes_dir ./data/surfaces/ --out_dir ./data/surfaces_smoothed/ --iterations 50 --mu1 0.5 --mu2 0.51
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
        default=50,
        help="Number of Taubin iterations (default: 50)",
    )
    parser.add_argument(
        "--mu1",
        type=float,
        default=0.5,
        help="Smoothing factor (default: 0.5)",
    )
    parser.add_argument(
        "--mu2",
        type=float,
        default=0.51,
        help="Inflation factor, typically slightly > mu1 to prevent shrinkage (default: 0.51)",
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

            if poly is None or poly.GetPoints() is None or poly.GetNumberOfPoints() == 0:
                logger.warning(f"Skipping {mesh_file}: mesh has no points (empty or corrupted)")
                continue

            smoothed = taubin_smooth_mesh(
                poly,
                it=args.iterations,
                mu1=args.mu1,
                mu2=args.mu2,
            )

            vf.write_geo(output_path, smoothed)
            logger.info(f"Done: {mesh_file} -> {output_path}")
        except Exception as e:
            logger.error(f"Failed to process {mesh_file}: {e}")
            raise

    logger.info("All done.")
