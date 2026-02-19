"""
Test for loop_over_mesh.py: iterative mesh processing along centerline
with Laplacian smoothing as model_forward.

Run with your own data:
  python tests/test_loop_over_mesh.py -s surface.vtp -c centerline.vtp -o output.vtp
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _laplacian_smoothing(vertices, edge_index, faces, lambda_factor=0.5):
    """
    Simple Laplacian smoothing: each vertex moves toward the centroid of its neighbors.
    new_pos = (1 - lambda) * old_pos + lambda * mean(neighbor_positions)
    edge_index is undirected: (2, E), each column is one edge (u, v).
    """
    n_vertices = vertices.shape[0]
    device = vertices.device

    # For each vertex, sum neighbor positions and count neighbors
    neighbor_sum = torch.zeros_like(vertices)
    neighbor_count = torch.zeros(n_vertices, 1, device=device, dtype=vertices.dtype)

    # Process both directions (undirected edges)
    for i in range(2):
        src = edge_index[i]
        dst = edge_index[1 - i]
        neighbor_sum.scatter_add_(
            0, src.unsqueeze(-1).expand(-1, 3), vertices[dst]
        )
        neighbor_count.scatter_add_(
            0, src.unsqueeze(-1),
            torch.ones(src.shape[0], 1, device=device, dtype=vertices.dtype),
        )

    # Avoid div by zero for isolated vertices
    neighbor_count = neighbor_count.clamp(min=1)
    neighbor_mean = neighbor_sum / neighbor_count

    # Blend: new = (1 - lambda) * old + lambda * neighbor_mean
    smoothed = (1 - lambda_factor) * vertices + lambda_factor * neighbor_mean
    return smoothed


def _load_loop_module():
    """Load process_mesh_along_centerline (cannot use 'from global.X' - global is a keyword)."""
    loop_path = project_root / "global" / "loop_over_mesh.py"
    spec = importlib.util.spec_from_file_location("loop_over_mesh", loop_path)
    loop_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop_mod)
    return loop_mod.process_mesh_along_centerline


def run_laplacian_loop(surface_path, centerline_path, output_path, **kwargs):
    """Run process_mesh_along_centerline with Laplacian smoothing."""
    def model_forward(vertices, edge_index, faces):
        return _laplacian_smoothing(vertices, edge_index, faces, lambda_factor=0.3)

    process_mesh_along_centerline = _load_loop_module()
    process_mesh_along_centerline(
        surface_path=surface_path,
        centerline_path=centerline_path,
        model_forward=model_forward,
        output_path=output_path,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run loop_over_mesh with Laplacian smoothing on surface + centerline."
    )
    parser.add_argument("-s", "--surface", required=True, help="Path to surface mesh .vtp")
    parser.add_argument("-c", "--centerline", required=True, help="Path to centerline .vtp")
    parser.add_argument("-o", "--output", required=True, help="Path to output .vtp")
    parser.add_argument("--box_size_factor", type=float, default=2.0)
    parser.add_argument("--move_dist", type=float, default=2.0)
    args = parser.parse_args()

    run_laplacian_loop(
        args.surface,
        args.centerline,
        args.output,
        box_size_factor=args.box_size_factor,
        move_dist=args.move_dist,
        verbose=True,
    )
