"""
Iteratively process a surface mesh along its centerline, passing localized
submeshes through a model for vertex updates.
"""

import argparse

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk

from modules import vtk_functions as vf
from modules.sampling_functions import (
    sort_centerline,
    find_next_point,
    get_longest_centerline,
    sort_centerline_by_length,
    flip_radius,
)


def _remove_isolated_vertices(vertices, faces, edge_index):
    """
    Remove vertices not used in faces or edges; reindex faces and edge_index.
    Returns (new_vertices, new_faces, new_edge_index, used_indices) where
    used_indices are the original vertex indices that were kept.
    """
    used = set()
    for i in range(faces.shape[0]):
        for j in range(3):
            used.add(int(faces[i, j]))
    for i in range(edge_index.shape[1]):
        used.add(int(edge_index[0, i]))
        used.add(int(edge_index[1, i]))
    used = sorted(used)
    old_to_new = {old: i for i, old in enumerate(used)}
    new_vertices = vertices[torch.tensor(used, dtype=torch.long)]
    new_faces = torch.tensor(
        [[old_to_new[int(v)] for v in f] for f in faces.tolist()],
        dtype=torch.long,
    )
    new_edge_index = torch.tensor(
        [[old_to_new[int(e[0])], old_to_new[int(e[1])]] for e in edge_index.t().tolist()],
        dtype=torch.long,
    ).t()
    return new_vertices, new_faces, new_edge_index, used


def _get_points_cells_polydata(polydata):
    """Get points and cells from polydata as numpy arrays."""
    points = v2n(polydata.GetPoints().GetData())
    cells = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        cell_points = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        cells.append(cell_points)
    return points, np.array(cells)


def extract_mesh_inside_box(polydata, center, radius, box_size_factor=2.0):
    """
    Extract vertices, faces, and edges of the surface mesh that lie inside
    an axis-aligned bounding box centered at `center` with half-extent
    `box_size_factor * radius` in each dimension.

    Returns a dict with (same structure as _extract_mesh_info_from_vtp):
        - vertices: (N, 3) torch.float32
        - faces: (M, 3) torch.long
        - edge_index: (2, E) torch.long
        - global_vertex_ids: (N,) numpy, for mapping back to global mesh
    """
    # --- Triangulate if needed ---
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(polydata)
    triangulator.Update()
    poly = triangulator.GetOutput()

    points, cells = _get_points_cells_polydata(poly)

    # Define bounding box (voi_min, voi_max)
    radius = float(radius)
    half_extent = box_size_factor * radius
    voi_min = np.array(center) - half_extent
    voi_max = np.array(center) + half_extent

    # Vertices inside box
    inside_mask = np.all((points >= voi_min) & (points <= voi_max), axis=1)
    global_vertex_ids = np.where(inside_mask)[0]

    if len(global_vertex_ids) == 0:
        return None

    # Map global -> local vertex index
    global_to_local = {g: i for i, g in enumerate(global_vertex_ids)}

    # Faces where ALL 3 vertices are inside (fully contained triangles)
    local_faces = []
    for cell in cells:
        if len(cell) != 3:
            continue
        if all(v in global_to_local for v in cell):
            local_faces.append([global_to_local[v] for v in cell])

    if len(local_faces) == 0:
        return None

    local_faces = np.array(local_faces, dtype=np.int64)
    vertices = points[global_vertex_ids].astype(np.float32)

    # Build edges from faces (undirected)
    edges = set()
    for a, b, c in local_faces:
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(a, c), max(a, c)))
    edge_index = np.array(list(edges), dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)

    # Convert to torch and remove isolated vertices (same format as _extract_mesh_info_from_vtp)
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(local_faces, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    new_vertices, new_faces, new_edge_index, used = _remove_isolated_vertices(
        vertices, faces, edge_index
    )
    new_global_vertex_ids = global_vertex_ids[used]

    return {
        "vertices": new_vertices,
        "faces": new_faces,
        "global_vertex_ids": new_global_vertex_ids,
        "edge_index": new_edge_index,
    }


def update_global_vertices(global_vertices, local_vertices, global_vertex_ids):
    """
    Update global vertex coordinates with values from a local submesh.

    Args:
        global_vertices: (N_total, 3) mutable array of all vertex coordinates
        local_vertices: (K, 3) updated coordinates for the local subset
        global_vertex_ids: (K,) global indices for each local vertex
    """
    global_vertices[global_vertex_ids] = local_vertices


def process_mesh_along_centerline(
    surface_path,
    centerline_path,
    model_forward,
    output_path,
    box_size_factor=2.0,
    radius_add=0.0,
    radius_scale=1.0,
    move_dist=2.0,
    verbose=True,
):
    """
    Iterate along the centerline and process localized submeshes through
    the model, updating the global mesh vertices at each step.

    Args:
        surface_path: Path to surface mesh .vtp
        centerline_path: Path to centerline .vtp
        model_forward: Callable(vertices, edge_index, faces) -> updated_vertices
            Accepts numpy arrays; may return numpy or torch (will be converted).
        output_path: Path to write the updated surface mesh
        box_size_factor: Half-extent of bounding box = box_size_factor * radius
        radius_add: Added to radii (as in main.py)
        radius_scale: Scale applied to radii (as in main.py)
        move_dist: Step along centerline in units of radius
        verbose: Print progress
    """
    # Load surface and centerline
    surface = vf.read_geo(surface_path).GetOutput()
    centerline = vf.read_geo(centerline_path).GetOutput()

    # Sort centerline (same as main.py)
    (_, c_loc, radii, cent_ids, bifurc_id, num_cent) = sort_centerline(centerline)
    radii = radii + radius_add
    radii = radii * radius_scale
    cent_ids = flip_radius(cent_ids, radii)

    # Start from longest centerline
    ip_longest = get_longest_centerline(cent_ids, c_loc)
    ips_sorted = sort_centerline_by_length(cent_ids, c_loc)

    # Global vertex array (we'll update in-place)
    points = surface.GetPoints()
    n_total = points.GetNumberOfPoints()
    global_vertices = np.array([points.GetPoint(i) for i in range(n_total)], dtype=np.float32)

    config = {
        "MOVE_DIST": move_dist,
        "MOVE_SLOWER_LARGE": 1.0,
        "MOVE_SLOWER_BIFURC": 1.0,
    }

    n_processed = 0
    for ip in ips_sorted:
        ids = cent_ids[ip]
        if len(ids) == 0:
            continue

        locs = c_loc[ids]
        rads = radii[ids]
        bifurc = bifurc_id[ids]

        on_cent = True
        count = 0

        while on_cent:
            center = locs[count]
            radius = rads[count]

            # Extract submesh inside bounding box
            inside = extract_mesh_inside_box(
                surface, center, radius, box_size_factor=box_size_factor
            )

            if inside is not None and inside["vertices"].shape[0] > 0:
                vertices = inside["vertices"]
                edge_index = inside["edge_index"]
                faces = inside["faces"]
                global_vertex_ids = inside["global_vertex_ids"]

                # Call model
                updated_vertices = model_forward(vertices, edge_index, faces)

                # Convert to numpy if needed (e.g. from torch)
                if hasattr(updated_vertices, "detach"):
                    updated_vertices = updated_vertices.detach().cpu().numpy()
                updated_vertices = np.asarray(updated_vertices, dtype=np.float32)

                if updated_vertices.shape[0] == vertices.shape[0]:
                    update_global_vertices(
                        global_vertices, updated_vertices, global_vertex_ids
                    )
                    # Update surface polydata so next extraction uses current geometry
                    vtk_array = numpy_to_vtk(global_vertices, deep=True)
                    surface.GetPoints().SetData(vtk_array)
                    surface.Modified()
                    n_processed += 1

                    if verbose:
                        print(
                            f"  Point {count}: {vertices.shape[0]} vertices, "
                            f"radius={radius:.3f}"
                        )

            count, on_cent = find_next_point(count, locs, rads, bifurc, config, on_cent)

    # --- Write predicted global mesh after iterating ---
    vtk_array = numpy_to_vtk(global_vertices, deep=True)
    surface.GetPoints().SetData(vtk_array)
    surface.Modified()
    vf.write_geo(output_path, surface)

    if verbose:
        print(f"Processed {n_processed} submesh regions. Predicted mesh saved to {output_path}")


# --- Legacy / helper functions (from original loop_over_mesh) ---


def _extract_mesh_info_from_vtp(self, filepath: str):
    # --- Read mesh ---
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    polydata = reader.GetOutput()

    # --- Triangulate (IMPORTANT) ---
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(polydata)
    triangulator.Update()
    polydata = triangulator.GetOutput()

    # --- Vertices ---
    points = polydata.GetPoints()
    n_points = points.GetNumberOfPoints()
    vertices = np.array([points.GetPoint(i) for i in range(n_points)])
    vertices = torch.tensor(vertices, dtype=torch.float32)

    # --- Faces (triangles) ---
    faces = []
    polys = polydata.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() != 3:
            continue  # safety (should not happen after triangulation)
        faces.append([
            id_list.GetId(0),
            id_list.GetId(1),
            id_list.GetId(2),
        ])
    faces = torch.tensor(faces, dtype=torch.long)

    # --- Edges (from cells, SAFE) ---
    edges = set()
    polys = polydata.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    while polys.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        for i in range(n):
            u = id_list.GetId(i)
            v = id_list.GetId((i + 1) % n)
            edges.add((u, v))
            edges.add((v, u))

    # --- Convert to tensor ---
    edge_index = torch.tensor(list(edges), dtype=torch.long).t()
    new_vertices, new_faces, new_edge_index, _ = _remove_isolated_vertices(
        vertices, faces, edge_index
    )

    return new_vertices, new_edge_index, new_faces


def update_vtp_vertices(vertices, input_path, output_path):
    # --- Load the VTP mesh ---
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(input_path))
    reader.Update()
    polydata = reader.GetOutput()

    # --- Convert torch tensor to numpy if necessary ---
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()

    vertices = np.array(vertices, dtype=np.float32)
    n_points = polydata.GetNumberOfPoints()

    # --- Check size consistency ---
    if vertices.shape[0] != n_points:
        raise ValueError(
            f"Number of vertices mismatch: polydata has {n_points}, "
            f"new_vertices has {vertices.shape[0]}"
        )

    # --- Update the points ---
    vtk_array = numpy_to_vtk(vertices, deep=True)
    polydata.GetPoints().SetData(vtk_array)
    polydata.Modified()  # Important!

    # --- Write the updated mesh ---
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(polydata)
    writer.Write()

    print(f"Updated mesh saved to: {output_path}")


def model_forward(vertices, edge_index, faces):
    """
    Placeholder model forward pass. Replace with your actual model.
    Receives same structure as _extract_mesh_info_from_vtp outputs.

    Args:
        vertices: (N, 3) torch.float32
        edge_index: (2, E) torch.long
        faces: (M, 3) torch.long

    Returns:
        updated_vertices: (N, 3) same shape as vertices (torch or numpy)
    """
    return vertices


def main():
    # Example:
    #   python global/loop_over_mesh.py -s surface.vtp -c centerline.vtp -o output.vtp
    #   python global/loop_over_mesh.py -s surface.vtp -c centerline.vtp -o output.vtp --box_size_factor 2.5 --move_dist 2.0
    parser = argparse.ArgumentParser(
        description="Process a surface mesh along its centerline, passing "
        "localized submeshes through a model for vertex updates."
    )
    parser.add_argument(
        "-s",
        "--surface",
        required=True,
        type=str,
        help="Path to surface mesh .vtp",
    )
    parser.add_argument(
        "-c",
        "--centerline",
        required=True,
        type=str,
        help="Path to centerline .vtp",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to write the predicted surface mesh",
    )
    parser.add_argument(
        "--box_size_factor",
        type=float,
        default=2.0,
        help="Half-extent of bounding box = box_size_factor * radius (default: 2.0)",
    )
    parser.add_argument(
        "--radius_add",
        type=float,
        default=0.0,
        help="Add to radii (default: 0.0)",
    )
    parser.add_argument(
        "--radius_scale",
        type=float,
        default=1.0,
        help="Scale applied to radii (default: 1.0)",
    )
    parser.add_argument(
        "--move_dist",
        type=float,
        default=2.0,
        help="Step along centerline in units of radius (default: 2.0)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    process_mesh_along_centerline(
        surface_path=args.surface,
        centerline_path=args.centerline,
        model_forward=model_forward,
        output_path=args.output,
        box_size_factor=args.box_size_factor,
        radius_add=args.radius_add,
        radius_scale=args.radius_scale,
        move_dist=args.move_dist,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
