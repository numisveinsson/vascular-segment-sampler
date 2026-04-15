"""Surface and volume metrics: Dice (optionally by MIS radius bin), ASSD, HD95, normal angular error."""

from __future__ import annotations

import math
import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from modules import vtk_functions as vf


def _vtk_mutable_int():
    if hasattr(vtk, "mutable"):
        return vtk.mutable(0)
    return vtk.reference(0)


def _vtk_mutable_double():
    if hasattr(vtk, "mutable"):
        return vtk.mutable(0.0)
    return vtk.reference(0.0)


def _cell_locator_for_surface(poly: vtk.vtkPolyData) -> vtk.vtkCellLocator:
    """Triangle surface + vtkCellLocator (avoids vtkImplicitPolyDataDistance segfaults on open meshes)."""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    surf = tri.GetOutput()
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(surf)
    if hasattr(loc, "SetNumberOfCellsPerBucket"):
        loc.SetNumberOfCellsPerBucket(25)
    if hasattr(loc, "AutomaticOn"):
        loc.AutomaticOn()
    loc.BuildLocator()
    return loc


def _closest_distances_to_surface(locator: vtk.vtkCellLocator, points: np.ndarray) -> np.ndarray:
    """Per-row Euclidean distance from point to closest point on located surface."""
    n = points.shape[0]
    out = np.empty(n, dtype=np.float64)
    closest = [0.0, 0.0, 0.0]
    cell = vtk.vtkGenericCell()
    cell_id = _vtk_mutable_int()
    sub_id = _vtk_mutable_int()
    dist2 = _vtk_mutable_double()
    for i in range(n):
        q = points[i]
        locator.FindClosestPoint(
            [float(q[0]), float(q[1]), float(q[2])],
            closest,
            cell,
            cell_id,
            sub_id,
            dist2,
        )
        out[i] = math.sqrt(max(0.0, dist2.get()))
    return out


def _poly_normals(poly):
    n = vtk.vtkPolyDataNormals()
    n.SetInputData(poly)
    # VTK 9.3+ uses Set*(bool); older builds use *On() / *Off().
    if hasattr(n, "SetComputePointNormals"):
        n.SetComputePointNormals(True)
    else:
        n.SetComputePointNormalsOn()
    if hasattr(n, "SetComputeCellNormals"):
        n.SetComputeCellNormals(True)
    else:
        n.SetComputeCellNormalsOn()
    if hasattr(n, "SetSplitting"):
        n.SetSplitting(False)
    else:
        n.SplittingOff()
    if hasattr(n, "SetConsistency"):
        n.SetConsistency(True)
    else:
        n.ConsistencyOn()
    if hasattr(n, "SetAutoOrientNormals"):
        n.SetAutoOrientNormals(True)
    else:
        n.AutoOrientNormalsOn()
    n.Update()
    return n.GetOutput()


def combined_bounds(poly_a, poly_b, margin_mm: float):
    b1 = np.array(poly_a.GetBounds(), dtype=np.float64)
    b2 = np.array(poly_b.GetBounds(), dtype=np.float64)
    lo = np.minimum(b1[[0, 2, 4]], b2[[0, 2, 4]]) - margin_mm
    hi = np.maximum(b1[[1, 3, 5]], b2[[1, 3, 5]]) + margin_mm
    return lo, hi


def make_reference_image(
    lo,
    hi,
    spacing_mm: float,
    max_dim: int | None = None,
):
    """
    VTK image covering [lo, hi] with isotropic spacing.

    If ``max_dim`` is set (>= 2), spacing is increased as needed so each axis
    has at most ``max_dim`` samples (caps memory for Dice rasterization).

    Returns
    -------
    im : vtk.vtkImageData
    effective_spacing : float
        Isotropic spacing actually used (>= ``spacing_mm`` when capped).
    """
    extent = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    extent = np.maximum(extent, 1e-12)
    spacing = float(spacing_mm)
    if max_dim is not None and int(max_dim) >= 2:
        md = int(max_dim)
        denom = float(max(md - 1, 1))
        s_floor = float(np.max(extent / denom))
        spacing = max(spacing, s_floor)
    dims = np.maximum(np.ceil(extent / spacing).astype(np.int32) + 1, 2)
    if max_dim is not None and int(max_dim) >= 2 and np.any(dims > int(max_dim)):
        md = int(max_dim)
        spacing = float(spacing * (float(np.max(dims)) / md) * 1.000001)
        dims = np.maximum(np.ceil(extent / spacing).astype(np.int32) + 1, 2)
    sp3 = (spacing, spacing, spacing)
    im = vtk.vtkImageData()
    im.SetOrigin(float(lo[0]), float(lo[1]), float(lo[2]))
    im.SetSpacing(sp3[0], sp3[1], sp3[2])
    im.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    im.AllocateScalars(vtk.VTK_INT, 1)
    scal = im.GetPointData().GetScalars()
    scal.FillComponent(0, 0.0)
    return im, spacing


def poly_to_binary_volume(poly, ref_im):
    """Rasterize closed surface into reference image; returns bool array (X,Y,Z) Fortran order."""
    ref_im.GetPointData().SetScalars(
        n2v(np.zeros(v2n(ref_im.GetPointData().GetScalars()).shape, dtype=np.int32))
    )
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.01)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing())
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    out = stencil.GetOutput()
    arr = v2n(out.GetPointData().GetScalars()).astype(np.int32)
    dims = ref_im.GetDimensions()
    vol = arr.reshape(dims[0], dims[1], dims[2], order="F") > 0
    return vol


def dice_binary(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt.astype(bool)
    p = pred.astype(bool)
    inter = np.count_nonzero(g & p)
    denom = np.count_nonzero(g) + np.count_nonzero(p)
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def _voxel_centers(ref_im: vtk.vtkImageData) -> np.ndarray:
    ox, oy, oz = ref_im.GetOrigin()
    sx, sy, sz = ref_im.GetSpacing()
    nx, ny, nz = ref_im.GetDimensions()
    xs = ox + sx * (np.arange(nx) + 0.5)
    ys = oy + sy * (np.arange(ny) + 0.5)
    zs = oz + sz * (np.arange(nz) + 0.5)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")], axis=1)


def global_quantile_bin_edges(pooled_radii: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Bin edges (length n_bins+1) from quantiles of pooled MIS radii — same edges for every case
    when used as bin_edges_mm in radius_bins_from_centerline.
    """
    r = np.asarray(pooled_radii, dtype=np.float64).ravel()
    if r.size == 0 or int(n_bins) <= 0:
        raise ValueError("global_quantile_bin_edges needs non-empty radii and n_bins > 0")
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(r, qs).astype(np.float64)
    for j in range(1, len(edges)):
        if edges[j] <= edges[j - 1]:
            edges[j] = edges[j - 1] + 1e-9
    return edges


def radius_bins_from_centerline(
    ref_im: vtk.vtkImageData,
    foreground_mask: np.ndarray,
    cl_points: np.ndarray,
    cl_radii: np.ndarray,
    n_bins: int,
    bin_edges_mm: np.ndarray | None = None,
):
    """
    Assign each foreground voxel a bin index 0..K-1 by nearest centerline MIS radius.

    If bin_edges_mm is None, uses quantiles of **this case's** cl_radii for K = n_bins bins.
    If bin_edges_mm is set (e.g. from global_quantile_bin_edges), uses those shared edges.
    Returns (bin_index_per_voxel_flat, bin_edges, bin_labels) where flat matches Fortran ravel.
    """
    if cl_points.size == 0 or cl_radii.size == 0 or n_bins <= 0:
        return None, None, []

    r = np.asarray(cl_radii, dtype=np.float64).ravel()
    if bin_edges_mm is None:
        qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
        edges = np.quantile(r, qs).astype(np.float64)
        for j in range(1, len(edges)):
            if edges[j] <= edges[j - 1]:
                edges[j] = edges[j - 1] + 1e-9
    else:
        edges = np.asarray(bin_edges_mm, dtype=np.float64)

    n_bin = len(edges) - 1
    labels = [
        f"r in [{edges[i]:.4g}, {edges[i + 1]:.4g})" for i in range(n_bin)
    ]

    tree = cKDTree(np.asarray(cl_points, dtype=np.float64))
    flat_mask = foreground_mask.ravel(order="F")
    idx = np.flatnonzero(flat_mask)
    if idx.size == 0:
        bin_flat = np.full(flat_mask.shape[0], -1, dtype=np.int32)
        return bin_flat, edges, labels

    centers = _voxel_centers(ref_im)
    pts = centers[idx]
    _, nn = tree.query(pts, k=1)
    rad = r[np.clip(nn, 0, len(r) - 1)]
    bins = np.searchsorted(edges, rad, side="right") - 1
    bins = np.clip(bins, 0, n_bin - 1)

    bin_flat = np.full(flat_mask.shape[0], -1, dtype=np.int32)
    bin_flat[idx] = bins.astype(np.int32)
    return bin_flat, edges, labels


def dice_per_radius_bin(
    gt_vol: np.ndarray,
    pred_vol: np.ndarray,
    bin_flat: np.ndarray,
    n_bins: int,
    labels: list,
):
    """Dice restricted to voxels whose bin index matches (stratified by vessel caliber)."""
    g = gt_vol.ravel(order="F")
    p = pred_vol.ravel(order="F")
    b = bin_flat
    rows = []
    for bi in range(int(n_bins)):
        m = b == bi
        if not np.any(m):
            rows.append(
                {
                    "radius_bin": bi,
                    "radius_bin_label": labels[bi] if bi < len(labels) else str(bi),
                    "dice": float("nan"),
                    "n_voxels": 0,
                }
            )
            continue
        dice = dice_binary(g[m] > 0, p[m] > 0)
        rows.append(
            {
                "radius_bin": bi,
                "radius_bin_label": labels[bi] if bi < len(labels) else str(bi),
                "dice": dice,
                "n_voxels": int(np.count_nonzero(m)),
            }
        )
    return rows


def _sample_triangle_surface(vertices: np.ndarray, faces: np.ndarray, n_samples: int, rng: np.random.Generator):
    """Area-weighted random points on triangle mesh. faces: (n,3) int vertex indices."""
    if faces.size == 0 or n_samples <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    verts = np.asarray(vertices, dtype=np.float64)
    row_ok = np.isfinite(verts).all(axis=1)
    if not np.any(row_ok):
        return np.zeros((0, 3), dtype=np.float64)
    face_ok = row_ok[faces[:, 0]] & row_ok[faces[:, 1]] & row_ok[faces[:, 2]]
    faces = faces[face_ok]
    if faces.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Center + isotropic scale so cross-products stay in float64 range (large world coords,
    # corrupted faces with huge edges, etc.). Relative triangle areas are unchanged.
    center = np.mean(verts[row_ok], axis=0)
    v0c = verts - center
    vids = np.unique(faces)
    amax = float(np.max(np.abs(v0c[vids])))
    if not np.isfinite(amax) or amax <= 0:
        scale = 1.0
    else:
        scale = amax
    V = v0c / scale

    v0 = V[faces[:, 0]]
    v1 = V[faces[:, 1]]
    v2 = V[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cp = np.cross(e1, e2, axis=1)
    cross = np.sqrt(np.maximum(np.sum(cp * cp, axis=1), 0.0))
    area = 0.5 * cross
    valid = np.isfinite(area) & (area > 1e-30)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    faces = faces[valid]
    area = area[valid]
    total = float(np.sum(area))
    if not np.isfinite(total) or total <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    cdf = np.cumsum(area / total)
    r = rng.random(n_samples)
    tri_idx = np.searchsorted(cdf, r)
    tri_idx = np.clip(tri_idx, 0, len(faces) - 1)

    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    sr1 = np.sqrt(r1)
    u = 1.0 - sr1
    v = sr1 * r2
    w = 1.0 - u - v
    f = faces[tri_idx]
    p_scaled = (
        u[:, None] * V[f[:, 0]]
        + v[:, None] * V[f[:, 1]]
        + w[:, None] * V[f[:, 2]]
    )
    p = p_scaled * scale + center
    return p.astype(np.float64)


def sample_surface_points(poly, n_samples: int, seed: int = 0):
    verts, cells = vf.get_points_cells_pd(poly)
    faces = np.array([c for c in cells if len(c) == 3], dtype=np.int64)
    if faces.size == 0:
        return np.asarray(verts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return _sample_triangle_surface(np.asarray(verts, dtype=np.float64), faces, n_samples, rng)


def directed_distances(points: np.ndarray, target_poly: vtk.vtkPolyData):
    """Euclidean distance from each point to closest point on target surface."""
    if points.shape[0] == 0:
        return np.array([], dtype=np.float64)
    loc = _cell_locator_for_surface(target_poly)
    return _closest_distances_to_surface(loc, points)


def assd_and_hd95(poly_a: vtk.vtkPolyData, poly_b: vtk.vtkPolyData, n_samples: int, seed: int):
    """Symmetric ASSD and HD95 in mesh length units from surface sampling."""
    pa = sample_surface_points(poly_a, n_samples, seed)
    pb = sample_surface_points(poly_b, n_samples, seed + 1)
    dab = directed_distances(pa, poly_b)
    dba = directed_distances(pb, poly_a)
    if dab.size == 0 or dba.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    assd = float(0.5 * (dab.mean() + dba.mean()))
    hd95 = float(
        0.5
        * (
            np.percentile(dab, 95)
            + np.percentile(dba, 95)
        )
    )
    return assd, hd95, float(dab.mean()), float(dba.mean())


def normal_angle_errors_deg(poly_pred: vtk.vtkPolyData, poly_gt: vtk.vtkPolyData, n_samples: int, seed: int):
    """
    Sample the predicted surface; GT normal at the nearest GT mesh vertex (cKDTree).
    Compares to nearest vertex normal on the predicted mesh. Stable on open / non-manifold
    surfaces (avoids vtkImplicitPolyDataDistance, which can segfault).
    """
    gt_n = _poly_normals(poly_gt)
    pr_n = _poly_normals(poly_pred)

    gt_pts = np.asarray(v2n(gt_n.GetPoints().GetData()), dtype=np.float64)
    gt_norm = gt_n.GetPointData().GetNormals()
    if gt_norm is None or gt_pts.shape[0] == 0:
        return float("nan"), float("nan")
    gt_norm = np.asarray(v2n(gt_norm), dtype=np.float64)
    tree_gt = cKDTree(gt_pts)

    pts = sample_surface_points(pr_n, n_samples, seed)
    pnorm = pr_n.GetPointData().GetNormals()
    if pnorm is None or pts.shape[0] == 0:
        return float("nan"), float("nan")
    pnorm = np.asarray(v2n(pnorm), dtype=np.float64)
    pred_verts = np.asarray(v2n(pr_n.GetPoints().GetData()), dtype=np.float64)
    n_pv = pred_verts.shape[0]
    if pnorm.shape[0] != n_pv:
        m = min(n_pv, pnorm.shape[0])
        pred_verts = pred_verts[:m]
        pnorm = pnorm[:m]
        n_pv = m
    if n_pv == 0:
        return float("nan"), float("nan")
    pred_tree = cKDTree(pred_verts)
    _, vid_p = pred_tree.query(pts, k=1)
    vid_p = np.asarray(vid_p, dtype=np.int64).reshape(-1)
    vid_p = np.clip(vid_p, 0, n_pv - 1)
    n_pred = pnorm[vid_p]

    _, vid_g = tree_gt.query(pts, k=1)
    vid_g = np.asarray(vid_g, dtype=np.int64).reshape(-1)
    vid_g = np.clip(vid_g, 0, gt_norm.shape[0] - 1)
    n_gt = gt_norm[vid_g]

    angles = []
    for i in range(pts.shape[0]):
        a = n_pred[i]
        b = n_gt[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            continue
        c = np.clip(float(np.dot(a, b) / (na * nb)), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(c)))

    if not angles:
        return float("nan"), float("nan")
    ang = np.array(angles, dtype=np.float64)
    return float(np.mean(ang)), float(np.max(ang))
