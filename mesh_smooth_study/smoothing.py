"""Smoothing backends: Laplacian (VTK), Taubin windowed-sinc, Taubin cotangent Laplacian."""

import vtk

from modules import vtk_functions as vf


def triangle_mesh(poly):
    """Ensure triangle cells for cotangent / Laplacian pipelines."""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    return tri.GetOutput()


def laplacian_smooth(poly, iterations=25, relaxation_factor=0.1, boundary_smoothing=True):
    """
    VTK Laplacian smoothing (vtkSmoothPolyDataFilter).

    Parameters
    ----------
    iterations : int
        Number of smoothing passes.
    relaxation_factor : float
        Each pass moves vertices toward neighbor centroid by this fraction (0–1).
    boundary_smoothing : bool
        If False, boundary edges are fixed.
    """
    poly = triangle_mesh(poly)
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputData(poly)
    smooth.SetNumberOfIterations(int(iterations))
    smooth.SetRelaxationFactor(float(relaxation_factor))
    smooth.SetBoundarySmoothing(bool(boundary_smoothing))
    smooth.FeatureEdgeSmoothingOff()
    smooth.Update()
    return smooth.GetOutput()


def taubin_windowed_sinc(
    poly,
    iteration=25,
    boundary=False,
    feature=False,
    smoothing_factor=0.0,
):
    """Taubin-style smoothing via vtkWindowedSincPolyDataFilter (same as global/taubin_smooth_meshes.py)."""
    poly = triangle_mesh(poly)
    return vf.smooth_polydata(
        poly,
        iteration=int(iteration),
        boundary=bool(boundary),
        feature=bool(feature),
        smoothingFactor=float(smoothing_factor),
    )


def taubin_cotangent(poly, it=50, mu1=0.5, mu2=0.51):
    """λ–μ Taubin with cotangent Laplacian (numpy; same as global/taubin_smooth_meshes_cotlaplacian.py)."""
    poly = triangle_mesh(poly)
    return vf.taubin_smooth_polydata(poly, it=int(it), mu1=float(mu1), mu2=float(mu2))


def apply_method(name, poly, params):
    """
    Apply smoothing method `name` with dict `params` (keys match CLI / grid).

    name: 'none' | 'laplacian' | 'taubin' | 'taubin_cot'
    """
    name = name.lower().strip()
    if name == "none":
        return triangle_mesh(poly)
    if name == "laplacian":
        return laplacian_smooth(
            poly,
            iterations=params.get("iterations", 25),
            relaxation_factor=params.get("relaxation_factor", 0.1),
            boundary_smoothing=params.get("boundary_smoothing", True),
        )
    if name == "taubin":
        return taubin_windowed_sinc(
            poly,
            iteration=params.get("iterations", params.get("iteration", 25)),
            boundary=params.get("boundary", False),
            feature=params.get("feature", False),
            smoothing_factor=params.get("smoothing_factor", 0.0),
        )
    if name in ("taubin_cot", "taubin_cotlaplacian"):
        return taubin_cotangent(
            poly,
            it=params.get("iterations", params.get("it", 50)),
            mu1=params.get("mu1", 0.5),
            mu2=params.get("mu2", 0.51),
        )
    raise ValueError(f"Unknown smoothing method: {name}")
