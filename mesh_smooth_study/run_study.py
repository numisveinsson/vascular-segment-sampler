#!/usr/bin/env python3
"""
Mesh smoothing study: sweep Laplacian / Taubin / Taubin–cotangent settings on MC surfaces
and compare to GT smooth surfaces (Dice, ASSD, HD95, normal angular error; Dice by MIS radius).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# Project root on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules import vtk_functions as vf
from modules.logger import get_logger

from mesh_smooth_study import metrics as M
from mesh_smooth_study import smoothing as S

logger = get_logger(__name__)

# Keep in sync with mesh_smooth_study/default_param_grid.json (or load via --param_grid_json).
DEFAULT_PARAM_GRID = {
    "none": [{}],
    "laplacian": [
        {"iterations": 10, "relaxation_factor": 0.05, "boundary_smoothing": True},
        {"iterations": 10, "relaxation_factor": 0.1, "boundary_smoothing": True},
        {"iterations": 15, "relaxation_factor": 0.1, "boundary_smoothing": True},
        {"iterations": 25, "relaxation_factor": 0.05, "boundary_smoothing": True},
        {"iterations": 25, "relaxation_factor": 0.1, "boundary_smoothing": True},
        {"iterations": 25, "relaxation_factor": 0.15, "boundary_smoothing": True},
        {"iterations": 25, "relaxation_factor": 0.2, "boundary_smoothing": True},
        {"iterations": 25, "relaxation_factor": 0.25, "boundary_smoothing": True},
        {"iterations": 40, "relaxation_factor": 0.1, "boundary_smoothing": True},
        {"iterations": 40, "relaxation_factor": 0.2, "boundary_smoothing": True},
        {"iterations": 50, "relaxation_factor": 0.1, "boundary_smoothing": True},
        {"iterations": 50, "relaxation_factor": 0.15, "boundary_smoothing": True},
        {"iterations": 75, "relaxation_factor": 0.1, "boundary_smoothing": True},
        {"iterations": 75, "relaxation_factor": 0.2, "boundary_smoothing": True},
        {"iterations": 100, "relaxation_factor": 0.1, "boundary_smoothing": True},
        {"iterations": 100, "relaxation_factor": 0.2, "boundary_smoothing": True},
        {"iterations": 25, "relaxation_factor": 0.1, "boundary_smoothing": False},
        {"iterations": 50, "relaxation_factor": 0.15, "boundary_smoothing": False},
        {"iterations": 75, "relaxation_factor": 0.1, "boundary_smoothing": False},
    ],
    "taubin": [
        {"iterations": 10, "boundary": False, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 15, "boundary": False, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 25, "boundary": False, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 35, "boundary": False, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 50, "boundary": False, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 75, "boundary": False, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 100, "boundary": False, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 25, "boundary": False, "feature": False, "smoothing_factor": 0.25},
        {"iterations": 25, "boundary": False, "feature": False, "smoothing_factor": 0.5},
        {"iterations": 25, "boundary": False, "feature": False, "smoothing_factor": 0.75},
        {"iterations": 50, "boundary": False, "feature": False, "smoothing_factor": 0.25},
        {"iterations": 50, "boundary": False, "feature": False, "smoothing_factor": 0.5},
        {"iterations": 50, "boundary": False, "feature": False, "smoothing_factor": 0.75},
        {"iterations": 50, "boundary": False, "feature": False, "smoothing_factor": 1.0},
        {"iterations": 75, "boundary": False, "feature": False, "smoothing_factor": 0.5},
        {"iterations": 100, "boundary": False, "feature": False, "smoothing_factor": 0.5},
        {"iterations": 25, "boundary": True, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 50, "boundary": True, "feature": False, "smoothing_factor": 0.0},
        {"iterations": 50, "boundary": True, "feature": False, "smoothing_factor": 0.5},
        {"iterations": 25, "boundary": False, "feature": True, "smoothing_factor": 0.0},
        {"iterations": 50, "boundary": False, "feature": True, "smoothing_factor": 0.0},
    ],
    "taubin_cot": [
        {"iterations": 20, "mu1": 0.5, "mu2": 0.51},
        {"iterations": 30, "mu1": 0.5, "mu2": 0.51},
        {"iterations": 50, "mu1": 0.5, "mu2": 0.51},
        {"iterations": 75, "mu1": 0.5, "mu2": 0.51},
        {"iterations": 100, "mu1": 0.5, "mu2": 0.51},
        {"iterations": 150, "mu1": 0.5, "mu2": 0.51},
        {"iterations": 50, "mu1": 0.45, "mu2": 0.455},
        {"iterations": 50, "mu1": 0.48, "mu2": 0.52},
        {"iterations": 50, "mu1": 0.52, "mu2": 0.53},
        {"iterations": 100, "mu1": 0.48, "mu2": 0.52},
        {"iterations": 100, "mu1": 0.5, "mu2": 0.505},
        {"iterations": 75, "mu1": 0.49, "mu2": 0.51},
        {"iterations": 100, "mu1": 0.45, "mu2": 0.48},
        {"iterations": 30, "mu1": 0.5, "mu2": 0.52},
        {"iterations": 50, "mu1": 0.5, "mu2": 0.52},
        {"iterations": 50, "mu1": 0.47, "mu2": 0.51},
        {"iterations": 100, "mu1": 0.47, "mu2": 0.51},
    ],
}


def _list_vtp(dir_path: str) -> list[str]:
    p = Path(dir_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Not a directory: {dir_path}")
    files = sorted(f.name for f in p.glob("*.vtp"))
    return files


def _load_centerline_radii(centerline_vtp: str, *, quiet: bool = False):
    """Return (N,3) points and (N,) MIS radii, or (pts, None) if radii missing."""
    from vtk.util.numpy_support import vtk_to_numpy as v2n

    poly = vf.read_geo(centerline_vtp).GetOutput()
    pts = np.asarray(v2n(poly.GetPoints().GetData()), dtype=np.float64)
    arrays = vf.collect_arrays(poly.GetPointData())
    if "MaximumInscribedSphereRadius" in arrays:
        r = np.asarray(arrays["MaximumInscribedSphereRadius"], dtype=np.float64).ravel()
        if r.shape[0] != pts.shape[0]:
            if not quiet:
                logger.warning(
                    "MIS radius length mismatch with points in %s; skipping radius strata",
                    centerline_vtp,
                )
            return pts, None
        return pts, r
    if not quiet:
        logger.warning("No MaximumInscribedSphereRadius in %s; Dice-by-radius disabled", centerline_vtp)
    return pts, None


def _pool_mis_radii_from_centerlines(common_vtp: list[str], centerline_dir: str) -> np.ndarray | None:
    """Concatenate MIS radii from all listed centerlines that exist and have valid arrays."""
    chunks: list[np.ndarray] = []
    for fname in common_vtp:
        path = os.path.join(centerline_dir, fname)
        if not os.path.isfile(path):
            continue
        _, r = _load_centerline_radii(path, quiet=True)
        if r is not None and r.size > 0:
            chunks.append(np.asarray(r, dtype=np.float64).ravel())
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _resolve_radius_bin_edges(
    radius_bin_edges_mm_arg: str | None,
    n_radius_bins: int,
    centerline_dir: str | None,
    per_case_radius_bins: bool,
    common_vtp: list[str],
) -> tuple[np.ndarray | None, str]:
    """
    Returns (edges_array_or_None, mode) where mode is fixed | global_quantile | per_case | off.
    """
    user_edges = _parse_bin_edges(radius_bin_edges_mm_arg)
    if user_edges is not None:
        return user_edges, "fixed"
    if n_radius_bins <= 0:
        return None, "off"
    if per_case_radius_bins:
        return None, "per_case"
    if not centerline_dir:
        return None, "off"
    pooled = _pool_mis_radii_from_centerlines(common_vtp, centerline_dir)
    if pooled is None or pooled.size == 0:
        logger.warning(
            "Could not pool MIS radii for global bins (missing centerlines or arrays); "
            "radius-stratum Dice disabled unless you set --radius_bin_edges_mm"
        )
        return None, "off"
    edges = M.global_quantile_bin_edges(pooled, n_radius_bins)
    logger.info(
        "Global radius bins: %d MIS samples from %s → edges (mesh length units) %s",
        pooled.size,
        centerline_dir,
        np.array2string(edges, precision=4, separator=", "),
    )
    return edges, "global_quantile"


def _parse_bin_edges(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) < 2:
        raise ValueError("--radius_bin_edges_mm needs at least two comma-separated edge values")
    return np.array(parts, dtype=np.float64)


def _params_sha10(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def _expected_config_keys(methods_and_params: dict[str, list[dict]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for method, param_list in methods_and_params.items():
        for params in param_list:
            keys.add((method, _params_sha10(params)))
    return keys


def _study_settings_hash(
    dice_spacing_mm: float,
    dice_max_dim: int | None,
    margin_mm: float,
    n_surface_samples: int,
    n_radius_bins: int,
    radius_bin_edges_mm: np.ndarray | None,
    radius_bin_resolution: str,
    seed: int,
) -> str:
    payload = {
        "dice_spacing_mm": dice_spacing_mm,
        "dice_max_dim": dice_max_dim,
        "margin_mm": margin_mm,
        "n_surface_samples": n_surface_samples,
        "n_radius_bins": n_radius_bins,
        "radius_bin_edges_mm": None if radius_bin_edges_mm is None else radius_bin_edges_mm.tolist(),
        "radius_bin_resolution": radius_bin_resolution,
        "seed": seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _case_already_complete(
    df: pd.DataFrame,
    case_id: str,
    expected_keys: set[tuple[str, str]],
    study_settings_hash: str,
) -> bool:
    if df.empty or "case_id" not in df.columns or "metric_scope" not in df.columns:
        return False
    if "study_settings_hash" not in df.columns:
        return False
    sub = df[(df["case_id"] == case_id) & (df["metric_scope"] == "overall")]
    sub = sub[sub["study_settings_hash"].astype(str) == study_settings_hash]
    if sub.empty:
        return False
    if "method" not in sub.columns or "params_sha10" not in sub.columns:
        return False
    done = set(zip(sub["method"].astype(str), sub["params_sha10"].astype(str)))
    return expected_keys.issubset(done)


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)


def run_case(
    case_stem: str,
    path_mc: str,
    path_gt: str,
    path_cl: str | None,
    methods_and_params: dict[str, list[dict]],
    dice_spacing_mm: float,
    dice_max_dim: int | None,
    margin_mm: float,
    n_surface_samples: int,
    n_radius_bins: int,
    radius_bin_edges_mm: np.ndarray | None,
    radius_bin_resolution: str,
    seed: int,
    save_smoothed_dir: str | None,
    study_settings_hash: str,
):
    gt_reader = vf.read_geo(path_gt)
    gt_poly = gt_reader.GetOutput()
    mc_reader = vf.read_geo(path_mc)
    mc_poly = mc_reader.GetOutput()

    cl_pts, cl_radii = (None, None)
    if path_cl and os.path.isfile(path_cl):
        cl_pts, cl_radii = _load_centerline_radii(path_cl)

    lo, hi = M.combined_bounds(gt_poly, mc_poly, margin_mm)
    ref_im, eff_sp = M.make_reference_image(lo, hi, dice_spacing_mm, dice_max_dim)
    if eff_sp > float(dice_spacing_mm) * 1.00001:
        logger.warning(
            "Case %s: Dice raster spacing raised from %s to %s (dice_max_dim=%s)",
            case_stem,
            dice_spacing_mm,
            eff_sp,
            dice_max_dim,
        )

    rows_out = []

    for method, param_list in methods_and_params.items():
        for params in param_list:
            try:
                smoothed = S.apply_method(method, mc_poly, params)
            except Exception as e:
                logger.exception("Smoothing failed case=%s method=%s params=%s: %s", case_stem, method, params, e)
                continue

            param_str = json.dumps(params, sort_keys=True)
            params_sha10 = _params_sha10(params)
            if save_smoothed_dir:
                os.makedirs(save_smoothed_dir, exist_ok=True)
                out_vtp = os.path.join(
                    save_smoothed_dir, f"{case_stem}__{method}__{params_sha10}.vtp"
                )
                vf.write_geo(out_vtp, smoothed)

            gt_vol = M.poly_to_binary_volume(gt_poly, ref_im)
            pr_vol = M.poly_to_binary_volume(smoothed, ref_im)
            dice_all = M.dice_binary(gt_vol, pr_vol)

            assd, hd95, m_ab, m_ba = M.assd_and_hd95(smoothed, gt_poly, n_surface_samples, seed)
            n_mean, n_max = M.normal_angle_errors_deg(smoothed, gt_poly, n_surface_samples, seed + 17)

            base_row = {
                "case_id": case_stem,
                "method": method,
                "params_json": param_str,
                "params_sha10": params_sha10,
                "study_settings_hash": study_settings_hash,
                "radius_bin_resolution": radius_bin_resolution,
                "radius_bin_edges_json": (
                    json.dumps(radius_bin_edges_mm.tolist()) if radius_bin_edges_mm is not None else ""
                ),
                "dice_spacing_mm": dice_spacing_mm,
                "dice_effective_spacing": eff_sp,
                "dice_max_dim": dice_max_dim,
                "dice_overall": dice_all,
                "assd_mm": assd,
                "hd95_mm": hd95,
                "mean_directed_dist_pred_to_gt_mm": m_ab,
                "mean_directed_dist_gt_to_pred_mm": m_ba,
                "normal_error_mean_deg": n_mean,
                "normal_error_max_deg": n_max,
            }

            fg = gt_vol | pr_vol
            if cl_pts is None or cl_radii is None or radius_bin_resolution == "off":
                do_radius_strata = False
            elif radius_bin_resolution in ("fixed", "global_quantile"):
                do_radius_strata = (
                    radius_bin_edges_mm is not None and len(radius_bin_edges_mm) >= 2
                )
            else:
                do_radius_strata = n_radius_bins > 0
            if do_radius_strata:
                bin_flat, edges, labels = M.radius_bins_from_centerline(
                    ref_im, fg, cl_pts, cl_radii, n_radius_bins, radius_bin_edges_mm
                )
                if bin_flat is not None and edges is not None:
                    n_strata = len(edges) - 1
                    strat = M.dice_per_radius_bin(gt_vol, pr_vol, bin_flat, n_strata, labels)
                    for srow in strat:
                        r = {**base_row, **srow}
                        r["metric_scope"] = "radius_stratum"
                        rows_out.append(r)
            base_row_overall = {
                **base_row,
                "radius_bin": -1,
                "radius_bin_label": "overall",
                "dice": dice_all,
                "n_voxels": int(np.count_nonzero(fg)),
                "metric_scope": "overall",
            }
            rows_out.append(base_row_overall)

    return rows_out


def _mp_case_worker(task: dict) -> tuple[str, list, str | None]:
    """
    Picklable entry point for Process workers. Returns (case_stem, rows, error_str_or_None).
    """
    stem = task["case_stem"]
    try:
        edges = task["radius_bin_edges_mm"]
        if edges is not None:
            edges = np.asarray(edges, dtype=np.float64)
        dmd = task.get("dice_max_dim")
        rows = run_case(
            stem,
            task["path_mc"],
            task["path_gt"],
            task["path_cl"],
            task["methods_and_params"],
            float(task["dice_spacing_mm"]),
            int(dmd) if dmd is not None else None,
            float(task["margin_mm"]),
            int(task["n_surface_samples"]),
            int(task["n_radius_bins"]),
            edges,
            str(task["radius_bin_resolution"]),
            int(task["seed"]),
            task["save_smoothed_dir"],
            str(task["study_settings_hash"]),
        )
        return stem, rows, None
    except Exception:
        return stem, [], traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Compare mesh smoothing methods against GT smooth surfaces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Directory with ground-truth smooth surfaces (.vtp), matched by filename",
    )
    parser.add_argument(
        "--mc_dir",
        type=str,
        required=True,
        help="Directory with marching-cubes / rough surfaces (.vtp)",
    )
    parser.add_argument(
        "--centerline_dir",
        type=str,
        default=None,
        help="Directory with centerlines (.vtp), same basename as meshes; MIS radius used for Dice strata",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path for all result rows",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="laplacian,taubin,taubin_cot",
        help="Comma-separated: none, laplacian, taubin, taubin_cot",
    )
    parser.add_argument(
        "--param_grid_json",
        type=str,
        default=None,
        help="JSON file overriding DEFAULT_PARAM_GRID (keys = method names, values = list of param dicts)",
    )
    parser.add_argument(
        "--dice_spacing_mm",
        type=float,
        default=0.4,
        help="Isotropic voxel spacing in the same length unit as the meshes (name is historical: "
        "use e.g. 0.04 if geometry is in cm and you want ~0.4 mm resolution). Coarser = faster.",
    )
    parser.add_argument(
        "--dice_max_dim",
        type=int,
        default=None,
        metavar="N",
        help="Cap each axis of the Dice raster grid at N samples (raises spacing if needed). "
        "Omit for no cap. Example: 256 limits memory vs huge bounds.",
    )
    parser.add_argument(
        "--margin_mm",
        type=float,
        default=2.0,
        help="Padding around combined GT/MC bounds, same unit as meshes (flag name is historical).",
    )
    parser.add_argument(
        "--n_surface_samples",
        type=int,
        default=30000,
        help="Surface samples per direction for ASSD / HD95 / normals",
    )
    parser.add_argument(
        "--n_radius_bins",
        type=int,
        default=5,
        help="Number of bins from MIS radius (0 disables strata). With --centerline_dir, edges are "
        "dataset-global quantiles by default (see --per_case_radius_bins).",
    )
    parser.add_argument(
        "--radius_bin_edges_mm",
        type=str,
        default=None,
        help="Fixed comma-separated MIS-radius bin edges in the same unit as centerlines/meshes "
        "(e.g. mm or cm; flag name is historical). Same for every case. Overrides quantile binning.",
    )
    parser.add_argument(
        "--per_case_radius_bins",
        action="store_true",
        help="Use quantiles of each case's centerline only (bins differ by case). Default is global "
        "quantiles pooled over all centerlines in this run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for surface sampling",
    )
    parser.add_argument(
        "--save_smoothed_dir",
        type=str,
        default=None,
        help="If set, write each smoothed mesh here (can use a lot of disk)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated basenames (without .vtp) to run; default = all common to gt and mc",
    )
    parser.add_argument(
        "--out_plots_dir",
        type=str,
        default=None,
        help="If set, write metric figures here after the CSV (see mesh_smooth_study/plots.py)",
    )
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=150,
        help="Figure DPI when --out_plots_dir is set",
    )
    parser.add_argument(
        "--plot_format",
        type=str,
        default="png",
        choices=("png", "pdf", "svg"),
        help="Figure format when --out_plots_dir is set",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Recompute every case even if present in --out_csv with matching settings",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=1,
        metavar="N",
        dest="num_workers",
        help="Number of parallel worker processes (one case per task). 1 = sequential.",
    )

    args = parser.parse_args()
    if args.num_workers < 1:
        raise SystemExit("--np must be >= 1")
    dice_max_dim = args.dice_max_dim if args.dice_max_dim is not None and args.dice_max_dim >= 2 else None
    if args.dice_max_dim is not None and args.dice_max_dim < 2:
        raise SystemExit("--dice_max_dim must be >= 2 or omitted")

    grid = json.loads(json.dumps(DEFAULT_PARAM_GRID))
    if args.param_grid_json:
        with open(args.param_grid_json, "r", encoding="utf-8") as f:
            user_grid = json.load(f)
        for k, v in user_grid.items():
            grid[k] = v

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods_and_params = {m: grid[m] for m in methods if m in grid}
    missing = [m for m in methods if m not in grid]
    if missing:
        raise SystemExit(f"Unknown method(s) {missing}. Known: {list(grid.keys())}")

    gt_files = _list_vtp(args.gt_dir)
    mc_files = _list_vtp(args.mc_dir)
    gt_set = set(gt_files)
    mc_set = set(mc_files)
    common = sorted(gt_set & mc_set)
    if args.cases:
        want = {c.strip() + ".vtp" for c in args.cases.split(",") if c.strip()}
        common = [f for f in common if f in want]
    if not common:
        raise SystemExit("No matching .vtp filenames between --gt_dir and --mc_dir")

    radius_edges, radius_mode = _resolve_radius_bin_edges(
        args.radius_bin_edges_mm,
        args.n_radius_bins,
        args.centerline_dir,
        args.per_case_radius_bins,
        common,
    )
    expected_keys = _expected_config_keys(methods_and_params)
    settings_hash = _study_settings_hash(
        args.dice_spacing_mm,
        dice_max_dim,
        args.margin_mm,
        args.n_surface_samples,
        args.n_radius_bins,
        radius_edges,
        radius_mode,
        args.seed,
    )

    out_csv = str(Path(args.out_csv).expanduser().resolve())
    if os.path.isfile(out_csv):
        df = pd.read_csv(out_csv)
    else:
        df = pd.DataFrame()

    def _paths_for(fname: str) -> tuple[str, str, str, str | None]:
        stem = Path(fname).stem
        path_gt = os.path.join(args.gt_dir, fname)
        path_mc = os.path.join(args.mc_dir, fname)
        path_cl = None
        if args.centerline_dir:
            cand = os.path.join(args.centerline_dir, fname)
            if os.path.isfile(cand):
                path_cl = cand
            else:
                logger.warning("No centerline %s; strata use overall Dice only", cand)
        return stem, path_gt, path_mc, path_cl

    tasks: list[dict] = []
    for fname in common:
        stem, path_gt, path_mc, path_cl = _paths_for(fname)
        if not args.no_skip and _case_already_complete(df, stem, expected_keys, settings_hash):
            logger.info("Case %s — skip (already in CSV for this settings hash)", stem)
            continue
        save_sub = os.path.join(args.save_smoothed_dir, stem) if args.save_smoothed_dir else None
        tasks.append(
            {
                "case_stem": stem,
                "path_mc": path_mc,
                "path_gt": path_gt,
                "path_cl": path_cl,
                "methods_and_params": methods_and_params,
                "dice_spacing_mm": args.dice_spacing_mm,
                "dice_max_dim": dice_max_dim,
                "margin_mm": args.margin_mm,
                "n_surface_samples": args.n_surface_samples,
                "n_radius_bins": args.n_radius_bins,
                "radius_bin_edges_mm": None if radius_edges is None else radius_edges.copy(),
                "radius_bin_resolution": radius_mode,
                "seed": args.seed,
                "save_smoothed_dir": save_sub,
                "study_settings_hash": settings_hash,
            }
        )

    if args.num_workers <= 1:
        for task in tasks:
            stem = task["case_stem"]
            logger.info("Case %s — running", stem)
            rows = run_case(
                stem,
                task["path_mc"],
                task["path_gt"],
                task["path_cl"],
                methods_and_params,
                args.dice_spacing_mm,
                dice_max_dim,
                args.margin_mm,
                args.n_surface_samples,
                args.n_radius_bins,
                radius_edges,
                radius_mode,
                args.seed,
                task["save_smoothed_dir"],
                settings_hash,
            )
            if not df.empty and "case_id" in df.columns:
                df = df[df["case_id"] != stem]
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            _atomic_write_csv(df, out_csv)
            logger.info("Case %s — wrote CSV (%d total rows)", stem, len(df))
    else:
        logger.info("Running %d cases on %d workers", len(tasks), args.num_workers)
        try:
            with Pool(processes=args.num_workers) as pool:
                for stem, rows, err in pool.imap_unordered(_mp_case_worker, tasks, chunksize=1):
                    if err:
                        logger.error("Case %s failed:\n%s", stem, err)
                        continue
                    logger.info("Case %s — finished (%d rows)", stem, len(rows))
                    if not df.empty and "case_id" in df.columns:
                        df = df[df["case_id"] != stem]
                    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                    _atomic_write_csv(df, out_csv)
                    logger.info("Case %s — wrote CSV (%d total rows)", stem, len(df))
        except KeyboardInterrupt:
            logger.warning("Interrupted; pool terminated")
            raise

    logger.info("Finished all cases: %d rows in %s", len(df), out_csv)

    if args.out_plots_dir:
        from mesh_smooth_study.plots import generate_study_plots

        plot_paths = generate_study_plots(
            out_csv,
            args.out_plots_dir,
            dpi=args.plot_dpi,
            fig_format=args.plot_format,
        )
        for pth in plot_paths:
            logger.info("Wrote plot %s", pth)


if __name__ == "__main__":
    main()
