#!/usr/bin/env python3
"""
Mesh smoothing study: sweep Laplacian / Taubin / Taubin–cotangent settings on MC surfaces
and compare to GT smooth surfaces (Dice, ASSD, HD95, normal angular error,
relative volume / surface-area error; Dice by MIS radius).
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

# Selectable metric groups (see --metrics). "dice" also drives radius-stratum Dice.
# "smoothness" = intrinsic surface roughness of the smoothed mesh (no GT needed).
ALL_METRICS = ("dice", "distance", "normal", "volume", "surface_area", "smoothness")

# True 2-D exhaustive (Cartesian) sweep per method; kept in sync with
# mesh_smooth_study/default_param_grid.json. Override either via --param_grid_json.
_LAPLACIAN_ITERATIONS = (10, 25, 50, 75, 100)
_LAPLACIAN_RELAXATION = (0.05, 0.1, 0.15, 0.2, 0.25)
_TAUBIN_ITERATIONS = (10, 25, 50, 75, 100)
_TAUBIN_SMOOTHING_FACTOR = (0.0, 0.25, 0.5, 0.75, 1.0)
_TAUBIN_COT_ITERATIONS = (20, 50, 75, 100, 150)
_TAUBIN_COT_MU1 = (0.45, 0.5, 0.55, 0.6, 0.65)

DEFAULT_PARAM_GRID = {
    "none": [{}],
    "laplacian": [
        {"iterations": it, "relaxation_factor": rf, "boundary_smoothing": True}
        for it in _LAPLACIAN_ITERATIONS
        for rf in _LAPLACIAN_RELAXATION
    ],
    "taubin": [
        {"iterations": it, "boundary": False, "feature": False, "smoothing_factor": sf}
        for it in _TAUBIN_ITERATIONS
        for sf in _TAUBIN_SMOOTHING_FACTOR
    ],
    "taubin_cot": [
        {"iterations": it, "mu1": mu1, "mu2": round(mu1 + 0.01, 3)}
        for it in _TAUBIN_COT_ITERATIONS
        for mu1 in _TAUBIN_COT_MU1
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


# Short, readable names for common smoothing params (others fall back to the raw key).
_PARAM_NAME_ABBREV = {
    "iterations": "iters",
    "relaxation_factor": "relax",
    "boundary_smoothing": "bnd",
    "feature": "feat",
    "feature_angle": "fangle",
    "pass_band": "pb",
}


def _params_filename_str(params: dict) -> str:
    """Compact, filesystem-safe encoding of smoothing params for filenames.

    Examples: ``iters100_relax0.05``, ``iters50_mu1-0.5_mu2-0.51``, ``default``.
    Encodes every key/value, so distinct params never collide.
    """
    if not params:
        return "default"

    parts = []
    for key, val in sorted(params.items()):
        name = _PARAM_NAME_ABBREV.get(key, str(key))
        if isinstance(val, bool):
            val_str = "on" if val else "off"
        elif isinstance(val, float):
            val_str = f"{val:g}"
        else:
            val_str = str(val)
        # Keep alphanumerics, dot and minus (all valid in filenames); drop the rest.
        safe = "".join(c for c in val_str if c.isalnum() or c in ".-")
        parts.append(f"{name}{safe}" if name in _PARAM_NAME_ABBREV.values() else f"{name}-{safe}")
    return "_".join(parts)


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
    metrics: set[str] | None = None,
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
        "metrics": sorted(metrics) if metrics else list(ALL_METRICS),
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


def _append_rows_csv(rows: list[dict], path: str) -> int:
    """Append result rows to the CSV without rewriting existing content.

    Writes the header only when the file is new. If new rows introduce columns not present
    in the existing header (rare), the file is rewritten once to keep columns aligned.
    Returns the number of rows appended.
    """
    if not rows:
        return 0
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if out.is_file():
        existing_cols = list(pd.read_csv(out, nrows=0).columns)
        extra = [c for c in new_df.columns if c not in existing_cols]
        cols = existing_cols + extra
        new_df = new_df.reindex(columns=cols)
        if extra:
            old = pd.read_csv(out).reindex(columns=cols)
            full = pd.concat([old, new_df], ignore_index=True)
            _atomic_write_csv(full, str(out))
        else:
            new_df.to_csv(out, mode="a", header=False, index=False)
    else:
        new_df.to_csv(out, index=False)
    return len(new_df)


def _done_file_for(out_csv: str) -> str:
    """Sibling ledger of completed cases for a given results CSV."""
    p = Path(out_csv)
    return str(p.with_name(p.stem + ".done.txt"))


def _load_done_cases(done_file: str, study_settings_hash: str) -> set[str]:
    """Read case stems marked done for this settings hash. Lines are 'stem<TAB>settings_hash'."""
    done: set[str] = set()
    p = Path(done_file)
    if not p.is_file():
        return done
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        stem = parts[0]
        h = parts[1] if len(parts) > 1 else ""
        if h == study_settings_hash:
            done.add(stem)
    return done


def _mark_case_done(done_file: str, stem: str, study_settings_hash: str) -> None:
    p = Path(done_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{stem}\t{study_settings_hash}\n")


def _bbox_diag(poly) -> float:
    """Bounding-box diagonal length of a polydata (nan if bounds are non-finite)."""
    b = np.asarray(poly.GetBounds(), dtype=np.float64)
    if not np.all(np.isfinite(b)):
        return float("nan")
    d = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]], dtype=np.float64)
    return float(np.sqrt(np.sum(d * d)))


def _smoothing_diverged(poly, ref_diag: float, max_factor: float = 50.0) -> bool:
    """True if the smoothed mesh exploded (non-finite points or bbox >> the input scale).

    Unstable Taubin / Taubin-cotangent settings can send vertices to ~1e6+, which yields NaN
    Dice and makes the cell-locator distance queries crawl. We detect that cheaply from bounds.
    """
    diag = _bbox_diag(poly)
    if not np.isfinite(diag):
        return True
    if np.isfinite(ref_diag) and ref_diag > 0 and diag > max_factor * ref_diag:
        return True
    return False


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
    metrics: set[str] | None = None,
):
    metrics = set(metrics) if metrics else set(ALL_METRICS)
    do_dice = "dice" in metrics
    do_distance = "distance" in metrics
    do_normal = "normal" in metrics
    do_volume = "volume" in metrics
    do_area = "surface_area" in metrics
    do_smoothness = "smoothness" in metrics

    gt_reader = vf.read_geo(path_gt)
    gt_poly = gt_reader.GetOutput()
    mc_reader = vf.read_geo(path_mc)
    mc_poly = mc_reader.GetOutput()

    # Cap open boundaries (clipped ends, MC gaps) before any smoothing so free edges
    # are not pulled inward by the smoothing passes.
    mc_poly = S.fill_holes(mc_poly)

    cl_pts, cl_radii = (None, None)
    if path_cl and os.path.isfile(path_cl):
        cl_pts, cl_radii = _load_centerline_radii(path_cl)

    # Scale reference for divergence detection: largest of the GT / MC bbox diagonals.
    ref_diag = max(_bbox_diag(gt_poly), _bbox_diag(mc_poly))

    ref_im, eff_sp = (None, float(dice_spacing_mm))
    if do_dice:
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
            param_str = json.dumps(params, sort_keys=True)
            params_sha10 = _params_sha10(params)

            nan = float("nan")
            smoothed = None
            fail_reason = ""
            try:
                smoothed = S.apply_method(method, mc_poly, params)
            except Exception as e:
                logger.exception("Smoothing failed case=%s method=%s params=%s: %s", case_stem, method, params, e)
                fail_reason = "smoothing_error"

            if smoothed is not None and save_smoothed_dir:
                os.makedirs(save_smoothed_dir, exist_ok=True)
                params_name = _params_filename_str(params)
                out_vtp = os.path.join(
                    save_smoothed_dir, f"{case_stem}__{method}__{params_name}.vtp"
                )
                vf.write_geo(out_vtp, smoothed)

            diverged = smoothed is not None and _smoothing_diverged(smoothed, ref_diag)
            if diverged:
                fail_reason = "diverged"
                logger.warning(
                    "Case %s method=%s params=%s: smoothing diverged "
                    "(bbox diag %.3g vs input %.3g); recording NaN and skipping metrics",
                    case_stem,
                    method,
                    params_sha10,
                    _bbox_diag(smoothed),
                    ref_diag,
                )
            # A "failed" config is one we could not score: smoothing raised, or the mesh blew up.
            failed = fail_reason != ""

            gt_vol = pr_vol = fg = None
            dice_all = nan
            if do_dice and not failed:
                gt_vol = M.poly_to_binary_volume(gt_poly, ref_im)
                pr_vol = M.poly_to_binary_volume(smoothed, ref_im)
                dice_all = M.dice_binary(gt_vol, pr_vol)
                fg = gt_vol | pr_vol

            assd = hd95 = m_ab = m_ba = nan
            if do_distance and not failed:
                assd, hd95, m_ab, m_ba = M.assd_and_hd95(smoothed, gt_poly, n_surface_samples, seed)

            n_mean = n_max = nan
            if do_normal and not failed:
                n_mean, n_max = M.normal_angle_errors_deg(smoothed, gt_poly, n_surface_samples, seed + 17)

            vol_err = (
                M.volume_error_metrics(gt_poly, smoothed)
                if do_volume and not failed
                else {"volume_gt": nan, "volume_pred": nan, "volume_error_abs": nan, "volume_error_rel": nan}
            )
            area_err = (
                M.surface_area_error_metrics(gt_poly, smoothed)
                if do_area and not failed
                else {
                    "surface_area_gt": nan,
                    "surface_area_pred": nan,
                    "surface_area_error_abs": nan,
                    "surface_area_error_rel": nan,
                }
            )
            smooth_m = (
                M.smoothness_metrics(smoothed)
                if do_smoothness and not failed
                else {"mean_curvature_rms": nan, "dihedral_angle_p95_deg": nan}
            )

            base_row = {
                "case_id": case_stem,
                "method": method,
                "params_json": param_str,
                "params_sha10": params_sha10,
                "study_settings_hash": study_settings_hash,
                "metrics": ",".join(sorted(metrics)),
                "diverged": bool(diverged),
                "failed": bool(failed),
                "fail_reason": fail_reason,
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
                "volume_gt": vol_err["volume_gt"],
                "volume_pred": vol_err["volume_pred"],
                "volume_error_abs": vol_err["volume_error_abs"],
                "volume_error_rel": vol_err["volume_error_rel"],
                "surface_area_gt": area_err["surface_area_gt"],
                "surface_area_pred": area_err["surface_area_pred"],
                "surface_area_error_abs": area_err["surface_area_error_abs"],
                "surface_area_error_rel": area_err["surface_area_error_rel"],
                "mean_curvature_rms": smooth_m["mean_curvature_rms"],
                "dihedral_angle_p95_deg": smooth_m["dihedral_angle_p95_deg"],
            }

            do_radius_strata = do_dice and fg is not None
            if not do_radius_strata or cl_pts is None or cl_radii is None or radius_bin_resolution == "off":
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
                "n_voxels": int(np.count_nonzero(fg)) if fg is not None else -1,
                "metric_scope": "overall",
            }
            rows_out.append(base_row_overall)

            logger.info(
                "  %s/%s %s | Dice=%.4f ASSD=%.4g HD95=%.4g normal(mean/max)=%.2f/%.2f deg "
                "dV_rel=%+.3f dS_rel=%+.3f Hrms=%.4g dihedral_p95=%.2f deg",
                case_stem,
                method,
                params_sha10,
                dice_all,
                assd,
                hd95,
                n_mean,
                n_max,
                vol_err["volume_error_rel"],
                area_err["surface_area_error_rel"],
                smooth_m["mean_curvature_rms"],
                smooth_m["dihedral_angle_p95_deg"],
            )

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
            set(task["metrics"]),
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
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated metric groups to compute: all, or any of "
        f"{','.join(ALL_METRICS)}. 'dice' = volumetric Dice (and radius-stratum Dice with "
        "--centerline_dir); 'distance' = ASSD/HD95; 'normal' = normal angle error; "
        "'volume' / 'surface_area' = relative volume / surface-area error; "
        "'smoothness' = area-weighted RMS mean curvature and 95th-pct adjacent-face "
        "dihedral angle of the smoothed mesh. Unselected metrics are written as NaN.",
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
        help="Recompute every case even if listed as done for the current settings hash. "
        "Appends new rows (does not delete prior rows for the case).",
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

    metric_tokens = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    if not metric_tokens or "all" in metric_tokens:
        metrics = set(ALL_METRICS)
    else:
        metrics = set(metric_tokens)
        bad = sorted(metrics - set(ALL_METRICS))
        if bad:
            raise SystemExit(f"Unknown metric(s) {bad}. Known: all,{','.join(ALL_METRICS)}")
    logger.info("Computing metrics: %s", ",".join(sorted(metrics)))

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
        metrics,
    )

    out_csv = str(Path(args.out_csv).expanduser().resolve())
    done_file = str(Path(_done_file_for(out_csv)).expanduser().resolve())

    # Bootstrap the ledger from a pre-existing CSV (older runs without a done.txt) so we can
    # resume them without recomputation.
    if not os.path.isfile(done_file) and os.path.isfile(out_csv):
        try:
            old_df = pd.read_csv(out_csv)
            for stem in sorted(set(old_df.get("case_id", pd.Series(dtype=str)).astype(str))):
                if _case_already_complete(old_df, stem, expected_keys, settings_hash):
                    _mark_case_done(done_file, stem, settings_hash)
            logger.info("Bootstrapped %s from existing CSV", done_file)
        except Exception as e:
            logger.warning("Could not bootstrap done file from CSV: %s", e)

    done_cases = _load_done_cases(done_file, settings_hash)

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
        if not args.no_skip and stem in done_cases:
            logger.info("Case %s — skip (in %s for this settings hash)", stem, os.path.basename(done_file))
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
                "metrics": sorted(metrics),
            }
        )

    total_rows = 0
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
                metrics,
            )
            n = _append_rows_csv(rows, out_csv)
            total_rows += n
            _mark_case_done(done_file, stem, settings_hash)
            done_cases.add(stem)
            logger.info("Case %s — appended %d rows (%d this run)", stem, n, total_rows)
    else:
        logger.info("Running %d cases on %d workers", len(tasks), args.num_workers)
        try:
            with Pool(processes=args.num_workers) as pool:
                for stem, rows, err in pool.imap_unordered(_mp_case_worker, tasks, chunksize=1):
                    if err:
                        logger.error("Case %s failed:\n%s", stem, err)
                        continue
                    n = _append_rows_csv(rows, out_csv)
                    total_rows += n
                    _mark_case_done(done_file, stem, settings_hash)
                    done_cases.add(stem)
                    logger.info("Case %s — finished, appended %d rows (%d this run)", stem, n, total_rows)
        except KeyboardInterrupt:
            logger.warning("Interrupted; pool terminated")
            raise

    logger.info("Finished: appended %d rows this run to %s (%d cases done)", total_rows, out_csv, len(done_cases))

    if args.out_plots_dir:
        from mesh_smooth_study.plots import generate_study_plots, write_summary_csv

        summary_path = write_summary_csv(out_csv, os.path.join(args.out_plots_dir, "summary.csv"))
        if summary_path is not None:
            logger.info("Wrote summary %s", summary_path)

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
