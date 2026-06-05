# Mesh smoothing study

Scripts to compare **Laplacian** (`vtkSmoothPolyDataFilter`), **Taubin** (windowed sinc / `smooth_polydata`), and **Taubin with cotangent Laplacian** (`taubin_smooth_polydata`) on marching-cubes surfaces against **ground-truth smooth** surfaces.

## Units (important)

All geometry is used **as-is**: **no unit conversion**. Whatever length unit your **meshes and centerlines** use (often **mm** in DICOM pipelines, sometimes **cm**) applies to:

- `--dice_spacing_mm`, `--margin_mm`, `--radius_bin_edges_mm` (names say `_mm` for historical reasons only — values are in **your mesh unit**).
- MIS radii from the centerline and global/per-case bin edges.
- CSV columns named `*_mm` (`assd_mm`, `hd95_mm`, directed distances): numbers are in the **same unit as the surfaces**, not necessarily millimeters.

**Example (meshes in cm):** default `--dice_spacing_mm 0.4` means **0.4 cm** voxels (~4 mm). For ~0.04 cm (~0.4 mm) voxels, pass `--dice_spacing_mm 0.04`.

## Input layout

Use the **same basename** for each case (example: `case_001.vtp`):

| Role | Directory flag | Content |
|------|----------------|---------|
| GT smooth | `--gt_dir` | Reference smooth surface |
| MC rough | `--mc_dir` | Input surface to smooth (e.g. discrete marching cubes) |
| Centerline (optional) | `--centerline_dir` | VTP with `MaximumInscribedSphereRadius` point array for vessel-caliber strata |

Files are paired by filename across folders.

## Metrics

- **Dice (overall)**: Both meshes rasterized into a shared isotropic grid (`--dice_spacing_mm`, same length unit as meshes), then standard overlap Dice. Optional **`--dice_max_dim N`** caps each axis at **N** voxels by **increasing spacing** if the padded bounds would exceed that (saves memory/time on large extents). The CSV records **`dice_effective_spacing`** (actual spacing used) and **`dice_max_dim`** when set.
- **Dice by vessel size**: Foreground voxels (GT ∪ prediction) are labeled by **nearest centerline point’s MIS radius**, then binned using **shared edges** so strata match across cases for pooling/plots:
  - **Default** (with `--centerline_dir`, no fixed edges): **global quantiles** — edges are computed once from **all MIS radii pooled** over every centerline in this run (same `--cases` / GT∩MC list), then reused for every case.
  - **`--radius_bin_edges_mm`**: fixed MIS-radius edges in mesh units, same for all cases.
  - **`--per_case_radius_bins`**: revert to **per-case quantiles** (edges differ by case; worse for cross-case plots).
  - Each row records **`radius_bin_resolution`** (`fixed` / `global_quantile` / `per_case` / `off`) and **`radius_bin_edges_json`** when edges are defined.
- **ASSD / HD95** (`assd_mm` / `hd95_mm` in CSV): Area-weighted surface samples; distances in **mesh length units**; computed with **`vtkCellLocator`** (not `vtkImplicitPolyDataDistance`, which can crash on open/non-manifold vascular surfaces).
- **Normal error (deg)**: Samples on the smoothed surface; **GT normal** = normal at the **nearest GT vertex** (cKDTree); **pred normal** = nearest pred vertex to the sample; mean and max angle between them.
- **Divergence guard**: unstable Taubin / Taubin-cotangent settings can blow up (vertices fly to ~1e6+), which produces NaN Dice and makes the distance queries crawl. After smoothing, the mesh bbox is compared to the input scale; if it exploded (non-finite bounds or bbox diagonal > 50× the GT/MC diagonal) the config is flagged `diverged=True`, all metrics are written as `NaN`, and the expensive Dice / distance / normal computations are **skipped** for that config (avoids the slowdown). Check the `diverged` CSV column to spot unstable parameter regions.

## Resume and incremental CSV

- Results are **appended per case**: after each case finishes its rows are appended to `--out_csv` (the existing file is **not** rewritten — only new rows are added; header is written once when the file is created).
- Completed cases are tracked in a **`done.txt` ledger** next to the CSV (`<out_csv stem>.done.txt`). Each line is `case<TAB>study_settings_hash`. On startup the ledger is loaded and matching cases are **skipped**, so an interrupted run resumes without recomputation.
- The **`study_settings_hash`** encodes Dice spacing, margin, surface samples, radius bin **edges + resolution mode**, seed, and the selected **`--metrics`** set. Changing any of these (or the case list, which can shift **global** quantile edges) yields a new hash, so those cases are treated as not-done and rerun.
- For a pre-existing CSV from before this ledger existed, the `done.txt` is **bootstrapped once** from the CSV (cases with a complete set of `(method, params_sha10)` rows for the current hash are marked done).
- **`--no_skip`**: process every case regardless of the ledger. Note: rows are **appended**, so rerunning a case adds duplicate rows rather than replacing them.

## Usage

From the **repository root** (so `modules` imports resolve):

```bash
python mesh_smooth_study/run_study.py \
  --gt_dir /path/to/gt_smooth_vtp \
  --mc_dir /path/to/mc_rough_vtp \
  --centerline_dir /path/to/centerlines_vtp \
  --out_csv ./results/smoothing_study.csv \
  --methods laplacian,taubin,taubin_cot
```

Optional:

- **`--param_grid_json`** — override the built-in parameter lists per method. Bundled files:
  - **`mesh_smooth_study/default_param_grid.json`** — full **2-D exhaustive (Cartesian) sweep** per method (5×5 = 25 configs each, plus `none`): `laplacian` = iterations × relaxation_factor; `taubin` = iterations × smoothing_factor; `taubin_cot` = iterations × mu1 (with `mu2 = mu1 + 0.01`). This is the built-in default (mirrored by `DEFAULT_PARAM_GRID` in `run_study.py`).
  - **`mesh_smooth_study/param_grid_small.json`** — small 3×3 Cartesian sweep per method for quick interaction checks.
  - **`mesh_smooth_study/param_grid_quick.json`** — compact curated grid for pilots; within each method, entries are ordered **mild → strong** smoothing (`none` first).
- **`--methods none,laplacian,...`** — include `none` to score the raw MC mesh without smoothing.
- **`--metrics all`** — choose which metric groups to compute (comma-separated). Groups: `dice` (volumetric Dice, plus radius-stratum Dice when `--centerline_dir` is given), `distance` (ASSD/HD95 + directed distances), `normal` (normal angle error mean/max), `volume` (relative/absolute volume error), `surface_area` (relative/absolute surface-area error). Default `all`. Unselected metrics are written as `NaN` and skip their (often expensive) computation — e.g. `--metrics dice,volume,surface_area` skips the surface-sampling-heavy distance and normal metrics. The chosen set is part of the `study_settings_hash`, so changing it reruns cases, and each row records a `metrics` column.
- **`--save_smoothed_dir ./out_smoothed`** — writes one `.vtp` per case × method × parameter set (large), named with the smoothing params (`{case}__{method}__{params}.vtp`, e.g. `case01__laplacian__iters100_relax0.05.vtp`).
- **`--n_radius_bins 0`** — disable quantile-based global bins (no strata unless you still pass `--radius_bin_edges_mm`).
- **`--per_case_radius_bins`** — per-case quantile bins instead of dataset-global (see Metrics above).
- **`--np N`** — run **N** cases in parallel (`N=1` default). Each worker loads VTK/meshes independently; the main process still updates the CSV **serially** after each case finishes (same resume behavior). If you see crashes with VTK, try `--np 1`.

## Output CSV

Long format: one row per **overall** metric block and additional rows per **radius stratum** (`metric_scope` = `overall` or `radius_stratum`). Stratified rows repeat ASSD/HD95/normals for that same run (those metrics are global to the mesh pair).

Saved meshes from `--save_smoothed_dir` are named `{case}__{method}__{params}.vtp` (params encoded readably, e.g. `iters50_mu1-0.5_mu2-0.51`). The `params_sha10` / `params_json` columns identify the same configuration in the CSV.

## Plots

After a run, generate figures from the CSV in two ways:

1. **Same command as the study** — add `--out_plots_dir ./plots` (optional: `--plot_dpi`, `--plot_format pdf`). Also writes `./plots/summary.csv`.
2. **Standalone** (no VTK) — `python mesh_smooth_study/plot_summary.py --csv ./results.csv --out_dir ./plots` (optional `--summary_csv ./plots/summary.csv`).

Add **`--improve_over_none`** (standalone) to plot each metric as the **per-case improvement relative to the `none` baseline** (matched on `case_id`, oriented so positive = better) instead of raw values. The `none` group is dropped and a zero reference line is drawn; outputs get an `_improve_over_none` suffix (e.g. `overall_metrics_boxplots_improve_over_none.png`, `summary_improve_over_none.csv`) so raw figures are not overwritten.

Figures written (when applicable):

| File | Content |
|------|---------|
| `summary.csv` | One row per `(method, params_sha10)`: mean/std/median of each metric across cases, sorted by mean Dice |
| `overall_metrics_boxplots.png` | Dice, ASSD, HD95, normal mean/max — one subplot each; box + strip = spread across cases |
| `overall_metrics_mean_std.png` | Same metrics as mean ± std bars |
| `dice_by_vessel_radius.png` | Facet per method: Dice vs radius stratum (only if radius-stratum rows exist) |

## Dependencies

Uses the repo environment: **VTK**, **numpy**, **scipy**, **pandas**, and `modules.vtk_functions`.
