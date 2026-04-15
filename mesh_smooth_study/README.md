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

## Resume and incremental CSV

- Results are **appended per case**: after each case finishes, the full table is written to `--out_csv` (atomic replace via a `.tmp` file).
- A case is **skipped** when the CSV already has an `overall` row for **every** `(method, params_sha10)` in the current grid **and** the same **`study_settings_hash`** (Dice spacing, margin, surface samples, radius bin **edges + resolution mode**, seed). Changing the case list can change **global** quantile edges, so the hash updates and cases may rerun.
- CSVs from older runs **without** `study_settings_hash` are treated as incomplete: every case is recomputed once (rows for that case are replaced).
- **`--no_skip`**: process all cases regardless of the CSV.

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
  - **`mesh_smooth_study/default_param_grid.json`** — full sweep (many configs per method).
  - **`mesh_smooth_study/param_grid_quick.json`** — compact grid for pilots; within each method, entries are ordered **mild → strong** smoothing (`none` first).
- **`--methods none,laplacian,...`** — include `none` to score the raw MC mesh without smoothing.
- **`--save_smoothed_dir ./out_smoothed`** — writes one `.vtp` per case × method × parameter set (large).
- **`--n_radius_bins 0`** — disable quantile-based global bins (no strata unless you still pass `--radius_bin_edges_mm`).
- **`--per_case_radius_bins`** — per-case quantile bins instead of dataset-global (see Metrics above).
- **`--np N`** — run **N** cases in parallel (`N=1` default). Each worker loads VTK/meshes independently; the main process still updates the CSV **serially** after each case finishes (same resume behavior). If you see crashes with VTK, try `--np 1`.

## Output CSV

Long format: one row per **overall** metric block and additional rows per **radius stratum** (`metric_scope` = `overall` or `radius_stratum`). Stratified rows repeat ASSD/HD95/normals for that same run (those metrics are global to the mesh pair).

The column **`params_sha10`** matches the suffix in filenames written with `--save_smoothed_dir` (`{case}__{method}__{params_sha10}.vtp`).

## Plots

After a run, generate figures from the CSV in two ways:

1. **Same command as the study** — add `--out_plots_dir ./plots` (optional: `--plot_dpi`, `--plot_format pdf`).
2. **Standalone** (no VTK) — `python mesh_smooth_study/plot_study_results.py --csv ./results.csv --out_dir ./plots`

Figures written (when applicable):

| File | Content |
|------|---------|
| `overall_metrics_boxplots.png` | Dice, ASSD, HD95, normal mean/max — one subplot each; box + strip = spread across cases |
| `overall_metrics_mean_std.png` | Same metrics as mean ± std bars |
| `dice_by_vessel_radius.png` | Facet per method: Dice vs radius stratum (only if radius-stratum rows exist) |

## Dependencies

Uses the repo environment: **VTK**, **numpy**, **scipy**, **pandas**, and `modules.vtk_functions`.
