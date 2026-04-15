"""Plot pred-vs-pred metric differences: reference vs others vs others amongst themselves.

Reads the per-comparison CSV files from compute_metrics_meshes_pred_vs_pred and
designates one prediction as the 'reference'. Compares:
  1) Reference vs Others: metric values from comparisons involving the reference
  2) Others vs Others: metric values from pairwise comparisons between non-reference preds

Violin plots show whether the reference is closer or further from the other preds
than they are from each other.

With --point-comparison: for each case (e.g. data0), compute |pairwise differences|
within each group. E.g. |data0_obs0 - data0_obs1| = |(ref vs pred1) - (ref vs pred2)|
for the same case. Plots the distribution of these absolute case-wise differences.

Usage:
	python analysis/plot_pred_vs_pred_reference.py /path/to/output_dir --reference obs1
	python -m analysis.plot_pred_vs_pred_reference /path/to/output_dir --reference obs1  # if installed as package
	python analysis/plot_pred_vs_pred_reference.py /path/to/output_dir --reference obs1 --point-comparison
	python analysis/plot_pred_vs_pred_reference.py /path/to/output_dir --reference obs1 --radar
	python analysis/plot_pred_vs_pred_reference.py /path/to/output_dir --reference obs1 --metrics dice,hausdorff_sym,volume_error_rel
"""

from __future__ import annotations

import argparse
from typing import Callable
import glob
import itertools
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
	sys.path.insert(0, _repo_root)
from analysis.plotting.latex_table import (
	write_point_diff_latex_table,
	write_pred_vs_pred_reference_latex_table,
)
from analysis.plotting.violin_plot_functions import (
	NATURE_COLORS_DEFAULT as NATURE_COLORS,
	apply_nature_style,
	add_wilcoxon_signed_rank_bracket,
	draw_violin_ax,
	save_violin_figure,
)


# Patient palette and mappings for violin styling
PATIENT_PALETTE = {
	"P1": "#4477AA",
	"P2": "#EE6677",
	"P3": "#228833",
	"P4": "#CCBB44",
	"P5": "#66CCEE",
	"P6": "#AA3377",
}

# Case ID (e.g. 0032) -> patient label (P1, P2, ...)
CASE_TO_PATIENT = {
	"0032": "P1",
	"0034": "P2",
	"0036": "P3",
	"0038": "P4",
	"0042": "P5",
	"0043": "P6",
}

# Operator display names
OP_DISPLAY_NAMES = {
	"Vmr": "Op1",
	"gala": "Op2",
	"numi": "Op3",
}

# Violin styling
VIOLIN_EDGE_COLOR = "#6D6E71"
VIOLIN_FACE_COLOR = "#E0E0E0"  # light gray for body

# Main scalar metrics to plot (excludes case, n_*, and radius-bucket columns)
DEFAULT_METRICS = [
	'dice',
	'hausdorff_sym',
	'hd95_sym',
	'assd',
	'volume_error_rel',
	'surface_area_error_rel',
	'surface_dice_t1',
	'surface_dice_t2',
	'centerline_overlap',
]

# Display names for axis labels and titles
METRIC_DISPLAY_NAMES = {
	'hausdorff_sym': 'Hausdorff (sym)',
	'hd95_sym': 'HD 95th',
	'assd': 'ASSD',
	'dice': 'Dice',
	'volume_error_rel': 'Volume error (rel)',
	'surface_area_error_rel': 'Relative Surface Area Error',
	'surface_dice_t1': 'Surface Dice τ1',
	'surface_dice_t2': 'Surface Dice τ2',
	'centerline_overlap': 'Centerline overlap',
	'mean_normal_angular_error_gt_to_pred': 'Average Normal Difference (°)',
	'max_normal_angular_error_gt_to_pred': 'Max Normal Difference (°)',
	'std_normal_angular_error_gt_to_pred': 'Std Normal Difference (°)',
}

# Dice-like metrics: [0, 1] range, ylim upper = 1
DICE_METRICS = {'dice', 'surface_dice_t1', 'surface_dice_t2', 'centerline_overlap'}
# Distance/error metrics: lower bound 0
DISTANCE_METRICS = {
	'hausdorff_sym', 'hd95_sym', 'assd', 'volume_error_rel', 'surface_area_error_rel',
	'mean_normal_angular_error_gt_to_pred', 'max_normal_angular_error_gt_to_pred', 'std_normal_angular_error_gt_to_pred',
}
# Relative error metrics: use absolute value before plotting/stats
RELATIVE_ERROR_METRICS = {'volume_error_rel', 'surface_area_error_rel'}


def _metric_label(metric: str) -> str:
	return METRIC_DISPLAY_NAMES.get(metric, metric)


def _case_to_patient(case: str) -> str:
	"""Map case ID to patient label (P1..P6). Extracts numeric part if case is e.g. data0_0032."""
	s = str(case).strip()
	# Try direct match first
	if s in CASE_TO_PATIENT:
		return CASE_TO_PATIENT[s]
	# Extract 4-digit ID (0032, 0034, etc.) from strings like data0_0032
	m = re.search(r'(\d{4})', s)
	if m:
		cid = m.group(1)
		return CASE_TO_PATIENT.get(cid, s)
	return s


def _tick_labels(reference: str) -> dict[str, str]:
	"""X-axis tick labels for the two groups. Multi-line for straight (non-diagonal) display."""
	if reference.lower() == 'seqseg':
		return {
			'Reference vs Others': 'SeqSeg\nvs\nManual',
			'Others vs Others': 'Manual\nvs\nManual',
		}
	return {'Reference vs Others': f'{reference}\nvs\nOthers', 'Others vs Others': 'Others\nvs\nOthers'}


def _set_ylim(ax, metric: str) -> None:
	"""Set y-axis limits: upper 1 for dice (lower auto), lower 0 for distances."""
	if metric in DICE_METRICS:
		ax.set_ylim(top=1)
	elif metric in DISTANCE_METRICS:
		ax.set_ylim(bottom=0)


def _patient_legend_handles():
	"""Return (handles, labels) for patient color legend (P1-P6)."""
	from matplotlib.patches import Patch
	return (
		[Patch(facecolor=c, label=p) for p, c in PATIENT_PALETTE.items()],
		list(PATIENT_PALETTE.keys()),
	)


def _draw_violin_ax_reference_style(
	ax,
	vals_by_group: list[np.ndarray],
	positions: list[int],
	labels: list[str],
	patients_by_group: list[np.ndarray] | None,
	ylabel: str,
	*,
	group_order: list[str],
	set_ylim: Callable[[object], None] | None = None,
	add_wilcoxon: bool = True,
	subplot_label: str | None = None,
) -> None:
	"""Draw violin plot with reference style: gray body, patient-colored scatter overlay.

	- Violin body: gray fill, edge #6D6E71
	- Others vs Others (manual vs manual): hatch '//'
	- Mean: dashed line (--), Median: solid line
	- Scatter points overlaid and colored by patient
	"""
	if subplot_label:
		ax.text(
			-0.28, 1.05, subplot_label,
			transform=ax.transAxes,
			fontsize=8,
			fontweight='bold',
			va='top',
			ha='right',
		)

	if not vals_by_group:
		ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		return

	parts = ax.violinplot(
		vals_by_group,
		positions=positions,
		showmeans=True,
		showmedians=True,
	)
	for i, pc in enumerate(parts['bodies']):
		pos = positions[i] if i < len(positions) else i
		group_name = group_order[pos] if pos < len(group_order) else None
		pc.set_facecolor(VIOLIN_FACE_COLOR)
		pc.set_alpha(0.9)
		pc.set_edgecolor(VIOLIN_EDGE_COLOR)
		pc.set_linewidth(0.5)
		if group_name == 'Others vs Others':
			pc.set_hatch('//')
	if parts.get('cmeans'):
		parts['cmeans'].set_linestyle('--')
		parts['cmeans'].set_linewidth(0.75)
		parts['cmeans'].set_color('black')
	if parts.get('cmedians'):
		parts['cmedians'].set_linestyle('-')
		parts['cmedians'].set_linewidth(0.75)
		parts['cmedians'].set_color('black')

	# Overlay scatter points colored by patient
	if patients_by_group is not None and len(patients_by_group) == len(vals_by_group):
		for i, (vals, patients) in enumerate(zip(vals_by_group, patients_by_group)):
			if len(vals) == 0 or len(patients) == 0:
				continue
			pos = positions[i]
			# Jitter x for visibility
			np.random.seed(42)
			jitter = 0.04 * (np.random.rand(len(vals)) - 0.5)
			x = np.full(len(vals), pos) + jitter
			for patient in np.unique(patients):
				mask = patients == patient
				color = PATIENT_PALETTE.get(patient, '#666666')
				ax.scatter(x[mask], vals[mask], c=color, s=8, alpha=0.7, zorder=3)

	ax.set_xticks(positions)
	ax.set_xticklabels(labels, rotation=0, ha='center')
	ax.set_ylabel(ylabel)
	if set_ylim:
		set_ylim(ax)
	if add_wilcoxon and len(vals_by_group) == 2 and patients_by_group is not None and len(patients_by_group) == 2:
		add_wilcoxon_signed_rank_bracket(
			ax,
			vals_by_group[0], vals_by_group[1],
			patients_by_group[0], patients_by_group[1],
			positions[0], positions[1],
		)
	ax.grid(axis='y', alpha=0.25, linewidth=0.25)
	# Closed plot boxes (show all spines)
	ax.spines['top'].set_visible(True)
	ax.spines['right'].set_visible(True)


def _find_comparison_csvs(indir: str) -> list[tuple[str, str]]:
	"""Find *_vs_*.csv files. Returns [(comparison_name, path), ...]."""
	pattern = os.path.join(indir, '*_vs_*.csv')
	paths = glob.glob(pattern)
	results = []
	for p in paths:
		basename = os.path.basename(p)
		name = basename[:-4] if basename.endswith('.csv') else basename
		if '_vs_' in name:
			results.append((name, p))
	return sorted(results, key=lambda x: x[0])


def _parse_comparison_name(name: str) -> tuple[str, str]:
	"""Parse 'A_vs_B' -> (A, B)."""
	parts = name.split('_vs_', 1)
	if len(parts) != 2:
		raise ValueError(f'Invalid comparison name: {name}')
	return parts[0], parts[1]


def _load_all_comparisons(indir: str) -> pd.DataFrame | None:
	"""Load all per-comparison CSVs into one DataFrame with 'comparison' column."""
	pairs = _find_comparison_csvs(indir)
	if not pairs:
		return None

	dfs = []
	for comp_name, path in pairs:
		try:
			df = pd.read_csv(path)
			df = df[df['case'].astype(str).str.upper() != 'MEAN']
			df = df[df['case'].astype(str).str.upper() != 'STD']
			df['comparison'] = comp_name
			dfs.append(df)
		except Exception as e:
			print(f'Warning: Could not read {path}: {e}', file=sys.stderr)

	if not dfs:
		return None
	return pd.concat(dfs, ignore_index=True)


def _assign_group(comparison_name: str, reference: str) -> str:
	"""Assign comparison to 'Reference vs Others' or 'Others vs Others'."""
	a, b = _parse_comparison_name(comparison_name)
	if a == reference or b == reference:
		return 'Reference vs Others'
	return 'Others vs Others'


def _get_numeric_metric_columns(df: pd.DataFrame, metrics: list[str] | None) -> list[str]:
	"""Return list of numeric metric columns to plot."""
	exclude = {'case', 'comparison'}
	radius_prefixes = ('volume_error_radii_', 'dice_radii_', 'volume_radii_')
	numeric = []
	for c in df.columns:
		if c in exclude:
			continue
		if any(c.startswith(p) for p in radius_prefixes):
			continue
		if df[c].dtype in (np.float64, np.int64, np.float32, np.int32):
			numeric.append(c)

	if metrics is not None:
		# Preserve order from metrics list (e.g. dice first)
		numeric = [c for c in metrics if c in numeric]
		return numeric
	return sorted(numeric)


def _get_available_references(indir: str) -> set[str]:
	"""Return set of pred names that appear in comparison files."""
	pairs = _find_comparison_csvs(indir)
	names = set()
	for comp_name, _ in pairs:
		a, b = _parse_comparison_name(comp_name)
		names.add(a)
		names.add(b)
	return names


def _compute_point_differences(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
	"""For each case, compute pairwise metric differences within each group.

	Returns: (vals_dict, patients_dict) where each is
	{metric: {'Reference vs Others': array, 'Others vs Others': array}}
	"""
	ref_comps = [
		c for c in df['comparison'].unique()
		if _assign_group(c, reference) == 'Reference vs Others'
	]
	others_comps = [
		c for c in df['comparison'].unique()
		if _assign_group(c, reference) == 'Others vs Others'
	]

	vals_result = {m: {'Reference vs Others': [], 'Others vs Others': []} for m in metric_cols}
	patients_result = {m: {'Reference vs Others': [], 'Others vs Others': []} for m in metric_cols}

	for case in df['case'].unique():
		patient = _case_to_patient(case)
		ref_rows = df[(df['case'] == case) & (df['comparison'].isin(ref_comps))]
		others_rows = df[(df['case'] == case) & (df['comparison'].isin(others_comps))]

		for metric in metric_cols:
			ref_vals = ref_rows.set_index('comparison')[metric].dropna()
			if len(ref_vals) >= 2:
				for (c1, v1), (c2, v2) in itertools.combinations(ref_vals.items(), 2):
					vals_result[metric]['Reference vs Others'].append(abs(v1 - v2))
					patients_result[metric]['Reference vs Others'].append(patient)

			others_vals = others_rows.set_index('comparison')[metric].dropna()
			if len(others_vals) >= 2:
				for (c1, v1), (c2, v2) in itertools.combinations(others_vals.items(), 2):
					vals_result[metric]['Others vs Others'].append(abs(v1 - v2))
					patients_result[metric]['Others vs Others'].append(patient)

	# Average per patient (independent observations for Wilcoxon)
	for metric in metric_cols:
		for group in vals_result[metric]:
			raw_vals = vals_result[metric][group]
			raw_patients = patients_result[metric][group]
			if len(raw_vals) > 0:
				agg_df = pd.DataFrame({'val': raw_vals, 'patient': raw_patients})
				patient_means = agg_df.groupby('patient', as_index=False)['val'].mean()
				vals_result[metric][group] = patient_means['val'].values
				patients_result[metric][group] = patient_means['patient'].values
			else:
				vals_result[metric][group] = np.array([])
				patients_result[metric][group] = np.array([])
	return vals_result, patients_result


def plot_reference_violins(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
	out_path: str,
	figsize_per_metric: tuple[float, float] = (2.4, 2.0),
	format: str = 'png',
) -> None:
	"""Create violin plots: one subplot per metric, two violins (Reference vs Others, Others vs Others).
	Uses Nature journal styling: Arial, 7pt, colorblind-safe palette, 600 DPI."""
	apply_nature_style()
	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows))
	if n == 1:
		axes = np.array([axes])
	axes = axes.flatten()

	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)

	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		data = df[['group', 'case', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
			continue

		vals_by_group = []
		patients_by_group = []
		positions = []
		labels = []
		for i, group in enumerate(group_order):
			group_data = data[data['group'] == group]
			# Average metric per patient (independent observations for Wilcoxon)
			patient_means = group_data.groupby('case', as_index=False)[metric].mean()
			patient_means['patient'] = patient_means['case'].apply(_case_to_patient)
			vals = patient_means[metric].values
			if len(vals) > 0:
				vals_by_group.append(vals)
				patients_by_group.append(patient_means['patient'].values)
				positions.append(i)
				labels.append(tick_labels[group])

		if vals_by_group:
			_draw_violin_ax_reference_style(
				ax,
				vals_by_group,
				positions,
				labels,
				patients_by_group,
				_metric_label(metric),
				group_order=group_order,
				set_ylim=lambda a, m=metric: _set_ylim(a, m),
				subplot_label=chr(97 + idx),
			)
		else:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

	for j in range(idx + 1, len(axes)):
		axes[j].set_visible(False)
	handles, labels = _patient_legend_handles()
	fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0, 0.7), fontsize=6)
	fig.tight_layout()
	save_violin_figure(fig, out_path, format=format or 'png')


def plot_reference_violins_single(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
	out_dir: str,
	figsize: tuple[float, float] = (3.5, 2.5),
	format: str = 'png',
) -> None:
	"""Create one violin plot per metric, saved as separate files. Nature-style."""
	apply_nature_style()
	os.makedirs(out_dir, exist_ok=True)
	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)

	for metric in metric_cols:
		fig, ax = plt.subplots(figsize=figsize)
		data = df[['group', 'case', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		else:
			vals_by_group = []
			patients_by_group = []
			positions = []
			labels = []
			for i, group in enumerate(group_order):
				group_data = data[data['group'] == group]
				# Average metric per patient (independent observations for Wilcoxon)
				patient_means = group_data.groupby('case', as_index=False)[metric].mean()
				patient_means['patient'] = patient_means['case'].apply(_case_to_patient)
				vals = patient_means[metric].values
				if len(vals) > 0:
					vals_by_group.append(vals)
					patients_by_group.append(patient_means['patient'].values)
					positions.append(i)
					labels.append(tick_labels[group])

			if vals_by_group:
				_draw_violin_ax_reference_style(
					ax,
					vals_by_group,
					positions,
					labels,
					patients_by_group,
					_metric_label(metric),
					group_order=group_order,
					set_ylim=lambda a, m=metric: _set_ylim(a, m),
				)
			else:
				ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		handles, labels = _patient_legend_handles()
		fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 0.7), fontsize=6)
		fig.tight_layout()
		safe_name = re.sub(r'[^\w\-]', '_', metric)
		out_path = os.path.join(out_dir, f'reference_violin_{safe_name}.{format}')
		save_violin_figure(fig, out_path, format=format)
		print(f'Saved {out_path}')


def plot_point_diff_violins(
	point_diffs: dict[str, dict[str, np.ndarray]],
	point_patients: dict[str, dict[str, np.ndarray]],
	metric_cols: list[str],
	reference: str,
	out_path: str,
	figsize_per_metric: tuple[float, float] = (2.4, 2.0),
	format: str = 'png',
) -> None:
	"""Violin plots of case-wise pairwise metric differences. Nature-style."""
	apply_nature_style()
	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows))
	if n == 1:
		axes = np.array([axes])
	axes = axes.flatten()

	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)

	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		vals_by_group = []
		patients_by_group = []
		positions = []
		labels = []
		for i, group in enumerate(group_order):
			vals = point_diffs.get(metric, {}).get(group, np.array([]))
			vals = np.asarray(vals)
			if len(vals) > 0:
				vals_by_group.append(vals)
				patients = np.asarray(point_patients.get(metric, {}).get(group, []))
				patients_by_group.append(patients if len(patients) == len(vals) else np.array([]))
				positions.append(i)
				labels.append(tick_labels[group])

		if vals_by_group:
			_draw_violin_ax_reference_style(
				ax,
				vals_by_group,
				positions,
				labels,
				patients_by_group,
				f'{_metric_label(metric)} (|pairwise diff|)',
				group_order=group_order,
				set_ylim=lambda a, m=metric: _set_ylim(a, m),
				subplot_label=chr(97 + idx),
			)
		else:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

	for j in range(idx + 1, len(axes)):
		axes[j].set_visible(False)
	handles, labels = _patient_legend_handles()
	fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 0.7), fontsize=6)
	fig.tight_layout()
	save_violin_figure(fig, out_path, format=format or 'png')


def _format_radar_tick_value(x: float) -> str:
	"""Short label for polar radial ticks (raw metric values)."""
	if not np.isfinite(x):
		return ''
	ax = abs(x)
	if ax >= 1000 or (0 < ax < 0.01):
		return f'{x:.1e}'
	return f'{x:.3g}'


def _radar_radial_distance(
	v: float,
	vmin: float,
	vmax: float,
	lower_is_better: bool,
) -> float:
	"""Radial distance from center: 0 = best for this metric, larger = worse."""
	span = vmax - vmin
	if span <= 0 or not np.isfinite(v):
		return span if span > 0 else 0.0
	if lower_is_better:
		return float(np.clip(v - vmin, 0.0, span))
	return float(np.clip(vmax - v, 0.0, span))


def _radar_axis_config(
	all_vals: np.ndarray,
	lower_is_better: bool,
	*,
	n_ticks: int = 5,
) -> tuple[float, float, float, float, np.ndarray, list[str]]:
	"""Return vmin, vmax (possibly expanded), span, r_max, ytick positions, ytick labels (raw values)."""
	vmin = float(np.nanmin(all_vals))
	vmax = float(np.nanmax(all_vals))
	span = vmax - vmin
	if span <= 0:
		eps = max(1e-12, abs(vmin) * 1e-9 + 1e-12)
		vmin -= eps
		vmax += eps
		span = vmax - vmin
	pad = 0.05 * span
	r_max = span + pad
	tick_r = np.linspace(0.0, span, n_ticks)
	if lower_is_better:
		tick_raw = vmin + tick_r
	else:
		tick_raw = vmax - tick_r
	labels = [_format_radar_tick_value(float(x)) for x in tick_raw]
	return vmin, vmax, span, r_max, tick_r, labels


def _patient_order(patients: set[str]) -> list[str]:
	"""Return patients sorted: P1-P6 first (by PATIENT_PALETTE), then any others."""
	ordered = [p for p in PATIENT_PALETTE if p in patients]
	rest = sorted(patients - set(ordered))
	return ordered + rest


def _radar_radial_digits_to_front(ax) -> None:
	"""Draw radial (numeric) tick labels above filled polygons and lines on polar axes."""
	for t in ax.get_yticklabels():
		t.set_zorder(100)
	# Keep concentric grid faintly behind data
	for gl in ax.yaxis.get_gridlines():
		gl.set_zorder(0)


def plot_reference_radar(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
	out_path: str,
	figsize_per_metric: tuple[float, float] = (4.0, 4.0),
	format: str = 'png',
) -> None:
	"""One figure: grid of radar plots (one per metric), panel labels a, b, c, …

	Each arm = patient; two polygons (Reference vs Others, Others vs Others). Better at center; radial ticks = raw values.
	"""
	apply_nature_style()
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)
	colors = list(NATURE_COLORS.values()) if isinstance(NATURE_COLORS, dict) else NATURE_COLORS

	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(
		nrows,
		ncols,
		figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows),
		subplot_kw=dict(projection='polar'),
	)
	if n == 1:
		axes = np.array([axes])
	axes = axes.flatten()

	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		ax.text(
			-0.35,
			1.12,
			chr(97 + idx),
			transform=ax.transAxes,
			fontsize=12,
			fontweight='bold',
			va='top',
			ha='right',
		)

		data = df[['group', 'case', metric]].dropna(subset=[metric])
		if data.empty:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=9)
			ax.set_axis_off()
			continue

		vals_by_group: dict[str, dict[str, float]] = {g: {} for g in group_order}
		for group in group_order:
			group_data = data[data['group'] == group]
			patient_means = group_data.groupby('case', as_index=False)[metric].mean()
			patient_means['patient'] = patient_means['case'].apply(_case_to_patient)
			patient_means = patient_means.groupby('patient', as_index=False)[metric].mean()
			for _, row in patient_means.iterrows():
				vals_by_group[group][row['patient']] = row[metric]

		patients = set()
		for g in group_order:
			patients.update(vals_by_group[g].keys())
		if not patients:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=9)
			ax.set_axis_off()
			continue

		patient_list = _patient_order(patients)
		n_pat = len(patient_list)
		all_vals = data[metric].dropna().values.astype(float)
		lower_is_better = metric in DISTANCE_METRICS
		vmin, vmax, span, r_max, yticks, yticklabels = _radar_axis_config(all_vals, lower_is_better)

		angles = np.linspace(0, 2 * np.pi, n_pat, endpoint=False).tolist()
		angles += angles[:1]

		for i, group in enumerate(group_order):
			values = []
			for p in patient_list:
				v = vals_by_group[group].get(p, np.nan)
				if np.isnan(v):
					r = span
				else:
					r = _radar_radial_distance(float(v), vmin, vmax, lower_is_better)
				values.append(r)
			values += values[:1]
			ax.plot(angles, values, 'o-', linewidth=1.5, label=tick_labels[group], color=colors[i % len(colors)], zorder=3)
			ax.fill(angles, values, alpha=0.2, color=colors[i % len(colors)], zorder=1)

		ax.set_xticks(angles[:-1])
		ax.set_xticklabels(patient_list, size=12)
		ax.set_ylim(0, r_max)
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticklabels, size=10)
		ax.set_title(_metric_label(metric), fontsize=12)
		_radar_radial_digits_to_front(ax)

	for j in range(len(metric_cols), len(axes)):
		axes[j].set_visible(False)

	legend_handles = [
		Line2D([0], [0], color=colors[0], marker='o', linewidth=1.5, markersize=5, label=tick_labels[group_order[0]]),
		Line2D([0], [0], color=colors[1], marker='o', linewidth=1.5, markersize=5, label=tick_labels[group_order[1]]),
	]
	fig.legend(
		handles=legend_handles,
		loc='lower right',
		bbox_to_anchor=(1.05, 0.4),
		fontsize=10,
		frameon=True,
	)
	if nrows > 1:
		fig.tight_layout(h_pad=0.12)
		fig.subplots_adjust(hspace=-0.22)
	else:
		fig.tight_layout()
	save_violin_figure(fig, out_path, format=format or 'png')
	print(f'Saved {out_path}')


def plot_point_diff_radar(
	point_diffs: dict[str, dict[str, np.ndarray]],
	point_patients: dict[str, dict[str, np.ndarray]],
	metric_cols: list[str],
	reference: str,
	out_path: str,
	figsize_per_metric: tuple[float, float] = (4.0, 4.0),
	format: str = 'png',
) -> None:
	"""One figure: grid of radar plots for |pairwise diff|, panel labels a, b, c, …"""
	apply_nature_style()
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)
	colors = list(NATURE_COLORS.values()) if isinstance(NATURE_COLORS, dict) else NATURE_COLORS

	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(
		nrows,
		ncols,
		figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows),
		subplot_kw=dict(projection='polar'),
	)
	if n == 1:
		axes = np.array([axes])
	axes = axes.flatten()

	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		ax.text(
			-0.35,
			1.12,
			chr(97 + idx),
			transform=ax.transAxes,
			fontsize=12,
			fontweight='bold',
			va='top',
			ha='right',
		)

		vals_by_group: dict[str, dict[str, float]] = {g: {} for g in group_order}
		all_vals_list = []
		for group in group_order:
			vals = point_diffs.get(metric, {}).get(group, np.array([]))
			patients_arr = point_patients.get(metric, {}).get(group, np.array([]))
			vals = np.asarray(vals)
			patients_arr = np.asarray(patients_arr)
			if len(vals) > 0 and len(patients_arr) == len(vals):
				agg = pd.DataFrame({'val': vals, 'patient': patients_arr}).groupby('patient', as_index=False)['val'].mean()
				for _, row in agg.iterrows():
					vals_by_group[group][row['patient']] = row['val']
				all_vals_list.extend(vals.tolist())

		patients = set()
		for g in group_order:
			patients.update(vals_by_group[g].keys())
		if not patients:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=9)
			ax.set_axis_off()
			continue

		patient_list = _patient_order(patients)
		n_pat = len(patient_list)
		all_vals = np.array(all_vals_list, dtype=float) if all_vals_list else np.array([0.0, 1.0])
		vmin, vmax, span, r_max, yticks, yticklabels = _radar_axis_config(all_vals, lower_is_better=True)

		angles = np.linspace(0, 2 * np.pi, n_pat, endpoint=False).tolist()
		angles += angles[:1]

		for i, group in enumerate(group_order):
			values = []
			for p in patient_list:
				v = vals_by_group[group].get(p, np.nan)
				if np.isnan(v):
					r = span
				else:
					r = _radar_radial_distance(float(v), vmin, vmax, lower_is_better=True)
				values.append(r)
			values += values[:1]
			ax.plot(angles, values, 'o-', linewidth=1.5, label=tick_labels[group], color=colors[i % len(colors)], zorder=3)
			ax.fill(angles, values, alpha=0.2, color=colors[i % len(colors)], zorder=1)

		ax.set_xticks(angles[:-1])
		ax.set_xticklabels(patient_list, size=12)
		ax.set_ylim(0, r_max)
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticklabels, size=10)
		ax.set_title(f'{_metric_label(metric)} (|pairwise diff|)', fontsize=12)
		_radar_radial_digits_to_front(ax)

	for j in range(len(metric_cols), len(axes)):
		axes[j].set_visible(False)

	legend_handles = [
		Line2D([0], [0], color=colors[0], marker='o', linewidth=1.5, markersize=5, label=tick_labels[group_order[0]]),
		Line2D([0], [0], color=colors[1], marker='o', linewidth=1.5, markersize=5, label=tick_labels[group_order[1]]),
	]
	fig.legend(
		handles=legend_handles,
		loc='lower right',
		bbox_to_anchor=(0.99, 0.07),
		fontsize=10,
		frameon=True,
	)
	if nrows > 1:
		fig.tight_layout(h_pad=0.12)
		fig.subplots_adjust(hspace=-0.22)
	else:
		fig.tight_layout()
	save_violin_figure(fig, out_path, format=format or 'png')
	print(f'Saved {out_path}')


def plot_point_diff_violins_single(
	point_diffs: dict[str, dict[str, np.ndarray]],
	point_patients: dict[str, dict[str, np.ndarray]],
	metric_cols: list[str],
	reference: str,
	out_dir: str,
	figsize: tuple[float, float] = (3.5, 2.5),
	format: str = 'png',
) -> None:
	"""One violin plot per metric for point differences. Nature-style."""
	apply_nature_style()
	os.makedirs(out_dir, exist_ok=True)
	group_order = ['Reference vs Others', 'Others vs Others']
	tick_labels = _tick_labels(reference)

	for metric in metric_cols:
		fig, ax = plt.subplots(figsize=figsize)
		vals_by_group = []
		patients_by_group = []
		positions = []
		labels = []
		for i, group in enumerate(group_order):
			vals = point_diffs.get(metric, {}).get(group, np.array([]))
			vals = np.asarray(vals)
			if len(vals) > 0:
				vals_by_group.append(vals)
				patients = np.asarray(point_patients.get(metric, {}).get(group, []))
				patients_by_group.append(patients if len(patients) == len(vals) else np.array([]))
				positions.append(i)
				labels.append(tick_labels[group])

		if vals_by_group:
			_draw_violin_ax_reference_style(
				ax,
				vals_by_group,
				positions,
				labels,
				patients_by_group,
				f'{_metric_label(metric)} (|pairwise diff|)',
				group_order=group_order,
				set_ylim=lambda a, m=metric: _set_ylim(a, m),
			)
		else:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		handles, labels = _patient_legend_handles()
		fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 0.7), fontsize=6)
		fig.tight_layout()
		safe_name = re.sub(r'[^\w\-]', '_', metric)
		out_path = os.path.join(out_dir, f'point_diff_violin_{safe_name}.{format}')
		save_violin_figure(fig, out_path, format=format)
		print(f'Saved {out_path}')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Plot pred-vs-pred metrics: reference vs others vs others amongst themselves'
	)
	p.add_argument(
		'input_dir',
		help='Directory containing *_vs_*.csv files from compute_metrics_meshes_pred_vs_pred',
	)
	p.add_argument(
		'--reference',
		required=True,
		help='Name of the prediction to use as reference (e.g. obs1, obs2)',
	)
	p.add_argument(
		'--out-dir',
		default=None,
		help='Output directory for figures (default: input_dir)',
	)
	p.add_argument(
		'--out',
		default=None,
		help='Output path for combined figure (default: reference_violin_metrics.png in out-dir)',
	)
	p.add_argument(
		'--metrics',
		default=None,
		metavar='M1,M2,...',
		help='Comma-separated metrics to plot. Default: %s' % ','.join(DEFAULT_METRICS[:5]) + ',...',
	)
	p.add_argument(
		'--separate',
		action='store_true',
		help='Save one figure per metric instead of a combined grid',
	)
	p.add_argument(
		'--point-comparison',
		action='store_true',
		help='Plot pairwise metric differences per case (e.g. data0_obs0 - data0_obs1) instead of raw values',
	)
	p.add_argument(
		'--radar',
		action='store_true',
		help='Create one combined star/radar figure (≤3 columns): panel labels a,b,c,…; each arm = patient',
	)
	p.add_argument(
		'--format',
		choices=['png', 'pdf'],
		default='png',
		help='Output format: png (600 DPI) or pdf (vector, preferred by Nature). Default: png',
	)
	return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	indir = args.input_dir
	out_dir = args.out_dir or indir
	reference = args.reference

	if not os.path.isdir(indir):
		print(f'Error: Not a directory: {indir}', file=sys.stderr)
		return 1

	df = _load_all_comparisons(indir)
	if df is None or df.empty:
		print(f'Error: No *_vs_*.csv files found in {indir}', file=sys.stderr)
		return 2

	available = _get_available_references(indir)
	if reference not in available:
		print(f'Error: Reference "{reference}" not found. Available: {sorted(available)}', file=sys.stderr)
		return 3

	df['group'] = df['comparison'].apply(lambda c: _assign_group(c, reference))

	metrics_to_plot = [m.strip() for m in args.metrics.split(',')] if args.metrics else DEFAULT_METRICS
	metric_cols = _get_numeric_metric_columns(df, metrics_to_plot)
	if not metric_cols:
		print('Error: No matching numeric metric columns found.', file=sys.stderr)
		return 4

	for m in metric_cols:
		if m in RELATIVE_ERROR_METRICS and m in df.columns:
			df[m] = np.abs(df[m])

	os.makedirs(out_dir, exist_ok=True)

	fmt = args.format
	if args.radar:
		if args.point_comparison:
			point_diffs, point_patients = _compute_point_differences(df, metric_cols, reference)
			out_path = args.out or os.path.join(out_dir, f'point_diff_radar_metrics_{reference}.png')
			plot_point_diff_radar(point_diffs, point_patients, metric_cols, reference, out_path, format=fmt)
		else:
			out_path = args.out or os.path.join(out_dir, f'reference_radar_metrics_{reference}.png')
			plot_reference_radar(df, metric_cols, reference, out_path, format=fmt)
	elif args.point_comparison:
		point_diffs, point_patients = _compute_point_differences(df, metric_cols, reference)
		if args.separate:
			plot_point_diff_violins_single(point_diffs, point_patients, metric_cols, reference, out_dir, format=fmt)
		else:
			out_path = args.out or os.path.join(out_dir, f'point_diff_violin_metrics_{reference}.png')
			plot_point_diff_violins(point_diffs, point_patients, metric_cols, reference, out_path, format=fmt)
			base, _ = os.path.splitext(out_path)
			print(f'Saved {base}.{fmt}')
		latex_path = os.path.join(out_dir, f'latex_table_point_diff_{reference}.txt')
		write_point_diff_latex_table(
			point_diffs, metric_cols, reference, latex_path,
			metric_display_names=METRIC_DISPLAY_NAMES,
			point_patients=point_patients,
		)
		print(f'Wrote {latex_path}')
	else:
		if args.separate:
			plot_reference_violins_single(df, metric_cols, reference, out_dir, format=fmt)
		else:
			out_path = args.out or os.path.join(out_dir, f'reference_violin_metrics_{reference}.png')
			plot_reference_violins(df, metric_cols, reference, out_path, format=fmt)
			base, _ = os.path.splitext(out_path)
			print(f'Saved {base}.{fmt}')
		latex_path = os.path.join(out_dir, f'latex_table_{reference}.txt')
		write_pred_vs_pred_reference_latex_table(
			df, metric_cols, reference, latex_path,
			metric_display_names=METRIC_DISPLAY_NAMES,
		)
		print(f'Wrote {latex_path}')

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
