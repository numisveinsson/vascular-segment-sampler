"""Plot scalar mesh metrics from compute_metrics_meshes_comparison output as violin plots.

Reads the per-prediction-folder per-case CSV files written by
compute_metrics_meshes_comparison.py in --predictions-root mode (e.g. mesh_metrics_mc.csv,
mesh_metrics_taubin.csv) and creates violin plots showing the distribution of each scalar
metric across prediction folders (one violin per folder per metric). A single-folder run
(mesh_metrics.csv) is also supported and shown as one violin.

This is the GT-vs-pred counterpart to plot_pred_vs_pred_metrics.py (which reads *_vs_*.csv).
For the per-radius-bucket metrics (volume_radii / dice_radii) use plot_volume_per_radii.py /
plot_dice_per_radii.py instead, which read summary.csv.

Usage:
	python -m analysis.plot_mesh_metrics /path/to/output_dir
	python -m analysis.plot_mesh_metrics /path/to/output_dir --out-dir /path/to/figures
	python -m analysis.plot_mesh_metrics /path/to/output_dir --metrics mean_curvature_rms,dihedral_angle_p95_deg
	python -m analysis.plot_mesh_metrics /path/to/output_dir --separate --format pdf

	# per anatomy: group violins by category (x-axis), methods shown as colors
	python -m analysis.plot_mesh_metrics /path/to/output_dir --categories analysis/categories.json

Rows with status='diverged' (meshes that blew up during smoothing) carry NaN metrics and are
automatically dropped from the plots.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
	sys.path.insert(0, _repo_root)

# Reuse the shared metric metadata and y-limit helpers so labels/axes stay consistent
# with the pred-vs-pred plots (and so newly added metrics like smoothness are picked up).
from analysis.plot_pred_vs_pred_metrics import (  # noqa: E402
	DEFAULT_METRICS,
	RELATIVE_ERROR_METRICS,
	_get_numeric_metric_columns,
	_metric_label,
	_set_ylim,
)
from analysis.plotting.violin_plot_functions import (  # noqa: E402
	apply_nature_style,
	draw_violin_ax,
	get_nature_colors,
	save_violin_figure,
)


def _find_per_folder_csvs(indir: str, stem: str) -> list[tuple[str, str]]:
	"""Find per-prediction-folder CSVs. Returns [(folder_name, path), ...].

	Matches ``<stem>_<folder>.csv`` (predictions-root mode) and, if present, a plain
	``<stem>.csv`` from a single-pred run (labelled 'pred'). Excludes the raw per-radii
	CSVs (volume_radii_raw_*, dice_radii_raw_*) and summary.csv.
	"""
	results: list[tuple[str, str]] = []
	for path in sorted(glob.glob(os.path.join(indir, f'{stem}_*.csv'))):
		name = os.path.basename(path)[:-4]  # strip .csv
		folder = name[len(stem) + 1:]  # strip '<stem>_'
		if not folder:
			continue
		results.append((folder, path))
	single = os.path.join(indir, f'{stem}.csv')
	if os.path.isfile(single):
		results.append(('pred', single))
	return sorted(results, key=lambda x: x[0])


def _load_all_folders(indir: str, stem: str) -> pd.DataFrame | None:
	"""Load all per-folder CSVs into one DataFrame with a 'pred_folder' column."""
	pairs = _find_per_folder_csvs(indir, stem)
	if not pairs:
		return None

	dfs = []
	for folder, path in pairs:
		try:
			df = pd.read_csv(path)
		except Exception as e:
			print(f'Warning: Could not read {path}: {e}', file=sys.stderr)
			continue
		if 'case' in df.columns:
			# Exclude appended MEAN/STD summary rows
			df = df[~df['case'].astype(str).str.upper().isin({'MEAN', 'STD'})]
		# Note: diverged rows (status='diverged', NaN metrics) are intentionally KEPT here so
		# failure counts can be reported; they drop out of distributions via per-metric dropna.
		df['pred_folder'] = folder
		dfs.append(df)

	if not dfs:
		return None
	df = pd.concat(dfs, ignore_index=True)
	for m in RELATIVE_ERROR_METRICS:
		if m in df.columns:
			df[m] = np.abs(df[m])
	return df


def _format_folder_label(name: str) -> str:
	return name.replace('_', ' ')


def _diverged_counts(df: pd.DataFrame, group_col: str) -> dict[str, tuple[int, int]]:
	"""Per group: (n_diverged_cases, n_total_cases). Failure = status == 'diverged'."""
	out: dict[str, tuple[int, int]] = {}
	has_status = 'status' in df.columns
	for g, sub in df.groupby(group_col):
		total = sub['case'].nunique() if 'case' in sub.columns else len(sub)
		ndiv = 0
		if has_status:
			div = sub[sub['status'].astype(str).str.lower() == 'diverged']
			ndiv = div['case'].nunique() if 'case' in div.columns else len(div)
		out[str(g)] = (int(ndiv), int(total))
	return out


def _diverged_counts_2d(df: pd.DataFrame, row_col: str, col_col: str) -> dict[tuple[str, str], int]:
	"""(row_value, col_value) -> n_diverged_cases. Failure = status == 'diverged'."""
	out: dict[tuple[str, str], int] = {}
	if 'status' not in df.columns:
		return out
	div = df[df['status'].astype(str).str.lower() == 'diverged']
	for (rv, cv), sub in div.groupby([row_col, col_col]):
		n = sub['case'].nunique() if 'case' in sub.columns else len(sub)
		out[(str(rv), str(cv))] = int(n)
	return out


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
	"""Rows excluding diverged/failed cases (used for distributions, not for counts)."""
	if 'status' in df.columns:
		return df[df['status'].astype(str).str.lower() != 'diverged']
	return df


def _label_with_failures(name: str, counts: dict[str, tuple[int, int]]) -> str:
	"""Method label with appended 'k/N failed' only when at least one case diverged."""
	base = _format_folder_label(name)
	if name in counts:
		nd, nt = counts[name]
		if nd > 0:
			base += f'\n{nd}/{nt} failed'
	return base


def _assign_categories(
	df: pd.DataFrame,
	categories_path: str,
	drop_uncategorized: bool,
) -> tuple[pd.DataFrame, list[str]]:
	"""Add an anatomy 'category' column from categories.json {category: [cases]}.

	Returns (df_with_category, ordered_category_names). Category order follows the JSON
	file; cases not listed become 'uncategorized' (placed last) unless dropped.
	"""
	with open(categories_path) as f:
		data = json.load(f)
	case_to_cat: dict[str, str] = {}
	order: list[str] = []
	for cat, cases in data.items():
		order.append(cat)
		for c in cases:
			case_to_cat[str(c)] = cat

	df = df.copy()
	df['category'] = df['case'].astype(str).map(case_to_cat)
	if drop_uncategorized:
		df = df[df['category'].notna()]
	else:
		df['category'] = df['category'].fillna('uncategorized')

	present = set(df['category'].unique())
	categories = [c for c in order if c in present]
	if not drop_uncategorized and 'uncategorized' in present:
		categories.append('uncategorized')
	return df, categories


def _inline_label_with_failures(name: str, counts: dict[str, tuple[int, int]]) -> str:
	"""Single-line method label for legends: 'method (k/N failed)' only when any diverged."""
	base = _format_folder_label(name)
	if name in counts:
		nd, nt = counts[name]
		if nd > 0:
			base += f' ({nd}/{nt} failed)'
	return base


def _draw_failure_panel(
	ax,
	df: pd.DataFrame,
	categories: list[str],
	methods: list[str],
	method_colors: dict[str, str],
	rotation: int,
	as_percent: bool = False,
) -> None:
	"""Grouped bar panel of diverged-case counts per anatomy category and method."""
	div2d = _diverged_counts_2d(df, 'category', 'pred_folder')
	totals = {}
	if as_percent:
		for cat in categories:
			for m in methods:
				sub = df[(df['category'] == cat) & (df['pred_folder'] == m)]
				totals[(cat, m)] = sub['case'].nunique() if 'case' in sub.columns else len(sub)

	n_m = max(len(methods), 1)
	width = 0.8 / n_m
	x = np.arange(len(categories))
	any_fail = False
	for mi, method in enumerate(methods):
		heights = []
		for cat in categories:
			n = div2d.get((cat, method), 0)
			if as_percent:
				tot = totals.get((cat, method), 0)
				heights.append(100.0 * n / tot if tot else 0.0)
			else:
				heights.append(n)
			if n > 0:
				any_fail = True
		offset = (mi - (n_m - 1) / 2.0) * width
		ax.bar(x + offset, heights, width * 0.9, color=method_colors.get(method, '#666666'), label=_format_folder_label(method))
	ax.set_xticks(range(len(categories)))
	ax.set_xticklabels([_format_folder_label(c) for c in categories], rotation=rotation, ha='right')
	ax.set_ylabel('Diverged cases (%)' if as_percent else 'Diverged cases (count)')
	ax.set_title('Failures (diverged)')
	ax.grid(axis='y', alpha=0.25, linewidth=0.25)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	if not any_fail:
		ax.text(0.5, 0.9, 'no failures', ha='center', va='top', transform=ax.transAxes, fontsize=6)


def _draw_grouped_violins(
	ax,
	df: pd.DataFrame,
	metric: str,
	categories: list[str],
	methods: list[str],
	method_colors: dict[str, str],
	rotation: int,
	set_ylim_fn=None,
	show_legend: bool = True,
	method_labels: dict[str, str] | None = None,
) -> None:
	"""Violins grouped by anatomy category on the x-axis, one colored sub-violin per method."""
	from matplotlib.patches import Patch

	n_m = max(len(methods), 1)
	width = 0.8 / n_m
	df_ok = _ok_rows(df)
	vals_all: list[np.ndarray] = []
	positions: list[float] = []
	body_colors: list[str] = []
	for ci, cat in enumerate(categories):
		for mi, method in enumerate(methods):
			sub = df_ok[(df_ok['category'] == cat) & (df_ok['pred_folder'] == method)]
			v = sub[metric].dropna().values
			if len(v) == 0:
				continue
			offset = (mi - (n_m - 1) / 2.0) * width
			positions.append(ci + offset)
			vals_all.append(v)
			body_colors.append(method_colors.get(method, '#666666'))

	ax.set_ylabel(_metric_label(metric))
	if not vals_all:
		ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		return

	parts = ax.violinplot(vals_all, positions=positions, widths=width * 0.9, showmeans=False, showmedians=True)
	for pc, c in zip(parts['bodies'], body_colors):
		pc.set_facecolor(c)
		pc.set_alpha(0.75)
		pc.set_edgecolor(c)
		pc.set_linewidth(0.5)
	if parts.get('cmedians') is not None:
		parts['cmedians'].set_linewidth(0.75)
		parts['cmedians'].set_color('black')

	ax.set_xticks(range(len(categories)))
	ax.set_xticklabels([_format_folder_label(c) for c in categories], rotation=rotation, ha='right')
	if set_ylim_fn:
		set_ylim_fn(ax)
	ax.grid(axis='y', alpha=0.25, linewidth=0.25)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	if show_legend and len(methods) > 1:
		handles = [
			Patch(
				facecolor=method_colors.get(m, '#666666'), edgecolor=method_colors.get(m, '#666666'), alpha=0.75,
				label=(method_labels.get(m) if method_labels else _format_folder_label(m)),
			)
			for m in methods
		]
		ax.legend(handles=handles, fontsize=5, frameon=False, loc='best')


def plot_violins_by_category(
	df: pd.DataFrame,
	metric_cols: list[str],
	categories: list[str],
	methods: list[str],
	method_colors: dict[str, str],
	out_path: str,
	figsize_per_metric: tuple[float, float] = (3.0, 2.2),
	rotation: int = 20,
	fmt: str = 'png',
) -> None:
	"""Combined grid: one subplot per metric, anatomy categories on x-axis, methods as colors."""
	import matplotlib.pyplot as plt

	apply_nature_style()
	fail_counts = _diverged_counts(df, 'pred_folder')
	method_labels = {m: _inline_label_with_failures(m, fail_counts) for m in methods}
	# One panel per metric, plus a final "Failures" panel
	n = len(metric_cols) + 1
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows))
	axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

	for idx, metric in enumerate(metric_cols):
		_draw_grouped_violins(
			axes[idx], df, metric, categories, methods, method_colors, rotation,
			set_ylim_fn=lambda a, m=metric: _set_ylim(a, m),
			show_legend=(idx == 0),
			method_labels=method_labels,
		)
	_draw_failure_panel(axes[len(metric_cols)], df, categories, methods, method_colors, rotation)
	for j in range(len(metric_cols) + 1, len(axes)):
		axes[j].set_visible(False)
	fig.tight_layout()
	save_violin_figure(fig, out_path, format=fmt)


def plot_violins_by_category_single_figure(
	df: pd.DataFrame,
	metric_cols: list[str],
	categories: list[str],
	methods: list[str],
	method_colors: dict[str, str],
	out_dir: str,
	figsize: tuple[float, float] = (4.5, 2.8),
	rotation: int = 20,
	fmt: str = 'png',
) -> None:
	"""One figure per metric, anatomy categories on x-axis, methods as colors."""
	import matplotlib.pyplot as plt

	os.makedirs(out_dir, exist_ok=True)
	apply_nature_style()
	fail_counts = _diverged_counts(df, 'pred_folder')
	method_labels = {m: _inline_label_with_failures(m, fail_counts) for m in methods}
	for metric in metric_cols:
		fig, ax = plt.subplots(figsize=figsize)
		_draw_grouped_violins(
			ax, df, metric, categories, methods, method_colors, rotation,
			set_ylim_fn=lambda a, m=metric: _set_ylim(a, m),
			show_legend=True,
			method_labels=method_labels,
		)
		fig.tight_layout()
		safe_name = re.sub(r'[^\w\-]', '_', metric)
		out_path = os.path.join(out_dir, f'violin_{safe_name}_by_category.{fmt}')
		save_violin_figure(fig, out_path, format=fmt)
		print(f'Saved {out_path}')

	# Dedicated failures figure
	fig, ax = plt.subplots(figsize=figsize)
	_draw_failure_panel(ax, df, categories, methods, method_colors, rotation)
	if len(methods) > 1:
		ax.legend(fontsize=6, frameon=False, loc='best')
	fig.tight_layout()
	out_path = os.path.join(out_dir, f'failures_by_category.{fmt}')
	save_violin_figure(fig, out_path, format=fmt)
	print(f'Saved {out_path}')


def plot_violins(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_path: str,
	figsize_per_metric: tuple[float, float] = (2.4, 2.0),
	rotation: int = 15,
	fmt: str = 'png',
) -> None:
	"""Combined grid: one subplot per metric, prediction folders on the x-axis."""
	import matplotlib.pyplot as plt

	apply_nature_style()
	n = len(metric_cols)
	ncols = min(3, n)
	nrows = (n + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_metric[0] * ncols, figsize_per_metric[1] * nrows))
	axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

	folders = sorted(df['pred_folder'].unique())
	colors = get_nature_colors(len(folders))
	fail_counts = _diverged_counts(df, 'pred_folder')
	df_ok = _ok_rows(df)

	has_case = 'case' in df_ok.columns

	idx = -1
	for idx, metric in enumerate(metric_cols):
		ax = axes[idx]
		cols = ['pred_folder', 'case', metric] if has_case else ['pred_folder', metric]
		data = df_ok[cols].dropna(subset=[metric])
		groups = []
		for c in folders:
			sub = data[data['pred_folder'] == c]
			if len(sub):
				cases = sub['case'].astype(str).values if has_case else None
				groups.append((c, sub[metric].values, cases))
		if not groups:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
			continue
		folders_used, values_list, cases_list = zip(*groups)
		positions = list(range(len(folders_used)))
		labels = [_label_with_failures(c, fail_counts) for c in folders_used]
		patients = list(cases_list) if has_case else None
		draw_violin_ax(
			ax,
			list(values_list),
			positions,
			labels,
			colors,
			_metric_label(metric),
			group_order=None,
			set_ylim=lambda a, m=metric: _set_ylim(a, m),
			add_wilcoxon=2 <= len(folders_used) <= 3,
			patients_by_group=patients,
			subplot_label=chr(97 + idx),
			xtick_rotation=rotation,
		)

	for j in range(idx + 1, len(axes)):
		axes[j].set_visible(False)
	fig.tight_layout()
	save_violin_figure(fig, out_path, format=fmt)


def plot_violins_single_figure(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_dir: str,
	figsize: tuple[float, float] = (3.5, 2.5),
	rotation: int = 15,
	fmt: str = 'png',
) -> None:
	"""One figure per metric saved separately."""
	import matplotlib.pyplot as plt

	os.makedirs(out_dir, exist_ok=True)
	apply_nature_style()
	folders = sorted(df['pred_folder'].unique())
	colors = get_nature_colors(len(folders))
	fail_counts = _diverged_counts(df, 'pred_folder')
	df_ok = _ok_rows(df)
	has_case = 'case' in df_ok.columns
	for metric in metric_cols:
		fig, ax = plt.subplots(figsize=figsize)
		cols = ['pred_folder', 'case', metric] if has_case else ['pred_folder', metric]
		data = df_ok[cols].dropna(subset=[metric])
		groups = []
		for c in folders:
			sub = data[data['pred_folder'] == c]
			if len(sub):
				cases = sub['case'].astype(str).values if has_case else None
				groups.append((c, sub[metric].values, cases))
		if not groups:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
		else:
			folders_used, values_list, cases_list = zip(*groups)
			positions = list(range(len(folders_used)))
			labels = [_label_with_failures(c, fail_counts) for c in folders_used]
			patients = list(cases_list) if has_case else None
			draw_violin_ax(
				ax,
				list(values_list),
				positions,
				labels,
				colors,
				_metric_label(metric),
				group_order=None,
				set_ylim=lambda a, m=metric: _set_ylim(a, m),
				add_wilcoxon=2 <= len(folders_used) <= 3,
				patients_by_group=patients,
				xtick_rotation=rotation,
			)
		fig.tight_layout()
		safe_name = re.sub(r'[^\w\-]', '_', metric)
		out_path = os.path.join(out_dir, f'violin_{safe_name}.{fmt}')
		save_violin_figure(fig, out_path, format=fmt)
		print(f'Saved {out_path}')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Plot scalar mesh metrics (GT vs pred) as violin plots, grouped by prediction folder'
	)
	p.add_argument(
		'input_dir',
		help='Directory containing mesh_metrics_<folder>.csv files from compute_metrics_meshes_comparison (--predictions-root mode)',
	)
	p.add_argument(
		'--stem',
		default='mesh_metrics',
		help="Filename stem of the per-folder CSVs (default: mesh_metrics, i.e. mesh_metrics_<folder>.csv)",
	)
	p.add_argument('--out-dir', default=None, help='Output directory for figures (default: input_dir)')
	p.add_argument('--out', default=None, help='Output path for combined figure (default: violin_mesh_metrics.<fmt> in out-dir)')
	p.add_argument('--metrics', default=None, metavar='M1,M2,...', help='Comma-separated metrics to plot. Default: standard scalar metric set.')
	p.add_argument('--categories', default=None, metavar='JSON', help='Path to categories.json {category: [cases]}. If set, violins are grouped by anatomy category on the x-axis (methods shown as colors).')
	p.add_argument('--drop-uncategorized', action='store_true', help='With --categories, drop cases not listed in the JSON instead of grouping them as "uncategorized".')
	p.add_argument('--separate', action='store_true', help='Save one figure per metric instead of a combined grid')
	p.add_argument('--rotation', type=int, default=15, help='Rotation for x-axis labels (default: 15)')
	p.add_argument('--format', choices=['png', 'pdf'], default='png', help='Output format: png (600 DPI) or pdf. Default: png')
	return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	indir = args.input_dir
	out_dir = args.out_dir or indir

	if not os.path.isdir(indir):
		print(f'Error: Not a directory: {indir}', file=sys.stderr)
		return 1

	df = _load_all_folders(indir, args.stem)
	if df is None or df.empty:
		print(f'Error: No {args.stem}_*.csv (or {args.stem}.csv) files found in {indir}', file=sys.stderr)
		return 2

	metrics_to_plot = [m.strip() for m in args.metrics.split(',')] if args.metrics else DEFAULT_METRICS
	metric_cols = _get_numeric_metric_columns(df, metrics_to_plot)
	# Grouping columns are never metrics to plot
	metric_cols = [c for c in metric_cols if c not in ('pred_folder', 'category')]
	if not metric_cols:
		print('Error: No matching numeric metric columns found.', file=sys.stderr)
		return 3

	os.makedirs(out_dir, exist_ok=True)
	fmt = args.format

	# Per-anatomy mode: group violins by category on the x-axis, methods as colors
	if args.categories:
		if not os.path.isfile(args.categories):
			print(f'Error: Categories file not found: {args.categories}', file=sys.stderr)
			return 1
		df, categories = _assign_categories(df, args.categories, args.drop_uncategorized)
		if not categories:
			print('Error: No cases matched any category in the categories JSON.', file=sys.stderr)
			return 3
		methods = sorted(df['pred_folder'].unique())
		method_colors = dict(zip(methods, get_nature_colors(len(methods))))
		if args.separate:
			plot_violins_by_category_single_figure(df, metric_cols, categories, methods, method_colors, out_dir, rotation=args.rotation, fmt=fmt)
		else:
			out_path = args.out or os.path.join(out_dir, f'violin_mesh_metrics_by_category.{fmt}')
			plot_violins_by_category(df, metric_cols, categories, methods, method_colors, out_path, rotation=args.rotation, fmt=fmt)
			print(f'Saved {out_path}')
		return 0

	if args.separate:
		plot_violins_single_figure(df, metric_cols, out_dir, rotation=args.rotation, fmt=fmt)
	else:
		out_path = args.out or os.path.join(out_dir, f'violin_mesh_metrics.{fmt}')
		plot_violins(df, metric_cols, out_path, rotation=args.rotation, fmt=fmt)
		print(f'Saved {out_path}')

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
