"""Plot dice-per-radii metrics as violin plots (from compute_metrics_meshes_comparison.py).

Creates grouped violin plots showing the per-case distribution of Dice within each
radii bucket: x-axis = radii buckets, one colored sub-violin per method. A p-value
comparing the methods is computed per radii bucket and annotated above each group
(paired Wilcoxon signed-rank for two methods, Kruskal-Wallis omnibus for more).

The violins require the per-case distributions, so this reads the per-prediction-folder
per-case CSV files (``mesh_metrics_<folder>.csv``) written by
compute_metrics_meshes_comparison.py, not the aggregated summary.csv. Pass either the
directory containing them or any file inside it (e.g. summary.csv).

Usage:
	python -m analysis.plot_dice_per_radii /path/to/output_dir
	python -m analysis.plot_dice_per_radii /path/to/output_dir --out dice_radii.png

	# per anatomy: one grouped-violin subplot per category, re-aggregating the per-case
	# mesh_metrics_<folder>.csv files
	python -m analysis.plot_dice_per_radii /path/to/output_dir --categories analysis/categories.json --out dice_radii_by_anatomy.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
	sys.path.insert(0, _repo_root)

from scipy.stats import kruskal, mannwhitneyu, wilcoxon  # noqa: E402

from analysis.plotting.violin_plot_functions import (  # noqa: E402
	apply_nature_style,
	format_pvalue,
	get_nature_colors,
	save_violin_figure,
)

# Font sizes (points). Larger than the Nature defaults for better on-screen readability.
_ANNOT_FONTSIZE = 12
_LEGEND_FONTSIZE = 15

# Axis label for the radius-bucket axis; the unit lives here (not on each tick).
_RADIUS_AXIS_LABEL = 'Radius (mm)'


def _format_bucket_range(label: str) -> str:
	"""Format a raw bucket label (e.g. '0.0_0.1', '0.4_inf') as a clear mm range.

	Bucket bounds are scaled by 10 for display (e.g. 0.1 -> 1.0 mm). Returns a closed
	range like '0.0 - 1.0' for finite buckets and '≥ 4.0' for the open-ended bucket.
	"""
	parts = label.split('_')
	if len(parts) >= 2:
		lo = float(parts[0]) * 10
		if parts[1] == 'inf':
			return f'≥ {lo:.1f}'
		hi = float(parts[1]) * 10
		return f'{lo:.1f} – {hi:.1f}'
	return label


def _apply_style() -> None:
	"""Apply the Nature base style, then enlarge fonts for readability."""
	apply_nature_style()
	plt.rcParams.update({
		'font.size': 14,
		'axes.labelsize': 16,
		'axes.titlesize': 16,
		'xtick.labelsize': 13,
		'ytick.labelsize': 13,
		'legend.fontsize': _LEGEND_FONTSIZE,
	})


def _format_method_display_name(folder: str) -> str:
	"""Map pred_folder to professional display name."""
	folder = str(folder).strip()
	if folder == 'global_pred':
		return 'EGNN (ours)'
	if folder == 'mc':
		return 'Marching Cubes'
	# Taubin variants: taubin_on_mc_04_50 -> Taubin -0.4/50, taubin_on_mc_025_400 -> Taubin -0.25/400
	m = re.match(r'.*taubin.*_(\d+)_(\d+)$', folder, re.IGNORECASE)
	if m:
		a, b = m.group(1), m.group(2)
		# Convert 04 -> 0.4, 025 -> 0.25, 02 -> 0.2
		if a.startswith('0') and len(a) >= 2:
			val = int(a) / (10 ** (len(a) - 1))
		else:
			val = int(a)
		return f'Taubin {val}/{b}'
	if 'taubin' in folder.lower():
		return 'Taubin'
	return folder


def _method_order_key(folder: str) -> int:
	"""Sort key: MC first, then Taubin, then EGNN (global_pred), then others."""
	f = str(folder).strip().lower()
	return 0 if f == 'mc' else 1 if 'taubin' in f else 2 if f == 'global_pred' else 3


def _assign_categories(
	df: pd.DataFrame,
	categories_path: str,
	drop_uncategorized: bool,
) -> tuple[pd.DataFrame, list[str]]:
	"""Add an anatomy 'category' column from categories.json {category: [cases]}.

	Returns (df_with_category, ordered_category_names). Category order follows the JSON;
	unlisted cases become 'uncategorized' (placed last) unless dropped.
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


def _find_per_case_csvs(indir: str, stem: str) -> list[tuple[str, str]]:
	"""Per-prediction-folder per-case CSVs: [(folder, path), ...]. Includes a plain <stem>.csv."""
	results: list[tuple[str, str]] = []
	for path in sorted(glob.glob(os.path.join(indir, f'{stem}_*.csv'))):
		folder = os.path.basename(path)[:-4][len(stem) + 1:]
		if folder:
			results.append((folder, path))
	single = os.path.join(indir, f'{stem}.csv')
	if os.path.isfile(single):
		results.append(('pred', single))
	return results


def _load_per_case(indir: str, stem: str) -> pd.DataFrame | None:
	"""Load all per-folder per-case CSVs into one DataFrame with a 'pred_folder' column.

	Skips MEAN/STD summary rows and drops diverged cases (status='diverged').
	"""
	dfs = []
	for folder, path in _find_per_case_csvs(indir, stem):
		try:
			df = pd.read_csv(path)
		except Exception as e:
			print(f'Warning: Could not read {path}: {e}', file=sys.stderr)
			continue
		if 'case' in df.columns:
			df = df[~df['case'].astype(str).str.upper().isin({'MEAN', 'STD'})]
		# Keep diverged rows (status='diverged', NaN buckets) so failures can be reported;
		# they drop out of the per-bucket means via to_numeric(...).dropna().
		df['pred_folder'] = folder
		dfs.append(df)
	if not dfs:
		return None
	return pd.concat(dfs, ignore_index=True)


def _diverged_2d(df: pd.DataFrame) -> dict[tuple[str, str], int]:
	"""(category, pred_folder) -> n_diverged_cases. Failure = status == 'diverged'."""
	out: dict[tuple[str, str], int] = {}
	if 'status' not in df.columns or 'category' not in df.columns:
		return out
	div = df[df['status'].astype(str).str.lower() == 'diverged']
	for (cat, method), sub in div.groupby(['category', 'pred_folder']):
		n = sub['case'].nunique() if 'case' in sub.columns else len(sub)
		out[(str(cat), str(method))] = int(n)
	return out


def _diverged_total(df: pd.DataFrame, method: str) -> tuple[int, int]:
	"""(n_diverged, n_total) cases for a method across all categories."""
	sub = df[df['pred_folder'].astype(str) == method]
	total = sub['case'].nunique() if 'case' in sub.columns else len(sub)
	ndiv = 0
	if 'status' in sub.columns:
		div = sub[sub['status'].astype(str).str.lower() == 'diverged']
		ndiv = div['case'].nunique() if 'case' in div.columns else len(div)
	return int(ndiv), int(total)


def _percase_dice_radii_columns(df: pd.DataFrame) -> list[str]:
	"""Per-case dice-per-radii bucket columns (e.g. 'dice_radii_0.0_0.1'), excluding _mean/_std."""
	cols = [
		c for c in df.columns
		if c.startswith('dice_radii_') and not c.endswith('_mean') and not c.endswith('_std')
	]
	return sorted(cols)


def _bucket_pvalue(
	sub: pd.DataFrame,
	bucket_col: str,
	methods_raw: list[str],
) -> float | None:
	"""Compare methods within a single radii bucket and return a p-value (or None).

	Two methods: paired Wilcoxon signed-rank matched by case (falls back to the
	unpaired Mann-Whitney U test if pairing/Wilcoxon is not possible). More than two
	methods: Kruskal-Wallis omnibus test across methods. Returns None if there is not
	enough data to compute a test.
	"""
	per_method: dict[str, pd.Series] = {}
	for m in methods_raw:
		md = sub[sub['pred_folder'].astype(str) == m]
		vals = pd.to_numeric(md[bucket_col], errors='coerce')
		if 'case' in md.columns:
			s = pd.Series(vals.values, index=md['case'].astype(str).values).dropna()
			s = s[~s.index.duplicated(keep='first')]
		else:
			s = pd.Series(vals.dropna().values)
		if len(s):
			per_method[m] = s
	if len(per_method) < 2:
		return None

	present = [m for m in methods_raw if m in per_method]
	if len(present) == 2:
		a, b = per_method[present[0]], per_method[present[1]]
		common = a.index.intersection(b.index)
		if len(common) >= 2:
			pa, pb = a.loc[common].values, b.loc[common].values
			if np.allclose(pa, pb):
				return 1.0
			try:
				_, p = wilcoxon(pa, pb, alternative='two-sided')
				return float(p)
			except Exception:
				pass
		try:
			_, p = mannwhitneyu(a.values, b.values, alternative='two-sided')
			return float(p)
		except Exception:
			return None

	arrays = [per_method[m].values for m in present if len(per_method[m]) >= 1]
	if len(arrays) < 2:
		return None
	try:
		_, p = kruskal(*arrays)
		return float(p)
	except Exception:
		return None


def _annotate_bucket_pvalues(
	ax,
	sub: pd.DataFrame,
	cols: list[str],
	methods_raw: list[str],
) -> None:
	"""Annotate the p-value comparing methods above each radii-bucket group of violins."""
	ymin, ymax = ax.get_ylim()
	yspan = ymax - ymin if ymax > ymin else 1.0
	texts: list[tuple[int, float, str]] = []
	top_needed = ymax
	for bi, col in enumerate(cols):
		p = _bucket_pvalue(sub, col, methods_raw)
		if p is None:
			continue
		local_max = ymin
		for method in methods_raw:
			md = sub[sub['pred_folder'].astype(str) == method]
			v = pd.to_numeric(md[col], errors='coerce').dropna().values
			if len(v):
				local_max = max(local_max, float(np.max(v)))
		y = local_max + 0.03 * yspan
		texts.append((bi, y, format_pvalue(p)))
		top_needed = max(top_needed, y + 0.06 * yspan)
	if texts:
		ax.set_ylim(ymin, top_needed)
		for bi, y, txt in texts:
			ax.text(bi, y, txt, ha='center', va='bottom', fontsize=_ANNOT_FONTSIZE, rotation=0)


def _draw_grouped_violins(
	ax,
	sub: pd.DataFrame,
	methods_raw: list[str],
	methods_display: list[str],
	cols: list[str],
	display_labels: list[str],
	ylabel: str,
	method_colors: dict[str, str],
	title: str | None = None,
	show_legend: bool = True,
	add_pvalue: bool = True,
) -> None:
	"""Grouped violins on one axes: x = radius buckets, one colored sub-violin per method."""
	from matplotlib.patches import Patch

	n_m = max(len(methods_raw), 1)
	width = 0.8 / n_m
	vals_all: list[np.ndarray] = []
	positions: list[float] = []
	body_colors: list[str] = []
	for bi, col in enumerate(cols):
		for mi, method in enumerate(methods_raw):
			md = sub[sub['pred_folder'].astype(str) == method]
			v = pd.to_numeric(md[col], errors='coerce').dropna().values
			if len(v) == 0:
				continue
			offset = (mi - (n_m - 1) / 2.0) * width
			positions.append(bi + offset)
			vals_all.append(v)
			body_colors.append(method_colors.get(method, '#666666'))

	ax.set_xlabel(_RADIUS_AXIS_LABEL)
	ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)
	ax.set_xticks(range(len(display_labels)))
	ax.set_xticklabels(display_labels, rotation=45, ha='right')
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

	ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
	ax.grid(axis='y', alpha=0.25, linewidth=0.25)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	if show_legend and len(methods_raw) > 1:
		handles = [
			Patch(facecolor=method_colors.get(m, '#666666'), edgecolor=method_colors.get(m, '#666666'), alpha=0.75, label=md)
			for m, md in zip(methods_raw, methods_display)
		]
		ax.legend(handles=handles, fontsize=_LEGEND_FONTSIZE, frameon=False, loc='best')
	if add_pvalue and len(methods_raw) >= 2:
		_annotate_bucket_pvalues(ax, sub, cols, methods_raw)


def _draw_failure_panel(
	fax,
	df: pd.DataFrame,
	categories: list[str],
	methods_raw: list[str],
	method_colors: dict[str, str],
) -> None:
	"""Grouped bar panel of diverged-case counts per anatomy category and method."""
	div2d = _diverged_2d(df)
	n_m = max(len(methods_raw), 1)
	bar_width = 0.8 / n_m
	x = np.arange(len(categories))
	offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, n_m)
	any_fail = False
	for mi, method in enumerate(methods_raw):
		heights = [div2d.get((cat, method), 0) for cat in categories]
		any_fail = any_fail or any(h > 0 for h in heights)
		fax.bar(x + offsets[mi], heights, bar_width * 0.9, color=method_colors.get(method, '#666666'), label=_format_method_display_name(method))
	fax.set_xlabel('Anatomy')
	fax.set_ylabel('Diverged cases (count)')
	fax.set_title('Failures (diverged)')
	fax.set_xticks(x)
	fax.set_xticklabels(categories, rotation=45, ha='right')
	fax.grid(axis='y', alpha=0.25, linewidth=0.25)
	fax.spines['top'].set_visible(False)
	fax.spines['right'].set_visible(False)
	if not any_fail:
		fax.text(0.5, 0.9, 'no failures', ha='center', va='top', transform=fax.transAxes, fontsize=_ANNOT_FONTSIZE)


def plot_dice_per_radii_by_category(
	indir: str,
	categories_json: str,
	out_path: str | None = None,
	figsize: tuple[float, float] = (6, 4),
	ylabel: str | None = None,
	stem: str = 'mesh_metrics',
	drop_uncategorized: bool = False,
	fmt: str = 'png',
) -> None:
	"""Per-anatomy dice-per-radii: one grouped-violin subplot per anatomy category.

	Reads the per-case CSVs (<stem>_<folder>.csv) in ``indir``, groups cases by anatomy
	via ``categories_json``, and shows the per-case Dice distribution for each radii
	bucket as violins (one colored sub-violin per method), with a per-bucket p-value
	annotated above each group. ``figsize`` is interpreted as the size per subplot.
	"""
	df = _load_per_case(indir, stem)
	if df is None or df.empty:
		raise ValueError(f'No {stem}_*.csv (or {stem}.csv) per-case files found in {indir}')
	cols = _percase_dice_radii_columns(df)
	if not cols:
		raise ValueError(
			'No per-case dice-per-radii columns found. '
			'Ensure the metrics were generated with --centerline-dir and dice_radii.'
		)

	df, categories = _assign_categories(df, categories_json, drop_uncategorized)
	if not categories:
		raise ValueError('No cases matched any category in the categories JSON.')

	methods_raw = sorted(df['pred_folder'].astype(str).unique(), key=lambda m: (_method_order_key(m), m))
	# Legend labels include each method's overall failure count
	method_fail = {m: _diverged_total(df, m) for m in methods_raw}
	methods_display = [
		f'{_format_method_display_name(m)} ({method_fail[m][0]}/{method_fail[m][1]} failed)'
		if method_fail[m][0] else _format_method_display_name(m)
		for m in methods_raw
	]
	bucket_labels = [c[len('dice_radii_'):] for c in cols]
	display_labels = [_format_bucket_range(b) for b in bucket_labels]
	ax_ylabel = ylabel if ylabel is not None else 'Dice coefficient'

	_apply_style()
	method_colors = dict(zip(methods_raw, get_nature_colors(len(methods_raw))))

	# One panel per category, plus a final "Failures" panel
	n = len(categories) + 1
	ncols = min(2, n)
	nrows = math.ceil(n / ncols)
	fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
	axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

	df_ok = df[df['status'].astype(str).str.lower() != 'diverged'] if 'status' in df.columns else df
	for idx, cat in enumerate(categories):
		ax = axes[idx]
		sub = df_ok[df_ok['category'] == cat]
		# Successful (scored) cases per category (sub excludes diverged); failures from full df
		n_ok = int(sub['case'].nunique()) if 'case' in sub.columns else len(sub)
		full_sub = df[df['category'] == cat]
		if 'status' in full_sub.columns:
			div = full_sub[full_sub['status'].astype(str).str.lower() == 'diverged']
			n_div = int(div['case'].nunique()) if 'case' in div.columns else len(div)
		else:
			n_div = 0
		title = f'{cat} (n={n_ok}' + (f', {n_div} failed)' if n_div else ')')
		_draw_grouped_violins(
			ax, sub, methods_raw, methods_display, cols, display_labels, ax_ylabel,
			method_colors, title=title, show_legend=(idx == 0), add_pvalue=True,
		)

	_draw_failure_panel(axes[len(categories)], df, categories, methods_raw, method_colors)

	for j in range(len(categories) + 1, len(axes)):
		axes[j].set_visible(False)
	fig.tight_layout()
	if out_path:
		save_violin_figure(fig, out_path, format=fmt)
		print(f'Saved figure to {out_path}')
	else:
		plt.show()


def plot_dice_per_radii(
	indir: str,
	out_path: str | None = None,
	figsize: tuple[float, float] = (10, 6),
	ylabel: str | None = None,
	stem: str = 'mesh_metrics',
	fmt: str = 'png',
) -> None:
	"""Plot dice-per-radii as grouped violins with a per-bucket p-value.

	X-axis: radius buckets. Each bucket shows one colored violin per method
	(the per-case Dice distribution), and the p-value comparing the methods within
	the bucket is annotated above the group.

	Parameters
	----------
	indir : str
		Directory containing the per-case ``<stem>_<folder>.csv`` files from
		compute_metrics_meshes_comparison.py (--predictions-root mode). A path to a
		file inside that directory (e.g. summary.csv) is also accepted.
	out_path : str, optional
		Path to save the figure
	figsize : tuple
		Figure size (width, height)
	ylabel : str, optional
		Y-axis label (default: 'Dice coefficient')
	stem : str
		Filename stem of the per-case CSVs (default: mesh_metrics).
	fmt : str
		Output format passed to the figure saver ('png' or 'pdf').
	"""
	if not os.path.isdir(indir):
		indir = os.path.dirname(indir) or '.'
	df = _load_per_case(indir, stem)
	if df is None or df.empty:
		raise ValueError(f'No per-case {stem}_*.csv (or {stem}.csv) files found in {indir}')

	cols = _percase_dice_radii_columns(df)
	if not cols:
		raise ValueError(
			'No per-case dice-per-radii columns found. '
			'Ensure the metrics were generated with --centerline-dir and dice_radii.'
		)

	methods_raw = sorted(df['pred_folder'].astype(str).unique(), key=lambda m: (_method_order_key(m), m))
	method_fail = {m: _diverged_total(df, m) for m in methods_raw}
	methods_display = [
		f'{_format_method_display_name(m)} ({method_fail[m][0]}/{method_fail[m][1]} failed)'
		if method_fail[m][0] else _format_method_display_name(m)
		for m in methods_raw
	]
	bucket_labels = [c[len('dice_radii_'):] for c in cols]
	display_labels = [_format_bucket_range(b) for b in bucket_labels]
	ax_ylabel = ylabel if ylabel is not None else 'Dice coefficient'

	_apply_style()
	method_colors = dict(zip(methods_raw, get_nature_colors(len(methods_raw))))
	df_ok = df[df['status'].astype(str).str.lower() != 'diverged'] if 'status' in df.columns else df

	fig, ax = plt.subplots(figsize=figsize)
	_draw_grouped_violins(
		ax, df_ok, methods_raw, methods_display, cols, display_labels, ax_ylabel,
		method_colors, show_legend=True, add_pvalue=True,
	)
	fig.tight_layout()
	if out_path:
		save_violin_figure(fig, out_path, format=fmt)
		print(f'Saved figure to {out_path}')
	else:
		plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Plot dice-per-radii as grouped violins: x-axis=buckets, violins=methods, with a per-bucket p-value'
	)
	p.add_argument(
		'input_dir',
		help='Directory containing the per-case mesh_metrics_<folder>.csv files from compute_metrics_meshes_comparison.py (--predictions-root mode). A path to a file inside that directory (e.g. summary.csv) is also accepted.',
	)
	p.add_argument(
		'--out', '-o',
		default=None,
		help='Output path for figure (default: display only)',
	)
	p.add_argument(
		'--figsize',
		default='10,6',
		help='Figure size as width,height (default: 10,6). With --categories this is the size per subplot.',
	)
	p.add_argument(
		'--categories',
		default=None,
		metavar='JSON',
		help='Path to categories.json {category: [cases]}. If set, plots one grouped-violin subplot per anatomy category, re-aggregating the per-case mesh_metrics_<folder>.csv files.',
	)
	p.add_argument(
		'--stem',
		default='mesh_metrics',
		help="Filename stem of the per-case CSVs (default: mesh_metrics, i.e. mesh_metrics_<folder>.csv).",
	)
	p.add_argument(
		'--drop-uncategorized',
		action='store_true',
		help="With --categories: drop cases not listed in the JSON instead of grouping them as 'uncategorized'.",
	)
	p.add_argument(
		'--format',
		choices=['png', 'pdf'],
		default='png',
		help='Output format: png (600 DPI) or pdf. Default: png',
	)
	return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	try:
		figsize = tuple(float(x) for x in args.figsize.split(','))
	except ValueError:
		print(f'Error: --figsize must be width,height (e.g. 10,6)', file=sys.stderr)
		return 1

	if os.path.isdir(args.input_dir):
		indir = args.input_dir
	else:
		indir = os.path.dirname(args.input_dir) or '.'

	# Per-anatomy mode: re-aggregate the per-case CSVs by category
	if args.categories:
		if not os.path.isfile(args.categories):
			print(f'Error: Categories file not found: {args.categories}', file=sys.stderr)
			return 1
		try:
			plot_dice_per_radii_by_category(
				indir,
				args.categories,
				out_path=args.out,
				figsize=figsize if args.figsize != '10,6' else (6, 4),
				stem=args.stem,
				drop_uncategorized=args.drop_uncategorized,
				fmt=args.format,
			)
		except ValueError as e:
			print(f'Error: {e}', file=sys.stderr)
			return 2
		return 0

	try:
		plot_dice_per_radii(
			indir,
			out_path=args.out,
			figsize=figsize,
			stem=args.stem,
			fmt=args.format,
		)
	except ValueError as e:
		print(f'Error: {e}', file=sys.stderr)
		return 2
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
