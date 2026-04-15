"""LaTeX table output for journal-ready tables.

Provides functions to write metrics as rows in LaTeX format, suitable for
inclusion in manuscripts (e.g. Nature, IEEE).
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon


def _fmt_num(x: float, decimals: int = 3) -> str:
	"""Format number for LaTeX; use scientific notation when compact."""
	if math.isnan(x):
		return '---'
	if abs(x) >= 1000 or (0 < abs(x) < 0.001 and x != 0):
		return f'{x:.2e}'
	return f'{x:.{decimals}f}'


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
	"""Format as 'mean ± std' for LaTeX."""
	if math.isnan(mean):
		return '---'
	if math.isnan(std) or std == 0:
		return _fmt_num(mean, decimals)
	return f'{_fmt_num(mean, decimals)} $\\pm$ {_fmt_num(std, decimals)}'


def _escape_latex(s: str) -> str:
	"""Escape special LaTeX characters in table cell text."""
	return s.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')


def _format_p_value(p: float) -> str:
	"""Format p-value for LaTeX: p < 0.001 for very small, else p = X.XXX."""
	if math.isnan(p) or p < 0:
		return '---'
	if p < 0.001:
		return '$p < 0.001$'
	return f'$p = {_fmt_num(p, 3)}$'


def write_pred_vs_pred_reference_latex_table(
	df: pd.DataFrame,
	metric_cols: list[str],
	reference: str,
	out_path: str,
	metric_display_names: Optional[dict[str, str]] = None,
	means_only: bool = False,
	include_pvalues: bool = True,
) -> None:
	"""Write LaTeX table with metrics as rows, groups as columns.

	Computes mean ± std per metric for 'Reference vs Others' and 'Others vs Others'.
	P-values from Mann-Whitney U test (two-sided). Output is journal-ready (e.g. Nature, IEEE).

	Args:
		df: DataFrame with 'group' column and metric columns.
		metric_cols: List of metric column names to include.
		reference: Reference prediction name (used for column headers).
		out_path: Path to write .txt file containing LaTeX.
		metric_display_names: Optional {metric_key: display_name} for row labels.
		means_only: If True, show only mean values (no ± std).
		include_pvalues: If True, add p-value column (Mann-Whitney U test).
	"""
	display_names = metric_display_names or {}
	group_order = ['Reference vs Others', 'Others vs Others']

	# Column headers: use reference-specific labels
	if reference.lower() == 'seqseg':
		col_ref = 'SeqSeg vs Manual'
		col_others = 'Manual vs Manual'
	else:
		col_ref = f'{reference} vs Others'
		col_others = 'Others vs Others'

	# Compute mean ± std and p-value per metric per group
	rows: list[tuple[str, str, str, str]] = []
	for metric in metric_cols:
		if metric not in df.columns:
			continue
		label = display_names.get(metric, metric.replace('_', ' ').title())
		label_tex = _escape_latex(label)

		# Average per patient (independent observations), then Wilcoxon signed-rank (paired)
		ref_df = df[df['group'] == group_order[0]][['case', metric]].dropna()
		others_df = df[df['group'] == group_order[1]][['case', metric]].dropna()
		vals_ref = ref_df.groupby('case')[metric].mean().reset_index()
		vals_others = others_df.groupby('case')[metric].mean().reset_index()
		# Align by case (inner join)
		merged = vals_ref.merge(vals_others, on='case', suffixes=('_ref', '_others'))
		vals_ref = merged[f'{metric}_ref'].values
		vals_others = merged[f'{metric}_others'].values

		cells = []
		for vals in (vals_ref, vals_others):
			if len(vals) == 0:
				cells.append('---')
			else:
				mean_val = float(np.mean(vals))
				std_val = float(np.std(vals))
				if means_only:
					cells.append(_fmt_num(mean_val))
				else:
					cells.append(format_mean_std(mean_val, std_val))

		# Wilcoxon signed-rank (paired, two-sided)
		if include_pvalues and len(vals_ref) >= 2:
			try:
				_, p = wilcoxon(vals_ref, vals_others, alternative='two-sided')
				p_str = _format_p_value(float(p))
			except Exception:
				p_str = '---'
		else:
			p_str = '---'

		if len(cells) == 2:
			rows.append((label_tex, cells[0], cells[1], p_str))

	if not rows:
		return

	# Build LaTeX
	lines = []
	lines.append('% Pred-vs-pred reference comparison. Metrics as rows.')
	lines.append('% P-values: Wilcoxon signed-rank test (paired, two-sided).')
	lines.append('% Copy into your LaTeX document. Ensure \\usepackage{booktabs} in preamble.')
	lines.append('')
	lines.append('\\begin{table}[htbp]')
	lines.append('\\centering')
	caption = 'Mean per metric.' if means_only else 'Mean $\\pm$ std per metric.'
	if include_pvalues:
		caption += ' P-values from Wilcoxon signed-rank test.'
	lines.append(f'\\caption{{{caption}}}')
	lines.append('\\label{tab:pred_vs_pred_reference}')
	col_spec = 'lccc' if include_pvalues else 'lcc'
	lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
	lines.append('\\toprule')
	header = f'Metric & {_escape_latex(col_ref)} & {_escape_latex(col_others)}'
	if include_pvalues:
		header += ' & $p$-value'
	lines.append(header + ' \\\\')
	lines.append('\\midrule')
	for label_tex, cell_ref, cell_others, p_str in rows:
		row = f'{label_tex} & {cell_ref} & {cell_others}'
		if include_pvalues:
			row += f' & {p_str}'
		lines.append(row + ' \\\\')
	lines.append('\\bottomrule')
	lines.append('\\end{tabular}')
	lines.append('\\end{table}')

	with open(out_path, 'w') as f:
		f.write('\n'.join(lines))


def write_point_diff_latex_table(
	point_diffs: dict[str, dict[str, np.ndarray]],
	metric_cols: list[str],
	reference: str,
	out_path: str,
	metric_display_names: Optional[dict[str, str]] = None,
	means_only: bool = False,
	include_pvalues: bool = True,
	point_patients: Optional[dict[str, dict[str, np.ndarray]]] = None,
) -> None:
	"""Write LaTeX table for point-wise pairwise differences.

	Metrics as rows; columns are 'Reference vs Others' and 'Others vs Others'
	with mean ± std of |pairwise differences| per patient. P-values from Wilcoxon signed-rank test.

	Args:
		point_diffs: {metric: {group: array of patient-level values}} from _compute_point_differences.
		point_patients: {metric: {group: array of patient labels}} for paired Wilcoxon test.
		metric_cols: List of metric names.
		reference: Reference name for column headers.
		out_path: Output .txt path.
		metric_display_names: Optional display names for metrics.
		means_only: If True, show only mean.
		include_pvalues: If True, add p-value column (Mann-Whitney U test).
	"""
	display_names = metric_display_names or {}
	group_order = ['Reference vs Others', 'Others vs Others']

	if reference.lower() == 'seqseg':
		col_ref = 'SeqSeg vs Manual'
		col_others = 'Manual vs Manual'
	else:
		col_ref = f'{reference} vs Others'
		col_others = 'Others vs Others'

	rows: list[tuple[str, str, str, str]] = []
	for metric in metric_cols:
		label = display_names.get(metric, metric.replace('_', ' ').title())
		label_tex = _escape_latex(label) + ' (|pairwise diff|)'

		vals_ref = np.asarray(point_diffs.get(metric, {}).get(group_order[0], np.array([])))
		vals_others = np.asarray(point_diffs.get(metric, {}).get(group_order[1], np.array([])))
		patients_ref = np.asarray(point_patients.get(metric, {}).get(group_order[0], [])) if point_patients else np.array([])
		patients_others = np.asarray(point_patients.get(metric, {}).get(group_order[1], [])) if point_patients else np.array([])

		cells = []
		for vals in (vals_ref, vals_others):
			if len(vals) == 0:
				cells.append('---')
			else:
				mean_val = float(np.mean(vals))
				std_val = float(np.std(vals))
				if means_only:
					cells.append(_fmt_num(mean_val))
				else:
					cells.append(format_mean_std(mean_val, std_val))

		# Wilcoxon signed-rank (paired, two-sided) when patient labels available
		if include_pvalues and len(vals_ref) >= 2 and len(vals_others) >= 2:
			try:
				if point_patients is not None and len(patients_ref) == len(vals_ref) and len(patients_others) == len(vals_others):
					common = np.intersect1d(patients_ref, patients_others)
					if len(common) >= 2:
						paired1 = np.array([vals_ref[np.where(patients_ref == p)[0][0]] for p in common])
						paired2 = np.array([vals_others[np.where(patients_others == p)[0][0]] for p in common])
						_, p = wilcoxon(paired1, paired2, alternative='two-sided')
					else:
						raise ValueError('Insufficient paired data')
				else:
					raise ValueError('Patient labels required for Wilcoxon')
				p_str = _format_p_value(float(p))
			except (ValueError, Exception):
				p_str = '---'
		else:
			p_str = '---'

		rows.append((label_tex, cells[0], cells[1], p_str))

	if not rows:
		return

	lines = []
	lines.append('% Point-wise pairwise metric differences. Metrics as rows.')
	lines.append('% P-values: Wilcoxon signed-rank test (paired, two-sided).')
	lines.append('% Copy into your LaTeX document. Ensure \\usepackage{booktabs} in preamble.')
	lines.append('')
	lines.append('\\begin{table}[htbp]')
	lines.append('\\centering')
	caption = 'Mean |pairwise diff| per metric.' if means_only else 'Mean $\\pm$ std |pairwise diff| per metric.'
	if include_pvalues:
		caption += ' P-values from Wilcoxon signed-rank test.'
	lines.append(f'\\caption{{{caption}}}')
	lines.append('\\label{tab:point_diff_reference}')
	col_spec = 'lccc' if include_pvalues else 'lcc'
	lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
	lines.append('\\toprule')
	header = f'Metric & {_escape_latex(col_ref)} & {_escape_latex(col_others)}'
	if include_pvalues:
		header += ' & $p$-value'
	lines.append(header + ' \\\\')
	lines.append('\\midrule')
	for label_tex, cell_ref, cell_others, p_str in rows:
		row = f'{label_tex} & {cell_ref} & {cell_others}'
		if include_pvalues:
			row += f' & {p_str}'
		lines.append(row + ' \\\\')
	lines.append('\\bottomrule')
	lines.append('\\end{tabular}')
	lines.append('\\end{table}')

	with open(out_path, 'w') as f:
		f.write('\n'.join(lines))


def _pvalue_friedman_across_comparisons(
	df: pd.DataFrame,
	comparisons: list[str],
	metric: str,
) -> float | None:
	"""Omnibus p-value: Friedman test on case-matched values across comparisons.

	Uses the same cases present in every comparison (repeated-measures blocks).
	SciPy's Friedman requires at least three conditions; with exactly two
	comparisons, uses Wilcoxon signed-rank on paired case values.

	Returns None if the test cannot be run (e.g. no ``case`` column, insufficient
	overlap, or Wilcoxon failure such as all pairwise differences zero).
	"""
	if 'case' not in df.columns or len(comparisons) < 2:
		return None

	series_list: list[pd.Series] = []
	for comp in comparisons:
		sub = df[df['comparison'] == comp][['case', metric]].dropna(subset=[metric])
		if sub.empty:
			return None
		sub = sub.drop_duplicates(subset=['case'], keep='first')
		series_list.append(sub.set_index(sub['case'].astype(str))[metric])

	common: set[str] = set(series_list[0].index)
	for s in series_list[1:]:
		common &= set(s.index)
	if len(common) < 2:
		return None

	sorted_cases = sorted(common)
	samples = [np.array([float(s.loc[c]) for c in sorted_cases], dtype=float) for s in series_list]

	try:
		if len(samples) == 2:
			_, p = wilcoxon(samples[0], samples[1], alternative='two-sided')
			return float(p)
		_, p = friedmanchisquare(*samples)
		return float(p)
	except (ValueError, RuntimeError):
		return None


def _wilcoxon_pvalue_two_comparisons_paired(
	df: pd.DataFrame,
	comp_a: str,
	comp_b: str,
	metric: str,
) -> float | None:
	"""Two-sided Wilcoxon signed-rank on case-matched values for two comparison groups."""
	if 'case' not in df.columns or metric not in df.columns:
		return None
	a = df[df['comparison'] == comp_a][['case', metric]].dropna(subset=[metric])
	b = df[df['comparison'] == comp_b][['case', metric]].dropna(subset=[metric])
	if a.empty or b.empty:
		return None
	a = a.drop_duplicates(subset=['case'], keep='first')
	b = b.drop_duplicates(subset=['case'], keep='first')
	m = a.merge(b, on='case', suffixes=('_a', '_b'), how='inner')
	if len(m) < 2:
		return None
	col_a, col_b = f'{metric}_a', f'{metric}_b'
	if col_a not in m.columns or col_b not in m.columns:
		return None
	x = m[col_a].astype(float).values
	y = m[col_b].astype(float).values
	try:
		_, p = wilcoxon(x, y, alternative='two-sided')
		return float(p)
	except (ValueError, RuntimeError):
		return None


def write_pred_vs_pred_all_comparisons_latex_table(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_path: str,
	metric_display_names: Optional[dict[str, str]] = None,
	metric_abbreviations: Optional[dict[str, str]] = None,
	comparison_display_names: Optional[dict[str, str]] = None,
	means_only: bool = False,
	include_pvalues: bool = True,
) -> None:
	"""Write LaTeX table with comparisons as rows, metrics as columns.

	Computes mean ± std per metric per comparison. P-values from Friedman test on
	case-matched values (repeated measures across comparisons); if only two
	comparisons exist, paired Wilcoxon signed-rank is used instead. Output is
	journal-ready (e.g. Nature, IEEE).

	Args:
		df: DataFrame with ``case``, ``comparison``, and metric columns.
		metric_cols: List of metric column names to include.
		out_path: Path to write .txt file containing LaTeX.
		metric_display_names: Optional {metric_key: display_name} for caption abbreviations.
		metric_abbreviations: Optional {metric_key: abbrev} for column headers; falls back to display_names.
		comparison_display_names: Optional {comparison_key: display_name} for row labels.
		means_only: If True, show only mean values (no ± std).
		include_pvalues: If True, add p-value column (Friedman / paired Wilcoxon).
	"""
	display_names = metric_display_names or {}
	abbrevs = metric_abbreviations or display_names
	comp_display = comparison_display_names or {}

	comparisons = sorted(df['comparison'].unique())

	# Filter to metrics that exist
	metric_cols = [m for m in metric_cols if m in df.columns]
	if not metric_cols:
		return

	# Compute mean ± std per comparison per metric; each row = one comparison
	rows: list[tuple[str, list[str]]] = []
	for comp in comparisons:
		comp_label = comp_display.get(comp, comp.replace('_vs_', ' vs '))
		comp_label_tex = _escape_latex(comp_label)

		cells = []
		for metric in metric_cols:
			vals = df[df['comparison'] == comp][metric].dropna().values
			if len(vals) == 0:
				cells.append('---')
			else:
				mean_val = float(np.mean(vals))
				std_val = float(np.std(vals))
				if means_only:
					cells.append(_fmt_num(mean_val))
				else:
					cells.append(format_mean_std(mean_val, std_val))

		rows.append((comp_label_tex, cells))

	# Column headers: abbreviations (or full names if no abbreviations)
	col_headers = []
	for metric in metric_cols:
		abbrev = abbrevs.get(metric, display_names.get(metric, metric.replace('_', ' ').title()))
		# Don't escape if already contains LaTeX math (e.g. SD$_{\tau 1}$)
		col_headers.append(abbrev if '$' in abbrev else _escape_latex(abbrev))

	# Friedman test (case-matched) per metric; summary row
	pvalue_row: list[str] = []
	if include_pvalues:
		for metric in metric_cols:
			pv = _pvalue_friedman_across_comparisons(df, comparisons, metric)
			if pv is not None:
				pvalue_row.append(_format_p_value(pv))
			else:
				pvalue_row.append('---')

	# Build LaTeX
	lines = []
	lines.append('% Pred-vs-pred all comparisons. Comparisons as rows, metrics as columns.')
	lines.append('% P-values: Friedman test on case-matched values (Wilcoxon if only 2 comparisons).')
	lines.append('% Copy into your LaTeX document. Ensure \\usepackage{booktabs} in preamble.')
	lines.append('')
	# Build caption with abbreviation descriptions
	caption = 'Mean per metric per comparison.' if means_only else 'Mean $\\pm$ std per metric per comparison.'
	if include_pvalues:
		caption += (
			' P-values from Friedman test on matched cases (paired Wilcoxon if two comparisons).'
		)
	abbrev_desc_pairs = [
		f'{abbrevs.get(m, m)}: {_escape_latex(display_names.get(m, m.replace("_", " ").title()))}'
		for m in metric_cols
	]
	if abbrev_desc_pairs:
		caption += ' Abbreviations: ' + '; '.join(abbrev_desc_pairs) + '.'
	lines.append('\\begin{table}[htbp]')
	lines.append('\\centering')
	lines.append(f'\\caption{{{caption}}}')
	lines.append('\\label{tab:pred_vs_pred_all_comparisons}')
	col_spec = 'l' + 'c' * len(metric_cols)
	lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
	lines.append('\\toprule')
	lines.append('Comparison & ' + ' & '.join(col_headers) + ' \\\\')
	lines.append('\\midrule')
	for comp_label_tex, cells in rows:
		lines.append(comp_label_tex + ' & ' + ' & '.join(cells) + ' \\\\')
	if include_pvalues and pvalue_row:
		lines.append('\\midrule')
		lines.append('$p$ (Friedman) & ' + ' & '.join(pvalue_row) + ' \\\\')
	lines.append('\\bottomrule')
	lines.append('\\end{tabular}')
	lines.append('\\end{table}')

	with open(out_path, 'w') as f:
		f.write('\n'.join(lines))


def write_pred_vs_pred_pairwise_wilcoxon_bonferroni_latex_table(
	df: pd.DataFrame,
	metric_cols: list[str],
	out_path: str,
	metric_display_names: Optional[dict[str, str]] = None,
	metric_abbreviations: Optional[dict[str, str]] = None,
	comparison_display_names: Optional[dict[str, str]] = None,
) -> None:
	"""Pairwise post-hoc Wilcoxon signed-rank (paired by case) with Bonferroni correction.

	For each metric, runs a two-sided Wilcoxon on case-matched values between every
	unordered pair of ``comparison`` groups. Raw *p*-values are multiplied by the
	number of comparison groups $k$ (same as the number of ``comparison`` levels)
	and capped at 1.0 (Bonferroni).

	Requires a ``case`` column. Writes nothing if fewer than two comparison groups.

	Args:
		df: DataFrame with ``case``, ``comparison``, and metric columns.
		metric_cols: Metrics as columns in the output table.
		out_path: Path for the LaTeX ``.txt`` file.
		metric_display_names: Optional row/caption names by metric key.
		metric_abbreviations: Optional short column headers.
		comparison_display_names: Optional labels for comparison pair rows.
	"""
	display_names = metric_display_names or {}
	abbrevs = metric_abbreviations or display_names
	comp_display = comparison_display_names or {}

	if 'case' not in df.columns:
		return

	comparisons = sorted(df['comparison'].unique())
	metric_cols = [m for m in metric_cols if m in df.columns]
	if len(comparisons) < 2 or not metric_cols:
		return

	pairs: list[tuple[str, str]] = list(combinations(comparisons, 2))
	if not pairs:
		return

	bonferroni_m = len(comparisons)

	def _pair_row_label(ca: str, cb: str) -> str:
		la = comp_display.get(ca, ca.replace('_vs_', ' vs '))
		lb = comp_display.get(cb, cb.replace('_vs_', ' vs '))
		# Semicolon avoids stacking a third "vs" onto comparison names that already contain "vs".
		return _escape_latex(f'{la}; {lb}')

	col_headers: list[str] = []
	for metric in metric_cols:
		abbrev = abbrevs.get(metric, display_names.get(metric, metric.replace('_', ' ').title()))
		col_headers.append(abbrev if '$' in abbrev else _escape_latex(abbrev))

	lines: list[str] = []
	lines.append('% Pairwise post-hoc Wilcoxon signed-rank tests (case-matched, two-sided).')
	lines.append(f'% Bonferroni: each p-value multiplied by {bonferroni_m} (number of comparison groups).')
	lines.append('% Copy into your LaTeX document. Ensure \\usepackage{booktabs} in preamble.')
	lines.append('')
	lines.append('\\begin{table}[htbp]')
	lines.append('\\centering')
	caption = (
		'Pairwise Wilcoxon signed-rank $p$-values (matched cases) between comparison groups, '
		f'Bonferroni-adjusted (multiply by $k = {bonferroni_m}$ comparison groups).'
	)
	abbrev_desc_pairs = [
		f'{abbrevs.get(m, m)}: {_escape_latex(display_names.get(m, m.replace("_", " ").title()))}'
		for m in metric_cols
	]
	if abbrev_desc_pairs:
		caption += ' Abbreviations: ' + '; '.join(abbrev_desc_pairs) + '.'
	lines.append(f'\\caption{{{caption}}}')
	lines.append('\\label{tab:pred_vs_pred_pairwise_wilcoxon_bonferroni}')
	col_spec = 'l' + 'c' * len(metric_cols)
	lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
	lines.append('\\toprule')
	lines.append('Pair & ' + ' & '.join(col_headers) + ' \\\\')
	lines.append('\\midrule')

	for ca, cb in pairs:
		row_label = _pair_row_label(ca, cb)
		cells: list[str] = []
		for metric in metric_cols:
			p_raw = _wilcoxon_pvalue_two_comparisons_paired(df, ca, cb, metric)
			if p_raw is None or math.isnan(p_raw):
				cells.append('---')
			else:
				p_adj = min(1.0, p_raw * bonferroni_m)
				cells.append(_format_p_value(p_adj))
		lines.append(row_label + ' & ' + ' & '.join(cells) + ' \\\\')

	lines.append('\\bottomrule')
	lines.append('\\end{tabular}')
	lines.append('\\end{table}')

	with open(out_path, 'w') as f:
		f.write('\n'.join(lines))
