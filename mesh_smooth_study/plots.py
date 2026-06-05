"""Generate figures from mesh_smooth_study CSV output."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _as_bool(v) -> bool:
    """Coerce CSV-roundtripped values (True/False, "True"/"False", 1/0) to bool."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    if pd.isna(v):
        return False
    return bool(v)


def _parse_params(params_json) -> dict:
    try:
        p = json.loads(params_json) if isinstance(params_json, str) else params_json
    except (json.JSONDecodeError, TypeError):
        return {}
    return p if isinstance(p, dict) else {}


def _compact_params_label(method: str, params_json: str) -> str:
    """Short multi-line x-tick label focused on iteration count and smoothing strength.

    Method is conveyed by colour/legend, so it is omitted here to keep labels narrow.
    Each parameter goes on its own line, e.g. ``iters=50`` / ``\u03bc1=0.5`` / ``\u03bc2=0.51``.
    """
    p = _parse_params(params_json)
    if not p:
        return "(no smoothing)"

    lines = []
    iters = p.get("iterations")
    if iters is not None:
        lines.append(f"iters={iters}")

    # Taubin uses mu1/mu2 (and pass-band variants); one per line to stay narrow.
    for key, sym in (("mu1", "\u03bc1"), ("mu2", "\u03bc2"), ("mu", "\u03bc"), ("pass_band", "pass_band")):
        if key in p:
            lines.append(f"{sym}={p[key]}")

    # Laplacian uses a relaxation factor instead of mus.
    if "relaxation_factor" in p:
        lines.append(f"relax={p['relaxation_factor']}")

    return "\n".join(lines) if lines else "(default)"


# Stable, colour-blind-friendly colour per smoothing method.
def _method_colors(methods) -> dict:
    palette = sns.color_palette("colorblind", n_colors=max(3, len(methods)))
    return {m: palette[i] for i, m in enumerate(sorted(methods))}


def _mark_method_groups(ax, methods: list[str], method_colors: dict | None = None) -> None:
    """Separate method groups with dashed lines and label each group at the top."""
    if not methods:
        return
    # Boundaries between consecutive method groups.
    starts = [0]
    for i in range(1, len(methods)):
        if methods[i] != methods[i - 1]:
            ax.axvline(i - 0.5, color="0.7", linewidth=0.8, linestyle="--", zorder=0)
            starts.append(i)
    starts.append(len(methods))

    # Centred method name above each group.
    for s, e in zip(starts[:-1], starts[1:]):
        center = (s + e - 1) / 2.0
        m = methods[s]
        color = (method_colors or {}).get(m, "0.2")
        ax.text(
            center,
            1.005,
            m,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color,
            clip_on=False,
        )


def _method_legend(fig, ax, method_colors: dict) -> None:
    """Use a hidden axis to host a legend mapping colour → method."""
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, label=m) for m, c in sorted(method_colors.items())]
    ax.set_visible(True)
    ax.axis("off")
    ax.legend(handles=handles, title="method", loc="center", fontsize=9, title_fontsize=10)


def _add_config_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_cfg_id"] = out["method"].astype(str) + "|" + out["params_sha10"].astype(str)
    out["_cfg_lbl"] = [
        _compact_params_label(m, pj) for m, pj in zip(out["method"], out["params_json"])
    ]
    return out


# Fixed method order (unlisted methods come after, alphabetically).
_METHOD_ORDER = ["none", "laplacian", "taubin", "taubin_cot", "windowed_sinc"]
# Within a method, sort configs by these params in this priority (numeric, ascending).
_PARAM_SORT_KEYS = ["iterations", "relaxation_factor", "mu", "mu1", "mu2", "pass_band", "feature_angle"]


def _method_rank(method: str):
    m = str(method)
    return (_METHOD_ORDER.index(m), "") if m in _METHOD_ORDER else (len(_METHOD_ORDER), m)


def _cfg_sort_key(params_json) -> tuple:
    """Numeric sort key from params so configs order consistently (missing keys last)."""
    p = _parse_params(params_json)
    key = []
    for k in _PARAM_SORT_KEYS:
        v = p.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            key.append((0, float(v)))
        else:
            key.append((1, 0.0))
    return tuple(key)


def _cfg_order_and_labels(df: pd.DataFrame, metric: str | None = None, ascending: bool = False):
    """Deterministic, metric-independent config ordering.

    Configs are ordered by a fixed method order and then by their parameter values,
    so the same setting always sits at the same x position across every subplot.
    ``metric``/``ascending`` are kept for call-site compatibility and only used to
    drop configs that have no data for the given metric.
    """
    sub = df
    if metric is not None and metric in df.columns:
        sub = df.dropna(subset=[metric])
    cfgs = sub.drop_duplicates("_cfg_id")
    if cfgs.empty:
        return [], [], []

    records = list(zip(cfgs["_cfg_id"], cfgs["method"], cfgs["params_json"], cfgs["_cfg_lbl"]))
    records.sort(key=lambda r: (_method_rank(r[1]), _cfg_sort_key(r[2]), str(r[0])))

    order = [r[0] for r in records]
    methods = [r[1] for r in records]
    labels = [r[3] for r in records]
    return order, labels, methods


# Metrics aggregated across cases in the summary CSV. (column, higher_is_better)
_SUMMARY_METRICS = [
    ("dice_overall", True),
    ("assd_mm", False),
    ("hd95_mm", False),
    ("normal_error_mean_deg", False),
    ("normal_error_max_deg", False),
    ("volume_error_rel", False),
    ("volume_error_abs", False),
    ("surface_area_error_rel", False),
    ("surface_area_error_abs", False),
    ("mean_curvature_rms", False),
    ("dihedral_angle_p95_deg", False),
]


def _improvement_over_none(
    df: pd.DataFrame,
    metric_cols: list[str],
    key_cols: list[str],
    higher_better: dict[str, bool],
) -> pd.DataFrame:
    """Replace each metric by its improvement relative to the `none` baseline.

    The baseline is the ``method == "none"`` value matched on ``key_cols`` (e.g. the
    same case, or the same case + radius bin). Improvement is oriented so that a
    positive value always means *better* (Dice up, errors/distances down). ``none``
    rows and any rows without a matching baseline are dropped.
    """
    if "method" not in df.columns:
        return df.iloc[0:0].copy()

    base = df[df["method"] == "none"]
    out = df[df["method"] != "none"].copy()
    cols = [c for c in metric_cols if c in df.columns]
    if base.empty or out.empty or not cols:
        return out.iloc[0:0].copy()

    present_keys = [k for k in key_cols if k in df.columns]
    if present_keys:
        base_vals = base.groupby(present_keys, as_index=False)[cols].mean()
        merged = out.merge(base_vals, on=present_keys, how="left", suffixes=("", "__base"))
    else:
        means = base[cols].mean()
        merged = out.copy()
        for col in cols:
            merged[f"{col}__base"] = means[col]

    for col in cols:
        delta = pd.to_numeric(merged[col], errors="coerce") - pd.to_numeric(
            merged[f"{col}__base"], errors="coerce"
        )
        if not higher_better.get(col, True):
            delta = -delta
        merged[col] = delta

    return merged.drop(columns=[f"{c}__base" for c in cols])


def write_summary_csv(
    csv_path: str | os.PathLike,
    out_csv: str | os.PathLike,
    *,
    improve_over_none: bool = False,
) -> Path | None:
    """
    Aggregate per-case `overall` rows by smoothing configuration and write mean/std/median
    per metric to ``out_csv``. Returns the written path (or None if nothing to write).

    One row per (method, params_sha10); sorted by mean Dice (best first).
    """
    csv_path = Path(csv_path)
    out_csv = Path(out_csv)

    df = pd.read_csv(csv_path)
    if df.empty or "metric_scope" not in df.columns:
        return None

    if "params_sha10" not in df.columns and "params_json" in df.columns:
        df["params_sha10"] = df["params_json"].astype(str).map(
            lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        )

    overall = df[df["metric_scope"] == "overall"].copy()
    if overall.empty:
        return None

    metrics = [(c, hb) for c, hb in _SUMMARY_METRICS if c in overall.columns]
    if not metrics:
        return None

    group_cols = ["method", "params_sha10"]

    # Failure counts come from the raw overall rows (including `none` and any rows dropped by
    # the improvement transform below) so they always reflect every attempted config.
    fail_counts: dict = {}
    if "failed" in overall.columns:
        flags = overall.assign(_failed=overall["failed"].map(_as_bool))
        for keys, sub in flags.groupby(group_cols, dropna=False):
            fail_counts[keys] = (int(sub["_failed"].sum()), int(len(sub)))

    if improve_over_none:
        hb_map = {c: hb for c, hb in metrics}
        overall = _improvement_over_none(overall, list(hb_map), ["case_id"], hb_map)
        if overall.empty:
            return None

    rows: list[dict] = []
    for keys, sub in overall.groupby(group_cols, dropna=False):
        method, params_sha10 = keys
        params_json = ""
        if "params_json" in sub.columns and not sub["params_json"].empty:
            params_json = str(sub["params_json"].iloc[0])
        n_failed, n_attempts = fail_counts.get(keys, (0, 0))
        row: dict = {
            "method": method,
            "params_sha10": params_sha10,
            "params_json": params_json,
            "n_cases": int(sub["case_id"].nunique()) if "case_id" in sub.columns else len(sub),
            "n_failed": int(n_failed),
            "frac_failed": float(n_failed / n_attempts) if n_attempts else float("nan"),
        }
        for col, _ in metrics:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std())
            row[f"{col}_median"] = float(vals.median())
        rows.append(row)

    summary = pd.DataFrame(rows)
    sort_col = "dice_overall_mean" if "dice_overall_mean" in summary.columns else None
    if sort_col is not None:
        summary = summary.sort_values(sort_col, ascending=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return out_csv


def _write_failed_cases_txt(
    failed: pd.DataFrame,
    out_txt: Path,
    fail_label: str,
    written: list[Path],
) -> None:
    """Write a human-readable list of the failed (case, method, params) configurations."""
    if failed.empty:
        return

    has_reason = "fail_reason" in failed.columns
    lines = [
        f"# Failed smoothing configurations ({fail_label})",
        f"# total: {len(failed)} across {failed['case_id'].nunique()} cases",
        "",
    ]
    sort_cols = [c for c in ("method", "_cfg_lbl", "case_id") if c in failed.columns]
    for _, r in failed.sort_values(sort_cols).iterrows():
        case = r.get("case_id", "?")
        method = r.get("method", "?")
        params = str(r.get("params_json", "")).replace("\n", " ")
        line = f"{case}\t{method}\t{params}"
        if has_reason:
            reason = str(r.get("fail_reason", "")).strip()
            if reason:
                line += f"\t{reason}"
        lines.append(line)

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(out_txt)


def _write_failed_case_counts_txt(
    failed: pd.DataFrame,
    out_txt: Path,
    fail_label: str,
    written: list[Path],
) -> None:
    """Write per-case failure counts (cases that failed at least once), most failures first."""
    if failed.empty or "case_id" not in failed.columns:
        return

    counts = failed.groupby("case_id").size().sort_values(ascending=False)
    lines = [
        f"# Cases that failed at least once ({fail_label})",
        f"# {len(counts)} cases, {int(counts.sum())} total failures",
        "# case_id\tn_failures",
        "",
    ]
    lines.extend(f"{case}\t{int(n)}" for case, n in counts.items())

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(out_txt)


def generate_study_plots(
    csv_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    *,
    dpi: int = 150,
    fig_format: str = "png",
    improve_over_none: bool = False,
) -> list[Path]:
    """
    Read study CSV and write metric figures into out_dir.

    If ``improve_over_none`` is set, every metric is plotted as the per-case
    improvement relative to the ``none`` (no-smoothing) baseline, oriented so that
    positive always means better. Returns list of written file paths.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    if "params_sha10" not in df.columns and "params_json" in df.columns:
        df["params_sha10"] = df["params_json"].astype(str).map(
            lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        )

    written: list[Path] = []
    sns.set_theme(style="whitegrid", context="notebook")
    suffix = "_improve_over_none" if improve_over_none else ""

    overall = df[df["metric_scope"] == "overall"].copy()
    if not overall.empty:
        overall = _add_config_columns(overall)

        metrics_spec = [
            ("dice_overall", "Dice (overall)", True),
            ("assd_mm", "ASSD (length)", False),
            ("hd95_mm", "HD95 (length)", False),
            ("normal_error_mean_deg", "Normal error mean (deg)", False),
            ("normal_error_max_deg", "Normal error max (deg)", False),
            ("mean_curvature_rms", "RMS mean curvature (1/length)", False),
            ("dihedral_angle_p95_deg", "Dihedral angle p95 (deg)", False),
        ]

        if improve_over_none:
            hb_map = {c: hb for c, _, hb in metrics_spec}
            overall = _improvement_over_none(overall, list(hb_map), ["case_id"], hb_map)

    if not overall.empty:
        method_colors = _method_colors(overall["method"].unique())

        fig, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
        axes_flat = axes.ravel()
        for ax, (col, title, higher_better) in zip(axes_flat, metrics_spec):
            sub = overall.dropna(subset=[col])
            if sub.empty:
                ax.set_visible(False)
                continue
            order, tick_lbls, methods = _cfg_order_and_labels(sub, col, ascending=not higher_better)
            cfg_palette = {cid: method_colors[m] for cid, m in zip(order, methods)}
            sns.boxplot(
                data=sub,
                x="_cfg_id",
                y=col,
                order=order,
                hue="_cfg_id",
                palette=cfg_palette,
                legend=False,
                ax=ax,
                width=0.6,
                fliersize=2,
            )
            sns.stripplot(
                data=sub,
                x="_cfg_id",
                y=col,
                order=order,
                ax=ax,
                color="0.15",
                size=2.5,
                alpha=0.65,
                jitter=0.12,
            )
            _mark_method_groups(ax, methods, method_colors)
            ax.set_title(f"\u0394 {title}" if improve_over_none else title, pad=20)
            ax.set_xlabel("iterations / smoothing strength")
            ax.set_ylabel("improvement vs none" if improve_over_none else "")
            xt = np.arange(len(order))
            ax.set_xticks(xt)
            ax.set_xticklabels(tick_lbls, rotation=0, ha="center", fontsize=6.5)
            if improve_over_none:
                ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="-", zorder=1)
                if col == "mean_curvature_rms":
                    ax.set_ylim(-15.0, 15.0)
            elif col == "dice_overall":
                ax.set_ylim(0.9, 1.0)
            elif col == "mean_curvature_rms":
                ax.set_ylim(0.0, 40.0)

        for ax in axes_flat[len(metrics_spec):]:
            ax.set_visible(False)
        _method_legend(fig, axes_flat[-1], method_colors)
        _suptitle = (
            "Improvement over no smoothing (positive = better; box = case spread)"
            if improve_over_none
            else "Overall metrics by smoothing method (box = case spread)"
        )
        fig.suptitle(_suptitle, fontsize=12, y=1.02)
        p = out_dir / f"overall_metrics_boxplots{suffix}.{fig_format}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

        # Mean ± std summary bars (aggregated across cases)
        fig2, axes2 = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
        axes2_flat = axes2.ravel()
        for ax, (col, title, higher_better) in zip(axes2_flat, metrics_spec):
            sub = overall.dropna(subset=[col])
            if sub.empty:
                ax.set_visible(False)
                continue
            order, tick_lbls, methods = _cfg_order_and_labels(sub, col, ascending=not higher_better)
            agg = sub.groupby("_cfg_id")[col].agg(mean="mean", std="std").reindex(order)
            x = np.arange(len(order))
            ax.bar(
                x,
                agg["mean"],
                yerr=agg["std"].fillna(0.0),
                capsize=2,
                color=[method_colors[m] for m in methods],
                ecolor="0.35",
                error_kw={"linewidth": 1},
            )
            _mark_method_groups(ax, methods, method_colors)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_lbls, rotation=0, ha="center", fontsize=6.5)
            ax.set_title(f"\u0394 {title}" if improve_over_none else title, pad=20)
            ax.set_xlabel("iterations / smoothing strength")
            ax.set_ylabel("improvement vs none" if improve_over_none else "")
            if improve_over_none:
                ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="-", zorder=1)
                if col == "mean_curvature_rms":
                    ax.set_ylim(-15.0, 15.0)
            elif col == "dice_overall":
                ax.set_ylim(0.9, 1.0)
            elif col == "mean_curvature_rms":
                ax.set_ylim(0.0, 40.0)

        for ax in axes2_flat[len(metrics_spec):]:
            ax.set_visible(False)
        _method_legend(fig2, axes2_flat[-1], method_colors)
        _suptitle2 = (
            "Mean \u00b1 std improvement over no smoothing (positive = better)"
            if improve_over_none
            else "Mean \u00b1 std across cases by smoothing method"
        )
        fig2.suptitle(_suptitle2, fontsize=12, y=1.02)
        p2 = out_dir / f"overall_metrics_mean_std{suffix}.{fig_format}"
        fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        written.append(p2)

    rad = df[df["metric_scope"] == "radius_stratum"].copy()
    if improve_over_none and not rad.empty:
        rad = _improvement_over_none(rad, ["dice"], ["case_id", "radius_bin"], {"dice": True})
    if not rad.empty and rad["dice"].notna().any():
        rad = _add_config_columns(rad)
        methods = sorted(rad["method"].unique())
        n_m = max(1, len(methods))
        fig3, axes3 = plt.subplots(1, n_m, figsize=(6 * n_m, 4.5), squeeze=False, constrained_layout=True)
        dice_ylabel = "\u0394 Dice vs none" if improve_over_none else "Dice"
        for ax, method in zip(axes3[0], methods):
            sub = rad[rad["method"] == method].dropna(subset=["dice"])
            if sub.empty:
                ax.set_visible(False)
                continue
            order_bins = sorted(sub["radius_bin"].unique())
            sns.lineplot(
                data=sub,
                x="radius_bin",
                y="dice",
                hue="_cfg_lbl",
                style="_cfg_lbl",
                dashes=False,
                markers=True,
                err_style="band",
                ax=ax,
                legend="brief",
            )
            if improve_over_none:
                ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="-", zorder=1)
            ax.set_xticks(order_bins)
            lbl_map = sub.drop_duplicates("radius_bin").set_index("radius_bin")["radius_bin_label"]
            ax.set_xticklabels([str(lbl_map.get(b, b)) for b in order_bins], rotation=20, ha="right", fontsize=7)
            ax.set_xlabel("Vessel size (radius bin)")
            ax.set_ylabel(dice_ylabel)
            ax.set_title(f"{method}: {dice_ylabel} by caliber")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, title="config")
        _rad_suptitle = (
            "Stratified Dice improvement over no smoothing (positive = better)"
            if improve_over_none
            else "Stratified Dice (foreground voxels by nearest MIS radius)"
        )
        fig3.suptitle(_rad_suptitle, fontsize=11, y=1.05)
        p3 = out_dir / f"dice_by_vessel_radius{suffix}.{fig_format}"
        fig3.savefig(p3, dpi=dpi, bbox_inches="tight")
        plt.close(fig3)
        written.append(p3)

    # Smoothing-failure counts per configuration (diverged meshes or smoothing exceptions).
    # Prefer the combined `failed` column; fall back to `diverged` for older CSVs.
    fail_col = "failed" if "failed" in df.columns else ("diverged" if "diverged" in df.columns else None)
    fail_src = df[df["metric_scope"] == "overall"].copy()
    if not fail_src.empty and fail_col is not None:
        fail_label = "diverged / smoothing error" if fail_col == "failed" else "diverged"
        fail_src = _add_config_columns(fail_src)
        fail_src["_failed"] = fail_src[fail_col].map(_as_bool)
        if int(fail_src["_failed"].sum()) > 0:
            _failed_rows = fail_src[fail_src["_failed"]]
            _write_failed_cases_txt(
                _failed_rows,
                out_dir / "smoothing_failures.txt",
                fail_label,
                written,
            )
            _write_failed_case_counts_txt(
                _failed_rows,
                out_dir / "smoothing_failures_by_case.txt",
                fail_label,
                written,
            )
            order, tick_lbls, methods = _cfg_order_and_labels(fail_src)
            if order:
                method_colors = _method_colors(fail_src["method"].unique())
                counts = fail_src.groupby("_cfg_id")["_failed"].sum().reindex(order).fillna(0.0)
                fig4, (axc, axl) = plt.subplots(
                    1,
                    2,
                    figsize=(max(10.0, 0.45 * len(order) + 4.0), 5.0),
                    gridspec_kw={"width_ratios": [10, 1.4]},
                    constrained_layout=True,
                )
                x = np.arange(len(order))
                axc.bar(x, counts.values, color=[method_colors[m] for m in methods])
                _mark_method_groups(axc, methods, method_colors)
                axc.set_xticks(x)
                axc.set_xticklabels(tick_lbls, rotation=0, ha="center", fontsize=6.5)
                axc.set_xlabel("iterations / smoothing strength")
                axc.set_ylabel(f"# cases failed ({fail_label})")
                axc.set_title("Smoothing failures by configuration", pad=20)
                axl.set_visible(True)
                _method_legend(fig4, axl, method_colors)
                total_fail = int(fail_src["_failed"].sum())
                n_cases = int(fail_src["case_id"].nunique()) if "case_id" in fail_src.columns else 0
                fig4.suptitle(
                    f"Failed smoothing attempts (total {total_fail} across {n_cases} cases)",
                    fontsize=12,
                    y=1.04,
                )
                p4 = out_dir / f"smoothing_failures.{fig_format}"
                fig4.savefig(p4, dpi=dpi, bbox_inches="tight")
                plt.close(fig4)
                written.append(p4)

    return written
