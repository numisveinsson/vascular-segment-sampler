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


def _compact_params_label(method: str, params_json: str) -> str:
    try:
        p = json.loads(params_json) if isinstance(params_json, str) else params_json
    except (json.JSONDecodeError, TypeError):
        return method
    if not p:
        return method
    # Drop flags that clutter legends
    skip = {"boundary", "feature", "boundary_smoothing"}
    items = [(k, v) for k, v in sorted(p.items()) if k not in skip]
    if not items:
        return method
    s = ", ".join(f"{k}={v}" for k, v in items[:5])
    return f"{method}: {s}"


def _add_config_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_cfg_id"] = out["method"].astype(str) + "|" + out["params_sha10"].astype(str)
    out["_cfg_lbl"] = [
        _compact_params_label(m, pj) for m, pj in zip(out["method"], out["params_json"])
    ]
    return out


def _cfg_order_and_labels(df: pd.DataFrame, metric: str, ascending: bool = False):
    """Sort configs by mean metric (higher dice is better → ascending=False)."""
    g = df.groupby("_cfg_id", as_index=False)[metric].mean().dropna()
    g = g.sort_values(metric, ascending=ascending)
    order = g["_cfg_id"].tolist()
    id_to_lbl = df.drop_duplicates("_cfg_id").set_index("_cfg_id")["_cfg_lbl"].to_dict()
    labels = [id_to_lbl.get(i, i) for i in order]
    return order, labels


def generate_study_plots(
    csv_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    *,
    dpi: int = 150,
    fig_format: str = "png",
) -> list[Path]:
    """
    Read study CSV and write metric figures into out_dir.

    Returns list of written file paths.
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

    overall = df[df["metric_scope"] == "overall"].copy()
    if not overall.empty:
        overall = _add_config_columns(overall)

        metrics_spec = [
            ("dice_overall", "Dice (overall)", True),
            ("assd_mm", "ASSD (length)", False),
            ("hd95_mm", "HD95 (length)", False),
            ("normal_error_mean_deg", "Normal error mean (deg)", False),
            ("normal_error_max_deg", "Normal error max (deg)", False),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        axes_flat = axes.ravel()
        for ax, (col, title, higher_better) in zip(axes_flat, metrics_spec):
            sub = overall.dropna(subset=[col])
            if sub.empty:
                ax.set_visible(False)
                continue
            order, tick_lbls = _cfg_order_and_labels(sub, col, ascending=not higher_better)
            sns.boxplot(
                data=sub,
                x="_cfg_id",
                y=col,
                order=order,
                ax=ax,
                color="#a8d5e5",
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
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("")
            xt = np.arange(len(order))
            ax.set_xticks(xt)
            ax.set_xticklabels(tick_lbls, rotation=38, ha="right", fontsize=7)

        axes_flat[-1].set_visible(False)
        fig.suptitle("Overall metrics by smoothing configuration (box = case spread)", fontsize=12, y=1.02)
        p = out_dir / f"overall_metrics_boxplots.{fig_format}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

        # Mean ± std summary bars (aggregated across cases)
        fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        axes2_flat = axes2.ravel()
        for ax, (col, title, higher_better) in zip(axes2_flat, metrics_spec):
            sub = overall.dropna(subset=[col])
            if sub.empty:
                ax.set_visible(False)
                continue
            order, tick_lbls = _cfg_order_and_labels(sub, col, ascending=not higher_better)
            agg = sub.groupby("_cfg_id")[col].agg(mean="mean", std="std").reindex(order)
            x = np.arange(len(order))
            ax.bar(
                x,
                agg["mean"],
                yerr=agg["std"].fillna(0.0),
                capsize=2,
                color="#6a9fb5",
                ecolor="0.35",
                error_kw={"linewidth": 1},
            )
            ax.set_xticks(x)
            ax.set_xticklabels(tick_lbls, rotation=38, ha="right", fontsize=7)
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("")

        axes2_flat[-1].set_visible(False)
        fig2.suptitle("Mean ± std across cases", fontsize=12, y=1.02)
        p2 = out_dir / f"overall_metrics_mean_std.{fig_format}"
        fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        written.append(p2)

    rad = df[df["metric_scope"] == "radius_stratum"].copy()
    if not rad.empty and rad["dice"].notna().any():
        rad = _add_config_columns(rad)
        methods = sorted(rad["method"].unique())
        n_m = max(1, len(methods))
        fig3, axes3 = plt.subplots(1, n_m, figsize=(6 * n_m, 4.5), squeeze=False, constrained_layout=True)
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
            ax.set_xticks(order_bins)
            lbl_map = sub.drop_duplicates("radius_bin").set_index("radius_bin")["radius_bin_label"]
            ax.set_xticklabels([str(lbl_map.get(b, b)) for b in order_bins], rotation=20, ha="right", fontsize=7)
            ax.set_xlabel("Vessel size (radius bin)")
            ax.set_ylabel("Dice")
            ax.set_title(f"{method}: Dice by caliber")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, title="config")
        fig3.suptitle("Stratified Dice (foreground voxels by nearest MIS radius)", fontsize=11, y=1.05)
        p3 = out_dir / f"dice_by_vessel_radius.{fig_format}"
        fig3.savefig(p3, dpi=dpi, bbox_inches="tight")
        plt.close(fig3)
        written.append(p3)

    return written
