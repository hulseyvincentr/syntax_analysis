#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bc_batch_lesion_group_summary_with_remaining_v1.py

Post-process a batch of per-bird *_cluster_bc_summary.csv files and make
lesion-hit-type comparison plots for three cluster sets:

  1) all qualifying clusters
  2) high-variance phrase-duration clusters (top-fraction set marked by the wrapper)
  3) remaining non-high-variance clusters (the complement of #2)

For each set, the script makes both:
  - cluster-level plots: every cluster is one point
  - bird-level plots: clusters are first aggregated within each bird, then each bird is one point

Expected input files are the per-bird outputs from bc_cluster_qc_and_summaries_*.py,
especially the files named:
    <ANIMAL_ID>_cluster_bc_summary.csv

Required columns in each cluster summary:
    animal_id, cluster_id, passes_min_balanced_duration, is_high_variance_cluster,
    bc_pre, bc_post, bc_prepost

The script can also use method-specific columns if present, for example:
    bc_pre_selected_bins, bc_post_selected_bins, bc_prepost_selected_bins
    bc_pre_full_contiguous_selected_runs, ...
    bc_pre_run_weighted_full_contiguous, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import math
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.stats import wilcoxon, kruskal
except Exception:  # pragma: no cover
    wilcoxon = None
    kruskal = None


SET_LABELS = {
    "all_clusters": "All qualifying clusters",
    "high_variance_clusters": "Top 30% phrase-duration variance clusters",
    "remaining_non_high_variance_clusters": "Remaining non-top-variance clusters",
}

SET_FILENAME_TOKEN = {
    "all_clusters": "all_clusters",
    "high_variance_clusters": "high_variance_clusters",
    "remaining_non_high_variance_clusters": "remaining_non_high_variance_clusters",
}

DEFAULT_LESION_ORDER = [
    "Complete Medial and Lateral lesion",
    "Partial Medial and Lateral lesion",
    "Lateral lesion only",
    "sham saline injection",
    "unknown",
]

DEFAULT_LESION_COLORS = {
    "Complete Medial and Lateral lesion": "#2B004F",
    "Partial Medial and Lateral lesion": "#6F4BAE",
    "Lateral lesion only": "#9A8FBF",
    "sham saline injection": "#707070",
    "unknown": "#BDBDBD",
}

METHOD_DEFS = {
    "selected_bins": (
        ["bc_pre_selected_bins", "bc_pre"],
        ["bc_post_selected_bins", "bc_post"],
        ["bc_prepost_selected_bins", "bc_prepost"],
        "Selected equal time bins",
    ),
    "full_contiguous_selected_runs": (
        ["bc_pre_full_contiguous_selected_runs"],
        ["bc_post_full_contiguous_selected_runs"],
        ["bc_prepost_full_contiguous_selected_runs"],
        "Full-contiguous selected runs",
    ),
    "run_weighted_full_contiguous": (
        ["bc_pre_run_weighted_full_contiguous"],
        ["bc_post_run_weighted_full_contiguous"],
        ["bc_prepost_run_weighted_full_contiguous"],
        "Run-weighted full-contiguous selected runs",
    ),
}


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin(["true", "1", "yes", "y", "t"])


def _normalize_animal_id(x: Any) -> str:
    return str(x).strip()


def _normalize_hit_type(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unknown"
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)

    sham = {"sham saline injection", "sham", "saline", "sham control", "control"}
    lateral = {
        "lateral lesion only",
        "lateral hit only",
        "area x visible (single hit)",
        "single hit",
        "single lateral hit",
        "area x visible (lateral only)",
    }
    partial = {
        "partial medial and lateral lesion",
        "area x visible (medial+lateral hit)",
        "area x visible medial+lateral hit",
        "medial+lateral hit",
        "m+l hit",
        "medial+lateral",
        "medial lateral",
        "area x visible (medial + lateral hit)",
    }
    complete = {
        "complete medial and lateral lesion",
        "large lesion area x not visible",
        "large lesion, area x not visible",
        "area x not visible",
        "large lesion",
        "area x visible and large lesion area x not visible",
    }

    if s in sham:
        return "sham saline injection"
    if s in lateral:
        return "Lateral lesion only"
    if s in partial:
        return "Partial Medial and Lateral lesion"
    if s in complete:
        return "Complete Medial and Lateral lesion"

    # Catch partial phrases conservatively.
    if "sham" in s or "saline" in s:
        return "sham saline injection"
    if "single" in s or "lateral only" in s or "lateral hit only" in s:
        return "Lateral lesion only"
    if "large lesion" in s or "not visible" in s:
        return "Complete Medial and Lateral lesion"
    if "medial" in s and "lateral" in s:
        return "Partial Medial and Lateral lesion"
    return str(x).strip() or "unknown"


def _load_metadata_hit_types(
    excel_path: Path,
    *,
    sheet_name: str,
    animal_col: str,
    hit_type_col: str,
) -> pd.DataFrame:
    excel_path = Path(excel_path)
    try:
        meta = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception:
        meta = pd.read_excel(excel_path, sheet_name=0)
    if animal_col not in meta.columns:
        raise ValueError(f"Metadata column {animal_col!r} not found. Found columns: {list(meta.columns)}")
    if hit_type_col not in meta.columns:
        raise ValueError(f"Metadata column {hit_type_col!r} not found. Found columns: {list(meta.columns)}")
    out = meta[[animal_col, hit_type_col]].copy()
    out = out.rename(columns={animal_col: "animal_id", hit_type_col: "raw_lesion_hit_type"})
    out["animal_id"] = out["animal_id"].map(_normalize_animal_id)
    out = out.dropna(subset=["animal_id"]).drop_duplicates(subset=["animal_id"], keep="first")
    out["lesion_hit_type"] = out["raw_lesion_hit_type"].map(_normalize_hit_type)
    return out[["animal_id", "raw_lesion_hit_type", "lesion_hit_type"]]


def _load_color_json(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return dict(DEFAULT_LESION_COLORS)
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    colors = dict(DEFAULT_LESION_COLORS)
    if isinstance(raw, dict):
        # Supports either {group: color} or {"colors": {group: color}}
        if "colors" in raw and isinstance(raw["colors"], dict):
            raw = raw["colors"]
        for k, v in raw.items():
            if isinstance(v, str):
                colors[str(k)] = v
    return colors


def _find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _available_methods(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    methods = []
    for method in requested:
        pre_cands, post_cands, prepost_cands, _ = METHOD_DEFS[method]
        if _find_first_existing_column(df, pre_cands) and _find_first_existing_column(df, post_cands):
            methods.append(method)
    return methods


def collect_cluster_summaries(
    bc_root: Path,
    metadata_excel: Path,
    *,
    metadata_sheet: str,
    metadata_animal_col: str,
    metadata_hit_type_col: str,
    min_balanced_duration_s: Optional[float] = None,
) -> pd.DataFrame:
    bc_root = Path(bc_root)
    files = sorted(p for p in bc_root.rglob("*_cluster_bc_summary.csv") if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No *_cluster_bc_summary.csv files found under: {bc_root}")

    parts = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
            continue
        if df.empty:
            continue
        if "animal_id" not in df.columns:
            animal = p.name.replace("_cluster_bc_summary.csv", "")
            df["animal_id"] = animal
        df["source_csv"] = str(p)
        parts.append(df)
    if not parts:
        raise ValueError("No readable non-empty cluster summary CSVs found.")

    out = pd.concat(parts, ignore_index=True)
    out["animal_id"] = out["animal_id"].map(_normalize_animal_id)

    if "passes_min_balanced_duration" in out.columns:
        out = out[_bool_series(out["passes_min_balanced_duration"])].copy()
    if min_balanced_duration_s is not None and "balanced_duration_s_per_group" in out.columns:
        out["balanced_duration_s_per_group"] = pd.to_numeric(out["balanced_duration_s_per_group"], errors="coerce")
        out = out[out["balanced_duration_s_per_group"] >= float(min_balanced_duration_s)].copy()

    if "is_high_variance_cluster" not in out.columns:
        raise ValueError(
            "Cluster summaries do not contain 'is_high_variance_cluster'. "
            "Re-run the wrapper with --phrase-csv so high-vs-remaining sets can be defined."
        )
    out["is_high_variance_cluster"] = _bool_series(out["is_high_variance_cluster"])

    meta = _load_metadata_hit_types(
        metadata_excel,
        sheet_name=metadata_sheet,
        animal_col=metadata_animal_col,
        hit_type_col=metadata_hit_type_col,
    )
    out = out.merge(meta, on="animal_id", how="left")
    out["lesion_hit_type"] = out["lesion_hit_type"].fillna("unknown")
    return out.reset_index(drop=True)


def build_analysis_tables(
    cluster_df: pd.DataFrame,
    *,
    methods: Sequence[str],
    bird_aggregate: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    set_masks = {
        "all_clusters": np.ones(len(cluster_df), dtype=bool),
        "high_variance_clusters": cluster_df["is_high_variance_cluster"].to_numpy(dtype=bool),
        "remaining_non_high_variance_clusters": (~cluster_df["is_high_variance_cluster"]).to_numpy(dtype=bool),
    }

    for method in methods:
        pre_cands, post_cands, prepost_cands, method_label = METHOD_DEFS[method]
        pre_col = _find_first_existing_column(cluster_df, pre_cands)
        post_col = _find_first_existing_column(cluster_df, post_cands)
        prepost_col = _find_first_existing_column(cluster_df, prepost_cands)
        if pre_col is None or post_col is None:
            print(f"[WARN] Skipping {method}: required BC columns not found.")
            continue

        for set_name, mask in set_masks.items():
            sub = cluster_df.loc[mask].copy()
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                bc_pre = pd.to_numeric(pd.Series([r.get(pre_col)]), errors="coerce").iloc[0]
                bc_post = pd.to_numeric(pd.Series([r.get(post_col)]), errors="coerce").iloc[0]
                bc_prepost = np.nan
                if prepost_col is not None:
                    bc_prepost = pd.to_numeric(pd.Series([r.get(prepost_col)]), errors="coerce").iloc[0]
                if not (np.isfinite(bc_pre) and np.isfinite(bc_post)):
                    continue
                rows.append({
                    "set_name": set_name,
                    "set_label": SET_LABELS.get(set_name, set_name),
                    "bc_method": method,
                    "bc_method_label": method_label,
                    "animal_id": r.get("animal_id"),
                    "cluster_id": r.get("cluster_id"),
                    "cluster_token": r.get("cluster_token", r.get("cluster_id")),
                    "lesion_hit_type": r.get("lesion_hit_type", "unknown"),
                    "raw_lesion_hit_type": r.get("raw_lesion_hit_type", ""),
                    "bc_pre": float(bc_pre),
                    "bc_post": float(bc_post),
                    "bc_prepost": float(bc_prepost) if np.isfinite(bc_prepost) else np.nan,
                    "bc_delta_post_minus_pre": float(bc_post - bc_pre),
                    "source_csv": r.get("source_csv", ""),
                })

    cluster_long = pd.DataFrame(rows)
    if cluster_long.empty:
        raise ValueError("No finite BC rows available after filtering.")

    # Bird-level: aggregate clusters within each bird/set/method.
    agg_func = np.nanmedian if bird_aggregate == "median" else np.nanmean
    bird_rows: List[Dict[str, Any]] = []
    group_cols = ["set_name", "set_label", "bc_method", "bc_method_label", "animal_id", "lesion_hit_type", "raw_lesion_hit_type"]
    for keys, g in cluster_long.groupby(group_cols, dropna=False, sort=False):
        d = dict(zip(group_cols, keys))
        d.update({
            "n_clusters": int(len(g)),
            "bc_pre": float(agg_func(g["bc_pre"].to_numpy(dtype=float))),
            "bc_post": float(agg_func(g["bc_post"].to_numpy(dtype=float))),
            "bc_prepost": float(agg_func(g["bc_prepost"].to_numpy(dtype=float))) if np.isfinite(g["bc_prepost"]).any() else np.nan,
        })
        d["bc_delta_post_minus_pre"] = float(d["bc_post"] - d["bc_pre"])
        d["bird_aggregate"] = bird_aggregate
        bird_rows.append(d)
    bird_level = pd.DataFrame(bird_rows)
    return cluster_long, bird_level


def _ordered_groups(df: pd.DataFrame, order: Sequence[str]) -> List[str]:
    present = list(dict.fromkeys(df["lesion_hit_type"].astype(str)))
    return [g for g in order if g in present] + [g for g in present if g not in order]


def _jitter(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return rng.uniform(-scale, scale, size=n)


def _boxplot_with_points(
    ax: plt.Axes,
    arrays: List[np.ndarray],
    positions: List[float],
    colors: List[str],
    *,
    width: float,
    point_size: float,
    rng: np.random.Generator,
    alpha: float = 0.75,
) -> None:
    for arr, pos, color in zip(arrays, positions, colors):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        ax.boxplot(
            [arr],
            positions=[pos],
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            boxprops=dict(facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.25),
            whiskerprops=dict(color="black", linewidth=0.9),
            capprops=dict(color="black", linewidth=0.9),
        )
        x = pos + _jitter(arr.size, width * 0.24, rng)
        ax.scatter(x, arr, s=point_size, color=color, alpha=alpha, edgecolors="none", zorder=3)


def plot_pre_post_by_lesion(
    df: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    colors: Dict[str, str],
    order: Sequence[str],
    level_label: str,
    dpi: int,
) -> None:
    _safe_mkdir(Path(out_png).parent)
    groups = _ordered_groups(df, order)
    rng = np.random.default_rng(0)
    fig_w = max(9.5, 2.0 * len(groups) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    arrays, positions, cols = [], [], []
    for i, group in enumerate(groups):
        sub = df[df["lesion_hit_type"].astype(str) == group]
        arrays.extend([sub["bc_pre"].to_numpy(dtype=float), sub["bc_post"].to_numpy(dtype=float)])
        positions.extend([i - 0.18, i + 0.18])
        base = colors.get(group, "#808080")
        cols.extend([base, base])
    _boxplot_with_points(ax, arrays, positions, cols, width=0.28, point_size=18 if level_label == "cluster" else 40, rng=rng)

    # Draw paired lines within animal/cluster if the dataset is not too large.
    id_cols = ["animal_id"] if level_label == "bird" else ["animal_id", "cluster_id"]
    if len(df) <= 550:
        for _, r in df.iterrows():
            group = str(r["lesion_hit_type"])
            if group not in groups:
                continue
            x0 = groups.index(group) - 0.18
            x1 = groups.index(group) + 0.18
            ax.plot([x0, x1], [r["bc_pre"], r["bc_post"]], color="0.78", linewidth=0.55, zorder=1)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel("Bhattacharyya coefficient")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", markerfacecolor="white", markersize=7, label="Pre: early vs late"),
        Line2D([0], [0], marker="o", linestyle="None", color="black", markerfacecolor="black", markersize=7, label="Post: early vs late"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_delta_by_lesion(
    df: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    colors: Dict[str, str],
    order: Sequence[str],
    level_label: str,
    dpi: int,
) -> None:
    _safe_mkdir(Path(out_png).parent)
    groups = _ordered_groups(df, order)
    rng = np.random.default_rng(1)
    fig_w = max(8.8, 1.65 * len(groups) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))
    arrays, positions, cols = [], [], []
    for i, group in enumerate(groups):
        sub = df[df["lesion_hit_type"].astype(str) == group]
        arrays.append(sub["bc_delta_post_minus_pre"].to_numpy(dtype=float))
        positions.append(i)
        cols.append(colors.get(group, "#808080"))
    _boxplot_with_points(ax, arrays, positions, cols, width=0.55, point_size=18 if level_label == "cluster" else 44, rng=rng)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1.1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel("Post BC − Pre BC")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def compute_stats(df: pd.DataFrame, *, level: str, order: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["set_name", "set_label", "bc_method", "bc_method_label"]
    for keys, sub0 in df.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        for lesion, sub in sub0.groupby("lesion_hit_type", sort=False):
            pre = sub["bc_pre"].to_numpy(dtype=float)
            post = sub["bc_post"].to_numpy(dtype=float)
            delta = sub["bc_delta_post_minus_pre"].to_numpy(dtype=float)
            mask = np.isfinite(pre) & np.isfinite(post)
            p_w = np.nan
            if wilcoxon is not None and mask.sum() >= 2:
                try:
                    p_w = float(wilcoxon(pre[mask], post[mask], alternative="two-sided").pvalue)
                except Exception:
                    p_w = np.nan
            rows.append({
                **base,
                "analysis_level": level,
                "lesion_hit_type": lesion,
                "n_observations": int(mask.sum()),
                "n_animals": int(sub["animal_id"].nunique()),
                "mean_bc_pre": float(np.nanmean(pre)),
                "mean_bc_post": float(np.nanmean(post)),
                "mean_delta_post_minus_pre": float(np.nanmean(delta)),
                "median_bc_pre": float(np.nanmedian(pre)),
                "median_bc_post": float(np.nanmedian(post)),
                "median_delta_post_minus_pre": float(np.nanmedian(delta)),
                "paired_pre_post_wilcoxon_p": p_w,
                "paired_pre_post_wilcoxon_p_formatted": _fmt_p(p_w),
            })
        # Across-lesion Kruskal-Wallis on delta.
        if kruskal is not None:
            arrays = []
            labels = []
            for lesion in _ordered_groups(sub0, order):
                vals = sub0.loc[sub0["lesion_hit_type"].astype(str) == lesion, "bc_delta_post_minus_pre"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size >= 2:
                    arrays.append(vals)
                    labels.append(lesion)
            if len(arrays) >= 2:
                try:
                    stat, p = kruskal(*arrays)
                except Exception:
                    stat, p = np.nan, np.nan
                rows.append({
                    **base,
                    "analysis_level": level,
                    "lesion_hit_type": "ACROSS_LESION_GROUPS",
                    "n_observations": int(sum(len(a) for a in arrays)),
                    "n_animals": int(sub0["animal_id"].nunique()),
                    "mean_bc_pre": np.nan,
                    "mean_bc_post": np.nan,
                    "mean_delta_post_minus_pre": np.nan,
                    "median_bc_pre": np.nan,
                    "median_bc_post": np.nan,
                    "median_delta_post_minus_pre": np.nan,
                    "paired_pre_post_wilcoxon_p": np.nan,
                    "paired_pre_post_wilcoxon_p_formatted": "",
                    "delta_kruskal_statistic": float(stat),
                    "delta_kruskal_p": float(p),
                    "delta_kruskal_p_formatted": _fmt_p(float(p)),
                    "delta_kruskal_groups": ";".join(labels),
                })
    return pd.DataFrame(rows)


def make_all_plots(
    cluster_long: pd.DataFrame,
    bird_level: pd.DataFrame,
    out_dir: Path,
    *,
    colors: Dict[str, str],
    lesion_order: Sequence[str],
    dpi: int,
) -> List[Path]:
    paths: List[Path] = []
    for level_label, df in [("cluster", cluster_long), ("bird", bird_level)]:
        level_dir = Path(out_dir) / f"{level_label}_level"
        _safe_mkdir(level_dir)
        for (set_name, method), sub in df.groupby(["set_name", "bc_method"], sort=False):
            if sub.empty:
                continue
            set_label = SET_LABELS.get(set_name, set_name)
            method_label = METHOD_DEFS.get(method, (None, None, None, method))[3]
            n_obs = len(sub)
            n_birds = sub["animal_id"].nunique()
            title_base = f"{level_label.capitalize()} level: {set_label}\n{method_label} | N={n_obs} {level_label}s, {n_birds} birds"
            token = f"{level_label}_level_{_clean_filename(method)}_{SET_FILENAME_TOKEN.get(set_name, _clean_filename(set_name))}"
            p1 = level_dir / f"{token}_pre_vs_post_by_lesion.png"
            plot_pre_post_by_lesion(
                sub,
                p1,
                title=title_base,
                colors=colors,
                order=lesion_order,
                level_label=level_label,
                dpi=dpi,
            )
            paths.append(p1)
            p2 = level_dir / f"{token}_delta_post_minus_pre_by_lesion.png"
            plot_delta_by_lesion(
                sub,
                p2,
                title=title_base,
                colors=colors,
                order=lesion_order,
                level_label=level_label,
                dpi=dpi,
            )
            paths.append(p2)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch lesion-hit-type BC summary plots, including remaining non-top-variance clusters."
    )
    p.add_argument("--bc-root", required=True, help="Root folder containing per-bird *_cluster_bc_summary.csv files.")
    p.add_argument("--metadata-excel", required=True, help="Excel file with animal IDs and lesion hit types.")
    p.add_argument("--out-dir", default=None, help="Output folder. Defaults to <bc-root>/_batch_lesion_group_summaries.")
    p.add_argument("--metadata-sheet", default="animal_hit_type_summary")
    p.add_argument("--metadata-animal-col", default="Animal ID")
    p.add_argument("--metadata-hit-type-col", default="Lesion hit type")
    p.add_argument("--bc-method", choices=list(METHOD_DEFS.keys()) + ["all"], default="selected_bins")
    p.add_argument("--bird-aggregate", choices=["median", "mean"], default="median")
    p.add_argument("--min-balanced-duration-s", type=float, default=None, help="Optional extra duration filter. Usually not needed because each cluster summary already has passes_min_balanced_duration.")
    p.add_argument("--color-json", default=None, help="Optional lesion-group color JSON. Supports either {group: color} or {'colors': {group: color}}.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bc_root = Path(args.bc_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else bc_root / "_batch_lesion_group_summaries"
    _safe_mkdir(out_dir)

    cluster_df = collect_cluster_summaries(
        bc_root,
        Path(args.metadata_excel).expanduser().resolve(),
        metadata_sheet=str(args.metadata_sheet),
        metadata_animal_col=str(args.metadata_animal_col),
        metadata_hit_type_col=str(args.metadata_hit_type_col),
        min_balanced_duration_s=args.min_balanced_duration_s,
    )

    requested_methods = list(METHOD_DEFS.keys()) if args.bc_method == "all" else [args.bc_method]
    methods = _available_methods(cluster_df, requested_methods)
    if not methods:
        raise ValueError(f"None of the requested BC methods are available. Requested: {requested_methods}. Columns: {list(cluster_df.columns)}")

    cluster_long, bird_level = build_analysis_tables(
        cluster_df,
        methods=methods,
        bird_aggregate=str(args.bird_aggregate),
    )

    colors = _load_color_json(Path(args.color_json).expanduser().resolve() if args.color_json else None)
    lesion_order = [g for g in DEFAULT_LESION_ORDER if g in set(cluster_long["lesion_hit_type"].astype(str))]
    lesion_order += [g for g in sorted(set(cluster_long["lesion_hit_type"].astype(str))) if g not in lesion_order]

    cluster_csv = out_dir / "bc_batch_cluster_level_long.csv"
    bird_csv = out_dir / "bc_batch_bird_level_summary.csv"
    cluster_long.to_csv(cluster_csv, index=False)
    bird_level.to_csv(bird_csv, index=False)
    print(f"[SAVE] {cluster_csv}")
    print(f"[SAVE] {bird_csv}")

    stats = pd.concat([
        compute_stats(cluster_long, level="cluster", order=lesion_order),
        compute_stats(bird_level, level="bird", order=lesion_order),
    ], ignore_index=True)
    stats_csv = out_dir / "bc_batch_lesion_group_stats.csv"
    stats.to_csv(stats_csv, index=False)
    print(f"[SAVE] {stats_csv}")

    paths = make_all_plots(
        cluster_long,
        bird_level,
        out_dir,
        colors=colors,
        lesion_order=lesion_order,
        dpi=int(args.dpi),
    )
    for p in paths:
        print(f"[SAVE] {p}")

    # A compact manifest to remind you what was included.
    manifest = pd.DataFrame({
        "bc_root": [str(bc_root)],
        "out_dir": [str(out_dir)],
        "n_cluster_summary_files": [cluster_df["source_csv"].nunique()],
        "n_animals": [cluster_long["animal_id"].nunique()],
        "n_cluster_level_rows": [len(cluster_long)],
        "n_bird_level_rows": [len(bird_level)],
        "bc_methods": [";".join(methods)],
        "sets": [";".join(SET_LABELS.keys())],
    })
    manifest_csv = out_dir / "bc_batch_lesion_group_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    print(f"[SAVE] {manifest_csv}")


if __name__ == "__main__":
    main()
