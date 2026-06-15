#!/usr/bin/env python3
"""
aggregate_phrase_position_local_rate_groups_v1.py

Aggregate phrase-position acoustic/cross-correlation results across syllables.

Designed for outputs from:
    phrase_position_acoustic_crosscorr_v17_consistent_timescales.py

Main goals:
1) Compare syllables in the top 30% phrase-duration-variance set,
   syllables in the lower 70%, and all syllables combined.
2) Highlight syllable labels with the largest early-vs-late within-phrase
   changes in spectrogram cross-correlation.
3) Highlight syllable labels with the highest local acoustic-change rates,
   including local correlation-drop rate.

Inputs:
- Combined segment CSV, usually:
    <v17-root>/<animal>_all_labels_phrase_position_segment_features.csv
- Combined local pair CSV, usually:
    <v17-root>/<animal>_all_labels_local_shortlag_pairs.csv
- Top-30% phrase-duration-variance CSV, usually:
    /Volumes/my_own_SSD/updated_AreaX_outputs/phrase_duration_batch_outputs/_batch_summary/batch_aligned_phrase_duration_variance_top30.csv

Outputs:
- group-comparison figures for top30/lower70/all
- ranked syllable figures and CSVs
- per-label line plots for top highlighted syllables

Notes:
- The group-comparison line plots are label-balanced by default:
  the script first computes a median per label per x-bin, then takes the
  median across labels. This avoids letting one long stuttered phrase dominate
  the aggregate curve.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PERIOD_ORDER = ["early_pre", "late_pre", "early_post", "late_post"]
PRE_POST_ORDER = ["pre", "post"]
GROUP_ORDER = [
    "Top 30% phrase-duration variance",
    "Lower 70% phrase-duration variance",
    "All syllables",
]


def ensure_dir(path: os.PathLike | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_label_value(label) -> str:
    """Stable string representation for labels like 19, 19.0, '19'."""
    if isinstance(label, bytes):
        label = label.decode(errors="replace")
    if label is None:
        return ""
    try:
        if pd.isna(label):
            return ""
    except Exception:
        pass
    s = str(label).strip()
    if s == "":
        return ""
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def add_normalized_label_column(df: pd.DataFrame, src_col: str = "label", dst_col: str = "label_norm") -> pd.DataFrame:
    out = df.copy()
    if src_col not in out.columns:
        raise ValueError(f"Missing required label column: {src_col}")
    out[dst_col] = [normalize_label_value(x) for x in out[src_col]]
    return out


def find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def default_csv_paths(v17_root: Path, animal_id: str) -> Tuple[Path, Path]:
    """Infer combined v17 CSV paths from root and animal id."""
    seg_candidates = [
        v17_root / f"{animal_id}_all_labels_phrase_position_segment_features.csv",
        v17_root / "all_labels_phrase_position_segment_features.csv",
    ]
    local_candidates = [
        v17_root / f"{animal_id}_all_labels_local_shortlag_pairs.csv",
        v17_root / "all_labels_local_shortlag_pairs.csv",
    ]
    seg = find_first_existing(seg_candidates)
    local = find_first_existing(local_candidates)
    if seg is None:
        raise FileNotFoundError(
            "Could not find combined segment CSV. Tried:\n"
            + "\n".join(str(x) for x in seg_candidates)
            + "\nPass --segment-csv explicitly if it lives elsewhere."
        )
    if local is None:
        raise FileNotFoundError(
            "Could not find combined local-pair CSV. Tried:\n"
            + "\n".join(str(x) for x in local_candidates)
            + "\nPass --local-pair-csv explicitly if it lives elsewhere."
        )
    return seg, local


def choose_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True, description: str = "column") -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    canon = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for c in candidates:
        key = str(c).strip().lower().replace(" ", "_")
        if key in canon:
            return canon[key]
    if required:
        raise ValueError(
            f"Could not identify {description}. Tried candidates: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_top30_labels(top30_csv: Path, animal_id: str, label_col: Optional[str] = None, animal_col: Optional[str] = None) -> set[str]:
    """Load top-30 phrase-duration-variance labels for the requested animal."""
    df = pd.read_csv(top30_csv)
    if animal_col is None:
        animal_col = choose_column(
            df,
            ["animal_id", "Animal ID", "animal", "bird", "Animal", "Bird", "subject"],
            required=False,
            description="animal ID column in top30 CSV",
        )
    if animal_col is not None:
        df = df[df[animal_col].astype(str) == str(animal_id)].copy()

    if label_col is None:
        label_col = choose_column(
            df,
            [
                "label",
                "cluster_label",
                "syllable_label",
                "hdbscan_label",
                "Syllable label",
                "Syllable",
                "syllable",
                "mapped_syllable_order",
                "hdbscan_labels",
            ],
            required=True,
            description="label column in top30 CSV",
        )

    labels = {normalize_label_value(x) for x in df[label_col].dropna().tolist()}
    labels = {x for x in labels if x != "" and x != "-1"}
    return labels


def add_variance_groups(df: pd.DataFrame, top30_labels: set[str], label_col: str = "label_norm") -> pd.DataFrame:
    """Add top30/lower70 group, and duplicate rows for All syllables."""
    if df.empty:
        return df.copy()
    base = df.copy()
    if label_col not in base.columns:
        base = add_normalized_label_column(base, src_col="label", dst_col=label_col)
    base["variance_group"] = np.where(
        base[label_col].isin(top30_labels),
        "Top 30% phrase-duration variance",
        "Lower 70% phrase-duration variance",
    )
    all_rows = base.copy()
    all_rows["variance_group"] = "All syllables"
    return pd.concat([base, all_rows], ignore_index=True)


def to_numeric_inplace(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def round_to_bin(x: pd.Series, bin_width: float) -> pd.Series:
    if bin_width <= 0:
        return x
    return np.round(pd.to_numeric(x, errors="coerce") / bin_width) * bin_width


def summarize_line_label_balanced(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_bin_width: float,
    min_rows_per_label_bin: int = 1,
    min_labels_per_group_bin: int = 1,
) -> pd.DataFrame:
    """Aggregate curves by first taking median per label/x-bin, then across labels."""
    need = ["variance_group", "pre_post", "label_norm", x_col, y_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for line summary: {missing}")

    work = df[need].copy()
    work = to_numeric_inplace(work, [x_col, y_col])
    work = work[np.isfinite(work[x_col]) & np.isfinite(work[y_col])].copy()
    if work.empty:
        return pd.DataFrame()

    work["x_bin"] = round_to_bin(work[x_col], x_bin_width)

    label_bin = (
        work.groupby(["variance_group", "pre_post", "label_norm", "x_bin"], dropna=False)
        .agg(
            label_bin_median=(y_col, "median"),
            label_bin_mean=(y_col, "mean"),
            n_rows=(y_col, "size"),
        )
        .reset_index()
    )
    label_bin = label_bin[label_bin["n_rows"] >= int(min_rows_per_label_bin)].copy()
    if label_bin.empty:
        return pd.DataFrame()

    def q25(x):
        return float(np.nanpercentile(x, 25))

    def q75(x):
        return float(np.nanpercentile(x, 75))

    summary = (
        label_bin.groupby(["variance_group", "pre_post", "x_bin"], dropna=False)
        .agg(
            median=("label_bin_median", "median"),
            mean=("label_bin_median", "mean"),
            q25=("label_bin_median", q25),
            q75=("label_bin_median", q75),
            n_labels=("label_norm", "nunique"),
            n_label_bins=("label_bin_median", "size"),
        )
        .reset_index()
    )
    summary = summary[summary["n_labels"] >= int(min_labels_per_group_bin)].copy()
    return summary


def plot_group_panels(
    summary: pd.DataFrame,
    out_png: Path,
    title: str,
    y_label: str,
    x_label: str,
    x_max: Optional[float] = None,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    if summary.empty:
        print(f"[WARN] No data for {out_png.name}; skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    prepost_style = {
        "pre": {"marker": "o", "linestyle": "-", "label": "Pre"},
        "post": {"marker": "o", "linestyle": "-", "label": "Post"},
    }

    if y_lim is None:
        y = summary[["q25", "q75", "median"]].to_numpy(dtype=float).ravel()
        y = y[np.isfinite(y)]
        if y.size:
            lo, hi = np.nanpercentile(y, [1, 99])
            pad = 0.08 * (hi - lo) if hi > lo else 0.1
            y_lim = (lo - pad, hi + pad)

    for ax, group in zip(axes, GROUP_ORDER):
        sub_g = summary[summary["variance_group"] == group].copy()
        for prepost in PRE_POST_ORDER:
            sub = sub_g[sub_g["pre_post"] == prepost].sort_values("x_bin")
            if sub.empty:
                continue
            x = sub["x_bin"].to_numpy(dtype=float)
            med = sub["median"].to_numpy(dtype=float)
            q25 = sub["q25"].to_numpy(dtype=float)
            q75 = sub["q75"].to_numpy(dtype=float)
            ax.plot(x, med, **prepost_style[prepost])
            ax.fill_between(x, q25, q75, alpha=0.16)

        ax.set_title(group)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)
        if x_max is not None:
            ax.set_xlim(left=0, right=x_max)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        n_by_group = sub_g.groupby("pre_post")["n_labels"].max().to_dict()
        if n_by_group:
            txt = ", ".join([f"{k}: {int(v)} labels" for k, v in n_by_group.items() if np.isfinite(v)])
            ax.text(
                0.02, 0.98, txt,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=2),
            )

    axes[0].set_ylabel(y_label)
    handles, labels = axes[-1].get_legend_handles_labels()
    seen = set()
    handles2, labels2 = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            handles2.append(h)
            labels2.append(lab)
            seen.add(lab)
    fig.legend(handles2, labels2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(title, y=1.12, fontsize=14)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_overlay_groups(
    summary: pd.DataFrame,
    out_png: Path,
    title: str,
    y_label: str,
    x_label: str,
    prepost: str = "post",
    x_max: Optional[float] = None,
) -> None:
    """One-axis overlay of top30/lower70/all, usually post only."""
    if summary.empty:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2), constrained_layout=True)
    for group in GROUP_ORDER:
        sub = summary[(summary["variance_group"] == group) & (summary["pre_post"] == prepost)].sort_values("x_bin")
        if sub.empty:
            continue
        ax.plot(sub["x_bin"], sub["median"], marker="o", linewidth=2, label=group)
        ax.fill_between(
            sub["x_bin"].to_numpy(dtype=float),
            sub["q25"].to_numpy(dtype=float),
            sub["q75"].to_numpy(dtype=float),
            alpha=0.12,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    if x_max is not None:
        ax.set_xlim(0, x_max)
    ax.legend(fontsize=8)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def add_local_rate_columns(local_df: pd.DataFrame) -> pd.DataFrame:
    out = local_df.copy()
    needed_numeric = [
        "shortlag_pair_corr",
        "pair_lag_repeats",
        "local_rate_spectral_displacement_per_repeat",
        "local_rate_spectral_displacement_sq_per_repeat",
        "n_previous_segments",
        "repeat_index_in_phrase",
        "elapsed_time_in_phrase_s",
    ]
    out = to_numeric_inplace(out, needed_numeric)
    if "pair_lag_repeats" in out.columns and "shortlag_pair_corr" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["local_corr_drop_per_repeat"] = (1.0 - out["shortlag_pair_corr"]) / out["pair_lag_repeats"]
    else:
        out["local_corr_drop_per_repeat"] = np.nan
    return out


def compute_early_late_corr_change(
    segment_df: pd.DataFrame,
    corr_feature: str = "corr_to_phrase_early_template",
    early_frac: float = 0.33,
    late_frac: float = 0.67,
    min_points: int = 3,
) -> pd.DataFrame:
    """Per-label early-vs-late summary for a cross-correlation feature."""
    required = ["label_norm", "pre_post", "fraction_through_phrase", corr_feature]
    missing = [c for c in required if c not in segment_df.columns]
    if missing:
        raise ValueError(f"Missing columns for early/late correlation summary: {missing}")

    work = segment_df[required].copy()
    work = to_numeric_inplace(work, ["fraction_through_phrase", corr_feature])
    rows = []
    for (label, prepost), sub in work.groupby(["label_norm", "pre_post"], dropna=False):
        early = sub[(sub["fraction_through_phrase"] <= float(early_frac)) & np.isfinite(sub[corr_feature])]
        late = sub[(sub["fraction_through_phrase"] >= float(late_frac)) & np.isfinite(sub[corr_feature])]
        if len(early) < min_points or len(late) < min_points:
            continue
        early_med = float(np.nanmedian(early[corr_feature]))
        late_med = float(np.nanmedian(late[corr_feature]))
        rows.append({
            "label_norm": label,
            "pre_post": prepost,
            "feature": corr_feature,
            "early_frac_max": early_frac,
            "late_frac_min": late_frac,
            "n_early": int(len(early)),
            "n_late": int(len(late)),
            "early_median_corr": early_med,
            "late_median_corr": late_med,
            "late_minus_early_corr": late_med - early_med,
            "early_minus_late_corr": early_med - late_med,
        })
    long = pd.DataFrame(rows)
    if long.empty:
        return long

    pivot_cols = [
        "early_median_corr",
        "late_median_corr",
        "late_minus_early_corr",
        "early_minus_late_corr",
        "n_early",
        "n_late",
    ]
    wide_parts = []
    labels = sorted(long["label_norm"].unique(), key=lambda x: (len(str(x)), str(x)))
    for lab in labels:
        row = {"label_norm": lab}
        sub_lab = long[long["label_norm"] == lab]
        for _, rec in sub_lab.iterrows():
            prepost = str(rec["pre_post"])
            for c in pivot_cols:
                row[f"{prepost}_{c}"] = rec[c]
        row["post_abs_late_minus_early_corr"] = abs(row.get("post_late_minus_early_corr", np.nan))
        row["post_corr_decline_early_minus_late"] = row.get("post_early_minus_late_corr", np.nan)
        row["pre_corr_decline_early_minus_late"] = row.get("pre_early_minus_late_corr", np.nan)
        row["post_minus_pre_late_minus_early_corr"] = (
            row.get("post_late_minus_early_corr", np.nan)
            - row.get("pre_late_minus_early_corr", np.nan)
        )
        row["post_minus_pre_decline"] = (
            row.get("post_corr_decline_early_minus_late", np.nan)
            - row.get("pre_corr_decline_early_minus_late", np.nan)
        )
        wide_parts.append(row)
    return pd.DataFrame(wide_parts)


def compute_local_rate_label_summary(local_df: pd.DataFrame, min_pairs: int = 5) -> pd.DataFrame:
    """Per-label local rate summary."""
    required = ["label_norm", "pre_post", "local_corr_drop_per_repeat", "local_rate_spectral_displacement_per_repeat"]
    missing = [c for c in required if c not in local_df.columns]
    if missing:
        raise ValueError(f"Missing columns for local rate label summary: {missing}")

    rows = []
    for (label, prepost), sub in local_df.groupby(["label_norm", "pre_post"], dropna=False):
        corr_drop = pd.to_numeric(sub["local_corr_drop_per_repeat"], errors="coerce").to_numpy(dtype=float)
        d_rate = pd.to_numeric(sub["local_rate_spectral_displacement_per_repeat"], errors="coerce").to_numpy(dtype=float)
        corr_drop = corr_drop[np.isfinite(corr_drop)]
        d_rate = d_rate[np.isfinite(d_rate)]
        if corr_drop.size < min_pairs and d_rate.size < min_pairs:
            continue
        rows.append({
            "label_norm": label,
            "pre_post": prepost,
            "n_pairs_corr_drop": int(corr_drop.size),
            "median_local_corr_drop_per_repeat": float(np.nanmedian(corr_drop)) if corr_drop.size else np.nan,
            "mean_local_corr_drop_per_repeat": float(np.nanmean(corr_drop)) if corr_drop.size else np.nan,
            "n_pairs_d_rate": int(d_rate.size),
            "median_local_d_per_repeat": float(np.nanmedian(d_rate)) if d_rate.size else np.nan,
            "mean_local_d_per_repeat": float(np.nanmean(d_rate)) if d_rate.size else np.nan,
        })
    long = pd.DataFrame(rows)
    if long.empty:
        return long

    labels = sorted(long["label_norm"].unique(), key=lambda x: (len(str(x)), str(x)))
    wide_rows = []
    for lab in labels:
        row = {"label_norm": lab}
        sub_lab = long[long["label_norm"] == lab]
        for _, rec in sub_lab.iterrows():
            prepost = str(rec["pre_post"])
            for c in [
                "n_pairs_corr_drop",
                "median_local_corr_drop_per_repeat",
                "mean_local_corr_drop_per_repeat",
                "n_pairs_d_rate",
                "median_local_d_per_repeat",
                "mean_local_d_per_repeat",
            ]:
                row[f"{prepost}_{c}"] = rec[c]
        row["post_minus_pre_median_local_corr_drop"] = (
            row.get("post_median_local_corr_drop_per_repeat", np.nan)
            - row.get("pre_median_local_corr_drop_per_repeat", np.nan)
        )
        row["post_minus_pre_median_local_d"] = (
            row.get("post_median_local_d_per_repeat", np.nan)
            - row.get("pre_median_local_d_per_repeat", np.nan)
        )
        wide_rows.append(row)
    return pd.DataFrame(wide_rows)


def add_group_to_summary(summary_df: pd.DataFrame, top30_labels: set[str]) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    out = summary_df.copy()
    out["variance_group"] = np.where(
        out["label_norm"].astype(str).isin(top30_labels),
        "Top 30% phrase-duration variance",
        "Lower 70% phrase-duration variance",
    )
    return out


def _sort_for_plot(df: pd.DataFrame, value_col: str, top_n: int, ascending: bool = False) -> pd.DataFrame:
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work[np.isfinite(work[value_col])].copy()
    work = work.sort_values(value_col, ascending=ascending).head(int(top_n)).copy()
    return work


def plot_ranked_bar(
    df: pd.DataFrame,
    value_col: str,
    out_png: Path,
    title: str,
    x_label: str,
    top_n: int = 12,
    ascending: bool = False,
) -> None:
    work = _sort_for_plot(df, value_col=value_col, top_n=top_n, ascending=ascending)
    if work.empty:
        print(f"[WARN] No ranked data for {out_png.name}; skipping.")
        return
    work = work.iloc[::-1].copy()
    labels = work["label_norm"].astype(str).tolist()
    values = pd.to_numeric(work[value_col], errors="coerce").to_numpy(dtype=float)

    fig_h = max(4.5, 0.38 * len(work) + 1.8)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, fig_h), constrained_layout=True)
    ax.barh(labels, values)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Syllable label")
    ax.grid(True, axis="x", alpha=0.25)
    if "variance_group" in work.columns:
        for y, (_, row) in enumerate(work.iterrows()):
            short = "top30" if row["variance_group"].startswith("Top") else "lower70"
            ax.text(values[y], y, f"  {short}", va="center", fontsize=8, alpha=0.8)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_top_label_curves(
    df: pd.DataFrame,
    labels: Sequence[str],
    y_col: str,
    x_col: str,
    out_png: Path,
    title: str,
    y_label: str,
    x_label: str,
    x_bin_width: float = 1.0,
    max_cols: int = 3,
) -> None:
    labels = [normalize_label_value(x) for x in labels if normalize_label_value(x)]
    labels = list(dict.fromkeys(labels))
    if not labels:
        return
    work = df[df["label_norm"].isin(labels)].copy()
    if work.empty:
        print(f"[WARN] No top-label curve data for {out_png.name}; skipping.")
        return
    work = to_numeric_inplace(work, [x_col, y_col])
    work = work[np.isfinite(work[x_col]) & np.isfinite(work[y_col])].copy()
    if work.empty:
        return
    work["x_bin"] = round_to_bin(work[x_col], x_bin_width)
    agg = (
        work.groupby(["label_norm", "pre_post", "x_bin"], dropna=False)
        .agg(median=(y_col, "median"), n=(y_col, "size"))
        .reset_index()
    )

    n = len(labels)
    ncols = min(max_cols, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, lab in zip(axes, labels):
        sub_lab = agg[agg["label_norm"] == lab]
        for prepost in PRE_POST_ORDER:
            sub = sub_lab[sub_lab["pre_post"] == prepost].sort_values("x_bin")
            if sub.empty:
                continue
            ax.plot(sub["x_bin"], sub["median"], marker="o", linewidth=2, label=prepost)
        ax.set_title(f"Label {lab}")
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(labels):]:
        ax.set_axis_off()

    axes[0].set_ylabel(y_label)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, y=1.08, fontsize=14)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def save_group_counts(segment_df: pd.DataFrame, local_df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for name, df in [("segment_features", segment_df), ("local_shortlag_pairs", local_df)]:
        if df.empty or "variance_group" not in df.columns:
            continue
        for group in GROUP_ORDER:
            sub = df[df["variance_group"] == group]
            rows.append({
                "table": name,
                "variance_group": group,
                "n_rows": int(len(sub)),
                "n_labels": int(sub["label_norm"].nunique()) if "label_norm" in sub.columns else np.nan,
                "n_phrases": int(sub["phrase_id"].nunique()) if "phrase_id" in sub.columns else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "group_counts.csv", index=False)
    print(f"[SAVED] {out_dir / 'group_counts.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate v17 phrase-position local-rate/cross-correlation results by top30/lower70 phrase-duration-variance groups."
    )
    parser.add_argument("--v17-root", required=True, help="Root folder containing combined all-label v17 CSVs for this animal.")
    parser.add_argument("--animal-id", required=True)
    parser.add_argument("--top30-csv", required=True, help="CSV containing top 30% phrase-duration-variance syllables.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--segment-csv", default=None, help="Optional explicit combined segment features CSV.")
    parser.add_argument("--local-pair-csv", default=None, help="Optional explicit combined local short-lag pair CSV.")
    parser.add_argument("--top30-label-col", default=None, help="Optional label column in top30 CSV.")
    parser.add_argument("--top30-animal-col", default=None, help="Optional animal-id column in top30 CSV.")
    parser.add_argument("--corr-feature", default="corr_to_phrase_early_template",
                        help="Cross-correlation feature used for early-vs-late phrase change ranking.")
    parser.add_argument("--early-frac", type=float, default=0.33, help="Early phrase fraction cutoff for early-vs-late summary.")
    parser.add_argument("--late-frac", type=float, default=0.67, help="Late phrase fraction cutoff for early-vs-late summary.")
    parser.add_argument("--min-early-late-points", type=int, default=3)
    parser.add_argument("--min-local-pairs-per-label", type=int, default=5)
    parser.add_argument("--repeat-bin-width", type=float, default=1.0)
    parser.add_argument("--elapsed-bin-width-s", type=float, default=0.25)
    parser.add_argument("--min-labels-per-group-bin", type=int, default=1)
    parser.add_argument("--top-n-labels", type=int, default=12)
    parser.add_argument("--max-repeat-x", type=float, default=None)
    parser.add_argument("--max-elapsed-x-s", type=float, default=None)
    args = parser.parse_args()

    v17_root = Path(args.v17_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.segment_csv is None or args.local_pair_csv is None:
        default_seg, default_local = default_csv_paths(v17_root, args.animal_id)
    else:
        default_seg = default_local = None

    segment_csv = Path(args.segment_csv) if args.segment_csv else default_seg
    local_pair_csv = Path(args.local_pair_csv) if args.local_pair_csv else default_local

    print(f"[INFO] Segment CSV: {segment_csv}")
    print(f"[INFO] Local pair CSV: {local_pair_csv}")
    print(f"[INFO] Top30 CSV: {args.top30_csv}")

    segment_df = pd.read_csv(segment_csv)
    local_df = pd.read_csv(local_pair_csv)

    segment_df = add_normalized_label_column(segment_df, src_col="label", dst_col="label_norm")
    local_df = add_normalized_label_column(local_df, src_col="label", dst_col="label_norm")
    local_df = add_local_rate_columns(local_df)

    top30_labels = load_top30_labels(
        Path(args.top30_csv),
        animal_id=args.animal_id,
        label_col=args.top30_label_col,
        animal_col=args.top30_animal_col,
    )
    present_labels = set(segment_df["label_norm"].unique()) | set(local_df["label_norm"].unique())
    overlap = top30_labels & present_labels
    print(f"[INFO] Top30 labels loaded: {len(top30_labels)}")
    print(f"[INFO] Top30 labels present in v17 outputs: {len(overlap)}")
    if len(overlap) == 0:
        print("[WARN] No top30 labels overlap with v17 output labels. Check top30 CSV label column / animal column.")

    segment_grouped = add_variance_groups(segment_df, top30_labels, label_col="label_norm")
    local_grouped = add_variance_groups(local_df, top30_labels, label_col="label_norm")
    save_group_counts(segment_grouped, local_grouped, out_dir)

    membership = pd.DataFrame({
        "label_norm": sorted(present_labels, key=lambda x: (len(str(x)), str(x)))
    })
    membership["variance_group"] = np.where(
        membership["label_norm"].isin(top30_labels),
        "Top 30% phrase-duration variance",
        "Lower 70% phrase-duration variance",
    )
    membership.to_csv(out_dir / "label_variance_group_membership.csv", index=False)
    print(f"[SAVED] {out_dir / 'label_variance_group_membership.csv'}")

    local_metrics = [
        (
            "local_corr_drop_per_repeat",
            "Local correlation drop per repeat, (1 - corr) / lag",
            "combined_groups_local_corr_drop_vs_previous_repeats",
        ),
        (
            "local_rate_spectral_displacement_per_repeat",
            "Local spectral displacement d per repeat",
            "combined_groups_local_displacement_vs_previous_repeats",
        ),
    ]

    for y_col, y_label, stem in local_metrics:
        if y_col not in local_grouped.columns:
            print(f"[WARN] Missing {y_col}; skipping.")
            continue
        summary = summarize_line_label_balanced(
            local_grouped,
            x_col="n_previous_segments",
            y_col=y_col,
            x_bin_width=args.repeat_bin_width,
            min_rows_per_label_bin=1,
            min_labels_per_group_bin=args.min_labels_per_group_bin,
        )
        summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
        print(f"[SAVED] {out_dir / f'{stem}_summary.csv'}")
        plot_group_panels(
            summary,
            out_png=out_dir / f"{stem}_panels.png",
            title=f"{args.animal_id}: {y_label}",
            y_label=y_label,
            x_label="Number of previous analyzed syllables in phrase",
            x_max=args.max_repeat_x,
        )
        plot_overlay_groups(
            summary,
            out_png=out_dir / f"{stem}_post_overlay.png",
            title=f"{args.animal_id}: post-lesion {y_label}",
            y_label=y_label,
            x_label="Number of previous analyzed syllables in phrase",
            prepost="post",
            x_max=args.max_repeat_x,
        )

    if args.corr_feature in segment_grouped.columns:
        corr_summary = summarize_line_label_balanced(
            segment_grouped,
            x_col="n_previous_segments",
            y_col=args.corr_feature,
            x_bin_width=args.repeat_bin_width,
            min_rows_per_label_bin=1,
            min_labels_per_group_bin=args.min_labels_per_group_bin,
        )
        corr_summary.to_csv(out_dir / "combined_groups_template_corr_vs_previous_repeats_summary.csv", index=False)
        print(f"[SAVED] {out_dir / 'combined_groups_template_corr_vs_previous_repeats_summary.csv'}")
        plot_group_panels(
            corr_summary,
            out_png=out_dir / "combined_groups_template_corr_vs_previous_repeats_panels.png",
            title=f"{args.animal_id}: spectrogram correlation to early-phrase template",
            y_label="Spectrogram corr. to early-phrase template",
            x_label="Number of previous analyzed syllables in phrase",
            x_max=args.max_repeat_x,
            y_lim=(-0.05, 1.05),
        )
        plot_overlay_groups(
            corr_summary,
            out_png=out_dir / "combined_groups_template_corr_vs_previous_repeats_post_overlay.png",
            title=f"{args.animal_id}: post-lesion spectrogram correlation to early-phrase template",
            y_label="Spectrogram corr. to early-phrase template",
            x_label="Number of previous analyzed syllables in phrase",
            prepost="post",
            x_max=args.max_repeat_x,
        )
    else:
        print(f"[WARN] Missing corr feature {args.corr_feature}; skipping template-corr group plots.")

    if "elapsed_time_in_phrase_s" in local_grouped.columns:
        for y_col, y_label, stem in local_metrics:
            if y_col not in local_grouped.columns:
                continue
            elapsed_summary = summarize_line_label_balanced(
                local_grouped,
                x_col="elapsed_time_in_phrase_s",
                y_col=y_col,
                x_bin_width=args.elapsed_bin_width_s,
                min_rows_per_label_bin=1,
                min_labels_per_group_bin=args.min_labels_per_group_bin,
            )
            elapsed_summary.to_csv(out_dir / f"{stem.replace('previous_repeats', 'elapsed_time')}_summary.csv", index=False)
            plot_group_panels(
                elapsed_summary,
                out_png=out_dir / f"{stem.replace('previous_repeats', 'elapsed_time')}_panels.png",
                title=f"{args.animal_id}: {y_label} vs elapsed phrase time",
                y_label=y_label,
                x_label="Elapsed time in phrase (s)",
                x_max=args.max_elapsed_x_s,
            )

    corr_change = compute_early_late_corr_change(
        segment_df,
        corr_feature=args.corr_feature,
        early_frac=args.early_frac,
        late_frac=args.late_frac,
        min_points=args.min_early_late_points,
    )
    corr_change = add_group_to_summary(corr_change, top30_labels)
    corr_change.to_csv(out_dir / "rank_syllables_early_late_corr_change.csv", index=False)
    print(f"[SAVED] {out_dir / 'rank_syllables_early_late_corr_change.csv'}")

    plot_ranked_bar(
        corr_change,
        value_col="post_abs_late_minus_early_corr",
        out_png=out_dir / "rank_syllables_greatest_abs_post_early_late_corr_change.png",
        title=f"{args.animal_id}: syllables with greatest post early-vs-late corr. change",
        x_label="|late median corr - early median corr|, post",
        top_n=args.top_n_labels,
        ascending=False,
    )
    plot_ranked_bar(
        corr_change,
        value_col="post_corr_decline_early_minus_late",
        out_png=out_dir / "rank_syllables_greatest_post_corr_decline_late_in_phrase.png",
        title=f"{args.animal_id}: syllables with strongest late-phrase correlation decline",
        x_label="Early median corr - late median corr, post",
        top_n=args.top_n_labels,
        ascending=False,
    )
    plot_ranked_bar(
        corr_change,
        value_col="post_minus_pre_decline",
        out_png=out_dir / "rank_syllables_greatest_post_minus_pre_corr_decline.png",
        title=f"{args.animal_id}: syllables with strongest post-specific corr. decline",
        x_label="Post decline - pre decline",
        top_n=args.top_n_labels,
        ascending=False,
    )

    top_corr_labels = (
        _sort_for_plot(corr_change, "post_abs_late_minus_early_corr", args.top_n_labels, ascending=False)["label_norm"]
        .astype(str).tolist()
        if not corr_change.empty else []
    )
    plot_top_label_curves(
        segment_df,
        labels=top_corr_labels[: min(args.top_n_labels, 12)],
        y_col=args.corr_feature,
        x_col="n_previous_segments",
        out_png=out_dir / "top_corr_change_syllables_corr_vs_previous_repeats.png",
        title=f"{args.animal_id}: top early-vs-late corr-change syllables",
        y_label="Spectrogram corr. to early-phrase template",
        x_label="Number of previous analyzed syllables in phrase",
        x_bin_width=args.repeat_bin_width,
    )

    local_summary = compute_local_rate_label_summary(local_df, min_pairs=args.min_local_pairs_per_label)
    local_summary = add_group_to_summary(local_summary, top30_labels)
    local_summary.to_csv(out_dir / "rank_syllables_high_local_rate.csv", index=False)
    print(f"[SAVED] {out_dir / 'rank_syllables_high_local_rate.csv'}")

    plot_ranked_bar(
        local_summary,
        value_col="post_median_local_corr_drop_per_repeat",
        out_png=out_dir / "rank_syllables_highest_post_local_corr_drop_rate.png",
        title=f"{args.animal_id}: syllables with highest post local corr-drop rate",
        x_label="Median local corr-drop per repeat, post",
        top_n=args.top_n_labels,
        ascending=False,
    )
    plot_ranked_bar(
        local_summary,
        value_col="post_median_local_d_per_repeat",
        out_png=out_dir / "rank_syllables_highest_post_local_d_rate.png",
        title=f"{args.animal_id}: syllables with highest post local spectral displacement rate",
        x_label="Median local spectral displacement d per repeat, post",
        top_n=args.top_n_labels,
        ascending=False,
    )
    plot_ranked_bar(
        local_summary,
        value_col="post_minus_pre_median_local_corr_drop",
        out_png=out_dir / "rank_syllables_greatest_post_minus_pre_local_corr_drop_rate.png",
        title=f"{args.animal_id}: syllables with greatest post-minus-pre local corr-drop rate",
        x_label="Post - pre median local corr-drop per repeat",
        top_n=args.top_n_labels,
        ascending=False,
    )

    top_local_labels = (
        _sort_for_plot(local_summary, "post_median_local_corr_drop_per_repeat", args.top_n_labels, ascending=False)["label_norm"]
        .astype(str).tolist()
        if not local_summary.empty else []
    )
    plot_top_label_curves(
        local_df,
        labels=top_local_labels[: min(args.top_n_labels, 12)],
        y_col="local_corr_drop_per_repeat",
        x_col="n_previous_segments",
        out_png=out_dir / "top_high_local_rate_syllables_corr_drop_vs_previous_repeats.png",
        title=f"{args.animal_id}: top high-local-rate syllables",
        y_label="Local corr-drop per repeat",
        x_label="Number of previous analyzed syllables in phrase",
        x_bin_width=args.repeat_bin_width,
    )

    readme = out_dir / "README_aggregate_phrase_position_groups.txt"
    readme.write_text(
        "Aggregate phrase-position local-rate/cross-correlation group plots.\n\n"
        "Key outputs:\n"
        "- combined_groups_*_panels.png: three-panel comparison of top 30%, lower 70%, and all syllables.\n"
        "- combined_groups_*_post_overlay.png: one-axis overlay of top 30%, lower 70%, and all syllables for post-lesion data.\n"
        "- rank_syllables_early_late_corr_change.csv: per-syllable early-vs-late correlation summaries.\n"
        "- rank_syllables_high_local_rate.csv: per-syllable local-rate summaries.\n"
        "- top_corr_change_syllables_corr_vs_previous_repeats.png: curves for labels with greatest early-vs-late correlation changes.\n"
        "- top_high_local_rate_syllables_corr_drop_vs_previous_repeats.png: curves for labels with high local corr-drop rates.\n\n"
        "Interpretation notes:\n"
        "- local_corr_drop_per_repeat = (1 - shortlag_pair_corr) / pair_lag_repeats. Higher values mean neighboring syllables are less similar.\n"
        "- local spectral displacement d = sqrt(2 * (1 - corr)) for unit-normalized spectrogram vectors.\n"
        "- Early/late phrase summaries use fraction_through_phrase by default: early <= early_frac and late >= late_frac.\n"
        "- Group curves are label-balanced by default: median per label/bin first, then median across labels.\n"
    )
    print(f"[SAVED] {readme}")
    print("[DONE]")


if __name__ == "__main__":
    main()
