#!/usr/bin/env python3
"""
Visualize Supplemental Table 2 animal metadata, sampling coverage, SD results,
and (optionally) coefficient of variation (CV) results.

This version pools partial and complete medial+lateral lesions into one M+L group.

PNG outputs always produced for SD:
----------------------------------
1. table2_recording_days_dumbbell.png
2. table2_selected_syllables_by_bird.png
3. table2_balanced_phrase_counts_by_bird.png
4. table2_pre_vs_post_sd_dumbbell.png
5. table2_sd_delta_by_group_pooled_ML.png
6. table2_sampling_coverage_composite.png
7. table2_bootstrap_ci_by_group_pooled_ML_SD.png

Optional CV outputs (if CV data are available):
-----------------------------------------------
8.  table2_pre_vs_post_cv_dumbbell.png
9.  table2_cv_delta_by_group_pooled_ML.png
10. table2_bootstrap_ci_by_group_pooled_ML_CV.png
11. table2_cv_group_summary_pooled_ML.csv

How CV is obtained
------------------
The script tries, in order:
1. Direct CV columns already present in Supplemental Table 2.
2. Mean-duration columns in Supplemental Table 2, using CV = SD / mean.
3. A separate --cv-input CSV merged by animal_id.

For --cv-input, the script looks for animal_id plus one of the following
column pairs:
- pre_cv / post_cv
- late_pre_cv / post_cv
- late_pre_cv_value / post_cv_value
- cv_pre / cv_post
- cv_late_pre / cv_post

If none of those are found, the script will still run all SD figures and
will simply skip the CV figures with a clear message.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/Volumes/my_own_SSD/updated_AreaX_outputs/"
    "Figure3_robust_stats_tables_v2/"
    "Supplemental_Table_2_animal_metadata_and_sample_sizes.csv"
)
DEFAULT_PRIMARY_METHOD = "Pooled pre+post-selected top 30%"
POOLED_ML_NAME = "Complete and partial medial and lateral lesion"

ORIGINAL_GROUP_ORDER = [
    "sham saline injection",
    "Lateral lesion only",
    "Partial Medial and Lateral lesion",
    "Complete Medial and Lateral lesion",
]

PLOTTED_GROUP_ORDER = [
    "sham saline injection",
    "Lateral lesion only",
    POOLED_ML_NAME,
]

GROUP_LABELS = {
    "sham saline injection": "Sham",
    "Lateral lesion only": "Lateral only",
    POOLED_ML_NAME: "Pooled M+L",
    "Partial Medial and Lateral lesion": "Partial M+L",
    "Complete Medial and Lateral lesion": "Complete M+L",
}

GROUP_COLORS = {
    "sham saline injection": "#2A9D8F",
    "Lateral lesion only": "#457B9D",
    POOLED_ML_NAME: "#7A68C7",
    "Partial Medial and Lateral lesion": "#8D5FD3",
    "Complete Medial and Lateral lesion": "#6C757D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG visualizations from Supplemental Table 2, pooling "
            "partial and complete medial+lateral lesions, with optional CV plots."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Supplemental Table 2 CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--bootstrap-summary",
        type=Path,
        default=None,
        help=(
            "Optional Supplemental Table 1 group-summary CSV containing "
            "precomputed SD bootstrap confidence intervals. If omitted, the script "
            "looks beside the input CSV."
        ),
    )
    parser.add_argument(
        "--cv-input",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing animal-level CV values to merge by animal_id. "
            "Use this if Supplemental Table 2 does not already contain CV or mean columns."
        ),
    )
    parser.add_argument(
        "--primary-method",
        default=DEFAULT_PRIMARY_METHOD,
        help=(
            "Selection-method row to use from the bootstrap summary. "
            + f"Default: {DEFAULT_PRIMARY_METHOD!r}".replace("%", "%%")
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: a 'table2_visualizations_pooled_ML_with_CV' "
            "folder beside the input CSV."
        ),
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=5000,
        help=(
            "Bootstrap iterations used only if the precomputed SD summary is "
            "unavailable, and always used for CV summaries. Default: 5000"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for jitter and bootstrap. Default: 123",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


def choose_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        "None of the expected columns were present:\n"
        + "\n".join(f"  - {column}" for column in candidates)
    )


def pooled_group_name(group: str) -> str:
    if group in {
        "Partial Medial and Lateral lesion",
        "Complete Medial and Lateral lesion",
    }:
        return POOLED_ML_NAME
    return group


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int, show: bool) -> None:
    fig.tight_layout()
    output_path = output_dir / f"{stem}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def group_legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=GROUP_COLORS[group],
            edgecolor="none",
            label=GROUP_LABELS[group],
        )
        for group in PLOTTED_GROUP_ORDER
    ]


def add_group_boundaries(ax: plt.Axes, df: pd.DataFrame, group_col: str = "display_group_pooled") -> None:
    previous_group = None
    for index, group in enumerate(df[group_col]):
        if previous_group is not None and group != previous_group:
            ax.axhline(index - 0.5, color="0.85", linewidth=0.8, zorder=0)
        previous_group = group


def percentile_bootstrap_ci(values: np.ndarray, *, reps: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, values.size), replace=True)
    bootstrap_medians = np.nanmedian(samples, axis=1)
    low, high = np.nanpercentile(bootstrap_medians, [2.5, 97.5])
    return float(np.nanmedian(values)), float(low), float(high)


def load_and_prepare(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found:\n{input_csv}")

    df = pd.read_csv(input_csv)

    required = [
        "animal_id",
        "display_group",
        "treatment_date",
        "n_selected_syllables",
        "n_pre_days",
        "n_post_days",
        "pre_sd_s",
        "post_sd_s",
        "sd_delta_s",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The input CSV is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    numeric_columns = [
        "n_selected_syllables",
        "n_pre_days",
        "n_post_days",
        "pre_sd_s",
        "post_sd_s",
        "sd_delta_s",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["animal_id"] = df["animal_id"].astype(str)
    df["display_group"] = df["display_group"].astype(str)
    df["treatment_date"] = pd.to_datetime(df["treatment_date"], errors="coerce")

    unknown_groups = sorted(set(df["display_group"]) - set(ORIGINAL_GROUP_ORDER))
    if unknown_groups:
        raise ValueError(
            "Unexpected display_group values were found:\n"
            + "\n".join(f"  - {group}" for group in unknown_groups)
        )

    pre_phrase_column = choose_first_existing(
        df, ["n_late_pre_phrases", "total_late_pre_phrases_selected"]
    )
    post_phrase_column = choose_first_existing(
        df, ["n_post_phrases", "total_post_phrases_selected"]
    )

    df[pre_phrase_column] = pd.to_numeric(df[pre_phrase_column], errors="coerce")
    df[post_phrase_column] = pd.to_numeric(df[post_phrase_column], errors="coerce")
    df["balanced_phrases_per_epoch"] = df[[pre_phrase_column, post_phrase_column]].median(axis=1, skipna=True)

    df["display_group_pooled"] = df["display_group"].map(pooled_group_name)
    rank = {group: i for i, group in enumerate(PLOTTED_GROUP_ORDER)}
    df["_group_rank"] = df["display_group_pooled"].map(rank)
    df = df.sort_values(
        ["_group_rank", "display_group", "treatment_date", "animal_id"],
        na_position="last",
    ).reset_index(drop=True)

    df["group_short"] = df["display_group_pooled"].map(GROUP_LABELS)
    df["plot_color"] = df["display_group_pooled"].map(GROUP_COLORS)
    df["bird_label"] = df["animal_id"] + "   (" + df["group_short"] + ")"

    return df


def infer_bootstrap_summary_path(input_csv: Path, requested_path: Path | None) -> Path | None:
    if requested_path is not None:
        requested_path = requested_path.expanduser().resolve()
        if not requested_path.exists():
            raise FileNotFoundError(
                f"Bootstrap group-summary CSV not found:\n{requested_path}"
            )
        return requested_path

    candidate = input_csv.parent / "Supplemental_Table_1_selection_robustness_group_summary.csv"
    if candidate.exists():
        return candidate
    return None


def load_or_calculate_sd_bootstrap_stats(
    df: pd.DataFrame,
    summary_path: Path | None,
    *,
    primary_method: str,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    required_summary_columns = [
        "selection_method",
        "display_group",
        "n_birds",
        "median_delta_sd_s",
        "delta_sd_CI95_low_s",
        "delta_sd_CI95_high_s",
    ]

    if summary_path is not None:
        summary = pd.read_csv(summary_path)
        missing = [c for c in required_summary_columns if c not in summary.columns]
        if missing:
            raise ValueError(
                "The SD bootstrap group-summary CSV is missing columns:\n"
                + "\n".join(f"  - {column}" for column in missing)
            )

        selected = summary[
            summary["selection_method"].astype(str).eq(primary_method)
            & summary["display_group"].astype(str).isin(PLOTTED_GROUP_ORDER)
        ].copy()

        if selected.empty:
            methods = sorted(summary["selection_method"].dropna().astype(str).unique())
            raise ValueError(
                f"No rows matched --primary-method {primary_method!r}.\n"
                "Available selection methods include:\n"
                + "\n".join(f"  - {method}" for method in methods)
            )

        selected["display_group"] = selected["display_group"].astype(str)
        selected = selected.drop_duplicates("display_group")
        selected = selected.set_index("display_group").loc[PLOTTED_GROUP_ORDER].reset_index()
        selected = selected[
            ["display_group", "n_birds", "median_delta_sd_s", "delta_sd_CI95_low_s", "delta_sd_CI95_high_s"]
        ].copy()

        source_note = (
            f"Precomputed animal-level bootstrap CIs from {summary_path.name}; "
            f"selection method: {primary_method}; partial and complete M+L pooled"
        )
        return selected, source_note

    rows = []
    for group_index, group in enumerate(PLOTTED_GROUP_ORDER):
        values = df.loc[df["display_group_pooled"] == group, "sd_delta_s"].to_numpy(dtype=float)
        median, low, high = percentile_bootstrap_ci(values, reps=bootstrap_reps, seed=seed + group_index)
        rows.append(
            {
                "display_group": group,
                "n_birds": int(np.isfinite(values).sum()),
                "median_delta_sd_s": median,
                "delta_sd_CI95_low_s": low,
                "delta_sd_CI95_high_s": high,
            }
        )

    fallback = pd.DataFrame(rows)
    source_note = (
        f"Recalculated animal-level percentile bootstrap CIs "
        f"({bootstrap_reps:,} resamples; median statistic); partial and complete M+L pooled"
    )
    return fallback, source_note


# ----------------------------- CV handling --------------------------------- #

def _normalize_cv_table(cv_df: pd.DataFrame) -> pd.DataFrame:
    cv_df = cv_df.copy()
    cv_df.columns = [str(c) for c in cv_df.columns]

    animal_candidates = ["animal_id", "bird", "bird_id", "Animal", "animal"]
    animal_col = None
    for c in animal_candidates:
        if c in cv_df.columns:
            animal_col = c
            break
    if animal_col is None:
        raise ValueError(
            "CV input CSV is missing an animal identifier column. Expected one of:\n"
            + "\n".join(f"  - {c}" for c in animal_candidates)
        )

    candidate_pairs = [
        ("pre_cv", "post_cv"),
        ("late_pre_cv", "post_cv"),
        ("late_pre_cv_value", "post_cv_value"),
        ("cv_pre", "cv_post"),
        ("cv_late_pre", "cv_post"),
    ]

    pre_col = post_col = None
    for p, q in candidate_pairs:
        if p in cv_df.columns and q in cv_df.columns:
            pre_col, post_col = p, q
            break

    if pre_col is None:
        raise ValueError(
            "CV input CSV does not contain a recognized pre/post CV column pair.\n"
            "Supported pairs include:\n"
            + "\n".join(f"  - {p} / {q}" for p, q in candidate_pairs)
        )

    out = pd.DataFrame({
        "animal_id": cv_df[animal_col].astype(str),
        "pre_cv": pd.to_numeric(cv_df[pre_col], errors="coerce"),
        "post_cv": pd.to_numeric(cv_df[post_col], errors="coerce"),
    })
    out["cv_delta"] = out["post_cv"] - out["pre_cv"]
    return out


def maybe_get_cv_table(df: pd.DataFrame, cv_input: Path | None) -> tuple[pd.DataFrame | None, str]:
    """
    Return a normalized CV table with columns:
    animal_id, pre_cv, post_cv, cv_delta
    """
    direct_pairs = [
        ("pre_cv", "post_cv"),
        ("late_pre_cv", "post_cv"),
        ("cv_pre", "cv_post"),
    ]
    for pre_col, post_col in direct_pairs:
        if pre_col in df.columns and post_col in df.columns:
            cv_df = pd.DataFrame({
                "animal_id": df["animal_id"].astype(str),
                "pre_cv": pd.to_numeric(df[pre_col], errors="coerce"),
                "post_cv": pd.to_numeric(df[post_col], errors="coerce"),
            })
            cv_df["cv_delta"] = cv_df["post_cv"] - cv_df["pre_cv"]
            return cv_df, f"CV source: direct columns in Supplemental Table 2 ({pre_col}, {post_col})"

    mean_pairs = [
        ("pre_mean_s", "post_mean_s"),
        ("late_pre_mean_s", "post_mean_s"),
        ("mean_pre_s", "mean_post_s"),
    ]
    for pre_mean_col, post_mean_col in mean_pairs:
        if pre_mean_col in df.columns and post_mean_col in df.columns:
            cv_df = pd.DataFrame({
                "animal_id": df["animal_id"].astype(str),
                "pre_cv": pd.to_numeric(df["pre_sd_s"], errors="coerce")
                          / pd.to_numeric(df[pre_mean_col], errors="coerce"),
                "post_cv": pd.to_numeric(df["post_sd_s"], errors="coerce")
                           / pd.to_numeric(df[post_mean_col], errors="coerce"),
            })
            cv_df["cv_delta"] = cv_df["post_cv"] - cv_df["pre_cv"]
            return cv_df, (
                f"CV source: calculated within this script as SD / mean using "
                f"{pre_mean_col} and {post_mean_col}"
            )

    if cv_input is not None:
        cv_input = cv_input.expanduser().resolve()
        if not cv_input.exists():
            raise FileNotFoundError(f"CV input CSV not found:\n{cv_input}")
        raw_cv = pd.read_csv(cv_input)
        cv_df = _normalize_cv_table(raw_cv)
        return cv_df, f"CV source: merged from {cv_input.name}"

    return None, (
        "CV not available: Supplemental Table 2 does not contain direct CV columns "
        "or mean-duration columns, and no --cv-input CSV was provided."
    )


def prepare_cv_for_plotting(main_df: pd.DataFrame, cv_input: Path | None) -> tuple[pd.DataFrame | None, str]:
    cv_df, source_note = maybe_get_cv_table(main_df, cv_input)
    if cv_df is None:
        return None, source_note

    merged = main_df[["animal_id", "display_group", "display_group_pooled", "bird_label", "treatment_date"]].merge(
        cv_df, on="animal_id", how="left"
    )
    merged = merged.sort_values(
        ["display_group_pooled", "display_group", "treatment_date", "animal_id"]
    ).reset_index(drop=True)
    return merged, source_note


def calculate_cv_group_summary(cv_plot_df: pd.DataFrame, *, bootstrap_reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, group in enumerate(PLOTTED_GROUP_ORDER):
        sub = cv_plot_df.loc[cv_plot_df["display_group_pooled"] == group].copy()
        pre_vals = sub["pre_cv"].to_numpy(dtype=float)
        post_vals = sub["post_cv"].to_numpy(dtype=float)
        delta_vals = sub["cv_delta"].to_numpy(dtype=float)

        median_delta, ci_low, ci_high = percentile_bootstrap_ci(
            delta_vals, reps=bootstrap_reps, seed=seed + i
        )
        rows.append({
            "display_group": group,
            "n_birds": int(np.isfinite(delta_vals).sum()),
            "median_pre_cv": float(np.nanmedian(pre_vals)),
            "median_post_cv": float(np.nanmedian(post_vals)),
            "median_delta_cv": median_delta,
            "mean_delta_cv": float(np.nanmean(delta_vals)),
            "delta_cv_CI95_low": ci_low,
            "delta_cv_CI95_high": ci_high,
            "resampling_unit": "animal",
            "summary_statistic": "median delta CV",
            "bootstrap_reps": int(bootstrap_reps),
        })
    return pd.DataFrame(rows)


# ----------------------------- SD plots ------------------------------------ #

def plot_recording_days(df: pd.DataFrame, output_dir: Path, dpi: int, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        color = row["plot_color"]
        ax.plot([row["n_pre_days"], row["n_post_days"]], [i, i], color=color, linewidth=1.5, alpha=0.85, zorder=1)
        ax.scatter(row["n_pre_days"], i, marker="o", s=52, color=color, zorder=2)
        ax.scatter(row["n_post_days"], i, marker="s", s=52, color=color, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(df["bird_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of recording days")
    ax.set_ylabel("Bird")
    ax.set_title("Recording coverage by bird")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, df)
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", label="Pre days", markersize=7),
        Line2D([0], [0], marker="s", linestyle="None", color="black", label="Post days", markersize=7),
    ]
    ax.legend(handles=group_legend_handles() + marker_handles, frameon=False, loc="lower right")
    save_figure(fig, output_dir, "table2_recording_days_dumbbell", dpi, show)


def plot_selected_syllables(df: pd.DataFrame, output_dir: Path, dpi: int, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))
    ax.barh(y, df["n_selected_syllables"], color=df["plot_color"], alpha=0.9)
    x_range = df["n_selected_syllables"].max() - df["n_selected_syllables"].min()
    label_offset = max(0.12, x_range * 0.02)
    for i, row in df.iterrows():
        ax.text(row["n_selected_syllables"] + label_offset, i, f"{int(row['n_selected_syllables'])}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["bird_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of selected syllables")
    ax.set_ylabel("Bird")
    ax.set_title("Selected syllables per bird")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, df)
    ax.legend(handles=group_legend_handles(), frameon=False, loc="lower right")
    save_figure(fig, output_dir, "table2_selected_syllables_by_bird", dpi, show)


def plot_balanced_phrase_counts(df: pd.DataFrame, output_dir: Path, dpi: int, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        value = row["balanced_phrases_per_epoch"]
        ax.scatter(value, i, s=62, color=row["plot_color"], zorder=2)
        ax.annotate(f"{int(value):,}", xy=(value, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["bird_label"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Balanced phrase observations per epoch (log scale)")
    ax.set_ylabel("Bird")
    ax.set_title("Balanced phrase count per epoch")
    ax.grid(axis="x", which="both", alpha=0.25)
    add_group_boundaries(ax, df)
    ax.legend(handles=group_legend_handles(), frameon=False, loc="lower right")
    save_figure(fig, output_dir, "table2_balanced_phrase_counts_by_bird", dpi, show)


def plot_pre_post_sd(df: pd.DataFrame, output_dir: Path, dpi: int, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))
    sd_values = np.concatenate([df["pre_sd_s"].dropna().to_numpy(), df["post_sd_s"].dropna().to_numpy()])
    label_offset = max(0.015, np.ptp(sd_values) * 0.02)
    for i, row in df.iterrows():
        color = row["plot_color"]
        ax.plot([row["pre_sd_s"], row["post_sd_s"]], [i, i], color=color, linewidth=1.5, alpha=0.85, zorder=1)
        ax.scatter(row["pre_sd_s"], i, marker="o", s=52, color=color, zorder=2)
        ax.scatter(row["post_sd_s"], i, marker="s", s=52, color=color, zorder=2)
        ax.text(max(row["pre_sd_s"], row["post_sd_s"]) + label_offset, i, f"Δ={row['sd_delta_s']:+.3f} s", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["bird_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Phrase duration SD (s)")
    ax.set_ylabel("Bird")
    ax.set_title("Late-pre versus post phrase-duration variability")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, df)
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", label="Late pre", markersize=7),
        Line2D([0], [0], marker="s", linestyle="None", color="black", label="Post", markersize=7),
    ]
    ax.legend(handles=group_legend_handles() + marker_handles, frameon=False, loc="lower right")
    save_figure(fig, output_dir, "table2_pre_vs_post_sd_dumbbell", dpi, show)


def plot_sd_delta_by_group(df: pd.DataFrame, output_dir: Path, dpi: int, show: bool, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6))
    rng = np.random.default_rng(seed)
    for group_index, group in enumerate(PLOTTED_GROUP_ORDER):
        values = df.loc[df["display_group_pooled"] == group, "sd_delta_s"].dropna().to_numpy()
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        x = np.full(len(values), group_index, dtype=float) + jitter
        ax.scatter(x, values, s=62, color=GROUP_COLORS[group], alpha=0.9, zorder=2)
        median = np.median(values); q1, q3 = np.percentile(values, [25, 75])
        ax.vlines(group_index, q1, q3, color="black", linewidth=2, zorder=3)
        ax.plot([group_index - 0.18, group_index + 0.18], [median, median], color="black", linewidth=2, zorder=3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(PLOTTED_GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in PLOTTED_GROUP_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Change in phrase duration SD\n(post − late pre, s)")
    ax.set_xlabel("Lesion group")
    ax.set_title("Bird-level change in phrase-duration SD")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "table2_sd_delta_by_group_pooled_ML", dpi, show)


def plot_bootstrap_ci_by_group_sd(
    df: pd.DataFrame,
    bootstrap_stats: pd.DataFrame,
    source_note: str,
    output_dir: Path,
    dpi: int,
    show: bool,
    seed: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    rng = np.random.default_rng(seed)
    stats_by_group = bootstrap_stats.set_index("display_group")
    for group_index, group in enumerate(PLOTTED_GROUP_ORDER):
        values = df.loc[df["display_group_pooled"] == group, "sd_delta_s"].dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        x_values = np.full(len(values), group_index, dtype=float) + jitter
        ax.scatter(x_values, values, s=48, facecolor=GROUP_COLORS[group], edgecolor="white", linewidth=0.6, alpha=0.82, zorder=2)
        row = stats_by_group.loc[group]
        estimate = float(row["median_delta_sd_s"])
        ci_low = float(row["delta_sd_CI95_low_s"])
        ci_high = float(row["delta_sd_CI95_high_s"])
        ax.errorbar(
            group_index, estimate,
            yerr=np.array([[estimate - ci_low], [ci_high - estimate]]),
            fmt="D", markersize=7, markerfacecolor="black", markeredgecolor="black",
            ecolor="black", elinewidth=2, capsize=5, capthick=1.8, zorder=4,
        )
        ax.annotate(f"n={int(row['n_birds'])}", xy=(group_index, ci_high), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="0.25", linestyle="--", linewidth=1.1, zorder=0)
    ax.set_xticks(range(len(PLOTTED_GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in PLOTTED_GROUP_ORDER], rotation=18, ha="right")
    ax.set_xlabel("Lesion group")
    ax.set_ylabel("Change in phrase duration SD\n(post − late pre, s)")
    ax.set_title("Phrase-duration SD change by lesion group\nMedian with animal-level bootstrap 95% CI")
    ax.grid(axis="y", alpha=0.22)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="0.55", markeredgecolor="white", label="Individual bird", markersize=7),
        Line2D([0], [0], marker="D", linestyle="None", color="black", label="Group median", markersize=7),
        Line2D([0], [0], color="black", marker="|", linestyle="-", label="95% bootstrap CI", markersize=10),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")
    fig.text(0.01, 0.005, source_note, ha="left", va="bottom", fontsize=7.5, color="0.35")
    save_figure(fig, output_dir, "table2_bootstrap_ci_by_group_pooled_ML_SD", dpi, show)


def plot_sampling_coverage_composite(df: pd.DataFrame, output_dir: Path, dpi: int, show: bool) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 0.9, 1.2]})
    y = np.arange(len(df))
    ax = axes[0]
    for i, row in df.iterrows():
        color = row["plot_color"]
        ax.plot([row["n_pre_days"], row["n_post_days"]], [i, i], color=color, linewidth=1.4, alpha=0.85)
        ax.scatter(row["n_pre_days"], i, marker="o", s=42, color=color)
        ax.scatter(row["n_post_days"], i, marker="s", s=42, color=color)
    ax.set_yticks(y); ax.set_yticklabels(df["animal_id"]); ax.invert_yaxis()
    ax.set_xlabel("Recording days"); ax.set_ylabel("Bird")
    ax.set_title("A  Recording coverage", loc="left"); ax.grid(axis="x", alpha=0.25); add_group_boundaries(ax, df)

    ax = axes[1]
    ax.barh(y, df["n_selected_syllables"], color=df["plot_color"], alpha=0.9)
    ax.set_xlabel("Selected syllables"); ax.set_title("B  Syllable inclusion", loc="left")
    ax.grid(axis="x", alpha=0.25); add_group_boundaries(ax, df)

    ax = axes[2]
    for i, row in df.iterrows():
        ax.scatter(row["balanced_phrases_per_epoch"], i, s=52, color=row["plot_color"])
    ax.set_xscale("log"); ax.set_xlabel("Balanced phrases per epoch\n(log scale)")
    ax.set_title("C  Phrase observations", loc="left"); ax.grid(axis="x", which="both", alpha=0.25); add_group_boundaries(ax, df)

    legend_handles = group_legend_handles() + [
        Line2D([0], [0], marker="o", linestyle="None", color="black", label="Pre days", markersize=7),
        Line2D([0], [0], marker="s", linestyle="None", color="black", label="Post days", markersize=7),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.subplots_adjust(wspace=0.12)
    save_figure(fig, output_dir, "table2_sampling_coverage_composite", dpi, show)


# ----------------------------- CV plots ------------------------------------ #

def plot_pre_post_cv(cv_df: pd.DataFrame, output_dir: Path, dpi: int, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(cv_df))
    vals = np.concatenate([cv_df["pre_cv"].dropna().to_numpy(), cv_df["post_cv"].dropna().to_numpy()])
    label_offset = max(0.003, np.ptp(vals) * 0.02 if len(vals) > 1 else 0.003)

    for i, row in cv_df.iterrows():
        color = GROUP_COLORS[row["display_group_pooled"]]
        ax.plot([row["pre_cv"], row["post_cv"]], [i, i], color=color, linewidth=1.5, alpha=0.85, zorder=1)
        ax.scatter(row["pre_cv"], i, marker="o", s=52, color=color, zorder=2)
        ax.scatter(row["post_cv"], i, marker="s", s=52, color=color, zorder=2)
        ax.text(max(row["pre_cv"], row["post_cv"]) + label_offset, i, f"Δ={row['cv_delta']:+.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(cv_df["bird_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient of variation (CV)")
    ax.set_ylabel("Bird")
    ax.set_title("Late-pre versus post coefficient of variation")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, cv_df, group_col="display_group_pooled")
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", label="Late pre", markersize=7),
        Line2D([0], [0], marker="s", linestyle="None", color="black", label="Post", markersize=7),
    ]
    ax.legend(handles=group_legend_handles() + marker_handles, frameon=False, loc="lower right")
    save_figure(fig, output_dir, "table2_pre_vs_post_cv_dumbbell", dpi, show)


def plot_cv_delta_by_group(cv_df: pd.DataFrame, output_dir: Path, dpi: int, show: bool, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6))
    rng = np.random.default_rng(seed + 1000)

    for group_index, group in enumerate(PLOTTED_GROUP_ORDER):
        values = cv_df.loc[cv_df["display_group_pooled"] == group, "cv_delta"].dropna().to_numpy()
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        x = np.full(len(values), group_index, dtype=float) + jitter
        ax.scatter(x, values, s=62, color=GROUP_COLORS[group], alpha=0.9, zorder=2)
        median = np.median(values); q1, q3 = np.percentile(values, [25, 75])
        ax.vlines(group_index, q1, q3, color="black", linewidth=2, zorder=3)
        ax.plot([group_index - 0.18, group_index + 0.18], [median, median], color="black", linewidth=2, zorder=3)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(PLOTTED_GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in PLOTTED_GROUP_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Change in coefficient of variation\n(post − late pre)")
    ax.set_xlabel("Lesion group")
    ax.set_title("Bird-level change in coefficient of variation")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "table2_cv_delta_by_group_pooled_ML", dpi, show)


def plot_bootstrap_ci_by_group_cv(
    cv_df: pd.DataFrame,
    cv_summary: pd.DataFrame,
    source_note: str,
    output_dir: Path,
    dpi: int,
    show: bool,
    seed: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    rng = np.random.default_rng(seed + 2000)
    stats_by_group = cv_summary.set_index("display_group")

    for group_index, group in enumerate(PLOTTED_GROUP_ORDER):
        values = cv_df.loc[cv_df["display_group_pooled"] == group, "cv_delta"].dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        x_values = np.full(len(values), group_index, dtype=float) + jitter
        ax.scatter(x_values, values, s=48, facecolor=GROUP_COLORS[group], edgecolor="white", linewidth=0.6, alpha=0.82, zorder=2)

        row = stats_by_group.loc[group]
        estimate = float(row["median_delta_cv"])
        ci_low = float(row["delta_cv_CI95_low"])
        ci_high = float(row["delta_cv_CI95_high"])
        ax.errorbar(
            group_index, estimate,
            yerr=np.array([[estimate - ci_low], [ci_high - estimate]]),
            fmt="D", markersize=7, markerfacecolor="black", markeredgecolor="black",
            ecolor="black", elinewidth=2, capsize=5, capthick=1.8, zorder=4,
        )
        ax.annotate(f"n={int(row['n_birds'])}", xy=(group_index, ci_high), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="0.25", linestyle="--", linewidth=1.1, zorder=0)
    ax.set_xticks(range(len(PLOTTED_GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in PLOTTED_GROUP_ORDER], rotation=18, ha="right")
    ax.set_xlabel("Lesion group")
    ax.set_ylabel("Change in coefficient of variation\n(post − late pre)")
    ax.set_title("Coefficient of variation change by lesion group\nMedian with animal-level bootstrap 95% CI")
    ax.grid(axis="y", alpha=0.22)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="0.55", markeredgecolor="white", label="Individual bird", markersize=7),
        Line2D([0], [0], marker="D", linestyle="None", color="black", label="Group median", markersize=7),
        Line2D([0], [0], color="black", marker="|", linestyle="-", label="95% bootstrap CI", markersize=10),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")
    fig.text(0.01, 0.005, source_note, ha="left", va="bottom", fontsize=7.5, color="0.35")
    save_figure(fig, output_dir, "table2_bootstrap_ci_by_group_pooled_ML_CV", dpi, show)


def main() -> None:
    args = parse_args()
    input_csv = args.input.expanduser().resolve()

    if args.output is None:
        output_dir = input_csv.parent / "table2_visualizations_pooled_ML_with_CV"
    else:
        output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    df = load_and_prepare(input_csv)

    # SD plots
    summary_path = infer_bootstrap_summary_path(input_csv, args.bootstrap_summary)
    sd_bootstrap_stats, sd_source_note = load_or_calculate_sd_bootstrap_stats(
        df,
        summary_path,
        primary_method=args.primary_method,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    plot_recording_days(df, output_dir, args.dpi, args.show)
    plot_selected_syllables(df, output_dir, args.dpi, args.show)
    plot_balanced_phrase_counts(df, output_dir, args.dpi, args.show)
    plot_pre_post_sd(df, output_dir, args.dpi, args.show)
    plot_sd_delta_by_group(df, output_dir, args.dpi, args.show, args.seed)
    plot_bootstrap_ci_by_group_sd(df, sd_bootstrap_stats, sd_source_note, output_dir, args.dpi, args.show, args.seed)
    plot_sampling_coverage_composite(df, output_dir, args.dpi, args.show)

    # CV plots
    cv_plot_df, cv_source_note = prepare_cv_for_plotting(df, args.cv_input)
    if cv_plot_df is None:
        print("\nCV plots skipped.")
        print(cv_source_note)
    else:
        cv_summary = calculate_cv_group_summary(
            cv_plot_df,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed + 3000,
        )
        cv_summary.to_csv(output_dir / "table2_cv_group_summary_pooled_ML.csv", index=False)
        print(f"Saved: {output_dir / 'table2_cv_group_summary_pooled_ML.csv'}")

        plot_pre_post_cv(cv_plot_df, output_dir, args.dpi, args.show)
        plot_cv_delta_by_group(cv_plot_df, output_dir, args.dpi, args.show, args.seed)
        plot_bootstrap_ci_by_group_cv(
            cv_plot_df, cv_summary, cv_source_note, output_dir, args.dpi, args.show, args.seed
        )

        print("\nCV group summary:")
        for _, row in cv_summary.iterrows():
            print(
                f"  {GROUP_LABELS[row['display_group']]}: "
                f"median ΔCV = {row['median_delta_cv']:+.4f} "
                f"[{row['delta_cv_CI95_low']:+.4f}, {row['delta_cv_CI95_high']:+.4f}]"
            )


if __name__ == "__main__":
    main()
