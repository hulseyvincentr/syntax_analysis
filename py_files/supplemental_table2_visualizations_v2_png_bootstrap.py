#!/usr/bin/env python3
"""
Visualize Supplemental Table 2 animal metadata, sampling coverage, and SD results.

The script saves PNG files only.

Outputs
-------
1. table2_recording_days_dumbbell.png
2. table2_selected_syllables_by_bird.png
3. table2_balanced_phrase_counts_by_bird.png
4. table2_pre_vs_post_sd_dumbbell.png
5. table2_sd_delta_by_group.png
6. table2_sampling_coverage_composite.png
7. table2_bootstrap_ci_by_lesion_hit_type.png

The bootstrap-CI plot uses the precomputed group summary from the original
Figure 3 robustness run when that file is available. By default, the script
looks beside Supplemental Table 2 for:

    Supplemental_Table_1_selection_robustness_group_summary.csv

If that file is unavailable, the script calculates percentile bootstrap
confidence intervals directly from the animal-level sd_delta_s values in
Supplemental Table 2.

Example
-------
python supplemental_table2_visualizations.py \
    --input "/Volumes/my_own_SSD/updated_AreaX_outputs/Figure3_robust_stats_tables_v2/Supplemental_Table_2_animal_metadata_and_sample_sizes.csv" \
    --bootstrap-summary "/Volumes/my_own_SSD/updated_AreaX_outputs/Figure3_robust_stats_tables_v2/Supplemental_Table_1_selection_robustness_group_summary.csv" \
    --output "/Volumes/my_own_SSD/updated_AreaX_outputs/Figure3_robust_stats_tables_v2/table2_visualizations"
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

GROUP_ORDER = [
    "sham saline injection",
    "Lateral lesion only",
    "Partial Medial and Lateral lesion",
    "Complete Medial and Lateral lesion",
]

GROUP_LABELS = {
    "sham saline injection": "Sham",
    "Lateral lesion only": "Lateral only",
    "Partial Medial and Lateral lesion": "Partial M+L",
    "Complete Medial and Lateral lesion": "Complete M+L",
}

# Edit these values if you want to match a different manuscript palette.
GROUP_COLORS = {
    "sham saline injection": "#2A9D8F",
    "Lateral lesion only": "#457B9D",
    "Partial Medial and Lateral lesion": "#8D5FD3",
    "Complete Medial and Lateral lesion": "#6C757D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG visualizations from Supplemental Table 2, including "
            "bootstrap 95% confidence intervals by lesion hit type."
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
            "Optional Supplemental Table 1 group-summary CSV containing the "
            "precomputed bootstrap confidence intervals. If omitted, the script "
            "looks beside the input CSV. If no summary is found, confidence "
            "intervals are recalculated from Supplemental Table 2."
        ),
    )
    parser.add_argument(
        "--primary-method",
        default=DEFAULT_PRIMARY_METHOD,
        help=(
            "Selection-method row to use from the bootstrap summary. "
            f"Default: {DEFAULT_PRIMARY_METHOD!r}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: a 'table2_visualizations' folder beside "
            "the input CSV."
        ),
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=5000,
        help=(
            "Bootstrap iterations used only if the precomputed summary is "
            "unavailable. Default: 5000"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for jitter and fallback bootstrap. Default: 123",
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


def choose_first_existing(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        "None of the expected phrase-count columns were present:\n"
        + "\n".join(f"  - {column}" for column in candidates)
    )


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
    df["treatment_date"] = pd.to_datetime(
        df["treatment_date"], errors="coerce"
    )

    unknown_groups = sorted(set(df["display_group"]) - set(GROUP_ORDER))
    if unknown_groups:
        raise ValueError(
            "Unexpected display_group values were found:\n"
            + "\n".join(f"  - {group}" for group in unknown_groups)
            + "\nUpdate GROUP_ORDER, GROUP_LABELS, and GROUP_COLORS in the script."
        )

    pre_phrase_column = choose_first_existing(
        df,
        [
            "n_late_pre_phrases",
            "total_late_pre_phrases_selected",
        ],
    )
    post_phrase_column = choose_first_existing(
        df,
        [
            "n_post_phrases",
            "total_post_phrases_selected",
        ],
    )

    df[pre_phrase_column] = pd.to_numeric(
        df[pre_phrase_column], errors="coerce"
    )
    df[post_phrase_column] = pd.to_numeric(
        df[post_phrase_column], errors="coerce"
    )

    # The Figure 3 analysis was usage-balanced, so these values should usually
    # match. The row-wise median remains safe if a future table contains a small
    # mismatch.
    df["balanced_phrases_per_epoch"] = df[
        [pre_phrase_column, post_phrase_column]
    ].median(axis=1, skipna=True)

    mismatched = df[
        ~np.isclose(
            df[pre_phrase_column],
            df[post_phrase_column],
            equal_nan=True,
        )
    ]
    if not mismatched.empty:
        print(
            "Warning: late-pre and post phrase counts differ for "
            f"{len(mismatched)} bird(s). The plotted balanced count is the "
            "row-wise median."
        )

    group_rank = {group: index for index, group in enumerate(GROUP_ORDER)}
    df["_group_rank"] = df["display_group"].map(group_rank)
    df = df.sort_values(
        ["_group_rank", "treatment_date", "animal_id"],
        na_position="last",
    ).reset_index(drop=True)

    df["group_short"] = df["display_group"].map(GROUP_LABELS)
    df["plot_color"] = df["display_group"].map(GROUP_COLORS)
    df["bird_label"] = (
        df["animal_id"] + "   (" + df["group_short"] + ")"
    )

    return df


def infer_bootstrap_summary_path(
    input_csv: Path,
    requested_path: Path | None,
) -> Path | None:
    if requested_path is not None:
        requested_path = requested_path.expanduser().resolve()
        if not requested_path.exists():
            raise FileNotFoundError(
                f"Bootstrap group-summary CSV not found:\n{requested_path}"
            )
        return requested_path

    candidate = (
        input_csv.parent
        / "Supplemental_Table_1_selection_robustness_group_summary.csv"
    )
    if candidate.exists():
        return candidate

    return None


def percentile_bootstrap_ci(
    values: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    """
    Return the observed median and percentile bootstrap 95% CI.

    Resampling unit: animal.
    Summary statistic: median.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values,
        size=(reps, values.size),
        replace=True,
    )
    bootstrap_medians = np.nanmedian(samples, axis=1)
    low, high = np.nanpercentile(bootstrap_medians, [2.5, 97.5])

    return (
        float(np.nanmedian(values)),
        float(low),
        float(high),
    )


def load_or_calculate_bootstrap_stats(
    df: pd.DataFrame,
    summary_path: Path | None,
    *,
    primary_method: str,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    """
    Return one row per distinct lesion hit type.

    The original Figure 3 robustness script bootstrapped the median animal-level
    delta. Therefore, the plotted point estimate is median_delta_sd_s and the
    interval is delta_sd_CI95_low_s to delta_sd_CI95_high_s.
    """
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
        missing = [
            column
            for column in required_summary_columns
            if column not in summary.columns
        ]
        if missing:
            raise ValueError(
                "The bootstrap group-summary CSV is missing columns:\n"
                + "\n".join(f"  - {column}" for column in missing)
            )

        selected = summary[
            summary["selection_method"].astype(str).eq(primary_method)
            & summary["display_group"].astype(str).isin(GROUP_ORDER)
        ].copy()

        if selected.empty:
            methods = sorted(
                summary["selection_method"].dropna().astype(str).unique()
            )
            raise ValueError(
                f"No rows matched --primary-method {primary_method!r}.\n"
                "Available selection methods include:\n"
                + "\n".join(f"  - {method}" for method in methods)
            )

        selected["display_group"] = selected["display_group"].astype(str)
        selected = selected.drop_duplicates("display_group")

        missing_groups = [
            group
            for group in GROUP_ORDER
            if group not in set(selected["display_group"])
        ]
        if missing_groups:
            raise ValueError(
                "The selected bootstrap-summary rows are missing groups:\n"
                + "\n".join(f"  - {group}" for group in missing_groups)
            )

        selected = selected.set_index("display_group").loc[GROUP_ORDER].reset_index()
        selected = selected[
            [
                "display_group",
                "n_birds",
                "median_delta_sd_s",
                "delta_sd_CI95_low_s",
                "delta_sd_CI95_high_s",
            ]
        ].copy()

        source_note = (
            f"Precomputed animal-level bootstrap CIs from {summary_path.name}; "
            f"selection method: {primary_method}"
        )
        return selected, source_note

    rows = []
    for group_index, group in enumerate(GROUP_ORDER):
        values = df.loc[
            df["display_group"] == group,
            "sd_delta_s",
        ].to_numpy(dtype=float)

        median, low, high = percentile_bootstrap_ci(
            values,
            reps=bootstrap_reps,
            seed=seed + group_index,
        )
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
        f"({bootstrap_reps:,} resamples; median statistic)"
    )
    return fallback, source_note


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
        }
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
    show: bool,
) -> None:
    """Save one PNG file only."""
    fig.tight_layout()
    output_path = output_dir / f"{stem}.png"
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def group_legend_handles(df: pd.DataFrame) -> list[Patch]:
    observed = set(df["display_group"])
    return [
        Patch(
            facecolor=GROUP_COLORS[group],
            edgecolor="none",
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
        if group in observed
    ]


def add_group_boundaries(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    """Add subtle horizontal separators between lesion groups."""
    previous_group = None
    for index, group in enumerate(df["display_group"]):
        if previous_group is not None and group != previous_group:
            ax.axhline(
                index - 0.5,
                color="0.85",
                linewidth=0.8,
                zorder=0,
            )
        previous_group = group


def plot_recording_days(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))

    for i, row in df.iterrows():
        color = row["plot_color"]
        ax.plot(
            [row["n_pre_days"], row["n_post_days"]],
            [i, i],
            color=color,
            linewidth=1.5,
            alpha=0.85,
            zorder=1,
        )
        ax.scatter(
            row["n_pre_days"],
            i,
            marker="o",
            s=52,
            color=color,
            zorder=2,
        )
        ax.scatter(
            row["n_post_days"],
            i,
            marker="s",
            s=52,
            color=color,
            zorder=2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["bird_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of recording days")
    ax.set_ylabel("Bird")
    ax.set_title("Recording coverage by bird")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, df)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="black",
            label="Pre days",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="black",
            label="Post days",
            markersize=7,
        ),
    ]
    ax.legend(
        handles=group_legend_handles(df) + marker_handles,
        frameon=False,
        loc="lower right",
    )

    save_figure(
        fig,
        output_dir,
        "table2_recording_days_dumbbell",
        dpi,
        show,
    )


def plot_selected_syllables(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))

    ax.barh(
        y,
        df["n_selected_syllables"],
        color=df["plot_color"],
        alpha=0.9,
    )

    x_range = (
        df["n_selected_syllables"].max()
        - df["n_selected_syllables"].min()
    )
    label_offset = max(0.12, x_range * 0.02)

    for i, row in df.iterrows():
        value = row["n_selected_syllables"]
        ax.text(
            value + label_offset,
            i,
            f"{int(value)}",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["bird_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of selected syllables")
    ax.set_ylabel("Bird")
    ax.set_title("Selected syllables per bird")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, df)
    ax.legend(
        handles=group_legend_handles(df),
        frameon=False,
        loc="lower right",
    )

    save_figure(
        fig,
        output_dir,
        "table2_selected_syllables_by_bird",
        dpi,
        show,
    )


def plot_balanced_phrase_counts(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))

    for i, row in df.iterrows():
        value = row["balanced_phrases_per_epoch"]
        ax.scatter(
            value,
            i,
            s=62,
            color=row["plot_color"],
            zorder=2,
        )
        ax.annotate(
            f"{int(value):,}",
            xy=(value, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["bird_label"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Balanced phrase observations per epoch (log scale)")
    ax.set_ylabel("Bird")
    ax.set_title("Balanced phrase count per epoch")
    ax.grid(axis="x", which="both", alpha=0.25)
    add_group_boundaries(ax, df)
    ax.legend(
        handles=group_legend_handles(df),
        frameon=False,
        loc="lower right",
    )

    save_figure(
        fig,
        output_dir,
        "table2_balanced_phrase_counts_by_bird",
        dpi,
        show,
    )


def plot_pre_post_sd(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))

    sd_values = np.concatenate(
        [
            df["pre_sd_s"].dropna().to_numpy(),
            df["post_sd_s"].dropna().to_numpy(),
        ]
    )
    label_offset = max(0.015, np.ptp(sd_values) * 0.02)

    for i, row in df.iterrows():
        color = row["plot_color"]
        ax.plot(
            [row["pre_sd_s"], row["post_sd_s"]],
            [i, i],
            color=color,
            linewidth=1.5,
            alpha=0.85,
            zorder=1,
        )
        ax.scatter(
            row["pre_sd_s"],
            i,
            marker="o",
            s=52,
            color=color,
            zorder=2,
        )
        ax.scatter(
            row["post_sd_s"],
            i,
            marker="s",
            s=52,
            color=color,
            zorder=2,
        )
        ax.text(
            max(row["pre_sd_s"], row["post_sd_s"]) + label_offset,
            i,
            f"Δ={row['sd_delta_s']:+.3f} s",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["bird_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Phrase duration SD (s)")
    ax.set_ylabel("Bird")
    ax.set_title("Late-pre versus post phrase-duration variability")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, df)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="black",
            label="Late pre",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="black",
            label="Post",
            markersize=7,
        ),
    ]
    ax.legend(
        handles=group_legend_handles(df) + marker_handles,
        frameon=False,
        loc="lower right",
    )

    save_figure(
        fig,
        output_dir,
        "table2_pre_vs_post_sd_dumbbell",
        dpi,
        show,
    )


def plot_sd_delta_by_group(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
    seed: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(seed)

    for group_index, group in enumerate(GROUP_ORDER):
        values = (
            df.loc[df["display_group"] == group, "sd_delta_s"]
            .dropna()
            .to_numpy()
        )
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        x = np.full(len(values), group_index, dtype=float) + jitter

        ax.scatter(
            x,
            values,
            s=62,
            color=GROUP_COLORS[group],
            alpha=0.9,
            zorder=2,
        )

        median = np.median(values)
        q1, q3 = np.percentile(values, [25, 75])
        ax.vlines(
            group_index,
            q1,
            q3,
            color="black",
            linewidth=2,
            zorder=3,
        )
        ax.plot(
            [group_index - 0.18, group_index + 0.18],
            [median, median],
            color="black",
            linewidth=2,
            zorder=3,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(
        [GROUP_LABELS[group] for group in GROUP_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Change in phrase duration SD\n(post − late pre, s)")
    ax.set_xlabel("Lesion hit type")
    ax.set_title("Bird-level change in phrase-duration SD")
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        output_dir,
        "table2_sd_delta_by_group",
        dpi,
        show,
    )


def plot_bootstrap_ci_by_lesion_hit_type(
    df: pd.DataFrame,
    bootstrap_stats: pd.DataFrame,
    source_note: str,
    output_dir: Path,
    dpi: int,
    show: bool,
    seed: int,
) -> None:
    """
    Plot individual animal ΔSD values plus group median and bootstrap 95% CI.

    The four distinct lesion hit types are shown separately. The pooled
    complete+partial M+L row is intentionally excluded because it is not a
    distinct hit type and would duplicate animals already displayed.
    """
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    rng = np.random.default_rng(seed)

    stats_by_group = bootstrap_stats.set_index("display_group")

    for group_index, group in enumerate(GROUP_ORDER):
        values = (
            df.loc[df["display_group"] == group, "sd_delta_s"]
            .dropna()
            .to_numpy(dtype=float)
        )
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        x_values = np.full(len(values), group_index, dtype=float) + jitter

        ax.scatter(
            x_values,
            values,
            s=48,
            facecolor=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.82,
            zorder=2,
        )

        row = stats_by_group.loc[group]
        estimate = float(row["median_delta_sd_s"])
        ci_low = float(row["delta_sd_CI95_low_s"])
        ci_high = float(row["delta_sd_CI95_high_s"])

        lower_error = estimate - ci_low
        upper_error = ci_high - estimate

        ax.errorbar(
            group_index,
            estimate,
            yerr=np.array([[lower_error], [upper_error]]),
            fmt="D",
            markersize=7,
            markerfacecolor="black",
            markeredgecolor="black",
            ecolor="black",
            elinewidth=2,
            capsize=5,
            capthick=1.8,
            zorder=4,
        )

        ax.annotate(
            f"n={int(row['n_birds'])}",
            xy=(group_index, ci_high),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(
        0,
        color="0.25",
        linestyle="--",
        linewidth=1.1,
        zorder=0,
    )
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(
        [GROUP_LABELS[group] for group in GROUP_ORDER],
        rotation=18,
        ha="right",
    )
    ax.set_xlabel("Lesion hit type")
    ax.set_ylabel("Change in phrase duration SD\n(post − late pre, s)")
    ax.set_title(
        "Phrase-duration SD change by lesion hit type\n"
        "Median with animal-level bootstrap 95% CI"
    )
    ax.grid(axis="y", alpha=0.22)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="0.55",
            markeredgecolor="white",
            label="Individual bird",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            color="black",
            label="Group median",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="|",
            linestyle="-",
            label="95% bootstrap CI",
            markersize=10,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
    )

    fig.text(
        0.01,
        0.005,
        source_note,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.35",
    )

    save_figure(
        fig,
        output_dir,
        "table2_bootstrap_ci_by_lesion_hit_type",
        dpi,
        show,
    )


def plot_sampling_coverage_composite(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    """
    Compact three-panel summary:
    A. Pre/post recording days
    B. Selected syllables
    C. Balanced phrase observations per epoch
    """
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.1, 0.9, 1.2]},
    )
    y = np.arange(len(df))

    # Panel A: days
    ax = axes[0]
    for i, row in df.iterrows():
        color = row["plot_color"]
        ax.plot(
            [row["n_pre_days"], row["n_post_days"]],
            [i, i],
            color=color,
            linewidth=1.4,
            alpha=0.85,
        )
        ax.scatter(row["n_pre_days"], i, marker="o", s=42, color=color)
        ax.scatter(row["n_post_days"], i, marker="s", s=42, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(df["animal_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Recording days")
    ax.set_ylabel("Bird")
    ax.set_title("A  Recording coverage", loc="left")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, df)

    # Panel B: selected syllables
    ax = axes[1]
    ax.barh(
        y,
        df["n_selected_syllables"],
        color=df["plot_color"],
        alpha=0.9,
    )
    ax.set_xlabel("Selected syllables")
    ax.set_title("B  Syllable inclusion", loc="left")
    ax.grid(axis="x", alpha=0.25)
    add_group_boundaries(ax, df)

    # Panel C: balanced phrases
    ax = axes[2]
    for i, row in df.iterrows():
        ax.scatter(
            row["balanced_phrases_per_epoch"],
            i,
            s=52,
            color=row["plot_color"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Balanced phrases per epoch\n(log scale)")
    ax.set_title("C  Phrase observations", loc="left")
    ax.grid(axis="x", which="both", alpha=0.25)
    add_group_boundaries(ax, df)

    legend_handles = group_legend_handles(df) + [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="black",
            label="Pre days",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="black",
            label="Post days",
            markersize=7,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )

    fig.subplots_adjust(wspace=0.12)

    save_figure(
        fig,
        output_dir,
        "table2_sampling_coverage_composite",
        dpi,
        show,
    )


def print_summary(
    df: pd.DataFrame,
    bootstrap_stats: pd.DataFrame,
    source_note: str,
    output_dir: Path,
) -> None:
    print("\nCompleted Supplemental Table 2 visualizations.")
    print(f"Output directory:\n{output_dir}")
    print("File format: PNG only")

    print("\nBird counts by group:")
    counts = (
        df["group_short"]
        .value_counts()
        .reindex([GROUP_LABELS[group] for group in GROUP_ORDER])
        .dropna()
        .astype(int)
    )
    for group, count in counts.items():
        print(f"  {group}: {count}")

    print("\nBootstrap CI source:")
    print(f"  {source_note}")

    print("\nBootstrap median ΔSD and 95% CI:")
    for _, row in bootstrap_stats.iterrows():
        group = row["display_group"]
        estimate = row["median_delta_sd_s"]
        low = row["delta_sd_CI95_low_s"]
        high = row["delta_sd_CI95_high_s"]
        print(
            f"  {GROUP_LABELS[group]}: "
            f"{estimate:+.4f} s [{low:+.4f}, {high:+.4f}]"
        )


def main() -> None:
    args = parse_args()
    input_csv = args.input.expanduser().resolve()

    if args.output is None:
        output_dir = input_csv.parent / "table2_visualizations"
    else:
        output_dir = args.output.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    df = load_and_prepare(input_csv)

    summary_path = infer_bootstrap_summary_path(
        input_csv,
        args.bootstrap_summary,
    )
    bootstrap_stats, source_note = load_or_calculate_bootstrap_stats(
        df,
        summary_path,
        primary_method=args.primary_method,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    plot_recording_days(
        df, output_dir, args.dpi, args.show
    )
    plot_selected_syllables(
        df, output_dir, args.dpi, args.show
    )
    plot_balanced_phrase_counts(
        df, output_dir, args.dpi, args.show
    )
    plot_pre_post_sd(
        df, output_dir, args.dpi, args.show
    )
    plot_sd_delta_by_group(
        df, output_dir, args.dpi, args.show, args.seed
    )
    plot_bootstrap_ci_by_lesion_hit_type(
        df,
        bootstrap_stats,
        source_note,
        output_dir,
        args.dpi,
        args.show,
        args.seed,
    )
    plot_sampling_coverage_composite(
        df, output_dir, args.dpi, args.show
    )

    print_summary(
        df,
        bootstrap_stats,
        source_note,
        output_dir,
    )


if __name__ == "__main__":
    main()
