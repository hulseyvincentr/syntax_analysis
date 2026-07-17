#!/usr/bin/env python3
"""
Visualize Supplemental Table 2 animal metadata, sampling coverage, and SD results.

Outputs
-------
1. table2_recording_days_dumbbell
2. table2_selected_syllables_by_bird
3. table2_balanced_phrase_counts_by_bird
4. table2_pre_vs_post_sd_dumbbell
5. table2_sd_delta_by_group
6. table2_sampling_coverage_composite

Each figure is saved as both PNG and PDF by default.

Example
-------
python supplemental_table2_visualizations.py \
    --input "/Volumes/my_own_SSD/updated_AreaX_outputs/Figure3_robust_stats_tables_v2/Supplemental_Table_2_animal_metadata_and_sample_sizes.csv" \
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

# These colors can be edited to match the final manuscript palette.
GROUP_COLORS = {
    "sham saline injection": "#2A9D8F",
    "Lateral lesion only": "#457B9D",
    "Partial Medial and Lateral lesion": "#8D5FD3",
    "Complete Medial and Lateral lesion": "#6C757D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-style visualizations from Supplemental Table 2."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV. Default: {DEFAULT_INPUT}",
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
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="Output formats. Default: png pdf",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster image resolution. Default: 300",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


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


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
    show: bool,
) -> None:
    fig.tight_layout()

    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
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
    orientation: str = "horizontal",
) -> None:
    """Add subtle separators between lesion groups."""
    previous_group = None
    for index, group in enumerate(df["display_group"]):
        if previous_group is not None and group != previous_group:
            boundary = index - 0.5
            if orientation == "horizontal":
                ax.axhline(boundary, color="0.85", linewidth=0.8, zorder=0)
            else:
                ax.axvline(boundary, color="0.85", linewidth=0.8, zorder=0)
        previous_group = group


def plot_recording_days(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
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
        formats,
        dpi,
        show,
    )


def plot_selected_syllables(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
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
        formats,
        dpi,
        show,
    )


def plot_balanced_phrase_counts(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
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
        formats,
        dpi,
        show,
    )


def plot_pre_post_sd(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
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
        formats,
        dpi,
        show,
    )


def plot_sd_delta_by_group(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    observed_groups = [
        group for group in GROUP_ORDER if group in set(df["display_group"])
    ]

    rng = np.random.default_rng(123)

    for group_index, group in enumerate(observed_groups):
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
    ax.set_xticks(range(len(observed_groups)))
    ax.set_xticklabels(
        [GROUP_LABELS[group] for group in observed_groups],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Change in phrase duration SD (post − late pre, s)")
    ax.set_xlabel("Group")
    ax.set_title("Bird-level change in phrase-duration SD")
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        output_dir,
        "table2_sd_delta_by_group",
        formats,
        dpi,
        show,
    )


def plot_sampling_coverage_composite(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
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
        formats,
        dpi,
        show,
    )


def print_summary(df: pd.DataFrame, output_dir: Path) -> None:
    print("\nCompleted Supplemental Table 2 visualizations.")
    print(f"Output directory:\n{output_dir}")

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

    print("\nBalanced phrase observations per epoch:")
    print(
        f"  Range: {int(df['balanced_phrases_per_epoch'].min()):,}"
        f"–{int(df['balanced_phrases_per_epoch'].max()):,}"
    )

    print("\nSelected syllables:")
    print(
        f"  Range: {int(df['n_selected_syllables'].min())}"
        f"–{int(df['n_selected_syllables'].max())}"
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

    plot_recording_days(
        df, output_dir, args.formats, args.dpi, args.show
    )
    plot_selected_syllables(
        df, output_dir, args.formats, args.dpi, args.show
    )
    plot_balanced_phrase_counts(
        df, output_dir, args.formats, args.dpi, args.show
    )
    plot_pre_post_sd(
        df, output_dir, args.formats, args.dpi, args.show
    )
    plot_sd_delta_by_group(
        df, output_dir, args.formats, args.dpi, args.show
    )
    plot_sampling_coverage_composite(
        df, output_dir, args.formats, args.dpi, args.show
    )

    print_summary(df, output_dir)


if __name__ == "__main__":
    main()
