#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Figure-4-style Q75 Pre/Post BC plots separately for:
    1) sham saline injection
    2) lateral lesion only
    3) pooled medial+lateral lesion

This script is a companion to:
    figure4_Q75_prepost_bc_Fig3colors_v2.py

It deliberately reuses that script's tested input parsing and statistics so it
works with the wide BC batch table containing columns such as:
    animal_id, cluster_id / cluster_token, bc_pre, bc_post, lesion_hit_type

For each lesion group, it:
    - inner-merges BC rows to the exact Figure 3 qualifying syllable universe
    - separates >=Q75 ΔCV from <Q75 ΔCV syllables
    - summarizes each subset by median BC per bird
    - uses all valid birds independently for each subset
    - runs the same raw two-sided paired Wilcoxon displayed in Figure 4
    - writes additional statistics through the base script's run_tests()
    - uses the Figure 3 teal/purple palette
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figure4_Q75_prepost_bc_Fig3colors_v2 as base


# Exact Figure 3 colors used in the current analysis.
GROUP_SPECS = [
    {
        "key": base.SHAM,
        "label": "Sham saline injection",
        "slug": "sham",
        "color": "#1B9E77",
    },
    {
        "key": base.LATERAL,
        "label": "Lateral lesion only",
        "slug": "lateral_only",
        "color": "#A88BD9",
    },
    {
        "key": base.POOLED_ML,
        "label": "Medial and lateral lesion",
        "slug": "medial_and_lateral",
        "color": "#7A4FB7",
    },
]


def make_group_bird_table(merged: pd.DataFrame, target_group: str) -> pd.DataFrame:
    """
    One Pre/Post BC value per bird for each Q75 subset within one lesion group.

    Birds may contribute to either subset independently. Only the direct
    Q75-vs-below-Q75 contrast requires both subsets, matching the corrected v2
    logic used for the pooled M+L Figure 4 analysis.
    """
    group_df = merged[merged["pooled_group"] == target_group].copy()

    grouped = (
        group_df.groupby(["animal_id", "subset"], dropna=False)
        .agg(
            n_syllables=("syllable", "nunique"),
            pre_bc=("pre_bc", "median"),
            post_bc=("post_bc", "median"),
        )
        .reset_index()
    )
    grouped["delta_bc"] = grouped["post_bc"] - grouped["pre_bc"]

    if grouped.empty:
        return pd.DataFrame(
            columns=[
                "animal_id",
                "q75_n_syllables",
                "q75_pre_bc",
                "q75_post_bc",
                "q75_delta_bc",
                "below_q75_n_syllables",
                "below_q75_pre_bc",
                "below_q75_post_bc",
                "below_q75_delta_bc",
                "q75_minus_below_delta_bc",
            ]
        )

    wide = grouped.pivot(index="animal_id", columns="subset")
    wide.columns = [
        "_".join(map(str, col)).strip("_")
        for col in wide.columns.to_flat_index()
    ]
    wide = wide.reset_index().rename(
        columns={
            "n_syllables_q75_or_above": "q75_n_syllables",
            "pre_bc_q75_or_above": "q75_pre_bc",
            "post_bc_q75_or_above": "q75_post_bc",
            "delta_bc_q75_or_above": "q75_delta_bc",
            "n_syllables_below_q75": "below_q75_n_syllables",
            "pre_bc_below_q75": "below_q75_pre_bc",
            "post_bc_below_q75": "below_q75_post_bc",
            "delta_bc_below_q75": "below_q75_delta_bc",
        }
    )

    expected = [
        "q75_n_syllables",
        "q75_pre_bc",
        "q75_post_bc",
        "q75_delta_bc",
        "below_q75_n_syllables",
        "below_q75_pre_bc",
        "below_q75_post_bc",
        "below_q75_delta_bc",
    ]
    for col in expected:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["q75_delta_bc"] = wide["q75_post_bc"] - wide["q75_pre_bc"]
    wide["below_q75_delta_bc"] = (
        wide["below_q75_post_bc"] - wide["below_q75_pre_bc"]
    )
    wide["q75_minus_below_delta_bc"] = (
        wide["q75_delta_bc"] - wide["below_q75_delta_bc"]
    )

    return wide


def draw_box(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    alpha: float,
) -> None:
    result = ax.boxplot(
        [values],
        positions=[position],
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for patch in result["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(alpha)
        patch.set_linewidth(1.4)
    for artist in result["whiskers"] + result["caps"]:
        artist.set_color("0.25")
        artist.set_linewidth(1.2)
    for artist in result["medians"]:
        artist.set_color("black")
        artist.set_linewidth(1.8)


def plot_group_panel(
    by_bird: pd.DataFrame,
    tests: pd.DataFrame,
    group_label: str,
    color: str,
    output: Path,
    dpi: int,
    show: bool,
) -> None:
    positions = {
        "q75_pre": 0.0,
        "q75_post": 1.0,
        "below_pre": 3.1,
        "below_post": 4.1,
    }

    q75_mask = (
        np.isfinite(by_bird["q75_pre_bc"].to_numpy(float))
        & np.isfinite(by_bird["q75_post_bc"].to_numpy(float))
    )
    below_mask = (
        np.isfinite(by_bird["below_q75_pre_bc"].to_numpy(float))
        & np.isfinite(by_bird["below_q75_post_bc"].to_numpy(float))
    )

    arrays = {
        "q75_pre": by_bird.loc[q75_mask, "q75_pre_bc"].to_numpy(float),
        "q75_post": by_bird.loc[q75_mask, "q75_post_bc"].to_numpy(float),
        "below_pre": by_bird.loc[below_mask, "below_q75_pre_bc"].to_numpy(float),
        "below_post": by_bird.loc[below_mask, "below_q75_post_bc"].to_numpy(float),
    }

    nonempty = [v for v in arrays.values() if len(v) > 0]
    if not nonempty:
        raise ValueError(f"No valid BC values available for {group_label}")

    fig, ax = plt.subplots(figsize=(8.3, 5.6))

    draw_box(ax, arrays["q75_pre"], positions["q75_pre"], color, alpha=0.28)
    draw_box(ax, arrays["q75_post"], positions["q75_post"], color, alpha=0.50)
    draw_box(ax, arrays["below_pre"], positions["below_pre"], color, alpha=0.28)
    draw_box(ax, arrays["below_post"], positions["below_post"], color, alpha=0.50)

    rng = np.random.default_rng(20260810)
    jitter_scale = 0.055

    for key, values in arrays.items():
        if len(values) == 0:
            continue
        x = (
            np.full(len(values), positions[key])
            + rng.uniform(-jitter_scale, jitter_scale, len(values))
        )
        point_alpha = 0.82 if "pre" in key else 0.95
        ax.scatter(
            x,
            values,
            s=40,
            color=color,
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["Pre", "Post", "Pre", "Post"], fontsize=16)
    ax.set_ylabel("Bhattacharyya coefficient", fontsize=19)
    ax.set_title(group_label, fontsize=17, pad=12)

    trans = ax.get_xaxis_transform()
    ax.text(
        0.5,
        -0.16,
        "≥ Q75 ΔCV\nsyllables",
        ha="center",
        va="top",
        transform=trans,
        fontsize=15,
    )
    ax.text(
        3.6,
        -0.16,
        "< Q75 ΔCV\nsyllables",
        ha="center",
        va="top",
        transform=trans,
        fontsize=15,
    )

    all_values = np.concatenate(nonempty)
    data_min = float(np.nanmin(all_values))
    data_max = float(np.nanmax(all_values))
    data_span = max(data_max - data_min, 0.015)

    lower = max(0.0, data_min - 0.10 * data_span)
    bracket_base = data_max + 0.09 * data_span
    bracket_height = 0.035 * data_span

    q75_row = tests.loc[tests["subset"] == ">= Q75 ΔCV syllables"].iloc[0]
    below_row = tests.loc[tests["subset"] == "< Q75 ΔCV syllables"].iloc[0]

    base.add_bracket(
        ax,
        positions["q75_pre"],
        positions["q75_post"],
        bracket_base,
        bracket_height,
        base.p_label(float(q75_row["wilcoxon_two_sided_p"])),
    )
    base.add_bracket(
        ax,
        positions["below_pre"],
        positions["below_post"],
        bracket_base,
        bracket_height,
        base.p_label(float(below_row["wilcoxon_two_sided_p"])),
    )

    ax.set_ylim(lower, bracket_base + 0.20 * data_span)
    ax.set_xlim(-0.8, 4.9)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.31)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)

    if show:
        plt.show()
    else:
        plt.close(fig)


def write_group_summary(
    path: Path,
    group_label: str,
    by_bird: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    n_q75 = int(np.sum(
        np.isfinite(by_bird["q75_pre_bc"].to_numpy(float))
        & np.isfinite(by_bird["q75_post_bc"].to_numpy(float))
    ))
    n_below = int(np.sum(
        np.isfinite(by_bird["below_q75_pre_bc"].to_numpy(float))
        & np.isfinite(by_bird["below_q75_post_bc"].to_numpy(float))
    ))
    n_both = int(np.sum(
        np.isfinite(by_bird["q75_delta_bc"].to_numpy(float))
        & np.isfinite(by_bird["below_q75_delta_bc"].to_numpy(float))
    ))

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{group_label}: Q75 Figure 4 BC analysis\n")
        handle.write("=" * 70 + "\n\n")
        handle.write(f"Birds with >=Q75 BC: {n_q75}\n")
        handle.write(f"Birds with below-Q75 BC: {n_below}\n")
        handle.write(f"Birds with both subsets: {n_both}\n")
        handle.write(
            "Bird-level statistic: median BC across qualifying syllables "
            "within each subset.\n"
        )
        handle.write(
            "Displayed test: raw two-sided paired Wilcoxon, matching the "
            "Figure 4 convention.\n\n"
        )

        for subset in [">= Q75 ΔCV syllables", "< Q75 ΔCV syllables"]:
            row = tests.loc[tests["subset"] == subset].iloc[0]
            handle.write(f"{subset}\n")
            handle.write("-" * len(subset) + "\n")
            handle.write(f"n birds: {int(row['n_birds'])}\n")
            handle.write(f"median Pre BC: {row['median_pre_bc']:.8g}\n")
            handle.write(f"median Post BC: {row['median_post_bc']:.8g}\n")
            handle.write(
                "median ΔBC (Post - Pre): "
                f"{row['median_delta_bc_post_minus_pre']:.8g}\n"
            )
            handle.write(
                "raw two-sided Wilcoxon p: "
                f"{row['wilcoxon_two_sided_p']:.8g}\n"
            )
            handle.write(
                "Holm-adjusted two-sided Wilcoxon p: "
                f"{row['wilcoxon_two_sided_holm_p']:.8g}\n\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate separate sham, lateral-only, and pooled M+L Q75 "
            "Pre/Post BC plots using the same parsing/statistics as "
            "figure4_Q75_prepost_bc_Fig3colors_v2.py."
        )
    )
    parser.add_argument("--bc-csv", required=True)
    parser.add_argument("--quantile-selection-csv", required=True)
    parser.add_argument("--selection-column", default="selected_Q75")
    parser.add_argument("--metadata-excel", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = base.read_quantile_selection(
        args.quantile_selection_csv,
        selection_column=args.selection_column,
    )
    all_fig3_pairs = selection[["animal_id", "syllable"]].drop_duplicates()

    # This is the important fix relative to the first by-group script:
    # base.read_bc_table() correctly handles the actual wide BC CSV schema
    # containing cluster_id/cluster_token plus bc_pre and bc_post.
    bc, identifier_report = base.read_bc_table(
        args.bc_csv,
        args.metadata_excel,
        all_fig3_pairs,
    )
    identifier_report.to_csv(
        out_dir / "figure4_Q75_by_group_bc_identifier_column_match_report.csv",
        index=False,
    )

    merged = bc.merge(
        selection,
        on=["animal_id", "syllable"],
        how="inner",
    )
    merged["subset"] = np.where(
        merged["is_q75"],
        "q75_or_above",
        "below_q75",
    )

    merged.to_csv(
        out_dir / "figure4_Q75_by_group_cluster_rows.csv",
        index=False,
    )

    summary_rows = []

    for spec in GROUP_SPECS:
        by_bird = make_group_bird_table(merged, spec["key"])
        if by_bird.empty:
            print(f"[WARN] No matching rows for {spec['label']}; skipping.")
            continue

        n_q75 = int(np.sum(
            np.isfinite(by_bird["q75_pre_bc"].to_numpy(float))
            & np.isfinite(by_bird["q75_post_bc"].to_numpy(float))
        ))
        n_below = int(np.sum(
            np.isfinite(by_bird["below_q75_pre_bc"].to_numpy(float))
            & np.isfinite(by_bird["below_q75_post_bc"].to_numpy(float))
        ))
        n_both = int(np.sum(
            np.isfinite(by_bird["q75_delta_bc"].to_numpy(float))
            & np.isfinite(by_bird["below_q75_delta_bc"].to_numpy(float))
        ))

        print(
            f"[INFO] {spec['label']}: "
            f">=Q75 n={n_q75}, below-Q75 n={n_below}, both n={n_both}"
        )

        # base.run_tests() uses each subset's available birds independently,
        # exactly as in the corrected pooled-M+L v2 script.
        tests = base.run_tests(by_bird)

        bird_path = out_dir / f"figure4_Q75_{spec['slug']}_by_bird.csv"
        test_path = out_dir / f"figure4_Q75_{spec['slug']}_tests.csv"
        fig_path = out_dir / f"Figure4_Q75_prepost_BC_{spec['slug']}_Fig3colors.png"
        text_path = out_dir / f"figure4_Q75_{spec['slug']}_summary.txt"

        by_bird.to_csv(bird_path, index=False)
        tests.to_csv(test_path, index=False)

        plot_group_panel(
            by_bird=by_bird,
            tests=tests,
            group_label=spec["label"],
            color=spec["color"],
            output=fig_path,
            dpi=args.dpi,
            show=args.show,
        )
        write_group_summary(
            path=text_path,
            group_label=spec["label"],
            by_bird=by_bird,
            tests=tests,
        )

        summary_rows.append(
            {
                "lesion_group": spec["label"],
                "n_q75_birds": n_q75,
                "n_below_q75_birds": n_below,
                "n_both_subsets_birds": n_both,
                "figure": str(fig_path),
                "bird_table": str(bird_path),
                "tests": str(test_path),
                "summary": str(text_path),
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "figure4_Q75_by_lesion_group_manifest.csv",
        index=False,
    )

    print(f"[OK] Wrote lesion-group-specific Q75 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
