#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

SHAM_COLOR = "#1B9E77"
LATERAL_COLOR = "#A88BD9"
ML_COLOR = "#7A4FB7"

GROUP_COLORS = {
    "sham saline injection": SHAM_COLOR,
    "Lateral lesion only": LATERAL_COLOR,
    "Medial and Lateral lesion": ML_COLOR,
}

GROUP_ORDER = [
    "sham saline injection",
    "Lateral lesion only",
    "Medial and Lateral lesion",
]


def clean_cluster_label(x):
    s = str(x).strip()
    s = s.replace("label", "")
    s = s.replace(".0", "")
    return s


def load_selection_table(selection_csv, selection_column):
    sel = pd.read_csv(selection_csv).copy()

    possible_animal_cols = ["animal_id", "Animal ID", "bird", "Bird", "bird_id"]
    possible_label_cols = ["syllable", "label", "cluster_label", "cluster", "syllable_label"]

    animal_col = next((c for c in possible_animal_cols if c in sel.columns), None)
    label_col = next((c for c in possible_label_cols if c in sel.columns), None)

    if animal_col is None or label_col is None:
        raise ValueError(
            f"Could not find animal/label columns in selection CSV. Columns: {list(sel.columns)}"
        )

    if selection_column not in sel.columns:
        raise ValueError(
            f"Selection column '{selection_column}' not found. Columns: {list(sel.columns)}"
        )

    out = sel[[animal_col, label_col, selection_column]].copy()
    out = out.rename(columns={
        animal_col: "animal_id",
        label_col: "cluster_label",
        selection_column: "selected_q75",
    })
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out["cluster_label"] = out["cluster_label"].map(clean_cluster_label)
    out["selected_q75"] = out["selected_q75"].astype(bool)
    return out


def load_metadata(metadata_excel):
    meta = pd.read_excel(metadata_excel, sheet_name="animal_hit_type_summary").copy()

    possible_animal_cols = ["Animal ID", "animal_id", "Animal", "bird"]
    possible_group_cols = ["Lesion hit type", "lesion_hit_type", "hit_type"]

    animal_col = next((c for c in possible_animal_cols if c in meta.columns), None)
    group_col = next((c for c in possible_group_cols if c in meta.columns), None)

    if animal_col is None or group_col is None:
        raise ValueError(
            f"Could not find animal/group columns in metadata. Columns: {list(meta.columns)}"
        )

    meta = meta[[animal_col, group_col]].copy()
    meta = meta.rename(columns={animal_col: "animal_id", group_col: "lesion_group"})
    meta["animal_id"] = meta["animal_id"].astype(str).str.strip()
    return meta


def load_bc_table(bc_csv):
    bc = pd.read_csv(bc_csv).copy()

    possible_animal_cols = ["animal_id", "Animal ID", "bird", "Bird"]
    possible_label_cols = ["cluster_label", "label", "syllable", "cluster"]
    possible_period_cols = ["period", "epoch"]
    possible_bc_cols = ["bc", "BC", "bhattacharyya_coefficient", "Bhattacharyya coefficient"]

    animal_col = next((c for c in possible_animal_cols if c in bc.columns), None)
    label_col = next((c for c in possible_label_cols if c in bc.columns), None)
    period_col = next((c for c in possible_period_cols if c in bc.columns), None)
    bc_col = next((c for c in possible_bc_cols if c in bc.columns), None)

    if None in (animal_col, label_col, period_col, bc_col):
        raise ValueError(
            f"Could not identify required columns in BC CSV. Columns: {list(bc.columns)}"
        )

    bc = bc.rename(columns={
        animal_col: "animal_id",
        label_col: "cluster_label",
        period_col: "period",
        bc_col: "bc",
    })

    bc["animal_id"] = bc["animal_id"].astype(str).str.strip()
    bc["cluster_label"] = bc["cluster_label"].map(clean_cluster_label)
    bc["period"] = bc["period"].astype(str).str.strip().str.lower()

    return bc[["animal_id", "cluster_label", "period", "bc"]].copy()


def collapse_periods_to_pre_post(df):
    pre_periods = {"early_pre", "late_pre", "pre"}
    post_periods = {"early_post", "late_post", "post"}

    temp = df.copy()
    temp["prepost"] = np.where(
        temp["period"].isin(pre_periods),
        "Pre",
        np.where(temp["period"].isin(post_periods), "Post", np.nan),
    )
    temp = temp.dropna(subset=["prepost"]).copy()

    return (
        temp.groupby(["animal_id", "cluster_label", "prepost"], as_index=False)["bc"]
        .mean()
    )


def bird_level_median(df):
    return (
        df.groupby(["animal_id", "lesion_group", "subset", "prepost"], as_index=False)["bc"]
        .median()
    )


def paired_wilcoxon_from_long(df_group_subset):
    wide = df_group_subset.pivot(index="animal_id", columns="prepost", values="bc").dropna()

    if len(wide) < 2 or "Pre" not in wide.columns or "Post" not in wide.columns:
        return {"n_birds": len(wide), "p_value": np.nan, "statistic": np.nan, "label": "n.s."}

    try:
        stat, p = wilcoxon(wide["Pre"], wide["Post"], alternative="two-sided")
    except ValueError:
        stat, p = np.nan, np.nan

    if pd.isna(p):
        label = "n.s."
    elif p < 0.01:
        label = "**"
    elif p < 0.05:
        label = "*"
    else:
        label = "n.s."

    return {"n_birds": len(wide), "p_value": p, "statistic": stat, "label": label}


def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=18):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.6, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def draw_group_figure(group_name, group_df, out_dir):
    color = GROUP_COLORS[group_name]
    bird_long = bird_level_median(group_df)

    safe_group = group_name.replace(" ", "_").replace("/", "_").replace("__", "_")
    bird_long.to_csv(
        os.path.join(out_dir, f"{safe_group}_bird_level_prepost_bc.csv"),
        index=False,
    )

    subsets = [
        ("≥ Q75 ΔCV syllables", "high"),
        ("< Q75 ΔCV syllables", "low"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    positions = [1, 2, 4, 5]
    labels = ["Pre", "Post", "Pre", "Post"]

    plot_data = []
    stats_rows = []

    for _, subset_key in subsets:
        sub = bird_long[bird_long["subset"] == subset_key].copy()
        stats = paired_wilcoxon_from_long(sub)
        stats["lesion_group"] = group_name
        stats["subset"] = subset_key
        stats_rows.append(stats)

        wide = sub.pivot(index="animal_id", columns="prepost", values="bc")
        pre_vals = wide["Pre"].dropna().values if "Pre" in wide.columns else np.array([])
        post_vals = wide["Post"].dropna().values if "Post" in wide.columns else np.array([])
        plot_data.extend([pre_vals, post_vals])

    bp = ax.boxplot(
        plot_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#444444", linewidth=1.5),
        capprops=dict(color="#444444", linewidth=1.5),
        boxprops=dict(edgecolor=color, linewidth=2),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(plot_data):
        if len(vals) == 0:
            continue
        x0 = positions[i]
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), x0) + jitter,
            vals,
            s=55,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            alpha=0.9,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel("Bhattacharyya coefficient", fontsize=24)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xlim(0, 6)
    ax.set_title(f"Bird level: {group_name}\nSelected equal time bins", fontsize=22, pad=18)

    stats_df = pd.DataFrame(stats_rows)

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    subset_text_y = ymin - 0.08 * yr
    p_text_y = ymin - 0.16 * yr

    ax.text(1.5, subset_text_y, "≥ Q75 ΔCV\nsyllables", ha="center", va="top", fontsize=18)
    ax.text(4.5, subset_text_y, "< Q75 ΔCV\nsyllables", ha="center", va="top", fontsize=18)

    left_stats = stats_df[stats_df["subset"] == "high"].iloc[0]
    right_stats = stats_df[stats_df["subset"] == "low"].iloc[0]

    ax.text(
        1.5,
        p_text_y,
        f"pre/post p={left_stats['p_value']:.3g}" if pd.notna(left_stats["p_value"]) else "pre/post p=n.a.",
        ha="center",
        va="top",
        fontsize=16,
    )
    ax.text(
        4.5,
        p_text_y,
        f"pre/post p={right_stats['p_value']:.3g}" if pd.notna(right_stats["p_value"]) else "pre/post p=n.a.",
        ha="center",
        va="top",
        fontsize=16,
    )

    vals_nonempty = [v for v in plot_data if len(v) > 0]
    data_max = max(np.max(v) for v in vals_nonempty)
    data_min = min(np.min(v) for v in vals_nonempty)
    data_range = data_max - data_min if data_max > data_min else 1.0

    add_sig_bracket(ax, 1, 2, data_max + 0.04 * data_range, 0.02 * data_range, left_stats["label"], fontsize=20)
    add_sig_bracket(ax, 4, 5, data_max + 0.12 * data_range, 0.02 * data_range, right_stats["label"], fontsize=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(bottom=0.26, top=0.86)

    fig_path = os.path.join(out_dir, f"{safe_group}_Q75_prepost_bc.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    stats_df.to_csv(
        os.path.join(out_dir, f"{safe_group}_Q75_prepost_bc_stats.csv"),
        index=False,
    )

    return fig_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-csv", required=True)
    parser.add_argument("--quantile-selection-csv", required=True)
    parser.add_argument("--selection-column", default="selected_Q75")
    parser.add_argument("--metadata-excel", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    bc = load_bc_table(args.bc_csv)
    sel = load_selection_table(args.quantile_selection_csv, args.selection_column)
    meta = load_metadata(args.metadata_excel)

    bc = collapse_periods_to_pre_post(bc)
    merged = bc.merge(sel, on=["animal_id", "cluster_label"], how="inner")
    merged = merged.merge(meta, on="animal_id", how="left")
    merged["subset"] = np.where(merged["selected_q75"], "high", "low")

    merged.to_csv(
        os.path.join(args.out_dir, "merged_bc_q75_by_group_rows.csv"),
        index=False,
    )

    summary_rows = []
    figure_paths = []

    for group_name in GROUP_ORDER:
        sub = merged[merged["lesion_group"] == group_name].copy()
        if sub.empty:
            continue

        fig_path = draw_group_figure(group_name, sub, args.out_dir)
        figure_paths.append(fig_path)

        summary_rows.append({
            "lesion_group": group_name,
            "n_birds": sub["animal_id"].nunique(),
            "n_clusters": sub[["animal_id", "cluster_label"]].drop_duplicates().shape[0],
            "figure_path": fig_path,
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, "figure_summary_by_group.csv"),
        index=False,
    )

    print("\n[DONE] Wrote lesion-group-specific Q75 BC figures to:")
    print(args.out_dir)
    for p in figure_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
