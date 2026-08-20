#!/usr/bin/env python3
"""
plot_quantile_threshold_scatter_deltaCV_timecourse.py

For each within-bird delta-CV quantile threshold (e.g. Q0, Q25, Q50, Q75, Q90), generate:
1) scatterplot of pre_cv vs post_cv, each point = one syllable
2) bird-level delta-CV boxplot by lesion group
3) equal-bird rolling-median longitudinal CV plots using only syllables at or above the threshold\n   - one with sham, lateral-only, and pooled M+L\n   - one with sham and pooled M+L only

Selection rule:
    for each bird, keep syllables with delta_cv >= bird-specific QXX

Default lesion grouping:
    sham, lateral-only, pooled medial+lateral

Scatterplot axis limits:\n    xlim = min/max pre_cv of points actually displayed\n    ylim = min/max post_cv of points actually displayed\n\nFigure 3 palette:
    sham         #1B9E77
    lateral-only #A88BD9
    pooled M+L   #7A4FB7
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL = "Partial Medial and Lateral lesion"
COMPLETE = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"

GROUP_ORDER = [SHAM, LATERAL, POOLED_ML]

COLORS = {
    SHAM: "#1B9E77",
    LATERAL: "#A88BD9",
    POOLED_ML: "#7A4FB7",
}

GROUP_LABELS = {
    SHAM: "Sham",
    LATERAL: "Lateral-only",
    POOLED_ML: "Medial+lateral",
}

BOX_LABELS = {
    SHAM: "Sham\nsaline",
    LATERAL: "Lateral\nonly",
    POOLED_ML: "Medial + lateral",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-metrics", type=Path, required=True)
    p.add_argument("--duration-long-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.50, 0.75, 0.90])
    p.add_argument("--late-pre-start", type=int, default=-14)
    p.add_argument("--late-pre-end", type=int, default=-1)
    p.add_argument("--x-min", type=int, default=-30)
    p.add_argument("--x-max", type=int, default=30)
    p.add_argument("--rolling-window-days", type=int, default=7)
    p.add_argument("--min-daily-phrases", type=int, default=5)
    p.add_argument("--min-birds-for-summary", type=int, default=2)
    p.add_argument("--normalize-mode", choices=["subtract", "ratio"], default="subtract")
    p.add_argument(
        "--timecourse-x-buffer",
        type=float,
        default=0.25,
        help="Extra padding added to the left and right x-limits of longitudinal plots. Default: 0.25",
    )
    p.add_argument(
        "--timecourse-y-buffer",
        type=float,
        default=0.25,
        help="Extra padding added below and above the data range of longitudinal plots. Default: 0.25",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--hide-nonselected",
        action="store_true",
        help=(
            "Do not plot nonselected syllables as grey background points in "
            "the scatterplots. Useful when including Q0 as the all-syllable panel."
        ),
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def choose_first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Could not find any of the expected columns:\n" + "\n".join(f"  - {x}" for x in candidates))


def canonical_detail_group(value: object) -> str:
    s = " ".join(str(value).strip().split()).lower().replace("_", " ")
    if s in {"sham saline injection", "sham", "sham saline", "saline"}:
        return SHAM
    if s in {"lateral lesion only", "lateral only", "lateral-only", "lateral hit only"}:
        return LATERAL
    if "partial" in s and "medial" in s and "lateral" in s:
        return PARTIAL
    if "complete" in s and "medial" in s and "lateral" in s:
        return COMPLETE
    raise ValueError(f"Unrecognized lesion group: {value!r}")


def pooled_group(detail_group: str) -> str:
    return POOLED_ML if detail_group in {PARTIAL, COMPLETE} else detail_group


def qlabel(q: float) -> str:
    return f"Q{int(round(100*q))}"


def apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=10, width=1.0, length=4)


def save_figure(fig: plt.Figure, path: Path, dpi: int, show: bool):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def load_pair_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"animal_id", "syllable", "lesion_group_detailed", "pre_cv", "post_cv", "delta_cv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns from pair-metrics CSV: {sorted(missing)}")

    df = df.copy()
    df["animal_id"] = df["animal_id"].astype(str).str.strip()
    df["syllable"] = df["syllable"].astype(str).str.strip()
    df["detail_group"] = df["lesion_group_detailed"].map(canonical_detail_group)
    df["group"] = df["detail_group"].map(pooled_group)

    for col in ["pre_cv", "post_cv", "delta_cv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[np.isfinite(df["pre_cv"]) & np.isfinite(df["post_cv"]) & np.isfinite(df["delta_cv"])].copy()


def build_quantile_selection(pair_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    for bird, sub in pair_df.groupby("animal_id", sort=True):
        vals = sub["delta_cv"].to_numpy(float)
        thresholds = {q: np.quantile(vals, q, method="linear") for q in quantiles}
        tmp = sub.copy()
        for q in quantiles:
            tmp[f"selected_{qlabel(q)}"] = tmp["delta_cv"] >= thresholds[q]
            tmp[f"threshold_{qlabel(q)}"] = thresholds[q]
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def plot_scatter_for_quantile(sel_df: pd.DataFrame, q: float, out_dir: Path, dpi: int, show: bool, hide_nonselected: bool = False):
    qname = qlabel(q)
    flag = f"selected_{qname}"
    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    if not hide_nonselected:
        ax.scatter(
            sel_df["pre_cv"], sel_df["post_cv"],
            s=18, facecolor="#CFCFCF", edgecolor="none",
            alpha=0.45, label="All qualifying syllables", zorder=1
        )

    for group in GROUP_ORDER:
        sub = sel_df[(sel_df["group"] == group) & (sel_df[flag])]
        ax.scatter(
            sub["pre_cv"], sub["post_cv"],
            s=28, facecolor=COLORS[group], edgecolor="white",
            linewidth=0.4, alpha=0.95,
            label=f"{GROUP_LABELS[group]} selected", zorder=2
        )

    # Set each axis to the exact range of the points actually plotted.
    #
    # If --hide-nonselected is used, only selected syllables determine the
    # limits. Otherwise, the grey background contains all qualifying
    # syllables, so all qualifying syllables determine the limits.
    if hide_nonselected:
        plotted_df = sel_df[sel_df[flag]].copy()
    else:
        plotted_df = sel_df.copy()

    if plotted_df.empty:
        raise ValueError(
            f"No syllables available to plot for {qname}; cannot set axis limits."
        )

    x_min = float(np.nanmin(plotted_df["pre_cv"].to_numpy(float)))
    x_max = float(np.nanmax(plotted_df["pre_cv"].to_numpy(float)))
    y_min = float(np.nanmin(plotted_df["post_cv"].to_numpy(float)))
    y_max = float(np.nanmax(plotted_df["post_cv"].to_numpy(float)))

    # Guard against a degenerate axis if all displayed values are identical.
    if x_min == x_max:
        eps = max(abs(x_min) * 1e-6, 1e-6)
        x_min -= eps
        x_max += eps
    if y_min == y_max:
        eps = max(abs(y_min) * 1e-6, 1e-6)
        y_min -= eps
        y_max += eps

    # Identity line. It is drawn across the union of the x/y ranges and then
    # clipped automatically to the visible axes.
    identity_min = min(x_min, y_min)
    identity_max = max(x_max, y_max)
    ax.plot(
        [identity_min, identity_max],
        [identity_min, identity_max],
        linestyle="--",
        color="#666666",
        linewidth=1.0,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Late pre phrase-duration CV")
    ax.set_ylabel("Post phrase-duration CV")
    ax.set_title(f"{qname} threshold: selected syllables highlighted")
    apply_style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    save_figure(fig, out_dir / f"{qname}_scatter_preCV_vs_postCV_selected_syllables.png", dpi, show)


def summarize_bird_delta_cv(sel_df: pd.DataFrame, q: float) -> pd.DataFrame:
    qname = qlabel(q)
    flag = f"selected_{qname}"
    rows = []
    for (bird, group), sub in sel_df.groupby(["animal_id", "group"], sort=True):
        picked = sub[sub[flag]].copy()
        if picked.empty:
            continue
        rows.append({
            "animal_id": bird,
            "group": group,
            "n_selected_syllables": len(picked),
            "bird_median_delta_cv": float(np.median(picked["delta_cv"])),
        })
    return pd.DataFrame(rows)


def permutation_p(x: np.ndarray, y: np.ndarray, alternative: str, rng: np.random.Generator, max_exact: int = 200000, mc_reps: int = 200000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    nx = len(x)
    total = math.comb(len(pooled), nx)

    def extreme(v):
        if alternative == "greater":
            return v >= observed - 1e-12
        if alternative == "two-sided":
            return abs(v) >= abs(observed) - 1e-12
        raise ValueError(alternative)

    if total <= max_exact:
        count = 0
        idx = np.arange(len(pooled))
        for combo in itertools.combinations(idx, nx):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(combo)] = True
            diff = float(np.mean(pooled[mask]) - np.mean(pooled[~mask]))
            count += int(extreme(diff))
        return count / total

    count = 0
    for _ in range(mc_reps):
        perm = rng.permutation(pooled)
        diff = float(np.mean(perm[:nx]) - np.mean(perm[nx:]))
        count += int(extreme(diff))
    return (count + 1) / (mc_reps + 1)


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        out[idx] = min(running, 1.0)
    return out


def compute_boxplot_stats(bird_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = [
        (POOLED_ML, SHAM, "greater", "M+L vs sham"),
        (POOLED_ML, LATERAL, "greater", "M+L vs lateral-only"),
        (LATERAL, SHAM, "two-sided", "Lateral-only vs sham"),
    ]

    pvals = []
    for g1, g2, alt, label in comparisons:
        x = bird_df.loc[bird_df["group"] == g1, "bird_median_delta_cv"].to_numpy(float)
        y = bird_df.loc[bird_df["group"] == g2, "bird_median_delta_cv"].to_numpy(float)
        p = permutation_p(x, y, alt, rng)
        pvals.append(p)
        rows.append({
            "comparison": label,
            "group1": g1,
            "group2": g2,
            "alternative": alt,
            "mean_difference": float(np.mean(x) - np.mean(y)),
            "p_raw": float(p),
        })

    padj = holm_adjust(pvals)
    for row, adj in zip(rows, padj):
        row["p_holm"] = float(adj)

    return pd.DataFrame(rows)


def p_to_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def draw_bracket(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", linewidth=1.0, clip_on=False)
    ax.text((x1+x2)/2, y+h, label, ha="center", va="bottom", fontsize=10)


def plot_delta_cv_boxplot(bird_df: pd.DataFrame, stat_df: pd.DataFrame, q: float, out_dir: Path, dpi: int, show: bool):
    qname = qlabel(q)
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    positions = np.arange(1, 4)
    rng = np.random.default_rng(123 + int(round(q * 1000)))

    values = {
        g: bird_df.loc[bird_df["group"] == g, "bird_median_delta_cv"].dropna().to_numpy(float)
        for g in GROUP_ORDER
    }

    for xpos, g in zip(positions, GROUP_ORDER):
        vals = values[g]
        q5, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        width = 0.56

        ax.add_patch(
            plt.Rectangle(
                (xpos - width/2, q25), width, q75-q25,
                facecolor=COLORS[g], edgecolor=COLORS[g],
                linewidth=1.4, alpha=0.24, zorder=1
            )
        )
        ax.plot([xpos-width/2, xpos+width/2], [q50, q50], color=COLORS[g], linewidth=2.0)
        ax.plot([xpos, xpos], [q5, q25], color=COLORS[g], linewidth=1.4)
        ax.plot([xpos, xpos], [q75, q95], color=COLORS[g], linewidth=1.4)
        cap = 0.18
        ax.plot([xpos-cap, xpos+cap], [q5, q5], color=COLORS[g], linewidth=1.4)
        ax.plot([xpos-cap, xpos+cap], [q95, q95], color=COLORS[g], linewidth=1.4)

    for xpos, g in zip(positions, GROUP_ORDER):
        vals = values[g]
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(
            np.full(len(vals), xpos) + jitter, vals,
            s=44, facecolor=COLORS[g], edgecolor="white",
            linewidth=0.6, alpha=0.98, zorder=4
        )
        ax.text(
            xpos, -0.13, f"n={len(vals)}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9
        )

    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([BOX_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel(f"Bird median ΔCV\n(selected syllables, {qname}+)")
    ax.set_title(f"{qname} threshold")
    apply_style(ax)

    xpos = {g: i for i, g in enumerate(GROUP_ORDER, start=1)}
    stat_df = stat_df.copy()
    stat_df["_order"] = stat_df["comparison"].map({
        "Lateral-only vs sham": 0,
        "M+L vs lateral-only": 1,
        "M+L vs sham": 2,
    })
    stat_df = stat_df.sort_values("_order")

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    data_max = max(np.max(v) for v in values.values())
    base = max(data_max, 0) + 0.07 * span
    step = 0.10 * span
    h = 0.022 * span

    for i, (_, row) in enumerate(stat_df.iterrows()):
        draw_bracket(
            ax,
            min(xpos[row["group1"]], xpos[row["group2"]]),
            max(xpos[row["group1"]], xpos[row["group2"]]),
            base + i * step,
            h,
            p_to_label(float(row["p_holm"])),
        )

    top_needed = base + (len(stat_df)-1) * step + 0.08 * span
    ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], top_needed))

    save_figure(fig, out_dir / f"{qname}_deltaCV_boxplot_by_group.png", dpi, show)


def load_long_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    animal_col = choose_first_existing(df, ["animal_id", "bird", "bird_id", "Animal ID"])
    syll_col = choose_first_existing(df, ["syllable", "syllable_id", "cluster", "label"])
    day_col = choose_first_existing(df, ["relative_day", "day", "rel_day"])
    dur_col = choose_first_existing(df, ["phrase_duration_s", "duration_s", "phrase_duration", "duration"])

    out = df.copy().rename(columns={
        animal_col: "animal_id",
        syll_col: "syllable",
        day_col: "relative_day",
        dur_col: "phrase_duration_s",
    })

    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out["syllable"] = out["syllable"].astype(str).str.strip()
    out["relative_day"] = pd.to_numeric(out["relative_day"], errors="coerce")
    out["phrase_duration_s"] = pd.to_numeric(out["phrase_duration_s"], errors="coerce")

    return out[np.isfinite(out["relative_day"]) & np.isfinite(out["phrase_duration_s"])].copy()


def build_daily_bird_cv(long_df: pd.DataFrame, selected_pairs: pd.DataFrame, min_daily_phrases: int) -> pd.DataFrame:
    keep = selected_pairs[["animal_id", "syllable", "group"]].drop_duplicates()
    merged = long_df.merge(keep, on=["animal_id", "syllable"], how="inner")

    daily_syll = (
        merged.groupby(["group", "animal_id", "relative_day", "syllable"])["phrase_duration_s"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    daily_syll = daily_syll[daily_syll["count"] >= min_daily_phrases].copy()
    daily_syll["cv"] = daily_syll["std"] / daily_syll["mean"]
    daily_syll = daily_syll[np.isfinite(daily_syll["cv"])].copy()

    return (
        daily_syll.groupby(["group", "animal_id", "relative_day"])["cv"]
        .median()
        .reset_index()
        .rename(columns={"cv": "bird_day_cv"})
    )


def normalize_to_late_pre(bird_day: pd.DataFrame, late_pre_start: int, late_pre_end: int, mode: str) -> pd.DataFrame:
    base = (
        bird_day[
            (bird_day["relative_day"] >= late_pre_start)
            & (bird_day["relative_day"] <= late_pre_end)
        ]
        .groupby("animal_id")["bird_day_cv"]
        .median()
        .rename("late_pre_baseline")
        .reset_index()
    )

    out = bird_day.merge(base, on="animal_id", how="left")
    out = out[np.isfinite(out["late_pre_baseline"])].copy()

    if mode == "subtract":
        out["cv_norm"] = out["bird_day_cv"] - out["late_pre_baseline"]
    else:
        out["cv_norm"] = out["bird_day_cv"] / out["late_pre_baseline"]

    return out


def smooth_equal_bird(norm_df: pd.DataFrame, x_min: int, x_max: int, rolling_window_days: int) -> pd.DataFrame:
    all_days = np.arange(x_min, x_max + 1)
    pieces = []

    for (group, bird), sub in norm_df.groupby(["group", "animal_id"], sort=True):
        tmp = (
            sub[["relative_day", "cv_norm"]]
            .drop_duplicates(subset=["relative_day"])
            .set_index("relative_day")
            .sort_index()
        )
        tmp = tmp.reindex(all_days)
        tmp.index.name = "relative_day"
        tmp["group"] = group
        tmp["animal_id"] = bird
        tmp["cv_norm_smooth"] = (
            tmp["cv_norm"]
            .rolling(window=rolling_window_days, center=True, min_periods=1)
            .median()
        )
        pieces.append(tmp.reset_index())

    return pd.concat(pieces, ignore_index=True)


def summarize_group_timecourse(smoothed_df: pd.DataFrame, min_birds_for_summary: int) -> pd.DataFrame:
    grp = (
        smoothed_df.groupby(["group", "relative_day"])["cv_norm_smooth"]
        .agg(
            n_birds=lambda x: x.notna().sum(),
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    return grp[grp["n_birds"] >= min_birds_for_summary].copy()


def plot_longitudinal(
    time_summary: pd.DataFrame,
    q: float,
    out_dir: Path,
    x_min: int,
    x_max: int,
    normalize_mode: str,
    dpi: int,
    show: bool,
    groups_to_plot=None,
    filename_suffix="",
    title_suffix="",
    x_buffer: float = 0.25,
    y_buffer: float = 0.25,
):
    qname = qlabel(q)

    if groups_to_plot is None:
        groups_to_plot = GROUP_ORDER

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for g in groups_to_plot:
        sub = time_summary[
            time_summary["group"] == g
        ].sort_values("relative_day")

        if sub.empty:
            continue

        ax.plot(
            sub["relative_day"],
            sub["median"],
            color=COLORS[g],
            linewidth=2.0,
            label=GROUP_LABELS[g],
        )

        ax.fill_between(
            sub["relative_day"],
            sub["q25"],
            sub["q75"],
            color=COLORS[g],
            alpha=0.18,
            linewidth=0,
        )

    ax.axvline(
        0,
        color="#D62728",
        linestyle="--",
        linewidth=1.0,
    )

    ax.axhline(
        0 if normalize_mode == "subtract" else 1,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
    )

    plotted = time_summary[time_summary["group"].isin(groups_to_plot)].copy()
    if plotted.empty:
        raise ValueError("No timecourse data available for the requested groups.")

    y_candidates = np.concatenate([
        plotted["q25"].to_numpy(float),
        plotted["q75"].to_numpy(float),
        plotted["median"].to_numpy(float),
    ])
    y_candidates = y_candidates[np.isfinite(y_candidates)]

    if len(y_candidates) == 0:
        raise ValueError("No finite y-values available for longitudinal plotting.")

    y_lower = float(np.nanmin(y_candidates)) - y_buffer
    y_upper = float(np.nanmax(y_candidates)) + y_buffer

    reference_y = 0 if normalize_mode == "subtract" else 1
    y_lower = min(y_lower, reference_y - 0.02)
    y_upper = max(y_upper, reference_y + 0.02)

    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlabel("Days relative to lesion")

    if normalize_mode == "subtract":
        ax.set_ylabel(
            "Baseline-normalized daily phrase-duration CV\n"
            "(selected syllables)"
        )
    else:
        ax.set_ylabel(
            "Daily phrase-duration CV / late-pre baseline\n"
            "(selected syllables)"
        )

    title = f"{qname} threshold"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)

    apply_style(ax)
    ax.legend(frameon=False, loc="upper left")

    outfile = (
        out_dir
        / f"{qname}_longitudinal_equal_bird_timecourse{filename_suffix}.png"
    )

    save_figure(
        fig,
        outfile,
        dpi,
        show,
    )



def main():
    args = parse_args()
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_df = load_pair_metrics(args.pair_metrics.expanduser().resolve())
    sel_df = build_quantile_selection(pair_df, args.quantiles)
    sel_df.to_csv(out_dir / "syllable_level_quantile_selection.csv", index=False)

    long_df = load_long_csv(args.duration_long_csv.expanduser().resolve())

    all_bird_summaries = []
    all_box_stats = []

    for q in args.quantiles:
        qname = qlabel(q)
        flag = f"selected_{qname}"

        plot_scatter_for_quantile(
            sel_df,
            q,
            out_dir,
            args.dpi,
            args.show,
            hide_nonselected=args.hide_nonselected,
        )

        bird_df = summarize_bird_delta_cv(sel_df, q)
        bird_df["quantile_label"] = qname
        all_bird_summaries.append(bird_df)

        stat_df = compute_boxplot_stats(
            bird_df,
            seed=args.seed + int(round(q * 1000)),
        )
        stat_df["quantile_label"] = qname
        all_box_stats.append(stat_df)

        plot_delta_cv_boxplot(
            bird_df, stat_df, q,
            out_dir, args.dpi, args.show
        )

        selected_pairs = (
            sel_df[sel_df[flag]][["animal_id", "syllable", "group"]]
            .drop_duplicates()
        )

        bird_day = build_daily_bird_cv(
            long_df, selected_pairs, args.min_daily_phrases
        )

        norm = normalize_to_late_pre(
            bird_day,
            late_pre_start=args.late_pre_start,
            late_pre_end=args.late_pre_end,
            mode=args.normalize_mode,
        )

        smoothed = smooth_equal_bird(
            norm,
            x_min=args.x_min,
            x_max=args.x_max,
            rolling_window_days=args.rolling_window_days,
        )

        summary = summarize_group_timecourse(
            smoothed,
            min_birds_for_summary=args.min_birds_for_summary,
        )

        summary.to_csv(
            out_dir / f"{qname}_timecourse_summary.csv",
            index=False,
        )

        # Full three-group longitudinal plot:
        # sham, lateral-only, and pooled medial+lateral.
        plot_longitudinal(
            summary,
            q,
            out_dir,
            x_min=args.x_min,
            x_max=args.x_max,
            normalize_mode=args.normalize_mode,
            dpi=args.dpi,
            show=args.show,
            x_buffer=args.timecourse_x_buffer,
            y_buffer=args.timecourse_y_buffer,
        )

        # Second longitudinal visualization using the exact same
        # bird/day summaries, but displaying only sham and pooled M+L.
        # Lateral-only birds are not plotted in this version.
        plot_longitudinal(
            summary,
            q,
            out_dir,
            x_min=args.x_min,
            x_max=args.x_max,
            normalize_mode=args.normalize_mode,
            dpi=args.dpi,
            show=args.show,
            groups_to_plot=[SHAM, POOLED_ML],
            filename_suffix="_sham_vs_ML_only",
            title_suffix="Sham vs medial+lateral",
            x_buffer=args.timecourse_x_buffer,
            y_buffer=args.timecourse_y_buffer,
        )

    pd.concat(all_bird_summaries, ignore_index=True).to_csv(
        out_dir / "bird_level_delta_cv_by_quantile_threshold.csv",
        index=False,
    )

    pd.concat(all_box_stats, ignore_index=True).to_csv(
        out_dir / "bird_level_delta_cv_group_stats_by_quantile_threshold.csv",
        index=False,
    )

    print("\nDone.")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
