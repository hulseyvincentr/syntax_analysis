#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL = "Partial Medial and Lateral lesion"
COMPLETE = "Complete Medial and Lateral lesion"

DETAIL_ORDER = [SHAM, LATERAL, PARTIAL, COMPLETE]

COLORS = {
    SHAM: "#1B9E77",
    LATERAL: "#A88BD9",
    PARTIAL: "#7A4FB7",
    COMPLETE: "#3F007D",
}

LABELS = {
    SHAM: "Sham\nsaline",
    LATERAL: "Lateral\nonly",
    PARTIAL: "Partial\nM+L",
    COMPLETE: "Complete\nM+L",
}

SHORT = {
    SHAM: "Sham",
    LATERAL: "Lateral-only",
    PARTIAL: "Partial M+L",
    COMPLETE: "Complete M+L",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.75, 0.90])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    return p.parse_args()

def canonical_detail_group(value):
    s = " ".join(str(value).strip().split()).lower().replace("_", " ")
    if s in {"sham saline injection", "sham", "sham saline", "saline"}:
        return SHAM
    if s in {"lateral lesion only", "lateral only", "lateral-only", "lateral hit only"}:
        return LATERAL
    if "partial" in s and "medial" in s and "lateral" in s:
        return PARTIAL
    if "complete" in s and "medial" in s and "lateral" in s:
        return COMPLETE
    raise ValueError(f"Unrecognized lesion_group_detailed value: {value!r}")

def qlabel(q):
    return f"Q{int(round(100*q))}"

def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=10, width=1.0, length=4)

def build_bird_quantiles(pair_df, quantiles):
    required = {"animal_id", "lesion_group_detailed", "delta_cv"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    work = pair_df.copy()
    work["animal_id"] = work["animal_id"].astype(str).str.strip()
    work["group"] = work["lesion_group_detailed"].map(canonical_detail_group)
    work["delta_cv"] = pd.to_numeric(work["delta_cv"], errors="coerce")
    work = work[np.isfinite(work["delta_cv"])].copy()

    rows = []
    for (bird, group), sub in work.groupby(["animal_id", "group"], sort=True):
        vals = sub["delta_cv"].to_numpy(float)
        row = {
            "animal_id": bird,
            "group": group,
            "group_label": SHORT[group],
            "n_qualifying_syllables": len(vals),
        }
        for q in quantiles:
            row[qlabel(q)] = float(np.quantile(vals, q, method="linear"))
        rows.append(row)
    return pd.DataFrame(rows)

def permutation_p(x, y, alternative, rng, max_exact=200000, mc_reps=200000):
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
        return count / total, "exact", total

    count = 0
    for _ in range(mc_reps):
        perm = rng.permutation(pooled)
        diff = float(np.mean(perm[:nx]) - np.mean(perm[nx:]))
        count += int(extreme(diff))
    return (count + 1) / (mc_reps + 1), "monte_carlo", mc_reps

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        out[idx] = min(running, 1.0)
    return out

def calculate_stats(bird_df, quantiles, seed):
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = [
        (LATERAL, SHAM, "two-sided", "Lateral-only vs sham"),
        (PARTIAL, SHAM, "greater", "Partial M+L vs sham"),
        (COMPLETE, SHAM, "greater", "Complete M+L vs sham"),
    ]
    for q in quantiles:
        col = qlabel(q)
        qrows = []
        for g1, g2, alt, label in comparisons:
            x = bird_df.loc[bird_df["group"] == g1, col].to_numpy(float)
            y = bird_df.loc[bird_df["group"] == g2, col].to_numpy(float)
            p, method, nperm = permutation_p(x, y, alt, rng)
            qrows.append({
                "quantile": q,
                "quantile_label": col,
                "comparison": label,
                "group1": g1,
                "group2": g2,
                "alternative": alt,
                "n_group1": len(x),
                "n_group2": len(y),
                "group1_mean": float(np.mean(x)),
                "group2_mean": float(np.mean(y)),
                "mean_difference": float(np.mean(x) - np.mean(y)),
                "p_raw": float(p),
                "permutation_method": method,
                "permutations_or_assignments": nperm,
            })
        adj = holm_adjust([r["p_raw"] for r in qrows])
        for row, padj in zip(qrows, adj):
            row["p_holm_within_quantile"] = float(padj)
            rows.append(row)
    return pd.DataFrame(rows)

def p_to_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def draw_bracket(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", linewidth=1.0, clip_on=False)
    ax.text((x1+x2)/2, y+h, label, ha="center", va="bottom", fontsize=11)

def plot_one_quantile(bird_df, stats_df, q, out_dir, dpi, seed, show, ax=None):
    col = qlabel(q)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
    else:
        fig = ax.figure

    positions = np.arange(1, len(DETAIL_ORDER)+1)
    values = {g: bird_df.loc[bird_df["group"] == g, col].dropna().to_numpy(float) for g in DETAIL_ORDER}

    for xpos, group in zip(positions, DETAIL_ORDER):
        vals = values[group]
        q5, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        width = 0.54
        ax.add_patch(plt.Rectangle((xpos-width/2, q25), width, q75-q25,
                                   facecolor=COLORS[group], edgecolor=COLORS[group],
                                   linewidth=1.4, alpha=0.22, zorder=1))
        ax.plot([xpos-width/2, xpos+width/2], [q50, q50], color=COLORS[group], linewidth=2.0, zorder=2)
        ax.plot([xpos, xpos], [q5, q25], color=COLORS[group], linewidth=1.4)
        ax.plot([xpos, xpos], [q75, q95], color=COLORS[group], linewidth=1.4)
        cap = 0.18
        ax.plot([xpos-cap, xpos+cap], [q5, q5], color=COLORS[group], linewidth=1.4)
        ax.plot([xpos-cap, xpos+cap], [q95, q95], color=COLORS[group], linewidth=1.4)

    rng = np.random.default_rng(seed + int(round(q*1000)))
    for xpos, group in zip(positions, DETAIL_ORDER):
        vals = values[group]
        jitter = rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(np.full(len(vals), xpos)+jitter, vals, s=42, facecolor=COLORS[group],
                   edgecolor="white", linewidth=0.7, alpha=0.95, zorder=4)

    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[g] for g in DETAIL_ORDER])
    ax.set_ylabel(f"Bird-level {col} of ΔCV\n(Post − Late Pre)")
    apply_style(ax)

    for xpos, group in zip(positions, DETAIL_ORDER):
        ax.text(xpos, -0.13, f"n={len(values[group])}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)

    xpos = {g: i for i, g in enumerate(DETAIL_ORDER, start=1)}
    qstats = stats_df[stats_df["quantile"] == q].copy()
    order = {
        "Lateral-only vs sham": 0,
        "Partial M+L vs sham": 1,
        "Complete M+L vs sham": 2,
    }
    qstats["_order"] = qstats["comparison"].map(order)
    qstats = qstats.sort_values("_order")

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    data_max = max(np.max(v) for v in values.values())
    base = max(data_max, 0) + 0.07*span
    step = 0.10*span
    h = 0.022*span

    for i, (_, row) in enumerate(qstats.iterrows()):
        draw_bracket(ax,
                     min(xpos[row["group1"]], xpos[row["group2"]]),
                     max(xpos[row["group1"]], xpos[row["group2"]]),
                     base + i*step, h,
                     p_to_label(float(row["p_holm_within_quantile"])))

    top_needed = base + (len(qstats)-1)*step + 0.08*span
    ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], top_needed))

    if own_fig:
        fig.tight_layout()
        out = out_dir / f"{col}_deltaCV_separated_ML_vs_sham_only_boxplot_with_ns.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
        if show: plt.show()
        else: plt.close(fig)
    return fig

def main():
    args = parse_args()
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_csv(args.pair_metrics.expanduser().resolve())
    bird_df = build_bird_quantiles(pair_df, args.quantiles)
    bird_df.to_csv(out_dir / "bird_quantiles_by_detailed_hit_type_vs_sham_only.csv", index=False)

    stats_df = calculate_stats(bird_df, args.quantiles, args.seed)
    stats_df.to_csv(out_dir / "pairwise_stats_detailed_groups_vs_sham_only.csv", index=False)

    for q in args.quantiles:
        plot_one_quantile(bird_df, stats_df, q, out_dir, args.dpi, args.seed, args.show)

    if len(args.quantiles) >= 2:
        fig, axes = plt.subplots(1, len(args.quantiles), figsize=(6.2*len(args.quantiles), 5.3))
        axes = np.atleast_1d(axes)
        for ax, q in zip(axes, args.quantiles):
            plot_one_quantile(bird_df, stats_df, q, out_dir, args.dpi, args.seed, False, ax=ax)
            ax.set_title(qlabel(q))
        for i, ax in enumerate(axes):
            ax.text(-0.12, 1.04, chr(ord("A")+i), transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top")
        fig.tight_layout(w_pad=2.0)
        out = out_dir / "Q75_Q90_separated_ML_vs_sham_only_boxplots_with_ns.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
        if args.show: plt.show()
        else: plt.close(fig)

    print("\nBird counts:")
    for g in DETAIL_ORDER:
        print(f"  {SHORT[g]}: n={(bird_df['group'] == g).sum()}")

    print("\nPairwise statistics:")
    print(stats_df[["quantile_label","comparison","alternative","mean_difference","p_raw","p_holm_within_quantile"]].to_string(index=False))
    print(f"\nOutputs: {out_dir}")

if __name__ == "__main__":
    main()
