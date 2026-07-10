#!/usr/bin/env python3
"""
Audit percentile math for phrase-duration prolongation outputs.

Why this exists:
  The paper-style q99 scatter from make_prolongation_paper_style_figures.py uses
  animal_level_prolongation_metrics.csv, where each point is ONE BIRD and q99 is
  the median/mean across that bird's selected syllables. Therefore an individual
  bird x syllable with 25-30 s outliers can be compressed in the animal-level plot.

This script checks the existing q99 calculations and makes syllable-level plots
that should show the long-tail syllables more directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Match the lesion-paper colors used in the Figure 3 style plots.
COLORS = {
    "sham": "#1b9e77",
    "lateral_only": "#b39ddb",
    "medial_lateral": "#8e6bd1",
    "complete_medial_lateral": "#5e1a9a",
    "partial_medial_lateral": "#8e6bd1",
}

GROUP_LABELS = {
    "sham": "sham saline injection",
    "lateral_only": "Lateral lesion only",
    "medial_lateral": "Complete and partial medial and lateral lesion",
    "complete_medial_lateral": "Complete Medial and Lateral lesion",
    "partial_medial_lateral": "Partial Medial and Lateral lesion",
}


def norm_group(x: object) -> str:
    s = str(x).strip().lower()
    if "sham" in s or "saline" in s:
        return "sham"
    if "lateral" in s and ("only" in s or "single" in s) and "medial" not in s:
        return "lateral_only"
    if "complete" in s and ("m+l" in s or "medial" in s or "large lesion" in s):
        return "complete_medial_lateral"
    if "partial" in s and ("m+l" in s or "medial" in s):
        return "partial_medial_lateral"
    if "combined" in s or "m+l" in s or "medial" in s:
        return "medial_lateral"
    return s.replace(" ", "_")


def coerce_syllable_string(df: pd.DataFrame, cols: Iterable[str] = ("bird", "syllable")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(width=1.2, length=5, labelsize=11)


def add_identity(ax, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vals = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if vals.size == 0:
        return
    lo = max(0.0, float(np.nanmin(vals)) * 0.95)
    hi = float(np.nanmax(vals)) * 1.05
    if hi <= lo:
        hi = lo + 1.0
    ax.plot([lo, hi], [lo, hi], "--", color="#d62728", linewidth=1.5, alpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def pre_post_scatter(df: pd.DataFrame, pre_col: str, post_col: str, out: Path, xlabel: str, ylabel: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    handles = []
    labels = []
    order = ["complete_medial_lateral", "partial_medial_lateral", "medial_lateral", "lateral_only", "sham"]
    for group in order:
        g = df[df["plot_group"] == group]
        if g.empty:
            continue
        h = ax.scatter(
            g[pre_col], g[post_col],
            s=34, alpha=0.85,
            color=COLORS.get(group, "0.4"), edgecolor="none",
        )
        handles.append(h)
        labels.append(GROUP_LABELS.get(group, group))
    add_identity(ax, df[pre_col], df[post_col])
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=13)
    style_axes(ax)
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=450)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def infer_col(df: pd.DataFrame, explicit: str | None, candidates: list[str], label: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested {label} '{explicit}' not found. Available columns: {list(df.columns)}")
        return explicit
    norm = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    raise ValueError(f"Could not infer {label}. Available columns: {list(df.columns)}")


def load_selected(selected_csv: Path | None) -> pd.DataFrame | None:
    if selected_csv is None:
        return None
    sel = pd.read_csv(selected_csv)
    sel = coerce_syllable_string(sel)
    return sel[["bird", "syllable"]].drop_duplicates()


def audit_existing_metrics(args) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    syll = pd.read_csv(args.syllable_metrics_csv)
    syll = coerce_syllable_string(syll)
    if "plot_group" not in syll.columns:
        syll["plot_group"] = syll["group"].map(norm_group)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Syllable-level q99 plot: this should reveal long-tail syllables hidden by bird-level aggregation.
    pre_post_scatter(
        syll,
        "pre_q99_duration_s",
        "post_q99_duration_s",
        args.out_dir / "syllable_level_pre_vs_post_q99_duration_s",
        "Late pre-lesion 99th percentile phrase duration (s)",
        "Post-lesion 99th percentile phrase duration (s)",
        title="One point = one selected bird × syllable",
    )

    top = syll.sort_values("post_q99_duration_s", ascending=False).head(args.top_n).copy()
    top.to_csv(args.out_dir / "top_syllable_level_post_q99_values.csv", index=False)

    animal_audit = None
    if args.animal_level_csv is not None and args.animal_level_csv.exists():
        animal = pd.read_csv(args.animal_level_csv)
        animal = coerce_syllable_string(animal, cols=("bird",))
        animal_audit = (
            syll.groupby(["bird", "group"], as_index=False)
            .agg(
                n_selected_syllables=("syllable", "nunique"),
                median_syllable_pre_q99_s=("pre_q99_duration_s", "median"),
                median_syllable_post_q99_s=("post_q99_duration_s", "median"),
                mean_syllable_post_q99_s=("post_q99_duration_s", "mean"),
                max_syllable_post_q99_s=("post_q99_duration_s", "max"),
                max_syllable_delta_q99_s=("delta_q99_duration_s", "max"),
            )
        )
        keep = [
            "bird", "pre_q99_duration_s", "post_q99_duration_s", "delta_q99_duration_s",
            "selected_syllables",
        ]
        keep = [c for c in keep if c in animal.columns]
        animal_audit = animal_audit.merge(
            animal[keep].rename(columns={
                "pre_q99_duration_s": "animal_level_pre_q99_s",
                "post_q99_duration_s": "animal_level_post_q99_s",
                "delta_q99_duration_s": "animal_level_delta_q99_s",
            }),
            on="bird", how="left",
        )
        animal_audit.to_csv(args.out_dir / "animal_level_vs_syllable_level_q99_audit.csv", index=False)

    return syll, animal_audit


def audit_raw_long(args, syll: pd.DataFrame) -> pd.DataFrame | None:
    if args.long_csv is None:
        return None
    long = pd.read_csv(args.long_csv)
    bird_col = infer_col(long, args.bird_col, ["bird", "animal_id", "Animal ID", "bird_id"], "bird column")
    syll_col = infer_col(long, args.syllable_col, ["syllable", "label", "cluster", "hdbscan_label"], "syllable column")
    epoch_col = infer_col(long, args.epoch_col, ["analysis_epoch", "epoch", "Group", "group"], "epoch column")
    dur_col = infer_col(long, args.duration_col, ["Phrase Duration (ms)", "duration_ms", "phrase_duration_ms", "duration_s"], "duration column")
    group_col = infer_col(long, args.group_col, ["lesion_group", "group", "Lesion hit type"], "group column")

    long = long.rename(columns={bird_col: "bird", syll_col: "syllable", epoch_col: "epoch", dur_col: "duration", group_col: "group"})
    long = coerce_syllable_string(long)
    long["duration"] = pd.to_numeric(long["duration"], errors="coerce")
    long = long[np.isfinite(long["duration"])]
    if args.duration_unit == "ms":
        long["duration_s"] = long["duration"] / 1000.0
    else:
        long["duration_s"] = long["duration"]
    long["epoch"] = long["epoch"].astype(str).str.lower()
    pre_vals = {x.strip().lower() for x in args.pre_epoch_values.split(",")}
    post_vals = {x.strip().lower() for x in args.post_epoch_values.split(",")}
    long = long[long["epoch"].isin(pre_vals | post_vals)].copy()
    long["epoch2"] = np.where(long["epoch"].isin(pre_vals), "pre", "post")

    selected = load_selected(args.selected_csv)
    if selected is None:
        selected = syll[["bird", "syllable"]].drop_duplicates()
    long = long.merge(selected.assign(selected=True), on=["bird", "syllable"], how="inner")

    rows = []
    for (bird, group, syllable, epoch), g in long.groupby(["bird", "group", "syllable", "epoch2"], sort=False):
        x = g["duration_s"].dropna().to_numpy(float)
        if len(x) == 0:
            continue
        rows.append({
            "bird": bird,
            "group": norm_group(group),
            "syllable": syllable,
            "epoch": epoch,
            "n_raw": len(x),
            "mean_s": float(np.mean(x)),
            "median_s": float(np.median(x)),
            "q95_s": float(np.quantile(x, 0.95)),
            "q99_s": float(np.quantile(x, 0.99)),
            "q995_s": float(np.quantile(x, 0.995)),
            "q999_s": float(np.quantile(x, 0.999)),
            "max_s": float(np.max(x)),
            "n_ge_10s": int(np.sum(x >= 10.0)),
            "n_ge_20s": int(np.sum(x >= 20.0)),
            "n_ge_30s": int(np.sum(x >= 30.0)),
            "prop_ge_10s": float(np.mean(x >= 10.0)),
            "prop_ge_20s": float(np.mean(x >= 20.0)),
            "prop_ge_30s": float(np.mean(x >= 30.0)),
        })
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("No raw rows found after filtering selected syllables and pre/post epochs.")

    wide = raw.pivot_table(index=["bird", "group", "syllable"], columns="epoch", values=[
        "n_raw", "mean_s", "median_s", "q95_s", "q99_s", "q995_s", "q999_s", "max_s",
        "n_ge_10s", "n_ge_20s", "n_ge_30s", "prop_ge_10s", "prop_ge_20s", "prop_ge_30s",
    ], aggfunc="first")
    wide.columns = [f"{epoch}_{metric}" for metric, epoch in wide.columns]
    wide = wide.reset_index()
    wide["plot_group"] = wide["group"]
    for metric in ["q99_s", "q995_s", "q999_s", "max_s", "prop_ge_10s", "prop_ge_20s", "prop_ge_30s"]:
        a, b = f"pre_{metric}", f"post_{metric}"
        if a in wide.columns and b in wide.columns:
            wide[f"delta_{metric}"] = wide[b] - wide[a]

    # Compare raw q99 to existing syllable metric q99.
    existing = syll[["bird", "syllable", "pre_q99_duration_s", "post_q99_duration_s"]].copy()
    wide = wide.merge(existing, on=["bird", "syllable"], how="left")
    wide["pre_q99_existing_minus_raw_s"] = wide["pre_q99_duration_s"] - wide["pre_q99_s"]
    wide["post_q99_existing_minus_raw_s"] = wide["post_q99_duration_s"] - wide["post_q99_s"]
    wide.to_csv(args.out_dir / "raw_percentile_audit_selected_syllables.csv", index=False)

    pre_post_scatter(
        wide,
        "pre_q99_s",
        "post_q99_s",
        args.out_dir / "syllable_level_raw_pre_vs_post_q99_duration_s",
        "Late pre-lesion raw 99th percentile phrase duration (s)",
        "Post-lesion raw 99th percentile phrase duration (s)",
        title="Raw q99; one point = one selected bird × syllable",
    )
    pre_post_scatter(
        wide,
        "pre_max_s",
        "post_max_s",
        args.out_dir / "syllable_level_raw_pre_vs_post_max_duration_s",
        "Late pre-lesion max phrase duration (s)",
        "Post-lesion max phrase duration (s)",
        title="Raw maximum; one point = one selected bird × syllable",
    )

    return wide


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--syllable-metrics-csv", required=True, type=Path)
    p.add_argument("--animal-level-csv", type=Path, default=None)
    p.add_argument("--selected-csv", type=Path, default=None)
    p.add_argument("--long-csv", type=Path, default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--bird-col", default=None)
    p.add_argument("--syllable-col", default=None)
    p.add_argument("--epoch-col", default=None)
    p.add_argument("--duration-col", default=None)
    p.add_argument("--duration-unit", choices=["ms", "s"], default="ms")
    p.add_argument("--group-col", default=None)
    p.add_argument("--pre-epoch-values", default="pre")
    p.add_argument("--post-epoch-values", default="post")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    syll, animal_audit = audit_existing_metrics(args)
    raw = audit_raw_long(args, syll)

    print("[DONE] Wrote audit outputs to:", args.out_dir)
    print("\n[CHECK] Top syllable-level post q99 values from existing metrics:")
    cols = ["bird", "group", "syllable", "pre_q99_duration_s", "post_q99_duration_s", "delta_q99_duration_s", "n_post_raw"]
    cols = [c for c in cols if c in syll.columns]
    print(syll.sort_values("post_q99_duration_s", ascending=False).head(args.top_n)[cols].to_string(index=False))
    if animal_audit is not None:
        print("\n[CHECK] Animal-level aggregation audit, top max syllable q99 values:")
        print(animal_audit.sort_values("max_syllable_post_q99_s", ascending=False).head(args.top_n).to_string(index=False))
    if raw is not None:
        print("\n[CHECK] Raw selected-syllable max duration values:")
        print(raw.sort_values("post_max_s", ascending=False).head(args.top_n)[["bird", "group", "syllable", "post_q99_s", "post_q995_s", "post_q999_s", "post_max_s", "post_n_raw"]].to_string(index=False))


if __name__ == "__main__":
    main()
