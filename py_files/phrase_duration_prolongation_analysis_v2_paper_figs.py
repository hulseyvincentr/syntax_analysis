#!/usr/bin/env python3
"""
phrase_duration_prolongation_analysis.py

Parallel analysis for the AFP lesion paper: test whether selected syllables are
PROLONGED post-lesion, rather than only more variable.

This script keeps the same experimental-unit logic as the phrase-duration SD
analysis:
  - phrase renditions are the atomic observations
  - syllables are selected once, then held fixed
  - syllables are aggregated within bird
  - birds are the independent units for permutation tests

It computes several prolongation-oriented metrics for each bird x syllable:
  1. median phrase duration: central shift
  2. mean phrase duration: central shift sensitive to long tails
  3. 90th/95th/99th percentile phrase duration: upper-tail prolongation
  4. proportion of post renditions exceeding a late-pre threshold:
       - late-pre 95th percentile
       - late-pre 99th percentile
       - late-pre mean + 2 SD

Then it aggregates selected syllables to one value per bird and runs:
  - within-group sign-flip permutation tests for post > late-pre
  - between-group label-shuffle tests for medial+lateral > lateral-only/sham
  - bird-level bootstrap confidence intervals

Example:
python phrase_duration_prolongation_analysis.py \
  --long-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/<YOUR_LONG_PHRASE_ROWS>.csv" \
  --metadata-excel "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/prolongation_pooled_top30" \
  --selection-basis pooled_var \
  --top-percent 30 \
  --n-balance 200 \
  --n-perm 10000 \
  --n-boot 5000 \
  --alternative greater

If you already exported a CSV containing the exact selected animal x syllable
pairs from Figure 3, pass it with --selection-csv. In that case the script will
not recompute selection.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend so the script works on remote/terminal runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------- paper figure style ----------------------------

# Colors approximate the Figure 3 palette in the AFP lesion manuscript.
PAPER_COLORS = {
    "complete_medial_lateral": "#4B0082",   # dark purple
    "partial_medial_lateral": "#7E57C2",    # medium purple
    "medial_lateral": "#8E6CCB",            # pooled ML purple
    "lateral_only": "#B39DDB",              # light purple
    "sham": "#1B9E77",                      # teal/green
}

PAPER_SCATTER_ORDER = [
    "complete_medial_lateral",
    "partial_medial_lateral",
    "medial_lateral",
    "lateral_only",
    "sham",
]

PAPER_SCATTER_LABELS = {
    "complete_medial_lateral": "Complete Medial and Lateral lesion",
    "partial_medial_lateral": "Partial Medial and Lateral lesion",
    "medial_lateral": "Complete and partial medial and lateral lesion",
    "lateral_only": "Lateral lesion only",
    "sham": "sham saline injection",
}

PAPER_GROUP_ORDER = ["sham", "lateral_only", "medial_lateral"]
PAPER_GROUP_LABELS = {
    "sham": "sham saline\ninjection",
    "lateral_only": "Lateral lesion\nonly",
    "medial_lateral": "Complete and partial\nmedial and lateral lesion",
}

PAPER_GROUP_FULL_LABELS = {
    "sham": "sham saline injection",
    "lateral_only": "Lateral lesion only",
    "medial_lateral": "Complete and partial medial and lateral lesion",
}

PAPER_METRIC_LABELS = {
    "mean_duration_s": "mean phrase duration (s)",
    "median_duration_s": "median phrase duration (s)",
    "q90_duration_s": "90th percentile phrase duration (s)",
    "q95_duration_s": "95th percentile phrase duration (s)",
    "q99_duration_s": "99th percentile phrase duration (s)",
    "prop_above_pre95": "proportion above late-pre 95th percentile",
    "prop_above_pre99": "proportion above late-pre 99th percentile",
    "prop_above_pre_mean2sd": "proportion above late-pre mean + 2 SD",
}

PAPER_FOCUS_BASES = ["q99_duration_s", "prop_above_pre99"]


def apply_paper_style(ax: plt.Axes) -> None:
    """Match the simple, clean matplotlib style used in the paper figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def p_to_label(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def pretty_metric_label(base: str) -> str:
    return PAPER_METRIC_LABELS.get(base, base.replace("_", " "))


def normalize_plot_group_value(x: object) -> str:
    """Map raw lesion labels onto plotting subgroups for Figure-3-style colors.

    This is intentionally finer than the statistical group. Complete and partial
    medial+lateral lesions remain one statistical group (medial_lateral), but are
    drawn with different purples when metadata distinguishes them.
    """
    v = _norm_value(x)
    compact = v.replace(" ", "")

    if "sham" in v or "saline" in v:
        return "sham"
    if "single hit" in v or "lateral only" in v or "lateral-only" in v:
        return "lateral_only"
    if "area x not visible" in v or "largelesion" in compact or "large lesion" in v or "complete" in v:
        return "complete_medial_lateral"
    if ("medial" in v and "lateral" in v) or "m+l" in compact or "medial+lateral" in v:
        return "partial_medial_lateral"
    # If only pooled group labels are available, use pooled plotting groups.
    g = normalize_group_value(x)
    if g in {"sham", "lateral_only", "medial_lateral"}:
        return g
    return "unknown"


# ----------------------------- column utilities -----------------------------

def _norm_name(s: object) -> str:
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum())


def _norm_value(s: object) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def find_col(
    df: pd.DataFrame,
    explicit: Optional[str],
    candidates: Sequence[str],
    required: bool = True,
    label: str = "column",
) -> Optional[str]:
    """Find a column by explicit name or by flexible candidate matching."""
    if explicit:
        if explicit in df.columns:
            return explicit
        raise ValueError(f"Requested {label} '{explicit}' was not found. Available columns: {list(df.columns)}")

    norm_to_actual = {_norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_name(cand)
        if key in norm_to_actual:
            return norm_to_actual[key]

    # Fallback: candidate appears inside a column name.
    for cand in candidates:
        key = _norm_name(cand)
        for col in df.columns:
            if key and key in _norm_name(col):
                return col

    if required:
        raise ValueError(
            f"Could not detect {label}. Tried candidates {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def split_values_csv(s: str) -> List[str]:
    return [_norm_value(x) for x in s.split(",") if str(x).strip()]


# ----------------------------- group utilities ------------------------------

def normalize_group_value(x: object) -> str:
    """Map common lesion/hit-type labels onto compact group names."""
    v = _norm_value(x)
    compact = v.replace(" ", "")

    if v in {"medial lateral", "medial+lateral", "medial and lateral", "ml", "partial medial+lateral", "complete medial+lateral"}:
        return "medial_lateral"
    if v in {"lateral only", "lateral-only", "lateral", "lo"}:
        return "lateral_only"
    if "sham" in v or "saline" in v:
        return "sham"

    # Common phrases from the current lesion metadata/supplement.
    if "medial" in v and "lateral" in v:
        return "medial_lateral"
    if "area x not visible" in v or "largelesion" in compact or "complete" in v:
        # In this dataset, complete Area X-not-visible lesions are ML lesions.
        return "medial_lateral"
    if "single hit" in v or ("visible" in v and "medial" not in v) or "lateral" in v:
        return "lateral_only"
    if "miss" in v:
        return "miss"

    return compact if compact else "unknown"


def load_metadata(
    metadata_excel: Path,
    metadata_sheet: Optional[str],
    metadata_bird_col: Optional[str],
    metadata_group_col: Optional[str],
) -> pd.DataFrame:
    if metadata_sheet:
        meta = pd.read_excel(metadata_excel, sheet_name=metadata_sheet)
    else:
        sheets = pd.read_excel(metadata_excel, sheet_name=None)
        # Prefer sheets that look like metadata or animal summaries.
        preferred = []
        for name, df in sheets.items():
            lower = str(name).lower()
            score = 0
            if "metadata" in lower:
                score += 2
            if "animal" in lower or "summary" in lower:
                score += 1
            if any(_norm_name(c) in {_norm_name(x) for x in ["Animal ID", "Bird ID", "bird", "animal"]} for c in df.columns):
                score += 2
            preferred.append((score, name, df))
        preferred.sort(reverse=True, key=lambda x: x[0])
        meta = preferred[0][2]

    bird_col = find_col(
        meta,
        metadata_bird_col,
        ["Animal ID", "Bird ID", "bird", "animal", "animal_id", "BirdID"],
        label="metadata bird column",
    )
    group_col = find_col(
        meta,
        metadata_group_col,
        [
            "Lesion group",
            "lesion_group",
            "lesion hit type grouping",
            "Lesion hit type",
            "hit type",
            "Treatment type",
            "Medial/Lateral Area X hit?",
            "Area X visible",
        ],
        label="metadata group column",
    )
    out = meta[[bird_col, group_col]].copy()
    out.columns = ["bird", "group_raw"]
    out["bird"] = out["bird"].astype(str)
    out["group"] = out["group_raw"].map(normalize_group_value)
    out["plot_group"] = out["group_raw"].map(normalize_plot_group_value)
    out = out.drop_duplicates(subset=["bird"])
    return out[["bird", "group", "group_raw", "plot_group"]]


# ---------------------------- input preparation -----------------------------

def load_long_table(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.long_csv)

    bird_col = find_col(
        df, args.bird_col,
        ["bird", "animal", "Animal ID", "Bird ID", "animal_id", "bird_id"],
        label="bird column",
    )
    syll_col = find_col(
        df, args.syllable_col,
        ["syllable", "syllable_label", "label", "cluster", "hdbscan_label", "hdbscan_labels"],
        label="syllable column",
    )
    epoch_col = find_col(
        df, args.epoch_col,
        ["epoch", "period", "group", "Group", "recording_period", "pre_post", "lesion_epoch"],
        label="epoch column",
    )
    dur_col = find_col(
        df, args.duration_col,
        [
            "duration_ms", "phrase_duration_ms", "duration", "duration_s",
            "phrase_duration_s", "phrase_duration", "phrase duration", "phrase_duration_seconds",
        ],
        label="duration column",
    )
    group_col = find_col(
        df, args.group_col,
        ["lesion_group", "Lesion group", "group", "lesion hit type", "Lesion hit type", "hit_type"],
        required=False,
        label="lesion group column",
    )

    out = pd.DataFrame({
        "bird": df[bird_col].astype(str),
        "syllable": df[syll_col].astype(str),
        "epoch_raw": df[epoch_col],
        "duration_raw": pd.to_numeric(df[dur_col], errors="coerce"),
    })
    if group_col is not None and group_col != epoch_col:
        out["group_raw"] = df[group_col]
        out["group"] = df[group_col].map(normalize_group_value)
        out["plot_group"] = df[group_col].map(normalize_plot_group_value)
    else:
        out["group_raw"] = np.nan
        out["group"] = np.nan
        out["plot_group"] = np.nan

    pre_values = split_values_csv(args.pre_epoch_values)
    post_values = split_values_csv(args.post_epoch_values)

    def map_epoch(x: object) -> Optional[str]:
        v = _norm_value(x)
        if v in pre_values:
            return "pre"
        if v in post_values:
            return "post"
        return None

    out["epoch"] = out["epoch_raw"].map(map_epoch)
    before = len(out)
    out = out[out["epoch"].isin(["pre", "post"])].copy()
    if out.empty:
        raise ValueError(
            "No rows matched the requested pre/post epoch values. "
            f"Unique epoch values seen: {sorted(pd.Series(df[epoch_col]).dropna().astype(str).unique())[:50]}\n"
            f"Current --pre-epoch-values: {args.pre_epoch_values}\n"
            f"Current --post-epoch-values: {args.post_epoch_values}"
        )
    print(f"[INFO] Kept {len(out):,}/{before:,} rows matching late-pre/post epochs.")

    out = out.dropna(subset=["duration_raw"]).copy()
    if args.duration_unit == "auto":
        med = float(out["duration_raw"].median())
        # Phrase durations in ms will usually have median >> 20; durations in s usually < 20.
        unit = "ms" if med > 20 else "s"
        print(f"[INFO] Auto-detected duration unit as {unit!r} from median raw duration={med:.3g}.")
    else:
        unit = args.duration_unit
    out["duration_s"] = out["duration_raw"] / 1000.0 if unit == "ms" else out["duration_raw"]

    # Merge metadata group labels if group was absent or mostly missing.
    if args.metadata_excel:
        need_meta = "group" not in out.columns or out["group"].isna().mean() > 0.5
        if need_meta or args.prefer_metadata_group:
            meta = load_metadata(
                Path(args.metadata_excel),
                args.metadata_sheet,
                args.metadata_bird_col,
                args.metadata_group_col,
            )
            out = out.drop(columns=["group", "group_raw", "plot_group"], errors="ignore").merge(meta, on="bird", how="left")
            missing = sorted(out.loc[out["group"].isna(), "bird"].unique())
            if missing:
                print(f"[WARN] Missing metadata group for {len(missing)} birds: {missing}")

    out["group"] = out["group"].map(lambda x: normalize_group_value(x) if pd.notna(x) else np.nan)
    if "plot_group" not in out.columns:
        out["plot_group"] = np.nan
    out["plot_group"] = out["plot_group"].where(out["plot_group"].notna(), out["group"])
    out["plot_group"] = out["plot_group"].map(lambda x: normalize_plot_group_value(x) if pd.notna(x) else np.nan)
    out = out.dropna(subset=["group"]).copy()
    out = out[out["group"].isin(args.keep_groups)].copy()
    print("[INFO] Rows by lesion group:")
    print(out.groupby("group").size().to_string())
    print("[INFO] Rows by plotting subgroup:")
    print(out.groupby("plot_group").size().to_string())
    return out


# ---------------------------- selection utilities ---------------------------

def load_selection_csv(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    sel = pd.read_csv(args.selection_csv)
    bird_col = find_col(
        sel, args.selection_bird_col,
        ["bird", "animal", "Animal ID", "Bird ID", "animal_id", "bird_id"],
        label="selection bird column",
    )
    syll_col = find_col(
        sel, args.selection_syllable_col,
        ["syllable", "syllable_label", "label", "cluster", "hdbscan_label", "hdbscan_labels"],
        label="selection syllable column",
    )
    selected_col = find_col(
        sel, args.selected_col,
        ["selected", "is_selected", "top30", "top_30", "include", "in_analysis"],
        required=False,
        label="selected column",
    )
    out = sel[[bird_col, syll_col]].copy()
    out.columns = ["bird", "syllable"]
    out["bird"] = out["bird"].astype(str)
    out["syllable"] = out["syllable"].astype(str)
    if selected_col is not None:
        mask = sel[selected_col]
        if mask.dtype != bool:
            mask = mask.astype(str).str.lower().isin(["1", "true", "t", "yes", "y", "selected"])
        out = out[mask.values].copy()
    out = out.drop_duplicates()
    print(f"[INFO] Loaded {len(out):,} selected bird x syllable pairs from {args.selection_csv}")
    return out


def compute_selection(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select syllables within bird using a fixed criterion, then hold fixed."""
    rows = []
    for (bird, syll), g in df.groupby(["bird", "syllable"], sort=False):
        pre = g.loc[g["epoch"] == "pre", "duration_s"].dropna().to_numpy()
        post = g.loc[g["epoch"] == "post", "duration_s"].dropna().to_numpy()
        if len(pre) < args.min_renditions or len(post) < args.min_renditions:
            continue
        if args.selection_basis == "all":
            score = 1.0
        elif args.selection_basis == "pooled_var":
            score = 0.5 * (np.var(pre, ddof=1) + np.var(post, ddof=1))
        elif args.selection_basis == "pre_var":
            score = np.var(pre, ddof=1)
        elif args.selection_basis == "post_var":
            score = np.var(post, ddof=1)
        elif args.selection_basis == "pooled_q95":
            score = 0.5 * (np.quantile(pre, 0.95) + np.quantile(post, 0.95))
        elif args.selection_basis == "pre_q95":
            score = np.quantile(pre, 0.95)
        elif args.selection_basis == "post_q95":
            score = np.quantile(post, 0.95)
        else:
            raise ValueError(f"Unsupported selection basis: {args.selection_basis}")
        rows.append({"bird": bird, "syllable": syll, "selection_score": score, "n_pre": len(pre), "n_post": len(post)})

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No bird x syllable candidates passed the minimum rendition filter.")

    selected = []
    for bird, g in candidates.groupby("bird", sort=False):
        if args.selection_basis == "all":
            selected.append(g.copy())
            continue
        n = max(1, int(math.ceil(len(g) * args.top_percent / 100.0)))
        selected.append(g.sort_values("selection_score", ascending=False).head(n).copy())
    out = pd.concat(selected, ignore_index=True)
    print(f"[INFO] Selected {len(out):,}/{len(candidates):,} candidate bird x syllable pairs using {args.selection_basis}, top {args.top_percent:g}%.")
    return out


# ---------------------------- metric calculation ----------------------------

def _draw(arr: np.ndarray, n: int, rng: np.random.Generator, replace: bool = False) -> np.ndarray:
    idx = rng.choice(len(arr), size=n, replace=replace)
    return arr[idx]


def compute_one_syllable_metrics(
    pre: np.ndarray,
    post: np.ndarray,
    n_balance: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Usage-balanced prolongation metrics for one bird x syllable."""
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    n = min(len(pre), len(post))
    if n < 2:
        raise ValueError("Need at least two observations in each epoch.")

    # Pre-defined thresholds for long/prolonged renditions. They use only late-pre data.
    thr_pre95 = float(np.quantile(pre, 0.95))
    thr_pre99 = float(np.quantile(pre, 0.99))
    thr_pre_mean2sd = float(np.mean(pre) + 2.0 * np.std(pre, ddof=1)) if len(pre) > 1 else float("nan")

    accum: Dict[str, List[float]] = {
        "pre_mean_duration_s": [], "post_mean_duration_s": [],
        "pre_median_duration_s": [], "post_median_duration_s": [],
        "pre_q90_duration_s": [], "post_q90_duration_s": [],
        "pre_q95_duration_s": [], "post_q95_duration_s": [],
        "pre_q99_duration_s": [], "post_q99_duration_s": [],
        "pre_prop_above_pre95": [], "post_prop_above_pre95": [],
        "pre_prop_above_pre99": [], "post_prop_above_pre99": [],
        "pre_prop_above_pre_mean2sd": [], "post_prop_above_pre_mean2sd": [],
    }

    for _ in range(n_balance):
        p0 = _draw(pre, n, rng, replace=False)
        p1 = _draw(post, n, rng, replace=False)
        accum["pre_mean_duration_s"].append(float(np.mean(p0)))
        accum["post_mean_duration_s"].append(float(np.mean(p1)))
        accum["pre_median_duration_s"].append(float(np.median(p0)))
        accum["post_median_duration_s"].append(float(np.median(p1)))
        for q, name in [(0.90, "q90"), (0.95, "q95"), (0.99, "q99")]:
            accum[f"pre_{name}_duration_s"].append(float(np.quantile(p0, q)))
            accum[f"post_{name}_duration_s"].append(float(np.quantile(p1, q)))
        accum["pre_prop_above_pre95"].append(float(np.mean(p0 > thr_pre95)))
        accum["post_prop_above_pre95"].append(float(np.mean(p1 > thr_pre95)))
        accum["pre_prop_above_pre99"].append(float(np.mean(p0 > thr_pre99)))
        accum["post_prop_above_pre99"].append(float(np.mean(p1 > thr_pre99)))
        accum["pre_prop_above_pre_mean2sd"].append(float(np.mean(p0 > thr_pre_mean2sd)))
        accum["post_prop_above_pre_mean2sd"].append(float(np.mean(p1 > thr_pre_mean2sd)))

    out = {k: float(np.mean(v)) for k, v in accum.items()}
    out.update({
        "n_balanced": int(n),
        "n_pre_raw": int(len(pre)),
        "n_post_raw": int(len(post)),
        "threshold_pre95_s": thr_pre95,
        "threshold_pre99_s": thr_pre99,
        "threshold_pre_mean2sd_s": thr_pre_mean2sd,
    })
    for base in [
        "mean_duration_s", "median_duration_s", "q90_duration_s", "q95_duration_s", "q99_duration_s",
        "prop_above_pre95", "prop_above_pre99", "prop_above_pre_mean2sd",
    ]:
        out[f"delta_{base}"] = out[f"post_{base}"] - out[f"pre_{base}"]
    return out


def compute_metrics(df: pd.DataFrame, selection: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    selected_key = selection[["bird", "syllable"]].drop_duplicates().copy()
    dsel = df.merge(selected_key.assign(selected=True), on=["bird", "syllable"], how="inner")
    if dsel.empty:
        raise ValueError("No long-table rows matched the selected bird x syllable pairs.")

    rows = []
    group_cols = ["bird", "group", "plot_group", "syllable"] if "plot_group" in dsel.columns else ["bird", "group", "syllable"]
    for key, g in dsel.groupby(group_cols, sort=False):
        if len(group_cols) == 4:
            bird, group, plot_group, syll = key
        else:
            bird, group, syll = key
            plot_group = group
        pre = g.loc[g["epoch"] == "pre", "duration_s"].to_numpy()
        post = g.loc[g["epoch"] == "post", "duration_s"].to_numpy()
        if len(pre) < args.min_renditions or len(post) < args.min_renditions:
            continue
        metrics = compute_one_syllable_metrics(pre, post, args.n_balance, rng)
        metrics.update({"bird": bird, "group": group, "plot_group": plot_group, "syllable": syll})
        rows.append(metrics)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No selected bird x syllable pairs passed the minimum rendition filter after selection.")
    print(f"[INFO] Computed prolongation metrics for {len(out):,} selected bird x syllable pairs.")
    return out


def aggregate_to_bird(syll_metrics: pd.DataFrame, agg: str) -> pd.DataFrame:
    metric_cols = [c for c in syll_metrics.columns if c.startswith(("pre_", "post_", "delta_", "threshold_"))]
    if agg == "median":
        fun = "median"
    elif agg == "mean":
        fun = "mean"
    else:
        raise ValueError("--syllable-agg must be 'mean' or 'median'")
    id_cols = ["bird", "group"]
    if "plot_group" in syll_metrics.columns:
        id_cols.append("plot_group")
    bird = syll_metrics.groupby(id_cols, as_index=False)[metric_cols].agg(fun)
    counts = syll_metrics.groupby(id_cols, as_index=False).agg(
        selected_syllables=("syllable", "nunique"),
        total_balanced_renditions=("n_balanced", "sum"),
        total_pre_raw=("n_pre_raw", "sum"),
        total_post_raw=("n_post_raw", "sum"),
    )
    bird = bird.merge(counts, on=id_cols, how="left")
    if "plot_group" not in bird.columns:
        bird["plot_group"] = bird["group"]
    return bird


# ------------------------------- statistics ---------------------------------

def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    stat: str = "median",
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, len(values), size=len(values))]
        reps[i] = np.median(sample) if stat == "median" else np.mean(sample)
    return (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975)))


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    stat: str = "median",
) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = a[rng.integers(0, len(a), size=len(a))]
        sb = b[rng.integers(0, len(b), size=len(b))]
        sta = np.median(sa) if stat == "median" else np.mean(sa)
        stb = np.median(sb) if stat == "median" else np.mean(sb)
        reps[i] = sta - stb
    return (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975)))


def signflip_p(
    values: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    alternative: str = "greater",
) -> Tuple[float, float]:
    """One-sample paired sign-flip test at bird level. Test statistic: mean delta."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    obs = float(np.mean(values))
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(values), replace=True)
        null[i] = float(np.mean(values * signs))
    if alternative == "greater":
        p = (1.0 + np.sum(null >= obs)) / (n_perm + 1.0)
    elif alternative == "less":
        p = (1.0 + np.sum(null <= obs)) / (n_perm + 1.0)
    elif alternative == "two-sided":
        p = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1.0)
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return obs, float(p)


def labelshuffle_p(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    alternative: str = "greater",
) -> Tuple[float, float]:
    """Two-group label-shuffle test at bird level. Test statistic: mean(a)-mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    obs = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        null[i] = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
    if alternative == "greater":
        p = (1.0 + np.sum(null >= obs)) / (n_perm + 1.0)
    elif alternative == "less":
        p = (1.0 + np.sum(null <= obs)) / (n_perm + 1.0)
    elif alternative == "two-sided":
        p = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1.0)
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return obs, float(p)


def run_stats(bird_metrics: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + 1)
    delta_cols = [c for c in bird_metrics.columns if c.startswith("delta_")]
    group_rows = []
    compare_rows = []

    for metric in delta_cols:
        for group, g in bird_metrics.groupby("group", sort=False):
            vals = g[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            obs_mean, p = signflip_p(vals, rng, args.n_perm, args.alternative)
            ci_lo, ci_hi = bootstrap_ci(vals, rng, args.n_boot, args.group_stat)
            group_rows.append({
                "metric": metric,
                "group": group,
                "n_birds": len(vals),
                "mean_delta": float(np.mean(vals)),
                "median_delta": float(np.median(vals)),
                "bootstrap_stat": args.group_stat,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "signflip_stat_mean_delta": obs_mean,
                "signflip_p": p,
                "alternative": args.alternative,
            })

        for a_group, b_group in args.comparisons:
            a = bird_metrics.loc[bird_metrics["group"] == a_group, metric].dropna().to_numpy(dtype=float)
            b = bird_metrics.loc[bird_metrics["group"] == b_group, metric].dropna().to_numpy(dtype=float)
            if len(a) == 0 or len(b) == 0:
                continue
            obs_diff, p = labelshuffle_p(a, b, rng, args.n_perm, args.alternative)
            ci_lo, ci_hi = bootstrap_diff_ci(a, b, rng, args.n_boot, args.group_stat)
            compare_rows.append({
                "metric": metric,
                "comparison": f"{a_group} - {b_group}",
                "group_a": a_group,
                "group_b": b_group,
                "n_a": len(a),
                "n_b": len(b),
                "mean_diff": float(np.mean(a) - np.mean(b)),
                "median_diff": float(np.median(a) - np.median(b)),
                "bootstrap_stat": args.group_stat,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "labelshuffle_stat_mean_diff": obs_diff,
                "labelshuffle_p": p,
                "alternative": args.alternative,
            })

    return pd.DataFrame(group_rows), pd.DataFrame(compare_rows)


# --------------------------------- figures ----------------------------------

def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def add_identity(ax, x: np.ndarray, y: np.ndarray) -> None:
    vals = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if len(vals) == 0:
        return
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def _finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def add_identity(ax, x: np.ndarray, y: np.ndarray, color: str = "#D62728") -> None:
    x, y = _finite_xy(x, y)
    if len(x) == 0:
        return
    vals = np.concatenate([x, y])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.3, color=color, alpha=0.9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def _get_group_p(group_stats: Optional[pd.DataFrame], metric: str, group: str) -> float:
    if group_stats is None or group_stats.empty:
        return np.nan
    rows = group_stats[(group_stats["metric"] == metric) & (group_stats["group"] == group)]
    if rows.empty or "signflip_p" not in rows.columns:
        return np.nan
    return float(rows.iloc[0]["signflip_p"])


def _get_comparison_p(comparison_stats: Optional[pd.DataFrame], metric: str, a: str, b: str) -> float:
    if comparison_stats is None or comparison_stats.empty:
        return np.nan
    rows = comparison_stats[
        (comparison_stats["metric"] == metric)
        & (comparison_stats["group_a"] == a)
        & (comparison_stats["group_b"] == b)
    ]
    if rows.empty or "labelshuffle_p" not in rows.columns:
        return np.nan
    return float(rows.iloc[0]["labelshuffle_p"])


def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str, h_frac: float = 0.03) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    h = h_frac * yr
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=10)


def paper_pre_vs_post_scatter(bird: pd.DataFrame, base: str, fig_dir: Path) -> None:
    pre_col = f"pre_{base}"
    post_col = f"post_{base}"
    if pre_col not in bird.columns or post_col not in bird.columns:
        return

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    present = set(bird.get("plot_group", bird["group"]))
    handles = []
    labels = []
    for pg in PAPER_SCATTER_ORDER:
        if pg not in present:
            continue
        g = bird[bird.get("plot_group", bird["group"]) == pg]
        if g.empty:
            continue
        h = ax.scatter(
            g[pre_col],
            g[post_col],
            s=32,
            alpha=0.85,
            color=PAPER_COLORS.get(pg, "0.4"),
            edgecolor="none",
            label=PAPER_SCATTER_LABELS.get(pg, pg),
        )
        handles.append(h)
        labels.append(PAPER_SCATTER_LABELS.get(pg, pg))

    # If only pooled groups exist, draw them here.
    if not handles:
        for group in PAPER_GROUP_ORDER:
            if group not in set(bird["group"]):
                continue
            g = bird[bird["group"] == group]
            h = ax.scatter(g[pre_col], g[post_col], s=32, alpha=0.85, color=PAPER_COLORS.get(group, "0.4"), edgecolor="none")
            handles.append(h)
            labels.append(PAPER_GROUP_FULL_LABELS.get(group, group))

    add_identity(ax, bird[pre_col].to_numpy(dtype=float), bird[post_col].to_numpy(dtype=float))
    label = pretty_metric_label(base)
    ax.set_xlabel(f"Late pre-lesion {label}", fontsize=13)
    ax.set_ylabel(f"Post-lesion {label}", fontsize=13)
    apply_paper_style(ax)
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / f"paper_pre_vs_post_{safe_filename(base)}.png", dpi=450)
    fig.savefig(fig_dir / f"paper_pre_vs_post_{safe_filename(base)}.pdf")
    plt.close(fig)


def paper_paired_boxplot(
    bird: pd.DataFrame,
    base: str,
    group_stats: Optional[pd.DataFrame],
    fig_dir: Path,
) -> None:
    pre_col = f"pre_{base}"
    post_col = f"post_{base}"
    metric = f"delta_{base}"
    if pre_col not in bird.columns or post_col not in bird.columns:
        return

    groups = [g for g in PAPER_GROUP_ORDER if g in set(bird["group"])]
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    pre_positions = []
    post_positions = []
    xticks = []
    xticklabels = []
    all_values = []

    for i, group in enumerate(groups):
        center = i * 2.2 + 1.0
        pre_pos = center - 0.22
        post_pos = center + 0.22
        pre_positions.append(pre_pos)
        post_positions.append(post_pos)
        xticks.append(center)
        xticklabels.append(PAPER_GROUP_LABELS.get(group, group))
        color = PAPER_COLORS.get(group, "0.4")
        g = bird[bird["group"] == group]
        pre_vals = g[pre_col].dropna().to_numpy(dtype=float)
        post_vals = g[post_col].dropna().to_numpy(dtype=float)
        all_values.extend(pre_vals.tolist() + post_vals.tolist())

        bp = ax.boxplot(
            [pre_vals, post_vals],
            positions=[pre_pos, post_pos],
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.25", "linewidth": 1.2},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
        )
        # late pre: white fill; post: colored fill
        bp["boxes"][0].set(facecolor="white", edgecolor=color, linewidth=1.4)
        bp["boxes"][1].set(facecolor=color, edgecolor=color, linewidth=1.4, alpha=0.42)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=11)
    label = pretty_metric_label(base)
    ax.set_ylabel(label[0].upper() + label[1:], fontsize=13)
    apply_paper_style(ax)

    # Legend matching Figure 3D: open box for Late Pre and filled box for Post.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="white", edgecolor="0.25", label="Late Pre"),
        Patch(facecolor="0.55", edgecolor="0.55", alpha=0.42, label="Post"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=10, loc="upper left", ncol=2)

    if all_values:
        ymin = float(np.nanmin(all_values))
        ymax = float(np.nanmax(all_values))
        yr = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin - 0.08 * yr, ymax + 0.25 * yr)
        for i, group in enumerate(groups):
            g = bird[bird["group"] == group]
            vals = pd.concat([g[pre_col], g[post_col]], ignore_index=True).dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            y = float(np.nanmax(vals)) + 0.06 * yr
            p = _get_group_p(group_stats, metric, group)
            add_bracket(ax, pre_positions[i], post_positions[i], y, p_to_label(p))

    fig.tight_layout()
    fig.savefig(fig_dir / f"paper_paired_box_{safe_filename(base)}.png", dpi=450)
    fig.savefig(fig_dir / f"paper_paired_box_{safe_filename(base)}.pdf")
    plt.close(fig)


def paper_delta_by_group(
    bird: pd.DataFrame,
    metric: str,
    comparison_stats: Optional[pd.DataFrame],
    fig_dir: Path,
) -> None:
    if metric not in bird.columns:
        return
    groups = [g for g in PAPER_GROUP_ORDER if g in set(bird["group"])]
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    rng = np.random.default_rng(123)
    all_vals = []
    positions = np.arange(1, len(groups) + 1)
    for pos, group in zip(positions, groups):
        color = PAPER_COLORS.get(group, "0.4")
        vals = bird.loc[bird["group"] == group, metric].dropna().to_numpy(dtype=float)
        all_vals.extend(vals.tolist())
        bp = ax.boxplot(
            [vals],
            positions=[pos],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.25", "linewidth": 1.2},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
        )
        bp["boxes"][0].set(facecolor=color, edgecolor=color, linewidth=1.4, alpha=0.35)
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=28, color=color, alpha=0.85, edgecolor="none", zorder=3)

    ax.axhline(0, linestyle="--", linewidth=1.0, color="0.35")
    ax.set_xticks(positions)
    ax.set_xticklabels([PAPER_GROUP_LABELS.get(g, g) for g in groups], fontsize=11)
    base = metric.replace("delta_", "")
    label = pretty_metric_label(base)
    if "proportion" in label:
        ax.set_ylabel(f"Δ {label}", fontsize=13)
    else:
        ax.set_ylabel(f"Δ {label}", fontsize=13)
    apply_paper_style(ax)

    if all_vals:
        ymin = float(np.nanmin(all_vals))
        ymax = float(np.nanmax(all_vals))
        yr = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin - 0.18 * yr, ymax + 0.28 * yr)
        # Primary anatomical specificity bracket: medial+lateral vs lateral-only.
        if "lateral_only" in groups and "medial_lateral" in groups:
            x1 = groups.index("lateral_only") + 1
            x2 = groups.index("medial_lateral") + 1
            p = _get_comparison_p(comparison_stats, metric, "medial_lateral", "lateral_only")
            add_bracket(ax, x1, x2, ymax + 0.07 * yr, p_to_label(p))

    fig.tight_layout()
    fig.savefig(fig_dir / f"paper_delta_by_group_{safe_filename(metric)}.png", dpi=450)
    fig.savefig(fig_dir / f"paper_delta_by_group_{safe_filename(metric)}.pdf")
    plt.close(fig)


def make_figures(
    bird: pd.DataFrame,
    out_dir: Path,
    group_stats: Optional[pd.DataFrame] = None,
    comparison_stats: Optional[pd.DataFrame] = None,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    scatter_bases = [
        "median_duration_s",
        "mean_duration_s",
        "q90_duration_s",
        "q95_duration_s",
        "q99_duration_s",
        "prop_above_pre95",
        "prop_above_pre99",
        "prop_above_pre_mean2sd",
    ]
    groups = [g for g in PAPER_GROUP_ORDER if g in set(bird["group"])]

    # Original broad output set, now using the paper colors.
    for base in scatter_bases:
        pre_col = f"pre_{base}"
        post_col = f"post_{base}"
        if pre_col not in bird.columns or post_col not in bird.columns:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for group in groups:
            g = bird[bird["group"] == group]
            ax.scatter(g[pre_col], g[post_col], label=f"{PAPER_GROUP_FULL_LABELS.get(group, group)} (n={len(g)})", alpha=0.85, color=PAPER_COLORS.get(group, "0.4"), edgecolor="none")
        add_identity(ax, bird[pre_col].to_numpy(dtype=float), bird[post_col].to_numpy(dtype=float))
        ax.set_xlabel(f"Late pre {pretty_metric_label(base)}")
        ax.set_ylabel(f"Post {pretty_metric_label(base)}")
        apply_paper_style(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"pre_vs_post_{safe_filename(base)}.png", dpi=300)
        fig.savefig(fig_dir / f"pre_vs_post_{safe_filename(base)}.pdf")
        plt.close(fig)

    delta_cols = [c for c in bird.columns if c.startswith("delta_")]
    for metric in delta_cols:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        data = [bird.loc[bird["group"] == group, metric].dropna().to_numpy(dtype=float) for group in groups]
        positions = np.arange(1, len(groups) + 1)
        bp = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False, patch_artist=True)
        for box, group in zip(bp["boxes"], groups):
            color = PAPER_COLORS.get(group, "0.4")
            box.set(facecolor=color, edgecolor=color, alpha=0.35, linewidth=1.4)
        for median in bp["medians"]:
            median.set(color="0.25", linewidth=1.2)
        rng = np.random.default_rng(123)
        for pos, vals, group in zip(positions, data, groups):
            color = PAPER_COLORS.get(group, "0.4")
            jitter = rng.normal(loc=0, scale=0.04, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, alpha=0.85, s=25, color=color, edgecolor="none")
        ax.axhline(0, linestyle="--", linewidth=1, color="0.35")
        ax.set_xticks(positions)
        ax.set_xticklabels([PAPER_GROUP_LABELS.get(g, g) for g in groups], rotation=0, ha="center")
        ax.set_ylabel("Δ " + pretty_metric_label(metric.replace("delta_", "")))
        apply_paper_style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"delta_by_group_{safe_filename(metric)}.png", dpi=300)
        fig.savefig(fig_dir / f"delta_by_group_{safe_filename(metric)}.pdf")
        plt.close(fig)

    # Focused, manuscript-style figures for the main prolongation follow-up.
    for base in PAPER_FOCUS_BASES:
        paper_pre_vs_post_scatter(bird, base, fig_dir)
        paper_paired_boxplot(bird, base, group_stats, fig_dir)
        paper_delta_by_group(bird, f"delta_{base}", comparison_stats, fig_dir)


# ----------------------------------- main -----------------------------------

def parse_comparisons(raw: str) -> List[Tuple[str, str]]:
    pairs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("Comparisons must look like 'medial_lateral:lateral_only,medial_lateral:sham'")
        a, b = part.split(":", 1)
        pairs.append((a.strip(), b.strip()))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check phrase-duration prolongation after AFP lesions.")
    p.add_argument("--long-csv", required=True, type=Path, help="Long-format phrase-duration CSV, one row per phrase rendition.")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    p.add_argument("--metadata-excel", type=Path, default=None, help="Optional lesion metadata Excel file for group labels.")
    p.add_argument("--metadata-sheet", default=None)
    p.add_argument("--metadata-bird-col", default=None)
    p.add_argument("--metadata-group-col", default=None)
    p.add_argument("--prefer-metadata-group", action="store_true", help="Use metadata group labels even if long CSV has a group column.")

    p.add_argument("--bird-col", default=None)
    p.add_argument("--syllable-col", default=None)
    p.add_argument("--epoch-col", default=None)
    p.add_argument("--duration-col", default=None)
    p.add_argument("--group-col", default=None)
    p.add_argument("--duration-unit", choices=["auto", "ms", "s"], default="auto")
    p.add_argument(
        "--pre-epoch-values",
        default="Late Pre,late_pre,late pre,late pre-lesion,late pre lesion,pre",
        help="Comma-separated epoch values to treat as late pre-lesion. Exact matching after case/underscore normalization.",
    )
    p.add_argument(
        "--post-epoch-values",
        default="Post,post-lesion,post lesion,post",
        help="Comma-separated epoch values to treat as post-lesion. Exact matching after case/underscore normalization.",
    )
    p.add_argument("--keep-groups", nargs="+", default=["sham", "lateral_only", "medial_lateral"])

    p.add_argument("--selection-csv", type=Path, default=None, help="Optional CSV of selected bird x syllable pairs.")
    p.add_argument("--selection-bird-col", default=None)
    p.add_argument("--selection-syllable-col", default=None)
    p.add_argument("--selected-col", default=None)
    p.add_argument(
        "--selection-basis",
        choices=["pooled_var", "pre_var", "post_var", "pooled_q95", "pre_q95", "post_q95", "all"],
        default="pooled_var",
        help="Selection basis used if --selection-csv is not provided.",
    )
    p.add_argument("--top-percent", type=float, default=30.0)
    p.add_argument("--min-renditions", type=int, default=10)

    p.add_argument("--n-balance", type=int, default=200, help="Usage-balancing draws for each bird x syllable.")
    p.add_argument("--syllable-agg", choices=["mean", "median"], default="median", help="How to aggregate selected syllables within bird.")
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--group-stat", choices=["mean", "median"], default="median")
    p.add_argument("--alternative", choices=["greater", "less", "two-sided"], default="greater")
    p.add_argument(
        "--comparisons",
        type=parse_comparisons,
        default=parse_comparisons("medial_lateral:lateral_only,medial_lateral:sham,lateral_only:sham"),
        help="Comma-separated group pairs A:B, interpreted as A - B.",
    )
    p.add_argument("--seed", type=int, default=12345)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_long_table(args)

    if args.selection_csv:
        selection = load_selection_csv(args, df)
    else:
        selection = compute_selection(df, args)
    selection.to_csv(args.out_dir / "selected_syllables_used_for_prolongation.csv", index=False)

    syll_metrics = compute_metrics(df, selection, args)
    syll_metrics.to_csv(args.out_dir / "animal_syllable_prolongation_metrics.csv", index=False)

    bird_metrics = aggregate_to_bird(syll_metrics, args.syllable_agg)
    bird_metrics.to_csv(args.out_dir / "animal_level_prolongation_metrics.csv", index=False)

    group_stats, comparison_stats = run_stats(bird_metrics, args)
    group_stats.to_csv(args.out_dir / "group_prolongation_stats.csv", index=False)
    comparison_stats.to_csv(args.out_dir / "group_comparison_prolongation_stats.csv", index=False)

    make_figures(bird_metrics, args.out_dir, group_stats=group_stats, comparison_stats=comparison_stats)

    print("\n[Done] Wrote outputs to:", args.out_dir)
    print("\n[Key output files]")
    for name in [
        "selected_syllables_used_for_prolongation.csv",
        "animal_syllable_prolongation_metrics.csv",
        "animal_level_prolongation_metrics.csv",
        "group_prolongation_stats.csv",
        "group_comparison_prolongation_stats.csv",
        "figures/",
    ]:
        print(" -", args.out_dir / name)

    print("\n[Quick view: within-group stats]")
    if not group_stats.empty:
        cols = ["metric", "group", "n_birds", "median_delta", "ci_low", "ci_high", "signflip_p"]
        print(group_stats[cols].to_string(index=False))
    print("\n[Quick view: between-group stats]")
    if not comparison_stats.empty:
        cols = ["metric", "comparison", "median_diff", "ci_low", "ci_high", "labelshuffle_p"]
        print(comparison_stats[cols].to_string(index=False))


if __name__ == "__main__":
    main()
