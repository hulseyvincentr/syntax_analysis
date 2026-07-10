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
    out = out.drop_duplicates(subset=["bird"])
    return out[["bird", "group", "group_raw"]]


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
        out["group"] = df[group_col].map(normalize_group_value)
    else:
        out["group"] = np.nan

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
            out = out.drop(columns=["group"], errors="ignore").merge(meta[["bird", "group"]], on="bird", how="left")
            missing = sorted(out.loc[out["group"].isna(), "bird"].unique())
            if missing:
                print(f"[WARN] Missing metadata group for {len(missing)} birds: {missing}")

    out["group"] = out["group"].map(lambda x: normalize_group_value(x) if pd.notna(x) else np.nan)
    out = out.dropna(subset=["group"]).copy()
    out = out[out["group"].isin(args.keep_groups)].copy()
    print("[INFO] Rows by lesion group:")
    print(out.groupby("group").size().to_string())
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
    for (bird, group, syll), g in dsel.groupby(["bird", "group", "syllable"], sort=False):
        pre = g.loc[g["epoch"] == "pre", "duration_s"].to_numpy()
        post = g.loc[g["epoch"] == "post", "duration_s"].to_numpy()
        if len(pre) < args.min_renditions or len(post) < args.min_renditions:
            continue
        metrics = compute_one_syllable_metrics(pre, post, args.n_balance, rng)
        metrics.update({"bird": bird, "group": group, "syllable": syll})
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
    bird = syll_metrics.groupby(["bird", "group"], as_index=False)[metric_cols].agg(fun)
    counts = syll_metrics.groupby(["bird", "group"], as_index=False).agg(
        selected_syllables=("syllable", "nunique"),
        total_balanced_renditions=("n_balanced", "sum"),
        total_pre_raw=("n_pre_raw", "sum"),
        total_post_raw=("n_post_raw", "sum"),
    )
    bird = bird.merge(counts, on=["bird", "group"], how="left")
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


def make_figures(bird: pd.DataFrame, out_dir: Path) -> None:
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
    groups = [g for g in ["sham", "lateral_only", "medial_lateral"] if g in set(bird["group"])]

    for base in scatter_bases:
        pre_col = f"pre_{base}"
        post_col = f"post_{base}"
        if pre_col not in bird.columns or post_col not in bird.columns:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for group in groups:
            g = bird[bird["group"] == group]
            ax.scatter(g[pre_col], g[post_col], label=f"{group} (n={len(g)})", alpha=0.85)
        add_identity(ax, bird[pre_col].to_numpy(dtype=float), bird[post_col].to_numpy(dtype=float))
        ax.set_xlabel(f"Late pre {base.replace('_', ' ')}")
        ax.set_ylabel(f"Post {base.replace('_', ' ')}")
        ax.set_title(f"Animal-level prolongation: {base.replace('_', ' ')}")
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
        ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
        rng = np.random.default_rng(123)
        for pos, vals in zip(positions, data):
            jitter = rng.normal(loc=0, scale=0.04, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, alpha=0.85, s=25)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=25, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Animal-level Δ {metric.replace('delta_', '').replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"delta_by_group_{safe_filename(metric)}.png", dpi=300)
        fig.savefig(fig_dir / f"delta_by_group_{safe_filename(metric)}.pdf")
        plt.close(fig)


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

    make_figures(bird_metrics, args.out_dir)

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
