#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure3_robust_phrase_duration_stats_v1.py

Build reviewer-facing Figure 3 robustness statistics and supplemental tables.

What this script does
---------------------
1) Recomputes phrase-duration SD robustness analyses from the animal × syllable
   pre/post summary table, using bird as the independent unit.

2) Produces a Supplemental Table 1-style robustness table with:
      - post-selected top 20%, 30%, 40%
      - pooled late-pre/post-selected top 30%
      - late-pre-selected top 30% validation
      - optional held-out post-selected validation

3) Produces a Supplemental Table 2-style animal/sample-size table for the
   primary pooled late-pre/post-selected top 30% analysis.

4) Produces supplemental plots EXCEPT the three-group animal-level delta SD plot
   you already made:
      - robustness contrast plot: medial+lateral minus lateral-only ΔSD
      - selected syllable counts by animal/group
      - optional held-out selection-frequency diagnostics

Important note about held-out validation
----------------------------------------
The strongest held-out analysis uses one row per phrase rendition. If you have a
long-format rendition table, pass it with --rendition-csv. If you do not pass a
rendition table, the script can run a day-level held-out validation from the
daily variance table using --daily-csv. The day-level version is a useful
robustness check, but it should be described as day-level validation because it
splits post-lesion days rather than individual phrase renditions.

Example
-------
python figure3_robust_phrase_duration_stats_v1.py \
  --scatter-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv" \
  --daily-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/phrase_duration_batch_outputs/_batch_summary/batch_aligned_phrase_duration_variance_all.csv" \
  --metadata-excel "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --out-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/Figure3_robust_stats_tables_v1" \
  --primary-method "Pooled pre+post-selected top 30%" \
  --n-bootstrap 5000 \
  --n-permutations 10000 \
  --n-heldout-splits 500 \
  --seed 123
"""

from __future__ import annotations

import argparse
import itertools
import math
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


# ──────────────────────────────────────────────────────────────────────────────
# Lesion group labels
# ──────────────────────────────────────────────────────────────────────────────
SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL_ML = "Partial Medial and Lateral lesion"
COMPLETE_ML = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"
ML_COMPONENTS = [PARTIAL_ML, COMPLETE_ML]

GROUP_ORDER = [SHAM, LATERAL, PARTIAL_ML, COMPLETE_ML, POOLED_ML]
PLOTTED_GROUP_ORDER = [SHAM, LATERAL, POOLED_ML]

GROUP_COLORS = {
    SHAM: "#1B9E77",
    LATERAL: "#A88BD9",
    PARTIAL_ML: "#7A4FB7",
    COMPLETE_ML: "#3F007D",
    POOLED_ML: "#6F4AB6",
    "unknown": "#4D4D4D",
}

RUNS = [
    {
        "selection_method": "Post-selected top 20%",
        "selection_basis": "post-lesion selection",
        "selection_fraction": "top 20%",
        "rank_on": "post",
        "top_frac": 0.20,
        "source": "pre/post summary",
    },
    {
        "selection_method": "Post-selected top 30%",
        "selection_basis": "post-lesion selection",
        "selection_fraction": "top 30%",
        "rank_on": "post",
        "top_frac": 0.30,
        "source": "pre/post summary",
    },
    {
        "selection_method": "Post-selected top 40%",
        "selection_basis": "post-lesion selection",
        "selection_fraction": "top 40%",
        "rank_on": "post",
        "top_frac": 0.40,
        "source": "pre/post summary",
    },
    {
        "selection_method": "Pooled pre+post-selected top 30%",
        "selection_basis": "pooled late-pre/post selection",
        "selection_fraction": "top 30%",
        "rank_on": "pooled",
        "top_frac": 0.30,
        "source": "pre/post summary",
    },
    {
        "selection_method": "Pre-selected top 30% validation",
        "selection_basis": "late pre-lesion selection",
        "selection_fraction": "top 30%",
        "rank_on": "pre",
        "top_frac": 0.30,
        "source": "pre/post summary",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# General utilities
# ──────────────────────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def lower(x: Any) -> str:
    return clean_text(x).lower()


def canonical_group(raw: Any) -> str:
    s = lower(raw)
    if not s or s == "nan":
        return "unknown"

    if "sham" in s or ("saline" in s and "lesion" not in s):
        return SHAM

    if "lateral lesion only" in s or "lateral only" in s or "lateral hit only" in s:
        return LATERAL
    if "area x visible" in s and "lateral" in s and "medial" not in s:
        return LATERAL
    if "single hit" in s:
        return LATERAL

    if "complete" in s and "medial" in s and "lateral" in s:
        return COMPLETE_ML
    if "large lesion" in s or "not visible" in s:
        return COMPLETE_ML

    if "partial" in s and "medial" in s and "lateral" in s:
        return PARTIAL_ML
    if "medial" in s and "lateral" in s:
        return PARTIAL_ML
    if "m+l" in s:
        return PARTIAL_ML

    return clean_text(raw) or "unknown"


def pretty_group_label(group: str) -> str:
    mapping = {
        SHAM: "Sham",
        LATERAL: "Lateral-only",
        PARTIAL_ML: "Partial ML",
        COMPLETE_ML: "Complete ML",
        POOLED_ML: "Medial+lateral",
        "unknown": "unknown",
    }
    return mapping.get(str(group), str(group))


def format_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def stars_from_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    exact = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in exact:
            return exact[cand.lower()]
    # fuzzy
    for cand in candidates:
        c0 = cand.lower()
        for c in cols:
            if c0 in c.lower():
                return c
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────────────
def load_metadata(metadata_excel: Optional[Path], animal_col: str = "Animal ID") -> pd.DataFrame:
    if metadata_excel is None:
        return pd.DataFrame(columns=["animal_id", "display_group"])

    metadata_excel = Path(metadata_excel)
    if not metadata_excel.exists():
        raise FileNotFoundError(f"metadata_excel not found: {metadata_excel}")

    xls = pd.ExcelFile(metadata_excel)
    chosen = None
    for sheet in xls.sheet_names:
        df = pd.read_excel(metadata_excel, sheet_name=sheet)
        if animal_col in df.columns:
            chosen = df.copy()
            chosen["_metadata_sheet"] = sheet
            break
    if chosen is None:
        raise ValueError(f"No metadata sheet contained animal column {animal_col!r}")

    df = chosen.copy()
    df["animal_id"] = df[animal_col].astype(str).str.strip()

    hit_col = find_col(
        list(df.columns),
        [
            "Lesion hit type",
            "Hit type",
            "Hit Type",
            "lesion_hit_type",
            "Lesion hit type group",
            "Treatment type",
            "Group",
        ],
    )
    if hit_col is None:
        warnings.warn("Could not find lesion hit type column in metadata; using unknown groups.")
        df["display_group"] = "unknown"
    else:
        df["display_group"] = df[hit_col].apply(canonical_group)

    pct_col = find_col(
        list(df.columns),
        [
            "% Area X lesioned",
            "Percent Area X lesioned",
            "Area X lesioned",
            "avg % Area X lesioned",
            "Area X lesion %",
        ],
    )
    date_col = find_col(list(df.columns), ["Treatment date", "Lesion date", "treatment_date"])

    keep_cols = ["animal_id", "display_group", "_metadata_sheet"]
    rename_map = {}
    if hit_col:
        keep_cols.append(hit_col)
        rename_map[hit_col] = "raw_hit_type"
    if pct_col:
        keep_cols.append(pct_col)
        rename_map[pct_col] = "percent_area_x_lesioned"
    if date_col:
        keep_cols.append(date_col)
        rename_map[date_col] = "treatment_date"

    out = df[keep_cols].rename(columns=rename_map).drop_duplicates("animal_id")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pre/post summary table
# ──────────────────────────────────────────────────────────────────────────────
def prepare_prepost_table(
    scatter_csv: Path,
    metadata: pd.DataFrame,
    *,
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    group_col: str = "Group",
    variance_col: str = "Variance_ms2",
    nphrases_col: str = "N_phrases",
    min_n_phrases: int = 5,
) -> pd.DataFrame:
    df = pd.read_csv(scatter_csv)
    required = [animal_col, syllable_col, group_col, variance_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"scatter_csv missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[df[group_col].astype(str).isin([pre_group, post_group])].copy()
    df[variance_col] = pd.to_numeric(df[variance_col], errors="coerce")

    if nphrases_col in df.columns:
        df[nphrases_col] = pd.to_numeric(df[nphrases_col], errors="coerce")
        df = df[df[nphrases_col].fillna(0) >= min_n_phrases].copy()

    # Average repeated rows for variance; sum phrase counts.
    val = (
        df.groupby([animal_col, syllable_col, group_col], dropna=False)[variance_col]
        .mean()
        .reset_index()
    )
    wide = val.pivot_table(
        index=[animal_col, syllable_col],
        columns=group_col,
        values=variance_col,
        aggfunc="mean",
    ).reset_index()

    if pre_group not in wide.columns or post_group not in wide.columns:
        raise ValueError(f"Could not find both {pre_group!r} and {post_group!r} after pivoting.")

    wide = wide.rename(columns={pre_group: "pre_variance_ms2", post_group: "post_variance_ms2"})
    wide = wide.dropna(subset=["pre_variance_ms2", "post_variance_ms2"])
    wide = wide[(wide["pre_variance_ms2"] > 0) & (wide["post_variance_ms2"] > 0)].copy()

    # Phrase counts, if available.
    if nphrases_col in df.columns:
        nval = (
            df.groupby([animal_col, syllable_col, group_col], dropna=False)[nphrases_col]
            .sum()
            .reset_index()
        )
        nw = nval.pivot_table(
            index=[animal_col, syllable_col],
            columns=group_col,
            values=nphrases_col,
            aggfunc="sum",
        ).reset_index()
        nw = nw.rename(columns={pre_group: "n_late_pre_phrases", post_group: "n_post_phrases"})
        wide = wide.merge(nw[[animal_col, syllable_col, "n_late_pre_phrases", "n_post_phrases"]], on=[animal_col, syllable_col], how="left")
    else:
        wide["n_late_pre_phrases"] = np.nan
        wide["n_post_phrases"] = np.nan

    wide["pre_sd_ms"] = np.sqrt(wide["pre_variance_ms2"])
    wide["post_sd_ms"] = np.sqrt(wide["post_variance_ms2"])
    wide["pre_sd_s"] = wide["pre_sd_ms"] / 1000.0
    wide["post_sd_s"] = wide["post_sd_ms"] / 1000.0
    wide["sd_delta_s"] = wide["post_sd_s"] - wide["pre_sd_s"]
    wide["sd_log2_ratio"] = np.log2(wide["post_sd_s"] / wide["pre_sd_s"])

    wide["rank_post"] = wide["post_variance_ms2"]
    wide["rank_pre"] = wide["pre_variance_ms2"]
    wide["rank_pooled"] = wide[["pre_variance_ms2", "post_variance_ms2"]].mean(axis=1)
    wide["rank_max"] = wide[["pre_variance_ms2", "post_variance_ms2"]].max(axis=1)

    wide = wide.rename(columns={animal_col: "animal_id", syllable_col: "syllable"})
    wide["animal_id"] = wide["animal_id"].astype(str)
    wide["syllable"] = wide["syllable"].astype(str)

    if metadata is not None and not metadata.empty:
        wide = wide.merge(metadata, on="animal_id", how="left")
    if "display_group" not in wide.columns:
        wide["display_group"] = "unknown"
    wide["display_group"] = wide["display_group"].fillna("unknown").apply(canonical_group)

    return wide


def select_top_syllables(prepost: pd.DataFrame, rank_on: str, top_frac: float) -> pd.DataFrame:
    rank_col = f"rank_{rank_on}"
    if rank_col not in prepost.columns:
        raise ValueError(f"Unknown rank_on={rank_on!r}; expected one of pre, post, pooled, max")

    parts = []
    for animal, g in prepost.groupby("animal_id", dropna=False):
        vals = g[rank_col].dropna()
        if vals.empty:
            continue
        threshold = np.nanpercentile(vals.to_numpy(float), 100.0 * (1.0 - top_frac))
        keep = g[g[rank_col] >= threshold].copy()
        keep["selection_rank_metric"] = keep[rank_col]
        keep["selection_threshold"] = threshold
        keep["rank_on"] = rank_on
        keep["top_frac"] = top_frac
        parts.append(keep)
    if not parts:
        return prepost.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def animal_level_from_selected(selected: pd.DataFrame) -> pd.DataFrame:
    def first_nonempty(x: pd.Series) -> str:
        vals = [str(v) for v in x.dropna().astype(str).tolist() if str(v) != "" and str(v) != "nan"]
        return vals[0] if vals else "unknown"

    agg = (
        selected.groupby("animal_id", dropna=False)
        .agg(
            display_group=("display_group", first_nonempty),
            n_selected_syllables=("syllable", "nunique"),
            pre_sd_s=("pre_sd_s", "mean"),
            post_sd_s=("post_sd_s", "mean"),
            sd_delta_s=("sd_delta_s", "mean"),
            sd_log2_ratio=("sd_log2_ratio", "mean"),
            n_late_pre_phrases=("n_late_pre_phrases", "sum"),
            n_post_phrases=("n_post_phrases", "sum"),
        )
        .reset_index()
    )
    return agg


def add_pooled_ml(df: pd.DataFrame, group_col: str = "display_group") -> pd.DataFrame:
    base = df.copy()
    ml = base[base[group_col].isin(ML_COMPONENTS)].copy()
    if ml.empty:
        return base
    pooled = ml.copy()
    pooled[group_col] = POOLED_ML
    return pd.concat([base, pooled], ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────
def paired_t_post_gt_pre(pre: np.ndarray, post: np.ndarray) -> tuple[float, float, int]:
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    n = int(len(pre))
    if n < 2 or scipy_stats is None:
        return np.nan, np.nan, n
    res = scipy_stats.ttest_rel(post, pre, alternative="greater")
    return float(res.statistic), float(res.pvalue), n


def signflip_p_greater(deltas: np.ndarray, rng: np.random.Generator, n_perm: int = 10000, stat: str = "median") -> float:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return np.nan
    obs = np.nanmedian(deltas) if stat == "median" else np.nanmean(deltas)

    # exact for small n
    if len(deltas) <= 16:
        vals = []
        for signs in itertools.product([-1, 1], repeat=len(deltas)):
            d = deltas * np.asarray(signs)
            vals.append(np.nanmedian(d) if stat == "median" else np.nanmean(d))
        vals = np.asarray(vals)
        return float((np.sum(vals >= obs - 1e-15) + 1) / (len(vals) + 1))

    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(deltas))
        d = deltas * signs
        val = np.nanmedian(d) if stat == "median" else np.nanmean(d)
        if val >= obs - 1e-15:
            count += 1
    return float((count + 1) / (n_perm + 1))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000, stat: str = "median") -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    out = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        out.append(np.nanmedian(sample) if stat == "median" else np.nanmean(sample))
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def difference_stat(a: np.ndarray, b: np.ndarray, stat: str = "median") -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if stat == "median":
        return float(np.nanmedian(a) - np.nanmedian(b))
    return float(np.nanmean(a) - np.nanmean(b))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 5000, stat: str = "median") -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals.append(difference_stat(aa, bb, stat=stat))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def labelshuffle_p_a_gt_b(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 10000,
    stat: str = "median",
) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    obs = difference_stat(a, b, stat=stat)
    pooled = np.concatenate([a, b])
    n_a = len(a)

    # exact if not too many combinations
    n_total = len(pooled)
    n_comb = math.comb(n_total, n_a) if n_total <= 30 else n_perm + 1
    if n_comb <= 60000:
        count = 0
        total = 0
        idx_all = np.arange(n_total)
        for idx_a in itertools.combinations(idx_all, n_a):
            mask = np.zeros(n_total, dtype=bool)
            mask[list(idx_a)] = True
            val = difference_stat(pooled[mask], pooled[~mask], stat=stat)
            if val >= obs - 1e-15:
                count += 1
            total += 1
        return float((count + 1) / (total + 1))

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        aa = perm[:n_a]
        bb = perm[n_a:]
        val = difference_stat(aa, bb, stat=stat)
        if val >= obs - 1e-15:
            count += 1
    return float((count + 1) / (n_perm + 1))


def compute_group_stats(
    animal_df: pd.DataFrame,
    *,
    method_info: dict[str, Any],
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
    stat: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = add_pooled_ml(animal_df, "display_group")

    summary_rows = []
    for group in GROUP_ORDER:
        g = work[work["display_group"] == group].copy()
        if g.empty:
            continue
        pre = g["pre_sd_s"].to_numpy(float)
        post = g["post_sd_s"].to_numpy(float)
        delta = g["sd_delta_s"].to_numpy(float)
        t_stat, t_p, n = paired_t_post_gt_pre(pre, post)
        sign_p = signflip_p_greater(delta, rng, n_perm=n_perm, stat=stat)
        ci_low, ci_high = bootstrap_ci(delta, rng, n_boot=n_boot, stat=stat)

        row = dict(method_info)
        row.update({
            "display_group": group,
            "n_birds": int(g["animal_id"].nunique()),
            "median_pre_sd_s": float(np.nanmedian(pre)),
            "median_post_sd_s": float(np.nanmedian(post)),
            "median_delta_sd_s": float(np.nanmedian(delta)),
            "mean_delta_sd_s": float(np.nanmean(delta)),
            "delta_sd_CI95_low_s": ci_low,
            "delta_sd_CI95_high_s": ci_high,
            "paired_t_post_gt_pre": t_stat,
            "paired_p_post_gt_pre": t_p,
            "signflip_p_delta_gt_0": sign_p,
            "paired_significance": stars_from_p(t_p),
            "signflip_significance": stars_from_p(sign_p),
            "median_n_selected_syllables": float(np.nanmedian(g["n_selected_syllables"])),
            "total_selected_syllables": int(np.nansum(g["n_selected_syllables"])),
            "resampling_unit": "animal",
        })
        summary_rows.append(row)

    contrast_specs = [
        ("primary specificity: medial+lateral > lateral-only", POOLED_ML, LATERAL),
        ("secondary: medial+lateral > sham", POOLED_ML, SHAM),
        ("secondary: lateral-only > sham", LATERAL, SHAM),
    ]
    contrast_rows = []
    for contrast_type, group_a, group_b in contrast_specs:
        a = work.loc[work["display_group"] == group_a, "sd_delta_s"].to_numpy(float)
        b = work.loc[work["display_group"] == group_b, "sd_delta_s"].to_numpy(float)
        obs = difference_stat(a, b, stat=stat)
        ci_low, ci_high = bootstrap_diff_ci(a, b, rng, n_boot=n_boot, stat=stat)
        p = labelshuffle_p_a_gt_b(a, b, rng, n_perm=n_perm, stat=stat)

        row = dict(method_info)
        row.update({
            "contrast_type": contrast_type,
            "group_A": group_a,
            "group_B": group_b,
            "n_A_birds": int(np.sum(np.isfinite(a))),
            "n_B_birds": int(np.sum(np.isfinite(b))),
            "observed_delta_difference_s_A_minus_B": obs,
            "difference_bootstrap_CI95_low_s": ci_low,
            "difference_bootstrap_CI95_high_s": ci_high,
            "labelshuffle_p_A_gt_B": p,
            "significance": stars_from_p(p),
            "resampling_unit": "animal",
        })
        contrast_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Held-out post-selection from daily variance table
# ──────────────────────────────────────────────────────────────────────────────
def heldout_day_split_validation(
    daily_csv: Path,
    metadata: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_splits: int = 500,
    top_frac: float = 0.30,
    x_min: int = -30,
    x_max: int = 30,
    min_pre_days: int = 3,
    min_train_post_days: int = 2,
    min_test_post_days: int = 2,
    aggregate_stat: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(daily_csv)
    required = ["animal_id", "syllable", "relative_day", "Variance (ms^2)"]
    missing = [c for c in required if c not in daily.columns]
    if missing:
        raise ValueError(f"daily_csv missing required columns for held-out validation: {missing}")

    daily = daily.copy()
    daily["animal_id"] = daily["animal_id"].astype(str)
    daily["syllable"] = daily["syllable"].astype(str)
    daily["relative_day"] = pd.to_numeric(daily["relative_day"], errors="coerce")
    daily["variance_ms2"] = pd.to_numeric(daily["Variance (ms^2)"], errors="coerce")
    daily = daily.dropna(subset=["relative_day", "variance_ms2"])
    daily = daily[(daily["relative_day"] >= x_min) & (daily["relative_day"] <= x_max)].copy()
    daily = daily[daily["variance_ms2"] > 0].copy()
    daily["daily_sd_s"] = np.sqrt(daily["variance_ms2"]) / 1000.0

    if metadata is not None and not metadata.empty:
        daily = daily.merge(metadata[["animal_id", "display_group"]].drop_duplicates("animal_id"), on="animal_id", how="left")
    elif "lesion_group" in daily.columns:
        daily["display_group"] = daily["lesion_group"].apply(canonical_group)
    else:
        daily["display_group"] = "unknown"
    daily["display_group"] = daily["display_group"].fillna("unknown").apply(canonical_group)

    split_animal_rows = []
    selection_count_rows = []

    animals = sorted(daily["animal_id"].unique())
    for split_i in range(n_splits):
        for animal in animals:
            adf = daily[daily["animal_id"] == animal].copy()
            if adf.empty:
                continue
            display_group = adf["display_group"].dropna().astype(str).iloc[0] if adf["display_group"].notna().any() else "unknown"

            # Split post-lesion days within animal.
            post_days = np.array(sorted(adf.loc[adf["relative_day"] > 0, "relative_day"].dropna().unique()), dtype=float)
            if len(post_days) < (min_train_post_days + min_test_post_days):
                continue
            rng.shuffle(post_days)
            cut = max(min_train_post_days, len(post_days) // 2)
            cut = min(cut, len(post_days) - min_test_post_days)
            train_days = set(post_days[:cut].tolist())
            test_days = set(post_days[cut:].tolist())

            pre_df = adf[adf["relative_day"] < 0]
            train_df = adf[adf["relative_day"].isin(train_days)]
            test_df = adf[adf["relative_day"].isin(test_days)]

            candidate_rows = []
            for syl, s_train in train_df.groupby("syllable"):
                s_pre = pre_df[pre_df["syllable"] == syl]
                s_test = test_df[test_df["syllable"] == syl]
                if s_pre["relative_day"].nunique() < min_pre_days:
                    continue
                if s_train["relative_day"].nunique() < min_train_post_days:
                    continue
                if s_test["relative_day"].nunique() < min_test_post_days:
                    continue
                rank_metric = float(np.nanmean(s_train["variance_ms2"].to_numpy(float)))
                pre_val = float(np.nanmedian(s_pre["daily_sd_s"])) if aggregate_stat == "median" else float(np.nanmean(s_pre["daily_sd_s"]))
                post_val = float(np.nanmedian(s_test["daily_sd_s"])) if aggregate_stat == "median" else float(np.nanmean(s_test["daily_sd_s"]))
                candidate_rows.append({
                    "split_i": split_i,
                    "animal_id": animal,
                    "syllable": syl,
                    "display_group": display_group,
                    "rank_metric_train_post_variance_ms2": rank_metric,
                    "pre_sd_s": pre_val,
                    "post_sd_s": post_val,
                    "sd_delta_s": post_val - pre_val,
                    "n_pre_days": int(s_pre["relative_day"].nunique()),
                    "n_train_post_days": int(s_train["relative_day"].nunique()),
                    "n_test_post_days": int(s_test["relative_day"].nunique()),
                })

            if not candidate_rows:
                continue
            cand = pd.DataFrame(candidate_rows)
            threshold = np.nanpercentile(cand["rank_metric_train_post_variance_ms2"].to_numpy(float), 100.0 * (1.0 - top_frac))
            selected = cand[cand["rank_metric_train_post_variance_ms2"] >= threshold].copy()
            if selected.empty:
                continue

            for _, r in selected.iterrows():
                selection_count_rows.append({
                    "animal_id": animal,
                    "syllable": r["syllable"],
                    "display_group": display_group,
                    "n_selected_in_split": 1,
                })

            split_animal_rows.append({
                "split_i": split_i,
                "animal_id": animal,
                "display_group": display_group,
                "n_selected_syllables": int(selected["syllable"].nunique()),
                "pre_sd_s": float(np.nanmean(selected["pre_sd_s"])),
                "post_sd_s": float(np.nanmean(selected["post_sd_s"])),
                "sd_delta_s": float(np.nanmean(selected["sd_delta_s"])),
                "median_n_pre_days": float(np.nanmedian(selected["n_pre_days"])),
                "median_n_train_post_days": float(np.nanmedian(selected["n_train_post_days"])),
                "median_n_test_post_days": float(np.nanmedian(selected["n_test_post_days"])),
            })

    split_df = pd.DataFrame(split_animal_rows)
    if split_df.empty:
        return split_df, pd.DataFrame(selection_count_rows)

    # Average split-level animal effects so each bird contributes one row.
    animal = (
        split_df.groupby("animal_id", dropna=False)
        .agg(
            display_group=("display_group", "first"),
            n_splits_contributed=("split_i", "nunique"),
            n_selected_syllables=("n_selected_syllables", "mean"),
            pre_sd_s=("pre_sd_s", "mean"),
            post_sd_s=("post_sd_s", "mean"),
            sd_delta_s=("sd_delta_s", "mean"),
            median_n_pre_days=("median_n_pre_days", "median"),
            median_n_train_post_days=("median_n_train_post_days", "median"),
            median_n_test_post_days=("median_n_test_post_days", "median"),
        )
        .reset_index()
    )

    selection_counts = pd.DataFrame(selection_count_rows)
    if not selection_counts.empty:
        selection_counts = (
            selection_counts.groupby(["animal_id", "syllable", "display_group"], dropna=False)["n_selected_in_split"]
            .sum()
            .reset_index()
        )
        selection_counts["selection_frequency"] = selection_counts["n_selected_in_split"] / float(n_splits)

    return animal, selection_counts


# ──────────────────────────────────────────────────────────────────────────────
# Held-out post-selection from rendition-level table
# ──────────────────────────────────────────────────────────────────────────────
def detect_rendition_cols(df: pd.DataFrame) -> dict[str, str]:
    cols = list(df.columns)
    animal = find_col(cols, ["animal_id", "Animal ID", "Animal", "bird", "Bird"])
    syllable = find_col(cols, ["syllable", "Syllable", "label", "cluster", "hdbscan_label"])
    duration = find_col(cols, ["duration_ms", "phrase_duration_ms", "Phrase Duration (ms)", "phrase_duration", "duration"])
    epoch = find_col(cols, ["Group", "group", "epoch", "Epoch", "period", "Period"])
    rel_day = find_col(cols, ["relative_day", "Relative day", "days_relative_to_lesion", "day_relative_to_lesion"])

    if animal is None or syllable is None or duration is None:
        raise ValueError(
            "Could not detect required rendition columns. Need animal, syllable, and duration columns. "
            f"Found columns: {cols}"
        )
    return {"animal": animal, "syllable": syllable, "duration": duration, "epoch": epoch, "relative_day": rel_day}


def heldout_rendition_validation(
    rendition_csv: Path,
    metadata: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_splits: int = 500,
    top_frac: float = 0.30,
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    min_pre_renditions: int = 10,
    min_train_post_renditions: int = 10,
    min_test_post_renditions: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(rendition_csv)
    cc = detect_rendition_cols(df)

    work = pd.DataFrame({
        "animal_id": df[cc["animal"]].astype(str),
        "syllable": df[cc["syllable"]].astype(str),
        "duration_ms": pd.to_numeric(df[cc["duration"]], errors="coerce"),
    })
    if cc["epoch"] is not None:
        work["epoch"] = df[cc["epoch"]].astype(str)
    elif cc["relative_day"] is not None:
        rd = pd.to_numeric(df[cc["relative_day"]], errors="coerce")
        work["epoch"] = np.where(rd < 0, pre_group, np.where(rd > 0, post_group, "Treatment day"))
    else:
        raise ValueError("Rendition CSV needs either an epoch/group column or a relative_day column.")

    if cc["relative_day"] is not None:
        work["relative_day"] = pd.to_numeric(df[cc["relative_day"]], errors="coerce")
    else:
        work["relative_day"] = np.nan

    work = work.dropna(subset=["duration_ms"])
    work = work[work["duration_ms"] > 0].copy()

    if metadata is not None and not metadata.empty:
        work = work.merge(metadata[["animal_id", "display_group"]].drop_duplicates("animal_id"), on="animal_id", how="left")
    else:
        work["display_group"] = "unknown"
    work["display_group"] = work["display_group"].fillna("unknown").apply(canonical_group)

    pre_mask = work["epoch"].astype(str).eq(pre_group)
    post_mask = work["epoch"].astype(str).eq(post_group)
    work = work[pre_mask | post_mask].copy()

    split_animal_rows = []
    selection_count_rows = []

    for split_i in range(n_splits):
        for animal, adf in work.groupby("animal_id"):
            display_group = adf["display_group"].dropna().astype(str).iloc[0] if adf["display_group"].notna().any() else "unknown"

            candidate_rows = []
            for syl, sdf in adf.groupby("syllable"):
                pre_vals = sdf.loc[sdf["epoch"].astype(str).eq(pre_group), "duration_ms"].dropna().to_numpy(float)
                post_vals = sdf.loc[sdf["epoch"].astype(str).eq(post_group), "duration_ms"].dropna().to_numpy(float)
                if len(pre_vals) < min_pre_renditions:
                    continue
                if len(post_vals) < (min_train_post_renditions + min_test_post_renditions):
                    continue

                # Split post renditions at random. If relative days are present and enough days exist,
                # split by day to reduce temporal leakage.
                post_sdf = sdf[sdf["epoch"].astype(str).eq(post_group)].copy()
                use_day_split = post_sdf["relative_day"].notna().any() and post_sdf["relative_day"].nunique() >= 4
                if use_day_split:
                    days = np.array(sorted(post_sdf["relative_day"].dropna().unique()), dtype=float)
                    rng.shuffle(days)
                    cut = max(1, len(days) // 2)
                    train_days = set(days[:cut].tolist())
                    test_days = set(days[cut:].tolist())
                    train_vals = post_sdf.loc[post_sdf["relative_day"].isin(train_days), "duration_ms"].dropna().to_numpy(float)
                    test_vals = post_sdf.loc[post_sdf["relative_day"].isin(test_days), "duration_ms"].dropna().to_numpy(float)
                else:
                    idx = rng.permutation(len(post_vals))
                    cut = max(min_train_post_renditions, len(post_vals) // 2)
                    cut = min(cut, len(post_vals) - min_test_post_renditions)
                    train_vals = post_vals[idx[:cut]]
                    test_vals = post_vals[idx[cut:]]

                if len(train_vals) < min_train_post_renditions or len(test_vals) < min_test_post_renditions:
                    continue

                pre_sd_s = float(np.nanstd(pre_vals, ddof=1) / 1000.0)
                train_sd_ms = float(np.nanstd(train_vals, ddof=1))
                test_sd_s = float(np.nanstd(test_vals, ddof=1) / 1000.0)
                candidate_rows.append({
                    "split_i": split_i,
                    "animal_id": animal,
                    "syllable": syl,
                    "display_group": display_group,
                    "rank_metric_train_post_sd_ms": train_sd_ms,
                    "pre_sd_s": pre_sd_s,
                    "post_sd_s": test_sd_s,
                    "sd_delta_s": test_sd_s - pre_sd_s,
                    "n_pre_renditions": len(pre_vals),
                    "n_train_post_renditions": len(train_vals),
                    "n_test_post_renditions": len(test_vals),
                })

            if not candidate_rows:
                continue
            cand = pd.DataFrame(candidate_rows)
            threshold = np.nanpercentile(cand["rank_metric_train_post_sd_ms"], 100.0 * (1.0 - top_frac))
            selected = cand[cand["rank_metric_train_post_sd_ms"] >= threshold].copy()
            if selected.empty:
                continue

            for _, r in selected.iterrows():
                selection_count_rows.append({
                    "animal_id": animal,
                    "syllable": r["syllable"],
                    "display_group": display_group,
                    "n_selected_in_split": 1,
                })

            split_animal_rows.append({
                "split_i": split_i,
                "animal_id": animal,
                "display_group": display_group,
                "n_selected_syllables": int(selected["syllable"].nunique()),
                "pre_sd_s": float(np.nanmean(selected["pre_sd_s"])),
                "post_sd_s": float(np.nanmean(selected["post_sd_s"])),
                "sd_delta_s": float(np.nanmean(selected["sd_delta_s"])),
                "median_n_pre_renditions": float(np.nanmedian(selected["n_pre_renditions"])),
                "median_n_train_post_renditions": float(np.nanmedian(selected["n_train_post_renditions"])),
                "median_n_test_post_renditions": float(np.nanmedian(selected["n_test_post_renditions"])),
            })

    split_df = pd.DataFrame(split_animal_rows)
    if split_df.empty:
        return split_df, pd.DataFrame(selection_count_rows)

    animal = (
        split_df.groupby("animal_id", dropna=False)
        .agg(
            display_group=("display_group", "first"),
            n_splits_contributed=("split_i", "nunique"),
            n_selected_syllables=("n_selected_syllables", "mean"),
            pre_sd_s=("pre_sd_s", "mean"),
            post_sd_s=("post_sd_s", "mean"),
            sd_delta_s=("sd_delta_s", "mean"),
            median_n_pre_renditions=("median_n_pre_renditions", "median"),
            median_n_train_post_renditions=("median_n_train_post_renditions", "median"),
            median_n_test_post_renditions=("median_n_test_post_renditions", "median"),
        )
        .reset_index()
    )

    selection_counts = pd.DataFrame(selection_count_rows)
    if not selection_counts.empty:
        selection_counts = (
            selection_counts.groupby(["animal_id", "syllable", "display_group"], dropna=False)["n_selected_in_split"]
            .sum()
            .reset_index()
        )
        selection_counts["selection_frequency"] = selection_counts["n_selected_in_split"] / float(n_splits)

    return animal, selection_counts


# ──────────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────────
def build_primary_animal_metadata_table(
    primary_selected: pd.DataFrame,
    primary_animal: pd.DataFrame,
    metadata: pd.DataFrame,
    daily_csv: Optional[Path] = None,
) -> pd.DataFrame:
    syl_counts = (
        primary_selected.groupby("animal_id")
        .agg(
            n_selected_syllables=("syllable", "nunique"),
            total_late_pre_phrases_selected=("n_late_pre_phrases", "sum"),
            total_post_phrases_selected=("n_post_phrases", "sum"),
            median_late_pre_phrases_per_syllable=("n_late_pre_phrases", "median"),
            median_post_phrases_per_syllable=("n_post_phrases", "median"),
        )
        .reset_index()
    )

    out = primary_animal.copy()
    out = out.merge(syl_counts, on="animal_id", how="left", suffixes=("", "_from_selected"))

    if metadata is not None and not metadata.empty:
        keep = [c for c in ["animal_id", "raw_hit_type", "display_group", "percent_area_x_lesioned", "treatment_date"] if c in metadata.columns]
        meta2 = metadata[keep].drop_duplicates("animal_id")
        out = out.drop(columns=["display_group"], errors="ignore").merge(meta2, on="animal_id", how="left")

    if daily_csv is not None and Path(daily_csv).exists():
        try:
            daily = pd.read_csv(daily_csv)
            if {"animal_id", "relative_day"}.issubset(daily.columns):
                daily["animal_id"] = daily["animal_id"].astype(str)
                daily["relative_day"] = pd.to_numeric(daily["relative_day"], errors="coerce")
                dd = (
                    daily.dropna(subset=["relative_day"])
                    .groupby("animal_id")
                    .agg(
                        n_pre_days=("relative_day", lambda x: int(np.sum(np.unique(x) < 0))),
                        n_post_days=("relative_day", lambda x: int(np.sum(np.unique(x) > 0))),
                    )
                    .reset_index()
                )
                out = out.merge(dd, on="animal_id", how="left")
        except Exception as e:
            warnings.warn(f"Could not add daily day counts to animal metadata table: {e}")

    ordered = [
        "animal_id",
        "display_group",
        "raw_hit_type",
        "percent_area_x_lesioned",
        "treatment_date",
        "n_selected_syllables",
        "pre_sd_s",
        "post_sd_s",
        "sd_delta_s",
        "total_late_pre_phrases_selected",
        "total_post_phrases_selected",
        "median_late_pre_phrases_per_syllable",
        "median_post_phrases_per_syllable",
        "n_pre_days",
        "n_post_days",
    ]
    ordered = [c for c in ordered if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest].sort_values(["display_group", "animal_id"], kind="stable")


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_robustness_contrast(contrast_df: pd.DataFrame, out_path: Path) -> Path:
    primary = contrast_df[
        contrast_df["contrast_type"].astype(str).str.contains("primary specificity", na=False)
    ].copy()
    if primary.empty:
        return out_path

    order = [
        "Post-selected top 20%",
        "Post-selected top 30%",
        "Post-selected top 40%",
        "Pooled pre+post-selected top 30%",
        "Pre-selected top 30% validation",
        "Held-out post-selected top 30% (day split)",
        "Held-out post-selected top 30% (rendition split)",
    ]
    primary["method_order"] = primary["selection_method"].map({m: i for i, m in enumerate(order)}).fillna(999)
    primary = primary.sort_values("method_order", ascending=False).reset_index(drop=True)

    y = np.arange(len(primary))
    x = primary["observed_delta_difference_s_A_minus_B"].to_numpy(float)
    lo = primary["difference_bootstrap_CI95_low_s"].to_numpy(float)
    hi = primary["difference_bootstrap_CI95_high_s"].to_numpy(float)

    fig_h = max(3.8, 0.55 * len(primary) + 1.2)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)

    xerr_low = np.maximum(0, x - lo)
    xerr_high = np.maximum(0, hi - x)
    ax.errorbar(
        x,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        markersize=5,
        zorder=3,
    )

    # Add p-value labels.
    for xi, yi, p, sig in zip(x, y, primary["labelshuffle_p_A_gt_B"], primary["significance"]):
        txt = f"{sig}, p={format_p(float(p))}"
        ax.text(xi, yi + 0.17, txt, ha="center", va="bottom", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(primary["selection_method"].tolist(), fontsize=10)
    ax.set_xlabel("Medial+lateral − lateral-only Δ phrase duration SD (s)", fontsize=11)
    ax.set_title("Direct anatomical-specificity contrast across selection methods", fontsize=12, pad=10)
    ax.tick_params(axis="x", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def plot_selected_counts(animal_table: pd.DataFrame, out_path: Path) -> Path:
    df = animal_table.copy()
    if "display_group" not in df.columns or "n_selected_syllables" not in df.columns:
        return out_path
    df = df[df["display_group"].isin(PLOTTED_GROUP_ORDER)].copy()
    if df.empty:
        return out_path

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    rng = np.random.default_rng(2)
    for i, group in enumerate(PLOTTED_GROUP_ORDER):
        g = df[df["display_group"] == group]
        if g.empty:
            continue
        vals = g["n_selected_syllables"].to_numpy(float)
        color = GROUP_COLORS.get(group, "gray")
        bp = ax.boxplot([vals], positions=[i], widths=0.45, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.35)
        for item in bp["whiskers"] + bp["caps"] + bp["medians"]:
            item.set_color(color)
            item.set_linewidth(1.4)
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, s=30, alpha=0.85, edgecolors="none")

    ax.set_xticks(range(len(PLOTTED_GROUP_ORDER)))
    ax.set_xticklabels([pretty_group_label(g) for g in PLOTTED_GROUP_ORDER], fontsize=10)
    ax.set_ylabel("Selected syllables per bird", fontsize=11)
    ax.set_title("Primary pooled pre/post-selected top 30% set", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def plot_heldout_selection_frequency(selection_counts: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if selection_counts is None or selection_counts.empty or "selection_frequency" not in selection_counts.columns:
        return None
    df = selection_counts.copy()
    df = df[df["display_group"].isin(PLOTTED_GROUP_ORDER)]
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    rng = np.random.default_rng(3)
    for i, group in enumerate(PLOTTED_GROUP_ORDER):
        g = df[df["display_group"] == group]
        if g.empty:
            continue
        vals = g["selection_frequency"].to_numpy(float)
        color = GROUP_COLORS.get(group, "gray")
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, s=18, alpha=0.55, edgecolors="none")
        med = np.nanmedian(vals)
        ax.plot([i - 0.18, i + 0.18], [med, med], color="black", linewidth=1.5)

    ax.set_xticks(range(len(PLOTTED_GROUP_ORDER)))
    ax.set_xticklabels([pretty_group_label(g) for g in PLOTTED_GROUP_ORDER], fontsize=10)
    ax.set_ylabel("Held-out selection frequency", fontsize=11)
    ax.set_title("Held-out post-selection stability by syllable", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="y", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build robust Figure 3 phrase-duration SD stats, tables, and supplemental plots."
    )
    parser.add_argument("--scatter-csv", required=True, type=Path)
    parser.add_argument("--daily-csv", type=Path, default=None)
    parser.add_argument("--rendition-csv", type=Path, default=None)
    parser.add_argument("--metadata-excel", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)

    parser.add_argument("--pre-group", default="Late Pre")
    parser.add_argument("--post-group", default="Post")
    parser.add_argument("--primary-method", default="Pooled pre+post-selected top 30%")
    parser.add_argument("--top-frac-primary", type=float, default=0.30)

    parser.add_argument("--min-n-phrases", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--stat", choices=["median", "mean"], default="median")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--n-heldout-splits", type=int, default=500)
    parser.add_argument("--heldout-x-min", type=int, default=-30)
    parser.add_argument("--heldout-x-max", type=int, default=30)
    parser.add_argument("--skip-heldout-day", action="store_true")
    parser.add_argument("--skip-xlsx", action="store_true")

    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir)
    plot_dir = ensure_dir(out_dir / "plots")
    rng = np.random.default_rng(args.seed)

    print("[INFO] Loading metadata...")
    metadata = load_metadata(args.metadata_excel) if args.metadata_excel else pd.DataFrame(columns=["animal_id", "display_group"])

    print("[INFO] Preparing pre/post summary table...")
    prepost = prepare_prepost_table(
        args.scatter_csv,
        metadata,
        pre_group=args.pre_group,
        post_group=args.post_group,
        min_n_phrases=args.min_n_phrases,
    )
    prepost.to_csv(out_dir / "prepost_animal_syllable_all_candidates.csv", index=False)

    group_summaries = []
    contrasts = []
    animal_tables = []
    selected_tables = []

    primary_selected = None
    primary_animal = None

    for run in RUNS:
        print(f"[INFO] Running {run['selection_method']}...")
        selected = select_top_syllables(prepost, run["rank_on"], run["top_frac"])
        selected["selection_method"] = run["selection_method"]
        selected["selection_basis"] = run["selection_basis"]
        selected["selection_fraction"] = run["selection_fraction"]
        selected["source"] = run["source"]

        animal = animal_level_from_selected(selected)
        animal["selection_method"] = run["selection_method"]
        animal["selection_basis"] = run["selection_basis"]
        animal["selection_fraction"] = run["selection_fraction"]
        animal["source"] = run["source"]

        method_info = {
            "selection_method": run["selection_method"],
            "selection_basis": run["selection_basis"],
            "selection_fraction": run["selection_fraction"],
            "source": run["source"],
        }
        gs, ct = compute_group_stats(
            animal,
            method_info=method_info,
            rng=rng,
            n_boot=args.n_bootstrap,
            n_perm=args.n_permutations,
            stat=args.stat,
        )

        group_summaries.append(gs)
        contrasts.append(ct)
        animal_tables.append(animal)
        selected_tables.append(selected)

        if run["selection_method"] == args.primary_method:
            primary_selected = selected.copy()
            primary_animal = animal.copy()

    # Held-out validation: prefer rendition-level if provided; also run day-level if daily_csv provided.
    heldout_selection_counts = []
    if args.rendition_csv is not None and args.rendition_csv.exists():
        print("[INFO] Running held-out post-selection from rendition-level table...")
        try:
            h_animal, h_counts = heldout_rendition_validation(
                args.rendition_csv,
                metadata,
                rng=rng,
                n_splits=args.n_heldout_splits,
                top_frac=args.top_frac_primary,
                pre_group=args.pre_group,
                post_group=args.post_group,
            )
            if not h_animal.empty:
                method_info = {
                    "selection_method": "Held-out post-selected top 30% (rendition split)",
                    "selection_basis": "held-out post-lesion validation",
                    "selection_fraction": "top 30%",
                    "source": "rendition-level table",
                }
                h_animal["selection_method"] = method_info["selection_method"]
                h_animal["selection_basis"] = method_info["selection_basis"]
                h_animal["selection_fraction"] = method_info["selection_fraction"]
                h_animal["source"] = method_info["source"]
                gs, ct = compute_group_stats(
                    h_animal,
                    method_info=method_info,
                    rng=rng,
                    n_boot=args.n_bootstrap,
                    n_perm=args.n_permutations,
                    stat=args.stat,
                )
                group_summaries.append(gs)
                contrasts.append(ct)
                animal_tables.append(h_animal)
                h_animal.to_csv(out_dir / "heldout_rendition_split_animal_level.csv", index=False)
                if not h_counts.empty:
                    h_counts.to_csv(out_dir / "heldout_rendition_split_selection_frequencies.csv", index=False)
                    heldout_selection_counts.append(("rendition", h_counts))
        except Exception as e:
            warnings.warn(f"Rendition-level held-out validation failed and was skipped: {e}")

    if args.daily_csv is not None and args.daily_csv.exists() and not args.skip_heldout_day:
        print("[INFO] Running held-out post-selection from day-level daily variance table...")
        try:
            h_animal, h_counts = heldout_day_split_validation(
                args.daily_csv,
                metadata,
                rng=rng,
                n_splits=args.n_heldout_splits,
                top_frac=args.top_frac_primary,
                x_min=args.heldout_x_min,
                x_max=args.heldout_x_max,
            )
            if not h_animal.empty:
                method_info = {
                    "selection_method": "Held-out post-selected top 30% (day split)",
                    "selection_basis": "held-out post-lesion day validation",
                    "selection_fraction": "top 30%",
                    "source": "daily variance table",
                }
                h_animal["selection_method"] = method_info["selection_method"]
                h_animal["selection_basis"] = method_info["selection_basis"]
                h_animal["selection_fraction"] = method_info["selection_fraction"]
                h_animal["source"] = method_info["source"]
                gs, ct = compute_group_stats(
                    h_animal,
                    method_info=method_info,
                    rng=rng,
                    n_boot=args.n_bootstrap,
                    n_perm=args.n_permutations,
                    stat=args.stat,
                )
                group_summaries.append(gs)
                contrasts.append(ct)
                animal_tables.append(h_animal)
                h_animal.to_csv(out_dir / "heldout_day_split_animal_level.csv", index=False)
                if not h_counts.empty:
                    h_counts.to_csv(out_dir / "heldout_day_split_selection_frequencies.csv", index=False)
                    heldout_selection_counts.append(("day", h_counts))
        except Exception as e:
            warnings.warn(f"Day-level held-out validation failed and was skipped: {e}")

    group_summary = pd.concat(group_summaries, ignore_index=True) if group_summaries else pd.DataFrame()
    contrast_summary = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    all_animals = pd.concat(animal_tables, ignore_index=True) if animal_tables else pd.DataFrame()
    all_selected = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()

    # Primary animal metadata/sample table.
    if primary_selected is None or primary_animal is None:
        warnings.warn(f"Primary method {args.primary_method!r} not found; using pooled pre+post-selected top 30%.")
        run = [r for r in RUNS if r["selection_method"] == "Pooled pre+post-selected top 30%"][0]
        primary_selected = select_top_syllables(prepost, run["rank_on"], run["top_frac"])
        primary_animal = animal_level_from_selected(primary_selected)

    animal_metadata = build_primary_animal_metadata_table(primary_selected, primary_animal, metadata, daily_csv=args.daily_csv)

    # Combined copy-paste table: group summary and primary contrasts in one CSV.
    group_block = group_summary.copy()
    contrast_block = contrast_summary.copy()
    group_block.insert(0, "table_section", "Group summary")
    contrast_block.insert(0, "table_section", "Pairwise contrast")
    combined = pd.concat([group_block, contrast_block], ignore_index=True, sort=False)

    # Save tables.
    out_group = out_dir / "Supplemental_Table_1_selection_robustness_group_summary.csv"
    out_contrast = out_dir / "Supplemental_Table_1_selection_robustness_pairwise_contrasts.csv"
    out_combined = out_dir / "Supplemental_Table_1_selection_robustness_combined_for_copy_paste.csv"
    out_animal = out_dir / "Supplemental_Table_2_animal_metadata_and_sample_sizes.csv"
    out_all_animals = out_dir / "all_selection_methods_animal_level.csv"
    out_all_selected = out_dir / "all_selection_methods_selected_animal_syllable_pairs.csv"

    group_summary.to_csv(out_group, index=False)
    contrast_summary.to_csv(out_contrast, index=False)
    combined.to_csv(out_combined, index=False)
    animal_metadata.to_csv(out_animal, index=False)
    all_animals.to_csv(out_all_animals, index=False)
    all_selected.to_csv(out_all_selected, index=False)

    # Save plots.
    print("[INFO] Writing supplemental plots...")
    contrast_plot = plot_robustness_contrast(contrast_summary, plot_dir / "Supplemental_Fig_selection_robustness_ML_vs_lateral_contrast.png")
    count_plot = plot_selected_counts(animal_metadata, plot_dir / "Supplemental_Fig_primary_selected_syllable_counts_by_bird.png")

    for label, counts in heldout_selection_counts:
        plot_heldout_selection_frequency(
            counts,
            plot_dir / f"Supplemental_Fig_heldout_{label}_selection_frequency.png",
        )

    # Optional Excel workbook for copy/paste convenience.
    if not args.skip_xlsx:
        xlsx_path = out_dir / "Figure3_robust_stats_tables.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            group_summary.to_excel(writer, sheet_name="Table1_group_summary", index=False)
            contrast_summary.to_excel(writer, sheet_name="Table1_contrasts", index=False)
            combined.to_excel(writer, sheet_name="Table1_combined", index=False)
            animal_metadata.to_excel(writer, sheet_name="Table2_animals", index=False)
            all_animals.to_excel(writer, sheet_name="All_animal_level", index=False)
            all_selected.to_excel(writer, sheet_name="Selected_pairs", index=False)
            if (out_dir / "heldout_day_split_animal_level.csv").exists():
                pd.read_csv(out_dir / "heldout_day_split_animal_level.csv").to_excel(writer, sheet_name="Heldout_day_animals", index=False)
            if (out_dir / "heldout_rendition_split_animal_level.csv").exists():
                pd.read_csv(out_dir / "heldout_rendition_split_animal_level.csv").to_excel(writer, sheet_name="Heldout_rend_animals", index=False)

    # Print key rows for quick review.
    print("\n[OK] Wrote tables:")
    for p in [out_group, out_contrast, out_combined, out_animal, out_all_animals, out_all_selected]:
        print(f"  {p}")
    if not args.skip_xlsx:
        print(f"  {out_dir / 'Figure3_robust_stats_tables.xlsx'}")

    print("\n[OK] Wrote plots:")
    print(f"  {contrast_plot}")
    print(f"  {count_plot}")

    print("\nKey primary specificity contrast rows:")
    if not contrast_summary.empty:
        key = contrast_summary[contrast_summary["contrast_type"].astype(str).str.contains("primary specificity", na=False)]
        cols = [
            "selection_method",
            "source",
            "observed_delta_difference_s_A_minus_B",
            "difference_bootstrap_CI95_low_s",
            "difference_bootstrap_CI95_high_s",
            "labelshuffle_p_A_gt_B",
            "significance",
        ]
        cols = [c for c in cols if c in key.columns]
        print(key[cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
