#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seasonal_contemporaneous_sham_vs_ML_v1.py

Supplemental seasonal-control analysis for Figure 3 phrase-duration SD results.

Starting point
--------------
Adapted from plot_top30_variance_from_csv_v22.py. The older script plotted
top-30% daily phrase-duration variance trajectories aligned to lesion day and
handled lesion-hit-type metadata. This version adds a reviewer-facing seasonal
control:

1) Use the same selected animal × syllable pairs as Figure 3 when provided.
2) Convert daily phrase-duration variance to SD in seconds.
3) Compute animal-level pre/post ΔSD from daily summaries.
4) Plot ΔSD by actual treatment calendar date.
5) Match each sham bird to the nearest-in-calendar-time medial+lateral lesion
   bird and compare their animal-level ΔSD values.
6) Output matched-pair tables and stats.

Recommended input
-----------------
Use the all-syllable daily variance table plus the Figure 3 selected-pairs CSV:

  --daily-csv .../batch_aligned_phrase_duration_variance_all.csv
  --selected-pairs-csv .../panel_C_selected_animal_syllable_pairs.csv

If --selected-pairs-csv is omitted, the script uses all syllable/day rows in
the daily table. For consistency with Figure 3, provide the selected-pairs CSV.

Outputs
-------
- seasonal_control_combined.png/pdf
- seasonal_control_delta_by_treatment_date.png/pdf
- seasonal_control_matched_pairs_delta_sd.png/pdf
- seasonal_control_animal_day_summary.csv
- seasonal_control_animal_delta_sd.csv
- seasonal_control_matched_pairs.csv
- seasonal_control_matched_pair_stats.csv
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
import matplotlib.dates as mdates

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


# ──────────────────────────────────────────────────────────────────────────────
# Labels/colors, kept close to recent Figure 3 palette
# ──────────────────────────────────────────────────────────────────────────────
SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL_ML = "Partial Medial and Lateral lesion"
COMPLETE_ML = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"

GROUP_COLORS = {
    SHAM: "#1B9E77",
    LATERAL: "#A88BD9",
    PARTIAL_ML: "#7A4FB7",
    COMPLETE_ML: "#3F007D",
    POOLED_ML: "#6F4AB6",
    "Sham": "#1B9E77",
    "Medial+lateral lesion": "#6F4AB6",
    "unknown": "#4D4D4D",
}

PLOT_GROUP_ORDER = ["Sham", "Medial+lateral lesion"]


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def lower(x: Any) -> str:
    return clean_text(x).lower()


def canonical_group(raw: Any) -> str:
    """Map metadata/daily lesion labels into current Figure 3 hit-type groups."""
    s = lower(raw)
    if not s or s == "nan":
        return "unknown"

    if "sham" in s or ("saline" in s and "lesion" not in s):
        return SHAM

    if (
        "lateral lesion only" in s
        or "lateral only" in s
        or "lateral hit only" in s
        or "area x visible (single hit)" in s
        or "single lateral hit" in s
        or s == "single hit"
    ):
        return LATERAL

    if (
        "complete" in s and "medial" in s and "lateral" in s
    ) or "large lesion" in s or "not visible" in s:
        return COMPLETE_ML

    if (
        "partial" in s and "medial" in s and "lateral" in s
    ) or "area x visible (medial+lateral hit)" in s or "medial+lateral" in s or "m+l" in s:
        return PARTIAL_ML

    return clean_text(raw) or "unknown"


def coarse_group(display_group: str) -> str:
    if display_group == SHAM:
        return "Sham"
    if display_group in {PARTIAL_ML, COMPLETE_ML, POOLED_ML}:
        return "Medial+lateral lesion"
    return "Other"


def find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    exact = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in exact:
            return exact[cand.lower()]
    for cand in candidates:
        c0 = cand.lower()
        for c in cols:
            if c0 in str(c).lower():
                return c
    return None


def format_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────────────
def load_metadata(
    metadata_excel: Path,
    *,
    animal_col: str = "Animal ID",
    preferred_hit_sheet: str = "animal_hit_type_summary",
    hit_type_col: str = "Lesion hit type",
    treatment_date_col: str = "Treatment date",
) -> pd.DataFrame:
    """Load animal ID, treatment date, and lesion hit type.

    Hit types and treatment dates may be stored on different sheets, so this
    function searches the workbook and merges the best available information.
    """
    metadata_excel = Path(metadata_excel)
    if not metadata_excel.exists():
        raise FileNotFoundError(f"metadata_excel not found: {metadata_excel}")

    xls = pd.ExcelFile(metadata_excel)

    # Hit type sheet/column.
    hit_records = []
    hit_df_used = None
    hit_sheet_used = None
    hit_col_used = None

    hit_candidates = [
        hit_type_col,
        "Lesion hit type",
        "lesion hit type",
        "Hit type",
        "Hit Type",
        "lesion_hit_type",
        "lesion hit type grouping",
        "Group",
        "Treatment type",
    ]

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(metadata_excel, sheet_name=sheet)
        except Exception:
            continue
        if animal_col not in df.columns:
            continue
        possible_cols = [c for c in hit_candidates if c in df.columns]
        for c in df.columns:
            cl = str(c).lower()
            if c not in possible_cols and (("hit" in cl and "type" in cl) or ("lesion" in cl and "group" in cl)):
                possible_cols.append(c)
        for col in possible_cols:
            groups = df[col].apply(canonical_group)
            n_known = int(groups.isin([SHAM, LATERAL, PARTIAL_ML, COMPLETE_ML]).sum())
            n_ml = int(groups.isin([PARTIAL_ML, COMPLETE_ML]).sum())
            bonus = 50 if sheet == preferred_hit_sheet else 0
            if "treatment type" in str(col).lower():
                bonus -= 40
            score = n_known + n_ml + bonus
            hit_records.append((score, sheet, col, df.copy()))

    if not hit_records:
        raise ValueError("Could not find a metadata sheet with animal IDs and lesion hit types.")

    hit_records.sort(key=lambda x: x[0], reverse=True)
    _, hit_sheet_used, hit_col_used, hit_df_used = hit_records[0]

    hit_out = pd.DataFrame({
        "animal_id": hit_df_used[animal_col].astype(str).str.strip(),
        "raw_hit_type": hit_df_used[hit_col_used],
    })
    hit_out["display_group"] = hit_out["raw_hit_type"].apply(canonical_group)
    hit_out = hit_out.dropna(subset=["animal_id"]).drop_duplicates("animal_id")

    # Treatment date sheet/column.
    date_records = []
    date_candidates = [treatment_date_col, "Treatment date", "Lesion date", "treatment_date", "surgery date"]
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(metadata_excel, sheet_name=sheet)
        except Exception:
            continue
        if animal_col not in df.columns:
            continue
        col = find_col(list(df.columns), date_candidates)
        if col is None:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        n_dates = int(parsed.notna().sum())
        bonus = 10 if sheet.lower() == "metadata" else 0
        date_records.append((n_dates + bonus, sheet, col, df.copy()))

    if not date_records:
        raise ValueError("Could not find treatment dates in metadata workbook.")

    date_records.sort(key=lambda x: x[0], reverse=True)
    _, date_sheet_used, date_col_used, date_df = date_records[0]

    date_out = pd.DataFrame({
        "animal_id": date_df[animal_col].astype(str).str.strip(),
        "treatment_date": pd.to_datetime(date_df[date_col_used], errors="coerce"),
    }).drop_duplicates("animal_id")

    # Optional percent lesion.
    pct_col = find_col(
        list(hit_df_used.columns),
        ["% Area X lesioned", "Percent Area X lesioned", "Area X lesioned", "Area X lesion %"],
    )
    if pct_col is not None:
        hit_out["percent_area_x_lesioned"] = hit_df_used[pct_col]

    out = hit_out.merge(date_out, on="animal_id", how="left")
    out["coarse_group"] = out["display_group"].apply(coarse_group)

    print(f"[INFO] Hit type source: sheet={hit_sheet_used!r}, column={hit_col_used!r}")
    print(f"[INFO] Treatment date source: sheet={date_sheet_used!r}, column={date_col_used!r}")
    print("[INFO] Metadata group counts:")
    print(out["display_group"].value_counts(dropna=False).to_string())
    print("[INFO] Coarse seasonal-control group counts:")
    print(out["coarse_group"].value_counts(dropna=False).to_string())

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Daily table preparation
# ──────────────────────────────────────────────────────────────────────────────
def resolve_daily_cols(df: pd.DataFrame) -> dict[str, str]:
    cols = list(df.columns)
    animal = find_col(cols, ["animal_id", "Animal ID", "Animal", "bird", "Bird"])
    syllable = find_col(cols, ["syllable", "Syllable", "label", "cluster", "hdbscan_label"])
    rel_day = find_col(cols, ["relative_day", "Relative day", "days_relative_to_lesion", "day_relative_to_lesion"])
    variance = find_col(cols, ["Variance (ms^2)", "variance_ms2", "Variance_ms2", "variance", "Variance"])
    date = find_col(cols, ["date", "Date", "recording_date", "Recording date"])
    n_col = find_col(cols, ["N_phrases", "n_phrases", "occurrences", "count", "n", "N"])

    missing = []
    if animal is None:
        missing.append("animal_id")
    if syllable is None:
        missing.append("syllable")
    if rel_day is None:
        missing.append("relative_day")
    if variance is None:
        missing.append("Variance (ms^2)")
    if missing:
        raise ValueError(f"daily CSV is missing required columns or aliases: {missing}. Found: {cols}")

    return {"animal": animal, "syllable": syllable, "relative_day": rel_day, "variance": variance, "date": date, "n": n_col}


def load_selected_pairs(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"selected pairs CSV not found: {path}")
    df = pd.read_csv(path)
    a_col = find_col(list(df.columns), ["animal_id", "Animal ID", "Animal"])
    s_col = find_col(list(df.columns), ["syllable", "Syllable"])
    if a_col is None or s_col is None:
        raise ValueError(f"selected pairs CSV needs animal_id and syllable columns. Found: {list(df.columns)}")
    out = pd.DataFrame({
        "animal_id": df[a_col].astype(str),
        "syllable": df[s_col].astype(str),
    }).drop_duplicates()
    return out


def prepare_daily_table(
    daily_csv: Path,
    metadata: pd.DataFrame,
    *,
    selected_pairs_csv: Optional[Path] = None,
    x_min: int = -30,
    x_max: int = 30,
    min_daily_n: int = 1,
) -> pd.DataFrame:
    df = pd.read_csv(daily_csv)
    cc = resolve_daily_cols(df)

    work = pd.DataFrame({
        "animal_id": df[cc["animal"]].astype(str),
        "syllable": df[cc["syllable"]].astype(str),
        "relative_day": pd.to_numeric(df[cc["relative_day"]], errors="coerce"),
        "variance_ms2": pd.to_numeric(df[cc["variance"]], errors="coerce"),
    })
    if cc["date"] is not None:
        work["recording_date_raw"] = pd.to_datetime(df[cc["date"]], errors="coerce")
    else:
        work["recording_date_raw"] = pd.NaT
    if cc["n"] is not None:
        work["daily_n"] = pd.to_numeric(df[cc["n"]], errors="coerce")
    else:
        work["daily_n"] = np.nan

    work = work.dropna(subset=["relative_day", "variance_ms2"])
    work = work[(work["relative_day"] >= x_min) & (work["relative_day"] <= x_max)].copy()
    work = work[work["variance_ms2"] > 0].copy()

    if min_daily_n is not None and min_daily_n > 1 and "daily_n" in work.columns:
        before = len(work)
        work = work[work["daily_n"].fillna(0) >= int(min_daily_n)].copy()
        print(f"[INFO] Applied min_daily_n={min_daily_n}: kept {len(work):,}/{before:,} rows")

    selected_pairs = load_selected_pairs(selected_pairs_csv)
    if selected_pairs is not None:
        before_rows = len(work)
        before_pairs = work[["animal_id", "syllable"]].drop_duplicates().shape[0]
        work = work.merge(selected_pairs, on=["animal_id", "syllable"], how="inner")
        after_pairs = work[["animal_id", "syllable"]].drop_duplicates().shape[0]
        print(
            f"[INFO] Filtered to selected Figure 3 pairs: "
            f"{len(work):,}/{before_rows:,} rows; {after_pairs}/{before_pairs} animal×syllable pairs retained"
        )

    meta_cols = ["animal_id", "display_group", "coarse_group", "treatment_date", "raw_hit_type"]
    extra = [c for c in ["percent_area_x_lesioned"] if c in metadata.columns]
    work = work.merge(metadata[meta_cols + extra].drop_duplicates("animal_id"), on="animal_id", how="left")

    work["display_group"] = work["display_group"].fillna("unknown").apply(canonical_group)
    work["coarse_group"] = work["display_group"].apply(coarse_group)
    work["treatment_date"] = pd.to_datetime(work["treatment_date"], errors="coerce")

    # Derive recording date from treatment_date + relative_day if explicit dates absent.
    work["recording_date"] = work["recording_date_raw"]
    missing_date = work["recording_date"].isna()
    work.loc[missing_date, "recording_date"] = (
        work.loc[missing_date, "treatment_date"]
        + pd.to_timedelta(work.loc[missing_date, "relative_day"].round().astype("Int64"), unit="D")
    )

    work["sd_s"] = np.sqrt(work["variance_ms2"]) / 1000.0

    # Restrict to sham and ML for the seasonal-control figure.
    keep = work["coarse_group"].isin(PLOT_GROUP_ORDER)
    dropped = int((~keep).sum())
    if dropped:
        print(f"[INFO] Dropping {dropped:,} rows not in Sham or medial+lateral groups for seasonal-control plots.")
    work = work[keep].copy()

    if work.empty:
        raise ValueError("No sham or medial+lateral rows remain after filtering.")

    return work.sort_values(["coarse_group", "animal_id", "syllable", "relative_day"]).reset_index(drop=True)


def make_animal_day_summary(daily: pd.DataFrame, *, daily_stat: str = "median") -> pd.DataFrame:
    if daily_stat not in {"median", "mean"}:
        raise ValueError("--daily-stat must be median or mean")
    func = "median" if daily_stat == "median" else "mean"

    out = (
        daily.groupby(["animal_id", "coarse_group", "display_group", "raw_hit_type", "treatment_date", "recording_date", "relative_day"], dropna=False)
        .agg(
            daily_sd_s=("sd_s", func),
            n_syllable_day_rows=("sd_s", "size"),
            n_syllables=("syllable", "nunique"),
        )
        .reset_index()
    )
    return out


def make_animal_delta_table(animal_day: pd.DataFrame, *, epoch_stat: str = "median") -> pd.DataFrame:
    if epoch_stat not in {"median", "mean"}:
        raise ValueError("--epoch-stat must be median or mean")

    rows = []
    for animal, g in animal_day.groupby("animal_id", dropna=False):
        pre = g[g["relative_day"] < 0]["daily_sd_s"].dropna().to_numpy(float)
        post = g[g["relative_day"] > 0]["daily_sd_s"].dropna().to_numpy(float)
        if len(pre) == 0 or len(post) == 0:
            continue

        val = np.nanmedian if epoch_stat == "median" else np.nanmean
        rows.append({
            "animal_id": animal,
            "coarse_group": g["coarse_group"].iloc[0],
            "display_group": g["display_group"].iloc[0],
            "raw_hit_type": g["raw_hit_type"].iloc[0],
            "treatment_date": g["treatment_date"].iloc[0],
            "pre_sd_s": float(val(pre)),
            "post_sd_s": float(val(post)),
            "delta_sd_s": float(val(post) - val(pre)),
            "days_recorded_pre": int(g.loc[g["relative_day"] < 0, "relative_day"].nunique()),
            "days_recorded_post": int(g.loc[g["relative_day"] > 0, "relative_day"].nunique()),
            "calendar_ordinal": float(pd.Timestamp(g["treatment_date"].iloc[0]).toordinal()) if pd.notna(g["treatment_date"].iloc[0]) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Could not compute animal-level delta table.")
    return out.sort_values(["coarse_group", "treatment_date", "animal_id"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Matching/stats
# ──────────────────────────────────────────────────────────────────────────────
def match_sham_to_ml(animal_delta: pd.DataFrame, *, unique_ml: bool = True) -> pd.DataFrame:
    sham = animal_delta[animal_delta["coarse_group"] == "Sham"].copy()
    ml = animal_delta[animal_delta["coarse_group"] == "Medial+lateral lesion"].copy()

    sham = sham.dropna(subset=["treatment_date", "delta_sd_s"]).reset_index(drop=True)
    ml = ml.dropna(subset=["treatment_date", "delta_sd_s"]).reset_index(drop=True)

    if sham.empty or ml.empty:
        raise ValueError("Need at least one sham and one medial+lateral bird with treatment date and delta SD.")

    sham_dates = np.array([pd.Timestamp(d).toordinal() for d in sham["treatment_date"]], dtype=float)
    ml_dates = np.array([pd.Timestamp(d).toordinal() for d in ml["treatment_date"]], dtype=float)
    cost = np.abs(sham_dates[:, None] - ml_dates[None, :])

    pairs = []
    if unique_ml and len(ml) >= len(sham) and linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
        # Ensure every sham gets one ML; if more rows than shams, row_ind is all shams.
        for r, c in zip(row_ind, col_ind):
            if r >= len(sham):
                continue
            pairs.append((r, c))
    elif unique_ml and len(ml) >= len(sham):
        used = set()
        # Greedy fallback ordered by globally shortest distances.
        flat = [(cost[i, j], i, j) for i in range(len(sham)) for j in range(len(ml))]
        flat.sort(key=lambda x: x[0])
        matched_shams = set()
        for _, i, j in flat:
            if i in matched_shams or j in used:
                continue
            pairs.append((i, j))
            matched_shams.add(i)
            used.add(j)
            if len(matched_shams) == len(sham):
                break
    else:
        # With replacement: nearest ML for each sham.
        for i in range(len(sham)):
            pairs.append((i, int(np.argmin(cost[i]))))

    rows = []
    for pair_i, (si, mi) in enumerate(pairs, start=1):
        s = sham.iloc[si]
        m = ml.iloc[mi]
        days_apart = abs((pd.Timestamp(m["treatment_date"]) - pd.Timestamp(s["treatment_date"])).days)
        rows.append({
            "pair_id": pair_i,
            "sham_animal_id": s["animal_id"],
            "sham_treatment_date": pd.Timestamp(s["treatment_date"]).date().isoformat(),
            "sham_delta_sd_s": float(s["delta_sd_s"]),
            "ml_animal_id": m["animal_id"],
            "ml_treatment_date": pd.Timestamp(m["treatment_date"]).date().isoformat(),
            "ml_hit_type": m["raw_hit_type"],
            "ml_display_group": m["display_group"],
            "ml_delta_sd_s": float(m["delta_sd_s"]),
            "ml_minus_sham_delta_sd_s": float(m["delta_sd_s"] - s["delta_sd_s"]),
            "treatment_dates_days_apart": int(days_apart),
        })
    return pd.DataFrame(rows)


def paired_signflip_p_greater(diff: np.ndarray, *, n_perm: int = 10000, seed: int = 123) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan
    obs = np.nanmean(diff)

    if len(diff) <= 16:
        vals = []
        for signs in itertools.product([-1, 1], repeat=len(diff)):
            vals.append(np.nanmean(diff * np.asarray(signs)))
        vals = np.asarray(vals)
        return float((np.sum(vals >= obs - 1e-15) + 1) / (len(vals) + 1))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        if np.nanmean(diff * signs) >= obs - 1e-15:
            count += 1
    return float((count + 1) / (n_perm + 1))


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 5000, seed: int = 123) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(np.nanmean(sample))
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def matched_pair_stats(matched: pd.DataFrame, *, n_boot: int = 5000, n_perm: int = 10000, seed: int = 123) -> pd.DataFrame:
    diff = matched["ml_minus_sham_delta_sd_s"].to_numpy(float)
    mean_diff = float(np.nanmean(diff))
    median_diff = float(np.nanmedian(diff))
    ci_low, ci_high = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    signflip_p = paired_signflip_p_greater(diff, n_perm=n_perm, seed=seed)

    if scipy_stats is not None and len(diff) >= 2:
        t_res = scipy_stats.ttest_1samp(diff, popmean=0, alternative="greater")
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)
    else:
        t_stat = np.nan
        t_p = np.nan

    return pd.DataFrame([{
        "comparison": "nearest-date matched ML minus sham ΔSD",
        "n_pairs": int(len(diff)),
        "mean_ML_minus_sham_delta_sd_s": mean_diff,
        "median_ML_minus_sham_delta_sd_s": median_diff,
        "mean_difference_bootstrap_CI95_low_s": ci_low,
        "mean_difference_bootstrap_CI95_high_s": ci_high,
        "paired_t_greater": t_stat,
        "paired_t_p_greater": t_p,
        "paired_signflip_p_greater": signflip_p,
        "median_days_between_matched_treatment_dates": float(np.nanmedian(matched["treatment_dates_days_apart"])),
        "max_days_between_matched_treatment_dates": int(np.nanmax(matched["treatment_dates_days_apart"])),
    }])


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def setup_axis(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def plot_delta_by_treatment_date(animal_delta: pd.DataFrame, out_path: Path) -> Path:
    df = animal_delta[animal_delta["coarse_group"].isin(PLOT_GROUP_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)

    for group in PLOT_GROUP_ORDER:
        g = df[df["coarse_group"] == group].copy()
        if g.empty:
            continue
        ax.scatter(
            g["treatment_date"],
            g["delta_sd_s"],
            s=52,
            color=GROUP_COLORS[group],
            alpha=0.85,
            edgecolors="none",
            label=f"{group} (n={g['animal_id'].nunique()})",
            zorder=3,
        )
        for _, r in g.iterrows():
            ax.text(
                r["treatment_date"],
                r["delta_sd_s"],
                f" {r['animal_id']}",
                fontsize=7.5,
                color=GROUP_COLORS[group],
                ha="left",
                va="center",
            )

    ax.set_ylabel("Δ phrase duration SD (s)", fontsize=11)
    ax.set_xlabel("Treatment date", fontsize=11)
    ax.set_title("Seasonal control: animal-level ΔSD by treatment date", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30, ha="right")
    setup_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def plot_matched_pairs(matched: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)

    x_sham, x_ml = 0.0, 1.0
    for _, r in matched.iterrows():
        ax.plot(
            [x_sham, x_ml],
            [r["sham_delta_sd_s"], r["ml_delta_sd_s"]],
            color="0.55",
            linewidth=1.0,
            alpha=0.75,
            zorder=1,
        )
        ax.scatter([x_sham], [r["sham_delta_sd_s"]], color=GROUP_COLORS["Sham"], s=48, edgecolors="none", zorder=3)
        ax.scatter([x_ml], [r["ml_delta_sd_s"]], color=GROUP_COLORS["Medial+lateral lesion"], s=48, edgecolors="none", zorder=3)

    # Mean ± SEM overlay.
    for x, col, color in [
        (x_sham, "sham_delta_sd_s", GROUP_COLORS["Sham"]),
        (x_ml, "ml_delta_sd_s", GROUP_COLORS["Medial+lateral lesion"]),
    ]:
        vals = matched[col].to_numpy(float)
        mean = np.nanmean(vals)
        sem = np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals))) if np.sum(np.isfinite(vals)) > 1 else 0
        ax.errorbar([x], [mean], yerr=[sem], fmt="o", color="black", ecolor="black", capsize=4, markersize=5, zorder=4)
        ax.plot([x - 0.15, x + 0.15], [mean, mean], color=color, linewidth=2.2, zorder=4)

    ax.set_xticks([x_sham, x_ml])
    ax.set_xticklabels(["Nearest-date\nsham", "Nearest-date\nmedial+lateral"], fontsize=10)
    ax.set_ylabel("Δ phrase duration SD (s)", fontsize=11)
    ax.set_title("Nearest-date matched seasonal control", fontsize=12)
    setup_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def plot_calendar_timecourse(animal_day: pd.DataFrame, out_path: Path) -> Path:
    """Optional trajectory view by actual recording date."""
    df = animal_day[animal_day["coarse_group"].isin(PLOT_GROUP_ORDER)].copy()

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for (animal, group), g in df.groupby(["animal_id", "coarse_group"], dropna=False):
        g = g.sort_values("recording_date")
        ax.plot(
            g["recording_date"],
            g["daily_sd_s"],
            color=GROUP_COLORS[group],
            alpha=0.38,
            linewidth=1.2,
            marker="o",
            markersize=2.8,
            markeredgewidth=0,
        )

    # Group median by calendar month, if enough points.
    tmp = df.copy()
    tmp["month"] = tmp["recording_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        tmp.groupby(["coarse_group", "month"], dropna=False)
        .agg(monthly_sd_s=("daily_sd_s", "median"))
        .reset_index()
    )
    for group in PLOT_GROUP_ORDER:
        g = monthly[monthly["coarse_group"] == group].sort_values("month")
        if len(g) >= 2:
            ax.plot(
                g["month"],
                g["monthly_sd_s"],
                color=GROUP_COLORS[group],
                linewidth=2.5,
                label=f"{group} monthly median",
            )

    ax.set_ylabel("Daily phrase duration SD (s)", fontsize=11)
    ax.set_xlabel("Recording date", fontsize=11)
    ax.set_title("Daily phrase duration SD by calendar date", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30, ha="right")
    setup_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def plot_combined_figure(animal_delta: pd.DataFrame, matched: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path) -> Path:
    df = animal_delta[animal_delta["coarse_group"].isin(PLOT_GROUP_ORDER)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    for group in PLOT_GROUP_ORDER:
        g = df[df["coarse_group"] == group].copy()
        if g.empty:
            continue
        ax.scatter(
            g["treatment_date"],
            g["delta_sd_s"],
            s=48,
            color=GROUP_COLORS[group],
            alpha=0.85,
            edgecolors="none",
            label=f"{group} (n={g['animal_id'].nunique()})",
            zorder=3,
        )
    ax.set_ylabel("Δ phrase duration SD (s)", fontsize=11)
    ax.set_xlabel("Treatment date", fontsize=11)
    ax.set_title("A. ΔSD by time of year", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)
    setup_axis(ax)

    ax = axes[1]
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    for _, r in matched.iterrows():
        ax.plot([0, 1], [r["sham_delta_sd_s"], r["ml_delta_sd_s"]], color="0.55", linewidth=1.0, alpha=0.75)
        ax.scatter([0], [r["sham_delta_sd_s"]], color=GROUP_COLORS["Sham"], s=48, edgecolors="none", zorder=3)
        ax.scatter([1], [r["ml_delta_sd_s"]], color=GROUP_COLORS["Medial+lateral lesion"], s=48, edgecolors="none", zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nearest-date\nsham", "Nearest-date\nmedial+lateral"], fontsize=9)
    ax.set_ylabel("Δ phrase duration SD (s)", fontsize=11)
    p = stats_df["paired_signflip_p_greater"].iloc[0] if "paired_signflip_p_greater" in stats_df.columns and not stats_df.empty else np.nan
    ax.set_title(f"B. Matched comparison\nsign-flip p={format_p(float(p))}", fontsize=12)
    setup_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Make supplemental seasonal-control plots comparing sham vs contemporaneous M+L lesions.")
    p.add_argument("--daily-csv", required=True, type=Path)
    p.add_argument("--metadata-excel", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--selected-pairs-csv", type=Path, default=None, help="Optional Figure 3 selected animal×syllable pairs CSV.")
    p.add_argument("--x-min", type=int, default=-30)
    p.add_argument("--x-max", type=int, default=30)
    p.add_argument("--min-daily-n", type=int, default=1)
    p.add_argument("--daily-stat", choices=["median", "mean"], default="median")
    p.add_argument("--epoch-stat", choices=["median", "mean"], default="median")
    p.add_argument("--allow-ml-reuse", action="store_true", help="Match each sham to nearest ML with replacement. Default uses unique ML matches when possible.")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_excel)
    metadata.to_csv(out_dir / "seasonal_control_metadata_mapping_used.csv", index=False)

    daily = prepare_daily_table(
        args.daily_csv,
        metadata,
        selected_pairs_csv=args.selected_pairs_csv,
        x_min=args.x_min,
        x_max=args.x_max,
        min_daily_n=args.min_daily_n,
    )
    daily.to_csv(out_dir / "seasonal_control_filtered_daily_rows.csv", index=False)

    animal_day = make_animal_day_summary(daily, daily_stat=args.daily_stat)
    animal_day.to_csv(out_dir / "seasonal_control_animal_day_summary.csv", index=False)

    animal_delta = make_animal_delta_table(animal_day, epoch_stat=args.epoch_stat)
    animal_delta.to_csv(out_dir / "seasonal_control_animal_delta_sd.csv", index=False)

    matched = match_sham_to_ml(animal_delta, unique_ml=not args.allow_ml_reuse)
    matched.to_csv(out_dir / "seasonal_control_matched_pairs.csv", index=False)

    stats_df = matched_pair_stats(matched, n_boot=args.n_bootstrap, n_perm=args.n_permutations, seed=args.seed)
    stats_df.to_csv(out_dir / "seasonal_control_matched_pair_stats.csv", index=False)

    p1 = plot_delta_by_treatment_date(animal_delta, out_dir / "seasonal_control_delta_by_treatment_date.png")
    p2 = plot_matched_pairs(matched, out_dir / "seasonal_control_matched_pairs_delta_sd.png")
    p3 = plot_calendar_timecourse(animal_day, out_dir / "seasonal_control_calendar_daily_sd.png")
    p4 = plot_combined_figure(animal_delta, matched, stats_df, out_dir / "seasonal_control_combined.png")

    print("\n[OK] Wrote tables:")
    for name in [
        "seasonal_control_metadata_mapping_used.csv",
        "seasonal_control_filtered_daily_rows.csv",
        "seasonal_control_animal_day_summary.csv",
        "seasonal_control_animal_delta_sd.csv",
        "seasonal_control_matched_pairs.csv",
        "seasonal_control_matched_pair_stats.csv",
    ]:
        print(f"  {out_dir / name}")

    print("\n[OK] Wrote plots:")
    for path in [p1, p2, p3, p4]:
        print(f"  {path}")

    print("\nMatched-pair stats:")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
