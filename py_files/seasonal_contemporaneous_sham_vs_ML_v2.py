#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seasonal_contemporaneous_sham_vs_ML_v2.py

Seasonal-control analysis for Figure 3 phrase-duration SD.

Compared with v1, this version:
  1) explicitly allows cross-year nearest-date sham↔medial+lateral matching by default;
  2) runs the same seasonal-control tests for two syllable-selection sets:
       - pooled pre+post-selected top 30%
       - post-selected top 30%

Inputs
------
Recommended:
  --daily-csv      batch_aligned_phrase_duration_variance_all.csv
  --scatter-csv    usage_balanced_phrase_duration_stats.csv
  --metadata-excel Area_X_lesion_metadata_with_hit_types.xlsx

Optional:
  --pooled-selected-pairs-csv  existing Figure 3 selected-pairs CSV
  --post-selected-pairs-csv    existing post-selected top30 selected-pairs CSV

If selected-pairs CSVs are not provided, this script generates them from
--scatter-csv.
"""

from __future__ import annotations

import argparse
import itertools
import re
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

SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL_ML = "Partial Medial and Lateral lesion"
COMPLETE_ML = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"
PLOT_GROUPS = ["Sham", "Medial+lateral lesion"]
COLORS = {"Sham": "#1B9E77", "Medial+lateral lesion": "#6F4AB6", "Other": "0.5"}


def clean(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).strip().split())


def lower(x: Any) -> str:
    return clean(x).lower()


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()


def find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    exact = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in exact:
            return exact[cand.lower()]
    for cand in candidates:
        cand_l = cand.lower()
        for c in cols:
            if cand_l in str(c).lower():
                return c
    return None


def canonical_group(x: Any) -> str:
    s = lower(x)
    if not s or s == "nan":
        return "unknown"
    if "sham" in s or ("saline" in s and "lesion" not in s):
        return SHAM
    if (
        "lateral lesion only" in s
        or "lateral only" in s
        or "lateral hit only" in s
        or "single hit" in s
        or "single lateral" in s
        or "area x visible (single hit)" in s
    ):
        return LATERAL
    if ("complete" in s and "medial" in s and "lateral" in s) or "large lesion" in s or "not visible" in s:
        return COMPLETE_ML
    if (
        ("partial" in s and "medial" in s and "lateral" in s)
        or "area x visible (medial+lateral hit)" in s
        or "medial+lateral" in s
        or "medial lateral" in s
        or "m+l" in s
    ):
        return PARTIAL_ML
    return clean(x) or "unknown"


def coarse_group(g: str) -> str:
    if g == SHAM:
        return "Sham"
    if g in {PARTIAL_ML, COMPLETE_ML, POOLED_ML}:
        return "Medial+lateral lesion"
    return "Other"


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def load_metadata(path: Path, animal_col: str = "Animal ID") -> pd.DataFrame:
    xls = pd.ExcelFile(path)

    # Choose best hit-type sheet/column.
    hit_candidates = [
        "Lesion hit type", "Hit type", "Hit Type", "lesion_hit_type",
        "lesion hit type grouping", "Group", "Treatment type",
    ]
    best = None
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if animal_col not in df.columns:
            continue
        possible = [c for c in hit_candidates if c in df.columns]
        for c in df.columns:
            cl = str(c).lower()
            if c not in possible and (("hit" in cl and "type" in cl) or ("lesion" in cl and "group" in cl)):
                possible.append(c)
        for col in possible:
            groups = df[col].apply(canonical_group)
            n_known = int(groups.isin([SHAM, LATERAL, PARTIAL_ML, COMPLETE_ML]).sum())
            n_ml = int(groups.isin([PARTIAL_ML, COMPLETE_ML]).sum())
            score = n_known + n_ml + (50 if sheet == "animal_hit_type_summary" else 0)
            if "treatment type" in str(col).lower():
                score -= 40
            if best is None or score > best[0]:
                best = (score, sheet, col, df.copy())
    if best is None:
        raise ValueError("Could not find animal hit-type metadata.")
    _, hit_sheet, hit_col, hit_df = best

    meta = pd.DataFrame({
        "animal_id": hit_df[animal_col].astype(str).str.strip(),
        "raw_hit_type": hit_df[hit_col],
    })
    meta["display_group"] = meta["raw_hit_type"].apply(canonical_group)
    meta = meta.drop_duplicates("animal_id")

    # Choose treatment date sheet/column.
    best_date = None
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if animal_col not in df.columns:
            continue
        date_col = find_col(list(df.columns), ["Treatment date", "Lesion date", "treatment_date", "surgery date"])
        if date_col is None:
            continue
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        score = int(parsed.notna().sum()) + (10 if sheet.lower() == "metadata" else 0)
        if best_date is None or score > best_date[0]:
            best_date = (score, sheet, date_col, df.copy())
    if best_date is None:
        raise ValueError("Could not find treatment dates in metadata workbook.")
    _, date_sheet, date_col, date_df = best_date
    dates = pd.DataFrame({
        "animal_id": date_df[animal_col].astype(str).str.strip(),
        "treatment_date": pd.to_datetime(date_df[date_col], errors="coerce"),
    }).drop_duplicates("animal_id")

    meta = meta.merge(dates, on="animal_id", how="left")
    meta["coarse_group"] = meta["display_group"].apply(coarse_group)

    print(f"[INFO] Hit type source: sheet={hit_sheet!r}, column={hit_col!r}")
    print(f"[INFO] Treatment date source: sheet={date_sheet!r}, column={date_col!r}")
    print("[INFO] Metadata coarse group counts:")
    print(meta["coarse_group"].value_counts(dropna=False).to_string())
    return meta


def load_selected_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    a = find_col(list(df.columns), ["animal_id", "Animal ID", "Animal"])
    s = find_col(list(df.columns), ["syllable", "Syllable"])
    if a is None or s is None:
        raise ValueError(f"Selected-pairs CSV needs animal and syllable columns. Found: {list(df.columns)}")
    out = pd.DataFrame({"animal_id": df[a].astype(str), "syllable": df[s].astype(str)})
    return out.drop_duplicates()


def generate_selected_pairs(scatter_csv: Path, rank_on: str, top_frac: float, pre_group: str, post_group: str, min_n_phrases: int) -> pd.DataFrame:
    df = pd.read_csv(scatter_csv)
    animal = find_col(list(df.columns), ["Animal ID", "animal_id", "Animal"])
    syll = find_col(list(df.columns), ["Syllable", "syllable"])
    epoch = find_col(list(df.columns), ["Group", "group", "epoch", "Epoch"])
    var = find_col(list(df.columns), ["Variance_ms2", "Variance (ms^2)", "variance_ms2", "Variance"])
    ncol = find_col(list(df.columns), ["N_phrases", "n_phrases", "N", "n"])
    missing = [name for name, col in [("animal", animal), ("syllable", syll), ("epoch", epoch), ("variance", var)] if col is None]
    if missing:
        raise ValueError(f"scatter_csv missing columns: {missing}. Found: {list(df.columns)}")

    work = df.copy()
    work[animal] = work[animal].astype(str)
    work[syll] = work[syll].astype(str)
    work[var] = pd.to_numeric(work[var], errors="coerce")
    work = work[work[epoch].astype(str).isin([pre_group, post_group])].copy()
    if ncol is not None and min_n_phrases > 1:
        work[ncol] = pd.to_numeric(work[ncol], errors="coerce")
        work = work[work[ncol].fillna(0) >= min_n_phrases].copy()

    mean_var = work.groupby([animal, syll, epoch], dropna=False)[var].mean().reset_index()
    wide = mean_var.pivot_table(index=[animal, syll], columns=epoch, values=var, aggfunc="mean").reset_index()
    if pre_group not in wide.columns or post_group not in wide.columns:
        raise ValueError(f"Could not find {pre_group!r} and {post_group!r} in scatter_csv.")
    wide = wide.rename(columns={pre_group: "pre_variance_ms2", post_group: "post_variance_ms2"})
    wide = wide.dropna(subset=["pre_variance_ms2", "post_variance_ms2"])
    wide = wide[(wide["pre_variance_ms2"] > 0) & (wide["post_variance_ms2"] > 0)].copy()

    if rank_on == "pooled":
        wide["rank_metric"] = wide[["pre_variance_ms2", "post_variance_ms2"]].mean(axis=1)
    elif rank_on == "post":
        wide["rank_metric"] = wide["post_variance_ms2"]
    else:
        raise ValueError("rank_on must be pooled or post")

    parts = []
    for aid, g in wide.groupby(animal, dropna=False):
        vals = g["rank_metric"].dropna().to_numpy(float)
        if vals.size == 0:
            continue
        thresh = np.nanpercentile(vals, 100 * (1 - top_frac))
        keep = g[g["rank_metric"] >= thresh].copy()
        keep["rank_on"] = rank_on
        keep["selection_threshold"] = thresh
        parts.append(keep)
    if not parts:
        raise ValueError(f"No selected pairs generated for rank_on={rank_on}")
    selected = pd.concat(parts, ignore_index=True)
    selected = selected.rename(columns={animal: "animal_id", syll: "syllable"})
    return selected[["animal_id", "syllable", "rank_metric", "rank_on", "selection_threshold"]].drop_duplicates(["animal_id", "syllable"])


def daily_cols(df: pd.DataFrame) -> dict[str, Optional[str]]:
    cols = list(df.columns)
    out = {
        "animal": find_col(cols, ["animal_id", "Animal ID", "Animal", "bird", "Bird"]),
        "syllable": find_col(cols, ["syllable", "Syllable", "label", "cluster"]),
        "relative_day": find_col(cols, ["relative_day", "Relative day", "days_relative_to_lesion"]),
        "variance": find_col(cols, ["Variance (ms^2)", "variance_ms2", "Variance_ms2", "Variance"]),
        "date": find_col(cols, ["date", "Date", "recording_date", "Recording date"]),
        "n": find_col(cols, ["N_phrases", "n_phrases", "occurrences", "count", "n", "N"]),
    }
    missing = [k for k in ["animal", "syllable", "relative_day", "variance"] if out[k] is None]
    if missing:
        raise ValueError(f"daily_csv missing required columns: {missing}. Found: {cols}")
    return out


def prepare_daily(daily_csv: Path, metadata: pd.DataFrame, selected_pairs: pd.DataFrame, x_min: int, x_max: int, min_daily_n: int) -> pd.DataFrame:
    df = pd.read_csv(daily_csv)
    cc = daily_cols(df)
    work = pd.DataFrame({
        "animal_id": df[cc["animal"]].astype(str),
        "syllable": df[cc["syllable"]].astype(str),
        "relative_day": pd.to_numeric(df[cc["relative_day"]], errors="coerce"),
        "variance_ms2": pd.to_numeric(df[cc["variance"]], errors="coerce"),
    })
    work["recording_date_raw"] = pd.to_datetime(df[cc["date"]], errors="coerce") if cc["date"] else pd.NaT
    work["daily_n"] = pd.to_numeric(df[cc["n"]], errors="coerce") if cc["n"] else np.nan

    work = work.dropna(subset=["relative_day", "variance_ms2"])
    work = work[(work["relative_day"] >= x_min) & (work["relative_day"] <= x_max) & (work["variance_ms2"] > 0)].copy()
    if min_daily_n > 1:
        before = len(work)
        work = work[work["daily_n"].fillna(0) >= min_daily_n].copy()
        print(f"[INFO] min_daily_n={min_daily_n}: kept {len(work):,}/{before:,} rows")

    before_rows = len(work)
    before_pairs = work[["animal_id", "syllable"]].drop_duplicates().shape[0]
    work = work.merge(selected_pairs[["animal_id", "syllable"]].drop_duplicates(), on=["animal_id", "syllable"], how="inner")
    after_pairs = work[["animal_id", "syllable"]].drop_duplicates().shape[0]
    print(f"[INFO] Filtered daily rows to selected pairs: {len(work):,}/{before_rows:,} rows; {after_pairs}/{before_pairs} pairs")

    work = work.merge(metadata[["animal_id", "raw_hit_type", "display_group", "coarse_group", "treatment_date"]], on="animal_id", how="left")
    work["display_group"] = work["display_group"].fillna("unknown").apply(canonical_group)
    work["coarse_group"] = work["display_group"].apply(coarse_group)
    work["treatment_date"] = pd.to_datetime(work["treatment_date"], errors="coerce")

    work["recording_date"] = work["recording_date_raw"]
    missing_date = work["recording_date"].isna()
    rel_int = work.loc[missing_date, "relative_day"].round().astype("Int64")
    work.loc[missing_date, "recording_date"] = work.loc[missing_date, "treatment_date"] + pd.to_timedelta(rel_int, unit="D")

    work["sd_s"] = np.sqrt(work["variance_ms2"]) / 1000.0
    work = work[work["coarse_group"].isin(PLOT_GROUPS)].copy()
    if work.empty:
        raise ValueError("No Sham or Medial+lateral rows remain after filtering.")
    return work


def animal_day_summary(daily: pd.DataFrame, daily_stat: str) -> pd.DataFrame:
    func = "median" if daily_stat == "median" else "mean"
    return daily.groupby(
        ["animal_id", "coarse_group", "display_group", "raw_hit_type", "treatment_date", "recording_date", "relative_day"],
        dropna=False,
    ).agg(
        daily_sd_s=("sd_s", func),
        n_syllable_day_rows=("sd_s", "size"),
        n_syllables=("syllable", "nunique"),
    ).reset_index()


def animal_delta(day: pd.DataFrame, epoch_stat: str) -> pd.DataFrame:
    val = np.nanmedian if epoch_stat == "median" else np.nanmean
    rows = []
    for aid, g in day.groupby("animal_id", dropna=False):
        pre = g.loc[g["relative_day"] < 0, "daily_sd_s"].dropna().to_numpy(float)
        post = g.loc[g["relative_day"] > 0, "daily_sd_s"].dropna().to_numpy(float)
        if pre.size == 0 or post.size == 0:
            continue
        rows.append({
            "animal_id": aid,
            "coarse_group": g["coarse_group"].iloc[0],
            "display_group": g["display_group"].iloc[0],
            "raw_hit_type": g["raw_hit_type"].iloc[0],
            "treatment_date": g["treatment_date"].iloc[0],
            "pre_sd_s": float(val(pre)),
            "post_sd_s": float(val(post)),
            "delta_sd_s": float(val(post) - val(pre)),
            "days_recorded_pre": int(g.loc[g["relative_day"] < 0, "relative_day"].nunique()),
            "days_recorded_post": int(g.loc[g["relative_day"] > 0, "relative_day"].nunique()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Could not compute animal-level deltas.")
    return out.sort_values(["coarse_group", "treatment_date", "animal_id"]).reset_index(drop=True)


def match_sham_ml(delta: pd.DataFrame, unique_ml: bool, same_year_only: bool, max_days_apart: Optional[int]) -> pd.DataFrame:
    sham = delta[delta["coarse_group"] == "Sham"].dropna(subset=["treatment_date", "delta_sd_s"]).reset_index(drop=True)
    ml = delta[delta["coarse_group"] == "Medial+lateral lesion"].dropna(subset=["treatment_date", "delta_sd_s"]).reset_index(drop=True)
    if sham.empty or ml.empty:
        raise ValueError("Need sham and ML birds for matching.")

    sd = np.array([pd.Timestamp(d).toordinal() for d in sham["treatment_date"]], float)
    md = np.array([pd.Timestamp(d).toordinal() for d in ml["treatment_date"]], float)
    cost = np.abs(sd[:, None] - md[None, :])
    if same_year_only:
        sy = np.array([pd.Timestamp(d).year for d in sham["treatment_date"]])
        my = np.array([pd.Timestamp(d).year for d in ml["treatment_date"]])
        cost = np.where(sy[:, None] == my[None, :], cost, np.inf)
    if max_days_apart is not None:
        cost = np.where(cost <= max_days_apart, cost, np.inf)

    pairs = []
    if unique_ml and len(ml) >= len(sham) and linear_sum_assignment is not None:
        if not np.isfinite(cost).any():
            raise ValueError("No valid matches under matching constraints.")
        large = np.nanmax(cost[np.isfinite(cost)]) + 1e6
        safe = np.where(np.isfinite(cost), cost, large)
        ri, ci = linear_sum_assignment(safe)
        for r, c in zip(ri, ci):
            if r < len(sham) and np.isfinite(cost[r, c]):
                pairs.append((r, c))
    elif unique_ml and len(ml) >= len(sham):
        used_m, used_s = set(), set()
        flat = sorted((cost[i, j], i, j) for i in range(len(sham)) for j in range(len(ml)) if np.isfinite(cost[i, j]))
        for _, i, j in flat:
            if i in used_s or j in used_m:
                continue
            pairs.append((i, j)); used_s.add(i); used_m.add(j)
            if len(used_s) == len(sham):
                break
    else:
        for i in range(len(sham)):
            finite = np.where(np.isfinite(cost[i]))[0]
            if finite.size:
                pairs.append((i, int(finite[np.argmin(cost[i, finite])])))
    if not pairs:
        raise ValueError("No valid matches found.")

    rows = []
    for k, (si, mi) in enumerate(pairs, start=1):
        s = sham.iloc[si]; m = ml.iloc[mi]
        days = abs((pd.Timestamp(m["treatment_date"]) - pd.Timestamp(s["treatment_date"])).days)
        rows.append({
            "pair_id": k,
            "sham_animal_id": s["animal_id"],
            "sham_treatment_date": pd.Timestamp(s["treatment_date"]).date().isoformat(),
            "sham_delta_sd_s": float(s["delta_sd_s"]),
            "ml_animal_id": m["animal_id"],
            "ml_treatment_date": pd.Timestamp(m["treatment_date"]).date().isoformat(),
            "ml_hit_type": m["raw_hit_type"],
            "ml_display_group": m["display_group"],
            "ml_delta_sd_s": float(m["delta_sd_s"]),
            "ml_minus_sham_delta_sd_s": float(m["delta_sd_s"] - s["delta_sd_s"]),
            "treatment_dates_days_apart": int(days),
            "cross_year_match": bool(pd.Timestamp(m["treatment_date"]).year != pd.Timestamp(s["treatment_date"]).year),
        })
    return pd.DataFrame(rows)


def signflip_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan
    obs = np.nanmean(diff)
    vals = []
    for signs in itertools.product([-1, 1], repeat=len(diff)):
        vals.append(np.nanmean(diff * np.asarray(signs)))
    vals = np.asarray(vals)
    return float((np.sum(vals >= obs - 1e-15) + 1) / (len(vals) + 1))


def boot_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [np.nanmean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def matched_stats(matched: pd.DataFrame, label: str, basis: str, n_boot: int, seed: int) -> pd.DataFrame:
    diff = matched["ml_minus_sham_delta_sd_s"].to_numpy(float)
    ci_low, ci_high = boot_ci(diff, n_boot, seed)
    if scipy_stats is not None and len(diff) >= 2:
        t = scipy_stats.ttest_1samp(diff, popmean=0, alternative="greater")
        t_stat, t_p = float(t.statistic), float(t.pvalue)
    else:
        t_stat, t_p = np.nan, np.nan
    return pd.DataFrame([{
        "selection_label": label,
        "selection_basis": basis,
        "comparison": "nearest-date matched ML minus sham ΔSD",
        "n_pairs": int(len(diff)),
        "mean_ML_minus_sham_delta_sd_s": float(np.nanmean(diff)),
        "median_ML_minus_sham_delta_sd_s": float(np.nanmedian(diff)),
        "mean_difference_bootstrap_CI95_low_s": ci_low,
        "mean_difference_bootstrap_CI95_high_s": ci_high,
        "paired_t_greater": t_stat,
        "paired_t_p_greater": t_p,
        "paired_signflip_p_greater": signflip_p(diff),
        "median_days_between_matched_treatment_dates": float(np.nanmedian(matched["treatment_dates_days_apart"])),
        "max_days_between_matched_treatment_dates": int(np.nanmax(matched["treatment_dates_days_apart"])),
        "n_cross_year_matches": int(matched["cross_year_match"].sum()),
    }])


def group_summary(delta: pd.DataFrame, label: str, basis: str) -> pd.DataFrame:
    rows = []
    for group in PLOT_GROUPS:
        g = delta[delta["coarse_group"] == group]
        vals = g["delta_sd_s"].dropna().to_numpy(float)
        if vals.size == 0:
            continue
        rows.append({
            "selection_label": label,
            "selection_basis": basis,
            "coarse_group": group,
            "n_birds": int(g["animal_id"].nunique()),
            "median_delta_sd_s": float(np.nanmedian(vals)),
            "mean_delta_sd_s": float(np.nanmean(vals)),
            "min_delta_sd_s": float(np.nanmin(vals)),
            "max_delta_sd_s": float(np.nanmax(vals)),
        })
    return pd.DataFrame(rows)


def setup(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def plot_delta_date(delta: pd.DataFrame, out: Path, label: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    for group in PLOT_GROUPS:
        g = delta[delta["coarse_group"] == group]
        ax.scatter(g["treatment_date"], g["delta_sd_s"], color=COLORS[group], s=52, edgecolors="none", alpha=0.85, label=f"{group} (n={g['animal_id'].nunique()})")
        for _, r in g.iterrows():
            ax.text(r["treatment_date"], r["delta_sd_s"], f" {r['animal_id']}", fontsize=7.5, color=COLORS[group], ha="left", va="center")
    ax.set_ylabel("Δ phrase duration SD (s)", fontsize=11)
    ax.set_xlabel("Treatment date", fontsize=11)
    ax.set_title(f"Seasonal control: ΔSD by treatment date\n{label}", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30, ha="right")
    setup(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_matched(matched: pd.DataFrame, out: Path, label: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    for _, r in matched.iterrows():
        ax.plot([0, 1], [r["sham_delta_sd_s"], r["ml_delta_sd_s"]], color="0.55", linewidth=1, alpha=0.75)
        ax.scatter([0], [r["sham_delta_sd_s"]], color=COLORS["Sham"], s=48, edgecolors="none")
        ax.scatter([1], [r["ml_delta_sd_s"]], color=COLORS["Medial+lateral lesion"], s=48, edgecolors="none")
    for x, col, color in [(0, "sham_delta_sd_s", COLORS["Sham"]), (1, "ml_delta_sd_s", COLORS["Medial+lateral lesion"] )]:
        vals = matched[col].to_numpy(float)
        mean = np.nanmean(vals)
        sem = np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals))) if np.sum(np.isfinite(vals)) > 1 else 0
        ax.errorbar([x], [mean], yerr=[sem], fmt="o", color="black", ecolor="black", capsize=4, markersize=5)
        ax.plot([x - 0.15, x + 0.15], [mean, mean], color=color, linewidth=2.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nearest-date\nsham", "Nearest-date\nmedial+lateral"], fontsize=10)
    ax.set_ylabel("Δ phrase duration SD (s)", fontsize=11)
    ax.set_title(f"Nearest-date matched seasonal control\n{label}", fontsize=12)
    setup(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_calendar(day: pd.DataFrame, out: Path, label: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for (aid, group), g in day.groupby(["animal_id", "coarse_group"]):
        g = g.sort_values("recording_date")
        ax.plot(g["recording_date"], g["daily_sd_s"], color=COLORS[group], alpha=0.38, linewidth=1.2, marker="o", markersize=2.8, markeredgewidth=0)
    tmp = day.copy()
    tmp["month"] = tmp["recording_date"].dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby(["coarse_group", "month"]).agg(monthly_sd_s=("daily_sd_s", "median")).reset_index()
    for group in PLOT_GROUPS:
        g = monthly[monthly["coarse_group"] == group].sort_values("month")
        if len(g) >= 2:
            ax.plot(g["month"], g["monthly_sd_s"], color=COLORS[group], linewidth=2.5, label=f"{group} monthly median")
    ax.set_ylabel("Daily phrase duration SD (s)", fontsize=11)
    ax.set_xlabel("Recording date", fontsize=11)
    ax.set_title(f"Daily phrase duration SD by calendar date\n{label}", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30, ha="right")
    setup(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_combined(delta: pd.DataFrame, matched: pd.DataFrame, stats: pd.DataFrame, out: Path, label: str):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    for group in PLOT_GROUPS:
        g = delta[delta["coarse_group"] == group]
        ax.scatter(g["treatment_date"], g["delta_sd_s"], color=COLORS[group], s=48, edgecolors="none", alpha=0.85, label=f"{group} (n={g['animal_id'].nunique()})")
    ax.set_ylabel("Δ phrase duration SD (s)", fontsize=11)
    ax.set_xlabel("Treatment date", fontsize=11)
    ax.set_title("A. ΔSD by time of year", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)
    setup(ax)

    ax = axes[1]
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    for _, r in matched.iterrows():
        ax.plot([0, 1], [r["sham_delta_sd_s"], r["ml_delta_sd_s"]], color="0.55", linewidth=1, alpha=0.75)
        ax.scatter([0], [r["sham_delta_sd_s"]], color=COLORS["Sham"], s=48, edgecolors="none")
        ax.scatter([1], [r["ml_delta_sd_s"]], color=COLORS["Medial+lateral lesion"], s=48, edgecolors="none")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nearest-date\nsham", "Nearest-date\nmedial+lateral"], fontsize=9)
    ax.set_ylabel("Δ phrase duration SD (s)", fontsize=11)
    p = float(stats["paired_signflip_p_greater"].iloc[0])
    n_cross = int(stats["n_cross_year_matches"].iloc[0])
    ax.set_title(f"B. Matched comparison\nsign-flip p={fmt_p(p)}; cross-year={n_cross}", fontsize=12)
    setup(ax)
    fig.suptitle(label, fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_summary(stats: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    x = np.arange(len(stats))
    y = stats["mean_ML_minus_sham_delta_sd_s"].to_numpy(float)
    lo = stats["mean_difference_bootstrap_CI95_low_s"].to_numpy(float)
    hi = stats["mean_difference_bootstrap_CI95_high_s"].to_numpy(float)
    ax.errorbar(x, y, yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)], fmt="o", color="black", ecolor="black", capsize=4)
    for xi, yi, p in zip(x, y, stats["paired_signflip_p_greater"]):
        ax.text(xi, yi, f" p={fmt_p(float(p))}", ha="left", va="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(stats["selection_label"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Matched ML − sham ΔSD (s)", fontsize=11)
    ax.set_title("Cross-year nearest-date seasonal control", fontsize=12)
    setup(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def run_selection(args, meta, label, basis, pairs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"[INFO] Selection: {label}")
    print(f"[INFO] Selected pairs: {len(pairs):,}")
    pairs.to_csv(out_dir / "selected_pairs_used.csv", index=False)

    daily = prepare_daily(args.daily_csv, meta, pairs, args.x_min, args.x_max, args.min_daily_n)
    daily["selection_label"] = label; daily["selection_basis"] = basis
    daily.to_csv(out_dir / "seasonal_control_filtered_daily_rows.csv", index=False)

    day = animal_day_summary(daily, args.daily_stat)
    day["selection_label"] = label; day["selection_basis"] = basis
    day.to_csv(out_dir / "seasonal_control_animal_day_summary.csv", index=False)

    delta = animal_delta(day, args.epoch_stat)
    delta["selection_label"] = label; delta["selection_basis"] = basis
    delta.to_csv(out_dir / "seasonal_control_animal_delta_sd.csv", index=False)

    matched = match_sham_ml(delta, unique_ml=not args.allow_ml_reuse, same_year_only=args.same_year_only, max_days_apart=args.max_days_apart)
    matched["selection_label"] = label; matched["selection_basis"] = basis
    matched.to_csv(out_dir / "seasonal_control_matched_pairs.csv", index=False)

    stats = matched_stats(matched, label, basis, args.n_bootstrap, args.seed)
    stats.to_csv(out_dir / "seasonal_control_matched_pair_stats.csv", index=False)

    gsum = group_summary(delta, label, basis)
    gsum.to_csv(out_dir / "seasonal_control_group_delta_summary.csv", index=False)

    plot_delta_date(delta, out_dir / "seasonal_control_delta_by_treatment_date.png", label)
    plot_matched(matched, out_dir / "seasonal_control_matched_pairs_delta_sd.png", label)
    plot_calendar(day, out_dir / "seasonal_control_calendar_daily_sd.png", label)
    plot_combined(delta, matched, stats, out_dir / "seasonal_control_combined.png", label)

    print("[INFO] Matched-pair stats:")
    print(stats.to_string(index=False))
    return daily, day, delta, matched, stats, gsum


def main():
    p = argparse.ArgumentParser(description="Seasonal-control sham vs ML comparison for pooled and post-selected top30 syllables.")
    p.add_argument("--daily-csv", required=True, type=Path)
    p.add_argument("--metadata-excel", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--scatter-csv", type=Path, default=None)
    p.add_argument("--pooled-selected-pairs-csv", type=Path, default=None)
    p.add_argument("--post-selected-pairs-csv", type=Path, default=None)
    p.add_argument("--top-frac", type=float, default=0.30)
    p.add_argument("--pre-group", default="Late Pre")
    p.add_argument("--post-group", default="Post")
    p.add_argument("--min-n-phrases", type=int, default=5)
    p.add_argument("--x-min", type=int, default=-30)
    p.add_argument("--x-max", type=int, default=30)
    p.add_argument("--min-daily-n", type=int, default=1)
    p.add_argument("--daily-stat", choices=["median", "mean"], default="median")
    p.add_argument("--epoch-stat", choices=["median", "mean"], default="median")
    p.add_argument("--allow-ml-reuse", action="store_true")
    p.add_argument("--same-year-only", action="store_true", help="Forbid cross-year matches. Default allows cross-year matches.")
    p.add_argument("--max-days-apart", type=int, default=None)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(args.metadata_excel)
    meta.to_csv(args.out_dir / "seasonal_control_metadata_mapping_used.csv", index=False)

    # Build pooled pre+post selected pairs.
    if args.pooled_selected_pairs_csv is not None:
        pooled_pairs = load_selected_pairs(args.pooled_selected_pairs_csv)
    else:
        if args.scatter_csv is None:
            raise ValueError("Need --scatter-csv or --pooled-selected-pairs-csv")
        pooled_pairs = generate_selected_pairs(args.scatter_csv, "pooled", args.top_frac, args.pre_group, args.post_group, args.min_n_phrases)
    pooled_label = f"Pooled pre+post-selected top {int(args.top_frac * 100)}%"
    pooled_basis = "pooled late-pre/post selection"
    pooled_pairs.to_csv(args.out_dir / f"{slug(pooled_label)}_selected_pairs.csv", index=False)

    # Build post selected pairs.
    if args.post_selected_pairs_csv is not None:
        post_pairs = load_selected_pairs(args.post_selected_pairs_csv)
    else:
        if args.scatter_csv is None:
            raise ValueError("Need --scatter-csv or --post-selected-pairs-csv")
        post_pairs = generate_selected_pairs(args.scatter_csv, "post", args.top_frac, args.pre_group, args.post_group, args.min_n_phrases)
    post_label = f"Post-selected top {int(args.top_frac * 100)}%"
    post_basis = "post-lesion selection"
    post_pairs.to_csv(args.out_dir / f"{slug(post_label)}_selected_pairs.csv", index=False)

    all_daily, all_day, all_delta, all_matched, all_stats, all_gsum = [], [], [], [], [], []
    for label, basis, pairs in [(pooled_label, pooled_basis, pooled_pairs), (post_label, post_basis, post_pairs)]:
        od = args.out_dir / slug(label)
        res = run_selection(args, meta, label, basis, pairs[["animal_id", "syllable"]].drop_duplicates(), od)
        daily, day, delta, matched, stats, gsum = res
        all_daily.append(daily); all_day.append(day); all_delta.append(delta); all_matched.append(matched); all_stats.append(stats); all_gsum.append(gsum)

    pd.concat(all_daily, ignore_index=True).to_csv(args.out_dir / "seasonal_control_all_filtered_daily_rows.csv", index=False)
    pd.concat(all_day, ignore_index=True).to_csv(args.out_dir / "seasonal_control_all_animal_day_summary.csv", index=False)
    pd.concat(all_delta, ignore_index=True).to_csv(args.out_dir / "seasonal_control_all_animal_delta_sd.csv", index=False)
    matched_all = pd.concat(all_matched, ignore_index=True)
    matched_all.to_csv(args.out_dir / "seasonal_control_all_matched_pairs.csv", index=False)
    stats_all = pd.concat(all_stats, ignore_index=True)
    stats_all.to_csv(args.out_dir / "seasonal_control_all_matched_pair_stats.csv", index=False)
    gsum_all = pd.concat(all_gsum, ignore_index=True)
    gsum_all.to_csv(args.out_dir / "seasonal_control_all_group_delta_summary.csv", index=False)

    summary = stats_all.merge(
        gsum_all.pivot_table(index=["selection_label", "selection_basis"], columns="coarse_group", values="median_delta_sd_s", aggfunc="first")
        .reset_index().rename(columns={"Sham": "sham_median_delta_sd_s", "Medial+lateral lesion": "ML_median_delta_sd_s"}),
        on=["selection_label", "selection_basis"], how="left"
    )
    summary.to_csv(args.out_dir / "seasonal_control_selection_summary.csv", index=False)
    plot_summary(stats_all, args.out_dir / "seasonal_control_summary_plot.png")

    print("\n" + "=" * 80)
    print("[OK] Wrote combined outputs to:", args.out_dir)
    print("\nCombined matched-pair stats:")
    print(stats_all.to_string(index=False))


if __name__ == "__main__":
    main()
