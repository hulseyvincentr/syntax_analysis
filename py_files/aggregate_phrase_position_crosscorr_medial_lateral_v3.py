#!/usr/bin/env python3
"""
Aggregate phrase-position spectrogram cross-correlation analyses across birds.

Input expected from phrase_position_acoustic_crosscorr_v2.py, for example:
  <root>/<ANIMAL>/<ANIMAL>_all_labels_phrase_position_segment_features.csv
or individual label files:
  <root>/<ANIMAL>/<ANIMAL>_label<LABEL>/<ANIMAL>_label<LABEL>_phrase_position_segment_features.csv

This script asks whether repeated syllables become less spectrogram-similar as the phrase
continues, and whether that relationship is stronger after lesion.

Main outputs:
- phrase_position_crosscorr_label_slopes.csv
- phrase_position_crosscorr_label_effects.csv
- phrase_position_crosscorr_bird_summary.csv
- phrase_position_crosscorr_bird_stats.csv
- slope post-pre by bird plots
- pre vs post degradation slope plots
- baseline-vs-lesion slope-change plots
- binned cross-correlation vs elapsed time / previous repeats plots
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

PERIOD_ORDER = ["early_pre", "late_pre", "early_post", "late_post"]
PRE_POST_ORDER = ["pre", "post"]

FEATURE_LABELS = {
    "corr_to_phrase_early_template": "Correlation to early-phrase template",
    "distance_from_phrase_early_template": "Distance from early-phrase template (1 - corr)",
    "mean_pitch_derivative_khz_per_s": "Mean pitch derivative (kHz/s)",
    "q95_pitch_derivative_khz_per_s": "95th percentile pitch derivative (kHz/s)",
    "mean_wiener_entropy": "Mean Wiener entropy",
    "q95_wiener_entropy": "95th percentile Wiener entropy",
}

ANIMAL_COL_CANDIDATES = ["Animal ID", "animal_id", "Animal", "animal", "Bird", "bird"]
LESION_HIT_TYPE_COL_CANDIDATES = [
    "Lesion hit type", "lesion_hit_type", "Hit type", "hit_type", "Lesion Type",
    "lesion type", "Treatment group", "treatment_group", "Group", "group",
]


def clean_values(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(x).strip().lower()).strip()


def find_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def is_medial_lateral_hit(hit_type) -> bool:
    s = normalize_text(hit_type)
    raw = str(hit_type).strip().lower().replace(" ", "")
    return ("medial" in s and "lateral" in s) or ("m+l" in raw) or ("medial+lateral" in raw)


def load_medial_lateral_animals(metadata_excel_path: str,
                                sheet_name: str = "animal_hit_type_summary",
                                animal_col: Optional[str] = None,
                                hit_type_col: Optional[str] = None) -> Tuple[set[str], pd.DataFrame]:
    meta = pd.read_excel(Path(metadata_excel_path).expanduser(), sheet_name=sheet_name)
    animal_col = animal_col or find_first_existing(meta.columns, ANIMAL_COL_CANDIDATES)
    hit_type_col = hit_type_col or find_first_existing(meta.columns, LESION_HIT_TYPE_COL_CANDIDATES)
    if animal_col is None:
        raise ValueError(f"Could not identify animal ID column in sheet {sheet_name!r}. Columns: {list(meta.columns)}")
    if hit_type_col is None:
        raise ValueError(f"Could not identify lesion hit type column in sheet {sheet_name!r}. Columns: {list(meta.columns)}")

    work = meta.copy()
    work[animal_col] = work[animal_col].astype(str).str.strip()
    label_ml = work[hit_type_col].map(is_medial_lateral_hit)

    medial_col = find_first_existing(work.columns, ["Medial Area X hit type", "Medial hit type", "medial_hit_type"])
    lateral_col = find_first_existing(work.columns, ["Lateral Area X hit type", "Lateral hit type", "lateral_hit_type"])
    if medial_col is not None and lateral_col is not None:
        medial_hit = work[medial_col].map(lambda x: normalize_text(x) not in {"", "miss", "nan", "none", "no"})
        lateral_hit = work[lateral_col].map(lambda x: normalize_text(x) not in {"", "miss", "nan", "none", "no"})
        parsed_ml = medial_hit & lateral_hit
    else:
        parsed_ml = pd.Series(False, index=work.index)

    work["_is_medial_lateral"] = label_ml | parsed_ml
    work["_medial_lateral_filter_source"] = np.where(
        label_ml, "lesion_hit_type_label",
        np.where(parsed_ml, "parsed_medial_lateral_hit_columns", "not_medial_lateral")
    )
    animals = set(work.loc[work["_is_medial_lateral"], animal_col].dropna().astype(str).str.strip())
    return animals, work


def parse_animal_label_from_filename(path: Path) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"(?P<animal>[^/\\]+)_label(?P<label>.+?)_phrase_position_segment_features\.csv$", path.name)
    if m:
        return m.group("animal"), m.group("label")
    m = re.search(r"(?P<animal>[^/\\]+)_all_labels_phrase_position_segment_features\.csv$", path.name)
    if m:
        return m.group("animal"), None
    return None, None


def find_segment_feature_files(root: Path) -> List[Path]:
    # Prefer per-animal combined all-label files when present; otherwise use individual label files.
    all_label_files = sorted(root.rglob("*_all_labels_phrase_position_segment_features.csv"))
    if all_label_files:
        return all_label_files
    return sorted(root.rglob("*_phrase_position_segment_features.csv"))


def read_segment_feature_tables(root: Path, allowed_animals: Optional[set[str]] = None,
                                min_segments_per_table: int = 1) -> pd.DataFrame:
    files = find_segment_feature_files(root)
    if not files:
        raise FileNotFoundError(f"No phrase-position segment feature CSVs found under: {root}")

    frames = []
    used_rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
            continue
        if df.empty or df.shape[0] < min_segments_per_table:
            continue
        animal_from_file, label_from_file = parse_animal_label_from_filename(f)
        if "animal_id" not in df.columns:
            df["animal_id"] = animal_from_file or "unknown"
        if "label" not in df.columns and label_from_file is not None:
            df["label"] = label_from_file
        df["animal_id"] = df["animal_id"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()

        if allowed_animals is not None:
            df = df[df["animal_id"].isin(allowed_animals)].copy()
            if df.empty:
                continue
        df["source_csv"] = str(f)
        frames.append(df)
        used_rows.append({
            "csv_path": str(f),
            "n_rows_used": int(df.shape[0]),
            "animals": ",".join(sorted(df["animal_id"].unique())),
            "labels": ",".join(sorted(df["label"].astype(str).unique())),
        })

    if not frames:
        raise RuntimeError("No usable segment-feature rows after filtering.")
    out = pd.concat(frames, ignore_index=True)
    return out, pd.DataFrame(used_rows)


def ensure_pre_post(df: pd.DataFrame) -> pd.DataFrame:
    if "pre_post" not in df.columns:
        df = df.copy()
        df["pre_post"] = np.where(df["period"].astype(str).str.contains("post"), "post", "pre")
    return df


def linear_slope_and_stats(x: np.ndarray, y: np.ndarray, min_n: int = 5) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(valid))
    row = {
        "n": n,
        "slope": np.nan,
        "intercept": np.nan,
        "r_value": np.nan,
        "p_value": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
    }
    if n < min_n:
        return row
    xv = x[valid]
    yv = y[valid]
    if np.unique(xv).size < 2 or np.unique(yv).size < 2:
        return row
    if SCIPY_AVAILABLE:
        lr = stats.linregress(xv, yv)
        sp = stats.spearmanr(xv, yv)
        row.update({
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "r_value": float(lr.rvalue),
            "p_value": float(lr.pvalue),
            "spearman_r": float(sp.correlation),
            "spearman_p": float(sp.pvalue),
        })
    else:
        slope, intercept = np.polyfit(xv, yv, deg=1)
        row.update({"slope": float(slope), "intercept": float(intercept)})
    return row


def degradation_multiplier(feature: str) -> float:
    # Convert feature slope to a "more acoustic degradation over phrase progression" slope.
    # For correlation, decreasing with phrase position means degradation, so multiply by -1.
    # For distance, increasing with phrase position means degradation, so multiply by +1.
    if feature == "corr_to_phrase_early_template":
        return -1.0
    if feature == "distance_from_phrase_early_template":
        return 1.0
    # For acoustic features, preserve signed slope rather than assigning degradation meaning.
    return 1.0


def compute_label_slopes(df: pd.DataFrame, features: List[str], min_n: int) -> pd.DataFrame:
    rows = []
    df = ensure_pre_post(df)
    group_specs = []
    for p in PERIOD_ORDER:
        group_specs.append(("period", p))
    for pp in PRE_POST_ORDER:
        group_specs.append(("pre_post", pp))

    for (animal, label), sub_label in df.groupby(["animal_id", "label"], dropna=False):
        for feature in features:
            if feature not in sub_label.columns:
                continue
            mult = degradation_multiplier(feature)
            for group_col, group in group_specs:
                sub = sub_label[sub_label[group_col].astype(str) == group].copy()
                for predictor, slope_col in [
                    ("elapsed_time_in_phrase_s", "per_s"),
                    ("n_previous_segments", "per_repeat"),
                ]:
                    res = linear_slope_and_stats(sub[predictor].to_numpy(dtype=float), sub[feature].to_numpy(dtype=float), min_n=min_n)
                    rows.append({
                        "animal_id": animal,
                        "label": label,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS.get(feature, feature),
                        "group_col": group_col,
                        "group": group,
                        "predictor": predictor,
                        "slope_type": slope_col,
                        "n": res["n"],
                        "slope": res["slope"],
                        "degradation_slope": mult * res["slope"] if np.isfinite(res["slope"]) else np.nan,
                        "intercept": res["intercept"],
                        "r_value": res["r_value"],
                        "p_value": res["p_value"],
                        "spearman_r": res["spearman_r"],
                        "spearman_p": res["spearman_p"],
                        "degradation_multiplier": mult,
                    })
    return pd.DataFrame(rows)


def pivot_value(slopes_df: pd.DataFrame, animal: str, label: str, feature: str, predictor: str,
                group_col: str, group: str, value_col: str = "degradation_slope") -> float:
    sub = slopes_df[
        (slopes_df["animal_id"].astype(str) == str(animal)) &
        (slopes_df["label"].astype(str) == str(label)) &
        (slopes_df["feature"] == feature) &
        (slopes_df["predictor"] == predictor) &
        (slopes_df["group_col"] == group_col) &
        (slopes_df["group"] == group)
    ]
    if sub.empty:
        return np.nan
    return float(sub[value_col].iloc[0])


def compute_label_effects(slopes_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_df = slopes_df[["animal_id", "label", "feature", "predictor"]].drop_duplicates()
    for _, key in key_df.iterrows():
        animal = str(key["animal_id"])
        label = str(key["label"])
        feature = key["feature"]
        predictor = key["predictor"]
        pre = pivot_value(slopes_df, animal, label, feature, predictor, "pre_post", "pre")
        post = pivot_value(slopes_df, animal, label, feature, predictor, "pre_post", "post")
        ep = pivot_value(slopes_df, animal, label, feature, predictor, "period", "early_pre")
        lp = pivot_value(slopes_df, animal, label, feature, predictor, "period", "late_pre")
        epost = pivot_value(slopes_df, animal, label, feature, predictor, "period", "early_post")
        lpost = pivot_value(slopes_df, animal, label, feature, predictor, "period", "late_post")
        post_combined = post
        baseline_change = abs(lp - ep) if np.isfinite(lp) and np.isfinite(ep) else np.nan
        lesion_change = abs(post_combined - lp) if np.isfinite(post_combined) and np.isfinite(lp) else np.nan
        rows.append({
            "animal_id": animal,
            "label": label,
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "predictor": predictor,
            "pre_degradation_slope": pre,
            "post_degradation_slope": post,
            "post_minus_pre_degradation_slope": post - pre if np.isfinite(post) and np.isfinite(pre) else np.nan,
            "early_pre_degradation_slope": ep,
            "late_pre_degradation_slope": lp,
            "early_post_degradation_slope": epost,
            "late_post_degradation_slope": lpost,
            "baseline_slope_change_abs": baseline_change,
            "lesion_slope_change_abs": lesion_change,
            "lesion_minus_baseline_slope_change_abs": lesion_change - baseline_change if np.isfinite(lesion_change) and np.isfinite(baseline_change) else np.nan,
        })
    return pd.DataFrame(rows)


def holm_adjust(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = (m - rank + 1) * p[idx]
        running = max(running, adj)
        out[idx] = min(running, 1.0)
    return out


def summarize_by_bird(effects_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    stat_rows = []
    for (feature, predictor), sub in effects_df.groupby(["feature", "predictor"]):
        bird_vals = []
        bird_lesion_minus_baseline = []
        for animal, sb in sub.groupby("animal_id"):
            r = {
                "animal_id": animal,
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "predictor": predictor,
                "n_labels": int(sb.shape[0]),
                "median_pre_degradation_slope": float(np.nanmedian(sb["pre_degradation_slope"])),
                "median_post_degradation_slope": float(np.nanmedian(sb["post_degradation_slope"])),
                "median_post_minus_pre_degradation_slope": float(np.nanmedian(sb["post_minus_pre_degradation_slope"])),
                "median_baseline_slope_change_abs": float(np.nanmedian(sb["baseline_slope_change_abs"])),
                "median_lesion_slope_change_abs": float(np.nanmedian(sb["lesion_slope_change_abs"])),
                "median_lesion_minus_baseline_slope_change_abs": float(np.nanmedian(sb["lesion_minus_baseline_slope_change_abs"])),
            }
            rows.append(r)
            if np.isfinite(r["median_post_minus_pre_degradation_slope"]):
                bird_vals.append(r["median_post_minus_pre_degradation_slope"])
            if np.isfinite(r["median_lesion_minus_baseline_slope_change_abs"]):
                bird_lesion_minus_baseline.append(r["median_lesion_minus_baseline_slope_change_abs"])

        bird_vals = clean_values(bird_vals)
        bird_lesion_minus_baseline = clean_values(bird_lesion_minus_baseline)
        stat = {
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "predictor": predictor,
            "n_birds_post_minus_pre": int(bird_vals.size),
            "median_bird_post_minus_pre_degradation_slope": float(np.nanmedian(bird_vals)) if bird_vals.size else np.nan,
            "wilcoxon_post_minus_pre_gt0_p": np.nan,
            "wilcoxon_post_minus_pre_twosided_p": np.nan,
            "n_birds_lesion_minus_baseline": int(bird_lesion_minus_baseline.size),
            "median_bird_lesion_minus_baseline_slope_change_abs": float(np.nanmedian(bird_lesion_minus_baseline)) if bird_lesion_minus_baseline.size else np.nan,
            "wilcoxon_lesion_minus_baseline_gt0_p": np.nan,
            "wilcoxon_lesion_minus_baseline_twosided_p": np.nan,
            "scipy_available": SCIPY_AVAILABLE,
        }
        if SCIPY_AVAILABLE and bird_vals.size >= 1:
            try:
                w = stats.wilcoxon(bird_vals, alternative="greater", zero_method="wilcox")
                stat["wilcoxon_post_minus_pre_gt0_p"] = float(w.pvalue)
                w2 = stats.wilcoxon(bird_vals, alternative="two-sided", zero_method="wilcox")
                stat["wilcoxon_post_minus_pre_twosided_p"] = float(w2.pvalue)
            except Exception:
                pass
        if SCIPY_AVAILABLE and bird_lesion_minus_baseline.size >= 1:
            try:
                w = stats.wilcoxon(bird_lesion_minus_baseline, alternative="greater", zero_method="wilcox")
                stat["wilcoxon_lesion_minus_baseline_gt0_p"] = float(w.pvalue)
                w2 = stats.wilcoxon(bird_lesion_minus_baseline, alternative="two-sided", zero_method="wilcox")
                stat["wilcoxon_lesion_minus_baseline_twosided_p"] = float(w2.pvalue)
            except Exception:
                pass
        stat_rows.append(stat)

    bird_df = pd.DataFrame(rows)
    stat_df = pd.DataFrame(stat_rows)
    if not stat_df.empty:
        for col in ["wilcoxon_post_minus_pre_gt0_p", "wilcoxon_lesion_minus_baseline_gt0_p"]:
            stat_df[f"{col}_holm"] = holm_adjust(stat_df[col].tolist())
    return bird_df, stat_df


def fmt_p(p) -> str:
    try:
        p = float(p)
    except Exception:
        return "NA"
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def get_stat_label(stat_df: pd.DataFrame, feature: str, predictor: str) -> str:
    if stat_df is None or stat_df.empty:
        return ""
    sub = stat_df[(stat_df["feature"] == feature) & (stat_df["predictor"] == predictor)]
    if sub.empty:
        return ""
    row = sub.iloc[0]
    return (
        f"post>pre degradation slope p={fmt_p(row.get('wilcoxon_post_minus_pre_gt0_p'))}; "
        f"lesion>baseline slope-change p={fmt_p(row.get('wilcoxon_lesion_minus_baseline_gt0_p'))}"
    )


def plot_post_minus_pre_by_bird(effects_df: pd.DataFrame, bird_df: pd.DataFrame, stat_df: pd.DataFrame,
                                feature: str, predictor: str, out_png: Path, title_prefix: str):
    sub_l = effects_df[(effects_df["feature"] == feature) & (effects_df["predictor"] == predictor)].copy()
    sub_b = bird_df[(bird_df["feature"] == feature) & (bird_df["predictor"] == predictor)].copy()
    if sub_l.empty or sub_b.empty:
        return
    sub_b = sub_b.sort_values("animal_id")
    birds = sub_b["animal_id"].tolist()
    xmap = {b: i for i, b in enumerate(birds)}
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in sub_l.iterrows():
        y = row["post_minus_pre_degradation_slope"]
        if not np.isfinite(y):
            continue
        jitter = ((hash((row["animal_id"], str(row["label"]), feature, predictor)) % 1000) / 1000.0 - 0.5) * 0.22
        ax.scatter(xmap[row["animal_id"]] + jitter, y, s=34, alpha=0.32)
    xs = np.arange(len(birds))
    ys = sub_b["median_post_minus_pre_degradation_slope"].to_numpy(dtype=float)
    ax.scatter(xs, ys, s=100, marker="D", edgecolors="black", label="Bird median")
    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(birds, rotation=30, ha="right")
    ylabel = "Post - pre degradation slope"
    ylabel += " per second" if predictor == "elapsed_time_in_phrase_s" else " per previous repeat"
    ax.set_ylabel(ylabel)
    stat_label = get_stat_label(stat_df, feature, predictor)
    ax.set_title(f"{title_prefix}\n{FEATURE_LABELS.get(feature, feature)}\n{stat_label}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_pre_vs_post_slope(effects_df: pd.DataFrame, bird_df: pd.DataFrame, stat_df: pd.DataFrame,
                           feature: str, predictor: str, out_png: Path, title_prefix: str):
    sub_l = effects_df[(effects_df["feature"] == feature) & (effects_df["predictor"] == predictor)].copy()
    sub_b = bird_df[(bird_df["feature"] == feature) & (bird_df["predictor"] == predictor)].copy()
    if sub_l.empty or sub_b.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    birds = sorted(sub_l["animal_id"].unique().tolist())
    cmap = plt.cm.get_cmap("tab10", len(birds))
    b2c = {b: cmap(i) for i, b in enumerate(birds)}
    xvals, yvals = [], []
    for _, row in sub_l.iterrows():
        x = row["pre_degradation_slope"]
        y = row["post_degradation_slope"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xvals.append(float(x)); yvals.append(float(y))
        ax.scatter(x, y, s=34, alpha=0.38, color=b2c[row["animal_id"]])
    for _, row in sub_b.iterrows():
        x = row["median_pre_degradation_slope"]
        y = row["median_post_degradation_slope"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xvals.append(float(x)); yvals.append(float(y))
        ax.scatter(x, y, s=105, marker="D", edgecolors="black", color=b2c.get(row["animal_id"], None))
        ax.text(x, y, row["animal_id"], fontsize=8, ha="left", va="bottom")
    if not xvals:
        plt.close(fig); return
    lo = min(min(xvals), min(yvals)); hi = max(max(xvals), max(yvals))
    span = hi - lo
    pad = max(0.06 * span, 0.02 * max(abs(lo), abs(hi), 1.0))
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="equal slope")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    unit = "per second" if predictor == "elapsed_time_in_phrase_s" else "per previous repeat"
    ax.set_xlabel(f"Pre-lesion degradation slope ({unit})")
    ax.set_ylabel(f"Post-lesion degradation slope ({unit})")
    stat_label = get_stat_label(stat_df, feature, predictor)
    ax.set_title(f"{title_prefix}\n{FEATURE_LABELS.get(feature, feature)}\n{stat_label}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_baseline_vs_lesion_slope_change(effects_df: pd.DataFrame, bird_df: pd.DataFrame, stat_df: pd.DataFrame,
                                         feature: str, predictor: str, out_png: Path, title_prefix: str):
    sub_l = effects_df[(effects_df["feature"] == feature) & (effects_df["predictor"] == predictor)].copy()
    sub_b = bird_df[(bird_df["feature"] == feature) & (bird_df["predictor"] == predictor)].copy()
    if sub_l.empty or sub_b.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    birds = sorted(sub_l["animal_id"].unique().tolist())
    cmap = plt.cm.get_cmap("tab10", len(birds))
    b2c = {b: cmap(i) for i, b in enumerate(birds)}
    xvals, yvals = [], []
    for _, row in sub_l.iterrows():
        x = row["baseline_slope_change_abs"]
        y = row["lesion_slope_change_abs"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xvals.append(float(x)); yvals.append(float(y))
        ax.scatter(x, y, s=34, alpha=0.38, color=b2c[row["animal_id"]])
    for _, row in sub_b.iterrows():
        x = row["median_baseline_slope_change_abs"]
        y = row["median_lesion_slope_change_abs"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xvals.append(float(x)); yvals.append(float(y))
        ax.scatter(x, y, s=105, marker="D", edgecolors="black", color=b2c.get(row["animal_id"], None))
        ax.text(x, y, row["animal_id"], fontsize=8, ha="left", va="bottom")
    if not xvals:
        plt.close(fig); return
    lo = min(min(xvals), min(yvals)); hi = max(max(xvals), max(yvals))
    span = hi - lo
    pad = max(0.06 * span, 0.02 * max(abs(lo), abs(hi), 1.0))
    lo = max(0.0, lo - pad); hi += pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="equal slope change")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    unit = "per second" if predictor == "elapsed_time_in_phrase_s" else "per previous repeat"
    ax.set_xlabel(f"Baseline slope change |late pre - early pre| ({unit})")
    ax.set_ylabel(f"Lesion slope change |post - late pre| ({unit})")
    stat_label = get_stat_label(stat_df, feature, predictor)
    ax.set_title(f"{title_prefix}\n{FEATURE_LABELS.get(feature, feature)}\n{stat_label}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def binned_mean_sem(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (x >= lo) & (x < hi) & np.isfinite(y)
        vals = y[mask]
        if vals.size == 0:
            continue
        rows.append({
            "x_mid": (lo + hi) / 2,
            "n": int(vals.size),
            "mean": float(np.nanmean(vals)),
            "sem": float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def plot_binned_feature_vs_position(df: pd.DataFrame, feature: str, predictor: str, out_png: Path,
                                    title_prefix: str, max_x: Optional[float] = None, n_bins: int = 30):
    if feature not in df.columns or predictor not in df.columns:
        return
    use = ensure_pre_post(df.copy())
    x = use[predictor].to_numpy(dtype=float)
    y = use[feature].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if max_x is not None:
        valid &= x <= max_x
    if np.sum(valid) < 5:
        return
    x_valid = x[valid]
    x_min, x_max = np.nanmin(x_valid), np.nanmax(x_valid)
    if x_min == x_max:
        return
    bins = np.linspace(x_min, x_max, n_bins + 1)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for group in PRE_POST_ORDER:
        sub = use[(use["pre_post"] == group)].copy()
        xx = sub[predictor].to_numpy(dtype=float)
        yy = sub[feature].to_numpy(dtype=float)
        if max_x is not None:
            keep = xx <= max_x
            xx = xx[keep]; yy = yy[keep]
        b = binned_mean_sem(xx, yy, bins)
        if b.empty:
            continue
        label = f"{group} (binned mean ± SEM)"
        ax.plot(b["x_mid"], b["mean"], lw=2.2, label=label)
        ax.fill_between(b["x_mid"], b["mean"] - b["sem"], b["mean"] + b["sem"], alpha=0.18)
    xlabel = "Elapsed time in phrase (s)" if predictor == "elapsed_time_in_phrase_s" else "Number of previous repeats"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
    ax.set_title(f"{title_prefix}\n{FEATURE_LABELS.get(feature, feature)} vs phrase position")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate phrase-position spectrogram cross-correlation analyses across Medial+Lateral lesion birds.")
    p.add_argument("--phrase-position-root", required=True,
                   help="Root folder containing outputs from phrase_position_acoustic_crosscorr_v2.py")
    p.add_argument("--out-dir", default=None,
                   help="Output folder. Default: <phrase-position-root>/aggregate_crosscorr_v3")
    p.add_argument("--metadata-excel-path", default=None,
                   help="Metadata Excel for Medial+Lateral bird filtering")
    p.add_argument("--lesion-filter", choices=["medial_lateral", "all"], default="medial_lateral")
    p.add_argument("--lesion-metadata-sheet", default="animal_hit_type_summary")
    p.add_argument("--lesion-animal-col", default=None)
    p.add_argument("--lesion-hit-type-col", default=None)
    p.add_argument("--animal-ids", default=None,
                   help="Optional comma-separated animal IDs to include after lesion filtering")
    p.add_argument("--features", default="corr_to_phrase_early_template,distance_from_phrase_early_template",
                   help="Comma-separated features to aggregate")
    p.add_argument("--min-values-per-group", type=int, default=5)
    p.add_argument("--max-plot-elapsed-s", type=float, default=20.0)
    p.add_argument("--max-plot-previous-repeats", type=float, default=80.0)
    p.add_argument("--title-prefix", default="Medial + Lateral birds")
    p.add_argument("--save-combined-segments", action="store_true",
                   help="Save combined segment-feature table; can be large.")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.phrase_position_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "aggregate_crosscorr_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_animals = None
    lesion_meta_used = None
    if args.lesion_filter == "medial_lateral":
        if args.metadata_excel_path is None:
            raise ValueError("--metadata-excel-path is required when --lesion-filter medial_lateral")
        allowed_animals, lesion_meta_used = load_medial_lateral_animals(
            metadata_excel_path=args.metadata_excel_path,
            sheet_name=args.lesion_metadata_sheet,
            animal_col=args.lesion_animal_col,
            hit_type_col=args.lesion_hit_type_col,
        )
        print(f"[INFO] Medial+Lateral animals: {sorted(allowed_animals)}")
        lesion_meta_used.to_csv(out_dir / "lesion_metadata_with_medial_lateral_filter.csv", index=False)
        print(f"[SAVED] {out_dir / 'lesion_metadata_with_medial_lateral_filter.csv'}")

    if args.animal_ids:
        requested = {x.strip() for x in args.animal_ids.split(",") if x.strip()}
        allowed_animals = requested if allowed_animals is None else (allowed_animals & requested)
        print(f"[INFO] Requested animals after lesion filtering: {sorted(allowed_animals)}")

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    segment_df, used_files_df = read_segment_feature_tables(root, allowed_animals=allowed_animals)
    segment_df = ensure_pre_post(segment_df)
    used_files_df.to_csv(out_dir / "input_phrase_position_segment_files_used.csv", index=False)
    print(f"[SAVED] {out_dir / 'input_phrase_position_segment_files_used.csv'}")
    if args.save_combined_segments:
        segment_df.to_csv(out_dir / "combined_phrase_position_segment_features_used.csv", index=False)
        print(f"[SAVED] {out_dir / 'combined_phrase_position_segment_features_used.csv'}")

    slopes_df = compute_label_slopes(segment_df, features=features, min_n=args.min_values_per_group)
    slopes_df.to_csv(out_dir / "phrase_position_crosscorr_label_slopes.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_crosscorr_label_slopes.csv'}")

    effects_df = compute_label_effects(slopes_df)
    effects_df.to_csv(out_dir / "phrase_position_crosscorr_label_effects.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_crosscorr_label_effects.csv'}")

    bird_df, stat_df = summarize_by_bird(effects_df)
    bird_df.to_csv(out_dir / "phrase_position_crosscorr_bird_summary.csv", index=False)
    stat_df.to_csv(out_dir / "phrase_position_crosscorr_bird_stats.csv", index=False)
    print(f"[SAVED] {out_dir / 'phrase_position_crosscorr_bird_summary.csv'}")
    print(f"[SAVED] {out_dir / 'phrase_position_crosscorr_bird_stats.csv'}")

    predictors = ["elapsed_time_in_phrase_s", "n_previous_segments"]
    for feature in features:
        for predictor in predictors:
            pred_tag = "elapsed_time" if predictor == "elapsed_time_in_phrase_s" else "previous_repeats"
            plot_post_minus_pre_by_bird(
                effects_df, bird_df, stat_df, feature, predictor,
                out_dir / f"post_minus_pre_degradation_slope_by_bird_{safe_name(feature)}_{pred_tag}.png",
                args.title_prefix,
            )
            plot_pre_vs_post_slope(
                effects_df, bird_df, stat_df, feature, predictor,
                out_dir / f"pre_vs_post_degradation_slope_{safe_name(feature)}_{pred_tag}.png",
                args.title_prefix,
            )
            plot_baseline_vs_lesion_slope_change(
                effects_df, bird_df, stat_df, feature, predictor,
                out_dir / f"baseline_vs_lesion_slope_change_{safe_name(feature)}_{pred_tag}.png",
                args.title_prefix,
            )
            max_x = args.max_plot_elapsed_s if predictor == "elapsed_time_in_phrase_s" else args.max_plot_previous_repeats
            plot_binned_feature_vs_position(
                segment_df, feature, predictor,
                out_dir / f"binned_{safe_name(feature)}_vs_{pred_tag}_pre_post.png",
                args.title_prefix,
                max_x=max_x,
            )

    readme = out_dir / "README_phrase_position_crosscorr_aggregate_v3.txt"
    readme.write_text(
        "Aggregate phrase-position cross-correlation analysis.\n\n"
        "Key interpretation:\n"
        "- For corr_to_phrase_early_template, more degradation over phrase position means a more negative raw slope.\n"
        "  The script converts this to degradation_slope = -raw slope.\n"
        "- For distance_from_phrase_early_template, more degradation means a more positive raw slope.\n"
        "  The script uses degradation_slope = raw slope.\n"
        "- post_minus_pre_degradation_slope > 0 means post-lesion phrases degrade more strongly with repeats/time than pre-lesion phrases.\n\n"
        "Main tables:\n"
        "- phrase_position_crosscorr_label_slopes.csv\n"
        "- phrase_position_crosscorr_label_effects.csv\n"
        "- phrase_position_crosscorr_bird_summary.csv\n"
        "- phrase_position_crosscorr_bird_stats.csv\n\n"
        "Main figures:\n"
        "- post_minus_pre_degradation_slope_by_bird_*.png\n"
        "- pre_vs_post_degradation_slope_*.png\n"
        "- baseline_vs_lesion_slope_change_*.png\n"
        "- binned_*_vs_elapsed_time_pre_post.png\n"
        "- binned_*_vs_previous_repeats_pre_post.png\n"
    )
    print(f"[SAVED] {readme}")
    print("[DONE]")


if __name__ == "__main__":
    main()
