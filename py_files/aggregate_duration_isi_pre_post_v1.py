#!/usr/bin/env python3
"""
Aggregate top-variance phrase-position duration/ISI outputs across birds.

This script reads CSVs produced by:
    phrase_position_duration_isi_majority_vote_v2_blue_red_mean.py

It tests whether high-phrase-duration-variance syllables change after lesion in:
    1) syllable duration: segment_duration_ms
    2) inter-syllabic interval: preceding_isi_ms

Main inference avoids treating thousands of segment rows as independent birds:
    segment rows -> animal x label pre/post summaries -> bird summaries -> across-bird tests

Outputs:
    duration_isi_all_segments_used.csv
    duration_isi_pre_post_label_tests.csv
    duration_isi_pre_post_bird_summary.csv
    duration_isi_pre_post_bird_stats.csv
    duration_isi_pre_post_feature_interpretation.csv
    figures/*.png
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

PRE_COLOR = "#1f77b4"   # blue
POST_COLOR = "#d62728"  # red
DELTA_COLOR = "#444444"
POINT_ALPHA = 0.75

ANIMAL_COL_CANDIDATES = [
    "Animal ID", "animal_id", "Animal", "animal", "Bird", "bird", "bird_id", "AnimalID"
]
LABEL_COL_CANDIDATES = [
    "label", "Label", "syllable", "Syllable", "syllable_label", "Syllable label",
    "cluster", "Cluster", "cluster_id", "Cluster ID", "hdbscan_label", "HDBSCAN label",
    "mapped_syllable_order", "syllable_cluster", "Syllable cluster", "chosen_label", "selected_label"
]
FEATURE_LABELS = {
    "segment_duration_ms": "Syllable duration (ms)",
    "preceding_isi_ms": "Preceding inter-syllabic interval (ms)",
    "following_isi_ms": "Following inter-syllabic interval (ms)",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def find_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_animal_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    m = re.search(r"(?:^|_)label([^_\s/\\]+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^label([^_\s/\\]+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def finite_values(x) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def safe_median(x) -> float:
    arr = finite_values(x)
    return float(np.median(arr)) if arr.size else np.nan


def safe_mean(x) -> float:
    arr = finite_values(x)
    return float(np.mean(arr)) if arr.size else np.nan


def safe_iqr(x) -> float:
    arr = finite_values(x)
    if arr.size == 0:
        return np.nan
    q25, q75 = np.percentile(arr, [25, 75])
    return float(q75 - q25)


def cliffs_delta(x, y) -> float:
    """Cliff's delta for y vs x: positive means y tends to be larger than x."""
    x = finite_values(x)
    y = finite_values(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    # Chunk over y so this stays memory-safe for large segment counts.
    greater = 0
    less = 0
    x_sorted = np.sort(x)
    for val in y:
        less += int(np.searchsorted(x_sorted, val, side="left"))
        greater += int(x.size - np.searchsorted(x_sorted, val, side="right"))
    return float((greater - less) / (x.size * y.size))


def mannwhitney(pre, post) -> tuple[float, float, float]:
    pre = finite_values(pre)
    post = finite_values(post)
    if pre.size < 1 or post.size < 1 or not SCIPY_AVAILABLE:
        return np.nan, np.nan, np.nan
    try:
        two = stats.mannwhitneyu(post, pre, alternative="two-sided")
        greater = stats.mannwhitneyu(post, pre, alternative="greater")
        less = stats.mannwhitneyu(post, pre, alternative="less")
        return float(two.statistic), float(two.pvalue), float(greater.pvalue), float(less.pvalue)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def one_sample_wilcoxon(x) -> tuple[float, float, float, float]:
    x = finite_values(x)
    x = x[x != 0]
    if x.size < 1 or not SCIPY_AVAILABLE:
        return np.nan, np.nan, np.nan, np.nan
    try:
        two = stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox")
        greater = stats.wilcoxon(x, alternative="greater", zero_method="wilcox")
        less = stats.wilcoxon(x, alternative="less", zero_method="wilcox")
        return float(two.statistic), float(two.pvalue), float(greater.pvalue), float(less.pvalue)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def sign_test_pvalues(x) -> tuple[int, int, int, float, float, float]:
    x = finite_values(x)
    x = x[x != 0]
    n_pos = int(np.sum(x > 0))
    n_neg = int(np.sum(x < 0))
    n = n_pos + n_neg
    if n == 0 or not SCIPY_AVAILABLE:
        return n_pos, n_neg, n, np.nan, np.nan, np.nan
    try:
        two = stats.binomtest(n_pos, n=n, p=0.5, alternative="two-sided").pvalue
        greater = stats.binomtest(n_pos, n=n, p=0.5, alternative="greater").pvalue
        less = stats.binomtest(n_pos, n=n, p=0.5, alternative="less").pvalue
        return n_pos, n_neg, n, float(two), float(greater), float(less)
    except Exception:
        return n_pos, n_neg, n, np.nan, np.nan, np.nan


def holm_adjust(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
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


def infer_animal_from_path(path: Path) -> str:
    m = re.search(r"(?P<animal>[^/\\]+)_all_labels_duration_isi_segment_timing\.csv$", path.name)
    if m:
        return m.group("animal")
    m = re.search(r"(?P<animal>[^/\\]+)_label.+?_duration_isi_segment_timing\.csv$", path.name)
    if m:
        return m.group("animal")
    return path.parent.name


def find_segment_timing_files(root: Path) -> list[Path]:
    all_label_files = sorted(root.rglob("*_all_labels_duration_isi_segment_timing.csv"))
    if all_label_files:
        # Prefer all-label files and use one per animal to avoid double-counting per-label files.
        by_animal = {}
        for f in all_label_files:
            animal = infer_animal_from_path(f)
            by_animal.setdefault(animal, f)
        return sorted(by_animal.values())
    return sorted(root.rglob("*_duration_isi_segment_timing.csv"))


def load_high_variance_pairs(path: str | None, animal_col: str | None = None, label_col: str | None = None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    table = pd.read_csv(Path(path).expanduser())
    animal_col = animal_col or find_first_existing(table.columns, ANIMAL_COL_CANDIDATES)
    label_col = label_col or find_first_existing(table.columns, LABEL_COL_CANDIDATES)
    if animal_col is None or label_col is None:
        raise ValueError(f"Could not find animal/label columns in {path}. Columns: {list(table.columns)}")
    pairs = set()
    for _, row in table.iterrows():
        animal = normalize_animal_id(row[animal_col])
        label = normalize_label_id(row[label_col])
        if animal and label:
            pairs.add((animal, label))
    return pairs


def read_all_segments(args) -> pd.DataFrame:
    root = Path(args.timing_root).expanduser()
    files = find_segment_timing_files(root)
    if not files:
        raise FileNotFoundError(f"No *_duration_isi_segment_timing.csv files found under {root}")

    frames = []
    used = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"[WARN] Could not read {f}: {exc}")
            continue
        if df.empty:
            continue
        if "animal_id" not in df.columns:
            df["animal_id"] = infer_animal_from_path(f)
        if "label" not in df.columns:
            print(f"[WARN] Skipping {f}; no label column")
            continue
        df["animal_id"] = df["animal_id"].map(normalize_animal_id)
        df["label"] = df["label"].map(normalize_label_id)
        if "pre_post" not in df.columns and "period" in df.columns:
            df["pre_post"] = np.where(df["period"].astype(str).str.contains("post", case=False, na=False), "post", "pre")
        if "pre_post" not in df.columns:
            print(f"[WARN] Skipping {f}; no pre_post or period column")
            continue
        df["source_csv"] = str(f)
        frames.append(df)
        used.append({
            "csv_path": str(f),
            "animal_id": ",".join(sorted(df["animal_id"].dropna().astype(str).unique())),
            "n_rows": int(df.shape[0]),
            "n_labels": int(df["label"].nunique()),
        })
    if not frames:
        raise RuntimeError("No usable segment timing rows found.")
    all_df = pd.concat(frames, ignore_index=True)
    used_df = pd.DataFrame(used)
    return all_df, used_df


def compute_label_tests(df: pd.DataFrame, features: list[str], min_n: int) -> pd.DataFrame:
    rows = []
    for (animal, label), sub in df.groupby(["animal_id", "label"], dropna=False):
        pre_rows = sub[sub["pre_post"].astype(str).str.lower() == "pre"]
        post_rows = sub[sub["pre_post"].astype(str).str.lower() == "post"]
        for feature in features:
            if feature not in sub.columns:
                print(f"[WARN] Missing feature column {feature}; skipping")
                continue
            pre = finite_values(pre_rows[feature])
            post = finite_values(post_rows[feature])
            n_pre = int(pre.size)
            n_post = int(post.size)
            if n_pre < min_n or n_post < min_n:
                status = "too_few_values"
            else:
                status = "ok"
            u, p_two, p_gt, p_lt = mannwhitney(pre, post) if status == "ok" else (np.nan, np.nan, np.nan, np.nan)
            pre_med = safe_median(pre)
            post_med = safe_median(post)
            pre_mean = safe_mean(pre)
            post_mean = safe_mean(post)
            delta_med = post_med - pre_med if np.isfinite(pre_med) and np.isfinite(post_med) else np.nan
            delta_mean = post_mean - pre_mean if np.isfinite(pre_mean) and np.isfinite(post_mean) else np.nan
            pct_med = 100.0 * delta_med / pre_med if np.isfinite(delta_med) and np.isfinite(pre_med) and pre_med != 0 else np.nan
            rows.append({
                "animal_id": animal,
                "label": label,
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "n_pre_segments": n_pre,
                "n_post_segments": n_post,
                "pre_median": pre_med,
                "post_median": post_med,
                "post_minus_pre_median": delta_med,
                "post_minus_pre_median_percent": pct_med,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "post_minus_pre_mean": delta_mean,
                "pre_iqr": safe_iqr(pre),
                "post_iqr": safe_iqr(post),
                "mannwhitney_u_post_vs_pre": u,
                "mannwhitney_p_two_sided": p_two,
                "mannwhitney_p_post_greater_pre": p_gt,
                "mannwhitney_p_post_less_pre": p_lt,
                "cliffs_delta_post_vs_pre": cliffs_delta(pre, post) if status == "ok" else np.nan,
                "test_status": status,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ["mannwhitney_p_two_sided", "mannwhitney_p_post_greater_pre", "mannwhitney_p_post_less_pre"]:
            out[col + "_holm_by_feature"] = np.nan
            for feat, idx in out.groupby("feature").groups.items():
                out.loc[idx, col + "_holm_by_feature"] = holm_adjust(out.loc[idx, col])
    return out


def compute_bird_summary(label_tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = label_tests[label_tests["test_status"] == "ok"].copy()
    for (animal, feature), sub in ok.groupby(["animal_id", "feature"], dropna=False):
        deltas = finite_values(sub["post_minus_pre_median"])
        pct = finite_values(sub["post_minus_pre_median_percent"])
        pre_meds = finite_values(sub["pre_median"])
        post_meds = finite_values(sub["post_median"])
        rows.append({
            "animal_id": animal,
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "n_labels": int(sub["label"].nunique()),
            "n_labels_post_gt_pre": int(np.sum(pd.to_numeric(sub["post_minus_pre_median"], errors="coerce") > 0)),
            "n_labels_post_lt_pre": int(np.sum(pd.to_numeric(sub["post_minus_pre_median"], errors="coerce") < 0)),
            "fraction_labels_post_gt_pre": float(np.mean(pd.to_numeric(sub["post_minus_pre_median"], errors="coerce") > 0)) if sub.shape[0] else np.nan,
            "bird_pre_median_across_labels": float(np.median(pre_meds)) if pre_meds.size else np.nan,
            "bird_post_median_across_labels": float(np.median(post_meds)) if post_meds.size else np.nan,
            "bird_post_minus_pre_median_of_label_medians": float(np.median(deltas)) if deltas.size else np.nan,
            "bird_post_minus_pre_mean_of_label_medians": float(np.mean(deltas)) if deltas.size else np.nan,
            "bird_post_minus_pre_median_percent": float(np.median(pct)) if pct.size else np.nan,
        })
    return pd.DataFrame(rows)


def compute_bird_stats(bird_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, sub in bird_summary.groupby("feature", dropna=False):
        delta = finite_values(sub["bird_post_minus_pre_median_of_label_medians"])
        pre = finite_values(sub["bird_pre_median_across_labels"])
        post = finite_values(sub["bird_post_median_across_labels"])
        w_stat, w_p_two, w_p_gt, w_p_lt = one_sample_wilcoxon(delta)
        n_pos, n_neg, n_sign, sign_p_two, sign_p_gt, sign_p_lt = sign_test_pvalues(delta)
        paired_delta = post - pre if pre.size == post.size else delta
        pw_stat, pw_p_two, pw_p_gt, pw_p_lt = one_sample_wilcoxon(paired_delta)
        rows.append({
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "n_birds": int(delta.size),
            "median_bird_delta": float(np.median(delta)) if delta.size else np.nan,
            "mean_bird_delta": float(np.mean(delta)) if delta.size else np.nan,
            "sem_bird_delta": float(np.std(delta, ddof=1) / np.sqrt(delta.size)) if delta.size > 1 else np.nan,
            "n_birds_post_gt_pre": n_pos,
            "n_birds_post_lt_pre": n_neg,
            "wilcoxon_delta_vs_zero_stat": w_stat,
            "wilcoxon_delta_vs_zero_p_two_sided": w_p_two,
            "wilcoxon_delta_vs_zero_p_post_greater_pre": w_p_gt,
            "wilcoxon_delta_vs_zero_p_post_less_pre": w_p_lt,
            "sign_test_n_nonzero": n_sign,
            "sign_test_p_two_sided": sign_p_two,
            "sign_test_p_post_greater_pre": sign_p_gt,
            "sign_test_p_post_less_pre": sign_p_lt,
            "paired_wilcoxon_pre_vs_post_stat": pw_stat,
            "paired_wilcoxon_pre_vs_post_p_two_sided": pw_p_two,
            "paired_wilcoxon_pre_vs_post_p_post_greater_pre": pw_p_gt,
            "paired_wilcoxon_pre_vs_post_p_post_less_pre": pw_p_lt,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        pcols = [c for c in out.columns if c.endswith("p_two_sided") or c.endswith("p_post_greater_pre") or c.endswith("p_post_less_pre")]
        for c in pcols:
            out[c + "_holm"] = holm_adjust(out[c])
    return out


def plot_bird_delta(bird_summary: pd.DataFrame, feature: str, out_png: Path):
    sub = bird_summary[bird_summary["feature"] == feature].copy().sort_values("animal_id")
    if sub.empty:
        return
    y = pd.to_numeric(sub["bird_post_minus_pre_median_of_label_medians"], errors="coerce")
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * sub.shape[0] + 3), 5))
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    x = np.arange(sub.shape[0])
    ax.scatter(x, y, color=DELTA_COLOR, alpha=POINT_ALPHA, s=55)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["animal_id"], rotation=45, ha="right")
    ax.set_ylabel(f"Post - pre median\n{FEATURE_LABELS.get(feature, feature)}")
    ax.set_title(f"Bird-level post-pre change: {FEATURE_LABELS.get(feature, feature)}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bird_pre_post(bird_summary: pd.DataFrame, feature: str, out_png: Path):
    sub = bird_summary[bird_summary["feature"] == feature].copy().sort_values("animal_id")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for _, row in sub.iterrows():
        pre = row["bird_pre_median_across_labels"]
        post = row["bird_post_median_across_labels"]
        if np.isfinite(pre) and np.isfinite(post):
            ax.plot([0, 1], [pre, post], color="0.75", linewidth=1.2, zorder=1)
            ax.scatter([0], [pre], color=PRE_COLOR, s=45, alpha=0.85, zorder=2)
            ax.scatter([1], [post], color=POST_COLOR, s=45, alpha=0.85, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pre", "post"])
    ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
    ax.set_title(f"Bird medians across high-variance labels: {FEATURE_LABELS.get(feature, feature)}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(stats_df: pd.DataFrame, out_csv: Path):
    rows = []
    for _, row in stats_df.iterrows():
        feature = row["feature"]
        delta = row["median_bird_delta"]
        if feature == "segment_duration_ms":
            measure = "syllable duration"
        elif feature == "preceding_isi_ms":
            measure = "time gap before the syllable / inter-syllabic interval"
        elif feature == "following_isi_ms":
            measure = "time gap after the syllable / inter-syllabic interval"
        else:
            measure = feature
        if np.isfinite(delta):
            direction = "increased post-lesion" if delta > 0 else "decreased post-lesion" if delta < 0 else "did not change in median direction"
        else:
            direction = "could not be summarized"
        rows.append({
            "feature": feature,
            "measure": measure,
            "direction_of_median_bird_delta": direction,
            "median_bird_delta": delta,
            "primary_two_sided_p": row.get("wilcoxon_delta_vs_zero_p_two_sided", np.nan),
            "primary_post_greater_pre_p": row.get("wilcoxon_delta_vs_zero_p_post_greater_pre", np.nan),
            "primary_post_less_pre_p": row.get("wilcoxon_delta_vs_zero_p_post_less_pre", np.nan),
            "plain_language": (
                f"Positive post-pre values mean {measure} is longer after lesion; "
                f"negative values mean {measure} is shorter after lesion."
            ),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Aggregate duration/ISI pre-vs-post changes across birds.")
    parser.add_argument("--timing-root", required=True, help="Root containing per-bird output folders from the timing script")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--features", default="segment_duration_ms,preceding_isi_ms",
                        help="Comma-separated features. Use preceding_isi_ms as the primary ISI measure.")
    parser.add_argument("--min-values-per-label", type=int, default=5)
    parser.add_argument("--high-variance-labels-csv", default=None,
                        help="Optional animal x label CSV to filter to top-variance labels again during aggregation.")
    parser.add_argument("--animal-col", default=None)
    parser.add_argument("--label-col", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    segments, used_files = read_all_segments(args)

    selected_pairs = load_high_variance_pairs(args.high_variance_labels_csv, args.animal_col, args.label_col)
    if selected_pairs is not None:
        before = segments.shape[0]
        segments = segments[
            [(a, l) in selected_pairs for a, l in zip(segments["animal_id"], segments["label"])]
        ].copy()
        print(f"[INFO] High-variance filter kept {segments.shape[0]} / {before} segment rows")
        if segments.empty:
            raise RuntimeError("High-variance filter removed all segment rows; check animal/label columns.")

    used_files.to_csv(out_dir / "duration_isi_input_files_used.csv", index=False)
    segments.to_csv(out_dir / "duration_isi_all_segments_used.csv", index=False)

    label_tests = compute_label_tests(segments, features, min_n=args.min_values_per_label)
    label_tests.to_csv(out_dir / "duration_isi_pre_post_label_tests.csv", index=False)

    bird_summary = compute_bird_summary(label_tests)
    bird_summary.to_csv(out_dir / "duration_isi_pre_post_bird_summary.csv", index=False)

    bird_stats = compute_bird_stats(bird_summary)
    bird_stats.to_csv(out_dir / "duration_isi_pre_post_bird_stats.csv", index=False)
    write_interpretation(bird_stats, out_dir / "duration_isi_pre_post_feature_interpretation.csv")

    for feature in features:
        plot_bird_delta(bird_summary, feature, fig_dir / f"bird_post_minus_pre_delta_{safe_name(feature)}.png")
        plot_bird_pre_post(bird_summary, feature, fig_dir / f"bird_pre_vs_post_medians_{safe_name(feature)}.png")

    print(f"[SAVED] {out_dir / 'duration_isi_pre_post_label_tests.csv'}")
    print(f"[SAVED] {out_dir / 'duration_isi_pre_post_bird_summary.csv'}")
    print(f"[SAVED] {out_dir / 'duration_isi_pre_post_bird_stats.csv'}")
    print(f"[SAVED] figures in {fig_dir}")


if __name__ == "__main__":
    main()
