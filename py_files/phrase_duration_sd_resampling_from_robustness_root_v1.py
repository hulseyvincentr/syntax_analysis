#!/usr/bin/env python3
"""
phrase_duration_sd_resampling_from_robustness_root_v1.py

Build Supplemental Table 1 and bird-level bootstrap/permutation contrasts from
an existing phrase_duration_robustness_YYYY-MM-DD folder.

This is designed for robustness folders made by:
    phrase_duration_robustness_with_pooled_selection_v1.py

It reads each run's:
    <robustness_root>/<run_folder>/panel_C_change_metrics_animal_level.csv

and writes:
    Supplemental_Table_1_group_summary.csv
    Supplemental_Table_1_pairwise_contrasts.csv
    Supplemental_Table_1_animal_level.csv
    Supplemental_Table_1_combined_for_copy_paste.csv
    phrase_duration_delta_bootstrap_contrast_<primary>.png/.pdf

This is a bird-level aggregate resampling analysis. It does NOT resample raw
phrase renditions; for that, use the rendition-level hierarchical script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL_ML = "Partial Medial and Lateral lesion"
COMPLETE_ML = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"

GROUP_ORDER = [SHAM, LATERAL, PARTIAL_ML, COMPLETE_ML, POOLED_ML]
PLOT_GROUP_ORDER = [SHAM, LATERAL, POOLED_ML]

COLOR_MAP = {
    SHAM: "#1FA187",
    LATERAL: "#B39DDB",
    PARTIAL_ML: "#7E57C2",
    COMPLETE_ML: "#4A148C",
    POOLED_ML: "#5E35B1",
}

RUNS = [
    ("post_selected_top20", "Post-selected top 20%", "post-lesion selection", "top 20%"),
    ("post_selected_top30", "Post-selected top 30%", "post-lesion selection", "top 30%"),
    ("post_selected_top40", "Post-selected top 40%", "post-lesion selection", "top 40%"),
    ("pooled_prepost_selected_top30", "Pooled pre+post-selected top 30%", "pooled pre/post selection", "top 30%"),
    ("pre_selected_top30_validation", "Pre-selected top 30% validation", "late pre-lesion selection", "top 30%"),
]

CONTRASTS = [
    (POOLED_ML, LATERAL, "primary specificity: medial+lateral > lateral-only"),
    (POOLED_ML, SHAM, "secondary control: medial+lateral > sham"),
    (LATERAL, SHAM, "secondary control: lateral-only > sham"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Supplemental Table 1 from an existing phrase-duration robustness output folder."
    )
    p.add_argument(
        "--robustness-root",
        type=Path,
        required=True,
        help="Folder containing post_selected_top20, post_selected_top30, etc.",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output folder. Default: robustness_root/supp_table_resampling")
    p.add_argument("--prefix", default="phrase_duration", help="Prefix for figure files.")
    p.add_argument("--primary-label", default="Post-selected top 30%", help="Selection method to use for main plot.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--stat", choices=["median", "mean"], default="median")
    p.add_argument("--alternative", choices=["greater", "two-sided"], default="greater")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    p.add_argument(
        "--allow-missing-runs",
        action="store_true",
        help="Skip missing run folders instead of raising an error.",
    )
    return p.parse_args()


def center(values: Iterable[float], stat: str = "median") -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if stat == "mean":
        return float(np.mean(arr))
    return float(np.median(arr))


def pvalue_from_null(obs: float, null: np.ndarray, alternative: str) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    if alternative == "greater":
        count = np.sum(null >= obs)
    elif alternative == "two-sided":
        count = np.sum(np.abs(null) >= abs(obs))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return float((count + 1) / (null.size + 1))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, B: int, stat: str) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    obs = center(values, stat=stat)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        boots[b] = center(rng.choice(values, size=values.size, replace=True), stat=stat)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, B: int, stat: str) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan
    obs = center(a, stat=stat) - center(b, stat=stat)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        aa = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        boots[i] = center(aa, stat=stat) - center(bb, stat=stat)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def signflip_test(values: np.ndarray, rng: np.random.Generator, P: int, stat: str, alternative: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    obs = center(values, stat=stat)
    null = np.empty(P, dtype=float)
    for i in range(P):
        signs = rng.choice(np.array([-1.0, 1.0]), size=values.size, replace=True)
        null[i] = center(values * signs, stat=stat)
    return float(obs), pvalue_from_null(obs, null, alternative=alternative)


def labelshuffle_test(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, P: int, stat: str, alternative: str) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    obs = center(a, stat=stat) - center(b, stat=stat)
    pooled = np.concatenate([a, b])
    n_a = a.size
    null = np.empty(P, dtype=float)
    for i in range(P):
        perm = rng.permutation(pooled)
        null[i] = center(perm[:n_a], stat=stat) - center(perm[n_a:], stat=stat)
    return float(obs), pvalue_from_null(obs, null, alternative=alternative)


def sig_label(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def normalize_group(x: str) -> str:
    s = str(x).strip()
    low = s.lower()
    if s in GROUP_ORDER:
        return s
    if "sham" in low:
        return SHAM
    if "lateral" in low and "medial" not in low:
        return LATERAL
    if "partial" in low and "medial" in low and "lateral" in low:
        return PARTIAL_ML
    if "complete" in low and "medial" in low and "lateral" in low:
        return COMPLETE_ML
    if "not visible" in low:
        return COMPLETE_ML
    if "medial" in low and "lateral" in low:
        return PARTIAL_ML
    return s


def read_one_metrics(metrics_path: Path, method: str, basis: str, fraction: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    required_any_animal = ["Animal ID", "animal_id"]
    if not any(c in df.columns for c in required_any_animal):
        raise ValueError(f"{metrics_path} needs Animal ID or animal_id column")
    if "Animal ID" not in df.columns:
        df["Animal ID"] = df["animal_id"]
    if "display_group" not in df.columns:
        raise ValueError(f"{metrics_path} needs display_group column")

    rename = {}
    if "pre_sd" in df.columns:
        rename["pre_sd"] = "median_pre_sd_ms"
    if "post_sd" in df.columns:
        rename["post_sd"] = "median_post_sd_ms"
    if "sd_delta" in df.columns:
        rename["sd_delta"] = "median_delta_sd_ms"
    df = df.rename(columns=rename)

    needed = ["median_pre_sd_ms", "median_post_sd_ms", "median_delta_sd_ms"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{metrics_path} is missing required metric columns after renaming: {missing}")

    out = df.copy()
    out["selection_method"] = method
    out["selection_basis"] = basis
    out["selection_fraction"] = fraction
    out["lesion_group"] = out["display_group"].map(normalize_group)
    if "n_syllables" not in out.columns:
        out["n_syllables"] = np.nan
    if "mean_delta_sd_ms" not in out.columns:
        out["mean_delta_sd_ms"] = out["median_delta_sd_ms"]

    keep = [
        "selection_method", "selection_basis", "selection_fraction", "Animal ID", "lesion_group",
        "n_syllables", "median_pre_sd_ms", "median_post_sd_ms", "median_delta_sd_ms", "mean_delta_sd_ms",
    ]
    for extra in ["sd_log2_ratio", "median_sd_log2_ratio", "pre_n", "post_n"]:
        if extra in out.columns:
            keep.append(extra)
    out = out[keep].copy()
    for col in ["n_syllables", "median_pre_sd_ms", "median_post_sd_ms", "median_delta_sd_ms", "mean_delta_sd_ms"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_pooled_ml_rows(animal: pd.DataFrame) -> pd.DataFrame:
    pooled = animal[animal["lesion_group"].isin([PARTIAL_ML, COMPLETE_ML])].copy()
    pooled["lesion_group"] = POOLED_ML
    return pd.concat([animal, pooled], ignore_index=True)


def load_robustness_root(root: Path, allow_missing: bool) -> pd.DataFrame:
    rows = []
    for folder, method, basis, fraction in RUNS:
        path = root / folder / "panel_C_change_metrics_animal_level.csv"
        if not path.exists():
            msg = f"Missing {method}: {path}"
            if allow_missing:
                print(f"[WARN] {msg}")
                continue
            raise FileNotFoundError(msg)
        print(f"Loading {method}: {path}")
        rows.append(read_one_metrics(path, method, basis, fraction))
    if not rows:
        raise RuntimeError("No animal-level metrics files were loaded.")
    animal = pd.concat(rows, ignore_index=True)
    animal = animal[animal["lesion_group"].isin([SHAM, LATERAL, PARTIAL_ML, COMPLETE_ML])].copy()
    animal = add_pooled_ml_rows(animal)
    return animal


def make_group_summary(animal: pd.DataFrame, rng: np.random.Generator, B: int, P: int, stat: str, alternative: str) -> pd.DataFrame:
    out_rows = []
    for (method, basis, fraction, group), g in animal.groupby(
        ["selection_method", "selection_basis", "selection_fraction", "lesion_group"], dropna=False
    ):
        values = g["median_delta_sd_ms"].to_numpy(dtype=float)
        obs, lo, hi = bootstrap_ci(values, rng, B, stat)
        _, p_sign = signflip_test(values, rng, P, stat, alternative)
        n_syllables_total = np.nan if g["n_syllables"].isna().all() else float(g["n_syllables"].sum())
        med_syllables = np.nan if g["n_syllables"].isna().all() else center(g["n_syllables"], "median")
        out_rows.append({
            "selection_method": method,
            "selection_basis": basis,
            "selection_fraction": fraction,
            "lesion_group": group,
            "n_birds": int(g["Animal ID"].nunique()),
            "total_selected_syllables": n_syllables_total,
            "median_selected_syllables_per_bird": med_syllables,
            "median_late_pre_SD_ms": center(g["median_pre_sd_ms"], stat),
            "median_post_SD_ms": center(g["median_post_sd_ms"], stat),
            f"{stat}_delta_SD_ms": obs,
            "delta_SD_bootstrap_CI95_low_ms": lo,
            "delta_SD_bootstrap_CI95_high_ms": hi,
            f"signflip_p_delta_gt_0_{alternative}": p_sign,
            "significance": sig_label(p_sign),
            "resampling_level": "bird-level aggregate",
        })
    out = pd.DataFrame(out_rows)
    method_order = {m: i for i, (_, m, _, _) in enumerate(RUNS)}
    group_order = {g: i for i, g in enumerate(GROUP_ORDER)}
    out["_method_order"] = out["selection_method"].map(method_order).fillna(99)
    out["_group_order"] = out["lesion_group"].map(group_order).fillna(99)
    return out.sort_values(["_method_order", "_group_order"]).drop(columns=["_method_order", "_group_order"])


def make_contrasts(animal: pd.DataFrame, rng: np.random.Generator, B: int, P: int, stat: str, alternative: str) -> pd.DataFrame:
    rows = []
    for (method, basis, fraction), mdf in animal.groupby(["selection_method", "selection_basis", "selection_fraction"], dropna=False):
        for group_a, group_b, contrast_type in CONTRASTS:
            a = mdf.loc[mdf["lesion_group"] == group_a, "median_delta_sd_ms"].to_numpy(dtype=float)
            b = mdf.loc[mdf["lesion_group"] == group_b, "median_delta_sd_ms"].to_numpy(dtype=float)
            diff, lo, hi = bootstrap_diff_ci(a, b, rng, B, stat)
            _, p = labelshuffle_test(a, b, rng, P, stat, alternative)
            rows.append({
                "selection_method": method,
                "selection_basis": basis,
                "selection_fraction": fraction,
                "contrast_type": contrast_type,
                "group_A": group_a,
                "group_B": group_b,
                "n_A_birds": int(np.isfinite(a).sum()),
                "n_B_birds": int(np.isfinite(b).sum()),
                f"observed_{stat}_delta_difference_ms_A_minus_B": diff,
                "difference_bootstrap_CI95_low_ms": lo,
                "difference_bootstrap_CI95_high_ms": hi,
                f"labelshuffle_p_A_gt_B_{alternative}": p,
                "significance": sig_label(p),
                "resampling_level": "bird-level aggregate",
            })
    return pd.DataFrame(rows)


def combined_table(group_summary: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    g = group_summary.copy()
    g.insert(0, "table_section", "group summary")
    g["comparison"] = g["lesion_group"]
    c = contrasts.copy()
    c.insert(0, "table_section", "pairwise contrast")
    c["comparison"] = c["group_A"] + " vs " + c["group_B"]
    tmp = pd.concat([g, c], ignore_index=True, sort=False)
    preferred = [
        "table_section", "selection_method", "selection_basis", "selection_fraction", "comparison",
        "n_birds", "n_A_birds", "n_B_birds", "total_selected_syllables", "median_selected_syllables_per_bird",
        "median_late_pre_SD_ms", "median_post_SD_ms", "median_delta_SD_ms", "mean_delta_SD_ms",
        "delta_SD_bootstrap_CI95_low_ms", "delta_SD_bootstrap_CI95_high_ms",
        "observed_median_delta_difference_ms_A_minus_B", "observed_mean_delta_difference_ms_A_minus_B",
        "difference_bootstrap_CI95_low_ms", "difference_bootstrap_CI95_high_ms",
        "significance", "resampling_level",
    ]
    cols = [x for x in preferred if x in tmp.columns] + [x for x in tmp.columns if x not in preferred]
    return tmp[cols]


def sanitize(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_")[:80]


def add_bracket(ax, x1, x2, y, h, text, fontsize=10):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="0.15", linewidth=1.2, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def plot_primary(animal: pd.DataFrame, contrasts: pd.DataFrame, label: str, out_dir: Path, prefix: str, dpi: int, stat: str, show: bool):
    d = animal[(animal["selection_method"] == label) & (animal["lesion_group"].isin(PLOT_GROUP_ORDER))].copy()
    if d.empty:
        print(f"[WARN] No rows for primary plot label: {label}")
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    jitter_rng = np.random.default_rng(321)
    ymin = np.nanmin(d["median_delta_sd_ms"].to_numpy(dtype=float))
    ymax = np.nanmax(d["median_delta_sd_ms"].to_numpy(dtype=float))
    yrange = max(ymax - ymin, 1.0)
    for i, group in enumerate(PLOT_GROUP_ORDER):
        g = d[d["lesion_group"] == group]
        vals = g["median_delta_sd_ms"].to_numpy(dtype=float)
        ax.scatter(
            np.full(vals.size, i) + jitter_rng.normal(0, 0.045, size=vals.size),
            vals,
            s=46,
            color=COLOR_MAP[group],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        obs, lo, hi = bootstrap_ci(vals, np.random.default_rng(777 + i), 3000, stat)
        ax.errorbar([i], [obs], yerr=[[obs - lo], [hi - obs]], fmt="o", color="black", ecolor="black", capsize=7, zorder=4)
        ax.hlines(obs, i - 0.22, i + 0.22, color="black", linewidth=2, zorder=4)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Sham", "Lateral AFP\nonly", "Medial+lateral\nAFP"])
    ax.set_ylabel("Δ phrase-duration SD\n(post-lesion − late pre-lesion, ms)")
    ax.set_title(f"Animal-level ΔSD with bootstrap CI\n{label}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    csub = contrasts[contrasts["selection_method"] == label]
    def get_p(a, b):
        row = csub[(csub["group_A"] == a) & (csub["group_B"] == b)]
        if row.empty:
            return np.nan
        pcols = [c for c in row.columns if c.startswith("labelshuffle_p_A_gt_B")]
        return float(row.iloc[0][pcols[0]]) if pcols else np.nan
    y_top = ymax + 0.12 * yrange
    h = 0.04 * yrange
    p1 = get_p(POOLED_ML, LATERAL)
    p2 = get_p(POOLED_ML, SHAM)
    if np.isfinite(p1):
        add_bracket(ax, 1, 2, y_top, h, f"ML vs lateral: {sig_label(p1)}\np={p1:.4g}")
    if np.isfinite(p2):
        add_bracket(ax, 0, 2, y_top + 0.18 * yrange, h, f"ML vs sham: {sig_label(p2)}\np={p2:.4g}")
    ax.set_ylim(ymin - 0.15 * yrange, y_top + 0.35 * yrange)
    fig.tight_layout()
    png = out_dir / f"{prefix}_delta_bootstrap_contrast_{sanitize(label)}.png"
    pdf = out_dir / f"{prefix}_delta_bootstrap_contrast_{sanitize(label)}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Wrote figure:\n  {png}\n  {pdf}")


def round_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(6)
    return out


def main() -> None:
    args = parse_args()
    root = args.robustness_root.expanduser().resolve()
    out_dir = (args.out_dir or (root / "supplemental_table_1_resampling")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    animal = load_robustness_root(root, allow_missing=args.allow_missing_runs)
    group_summary = make_group_summary(animal, rng, args.n_bootstrap, args.n_permutations, args.stat, args.alternative)
    contrasts = make_contrasts(animal, rng, args.n_bootstrap, args.n_permutations, args.stat, args.alternative)
    combined = combined_table(group_summary, contrasts)

    paths = {
        "group_summary": out_dir / "Supplemental_Table_1_group_summary.csv",
        "pairwise_contrasts": out_dir / "Supplemental_Table_1_pairwise_contrasts.csv",
        "animal_level": out_dir / "Supplemental_Table_1_animal_level.csv",
        "combined": out_dir / "Supplemental_Table_1_combined_for_copy_paste.csv",
    }
    round_df(group_summary).to_csv(paths["group_summary"], index=False)
    round_df(contrasts).to_csv(paths["pairwise_contrasts"], index=False)
    round_df(animal).to_csv(paths["animal_level"], index=False)
    round_df(combined).to_csv(paths["combined"], index=False)

    print("\nWrote tables:")
    for p in paths.values():
        print(f"  {p}")

    plot_primary(animal, contrasts, args.primary_label, out_dir, args.prefix, args.dpi, args.stat, args.show)

    print("\nKey primary contrast rows:")
    key = contrasts[contrasts["contrast_type"].str.contains("primary", na=False)]
    print(round_df(key).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
