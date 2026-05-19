#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_bc_by_lesion_type_v1.py

Aggregate per-bird cluster BC summaries produced by
bc_cluster_qc_and_summaries_v12_local_density_axes.py and make boxplots comparing
pre-lesion vs post-lesion Bhattacharyya coefficients within each lesion type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in low:
            return low[key]
    return None


def _stars_from_p(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _paired_wilcoxon(pre: Sequence[float], post: Sequence[float]) -> Tuple[float, str]:
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    if len(pre) < 2:
        return float("nan"), "n.s."
    try:
        p = float(wilcoxon(pre, post, alternative="two-sided").pvalue)
    except Exception:
        p = float("nan")
    return p, _stars_from_p(p)


def _clean_hit_type(x: Any, *, combine_medial_lateral: bool = True) -> str:
    s = str(x).strip()
    low = s.lower()
    if low in {"", "nan", "none"}:
        return "Unknown"
    if "sham" in low:
        return "sham saline injection"
    if "lateral lesion only" in low:
        return "Lateral lesion only"
    if "single" in low and "hit" in low:
        return "Lateral lesion only"

    is_partial_ml = (
        ("partial" in low and "medial" in low and "lateral" in low)
        or "medial+lateral" in low
        or ("medial lateral" in low and "partial" in low)
    )
    is_complete_ml = (
        ("complete" in low and "medial" in low and "lateral" in low)
        or "large lesion" in low
        or "area x not visible" in low
    )

    if combine_medial_lateral and (is_partial_ml or is_complete_ml or ("medial" in low and "lateral" in low)):
        return "Complete and partial Medial and Lateral lesion"
    if is_partial_ml:
        return "Partial Medial and Lateral lesion"
    if is_complete_ml:
        return "Complete Medial and Lateral lesion"
    return s


def load_metadata_hit_types(metadata_excel_path: Path, *, combine_medial_lateral: bool = True) -> pd.DataFrame:
    sheets = pd.read_excel(metadata_excel_path, sheet_name=None)
    best_df = None
    best_score = -1
    for _, df in sheets.items():
        animal_col = _pick_column(df, ["Animal ID", "animal_id", "animal id", "Bird ID", "bird"])
        hit_col = _pick_column(df, ["Lesion hit type", "hit type", "lesion_hit_type", "Treatment type", "treatment type"])
        score = int(animal_col is not None) + int(hit_col is not None)
        if score > best_score:
            best_df = df.copy()
            best_score = score

    if best_df is None:
        raise ValueError(f"Could not read metadata sheets from {metadata_excel_path}")

    animal_col = _pick_column(best_df, ["Animal ID", "animal_id", "animal id", "Bird ID", "bird"])
    hit_col = _pick_column(best_df, ["Lesion hit type", "hit type", "lesion_hit_type", "Treatment type", "treatment type"])
    if animal_col is None:
        raise KeyError(f"Could not find Animal ID column. Columns: {list(best_df.columns)}")
    if hit_col is None:
        raise KeyError(f"Could not find Lesion hit type column. Columns: {list(best_df.columns)}")

    out = best_df[[animal_col, hit_col]].copy()
    out.columns = ["animal_id", "lesion_hit_type_raw"]
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out = out[out["animal_id"].notna() & (out["animal_id"] != "") & (out["animal_id"].str.lower() != "nan")]
    out["lesion_hit_type"] = out["lesion_hit_type_raw"].map(
        lambda x: _clean_hit_type(x, combine_medial_lateral=combine_medial_lateral)
    )
    return out.drop_duplicates(subset=["animal_id"], keep="first").copy()


def load_all_cluster_summaries(bc_root: Path) -> pd.DataFrame:
    files = sorted(Path(bc_root).glob("*/*_cluster_bc_summary.csv"))
    if not files:
        raise FileNotFoundError(
            f"No per-bird cluster summary CSVs found under {bc_root}. "
            "Expected <bc-root>/<animal_id>/<animal_id>_cluster_bc_summary.csv"
        )

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_csv"] = str(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    if not frames:
        raise ValueError("Found summary paths, but none could be read.")

    out = pd.concat(frames, ignore_index=True)
    if "animal_id" not in out.columns:
        raise KeyError("Cluster summaries must contain 'animal_id'.")

    for col in ["bc_pre", "bc_post", "bc_prepost", "balanced_duration_s_per_group"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "passes_min_balanced_duration" in out.columns:
        s = out["passes_min_balanced_duration"]
        if s.dtype != bool:
            out["passes_min_balanced_duration"] = s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
    else:
        out["passes_min_balanced_duration"] = True

    if "is_high_variance_cluster" in out.columns:
        s = out["is_high_variance_cluster"]
        if s.dtype != bool:
            out["is_high_variance_cluster"] = s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
    else:
        out["is_high_variance_cluster"] = False

    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower().eq("done") | out["status"].isna()].copy()
    return out


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lesion_type, sub in df.groupby("lesion_hit_type", sort=False):
        pre = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
        post = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(pre) & np.isfinite(post)
        pre = pre[mask]
        post = post[mask]
        p, stars = _paired_wilcoxon(pre, post)
        rows.append({
            "lesion_hit_type": lesion_type,
            "n_cluster_rows": int(len(pre)),
            "n_animals": int(sub["animal_id"].nunique()),
            "mean_bc_pre": float(np.nanmean(pre)) if len(pre) else np.nan,
            "mean_bc_post": float(np.nanmean(post)) if len(post) else np.nan,
            "median_bc_pre": float(np.nanmedian(pre)) if len(pre) else np.nan,
            "median_bc_post": float(np.nanmedian(post)) if len(post) else np.nan,
            "mean_post_minus_pre": float(np.nanmean(post - pre)) if len(pre) else np.nan,
            "median_post_minus_pre": float(np.nanmedian(post - pre)) if len(pre) else np.nan,
            "wilcoxon_p": p,
            "significance": stars,
        })
    return pd.DataFrame(rows)


def plot_bc_by_lesion_type(
    df: pd.DataFrame,
    *,
    out_png: Path,
    title: str,
    y_min: Optional[float] = 0.4,
    y_max: Optional[float] = 1.15,
    dpi: int = 300,
) -> None:
    _safe_mkdir(out_png.parent)
    preferred = [
        "sham saline injection",
        "Lateral lesion only",
        "Complete and partial Medial and Lateral lesion",
        "Partial Medial and Lateral lesion",
        "Complete Medial and Lateral lesion",
        "Unknown",
    ]
    lesion_types = [x for x in preferred if x in set(df["lesion_hit_type"])]
    lesion_types += [x for x in df["lesion_hit_type"].dropna().unique().tolist() if x not in lesion_types]

    n = len(lesion_types)
    if n == 0:
        raise ValueError("No lesion types to plot.")

    fig, axes = plt.subplots(1, n, figsize=(max(6.5, 5.2 * n), 6.0), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, lesion_type in zip(axes, lesion_types):
        sub = df[df["lesion_hit_type"] == lesion_type].copy()
        pre = pd.to_numeric(sub["bc_pre"], errors="coerce").to_numpy(dtype=float)
        post = pd.to_numeric(sub["bc_post"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(pre) & np.isfinite(post)
        pre = pre[mask]
        post = post[mask]

        if len(pre) == 0:
            ax.set_title(f"{lesion_type}\nno finite rows")
            ax.axis("off")
            continue

        ax.boxplot(
            [pre, post],
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=0),
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.8),
            whiskerprops=dict(color="black", linewidth=1.5),
            capprops=dict(color="black", linewidth=1.5),
        )

        for a, b in zip(pre, post):
            ax.plot([1, 2], [a, b], color="0.78", linewidth=0.7, zorder=1)
        ax.scatter(np.full(len(pre), 1.0), pre, s=14, color="red", alpha=0.45, zorder=3)
        ax.scatter(np.full(len(post), 2.0), post, s=14, color="#5e2ca5", alpha=0.45, zorder=3)

        ax.hlines(float(np.nanmedian(pre)), 1 - 0.275, 1 + 0.275, colors="red", linewidth=2.0, zorder=4)
        ax.hlines(float(np.nanmedian(post)), 2 - 0.275, 2 + 0.275, colors="#5e2ca5", linewidth=2.0, zorder=4)

        p, stars = _paired_wilcoxon(pre, post)
        local_top = max(float(np.nanmax(pre)), float(np.nanmax(post)))
        local_bottom = min(float(np.nanmin(pre)), float(np.nanmin(post)))
        yr = max(0.02, local_top - local_bottom)
        yb = min(float(y_max) - 0.045, local_top + 0.13 * yr) if y_max is not None else local_top + 0.13 * yr
        ax.plot([1, 1, 2, 2], [yb - 0.02 * yr, yb, yb, yb - 0.02 * yr], color="black", linewidth=1.4)
        ax.text(1.5, yb + 0.012 * yr, stars, ha="center", va="bottom", fontsize=15)

        ax.set_title(f"{lesion_type}\nclusters={len(pre)}, animals={sub['animal_id'].nunique()}", fontsize=15, pad=8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre", "Post"], fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=12)
        if y_min is not None or y_max is not None:
            ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("Bhattacharyya coefficient\n(early vs. late, equal time-bin sample size)", fontsize=14)
    fig.suptitle(title, fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate per-bird BC summaries and plot pre/post BC by lesion type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bc-root", required=True, help="Root output folder containing <animal>/<animal>_cluster_bc_summary.csv files.")
    p.add_argument("--metadata-excel-path", required=True, help="Metadata Excel file containing Animal ID and Lesion hit type.")
    p.add_argument("--out-dir", required=True, help="Output folder for aggregate CSVs and plots.")
    p.add_argument("--min-balanced-duration-s", type=float, default=2.0)
    p.add_argument("--separate-complete-partial", action="store_true", help="Do not combine complete and partial medial/lateral lesions.")
    p.add_argument("--y-min", type=float, default=0.4)
    p.add_argument("--y-max", type=float, default=1.15)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    bc_root = Path(args.bc_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _safe_mkdir(out_dir)

    merged = load_all_cluster_summaries(bc_root).merge(
        load_metadata_hit_types(
            Path(args.metadata_excel_path).expanduser().resolve(),
            combine_medial_lateral=not bool(args.separate_complete_partial),
        ),
        on="animal_id",
        how="left",
    )
    merged["lesion_hit_type"] = merged["lesion_hit_type"].fillna("Unknown")

    if "balanced_duration_s_per_group" in merged.columns:
        merged["passes_requested_min_duration"] = (
            pd.to_numeric(merged["balanced_duration_s_per_group"], errors="coerce")
            >= float(args.min_balanced_duration_s)
        )
    else:
        merged["passes_requested_min_duration"] = merged["passes_min_balanced_duration"].astype(bool)

    finite = np.isfinite(pd.to_numeric(merged["bc_pre"], errors="coerce")) & np.isfinite(pd.to_numeric(merged["bc_post"], errors="coerce"))
    filtered = merged[
        merged["passes_requested_min_duration"].astype(bool)
        & merged["passes_min_balanced_duration"].astype(bool)
        & finite
    ].copy()

    all_csv = out_dir / "cluster_bc_summary_all_birds_with_lesion_type.csv"
    filtered.to_csv(all_csv, index=False)
    print(f"[SAVE] {all_csv}")

    if filtered.empty:
        raise ValueError("No rows survived filtering; check --bc-root and --min-balanced-duration-s.")

    all_stats_csv = out_dir / "lesion_type_bc_stats_all_clusters.csv"
    compute_stats(filtered).to_csv(all_stats_csv, index=False)
    print(f"[SAVE] {all_stats_csv}")

    all_plot = out_dir / "BC_by_lesion_type_all_clusters.png"
    plot_bc_by_lesion_type(
        filtered,
        out_png=all_plot,
        title=(
            "Bhattacharyya coefficient: pre vs post within lesion hit type\n"
            f"all clusters; min balanced duration/group = {float(args.min_balanced_duration_s):.1f} s"
        ),
        y_min=args.y_min,
        y_max=args.y_max,
        dpi=int(args.dpi),
    )
    print(f"[SAVE] {all_plot}")

    hv = filtered[filtered["is_high_variance_cluster"].astype(bool)].copy()
    if hv.empty:
        print("[WARN] No high-variance clusters found. Did you pass --phrase-csv to the per-bird QC script?")
    else:
        hv_stats_csv = out_dir / "lesion_type_bc_stats_high_variance_clusters.csv"
        compute_stats(hv).to_csv(hv_stats_csv, index=False)
        print(f"[SAVE] {hv_stats_csv}")

        hv_plot = out_dir / "BC_by_lesion_type_high_variance_clusters.png"
        plot_bc_by_lesion_type(
            hv,
            out_png=hv_plot,
            title=(
                "Bhattacharyya coefficient: pre vs post within lesion hit type\n"
                f"high-variance phrase-duration clusters only; min balanced duration/group = {float(args.min_balanced_duration_s):.1f} s"
            ),
            y_min=args.y_min,
            y_max=args.y_max,
            dpi=int(args.dpi),
        )
        print(f"[SAVE] {hv_plot}")


if __name__ == "__main__":
    main()
