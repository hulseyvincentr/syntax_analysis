#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entropy_pre_post_within_group.py

Within-group paired comparisons of total transition entropy (TE) pre vs post lesion.

For each experimental group (hit type):
  1) Combined (visible ML + not visible)
  2) Area X visible (single hit)
  3) Sham saline injection

This script:
  - Loads per-bird TE_pre and TE_post from entropy_batch_summary.csv
  - Looks up group membership from Area_X_lesion_metadata_with_hit_types.xlsx
  - For each group:
      * Plots Pre vs Post with paired lines (one per bird) + boxplots
      * Runs Wilcoxon signed-rank test (paired) across birds
      * Annotates significance bracket with asterisks + prints p-value
  - Saves:
      * 3 figures (one per group)
      * a stats .txt summary
      * a per-bird CSV used for plotting

Expected inputs
---------------
1) Batch summary CSV (created by your entropy wrapper):
   root_dir.parent / "entropy_figures" / "entropy_batch_summary.csv"
   Must contain columns: ['animal_id','TE_pre','TE_post'].

2) Metadata Excel (one row per animal):
   sheet: "animal_hit_type_summary"
   columns: ['Animal ID','Lesion hit type'].

Outputs
-------
Default out_dir:
  root_dir.parent / "entropy_figures" / "aggregate_within_group_pre_post"

Files:
  - TE_pre_post__<group_slug>.png  (3 plots)
  - TE_pre_post__per_bird.csv
  - TE_pre_post__within_group_stats.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for Wilcoxon signed-rank
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ──────────────────────────────────────────────────────────────────────────────
# Group mapping
# ──────────────────────────────────────────────────────────────────────────────
GROUP_COMBINED = "Combined (visible ML + not visible)"
GROUP_SINGLE   = "Area X visible (single hit)"
GROUP_SHAM     = "Sham saline injection"

GROUP_ORDER = (GROUP_COMBINED, GROUP_SINGLE, GROUP_SHAM)


def map_hit_type_to_group(hit_type: str) -> Optional[str]:
    if hit_type is None or (isinstance(hit_type, float) and np.isnan(hit_type)):
        return None
    s = str(hit_type).strip().lower()

    if "sham" in s or "saline" in s:
        return GROUP_SHAM
    if "single hit" in s:
        return GROUP_SINGLE
    if "medial+lateral" in s or ("medial" in s and "lateral" in s):
        return GROUP_COMBINED
    if ("not visible" in s) and ("large" in s):
        return GROUP_COMBINED
    return None


def slugify(s: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────
def p_to_stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Minimal rankdata implementation with average ranks for ties.
    Ranks are 1..N.
    """
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)

    # average ranks for ties
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = np.mean(np.arange(i + 1, j + 1, dtype=float))
            ranks[order[i:j]] = avg
        i = j
    return ranks


def wilcoxon_signed_rank_paired(pre: np.ndarray, post: np.ndarray) -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test.
    Returns (W, p_two_sided)

    If SciPy is available: scipy.stats.wilcoxon(pre, post)
    Else: permutation test on Wilcoxon W statistic using random sign flips.
    """
    pre = np.asarray(pre, float)
    post = np.asarray(post, float)
    ok = np.isfinite(pre) & np.isfinite(post)
    pre = pre[ok]
    post = post[ok]
    if len(pre) < 2:
        return np.nan, np.nan

    d = post - pre

    # remove zeros (standard Wilcoxon behavior)
    nz = d != 0
    d = d[nz]
    if len(d) < 2:
        return np.nan, np.nan

    if _HAVE_SCIPY:
        res = _scipy_stats.wilcoxon(pre[ok][nz], post[ok][nz], alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)

    # Fallback: permutation on signed ranks (two-sided)
    absd = np.abs(d)
    ranks = _rankdata_average_ties(absd)
    W_obs = float(np.sum(ranks[d > 0]))

    rng = np.random.default_rng(0)
    n_perm = 20000
    W_perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=len(d))
        d_flip = d * flips
        W_perm[i] = np.sum(ranks[d_flip > 0])

    # two-sided p: compare to distribution extremes around expected value
    # Expected W under null ~ sum(ranks)/2
    W0 = 0.5 * np.sum(ranks)
    dist = np.abs(W_perm - W0)
    obs = abs(W_obs - W0)
    p = (np.sum(dist >= obs) + 1) / (n_perm + 1)
    return W_obs, float(p)


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────
def load_entropy_batch_summary(summary_csv: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    required = {"animal_id", "TE_pre", "TE_post"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Batch summary missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    df["animal_id"] = df["animal_id"].astype(str).str.strip()
    df["TE_pre"] = pd.to_numeric(df["TE_pre"], errors="coerce")
    df["TE_post"] = pd.to_numeric(df["TE_post"], errors="coerce")
    return df


def load_hit_type_summary(metadata_xlsx: Union[str, Path], sheet_name: str = "animal_hit_type_summary") -> pd.DataFrame:
    meta = pd.read_excel(metadata_xlsx, sheet_name=sheet_name)
    needed = {"Animal ID", "Lesion hit type"}
    missing = needed - set(meta.columns)
    if missing:
        raise KeyError(f"Metadata sheet '{sheet_name}' missing columns: {sorted(missing)}. Found: {list(meta.columns)}")

    out = meta[["Animal ID", "Lesion hit type"]].copy()
    out = out.dropna(subset=["Animal ID"]).reset_index(drop=True)
    out["Animal ID"] = out["Animal ID"].astype(str).str.strip()
    out["Lesion hit type"] = out["Lesion hit type"].astype(str).str.strip()
    out["group"] = out["Lesion hit type"].apply(map_hit_type_to_group)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def _add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1.2, color="black")
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom")


def plot_pre_post_for_group(
    df_group: pd.DataFrame,
    *,
    group_name: str,
    out_path: Union[str, Path],
    title_prefix: str = "Total transition entropy",
) -> Dict[str, float]:
    """
    Plot paired pre/post TE for one group, with one marker per bird and lines connecting.
    Returns dict with W and p.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep finite pairs
    dfg = df_group.copy()
    dfg = dfg[np.isfinite(dfg["TE_pre"].astype(float)) & np.isfinite(dfg["TE_post"].astype(float))].copy()
    dfg = dfg.sort_values("animal_id").reset_index(drop=True)

    pre = dfg["TE_pre"].to_numpy(float)
    post = dfg["TE_post"].to_numpy(float)
    animal_ids = dfg["animal_id"].astype(str).tolist()

    W, p = wilcoxon_signed_rank_paired(pre, post)

    # Marker list (cycled if needed)
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "+", "x", "d", "h", "H", "p", "1", "2", "3", "4"]
    n = len(animal_ids)

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = plt.gca()

    # Boxplots
    ax.boxplot(
        [pre, post],
        positions=[1, 2],
        widths=0.5,
        showfliers=False,
        patch_artist=False,
    )

    # Paired lines + points
    for i, (aid, a_pre, a_post) in enumerate(zip(animal_ids, pre, post)):
        mk = markers[i % len(markers)]
        # connecting line
        ax.plot([1, 2], [a_pre, a_post], linewidth=1.2, alpha=0.8, color="0.5")
        # points (blue pre, red post)
        ax.scatter([1], [a_pre], marker=mk, s=60, edgecolor="blue", facecolor="none", linewidth=1.6)
        ax.scatter([2], [a_post], marker=mk, s=60, edgecolor="red",  facecolor="none", linewidth=1.6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pre lesion", "Post lesion"])
    ax.set_ylabel("Total Transition Entropy (bits)")
    ax.set_title(f"{title_prefix}\n{group_name}  (n = {n} birds)")

    # Significance bracket
    y_max = float(np.nanmax(np.r_[pre, post])) if n > 0 else 0.0
    y_min = float(np.nanmin(np.r_[pre, post])) if n > 0 else 0.0
    span = max(1e-6, y_max - y_min)
    base = y_max + 0.08 * span
    h = 0.03 * span
    _add_sig_bracket(ax, 1, 2, base, h, p_to_stars(p))

    # p-value text
    p_txt = "p = n/a" if not np.isfinite(p) else f"Wilcoxon signed-rank p = {p:.3g}"
    ax.text(0.02, 0.02, p_txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)

    # Bird legend (marker per bird)
    # Build fake handles with the same markers
    handles = []
    labels = []
    for i, aid in enumerate(animal_ids):
        mk = markers[i % len(markers)]
        hndl = ax.scatter([], [], marker=mk, s=60, edgecolor="black", facecolor="none", linewidth=1.4)
        handles.append(hndl)
        labels.append(aid)

    if handles:
        ax.legend(
            handles,
            labels,
            title="Bird",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=9,
            title_fontsize=10,
            borderaxespad=0.0,
        )

    plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave room for legend
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"W": float(W), "p": float(p), "n": float(n)}


# ──────────────────────────────────────────────────────────────────────────────
# Main driver: make 3 plots (one per group)
# ──────────────────────────────────────────────────────────────────────────────
def within_group_pre_post_TE_plots(
    *,
    root_dir: Union[str, Path],
    metadata_xlsx: Union[str, Path],
    batch_summary_csv: Optional[Union[str, Path]] = None,
    metadata_sheet_name: str = "animal_hit_type_summary",
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    root_dir = Path(root_dir)

    if batch_summary_csv is None:
        batch_summary_csv = root_dir.parent / "entropy_figures" / "entropy_batch_summary.csv"
    batch_summary_csv = Path(batch_summary_csv)

    if out_dir is None:
        out_dir = root_dir.parent / "entropy_figures" / "aggregate_within_group_pre_post"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = load_entropy_batch_summary(batch_summary_csv)
    meta = load_hit_type_summary(metadata_xlsx, sheet_name=metadata_sheet_name)

    merged = batch.merge(
        meta.rename(columns={"Animal ID": "animal_id"}),
        on="animal_id",
        how="left",
        validate="one_to_one",
    )

    merged = merged[merged["group"].notna()].copy()
    merged = merged.reset_index(drop=True)

    # Save the per-bird table used
    per_bird_csv = out_dir / "TE_pre_post__per_bird.csv"
    merged.to_csv(per_bird_csv, index=False)

    # Make one plot per group
    stats_lines = []
    stats_lines.append(f"Batch summary: {batch_summary_csv}")
    stats_lines.append(f"Metadata: {Path(metadata_xlsx)} (sheet={metadata_sheet_name})")
    stats_lines.append("")
    stats_lines.append("Within-group Wilcoxon signed-rank tests (paired pre vs post):")

    plot_paths = {}
    results = {}

    for g in GROUP_ORDER:
        dfg = merged[merged["group"] == g].copy()
        out_path = out_dir / f"TE_pre_post__{slugify(g)}.png"

        res = plot_pre_post_for_group(
            dfg,
            group_name=g,
            out_path=out_path,
            title_prefix="Total transition entropy",
        )

        plot_paths[g] = str(out_path)
        results[g] = res

        p = res["p"]
        n = int(res["n"])
        if np.isfinite(p):
            stats_lines.append(f"  {g}: n={n}, W={res['W']:.6g}, p={p:.6g}")
        else:
            stats_lines.append(f"  {g}: n={n}, W=n/a, p=n/a (not enough nonzero paired differences)")

    stats_txt = out_dir / "TE_pre_post__within_group_stats.txt"
    stats_txt.write_text("\n".join(stats_lines) + "\n")

    return {
        "paths": {
            "out_dir": str(out_dir),
            "per_bird_csv": str(per_bird_csv),
            "stats_txt": str(stats_txt),
            "plots": plot_paths,
        },
        "results": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# COMMENTED-OUT SPYDER CONSOLE EXAMPLE
# ──────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
import sys, importlib

code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import entropy_pre_post_within_group as epw
importlib.reload(epw)

root_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
metadata_xlsx = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx")

res = epw.within_group_pre_post_TE_plots(
    root_dir=root_dir,
    metadata_xlsx=metadata_xlsx,
    metadata_sheet_name="animal_hit_type_summary",
    # batch_summary_csv defaults to: root_dir.parent / "entropy_figures" / "entropy_batch_summary.csv"
    # out_dir defaults to: root_dir.parent / "entropy_figures" / "aggregate_within_group_pre_post"
)

print(res["paths"]["plots"])
"""
