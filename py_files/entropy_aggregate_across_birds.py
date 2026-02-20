#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entropy_aggregate_across_birds.py

Aggregate plots across birds using outputs from entropy_batch_wrapper.py.

This script expects a batch summary CSV produced by entropy_batch_wrapper.py, and an
Excel metadata sheet mapping Animal ID -> Lesion hit type.

What this script can plot
-------------------------
A) Bird-level Total Transition Entropy (TTE)
   - Pre vs Post paired within bird, shown separately for 3 lesion groups:
        • Combined (visible ML + not visible)
        • Area X visible (single hit)
        • Sham saline injection
   - Optional: ΔTTE = TTE_post - TTE_pre across groups

B) Syllable-level transition entropy H_a (optional; requires per-bird Ha table CSV path)
   - Overall ΔH_a distribution by group (points are syllable×bird; stats use bird-level
     means to reduce pseudo-replication).
   - Optional: per-syllable ΔH_a across birds (one figure per syllable)

NEW (requires per-bird output paths saved by updated entropy_batch_wrapper.py)
-----------------------------------------------------------------------------
C) Mean remaining syllables to end-of-song (per syllable; pre vs post)
   - Overall Δ(mean remaining) distribution by group (syllable×bird points; bird-mean stats).
   - Optional: per-syllable Δ(mean remaining) across birds

D) Variance-tier analysis: do high phrase-duration-variance syllables change more in H_a?
   - For each bird, selects top-variance syllables (default: top 30% by var_summary from
     aggregate_variance_points_by_bird_and_syllable.csv) and the remaining syllables.
   - Computes bird-level mean ΔH_a for each tier.
   - Produces separate boxplots for:
        • High-variance syllables (bird means)
        • Low-variance syllables (bird means)
   - Optional: paired within-bird plot (high vs low) for each lesion group.

Outputs
-------
Default out_dir:
  <root_dir.parent>/entropy_figures/aggregate

Key files (when enabled):
  - TTE_pre_post_by_group__paired.png
  - delta_TTE_by_group__boxplot.png
  - delta_Ha_overall_by_group__boxplot.png
  - mean_remaining_delta_overall_by_group__boxplot.png
  - variance_tier__delta_Ha_high_by_group__boxplot.png
  - variance_tier__delta_Ha_low_by_group__boxplot.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy
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
    """Map metadata 'Lesion hit type' → the 3-group scheme."""
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
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_entropy_batch_summary(summary_csv: Union[str, Path]) -> pd.DataFrame:
    """
    Loads the batch summary CSV produced by entropy_batch_wrapper.py.

    Required columns:
      - animal_id
      - TE_pre
      - TE_post

    Optional columns (newer wrappers may include these):
      - Ha_table_csv (or ha_csv_path)
      - mean_remaining_to_end_csv
      - variance_summary_csv
    """
    df = pd.read_csv(summary_csv)

    required = {"animal_id", "TE_pre", "TE_post"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Batch summary missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df["animal_id"] = df["animal_id"].astype(str).str.strip()
    df["TE_pre"] = pd.to_numeric(df["TE_pre"], errors="coerce")
    df["TE_post"] = pd.to_numeric(df["TE_post"], errors="coerce")

    # Normalize path-like columns to strings if present
    for c in ["ha_csv_path", "Ha_table_csv", "mean_remaining_to_end_csv", "variance_summary_csv"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


def load_hit_type_summary(
    metadata_xlsx: Union[str, Path],
    sheet_name: str = "animal_hit_type_summary",
) -> pd.DataFrame:
    meta = pd.read_excel(metadata_xlsx, sheet_name=sheet_name)
    needed = {"Animal ID", "Lesion hit type"}
    missing = needed - set(meta.columns)
    if missing:
        raise KeyError(
            f"Metadata sheet '{sheet_name}' missing columns: {sorted(missing)}. Found: {list(meta.columns)}"
        )

    out = meta[["Animal ID", "Lesion hit type"]].copy()
    out = out.dropna(subset=["Animal ID"]).reset_index(drop=True)
    out["Animal ID"] = out["Animal ID"].astype(str).str.strip()
    out["Lesion hit type"] = out["Lesion hit type"].astype(str).str.strip()
    out["group"] = out["Lesion hit type"].apply(map_hit_type_to_group)
    return out


def merge_batch_with_groups(
    *,
    root_dir: Union[str, Path],
    metadata_xlsx: Union[str, Path],
    batch_summary_csv: Optional[Union[str, Path]] = None,
    metadata_sheet_name: str = "animal_hit_type_summary",
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Returns merged df with columns (at least):
      animal_id, TE_pre, TE_post, group (+ any other batch columns)
    """
    root_dir = Path(root_dir)
    if batch_summary_csv is None:
        batch_summary_csv = root_dir.parent / "entropy_figures" / "entropy_batch_summary.csv"
    batch_summary_csv = Path(batch_summary_csv)

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
    return merged, batch_summary_csv, Path(metadata_xlsx)


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────
def _format_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _welch_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Welch's t-test (two-sided). Returns (t, p)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    if _HAVE_SCIPY:
        res = _scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)

    # Fallback: permutation on mean diff (two-sided)
    rng = np.random.default_rng(0)
    n_perm = 20000
    obs = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        rng.shuffle(pooled)
        perm[i] = float(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:]))
    p = float((np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1))
    return np.nan, p


def wilcoxon_signed_rank_paired(pre: np.ndarray, post: np.ndarray, *, seed: int = 0) -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test: compares (post - pre) within birds.
    Returns (W, p_two_sided).

    If SciPy unavailable, uses sign-flip permutation on mean difference.
    """
    pre = np.asarray(pre, float)
    post = np.asarray(post, float)
    ok = np.isfinite(pre) & np.isfinite(post)
    pre = pre[ok]
    post = post[ok]
    if len(pre) < 2:
        return np.nan, np.nan

    diffs = post - pre
    nz = diffs != 0
    pre = pre[nz]
    post = post[nz]
    diffs = diffs[nz]
    if len(pre) < 2:
        return np.nan, np.nan

    if _HAVE_SCIPY:
        res = _scipy_stats.wilcoxon(pre, post, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)

    rng = np.random.default_rng(seed)
    n_perm = 20000
    obs = float(np.mean(diffs))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size), replace=True)
    perm_means = (signs * diffs[None, :]).mean(axis=1)
    p = float((np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1))
    return np.nan, p


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def _remove_top_right_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    *,
    text_offset: float = 0.0,
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1.2, color="black")
    ax.text((x1 + x2) / 2.0, y + h + text_offset, text, ha="center", va="bottom")


def _jitter(n: int, *, seed: int = 0, scale: float = 0.06) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=n) if n > 0 else np.array([])


def _safe_span_from_groups(vals: List[np.ndarray], default: float = 1.0) -> float:
    nonempty = [v for v in vals if v is not None and len(v) > 0]
    if not nonempty:
        return default
    allv = np.concatenate(nonempty)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return default
    return max(1e-6, float(np.nanmax(allv) - np.nanmin(allv)))


def _safe_ymax_from_groups(vals: List[np.ndarray], default: float = 0.0) -> float:
    nonempty = [v for v in vals if v is not None and len(v) > 0]
    if not nonempty:
        return default
    allv = np.concatenate(nonempty)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return default
    return float(np.nanmax(allv))


# ──────────────────────────────────────────────────────────────────────────────
# TTE combined pre/post plot by group
# ──────────────────────────────────────────────────────────────────────────────
def plot_TTE_pre_post_by_group(
    merged: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Single combined figure:
      - Three groups on x-axis
      - For each group: Pre boxplot + Post boxplot, paired lines within bird
      - Paired test (Wilcoxon / sign-flip) per group
      - p and n are placed on separate lines (avoids overlap).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = merged.copy()
    df = df[np.isfinite(df["TE_pre"].astype(float)) & np.isfinite(df["TE_post"].astype(float))].copy()

    group_color = {
        GROUP_COMBINED: "C0",
        GROUP_SINGLE: "C1",
        GROUP_SHAM: "C2",
    }

    centers = np.array([1.0, 4.0, 7.0], dtype=float)
    dx = 0.35
    x_pre = centers - dx
    x_post = centers + dx

    all_vals = np.concatenate([df["TE_pre"].to_numpy(float), df["TE_post"].to_numpy(float)])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise ValueError("No finite TE_pre/TE_post values to plot.")

    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    span = max(1e-6, y_max - y_min)

    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    pre_handle = ax.scatter([], [], s=80, facecolor="none", edgecolor="black", linewidth=1.6, label="Pre (open)")
    post_handle = ax.scatter([], [], s=80, facecolor="black", edgecolor="black", linewidth=1.0, label="Post (filled)")

    stats_by_group: Dict[str, Dict[str, float]] = {}

    for gi, g in enumerate(GROUP_ORDER):
        dfg = df[df["group"] == g].copy().sort_values("animal_id")
        pre = dfg["TE_pre"].to_numpy(float)
        post = dfg["TE_post"].to_numpy(float)
        n = int(len(dfg))

        ax.boxplot(
            [pre, post],
            positions=[x_pre[gi], x_post[gi]],
            widths=0.42,
            showfliers=False,
            patch_artist=False,
        )

        jit = _jitter(n, seed=seed + gi, scale=0.02)
        c = group_color.get(g, "0.3")

        for i in range(n):
            ax.plot([x_pre[gi] + jit[i], x_post[gi] + jit[i]], [pre[i], post[i]],
                    color=c, alpha=0.45, linewidth=1.3)

        ax.scatter(
            np.full(n, x_pre[gi]) + jit,
            pre,
            s=70,
            facecolor="none",
            edgecolor=c,
            linewidth=1.9,
            zorder=3,
        )
        ax.scatter(
            np.full(n, x_post[gi]) + jit,
            post,
            s=70,
            facecolor=c,
            edgecolor=c,
            linewidth=1.0,
            zorder=3,
        )

        W, p = wilcoxon_signed_rank_paired(pre, post, seed=seed)
        stats_by_group[g] = {"n": float(n), "W": float(W) if np.isfinite(W) else np.nan, "p": float(p)}

        local_max = float(np.nanmax(np.r_[pre, post])) if n > 0 else y_max
        base = local_max + 0.07 * span
        h = 0.03 * span
        label = f"p={_format_p(p)}\\n(n={n})"
        _add_sig_bracket(
            ax,
            x_pre[gi],
            x_post[gi],
            base,
            h,
            label,
            text_offset=0.015 * span,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(list(GROUP_ORDER), rotation=15, ha="right")
    ax.set_ylabel("TTE (bits)")

    if title is None:
        title = "Bird-level Total Transition Entropy (TTE)\\n(Pre vs Post; paired within bird; weighted by syllable usage P(a))"
    ax.set_title(title)

    ax.set_ylim(y_min - 0.05 * span, y_max + 0.45 * span)

    _remove_top_right_spines(ax)
    ax.legend(handles=[pre_handle, post_handle], frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"stats_by_group": stats_by_group, "out_path": str(out_path)}


# ──────────────────────────────────────────────────────────────────────────────
# ΔTTE across groups
# ──────────────────────────────────────────────────────────────────────────────
def plot_delta_TTE_by_group(
    merged: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    seed: int = 0,
) -> Dict[str, object]:
    """
    Boxplot of ΔTTE = TE_post - TE_pre across groups, with between-group tests:
      - Combined vs Sham
      - Single vs Sham

    p-values are annotated on the plot.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = merged.copy()
    df = df[np.isfinite(df["TE_pre"].astype(float)) & np.isfinite(df["TE_post"].astype(float))].copy()
    df["delta_TTE"] = df["TE_post"] - df["TE_pre"]

    vals: List[np.ndarray] = []
    ns: List[int] = []
    for g in GROUP_ORDER:
        v = df.loc[df["group"] == g, "delta_TTE"].to_numpy(float)
        v = v[np.isfinite(v)]
        vals.append(v)
        ns.append(int(len(v)))

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.boxplot(vals, labels=[f"{g}\\n(n={n})" for g, n in zip(GROUP_ORDER, ns)], showfliers=True, whis=1.5)

    rng = np.random.default_rng(seed)
    for i, v in enumerate(vals, start=1):
        j = rng.normal(0.0, 0.06, size=len(v))
        ax.scatter(np.full(len(v), i) + j, v, s=30, alpha=0.85)

    ax.set_ylabel("ΔTTE = TTE_post − TTE_pre (bits)")
    ax.set_title("Change in bird-level TTE by lesion hit type")
    _remove_top_right_spines(ax)

    # Between-group tests vs Sham (indices: Combined=1, Single=2, Sham=3)
    comb = vals[0]
    single = vals[1]
    sham = vals[2]

    t1, p1 = _welch_ttest(comb, sham)
    t2, p2 = _welch_ttest(single, sham)

    y0 = _safe_ymax_from_groups(vals, default=0.0)
    span = _safe_span_from_groups(vals, default=1.0)
    base = y0 + 0.10 * span
    h = 0.03 * span

    _add_sig_bracket(ax, 1, 3, base, h, f"Welch p={_format_p(p1)}", text_offset=0.01 * span)
    _add_sig_bracket(ax, 2, 3, base + 0.10 * span, h, f"Welch p={_format_p(p2)}", text_offset=0.01 * span)

    ax.set_ylim(ax.get_ylim()[0], y0 + 0.30 * span)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "tests": {
            "combined_vs_sham": {"t": float(t1) if np.isfinite(t1) else np.nan, "p": float(p1)},
            "single_vs_sham": {"t": float(t2) if np.isfinite(t2) else np.nan, "p": float(p2)},
        },
        "out_path": str(out_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ΔH_a (optional) – requires per-bird Ha table path in batch summary
# ──────────────────────────────────────────────────────────────────────────────
def _get_ha_csv_path_col(merged: pd.DataFrame) -> Optional[str]:
    # Prefer updated wrapper column name, but allow legacy name too.
    return _first_existing_col(merged, ["Ha_table_csv", "ha_csv_path", "ha_table_csv", "ha_table_path"])


def build_delta_Ha_long_df(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a long table with one row per (bird, syllable):
      animal_id, group, syllable, H_pre, H_post, delta_Ha, n_pre, n_post
    """
    ha_col = _get_ha_csv_path_col(merged)
    if ha_col is None:
        raise KeyError(
            "Merged df has no per-bird Ha table path column. Expected one of: "
            "Ha_table_csv, ha_csv_path."
        )

    rows: List[Dict[str, object]] = []
    for _, r in merged.iterrows():
        aid = str(r["animal_id"])
        grp = str(r["group"])
        ha_path = Path(str(r[ha_col]))

        if not ha_path.exists():
            continue

        hdf = pd.read_csv(ha_path)

        needed = {"syllable", "H_pre", "H_post"}
        if not needed.issubset(set(hdf.columns)):
            continue

        hdf["syllable"] = hdf["syllable"].astype(str)
        hdf["H_pre"] = pd.to_numeric(hdf["H_pre"], errors="coerce")
        hdf["H_post"] = pd.to_numeric(hdf["H_post"], errors="coerce")

        if "n_pre" in hdf.columns:
            hdf["n_pre"] = pd.to_numeric(hdf["n_pre"], errors="coerce")
        else:
            hdf["n_pre"] = np.nan

        if "n_post" in hdf.columns:
            hdf["n_post"] = pd.to_numeric(hdf["n_post"], errors="coerce")
        else:
            hdf["n_post"] = np.nan

        ok = np.isfinite(hdf["H_pre"].to_numpy(float)) & np.isfinite(hdf["H_post"].to_numpy(float))
        hdf = hdf.loc[ok].copy()
        if hdf.empty:
            continue

        hdf["delta_Ha"] = hdf["H_post"] - hdf["H_pre"]

        for _, rr in hdf.iterrows():
            rows.append(
                {
                    "animal_id": aid,
                    "group": grp,
                    "syllable": str(rr["syllable"]),
                    "H_pre": float(rr["H_pre"]),
                    "H_post": float(rr["H_post"]),
                    "delta_Ha": float(rr["delta_Ha"]),
                    "n_pre": float(rr["n_pre"]) if np.isfinite(rr["n_pre"]) else np.nan,
                    "n_post": float(rr["n_post"]) if np.isfinite(rr["n_post"]) else np.nan,
                    "ha_csv_path": str(ha_path),
                }
            )

    return pd.DataFrame(rows)


def plot_delta_Ha_overall_by_group(
    delta_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    seed: int = 0,
) -> Dict[str, object]:
    """
    Overall ΔH_a distribution by group.
    Points: syllable×bird.
    Stats: bird-level mean ΔH_a (avoids pseudo-replication).
    Tests annotated: Combined vs Sham, Single vs Sham.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if delta_df is None or delta_df.empty:
        raise ValueError("delta_df is empty; nothing to plot.")

    bird_means = (
        delta_df.groupby(["animal_id", "group"], as_index=False)["delta_Ha"]
        .mean()
        .rename(columns={"delta_Ha": "mean_delta_Ha"})
    )

    vals: List[np.ndarray] = []
    n_pairs: List[int] = []
    for g in GROUP_ORDER:
        v = delta_df.loc[delta_df["group"] == g, "delta_Ha"].to_numpy(float)
        v = v[np.isfinite(v)]
        vals.append(v)
        n_pairs.append(int(len(v)))

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.boxplot(vals, labels=[f"{g}\\n(n_pairs={n})" for g, n in zip(GROUP_ORDER, n_pairs)], showfliers=True, whis=1.5)

    rng = np.random.default_rng(seed)
    for i, v in enumerate(vals, start=1):
        j = rng.normal(0.0, 0.06, size=len(v))
        ax.scatter(np.full(len(v), i) + j, v, s=18, alpha=0.55)

    ax.set_ylabel("ΔHₐ = Hₐ_post − Hₐ_pre (bits)")
    ax.set_title("Per-syllable entropy change (ΔHₐ) aggregated across birds")
    _remove_top_right_spines(ax)

    comb_b = bird_means.loc[bird_means["group"] == GROUP_COMBINED, "mean_delta_Ha"].to_numpy(float)
    single_b = bird_means.loc[bird_means["group"] == GROUP_SINGLE, "mean_delta_Ha"].to_numpy(float)
    sham_b = bird_means.loc[bird_means["group"] == GROUP_SHAM, "mean_delta_Ha"].to_numpy(float)

    t1, p1 = _welch_ttest(comb_b, sham_b)
    t2, p2 = _welch_ttest(single_b, sham_b)

    y0 = _safe_ymax_from_groups(vals, default=0.0)
    span = _safe_span_from_groups(vals, default=1.0)
    base = y0 + 0.10 * span
    h = 0.03 * span

    _add_sig_bracket(ax, 1, 3, base, h, f"Bird-mean Welch p={_format_p(p1)}", text_offset=0.01 * span)
    _add_sig_bracket(ax, 2, 3, base + 0.10 * span, h, f"Bird-mean Welch p={_format_p(p2)}", text_offset=0.01 * span)

    ax.set_ylim(ax.get_ylim()[0], y0 + 0.30 * span)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "tests_bird_mean": {
            "combined_vs_sham": {
                "t": float(t1) if np.isfinite(t1) else np.nan,
                "p": float(p1),
                "n_comb": int(len(comb_b)),
                "n_sham": int(len(sham_b)),
            },
            "single_vs_sham": {
                "t": float(t2) if np.isfinite(t2) else np.nan,
                "p": float(p2),
                "n_single": int(len(single_b)),
                "n_sham": int(len(sham_b)),
            },
        },
        "out_path": str(out_path),
    }


def plot_delta_Ha_by_syllable_across_birds(
    delta_df: pd.DataFrame,
    *,
    out_dir: Union[str, Path],
    min_birds_per_group: int = 3,
    seed: int = 0,
) -> Path:
    """
    Makes one figure per syllable: ΔHₐ across birds by group (boxplot + points),
    with tests vs Sham (Combined vs Sham; Single vs Sham).

    Writes a summary CSV with p-values to:
      out_dir.parent / "delta_Ha_by_syllable__stats.csv"

    Returns path to the summary CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if delta_df is None or delta_df.empty:
        raise ValueError("delta_df is empty; nothing to plot.")

    syllables = sorted(delta_df["syllable"].astype(str).unique().tolist())
    stats_rows: List[Dict[str, object]] = []

    rng = np.random.default_rng(seed)

    for syl in syllables:
        d = delta_df[delta_df["syllable"].astype(str) == str(syl)].copy()
        if d.empty:
            continue

        bird_vals = (
            d.groupby(["animal_id", "group"], as_index=False)["delta_Ha"]
            .mean()
            .rename(columns={"delta_Ha": "delta_Ha_bird"})
        )

        vals: List[np.ndarray] = []
        ns: List[int] = []
        for g in GROUP_ORDER:
            v = bird_vals.loc[bird_vals["group"] == g, "delta_Ha_bird"].to_numpy(float)
            v = v[np.isfinite(v)]
            vals.append(v)
            ns.append(int(len(v)))

        if ns[2] < min_birds_per_group:
            continue

        t1, p1 = _welch_ttest(vals[0], vals[2])  # combined vs sham
        t2, p2 = _welch_ttest(vals[1], vals[2])  # single vs sham

        fig, ax = plt.subplots(figsize=(10.0, 5.4))
        ax.boxplot(vals, labels=[f"{g}\\n(n={n})" for g, n in zip(GROUP_ORDER, ns)], showfliers=True, whis=1.5)

        for i, v in enumerate(vals, start=1):
            j = rng.normal(0.0, 0.06, size=len(v))
            ax.scatter(np.full(len(v), i) + j, v, s=26, alpha=0.85)

        ax.set_ylabel("ΔHₐ (bits)")
        ax.set_title(f"ΔHₐ by group for syllable {syl}")
        _remove_top_right_spines(ax)

        y0 = _safe_ymax_from_groups(vals, default=0.0)
        span = _safe_span_from_groups(vals, default=1.0)
        base = y0 + 0.10 * span
        h = 0.03 * span

        _add_sig_bracket(ax, 1, 3, base, h, f"Welch p={_format_p(p1)}", text_offset=0.01 * span)
        _add_sig_bracket(ax, 2, 3, base + 0.10 * span, h, f"Welch p={_format_p(p2)}", text_offset=0.01 * span)
        ax.set_ylim(ax.get_ylim()[0], y0 + 0.30 * span)

        fig.tight_layout()
        out_png = out_dir / f"delta_Ha__syllable_{str(syl)}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        stats_rows.append(
            {
                "syllable": str(syl),
                "n_combined": ns[0],
                "n_single": ns[1],
                "n_sham": ns[2],
                "combined_vs_sham_t": float(t1) if np.isfinite(t1) else np.nan,
                "combined_vs_sham_p": float(p1),
                "single_vs_sham_t": float(t2) if np.isfinite(t2) else np.nan,
                "single_vs_sham_p": float(p2),
                "plot_path": str(out_png),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    out_csv = out_dir.parent / "delta_Ha_by_syllable__stats.csv"
    stats_df.to_csv(out_csv, index=False)
    return out_csv


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Mean remaining syllables to end-of-song (per syllable)
# ──────────────────────────────────────────────────────────────────────────────
def _get_mean_remaining_csv_col(merged: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(merged, ["mean_remaining_to_end_csv", "mean_remaining_table_csv"])


def build_mean_remaining_long_df(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a long table with one row per (bird, syllable):
      animal_id, group, syllable, mean_remaining_pre, mean_remaining_post, delta_mean_remaining,
      n_songs_pre, n_songs_post
    """
    mr_col = _get_mean_remaining_csv_col(merged)
    if mr_col is None:
        raise KeyError(
            "Merged df has no per-bird mean-remaining table path column. Expected one of: "
            "mean_remaining_to_end_csv, mean_remaining_table_csv."
        )

    rows: List[Dict[str, object]] = []
    for _, r in merged.iterrows():
        aid = str(r["animal_id"])
        grp = str(r["group"])
        p = Path(str(r[mr_col]))
        if not p.exists():
            continue
        mdf = pd.read_csv(p)
        needed = {"syllable", "mean_remaining_pre", "mean_remaining_post"}
        if not needed.issubset(set(mdf.columns)):
            continue

        mdf["syllable"] = mdf["syllable"].astype(str)
        mdf["mean_remaining_pre"] = pd.to_numeric(mdf["mean_remaining_pre"], errors="coerce")
        mdf["mean_remaining_post"] = pd.to_numeric(mdf["mean_remaining_post"], errors="coerce")
        if "n_songs_pre" in mdf.columns:
            mdf["n_songs_pre"] = pd.to_numeric(mdf["n_songs_pre"], errors="coerce")
        else:
            mdf["n_songs_pre"] = np.nan
        if "n_songs_post" in mdf.columns:
            mdf["n_songs_post"] = pd.to_numeric(mdf["n_songs_post"], errors="coerce")
        else:
            mdf["n_songs_post"] = np.nan

        ok = np.isfinite(mdf["mean_remaining_pre"].to_numpy(float)) & np.isfinite(mdf["mean_remaining_post"].to_numpy(float))
        mdf = mdf.loc[ok].copy()
        if mdf.empty:
            continue

        mdf["delta_mean_remaining"] = mdf["mean_remaining_post"] - mdf["mean_remaining_pre"]

        for _, rr in mdf.iterrows():
            rows.append(
                {
                    "animal_id": aid,
                    "group": grp,
                    "syllable": str(rr["syllable"]),
                    "mean_remaining_pre": float(rr["mean_remaining_pre"]),
                    "mean_remaining_post": float(rr["mean_remaining_post"]),
                    "delta_mean_remaining": float(rr["delta_mean_remaining"]),
                    "n_songs_pre": float(rr["n_songs_pre"]) if np.isfinite(rr["n_songs_pre"]) else np.nan,
                    "n_songs_post": float(rr["n_songs_post"]) if np.isfinite(rr["n_songs_post"]) else np.nan,
                    "mean_remaining_csv_path": str(p),
                }
            )

    return pd.DataFrame(rows)


def plot_delta_mean_remaining_overall_by_group(
    delta_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    seed: int = 0,
) -> Dict[str, object]:
    """
    Overall Δ(mean remaining) distribution by group.
    Points: syllable×bird.
    Stats: bird-level mean Δ(mean remaining).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if delta_df is None or delta_df.empty:
        raise ValueError("delta_df is empty; nothing to plot.")

    bird_means = (
        delta_df.groupby(["animal_id", "group"], as_index=False)["delta_mean_remaining"]
        .mean()
        .rename(columns={"delta_mean_remaining": "mean_delta_mean_remaining"})
    )

    vals: List[np.ndarray] = []
    n_pairs: List[int] = []
    for g in GROUP_ORDER:
        v = delta_df.loc[delta_df["group"] == g, "delta_mean_remaining"].to_numpy(float)
        v = v[np.isfinite(v)]
        vals.append(v)
        n_pairs.append(int(len(v)))

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.boxplot(vals, labels=[f"{g}\\n(n_pairs={n})" for g, n in zip(GROUP_ORDER, n_pairs)], showfliers=True, whis=1.5)

    rng = np.random.default_rng(seed)
    for i, v in enumerate(vals, start=1):
        j = rng.normal(0.0, 0.06, size=len(v))
        ax.scatter(np.full(len(v), i) + j, v, s=18, alpha=0.55)

    ax.set_ylabel("Δ mean remaining = post − pre (syllables)")
    ax.set_title("Change in mean remaining syllables to song end (Δ remaining) aggregated across birds")
    _remove_top_right_spines(ax)

    comb_b = bird_means.loc[bird_means["group"] == GROUP_COMBINED, "mean_delta_mean_remaining"].to_numpy(float)
    single_b = bird_means.loc[bird_means["group"] == GROUP_SINGLE, "mean_delta_mean_remaining"].to_numpy(float)
    sham_b = bird_means.loc[bird_means["group"] == GROUP_SHAM, "mean_delta_mean_remaining"].to_numpy(float)

    t1, p1 = _welch_ttest(comb_b, sham_b)
    t2, p2 = _welch_ttest(single_b, sham_b)

    y0 = _safe_ymax_from_groups(vals, default=0.0)
    span = _safe_span_from_groups(vals, default=1.0)
    base = y0 + 0.10 * span
    h = 0.03 * span

    _add_sig_bracket(ax, 1, 3, base, h, f"Bird-mean Welch p={_format_p(p1)}", text_offset=0.01 * span)
    _add_sig_bracket(ax, 2, 3, base + 0.10 * span, h, f"Bird-mean Welch p={_format_p(p2)}", text_offset=0.01 * span)

    ax.set_ylim(ax.get_ylim()[0], y0 + 0.30 * span)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "tests_bird_mean": {
            "combined_vs_sham": {"t": float(t1) if np.isfinite(t1) else np.nan, "p": float(p1)},
            "single_vs_sham": {"t": float(t2) if np.isfinite(t2) else np.nan, "p": float(p2)},
        },
        "out_path": str(out_path),
    }


def plot_delta_mean_remaining_by_syllable_across_birds(
    delta_df: pd.DataFrame,
    *,
    out_dir: Union[str, Path],
    min_birds_per_group: int = 3,
    seed: int = 0,
) -> Path:
    """
    One figure per syllable: bird-level mean Δ(mean remaining) across groups.

    Writes summary CSV to:
      out_dir.parent / "delta_mean_remaining_by_syllable__stats.csv"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if delta_df is None or delta_df.empty:
        raise ValueError("delta_df is empty; nothing to plot.")

    syllables = sorted(delta_df["syllable"].astype(str).unique().tolist())
    stats_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for syl in syllables:
        d = delta_df[delta_df["syllable"].astype(str) == str(syl)].copy()
        if d.empty:
            continue

        bird_vals = (
            d.groupby(["animal_id", "group"], as_index=False)["delta_mean_remaining"]
            .mean()
            .rename(columns={"delta_mean_remaining": "delta_mean_remaining_bird"})
        )

        vals: List[np.ndarray] = []
        ns: List[int] = []
        for g in GROUP_ORDER:
            v = bird_vals.loc[bird_vals["group"] == g, "delta_mean_remaining_bird"].to_numpy(float)
            v = v[np.isfinite(v)]
            vals.append(v)
            ns.append(int(len(v)))

        if ns[2] < min_birds_per_group:
            continue

        t1, p1 = _welch_ttest(vals[0], vals[2])  # combined vs sham
        t2, p2 = _welch_ttest(vals[1], vals[2])  # single vs sham

        fig, ax = plt.subplots(figsize=(10.0, 5.4))
        ax.boxplot(vals, labels=[f"{g}\\n(n={n})" for g, n in zip(GROUP_ORDER, ns)], showfliers=True, whis=1.5)

        for i, v in enumerate(vals, start=1):
            j = rng.normal(0.0, 0.06, size=len(v))
            ax.scatter(np.full(len(v), i) + j, v, s=26, alpha=0.85)

        ax.set_ylabel("Δ mean remaining (syllables)")
        ax.set_title(f"Δ mean remaining by group for syllable {syl}")
        _remove_top_right_spines(ax)

        y0 = _safe_ymax_from_groups(vals, default=0.0)
        span = _safe_span_from_groups(vals, default=1.0)
        base = y0 + 0.10 * span
        h = 0.03 * span

        _add_sig_bracket(ax, 1, 3, base, h, f"Welch p={_format_p(p1)}", text_offset=0.01 * span)
        _add_sig_bracket(ax, 2, 3, base + 0.10 * span, h, f"Welch p={_format_p(p2)}", text_offset=0.01 * span)
        ax.set_ylim(ax.get_ylim()[0], y0 + 0.30 * span)

        fig.tight_layout()
        out_png = out_dir / f"delta_mean_remaining__syllable_{str(syl)}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        stats_rows.append(
            {
                "syllable": str(syl),
                "n_combined": ns[0],
                "n_single": ns[1],
                "n_sham": ns[2],
                "combined_vs_sham_t": float(t1) if np.isfinite(t1) else np.nan,
                "combined_vs_sham_p": float(p1),
                "single_vs_sham_t": float(t2) if np.isfinite(t2) else np.nan,
                "single_vs_sham_p": float(p2),
                "plot_path": str(out_png),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    out_csv = out_dir.parent / "delta_mean_remaining_by_syllable__stats.csv"
    stats_df.to_csv(out_csv, index=False)
    return out_csv


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Variance-tier ΔH_a (top-variance syllables vs low-variance syllables)
# ──────────────────────────────────────────────────────────────────────────────
def _get_variance_summary_csv_col(merged: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(merged, ["variance_summary_csv", "variance_summary_by_syllable_csv"])


def _load_variance_summary(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    needed = {"syllable", "var_summary"}
    if not needed.issubset(set(df.columns)):
        return None
    df = df.copy()
    df["syllable"] = df["syllable"].astype(str)
    df["var_summary"] = pd.to_numeric(df["var_summary"], errors="coerce")
    df = df[np.isfinite(df["var_summary"].to_numpy(float))].copy()
    return df


def build_variance_tier_delta_Ha_bird_df(
    merged: pd.DataFrame,
    *,
    top_variance_percentile: float = 70.0,
) -> pd.DataFrame:
    """
    Returns one row per bird:
      animal_id, group,
      mean_delta_Ha_high, mean_delta_Ha_low,
      n_high, n_low,
      high_syllables (comma-separated)

    Uses:
      - variance_summary_csv (syllable, var_summary) to choose "high variance" set
      - Ha_table_csv to compute delta_Ha per syllable
    """
    ha_col = _get_ha_csv_path_col(merged)
    var_col = _get_variance_summary_csv_col(merged)
    if ha_col is None or var_col is None:
        raise KeyError(
            "Need both per-bird Ha table and variance summary paths in the batch summary. "
            "Expected Ha_table_csv (or ha_csv_path) and variance_summary_csv."
        )

    rows: List[Dict[str, object]] = []

    for _, r in merged.iterrows():
        aid = str(r["animal_id"])
        grp = str(r["group"])

        ha_path = Path(str(r[ha_col]))
        var_path = Path(str(r[var_col]))

        if not ha_path.exists() or not var_path.exists():
            continue

        vdf = _load_variance_summary(var_path)
        if vdf is None or vdf.empty:
            continue

        # Pick top-variance syllables based on percentile threshold of var_summary
        q = np.nanpercentile(vdf["var_summary"].to_numpy(float), float(top_variance_percentile))
        high = vdf.loc[vdf["var_summary"] >= q, "syllable"].astype(str).tolist()
        high_set = set(high)

        hdf = pd.read_csv(ha_path)
        needed = {"syllable", "H_pre", "H_post"}
        if not needed.issubset(set(hdf.columns)):
            continue
        hdf = hdf.copy()
        hdf["syllable"] = hdf["syllable"].astype(str)
        hdf["H_pre"] = pd.to_numeric(hdf["H_pre"], errors="coerce")
        hdf["H_post"] = pd.to_numeric(hdf["H_post"], errors="coerce")
        ok = np.isfinite(hdf["H_pre"].to_numpy(float)) & np.isfinite(hdf["H_post"].to_numpy(float))
        hdf = hdf.loc[ok].copy()
        if hdf.empty:
            continue
        hdf["delta_Ha"] = hdf["H_post"] - hdf["H_pre"]

        high_df = hdf[hdf["syllable"].isin(high_set)].copy()
        low_df = hdf[~hdf["syllable"].isin(high_set)].copy()

        mean_high = float(np.nanmean(high_df["delta_Ha"].to_numpy(float))) if not high_df.empty else np.nan
        mean_low = float(np.nanmean(low_df["delta_Ha"].to_numpy(float))) if not low_df.empty else np.nan

        rows.append(
            {
                "animal_id": aid,
                "group": grp,
                "mean_delta_Ha_high": mean_high,
                "mean_delta_Ha_low": mean_low,
                "n_high": int(len(high_df)),
                "n_low": int(len(low_df)),
                "high_syllables": ",".join(sorted(high_set)),
                "variance_summary_csv": str(var_path),
                "Ha_table_csv": str(ha_path),
                "variance_percentile_threshold": float(top_variance_percentile),
                "var_summary_threshold": float(q) if np.isfinite(q) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["mean_delta_Ha_high"] = pd.to_numeric(out["mean_delta_Ha_high"], errors="coerce")
        out["mean_delta_Ha_low"] = pd.to_numeric(out["mean_delta_Ha_low"], errors="coerce")
    return out


def build_variance_tier_delta_Ha_long_df(
    merged: pd.DataFrame,
    *,
    top_variance_percentile: float = 70.0,
) -> pd.DataFrame:
    """Returns a LONG table with one row per (bird, syllable).

    Columns:
      animal_id, group, syllable, H_pre, H_post, delta_Ha, var_summary, variance_tier

    The 'high' tier is defined *within each bird* as syllables whose var_summary is
    >= the given percentile threshold (e.g., 70th percentile = top 30%).

    Requires per-bird paths in the batch summary:
      - Ha_table_csv (or ha_csv_path)
      - variance_summary_csv (or variance_summary_by_syllable_csv)
    """
    ha_col = _get_ha_csv_path_col(merged)
    var_col = _get_variance_summary_csv_col(merged)
    if ha_col is None or var_col is None:
        raise KeyError(
            "Need both per-bird Ha table and variance summary paths in the batch summary. "
            "Expected Ha_table_csv (or ha_csv_path) and variance_summary_csv."
        )

    rows: List[Dict[str, object]] = []

    for _, r in merged.iterrows():
        aid = str(r["animal_id"])
        grp = str(r["group"])

        ha_path = Path(str(r[ha_col]))
        var_path = Path(str(r[var_col]))

        if not ha_path.exists() or not var_path.exists():
            continue

        vdf = _load_variance_summary(var_path)
        if vdf is None or vdf.empty:
            continue

        q = np.nanpercentile(vdf["var_summary"].to_numpy(float), float(top_variance_percentile))
        high_set = set(vdf.loc[vdf["var_summary"] >= q, "syllable"].astype(str).tolist())

        hdf = pd.read_csv(ha_path)
        needed = {"syllable", "H_pre", "H_post"}
        if not needed.issubset(set(hdf.columns)):
            continue

        hdf = hdf.copy()
        hdf["syllable"] = hdf["syllable"].astype(str)
        hdf["H_pre"] = pd.to_numeric(hdf["H_pre"], errors="coerce")
        hdf["H_post"] = pd.to_numeric(hdf["H_post"], errors="coerce")
        ok = np.isfinite(hdf["H_pre"].to_numpy(float)) & np.isfinite(hdf["H_post"].to_numpy(float))
        hdf = hdf.loc[ok].copy()
        if hdf.empty:
            continue

        hdf["delta_Ha"] = hdf["H_post"] - hdf["H_pre"]

        # Attach var_summary per syllable (may be missing for some syllables)
        vdf_small = vdf[["syllable", "var_summary"]].copy()
        merged_s = hdf.merge(vdf_small, on="syllable", how="left")

        for _, rr in merged_s.iterrows():
            syl = str(rr["syllable"])
            tier = "high" if syl in high_set else "low"
            vs = rr.get("var_summary", float('nan'))
            rows.append(
                {
                    "animal_id": aid,
                    "group": grp,
                    "syllable": syl,
                    "H_pre": float(rr["H_pre"]),
                    "H_post": float(rr["H_post"]),
                    "delta_Ha": float(rr["delta_Ha"]),
                    "var_summary": float(vs) if np.isfinite(vs) else np.nan,
                    "variance_tier": tier,
                    "variance_summary_csv": str(var_path),
                    "Ha_table_csv": str(ha_path),
                    "variance_percentile_threshold": float(top_variance_percentile),
                    "var_summary_threshold": float(q) if np.isfinite(q) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["delta_Ha"] = pd.to_numeric(out["delta_Ha"], errors="coerce")
        out["var_summary"] = pd.to_numeric(out["var_summary"], errors="coerce")
    return out



def plot_variance_tier_boxplots_by_group(
    tier_df: pd.DataFrame,
    *,
    out_path_high: Union[str, Path],
    out_path_low: Union[str, Path],
    top_variance_percentile: Optional[float] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Variance-tier ΔHₐ plots that mirror the overall ΔHₐ plot style.

    This makes TWO figures (high-variance and low-variance).

    IMPORTANT: Every point is a (bird, syllable) pair (syllable×bird).
    Stats are computed on bird-level means within each tier (to avoid pseudo-replication).

    Expected columns in tier_df:
      - animal_id
      - group
      - variance_tier in {'high','low'}
      - delta_Ha  (preferred; per syllable)

    For backward compatibility, if delta_Ha is missing but delta_Ha_mean exists,
    it will plot bird-level points instead.
    """
    out_path_high = Path(out_path_high)
    out_path_low = Path(out_path_low)
    out_path_high.parent.mkdir(parents=True, exist_ok=True)
    out_path_low.parent.mkdir(parents=True, exist_ok=True)

    if tier_df is None or tier_df.empty:
        raise ValueError('tier_df is empty; nothing to plot.')

    ycol = 'delta_Ha' if 'delta_Ha' in tier_df.columns else 'delta_Ha_mean'
    if ycol not in tier_df.columns:
        raise KeyError("tier_df must contain 'delta_Ha' (syllable-level) or 'delta_Ha_mean' (bird-level).")

    def _plot_one(tier: str, out_path: Path) -> Dict[str, object]:
        d = tier_df[tier_df['variance_tier'].astype(str).str.lower() == tier].copy()
        if d.empty:
            raise ValueError(f'No rows for variance_tier={tier!r}.')

        vals: List[np.ndarray] = []
        ns: List[int] = []
        for g in GROUP_ORDER:
            v = pd.to_numeric(d.loc[d['group'] == g, ycol], errors='coerce').to_numpy(float)
            v = v[np.isfinite(v)]
            vals.append(v)
            ns.append(int(len(v)))

        fig, ax = plt.subplots(figsize=(10.8, 5.6))
        ax.boxplot(
            vals,
            labels=[f"{g}\n(n_pairs={n})" for g, n in zip(GROUP_ORDER, ns)],
            showfliers=True,
            whis=1.5,
        )

        rng = np.random.default_rng(seed)
        for i, v in enumerate(vals, start=1):
            j = rng.normal(0.0, 0.06, size=len(v))
            ax.scatter(np.full(len(v), i) + j, v, s=18, alpha=0.55)

        ax.set_ylabel('ΔHₐ = Hₐ_post − Hₐ_pre (bits)')

        ptxt = ''
        if top_variance_percentile is not None and np.isfinite(float(top_variance_percentile)):
            ptxt = f" (threshold: ≥{float(top_variance_percentile):g}th percentile within bird)"

        ax.set_title(f"Per-syllable entropy change (ΔHₐ) by group\n{tier.capitalize()}-variance syllables{ptxt}")
        _remove_top_right_spines(ax)

        # Stats on bird means within this tier
        if ycol == 'delta_Ha':
            bird_means = (
                d.groupby(['animal_id', 'group'], as_index=False)['delta_Ha']
                .mean()
                .rename(columns={'delta_Ha': 'mean_delta_Ha'})
            )
            comb_b = bird_means.loc[bird_means['group'] == GROUP_COMBINED, 'mean_delta_Ha'].to_numpy(float)
            single_b = bird_means.loc[bird_means['group'] == GROUP_SINGLE, 'mean_delta_Ha'].to_numpy(float)
            sham_b = bird_means.loc[bird_means['group'] == GROUP_SHAM, 'mean_delta_Ha'].to_numpy(float)
        else:
            comb_b = pd.to_numeric(d.loc[d['group'] == GROUP_COMBINED, ycol], errors='coerce').to_numpy(float)
            single_b = pd.to_numeric(d.loc[d['group'] == GROUP_SINGLE, ycol], errors='coerce').to_numpy(float)
            sham_b = pd.to_numeric(d.loc[d['group'] == GROUP_SHAM, ycol], errors='coerce').to_numpy(float)

        t1, p1 = _welch_ttest(comb_b, sham_b)
        t2, p2 = _welch_ttest(single_b, sham_b)

        y0 = _safe_ymax_from_groups(vals, default=0.0)
        span = _safe_span_from_groups(vals, default=1.0)
        base = y0 + 0.10 * span
        h = 0.03 * span

        _add_sig_bracket(ax, 1, 3, base, h, f"Bird-mean Welch p={_format_p(p1)}", text_offset=0.01 * span)
        _add_sig_bracket(ax, 2, 3, base + 0.10 * span, h, f"Bird-mean Welch p={_format_p(p2)}", text_offset=0.01 * span)

        ax.set_ylim(ax.get_ylim()[0], y0 + 0.30 * span)

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        return {
            'tests_bird_mean': {
                'combined_vs_sham': {
                    't': float(t1) if np.isfinite(t1) else np.nan,
                    'p': float(p1),
                    'n_comb': int(np.isfinite(comb_b).sum()),
                    'n_sham': int(np.isfinite(sham_b).sum()),
                },
                'single_vs_sham': {
                    't': float(t2) if np.isfinite(t2) else np.nan,
                    'p': float(p2),
                    'n_single': int(np.isfinite(single_b).sum()),
                    'n_sham': int(np.isfinite(sham_b).sum()),
                },
            },
            'n_pairs_by_group': dict(zip(GROUP_ORDER, ns)),
            'out_path': str(out_path),
        }

    high_res = _plot_one('high', out_path_high)
    low_res = _plot_one('low', out_path_low)

    return {
        'high': high_res,
        'low': low_res,
        'out_high': str(out_path_high),
        'out_low': str(out_path_low),
    }


def plot_variance_tier_paired_within_group(
    bird_tier_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    seed: int = 0,
) -> Dict[str, object]:
    """
    For each lesion group, compare within-bird:
      mean_delta_Ha_high vs mean_delta_Ha_low (paired).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = bird_tier_df.copy()
    df = df[np.isfinite(df["mean_delta_Ha_high"].to_numpy(float)) & np.isfinite(df["mean_delta_Ha_low"].to_numpy(float))].copy()
    if df.empty:
        raise ValueError("No birds with both high and low tier values.")

    group_color = {GROUP_COMBINED: "C0", GROUP_SINGLE: "C1", GROUP_SHAM: "C2"}

    centers = np.array([1.0, 4.0, 7.0], dtype=float)
    dx = 0.35
    x_high = centers - dx
    x_low = centers + dx

    all_vals = np.concatenate([df["mean_delta_Ha_high"].to_numpy(float), df["mean_delta_Ha_low"].to_numpy(float)])
    all_vals = all_vals[np.isfinite(all_vals)]
    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    span = max(1e-6, y_max - y_min)

    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    high_handle = ax.scatter([], [], s=80, facecolor="none", edgecolor="black", linewidth=1.6, label="Top-variance (open)")
    low_handle = ax.scatter([], [], s=80, facecolor="black", edgecolor="black", linewidth=1.0, label="Low-variance (filled)")

    stats_by_group: Dict[str, Dict[str, float]] = {}

    for gi, g in enumerate(GROUP_ORDER):
        dfg = df[df["group"] == g].copy().sort_values("animal_id")
        high = dfg["mean_delta_Ha_high"].to_numpy(float)
        low = dfg["mean_delta_Ha_low"].to_numpy(float)
        n = int(len(dfg))

        ax.boxplot(
            [high, low],
            positions=[x_high[gi], x_low[gi]],
            widths=0.42,
            showfliers=False,
            patch_artist=False,
        )

        jit = _jitter(n, seed=seed + gi, scale=0.02)
        c = group_color.get(g, "0.3")

        for i in range(n):
            ax.plot([x_high[gi] + jit[i], x_low[gi] + jit[i]], [high[i], low[i]],
                    color=c, alpha=0.45, linewidth=1.3)

        ax.scatter(
            np.full(n, x_high[gi]) + jit,
            high,
            s=70,
            facecolor="none",
            edgecolor=c,
            linewidth=1.9,
            zorder=3,
        )
        ax.scatter(
            np.full(n, x_low[gi]) + jit,
            low,
            s=70,
            facecolor=c,
            edgecolor=c,
            linewidth=1.0,
            zorder=3,
        )

        W, p = wilcoxon_signed_rank_paired(high, low, seed=seed)
        stats_by_group[g] = {"n": float(n), "W": float(W) if np.isfinite(W) else np.nan, "p": float(p)}

        local_max = float(np.nanmax(np.r_[high, low])) if n > 0 else y_max
        base = local_max + 0.07 * span
        h = 0.03 * span
        label = f"p={_format_p(p)}\\n(n={n})"
        _add_sig_bracket(ax, x_high[gi], x_low[gi], base, h, label, text_offset=0.015 * span)

    ax.set_xticks(centers)
    ax.set_xticklabels(list(GROUP_ORDER), rotation=15, ha="right")
    ax.set_ylabel("Mean ΔHₐ per bird (bits)")
    ax.set_title("Within-bird comparison: ΔHₐ in high-variance vs low-variance syllables")

    ax.set_ylim(y_min - 0.05 * span, y_max + 0.45 * span)

    _remove_top_right_spines(ax)
    ax.legend(handles=[high_handle, low_handle], frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"stats_by_group": stats_by_group, "out_path": str(out_path)}


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────────────────
def make_aggregate_entropy_figures(
    *,
    root_dir: Union[str, Path],
    metadata_xlsx: Union[str, Path],
    batch_summary_csv: Optional[Union[str, Path]] = None,
    metadata_sheet_name: str = "animal_hit_type_summary",
    out_dir: Optional[Union[str, Path]] = None,

    make_delta_TTE: bool = True,

    make_delta_Ha: bool = True,
    make_delta_Ha_per_syllable: bool = True,

    # NEW:
    make_mean_remaining: bool = True,
    make_mean_remaining_per_syllable: bool = False,

    make_variance_tier_delta_Ha: bool = True,
    top_variance_percentile: float = 70.0,      # top 30% variance syllables
    make_variance_tier_paired: bool = True,

    seed: int = 0,
) -> Dict[str, object]:
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir.parent / "entropy_figures" / "aggregate"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, batch_csv_p, meta_xlsx_p = merge_batch_with_groups(
        root_dir=root_dir,
        metadata_xlsx=metadata_xlsx,
        batch_summary_csv=batch_summary_csv,
        metadata_sheet_name=metadata_sheet_name,
    )

    per_bird_csv = out_dir / "aggregate__per_bird_TE_with_groups.csv"
    merged.to_csv(per_bird_csv, index=False)

    paths: Dict[str, str] = {"out_dir": str(out_dir), "per_bird_csv": str(per_bird_csv)}
    results: Dict[str, object] = {}

    # --- TTE combined plot (pre vs post within each group) ---
    tte_plot = out_dir / "TTE_pre_post_by_group__paired.png"
    tte_stats_txt = out_dir / "TTE_pre_post_by_group__paired_stats.txt"

    tte_res = plot_TTE_pre_post_by_group(
        merged,
        out_path=tte_plot,
        title=None,
        seed=seed,
    )
    paths["TTE_pre_post_by_group__paired"] = str(tte_plot)
    results["TTE_pre_post_by_group__paired"] = tte_res

    lines = []
    lines.append(f"Batch summary: {batch_csv_p}")
    lines.append(f"Metadata: {meta_xlsx_p} (sheet={metadata_sheet_name})")
    lines.append("")
    lines.append("Paired within-group tests (TE_pre vs TE_post):")
    stats_by_group = tte_res.get("stats_by_group", {})
    for g in GROUP_ORDER:
        d = stats_by_group.get(g, {})
        n = d.get("n", np.nan)
        W = d.get("W", np.nan)
        p = d.get("p", np.nan)
        lines.append(f"  {g}: n={int(n) if np.isfinite(n) else 'n/a'}, W={W}, p={p}")
    tte_stats_txt.write_text("\\n".join(lines) + "\\n")
    paths["TTE_pre_post_by_group__paired_stats"] = str(tte_stats_txt)

    # --- ΔTTE across groups ---
    if make_delta_TTE:
        dplot = out_dir / "delta_TTE_by_group__boxplot.png"
        dstat = out_dir / "delta_TTE_by_group__stats.txt"
        dres = plot_delta_TTE_by_group(merged, out_path=dplot, seed=seed)
        paths["delta_TTE_by_group__boxplot"] = str(dplot)
        results["delta_TTE_by_group"] = dres

        t = dres["tests"]
        dlines = []
        dlines.append("Between-group tests on ΔTTE = TE_post - TE_pre (Welch t-test):")
        dlines.append(f"  Combined vs Sham: t={t['combined_vs_sham']['t']}, p={t['combined_vs_sham']['p']}")
        dlines.append(f"  Single vs Sham:   t={t['single_vs_sham']['t']}, p={t['single_vs_sham']['p']}")
        dstat.write_text("\\n".join(dlines) + "\\n")
        paths["delta_TTE_by_group__stats"] = str(dstat)

    # --- ΔH_a (syllable-level) ---
    if make_delta_Ha:
        try:
            delta_df = build_delta_Ha_long_df(merged)
            delta_csv = out_dir / "delta_Ha__all_birds_all_syllables_long.csv"
            delta_df.to_csv(delta_csv, index=False)
            paths["delta_Ha_long_csv"] = str(delta_csv)

            dh_plot = out_dir / "delta_Ha_overall_by_group__boxplot.png"
            dh_res = plot_delta_Ha_overall_by_group(delta_df, out_path=dh_plot, seed=seed)
            paths["delta_Ha_overall_by_group__boxplot"] = str(dh_plot)
            results["delta_Ha_overall_by_group"] = dh_res

            if make_delta_Ha_per_syllable:
                syl_dir = out_dir / "delta_Ha_by_syllable"
                stats_csv = plot_delta_Ha_by_syllable_across_birds(
                    delta_df,
                    out_dir=syl_dir,
                    min_birds_per_group=3,
                    seed=seed,
                )
                paths["delta_Ha_by_syllable_dir"] = str(syl_dir)
                paths["delta_Ha_by_syllable_stats_csv"] = str(stats_csv)

        except Exception as e:
            results["delta_Ha_error"] = str(e)

    # --- NEW: mean remaining syllables to end-of-song ---
    if make_mean_remaining:
        try:
            mr_df = build_mean_remaining_long_df(merged)
            mr_csv = out_dir / "mean_remaining__all_birds_all_syllables_long.csv"
            mr_df.to_csv(mr_csv, index=False)
            paths["mean_remaining_long_csv"] = str(mr_csv)

            mr_plot = out_dir / "mean_remaining_delta_overall_by_group__boxplot.png"
            mr_res = plot_delta_mean_remaining_overall_by_group(mr_df, out_path=mr_plot, seed=seed)
            paths["mean_remaining_delta_overall_by_group__boxplot"] = str(mr_plot)
            results["mean_remaining_delta_overall_by_group"] = mr_res

            if make_mean_remaining_per_syllable:
                mr_syl_dir = out_dir / "mean_remaining_delta_by_syllable"
                mr_stats_csv = plot_delta_mean_remaining_by_syllable_across_birds(
                    mr_df,
                    out_dir=mr_syl_dir,
                    min_birds_per_group=3,
                    seed=seed,
                )
                paths["mean_remaining_delta_by_syllable_dir"] = str(mr_syl_dir)
                paths["mean_remaining_delta_by_syllable_stats_csv"] = str(mr_stats_csv)

        except Exception as e:
            results["mean_remaining_error"] = str(e)

    # --- NEW: variance-tier ΔH_a analysis ---
    if make_variance_tier_delta_Ha:
        try:
            # LONG table: one row per (bird, syllable) with variance tier labels
            tier_long_df = build_variance_tier_delta_Ha_long_df(
                merged,
                top_variance_percentile=float(top_variance_percentile),
            )
            tier_long_csv = out_dir / "variance_tier__delta_Ha_points_long.csv"
            tier_long_df.to_csv(tier_long_csv, index=False)
            paths["variance_tier_delta_Ha_points_long_csv"] = str(tier_long_csv)

            # Also keep bird-level means (used for paired high-vs-low plot)
            tier_df = build_variance_tier_delta_Ha_bird_df(
                merged,
                top_variance_percentile=float(top_variance_percentile),
            )
            tier_csv = out_dir / "variance_tier__delta_Ha_bird_means.csv"
            tier_df.to_csv(tier_csv, index=False)
            paths["variance_tier_delta_Ha_bird_means_csv"] = str(tier_csv)

            high_plot = out_dir / "variance_tier__delta_Ha_high_by_group__boxplot.png"
            low_plot = out_dir / "variance_tier__delta_Ha_low_by_group__boxplot.png"
            tier_res = plot_variance_tier_boxplots_by_group(
                tier_long_df,
                out_path_high=high_plot,
                out_path_low=low_plot,
                top_variance_percentile=float(top_variance_percentile),
                seed=seed,
            )
            paths["variance_tier__delta_Ha_high_by_group__boxplot"] = str(high_plot)
            paths["variance_tier__delta_Ha_low_by_group__boxplot"] = str(low_plot)
            results["variance_tier_boxplots"] = tier_res

            if make_variance_tier_paired:
                paired_plot = out_dir / "variance_tier__delta_Ha_high_vs_low__paired_within_group.png"
                paired_res = plot_variance_tier_paired_within_group(tier_df, out_path=paired_plot, seed=seed)
                paths["variance_tier__paired_plot"] = str(paired_plot)
                results["variance_tier_paired"] = paired_res

        except Exception as e:
            results["variance_tier_error"] = str(e)

    return {"paths": paths, "results": results}


# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL CLI
# ──────────────────────────────────────────────────────────────────────────────
def _main():
    import argparse
    p = argparse.ArgumentParser(
        description="Aggregate entropy plots across birds into entropy_figures/aggregate/"
    )
    p.add_argument("--root", required=True, type=str, help="Root directory containing bird folders")
    p.add_argument("--metadata-xlsx", required=True, type=str, help="Path to metadata Excel with hit types")
    p.add_argument("--sheet-name", default="animal_hit_type_summary", type=str, help="Excel sheet name")
    p.add_argument("--batch-summary-csv", default=None, type=str, help="Optional override to batch summary CSV")
    p.add_argument("--out-dir", default=None, type=str, help="Optional output directory")

    p.add_argument("--no-delta-tte", action="store_true", help="Disable ΔTTE across-group plot")
    p.add_argument("--no-delta-ha", action="store_true", help="Disable ΔHa plots")
    p.add_argument("--no-delta-ha-per-syllable", action="store_true", help="Disable per-syllable ΔHa plots")

    p.add_argument("--no-mean-remaining", action="store_true", help="Disable mean-remaining Δ plots")
    p.add_argument("--mean-remaining-per-syllable", action="store_true", help="Enable per-syllable mean-remaining Δ plots")

    p.add_argument("--no-variance-tier", action="store_true", help="Disable variance-tier ΔHa analysis")
    p.add_argument("--top-variance-percentile", default=70.0, type=float, help="Percentile threshold for TOP variance syllables (default=70)")
    p.add_argument("--no-variance-tier-paired", action="store_true", help="Disable paired high-vs-low variance plot")

    args = p.parse_args()

    res = make_aggregate_entropy_figures(
        root_dir=args.root,
        metadata_xlsx=args.metadata_xlsx,
        batch_summary_csv=args.batch_summary_csv,
        metadata_sheet_name=args.sheet_name,
        out_dir=args.out_dir,

        make_delta_TTE=not args.no_delta_tte,

        make_delta_Ha=not args.no_delta_ha,
        make_delta_Ha_per_syllable=not args.no_delta_ha_per_syllable,

        make_mean_remaining=not args.no_mean_remaining,
        make_mean_remaining_per_syllable=bool(args.mean_remaining_per_syllable),

        make_variance_tier_delta_Ha=not args.no_variance_tier,
        top_variance_percentile=float(args.top_variance_percentile),
        make_variance_tier_paired=not args.no_variance_tier_paired,

        seed=0,
    )
    print(res["paths"])


if __name__ == "__main__":
    _main()


# ──────────────────────────────────────────────────────────────────────────────
# COMMENTED-OUT SPYDER CONSOLE EXAMPLE
# ──────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
import sys, importlib

code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import entropy_aggregate_across_birds as eagg
importlib.reload(eagg)

root_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
metadata_xlsx = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx")

res = eagg.make_aggregate_entropy_figures(
    root_dir=root_dir,
    metadata_xlsx=metadata_xlsx,
    metadata_sheet_name="animal_hit_type_summary",

    make_delta_TTE=True,

    make_delta_Ha=True,
    make_delta_Ha_per_syllable=True,

    make_mean_remaining=True,
    make_mean_remaining_per_syllable=False,

    make_variance_tier_delta_Ha=True,
    top_variance_percentile=70.0,
    make_variance_tier_paired=True,
)

print(res["paths"]["variance_tier__delta_Ha_high_by_group__boxplot"])
"""
