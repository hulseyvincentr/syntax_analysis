# song_duration_comparison.py
# -*- coding: utf-8 -*-
"""
Pre/Post and Early-Pre vs Late-Pre vs Post song-duration comparisons
with readable legends and on-plot significance stars.

Depends on:
  - organize_song_detection_json.py
  - merge_potential_split_up_song.py

Multi-bird mode:
  - Finds <animal_id>_song_detection.json under a root directory
  - Looks up each bird's treatment date in a metadata Excel sheet (MATCHED BY animal_id)
  - Runs per-bird plots + writes a summary CSV

Also produces (multi-bird):
  1) Aggregate "spaghetti" plot: each bird's mean ± SEM across EarlyPre/LatePre/Post
     connected per bird and color-coded by Sham vs Bilateral NMA lesion.
     Can label each animal_id at Post AND show a right-side "key" legend of animal IDs.

  2) Aggregated epoch boxplots (across birds; per-bird means), one for Sham and one for NMA,
     EACH IN TWO VERSIONS:
       - CONNECTED: lines connect each bird's 3 points
       - NOLINES: no connecting lines

     Includes KW + pairwise MWU (BH) + bracket stars.

  3) Mean-delta boxplots (across birds; per-bird means), Sham and NMA,
     EACH IN TWO VERSIONS:
       A) Absolute differences (seconds):
            - (Late−Early)
            - (Post−Late)
       B) Percent change (TRUE percent units; multiplied by 100):
            - 100*(Late−Early)/Early
            - 100*(Post−Late)/Late

     Delta-plot significance:
       - Paired test across birds comparing the two deltas:
            (Post−Late) vs (Late−Early)
       - One bracket with stars.

  4) Variance-delta boxplots (across birds; per-bird within-epoch variance, s^2), Sham and NMA,
     EACH IN TWO VERSIONS:
       A) Absolute differences (s^2):
            - (Late−Early)
            - (Post−Late)
       B) Percent change (TRUE percent units; multiplied by 100):
            - 100*(Late−Early)/Early
            - 100*(Post−Late)/Late
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.stats import mannwhitneyu, kruskal
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from organize_song_detection_json import build_detected_song_segments
from merge_potential_split_up_song import build_detected_and_merged_songs


# ───────────── ID + metadata helpers ─────────────
def _normalize_animal_id(x: Any) -> str:
    """
    Normalize IDs from both JSON filenames and Excel cells so they match.
    Also strips macOS AppleDouble prefixes (._) and common Excel artifacts.
    """
    s = str(x).strip()

    # macOS AppleDouble prefix
    if s.startswith("._"):
        s = s[2:]

    # leading hidden-file dot
    s = s.lstrip(".")

    # remove spaces/underscores
    s = s.replace(" ", "").replace("_", "")

    # Excel artifact: "USA5288.0"
    if s.endswith(".0"):
        s = s[:-2]

    return s.strip().upper()


def _infer_animal_id_from_song_detection_json(path: Union[str, Path]) -> str:
    p = Path(path)
    name = p.name

    if name.endswith("_song_detection.json"):
        return _normalize_animal_id(name.replace("_song_detection.json", ""))

    stem = p.stem
    if "_song_detection" in stem:
        return _normalize_animal_id(stem.split("_song_detection")[0])

    return _normalize_animal_id(stem)


def _normalize_colname(s: str) -> str:
    return "".join(
        ch for ch in s.strip().lower().replace("_", " ")
        if ch.isalnum() or ch.isspace()
    ).strip()


def _guess_columns_for_treatment(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Strict inference: avoids generic 'id' to prevent wrong column matches.
    """
    cols = list(df.columns)
    norm = {_normalize_colname(c): c for c in cols}

    id_candidates = ["animal id", "animalid", "bird id", "birdid"]
    date_candidates = [
        "treatment date", "treatmentdate",
        "surgery date", "surgerydate",
        "lesion date", "date of treatment",
    ]

    def find_best(cands: List[str]) -> Optional[str]:
        for k in cands:
            if k in norm:
                return norm[k]
        for k in cands:
            for nk, orig in norm.items():
                if k in nk:
                    return orig
        return None

    id_col = find_best(id_candidates)
    date_col = find_best(date_candidates)
    if id_col is None or date_col is None:
        raise ValueError(
            "Could not infer required columns from metadata excel.\n"
            f"Found columns: {cols}\n"
            "Pass id_col=... and treatment_date_col=... explicitly."
        )
    return id_col, date_col


def _guess_group_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    norm = {_normalize_colname(c): c for c in cols}

    group_candidates = [
        "group",
        "treatment type", "treatmenttype",
        "condition",
        "lesion type", "lesiontype",
        "injection type", "injectiontype",
        "treatment",
    ]

    for k in group_candidates:
        if k in norm:
            return norm[k]
    for k in group_candidates:
        for nk, orig in norm.items():
            if k in nk:
                return orig
    return None


def _classify_sham_vs_nma(val: Any) -> str:
    """
    Map free-form metadata text to:
      - "Sham lesion"
      - "Bilateral NMA lesion"
      - "Unknown"
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Unknown"
    s = str(val).strip().lower()

    if ("sham" in s) or ("saline" in s) or ("control" in s):
        return "Sham lesion"

    if ("nma" in s) or ("lesion" in s) or ("experimental" in s):
        return "Bilateral NMA lesion"

    return "Unknown"


def _lookup_treatment_date_for_animal(
    meta_norm: pd.DataFrame,
    *,
    animal_id: str,
    id_col: str,
    treat_dt_col_norm: str = "_treat_dt",
) -> Optional[pd.Timestamp]:
    aid = _normalize_animal_id(animal_id)
    sub = meta_norm.loc[meta_norm[id_col] == aid, treat_dt_col_norm]
    if sub.empty:
        return None
    dt = pd.to_datetime(sub, errors="coerce").dropna()
    if dt.empty:
        return None
    return dt.min().normalize()


def _lookup_group_for_animal(
    meta_norm: pd.DataFrame,
    *,
    animal_id: str,
    id_col: str,
    group_col: Optional[str],
) -> str:
    if group_col is None or group_col not in meta_norm.columns:
        return "Unknown"

    aid = _normalize_animal_id(animal_id)
    sub = meta_norm.loc[meta_norm[id_col] == aid, group_col]
    if sub.empty:
        return "Unknown"

    labels = [_classify_sham_vs_nma(v) for v in sub.values]
    labels = [x for x in labels if x != "Unknown"]
    if not labels:
        return "Unknown"

    vals, counts = np.unique(labels, return_counts=True)
    return str(vals[np.argmax(counts)])


# ───────────── stats helpers ─────────────
def _try_mannwhitney(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    if _HAVE_SCIPY:
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(stat), float(p)
    # fallback (rough)
    ma, mb = np.mean(x), np.mean(y)
    va, vb = np.var(x, ddof=1), np.var(y, ddof=1)
    na, nb = len(x), len(y)
    t = (ma - mb) / np.sqrt(va/na + vb/nb + 1e-12)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p)


def _try_kruskal(groups: List[np.ndarray]) -> Tuple[float, float]:
    gs = [np.asarray(g, float)[~np.isnan(g)] for g in groups if len(g)]
    if len(gs) < 2:
        return float("nan"), float("nan")
    if _HAVE_SCIPY:
        H, p = kruskal(*gs)
        return float(H), float(p)
    # fallback (rough)
    ranks = np.argsort(np.argsort(np.concatenate(gs)))
    sizes = [len(g) for g in gs]
    splits = np.cumsum([0] + sizes)
    parts = [ranks[splits[i]:splits[i+1]] for i in range(len(sizes))]
    means = [np.mean(p) for p in parts]
    grand = np.mean(ranks)
    ss_between = sum(n * (m - grand)**2 for n, m in zip(sizes, means))
    ss_within = sum(((p - np.mean(p))**2).sum() for p in parts)
    from math import erf, sqrt
    F = (ss_between / (len(gs)-1 + 1e-12)) / (ss_within / (len(ranks)-len(gs) + 1e-12) + 1e-12)
    p = 2 * (1 - 0.5 * (1 + erf(abs(F) / sqrt(2))))
    return float(F), float(p)


def _p_adjust_bh(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan, dtype=float)
    if mask.sum() == 0:
        return out.tolist()
    ps = p[mask]
    n = ps.size
    order = np.argsort(ps)
    ranks = np.empty(n, int); ranks[order] = np.arange(1, n+1)
    adj = ps * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    res = np.empty(n, float)
    res[order] = np.minimum(adj_sorted, 1.0)
    out[mask] = res
    return out.tolist()


def _stars(p: float) -> str:
    if np.isnan(p): return "n.s."
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "n.s."


def _fmt_p(p: float) -> str:
    if np.isnan(p): return "nan"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def _try_paired_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, str]:
    """
    Paired test for x vs y (two-sided). Returns (stat, p, test_name).
    Uses SciPy Wilcoxon if available, otherwise paired-t fallback (rough).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan"), float("nan"), "paired"

    if _HAVE_SCIPY:
        try:
            from scipy.stats import wilcoxon
            d = y - x
            if np.allclose(d, 0):
                return 0.0, 1.0, "wilcoxon"
            stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            return float(stat), float(p), "wilcoxon"
        except Exception:
            pass

    # fallback: paired t-test on differences with rough normal p
    d = y - x
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan"), "paired_t_fallback"

    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan"), float("nan"), "paired_t_fallback"

    t = md / (sd / np.sqrt(d.size) + 1e-12)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p), "paired_t_fallback"


# ───────────── data prep ─────────────
def _prepare_table(
    json_input: Union[str, Path],
    *,
    merge_gap_ms: Optional[int] = None,
    songs_only: bool = True
) -> pd.DataFrame:
    if merge_gap_ms and merge_gap_ms > 0:
        out = build_detected_and_merged_songs(
            json_input,
            songs_only=songs_only,
            flatten_spec_params=True,
            max_gap_between_song_segments=int(merge_gap_ms),
        )
        df = out["detected_merged_songs"]
    else:
        df = build_detected_song_segments(
            json_input,
            songs_only=songs_only,
            drop_no_segments=True,
            flatten_spec_params=True,
            max_gap_between_song_segments=1000,
        )
    return df.copy()


def _split_pre_post(df: pd.DataFrame, treatment_date: Union[str, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(treatment_date, pd.Timestamp):
        treatment_date = pd.to_datetime(str(treatment_date)).normalize()
    d = df.copy()
    d = d[pd.notna(d["recording_datetime"])].copy()
    d["recording_datetime"] = pd.to_datetime(d["recording_datetime"])
    return d[d["recording_datetime"] < treatment_date].copy(), d[d["recording_datetime"] >= treatment_date].copy()


def _split_three_groups_balanced(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    pre_sorted  = pre_df.sort_values("recording_datetime").reset_index(drop=True)
    post_sorted = post_df.sort_values("recording_datetime").reset_index(drop=True)
    n = int(min(len(pre_sorted)//2, len(post_sorted)))
    if n <= 0:
        return (pre_sorted.iloc[0:0], pre_sorted.iloc[0:0], post_sorted.iloc[0:0], 0)
    early = pre_sorted.iloc[len(pre_sorted)-2*n : len(pre_sorted)-n]
    late  = pre_sorted.tail(n)
    postn = post_sorted.head(n)
    return early.copy(), late.copy(), postn.copy(), n


def _durations_seconds(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df["length_ms"], errors="coerce") / 1000.0
    return s[(pd.notna(s)) & (s > 0)]


def _mean_sem(x: Union[np.ndarray, pd.Series]) -> Tuple[float, float, int]:
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = float(np.mean(arr))
    if n < 2:
        return mu, float("nan"), n
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mu, sem, n


def _var_s2(x: Union[np.ndarray, pd.Series]) -> Tuple[float, int]:
    """
    Sample variance (ddof=1) in seconds^2. Returns (var, n).
    If n < 2, variance is NaN.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n < 2:
        return float("nan"), n
    return float(np.var(arr, ddof=1)), n


# ───────────── styling + annotation ─────────────
def _style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)


def _add_sig_brackets(
    ax,
    pairs: List[Tuple[int, int]],
    pvals_adj: List[float],
    data_max: float,
    y_pad_frac: float = 0.06
):
    """
    Draw bracketed significance comparisons between x-positions (1-indexed).
    Expands y-limits to make room.
    """
    y0, y1 = ax.get_ylim()
    ymax = max(y1, data_max)
    span = max(1e-6, ymax - y0)
    step = span * y_pad_frac
    y = ymax + step
    new_top = ymax + step * (len(pairs) + 1)
    ax.set_ylim(y0, new_top)

    for (i, j), p in zip(pairs, pvals_adj):
        ax.plot([i, i, j, j], [y, y+step*0.3, y+step*0.3, y],
                color="black", linewidth=1.2, clip_on=False, zorder=10)
        ax.text((i + j) / 2, y + step*0.35, _stars(p),
                ha="center", va="bottom", fontsize=12, color="black", zorder=11)
        y += step


# ───────────── Plot 1: Pre vs Post ─────────────
def plot_pre_post_boxplot(
    pre_s: pd.Series,
    post_s: pd.Series,
    *,
    treatment_date: Union[str, pd.Timestamp],
    animal_id: Optional[str] = None,
    merge_gap_ms: Optional[int] = None,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    jitter: float = 0.08,
    point_size: float = 12.0,
    point_alpha: float = 0.2,
    bw_box_alpha: float = 0.35
) -> Dict[str, Any]:
    if not isinstance(treatment_date, pd.Timestamp):
        treatment_date = pd.to_datetime(str(treatment_date)).normalize()

    data = []; xpos = []; labels = []; pos = 1
    if len(pre_s):  data.append(pre_s.values);  xpos.append(pos); labels.append(f"Pre\nn={pre_s.size}");  pos += 1
    if len(post_s): data.append(post_s.values); xpos.append(pos); labels.append(f"Post\nn={post_s.size}")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(right=0.78)

    if data:
        if len(pre_s):
            ax.scatter(np.random.uniform(-jitter, jitter, size=len(pre_s)) + xpos[0],
                       pre_s.values, s=point_size, alpha=point_alpha, linewidths=0,
                       edgecolors="none", zorder=1)
        if len(post_s):
            ax.scatter(np.random.uniform(-jitter, jitter, size=len(post_s)) + xpos[-1],
                       post_s.values, s=point_size, alpha=point_alpha, linewidths=0,
                       edgecolors="none", zorder=1)

        bp = ax.boxplot(
            data, positions=xpos, labels=labels, patch_artist=True,
            showmeans=True, meanline=True, showfliers=False,
            boxprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            medianprops=dict(color="black", linewidth=2.0, linestyle="-"),
            meanprops=dict(color="black", linewidth=2.0, linestyle="--"),
            zorder=2,
        )
        for b in bp["boxes"]:
            b.set_facecolor("white"); b.set_alpha(bw_box_alpha)

        if len(pre_s) and len(post_s):
            _, p = _try_mannwhitney(pre_s.values, post_s.values)
            data_max = np.nanmax(np.concatenate([pre_s.values, post_s.values]))
            _add_sig_brackets(ax, [(1, 2)], [p], data_max)

        ax.set_ylabel("Song duration (s)")
        prefix = f"{animal_id} — " if animal_id else ""
        t = title or f"{prefix}Song durations Pre vs Post (Treatment: {treatment_date.date()})"
        if merge_gap_ms and merge_gap_ms > 0:
            t += f"\n(Merged gap < {merge_gap_ms} ms)"
        ax.set_title(t)

        handles = [
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="Median"),
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="Mean"),
            Line2D([0], [0], marker="o", linestyle="None", color="black",
                   alpha=point_alpha, markersize=6, label="Song (point)"),
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, borderaxespad=0.0)
    else:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=12)
        ax.axis("off")

    _style_ax(ax)

    saved = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved = str(output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"fig": fig, "ax": ax, "saved_path": saved}


# ───────────── Plot 2: 3 groups + stars ─────────────
def plot_three_group_boxplot(
    early_s: pd.Series,
    late_s: pd.Series,
    post_s: pd.Series,
    *,
    treatment_date: Union[str, pd.Timestamp],
    animal_id: Optional[str] = None,
    merge_gap_ms: Optional[int] = None,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    jitter: float = 0.08,
    point_size: float = 12.0,
    point_alpha: float = 0.2,
    bw_box_alpha: float = 0.35,
    do_stats: bool = True
) -> Dict[str, Any]:
    if not isinstance(treatment_date, pd.Timestamp):
        treatment_date = pd.to_datetime(str(treatment_date)).normalize()

    ordered = [("Early Pre", early_s), ("Late Pre", late_s), ("Post", post_s)]
    present = [(lab, s) for (lab, s) in ordered if len(s)]

    groups = []
    labels = []
    arrs: Dict[str, np.ndarray] = {}
    for lab, s in present:
        groups.append(s.values)
        labels.append(f"{lab}\nn={s.size}")
        arrs[lab] = s.values

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.subplots_adjust(right=0.78)
    stats_out: Dict[str, Any] = {"kw": None, "pairwise": []}

    if groups:
        xpos = np.arange(1, len(groups) + 1)

        for i, (lab, s) in enumerate(present):
            ax.scatter(np.random.uniform(-jitter, jitter, size=len(s)) + xpos[i],
                       s.values, s=point_size, alpha=point_alpha,
                       linewidths=0, edgecolors="none", zorder=1)

        bp = ax.boxplot(
            groups, positions=xpos, labels=labels, patch_artist=True,
            showmeans=True, meanline=True, showfliers=False,
            boxprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            medianprops=dict(color="black", linewidth=2.0, linestyle="-"),
            meanprops=dict(color="black", linewidth=2.0, linestyle="--"),
            zorder=3,
        )
        for b in bp["boxes"]:
            b.set_facecolor("white"); b.set_alpha(bw_box_alpha)

        ax.set_ylabel("Song duration (s)")
        prefix = f"{animal_id} — " if animal_id else ""
        t = title or f"{prefix}Song durations: Early Pre / Late Pre / Post (Treatment: {treatment_date.date()})"
        if merge_gap_ms and merge_gap_ms > 0:
            t += f"\n(Balanced groups, merged gap < {merge_gap_ms} ms)"
        else:
            t += "\n(Balanced groups)"
        ax.set_title(t)

        handles = [
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="Median"),
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="Mean"),
            Line2D([0], [0], marker="o", linestyle="None", color="black",
                   alpha=point_alpha, markersize=6, label="Song (point)"),
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, borderaxespad=0.0)

        if do_stats and len(arrs) >= 2:
            order = [lab for (lab, _) in present]
            arrays = [np.asarray(arrs[g], float) for g in order]

            H, p_kw = _try_kruskal(arrays)
            stats_out["kw"] = {"groups": order, "H": H, "p": p_kw}

            from itertools import combinations
            pairs_lbl = list(combinations(order, 2))
            raw_p = []
            pair_stats = []
            for a, b in pairs_lbl:
                stat, p = _try_mannwhitney(arrs[a], arrs[b])
                raw_p.append(p)
                pair_stats.append({"a": a, "b": b, "stat": stat, "p_raw": p})

            adj = _p_adjust_bh(raw_p)
            for i, d in enumerate(pair_stats):
                d["p_bh"] = adj[i]
                d["stars"] = _stars(adj[i])
            stats_out["pairwise"] = pair_stats

            summary_lines = [f"Kruskal–Wallis: H={H:.3g}, p={_fmt_p(p_kw)}"]
            for d in pair_stats:
                summary_lines.append(f"{d['a']} vs {d['b']}: p(BH)={_fmt_p(d['p_bh'])}  [{d['stars']}]")
            fig.text(
                0.80, 0.98, "\n".join(summary_lines),
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95)
            )

            xmap = {lab: i+1 for i, lab in enumerate(order)}
            star_pairs = [(xmap[a], xmap[b]) for (a, b) in pairs_lbl]
            data_max = np.nanmax(np.concatenate(arrays))
            _add_sig_brackets(ax, star_pairs, adj, data_max)

    else:
        ax.text(0.5, 0.5, "Insufficient data for balanced three-group plot",
                ha="center", va="center", fontsize=12)
        ax.axis("off")

    _style_ax(ax)

    saved = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved = str(output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fig": fig, "ax": ax, "saved_path": saved, "stats": stats_out}


# ───────────── public entrypoint (single bird) ─────────────
def run_song_duration_comparison(
    json_input: Union[str, Path],
    *,
    treatment_date: Union[str, pd.Timestamp],
    animal_id: Optional[str] = None,
    merge_gap_ms: Optional[int] = None,
    songs_only: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    three_output_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Dict[str, Any]:
    if animal_id is None:
        animal_id = _infer_animal_id_from_song_detection_json(json_input)

    df = _prepare_table(json_input, merge_gap_ms=merge_gap_ms, songs_only=songs_only)

    pre_df, post_df = _split_pre_post(df, treatment_date=treatment_date)
    pre_s  = _durations_seconds(pre_df)
    post_s = _durations_seconds(post_df)

    plot1 = plot_pre_post_boxplot(
        pre_s, post_s,
        treatment_date=treatment_date,
        animal_id=animal_id,
        merge_gap_ms=merge_gap_ms,
        output_path=output_path,
        show=show,
    )

    early_df, late_df, post_bal_df, n = _split_three_groups_balanced(pre_df, post_df)
    early_s = _durations_seconds(early_df)
    late_s  = _durations_seconds(late_df)
    post_b  = _durations_seconds(post_bal_df)

    plot2 = plot_three_group_boxplot(
        early_s, late_s, post_b,
        treatment_date=treatment_date,
        animal_id=animal_id,
        merge_gap_ms=merge_gap_ms,
        output_path=three_output_path,
        show=show,
        do_stats=True,
    )

    return {
        "animal_id": animal_id,
        "df_used": df,
        "pre_df": pre_df, "post_df": post_df,
        "pre_seconds": pre_s, "post_seconds": post_s,
        "counts": {"pre_n": int(pre_s.shape[0]), "post_n": int(post_s.shape[0])},
        "early_df": early_df, "late_df": late_df, "post_balanced_df": post_bal_df,
        "early_seconds": early_s, "late_seconds": late_s, "post_balanced_seconds": post_b,
        "counts_three": {"balanced_n": int(n), "early_n": int(early_s.shape[0]),
                         "late_n": int(late_s.shape[0]), "post_n": int(post_b.shape[0])},
        "plots": {"pre_post_saved_path": plot1["saved_path"],
                  "three_group_saved_path": plot2["saved_path"]},
        "three_group_stats": plot2["stats"],
    }


# ───────────── Aggregate plot: per-bird means ± SEM connected ─────────────
def plot_multi_bird_three_group_means(
    summary_df: pd.DataFrame,
    *,
    group_col: str = "Group",
    id_col: str = "Animal ID",
    early_mean_col: str = "EarlyPre_mean_s",
    early_sem_col: str = "EarlyPre_sem_s",
    late_mean_col: str = "LatePre_mean_s",
    late_sem_col: str = "LatePre_sem_s",
    post_mean_col: str = "Post_mean_s",
    post_sem_col: str = "Post_sem_s",
    title: str = "Per-bird song duration mean ± SEM (balanced groups)",
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    label_ids_at_post: bool = True,
    show_animal_legend: bool = True,
    legend_max_animals: int = 40,
    sort_legend_by_group_then_id: bool = True,
    line_alpha: float = 0.65,
    marker_size: float = 6.0,
) -> Dict[str, Any]:
    df = summary_df.copy()

    required = [id_col, group_col,
                early_mean_col, early_sem_col,
                late_mean_col, late_sem_col,
                post_mean_col, post_sem_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"summary_df missing required column: {c}")

    df = df[pd.notna(df[early_mean_col]) & pd.notna(df[late_mean_col]) & pd.notna(df[post_mean_col])].copy()

    x = np.array([1, 2, 3], dtype=float)
    xticklabels = ["Early Pre", "Late Pre", "Post"]

    def color_for_group(g: str) -> str:
        g = str(g).strip().lower()
        if "sham" in g:
            return "tab:blue"
        if "nma" in g:
            return "tab:red"
        return "0.5"

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.subplots_adjust(right=0.72 if show_animal_legend else 0.90)

    if sort_legend_by_group_then_id and not df.empty:
        gkey = df[group_col].astype(str).str.lower().map(
            lambda s: 0 if "sham" in s else (1 if "nma" in s else 2)
        )
        df = df.assign(_gkey=gkey, _id=df[id_col].astype(str))
        df = df.sort_values(["_gkey", "_id"]).drop(columns=["_gkey", "_id"])

    animal_handles: List[Line2D] = []
    n_plotted = 0

    for _, row in df.iterrows():
        aid = str(row[id_col])
        grp = str(row[group_col])

        means = np.array([row[early_mean_col], row[late_mean_col], row[post_mean_col]], dtype=float)
        sems  = np.array([row[early_sem_col],  row[late_sem_col],  row[post_sem_col]],  dtype=float)

        c = color_for_group(grp)

        ax.plot(x, means, linewidth=1.5, alpha=line_alpha, color=c, zorder=2)
        ax.errorbar(
            x, means, yerr=sems,
            fmt="o", linestyle="None",
            markersize=marker_size,
            capsize=3,
            color=c,
            alpha=0.95,
            zorder=3,
        )

        if label_ids_at_post:
            ax.text(3.05, means[-1], aid, fontsize=9, va="center", ha="left", color=c, alpha=0.95)

        if show_animal_legend:
            animal_handles.append(Line2D([0], [0], color=c, linewidth=2.0, marker="o", label=aid))

        n_plotted += 1

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Song duration (s)")
    ax.set_title(title)

    group_handles = [
        Line2D([0], [0], color="tab:blue", marker="o", linestyle="-", label="Sham lesion"),
        Line2D([0], [0], color="tab:red", marker="o", linestyle="-", label="Bilateral NMA lesion"),
        Line2D([0], [0], color="0.5", marker="o", linestyle="-", label="Unknown"),
    ]
    group_leg = ax.legend(handles=group_handles, frameon=False, loc="upper left")
    ax.add_artist(group_leg)

    if show_animal_legend and animal_handles:
        handles_to_show = animal_handles[: int(legend_max_animals)]
        title_txt = "Animal IDs"
        if len(animal_handles) > legend_max_animals:
            title_txt += f" (showing {legend_max_animals}/{len(animal_handles)})"

        ax.legend(
            handles=handles_to_show,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title=title_txt,
            fontsize=9,
            title_fontsize=10,
        )

    _style_ax(ax)

    saved = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved = str(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fig": fig, "ax": ax, "saved_path": saved, "n_birds_plotted": int(n_plotted)}


# ───────────── Aggregated epoch boxplot + stats (across birds; per-bird means) ─────────────
def plot_group_three_epoch_boxplot_from_summary(
    summary_df: pd.DataFrame,
    *,
    group_filter: str,
    group_label: str,
    group_col: str = "Group",
    id_col: str = "Animal ID",
    early_mean_col: str = "EarlyPre_mean_s",
    late_mean_col: str = "LatePre_mean_s",
    post_mean_col: str = "Post_mean_s",
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    jitter: float = 0.06,
    point_size: float = 35.0,
    point_alpha: float = 0.85,
    box_alpha: float = 0.30,
    connect_birds: bool = True,
) -> Dict[str, Any]:
    df = summary_df.copy()
    if group_col not in df.columns:
        raise ValueError(f"summary_df missing '{group_col}' column")
    for c in (id_col, early_mean_col, late_mean_col, post_mean_col):
        if c not in df.columns:
            raise ValueError(f"summary_df missing required column: {c}")

    gf = str(group_filter).strip().lower()
    gser = df[group_col].astype(str).str.lower()
    df = df[gser.str.contains(gf, na=False)].copy()

    df[early_mean_col] = pd.to_numeric(df[early_mean_col], errors="coerce")
    df[late_mean_col]  = pd.to_numeric(df[late_mean_col],  errors="coerce")
    df[post_mean_col]  = pd.to_numeric(df[post_mean_col],  errors="coerce")
    df = df[pd.notna(df[early_mean_col]) & pd.notna(df[late_mean_col]) & pd.notna(df[post_mean_col])].copy()

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    if df.empty:
        ax.text(0.5, 0.5, f"No birds found for group: {group_label}", ha="center", va="center", fontsize=12)
        ax.axis("off")
        saved = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            saved = str(output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {"fig": fig, "ax": ax, "saved_path": saved, "stats": {"kw": None, "pairwise": []}, "n_birds": 0}

    x = np.array([1, 2, 3], dtype=float)
    xticklabels = ["Early Pre", "Late Pre", "Post"]

    early = df[early_mean_col].values.astype(float)
    late  = df[late_mean_col].values.astype(float)
    post  = df[post_mean_col].values.astype(float)

    bp = ax.boxplot(
        [early, late, post],
        positions=x,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
        boxprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color="black", linewidth=1.1),
        capprops=dict(color="black", linewidth=1.1),
        medianprops=dict(color="black", linewidth=2.0, linestyle="-"),
        meanprops=dict(color="black", linewidth=2.0, linestyle="--"),
        zorder=1,
    )
    for b in bp["boxes"]:
        b.set_facecolor("white")
        b.set_alpha(box_alpha)

    rng = np.random.default_rng(0)
    for _, row in df.iterrows():
        m = np.array([row[early_mean_col], row[late_mean_col], row[post_mean_col]], dtype=float)
        xj = x + rng.uniform(-jitter, jitter, size=3)

        if connect_birds:
            ax.plot(x, m, linewidth=1.2, alpha=0.55, color="0.35", zorder=2)

        ax.scatter(xj, m, s=point_size, alpha=point_alpha, edgecolors="none", color="0.1", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Song duration (s)")
    t = title or f"{group_label}: per-bird mean song duration across epochs\n(boxplots show distribution across birds)"
    ax.set_title(t)

    _style_ax(ax)

    stats_out: Dict[str, Any] = {"kw": None, "pairwise": []}

    arrays = [early, late, post]
    labs = ["Early Pre", "Late Pre", "Post"]

    H, p_kw = _try_kruskal([np.asarray(a, float) for a in arrays])
    stats_out["kw"] = {"groups": labs, "H": H, "p": p_kw}

    from itertools import combinations
    pairs_idx = list(combinations(range(3), 2))
    raw_p = []
    pair_stats = []
    for i, j in pairs_idx:
        stat, p = _try_mannwhitney(arrays[i], arrays[j])
        raw_p.append(p)
        pair_stats.append({"a": labs[i], "b": labs[j], "stat": stat, "p_raw": p})

    adj = _p_adjust_bh(raw_p)
    for k, d in enumerate(pair_stats):
        d["p_bh"] = adj[k]
        d["stars"] = _stars(adj[k])
    stats_out["pairwise"] = pair_stats

    summary_lines = [f"Kruskal–Wallis: H={H:.3g}, p={_fmt_p(p_kw)}"]
    for d in pair_stats:
        summary_lines.append(f"{d['a']} vs {d['b']}: p(BH)={_fmt_p(d['p_bh'])}  [{d['stars']}]")
    fig.text(
        0.74, 0.98, "\n".join(summary_lines),
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95),
    )

    data_max = float(np.nanmax(np.concatenate([early, late, post])))
    _add_sig_brackets(ax, [(1, 2), (1, 3), (2, 3)], adj, data_max)

    saved = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved = str(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fig": fig, "ax": ax, "saved_path": saved, "stats": stats_out, "n_birds": int(len(df))}


# ───────────── Delta plots (mean): absolute seconds OR percent change ─────────────
def plot_group_mean_delta_boxplot_from_summary(
    summary_df: pd.DataFrame,
    *,
    group_filter: str,
    group_label: str,
    mode: str = "absolute",  # "absolute" or "percent"
    group_col: str = "Group",
    id_col: str = "Animal ID",
    early_mean_col: str = "EarlyPre_mean_s",
    late_mean_col: str = "LatePre_mean_s",
    post_mean_col: str = "Post_mean_s",
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    jitter: float = 0.06,
    point_size: float = 38.0,
    point_alpha: float = 0.85,
    box_alpha: float = 0.30,
    connect_birds: bool = False,
    do_between_delta_stats: bool = True,
) -> Dict[str, Any]:
    mode = str(mode).strip().lower()
    if mode not in {"absolute", "percent"}:
        raise ValueError("mode must be 'absolute' or 'percent'")

    df = summary_df.copy()
    for c in (group_col, id_col, early_mean_col, late_mean_col, post_mean_col):
        if c not in df.columns:
            raise ValueError(f"summary_df missing required column: {c}")

    gf = str(group_filter).strip().lower()
    gser = df[group_col].astype(str).str.lower()
    df = df[gser.str.contains(gf, na=False)].copy()

    df[early_mean_col] = pd.to_numeric(df[early_mean_col], errors="coerce")
    df[late_mean_col]  = pd.to_numeric(df[late_mean_col],  errors="coerce")
    df[post_mean_col]  = pd.to_numeric(df[post_mean_col],  errors="coerce")
    df = df[pd.notna(df[early_mean_col]) & pd.notna(df[late_mean_col]) & pd.notna(df[post_mean_col])].copy()

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    if df.empty:
        ax.text(0.5, 0.5, f"No birds found for group: {group_label}", ha="center", va="center", fontsize=12)
        ax.axis("off")
        saved = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            saved = str(output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {"fig": fig, "ax": ax, "saved_path": saved, "stats": None, "n_birds": 0}

    early = df[early_mean_col].values.astype(float)
    late  = df[late_mean_col].values.astype(float)
    post  = df[post_mean_col].values.astype(float)

    if mode == "absolute":
        d1 = late - early
        d2 = post - late
        ylabel = "Δ mean song duration (s)"
        xticklabels = ["Δ mean: Late − Early", "Δ mean: Post − Late"]
        default_title = f"{group_label}: Δ(mean song duration) across birds (seconds)"
    else:
        d1 = np.where(early > 0, 100.0 * (late - early) / early, np.nan)
        d2 = np.where(late > 0,  100.0 * (post - late) / late,  np.nan)
        ylabel = "Percent change in mean song duration (%)"
        xticklabels = ["100*(Late−Early)/Early", "100*(Post−Late)/Late"]
        default_title = f"{group_label}: percent change in mean song duration across birds"

    mask = np.isfinite(d1) & np.isfinite(d2)
    d1 = d1[mask]
    d2 = d2[mask]
    df_used = df.loc[mask].copy()

    if d1.size == 0:
        ax.text(0.5, 0.5, f"No valid delta data for group: {group_label} ({mode})", ha="center", va="center", fontsize=12)
        ax.axis("off")
        saved = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            saved = str(output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {"fig": fig, "ax": ax, "saved_path": saved, "stats": None, "n_birds": 0}

    x = np.array([1, 2], dtype=float)

    bp = ax.boxplot(
        [d1, d2],
        positions=x,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
        boxprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color="black", linewidth=1.1),
        capprops=dict(color="black", linewidth=1.1),
        medianprops=dict(color="black", linewidth=2.0, linestyle="-"),
        meanprops=dict(color="black", linewidth=2.0, linestyle="--"),
        zorder=1,
    )
    for b in bp["boxes"]:
        b.set_facecolor("white")
        b.set_alpha(box_alpha)

    rng = np.random.default_rng(0)
    for _, row in df_used.iterrows():
        e = float(row[early_mean_col])
        l = float(row[late_mean_col])
        p = float(row[post_mean_col])

        if mode == "absolute":
            y = np.array([l - e, p - l], dtype=float)
        else:
            y = np.array(
                [100.0 * (l - e) / e if e > 0 else np.nan,
                 100.0 * (p - l) / l if l > 0 else np.nan],
                dtype=float
            )
        if not np.all(np.isfinite(y)):
            continue

        xj = x + rng.uniform(-jitter, jitter, size=2)
        if connect_birds:
            ax.plot(x, y, linewidth=1.1, alpha=0.45, color="0.35", zorder=2)
        ax.scatter(xj, y, s=point_size, alpha=point_alpha, edgecolors="none", color="0.1", zorder=3)

    ax.axhline(0.0, linewidth=1.2, linestyle="--", color="black", alpha=0.6, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title or default_title)
    _style_ax(ax)

    stats_out = {"n_birds": int(d1.size), "mode": mode, "between_deltas": None}

    if do_between_delta_stats:
        stat, pval, test_name = _try_paired_test(d1, d2)
        stats_out["between_deltas"] = {"test": test_name, "stat": stat, "p": pval}

        data_max = float(np.nanmax(np.concatenate([d1, d2])))
        _add_sig_brackets(ax, [(1, 2)], [pval], data_max)

        fig.text(
            0.70, 0.98,
            f"{test_name} (paired): {xticklabels[0]} vs {xticklabels[1]}\n"
            f"p={_fmt_p(pval)}  [{_stars(pval)}]   n birds={int(d1.size)}",
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95),
        )

    saved = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved = str(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fig": fig, "ax": ax, "saved_path": saved, "stats": stats_out, "n_birds": int(d1.size)}


# ───────────── Delta plots (variance): absolute s^2 OR percent change ─────────────
def plot_group_variance_delta_boxplot_from_summary(
    summary_df: pd.DataFrame,
    *,
    group_filter: str,
    group_label: str,
    mode: str = "absolute",  # "absolute" or "percent"
    group_col: str = "Group",
    id_col: str = "Animal ID",
    early_var_col: str = "EarlyPre_var_s2",
    late_var_col: str = "LatePre_var_s2",
    post_var_col: str = "Post_var_s2",
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    jitter: float = 0.06,
    point_size: float = 38.0,
    point_alpha: float = 0.85,
    box_alpha: float = 0.30,
    connect_birds: bool = False,
    do_between_delta_stats: bool = True,
) -> Dict[str, Any]:
    mode = str(mode).strip().lower()
    if mode not in {"absolute", "percent"}:
        raise ValueError("mode must be 'absolute' or 'percent'")

    df = summary_df.copy()
    for c in (group_col, id_col, early_var_col, late_var_col, post_var_col):
        if c not in df.columns:
            raise ValueError(f"summary_df missing required column: {c}")

    gf = str(group_filter).strip().lower()
    gser = df[group_col].astype(str).str.lower()
    df = df[gser.str.contains(gf, na=False)].copy()

    df[early_var_col] = pd.to_numeric(df[early_var_col], errors="coerce")
    df[late_var_col]  = pd.to_numeric(df[late_var_col],  errors="coerce")
    df[post_var_col]  = pd.to_numeric(df[post_var_col],  errors="coerce")

    df = df[pd.notna(df[early_var_col]) & pd.notna(df[late_var_col]) & pd.notna(df[post_var_col])].copy()

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    if df.empty:
        ax.text(0.5, 0.5, f"No birds found for group: {group_label}", ha="center", va="center", fontsize=12)
        ax.axis("off")
        saved = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            saved = str(output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {"fig": fig, "ax": ax, "saved_path": saved, "stats": None, "n_birds": 0}

    early = df[early_var_col].values.astype(float)
    late  = df[late_var_col].values.astype(float)
    post  = df[post_var_col].values.astype(float)

    if mode == "absolute":
        d1 = late - early
        d2 = post - late
        ylabel = "Δ variance of song duration (s²)"
        xticklabels = ["Δ var: Late − Early", "Δ var: Post − Late"]
        default_title = f"{group_label}: Δ(variance of song duration) across birds (s²)"
    else:
        d1 = np.where(early > 0, 100.0 * (late - early) / early, np.nan)
        d2 = np.where(late > 0,  100.0 * (post - late) / late,  np.nan)
        ylabel = "Percent change in variance (%)"
        xticklabels = ["100*(Late−Early)/Early (var)", "100*(Post−Late)/Late (var)"]
        default_title = f"{group_label}: percent change in variance of song duration across birds"

    mask = np.isfinite(d1) & np.isfinite(d2)
    d1 = d1[mask]
    d2 = d2[mask]
    df_used = df.loc[mask].copy()

    if d1.size == 0:
        ax.text(0.5, 0.5, f"No valid variance-delta data for group: {group_label} ({mode})",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        saved = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            saved = str(output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {"fig": fig, "ax": ax, "saved_path": saved, "stats": None, "n_birds": 0}

    x = np.array([1, 2], dtype=float)

    bp = ax.boxplot(
        [d1, d2],
        positions=x,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
        boxprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color="black", linewidth=1.1),
        capprops=dict(color="black", linewidth=1.1),
        medianprops=dict(color="black", linewidth=2.0, linestyle="-"),
        meanprops=dict(color="black", linewidth=2.0, linestyle="--"),
        zorder=1,
    )
    for b in bp["boxes"]:
        b.set_facecolor("white")
        b.set_alpha(box_alpha)

    rng = np.random.default_rng(0)
    for _, row in df_used.iterrows():
        e = float(row[early_var_col])
        l = float(row[late_var_col])
        p = float(row[post_var_col])

        if mode == "absolute":
            y = np.array([l - e, p - l], dtype=float)
        else:
            y = np.array(
                [100.0 * (l - e) / e if e > 0 else np.nan,
                 100.0 * (p - l) / l if l > 0 else np.nan],
                dtype=float
            )
        if not np.all(np.isfinite(y)):
            continue

        xj = x + rng.uniform(-jitter, jitter, size=2)
        if connect_birds:
            ax.plot(x, y, linewidth=1.1, alpha=0.45, color="0.35", zorder=2)
        ax.scatter(xj, y, s=point_size, alpha=point_alpha, edgecolors="none", color="0.1", zorder=3)

    ax.axhline(0.0, linewidth=1.2, linestyle="--", color="black", alpha=0.6, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title or default_title)
    _style_ax(ax)

    stats_out = {"n_birds": int(d1.size), "mode": mode, "between_deltas": None}

    if do_between_delta_stats:
        stat, pval, test_name = _try_paired_test(d1, d2)
        stats_out["between_deltas"] = {"test": test_name, "stat": stat, "p": pval}

        data_max = float(np.nanmax(np.concatenate([d1, d2])))
        _add_sig_brackets(ax, [(1, 2)], [pval], data_max)

        fig.text(
            0.70, 0.98,
            f"{test_name} (paired): {xticklabels[0]} vs {xticklabels[1]}\n"
            f"p={_fmt_p(pval)}  [{_stars(pval)}]   n birds={int(d1.size)}",
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95),
        )

    saved = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved = str(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fig": fig, "ax": ax, "saved_path": saved, "stats": stats_out, "n_birds": int(d1.size)}


# ───────────── multi-bird helpers ─────────────
def _find_song_detection_jsons(root_dir: Union[str, Path]) -> Dict[str, Path]:
    root = Path(root_dir)
    files = sorted(root.rglob("*_song_detection.json"))

    out: Dict[str, Path] = {}
    for f in files:
        if f.name.startswith("._") or f.name.startswith("."):
            continue
        if any(part == "__MACOSX" for part in f.parts):
            continue
        aid = _infer_animal_id_from_song_detection_json(f)
        out.setdefault(aid, f)
    return out


def run_song_duration_comparison_multi(
    root_dir: Union[str, Path],
    *,
    metadata_excel: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    id_col: Optional[str] = None,
    treatment_date_col: Optional[str] = None,
    group_col: Optional[str] = None,
    merge_gap_ms: Optional[int] = None,
    songs_only: bool = True,
    output_root: Optional[Union[str, Path]] = None,
    show: bool = False,
    include_ids: Optional[Iterable[str]] = None,
    exclude_ids: Optional[Iterable[str]] = None,
    save_summary_csv: bool = True,
    make_aggregate_plot: bool = True,
    aggregate_plot_path: Optional[Union[str, Path]] = None,
    make_group_boxplots: bool = True,
    make_delta_plots_absolute: bool = True,
    make_delta_plots_percent: bool = True,
    make_variance_delta_plots_absolute: bool = True,
    make_variance_delta_plots_percent: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    root = Path(root_dir)
    outroot = Path(output_root) if output_root else None

    meta_raw = pd.read_excel(Path(metadata_excel), sheet_name=sheet_name)

    # infer treatment columns if needed
    if id_col is None or treatment_date_col is None:
        guessed_id, guessed_date = _guess_columns_for_treatment(meta_raw)
        id_col_use = id_col or guessed_id
        date_col_use = treatment_date_col or guessed_date
    else:
        id_col_use = id_col
        date_col_use = treatment_date_col

    group_col_use = group_col or _guess_group_column(meta_raw)

    meta_norm = meta_raw.copy()
    meta_norm[id_col_use] = meta_norm[id_col_use].apply(_normalize_animal_id)
    meta_norm["_treat_dt"] = pd.to_datetime(meta_norm[date_col_use], errors="coerce").dt.normalize()

    song_jsons = _find_song_detection_jsons(root)

    if include_ids is not None:
        include_set = {_normalize_animal_id(x) for x in include_ids}
        song_jsons = {k: v for k, v in song_jsons.items() if _normalize_animal_id(k) in include_set}
    if exclude_ids is not None:
        exclude_set = {_normalize_animal_id(x) for x in exclude_ids}
        song_jsons = {k: v for k, v in song_jsons.items() if _normalize_animal_id(k) not in exclude_set}

    results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for animal_id, json_path in sorted(song_jsons.items()):
        animal_id_norm = _normalize_animal_id(animal_id)

        td = _lookup_treatment_date_for_animal(
            meta_norm,
            animal_id=animal_id_norm,
            id_col=id_col_use,
            treat_dt_col_norm="_treat_dt",
        )

        if td is None or pd.isna(td):
            if verbose:
                print(f"[SKIP] {animal_id_norm}: no treatment date found in metadata (matched on '{id_col_use}').")
            continue

        grp = _lookup_group_for_animal(
            meta_norm,
            animal_id=animal_id_norm,
            id_col=id_col_use,
            group_col=group_col_use,
        )

        # Output directory
        if outroot is None:
            bird_out_dir = json_path.parent / "figures" / "song_duration_comparison"
        else:
            bird_out_dir = outroot / animal_id_norm

        out1 = bird_out_dir / f"{animal_id_norm}_song_durations_pre_vs_post.png"
        out2 = bird_out_dir / f"{animal_id_norm}_song_durations_three_group_balanced.png"

        if verbose:
            print(f"[RUN] {animal_id_norm}")
            print(f"      json: {json_path}")
            print(f"      treatment_date used: {td.date()}")
            if group_col_use:
                print(f"      group ({group_col_use}): {grp}")
            else:
                print("      group: Unknown (no group-like column found)")
            print(f"      out: {bird_out_dir}")

        try:
            res = run_song_duration_comparison(
                json_input=json_path,
                treatment_date=td,
                animal_id=animal_id_norm,
                merge_gap_ms=merge_gap_ms,
                songs_only=songs_only,
                output_path=out1,
                three_output_path=out2,
                show=show,
            )
            results[animal_id_norm] = res

            # Per-bird mean±SEM for Early/Late/Post (balanced)
            e_mu, e_sem, e_n = _mean_sem(res["early_seconds"].values)
            l_mu, l_sem, l_n = _mean_sem(res["late_seconds"].values)
            p_mu, p_sem, p_n = _mean_sem(res["post_balanced_seconds"].values)

            # Per-bird variance for Early/Late/Post (balanced)
            e_var, _ = _var_s2(res["early_seconds"].values)
            l_var, _ = _var_s2(res["late_seconds"].values)
            p_var, _ = _var_s2(res["post_balanced_seconds"].values)

            kw = res.get("three_group_stats", {}).get("kw", None)
            kw_p = kw.get("p", np.nan) if isinstance(kw, dict) else np.nan

            # Pull BH-adjusted pairwise p-values into columns (if present)
            pairwise = res.get("three_group_stats", {}).get("pairwise", []) or []
            pw_map = {}
            for d in pairwise:
                a = d.get("a", "")
                b = d.get("b", "")
                key = f"pBH_{a.replace(' ', '')}_vs_{b.replace(' ', '')}"
                pw_map[key] = d.get("p_bh", np.nan)

            summary_rows.append({
                "Animal ID": animal_id_norm,
                "Group": grp,
                "Song JSON": str(json_path),
                "Treatment date": str(td.date()),

                "Pre_n": res["counts"]["pre_n"],
                "Post_n": res["counts"]["post_n"],
                "Balanced_n": res["counts_three"]["balanced_n"],
                "KW_p": kw_p,

                "EarlyPre_n": e_n, "EarlyPre_mean_s": e_mu, "EarlyPre_sem_s": e_sem, "EarlyPre_var_s2": e_var,
                "LatePre_n": l_n, "LatePre_mean_s": l_mu, "LatePre_sem_s": l_sem, "LatePre_var_s2": l_var,
                "Post_n_bal": p_n, "Post_mean_s": p_mu, "Post_sem_s": p_sem, "Post_var_s2": p_var,

                "PrePost_plot": res["plots"]["pre_post_saved_path"],
                "ThreeGroup_plot": res["plots"]["three_group_saved_path"],
                **pw_map,
            })

        except Exception as e:
            if verbose:
                print(f"[ERROR] {animal_id_norm}: {type(e).__name__}: {e}")

    summary_df = pd.DataFrame(summary_rows)

    # Write summary CSV
    summary_path = None
    if save_summary_csv:
        summary_path = (outroot / "song_duration_comparison_summary.csv") if outroot else (root / "song_duration_comparison_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"[OK] Wrote summary CSV: {summary_path}")

    # Aggregate plot (spaghetti)
    agg_plot_saved = None
    agg_plot_n = 0
    if make_aggregate_plot and not summary_df.empty:
        if aggregate_plot_path is not None:
            agg_path = Path(aggregate_plot_path)
        else:
            agg_path = (outroot / "song_duration_three_group_means_by_bird.png") if outroot else (root / "song_duration_three_group_means_by_bird.png")

        outp = plot_multi_bird_three_group_means(
            summary_df,
            output_path=agg_path,
            show=show,
            label_ids_at_post=True,
            show_animal_legend=True,
            legend_max_animals=60,
            title="Per-bird song duration mean ± SEM (Early Pre / Late Pre / Post)",
        )
        agg_plot_saved = outp["saved_path"]
        agg_plot_n = outp["n_birds_plotted"]
        if verbose and agg_plot_saved:
            print(f"[OK] Wrote aggregate plot: {agg_plot_saved}  (birds plotted: {agg_plot_n})")

    # ── Aggregated epoch boxplots (TWO versions: CONNECTED and NOLINES) ──
    sham_epoch_connected = None
    sham_epoch_nolines = None
    nma_epoch_connected = None
    nma_epoch_nolines = None
    sham_epoch_stats_connected = None
    sham_epoch_stats_nolines = None
    nma_epoch_stats_connected = None
    nma_epoch_stats_nolines = None

    if make_group_boxplots and not summary_df.empty:
        base = outroot if outroot else root

        sham_epoch_connected = base / "song_duration_means_boxplot_SHAM_CONNECTED.png"
        sham_epoch_nolines   = base / "song_duration_means_boxplot_SHAM_NOLINES.png"
        nma_epoch_connected  = base / "song_duration_means_boxplot_BILATERAL_NMA_CONNECTED.png"
        nma_epoch_nolines    = base / "song_duration_means_boxplot_BILATERAL_NMA_NOLINES.png"

        outA = plot_group_three_epoch_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            output_path=sham_epoch_connected,
            show=show,
            connect_birds=True,
        )
        sham_epoch_stats_connected = outA["stats"]
        if verbose and outA.get("saved_path"):
            print(f"[OK] Wrote sham epoch boxplot (CONNECTED): {outA['saved_path']}  (birds: {outA['n_birds']})")

        outB = plot_group_three_epoch_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            output_path=sham_epoch_nolines,
            show=show,
            connect_birds=False,
        )
        sham_epoch_stats_nolines = outB["stats"]
        if verbose and outB.get("saved_path"):
            print(f"[OK] Wrote sham epoch boxplot (NOLINES): {outB['saved_path']}  (birds: {outB['n_birds']})")

        outC = plot_group_three_epoch_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            output_path=nma_epoch_connected,
            show=show,
            connect_birds=True,
        )
        nma_epoch_stats_connected = outC["stats"]
        if verbose and outC.get("saved_path"):
            print(f"[OK] Wrote NMA epoch boxplot (CONNECTED): {outC['saved_path']}  (birds: {outC['n_birds']})")

        outD = plot_group_three_epoch_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            output_path=nma_epoch_nolines,
            show=show,
            connect_birds=False,
        )
        nma_epoch_stats_nolines = outD["stats"]
        if verbose and outD.get("saved_path"):
            print(f"[OK] Wrote NMA epoch boxplot (NOLINES): {outD['saved_path']}  (birds: {outD['n_birds']})")

    # ── Mean delta plots: seconds (TWO versions) ──
    sham_mean_delta_seconds_connected = None
    sham_mean_delta_seconds_nolines = None
    nma_mean_delta_seconds_connected = None
    nma_mean_delta_seconds_nolines = None
    sham_mean_delta_seconds_stats_connected = None
    sham_mean_delta_seconds_stats_nolines = None
    nma_mean_delta_seconds_stats_connected = None
    nma_mean_delta_seconds_stats_nolines = None

    if make_delta_plots_absolute and not summary_df.empty:
        base = outroot if outroot else root

        sham_mean_delta_seconds_connected = base / "song_duration_mean_deltas_SECONDS_SHAM_CONNECTED.png"
        sham_mean_delta_seconds_nolines   = base / "song_duration_mean_deltas_SECONDS_SHAM_NOLINES.png"
        nma_mean_delta_seconds_connected  = base / "song_duration_mean_deltas_SECONDS_BILATERAL_NMA_CONNECTED.png"
        nma_mean_delta_seconds_nolines    = base / "song_duration_mean_deltas_SECONDS_BILATERAL_NMA_NOLINES.png"

        outA = plot_group_mean_delta_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            mode="absolute",
            output_path=sham_mean_delta_seconds_connected,
            show=show,
            connect_birds=True,
            do_between_delta_stats=True,
        )
        sham_mean_delta_seconds_stats_connected = outA["stats"]
        if verbose and outA.get("saved_path"):
            print(f"[OK] Wrote sham mean delta (seconds) (CONNECTED): {outA['saved_path']} (birds: {outA['n_birds']})")

        outB = plot_group_mean_delta_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            mode="absolute",
            output_path=sham_mean_delta_seconds_nolines,
            show=show,
            connect_birds=False,
            do_between_delta_stats=True,
        )
        sham_mean_delta_seconds_stats_nolines = outB["stats"]
        if verbose and outB.get("saved_path"):
            print(f"[OK] Wrote sham mean delta (seconds) (NOLINES): {outB['saved_path']} (birds: {outB['n_birds']})")

        outC = plot_group_mean_delta_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            mode="absolute",
            output_path=nma_mean_delta_seconds_connected,
            show=show,
            connect_birds=True,
            do_between_delta_stats=True,
        )
        nma_mean_delta_seconds_stats_connected = outC["stats"]
        if verbose and outC.get("saved_path"):
            print(f"[OK] Wrote NMA mean delta (seconds) (CONNECTED): {outC['saved_path']} (birds: {outC['n_birds']})")

        outD = plot_group_mean_delta_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            mode="absolute",
            output_path=nma_mean_delta_seconds_nolines,
            show=show,
            connect_birds=False,
            do_between_delta_stats=True,
        )
        nma_mean_delta_seconds_stats_nolines = outD["stats"]
        if verbose and outD.get("saved_path"):
            print(f"[OK] Wrote NMA mean delta (seconds) (NOLINES): {outD['saved_path']} (birds: {outD['n_birds']})")

    # ── Mean delta plots: percent (TWO versions) ──
    sham_mean_delta_percent_connected = None
    sham_mean_delta_percent_nolines = None
    nma_mean_delta_percent_connected = None
    nma_mean_delta_percent_nolines = None
    sham_mean_delta_percent_stats_connected = None
    sham_mean_delta_percent_stats_nolines = None
    nma_mean_delta_percent_stats_connected = None
    nma_mean_delta_percent_stats_nolines = None

    if make_delta_plots_percent and not summary_df.empty:
        base = outroot if outroot else root

        sham_mean_delta_percent_connected = base / "song_duration_mean_deltas_PERCENT_SHAM_CONNECTED.png"
        sham_mean_delta_percent_nolines   = base / "song_duration_mean_deltas_PERCENT_SHAM_NOLINES.png"
        nma_mean_delta_percent_connected  = base / "song_duration_mean_deltas_PERCENT_BILATERAL_NMA_CONNECTED.png"
        nma_mean_delta_percent_nolines    = base / "song_duration_mean_deltas_PERCENT_BILATERAL_NMA_NOLINES.png"

        outA = plot_group_mean_delta_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            mode="percent",
            output_path=sham_mean_delta_percent_connected,
            show=show,
            connect_birds=True,
            do_between_delta_stats=True,
        )
        sham_mean_delta_percent_stats_connected = outA["stats"]
        if verbose and outA.get("saved_path"):
            print(f"[OK] Wrote sham mean delta (percent) (CONNECTED): {outA['saved_path']} (birds: {outA['n_birds']})")

        outB = plot_group_mean_delta_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            mode="percent",
            output_path=sham_mean_delta_percent_nolines,
            show=show,
            connect_birds=False,
            do_between_delta_stats=True,
        )
        sham_mean_delta_percent_stats_nolines = outB["stats"]
        if verbose and outB.get("saved_path"):
            print(f"[OK] Wrote sham mean delta (percent) (NOLINES): {outB['saved_path']} (birds: {outB['n_birds']})")

        outC = plot_group_mean_delta_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            mode="percent",
            output_path=nma_mean_delta_percent_connected,
            show=show,
            connect_birds=True,
            do_between_delta_stats=True,
        )
        nma_mean_delta_percent_stats_connected = outC["stats"]
        if verbose and outC.get("saved_path"):
            print(f"[OK] Wrote NMA mean delta (percent) (CONNECTED): {outC['saved_path']} (birds: {outC['n_birds']})")

        outD = plot_group_mean_delta_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            mode="percent",
            output_path=nma_mean_delta_percent_nolines,
            show=show,
            connect_birds=False,
            do_between_delta_stats=True,
        )
        nma_mean_delta_percent_stats_nolines = outD["stats"]
        if verbose and outD.get("saved_path"):
            print(f"[OK] Wrote NMA mean delta (percent) (NOLINES): {outD['saved_path']} (birds: {outD['n_birds']})")

    # ── Variance delta plots: s^2 (TWO versions) ──
    sham_var_delta_s2_connected = None
    sham_var_delta_s2_nolines = None
    nma_var_delta_s2_connected = None
    nma_var_delta_s2_nolines = None
    sham_var_delta_s2_stats_connected = None
    sham_var_delta_s2_stats_nolines = None
    nma_var_delta_s2_stats_connected = None
    nma_var_delta_s2_stats_nolines = None

    if make_variance_delta_plots_absolute and not summary_df.empty:
        base = outroot if outroot else root

        sham_var_delta_s2_connected = base / "song_duration_VARIANCE_deltas_S2_SHAM_CONNECTED.png"
        sham_var_delta_s2_nolines   = base / "song_duration_VARIANCE_deltas_S2_SHAM_NOLINES.png"
        nma_var_delta_s2_connected  = base / "song_duration_VARIANCE_deltas_S2_BILATERAL_NMA_CONNECTED.png"
        nma_var_delta_s2_nolines    = base / "song_duration_VARIANCE_deltas_S2_BILATERAL_NMA_NOLINES.png"

        outA = plot_group_variance_delta_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            mode="absolute",
            output_path=sham_var_delta_s2_connected,
            show=show,
            connect_birds=True,
            do_between_delta_stats=True,
        )
        sham_var_delta_s2_stats_connected = outA["stats"]
        if verbose and outA.get("saved_path"):
            print(f"[OK] Wrote sham variance delta (s^2) (CONNECTED): {outA['saved_path']} (birds: {outA['n_birds']})")

        outB = plot_group_variance_delta_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            mode="absolute",
            output_path=sham_var_delta_s2_nolines,
            show=show,
            connect_birds=False,
            do_between_delta_stats=True,
        )
        sham_var_delta_s2_stats_nolines = outB["stats"]
        if verbose and outB.get("saved_path"):
            print(f"[OK] Wrote sham variance delta (s^2) (NOLINES): {outB['saved_path']} (birds: {outB['n_birds']})")

        outC = plot_group_variance_delta_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            mode="absolute",
            output_path=nma_var_delta_s2_connected,
            show=show,
            connect_birds=True,
            do_between_delta_stats=True,
        )
        nma_var_delta_s2_stats_connected = outC["stats"]
        if verbose and outC.get("saved_path"):
            print(f"[OK] Wrote NMA variance delta (s^2) (CONNECTED): {outC['saved_path']} (birds: {outC['n_birds']})")

        outD = plot_group_variance_delta_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            mode="absolute",
            output_path=nma_var_delta_s2_nolines,
            show=show,
            connect_birds=False,
            do_between_delta_stats=True,
        )
        nma_var_delta_s2_stats_nolines = outD["stats"]
        if verbose and outD.get("saved_path"):
            print(f"[OK] Wrote NMA variance delta (s^2) (NOLINES): {outD['saved_path']} (birds: {outD['n_birds']})")

    # ── Variance delta plots: percent (TWO versions) ──
    sham_var_delta_percent_connected = None
    sham_var_delta_percent_nolines = None
    nma_var_delta_percent_connected = None
    nma_var_delta_percent_nolines = None
    sham_var_delta_percent_stats_connected = None
    sham_var_delta_percent_stats_nolines = None
    nma_var_delta_percent_stats_connected = None
    nma_var_delta_percent_stats_nolines = None

    if make_variance_delta_plots_percent and not summary_df.empty:
        base = outroot if outroot else root

        sham_var_delta_percent_connected = base / "song_duration_VARIANCE_deltas_PERCENT_SHAM_CONNECTED.png"
        sham_var_delta_percent_nolines   = base / "song_duration_VARIANCE_deltas_PERCENT_SHAM_NOLINES.png"
        nma_var_delta_percent_connected  = base / "song_duration_VARIANCE_deltas_PERCENT_BILATERAL_NMA_CONNECTED.png"
        nma_var_delta_percent_nolines    = base / "song_duration_VARIANCE_deltas_PERCENT_BILATERAL_NMA_NOLINES.png"

        outA = plot_group_variance_delta_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            mode="percent",
            output_path=sham_var_delta_percent_connected,
            show=show,
            connect_birds=True,
            do_between_delta_stats=True,
        )
        sham_var_delta_percent_stats_connected = outA["stats"]
        if verbose and outA.get("saved_path"):
            print(f"[OK] Wrote sham variance delta (percent) (CONNECTED): {outA['saved_path']} (birds: {outA['n_birds']})")

        outB = plot_group_variance_delta_boxplot_from_summary(
            summary_df,
            group_filter="sham",
            group_label="Sham lesion",
            mode="percent",
            output_path=sham_var_delta_percent_nolines,
            show=show,
            connect_birds=False,
            do_between_delta_stats=True,
        )
        sham_var_delta_percent_stats_nolines = outB["stats"]
        if verbose and outB.get("saved_path"):
            print(f"[OK] Wrote sham variance delta (percent) (NOLINES): {outB['saved_path']} (birds: {outB['n_birds']})")

        outC = plot_group_variance_delta_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            mode="percent",
            output_path=nma_var_delta_percent_connected,
            show=show,
            connect_birds=True,
            do_between_delta_stats=True,
        )
        nma_var_delta_percent_stats_connected = outC["stats"]
        if verbose and outC.get("saved_path"):
            print(f"[OK] Wrote NMA variance delta (percent) (CONNECTED): {outC['saved_path']} (birds: {outC['n_birds']})")

        outD = plot_group_variance_delta_boxplot_from_summary(
            summary_df,
            group_filter="nma",
            group_label="Bilateral NMA lesion",
            mode="percent",
            output_path=nma_var_delta_percent_nolines,
            show=show,
            connect_birds=False,
            do_between_delta_stats=True,
        )
        nma_var_delta_percent_stats_nolines = outD["stats"]
        if verbose and outD.get("saved_path"):
            print(f"[OK] Wrote NMA variance delta (percent) (NOLINES): {outD['saved_path']} (birds: {outD['n_birds']})")

    return {
        "results_by_animal": results,
        "summary_df": summary_df,
        "summary_csv": str(summary_path) if summary_path else None,

        "aggregate_plot": agg_plot_saved,
        "aggregate_plot_n_birds": int(agg_plot_n),

        # epoch boxplots (two versions)
        "sham_epoch_boxplot_connected": str(sham_epoch_connected) if sham_epoch_connected else None,
        "sham_epoch_boxplot_nolines": str(sham_epoch_nolines) if sham_epoch_nolines else None,
        "nma_epoch_boxplot_connected": str(nma_epoch_connected) if nma_epoch_connected else None,
        "nma_epoch_boxplot_nolines": str(nma_epoch_nolines) if nma_epoch_nolines else None,
        "sham_epoch_boxplot_stats_connected": sham_epoch_stats_connected,
        "sham_epoch_boxplot_stats_nolines": sham_epoch_stats_nolines,
        "nma_epoch_boxplot_stats_connected": nma_epoch_stats_connected,
        "nma_epoch_boxplot_stats_nolines": nma_epoch_stats_nolines,

        # mean delta seconds (two versions)
        "sham_mean_delta_seconds_connected": str(sham_mean_delta_seconds_connected) if sham_mean_delta_seconds_connected else None,
        "sham_mean_delta_seconds_nolines": str(sham_mean_delta_seconds_nolines) if sham_mean_delta_seconds_nolines else None,
        "nma_mean_delta_seconds_connected": str(nma_mean_delta_seconds_connected) if nma_mean_delta_seconds_connected else None,
        "nma_mean_delta_seconds_nolines": str(nma_mean_delta_seconds_nolines) if nma_mean_delta_seconds_nolines else None,
        "sham_mean_delta_seconds_stats_connected": sham_mean_delta_seconds_stats_connected,
        "sham_mean_delta_seconds_stats_nolines": sham_mean_delta_seconds_stats_nolines,
        "nma_mean_delta_seconds_stats_connected": nma_mean_delta_seconds_stats_connected,
        "nma_mean_delta_seconds_stats_nolines": nma_mean_delta_seconds_stats_nolines,

        # mean delta percent (two versions)
        "sham_mean_delta_percent_connected": str(sham_mean_delta_percent_connected) if sham_mean_delta_percent_connected else None,
        "sham_mean_delta_percent_nolines": str(sham_mean_delta_percent_nolines) if sham_mean_delta_percent_nolines else None,
        "nma_mean_delta_percent_connected": str(nma_mean_delta_percent_connected) if nma_mean_delta_percent_connected else None,
        "nma_mean_delta_percent_nolines": str(nma_mean_delta_percent_nolines) if nma_mean_delta_percent_nolines else None,
        "sham_mean_delta_percent_stats_connected": sham_mean_delta_percent_stats_connected,
        "sham_mean_delta_percent_stats_nolines": sham_mean_delta_percent_stats_nolines,
        "nma_mean_delta_percent_stats_connected": nma_mean_delta_percent_stats_connected,
        "nma_mean_delta_percent_stats_nolines": nma_mean_delta_percent_stats_nolines,

        # variance delta s^2 (two versions)
        "sham_var_delta_s2_connected": str(sham_var_delta_s2_connected) if sham_var_delta_s2_connected else None,
        "sham_var_delta_s2_nolines": str(sham_var_delta_s2_nolines) if sham_var_delta_s2_nolines else None,
        "nma_var_delta_s2_connected": str(nma_var_delta_s2_connected) if nma_var_delta_s2_connected else None,
        "nma_var_delta_s2_nolines": str(nma_var_delta_s2_nolines) if nma_var_delta_s2_nolines else None,
        "sham_var_delta_s2_stats_connected": sham_var_delta_s2_stats_connected,
        "sham_var_delta_s2_stats_nolines": sham_var_delta_s2_stats_nolines,
        "nma_var_delta_s2_stats_connected": nma_var_delta_s2_stats_connected,
        "nma_var_delta_s2_stats_nolines": nma_var_delta_s2_stats_nolines,

        # variance delta percent (two versions)
        "sham_var_delta_percent_connected": str(sham_var_delta_percent_connected) if sham_var_delta_percent_connected else None,
        "sham_var_delta_percent_nolines": str(sham_var_delta_percent_nolines) if sham_var_delta_percent_nolines else None,
        "nma_var_delta_percent_connected": str(nma_var_delta_percent_connected) if nma_var_delta_percent_connected else None,
        "nma_var_delta_percent_nolines": str(nma_var_delta_percent_nolines) if nma_var_delta_percent_nolines else None,
        "sham_var_delta_percent_stats_connected": sham_var_delta_percent_stats_connected,
        "sham_var_delta_percent_stats_nolines": sham_var_delta_percent_stats_nolines,
        "nma_var_delta_percent_stats_connected": nma_var_delta_percent_stats_connected,
        "nma_var_delta_percent_stats_nolines": nma_var_delta_percent_stats_nolines,

        "n_ran": int(len(results)),
        "n_found_jsons": int(len(song_jsons)),
        "metadata_id_col_used": id_col_use,
        "metadata_date_col_used": date_col_use,
        "metadata_group_col_used": group_col_use,
    }


# ───────────── CLI ─────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Song-duration comparisons (single bird or multi-bird from root dir + metadata excel)."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--json_input", type=str, help="Single-bird song_detection.json path")
    mode.add_argument("--root_dir", type=str, help="Root directory containing per-bird folders")

    # Single-bird requirement
    p.add_argument("--treatment_date", type=str, default=None, help="(single mode) treatment date YYYY-MM-DD")

    # Multi-bird requirement
    p.add_argument("--metadata_excel", type=str, default=None, help="(multi mode) metadata excel path")
    p.add_argument("--sheet_name", type=str, default="0", help="Excel sheet name or index (default 0)")
    p.add_argument("--id_col", type=str, default=None, help="Column name for animal IDs (optional)")
    p.add_argument("--treatment_date_col", type=str, default=None, help="Column name for treatment date (optional)")
    p.add_argument("--group_col", type=str, default=None, help="Column name for group/condition (optional)")

    # Common
    p.add_argument("--merge_gap_ms", type=int, default=None)
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--songs_only", action="store_true", default=True)
    p.add_argument("--output_root", type=str, default=None, help="If set, saves per-bird outputs under this folder")
    p.add_argument("--out1", type=str, default=None, help="(single mode) output path for pre/post plot")
    p.add_argument("--out2", type=str, default=None, help="(single mode) output path for three-group plot")

    p.add_argument("--no_aggregate_plot", action="store_true", help="(multi) disable aggregate means plot")
    p.add_argument("--aggregate_plot_path", type=str, default=None, help="(multi) path for aggregate plot png")
    p.add_argument("--no_group_boxplots", action="store_true", help="(multi) disable per-group epoch boxplot figs")
    p.add_argument("--no_delta_seconds", action="store_true", help="(multi) disable mean-delta (seconds) plots")
    p.add_argument("--no_delta_percent", action="store_true", help="(multi) disable mean-delta (percent) plots")
    p.add_argument("--no_var_delta_s2", action="store_true", help="(multi) disable variance-delta (s^2) plots")
    p.add_argument("--no_var_delta_percent", action="store_true", help="(multi) disable variance-delta (percent) plots")

    args = p.parse_args()

    # parse sheet_name to int if possible
    try:
        sheet_name: Union[int, str] = int(args.sheet_name)
    except Exception:
        sheet_name = args.sheet_name

    if args.json_input:
        if not args.treatment_date:
            raise SystemExit("Single-bird mode requires --treatment_date YYYY-MM-DD")

        res = run_song_duration_comparison(
            json_input=args.json_input,
            treatment_date=args.treatment_date,
            merge_gap_ms=args.merge_gap_ms,
            songs_only=args.songs_only,
            output_path=args.out1,
            three_output_path=args.out2,
            show=not args.no_show,
        )
        print("Animal ID:", res.get("animal_id"))
        print("Pre/Post counts:", res["counts"])
        print("Three-group counts:", res["counts_three"])
        print("Three-group stats:", res["three_group_stats"])
        if res["plots"]["pre_post_saved_path"]:
            print("Saved Plot 1:", res["plots"]["pre_post_saved_path"])
        if res["plots"]["three_group_saved_path"]:
            print("Saved Plot 2:", res["plots"]["three_group_saved_path"])

    else:
        if not args.metadata_excel:
            raise SystemExit("Multi-bird mode requires --metadata_excel PATH")

        out = run_song_duration_comparison_multi(
            root_dir=args.root_dir,
            metadata_excel=args.metadata_excel,
            sheet_name=sheet_name,
            id_col=args.id_col,
            treatment_date_col=args.treatment_date_col,
            group_col=args.group_col,
            merge_gap_ms=args.merge_gap_ms,
            songs_only=args.songs_only,
            output_root=args.output_root,
            show=not args.no_show,
            save_summary_csv=True,
            make_aggregate_plot=(not args.no_aggregate_plot),
            aggregate_plot_path=args.aggregate_plot_path,
            make_group_boxplots=(not args.no_group_boxplots),
            make_delta_plots_absolute=(not args.no_delta_seconds),
            make_delta_plots_percent=(not args.no_delta_percent),
            make_variance_delta_plots_absolute=(not args.no_var_delta_s2),
            make_variance_delta_plots_percent=(not args.no_var_delta_percent),
            verbose=True,
        )
        print("Ran birds:", out["n_ran"])
        print("Found JSONs:", out["n_found_jsons"])
        print("Summary CSV:", out["summary_csv"])
        print("Aggregate plot:", out["aggregate_plot"])
        print("Sham epoch boxplot (connected):", out["sham_epoch_boxplot_connected"])
        print("Sham epoch boxplot (nolines):", out["sham_epoch_boxplot_nolines"])
        print("NMA epoch boxplot (connected):", out["nma_epoch_boxplot_connected"])
        print("NMA epoch boxplot (nolines):", out["nma_epoch_boxplot_nolines"])
        print("Metadata columns used:", out["metadata_id_col_used"], "/", out["metadata_date_col_used"], "/", out["metadata_group_col_used"])


"""
from pathlib import Path
import sys, importlib

code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
sys.path.insert(0, str(code_dir))

import song_duration_comparison as sdc
importlib.reload(sdc)

json_input = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443_song_detection.json")

res = sdc.run_song_duration_comparison(
    json_input=json_input,
    treatment_date="2024-04-30",
    merge_gap_ms=250,
    output_path=Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/figures/song_duration_comparison/USA5443_pre_post.png"),
    three_output_path=Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/figures/song_duration_comparison/USA5443_three_group.png"),
    show=True,
)

print(res["counts"], res["counts_three"])
"""



"""
# Batch run
from pathlib import Path
import importlib
import song_duration_comparison as sdc

importlib.reload(sdc)

root_dir = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")  # contains per-bird folders
metadata_excel = root_dir / "Area_X_lesion_metadata.xlsx"

# Where to put outputs (optional). If None, saves under each bird folder:
#   <bird_folder>/figures/song_duration_comparison/
output_root = root_dir / "song_duration_batch_outputs"  # or set to None

out = sdc.run_song_duration_comparison_multi(
    root_dir=root_dir,
    metadata_excel=metadata_excel,
    sheet_name=0,                 # sheet index or name
    # id_col="Animal ID",         # optional: set explicitly if you want
    # treatment_date_col="Treatment date",
    merge_gap_ms=500,             # optional: merge song segments closer than 500 ms
    songs_only=True,
    output_root=output_root,      # None -> saves into each bird folder
    show=False,                   # batch mode usually False
    save_summary_csv=True,
    verbose=True,
)

print("Ran birds:", out["n_ran"])
print("Found JSONs:", out["n_found_jsons"])
print("Summary CSV:", out["summary_csv"])
print(out["summary_df"].head())


"""
