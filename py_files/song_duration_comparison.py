# song_duration_comparison.py
# -*- coding: utf-8 -*-
"""
Pre/Post and Early-Pre vs Late-Pre vs Post song-duration comparisons
with readable legends and on-plot significance stars.

Depends on:
  - organize_song_detection_json.py
  - merge_potential_split_up_song.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
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


# ───────────── data prep ─────────────
def _prepare_table(json_input: Union[str, Path], *, merge_gap_ms: Optional[int] = None,
                   songs_only: bool = True) -> pd.DataFrame:
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


# ───────────── styling + annotation ─────────────
def _style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

def _add_sig_brackets(ax, pairs: List[Tuple[int, int]], pvals_adj: List[float],
                      data_max: float, y_pad_frac: float = 0.06, text_offset: float = 0.01):
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
        ax.plot([i, i, j, j], [y, y+step*0.3, y+step*0.3, y], color="black", linewidth=1.2, clip_on=False, zorder=5)
        ax.text((i + j) / 2, y + step*0.35, _stars(p), ha="center", va="bottom",
                fontsize=12, color="black", zorder=6)
        y += step  # advance for next bracket


# ───────────── Plot 1: Pre vs Post ─────────────
def plot_pre_post_boxplot(pre_s: pd.Series, post_s: pd.Series, *,
                          treatment_date: Union[str, pd.Timestamp],
                          merge_gap_ms: Optional[int] = None,
                          title: Optional[str] = None,
                          output_path: Optional[Union[str, Path]] = None,
                          show: bool = True,
                          jitter: float = 0.08, point_size: float = 12.0,
                          point_alpha: float = 0.2, bw_box_alpha: float = 0.35) -> Dict[str, Any]:
    if not isinstance(treatment_date, pd.Timestamp):
        treatment_date = pd.to_datetime(str(treatment_date)).normalize()

    data = []; xpos = []; labels = []; pos = 1
    if len(pre_s):  data.append(pre_s.values);  xpos.append(pos); labels.append(f"Pre\nn={pre_s.size}");  pos += 1
    if len(post_s): data.append(post_s.values); xpos.append(pos); labels.append(f"Post\nn={post_s.size}")

    fig, ax = plt.subplots(figsize=(9, 6))
    # make room for outside legend on the right
    fig.subplots_adjust(right=0.78)

    if data:
        # scatter behind
        if len(pre_s):
            ax.scatter(np.random.uniform(-jitter, jitter, size=len(pre_s)) + xpos[0],
                       pre_s.values, s=point_size, alpha=point_alpha, linewidths=0,
                       edgecolors="none", zorder=1)
        if len(post_s):
            ax.scatter(np.random.uniform(-jitter, jitter, size=len(post_s)) + xpos[-1],
                       post_s.values, s=point_size, alpha=point_alpha, linewidths=0,
                       edgecolors="none", zorder=1)

        # boxes on top
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

        # significance for two groups
        if len(pre_s) and len(post_s):
            _, p = _try_mannwhitney(pre_s.values, post_s.values)
            data_max = np.nanmax(np.concatenate([pre_s.values, post_s.values]))
            _add_sig_brackets(ax, [(1, 2)], [p], data_max)

        ax.set_ylabel("Song duration (s)")
        t = title or f"Song durations Pre vs Post (Treatment: {treatment_date.date()})"
        if merge_gap_ms and merge_gap_ms > 0:
            t += f"\n(Merged gap < {merge_gap_ms} ms)"
        ax.set_title(t)

        # legend outside (right)
        handles = [
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="Median"),
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="Mean"),
            Line2D([0], [0], marker="o", linestyle="None", color="black",
                   alpha=point_alpha, markersize=6, label="Song (point)"),
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, borderaxespad=0.0)

    else:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=12); ax.axis("off")

    _style_ax(ax)

    saved = None
    if output_path:
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        saved = str(output_path)
    if show: plt.show()
    else: plt.close(fig)
    return {"fig": fig, "ax": ax, "saved_path": saved}


# ───────────── Plot 2: 3 groups + stars ─────────────
def plot_three_group_boxplot(early_s: pd.Series, late_s: pd.Series, post_s: pd.Series, *,
                             treatment_date: Union[str, pd.Timestamp],
                             merge_gap_ms: Optional[int] = None,
                             title: Optional[str] = None,
                             output_path: Optional[Union[str, Path]] = None,
                             show: bool = True,
                             jitter: float = 0.08, point_size: float = 12.0,
                             point_alpha: float = 0.2, bw_box_alpha: float = 0.35,
                             do_stats: bool = True) -> Dict[str, Any]:
    if not isinstance(treatment_date, pd.Timestamp):
        treatment_date = pd.to_datetime(str(treatment_date)).normalize()

    groups = []; labels = []; arrs: Dict[str, np.ndarray] = {}
    if len(early_s): groups.append(early_s.values); labels.append(f"Early Pre\nn={early_s.size}"); arrs["Early Pre"] = early_s.values
    if len(late_s):  groups.append(late_s.values);  labels.append(f"Late Pre\nn={late_s.size}");  arrs["Late Pre"]  = late_s.values
    if len(post_s):  groups.append(post_s.values);  labels.append(f"Post\nn={post_s.size}");      arrs["Post"]      = post_s.values

    # Leave EXTRA room on the right for legend + stats box
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.subplots_adjust(right=0.78)
    stats_out: Dict[str, Any] = {"kw": None, "pairwise": []}

    if groups:
        xpos = np.arange(1, len(groups) + 1)

        # scatter behind boxes
        for idx, s in enumerate([early_s, late_s, post_s]):
            if len(s):
                ax.scatter(np.random.uniform(-jitter, jitter, size=len(s)) + xpos[idx],
                           s.values, s=point_size, alpha=point_alpha,
                           linewidths=0, edgecolors="none", zorder=1)

        # boxes on top
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
        t = title or f"Song durations: Early Pre / Late Pre / Post (Treatment: {treatment_date.date()})"
        if merge_gap_ms and merge_gap_ms > 0:
            t += f"\n(Balanced groups, merged gap < {merge_gap_ms} ms)"
        else:
            t += "\n(Balanced groups)"
        ax.set_title(t)

        # legend outside (right)
        handles = [
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="Median"),
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="Mean"),
            Line2D([0], [0], marker="o", linestyle="None", color="black",
                   alpha=point_alpha, markersize=6, label="Song (point)"),
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, borderaxespad=0.0)

        # ----- statistics + on-plot stars -----
        if do_stats and len(arrs) >= 2:
            order = [g for g in ["Early Pre", "Late Pre", "Post"] if g in arrs]
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

            # summary OUTSIDE the axes, above the legend (right margin area)
            summary_lines = [f"Kruskal–Wallis: H={H:.3g}, p={_fmt_p(p_kw)}"]
            for d in pair_stats:
                summary_lines.append(f"{d['a']} vs {d['b']}: p(BH)={_fmt_p(d['p_bh'])}  [{d['stars']}]")
            fig.text(
                0.80, 0.98, "\n".join(summary_lines),
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95)
            )

            # star brackets ABOVE boxes (inside axes)
            xmap = {lab: i+1 for i, lab in enumerate(order)}
            star_pairs = [(xmap[a], xmap[b]) for (a, b) in pairs_lbl]
            data_max = np.nanmax(np.concatenate(arrays))
            _add_sig_brackets(ax, star_pairs, adj, data_max)

    else:
        ax.text(0.5, 0.5, "Insufficient data for balanced three-group plot",
                ha="center", va="center", fontsize=12); ax.axis("off")

    _style_ax(ax)

    saved = None
    if output_path:
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        saved = str(output_path)
    if show: plt.show()
    else: plt.close(fig)

    return {"fig": fig, "ax": ax, "saved_path": saved, "stats": stats_out}



# ───────────── public entrypoint ─────────────
def run_song_duration_comparison(json_input: Union[str, Path], *,
                                 treatment_date: Union[str, pd.Timestamp],
                                 merge_gap_ms: Optional[int] = None,
                                 songs_only: bool = True,
                                 output_path: Optional[Union[str, Path]] = None,
                                 three_output_path: Optional[Union[str, Path]] = None,
                                 show: bool = True) -> Dict[str, Any]:
    df = _prepare_table(json_input, merge_gap_ms=merge_gap_ms, songs_only=songs_only)

    pre_df, post_df = _split_pre_post(df, treatment_date=treatment_date)
    pre_s  = _durations_seconds(pre_df)
    post_s = _durations_seconds(post_df)

    plot1 = plot_pre_post_boxplot(
        pre_s, post_s, treatment_date=treatment_date,
        merge_gap_ms=merge_gap_ms, output_path=output_path, show=show
    )

    early_df, late_df, post_bal_df, n = _split_three_groups_balanced(pre_df, post_df)
    early_s = _durations_seconds(early_df)
    late_s  = _durations_seconds(late_df)
    post_b  = _durations_seconds(post_bal_df)

    plot2 = plot_three_group_boxplot(
        early_s, late_s, post_b, treatment_date=treatment_date,
        merge_gap_ms=merge_gap_ms, output_path=three_output_path, show=show, do_stats=True
    )

    return {
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


# ───────────── CLI ─────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Song-duration comparisons with readable legends and star annotations.")
    p.add_argument("json_input", type=str)
    p.add_argument("--treatment_date", type=str, required=True)
    p.add_argument("--merge_gap_ms", type=int, default=None)
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--out1", type=str, default=None)
    p.add_argument("--out2", type=str, default=None)
    args = p.parse_args()

    res = run_song_duration_comparison(
        json_input=args.json_input,
        treatment_date=args.treatment_date,
        merge_gap_ms=args.merge_gap_ms,
        output_path=args.out1,
        three_output_path=args.out2,
        show=not args.no_show,
    )
    print("Pre/Post counts:", res["counts"])
    print("Three-group counts:", res["counts_three"])
    print("Three-group stats:", res["three_group_stats"])
    if res["plots"]["pre_post_saved_path"]:
        print("Saved Plot 1:", res["plots"]["pre_post_saved_path"])
    if res["plots"]["three_group_saved_path"]:
        print("Saved Plot 2:", res["plots"]["three_group_saved_path"])
