# -*- coding: utf-8 -*-
# pre_and_post_syllable_plots.py
"""
For each syllable label (target), visualize which syllable PRECEDED it and which
FOLLOWED it in Early-Pre, Late-Pre, and Post (balanced groups), using merged
annotations from merge_annotations_from_split_songs.py.

Per target, produces one combined figure:
  • Left column  = PRECEDERS   (histogram + pies)
  • Right column = SUCCESSORS  (histogram + pies)
  • SINGLE shared legend for all pies (preceder+successor), with consistent colors
  • Each pie title shows: "<Group>\\n(n songs = ...)"

Low-frequency categories not in Top-K are aggregated into an "Other" bin (configurable).

Color policy (matches your HDBSCAN/UMAP script):
  - Integer labels → 60-color qualitative palette (tab20 + tab20b + tab20c)
  - -1 (noise)     → fixed gray
  - <START>, <END> → fixed colors
  - "Other"        → fixed light gray

REQUESTED CHANGE: Histogram bars are drawn in distinct SHADES OF GRAY per group
(Early/Late/Post), while pie wedges use the HDBSCAN label colors.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from merge_annotations_from_split_songs import build_decoded_with_split_labels


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

def _as_timestamp(d) -> pd.Timestamp:
    return pd.to_datetime(str(d)).normalize()

def _sequence_from_dict(label_to_intervals: dict) -> List[str]:
    """
    Flatten {label: [[on,off], ...]} → chronological sequence of labels (strings).
    Repeats are preserved.
    """
    if not isinstance(label_to_intervals, dict) or not label_to_intervals:
        return []
    pairs: List[Tuple[float, str]] = []
    for lab, ivals in label_to_intervals.items():
        if not isinstance(ivals, (list, tuple)):
            continue
        for itv in ivals:
            if isinstance(itv, (list, tuple)) and len(itv) >= 2:
                try:
                    on = float(itv[0])
                except Exception:
                    continue
                pairs.append((on, str(lab)))
    pairs.sort(key=lambda p: p[0])
    return [lab for _, lab in pairs]

def _balanced_split_by_treatment(
    df: pd.DataFrame,
    treatment_date: Union[str, pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Balanced split using 'Recording DateTime':
      early_pre = last n of the last 2n pre songs (excluding the last n)
      late_pre  = last n pre songs
      post_bal  = first n post songs
      n = min(len(pre)//2, len(post))
    """
    if "Recording DateTime" not in df.columns:
        if "recording_datetime" in df.columns:
            df = df.rename(columns={"recording_datetime": "Recording DateTime"})
        else:
            return df.iloc[0:0], df.iloc[0:0], df.iloc[0:0], 0

    d = df.copy()
    d = d[pd.notna(d["Recording DateTime"])].copy()
    d["Recording DateTime"] = pd.to_datetime(d["Recording DateTime"])

    tdate = _as_timestamp(treatment_date)
    pre   = d[d["Recording DateTime"] < tdate].sort_values("Recording DateTime").reset_index(drop=True)
    post  = d[d["Recording DateTime"] >= tdate].sort_values("Recording DateTime").reset_index(drop=True)

    n = int(min(len(pre)//2, len(post)))
    if n <= 0:
        return pre.iloc[0:0], pre.iloc[0:0], post.iloc[0:0], 0

    early = pre.iloc[len(pre)-2*n : len(pre)-n].copy()
    late  = pre.tail(n).copy()
    postn = post.head(n).copy()
    return early, late, postn, n

# ----- counts -----

def _preceder_counts_for_target(
    df: pd.DataFrame,
    target_label: str,
    include_start: bool = False,
    start_token: str = "<START>",
) -> pd.Series:
    from collections import Counter
    C = Counter()
    for _, row in df.iterrows():
        seq = _sequence_from_dict(row.get("syllable_onsets_offsets_ms_dict", {}))
        for i in range(1, len(seq)):
            prev_lab, cur_lab = seq[i-1], seq[i]
            if cur_lab == str(target_label):
                C[prev_lab] += 1
        if include_start and len(seq) and seq[0] == str(target_label):
            C[start_token] += 1
    return pd.Series(C, dtype=int).sort_values(ascending=False)

def _successor_counts_for_target(
    df: pd.DataFrame,
    target_label: str,
    include_end: bool = False,
    end_token: str = "<END>",
) -> pd.Series:
    from collections import Counter
    C = Counter()
    for _, row in df.iterrows():
        seq = _sequence_from_dict(row.get("syllable_onsets_offsets_ms_dict", {}))
        for i in range(0, len(seq)-1):
            cur_lab, next_lab = seq[i], seq[i+1]
            if cur_lab == str(target_label):
                C[next_lab] += 1
        if include_end and len(seq) and seq[-1] == str(target_label):
            C[end_token] += 1
    return pd.Series(C, dtype=int).sort_values(ascending=False)

def _select_target_labels_auto(merged_df: pd.DataFrame, top_k_targets: int = 8) -> List[str]:
    from collections import Counter
    C = Counter()
    for _, row in merged_df.iterrows():
        seq = _sequence_from_dict(row.get("syllable_onsets_offsets_ms_dict", {}))
        for i in range(1, len(seq)):  # exclude first position
            C[seq[i]] += 1
    if not C:
        return []
    s = pd.Series(C, dtype=int).sort_values(ascending=False)
    return [str(x) for x in s.head(top_k_targets).index.tolist()]

def _unified_order(top_k: int, *series: pd.Series) -> List[str]:
    total = None
    for s in series:
        total = s if total is None else total.add(s, fill_value=0)
    if total is None or total.empty:
        return []
    return total.sort_values(ascending=False).head(top_k).index.astype(str).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# HDBSCAN-style palette + robust Label→Color LUT
# ──────────────────────────────────────────────────────────────────────────────

def _get_tab60_palette() -> List[str]:
    """tab20 + tab20b + tab20c → 60 hex colors."""
    tab20  = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    return [mpl.colors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]

def _collect_numeric_labels_from_df(df: pd.DataFrame) -> List[int]:
    """Collect integer-like labels present as keys in syllable dicts."""
    labs = set()
    for _, row in df.iterrows():
        d = row.get("syllable_onsets_offsets_ms_dict", {})
        if isinstance(d, dict):
            for k in d.keys():
                try:
                    labs.add(int(k))
                except Exception:
                    pass
    return sorted(labs)

def _normalize_label_key(raw: Union[str, int, float],
                         start_token: str,
                         end_token: str,
                         other_label: str) -> str:
    """
    Normalize labels so color lookups don't fail:
     - strip whitespace
     - map numeric-like strings/floats to canonical integer string (e.g., '5.0' -> '5')
     - pass through tokens and 'Other'
    """
    s = str(raw).strip()
    if s in {str(start_token), str(end_token), str(other_label), str(-1)}:
        return s
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except Exception:
        return s

def _build_label_color_lut_from_df(
    df: pd.DataFrame,
    *,
    start_token: str,
    end_token: str,
    other_label: str,
    noise_color: str = "#7f7f7f",
) -> Dict[str, str]:
    """
    Mirror the HDBSCAN plotting code and ensure robust keys.
    """
    palette = _get_tab60_palette()
    int_labels = _collect_numeric_labels_from_df(df)

    lut: Dict[str, str] = {str(-1): noise_color}
    for i, lab in enumerate(int_labels):
        lut[str(lab)] = palette[i % len(palette)]

    # Tokens/aggregates: fixed, readable colors
    lut[str(start_token)] = "#000000"  # black
    lut[str(end_token)]   = "#FFD54F"  # amber
    lut[str(other_label)] = "#BDBDBD"  # light gray (distinct from noise gray)
    return lut


# ----- "Other" bin helpers -----

def _apply_other_bin(
    s: pd.Series,
    label_order: List[str],
    *,
    include_other: bool,
    other_label: str = "Other",
) -> pd.Series:
    """
    Reindex to label_order, aggregate anything NOT in label_order into 'Other'.
    If 'Other' already exists, we add to it (no duplicate index).
    """
    s = s.copy().astype(float)
    main = s.reindex(label_order, fill_value=0.0)
    leftover = s.drop(labels=label_order, errors="ignore").sum()

    if include_other and leftover > 0:
        if other_label in main.index:
            main.loc[other_label] = float(main.loc[other_label]) + float(leftover)
        else:
            main = pd.concat([main, pd.Series({other_label: float(leftover)})])

    if not main.index.is_unique:
        main = main.groupby(level=0).sum()

    return main

def _order_with_optional_other(
    label_order: List[str],
    s_list: List[pd.Series],
    *,
    include_other: bool,
    other_label: str = "Other",
) -> List[str]:
    """
    Append 'Other' once (at the end) iff at least one series has leftover>0
    for labels not in label_order.
    """
    if not include_other or other_label in label_order:
        return label_order

    for s in s_list:
        if s.drop(labels=label_order, errors="ignore").sum() > 0:
            return label_order + [other_label]
    return label_order


# ----- Legend sorting (numeric first, then tokens, then others) -----

def _sort_legend_labels(labels: List[str],
                        start_token: str,
                        end_token: str,
                        other_label: str) -> List[str]:
    """
    Sort legend labels:
      - numeric (incl. negatives like -1) → ascending integer order
      - then tokens in this order: <START>, <END>, Other
      - then any remaining non-numeric labels → lexicographic
    Expects already-normalized strings.
    """
    def _unique(seq): return list(dict.fromkeys(seq))

    nums: List[int] = []
    nonnums: List[str] = []
    for lab in labels:
        try:
            f = float(lab)
            if f.is_integer():
                nums.append(int(f))
            else:
                nonnums.append(lab)
        except Exception:
            nonnums.append(lab)

    nums = sorted(_unique(nums))
    numeric_sorted = [str(n) for n in nums]

    remainder = [x for x in _unique(nonnums) if x not in {start_token, end_token, other_label}]
    tokens = [tok for tok in (start_token, end_token, other_label) if tok in nonnums]

    return numeric_sorted + tokens + sorted(remainder)


# ──────────────────────────────────────────────────────────────────────────────
# Combined plot: PRECEDER (left) + SUCCESSOR (right) for one target label
# ──────────────────────────────────────────────────────────────────────────────

def _plot_pre_suc_panel(
    *,
    target_label: str,
    title_suffix: str,
    # PRECEDER inputs (expected to be already "Other"-processed for pies)
    counts_pre_by_group: Dict[str, pd.Series],
    order_pre: List[str],                 # hist order
    n_pre_by_group: Dict[str, int],
    # SUCCESSOR inputs
    counts_suc_by_group: Dict[str, pd.Series],
    order_suc: List[str],                 # hist order
    n_suc_by_group: Dict[str, int],
    # NEW: number of songs per group (Early/Late/Post)
    n_songs_by_group: Optional[Dict[str, int]] = None,
    # shared legend + "Other" behavior
    include_other_in_hist: bool = True,
    other_label: str = "Other",
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    # NEW: LUT + tokens to ensure robust lookup
    color_lut: Optional[Dict[str, str]] = None,
    start_token: str = "<START>",
    end_token: str = "<END>",
):
    """
    Make a single figure with two columns:
      left  = PRECEDER (hist + pies)
      right = SUCCESSOR (hist + pies)

    Pie colors are looked up through `color_lut` using normalized keys (so '5',
    '5.0', 5 match). Histogram bars are shaded greys by group (Early/Late/Post).
    """
    groups = ["Early Pre", "Late Pre", "Post"]
    color_lut = color_lut or {}

    def _lab_color(lab: Union[str, int, float]) -> str:
        key = _normalize_label_key(lab, start_token, end_token, other_label)
        return color_lut.get(key, "#808080")  # mid-gray fallback (should be rare)

    # Build union of labels actually appearing in pies (normalize for legend)
    pie_series_list = [
        counts_pre_by_group.get("Early Pre",  pd.Series(dtype=float)),
        counts_pre_by_group.get("Late Pre",   pd.Series(dtype=float)),
        counts_pre_by_group.get("Post",       pd.Series(dtype=float)),
        counts_suc_by_group.get("Early Pre",  pd.Series(dtype=float)),
        counts_suc_by_group.get("Late Pre",   pd.Series(dtype=float)),
        counts_suc_by_group.get("Post",       pd.Series(dtype=float)),
    ]
    union_labels: List[str] = []
    for s in pie_series_list:
        for lab in getattr(s, "index", []):
            key = _normalize_label_key(lab, start_token, end_token, other_label)
            if key not in union_labels:
                union_labels.append(key)
    union_labels_sorted = _sort_legend_labels(union_labels, start_token, end_token, other_label)

    # Layout
    fig = plt.figure(figsize=(16, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.3], wspace=0.28, hspace=0.38)
    fig.suptitle(f"Syllable {target_label}: Preceders & Successors {title_suffix}", fontsize=14)

    # Predefine grey shades for histograms (Early / Late / Post)
    GREYS = ["#4a4a4a", "#8c8c8c", "#cfcfcf"]

    # --------- PRECEDER column ---------
    ax_hist_pre = fig.add_subplot(gs[0, 0])
    gs_pies_pre = gs[1, 0].subgridspec(1, 3, wspace=0.06)
    ax_pie_pre = [fig.add_subplot(gs_pies_pre[0, i]) for i in range(3)]

    x_pre = np.arange(len(order_pre), dtype=float)
    width = 0.8 / max(1, len(groups))

    # PRECEDER histogram (%), GREY bars
    for i, g in enumerate(groups):
        s = counts_pre_by_group.get(g, pd.Series(dtype=float)).reindex(order_pre, fill_value=0.0)
        denom = s.sum()
        y = (s / denom * 100.0) if denom > 0 else np.zeros_like(s.values)
        ax_hist_pre.bar(x_pre + (i - (len(groups)-1)/2)*width, y, width=width,
                        label=f"{g} (n={n_pre_by_group.get(g, 0)})",
                        color=GREYS[i % len(GREYS)], edgecolor="#333333", linewidth=0.6,
                        zorder=2)
    ax_hist_pre.set_title("Preceders (percent)")
    ax_hist_pre.set_xticks(x_pre)
    ax_hist_pre.set_xticklabels(order_pre, rotation=45, ha="right")
    ax_hist_pre.set_ylabel("% of preceders")
    _style_ax(ax_hist_pre)
    ax_hist_pre.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
    ax_hist_pre.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)

    # PRECEDER pies (HDBSCAN colors)
    for ax, g in zip(ax_pie_pre, groups):
        s = counts_pre_by_group.get(g, pd.Series(dtype=float))
        s = s[s > 0]
        n_here_edges = n_pre_by_group.get(g, 0)
        n_songs_txt = (n_songs_by_group or {}).get(g, n_here_edges)
        if len(s) == 0:
            ax.text(0.5, 0.55, f"No data\n{g}\n(n songs = {n_songs_txt})",
                    ha="center", va="center")
            ax.axis("off"); continue
        vals = s.values.astype(float)
        labs_raw = list(s.index)
        labs_norm = [_normalize_label_key(l, start_token, end_token, other_label) for l in labs_raw]
        cols = [_lab_color(l) for l in labs_norm]
        ax.pie(vals, labels=None, startangle=90, colors=cols,
               wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
        ax.set_title(f"{g}\n(n songs = {n_songs_txt})", y=1.06, fontsize=11)
        _style_ax(ax)

    # --------- SUCCESSOR column ---------
    ax_hist_suc = fig.add_subplot(gs[0, 1])
    gs_pies_suc = gs[1, 1].subgridspec(1, 3, wspace=0.06)
    ax_pie_suc = [fig.add_subplot(gs_pies_suc[0, i]) for i in range(3)]

    x_suc = np.arange(len(order_suc), dtype=float)

    # SUCCESSOR histogram (%), GREY bars
    for i, g in enumerate(groups):
        s = counts_suc_by_group.get(g, pd.Series(dtype=float)).reindex(order_suc, fill_value=0.0)
        denom = s.sum()
        y = (s / denom * 100.0) if denom > 0 else np.zeros_like(s.values)
        ax_hist_suc.bar(x_suc + (i - (len(groups)-1)/2)*width, y, width=width,
                        label=f"{g} (n={n_suc_by_group.get(g, 0)})",
                        color=GREYS[i % len(GREYS)], edgecolor="#333333", linewidth=0.6,
                        zorder=2)
    ax_hist_suc.set_title("Successors (percent)")
    ax_hist_suc.set_xticks(x_suc)
    ax_hist_suc.set_xticklabels(order_suc, rotation=45, ha="right")
    ax_hist_suc.set_ylabel("% of successors")
    _style_ax(ax_hist_suc)
    ax_hist_suc.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
    ax_hist_suc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)

    # SUCCESSOR pies (HDBSCAN colors)
    for ax, g in zip(ax_pie_suc, groups):
        s = counts_suc_by_group.get(g, pd.Series(dtype=float))
        s = s[s > 0]
        n_here_edges = n_suc_by_group.get(g, 0)
        n_songs_txt = (n_songs_by_group or {}).get(g, n_here_edges)
        if len(s) == 0:
            ax.text(0.5, 0.55, f"No data\n{g}\n(n songs = {n_songs_txt})",
                    ha="center", va="center")
            ax.axis("off"); continue
        vals = s.values.astype(float)
        labs_raw = list(s.index)
        labs_norm = [_normalize_label_key(l, start_token, end_token, other_label) for l in labs_raw]
        cols = [_lab_color(l) for l in labs_norm]
        ax.pie(vals, labels=None, startangle=90, colors=cols,
               wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
        ax.set_title(f"{g}\n(n songs = {n_songs_txt})", y=1.06, fontsize=11)
        _style_ax(ax)

    # SINGLE shared legend (preceder + successor union), sorted numerically
    handles = [plt.Line2D([0], [0], marker="o", linestyle="None",
                          color=_lab_color(lab), markerfacecolor=_lab_color(lab),
                          markersize=7, label=lab) for lab in union_labels_sorted]
    if handles:
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(10, len(handles)), frameon=False,
                   bbox_to_anchor=(0.5, 0.01))

    saved = None
    if output_path:
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight"); saved = str(output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fig": fig, "saved_path": saved}


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PrePostPrecederSuccessorResult:
    merged_df: pd.DataFrame
    early_df: pd.DataFrame
    late_df: pd.DataFrame
    post_df: pd.DataFrame
    targets_used: List[str]
    per_target_outputs: Dict[str, str]  # target -> combined panel path (or None)

def run_pre_and_post_syllable_plots(
    *,
    song_detection_json: Union[str, Path],
    decoded_annotations_json: Union[str, Path],
    treatment_date: Union[str, pd.Timestamp],
    # merge/append options passed through
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    # selection + plotting
    target_labels: Optional[List[str]] = None,   # if None → auto-select top_k_targets
    top_k_targets: Optional[int] = 8,
    top_k_preceders: int = 12,
    top_k_successors: int = 12,
    include_start_as_preceder: bool = False,
    include_end_as_successor: bool = False,
    start_token: str = "<START>",
    end_token: str = "<END>",
    include_other_bin: bool = True,      # controls pies' "Other"
    include_other_in_hist: bool = True,  # controls hist "Other"
    other_label: str = "Other",
    combined_output_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    # allow passing an external LUT from your HDBSCAN script
    label_color_lut: Optional[Dict[Union[int, str], str]] = None,
) -> Dict[str, object]:
    """
    Build merged annotations, split balanced groups, then for each target syllable
    produce a single figure with PRECEDER and SUCCESSOR distributions.

    - Pie colors match the HDBSCAN palette (either built from data or provided via LUT).
    - Histogram bars are rendered as distinct shades of grey per group.
    """
    # 1) Merge songs + annotations (time-aligned; optional repeat coalescing)
    res = build_decoded_with_split_labels(
        decoded_database_json=decoded_annotations_json,
        song_detection_json=song_detection_json,
        only_song_present=True,
        compute_durations=True,
        add_recording_datetime=True,
        songs_only=True,
        flatten_spec_params=True,
        max_gap_between_song_segments=max_gap_between_song_segments,
        segment_index_offset=segment_index_offset,
        merge_repeated_syllables=merge_repeated_syllables,
        repeat_gap_ms=repeat_gap_ms,
        repeat_gap_inclusive=repeat_gap_inclusive,
    )
    merged_df = res.annotations_appended_df.copy()
    if merged_df.empty:
        return {
            "merged_df": merged_df,
            "early_df": merged_df, "late_df": merged_df, "post_df": merged_df,
            "targets_used": [],
            "per_target_outputs": {},
        }

    # 2) Balanced groups
    early_df, late_df, post_df, _ = _balanced_split_by_treatment(merged_df, treatment_date)

    # Song counts (shown above pies)
    n_songs_by_group = {
        "Early Pre": int(early_df.shape[0]),
        "Late Pre":  int(late_df.shape[0]),
        "Post":      int(post_df.shape[0]),
    }

    # 3) Choose targets
    if target_labels is None:
        cat_df = pd.concat([early_df, late_df, post_df], ignore_index=True)
        if top_k_targets is None:
            top_k_targets = 8
        targets = _select_target_labels_auto(cat_df, top_k_targets=top_k_targets)
    else:
        targets = [str(t) for t in target_labels]

    # 4) Build or adopt label→color LUT
    if label_color_lut is None:
        label_color_lut = _build_label_color_lut_from_df(
            merged_df,
            start_token=start_token,
            end_token=end_token,
            other_label=other_label,
        )
    else:
        # Normalize incoming LUT keys to strings ("5"), keeping values as given.
        normalized_lut: Dict[str, str] = {}
        for k, v in label_color_lut.items():
            key = _normalize_label_key(k, start_token, end_token, other_label)
            normalized_lut[str(key)] = v
        # Ensure tokens/Other exist with fixed colors (if omitted externally).
        normalized_lut.setdefault(str(start_token), "#000000")
        normalized_lut.setdefault(str(end_token), "#FFD54F")
        normalized_lut.setdefault(str(other_label), "#BDBDBD")
        normalized_lut.setdefault(str(-1), "#7f7f7f")
        label_color_lut = normalized_lut

    outputs: Dict[str, str] = {}
    if combined_output_dir:
        combined_output_dir = Path(combined_output_dir); combined_output_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = f"(balanced Early-Pre / Late-Pre / Post; Treatment {pd.to_datetime(str(treatment_date)).date()})"
    groups = ["Early Pre", "Late Pre", "Post"]

    for tgt in targets:
        # Preceder counts (raw)
        e_pre_raw = _preceder_counts_for_target(early_df, tgt, include_start=include_start_as_preceder, start_token=start_token)
        l_pre_raw = _preceder_counts_for_target(late_df,  tgt, include_start=include_start_as_preceder, start_token=start_token)
        p_pre_raw = _preceder_counts_for_target(post_df,  tgt, include_start=include_start_as_preceder, start_token=start_token)
        order_pre = _unified_order(top_k_preceders, e_pre_raw, l_pre_raw, p_pre_raw)

        # Successor counts (raw)
        e_suc_raw = _successor_counts_for_target(early_df, tgt, include_end=include_end_as_successor, end_token=end_token)
        l_suc_raw = _successor_counts_for_target(late_df,  tgt, include_end=include_end_as_successor, end_token=end_token)
        p_suc_raw = _successor_counts_for_target(post_df,  tgt, include_end=include_end_as_successor, end_token=end_token)
        order_suc = _unified_order(top_k_successors, e_suc_raw, l_suc_raw, p_suc_raw)

        # Optional "Other" aggregation
        order_pre_hist = _order_with_optional_other(order_pre, [e_pre_raw, l_pre_raw, p_pre_raw],
                                                    include_other=include_other_in_hist, other_label=other_label)
        order_pre_pie  = _order_with_optional_other(order_pre, [e_pre_raw, l_pre_raw, p_pre_raw],
                                                    include_other=include_other_bin, other_label=other_label)

        order_suc_hist = _order_with_optional_other(order_suc, [e_suc_raw, l_suc_raw, p_suc_raw],
                                                    include_other=include_other_in_hist, other_label=other_label)
        order_suc_pie  = _order_with_optional_other(order_suc, [e_suc_raw, l_suc_raw, p_suc_raw],
                                                    include_other=include_other_bin, other_label=other_label)

        # Build SERIES with "Other" applied for pies (and optionally hists)
        def _prep_counts(raw_s: pd.Series, order_for_hist: List[str], order_for_pie: List[str]):
            s_hist = _apply_other_bin(raw_s, order_for_hist, include_other=(include_other_in_hist), other_label=other_label)
            s_pie  = _apply_other_bin(raw_s, order_for_pie,  include_other=(include_other_bin),   other_label=other_label)
            return s_hist, s_pie

        e_pre_hist, e_pre_pie = _prep_counts(e_pre_raw, order_pre_hist, order_pre_pie)
        l_pre_hist, l_pre_pie = _prep_counts(l_pre_raw, order_pre_hist, order_pre_pie)
        p_pre_hist, p_pre_pie = _prep_counts(p_pre_raw, order_pre_hist, order_pre_pie)

        e_suc_hist, e_suc_pie = _prep_counts(e_suc_raw, order_suc_hist, order_suc_pie)
        l_suc_hist, l_suc_pie = _prep_counts(l_suc_raw, order_suc_hist, order_suc_pie)
        p_suc_hist, p_suc_pie = _prep_counts(p_suc_raw, order_suc_hist, order_suc_pie)

        counts_pre_by_group_hist = {"Early Pre": e_pre_hist, "Late Pre": l_pre_hist, "Post": p_pre_hist}
        counts_suc_by_group_hist = {"Early Pre": e_suc_hist, "Late Pre": l_suc_hist, "Post": p_suc_hist}
        counts_pre_by_group_pie  = {"Early Pre": e_pre_pie,  "Late Pre": l_pre_pie,  "Post": p_pre_pie}
        counts_suc_by_group_pie  = {"Early Pre": e_suc_pie,  "Late Pre": l_suc_pie,  "Post": p_suc_pie}

        n_pre_by_group = {g: int(counts_pre_by_group_pie[g].sum()) for g in groups}
        n_suc_by_group = {g: int(counts_suc_by_group_pie[g].sum()) for g in groups}

        # Final orders used for histograms (choose from any available group consistently)
        def _pick_order(d: Dict[str, pd.Series]) -> List[str]:
            for g in groups:
                if len(d[g].index) > 0:
                    return list(d[g].index)
            return []
        order_pre_used_hist = _pick_order(counts_pre_by_group_hist)
        order_suc_used_hist = _pick_order(counts_suc_by_group_hist)

        if len(order_pre_used_hist) == 0 and len(order_suc_used_hist) == 0:
            outputs[str(tgt)] = None
            continue

        out_path = None if not combined_output_dir else combined_output_dir / f"preceder_successor_target_{tgt}.png"
        _ = _plot_pre_suc_panel(
            target_label=str(tgt),
            title_suffix=title_suffix,
            counts_pre_by_group=counts_pre_by_group_pie,   # pies use these
            order_pre=order_pre_used_hist,
            n_pre_by_group=n_pre_by_group,
            counts_suc_by_group=counts_suc_by_group_pie,
            order_suc=order_suc_used_hist,
            n_suc_by_group=n_suc_by_group,
            n_songs_by_group=n_songs_by_group,
            include_other_in_hist=include_other_in_hist,
            other_label=other_label,
            output_path=out_path, show=show,
            color_lut=label_color_lut,              # HDBSCAN-style LUT for pies
            start_token=start_token, end_token=end_token,
        )
        outputs[str(tgt)] = str(out_path) if combined_output_dir else None

    return {
        "merged_df": merged_df,
        "early_df": early_df, "late_df": late_df, "post_df": post_df,
        "targets_used": targets,
        "per_target_outputs": outputs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI (optional)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Per-target PRECEDER+SUCCESSOR panels for balanced Early/Late/Post groups (shared pie legend, robust HDBSCAN colors; grey histogram bars).")
    p.add_argument("--song-detection", "-d", type=str, required=True)
    p.add_argument("--annotations", "-a", type=str, required=True)
    p.add_argument("--treatment-date", "-t", type=str, required=True)
    p.add_argument("--gap-ms", type=int, default=500)
    p.add_argument("--seg-offset", type=int, default=0)
    p.add_argument("--merge-repeats", action="store_true")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0)
    p.add_argument("--repeat-gap-inclusive", action="store_true")
    p.add_argument("--targets", nargs="*", default=None, help="Explicit target labels; omit to auto-select.")
    p.add_argument("--top-k-targets", type=int, default=8)
    p.add_argument("--top-k-preceders", type=int, default=12)
    p.add_argument("--top-k-successors", type=int, default=12)
    p.add_argument("--include-start", action="store_true", help="Count <START> as a preceder when target is first.")
    p.add_argument("--include-end", action="store_true", help="Count <END> as a successor when target is last.")
    p.add_argument("--start-token", type=str, default="<START>")
    p.add_argument("--end-token", type=str, default="<END>")
    p.add_argument("--no-other", action="store_true", help="Disable 'Other' bin aggregation in pies.")
    p.add_argument("--no-other-in-hist", action="store_true", help="Do not show 'Other' in histograms (still in pies if enabled).")
    p.add_argument("--other-label", type=str, default="Other")
    p.add_argument("--outdir", type=str, default=None, help="Directory to save combined panels.")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    res = run_pre_and_post_syllable_plots(
        song_detection_json=args.song_detection,
        decoded_annotations_json=args.annotations,
        treatment_date=args.treatment_date,
        max_gap_between_song_segments=args.gap_ms,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
        target_labels=args.targets,
        top_k_targets=args.top_k_targets,
        top_k_preceders=args.top_k_preceders,
        top_k_successors=args.top_k_successors,
        include_start_as_preceder=args.include_start,
        include_end_as_successor=args.include_end,
        start_token=args.start_token,
        end_token=args.end_token,
        include_other_bin=not args.no_other,
        include_other_in_hist=not args.no_other_in_hist,
        other_label=args.other_label,
        combined_output_dir=args.outdir,
        show=not args.no_show,
        # label_color_lut can be provided programmatically (not via CLI) if desired.
    )
    print("Targets used:", res["targets_used"])
    print("Per-target outputs:", res["per_target_outputs"])



"""
# --- Sample usage: make pies use the exact HDBSCAN colors ---

from pathlib import Path
import pre_and_post_syllable_plots as pps

detect  = Path("/Volumes/my_own_ssd/2024_AreaX_lesions_NMA_and_sham/AreaXlesion_TweetyBERT_outputs/new_outputs/USA5443_song_detection.json")
decoded = Path("/Volumes/my_own_ssd/2024_AreaX_lesions_NMA_and_sham/AreaXlesion_TweetyBERT_outputs/new_outputs/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5443_decoded_database.json")
tdate   = "2024-04-30"

res = pps.run_pre_and_post_syllable_plots(
    song_detection_json=detect,
    decoded_annotations_json=decoded,
    treatment_date=tdate,
    combined_output_dir=decoded.parent / "preceder_successor_panels",
    # no label_color_lut provided → colors are auto-built from merged annotations
)



"""