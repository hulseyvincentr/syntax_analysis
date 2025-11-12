# -*- coding: utf-8 -*-
# last_syllable_plots.py
"""
Make last-syllable plots (histogram + pies) for Early-Pre, Late-Pre, Post
balanced groups, using merged songs/annotations.

Changes requested:
  • Histograms use GREY bars (distinct shades for Early/Late/Post).
  • Pie wedges use the HDBSCAN-style 60-color palette (tab20+tab20b+tab20c).
  • Legend for pies is sorted numerically (e.g., -1, 0, 1, 2, ...).

Optional:
  • Provide an external label_color_lut={int/str->hex} to force exact colors
    used elsewhere (e.g., from your .npz/UMAP script). If omitted, a LUT is
    built from labels found in the merged annotations.

Example (Spyder):
    from pathlib import Path
    import importlib
    import last_syllable_plots as lsp
    importlib.reload(lsp)

    detect  = Path("/Volumes/my_own_ssd/2024_AreaX_lesions_NMA_and_sham/AreaXlesion_TweetyBERT_outputs/new_outputs/USA5443_song_detection.json")
    decoded = Path("/Volumes/my_own_ssd/2024_AreaX_lesions_NMA_and_sham/AreaXlesion_TweetyBERT_outputs/new_outputs/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5443_decoded_database.json")
    tdate   = "2024-04-30"

    res = lsp.run_last_syllable_plots(
        song_detection_json=detect,
        decoded_annotations_json=decoded,
        treatment_date=tdate,
        max_gap_between_song_segments=500,
        segment_index_offset=0,
        merge_repeated_syllables=True,
        repeat_gap_ms=10.0,
        repeat_gap_inclusive=False,
        top_k_labels=15,
        hist_output_path=decoded.parent / "last_syllable_hist.png",
        pies_output_path=decoded.parent / "last_syllable_pies.png",
        show=True,
        # label_color_lut=external_lut,  # (optional) pass your exact HDBSCAN LUT here
    )
    print(res["counts_three"])
    print(res["label_order"])
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
# Styling / Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

def _as_timestamp(d) -> pd.Timestamp:
    return pd.to_datetime(str(d)).normalize()

def _normalize_label_key(raw: Union[str, int, float]) -> str:
    """
    Normalize labels for LUT lookup:
      - Trim whitespace
      - Convert int-like strings/floats to canonical integer strings ('5.0'->'5')
    """
    s = str(raw).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except Exception:
        return s

def _sort_numeric_strings(labels: List[str]) -> List[str]:
    """Return labels sorted numerically when possible, otherwise lexicographically."""
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
    # dedupe preserving order
    def _unique(seq): return list(dict.fromkeys(seq))
    nums = sorted(_unique(nums))
    return [str(n) for n in nums] + sorted(_unique(nonnums))


# ──────────────────────────────────────────────────────────────────────────────
# HDBSCAN-style palette & LUT (tab20 + tab20b + tab20c)
# ──────────────────────────────────────────────────────────────────────────────

def _get_tab60_palette() -> List[str]:
    tab20  = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    return [mpl.colors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]  # 60 colors

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

def _build_label_color_lut_from_df(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a label→hex LUT:
      - numeric labels map to tab60 colors (repeats if >60)
      - -1 (noise) fixed to mid-gray
    """
    palette = _get_tab60_palette()
    int_labels = _collect_numeric_labels_from_df(df)
    lut: Dict[str, str] = {str(-1): "#7f7f7f"}  # fixed grey for noise
    for i, lab in enumerate(int_labels):
        lut[str(lab)] = palette[i % len(palette)]
    return lut


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def _last_syllable_from_dict(label_to_intervals: dict) -> Optional[str]:
    """
    Choose the label whose LAST interval ends latest in time.
    Returns None if the dict is empty or malformed.
    """
    if not isinstance(label_to_intervals, dict) or not label_to_intervals:
        return None
    best_lab = None
    best_off = -np.inf
    for lab, ivals in label_to_intervals.items():
        if not isinstance(ivals, (list, tuple)) or not ivals:
            continue
        try:
            offs = [float(v[1]) for v in ivals if isinstance(v, (list, tuple)) and len(v) >= 2]
        except Exception:
            offs = []
        if not offs:
            continue
        m = max(offs)
        if m > best_off:
            best_off = m
            best_lab = str(lab)
    return best_lab

def _extract_last_syllables(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series of last-syllable labels (strings), one per row (song),
    dropping rows with empty/invalid annotation dicts.
    """
    s = df["syllable_onsets_offsets_ms_dict"].apply(_last_syllable_from_dict)
    s = s.dropna().astype(str)
    return s

def _split_pre_post_balanced(
    df: pd.DataFrame,
    treatment_date: Union[str, pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Return (early_pre, late_pre, post_balanced, n) using the same
    2n/ n / n scheme (equal sizes) based on Recording DateTime.
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

def _counts_by_group(
    last_early: pd.Series,
    last_late: pd.Series,
    last_post: pd.Series,
    top_k: int = 15,
) -> Tuple[Dict[str, pd.Series], List[str]]:
    """
    Build {group -> counts Series} with a unified label order based on top_k by
    total frequency across all groups. Returns (counts_by_group, label_order).
    """
    groups = {
        "Early Pre": last_early.value_counts(),
        "Late Pre":  last_late.value_counts(),
        "Post":      last_post.value_counts(),
    }
    total = (groups["Early Pre"].add(groups["Late Pre"], fill_value=0)
                           .add(groups["Post"],      fill_value=0))
    top_labels = total.sort_values(ascending=False).head(top_k).index.tolist()
    # Normalize label strings consistently
    label_order = [_normalize_label_key(l) for l in top_labels]
    counts = {g: s.rename(index=_normalize_label_key).reindex(label_order, fill_value=0).astype(int)
              for g, s in groups.items()}
    return counts, label_order


# ──────────────────────────────────────────────────────────────────────────────
# Plotters
# ──────────────────────────────────────────────────────────────────────────────

def _plot_histogram(
    counts: Dict[str, pd.Series],
    label_order: List[str],
    title: str,
    n_by_group: Dict[str, int],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Grouped bar chart in GREYSCALE: x = label, y = count.
    Each group (Early/Late/Post) uses a distinct grey shade.
    Legend shows "<Group> (n=<count>)" and sits to the right.
    """
    groups = list(counts.keys())
    x = np.arange(len(label_order), dtype=float)
    width = 0.8 / max(1, len(groups))

    fig, ax = plt.subplots(figsize=(max(9, 0.5*len(label_order)+4), 6))
    fig.subplots_adjust(right=0.80)  # space for legend

    # Distinct grey shades for groups
    GREYS = ["#4a4a4a", "#8c8c8c", "#cfcfcf"]

    for i, g in enumerate(groups):
        y = counts[g].reindex(label_order, fill_value=0).to_numpy(dtype=float)
        label = f"{g} (n={n_by_group.get(g, 0)})"
        ax.bar(x + (i - (len(groups)-1)/2)*width, y, width=width,
               label=label, color=GREYS[i % len(GREYS)],
               edgecolor="#333333", linewidth=0.6, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    _style_ax(ax)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)

    # legend outside with n's
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

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


def _plot_pies(
    counts: Dict[str, pd.Series],
    label_order: List[str],
    title: str,
    n_by_group: Dict[str, int],
    color_lut: Dict[str, str],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    1×3 pies (Early Pre, Late Pre, Post) using HDBSCAN colors via color_lut.
    Legend is sorted numerically and placed at the bottom.
    """
    groups = list(counts.keys())
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(4*n + 4, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.18)

    # Build union label list from label_order (top-k) and any group leftovers
    union_labels = set(label_order)
    for g in groups:
        union_labels.update([_normalize_label_key(l) for l in counts[g].index.tolist()])
    union_labels = _sort_numeric_strings(list(union_labels))

    # Legend handles use LUT colors (fallback mid-grey if missing)
    def _lab_color(lab: str) -> str:
        return color_lut.get(lab, "#808080")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   color=_lab_color(lab), markerfacecolor=_lab_color(lab),
                   markersize=8, label=lab)
        for lab in union_labels
    ]

    for ax, g in zip(axes, groups):
        s = counts[g].reindex(label_order, fill_value=0)
        s = s[s > 0]
        n_here = n_by_group.get(g, 0)

        if len(s) == 0:
            ax.text(0.5, 0.5, f"No data\n({g} • n={n_here})", ha="center", va="center")
            ax.axis("off")
            continue

        vals = s.values.astype(float)
        labs = [_normalize_label_key(l) for l in s.index.tolist()]
        cols = [_lab_color(l) for l in labs]

        ax.pie(vals, labels=None, startangle=90, colors=cols,
               wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
        ax.set_title(f"{g} (n={n_here})")
        _style_ax(ax)

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(10, len(legend_handles)),
                   frameon=False)

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
    return {"fig": fig, "axes": axes, "saved_path": saved}


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LastSyllablePlotsResult:
    appended_df: pd.DataFrame
    early_df: pd.DataFrame
    late_df: pd.DataFrame
    post_df: pd.DataFrame
    last_early: pd.Series
    last_late: pd.Series
    last_post: pd.Series
    counts_three: Dict[str, int]
    label_order: List[str]
    hist_saved_path: Optional[str]
    pies_saved_path: Optional[str]

def run_last_syllable_plots(
    *,
    song_detection_json: Union[str, Path],
    decoded_annotations_json: Union[str, Path],
    treatment_date: Union[str, pd.Timestamp],
    # merge+coalescing knobs
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    # plotting/output
    top_k_labels: int = 15,
    hist_output_path: Optional[Union[str, Path]] = None,
    pies_output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    # optional external LUT (e.g., from your .npz/UMAP code)
    label_color_lut: Optional[Dict[Union[int, str], str]] = None,
) -> Dict[str, object]:
    """
    Build merged annotations, split balanced groups, plot histogram & pies of last syllables.
    - Histograms: grey bars per group.
    - Pies: HDBSCAN colors via LUT (auto-built from data unless externally provided).
    """
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
            "appended_df": merged_df,
            "early_df": merged_df, "late_df": merged_df, "post_df": merged_df,
            "last_early": pd.Series([], dtype=str),
            "last_late":  pd.Series([], dtype=str),
            "last_post":  pd.Series([], dtype=str),
            "counts_three": {"balanced_n": 0, "early_n": 0, "late_n": 0, "post_n": 0},
            "label_order": [],
            "hist_saved_path": None,
            "pies_saved_path": None,
        }

    # Balanced split
    early_df, late_df, post_df, n = _split_pre_post_balanced(merged_df, treatment_date)
    last_early = _extract_last_syllables(early_df)
    last_late  = _extract_last_syllables(late_df)
    last_post  = _extract_last_syllables(post_df)

    # Counts + unified label order (normalize keys)
    counts_by_group, label_order = _counts_by_group(last_early, last_late, last_post, top_k=top_k_labels)

    # Build or adopt LUT
    if label_color_lut is None:
        lut = _build_label_color_lut_from_df(merged_df)
    else:
        # normalize incoming LUT keys to strings
        lut = {}
        for k, v in label_color_lut.items():
            lut[_normalize_label_key(k)] = v
        lut.setdefault(str(-1), "#7f7f7f")

    # n per plotted group (after last-syllable extraction)
    n_by_group = {
        "Early Pre": int(last_early.shape[0]),
        "Late Pre":  int(last_late.shape[0]),
        "Post":      int(last_post.shape[0]),
    }

    # Titles
    title_base = f"Last syllables per song (balanced groups; Treatment {pd.to_datetime(str(treatment_date)).date()})"
    hist_title = title_base + f"\n(Top {top_k_labels} labels)"
    pies_title = "Last syllables: Early-Pre / Late-Pre / Post"

    # Plots
    hist_res = _plot_histogram(counts_by_group, label_order, hist_title,
                               n_by_group=n_by_group,
                               output_path=hist_output_path, show=show)
    pies_res = _plot_pies(counts_by_group, label_order, pies_title,
                          n_by_group=n_by_group, color_lut=lut,
                          output_path=pies_output_path, show=show)

    return {
        "appended_df": merged_df,
        "early_df": early_df, "late_df": late_df, "post_df": post_df,
        "last_early": last_early, "last_late": last_late, "last_post": last_post,
        "counts_three": {"balanced_n": int(n), "early_n": n_by_group["Early Pre"],
                         "late_n": n_by_group["Late Pre"], "post_n": n_by_group["Post"]},
        "label_order": label_order,
        "hist_saved_path": hist_res["saved_path"],
        "pies_saved_path": pies_res["saved_path"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI (optional)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Last-syllable histogram (grey) and pies (HDBSCAN colors) for balanced Early/Late/Post groups.")
    p.add_argument("--song-detection", "-d", type=str, required=True)
    p.add_argument("--annotations", "-a", type=str, required=True)
    p.add_argument("--treatment-date", "-t", type=str, required=True)
    p.add_argument("--gap-ms", type=int, default=500)
    p.add_argument("--seg-offset", type=int, default=0)
    p.add_argument("--merge-repeats", action="store_true")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0)
    p.add_argument("--repeat-gap-inclusive", action="store_true")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--hist-out", type=str, default=None)
    p.add_argument("--pies-out", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    out = run_last_syllable_plots(
        song_detection_json=args.song_detection,
        decoded_annotations_json=args.annotations,
        treatment_date=args.treatment_date,
        max_gap_between_song_segments=args.gap_ms,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
        top_k_labels=args.top_k,
        hist_output_path=args.hist_out,
        pies_output_path=args.pies_out,
        show=not args.no_show,
        # label_color_lut can be provided programmatically if desired.
    )
    print("Counts (balanced):", out["counts_three"])
    print("Saved histogram:", out["hist_saved_path"])
    print("Saved pies:", out["pies_saved_path"])
