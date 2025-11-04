# -*- coding: utf-8 -*-
# phrase_duration_pre_vs_post_grouped.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import math
import json
import ast
import inspect
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
TITLE_FS = 18
LABEL_FS = 16
TICK_FS  = 13

def _pretty_axes(ax, x_rotation: int = 0):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.xaxis.label.set_size(LABEL_FS)
    ax.yaxis.label.set_size(LABEL_FS)
    if x_rotation:
        for lab in ax.get_xticklabels():
            lab.set_rotation(x_rotation)
            lab.set_horizontalalignment("right")

plt.rcParams.update({
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
})

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import (prefer YOUR file name)
# ──────────────────────────────────────────────────────────────────────────────
_ORGANIZE = None
try:
    # Your module/file:
    from organize_decoded_serialTS_segments import (  # type: ignore
        build_organized_segments_with_durations as _ORGANIZE
    )
except Exception:
    # Fallback alternate filename:
    from organized_decoded_serialTS_segments import (  # type: ignore
        build_organized_segments_with_durations as _ORGANIZE
    )

# ──────────────────────────────────────────────────────────────────────────────
# Dataclass result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GroupedPlotsResult:
    early_pre_path: Optional[Path]
    late_pre_path: Optional[Path]
    post_path: Optional[Path]
    aggregate_path: Optional[Path]                          # per-day box+scatter
    aggregate_three_box_path: Optional[Path]                # 3 boxes + dashed connectors (medians)
    aggregate_three_box_auto_ylim_path: Optional[Path]      # same but auto y-limits
    # 3 boxes with *per-syllable duration points* (one dot per syllable per group)
    aggregate_three_box_with_syll_points_path: Optional[Path]
    # Variance figures by syllable on x-axis
    per_syllable_variance_path: Optional[Path]              # (DISABLED; will be None)
    per_syllable_variance_boxscatter_path: Optional[Path]   # variance box + scatter (x = syllable)
    # 3-group variance: boxes + syllable points (ylim fixed to 2.0e7)
    aggregate_three_box_variance_with_syll_points_path: Optional[Path]
    # NEW: 3-group variance: boxes + jitter points + stats (ylim fixed to 2.0e7)
    aggregate_three_box_variance_with_stats_path: Optional[Path]
    syllable_labels: List[str]
    y_limits: Tuple[float, float]

# ──────────────────────────────────────────────────────────────────────────────
# Utilities: dates, parsing, table building
# ──────────────────────────────────────────────────────────────────────────────
def _parse_date_like(s: Union[str, pd.Timestamp, None]) -> Union[pd.Timestamp, pd.NaT]:
    if s is None:
        return pd.NaT
    if isinstance(s, pd.Timestamp):
        return s
    s2 = str(s).replace(".", "-")
    return pd.to_datetime(s2, errors="coerce")

def _choose_datetime_series(df: pd.DataFrame) -> pd.Series:
    # Prefer your organizer’s column
    dt = df.get("Recording DateTime", pd.Series([pd.NaT]*len(df), index=df.index)).copy()
    if "creation_date" in df.columns:
        need = dt.isna()
        if need.any():
            dt.loc[need] = pd.to_datetime(df.loc[need, "creation_date"], errors="coerce")
    if "Date" in df.columns and "Time" in df.columns:
        need = dt.isna()
        if need.any():
            combo = pd.to_datetime(
                df.loc[need, "Date"].astype(str).str.replace(".", "-", regex=False)
                + " " + df.loc[need, "Time"].astype(str),
                errors="coerce"
            )
            dt.loc[need] = combo
    return dt

def _infer_animal_id(df: pd.DataFrame, decoded_path: Path) -> str:
    for col in ["animal_id", "Animal", "Animal ID"]:
        if col in df.columns and pd.notna(df[col]).any():
            val = str(df[col].dropna().iloc[0]).strip()
            if val:
                return val
    stem = decoded_path.stem
    tok  = stem.split("_")[0]
    return tok if tok else "unknown_animal"

def _maybe_parse_dict(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(obj)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return None

def _row_ms_per_bin(row: pd.Series) -> float | None:
    for k in ["time_bin_ms", "timebin_ms", "bin_ms", "ms_per_bin"]:
        if k in row and pd.notna(row[k]):
            try:
                return float(row[k])
            except Exception:
                pass
    return None

def _is_timebin_col(colname: str) -> bool:
    c = (colname or "").lower()
    return "timebin" in c or c.endswith("_bins") or "bin" in c

def _extract_durations_from_spans(spans, *, ms_per_bin: float | None, treat_as_bins: bool) -> List[float]:
    """
    Accepts:
      • [[on, off], ...]
      • [on, off]  (single flat pair)
      • [{"onset_ms":..., "offset_ms":...}, ...]
      • [{"onset_bin":..., "offset_bin":...}, ...]
    Converts from bins→ms if treat_as_bins and ms_per_bin is provided.
    """
    out: List[float] = []
    if spans is None:
        return out

    # If it's a single flat pair [on, off], wrap it
    if isinstance(spans, (list, tuple)) and len(spans) == 2 and all(isinstance(x, (int, float)) for x in spans):
        spans = [spans]

    # If it's a dict describing one interval, wrap it
    if isinstance(spans, dict):
        spans = [spans]

    if not isinstance(spans, (list, tuple)):
        return out

    for item in spans:
        on = off = None
        using_bins = False

        if isinstance(item, dict):
            if "onset_ms" in item or "on" in item:
                on = item.get("onset_ms", item.get("on"))
                off = item.get("offset_ms", item.get("off"))
            elif "onset_bin" in item or "on_bin" in item:
                on = item.get("onset_bin", item.get("on_bin"))
                off = item.get("offset_bin", item.get("off_bin"))
                using_bins = True

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            on, off = item[:2]
            using_bins = False  # unless treat_as_bins tells us otherwise

        # Convert to duration
        try:
            dur = float(off) - float(on)
            if dur < 0:
                continue
            if treat_as_bins or using_bins:
                if ms_per_bin:
                    dur *= float(ms_per_bin)
                else:
                    # No conversion available; skip to avoid mixing units
                    continue
            out.append(dur)
        except Exception:
            continue

    return out

def _collect_phrase_durations_per_song(row: pd.Series, spans_col: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    raw = row.get(spans_col, None)
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return out

    # If the column is already a dict of label -> intervals
    d = _maybe_parse_dict(raw) if not isinstance(raw, dict) else raw
    if not isinstance(d, dict):
        return out

    mpb = _row_ms_per_bin(row)
    treat_as_bins = _is_timebin_col(spans_col)

    for lbl, spans in d.items():
        vals = _extract_durations_from_spans(spans, ms_per_bin=mpb, treat_as_bins=treat_as_bins)
        if vals:
            out[str(lbl)] = vals
    return out

def _find_best_spans_column(df: pd.DataFrame) -> str | None:
    # Prefer ms, then timebins, then anything *_dict-like
    for c in [
        "syllable_onsets_offsets_ms_dict",
        "syllable_onsets_offsets_ms",
        "onsets_offsets_ms_dict",
    ]:
        if c in df.columns:
            return c
    for c in [
        "syllable_onsets_offsets_timebins",
        "syllable_onsets_offsets_timebins_dict",
    ]:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if lc.endswith("_dict") and df[c].notna().any():
            return c
    return None

def _build_durations_table(df: pd.DataFrame, labels: Sequence[str]) -> tuple[pd.DataFrame, str | None]:
    col = _find_best_spans_column(df)
    if col is None:
        return df.copy(), None

    per_song = df.apply(lambda r: _collect_phrase_durations_per_song(r, col), axis=1)
    out = df.copy()
    for lbl in labels:  # labels are strings
        out[f"syllable_{lbl}_durations"] = per_song.apply(lambda d: d.get(str(lbl), []))
    return out, col

# ──────────────────────────────────────────────────────────────────────────────
# Label handling + plotting helpers (fixed x-order, labels as strings)
# ──────────────────────────────────────────────────────────────────────────────
def _explode_for_plot(dataset: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """
    Tall DF: ['Syllable','Phrase Duration (ms)'] with Syllable categorical set to full order.
    """
    order = [str(x) for x in labels]
    parts = []

    for lbl in order:
        col = f"syllable_{lbl}_durations"
        if col not in dataset.columns:
            parts.append(pd.DataFrame({"Syllable": [lbl], "Phrase Duration (ms)": [np.nan]}))
            continue

        e = dataset[[col]].explode(col)
        if e.empty:
            parts.append(pd.DataFrame({"Syllable": [lbl], "Phrase Duration (ms)": [np.nan]}))
            continue

        e["Phrase Duration (ms)"] = pd.to_numeric(e[col], errors="coerce")
        e = e.dropna(subset=["Phrase Duration (ms)"])
        if e.empty:
            parts.append(pd.DataFrame({"Syllable": [lbl], "Phrase Duration (ms)": [np.nan]}))
            continue

        e["Syllable"] = lbl
        parts.append(e[["Syllable", "Phrase Duration (ms)"]])

    tall = pd.concat(parts, ignore_index=True)
    tall["Syllable"] = pd.Categorical(tall["Syllable"], categories=order, ordered=True)
    return tall

def _violin_kwargs():
    params = inspect.signature(sns.violinplot).parameters
    if "density_norm" in params:   # seaborn >= 0.13
        return dict(inner="quartile", density_norm="width", color="lightgray")
    return dict(inner="quartile", scale="width", color="lightgray")

def _violin_plus_strip(ax, tall_df, y_limits, order: Sequence[str]):
    sns.set(style="white")
    if tall_df.empty:
        ax.text(0.5, 0.5, "No phrase durations", ha="center", va="center", fontsize=TITLE_FS-2)
        _pretty_axes(ax); ax.set_axis_off(); return
    sns.violinplot(
        x="Syllable", y="Phrase Duration (ms)",
        data=tall_df, order=order, **_violin_kwargs(), ax=ax
    )
    sns.stripplot(
        x="Syllable", y="Phrase Duration (ms)",
        data=tall_df, order=order,
        jitter=True, size=5, color="#2E4845", alpha=0.6, ax=ax
    )
    ax.set_ylim(y_limits)
    ax.set_xlabel("Syllable Label")
    ax.set_ylabel("Phrase Duration (ms)")
    _pretty_axes(ax, 0)

def _quick_duration_summary(df, labels, tag):
    c = Counter()
    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        if col in df.columns:
            c[lbl] += int(df[col].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0).sum())
    print(f"[DEBUG] {tag} — durations per label:", dict(c), " (rows:", len(df), ")")

def _build_daily_aggregate(df, labels, dt_series):
    cols = [f"syllable_{l}_durations" for l in labels if f"syllable_{l}_durations" in df.columns]
    if not cols:
        return pd.DataFrame(columns=["Date", "Phrase Duration (ms)"])
    def _concat(row):
        out = []
        for c in cols:
            v = row.get(c, [])
            if isinstance(v, (list, tuple)):
                out.extend([x for x in v if isinstance(x, (int, float))])
        return out
    tmp = df.copy()
    tmp["_all_durs"] = tmp.apply(_concat, axis=1)
    if tmp["_all_durs"].map(len).sum() == 0:
        return pd.DataFrame(columns=["Date", "Phrase Duration (ms)"])
    tmp["Date"] = pd.to_datetime(dt_series.dt.date)
    tall = tmp[["Date", "_all_durs"]].explode("_all_durs").dropna()
    tall = tall.rename(columns={"_all_durs": "Phrase Duration (ms)"})
    tall["Phrase Duration (ms)"] = pd.to_numeric(tall["Phrase Duration (ms)"], errors="coerce")
    return tall.dropna(subset=["Phrase Duration (ms)"])

def _collect_all_durations(df_subset: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    arrs = []
    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        if col in df_subset.columns:
            vals = df_subset[col].explode().dropna()
            if not vals.empty:
                arrs.append(pd.to_numeric(vals, errors="coerce").dropna().to_numpy())
    if not arrs:
        return np.array([], dtype=float)
    return np.concatenate(arrs, axis=0)

def _per_syllable_medians(df_subset: pd.DataFrame, labels: Sequence[str]) -> Dict[str, float]:
    out = {}
    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        if col in df_subset.columns:
            s = pd.to_numeric(df_subset[col].explode(), errors="coerce").dropna()
            if not s.empty:
                out[str(lbl)] = float(s.median())
    return out

def _per_syllable_variances(df_subset: pd.DataFrame, labels: Sequence[str]) -> Dict[str, float]:
    """Return {label -> variance of phrase duration (ms^2)} for a subset."""
    out = {}
    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        if col in df_subset.columns:
            s = pd.to_numeric(df_subset[col].explode(), errors="coerce").dropna()
            if s.size > 1:
                out[str(lbl)] = float(np.var(s.to_numpy(), ddof=1))  # sample variance
            elif s.size == 1:
                out[str(lbl)] = 0.0
    return out

# ───── Stats helpers (Kruskal–Wallis omnibus + pairwise Mann–Whitney with BH FDR) ─────
def _try_mannwhitney(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        ma, mb = np.mean(x), np.mean(y)
        va, vb = np.var(x, ddof=1), np.var(y, ddof=1)
        na, nb = len(x), len(y)
        t = (ma - mb) / np.sqrt(va/na + vb/nb + 1e-12)
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(t), float(p)

def _try_kruskal(groups: List[np.ndarray]) -> Tuple[float, float]:
    try:
        from scipy.stats import kruskal
        stat, p = kruskal(*groups)
        return float(stat), float(p)
    except Exception:
        ranks = np.argsort(np.argsort(np.concatenate(groups)))
        sizes = [len(g) for g in groups]
        splits = np.cumsum([0] + sizes)
        parts = [ranks[splits[i]:splits[i+1]] for i in range(len(sizes))]
        means = [np.mean(p) for p in parts]
        grand = np.mean(ranks)
        ss_between = sum(n * (m - grand)**2 for n, m in zip(sizes, means))
        ss_within = sum(((p - np.mean(p))**2).sum() for p in parts)
        dfb = len(groups) - 1
        dfw = len(ranks) - len(groups)
        F = (ss_between / (dfb + 1e-12)) / (ss_within / (dfw + 1e-12) + 1e-12)
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(F) / sqrt(2))))
        return float(F), float(p)

def _p_adjust_bh(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, int); ranks[order] = np.arange(1, n+1)
    adj = p * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty(n, float)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out.tolist()

def _stars(p: float) -> str:
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "n.s."

def _fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

def _calc_global_ylim(datasets, labels):
    vals = []
    for ds in datasets:
        t = _explode_for_plot(ds, labels)
        if not t.empty:
            vv = pd.to_numeric(t["Phrase Duration (ms)"], errors="coerce").dropna().to_list()
            vals.extend(vv)
    if not vals:
        return (0.0, 1.0)
    return (float(np.nanmin(vals)), float(np.nanmax(vals)))

# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────
def run_phrase_duration_pre_vs_post_grouped(
    decoded_database_json, output_dir, treatment_date,
    grouping_mode="explicit", early_group_size=100, late_group_size=100, post_group_size=100,
    only_song_present=True, restrict_to_labels=None, y_max_ms=None,
    show_plots=True, make_aggregate_plot=True,
) -> GroupedPlotsResult:

    decoded_path = Path(decoded_database_json)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: Call YOUR organizer (compute_durations=False so we use robust parser here)
    out = _ORGANIZE(
        decoded_database_json=decoded_path,
        only_song_present=only_song_present,
        compute_durations=False,          # we’ll compute robustly (handles flat pairs & bins)
        add_recording_datetime=True
    )
    df = out.organized_df.copy()
    if only_song_present and "song_present" in df.columns:
        df = df[df["song_present"] == True].copy()

    # Datetime + sort
    dt = _choose_datetime_series(df)
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

    # Labels as STRINGS, using your organizer’s discovered labels unless user restricts
    labels = [str(x) for x in (restrict_to_labels or out.unique_syllable_labels)]
    df, spans_col = _build_durations_table(df, labels)
    print("[INFO] Using spans column:", spans_col)

    t_date = _parse_date_like(treatment_date)
    pre, post = df[df["_dt"] < t_date].copy(), df[df["_dt"] >= t_date].copy()

    if grouping_mode == "explicit":
        early_n, late_n, post_n = map(int, [early_group_size, late_group_size, post_group_size])
        late_pre = pre.tail(late_n)
        early_pre = pre.iloc[max(0, len(pre) - late_n - early_n):max(0, len(pre) - late_n)]
        post_g = post.head(post_n)
        print(f"[INFO] explicit sizes → early={len(early_pre)}, late={len(late_pre)}, post={len(post_g)} "
              f"(requested early={early_n}, late={late_n}, post={post_n})")
    else:
        # auto-balance rule: n = min( len(pre)//2, len(post) )
        n = min(len(pre) // 2, len(post))
        if n <= 0:
            late_pre  = pre.iloc[0:0]
            early_pre = pre.iloc[0:0]
            post_g    = post.iloc[0:0]
            print("[INFO] auto_balance → n=0 (insufficient data for balanced groups)")
        else:
            late_pre  = pre.tail(n)
            early_pre = pre.iloc[len(pre) - 2*n : len(pre) - n]
            post_g    = post.head(n)
            print(f"[INFO] auto_balance(min(half-pre, post)) → n={n} "
                  f"(early_pre={len(early_pre)}, late_pre={len(late_pre)}, post={len(post_g)})")

    for nm, part in [("early_pre", early_pre), ("late_pre", late_pre), ("post", post_g)]:
        _quick_duration_summary(part, labels, nm)

    y_min, y_max = _calc_global_ylim([early_pre, late_pre, post_g], labels)
    if y_max_ms:
        y_max = float(y_max_ms)
    animal_id = _infer_animal_id(df, decoded_path)

    # — per-group violin+scatter (fixed x-order)
    def _make_plot(ds, title, n, fname):
        tall = _explode_for_plot(ds, labels)
        fig, ax = plt.subplots(figsize=(12, 6))
        _violin_plus_strip(ax, tall, (y_min, y_max), labels)
        ax.set_title(f"{animal_id} — {title} (N={n})", fontsize=TITLE_FS)
        fig.tight_layout()
        outp = output_dir / f"{animal_id}_{fname}_phrase_durations.png"
        fig.savefig(outp, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig)
        return outp

    ep = lp = po = None
    if len(early_pre): ep = _make_plot(early_pre, "Early Pre-Treatment", len(early_pre), "early_pre")
    if len(late_pre):  lp = _make_plot(late_pre,  "Late Pre-Treatment",  len(late_pre),  "late_pre")
    if len(post_g):    po = _make_plot(post_g,    "Post-Treatment",      len(post_g),    "post")

    # — per-day aggregate (no stats)
    agg = None
    if make_aggregate_plot:
        daily = _build_daily_aggregate(df, labels, df["_dt"])
        if not daily.empty:
            daily["DateStr"] = daily["Date"].dt.strftime("%Y-%m-%d")
            date_order = sorted(daily["DateStr"].unique())

            fig, ax = plt.subplots(figsize=(16, 6))
            sns.set(style="white")
            sns.boxplot(x="DateStr", y="Phrase Duration (ms)", data=daily,
                        order=date_order, color="lightgray", fliersize=0, ax=ax)
            sns.stripplot(x="DateStr", y="Phrase Duration (ms)", data=daily,
                          order=date_order, jitter=0.25, size=3, alpha=0.6, ax=ax, color="#2E4845")

            t_str = t_date.strftime("%Y-%m-%d")
            if t_str in date_order:
                idx = date_order.index(t_str)
                ax.axvline(idx, color="red", ls="--", lw=1.2)

            ax.set_xlabel("Recording Date"); ax.set_ylabel("Phrase Duration (ms)")
            _pretty_axes(ax, 90)
            if y_max_ms: ax.set_ylim(0, float(y_max_ms))
            ax.set_title(f"{animal_id} — Pre/Post Aggregate (per-day box + scatter)", fontsize=TITLE_FS)

            fig.tight_layout()
            agg = output_dir / f"{animal_id}_aggregate_pre_post_phrase_durations.png"
            fig.savefig(agg, dpi=300, transparent=False)
            if show_plots: plt.show()
            else: plt.close(fig)
        else:
            print("[WARN] Aggregate: no durations found to plot.")

    # — three-group aggregate with per-syllable median dots + stats (DURATION)
    agg3 = None
    agg3_auto = None
    agg3_pts = None

    three_groups = []
    group_defs = [(early_pre, "Early Pre"), (late_pre, "Late Pre"), (post_g, "Post")]
    for part, label in group_defs:
        if len(part):
            vals = _collect_all_durations(part, labels)
            if vals.size:
                three_groups.append(pd.DataFrame({"Group": label, "Phrase Duration (ms)": vals}))

    if three_groups:
        tall3 = pd.concat(three_groups, ignore_index=True)
        order = ["Early Pre", "Late Pre", "Post"]
        xpos = {g: i for i, g in enumerate(order)}

        # per-syllable medians for each group (duration)
        med_map = {glabel: _per_syllable_medians(gdf, labels) for gdf, glabel in group_defs}

        # arrays per group for stats
        group_arrays = {g: tall3.loc[tall3["Group"] == g, "Phrase Duration (ms)"].to_numpy() for g in order}

        # omnibus Kruskal–Wallis
        kw_stat, kw_p = _try_kruskal([group_arrays[g] for g in order])

        # pairwise Mann–Whitney + BH adjust
        pairs = list(combinations(order, 2))
        raw_p = []
        for a, b in pairs:
            _, p = _try_mannwhitney(group_arrays[a], group_arrays[b])
            raw_p.append(p)
        adj_p = _p_adjust_bh(raw_p)

        def _summary_text():
            lines = [f"Kruskal–Wallis: H={kw_stat:.3g}, p={_fmt_p(kw_p)}"]
            for (a, b), p_raw, p_adj in zip(pairs, raw_p, adj_p):
                lines.append(f"{a} vs {b}: p(BH)={_fmt_p(p_adj)}  [{_stars(p_adj)}]")
            return "\n".join(lines)

        def _draw_three_box(ax, set_ylim: bool, title: str):
            sns.boxplot(
                x="Group", y="Phrase Duration (ms)", data=tall3,
                order=order, color="lightgray",
                whis=(0, 100), showfliers=False, ax=ax
            )
            # overlay per-syllable medians with dashed connectors
            for lbl in [str(x) for x in labels]:
                xs, ys = [], []
                for g in order:
                    y = med_map.get(g, {}).get(lbl, None)
                    if y is not None:
                        xs.append(xpos[g]); ys.append(y)
                if xs:
                    ax.plot(xs, ys, linestyle="--", linewidth=1, color="0.4", alpha=0.7, zorder=3)
                    ax.scatter(xs, ys, s=28, edgecolor="white", linewidth=0.5, zorder=4)
            ax.set_xlabel("")
            ax.set_ylabel("Phrase Duration (ms)")
            _pretty_axes(ax, 0)
            if set_ylim and y_max_ms:
                ax.set_ylim(0, float(y_max_ms))
            ax.set_title(title, fontsize=TITLE_FS)
            ax.text(
                0.99, 0.98, _summary_text(),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=LABEL_FS-2,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9)
            )

        # Figure 1: with ylim (if provided)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        _draw_three_box(ax1, set_ylim=True, title=f"{animal_id} — Early/Late Pre vs Post (Syllable medians + boxes)")
        fig1.tight_layout()
        agg3 = output_dir / f"{animal_id}_aggregate_three_group_phrase_durations.png"
        fig1.savefig(agg3, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig1)

        # Figure 2: auto y-limits
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        _draw_three_box(ax2, set_ylim=False, title=f"{animal_id} — Early/Late Pre vs Post (auto y-limits)")
        fig2.tight_layout()
        agg3_auto = output_dir / f"{animal_id}_aggregate_three_group_phrase_durations_auto_ylim.png"
        fig2.savefig(agg3_auto, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig2)

        # Figure 3: boxes + one *point per syllable* per group (duration)
        rows_pts = []
        for g in order:
            m = med_map.get(g, {})
            for lbl, y in m.items():
                rows_pts.append({"Group": g, "Syllable": str(lbl), "Phrase Duration (ms)": y})
        tall3_pts = pd.DataFrame(rows_pts)

        if not tall3_pts.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.set(style="white")
            sns.boxplot(
                x="Group", y="Phrase Duration (ms)",
                data=tall3, order=order,
                color="lightgray", whis=(0, 100), showfliers=False, ax=ax3
            )
            sns.stripplot(
                x="Group", y="Phrase Duration (ms)",
                data=tall3_pts, order=order,
                hue="Syllable", dodge=False, jitter=0.12, size=5, alpha=0.9, ax=ax3
            )
            ax3.set_xlabel("")
            ax3.set_ylabel("Phrase Duration (ms)")
            _pretty_axes(ax3, 0)
            if y_max_ms:
                ax3.set_ylim(0, float(y_max_ms))
            ax3.set_title(f"{animal_id} — Early/Late Pre vs Post (boxes + syllable points)", fontsize=TITLE_FS)
            ax3.legend(title="Syllable", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig3.tight_layout()
            agg3_pts = output_dir / f"{animal_id}_aggregate_three_group_phrase_durations_syllable_points.png"
            fig3.savefig(agg3_pts, dpi=300, transparent=False)
            if show_plots: plt.show()
            else: plt.close(fig3)
    else:
        print("[WARN] Three-group aggregate: no durations found to plot.")

    # ──────────────────────────────────────────────────────────────────────────
    # Variance analyses (per-syllable)
    # ──────────────────────────────────────────────────────────────────────────
    var_plot = None             # DISABLED: no per-syllable variance line plot
    var_boxscatter = None
    agg3_var_pts = None         # boxes + syllable points (ylim fixed to 2.0e7)
    agg3_var_stats = None       # NEW: boxes + jitter + stats (ylim fixed to 2.0e7)

    var_maps: Dict[str, Dict[str, float]] = {
        "Early Pre": _per_syllable_variances(early_pre, labels) if len(early_pre) else {},
        "Late Pre":  _per_syllable_variances(late_pre,  labels) if len(late_pre)  else {},
        "Post":      _per_syllable_variances(post_g,    labels) if len(post_g)    else {},
    }

    rows_var = []
    def _sort_key(lbl: str):
        try: return (0, int(lbl))
        except: return (1, lbl)
    syllables_sorted = sorted({str(l) for l in labels}, key=_sort_key)

    for g in ["Early Pre", "Late Pre", "Post"]:
        m = var_maps.get(g, {})
        for lbl in syllables_sorted:
            if lbl in m:
                rows_var.append({"Syllable": lbl, "Variance (ms^2)": m[lbl], "Group": g})

    if rows_var:
        df_var = pd.DataFrame(rows_var)

        # 1) Box + scatter of variances per syllable (x = syllable, points colored by group)
        fig_b, ax_b = plt.subplots(figsize=(12, 6))
        sns.set(style="white")
        sns.boxplot(
            x="Syllable", y="Variance (ms^2)",
            data=df_var, order=syllables_sorted,
            color="lightgray", fliersize=0, ax=ax_b
        )
        sns.stripplot(
            x="Syllable", y="Variance (ms^2)",
            data=df_var, order=syllables_sorted,
            hue="Group", dodge=True, jitter=0.15, size=5, alpha=0.9, ax=ax_b
        )
        ax_b.set_xlabel("Syllable Label")
        ax_b.set_ylabel("Variance of Phrase Duration (ms²)")
        ax_b.set_title(f"{animal_id} — Per-Syllable Variance (box + scatter by group)", fontsize=TITLE_FS)
        _pretty_axes(ax_b, x_rotation=0)
        ax_b.legend(title="Group", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig_b.tight_layout()
        var_boxscatter = output_dir / f"{animal_id}_per_syllable_variance_box_scatter.png"
        fig_b.savefig(var_boxscatter, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig_b)

        # 2) Three-group variance: boxes + syllable points (ylim fixed to 2.0e7)
        order = ["Early Pre", "Late Pre", "Post"]
        fig_g, ax_g = plt.subplots(figsize=(10, 6))
        sns.set(style="white")
        sns.boxplot(
            x="Group", y="Variance (ms^2)",
            data=df_var, order=order,
            color="lightgray", whis=(0, 100), showfliers=False, ax=ax_g
        )
        sns.stripplot(
            x="Group", y="Variance (ms^2)",
            data=df_var, order=order,
            hue="Syllable", dodge=False, jitter=0.12, size=5, alpha=0.9, ax=ax_g
        )
        ax_g.set_xlabel("")
        ax_g.set_ylabel("Variance of Phrase Duration (ms²)")
        ax_g.set_title(f"{animal_id} — Early/Late Pre vs Post (variance: boxes + syllable points)", fontsize=TITLE_FS)
        _pretty_axes(ax_g, 0)
        ax_g.legend(title="Syllable", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax_g.set_ylim(0, 2.0e7)  # fixed per request
        fig_g.tight_layout()
        agg3_var_pts = output_dir / f"{animal_id}_aggregate_three_group_variance_syllable_points.png"
        fig_g.savefig(agg3_var_pts, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig_g)

        # 3) NEW: Three-group variance: boxes + jitter + stats (ylim fixed to 2.0e7)
        fig_s, ax_s = plt.subplots(figsize=(10, 6))
        sns.set(style="white")
        sns.boxplot(
            x="Group", y="Variance (ms^2)",
            data=df_var, order=order,
            color="lightgray", whis=(0, 100), showfliers=False, ax=ax_s
        )
        sns.stripplot(
            x="Group", y="Variance (ms^2)",
            data=df_var, order=order,
            dodge=False, jitter=0.15, size=5, alpha=0.9, ax=ax_s
        )
        ax_s.set_xlabel("")
        ax_s.set_ylabel("Variance of Phrase Duration (ms²)")
        ax_s.set_title(f"{animal_id} — Early/Late Pre vs Post (variance: boxes + jitter + stats)", fontsize=TITLE_FS)
        _pretty_axes(ax_s, 0)
        ax_s.set_ylim(0, 2.0e7)  # fixed per request

        # Stats on variance distributions (one value per syllable per group)
        var_arrays = {g: df_var.loc[df_var["Group"] == g, "Variance (ms^2)"].to_numpy() for g in order}
        kw_stat_v, kw_p_v = _try_kruskal([var_arrays[g] for g in order])
        pairs = list(combinations(order, 2))
        raw_p_v = []
        for a, b in pairs:
            _, pv = _try_mannwhitney(var_arrays[a], var_arrays[b])
            raw_p_v.append(pv)
        adj_p_v = _p_adjust_bh(raw_p_v)

        lines = [f"Kruskal–Wallis: H={kw_stat_v:.3g}, p={_fmt_p(kw_p_v)}"]
        for (a, b), p_adj in zip(pairs, adj_p_v):
            lines.append(f"{a} vs {b}: p(BH)={_fmt_p(p_adj)}  [{_stars(p_adj)}]")
        ax_s.text(
            0.99, 0.98, "\n".join(lines),
            transform=ax_s.transAxes, ha="right", va="top",
            fontsize=LABEL_FS-2,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9)
        )

        fig_s.tight_layout()
        agg3_var_stats = output_dir / f"{animal_id}_aggregate_three_group_variance_box_scatter_STATS.png"
        fig_s.savefig(agg3_var_stats, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig_s)
    else:
        print("[WARN] Per-syllable variance plots: no variance data available.")

    return GroupedPlotsResult(
        early_pre_path=ep,
        late_pre_path=lp,
        post_path=po,
        aggregate_path=agg,
        aggregate_three_box_path=agg3,
        aggregate_three_box_auto_ylim_path=agg3_auto,
        aggregate_three_box_with_syll_points_path=agg3_pts,
        per_syllable_variance_path=None,                       # None (disabled)
        per_syllable_variance_boxscatter_path=var_boxscatter,
        aggregate_three_box_variance_with_syll_points_path=agg3_var_pts,
        aggregate_three_box_variance_with_stats_path=agg3_var_stats,
        syllable_labels=[str(x) for x in labels],
        y_limits=(y_min, y_max),
    )

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pre/Post grouped phrase-duration plots per syllable + aggregate.")
    p.add_argument("decoded_database_json", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("treatment_date", type=str, help='e.g., "2025-03-04" or "2025.03.04"')
    p.add_argument("--grouping_mode", type=str, default="explicit",
                   choices=["explicit", "auto_balance"],
                   help="explicit: use the provided sizes; "
                        "auto_balance: n = min(len(pre)//2, len(post)) for equal-sized groups")
    p.add_argument("--early_group_size", type=int, default=100)
    p.add_argument("--late_group_size", type=int, default=100)
    p.add_argument("--post_group_size", type=int, default=100)
    p.add_argument("--only_song_present", action="store_true", default=True)
    p.add_argument("--y_max_ms", type=float, default=None)
    p.add_argument("--labels", type=str, nargs="*", default=None, help="Restrict to these labels (e.g., 0 1 2 ...)")
    p.add_argument("--no_show", action="store_true", help="Disable plt.show()")
    p.add_argument("--no_aggregate", action="store_true", help="(legacy; ignored—aggregate stays on)")
    args = p.parse_args()

    _ = run_phrase_duration_pre_vs_post_grouped(
        decoded_database_json=args.decoded_database_json,
        output_dir=args.output_dir,
        treatment_date=args.treatment_date,
        grouping_mode=args.grouping_mode,
        early_group_size=args.early_group_size,
        late_group_size=args.late_group_size,
        post_group_size=args.post_group_size,
        only_song_present=args.only_song_present,
        restrict_to_labels=args.labels,
        y_max_ms=args.y_max_ms,
        show_plots=not args.no_show,
        make_aggregate_plot=True,
    )

"""
# Example (locks axis to 0..25)
from phrase_duration_pre_vs_post_grouped import run_phrase_duration_pre_vs_post_grouped

decoded = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2024/USA5283_decoded_database.json"
outdir  = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2024/USA5283_figures"
tdate   = "2024-03-05"

res = run_phrase_duration_pre_vs_post_grouped(
    decoded_database_json=decoded,
    output_dir=outdir,
    treatment_date=tdate,
    grouping_mode="auto_balance",
    restrict_to_labels=[str(i) for i in range(26)],  # force 0..25 on x-axis
    y_max_ms=40000,
    only_song_present=True,
    show_plots=True,
)
"""
