# -*- coding: utf-8 -*-
# phrase_duration_pre_vs_post_grouped.py  (standalone; auto-merge if no premerged DF provided)
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
# External merge builder (required to auto-merge)
# ──────────────────────────────────────────────────────────────────────────────
try:
    # Must be in your PYTHONPATH / same folder
    from merge_annotations_from_split_songs import build_decoded_with_split_labels
except Exception as e:
    build_decoded_with_split_labels = None
    _MERGE_IMPORT_ERR = e

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
    aggregate_three_box_with_syll_points_path: Optional[Path]
    per_syllable_variance_path: Optional[Path]              # (kept for API; None)
    per_syllable_variance_boxscatter_path: Optional[Path]
    aggregate_three_box_variance_with_syll_points_path: Optional[Path]
    aggregate_three_box_variance_with_stats_path: Optional[Path]
    syllable_labels: List[str]
    y_limits: Tuple[float, float]
    # New: per-syllable phrase-duration stats for Early/Late/Post
    phrase_duration_stats_df: pd.DataFrame

# ──────────────────────────────────────────────────────────────────────────────
# Utilities: dates, parsing, labels, durations extraction
# ──────────────────────────────────────────────────────────────────────────────
def _parse_date_like(s: Union[str, pd.Timestamp, None]) -> Union[pd.Timestamp, pd.NaT]:
    if s is None:
        return pd.NaT
    if isinstance(s, pd.Timestamp):
        return s
    s2 = str(s).replace(".", "-").replace("/", "-")
    return pd.to_datetime(s2, errors="coerce")

def _choose_datetime_series(df: pd.DataFrame) -> pd.Series:
    dt = df.get("Recording DateTime", pd.Series([pd.NaT]*len(df), index=df.index)).copy()
    # fallbacks
    for cand in ("recording_datetime", "creation_date"):
        if cand in df.columns:
            need = dt.isna()
            if need.any():
                dt.loc[need] = pd.to_datetime(df.loc[need, cand], errors="coerce")
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

def _infer_animal_id(df: Optional[pd.DataFrame], decoded_path: Optional[Path]) -> str:
    if df is not None:
        for col in ["animal_id", "Animal", "Animal ID"]:
            if col in df.columns and pd.notna(df[col]).any():
                val = str(df[col].dropna().iloc[0]).strip()
                if val:
                    return val
        # Try to parse from a file_name column (common in your tables)
        if "file_name" in df.columns and df["file_name"].notna().any():
            import re
            for s in df["file_name"].astype(str):
                m = re.search(r'(usa\d{4,6})', s, flags=re.IGNORECASE)
                if m:
                    return m.group(1).upper()
    if decoded_path is not None:
        tok = decoded_path.stem.split("_")[0]
        if tok:
            return tok
    return "unknown_animal"

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
    out: List[float] = []
    if spans is None:
        return out

    if isinstance(spans, (list, tuple)) and len(spans) == 2 and all(isinstance(x, (int, float)) for x in spans):
        spans = [spans]
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
            using_bins = False

        try:
            dur = float(off) - float(on)
            if dur < 0:
                continue
            if treat_as_bins or using_bins:
                if ms_per_bin:
                    dur *= float(ms_per_bin)
                else:
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
    for c in ["syllable_onsets_offsets_ms_dict", "syllable_onsets_offsets_ms", "onsets_offsets_ms_dict"]:
        if c in df.columns:
            return c
    for c in ["syllable_onsets_offsets_timebins", "syllable_onsets_offsets_timebins_dict"]:
        if c in df.columns:
            return c
    # Heuristic: any *_dict with content
    for c in df.columns:
        lc = c.lower()
        if lc.endswith("_dict") and df[c].notna().any():
            return c
    return None

def _collect_unique_labels_sorted(df: pd.DataFrame, dict_col: str) -> List[str]:
    labs: List[str] = []
    for d in df[dict_col].dropna():
        if isinstance(d, dict):
            labs.extend(list(map(str, d.keys())))
        else:
            dd = _maybe_parse_dict(d)
            if isinstance(dd, dict):
                labs.extend(list(map(str, dd.keys())))
    # preserve first-seen order
    labs = list(dict.fromkeys(labs))
    # numeric sort when possible
    try:
        as_int = [int(x) for x in labs]
        order = np.argsort(as_int)
        return [labs[i] for i in order]
    except Exception:
        return sorted(labs)

def _build_durations_table(df: pd.DataFrame, labels: Sequence[str]) -> tuple[pd.DataFrame, str | None]:
    col = _find_best_spans_column(df)
    if col is None:
        return df.copy(), None

    per_song = df.apply(lambda r: _collect_phrase_durations_per_song(r, col), axis=1)
    out = df.copy()
    for lbl in labels:
        out[f"syllable_{lbl}_durations"] = per_song.apply(lambda d: d.get(str(lbl), []))
    return out, col

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def _explode_for_plot(dataset: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
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
    out = {}
    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        if col in df_subset.columns:
            s = pd.to_numeric(df_subset[col].explode(), errors="coerce").dropna()
            if s.size > 1:
                out[str(lbl)] = float(np.var(s.to_numpy(), ddof=1))
            elif s.size == 1:
                out[str(lbl)] = 0.0
    return out

def _build_phrase_duration_stats_df(
    early_pre: pd.DataFrame,
    late_pre: pd.DataFrame,
    post_g: pd.DataFrame,
    labels: Sequence[str],
) -> pd.DataFrame:
    """
    Build a long-form DataFrame with per-syllable phrase-duration stats
    for each group (Early Pre, Late Pre, Post).

    Columns:
        Group          : "Early Pre" / "Late Pre" / "Post"
        Syllable       : syllable label (as string)
        N_phrases      : number of phrase-duration samples
        Mean_ms        : mean phrase duration (ms)
        SEM_ms         : standard error of the mean (ms)
        Median_ms      : median phrase duration (ms)
        Std_ms         : standard deviation of phrase duration (ms)
        Variance_ms2   : variance of phrase duration (ms^2)
    """
    groups = [
        ("Early Pre", early_pre),
        ("Late Pre",  late_pre),
        ("Post",      post_g),
    ]

    rows = []
    for gname, df_subset in groups:
        if df_subset is None or df_subset.empty:
            continue

        for lbl in labels:
            col = f"syllable_{lbl}_durations"
            if col not in df_subset.columns:
                continue

            s = df_subset[col].explode()
            s = pd.to_numeric(s, errors="coerce").dropna()
            n = int(s.size)
            if n == 0:
                continue

            arr = s.to_numpy(dtype=float)
            mean = float(arr.mean())
            if n > 1:
                std = float(arr.std(ddof=1))
                var = float(arr.var(ddof=1))
                sem = float(std / math.sqrt(n))
            else:
                std = 0.0
                var = 0.0
                sem = 0.0
            median = float(np.median(arr))

            rows.append({
                "Group": gname,
                "Syllable": str(lbl),
                "N_phrases": n,
                "Mean_ms": mean,
                "SEM_ms": sem,
                "Median_ms": median,
                "Std_ms": std,
                "Variance_ms2": var,
            })

    return pd.DataFrame(
        rows,
        columns=[
            "Group",
            "Syllable",
            "N_phrases",
            "Mean_ms",
            "SEM_ms",
            "Median_ms",
            "Std_ms",
            "Variance_ms2",
        ],
    )

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
# Main function (auto-merge if no premerged DF/path is provided)
# ──────────────────────────────────────────────────────────────────────────────
def run_phrase_duration_pre_vs_post_grouped(
    *,
    # If you already have a merged DF/path, pass one of these:
    premerged_annotations_df: Optional[pd.DataFrame] = None,
    premerged_annotations_path: Optional[Union[str, Path]] = None,

    # Otherwise we will auto-merge using these:
    decoded_database_json: Optional[Union[str, Path]] = None,
    song_detection_json: Optional[Union[str, Path]] = None,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,

    # Plotting / grouping
    output_dir: Optional[Union[str, Path]] = None,
    treatment_date: Union[str, pd.Timestamp] = None,
    grouping_mode: str = "explicit",      # "explicit" or "auto_balance"
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    y_max_ms: Optional[float] = None,
    show_plots: bool = True,

    # Optional override for titles/filenames
    animal_id_override: Optional[str] = None,
) -> GroupedPlotsResult:

    # 1) Decide source DF
    df = None
    if premerged_annotations_df is not None:
        df = premerged_annotations_df.copy()

    elif premerged_annotations_path is not None:
        p = Path(premerged_annotations_path)
        if not p.exists():
            raise FileNotFoundError(f"premerged_annotations_path not found: {p}")
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".json", ".ndjson"}:
            try:
                df = pd.read_json(p)
            except ValueError:
                df = pd.read_json(p, lines=True)
        else:
            raise ValueError(f"Unsupported file type for premerged_annotations_path: {p.suffix}")

    else:
        # Auto-merge path
        if build_decoded_with_split_labels is None:
            raise ImportError(
                "merge_annotations_from_split_songs.build_decoded_with_split_labels could not be imported.\n"
                f"Original import error: {_MERGE_IMPORT_ERR}"
            )
        if decoded_database_json is None or song_detection_json is None:
            raise ValueError(
                "To auto-merge, you must provide both decoded_database_json and song_detection_json."
            )
        decoded_database_json = Path(decoded_database_json)
        song_detection_json  = Path(song_detection_json)

        ann = build_decoded_with_split_labels(
            decoded_database_json=decoded_database_json,
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
        df = ann.annotations_appended_df.copy()

    if df is None or df.empty:
        raise ValueError("No data available after reading/merging annotations.")

    # 2) Output dir and animal id
    decoded_path = Path(decoded_database_json) if decoded_database_json else None
    if output_dir is None:
        base = (decoded_path.parent if decoded_path is not None else Path.cwd())
        output_dir = base / "phrase_durations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    animal_id = animal_id_override or _infer_animal_id(df, decoded_path)

    # 3) Datetime + sort
    dt = _choose_datetime_series(df)
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

    # 4) Labels
    if restrict_to_labels is None:
        col_hint = _find_best_spans_column(df)
        if col_hint is None:
            raise ValueError("Could not locate a syllable spans column (e.g., 'syllable_onsets_offsets_ms_dict').")
        labels = _collect_unique_labels_sorted(df, col_hint)
    else:
        labels = [str(x) for x in restrict_to_labels]

    # 5) Build per-syllable duration lists
    df, spans_col = _build_durations_table(df, labels)
    print("[INFO] Using spans column:", spans_col)

    # 6) Split by treatment date
    t_date = _parse_date_like(treatment_date)
    if pd.isna(t_date):
        raise ValueError("Valid treatment_date is required (e.g., '2025-05-22').")
    pre, post = df[df["_dt"] < t_date].copy(), df[df["_dt"] >= t_date].copy()

    # 7) Balanced groups
    if grouping_mode == "explicit":
        early_n, late_n, post_n = map(int, [early_group_size, late_group_size, post_group_size])
        late_pre = pre.tail(late_n)
        early_pre = pre.iloc[max(0, len(pre) - late_n - early_n):max(0, len(pre) - late_n)]
        post_g = post.head(post_n)
        print(f"[INFO] explicit sizes → early={len(early_pre)}, late={len(late_pre)}, post={len(post_g)} "
              f"(requested early={early_n}, late={late_n}, post={post_n})")
    else:
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

    # — three-group aggregate with per-syllable median dots + stats (duration)
    agg3 = agg3_auto = agg3_pts = None
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
        med_map = {glabel: _per_syllable_medians(gdf, labels) for gdf, glabel in group_defs}
        group_arrays = {g: tall3.loc[tall3["Group"] == g, "Phrase Duration (ms)"].to_numpy() for g in order}
        kw_stat, kw_p = _try_kruskal([group_arrays[g] for g in order])

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
            # dashed connectors for per-syllable medians
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

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        _draw_three_box(ax1, set_ylim=True, title=f"{animal_id} — Early/Late Pre vs Post (Syllable medians + boxes)")
        fig1.tight_layout()
        agg3 = output_dir / f"{animal_id}_aggregate_three_group_phrase_durations.png"
        fig1.savefig(agg3, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        _draw_three_box(ax2, set_ylim=False, title=f"{animal_id} — Early/Late Pre vs Post (auto y-limits)")
        fig2.tight_layout()
        agg3_auto = output_dir / f"{animal_id}_aggregate_three_group_phrase_durations_auto_ylim.png"
        fig2.savefig(agg3_auto, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig2)

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
    var_plot = None             # kept for API
    var_boxscatter = None
    agg3_var_pts = None
    agg3_var_stats = None

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

        # 2) Three-group variance: boxes + syllable points (ylim fixed)
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
        ax_g.set_ylim(0, 2.0e7)
        fig_g.tight_layout()
        agg3_var_pts = output_dir / f"{animal_id}_aggregate_three_group_variance_syllable_points.png"
        fig_g.savefig(agg3_var_pts, dpi=300, transparent=False)
        if show_plots: plt.show()
        else: plt.close(fig_g)

        # 3) Three-group variance: boxes + jitter + stats (ylim fixed)
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
        ax_s.set_ylim(0, 2.0e7)

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

    # Build per-syllable phrase-duration summary stats table
    stats_df = _build_phrase_duration_stats_df(early_pre, late_pre, post_g, labels)

    return GroupedPlotsResult(
        early_pre_path=ep,
        late_pre_path=lp,
        post_path=po,
        aggregate_path=agg,
        aggregate_three_box_path=agg3,
        aggregate_three_box_auto_ylim_path=agg3_auto,
        aggregate_three_box_with_syll_points_path=agg3_pts,
        per_syllable_variance_path=None,
        per_syllable_variance_boxscatter_path=var_boxscatter,
        aggregate_three_box_variance_with_syll_points_path=agg3_var_pts,
        aggregate_three_box_variance_with_stats_path=agg3_var_stats,
        syllable_labels=[str(x) for x in labels],
        y_limits=(y_min, y_max),
        phrase_duration_stats_df=stats_df,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Batch wrapper: use Excel metadata to drive per-animal runs
# ──────────────────────────────────────────────────────────────────────────────
def run_batch_phrase_duration_from_excel(
    excel_path: Union[str, Path],
    json_root: Union[str, Path],
    *,
    sheet_name: Union[int, str] = 0,
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    grouping_mode: str = "auto_balance",    # or "explicit"
    early_group_size: int = 100,
    late_group_size: int = 100,
    post_group_size: int = 100,
    restrict_to_labels: Optional[Sequence[Union[str, int]]] = None,
    y_max_ms: Optional[float] = None,
    show_plots: bool = True,
) -> Dict[str, GroupedPlotsResult]:
    """
    Batch wrapper that uses an Excel metadata sheet to supply treatment dates to
    the phrase-duration analysis for multiple animals.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel file with metadata.
    json_root : str or Path
        Root folder containing subfolders / JSON files for each animal.
        We search recursively under this root for:
            *{animal_id}*decoded_database.json
            *{animal_id}*song_detection.json
    sheet_name : int or str, default 0
        Sheet index or name for pd.read_excel.
    id_col : str, default "Animal ID"
        Column in the Excel file containing animal IDs (e.g., "USA5505").
    treatment_date_col : str, default "Treatment date"
        Column in the Excel file containing the treatment date for each animal.
    grouping_mode : {"auto_balance", "explicit"}, default "auto_balance"
        Passed through to run_phrase_duration_pre_vs_post_grouped.
    early_group_size, late_group_size, post_group_size : int
        Group sizes when grouping_mode="explicit".
    restrict_to_labels : sequence of str or int, optional
        If provided, restrict to these syllable labels.
    y_max_ms : float, optional
        Fixed y-axis upper limit in ms for duration plots.
    show_plots : bool, default True
        Whether to display plots interactively (Spyder / Jupyter).

    Returns
    -------
    results : dict
        Mapping {animal_id -> GroupedPlotsResult}
    """
    excel_path = Path(excel_path)
    json_root = Path(json_root)

    # 1) Read metadata from Excel
    meta_df = pd.read_excel(excel_path, sheet_name=sheet_name)

    if id_col not in meta_df.columns:
        raise ValueError(f"Column '{id_col}' not found in Excel file: {excel_path}")
    if treatment_date_col not in meta_df.columns:
        raise ValueError(f"Column '{treatment_date_col}' not found in Excel file: {excel_path}")

    # 2) Build mapping: animal_id -> treatment_date (first non-null per animal)
    animal_to_tdate: Dict[str, Union[str, pd.Timestamp, None]] = {}
    for aid, group in meta_df.groupby(id_col):
        vals = group[treatment_date_col].dropna().unique()
        tdate = vals[0] if len(vals) > 0 else None
        animal_to_tdate[str(aid)] = tdate

    # 3) Helper to find JSON files for each animal
    def _find_json_for_animal(
        root: Path,
        animal_id: str,
        decoded_suffix: str = "decoded_database.json",
        detect_suffix: str = "song_detection.json",
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Search recursively under `root` for:
          - *animal_id*decoded_suffix
          - *animal_id*detect_suffix

        Ignores macOS resource-fork files that start with '._'.
        """
        decoded_candidates = [
            p for p in root.rglob(f"*{animal_id}*{decoded_suffix}")
            if not p.name.startswith("._")
        ]
        detect_candidates = [
            p for p in root.rglob(f"*{animal_id}*{detect_suffix}")
            if not p.name.startswith("._")
        ]

        decoded_path = decoded_candidates[0] if decoded_candidates else None
        detect_path = detect_candidates[0] if detect_candidates else None
        return decoded_path, detect_path

    # 4) Iterate over animals and call the phrase-duration function
    results: Dict[str, GroupedPlotsResult] = {}

    # Normalize restrict_to_labels to strings
    if restrict_to_labels is not None:
        restrict_str: Optional[Sequence[str]] = [str(x) for x in restrict_to_labels]
    else:
        restrict_str = None

    for animal_id, tdate in animal_to_tdate.items():
        # Skip animals with no date
        if tdate is None or (isinstance(tdate, float) and pd.isna(tdate)):
            print(f"[WARN] {animal_id}: no valid treatment date in Excel, skipping.")
            continue

        decoded_path, detect_path = _find_json_for_animal(json_root, animal_id)

        if decoded_path is None or detect_path is None:
            print(
                f"[WARN] {animal_id}: could not find both JSONs under {json_root}.\n"
                f"       decoded: {decoded_path}\n"
                f"       detect : {detect_path}"
            )
            continue

        outdir = decoded_path.parent / "figures" / "phrase_durations"
        outdir.mkdir(parents=True, exist_ok=True)

        print(
            f"[RUN] {animal_id} | treatment_date={tdate} | "
            f"decoded={decoded_path.name} | detect={detect_path.name}"
        )

        res = run_phrase_duration_pre_vs_post_grouped(
            decoded_database_json=decoded_path,
            song_detection_json=detect_path,
            max_gap_between_song_segments=500,
            segment_index_offset=0,
            merge_repeated_syllables=True,
            repeat_gap_ms=10.0,
            repeat_gap_inclusive=False,
            output_dir=outdir,
            treatment_date=tdate,              # ← treatment date from Excel
            grouping_mode=grouping_mode,
            early_group_size=early_group_size,
            late_group_size=late_group_size,
            post_group_size=post_group_size,
            restrict_to_labels=restrict_str,
            y_max_ms=y_max_ms,
            show_plots=show_plots,
            animal_id_override=animal_id,
        )

        results[animal_id] = res

    return results

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description=("Phrase-duration grouped plots using merged annotations.\n"
                     "If --premerged is not provided, the script will MERGE automatically "
                     "using --annotations and --detect (via build_decoded_with_split_labels).")
    )
    # Option A: premerged file
    p.add_argument("--premerged", type=str, default=None,
                   help="Path to pre-merged annotations table (CSV or JSON).")

    # Option B: auto-merge inputs
    p.add_argument("--annotations", type=str, default=None,
                   help="Path to *_decoded_database.json (required if not using --premerged).")
    p.add_argument("--detect", type=str, default=None,
                   help="Path to *_song_detection.json (required if not using --premerged).")

    # Merge knobs (used only when auto-merging)
    p.add_argument("--ann-gap-ms", type=int, default=500)
    p.add_argument("--seg-offset", type=int, default=0)
    p.add_argument("--merge-repeats", action="store_true")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0)
    p.add_argument("--repeat-gap-inclusive", action="store_true")

    # Outputs / plotting
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (default: <annotations>.parent/phrase_durations or CWD/phrase_durations)")
    p.add_argument("--treatment-date", type=str, required=True)
    p.add_argument("--grouping_mode", type=str, default="auto_balance",
                   choices=["explicit", "auto_balance"])
    p.add_argument("--early_group_size", type=int, default=100)
    p.add_argument("--late_group_size", type=int, default=100)
    p.add_argument("--post_group_size", type=int, default=100)
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="Restrict to these labels (e.g., 0 1 2 ...). If omitted, inferred from data.")
    p.add_argument("--y_max_ms", type=float, default=None)
    p.add_argument("--animal-id", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    return p

def main():
    p = _build_arg_parser()
    args = p.parse_args()

    # Resolve label list
    restrict = [str(x) for x in args.labels] if args.labels else None

    res = run_phrase_duration_pre_vs_post_grouped(
        premerged_annotations_df=None,
        premerged_annotations_path=args.premerged,

        decoded_database_json=args.annotations,
        song_detection_json=args.detect,
        max_gap_between_song_segments=args.ann_gap_ms,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,

        output_dir=args.outdir,
        treatment_date=args.treatment_date,
        grouping_mode=args.grouping_mode,
        early_group_size=args.early_group_size,
        late_group_size=args.late_group_size,
        post_group_size=args.post_group_size,
        restrict_to_labels=restrict,
        y_max_ms=args.y_max_ms,
        show_plots=not args.no_show,
        animal_id_override=args.animal_id,
    )

    print("\n[OK] Plots saved:")
    print("  early_pre:", res.early_pre_path)
    print("  late_pre :", res.late_pre_path)
    print("  post     :", res.post_path)
    print("  aggregate:", res.aggregate_path)
    print("  3-group  :", res.aggregate_three_box_path)
    print("  3-group (auto y):", res.aggregate_three_box_auto_ylim_path)
    print("  3-group + syll pts:", res.aggregate_three_box_with_syll_points_path)
    print("  variance box+scatter:", res.per_syllable_variance_boxscatter_path)
    print("  variance 3-group pts :", res.aggregate_three_box_variance_with_syll_points_path)
    print("  variance 3-group stats:", res.aggregate_three_box_variance_with_stats_path)
    print("\n[INFO] Phrase-duration stats dataframe shape:", res.phrase_duration_stats_df.shape)

if __name__ == "__main__":
    main()


"""
Example interactive usage (single animal):

from pathlib import Path
import phrase_duration_pre_vs_post_grouped as pdpg

detect  = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/USA5288/USA5288_song_detection.json")
decoded = Path("/Volumes/my_own_ssd/2024_2025_Area_X_jsons_npzs/USA5288/USA5288_decoded_database.json")

outdir = decoded.parent / "figures" / "phrase_durations"
outdir.mkdir(parents=True, exist_ok=True)

res = pdpg.run_phrase_duration_pre_vs_post_grouped(
    decoded_database_json=decoded,
    song_detection_json=detect,
    max_gap_between_song_segments=500,
    segment_index_offset=0,
    merge_repeated_syllables=True,
    repeat_gap_ms=10.0,
    repeat_gap_inclusive=False,
    output_dir=outdir,
    treatment_date="2024-04-09",
    grouping_mode="auto_balance",
    restrict_to_labels=[str(i) for i in range(26)],
    y_max_ms=40000,
    show_plots=True,
)
print(res.phrase_duration_stats_df.head())

"""
