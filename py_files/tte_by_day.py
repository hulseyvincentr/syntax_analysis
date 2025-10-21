# -*- coding: utf-8 -*-
# tte_by_day.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import argparse
import inspect
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for stats (kept for future extensions; not used in 3-group plot)
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import strategy (prefer Excel-serial-based builder)
# ──────────────────────────────────────────────────────────────────────────────
_ORGANIZE_DEFAULT: Optional[Callable] = None
try:
    from organized_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _ORGANIZE_DEFAULT  # type: ignore
    )
except Exception:
    try:
        from organize_decoded_serialTS_segments import (
            build_organized_segments_with_durations as _ORGANIZE_DEFAULT  # type: ignore
        )
    except Exception:
        try:
            from organize_decoded_with_segments import (
                build_organized_segments_with_durations as _ORGANIZE_DEFAULT  # type: ignore
            )
        except Exception:
            _ORGANIZE_DEFAULT = None


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _safe_pad(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return "00"
    return str(s).zfill(2)

def _date_str(d) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")

def _most_common(strings: Iterable[str]) -> Optional[str]:
    from collections import Counter
    c = Counter(s for s in strings if s)
    return c.most_common(1)[0][0] if c else None

def _extract_id_from_text(text: str) -> Optional[str]:
    m = re.search(r'(usa\d{4,6})', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'(R\d{1,3})(?=[_\W]|$)', text)
    if m:
        return m.group(1)
    return None

def _infer_animal_id_from_df(organized_df: pd.DataFrame) -> Optional[str]:
    for col in ["animal_id", "Animal_ID", "animalID"]:
        if col in organized_df.columns and organized_df[col].notna().any():
            return _most_common(organized_df[col].dropna().astype(str))
    if "file_name" in organized_df.columns:
        matches = []
        for fn in organized_df["file_name"].dropna().astype(str):
            mid = _extract_id_from_text(fn)
            if mid:
                matches.append(mid)
        return _most_common(matches) if matches else None
    return None

def _infer_animal_id(decoded_path: Path,
                     metadata_path: Optional[Path] = None,
                     organized_df: Optional[pd.DataFrame] = None,
                     od: Optional[object] = None) -> str:
    if od is not None:
        for attr in ["animal_id", "animalID", "Animal_ID"]:
            if hasattr(od, attr):
                val = getattr(od, attr)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    if organized_df is not None:
        guess = _infer_animal_id_from_df(organized_df)
        if guess:
            return guess
    texts: List[str] = [str(decoded_path), decoded_path.stem]
    if metadata_path is not None:
        texts.extend([str(metadata_path), metadata_path.stem])
    for text in texts:
        guess = _extract_id_from_text(text)
        if guess:
            return guess
    return decoded_path.stem.split("_")[0]

def _save_dir(fig_dir: Union[str, Path, None], decoded_database_json: Path) -> Path:
    if fig_dir is None or str(fig_dir).strip() == "":
        fig_dir = decoded_database_json.parent
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

def _parse_treatment_date(treatment_date: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if treatment_date is None:
        return None
    if isinstance(treatment_date, pd.Timestamp):
        return pd.Timestamp(treatment_date.date())
    s = str(treatment_date).strip().replace(".", "-").replace("/", "-")
    try:
        dt = pd.to_datetime(s, errors="raise")
        return pd.Timestamp(dt.date())
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Syllable sequence → bigram counts → TTE
# ──────────────────────────────────────────────────────────────────────────────
def _derive_order_from_row(row: pd.Series) -> List[str]:
    dct = row.get("syllable_onsets_offsets_ms_dict") or row.get("syllable_onsets_offsets_ms")
    if not isinstance(dct, dict) or len(dct) == 0:
        dct = row.get("syllable_onsets_offsets_timebins")
        if not isinstance(dct, dict) or len(dct) == 0:
            return []
    events: List[Tuple[float, str]] = []
    for lab, spans in dct.items():
        if spans is None:
            continue
        try:
            for on_off in spans:
                if not on_off:
                    continue
                on = float(on_off[0])
                events.append((on, str(lab)))
        except Exception:
            continue
    if not events:
        return []
    events.sort(key=lambda x: x[0])
    seq = [events[0][1]]
    for _, lab in events[1:]:
        if lab != seq[-1]:
            seq.append(lab)
    return seq

def _bigram_counts_from_seq(seq: List[str]) -> Dict[Tuple[str, str], int]:
    if not seq or len(seq) < 2:
        return {}
    from collections import Counter
    pairs = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
    return dict(Counter(pairs))

def _generate_normalized_transition_matrix(
    transition_list: List[Tuple[str, str]],
    transition_counts: List[int],
    syllable_types: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(syllable_types)
    counts_mat = np.zeros((n, n), dtype=float)
    idx = {lab: i for i, lab in enumerate(syllable_types)}
    for (frm, to), cnt in zip(transition_list, transition_counts):
        if frm in idx and to in idx:
            counts_mat[idx[frm], idx[to]] += cnt
    row_sums = counts_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_mat = counts_mat / row_sums
    return norm_mat, counts_mat

def _transition_entropy_per_row(transition_matrix: np.ndarray, syllable_types: List[str]) -> Dict[str, float]:
    ent: Dict[str, float] = {}
    for i, lab in enumerate(syllable_types):
        probs = transition_matrix[i, :]
        probs = probs[probs > 0]
        ent[lab] = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
    return ent

def _total_transition_entropy(counts_mat: np.ndarray, syllable_types: List[str], ent_by_syll: Dict[str, float]) -> float:
    row_sums = counts_mat.sum(axis=1)
    total = row_sums.sum()
    if total <= 0:
        return 0.0
    probs = row_sums / total
    return float(sum(probs[i] * ent_by_syll.get(lab, 0.0) for i, lab in enumerate(syllable_types)))

def _aggregated_tte_for_subset(subdf: pd.DataFrame, syllable_types: List[str]) -> float:
    if subdf.empty:
        return float("nan")
    pooled: Dict[Tuple[str, str], int] = {}
    for freq in subdf["transition_frequencies"]:
        for tr, cnt in freq.items():
            pooled[tr] = pooled.get(tr, 0) + cnt
    if not pooled:
        return float("nan")
    tr_list = list(pooled.keys())
    tr_counts = list(pooled.values())
    norm_mat, counts_mat = _generate_normalized_transition_matrix(tr_list, tr_counts, syllable_types)
    ent_by = _transition_entropy_per_row(norm_mat, syllable_types)
    return _total_transition_entropy(counts_mat, syllable_types, ent_by)

def _per_file_ttes(subdf: pd.DataFrame, syllable_types: List[str]) -> List[float]:
    vals: List[float] = []
    for freq in subdf["transition_frequencies"]:
        if not freq:
            vals.append(0.0)
            continue
        tr_list = list(freq.keys())
        tr_counts = list(freq.values())
        norm_mat, counts_mat = _generate_normalized_transition_matrix(tr_list, tr_counts, syllable_types)
        ent_by = _transition_entropy_per_row(norm_mat, syllable_types)
        vals.append(_total_transition_entropy(counts_mat, syllable_types, ent_by))
    return vals

def _tte_from_freq(freq: Dict[Tuple[str, str], int], syllable_types: List[str]) -> float:
    """Compute TTE for a single file/segment given its transition_frequencies dict."""
    if not freq:
        return 0.0
    tr_list = list(freq.keys())
    tr_counts = list(freq.values())
    norm_mat, counts_mat = _generate_normalized_transition_matrix(tr_list, tr_counts, syllable_types)
    ent_by = _transition_entropy_per_row(norm_mat, syllable_types)
    return _total_transition_entropy(counts_mat, syllable_types, ent_by)

def _mean_sem(x: List[float]) -> Tuple[float, Optional[float], int]:
    n = len(x)
    if n == 0:
        return float("nan"), None, 0
    if n == 1:
        return float(x[0]), 0.0, 1
    arr = np.asarray(x, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(n)), n


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers (kept; not used in the new 3-group figure by default)
# ──────────────────────────────────────────────────────────────────────────────
def _p_to_stars(p: float) -> str:
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))

def _prepost_pvalue(pre_vals: List[float], post_vals: List[float]) -> Tuple[Optional[float], str]:
    pre = np.asarray(pre_vals, dtype=float)
    post = np.asarray(post_vals, dtype=float)
    if pre.size == 0 or post.size == 0:
        return None, "insufficient"

    if _HAVE_SCIPY:
        try:
            u = _scipy_stats.mannwhitneyu(pre, post, alternative="two-sided", method="auto")
            return float(u.pvalue), "mannwhitneyu"
        except Exception:
            pass

    rng = np.random.default_rng(0)
    pooled = np.concatenate([pre, post])
    n1 = pre.size
    obs = abs(pre.mean() - post.mean())
    iters = 20000 if pooled.size <= 200 else 5000
    count = 0
    for _ in range(iters):
        rng.shuffle(pooled)
        a = pooled[:n1]
        b = pooled[n1:]
        if abs(a.mean() - b.mean()) >= obs - 1e-12:
            count += 1
    pval = (count + 1) / (iters + 1)
    return float(pval), "perm_mean_diff"


# ──────────────────────────────────────────────────────────────────────────────
# Build per-file (per-segment) table with datetime + bigram counts
# ──────────────────────────────────────────────────────────────────────────────
def _build_per_file_table(organized_df: pd.DataFrame) -> pd.DataFrame:
    df = organized_df.copy()

    if "file_name" in df.columns:
        seg_extracted = df["file_name"].astype(str).str.extract(r'_(\d+)(?:\.\w+)?$', expand=False)
        if "Segment" not in df.columns:
            df["Segment"] = pd.to_numeric(seg_extracted, errors="coerce").fillna(0).astype(int)
        if "File Stem" not in df.columns:
            df["File Stem"] = df["file_name"].astype(str).str.replace(r'_(\d+)(?:\.\w+)?$', "", regex=True)

    dt = None
    for cand in ["creation_date", "Recording DateTime", "recording_datetime", "date_time"]:
        if cand in df.columns:
            dt = pd.to_datetime(df[cand], errors="coerce")
            break
    if dt is None:
        for c in ["Date", "Hour", "Minute", "Second"]:
            if c not in df.columns:
                df[c] = None
        dstr = pd.to_datetime(df["Date"], errors="coerce")
        hh = df["Hour"].apply(_safe_pad) if "Hour" in df.columns else "00"
        mm = df["Minute"].apply(_safe_pad) if "Minute" in df.columns else "00"
        ss = df["Second"].apply(_safe_pad) if "Second" in df.columns else "00"
        dt = pd.to_datetime(
            dstr.dt.strftime("%Y-%m-%d") + " " + hh.astype(str) + ":" + mm.astype(str) + ":" + ss.astype(str),
            errors="coerce"
        )

    df["date_time"] = dt
    df["day"] = df["date_time"].dt.date

    if "syllable_order" not in df.columns:
        df["syllable_order"] = df.apply(_derive_order_from_row, axis=1)
    else:
        def _coerce(x):
            if isinstance(x, (list, tuple)):
                return [str(v) for v in x]
            return []
        df["syllable_order"] = df["syllable_order"].apply(_coerce)

    df["transition_frequencies"] = df["syllable_order"].apply(_bigram_counts_from_seq)

    ok = (~df["date_time"].isna()) & (df["transition_frequencies"].apply(lambda d: isinstance(d, dict) and len(d) > 0))

    keep = ["date_time", "day", "transition_frequencies"]
    for name in ["file_name", "File Stem", "Segment", "segment"]:
        if name in df.columns:
            keep.append(name)

    return df.loc[ok, keep].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TTEByDayResult:
    results_df: pd.DataFrame
    figure_path_timecourse: Optional[Path]
    figure_path_prepost_box: Optional[Path]  # now points to the 3-group matched plot when treatment_date is given
    prepost_p_value: Optional[float]
    prepost_test: Optional[str]

__all__ = ["TTEByDayResult", "TTE_by_day", "main"]

def TTE_by_day(
    *,
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Optional[Union[str, Path]] = None,
    only_song_present: bool = True,
    compute_durations: bool = False,
    fig_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    organize_builder: Optional[Callable] = _ORGANIZE_DEFAULT,
    min_songs_per_day: int = 1,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    treatment_in: str = "post",  # "post", "pre", or "exclude"
) -> TTEByDayResult:
    """
    Compute aggregated TTE per day and render:
      (1) a time-course line plot (optional treatment date vertical line),
      (2) a Balanced 3-group per-song plot (Early-Pre, Late-Pre, Post) if treatment_date is given:
          n = min(floor(#pre_songs/2), #post_songs).
          Use last 2n pre songs (split into halves) and first n post songs.
    """
    decoded_path = Path(decoded_database_json)
    metadata_path = Path(creation_metadata_json) if creation_metadata_json else None
    tx_date = _parse_treatment_date(treatment_date)

    builder = organize_builder or _ORGANIZE_DEFAULT
    if builder is None:
        raise ImportError(
            "No organizer available. Ensure one of these modules is importable:\n"
            "  - organized_decoded_serialTS_segments.py\n"
            "  - organize_decoded_serialTS_segments.py\n"
            "  - organize_decoded_with_segments.py"
        )

    # Signature-aware call
    params = inspect.signature(builder).parameters
    kwargs = {
        "decoded_database_json": decoded_path,
        "only_song_present": only_song_present,
        "compute_durations": compute_durations,
    }
    if "add_recording_datetime" in params:
        kwargs["add_recording_datetime"] = True
    if "creation_metadata_json" in params:
        kwargs["creation_metadata_json"] = metadata_path

    od = builder(**kwargs)
    organized = od.organized_df
    animal_id = _infer_animal_id(decoded_path, metadata_path, organized_df=organized, od=od)

    # Per-file (per-segment) table
    pf = _build_per_file_table(organized)
    if pf.empty:
        print("No valid per-file transitions (check date_time and syllable dicts).")
        return TTEByDayResult(pd.DataFrame(), None, None, None, None)

    # Global syllable set for consistent axes
    all_transitions: Dict[Tuple[str, str], int] = {}
    for d in pf["transition_frequencies"]:
        for tr, cnt in d.items():
            all_transitions[tr] = all_transitions.get(tr, 0) + cnt
    syllable_types = sorted({a for a, _ in all_transitions} | {b for _, b in all_transitions})

    # Per-day aggregated TTE (+ per-file stats for reference)
    rows: List[Dict[str, object]] = []
    for day, grp in pf.groupby("day"):
        grp = grp.sort_values(["date_time", *(c for c in ["Segment", "segment", "file_name"] if c in grp.columns)])
        n = len(grp)
        if n < min_songs_per_day:
            continue

        agg_tte = _aggregated_tte_for_subset(grp, syllable_types)
        perfile_ttes = _per_file_ttes(grp, syllable_types)
        mean_pf, sem_pf, _ = _mean_sem(perfile_ttes)

        rows.append({
            "day": pd.to_datetime(day),
            "n_segments": int(n),
            "agg_TTE": float(agg_tte),
            "perfile_mean_TTE": float(mean_pf),
            "perfile_sem_TTE": (None if sem_pf is None else float(sem_pf)),
            "perfile_TTEs": perfile_ttes,
        })

    results_df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    if results_df.empty:
        print("No days satisfied the minimum segment requirement.")
        return TTEByDayResult(results_df, None, None, None, None)

    fig_dir = _save_dir(fig_dir, decoded_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 1: Time-course (every date tick; 90°; no grid; hide top/right)
    # ──────────────────────────────────────────────────────────────────────────
    dates = pd.to_datetime(results_df["day"])
    fig_w = max(12, 0.35 * len(dates))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.plot(dates, results_df["agg_TTE"], marker="o", linewidth=2)

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Transition Entropy (bits)")
    ax.set_title(f"{animal_id} – TTE by Day (aggregated per day)")

    ax.set_xticks(dates.to_list())
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=90, ha="center")

    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(top=False, right=False)

    if tx_date is not None:
        ax.axvline(pd.Timestamp(tx_date), linestyle="--", linewidth=1.5, color="red")
        ax.text(pd.Timestamp(tx_date), ax.get_ylim()[1], "Treatment",
                color="red", rotation=90, va="top", ha="right", fontsize=9, clip_on=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    fig_path_timecourse = fig_dir / f"{animal_id}_TTE_by_day.png"
    fig.savefig(fig_path_timecourse, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 2 (reworked): Balanced 3-group per-song plot (Early-Pre, Late-Pre, Post)
    # ──────────────────────────────────────────────────────────────────────────
    fig_path_prepost_box: Optional[Path] = None
    p_val: Optional[float] = None
    p_method: Optional[str] = None

    if tx_date is not None:
        # Build per-file TTE column once for selection/plotting
        pf_sorted = pf.sort_values(["date_time", *(c for c in ["Segment", "segment", "file_name"] if c in pf.columns)])
        pf_sorted = pf_sorted.assign(
            file_TTE=pf_sorted["transition_frequencies"].apply(lambda d: _tte_from_freq(d, syllable_types))
        )

        # Define pre/post masks per treatment_in policy
        dts = pf_sorted["date_time"]
        t0 = pd.Timestamp(tx_date)
        if treatment_in.lower() == "pre":
            pre_mask  = dts.dt.date <= t0.date()
            post_mask = dts.dt.date >  t0.date()
        elif treatment_in.lower() == "exclude":
            pre_mask  = dts.dt.date <  t0.date()
            post_mask = dts.dt.date >  t0.date()
        else:  # "post" includes the treatment day in post
            pre_mask  = dts.dt.date <  t0.date()
            post_mask = dts.dt.date >= t0.date()

        pre_df  = pf_sorted.loc[pre_mask].copy()
        post_df = pf_sorted.loc[post_mask].copy()

        # Sort within groups by time (ascending)
        pre_df  = pre_df.sort_values("date_time")
        post_df = post_df.sort_values("date_time")

        # n = min(floor(#pre/2), #post)
        n_pre   = len(pre_df)
        n_post  = len(post_df)
        n = min(n_pre // 2, n_post)

        if n <= 0:
            print("Balanced 3-group plot skipped: not enough songs to balance (need at least 1 in post and 2 in pre).")
        else:
            # Take last 2n pre songs → split into halves
            pre_last_2n = pre_df.tail(2 * n).reset_index(drop=True)
            early_pre_n = pre_last_2n.iloc[:n]   # older half of the last 2n
            late_pre_n  = pre_last_2n.iloc[n:]   # newer half of the last 2n

            # First n post songs
            post_first_n = post_df.head(n).reset_index(drop=True)

            # Grab per-song TTE lists
            vals_early = early_pre_n["file_TTE"].astype(float).tolist()
            vals_late  = late_pre_n["file_TTE"].astype(float).tolist()
            vals_post  = post_first_n["file_TTE"].astype(float).tolist()

            # Make 3-group box + jitter
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            data = [vals_early, vals_late, vals_post]
            labels = [f"Early Pre (n={n})", f"Late Pre (n={n})", f"Post (n={n})"]
            positions = [1, 2, 3]

            bp = ax3.boxplot(
                data, positions=positions, widths=0.6, showfliers=False,
                patch_artist=True, medianprops=dict(linewidth=2)
            )
            for patch in bp["boxes"]:
                patch.set_alpha(0.25)

            rng = np.random.default_rng(0)
            for pos, vals in zip(positions, data):
                if not vals:
                    continue
                x = np.full(len(vals), pos) + rng.uniform(-0.08, 0.08, size=len(vals))
                ax3.scatter(x, vals, s=22, alpha=0.9)

            ax3.set_xticks(positions)
            ax3.set_xticklabels(labels)
            ax3.set_ylabel("Per-song Total Transition Entropy (bits)")
            ax3.set_title(f"{animal_id} – Balanced per-song TTE (Early/Late Pre vs Post)")

            ax3.grid(False)
            for side in ("top", "right"):
                ax3.spines[side].set_visible(False)
            ax3.tick_params(top=False, right=False)

            fig3.tight_layout()
            fig_path_prepost_box = fig_dir / f"{animal_id}_TTE_pre_vs_post_balanced_3group_box.png"
            fig3.savefig(fig_path_prepost_box, dpi=200)
            if show:
                plt.show()
            else:
                plt.close(fig3)

    return TTEByDayResult(
        results_df=results_df,
        figure_path_timecourse=fig_path_timecourse,
        figure_path_prepost_box=fig_path_prepost_box,  # now 3-group balanced plot (if treatment_date provided)
        prepost_p_value=p_val,   # unchanged; 3-group plot does not compute stats by default
        prepost_test=p_method,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute aggregated TTE per day and plot:\n"
            " - Time-course by day\n"
            " - Balanced 3-group per-song plot if treatment_date is given "
            "(Early-Pre, Late-Pre, Post) with n = min(floor(#pre/2), #post)."
        )
    )
    p.add_argument("--decoded", required=True, help="Path to decoded_database.json")
    p.add_argument("--meta", default=None,
                   help="(Optional) Path to metadata .json — not required for Excel-serial organizer.")
    p.add_argument("--only_song_present", action="store_true",
                   help="Restrict to song_present rows in organizer")
    p.add_argument("--fig_dir", default=None, help="Directory to save figures (default: decoded file's folder)")
    p.add_argument("--min_songs_per_day", type=int, default=1,
                   help="Minimum segments required to include a day (default: 1)")
    p.add_argument("--treatment_date", default=None,
                   help="Optional treatment date: 'YYYY-MM-DD' (also accepts 'YYYY.MM.DD')")
    p.add_argument("--treatment_in", default="post", choices=["post", "pre", "exclude"],
                   help="Which side includes the treatment day for pre/post split (default: post)")
    p.add_argument("--no_show", action="store_true", help="Do not display plot windows")
    return p

def main():
    args = _build_arg_parser().parse_args()
    res = TTE_by_day(
        decoded_database_json=args.decoded,
        creation_metadata_json=args.meta,
        only_song_present=args.only_song_present,
        fig_dir=args.fig_dir,
        show=not args.no_show,
        min_songs_per_day=args.min_songs_per_day,
        treatment_date=args.treatment_date,
        treatment_in=args.treatment_in,
    )
    print("\nResults (first 10 rows):")
    if not res.results_df.empty:
        print(res.results_df.head(10).to_string(index=False))
    else:
        print("No results.")
    if res.figure_path_timecourse:
        print(f"Time-course plot → {res.figure_path_timecourse}")
    if res.figure_path_prepost_box:
        print(f"Balanced 3-group plot → {res.figure_path_prepost_box}")
    if res.prepost_p_value is not None:
        print(f"Pre vs Post: p={res.prepost_p_value:.3g} ({res.prepost_test})")

if __name__ == "__main__":
    main()


"""
import importlib, tte_by_day
importlib.reload(tte_by_day)
from tte_by_day import TTE_by_day

decoded = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08_RC6_Comp2_decoded_database.json"

out = TTE_by_day(
    decoded_database_json=decoded,
    creation_metadata_json=None,
    only_song_present=True,
    show=True,
    min_songs_per_day=5,
    treatment_date="2025-05-22",
    treatment_in="post",      # include the treatment day with post
)

print("p-value:", out.prepost_p_value, "method:", out.prepost_test)
print("Time-course:", out.figure_path_timecourse)
print("Pre/Post box:", out.figure_path_prepost_box)

"""