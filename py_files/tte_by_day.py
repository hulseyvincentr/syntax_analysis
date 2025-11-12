# -*- coding: utf-8 -*-
# tte_by_day.py  (merged-aware; supports pre-merged DataFrames; plain + stats 3-group plots)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import argparse
import re
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for omnibus + MWU
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# These imports are only used when we need to perform merging ourselves.
from merge_annotations_from_split_songs import build_decoded_with_split_labels
from merge_potential_split_up_song import build_detected_and_merged_songs

# Fallback organizer (legacy mode if neither premerged nor detect path is provided)
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


def _most_common(strings: Iterable[str]) -> Optional[str]:
    from collections import Counter
    c = Counter(s for s in strings if s)
    return c.most_common(1)[0][0] if c else None


def _extract_id_from_text(text: str) -> Optional[str]:
    m = re.search(r"(usa\d{4,6})", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(R\d{1,3})(?=[_\W]|$)", text)
    if m:
        return m.group(1)
    return None


def _infer_animal_id_from_df(df: pd.DataFrame) -> Optional[str]:
    for col in ["animal_id", "Animal_ID", "animalID"]:
        if col in df.columns and df[col].notna().any():
            return _most_common(df[col].dropna().astype(str))
    if "file_name" in df.columns:
        matches = []
        for fn in df["file_name"].dropna().astype(str):
            mid = _extract_id_from_text(fn)
            if mid:
                matches.append(mid)
        return _most_common(matches) if matches else None
    return None


def _infer_animal_id(
    decoded_path: Path,
    metadata_path: Optional[Path] = None,
    df: Optional[pd.DataFrame] = None,
    od: Optional[object] = None,
) -> str:
    if od is not None:
        for attr in ["animal_id", "animalID", "Animal_ID"]:
            if hasattr(od, attr):
                val = getattr(od, attr)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    if df is not None:
        guess = _infer_animal_id_from_df(df)
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
# Sequence → bigrams → entropy
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
    pairs = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
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


def _total_transition_entropy(
    counts_mat: np.ndarray, syllable_types: List[str], ent_by_syll: Dict[str, float]
) -> float:
    row_sums = counts_mat.sum(axis=1)
    total = row_sums.sum()
    if total <= 0:
        return 0.0
    probs = row_sums / total
    return float(sum(probs[i] * ent_by_syll.get(lab, 0.0) for i, lab in enumerate(syllable_types)))


def _per_file_tte_from_freq(freq: Dict[Tuple[str, str], int], syllable_types: List[str]) -> float:
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
# Stats helpers (Kruskal + MWU Holm; Cliff's delta; formatting/plotting)
# ──────────────────────────────────────────────────────────────────────────────
def _p_to_stars(p: float) -> str:
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))


def _holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = pvals[idx] * factor
        running = max(running, val)  # step-up
        adj[idx] = min(1.0, running)
    return [float(v) for v in adj]  # type: ignore


def _cliffs_delta(a: List[float], b: List[float]) -> float:
    a = list(a)
    b = list(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    wins = 0
    losses = 0
    for x in a:
        for y in b:
            if x > y:
                wins += 1
            elif x < y:
                losses += 1
    denom = len(a) * len(b)
    return (wins - losses) / denom if denom else float("nan")


def _pairwise_mwu(a: List[float], b: List[float]) -> float:
    if _HAVE_SCIPY:
        return float(_scipy_stats.mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue)
    # permutation fallback (mean diff)
    rng = np.random.default_rng(0)
    pooled = np.asarray(a + b, dtype=float)
    n1 = len(a)
    obs = abs(np.mean(a) - np.mean(b))
    iters = 20000 if pooled.size <= 200 else 5000
    count = 0
    for _ in range(iters):
        rng.shuffle(pooled)
        if abs(pooled[:n1].mean() - pooled[n1:].mean()) >= obs - 1e-12:
            count += 1
    return float((count + 1) / (iters + 1))


def _draw_sig_bracket(ax, x1: float, x2: float, y: float, h: float, label: str):
    """Draw a significance bracket from x1 to x2 at height y."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
    ax.text(
        (x1 + x2) / 2,
        y + h + (0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        label,
        ha="center",
        va="bottom",
        fontsize=11,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-song table from MERGED ANNOTATIONS
# ──────────────────────────────────────────────────────────────────────────────
def _build_per_song_from_merged_annotations(merged_ann_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_ann_df.copy()
    dt_col = None
    for cand in ["Recording DateTime", "recording_datetime", "creation_date", "date_time"]:
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise ValueError("Merged annotations missing a datetime column.")
    df["date_time"] = pd.to_datetime(df[dt_col], errors="coerce")
    df["day"] = df["date_time"].dt.date

    if "file_name" in df.columns and "File Stem" not in df.columns:
        df["File Stem"] = df["file_name"].astype(str)

    if "syllable_order" not in df.columns:
        df["syllable_order"] = df.apply(_derive_order_from_row, axis=1)
    else:
        df["syllable_order"] = df["syllable_order"].apply(
            lambda x: [str(v) for v in x] if isinstance(x, (list, tuple)) else []
        )
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
    figure_path_prepost_box_plain: Optional[Path]   # plain 3-group plot
    figure_path_prepost_box_stats: Optional[Path]   # stats 3-group plot
    # Back-compat pointer (stats if present else plain)
    figure_path_prepost_box: Optional[Path]
    prepost_p_value: Optional[float]
    prepost_test: Optional[str]
    merged_durations_df: Optional[pd.DataFrame]
    three_group_stats: Optional[Dict[str, object]]


__all__ = ["TTEByDayResult", "TTE_by_day", "main"]


def TTE_by_day(
    *,
    decoded_database_json: Union[str, Path],
    # NEW: Prefer these if provided (from wrapper)
    premerged_annotations_df: Optional[pd.DataFrame] = None,
    premerged_durations_df: Optional[pd.DataFrame] = None,
    animal_id_override: Optional[str] = None,

    # If premerged_* are not provided, we can still merge from JSONs:
    song_detection_json: Optional[Union[str, Path]] = None,

    # annotations merge knobs (only used when we merge ourselves)
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,

    # durations merge knob (only used when we merge ourselves)
    merged_song_gap_ms: Optional[int] = 500,

    # general
    fig_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    min_songs_per_day: int = 1,
    treatment_date: Optional[Union[str, pd.Timestamp]] = None,
    treatment_in: str = "post",

    # legacy fallbacks (only used if neither premerged nor detect are given)
    creation_metadata_json: Optional[Union[str, Path]] = None,
    only_song_present: bool = True,
    compute_durations: bool = False,
    organize_builder: Optional[Callable] = _ORGANIZE_DEFAULT,
) -> TTEByDayResult:
    """
    Build daily TTE from *pre-merged* song annotations (preferred if provided),
    otherwise from merged JSONs, otherwise legacy organizers.

    Precedence:
      1) premerged_annotations_df (and optional premerged_durations_df)
      2) song_detection_json + decoded_database_json (merge on the fly)
      3) legacy organizers (per-segment)

    Returns time-course figure + two balanced 3-group figures (plain & stats).
    """
    decoded_path = Path(decoded_database_json)
    det_path = Path(song_detection_json) if song_detection_json else None
    tx_date = _parse_treatment_date(treatment_date)

    # ── Source selection: premerged > merged-from-json > legacy
    if premerged_annotations_df is not None:
        work_df = premerged_annotations_df.copy()
        merged_dur_df = premerged_durations_df.copy() if premerged_durations_df is not None else None
        od_for_id = None
    elif det_path is not None:
        ann = build_decoded_with_split_labels(
            decoded_database_json=decoded_path,
            song_detection_json=det_path,
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
        work_df = ann.annotations_appended_df.copy()
        if merged_song_gap_ms and int(merged_song_gap_ms) > 0:
            det = build_detected_and_merged_songs(
                det_path,
                songs_only=True,
                flatten_spec_params=True,
                max_gap_between_song_segments=int(merged_song_gap_ms),
            )
            merged_dur_df = det["detected_merged_songs"].copy()
        else:
            merged_dur_df = None
        od_for_id = None
    else:
        builder = organize_builder or _ORGANIZE_DEFAULT
        if builder is None:
            raise ImportError(
                "No data source available: pass premerged_annotations_df, or song_detection_json, "
                "or ensure a legacy organizer is importable."
            )
        params = inspect.signature(builder).parameters
        kwargs = {
            "decoded_database_json": decoded_path,
            "only_song_present": only_song_present,
            "compute_durations": compute_durations,
        }
        if "add_recording_datetime" in params:
            kwargs["add_recording_datetime"] = True
        if "creation_metadata_json" in params and creation_metadata_json:
            kwargs["creation_metadata_json"] = Path(creation_metadata_json)
        od_for_id = builder(**kwargs)
        work_df = od_for_id.organized_df.copy()
        merged_dur_df = None

    # Animal ID preference: explicit override > infer
    if animal_id_override and str(animal_id_override).strip():
        animal_id = str(animal_id_override).strip()
    else:
        animal_id = _infer_animal_id(decoded_path, None, df=work_df, od=od_for_id)

    # Per-song rows (use merged-annotation builder if we used merged/premerged)
    if (premerged_annotations_df is not None) or (det_path is not None):
        pf = _build_per_song_from_merged_annotations(work_df)
    else:
        # legacy path: derive columns similarly to older organizer outputs
        pf = work_df.copy()
        dt = None
        for cand in ["creation_date", "Recording DateTime", "recording_datetime", "date_time"]:
            if cand in pf.columns:
                dt = pd.to_datetime(pf[cand], errors="coerce")
                break
        if dt is None:
            for c in ["Date", "Hour", "Minute", "Second"]:
                if c not in pf.columns:
                    pf[c] = None
            dstr = pd.to_datetime(pf["Date"], errors="coerce")
            hh = pf["Hour"].apply(_safe_pad) if "Hour" in pf.columns else "00"
            mm = pf["Minute"].apply(_safe_pad) if "Minute" in pf.columns else "00"
            ss = pf["Second"].apply(_safe_pad) if "Second" in pf.columns else "00"
            dt = pd.to_datetime(
                dstr.dt.strftime("%Y-%m-%d") + " " + hh.astype(str) + ":" + mm.astype(str) + ":" + ss.astype(str),
                errors="coerce",
            )
        pf["date_time"] = dt
        pf["day"] = pf["date_time"].dt.date
        if "syllable_order" not in pf.columns:
            pf["syllable_order"] = pf.apply(_derive_order_from_row, axis=1)
        else:
            pf["syllable_order"] = pf["syllable_order"].apply(
                lambda x: [str(v) for v in x] if isinstance(x, (list, tuple)) else []
            )
        pf["transition_frequencies"] = pf["syllable_order"].apply(_bigram_counts_from_seq)
        ok = (~pf["date_time"].isna()) & (pf["transition_frequencies"].apply(lambda d: isinstance(d, dict) and len(d) > 0))
        keep = ["date_time", "day", "transition_frequencies"]
        for name in ["file_name", "File Stem", "Segment", "segment"]:
            if name in pf.columns:
                keep.append(name)
        pf = pf.loc[ok, keep].reset_index(drop=True)

    if pf.empty:
        print("No valid per-song transitions after merging.")
        return TTEByDayResult(
            pd.DataFrame(), None, None, None, None, None, None, merged_durations_df=merged_dur_df, three_group_stats=None
        )

    # Global syllable set
    all_tr: Dict[Tuple[str, str], int] = {}
    for d in pf["transition_frequencies"]:
        for tr, cnt in d.items():
            all_tr[tr] = all_tr.get(tr, 0) + cnt
    syllable_types = sorted({a for a, _ in all_tr} | {b for _, b in all_tr})

    # Per-day aggregated TTE (+ per-song stats for ref)
    rows: List[Dict[str, object]] = []
    for day, grp in pf.groupby("day"):
        grp = grp.sort_values(["date_time", *(c for c in ["Segment", "segment", "file_name"] if c in grp.columns)])
        n = len(grp)
        if n < min_songs_per_day:
            continue

        per_song_tte = [_per_file_tte_from_freq(freq, syllable_types) for freq in grp["transition_frequencies"]]
        mean_pf, sem_pf, _ = _mean_sem(per_song_tte)

        pooled: Dict[Tuple[str, str], int] = {}
        for freq in grp["transition_frequencies"]:
            for tr, cnt in freq.items():
                pooled[tr] = pooled.get(tr, 0) + cnt
        tr_list = list(pooled.keys())
        tr_counts = list(pooled.values())
        norm_mat, counts_mat = _generate_normalized_transition_matrix(tr_list, tr_counts, syllable_types)
        ent_by = _transition_entropy_per_row(norm_mat, syllable_types)
        agg_tte = _total_transition_entropy(counts_mat, syllable_types, ent_by)

        rows.append(
            {
                "day": pd.to_datetime(day),
                "n_songs": int(n),
                "agg_TTE": float(agg_tte),
                "per_song_mean_TTE": float(mean_pf),
                "per_song_sem_TTE": (None if sem_pf is None else float(sem_pf)),
                "per_song_TTEs": per_song_tte,
            }
        )

    results_df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    if results_df.empty:
        print("No days satisfied the minimum song requirement.")
        return TTEByDayResult(
            results_df, None, None, None, None, None, None, merged_durations_df=merged_dur_df, three_group_stats=None
        )

    fig_dir = _save_dir(fig_dir, decoded_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 1: Time-course
    # ──────────────────────────────────────────────────────────────────────────
    dates = pd.to_datetime(results_df["day"])
    fig_w = max(12, 0.35 * len(dates))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.plot(dates, results_df["agg_TTE"], marker="o", linewidth=2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Total Transition Entropy (bits)")
    ax.set_title(f"{animal_id} – TTE by Day (merged-song annotations)")
    ax.set_xticks(dates.to_list())
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=90, ha="center")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(top=False, right=False)
    if tx_date is not None:
        ax.axvline(pd.Timestamp(tx_date), linestyle="--", linewidth=1.5, color="red")
        ax.text(
            pd.Timestamp(tx_date),
            ax.get_ylim()[1],
            "Treatment",
            color="red",
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            clip_on=False,
        )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig_path_timecourse = fig_dir / f"{animal_id}_TTE_by_day_merged.png"
    fig.savefig(fig_path_timecourse, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 2: Balanced 3-group per-song (plain + stats)
    # ──────────────────────────────────────────────────────────────────────────
    fig_path_plain: Optional[Path] = None
    fig_path_stats: Optional[Path] = None
    three_stats: Optional[Dict[str, object]] = None

    if tx_date is not None:
        pf_sorted = pf.sort_values(["date_time", *(c for c in ["Segment", "segment", "file_name"] if c in pf.columns)])
        pf_sorted = pf_sorted.assign(
            file_TTE=pf_sorted["transition_frequencies"].apply(lambda d: _per_file_tte_from_freq(d, syllable_types))
        )

        dts = pf_sorted["date_time"]
        t0 = pd.Timestamp(tx_date)
        if treatment_in.lower() == "pre":
            pre_mask = dts.dt.date <= t0.date()
            post_mask = dts.dt.date > t0.date()
        elif treatment_in.lower() == "exclude":
            pre_mask = dts.dt.date < t0.date()
            post_mask = dts.dt.date > t0.date()
        else:
            pre_mask = dts.dt.date < t0.date()
            post_mask = dts.dt.date >= t0.date()

        pre_df = pf_sorted.loc[pre_mask].copy().sort_values("date_time")
        post_df = pf_sorted.loc[post_mask].copy().sort_values("date_time")

        n = min(len(pre_df) // 2, len(post_df))
        if n > 0:
            pre_last_2n = pre_df.tail(2 * n).reset_index(drop=True)
            early_pre_n = pre_last_2n.iloc[:n]
            late_pre_n = pre_last_2n.iloc[n:]
            post_first_n = post_df.head(n).reset_index(drop=True)

            E = early_pre_n["file_TTE"].astype(float).tolist()
            L = late_pre_n["file_TTE"].astype(float).tolist()
            P = post_first_n["file_TTE"].astype(float).tolist()

            # ------ Plain plot ------
            figP, axP = plt.subplots(figsize=(8.5, 5))
            data = [E, L, P]
            labels = [f"Early Pre (n={n})", f"Late Pre (n={n})", f"Post (n={n})"]
            pos = [1, 2, 3]
            bp = axP.boxplot(
                data, positions=pos, widths=0.6, showfliers=False, patch_artist=True, medianprops=dict(linewidth=2)
            )
            for patch in bp["boxes"]:
                patch.set_alpha(0.25)
            rng = np.random.default_rng(0)
            for p, vals in zip(pos, data):
                if vals:
                    x = np.full(len(vals), p) + rng.uniform(-0.08, 0.08, size=len(vals))
                    axP.scatter(x, vals, s=22, alpha=0.9)
            axP.set_xticks(pos)
            axP.set_xticklabels(labels)
            axP.set_ylabel("Per-song Total Transition Entropy (bits)")
            axP.set_title(f"{animal_id} – Balanced per-song TTE (Early/Late Pre vs Post, merged)")
            for side in ("top", "right"):
                axP.spines[side].set_visible(False)
            axP.tick_params(top=False, right=False)
            figP.tight_layout()
            fig_path_plain = fig_dir / f"{animal_id}_TTE_pre_vs_post_balanced_3group_box_plain_merged.png"
            figP.savefig(fig_path_plain, dpi=200)
            if show:
                plt.show()
            else:
                plt.close(figP)

            # ------ Stats (omnibus + pairwise + panel + brackets) ------
            kw = None
            if _HAVE_SCIPY:
                try:
                    kw = _scipy_stats.kruskal(E, L, P, nan_policy="omit")
                except Exception:
                    kw = None

            raw_pairs = {
                "Early vs Late": _pairwise_mwu(E, L),
                "Early vs Post": _pairwise_mwu(E, P),
                "Late  vs Post": _pairwise_mwu(L, P),
            }
            adj_vals = _holm_bonferroni(list(raw_pairs.values()))
            pairs_adj = {k: a for k, a in zip(raw_pairs.keys(), adj_vals)}
            effects = {
                "Early vs Late": _cliffs_delta(E, L),
                "Early vs Post": _cliffs_delta(E, P),
                "Late  vs Post": _cliffs_delta(L, P),
            }
            three_stats = {
                "n_per_group": n,
                "omnibus": ({"H": float(kw.statistic), "p": float(kw.pvalue)} if kw is not None else None),
                "pairwise_raw_p": raw_pairs,
                "pairwise_holm_p": pairs_adj,
                "cliffs_delta": effects,
                "stars": {k: _p_to_stars(p) for k, p in pairs_adj.items()},
            }

            # Separate right-hand axis for stats text
            figS, (axS, axTxt) = plt.subplots(
                1,
                2,
                figsize=(10.8, 5),
                gridspec_kw={"width_ratios": [3.6, 1.4], "wspace": 0.30},
            )

            bp = axS.boxplot(
                data, positions=pos, widths=0.6, showfliers=False, patch_artist=True, medianprops=dict(linewidth=2)
            )
            for patch in bp["boxes"]:
                patch.set_alpha(0.25)
            rng = np.random.default_rng(0)
            for p, vals in zip(pos, data):
                if vals:
                    x = np.full(len(vals), p) + rng.uniform(-0.08, 0.08, size=len(vals))
                    axS.scatter(x, vals, s=22, alpha=0.9)

            axS.set_xticks(pos)
            axS.set_xticklabels(labels)
            axS.set_ylabel("Per-song Total Transition Entropy (bits)")
            axS.set_title(f"{animal_id} – Balanced per-song TTE (Early/Late Pre vs Post, merged)")
            for side in ("top", "right"):
                axS.spines[side].set_visible(False)
            axS.tick_params(top=False, right=False)

            # Brackets with stars/ns (Holm-adjusted p)
            y_max = max([max(vals) if len(vals) else 0 for vals in data] + [1.0])
            axS.set_ylim(0, y_max * 1.35)
            h = (axS.get_ylim()[1] - axS.get_ylim()[0]) * 0.02
            y1 = y_max * 1.08
            y2 = y_max * 1.17
            y3 = y_max * 1.26
            _draw_sig_bracket(axS, 1, 2, y1, h, _p_to_stars(pairs_adj["Early vs Late"]))
            _draw_sig_bracket(axS, 1, 3, y2, h, _p_to_stars(pairs_adj["Early vs Post"]))
            _draw_sig_bracket(axS, 2, 3, y3, h, _p_to_stars(pairs_adj["Late  vs Post"]))

            axTxt.axis("off")
            lines = []
            if kw is not None:
                lines.append(f"Kruskal–Wallis: H={kw.statistic:.3g}, p={kw.pvalue:.3g} ({_p_to_stars(kw.pvalue)})")
            else:
                lines.append("Kruskal–Wallis: unavailable (SciPy not found)")
            lines.append("")
            for k in ["Early vs Late", "Early vs Post", "Late  vs Post"]:
                p_adj = pairs_adj[k]
                delta = effects[k]
                lines.append(f"{k}: p_holm={p_adj:.3g} ({_p_to_stars(p_adj)})  δ={delta:.2f}")

            stats_text = "\n".join(lines)
            axTxt.text(0.0, 1.0, stats_text, transform=axTxt.transAxes, va="top", ha="left", fontsize=10, wrap=True)

            figS.tight_layout()
            fig_path_stats = fig_dir / f"{animal_id}_TTE_pre_vs_post_balanced_3group_box_stats_merged.png"
            figS.savefig(fig_path_stats, dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(figS)

    # Back-compat pointer: prefer stats plot, else plain
    fallback = fig_path_stats if fig_path_stats is not None else fig_path_plain

    return TTEByDayResult(
        results_df=results_df,
        figure_path_timecourse=fig_path_timecourse,
        figure_path_prepost_box_plain=fig_path_plain,
        figure_path_prepost_box_stats=fig_path_stats,
        figure_path_prepost_box=fallback,
        prepost_p_value=None,
        prepost_test=None,
        merged_durations_df=merged_dur_df,
        three_group_stats=three_stats,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI (unchanged; pre-merged usage is intended from Python, e.g., wrapper)
# ──────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute aggregated TTE per day from MERGED song annotations; "
            "plot time-course and (if treatment_date) two balanced per-song plots: "
            "plain and stats (with right-side panel + bracket stars)."
        )
    )
    p.add_argument("--decoded", required=True, help="Path to *_decoded_database.json")
    p.add_argument("--detect", required=False, default=None, help="Path to *_song_detection.json (recommended).")
    # merge knobs
    p.add_argument("--ann-gap-ms", type=int, default=500)
    p.add_argument("--seg-offset", type=int, default=0)
    p.add_argument("--merge-repeats", action="store_true")
    p.add_argument("--repeat-gap-ms", type=float, default=10.0)
    p.add_argument("--repeat-gap-inclusive", action="store_true")
    p.add_argument("--dur-merge-gap-ms", type=int, default=500, help="0 to skip duration merge.")
    # general
    p.add_argument("--fig_dir", default=None)
    p.add_argument("--min_songs_per_day", type=int, default=1)
    p.add_argument("--treatment_date", default=None)
    p.add_argument("--treatment_in", default="post", choices=["post", "pre", "exclude"])
    p.add_argument("--no_show", action="store_true")
    # legacy fallbacks (if --detect omitted)
    p.add_argument("--meta", default=None)
    p.add_argument("--only_song_present", action="store_true")
    p.add_argument("--compute_durations", action="store_true")
    return p


def main():
    args = _build_arg_parser().parse_args()
    res = TTE_by_day(
        decoded_database_json=args.decoded,
        song_detection_json=args.detect,
        max_gap_between_song_segments=args.ann_gap_ms,
        segment_index_offset=args.seg_offset,
        merge_repeated_syllables=args.merge_repeats,
        repeat_gap_ms=args.repeat_gap_ms,
        repeat_gap_inclusive=args.repeat_gap_inclusive,
        merged_song_gap_ms=(None if args.dur_merge_gap_ms == 0 else args.dur_merge_gap_ms),
        fig_dir=args.fig_dir,
        show=not args.no_show,
        min_songs_per_day=args.min_songs_per_day,
        treatment_date=args.treatment_date,
        treatment_in=args.treatment_in,
        # legacy:
        creation_metadata_json=args.meta,
        only_song_present=args.only_song_present,
        compute_durations=args.compute_durations,
    )
    print("\nResults (first 10 rows):")
    if not res.results_df.empty:
        print(res.results_df.head(10).to_string(index=False))
    else:
        print("No results.")
    if res.figure_path_timecourse:
        print(f"Time-course plot → {res.figure_path_timecourse}")
    if res.figure_path_prepost_box_plain:
        print(f"3-group (plain) → {res.figure_path_prepost_box_plain}")
    if res.figure_path_prepost_box_stats:
        print(f"3-group (stats) → {res.figure_path_prepost_box_stats}")
    if res.three_group_stats:
        print("\n[Three-group stats]")
        print(res.three_group_stats)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Example (Python usage with pre-merged DataFrames from your wrapper):
# ---------------------------------------------------------------------------
# from tte_by_day import TTE_by_day
# res = run_area_x_meta_bundle(... )  # your wrapper that returns merged tables
# tte = TTE_by_day(
#     decoded_database_json=decoded_json_path,
#     premerged_annotations_df=res.merged_annotations_df,
#     premerged_durations_df=res.merged_detected_df,   # optional
#     animal_id_override="R08",                        # optional
#     fig_dir=(Path(decoded_json_path).parent / "figures"),
#     show=True,
#     min_songs_per_day=5,
#     treatment_date="2025-05-22",
#     treatment_in="post",
# )
# print(tte.figure_path_timecourse, tte.figure_path_prepost_box_stats)
