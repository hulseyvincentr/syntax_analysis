# -*- coding: utf-8 -*-
# tte_set_hour_and_batch.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Callable, Union

import argparse
import re
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Preferred organizer: Excel-serial timestamps from filename
# Fallbacks: alt module name, then legacy organizer.
# ──────────────────────────────────────────────────────────────────────────────
_ORGANIZE_DEFAULT: Optional[Callable] = None
try:
    # preferred (your new module)
    from organized_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _ORGANIZE_DEFAULT  # type: ignore
    )
except Exception:
    try:
        # alternate filename
        from organize_decoded_serialTS_segments import (
            build_organized_segments_with_durations as _ORGANIZE_DEFAULT  # type: ignore
        )
    except Exception:
        try:
            # legacy organizer (uses metadata, but we can pass None)
            from organize_decoded_with_segments import (
                build_organized_segments_with_durations as _ORGANIZE_DEFAULT  # type: ignore
            )
        except Exception:
            _ORGANIZE_DEFAULT = None

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_pad(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return "00"
    return str(s).zfill(2)

def _minutes_since_midnight(dt: pd.Timestamp) -> int:
    return dt.hour * 60 + dt.minute

def _parse_time_range(spec: str) -> Tuple[int, int]:
    """
    Parse 'HH:MM-HH:MM' → (start_min_incl, end_min_incl).
    """
    spec = spec.strip()
    if "-" not in spec:
        raise ValueError(f"Time range must be 'HH:MM-HH:MM', got: {spec}")
    lhs, rhs = spec.split("-", 1)
    def to_min(hhmm: str) -> int:
        hh, mm = hhmm.strip().split(":")
        return int(hh) * 60 + int(mm)
    return to_min(lhs), to_min(rhs)

def _bigrams(labels: Iterable[str]) -> List[Tuple[str, str]]:
    arr = list(labels)
    return [(arr[i], arr[i + 1]) for i in range(len(arr) - 1)]

def _row_transition_frequencies(syllable_order: Iterable[str]) -> Dict[Tuple[str, str], int]:
    from collections import Counter
    if not syllable_order:
        return {}
    return dict(Counter(_bigrams(list(syllable_order))))

def _build_per_file_table(organized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-SEGMENT table:
      - date_time, day, tod_min
      - transition_frequencies (dict[(from,to)->count])
      - file_name/File Stem, Segment (if present; derive if missing)
    """
    df = organized_df.copy()

    # Derive Segment / File Stem from file_name if missing
    if "file_name" in df.columns:
        seg_extracted = df["file_name"].astype(str).str.extract(r'_(\d+)(?:\.\w+)?$', expand=False)
        if "Segment" not in df.columns:
            df["Segment"] = pd.to_numeric(seg_extracted, errors="coerce").fillna(0).astype(int)
        if "File Stem" not in df.columns:
            df["File Stem"] = df["file_name"].astype(str).str.replace(r'_(\d+)(?:\.\w+)?$', "", regex=True)

    # ---------------- Datetime ----------------
    dt = None
    for cand in ["creation_date", "Recording DateTime", "recording_datetime", "date_time"]:
        if cand in df.columns:
            dt = pd.to_datetime(df[cand], errors="coerce")
            break
    if dt is None:
        # Fallback to legacy split parts
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
    df["tod_min"] = df["date_time"].apply(lambda x: _minutes_since_midnight(x) if pd.notna(x) else np.nan)

    # ---------------- Syllable order ----------------
    def _derive_order_from_dict(row) -> list[str]:
        # Prefer ms, fall back to timebins
        dct = row.get("syllable_onsets_offsets_ms_dict") or row.get("syllable_onsets_offsets_ms")
        if not isinstance(dct, dict) or len(dct) == 0:
            dct = row.get("syllable_onsets_offsets_timebins")
            if not isinstance(dct, dict) or len(dct) == 0:
                return []
        events = []
        for lab, spans in dct.items():
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
            if lab != seq[-1]:  # collapse repeats
                seq.append(lab)
        return seq

    if "syllable_order" not in df.columns:
        df["syllable_order"] = df.apply(_derive_order_from_dict, axis=1)
    else:
        def _coerce_seq(x):
            if isinstance(x, (list, tuple)):
                return [str(v) for v in x]
            return []
        df["syllable_order"] = df["syllable_order"].apply(_coerce_seq)

    # ---------------- Transition frequencies ----------------
    def _freq(seq: list[str]) -> dict[tuple[str, str], int]:
        from collections import Counter
        if not seq or len(seq) < 2:
            return {}
        bigrams = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
        return dict(Counter(bigrams))

    df["transition_frequencies"] = df["syllable_order"].apply(_freq)

    ok = (~df["date_time"].isna()) & (df["transition_frequencies"].apply(lambda d: isinstance(d, dict) and len(d) > 0))
    keep_cols = ["date_time", "day", "tod_min", "transition_frequencies"]
    for name in ["file_name", "File Stem", "Segment", "segment"]:
        if name in df.columns:
            keep_cols.append(name)

    return df.loc[ok, keep_cols].reset_index(drop=True)

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

def _per_file_total_transition_entropy(freq_dict: Dict[Tuple[str, str], int], syllable_types: List[str]) -> float:
    if not freq_dict:
        return 0.0
    tr_list = list(freq_dict.keys())
    tr_counts = list(freq_dict.values())
    norm_mat, counts_mat = _generate_normalized_transition_matrix(tr_list, tr_counts, syllable_types)
    ent_by = _transition_entropy_per_row(norm_mat, syllable_types)
    return _total_transition_entropy(counts_mat, syllable_types, ent_by)

def _subset_aggregated_tte(subdf: pd.DataFrame, syllable_types: List[str]) -> float:
    """
    Sum bigram counts across all files in the subset, then compute one TTE.
    """
    if subdf.empty:
        return float("nan")
    counts: Dict[Tuple[str, str], int] = {}
    for freq in subdf["transition_frequencies"]:
        for tr, cnt in freq.items():
            counts[tr] = counts.get(tr, 0) + cnt
    if not counts:
        return float("nan")
    tr_list = list(counts.keys())
    tr_counts = list(counts.values())
    norm_mat, counts_mat = _generate_normalized_transition_matrix(tr_list, tr_counts, syllable_types)
    ent_by = _transition_entropy_per_row(norm_mat, syllable_types)
    return _total_transition_entropy(counts_mat, syllable_types, ent_by)

def _save_dir(fig_dir: Union[str, Path, None], decoded_database_json: Path) -> Path:
    if fig_dir is None or str(fig_dir).strip() == "":
        fig_dir = decoded_database_json.parent
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

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
    # A) Organizer attribute
    if od is not None:
        for attr in ["animal_id", "animalID", "Animal_ID"]:
            if hasattr(od, attr):
                val = getattr(od, attr)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    # B) DataFrame contents
    if organized_df is not None:
        guess = _infer_animal_id_from_df(organized_df)
        if guess:
            return guess
    # C) Paths (metadata optional)
    texts: List[str] = [str(decoded_path), decoded_path.stem]
    if metadata_path is not None:
        texts.extend([str(metadata_path), metadata_path.stem])
    for text in texts:
        guess = _extract_id_from_text(text)
        if guess:
            return guess
    # D) Fallback
    return decoded_path.stem.split("_")[0]

def _mean_sem(vals: List[float]) -> Tuple[float, Optional[float], int]:
    n = len(vals)
    if n == 0:
        return float("nan"), None, 0
    if n == 1:
        return float(vals[0]), 0.0, 1
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(n)), n

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SetHourAndBatchTTEResult:
    results_df: pd.DataFrame
    figure_path_paired: Optional[Path]
    figure_path_overlay: Optional[Path]
    figure_path_paired_sem: Optional[Path]
    figure_path_overlay_sem: Optional[Path]

def run_set_hour_and_batch_TTE(
    *,
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Optional[Union[str, Path]] = None,  # ← now optional
    range1: str = "00:00-12:00",
    range2: str = "12:00-23:59",
    batch_size: int = 10,
    min_required_per_range: Optional[int] = None,
    only_song_present: bool = True,
    compute_durations: bool = False,
    fig_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    organize_builder: Optional[Callable] = _ORGANIZE_DEFAULT,
) -> SetHourAndBatchTTEResult:

    decoded_database_json = Path(decoded_database_json)
    metadata_path = Path(creation_metadata_json) if creation_metadata_json else None

    builder = organize_builder or _ORGANIZE_DEFAULT
    if builder is None:
        raise ImportError(
            "No organizer available. Ensure one of these modules is importable:\n"
            "  - organized_decoded_serialTS_segments.py\n"
            "  - organize_decoded_serialTS_segments.py\n"
            "  - organize_decoded_with_segments.py"
        )

    # Call organizer with signature-aware kwargs (no metadata required)
    params = inspect.signature(builder).parameters
    kwargs = {
        "decoded_database_json": decoded_database_json,
        "only_song_present": only_song_present,
        "compute_durations": compute_durations,
    }
    if "add_recording_datetime" in params:
        kwargs["add_recording_datetime"] = True
    if "creation_metadata_json" in params:
        kwargs["creation_metadata_json"] = metadata_path  # may be None; organizer may ignore

    od = builder(**kwargs)

    organized = od.organized_df
    animal_id = _infer_animal_id(decoded_database_json, metadata_path, organized_df=organized, od=od)

    pf = _build_per_file_table(organized)
    if pf.empty:
        print("No valid per-file transitions (check dates or syllable_order).")
        return SetHourAndBatchTTEResult(pd.DataFrame(), None, None, None, None)

    # Global label set (consistent axes)
    all_transitions: Dict[Tuple[str, str], int] = {}
    for d in pf["transition_frequencies"]:
        for tr, cnt in d.items():
            all_transitions[tr] = all_transitions.get(tr, 0) + cnt
    syllable_types = sorted({a for a, _ in all_transitions} | {b for _, b in all_transitions})

    r1_start, r1_end = _parse_time_range(range1)
    r2_start, r2_end = _parse_time_range(range2)
    if min_required_per_range is None:
        min_required_per_range = batch_size

    rows = []
    for day, grp in pf.groupby("day"):
        # Segment-aware sort: date_time, then Segment/segment/file_name
        secondary = None
        for cand in ["Segment", "segment", "file_name"]:
            if cand in grp.columns:
                secondary = cand
                break
        sort_by = ["date_time"] + ([secondary] if secondary else [])
        g = grp.sort_values(sort_by, ascending=True).copy()

        g1 = g[(g["tod_min"] >= r1_start) & (g["tod_min"] <= r1_end)].copy()
        g2 = g[(g["tod_min"] >= r2_start) & (g["tod_min"] <= r2_end)].copy()

        firstN = g1.head(batch_size)
        lastN  = g2.tail(batch_size)

        # Per-file TTEs (for SEM and auditing)
        def tte_list(subdf: pd.DataFrame) -> List[float]:
            if subdf.empty:
                return []
            return [
                _per_file_total_transition_entropy(freq, syllable_types)
                for freq in subdf["transition_frequencies"]
            ]

        firstN_ttes = tte_list(firstN)
        lastN_ttes  = tte_list(lastN)

        firstN_files = list(firstN["file_name"]) if "file_name" in firstN.columns else []
        lastN_files  = list(lastN["file_name"]) if "file_name" in lastN.columns else []

        # Aggregated TTE (plotted & stored in *_avg_TTE)
        r1_agg = _subset_aggregated_tte(firstN, syllable_types) if len(firstN) >= min_required_per_range else float("nan")
        r2_agg = _subset_aggregated_tte(lastN,  syllable_types) if len(lastN)  >= min_required_per_range else float("nan")

        # SEM from per-file TTEs
        r1_mean_perfile, r1_sem, _ = _mean_sem(firstN_ttes)
        r2_mean_perfile, r2_sem, _ = _mean_sem(lastN_ttes)
        if len(firstN) < min_required_per_range:
            r1_sem = None
        if len(lastN) < min_required_per_range:
            r2_sem = None

        rows.append({
            "day": day,
            "range1_firstN_avg_TTE": r1_agg,
            "range2_lastN_avg_TTE":  r2_agg,
            "range1_sem": r1_sem,
            "range2_sem": r2_sem,
            "range1_count": int(len(firstN)),
            "range1_firstN_TTEs": firstN_ttes,
            "range1_firstN_file_names": firstN_files,
            "range2_count": int(len(lastN)),
            "range2_lastN_TTEs": lastN_ttes,
            "range2_lastN_file_names": lastN_files,
            "range1_firstN_avg_TTE_perfile": r1_mean_perfile,  # ref only
            "range2_lastN_avg_TTE_perfile": r2_mean_perfile,   # ref only
        })

    results_df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    fig_dir = _save_dir(fig_dir, decoded_database_json)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 1: Paired (no error bars)
    # ──────────────────────────────────────────────────────────────────────────
    fig_path_paired = None
    if not results_df.empty:
        plt.figure(figsize=(12, 6))
        handles = []; labels = []
        for _, row in results_df.iterrows():
            dstr = _date_str(row["day"])
            y1 = row["range1_firstN_avg_TTE"]
            y2 = row["range2_lastN_avg_TTE"]
            xs = []; ys = []
            if pd.notna(y1): xs.append(0); ys.append(y1)
            if pd.notna(y2): xs.append(1); ys.append(y2)
            if not xs: continue
            h, = plt.plot(xs, ys, marker="o", linewidth=2 if len(xs) == 2 else 1.5, label=dstr)
            handles.append(h); labels.append(dstr)

        plt.xticks([0, 1], [f"{range1}\n(first N)", f"{range2}\n(last N)"])
        plt.ylabel("Total Transition Entropy (bits)")
        plt.title(f"{animal_id} – Per-day aggregated TTE\n"
                  f"First {batch_size} in {range1} vs Last {batch_size} in {range2}")
        if handles:
            plt.legend(handles=handles, labels=labels, title="Day",
                       bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig_path_paired = fig_dir / f"{animal_id}_set_hour_and_batch_TTE_batch{batch_size}_paired.png"
        plt.savefig(fig_path_paired, dpi=200)
        if show: plt.show()
        else: plt.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 2: Overlay (no error bars)
    # ──────────────────────────────────────────────────────────────────────────
    fig_path_overlay = None
    if not results_df.empty:
        plt.figure(figsize=(12, 6))
        handles = []; labels = []
        for _, row in results_df.iterrows():
            dstr = _date_str(row["day"])
            y1 = row["range1_firstN_avg_TTE"]
            y2 = row["range2_lastN_avg_TTE"]
            xs = []; ys = []
            if pd.notna(y1): xs.append(0); ys.append(y1)
            if pd.notna(y2): xs.append(1); ys.append(y2)
            if not xs: continue
            h, = plt.plot(xs, ys, marker="o", linewidth=2 if len(xs) == 2 else 1.5, label=dstr)
            handles.append(h); labels.append(dstr)

        plt.xticks([0, 1], [f"{range1}\n(first {batch_size} songs)", f"{range2}\n(last {batch_size} songs)"])
        plt.ylabel("Total Transition Entropy (bits)")
        plt.title(f"{animal_id} – Per-day aggregated TTE (overlay)\n"
                  f"First {batch_size} in {range1} vs Last {batch_size} in {range2}")
        if handles:
            plt.legend(handles=handles, labels=labels, title="Day",
                       bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig_path_overlay = fig_dir / f"{animal_id}_set_hour_and_batch_TTE_batch{batch_size}_overlay.png"
        plt.savefig(fig_path_overlay, dpi=200)
        if show: plt.show()
        else: plt.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 3: Paired with SEM error bars
    # ──────────────────────────────────────────────────────────────────────────
    fig_path_paired_sem = None
    if not results_df.empty:
        x_vals, y_vals, y_sem, labels = [], [], [], []
        for _, row in results_df.iterrows():
            dstr = _date_str(row["day"])
            if pd.notna(row["range1_firstN_avg_TTE"]):
                x_vals.append(len(x_vals)); y_vals.append(row["range1_firstN_avg_TTE"])
                y_sem.append(row["range1_sem"] if row["range1_sem"] is not None else 0.0)
                labels.append(f"{dstr}\n({range1} first {batch_size})")
            if pd.notna(row["range2_lastN_avg_TTE"]):
                x_vals.append(len(x_vals)); y_vals.append(row["range2_lastN_avg_TTE"])
                y_sem.append(row["range2_sem"] if row["range2_sem"] is not None else 0.0)
                labels.append(f"{dstr}\n({range2} last {batch_size})")

        if x_vals:
            plt.figure(figsize=(max(10, len(x_vals) * 0.7), 6))
            plt.errorbar(x_vals, y_vals, yerr=y_sem, fmt='o', capsize=4, elinewidth=1.2)
            cursor = 0
            for _, row in results_df.iterrows():
                have_r1 = pd.notna(row["range1_firstN_avg_TTE"])
                have_r2 = pd.notna(row["range2_lastN_avg_TTE"])
                if have_r1 and have_r2:
                    plt.plot([cursor, cursor + 1],
                             [row["range1_firstN_avg_TTE"], row["range2_lastN_avg_TTE"]],
                             linestyle="--", linewidth=1)
                    cursor += 2
                elif have_r1 or have_r2:
                    cursor += 1

            plt.ylabel("Average TTE ± SEM (bits)")
            plt.title(f"{animal_id} – Per-day aggregated TTE with SEM\n"
                      f"First {batch_size} in {range1} vs Last {batch_size} in {range2}")
            plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
            plt.tight_layout()
            fig_path_paired_sem = fig_dir / f"{animal_id}_set_hour_and_batch_TTE_batch{batch_size}_paired_sem.png"
            plt.savefig(fig_path_paired_sem, dpi=200)
            if show: plt.show()
            else: plt.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 4: Overlay with SEM error bars (color-coded by day)
    # ──────────────────────────────────────────────────────────────────────────
    fig_path_overlay_sem = None
    if not results_df.empty:
        plt.figure(figsize=(12, 6))
        handles = []; legend_labels = []
        for _, row in results_df.iterrows():
            dstr = _date_str(row["day"])
            have_r1 = pd.notna(row["range1_firstN_avg_TTE"])
            have_r2 = pd.notna(row["range2_lastN_avg_TTE"])
            if not (have_r1 or have_r2):
                continue

            xs = []; ys = []; es = []
            if have_r1:
                xs.append(0); ys.append(row["range1_firstN_avg_TTE"])
                es.append(row["range1_sem"] if row["range1_sem"] is not None else 0.0)
            if have_r2:
                xs.append(1); ys.append(row["range2_lastN_avg_TTE"])
                es.append(row["range2_sem"] if row["range2_sem"] is not None else 0.0)

            h_line, = plt.plot(xs, ys, linewidth=2.0 if len(xs) == 2 else 1.5,
                               label=dstr, marker='o')
            line_color = h_line.get_color()
            plt.errorbar(xs, ys, yerr=es, fmt='none', capsize=4, elinewidth=1.2,
                         ecolor=line_color)

            handles.append(h_line); legend_labels.append(dstr)

        plt.xticks([0, 1], [f"{range1}\n(first {batch_size} songs)",
                            f"{range2}\n(last {batch_size} songs)"])
        plt.ylabel("Average TTE ± SEM (bits)")
        plt.title(f"{animal_id} – Per-day aggregated TTE (overlay) with SEM\n"
                  f"First {batch_size} in {range1} vs Last {batch_size} in {range2}")
        if handles:
            plt.legend(handles=handles, labels=legend_labels, title="Day",
                       bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig_path_overlay_sem = fig_dir / f"{animal_id}_set_hour_and_batch_TTE_batch{batch_size}_overlay_sem.png"
        plt.savefig(fig_path_overlay_sem, dpi=200)
        if show: plt.show()
        else: plt.close()

    return SetHourAndBatchTTEResult(
        results_df=results_df,
        figure_path_paired=fig_path_paired,
        figure_path_overlay=fig_path_overlay,
        figure_path_paired_sem=fig_path_paired_sem,
        figure_path_overlay_sem=fig_path_overlay_sem,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def summarize_tte_selection(results_df: pd.DataFrame, *, batch_size: int = 10) -> None:
    for _, row in results_df.iterrows():
        day = row["day"]
        r1_count = int(row.get("range1_count", 0))
        r2_count = int(row.get("range2_count", 0))
        r1_files = row.get("range1_firstN_file_names", []) or []
        r2_files = row.get("range2_lastN_file_names", []) or []

        print(f"\n=== {pd.to_datetime(day).date()} ===")
        print(f"Range1 (first {batch_size}): {r1_count} segments"
              + (f"  ⚠️ missing {batch_size - r1_count}" if r1_count < batch_size else ""))
        for fn in r1_files:
            print("   ", fn)

        print(f"\nRange2 (last {batch_size}): {r2_count} segments"
              + (f"  ⚠️ missing {batch_size - r2_count}" if r2_count < batch_size else ""))
        for fn in r2_files:
            print("   ", fn)

def summarize_tte_selection_or_raise(results_df: pd.DataFrame, *, batch_size: int = 10) -> None:
    summarize_tte_selection(results_df, batch_size=batch_size)
    problems = []
    for _, row in results_df.iterrows():
        day = pd.to_datetime(row["day"]).date()
        r1_count = int(row.get("range1_count", 0))
        r2_count = int(row.get("range2_count", 0))
        if r1_count < batch_size:
            problems.append(f"{day}: range1 captured {r1_count} < {batch_size}")
        if r2_count < batch_size:
            problems.append(f"{day}: range2 captured {r2_count} < {batch_size}")
    if problems:
        raise ValueError("Segment capture check failed:\n  - " + "\n  - ".join(problems))

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Average TTE of FIRST N songs in range1 vs LAST N songs in range2, per day."
    )
    p.add_argument("--decoded", required=True, help="Path to decoded_database.json")
    p.add_argument("--meta", default=None,
                   help="(Optional) Path to metadata .json — not required for Excel-serial organizer.")
    p.add_argument("--range1", default="00:00-12:00", help="Time range 1, e.g., '00:00-12:00'")
    p.add_argument("--range2", default="12:00-23:59", help="Time range 2, e.g., '12:00-23:59'")
    p.add_argument("--batch_size", type=int, default=10, help="Songs to take in each subset")
    p.add_argument("--min_required", type=int, default=None,
                   help="Minimum songs required per subset (default = batch_size). If unmet, avg is NaN.")
    p.add_argument("--only_song_present", action="store_true",
                   help="Restrict to song_present rows in organizer")
    p.add_argument("--fig_dir", default=None, help="Directory to save figures (default: decoded file's folder)")
    p.add_argument("--no_show", action="store_true", help="Do not display plot windows")
    return p

def main():
    args = _build_arg_parser().parse_args()
    res = run_set_hour_and_batch_TTE(
        decoded_database_json=args.decoded,
        creation_metadata_json=args.meta,  # may be None
        range1=args.range1,
        range2=args.range2,
        batch_size=args.batch_size,
        min_required_per_range=args.min_required,
        only_song_present=args.only_song_present,
        compute_durations=False,
        fig_dir=args.fig_dir,
        show=not args.no_show,
        organize_builder=_ORGANIZE_DEFAULT,
    )
    print("\nResults (first 10 rows):")
    if not res.results_df.empty:
        print(res.results_df.head(10).to_string(index=False))
    else:
        print("No results.")
    if res.figure_path_paired:
        print(f"Paired plot saved → {res.figure_path_paired}")
    if res.figure_path_overlay:
        print(f"Overlay plot saved → {res.figure_path_overlay}")
    if res.figure_path_paired_sem:
        print(f"Paired SEM plot saved → {res.figure_path_paired_sem}")
    if res.figure_path_overlay_sem:
        print(f"Overlay SEM plot saved → {res.figure_path_overlay_sem}")

if __name__ == "__main__":
    main()




"""
import importlib, tte_set_hour_and_batch as mod
importlib.reload(mod)
from tte_set_hour_and_batch import run_set_hour_and_batch_TTE, summarize_tte_selection_or_raise


decoded = "/Users/mirandahulsey-vincent/Desktop/USA1234_VALIDATION_Data/USA1234_decoded_database.json"

out = run_set_hour_and_batch_TTE(
    decoded_database_json=decoded,
    range1="00:00-12:00",
    range2="12:00-23:59",
    batch_size=30,
    min_required_per_range=30,
    only_song_present=True,
    fig_dir=None,
    show=True,
)

summarize_tte_selection_or_raise(out.results_df, batch_size=20)


"""