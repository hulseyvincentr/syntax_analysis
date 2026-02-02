# outlier_phrase_duration_wrapper.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import outlier_graphs as og

# Optional: used to build merged DF (recommended)
try:
    import merge_annotations_from_split_songs as mas
    _HAS_MERGE_ANNOT = True
except Exception:
    mas = None
    _HAS_MERGE_ANNOT = False


# -----------------------------------------------------------------------------
# JSON discovery helpers (nested folder structure + ignore AppleDouble ._ files)
# -----------------------------------------------------------------------------
def _is_bad_mac_file(p: Path) -> bool:
    n = p.name
    return (
        n.startswith("._") or n == ".DS_Store" or n.startswith(".DS_Store")
        or (n.startswith(".") and n.endswith(".json") and n.startswith("._"))
    )


def _score_dir_name(dir_name: str, animal_id: str) -> int:
    """Higher score = better match."""
    dn = dir_name.lower()
    aid = animal_id.lower()
    if dn == aid:
        return 100
    if dn.startswith(aid):
        return 80
    if aid in dn:
        return 60
    return 0


def _find_candidate_animal_dirs(outputs_root: Path, animal_id: str) -> List[Path]:
    """
    Find candidate directories under outputs_root whose folder name matches animal_id.
    Prefers immediate children, but falls back to recursive directory search.
    """
    outputs_root = Path(outputs_root)
    candidates: List[Path] = []

    # 1) Prefer immediate child dirs (fast)
    if outputs_root.exists():
        for p in outputs_root.iterdir():
            if p.is_dir() and _score_dir_name(p.name, animal_id) > 0:
                candidates.append(p)

    # 2) If nothing found, fall back to recursive dir scan
    if not candidates:
        for p in outputs_root.rglob("*"):
            if p.is_dir() and _score_dir_name(p.name, animal_id) > 0:
                candidates.append(p)

    def _sort_key(p: Path):
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        depth = len(p.parts)
        return (-_score_dir_name(p.name, animal_id), depth, -mtime)

    return sorted(candidates, key=_sort_key)


def _pick_best_file(files: List[Path], *, animal_id: str, prefer_tokens: Sequence[str] = ()) -> Optional[Path]:
    """Pick the most likely correct JSON among multiple matches (ignores '._' files)."""
    if not files:
        return None

    # Filter macOS AppleDouble + obvious junk
    files = [p for p in files if p.is_file() and not _is_bad_mac_file(p)]
    if not files:
        return None

    aid = animal_id.lower()
    toks = [t.lower() for t in prefer_tokens]

    def _score(p: Path) -> Tuple[int, float]:
        name = p.name.lower()
        score = 0

        # Strong penalty if hidden-ish file
        if p.name.startswith("._"):
            score -= 1000
        if p.name.startswith("."):
            score -= 50

        if aid and aid in name:
            score += 5
        for t in toks:
            if t and t in name:
                score += 1

        # Prefer larger files (real JSONs usually bigger than tiny stubs)
        try:
            size = p.stat().st_size
        except Exception:
            size = 0

        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0

        # (score, size, mtime)
        return (score, size + 0.0, mtime)

    files_sorted = sorted(files, key=lambda p: _score(p), reverse=True)
    return files_sorted[0]


def _find_jsons_in_dir(animal_dir: Path, animal_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Search within an animal directory (recursively) for decoded_database + song_detection JSONs.
    """
    animal_dir = Path(animal_dir)

    decoded_matches = list(animal_dir.rglob("*decoded_database.json"))
    if not decoded_matches:
        decoded_matches = list(animal_dir.rglob("*decoded*database*.json"))

    detect_matches = list(animal_dir.rglob("*song_detection.json"))
    if not detect_matches:
        detect_matches = list(animal_dir.rglob("*song*detect*.json"))

    decoded = _pick_best_file(decoded_matches, animal_id=animal_id, prefer_tokens=("decoded_database", "decoded"))
    detect = _pick_best_file(detect_matches, animal_id=animal_id, prefer_tokens=("song_detection", "detect"))

    return decoded, detect


def _find_jsons_for_animal(
    outputs_root: Union[str, Path],
    animal_id: str,
    *,
    verbose: bool = False
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    outputs_root structure:
      /.../updated_AreaX_outputs/
        ├─ R08/
        │    ├─ R08_decoded_database.json
        │    └─ R08_song_detection.json
        ├─ USA5288/
        │    └─ ...
        └─ ...
    """
    root = Path(outputs_root)

    candidates = _find_candidate_animal_dirs(root, animal_id)

    for d in candidates:
        decoded, detect = _find_jsons_in_dir(d, animal_id)
        if decoded is not None and detect is not None:
            if verbose:
                print(f"[FOUND] {animal_id}: using dir={d}")
                print(f"        decoded={decoded.name}")
                print(f"        detect ={detect.name}")
            return decoded, detect

    # Fallback brute-force
    decoded_matches = list(root.rglob(f"*{animal_id}*decoded*database*.json"))
    detect_matches  = list(root.rglob(f"*{animal_id}*song*detect*.json"))

    decoded = _pick_best_file(decoded_matches, animal_id=animal_id)
    detect  = _pick_best_file(detect_matches, animal_id=animal_id)

    if verbose:
        if decoded or detect:
            print(f"[FALLBACK FOUND] {animal_id}:")
            print(f"  decoded={decoded}")
            print(f"  detect ={detect}")
        else:
            print(f"[MISSING] {animal_id}: no decoded/detect JSONs found under {root}")

    return decoded, detect


# -----------------------------------------------------------------------------
# Metadata (treatment date + lesion/sham) helpers
# -----------------------------------------------------------------------------
def _load_metadata_excel(metadata_excel: Union[str, Path]) -> pd.DataFrame:
    """
    Loads the sheet that contains 'Animal ID' and returns it.
    Your uploaded file uses sheet_name='metadata'.
    """
    metadata_excel = Path(metadata_excel)
    xls = pd.ExcelFile(metadata_excel)

    for sheet in xls.sheet_names:
        df = pd.read_excel(metadata_excel, sheet_name=sheet)
        if any(str(c).strip().lower() == "animal id" for c in df.columns):
            return df
    # fallback: first sheet
    return pd.read_excel(metadata_excel, sheet_name=xls.sheet_names[0])


def _get_treatment_date_and_group(meta_df: pd.DataFrame, animal_id: str) -> Tuple[Optional[pd.Timestamp], str]:
    """
    Returns:
      treatment_date (normalized Timestamp or None),
      lesion_group in {"NMA lesion", "Sham lesion", "Unknown"}
    """
    # Column name normalization
    cols = {str(c).strip().lower(): c for c in meta_df.columns}

    aid_col = cols.get("animal id", None)
    date_col = cols.get("treatment date", None)
    type_col = cols.get("treatment type", None)

    if aid_col is None:
        return None, "Unknown"

    sub = meta_df.loc[meta_df[aid_col].astype(str) == str(animal_id)]
    if sub.empty:
        return None, "Unknown"

    # Treatment date
    tdt = None
    if date_col is not None and sub[date_col].notna().any():
        tdt_raw = sub[date_col].dropna().iloc[0]
        tdt = pd.to_datetime(tdt_raw, errors="coerce")
        if pd.notna(tdt):
            tdt = tdt.normalize()
        else:
            tdt = None

    # Lesion group
    group = "Unknown"
    if type_col is not None and sub[type_col].notna().any():
        tt = str(sub[type_col].dropna().iloc[0]).lower()
        if "nma" in tt:
            group = "NMA lesion"
        elif "saline" in tt or "sham" in tt:
            group = "Sham lesion"
        else:
            group = "Unknown"

    return tdt, group


# -----------------------------------------------------------------------------
# Merged DF -> daily median phrase duration per bird×syllable
# -----------------------------------------------------------------------------
def _parse_intervals_to_durations_ms(intervals_obj) -> List[float]:
    """
    Accepts dict value like [[on, off], ...] (ms), returns durations in ms.
    """
    out: List[float] = []
    if intervals_obj is None:
        return out

    if isinstance(intervals_obj, (list, tuple)) and len(intervals_obj) == 2 and all(
        isinstance(x, (int, float)) for x in intervals_obj
    ):
        intervals_obj = [intervals_obj]

    if not isinstance(intervals_obj, (list, tuple)):
        return out

    for itv in intervals_obj:
        if not (isinstance(itv, (list, tuple)) and len(itv) >= 2):
            continue
        try:
            on = float(itv[0])
            off = float(itv[1])
            dur = off - on
            if dur >= 0:
                out.append(dur)
        except Exception:
            continue
    return out


def _ensure_duration_column_from_dict(
    df: pd.DataFrame,
    syllable_label: str,
    *,
    dict_col: str = "syllable_onsets_offsets_ms_dict",
) -> pd.Series:
    """
    Builds a Series of list[float] durations for `syllable_label`
    using df[dict_col] dicts (ms intervals).
    """
    lab = str(syllable_label)

    def _get_list(row):
        d = row
        if not isinstance(d, dict):
            return []
        intervals = d.get(lab, None)
        return _parse_intervals_to_durations_ms(intervals)

    if dict_col not in df.columns:
        return pd.Series([[]] * len(df), index=df.index)

    # df[dict_col] may contain dicts or strings; merge builder should be dicts
    return df[dict_col].apply(lambda v: _get_list(v) if isinstance(v, dict) else [])


def _build_daily_medians_for_bird(
    merged_df: pd.DataFrame,
    animal_id: str,
    syllables: Sequence[str],
    treatment_date: pd.Timestamp,
    lesion_group: str,
) -> pd.DataFrame:
    """
    Returns long-form table:
      Animal ID, Syllable, DateDay, DaysRel, DailyMedianMs, LesionGroup
    """
    df = merged_df.copy()

    # Recording datetime
    if "Recording DateTime" not in df.columns:
        raise KeyError("Merged DF missing 'Recording DateTime' column.")

    dt = pd.to_datetime(df["Recording DateTime"], errors="coerce")
    df = df.assign(_dt=dt).dropna(subset=["_dt"])
    if df.empty:
        return pd.DataFrame(columns=["Animal ID", "Syllable", "DateDay", "DaysRel", "DailyMedianMs", "LesionGroup"])

    df["DateDay"] = df["_dt"].dt.normalize()

    rows = []
    for s in map(str, syllables):
        col = f"syllable_{s}_durations"

        if col in df.columns and df[col].notna().any():
            dur_lists = df[col]
        else:
            # fallback: compute from dict column
            dur_lists = _ensure_duration_column_from_dict(df, s)

        tmp = pd.DataFrame({"DateDay": df["DateDay"], "dur": dur_lists})
        tmp = tmp.explode("dur")
        tmp["dur"] = pd.to_numeric(tmp["dur"], errors="coerce")
        tmp = tmp.dropna(subset=["dur"])
        if tmp.empty:
            continue

        daily_med = tmp.groupby("DateDay", as_index=False)["dur"].median()
        daily_med = daily_med.rename(columns={"dur": "DailyMedianMs"})
        daily_med["Animal ID"] = str(animal_id)
        daily_med["Syllable"] = str(s)
        daily_med["LesionGroup"] = lesion_group
        daily_med["DaysRel"] = (daily_med["DateDay"] - treatment_date).dt.days.astype(int)

        rows.append(daily_med)

    if not rows:
        return pd.DataFrame(columns=["Animal ID", "Syllable", "DateDay", "DaysRel", "DailyMedianMs", "LesionGroup"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["Animal ID", "Syllable", "DaysRel"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Plotting (two aggregate plots)
# -----------------------------------------------------------------------------
def _despine_and_grid(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _auto_ylim(y: np.ndarray, pad_frac: float = 0.05) -> Tuple[float, float]:
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (0.0, 1.0)
    y_max = float(np.max(y))
    if y_max <= 0:
        return (0.0, 1.0)
    return (0.0, y_max * (1.0 + pad_frac))


def plot_aggregate_lines_color_by_lesion_group(
    daily_df: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    title: str = "Phrase duration vs days relative to treatment (one line per bird×syllable; color = lesion group)",
    figsize: Tuple[int, int] = (18, 7),
    alpha: float = 0.35,
    lw: float = 1.0,
    marker_size: float = 2.0,
    y_max_ms: Optional[float] = None,
    legend_outside: bool = True,
) -> None:
    """
    Plots one line per (Animal ID, Syllable) using daily median phrase duration.
    Color encodes lesion group (NMA vs Sham vs Unknown).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    color_map = {
        "NMA lesion": "tab:blue",
        "Sham lesion": "tab:red",
        "Unknown": "tab:gray",
    }

    fig, ax = plt.subplots(figsize=figsize)

    # vertical line at treatment (day 0)
    ax.axvline(0, linestyle="--", linewidth=1.2, color="k", alpha=0.6)

    # Plot each bird×syllable as its own series
    for (aid, syl), sub in daily_df.groupby(["Animal ID", "Syllable"], sort=False):
        grp = str(sub["LesionGroup"].iloc[0]) if "LesionGroup" in sub.columns else "Unknown"
        c = color_map.get(grp, "tab:gray")

        x = sub["DaysRel"].to_numpy()
        y = sub["DailyMedianMs"].to_numpy()

        ax.plot(x, y, color=c, alpha=alpha, linewidth=lw)
        ax.scatter(x, y, color=c, alpha=alpha, s=marker_size)

    ax.set_title(title)
    ax.set_xlabel("Days relative to treatment")
    ax.set_ylabel("Daily median phrase duration (ms)")
    _despine_and_grid(ax)

    # y-limits
    if y_max_ms is None:
        ymin, ymax = _auto_ylim(daily_df["DailyMedianMs"].to_numpy(), pad_frac=0.05)
    else:
        ymin, ymax = 0.0, float(y_max_ms)
    ax.set_ylim(ymin, ymax)

    # Legend
    handles = []
    labels = []
    for k in ["NMA lesion", "Sham lesion", "Unknown"]:
        handles.append(plt.Line2D([0], [0], color=color_map[k], lw=3))
        labels.append(k)

    if legend_outside:
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.subplots_adjust(right=0.82)
    else:
        ax.legend(handles, labels, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_aggregate_lines_color_by_animal_id(
    daily_df: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    title: str = "Phrase duration vs days relative to treatment (one line per bird×syllable; color = animal id)",
    figsize: Tuple[int, int] = (18, 7),
    alpha: float = 0.35,
    lw: float = 1.0,
    marker_size: float = 2.0,
    y_max_ms: Optional[float] = None,
    legend_outside: bool = True,
) -> None:
    """
    Same data, but color encodes Animal ID.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, linestyle="--", linewidth=1.2, color="k", alpha=0.6)

    animal_ids = list(dict.fromkeys(daily_df["Animal ID"].astype(str).tolist()))
    cmap = plt.get_cmap("tab20")
    aid_to_color = {aid: cmap(i % 20) for i, aid in enumerate(animal_ids)}

    for (aid, syl), sub in daily_df.groupby(["Animal ID", "Syllable"], sort=False):
        c = aid_to_color.get(str(aid), "tab:gray")
        x = sub["DaysRel"].to_numpy()
        y = sub["DailyMedianMs"].to_numpy()

        ax.plot(x, y, color=c, alpha=alpha, linewidth=lw)
        ax.scatter(x, y, color=c, alpha=alpha, s=marker_size)

    ax.set_title(title)
    ax.set_xlabel("Days relative to treatment")
    ax.set_ylabel("Daily median phrase duration (ms)")
    _despine_and_grid(ax)

    if y_max_ms is None:
        ymin, ymax = _auto_ylim(daily_df["DailyMedianMs"].to_numpy(), pad_frac=0.05)
    else:
        ymin, ymax = 0.0, float(y_max_ms)
    ax.set_ylim(ymin, ymax)

    # Legend: one entry per animal
    handles = [plt.Line2D([0], [0], color=aid_to_color[aid], lw=3) for aid in animal_ids]
    labels = animal_ids

    if legend_outside:
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
        fig.subplots_adjust(right=0.80)
    else:
        ax.legend(handles, labels, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Public wrapper:
#   top-variance syllables per bird -> merged DF -> aggregate lines plots
# -----------------------------------------------------------------------------
def plot_daily_phrase_durations_for_top_variance_syllables(
    *,
    compiled_csv: Union[str, Path],
    updated_areax_outputs_root: Union[str, Path],
    metadata_excel: Union[str, Path],
    out_dir: Union[str, Path],
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    variance_col: str = "Variance_ms2",
    min_n_phrases: int = 10,
    # IMPORTANT: top 30% highest variance per bird -> top_percentile=70
    top_percentile: float = 70.0,
    rank_on: str = "post",
    # merge builder params
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    # plotting style
    y_max_ms: Optional[float] = None,   # None => auto-fit to data
    xtick_every: int = 1,               # kept for compatibility (not used in aggregate)
    show_plots: bool = False,           # kept for compatibility
    # debug/preview
    print_merged_preview: bool = False,
    merged_preview_rows: int = 3,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    1) Select top variance syllables per bird (from compiled_csv).
    2) For each bird, find decoded+song_detection JSONs under updated_areax_outputs_root/<animal>/...
    3) Build merged annotations DF (singles+merged) via merge_annotations_from_split_songs.
    4) Compute DAILY MEDIAN phrase duration per (bird, syllable) and convert x-axis to days relative to treatment.
    5) Save TWO aggregate plots:
         A) color-coded by lesion group (NMA vs Sham vs Unknown)
         B) color-coded by animal_id

    Note on the line:
      Each line = ONE (bird × syllable) series.
      Each point on the line = daily median of all phrase durations for that (bird, syllable) on that day.
    """
    compiled_csv = Path(compiled_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _HAS_MERGE_ANNOT or mas is None:
        raise RuntimeError(
            "merge_annotations_from_split_songs could not be imported. "
            "Make sure merge_annotations_from_split_songs.py is on your PYTHONPATH."
        )

    # Load metadata once
    meta_df = _load_metadata_excel(metadata_excel)

    # Step 1: select top variance syllables per bird
    pairs_dir = out_dir / "_variance_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    variance_res = og.plot_pre_post_variance_scatter(
        csv_path=compiled_csv,
        out_dir=pairs_dir,
        pre_group=pre_group,
        post_group=post_group,
        variance_col=variance_col,
        min_n_phrases=min_n_phrases,
        top_percentile=float(top_percentile),
        rank_on=rank_on,
        log_scale=True,
        metadata_excel=None,
        make_continuous_lesion_pct_plot=False,
        compute_identity_stats=False,
        annotate_identity_stats=False,
        do_perm_test=False,
        annotate_perm_test=False,
        do_tail_metrics=False,
        annotate_tail_metrics=False,
        make_ratio_vs_volume_plots=False,
        show=False,
        verbose=False,
    )

    top_table = variance_res["table"].copy()
    if top_table.empty:
        raise RuntimeError(
            "Top-variance selection produced an empty table. "
            "Try lowering min_n_phrases or adjusting top_percentile."
        )

    # Step 2/3/4: per bird, build merged DF and compute daily medians per syllable
    outputs_root = Path(updated_areax_outputs_root)
    per_bird_outputs: Dict[str, str] = {}
    daily_all: List[pd.DataFrame] = []

    for aid, sub in top_table.groupby("Animal ID", sort=False):
        aid = str(aid)
        sylls = sub["Syllable"].astype(str).unique().tolist()

        # Get treatment date + group
        tdate, lesion_group = _get_treatment_date_and_group(meta_df, aid)
        if tdate is None:
            if verbose:
                print(f"[SKIP] {aid}: missing treatment date in metadata.")
            continue

        decoded_path, detect_path = _find_jsons_for_animal(outputs_root, aid, verbose=verbose)
        if decoded_path is None or detect_path is None:
            if verbose:
                print(f"[SKIP] {aid}: missing decoded/detect JSON(s)")
            continue

        bird_out = out_dir / "daily_phrase_duration" / aid
        bird_out.mkdir(parents=True, exist_ok=True)
        per_bird_outputs[aid] = str(bird_out)

        if verbose:
            print(f"[MERGE] {aid}: n_syllables={len(sylls)} group={lesion_group} tdate={tdate.date()}")

        merged_res = mas.build_decoded_with_split_labels(
            decoded_database_json=decoded_path,
            song_detection_json=detect_path,
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
        merged_df = merged_res.annotations_appended_df

        if print_merged_preview:
            cols = [c for c in ["file_name", "Segment", "was_merged", "merged_n_parts",
                                "Recording DateTime", "Date", "Hour", "Minute", "Second"]
                    if c in merged_df.columns]
            print(f"[MERGED PREVIEW] {aid} shape={merged_df.shape}")
            if cols:
                print(merged_df[cols].head(int(merged_preview_rows)).to_string(index=False))

        daily_df = _build_daily_medians_for_bird(
            merged_df=merged_df,
            animal_id=aid,
            syllables=sylls,
            treatment_date=tdate,
            lesion_group=lesion_group,
        )
        if daily_df.empty:
            if verbose:
                print(f"[SKIP] {aid}: no usable durations for selected syllables.")
            continue

        daily_all.append(daily_df)

    if not daily_all:
        raise RuntimeError("No birds produced daily median data. Check JSON discovery and metadata treatment dates.")

    daily_all_df = pd.concat(daily_all, ignore_index=True)

    # Save the aggregated table (useful for stats)
    daily_csv = out_dir / "aggregate_daily_median_phrase_duration_by_bird_syllable.csv"
    daily_all_df.to_csv(daily_csv, index=False)
    if verbose:
        print(f"[WROTE] {daily_csv}")

    # Make plots
    plotA = out_dir / "aggregate_phrase_duration_rel_treatment_color_by_lesion_group.png"
    plotB = out_dir / "aggregate_phrase_duration_rel_treatment_color_by_animal_id.png"

    plot_aggregate_lines_color_by_lesion_group(
        daily_all_df,
        plotA,
        y_max_ms=y_max_ms,  # None => auto-fit
        legend_outside=True,
    )
    plot_aggregate_lines_color_by_animal_id(
        daily_all_df,
        plotB,
        y_max_ms=y_max_ms,  # None => auto-fit
        legend_outside=True,
    )

    if verbose:
        print(f"[PLOT] {plotA}")
        print(f"[PLOT] {plotB}")

    return {
        "variance_result": variance_res,
        "top_table": top_table,
        "per_bird_outputs": per_bird_outputs,
        "pairs_dir": str(pairs_dir),
        "out_dir": str(out_dir),
        "daily_medians_df": daily_all_df,
        "daily_csv": str(daily_csv),
        "plot_by_group": str(plotA),
        "plot_by_animal": str(plotB),
    }


# -----------------------------------------------------------------------------
# Example usage (Spyder)
# -----------------------------------------------------------------------------
"""
from pathlib import Path
import sys, importlib

code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
sys.path.insert(0, str(code_dir))

import outlier_phrase_duration_wrapper as wrap
importlib.reload(wrap)

compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv")
updated_root = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
meta_xlsx    = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")

out_dir      = compiled_csv.parent / "top30pct_variance_daily_phrase_duration"

res = wrap.plot_daily_phrase_durations_for_top_variance_syllables(
    compiled_csv=compiled_csv,
    updated_areax_outputs_root=updated_root,
    metadata_excel=meta_xlsx,
    out_dir=out_dir,
    top_percentile=70,          # top 30%
    rank_on="post",
    min_n_phrases=10,
    y_max_ms=None,              # auto-fit to data
    verbose=True,
    print_merged_preview=True,
    merged_preview_rows=3,
)

print("Done.")
print("Plot A (group): ", res["plot_by_group"])
print("Plot B (animal):", res["plot_by_animal"])
"""
