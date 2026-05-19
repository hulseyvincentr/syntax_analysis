#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_top_variance_syllable_spectrograms.py

Batch-generate pre/post stitched spectrogram examples for the high-variance
syllables selected by the top-30% phrase-duration-variance BC analysis.

Typical workflow
----------------
1) Run top30_phrase_variance_bc_only_analysis_v4.py.
2) Use its merged_top30_bc_rows.csv as --top30-csv here.
3) This script finds each animal's .npz file, pulls the top-variance labels,
   then calls your existing spectrogram plotting function/script to save
   qualitative pre/post spectrogram examples.

The script is designed for qualitative checks, so it defaults to:
    --num-sample-spectrograms 1
    --adaptive-length

That prevents accidentally generating hundreds of plots and helps labels with
fewer bins still produce a useful shorter example.

Expected top30 CSV columns
--------------------------
Works with:
    merged_top30_bc_rows.csv

Expected columns include:
    animal_id
    syllable_key or label/cluster
    is_top_phrase_variance  (optional; if absent, all rows are treated as selected)
    phrase_variance_ms2     (optional; used for sorting)
    phrase_variance_rank_within_bird (optional; used for sorting/output)

Expected NPZ location
---------------------
By default, the script looks for:
    <npz_root>/<animal_id>/<animal_id>.npz

It also falls back to recursive searching under --npz-root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
import argparse
import importlib.util
import math
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in low:
            return low[key]
    return None


def _label_key(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def _label_to_int(label: Any) -> Optional[int]:
    key = _label_key(label)
    if key == "":
        return None
    try:
        return int(key)
    except Exception:
        return None


def _parse_list_arg(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    text = s.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "t"])


def _clean_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _load_module_from_path(path: Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Spectrogram script not found: {path}")

    spec = importlib.util.spec_from_file_location("spectrogram_module_for_top_variance_qc", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import spectrogram script from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "plot_pre_post_spectrogram_samples_for_labels",
        "get_treatment_date_from_metadata",
        "get_timebin_recording_dates",
        "make_pre_post_masks",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Spectrogram script {path} is missing expected function(s): {missing}"
        )

    return module


def _find_spectrogram_script(provided: Optional[str], search_dirs: Sequence[Path]) -> Path:
    if provided:
        return Path(provided).expanduser().resolve()

    patterns = [
        "pre_post_syllable_sample_spectrograms_long_rows_with_bouts_v*.py",
        "pre_post_syllable_sample_spectrograms*.py",
    ]

    candidates: List[Path] = []
    for d in search_dirs:
        d = Path(d).expanduser()
        if not d.exists():
            continue
        for pat in patterns:
            candidates.extend(sorted(d.glob(pat)))

    # Prefer newer/higher version-looking files; then modification time.
    candidates = list(dict.fromkeys([c.resolve() for c in candidates if c.is_file()]))
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-find your pre/post spectrogram script. "
            "Pass it explicitly with --spectrogram-script."
        )

    candidates.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)
    return candidates[0]


def _find_npz_for_animal(
    animal_id: str,
    *,
    npz_root: Path,
    npz_template: Optional[str] = None,
) -> Optional[Path]:
    animal_id = str(animal_id).strip()
    npz_root = Path(npz_root).expanduser()

    candidates: List[Path] = []

    if npz_template:
        formatted = npz_template.format(npz_root=str(npz_root), animal_id=animal_id)
        candidates.append(Path(formatted).expanduser())

    candidates.extend([
        npz_root / animal_id / f"{animal_id}.npz",
        npz_root / f"{animal_id}.npz",
    ])

    for p in candidates:
        if p.exists():
            return p.resolve()

    # Fall back to recursive search.
    recursive_patterns = [
        f"**/{animal_id}.npz",
        f"**/{animal_id}_*.npz",
        f"**/*{animal_id}*.npz",
    ]
    found: List[Path] = []
    for pat in recursive_patterns:
        found.extend(npz_root.glob(pat))
        if found:
            break

    found = [p for p in found if p.is_file()]
    if not found:
        return None

    # Prefer exact stem match.
    exact = [p for p in found if p.stem == animal_id]
    if exact:
        return sorted(exact, key=lambda p: len(str(p)))[0].resolve()

    return sorted(found, key=lambda p: len(str(p)))[0].resolve()


def read_top_variance_table(
    top30_csv: Path,
    *,
    animal_col: Optional[str] = None,
    label_col: Optional[str] = None,
    phrase_variance_col: Optional[str] = None,
    rank_col: Optional[str] = None,
    animal_ids: Optional[Sequence[str]] = None,
    max_labels_per_bird: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(top30_csv)

    if animal_col is None:
        animal_col = _pick_column(df, ["animal_id", "Animal ID", "animal id", "bird", "Bird ID"])
    if label_col is None:
        label_col = _pick_column(df, ["syllable_key", "label", "cluster", "Syllable", "syllable"])
    if phrase_variance_col is None:
        phrase_variance_col = _pick_column(df, ["phrase_variance_ms2", "Variance_ms2", "variance_ms2"])
    if rank_col is None:
        rank_col = _pick_column(df, ["phrase_variance_rank_within_bird", "rank", "variance_rank"])

    if animal_col is None:
        raise KeyError(f"Could not infer animal ID column in {top30_csv}. Columns: {list(df.columns)}")
    if label_col is None:
        raise KeyError(f"Could not infer label/cluster column in {top30_csv}. Columns: {list(df.columns)}")

    out = df.copy()
    out["animal_id"] = out[animal_col].astype(str).str.strip()
    out["label_int"] = out[label_col].map(_label_to_int)

    if out["label_int"].isna().any():
        bad = out.loc[out["label_int"].isna(), [animal_col, label_col]].head(10)
        print("[WARN] Some labels could not be converted to integers and will be dropped:")
        print(bad.to_string(index=False))
        out = out[out["label_int"].notna()].copy()

    out["label_int"] = out["label_int"].astype(int)

    # If the table contains all ranked syllables, keep only the selected top rows.
    if "is_top_phrase_variance" in out.columns:
        out = out[_as_bool_series(out["is_top_phrase_variance"])].copy()

    if animal_ids is not None:
        wanted = {str(a).strip() for a in animal_ids}
        out = out[out["animal_id"].isin(wanted)].copy()

    if phrase_variance_col is not None and phrase_variance_col in out.columns:
        out["_sort_phrase_variance"] = pd.to_numeric(out[phrase_variance_col], errors="coerce")
    else:
        out["_sort_phrase_variance"] = np.nan

    if rank_col is not None and rank_col in out.columns:
        out["_sort_rank"] = pd.to_numeric(out[rank_col], errors="coerce")
    else:
        out["_sort_rank"] = np.nan

    # Keep one row per animal/label, preferring higher phrase variance.
    out = (
        out.sort_values(
            ["animal_id", "_sort_phrase_variance", "_sort_rank"],
            ascending=[True, False, True],
            na_position="last",
        )
        .drop_duplicates(subset=["animal_id", "label_int"], keep="first")
        .copy()
    )

    # Within each bird, sort by variance descending/rank ascending.
    out = out.sort_values(
        ["animal_id", "_sort_phrase_variance", "_sort_rank", "label_int"],
        ascending=[True, False, True, True],
        na_position="last",
    ).copy()

    if max_labels_per_bird and max_labels_per_bird > 0:
        out = out.groupby("animal_id", group_keys=False).head(max_labels_per_bird).copy()

    return out


def compute_label_bin_counts(
    spectrogram_module,
    npz_path: Path,
    metadata_excel_path: Path,
    animal_id: str,
    *,
    selected_labels: Sequence[int],
    treatment_day_assignment: str,
    date_array_key: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[int, int]], str]:
    arr = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(arr["hdbscan_labels"]).astype(int)

    treatment_date = spectrogram_module.get_treatment_date_from_metadata(
        metadata_excel_path=metadata_excel_path,
        animal_id=animal_id,
    )
    timebin_dates = spectrogram_module.get_timebin_recording_dates(
        arr,
        expected_length=labels.shape[0],
        date_array_key=date_array_key,
    )
    pre_mask, post_mask = spectrogram_module.make_pre_post_masks(
        timebin_dates,
        treatment_date,
        treatment_day_assignment=treatment_day_assignment,
    )

    counts: Dict[int, Tuple[int, int]] = {}
    for lbl in selected_labels:
        pre_bins = int(np.sum((labels == int(lbl)) & pre_mask))
        post_bins = int(np.sum((labels == int(lbl)) & post_mask))
        counts[int(lbl)] = (pre_bins, post_bins)

    return counts, str(treatment_date)


def choose_spectrogram_length(
    *,
    requested_length: int,
    pre_bins: int,
    post_bins: int,
    adaptive_length: bool,
    minimum_length: int,
    round_to: int,
) -> Tuple[Optional[int], str]:
    available = min(int(pre_bins), int(post_bins))
    if available <= 0:
        return None, "no paired pre/post bins"

    if adaptive_length:
        chosen = min(int(requested_length), available)
        if round_to and round_to > 1 and chosen >= round_to:
            chosen = max(round_to, (chosen // round_to) * round_to)
    else:
        chosen = int(requested_length)

    if available < chosen:
        return None, f"not enough paired bins for requested length {chosen}"
    if chosen < int(minimum_length):
        return None, f"adaptive length {chosen} below minimum length {minimum_length}"

    return int(chosen), "ok"


def run_batch(
    *,
    top30_csv: Path,
    spectrogram_script: Optional[str],
    npz_root: Path,
    metadata_excel_path: Path,
    out_dir: Path,
    npz_template: Optional[str],
    animal_ids: Optional[Sequence[str]],
    max_labels_per_bird: int,
    spectrogram_length: int,
    adaptive_length: bool,
    minimum_length: int,
    round_adaptive_length_to: int,
    num_sample_spectrograms: int,
    sample_mode: str,
    random_seed: int,
    treatment_day_assignment: str,
    seconds_per_bin: float,
    x_tick_step_s: float,
    figure_width: float,
    row_height: float,
    underline_row_height: float,
    subplot_hspace: float,
    cmap: str,
    contrast_percentiles: Optional[Tuple[float, float]],
    show_bout_labels: bool,
    min_bout_label_bins: int,
    no_bout_underlines: bool,
    make_umap: bool,
    save_label_key: bool,
    dry_run: bool,
) -> pd.DataFrame:
    out_dir = Path(out_dir).expanduser()
    _safe_mkdir(out_dir)

    wrapper_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    spectrogram_script_path = _find_spectrogram_script(
        spectrogram_script,
        search_dirs=[wrapper_dir, cwd],
    )
    print(f"[INFO] Using spectrogram script: {spectrogram_script_path}")

    spectrogram_module = _load_module_from_path(spectrogram_script_path)

    top_df = read_top_variance_table(
        top30_csv=top30_csv,
        animal_ids=animal_ids,
        max_labels_per_bird=max_labels_per_bird,
    )

    if top_df.empty:
        raise ValueError("No selected high-variance syllables were found in the top30 CSV after filtering.")

    print(f"[INFO] Selected high-variance rows: {len(top_df)}")
    print(f"[INFO] Birds represented: {top_df['animal_id'].nunique()}")

    manifest_rows: List[Dict[str, Any]] = []

    for animal_id, bird_df in top_df.groupby("animal_id", sort=True):
        animal_id = str(animal_id)
        selected_labels = [int(x) for x in bird_df["label_int"].tolist()]

        npz_path = _find_npz_for_animal(
            animal_id,
            npz_root=npz_root,
            npz_template=npz_template,
        )

        if npz_path is None:
            print(f"[WARN] {animal_id}: could not find .npz file under {npz_root}; skipping.")
            for _, row in bird_df.iterrows():
                manifest_rows.append({
                    "animal_id": animal_id,
                    "label": int(row["label_int"]),
                    "status": "skipped",
                    "reason": "npz not found",
                    "npz_path": None,
                    "output_dir": None,
                })
            continue

        print()
        print("=" * 90)
        print(f"[INFO] {animal_id}: {len(selected_labels)} selected label(s)")
        print(f"[INFO] NPZ: {npz_path}")
        print("=" * 90)

        try:
            label_counts, treatment_date = compute_label_bin_counts(
                spectrogram_module,
                npz_path=npz_path,
                metadata_excel_path=metadata_excel_path,
                animal_id=animal_id,
                selected_labels=selected_labels,
                treatment_day_assignment=treatment_day_assignment,
            )
        except Exception as e:
            print(f"[ERROR] Could not compute label bin counts for {animal_id}: {e}")
            traceback.print_exc()
            for _, row in bird_df.iterrows():
                manifest_rows.append({
                    "animal_id": animal_id,
                    "label": int(row["label_int"]),
                    "status": "error",
                    "reason": f"bin count error: {e}",
                    "npz_path": str(npz_path),
                    "output_dir": None,
                })
            continue

        # Save the selected labels for this animal for easy inspection.
        animal_out_dir = out_dir / _clean_filename(animal_id)
        _safe_mkdir(animal_out_dir)
        selected_label_table = bird_df.copy()
        selected_label_table["npz_path"] = str(npz_path)
        selected_label_table["pre_bins"] = selected_label_table["label_int"].map(lambda x: label_counts.get(int(x), (0, 0))[0])
        selected_label_table["post_bins"] = selected_label_table["label_int"].map(lambda x: label_counts.get(int(x), (0, 0))[1])
        selected_label_table.to_csv(animal_out_dir / f"{animal_id}_selected_high_variance_labels.csv", index=False)

        for _, row in bird_df.iterrows():
            label = int(row["label_int"])
            pre_bins, post_bins = label_counts.get(label, (0, 0))

            chosen_length, reason = choose_spectrogram_length(
                requested_length=spectrogram_length,
                pre_bins=pre_bins,
                post_bins=post_bins,
                adaptive_length=adaptive_length,
                minimum_length=minimum_length,
                round_to=round_adaptive_length_to,
            )

            label_out_dir = animal_out_dir / f"label_{label}_pre{pre_bins}_post{post_bins}"
            status = "planned" if dry_run else "started"

            manifest_base = {
                "animal_id": animal_id,
                "label": label,
                "treatment_date": treatment_date,
                "pre_bins": pre_bins,
                "post_bins": post_bins,
                "requested_spectrogram_length": int(spectrogram_length),
                "chosen_spectrogram_length": chosen_length,
                "adaptive_length": bool(adaptive_length),
                "npz_path": str(npz_path),
                "output_dir": str(label_out_dir),
                "phrase_variance_ms2": row.get("phrase_variance_ms2", np.nan),
                "phrase_variance_rank_within_bird": row.get("phrase_variance_rank_within_bird", np.nan),
            }

            if chosen_length is None:
                print(f"[SKIP] {animal_id} label {label}: {reason}; pre={pre_bins}, post={post_bins}")
                manifest_rows.append({
                    **manifest_base,
                    "status": "skipped",
                    "reason": reason,
                })
                continue

            print(
                f"[RUN] {animal_id} label {label}: length={chosen_length}, "
                f"pre_bins={pre_bins}, post_bins={post_bins}"
            )

            if dry_run:
                manifest_rows.append({
                    **manifest_base,
                    "status": status,
                    "reason": "dry run",
                })
                continue

            _safe_mkdir(label_out_dir)

            try:
                summary = spectrogram_module.plot_pre_post_spectrogram_samples_for_labels(
                    npz_path=npz_path,
                    metadata_excel_path=metadata_excel_path,
                    output_dir=label_out_dir,
                    animal_id=animal_id,
                    selected_labels=[label],
                    skip_noise_label=True,
                    spectrogram_length=int(chosen_length),
                    num_sample_spectrograms=int(num_sample_spectrograms),
                    treatment_day_assignment=treatment_day_assignment,
                    sample_mode=sample_mode,
                    random_seed=int(random_seed),
                    cmap=cmap,
                    show_colorbar=False,
                    show_plots=False,
                    save_figures=True,
                    shared_intensity_scale=True,
                    contrast_percentiles=contrast_percentiles,
                    make_umap_pre_post_plot=bool(make_umap),
                    save_umap_label_key=bool(save_label_key),
                    figure_width=float(figure_width),
                    row_height=float(row_height),
                    underline_row_height=float(underline_row_height),
                    subplot_hspace=float(subplot_hspace),
                    seconds_per_bin=float(seconds_per_bin),
                    x_tick_step_s=float(x_tick_step_s),
                    show_bout_underlines=not bool(no_bout_underlines),
                    show_bout_labels=bool(show_bout_labels),
                    min_bout_label_bins=int(min_bout_label_bins),
                )
                samples_made = int(summary["samples_made"].max()) if isinstance(summary, pd.DataFrame) and not summary.empty and "samples_made" in summary.columns else np.nan
                manifest_rows.append({
                    **manifest_base,
                    "status": "done",
                    "reason": "ok",
                    "samples_made": samples_made,
                })
            except Exception as e:
                print(f"[ERROR] {animal_id} label {label}: {e}")
                traceback.print_exc()
                manifest_rows.append({
                    **manifest_base,
                    "status": "error",
                    "reason": repr(e),
                })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "top_variance_spectrogram_batch_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print()
    print("=" * 90)
    print("[DONE] Batch spectrogram generation complete")
    print(f"[SAVE] {manifest_path}")
    print("=" * 90)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate pre/post sample spectrograms for high phrase-duration-variance "
            "syllables selected by the top30 BC-only analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--top30-csv", required=True, help="Path to merged_top30_bc_rows.csv or selected_top_phrase_variance_syllables.csv.")
    p.add_argument("--npz-root", required=True, help="Root folder containing per-animal .npz outputs.")
    p.add_argument("--metadata-excel-path", required=True, help="Path to Area_X_lesion_metadata_with_hit_types.xlsx.")
    p.add_argument("--out-dir", required=True, help="Output directory for spectrogram examples.")
    p.add_argument("--spectrogram-script", default=None, help="Path to your pre/post spectrogram script. If omitted, this script searches the current folder.")
    p.add_argument("--npz-template", default=None, help="Optional template, e.g. '{npz_root}/{animal_id}/{animal_id}.npz'.")

    p.add_argument("--animal-ids", nargs="*", default=None, help="Optional animal IDs to include, e.g. --animal-ids USA5288 USA5325")
    p.add_argument("--max-labels-per-bird", type=int, default=0, help="Optional cap on high-variance labels per bird. 0 means all selected labels.")

    p.add_argument("--spectrogram-length", type=int, default=5000, help="Requested stitched bins per pre/post row.")
    p.add_argument("--no-adaptive-length", action="store_true", help="Disable adaptive shortening when a label has fewer bins than requested.")
    p.add_argument("--minimum-length", type=int, default=800, help="Minimum allowed adaptive spectrogram length.")
    p.add_argument("--round-adaptive-length-to", type=int, default=100, help="Round adaptive lengths down to this many bins. Use 1 to disable.")
    p.add_argument("--num-sample-spectrograms", type=int, default=1, help="Number of paired samples per label.")
    p.add_argument("--sample-mode", choices=["first", "random"], default="random", help="Sampling mode passed to the spectrogram script.")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--treatment-day-assignment", choices=["exclude", "pre", "post"], default="exclude")

    p.add_argument("--seconds-per-bin", type=float, default=0.0027)
    p.add_argument("--x-tick-step-s", type=float, default=2.0)
    p.add_argument("--figure-width", type=float, default=18.0)
    p.add_argument("--row-height", type=float, default=2.6)
    p.add_argument("--underline-row-height", type=float, default=0.18)
    p.add_argument("--subplot-hspace", type=float, default=0.005)
    p.add_argument("--cmap", default="gray_r")
    p.add_argument("--contrast-percentiles", nargs=2, type=float, default=[1, 99.5], metavar=("LOW", "HIGH"))
    p.add_argument("--no-contrast-percentiles", action="store_true")
    p.add_argument("--show-bout-labels", action="store_true")
    p.add_argument("--min-bout-label-bins", type=int, default=80)
    p.add_argument("--no-bout-underlines", action="store_true")
    p.add_argument("--make-umap", action="store_true", help="Also make UMAP pre/post plots. Usually leave off for batch QC.")
    p.add_argument("--save-label-key", action="store_true", help="Also save label color keys. Usually leave off for batch QC.")
    p.add_argument("--dry-run", action="store_true", help="Print/record what would run without making spectrograms.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    animal_ids = _parse_list_arg(args.animal_ids)
    contrast = None if args.no_contrast_percentiles else tuple(args.contrast_percentiles)

    run_batch(
        top30_csv=Path(args.top30_csv).expanduser(),
        spectrogram_script=args.spectrogram_script,
        npz_root=Path(args.npz_root).expanduser(),
        metadata_excel_path=Path(args.metadata_excel_path).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        npz_template=args.npz_template,
        animal_ids=animal_ids,
        max_labels_per_bird=int(args.max_labels_per_bird),
        spectrogram_length=int(args.spectrogram_length),
        adaptive_length=not bool(args.no_adaptive_length),
        minimum_length=int(args.minimum_length),
        round_adaptive_length_to=int(args.round_adaptive_length_to),
        num_sample_spectrograms=int(args.num_sample_spectrograms),
        sample_mode=args.sample_mode,
        random_seed=int(args.random_seed),
        treatment_day_assignment=args.treatment_day_assignment,
        seconds_per_bin=float(args.seconds_per_bin),
        x_tick_step_s=float(args.x_tick_step_s),
        figure_width=float(args.figure_width),
        row_height=float(args.row_height),
        underline_row_height=float(args.underline_row_height),
        subplot_hspace=float(args.subplot_hspace),
        cmap=args.cmap,
        contrast_percentiles=contrast,
        show_bout_labels=bool(args.show_bout_labels),
        min_bout_label_bins=int(args.min_bout_label_bins),
        no_bout_underlines=bool(args.no_bout_underlines),
        make_umap=bool(args.make_umap),
        save_label_key=bool(args.save_label_key),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
