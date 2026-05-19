#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_top_variance_syllable_spectrograms_multirow_epochs.py

Generate pre- and post-lesion sample spectrograms for high-variance syllables,
saving separate multi-row PNG figures for each epoch.

This version is designed for qualitative inspection of syllable detail:
- each row/panel is a stitched spectrogram segment, default 2,000 time bins long
- all rows within a figure use the same time scale
- if the last row has fewer than 2,000 bins, the spectrogram stops where data ends
  and the remaining x-axis space stays blank
- saves one PRE multi-row figure and one POST multi-row figure per selected label
- optionally truncates pre/post to equal bin counts for balanced comparison
- does not save summary CSV files
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import argparse
import importlib.util
import math
import re
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


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

    spec = importlib.util.spec_from_file_location("spectrogram_module_for_top_variance_qc_chunked", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import spectrogram script from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "get_treatment_date_from_metadata",
        "get_timebin_recording_dates",
        "make_pre_post_masks",
        "_orient_spectrogram_to_FxT",
        "_hide_spectrogram_ticks",
        "_set_stitched_time_axis",
        "_find_stitched_bout_spans",
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

    recursive_patterns = [f"**/{animal_id}.npz", f"**/{animal_id}_*.npz", f"**/*{animal_id}*.npz"]
    found: List[Path] = []
    for pat in recursive_patterns:
        found.extend(npz_root.glob(pat))
        if found:
            break
    found = [p for p in found if p.is_file()]
    if not found:
        return None
    exact = [p for p in found if p.stem == animal_id]
    if exact:
        return sorted(exact, key=lambda p: len(str(p)))[0].resolve()
    return sorted(found, key=lambda p: len(str(p)))[0].resolve()


def read_top_variance_table(
    input_csv: Path,
    *,
    animal_col: Optional[str] = None,
    label_col: Optional[str] = None,
    group_col: Optional[str] = None,
    nphrases_col: Optional[str] = None,
    phrase_variance_col: Optional[str] = None,
    rank_col: Optional[str] = None,
    animal_ids: Optional[Sequence[str]] = None,
    max_labels_per_bird: int = 0,
    top_fraction: float = 0.30,
    post_group_name: str = "Post",
    min_n_phrases: int = 100,
) -> pd.DataFrame:
    """
    Read either:
      1) a top-30 output table, e.g. merged_top30_bc_rows.csv, OR
      2) the original phrase-duration stats CSV, e.g. usage_balanced_phrase_duration_stats.csv.

    If the table has is_top_phrase_variance, this function uses that column.
    Otherwise, if it has a Group column, it filters to post_group_name and selects
    the top_fraction highest post phrase-duration variance syllables within each bird.
    """
    df = pd.read_csv(input_csv)

    if animal_col is None:
        animal_col = _pick_column(df, ["animal_id", "Animal ID", "animal id", "bird", "Bird ID"])
    if label_col is None:
        label_col = _pick_column(df, ["syllable_key", "label", "cluster", "Syllable", "syllable"])
    if group_col is None:
        group_col = _pick_column(df, ["Group", "group", "period", "phrase_group"])
    if nphrases_col is None:
        nphrases_col = _pick_column(df, ["N_phrases", "n_phrases", "phrase_n_phrases", "nphrases", "count"])
    if phrase_variance_col is None:
        phrase_variance_col = _pick_column(df, ["phrase_variance_ms2", "Variance_ms2", "variance_ms2", "Post_Variance_ms2"])
    if rank_col is None:
        rank_col = _pick_column(df, ["phrase_variance_rank_within_bird", "rank", "variance_rank"])

    if animal_col is None:
        raise KeyError(f"Could not infer animal ID column in {input_csv}. Columns: {list(df.columns)}")
    if label_col is None:
        raise KeyError(f"Could not infer label/cluster column in {input_csv}. Columns: {list(df.columns)}")
    if phrase_variance_col is None:
        raise KeyError(
            f"Could not infer phrase-variance column in {input_csv}. "
            f"Expected something like Variance_ms2 or phrase_variance_ms2. Columns: {list(df.columns)}"
        )

    out = df.copy()
    out["animal_id"] = out[animal_col].astype(str).str.strip()
    out["label_int"] = out[label_col].map(_label_to_int)
    out = out[out["label_int"].notna()].copy()
    out["label_int"] = out["label_int"].astype(int)
    out["_sort_phrase_variance"] = pd.to_numeric(out[phrase_variance_col], errors="coerce")

    if nphrases_col is not None and nphrases_col in out.columns:
        out["_n_phrases_for_filter"] = pd.to_numeric(out[nphrases_col], errors="coerce")
    else:
        out["_n_phrases_for_filter"] = np.nan

    # CASE 1: this is a previous top-variance output table.
    if "is_top_phrase_variance" in out.columns:
        out = out[_as_bool_series(out["is_top_phrase_variance"])].copy()

        if rank_col is not None and rank_col in out.columns:
            out["_sort_rank"] = pd.to_numeric(out[rank_col], errors="coerce")
        else:
            out["_sort_rank"] = np.nan

        out = (
            out.sort_values(
                ["animal_id", "_sort_phrase_variance", "_sort_rank"],
                ascending=[True, False, True],
                na_position="last",
            )
            .drop_duplicates(subset=["animal_id", "label_int"], keep="first")
            .copy()
        )

    # CASE 2: this is the original phrase-duration stats CSV.
    else:
        if group_col is not None and group_col in out.columns:
            out = out[
                out[group_col].astype(str).str.strip().str.lower()
                == str(post_group_name).strip().lower()
            ].copy()
            if out.empty:
                raise ValueError(
                    f"No rows matched post group '{post_group_name}' in column '{group_col}'."
                )
        else:
            print(
                "[WARN] No Group column found and no is_top_phrase_variance column found. "
                "Using all rows and selecting top fraction within each bird."
            )

        out = out[np.isfinite(out["_sort_phrase_variance"])].copy()

        if nphrases_col is not None and nphrases_col in out.columns:
            out = out[np.isfinite(out["_n_phrases_for_filter"])].copy()
            out = out[out["_n_phrases_for_filter"] >= int(min_n_phrases)].copy()

        if out.empty:
            raise ValueError(
                "No phrase-duration rows survived filtering. Try lowering --min-n-phrases "
                "or check --post-group-name."
            )

        # Collapse duplicate animal/label rows by keeping max post variance.
        out = (
            out.sort_values(
                ["animal_id", "label_int", "_sort_phrase_variance", "_n_phrases_for_filter"],
                ascending=[True, True, False, False],
                na_position="last",
            )
            .drop_duplicates(subset=["animal_id", "label_int"], keep="first")
            .copy()
        )

        if not (0 < float(top_fraction) <= 1):
            raise ValueError("--top-fraction must be >0 and <=1")

        ranked_parts: List[pd.DataFrame] = []
        for animal_id_value, sub in out.groupby("animal_id", sort=True):
            sub = sub.sort_values("_sort_phrase_variance", ascending=False).copy()
            n = len(sub)
            n_top = max(1, int(math.ceil(n * float(top_fraction))))
            sub["phrase_variance_rank_within_bird"] = np.arange(1, n + 1)
            sub["n_ranked_syllables_within_bird"] = n
            sub["n_top_syllables_within_bird"] = n_top
            sub["is_top_phrase_variance"] = sub["phrase_variance_rank_within_bird"] <= n_top
            sub["top_fraction"] = float(top_fraction)
            ranked_parts.append(sub.loc[sub["is_top_phrase_variance"]].copy())

        out = pd.concat(ranked_parts, ignore_index=True)

    if animal_ids is not None:
        wanted = {str(a).strip() for a in animal_ids}
        out = out[out["animal_id"].isin(wanted)].copy()

    # Stable sorting for output and filenames.
    if "phrase_variance_rank_within_bird" in out.columns:
        out["_sort_rank"] = pd.to_numeric(out["phrase_variance_rank_within_bird"], errors="coerce")
    elif rank_col is not None and rank_col in out.columns:
        out["_sort_rank"] = pd.to_numeric(out[rank_col], errors="coerce")
    else:
        out["_sort_rank"] = np.nan

    out = out.sort_values(
        ["animal_id", "_sort_rank", "_sort_phrase_variance", "label_int"],
        ascending=[True, True, False, True],
        na_position="last",
    ).copy()

    if max_labels_per_bird and max_labels_per_bird > 0:
        out = out.groupby("animal_id", group_keys=False).head(max_labels_per_bird).copy()

    return out


def _get_annotation_font(font_size: int = 28):
    if not HAVE_PIL:
        return None
    try:
        import matplotlib.font_manager as fm
        font_path = fm.findfont("DejaVu Sans")
        return ImageFont.truetype(font_path, font_size)
    except Exception:
        try:
            return ImageFont.truetype("Arial.ttf", font_size)
        except Exception:
            return ImageFont.load_default()


def _annotate_png_with_chunk_counts(
    png_path: Path,
    *,
    pre_chunks: int,
    post_chunks: int,
    font_size: int = 30,
) -> None:
    if not HAVE_PIL:
        print(f"[WARN] Pillow/PIL is not available; cannot annotate {png_path.name}")
        return

    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    font = _get_annotation_font(font_size=font_size)
    line = f"Stitched chunks in sampled bins: Pre = {int(pre_chunks):,}; Post = {int(post_chunks):,}"

    line_spacing = int(font_size * 0.35)
    top_pad = int(font_size * 0.55)
    bottom_pad = int(font_size * 0.55)
    strip_h = top_pad + bottom_pad + font_size + line_spacing

    new_img = Image.new("RGB", (w, h + strip_h), "white")
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    try:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w = len(line) * font_size * 0.5
        text_h = font_size

    x = max(0, int((w - text_w) / 2))
    y = h + top_pad
    draw.text((x, y), line, fill="black", font=font)
    new_img.save(png_path)


def _chunk_sizes(total_bins: int, max_bins_per_panel: int, max_panels: int) -> List[int]:
    total_bins = int(total_bins)
    max_bins_per_panel = max(1, int(max_bins_per_panel))
    if total_bins <= 0:
        return []

    n_full = total_bins // max_bins_per_panel
    remainder = total_bins % max_bins_per_panel
    sizes = [max_bins_per_panel] * n_full
    if remainder > 0:
        sizes.append(remainder)

    if max_panels and max_panels > 0:
        sizes = sizes[:max_panels]
    return sizes


def _sequential_chunks(idx: np.ndarray, sizes: Sequence[int]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    start = 0
    for size in sizes:
        end = start + int(size)
        out.append(np.asarray(idx[start:end], dtype=int))
        start = end
    return out


def _random_disjoint_chunks(idx: np.ndarray, sizes: Sequence[int], rng: np.random.Generator) -> List[np.ndarray]:
    total_needed = int(sum(int(s) for s in sizes))
    if total_needed <= 0:
        return []
    chosen = rng.choice(np.asarray(idx, dtype=int), size=total_needed, replace=False)
    out: List[np.ndarray] = []
    start = 0
    for size in sizes:
        end = start + int(size)
        out.append(np.sort(chosen[start:end]).astype(int))
        start = end
    return out



def _compute_shared_intensity_limits(
    *,
    S: np.ndarray,
    chunk_lists: Sequence[Sequence[np.ndarray]],
    contrast_percentiles: Optional[Tuple[float, float]],
    random_seed: int = 0,
    max_values_for_percentile: int = 2_000_000,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute a shared intensity scale across multiple epoch figures.

    For very large selections, this function samples values so percentile
    calculation does not create an unnecessarily huge temporary array.
    """
    chunks: List[np.ndarray] = []
    for chunk_list in chunk_lists:
        for ch in chunk_list:
            ch = np.asarray(ch, dtype=int)
            if ch.size > 0:
                chunks.append(ch)

    if not chunks:
        return None, None

    total_values = int(sum(S.shape[0] * ch.size for ch in chunks))
    rng = np.random.default_rng(random_seed)
    values_parts: List[np.ndarray] = []

    if total_values <= int(max_values_for_percentile):
        for ch in chunks:
            values_parts.append(np.asarray(S[:, ch], dtype=float).ravel())
    else:
        # Sample approximately evenly across chunks. This keeps contrast scaling
        # stable without copying every selected spectrogram value into memory.
        per_chunk = max(1, int(max_values_for_percentile // len(chunks)))
        for ch in chunks:
            panel = np.asarray(S[:, ch], dtype=float)
            flat = panel.ravel()
            if flat.size <= per_chunk:
                values_parts.append(flat)
            else:
                sample_idx = rng.choice(flat.size, size=per_chunk, replace=False)
                values_parts.append(flat[sample_idx])

    if not values_parts:
        return None, None

    values = np.concatenate(values_parts)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None

    if contrast_percentiles is None:
        return float(np.nanmin(values)), float(np.nanmax(values))

    lo, hi = contrast_percentiles
    vmin, vmax = np.nanpercentile(values, [lo, hi])
    return float(vmin), float(vmax)


def _plot_epoch_multirow(
    spectrogram_module,
    *,
    S: np.ndarray,
    sel_chunks: Sequence[np.ndarray],
    animal_id: str,
    label: int,
    epoch_name: str,
    treatment_date,
    seconds_per_bin: float,
    x_tick_step_s: float,
    figure_width: float,
    row_height: float,
    subplot_hspace: float,
    cmap: str,
    out_png: Path,
    max_bins_per_panel: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate_chunks: bool = False,
) -> None:
    """
    Make one multi-row figure for a single epoch, either Pre or Post.

    Each row is one stitched spectrogram chunk. Every row shares the same
    x-axis scale, based on max_bins_per_panel. If the final row has fewer bins,
    it is drawn only over its true duration and the rest of the panel remains
    blank.
    """
    sel_chunks = [np.asarray(ch, dtype=int) for ch in sel_chunks if len(ch) > 0]
    if not sel_chunks:
        raise ValueError(f"No chunks available to plot for {epoch_name}.")

    n_panels = len(sel_chunks)
    max_bins_per_panel = max(1, int(max_bins_per_panel))
    max_time_s = float(max_bins_per_panel) * float(seconds_per_bin)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(figure_width, row_height * n_panels + 0.7),
        sharex=True,
        gridspec_kw={"hspace": max(0.04, subplot_hspace)},
    )

    if n_panels == 1:
        axes = [axes]

    for i, (ax, sel) in enumerate(zip(axes, sel_chunks), start=1):
        S_panel = np.asarray(S[:, sel], dtype=float)
        actual_bins = int(S_panel.shape[1])
        actual_time_s = float(actual_bins) * float(seconds_per_bin)

        ax.imshow(
            S_panel,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, actual_time_s, 0, S_panel.shape[0]],
        )

        # This keeps every row on the same time scale. Short final panels end
        # early and leave white space to the right.
        ax.set_xlim(0, max_time_s)
        ax.set_facecolor("white")

        spectrogram_module._hide_spectrogram_ticks(ax)
        ax.set_ylabel(
            f"{epoch_name}\nrow {i}\n{actual_bins:,} bins",
            rotation=0,
            labelpad=44,
            va="center",
            fontsize=11,
        )

    # Show x-axis ticks only on the bottom row.
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    axes[-1].tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    xticks = np.arange(0, max_time_s + 1e-9, float(x_tick_step_s))
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{x:g}" for x in xticks])
    axes[-1].set_xlabel("Stitched label time (s)")

    fig.suptitle(
        f"{animal_id} — HDBSCAN label {label} — {epoch_name}-lesion spectrograms\n"
        f"Treatment date: {treatment_date} | {n_panels} panel(s), up to {max_bins_per_panel:,} bins each",
        y=0.995,
        fontsize=15,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.08)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    if annotate_chunks:
        try:
            all_sel = np.concatenate(sel_chunks)
            chunk_count = len(spectrogram_module._find_stitched_bout_spans(all_sel))
            # Reuse the existing annotation helper by putting this epoch's count
            # in the relevant field and zero in the other field.
            if str(epoch_name).lower().startswith("pre"):
                _annotate_png_with_chunk_counts(out_png, pre_chunks=chunk_count, post_chunks=0)
            else:
                _annotate_png_with_chunk_counts(out_png, pre_chunks=0, post_chunks=chunk_count)
        except Exception:
            # Annotation should never prevent saving the main figure.
            pass

def run_batch(
    *,
    input_csv: Path,
    spectrogram_script: Optional[str],
    npz_root: Path,
    metadata_excel_path: Path,
    out_dir: Path,
    npz_template: Optional[str],
    animal_ids: Optional[Sequence[str]],
    max_labels_per_bird: int,
    top_fraction: float,
    post_group_name: str,
    min_n_phrases: int,
    spectrogram_length: int,
    minimum_length: int,
    num_sample_spectrograms: int,
    sample_mode: str,
    random_seed: int,
    treatment_day_assignment: str,
    seconds_per_bin: float,
    x_tick_step_s: float,
    figure_width: float,
    row_height: float,
    subplot_hspace: float,
    cmap: str,
    contrast_percentiles: Optional[Tuple[float, float]],
    dry_run: bool,
    annotate_chunks: bool,
    match_pre_post_bin_count: bool,
) -> None:
    out_dir = Path(out_dir).expanduser()
    _safe_mkdir(out_dir)

    wrapper_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    spectrogram_script_path = _find_spectrogram_script(spectrogram_script, search_dirs=[wrapper_dir, cwd])
    print(f"[INFO] Using spectrogram script: {spectrogram_script_path}")
    spectrogram_module = _load_module_from_path(spectrogram_script_path)

    top_df = read_top_variance_table(
        input_csv=input_csv,
        animal_ids=animal_ids,
        max_labels_per_bird=max_labels_per_bird,
        top_fraction=float(top_fraction),
        post_group_name=str(post_group_name),
        min_n_phrases=int(min_n_phrases),
    )
    if top_df.empty:
        raise ValueError("No selected high-variance syllables were found in the input CSV after filtering.")

    print(f"[INFO] Selected high-variance rows: {len(top_df)}")
    print(f"[INFO] Birds represented: {top_df['animal_id'].nunique()}")

    base_rng = np.random.default_rng(random_seed)

    for animal_id, bird_df in top_df.groupby("animal_id", sort=True):
        animal_id = str(animal_id)
        selected_labels = [int(x) for x in bird_df["label_int"].tolist()]
        npz_path = _find_npz_for_animal(animal_id, npz_root=npz_root, npz_template=npz_template)

        if npz_path is None:
            print(f"[WARN] {animal_id}: could not find .npz file under {npz_root}; skipping.")
            continue

        print("\n" + "=" * 90)
        print(f"[INFO] {animal_id}: {len(selected_labels)} selected label(s)")
        print(f"[INFO] NPZ: {npz_path}")
        print("=" * 90)

        try:
            arr = np.load(npz_path, allow_pickle=True)
            labels = np.asarray(arr["hdbscan_labels"]).astype(int)
            S = spectrogram_module._orient_spectrogram_to_FxT(np.asarray(arr["s"]), labels)
            treatment_date = spectrogram_module.get_treatment_date_from_metadata(
                metadata_excel_path=metadata_excel_path,
                animal_id=animal_id,
            )
            timebin_dates = spectrogram_module.get_timebin_recording_dates(arr, expected_length=labels.shape[0], date_array_key=None)
            pre_mask, post_mask = spectrogram_module.make_pre_post_masks(
                timebin_dates,
                treatment_date,
                treatment_day_assignment=treatment_day_assignment,
            )
        except Exception as e:
            print(f"[ERROR] Could not load pre/post masks for {animal_id}: {e}")
            traceback.print_exc()
            continue

        animal_out_dir = out_dir / _clean_filename(animal_id)
        _safe_mkdir(animal_out_dir)

        for _, row in bird_df.iterrows():
            label = int(row["label_int"])
            pre_idx = np.flatnonzero((labels == label) & pre_mask)
            post_idx = np.flatnonzero((labels == label) & post_mask)

            if match_pre_post_bin_count:
                # Balanced mode: truncate pre and post to the same number of bins.
                # This preserves the comparison logic from the earlier paired-panel script.
                pre_total = post_total = int(min(pre_idx.size, post_idx.size))
                mode_note = f"matched pre/post bins={pre_total}"
            else:
                # All-available mode: pre and post figures can have different numbers
                # of rows if one epoch contains more examples than the other.
                pre_total = int(pre_idx.size)
                post_total = int(post_idx.size)
                mode_note = f"pre bins={pre_total}, post bins={post_total}"

            pre_panel_sizes = _chunk_sizes(pre_total, spectrogram_length, num_sample_spectrograms)
            post_panel_sizes = _chunk_sizes(post_total, spectrogram_length, num_sample_spectrograms)
            pre_panel_sizes = [int(s) for s in pre_panel_sizes if int(s) >= max(1, int(minimum_length))]
            post_panel_sizes = [int(s) for s in post_panel_sizes if int(s) >= max(1, int(minimum_length))]

            if not pre_panel_sizes and not post_panel_sizes:
                print(
                    f"[SKIP] {animal_id} label {label}: no pre or post panels passed "
                    f"minimum_length={minimum_length}. Raw counts: pre={pre_idx.size}, post={post_idx.size}."
                )
                continue

            if sample_mode == "first":
                pre_chunks = _sequential_chunks(pre_idx[:sum(pre_panel_sizes)], pre_panel_sizes)
                post_chunks = _sequential_chunks(post_idx[:sum(post_panel_sizes)], post_panel_sizes)
            elif sample_mode == "random":
                # Use a label-specific RNG so a run is reproducible but independent across labels.
                label_seed = int(base_rng.integers(0, 2**31 - 1))
                rng = np.random.default_rng(label_seed)
                pre_chunks = _random_disjoint_chunks(pre_idx, pre_panel_sizes, rng)
                post_chunks = _random_disjoint_chunks(post_idx, post_panel_sizes, rng)
            else:
                raise ValueError("sample_mode must be 'first' or 'random'.")

            print(
                f"[RUN] {animal_id} label {label}: {mode_note}, "
                f"max bins/panel={spectrogram_length}, "
                f"pre panels={len(pre_panel_sizes)} sizes={pre_panel_sizes}, "
                f"post panels={len(post_panel_sizes)} sizes={post_panel_sizes}"
            )
            if dry_run:
                continue

            rank_value = row.get("phrase_variance_rank_within_bird", np.nan)
            try:
                rank_prefix = f"rank{int(rank_value):02d}_"
            except Exception:
                rank_prefix = ""

            shared_vmin, shared_vmax = _compute_shared_intensity_limits(
                S=S,
                chunk_lists=[pre_chunks, post_chunks],
                contrast_percentiles=contrast_percentiles,
                random_seed=int(random_seed),
            )

            if pre_chunks:
                pre_total_selected = int(sum(len(ch) for ch in pre_chunks))
                pre_png = animal_out_dir / (
                    f"{rank_prefix}label{label}_PRE_multirow_"
                    f"{len(pre_chunks):02d}panels_N{pre_total_selected}_"
                    f"panelbins{int(spectrogram_length)}_no_bouts.png"
                )
                try:
                    _plot_epoch_multirow(
                        spectrogram_module,
                        S=S,
                        sel_chunks=pre_chunks,
                        animal_id=animal_id,
                        label=label,
                        epoch_name="Pre",
                        treatment_date=treatment_date,
                        seconds_per_bin=float(seconds_per_bin),
                        x_tick_step_s=float(x_tick_step_s),
                        figure_width=float(figure_width),
                        row_height=float(row_height),
                        subplot_hspace=float(subplot_hspace),
                        cmap=cmap,
                        out_png=pre_png,
                        max_bins_per_panel=int(spectrogram_length),
                        vmin=shared_vmin,
                        vmax=shared_vmax,
                        annotate_chunks=annotate_chunks,
                    )
                    print(f"[SAVE] {pre_png}")
                except Exception as e:
                    print(f"[ERROR] {animal_id} label {label} PRE figure: {e}")
                    traceback.print_exc()
            else:
                print(f"[SKIP] {animal_id} label {label}: no PRE panels to save.")

            if post_chunks:
                post_total_selected = int(sum(len(ch) for ch in post_chunks))
                post_png = animal_out_dir / (
                    f"{rank_prefix}label{label}_POST_multirow_"
                    f"{len(post_chunks):02d}panels_N{post_total_selected}_"
                    f"panelbins{int(spectrogram_length)}_no_bouts.png"
                )
                try:
                    _plot_epoch_multirow(
                        spectrogram_module,
                        S=S,
                        sel_chunks=post_chunks,
                        animal_id=animal_id,
                        label=label,
                        epoch_name="Post",
                        treatment_date=treatment_date,
                        seconds_per_bin=float(seconds_per_bin),
                        x_tick_step_s=float(x_tick_step_s),
                        figure_width=float(figure_width),
                        row_height=float(row_height),
                        subplot_hspace=float(subplot_hspace),
                        cmap=cmap,
                        out_png=post_png,
                        max_bins_per_panel=int(spectrogram_length),
                        vmin=shared_vmin,
                        vmax=shared_vmax,
                        annotate_chunks=annotate_chunks,
                    )
                    print(f"[SAVE] {post_png}")
                except Exception as e:
                    print(f"[ERROR] {animal_id} label {label} POST figure: {e}")
                    traceback.print_exc()
            else:
                print(f"[SKIP] {animal_id} label {label}: no POST panels to save.")

    print("\n" + "=" * 90)
    print("[DONE] Multi-row pre/post spectrogram generation complete")
    print(f"[SAVE] {out_dir}")
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate separate PRE and POST multi-row spectrogram figures for high "
            "phrase-duration-variance syllables, saving no-bout-line PNGs directly "
            "in one folder per bird."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--input-csv",
        default=None,
        help=(
            "Input CSV. Can be a top-variance output table or the original "
            "usage_balanced_phrase_duration_stats.csv phrase-duration table."
        ),
    )
    p.add_argument(
        "--top30-csv",
        default=None,
        help="Backward-compatible alias for --input-csv.",
    )
    p.add_argument(
        "--phrase-csv",
        default=None,
        help="Backward-compatible alias for --input-csv when using the phrase-duration stats CSV.",
    )
    p.add_argument("--npz-root", required=True, help="Root folder containing per-animal .npz outputs.")
    p.add_argument("--metadata-excel-path", required=True, help="Path to Area_X_lesion_metadata_with_hit_types.xlsx.")
    p.add_argument("--out-dir", required=True, help="Output directory for spectrogram examples.")
    p.add_argument("--spectrogram-script", default=None, help="Path to your pre/post spectrogram script. If omitted, this script searches the current folder.")
    p.add_argument("--npz-template", default=None, help="Optional template, e.g. '{npz_root}/{animal_id}/{animal_id}.npz'.")

    p.add_argument("--animal-ids", nargs="*", default=None, help="Optional animal IDs to include, e.g. --animal-ids USA5288 USA5325")
    p.add_argument("--max-labels-per-bird", type=int, default=0, help="Optional cap on high-variance labels per bird. 0 means all selected labels.")
    p.add_argument("--top-fraction", type=float, default=0.30, help="When using phrase-duration stats CSV, select this top fraction by post variance within each bird.")
    p.add_argument("--post-group-name", default="Post", help="When using phrase-duration stats CSV, group name used for post-lesion rows.")
    p.add_argument("--min-n-phrases", type=int, default=100, help="When using phrase-duration stats CSV, minimum N_phrases for ranking top-variance syllables.")

    p.add_argument("--spectrogram-length", type=int, default=2000, help="Maximum stitched bins per pre/post row.")
    p.add_argument("--minimum-length", type=int, default=1, help="Minimum allowed bins for a panel.")
    p.add_argument("--num-sample-spectrograms", type=int, default=0, help="Maximum number of panels per label. 0 means make all chunked panels.")
    p.add_argument("--sample-mode", choices=["first", "random"], default="first", help="How to choose bins within each label/epoch.")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--treatment-day-assignment", choices=["exclude", "pre", "post"], default="exclude")

    p.add_argument("--seconds-per-bin", type=float, default=0.0027)
    p.add_argument("--x-tick-step-s", type=float, default=1.0)
    p.add_argument("--figure-width", type=float, default=18.0)
    p.add_argument("--row-height", type=float, default=2.6)
    p.add_argument("--subplot-hspace", type=float, default=0.005)
    p.add_argument("--cmap", default="gray_r")
    p.add_argument("--contrast-percentiles", nargs=2, type=float, default=[1, 99.5], metavar=("LOW", "HIGH"))
    p.add_argument("--no-contrast-percentiles", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print what would run without making spectrograms.")
    p.add_argument("--no-chunk-annotation", action="store_true", help="Do not add stitched-chunk counts under the x-axis label.")
    p.add_argument(
        "--match-pre-post-bin-count",
        action="store_true",
        help=(
            "Truncate pre and post to the same number of bins before making multi-row figures. "
            "By default, the script uses all available bins separately for pre and post."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    animal_ids = _parse_list_arg(args.animal_ids)
    contrast = None if args.no_contrast_percentiles else tuple(args.contrast_percentiles)

    input_csv_arg = args.input_csv or args.top30_csv or args.phrase_csv
    if input_csv_arg is None:
        raise ValueError("Please provide --input-csv, --top30-csv, or --phrase-csv.")

    run_batch(
        input_csv=Path(input_csv_arg).expanduser(),
        spectrogram_script=args.spectrogram_script,
        npz_root=Path(args.npz_root).expanduser(),
        metadata_excel_path=Path(args.metadata_excel_path).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        npz_template=args.npz_template,
        animal_ids=animal_ids,
        max_labels_per_bird=int(args.max_labels_per_bird),
        top_fraction=float(args.top_fraction),
        post_group_name=str(args.post_group_name),
        min_n_phrases=int(args.min_n_phrases),
        spectrogram_length=int(args.spectrogram_length),
        minimum_length=int(args.minimum_length),
        num_sample_spectrograms=int(args.num_sample_spectrograms),
        sample_mode=args.sample_mode,
        random_seed=int(args.random_seed),
        treatment_day_assignment=args.treatment_day_assignment,
        seconds_per_bin=float(args.seconds_per_bin),
        x_tick_step_s=float(args.x_tick_step_s),
        figure_width=float(args.figure_width),
        row_height=float(args.row_height),
        subplot_hspace=float(args.subplot_hspace),
        cmap=args.cmap,
        contrast_percentiles=contrast,
        dry_run=bool(args.dry_run),
        annotate_chunks=not bool(args.no_chunk_annotation),
        match_pre_post_bin_count=bool(args.match_pre_post_bin_count),
    )


if __name__ == "__main__":
    main()
