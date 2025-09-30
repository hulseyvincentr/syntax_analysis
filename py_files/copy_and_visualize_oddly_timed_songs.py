# copy_and_visualize_oddly_time_songs.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import csv
import collections

# ───────── optional plotting (hour/date histograms & PNG spectrograms) ─────────
try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# ───────── spectrogram deps (required for PNGs) ─────────
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

# ───────── spectrogram tunables (match your script) ─────────
ROW_DUR = 10.0            # seconds per horizontal lane
SPEC_NPERSEG = 1024
SPEC_NOVERLAP = 512
SPEC_VMIN, SPEC_VMAX = -90, -20   # dB colour-range (gray_r)


@dataclass
class OffhourItem:
    file_name: str
    file_stem: Optional[str]
    segment: Optional[int]
    date_str: Optional[str]
    hour: Optional[int]


def _load_offhours_json(json_path: Path) -> Tuple[str, List[OffhourItem]]:
    with json_path.open("r") as f:
        data = json.load(f)
    animal_id = data.get("animal_id", "UNKNOWN_ANIMAL")
    items: List[OffhourItem] = []
    for rec in data.get("offhours_segments", []):
        items.append(
            OffhourItem(
                file_name=rec.get("file_name", ""),
                file_stem=rec.get("file_stem") or None,
                segment=rec.get("segment"),
                date_str=rec.get("date_str"),
                hour=rec.get("hour"),
            )
        )
    return animal_id, items


def _index_audio_files(parent_dir: Path, audio_exts=(".wav", ".flac", ".mp3")) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = collections.defaultdict(list)
    for ext in audio_exts:
        for p in parent_dir.rglob(f"*{ext}"):
            try:
                index[p.name].append(p)
                index[p.stem].append(p)
            except Exception:
                pass
    return index


def _resolve_candidates(item: OffhourItem, index: Dict[str, List[Path]], audio_exts) -> List[Path]:
    candidates: List[Path] = []

    # exact filename (with/without extension)
    if Path(item.file_name).suffix:
        candidates += index.get(item.file_name, [])
    else:
        for ext in audio_exts:
            candidates += index.get(item.file_name + ext, [])

    # stem match
    candidates += index.get(Path(item.file_name).stem, [])

    # optional: prefix fuzziness using file_stem
    if item.file_stem:
        candidates += index.get(item.file_stem, [])
        fuzzies = []
        prefix = item.file_stem
        for key, paths in index.items():
            if "." not in key and key.startswith(prefix):
                fuzzies.extend(paths)
        candidates += fuzzies

    # dedupe
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


# ─────────────────────────────────────────────────────────────
# Title helpers (adds date & time to figure titles)
# ─────────────────────────────────────────────────────────────
def _guess_datetime_from_name(name: str) -> Optional[str]:
    """
    Try to recover 'YYYY-MM-DD HH:MM:SS' from a filename.
    Supports common forms like:
      YYYYMMDD_HHMMSS, YYYY-MM-DD_HH-MM-SS, YYYYMMDDHHMMSS, etc.
    Returns a nice string or None.
    """
    stem = Path(name).stem

    # YYYY[-_]MM[-_]DD[T/_ -]?HH[-_]MM[-_]SS
    m = re.search(
        r'(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[T/_ -]?([01]\d|2[0-3])[-_]?([0-5]\d)[-_]?([0-5]\d)',
        stem
    )
    if m:
        y, mo, d, H, M, S = m.groups()
        return f"{y}-{mo}-{d} {H}:{M}:{S}"

    # Date only + time only (combine if both appear)
    md = re.search(r'(20\d{2})[-_]?(\d{2})[-_]?(\d{2})', stem)
    mt = re.search(r'([01]\d|2[0-3])[-_]?([0-5]\d)(?:[-_]?([0-5]\d))?', stem)
    if md and mt:
        y, mo, d = md.groups()
        H, M, S = mt.group(1), mt.group(2), mt.group(3) or '00'
        return f"{y}-{mo}-{d} {H}:{M}:{S}"

    return None


def _build_title_prefix(item: OffhourItem, path: Path) -> str:
    """
    Best-effort recording datetime string for the figure title.
    Priority:
      1) Parse full datetime from filename
      2) JSON date_str + hour (HH:00)
      3) File mtime
    """
    # 1) try filename
    guess = _guess_datetime_from_name(path.name)
    if guess:
        return guess

    # 2) JSON fields
    if item.date_str and item.hour is not None:
        return f"{item.date_str} {int(item.hour):02d}:00"
    if item.date_str:
        return item.date_str

    # 3) mtime as a last resort
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# Spectrogram drawing (whole file → stacked 10 s rows)
# ─────────────────────────────────────────────────────────────
def _draw_segment(ax, audio, sr, x_offset, seg_dur):
    """Draw a grayscale spectrogram of 'audio' (<= seg_dur) at x-offset 'x_offset'."""
    if seg_dur <= 0 or audio.size == 0:
        return

    nperseg = min(SPEC_NPERSEG, len(audio))
    if nperseg <= 0:
        return

    f, t, S = spectrogram(
        audio,
        fs=sr,
        nperseg=nperseg,
        noverlap=min(SPEC_NOVERLAP, max(0, nperseg // 2)),
        scaling="spectrum",
        mode="magnitude",
    )
    if S.size == 0:
        return
    S_db = 20 * np.log10(S + 1e-12)
    ax.pcolormesh(
        x_offset + t, f, S_db, shading="auto", cmap="gray_r",
        vmin=SPEC_VMIN, vmax=SPEC_VMAX
    )


def _save_spectrogram_png(
    wav_path: Path,
    out_png: Path,
    rows_per_fig: int = 6,
    title_prefix: Optional[str] = None
) -> None:
    """
    Render WAV into stacked 10-second spectrogram rows and save to out_png.
    No overlays (the off-hours JSON has no spans).
    If matplotlib is unavailable, silently skip creating PNGs.
    """
    if not _HAVE_MPL:
        return

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur_sec = len(audio) / float(sr) if sr > 0 else 0.0
    if dur_sec <= 0:
        # create an empty placeholder image to keep pipeline consistent
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis("off")
        base = f"{wav_path.name}  •  {dur_sec:.2f}s @ {sr} Hz"
        title = f"{title_prefix}  •  {base}" if title_prefix else base
        fig.suptitle(title, fontsize=11)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    # We'll fill each figure with up to 'rows_per_fig' rows, each ROW_DUR seconds wide
    remaining = dur_sec
    cursor = 0.0

    # If a file fits in one figure, just draw it and add a dashed boundary at the end.
    # For longer files: multiple figures numbered ..._p01.png, _p02.png, etc.
    fig_index = 1
    while remaining > 1e-9:
        fig, axes = plt.subplots(
            rows_per_fig, 1,
            figsize=(10, 2 * rows_per_fig),
            sharex=True, constrained_layout=True
        )
        axes = np.atleast_1d(axes)

        for ax in axes:
            if remaining <= 1e-9:
                ax.axis("off")
                continue

            # One row spans up to ROW_DUR seconds, but may include multiple fragments
            # from the same file if it’s longer than ROW_DUR.
            x_offset = 0.0
            row_left = ROW_DUR

            while row_left > 1e-9 and remaining > 1e-9:
                seg_dur = min(row_left, remaining)
                start_i = int(round(cursor * sr))
                seg_i = int(round(seg_dur * sr))
                seg_audio = audio[start_i:start_i + seg_i]
                _draw_segment(ax, seg_audio, sr, x_offset, seg_dur)

                cursor += seg_dur
                remaining -= seg_dur
                x_offset += seg_dur
                row_left -= seg_dur

            # aesthetics per row
            ax.set_xlim(0, ROW_DUR)
            ax.set_ylim(0, 10_000)
            ax.set_yticks([0, 2500, 5000, 7500, 10_000])
            ax.set_ylabel("Freq [Hz]")
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel("Time [s]")
        base = f"{wav_path.name}  •  {dur_sec:.2f}s @ {sr} Hz"
        title = f"{title_prefix}  •  {base}" if title_prefix else base
        fig.suptitle(title, fontsize=11)

        if dur_sec <= ROW_DUR * rows_per_fig and fig_index == 1:
            # simple (single figure)
            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            break
        else:
            # multi-figure: append page index
            paged_png = out_png.with_name(out_png.stem + f"_p{fig_index:02d}" + out_png.suffix)
            fig.savefig(paged_png, dpi=300)
            plt.close(fig)
            fig_index += 1


def copy_and_visualize_oddly_time_songs(
    json_path: str | Path,
    parent_directory: str | Path,
    output_directory: str | Path,
    *,
    audio_exts: Tuple[str, ...] = (".wav", ".flac", ".mp3"),
    overwrite: bool = False,
    make_histograms: bool = True,
    make_spectrogram_pngs: bool = True,
    rows_per_fig: int = 6,
    manifest_filename: str = "offhours_manifest.csv",
) -> Dict[str, object]:
    """
    Copy oddly-timed audio files into output_directory and (optionally) save spectrogram PNGs.

    - Spectrogram settings match your QC script:
      ROW_DUR=10 s, nperseg=1024, noverlap=512, grayscale, vmin=-90, vmax=-20.
    - Writes a manifest CSV with src, dest, and spectrogram path (if created).
    - Figure title now includes a best-effort recording datetime (parsed from name,
      else JSON date/hour, else file mtime).

    Returns
    -------
    dict with basic counts and paths.
    """
    json_path = Path(json_path).expanduser().resolve()
    parent_directory = Path(parent_directory).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    animal_id, items = _load_offhours_json(json_path)
    index = _index_audio_files(parent_directory, audio_exts=audio_exts)

    manifest_rows: List[Dict[str, object]] = []
    found_count = 0
    copied_count = 0

    for item in items:
        candidates = _resolve_candidates(item, index, audio_exts=audio_exts)
        if not candidates:
            manifest_rows.append({
                "animal_id": animal_id,
                "file_name": item.file_name,
                "segment": item.segment,
                "date_str": item.date_str,
                "hour": item.hour,
                "src_path": "",
                "copied_to": "",
                "spectrogram_png": "",
                "status": "NOT_FOUND",
            })
            continue

        # prefer exact stem match first
        preferred = None
        for c in candidates:
            if c.stem == item.file_name:
                preferred = c
                break
        preferred = preferred or candidates[0]
        found_count += 1

        # copy to output dir
        dest = output_directory / preferred.name
        if dest.exists() and not overwrite:
            status = "ALREADY_EXISTS"
        else:
            shutil.copy2(preferred, dest)
            status = "COPIED"
            copied_count += 1 if status == "COPIED" else 0

        # spectrogram PNG (one per audio; multi-page files get _p01, _p02…)
        png_path = ""
        if make_spectrogram_pngs and _HAVE_MPL:
            try:
                png_out = output_directory / (preferred.stem + ".png")
                title_prefix = _build_title_prefix(item, preferred)
                _save_spectrogram_png(
                    dest if dest.exists() else preferred,
                    png_out,
                    rows_per_fig=rows_per_fig,
                    title_prefix=title_prefix or None,
                )
                png_path = str(png_out)
            except Exception as e:
                # keep pipeline resilient; note the error in manifest
                png_path = f"ERROR: {e}"
        elif make_spectrogram_pngs and not _HAVE_MPL:
            png_path = "SKIPPED: matplotlib not available"

        manifest_rows.append({
            "animal_id": animal_id,
            "file_name": item.file_name,
            "segment": item.segment,
            "date_str": item.date_str,
            "hour": item.hour,
            "src_path": str(preferred),
            "copied_to": str(dest),
            "spectrogram_png": png_path,
            "status": status,
        })

    # write manifest
    manifest_path = output_directory / manifest_filename
    fieldnames = ["animal_id", "file_name", "segment", "date_str", "hour", "src_path", "copied_to", "spectrogram_png", "status"]
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    # quick histograms (hour/date) if requested
    if make_histograms and _HAVE_MPL:
        hour_counts = collections.Counter([r["hour"] for r in manifest_rows if r["hour"] is not None])
        if hour_counts:
            hours_sorted = sorted(hour_counts.keys())
            counts = [hour_counts[h] for h in hours_sorted]
            plt.figure()
            plt.bar(hours_sorted, counts)
            plt.xlabel("Hour of day")
            plt.ylabel("Off-hours segments (count)")
            plt.title(f"{animal_id}: off-hours counts by hour")
            plt.tight_layout()
            plt.savefig(output_directory / "offhours_by_hour.png", dpi=150)
            plt.close()

        date_counts = collections.Counter([r["date_str"] for r in manifest_rows if r["date_str"]])
        if date_counts:
            dates_sorted = sorted(date_counts.keys())
            counts = [date_counts[d] for d in dates_sorted]
            plt.figure()
            plt.bar(range(len(dates_sorted)), counts)
            plt.xticks(range(len(dates_sorted)), dates_sorted, rotation=45, ha="right")
            plt.xlabel("Date")
            plt.ylabel("Off-hours segments (count)")
            plt.title(f"{animal_id}: off-hours counts by date")
            plt.tight_layout()
            plt.savefig(output_directory / "offhours_by_date.png", dpi=150)
            plt.close()

    return {
        "animal_id": animal_id,
        "n_requested": len(items),
        "n_found": found_count,
        "n_copied": copied_count,
        "manifest_path": manifest_path,
        "output_dir": output_directory,
    }

"""
# -------------------------
# Example usage:
# -------------------------
from copy_and_visualize_oddly_timed_songs import copy_and_visualize_oddly_time_songs
out = copy_and_visualize_oddly_time_songs(
     json_path="/Users/mirandahulsey-vincent/Desktop/AFP_lesion_QC_detection/USA5483/USA5483_offhours_segments.json",
     parent_directory="/Volumes/GLABSSD/Gardner_447D_Comp1_Comp2_and_laptop_SONGS_24June2025/2025_AreaX/USA5483_RC6_Comp2",
     output_directory="/Volumes/GLABSSD/2025_song_detector_QC/USA5483",
     audio_exts=(".wav", ".flac"),
     overwrite=False,
     make_histograms=True,
     make_spectrogram_pngs=True,
     rows_per_fig=6,
 )
print(out)

"""
