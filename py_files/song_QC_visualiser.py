#!/usr/bin/env python
"""
song_feature_QC_visualiser.py   ·   v2025‑07‑17
==============================================
Visual QC for *detected* song intervals only
( single yellow overlay, no amplitude / entropy panels ).

JSON format required
--------------------
[
  {
    "file_name": "...wav",
    "file_path": ".../path/to/file.wav",
    "duration_seconds": 12.34,
    "contains_song": true | false,
    "detected_song_times": [[start, end], ...]   # seconds
  },
  ...
]

The script stacks spectrogram “rows” (default 10 s each),
draws transparent yellow bands wherever `detected_song_times`
fall, and places a dashed red line at every file boundary.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import soundfile as sf
from scipy.signal import spectrogram

# ───────── tunables ──────────
ROW_DUR         = 10.0        # seconds per horizontal lane
SPEC_NPERSEG    = 1024
SPEC_NOVERLAP   = 512
SPEC_VMIN, SPEC_VMAX = -90, -20   # dB colour‑range


# ────────────────── helpers ──────────────────
def _draw_segment(ax, audio, sr, offset, dur):
    """Draw a greyscale spectrogram of `audio[0:dur]` at x‑offset `offset`."""
    if dur <= 0:
        return
    
    audio_segment = audio[: int(round(dur * sr))]
    if len(audio_segment) == 0:
        return
        
    # Adjust nperseg if audio is too short
    nperseg = min(SPEC_NPERSEG, len(audio_segment))
    if nperseg == 0:
        return
        
    f, t, S = spectrogram(
        audio_segment,
        fs=sr, nperseg=nperseg, noverlap=SPEC_NOVERLAP,
        scaling="spectrum", mode="magnitude",
    )
    
    if S.size == 0:
        return
        
    S_db = 20 * np.log10(S + 1e-12)
    ax.pcolormesh(
        offset + t, f, S_db,
        shading="auto", cmap="gray_r",
        vmin=SPEC_VMIN, vmax=SPEC_VMAX,
    )


def _highlight(ax, spans, x_offset, seg_start, seg_dur, alpha=0.45, zorder=5):
    """
    Draw spans that intersect current window [seg_start, seg_start+seg_dur),
    coloring each by its syllable label via tab20.
    `spans` = list of (label, s, e).
    """
    seg_end = seg_start + seg_dur
    for label, s, e in spans:
        if e <= seg_start or s >= seg_end:
            continue
        ax.axvspan(
            x_offset + max(s, seg_start) - seg_start,
            x_offset + min(e, seg_end)   - seg_start,
            color=_color_for_label(label), alpha=alpha, zorder=zorder
        )



def _save_legend(out_dir: Path):
    """One-off legend PNG; keep panel clean."""
    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    ax.axis("off")
    handles = [
        mpatches.Patch(color=_tab20(0), alpha=0.45, label="Syllable spans (tab20 by label)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center")
    fig.savefig(out_dir / "QC_key.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ────────────────── core engine ──────────────────
def build_QC_panels(json_path: str | Path,
                    *,
                    sr: int = 44_100,
                    rows_per_fig: int = 6,
                    audio_base_path: str = "",
                    max_files: int = None):
    json_path  = Path(json_path)
    
    # Create output directory in plots/
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique folder name based on JSON filename
    json_name = json_path.stem  # Get filename without extension
    output_dir = plots_dir / f"qc_panels_{json_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for song_present status
    songs_present_dir = output_dir / "songs_present"
    songs_absent_dir = output_dir / "songs_absent"
    songs_present_dir.mkdir(parents=True, exist_ok=True)
    songs_absent_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created subdirectories: {songs_present_dir}, {songs_absent_dir}")

    with json_path.open() as f:
        recs_all = json.load(f)

    # Shuffle the records for random sampling
    import random
    random.shuffle(recs_all)
    print(f"[INFO] Shuffled {len(recs_all)} records for random sampling...")

    # Convert format - process all records and separate by song_present status
    recs = []
    songs_present_recs = []
    songs_absent_recs = []
    
    # Calculate max files per category (divide by 2)
    max_files_per_category = max_files // 2 if max_files else None
    
    for i, r in enumerate(recs_all):
        # Convert from your format to expected format
        segments = r.get("segments", [])
        detected_song_times = []
        
        for segment in segments:
            onset_ms = segment.get("onset_ms", 0)
            offset_ms = segment.get("offset_ms", 0)
            # Convert milliseconds to seconds
            onset_sec = onset_ms / 1000.0
            offset_sec = offset_ms / 1000.0
            detected_song_times.append([onset_sec, offset_sec])
        
        # Create converted record
        file_name = r.get("filename", "")
        if audio_base_path:
            # Search for the file recursively in the audio directory
            audio_dir = Path(audio_base_path)
            found_file = None
            for audio_file in audio_dir.rglob(file_name):
                found_file = audio_file
                break
            file_path = str(found_file) if found_file else file_name
        else:
            file_path = file_name
        
        converted_rec = {
            "file_name": file_name,
            "file_path": file_path,
            "duration_seconds": 30.0,  # Default duration - you may need to adjust this
            "contains_song": r.get("song_present", False),
            "detected_song_times": detected_song_times
        }
        
        # Separate by song_present status
        if converted_rec["contains_song"]:
            if max_files_per_category is None or len(songs_present_recs) < max_files_per_category:
                songs_present_recs.append(converted_rec)
        else:
            if max_files_per_category is None or len(songs_absent_recs) < max_files_per_category:
                songs_absent_recs.append(converted_rec)
        
        # Early exit if we have enough files in both categories
        if max_files_per_category and len(songs_present_recs) >= max_files_per_category and len(songs_absent_recs) >= max_files_per_category:
            print(f"[INFO] Found enough files in both categories, stopping early at record {i+1}")
            break
    
    print(f"[INFO] Separated records: {len(songs_present_recs)} with songs, {len(songs_absent_recs)} without songs")
    
    if not songs_present_recs and not songs_absent_recs:
        print("[WARN] Nothing to plot – no records found.")
        return

    # Process songs_present records
    if songs_present_recs:
        print(f"[INFO] Processing {len(songs_present_recs)} files with songs present...")
        _process_records(songs_present_recs, songs_present_dir, sr, rows_per_fig)
    
    # Process songs_absent records  
    if songs_absent_recs:
        print(f"[INFO] Processing {len(songs_absent_recs)} files with songs absent...")
        _process_records(songs_absent_recs, songs_absent_dir, sr, rows_per_fig)


def _process_records(recs, output_dir, sr, rows_per_fig):
    """Process a list of records and save QC panels to the specified output directory."""
    rec_idx, rec_progress, fig_no = 0, 0.0, 1
    legend_done = False
    
    print(f"[INFO] Starting to generate QC panels for {len(recs)} files...")

    while rec_idx < len(recs):
        fig, axes = plt.subplots(
            rows_per_fig, 1,
            figsize=(10, 2 * rows_per_fig),
            sharex=True, constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for ax in axes:
            if rec_idx >= len(recs):
                ax.axis("off")
                continue

            offset, titles = 0.0, []
            while offset < ROW_DUR and rec_idx < len(recs):
                rec      = recs[rec_idx]
                wav_path = Path(rec["file_path"])

                # ─── lazy‑load WAV once per recording ───
                if rec_progress == 0.0:
                    audio_full, sr_file = sf.read(wav_path)
                    if audio_full.ndim > 1:
                        audio_full = audio_full.mean(axis=1)
                    if sr_file != sr:
                        print(f"[WARN] {wav_path.name}: "
                              f"sr {sr_file} ≠ {sr}; using {sr_file}")
                        sr = sr_file

                remaining = rec["duration_seconds"] - rec_progress
                seg_dur   = min(remaining, ROW_DUR - offset)

                # slice audio for this row fragment
                start_idx = int(round(rec_progress * sr))
                seg_audio = audio_full[
                    start_idx : start_idx + int(round(seg_dur * sr))
                ]
                _draw_segment(ax, seg_audio, sr, offset, seg_dur)

                # overlay yellow spans
                shift = rec_progress
                shifted_spans = [
                    (s - shift, e - shift) for s, e in rec["detected_song_times"]
                ]
                _highlight(ax, shifted_spans, offset, seg_dur)

                # advance cursors
                rec_progress += seg_dur
                offset       += seg_dur

                finished = abs(rec_progress - rec["duration_seconds"]) < 1e-6
                if finished:
                    ax.axvline(offset, color="red", ls="--", lw=1.0, zorder=6)
                    titles.append(wav_path.name)
                    rec_idx     += 1
                    rec_progress = 0.0

            # aesthetics
            ax.set_xlim(0, ROW_DUR)
            ax.set_ylim(0, 10_000)
            ax.set_yticks([0, 2500, 5000, 7500, 10_000])
            ax.set_ylabel("Freq [Hz]")
            ax.tick_params(labelsize=8)
            if titles:
                ax.set_title(" + ".join(titles), fontsize=9, pad=4)

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Detected‑song QC panel {fig_no}", fontsize=12)
        out_png = output_dir / f"song_QC_panel_{fig_no:03d}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved {out_png.resolve()} (panel {fig_no}, processed {rec_idx}/{len(recs)} files)")

        if not legend_done:
            _save_legend(output_dir)
            legend_done = True

        fig_no += 1


# ────────────────── CLI entrypoint ──────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Visual QC for detected_song_times JSON."
    )
    ap.add_argument("--json_path",  required=True, type=Path)
    ap.add_argument("--sr",         default=44_100, type=int,
                    help="Expected sample rate (overridden if WAV differs).")
    ap.add_argument("--rows",       default=6, type=int, dest="rows_per_fig",
                    help="Spectrogram rows per PNG.")
    ap.add_argument("--audio_path", default="", type=str,
                    help="Base path to audio files (required if JSON contains only filenames).")
    ap.add_argument("--max_files", default=None, type=int,
                    help="Maximum number of files to process (divided by 2 for each category: songs_present and songs_absent).")
    args = ap.parse_args()

    build_QC_panels(
        args.json_path,
        sr=args.sr,
        rows_per_fig=args.rows_per_fig,
        audio_base_path=args.audio_path,
        max_files=args.max_files,
    )


#json_path = "/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class_clean/data_inputs/USA5510_unsegmented_songs/55_subsample_detected_song_intervals.json"
#output_dir = "/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class_clean/data_inputs/USA5510_unsegmented_songs/55_subsample_detected_song_qc_panels"
#build_QC_panels(json_path, output_dir,sr=44100,rows_per_fig=6)