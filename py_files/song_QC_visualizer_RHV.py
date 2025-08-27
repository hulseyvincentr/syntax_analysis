#!/usr/bin/env python
"""
qc_each_wav_with_annotations.py  ·  v2025-08-26d
One spectrogram PNG per .wav, overlaying annotated (syllable) spans.

Key features:
- Supports JSON where file_name removes ".wav" and appends a segment index:
    e.g., "USA5199_45106.45073191_6_29_12_31_13_0"
  All segments with the same base (before the final "_<int>" or "_segment_<int>")
  map to "<base>.wav".
- Robust JSON loader:
  * top-level object with "results" list
  * wrappers: records/items/data/decoded_database/entries
  * pointer files (value is a path string to JSON)
  * double-encoded JSON
  * NDJSON / JSONL
- User-controlled output directory
- Verbose diagnostics (counts, example keys)
- Label-based highlighting: tab20 palette per syllable label

Per-record schemas:
A) New schema (preferred):
   - file_name: "<wav_base_without_dotwav>_<segmentIndex>" OR "<wav_basename>.wav"
   - song_present: bool
   - syllable_onsets_offsets_ms: { "<label>": [[start_ms, end_ms], ...], ... }

B) Fallback:
   - detected_song_times: [[start_sec, end_sec], ...] (no labels; uses a single color)

Default output:
plots/qc_each_wav_<json_stem>/
  QC_key.png
  <wav_basename>_qc.png
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import soundfile as sf
from scipy.signal import spectrogram

# ───────── tunables (defaults; override by CLI) ─────────
SPEC_NPERSEG    = 1024
SPEC_NOVERLAP   = 512
SPEC_VMIN, SPEC_VMAX = -90, -20   # dB colour-range
FMAX_HZ = 10_000                   # y-axis limit

# ───────────────── helpers ─────────────────
# Accepts either "..._<int>" or "..._segment_<int>"
_SEG_SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?:_segment)?_(?P<seg>\d+)$")

def _load_records(json_path: Path) -> List[dict]:
    """
    Load annotations and return list[dict].
    Handles:
      - list[dict] (ideal)
      - dict wrappers like {"results":[...]} (plus: records/items/data/decoded_database/entries)
      - 'pointer' dict/list whose value(s) are path strings to another JSON
      - double-encoded JSON (top-level string containing JSON)
      - NDJSON / JSONL (one object per line)
    """
    WRAPPER_KEYS = ("results", "records", "items", "data", "decoded_database", "entries")

    def _try_interpret_as_records(obj) -> Optional[List[dict]]:
        # Direct list of dicts
        if isinstance(obj, list):
            if not obj:
                return []
            if isinstance(obj[0], dict):
                return obj
            if isinstance(obj[0], str):
                # List of possible file paths → follow the first that yields records
                for s in obj:
                    p = Path(s)
                    if p.exists() and p.suffix.lower() == ".json":
                        try:
                            with p.open("r") as f2:
                                nested = json.load(f2)
                            recs = _try_interpret_as_records(nested)
                            if recs is not None:
                                return recs
                        except Exception:
                            pass
                # List of JSON strings
                parsed = []
                for s in obj:
                    try:
                        parsed.append(json.loads(s))
                    except Exception:
                        return None
                return parsed if parsed and isinstance(parsed[0], dict) else None

        # Dict wrappers or pointers
        if isinstance(obj, dict):
            # Unwrap common wrapper keys (including "results")
            for k in WRAPPER_KEYS:
                if k in obj:
                    v = obj[k]
                    # Pointer value: wrapper key → string path to JSON
                    if isinstance(v, str):
                        p = Path(v)
                        if p.exists() and p.suffix.lower() == ".json":
                            try:
                                with p.open("r") as f2:
                                    nested = json.load(f2)
                                recs = _try_interpret_as_records(nested)
                                if recs is not None:
                                    return recs
                            except Exception:
                                pass
                    # Normal list payload
                    if isinstance(v, list):
                        recs = _try_interpret_as_records(v)
                        if recs is not None:
                            return recs
            # Dict of {id: record}
            vals = list(obj.values())
            if vals and isinstance(vals[0], dict):
                return vals
            return None

        # Double-encoded JSON string
        if isinstance(obj, str):
            try:
                return _try_interpret_as_records(json.loads(obj))
            except Exception:
                return None

        return None

    # Load primary file
    with json_path.open("r") as f:
        data = json.load(f)

    # Best-effort normalization
    recs = _try_interpret_as_records(data)
    if recs is not None:
        return recs

    # Last resort: NDJSON/JSONL (one object per line)
    records = []
    try:
        with json_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except Exception:
                    pass
        if records:
            return records
    except Exception:
        pass

    raise ValueError(
        f"Could not parse {json_path} into a list of records. "
        "Expected JSON list of objects, dict containing a list (e.g. 'results'), "
        "pointer(s) to a JSON file, or NDJSON."
    )

def _basename_with_wav(name: str) -> str:
    """
    Convert a JSON file_name to a WAV basename:
      - If it ends with ".wav" already, return as-is.
      - If it ends with "_<int>" or "_segment_<int>", strip that and add ".wav".
      - Otherwise, add ".wav".
    """
    if name.endswith(".wav"):
        return name
    m = _SEG_SUFFIX_RE.match(name)
    if m:
        return f"{m.group('base')}.wav"
    return f"{name}.wav"

# ─── label coloring (tab20) ───
_label_to_color: Dict[str, tuple] = {}
_tab20 = cm.get_cmap("tab20", 20)  # 20 distinct colors

def _color_for_label(label: str):
    """
    Map a label string (e.g., '1','2','7','song') to a stable color from tab20.
    Numeric labels use their int value; others use a hash modulo 20.
    """
    if label in _label_to_color:
        return _label_to_color[label]
    try:
        idx = int(label) % 20
    except Exception:
        idx = (hash(label) % 20)
    color = _tab20(idx)
    _label_to_color[label] = color
    return color

# ─── span extraction (keep labels) ───
def _spans_from_ms_dict(ms_dict: dict | None) -> List[Tuple[str, float, float]]:
    """
    Flatten syllable_onsets_offsets_ms to [(label, s_sec, e_sec), ...].
    Labels are the dict keys (e.g., "1","2","7").
    """
    out: List[Tuple[str, float, float]] = []
    if not ms_dict:
        return out
    for label, intervals in ms_dict.items():
        if not intervals:
            continue
        for pair in intervals:
            if not pair or len(pair) < 2:
                continue
            s_ms, e_ms = float(pair[0]), float(pair[1])
            if e_ms > s_ms:
                out.append((str(label), s_ms / 1000.0, e_ms / 1000.0))
    return out

def _to_seconds_spans(rec: dict) -> List[Tuple[str, float, float]]:
    """Return [(label, s, e), ...] from any supported schema on a single record."""
    # Preferred new schema (labeled syllable intervals)
    if "syllable_onsets_offsets_ms" in rec:
        return _spans_from_ms_dict(rec.get("syllable_onsets_offsets_ms"))

    # Legacy seconds schema (no labels)
    if "detected_song_times" in rec:
        spans = rec.get("detected_song_times") or []
        return [("song", float(s), float(e)) for s, e in spans if float(e) > float(s)]

    # Fallback: nothing usable
    return []

def _build_annotation_index(json_path: Path) -> Dict[str, List[Tuple[str, float, float]]]:
    """
    Map WAV basename (e.g., 'X.wav') → list of (label, start_sec, end_sec).
    Supports segmented names ('X_0', 'X_1' → 'X.wav') and merges spans across segments.
    """
    data = _load_records(json_path)

    idx: Dict[str, List[Tuple[str, float, float]]] = {}
    for rec in data:
        if not isinstance(rec, dict):
            continue
        name = rec.get("file_name") or rec.get("filename")
        if not name:
            continue

        wav_key = _basename_with_wav(name)
        spans   = _to_seconds_spans(rec)

        bucket = idx.setdefault(wav_key, [])
        if spans:
            bucket.extend(spans)

    return idx

def _save_legend(out_dir: Path):
    """One-off legend PNG so panels stay clean."""
    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    ax.axis("off")
    handles = [
        mpatches.Patch(color=_tab20(0), alpha=0.45, label="Syllable spans (tab20 by label)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center")
    fig.savefig(out_dir / "QC_key.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def _draw_segment(ax, audio, sr, x_offset, dur_s):
    """Greyscale spectrogram of 'audio[:dur_s]' drawn at x offset."""
    if dur_s <= 0:
        return
    n_samples = int(round(dur_s * sr))
    if n_samples <= 0:
        return
    audio_segment = audio[:n_samples]
    if np.size(audio_segment) == 0:
        return
    # adapt nperseg for very short segments
    nperseg = min(SPEC_NPERSEG, len(audio_segment))
    if nperseg <= 0:
        return

    f, t, S = spectrogram(
        audio_segment,
        fs=sr, nperseg=nperseg, noverlap=SPEC_NOVERLAP,
        scaling="spectrum", mode="magnitude",
    )
    if S.size == 0:
        return

    S_db = 20 * np.log10(S + 1e-12)
    ax.pcolormesh(x_offset + t, f, S_db, shading="auto",
                  cmap="gray_r", vmin=SPEC_VMIN, vmax=SPEC_VMAX)

def _highlight(ax,
               spans: List[Tuple[str, float, float]],
               x_offset: float,
               seg_start: float,
               seg_dur: float,
               alpha: float = 0.45,
               zorder: int = 5):
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

def _plot_single_wav(wav_path: Path,
                     spans_sec: List[Tuple[str, float, float]],
                     *,
                     row_dur: float,
                     expected_sr: int,
                     out_dir: Path):
    """Make one PNG for this wav with colored overlays per syllable label."""
    audio, sr = sf.read(wav_path)
    if np.ndim(audio) > 1:
        audio = np.mean(audio, axis=1)
    if sr != expected_sr:
        print(f"[WARN] {wav_path.name}: sr {sr} ≠ expected {expected_sr}; proceeding with {sr}")
    dur_s = len(audio) / sr if len(audio) else 0.0
    if dur_s <= 0:
        print(f"[WARN] {wav_path.name}: zero duration?")
        return

    rows_needed = max(1, int(np.ceil(dur_s / row_dur)))
    fig, axes = plt.subplots(
        rows_needed, 1,
        figsize=(10, 2 * rows_needed),
        sharex=False, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    t_cursor = 0.0
    for i, ax in enumerate(axes):
        if t_cursor >= dur_s:
            ax.axis("off")
            continue

        window_dur = min(row_dur, dur_s - t_cursor)

        s_idx = int(round(t_cursor * sr))
        e_idx = s_idx + int(round(window_dur * sr))
        seg_audio = audio[s_idx:e_idx]

        _draw_segment(ax, seg_audio, sr, x_offset=0.0, dur_s=window_dur)
        _highlight(ax, spans_sec, x_offset=0.0, seg_start=t_cursor, seg_dur=window_dur)

        ax.set_xlim(0, window_dur)
        ax.set_ylim(0, FMAX_HZ)
        ax.set_yticks([0, 2500, 5000, 7500, 10_000])
        ax.set_ylabel("Freq [Hz]")
        ax.tick_params(labelsize=8)
        ax.set_title(f"{wav_path.name}  (row {i+1}/{rows_needed})", fontsize=9, pad=4)

        t_cursor += window_dur

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Detected-song QC: {wav_path.name}", fontsize=12)
    out_png = out_dir / f"{wav_path.stem}_qc.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {out_png.resolve()}")

# ───────────────── public API (function-call style) ─────────────────
def run_qc_each_wav(json_path, wav_dir, *, row_dur=10.0, sr=44_100, recursive=False, out_dir=None, verbose=False):
    """
    Make QC spectrograms for each .wav with annotated spans.

    Parameters
    ----------
    json_path : str | Path
        Path to annotations JSON (or a file that points to the real JSON).
    wav_dir : str | Path
        Directory containing WAV files.
    row_dur : float
        Seconds per spectrogram row.
    sr : int
        Expected sample rate (used for warnings only).
    recursive : bool
        Whether to search subdirectories of wav_dir.
    out_dir : str | Path | None
        Output directory. If None, defaults to plots/qc_each_wav_<json_stem>.
    verbose : bool
        Print diagnostics about parsed JSON and mapping.
    """
    json_path = Path(json_path)
    wav_dir   = Path(wav_dir)

    if out_dir is None:
        plots_dir = Path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_dir = plots_dir / f"qc_each_wav_{json_path.stem}"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_idx = _build_annotation_index(json_path)

    if verbose:
        print(f"[DEBUG] JSON: {json_path}")
        try:
            data = _load_records(Path(json_path))
            print(f"[DEBUG] Loaded {len(data)} records total")
            n_named = sum(1 for r in data if isinstance(r, dict) and (r.get('file_name') or r.get('filename')))
            n_ms    = sum(1 for r in data if isinstance(r, dict) and isinstance(r.get('syllable_onsets_offsets_ms'), dict))
            n_times = sum(1 for r in data if isinstance(r, dict) and isinstance(r.get('detected_song_times'), list))
            print(f"[DEBUG] Records with file name: {n_named}")
            print(f"[DEBUG] Records with syllable_onsets_offsets_ms: {n_ms}")
            print(f"[DEBUG] Records with detected_song_times: {n_times}")
            if ann_idx:
                first_key = next(iter(ann_idx))
                print(f"[DEBUG] Example mapped key: {first_key}  (spans={len(ann_idx[first_key])})")
        except Exception as e:
            print(f"[DEBUG] Loader error: {e}")

    if not ann_idx:
        print("[WARN] No usable annotations found in JSON.")

    pattern = "**/*.wav" if recursive else "*.wav"
    wavs = sorted(wav_dir.glob(pattern))
    if not wavs:
        print(f"[WARN] No .wav files found in {wav_dir}")
        return out_dir

    _save_legend(out_dir)

    for wav in wavs:
        spans = ann_idx.get(wav.name, [])
        if not spans:
            print(f"[INFO] {wav.name}: no annotations found; plotting without overlays.")
        _plot_single_wav(
            wav, spans_sec=spans,
            row_dur=row_dur,
            expected_sr=sr,
            out_dir=out_dir
        )

    return out_dir

# ───────────────── CLI wrapper (optional) ─────────────────
def main():
    ap = argparse.ArgumentParser(description="Make one QC spectrogram per .wav with annotated spans.")
    ap.add_argument("--json_path",  required=True, type=Path, help="Annotations JSON (or a pointer file).")
    ap.add_argument("--wav_dir",    required=True, type=Path, help="Folder containing .wav files.")
    ap.add_argument("--row_dur",    default=10.0, type=float, help="Seconds per spectrogram row.")
    ap.add_argument("--sr",         default=44_100, type=int, help="Expected sample rate (messages only).")
    ap.add_argument("--recursive",  action="store_true", help="Recurse into subfolders of --wav_dir.")
    ap.add_argument("--out_dir",    default=None, type=Path, help="Output directory (default: plots/qc_each_wav_<json_stem>).")
    ap.add_argument("--verbose",    action="store_true", help="Print diagnostics about parsed JSON and mapping.")
    args = ap.parse_args()

    run_qc_each_wav(
        json_path=args.json_path,
        wav_dir=args.wav_dir,
        row_dur=args.row_dur,
        sr=args.sr,
        recursive=args.recursive,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()



"""
from song_QC_visualizer_RHV import run_qc_each_wav

json_path = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/example_wav_files/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5199_NEW_decoded_database.json"
wav_dir   = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/example_wav_files"
custom_out = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/example_wav_files/figures"

out_dir = run_qc_each_wav(
    json_path=json_path,
    wav_dir=wav_dir,
    row_dur=10.0,
    sr=44100,
    recursive=False,
    out_dir=custom_out,
    verbose=True,   # ← turn on debug to confirm parsing & filename mapping
)
print("Saved PNGs to:", out_dir)


"""