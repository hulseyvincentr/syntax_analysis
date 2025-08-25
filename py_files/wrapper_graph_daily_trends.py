# -*- coding: utf-8 -*-

#!/usr/bin/env python
"""
wrapper_graph_daily_trends.py
==================

One-stop wrapper to run your birdsong analysis steps against a *single* pair of:
  • decoded_database_json
  • creation_metadata_json

You can:
  • Choose which steps to run (and order)
  • Pass per-step options via JSON
  • Run from CLI or import and call run_pipeline(...)
  • Get a simple JSONL manifest of what ran

Steps wired:
  1) daily_first_order_transitions
  2) phrase_duration_over_days
  3) transition_entropy_daily
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import time
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Your modules (as provided)
# ──────────────────────────────────────────────────────────────────────────────
from build_daily_first_order_transitions_movie import run_daily_first_order_transitions
from graph_phrase_duration_over_days import graph_phrase_duration_over_days
from transition_entropy_from_decoded import analyze_transitions_for_each_day_from_decoded
from syllable_heatmap import (
    build_daily_avg_count_table,
    plot_log_scaled_syllable_counts,
)
from organize_decoded_with_durations import build_organized_dataset_with_durations


# ──────────────────────────────────────────────────────────────────────────────
# Config + Manifest
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    decoded_database_json: Path
    creation_metadata_json: Path
    outdir: Path
    steps: List[str]
    step_opts: Dict[str, Dict[str, Any]]
    resume: bool
    manifest_path: Path

@dataclass
class ManifestRecord:
    timestamp: str
    step: str
    status: str
    details: Dict[str, Any]

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _default_manifest(outdir: Path) -> Path:
    return outdir / f"syntax_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

def _load_completed(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    done: set[str] = set()
    with manifest.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done.add(rec.get("step", ""))
            except Exception:
                pass
    return done

def _append_manifest(manifest: Path, record: ManifestRecord) -> None:
    with manifest.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# Step adapters — thin shims that forward to your functions
# (Each returns a small dict of outputs for manifest readability)
# ──────────────────────────────────────────────────────────────────────────────

def step_daily_first_order_transitions(cfg: PipelineConfig, **kw) -> Dict[str, Any]:
    """
    Wraps: run_daily_first_order_transitions(...)
    Common useful kwargs:
      only_song_present: bool
      restrict_to_labels: list[str] | None
      min_row_total: int
      output_dir: str | Path
      save_csv: bool
      save_png: bool
      show_plots: bool
      save_movie_path: str | Path | None
      movie_fps: int
      movie_figsize: (w, h)
      enforce_consistent_order: bool
    """
    res = run_daily_first_order_transitions(
        decoded_database_json=str(cfg.decoded_database_json),
        creation_metadata_json=str(cfg.creation_metadata_json),
        **kw,
    )
    # Put a couple of handy counts in the manifest
    return {
        "days_exported": len(res),
        "movie_path": str(kw.get("save_movie_path")) if kw.get("save_movie_path") else None,
        "output_dir": str(kw.get("output_dir")) if kw.get("output_dir") else None,
    }

def step_phrase_duration_over_days(cfg: PipelineConfig, **kw) -> Dict[str, Any]:
    """
    Wraps: graph_phrase_duration_over_days(...)
    Common useful kwargs:
      save_output_to_this_file_path: str | Path   (REQUIRED here)
      only_song_present: bool
      y_max_ms: int
      point_alpha: float
      point_size: int
      jitter: float | bool
      dpi: int
      transparent: bool
      figsize: (w, h)
      font_size_labels: int
      xtick_fontsize: int
      xtick_every: int
      show_plots: bool
      syllables_subset: list[str] | None
    """
    out = graph_phrase_duration_over_days(
        decoded_database_json=str(cfg.decoded_database_json),
        creation_metadata_json=str(cfg.creation_metadata_json),
        **kw,
    )
    return {
        "save_dir": str(kw.get("save_output_to_this_file_path")),
        "unique_days": len(getattr(out, "unique_dates", []) or []),
        "n_labels": len(getattr(out, "unique_syllable_labels", []) or []),
        "treatment_date": getattr(out, "treatment_date", None),
    }

def step_transition_entropy_daily(cfg: PipelineConfig, **kw) -> Dict[str, Any]:
    """
    Wraps: analyze_transitions_for_each_day_from_decoded(...)
    Common useful kwargs:
      only_song_present: bool
      compute_durations: bool
      surgery_date_override: str | None  (YYYY-MM-DD)
    """
    organized, per_file_df, entropies_per_day, syll_types = analyze_transitions_for_each_day_from_decoded(
        decoded_database_json=str(cfg.decoded_database_json),
        creation_metadata_json=str(cfg.creation_metadata_json),
        **kw,
    )
    return {
        "n_days": len(entropies_per_day),
        "n_syllable_types": len(syll_types),
        "plotted": True,  # the step shows a plot inline
    }

def step_syllable_heatmap(cfg: PipelineConfig, **kw) -> Dict[str, Any]:
    """
    Build daily avg syllable-count table and save a log10-scaled heatmap.

    Step options (all optional):
      save_dir: str|Path   -> directory to save the PNG.
      pseudocount: float   -> default 1e-3
      figsize: (int,int)   -> default (16, 6)
      cmap: str            -> default 'Greys_r' (low≈white, high≈black)
      cbar_label: str
      sort_dates: bool
      date_format: str
      show: bool           -> default False (good for pipelines)
      mark_treatment: bool -> default True
      nearest_match: bool  -> default True
      max_days_off: int    -> default 1
      line_position: str   -> 'center' or 'boundary'
      line_kwargs: dict
    """
    # 1) Organize (uses your existing organizer)
    od = build_organized_dataset_with_durations(
        decoded_database_json=str(cfg.decoded_database_json),
        creation_metadata_json=str(cfg.creation_metadata_json),
        only_song_present=kw.pop("only_song_present", False),
        compute_durations=False,  # durations not needed for count heatmap
    )
    organized_df = od.organized_df

    # 2) Build count table (avg per song per day)
    count_table = build_daily_avg_count_table(
        organized_df,
        label_column="syllable_onsets_offsets_ms_dict",
        date_column="Date",
        syllable_labels=od.unique_syllable_labels,  # stable order
    )

    # 3) Resolve animal_id & treatment_date
    try:
        animal_id = str(organized_df["Animal ID"].dropna().iloc[0])
    except Exception:
        animal_id = "unknown_animal"
    treatment_date = getattr(od, "treatment_date", None)

    # 4) Build save path from save_dir + animal_id
    save_dir = kw.pop("save_dir", None)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{animal_id}_daily_syllable_heatmap.png"
    else:
        save_path = None

    # 5) Plot (forward any remaining style knobs in **kw)
    fig, ax = plot_log_scaled_syllable_counts(
        count_table,
        animal_id=animal_id,
        treatment_date=treatment_date,
        show=kw.pop("show", False),
        save_path=save_path,
        **kw,
    )

    return {
        "animal_id": animal_id,
        "saved_path": str(save_path) if save_path else None,
        "n_days": count_table.shape[1],
        "n_labels": count_table.shape[0],
    }


STEP_REGISTRY = {
    "daily_first_order_transitions": step_daily_first_order_transitions,
    "phrase_duration_over_days":    step_phrase_duration_over_days,
    "transition_entropy_daily":     step_transition_entropy_daily,
    "syllable_heatmap":             step_syllable_heatmap,   # ← new
}


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    decoded_database_json: str | Path,
    creation_metadata_json: str | Path,
    outdir: str | Path,
    steps: List[str],
    step_opts: Optional[Dict[str, Dict[str, Any]]] = None,
    resume: bool = False,
    manifest_path: Optional[str | Path] = None,
) -> None:
    """
    Programmatic entrypoint.

    Parameters
    ----------
    decoded_database_json : str | Path
    creation_metadata_json : str | Path
    outdir : str | Path
        A general outputs folder (some steps also take their own output dirs).
    steps : list[str]
        Any subset (order matters): daily_first_order_transitions, phrase_duration_over_days, transition_entropy_daily
    step_opts : dict[str, dict]
        Per-step kwargs. Example:
            {
              "daily_first_order_transitions": {
                "output_dir": ".../figures/daily_transitions",
                "save_movie_path": ".../figures/daily_transitions/first_order_daily.gif",
                "movie_fps": 2,
                "enforce_consistent_order": True,
                "only_song_present": False,
                "min_row_total": 0
              },
              "phrase_duration_over_days": {
                "save_output_to_this_file_path": ".../figures",
                "only_song_present": False,
                "y_max_ms": 25000,
                "show_plots": False
              },
              "transition_entropy_daily": {
                "only_song_present": False,
                "compute_durations": False,
                "surgery_date_override": None
              }
            }
    resume : bool
        If True, skip steps already marked ok in the manifest.
    manifest_path : str | Path | None
        Where to write JSONL manifest. Default: under outdir with timestamped name.
    """
    outdir = Path(outdir)
    _safe_mkdir(outdir)
    manifest = Path(manifest_path) if manifest_path else _default_manifest(outdir)

    cfg = PipelineConfig(
        decoded_database_json=Path(decoded_database_json),
        creation_metadata_json=Path(creation_metadata_json),
        outdir=outdir,
        steps=steps,
        step_opts=step_opts or {},
        resume=resume,
        manifest_path=manifest,
    )

    completed = _load_completed(cfg.manifest_path) if cfg.resume else set()

    print(f"[syntax-pipeline] Steps: {steps}")
    print(f"[syntax-pipeline] Manifest: {cfg.manifest_path}")

    for step in steps:
        if step not in STEP_REGISTRY:
            _append_manifest(cfg.manifest_path, ManifestRecord(
                timestamp=datetime.now().isoformat(),
                step=step,
                status="error",
                details={"error": f"Unknown step: {step}", "available": list(STEP_REGISTRY.keys())},
            ))
            print(f"[syntax-pipeline] Unknown step: {step}")
            continue

        if cfg.resume and step in completed:
            _append_manifest(cfg.manifest_path, ManifestRecord(
                timestamp=datetime.now().isoformat(),
                step=step,
                status="skipped",
                details={"reason": "resume"},
            ))
            print(f"[syntax-pipeline] Skipping (resume): {step}")
            continue

        fn = STEP_REGISTRY[step]
        kw = cfg.step_opts.get(step, {})

        try:
            t0 = time.time()
            outputs = fn(cfg, **kw)
            dt = round(time.time() - t0, 3)
            _append_manifest(cfg.manifest_path, ManifestRecord(
                timestamp=datetime.now().isoformat(),
                step=step,
                status="ok",
                details={"elapsed_s": dt, "outputs": outputs},
            ))
            print(f"[syntax-pipeline] OK: {step} ({dt}s)")
        except Exception as e:
            _append_manifest(cfg.manifest_path, ManifestRecord(
                timestamp=datetime.now().isoformat(),
                step=step,
                status="error",
                details={"error": str(e)},
            ))
            raise

    print("[syntax-pipeline] Done.")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run birdsong syntax pipeline steps.")
    p.add_argument("decoded", type=str, help="Path to *_decoded_database.json")
    p.add_argument("meta", type=str, help="Path to *_creation_data.json")
    p.add_argument("--outdir", type=str, default="./syntax_outputs", help="General output folder")
    p.add_argument(
        "--steps", nargs="+", required=True,
        help=f"Steps to run (order matters). Available: {list(STEP_REGISTRY.keys())}"
    )
    p.add_argument(
        "--step-opts", type=str, default=None,
        help="JSON with per-step kwargs (see docstring)."
    )
    p.add_argument("--resume", action="store_true", help="Skip steps already completed (per manifest).")
    p.add_argument("--manifest", type=str, default=None, help="Path to JSONL manifest file.")
    return p.parse_args()

def main():
    ns = _parse_cli()
    step_opts = json.loads(ns.step_opts) if ns.step_opts else {}
    run_pipeline(
        decoded_database_json=ns.decoded,
        creation_metadata_json=ns.meta,
        outdir=ns.outdir,
        steps=ns.steps,
        step_opts=step_opts,
        resume=ns.resume,
        manifest_path=ns.manifest,
    )

if __name__ == "__main__":
    main()


"""
from wrapper_graph_daily_trends import run_pipeline

decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"
outdir  = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/figures"

run_pipeline(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    outdir=outdir,
    steps=[
        "daily_first_order_transitions",
        "phrase_duration_over_days",
        "transition_entropy_daily",
        "syllable_heatmap",                    # ← add here
    ],
    step_opts={
        "daily_first_order_transitions": {
            "output_dir": f"{outdir}/daily_transitions",
            "save_movie_path": f"{outdir}/daily_transitions/first_order_daily.gif",
            "movie_fps": 2,
            "enforce_consistent_order": True,
            "only_song_present": False,
            "save_csv": False,
            "save_png": True,
            "show_plots": False,
        },
        "phrase_duration_over_days": {
            "save_output_to_this_file_path": outdir,
            "only_song_present": False,
            "y_max_ms": 25000,
            "show_plots": False,
        },
        "transition_entropy_daily": {
            "only_song_present": False,
            "compute_durations": False,
            "surgery_date_override": None,
            "save_dir": outdir   # -> {animal_id}_transition_entropy_daily.png
        },
        "syllable_heatmap": {
            "save_dir": outdir,  # -> {animal_id}_daily_syllable_heatmap.png
            "pseudocount": 1e-3,
            "cmap": "Greys",
            "show": False,
            "mark_treatment": True,
            "nearest_match": True,
            "max_days_off": 1,
            "line_position": "center",
            # "line_kwargs": {"color": "red", "linestyle": "--", "linewidth": 2},
        },
    },
    resume=True,
)


"""