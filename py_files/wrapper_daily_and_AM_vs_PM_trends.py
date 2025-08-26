# -*- coding: utf-8 -*-

# wrapper_graph_daily_and_AM_vs_PM_trends.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import json
import argparse
from datetime import datetime

# Import your two existing wrappers
from wrapper_graph_daily_trends import run_pipeline as run_daily_pipeline
from wrapper_graph_AM_vs_PM_trends import run_am_pm_pipeline as run_ampm_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_animal_id_from_decoded(decoded_database_json: str | Path) -> str:
    """Extract the animal ID from the decoded JSON database, with robust fallbacks."""
    decoded_path = Path(decoded_database_json)
    try:
        with decoded_path.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if data.get("animal_id"):
                return str(data["animal_id"])
            if data.get("Animal ID"):
                return str(data["Animal ID"])
    except Exception:
        pass
    # Fallback: use filename stem up to first underscore (e.g., USA5288_decoded_database -> USA5288)
    return decoded_path.stem.split("_")[0]


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_json_or_path(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """Accept either inline JSON or a path to a JSON file."""
    if not s:
        return None
    p = Path(s)
    if p.suffix.lower() in {".json", ".jsonl"} and p.exists():
        with p.open("r") as f:
            return json.load(f)
    return json.loads(s)


# ──────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def run_daily_and_am_pm_pipeline(
    *,
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Union[str, Path],
    outdir: Union[str, Path],

    # Which groups to run
    run_daily: bool = True,
    run_ampm: bool = True,

    # DAILY controls (passthrough to wrapper_graph_daily_trends.run_pipeline)
    daily_steps: Optional[Sequence[str]] = None,
    daily_step_opts: Optional[Dict[str, Dict[str, Any]]] = None,
    daily_resume: bool = True,
    daily_manifest: Optional[Union[str, Path]] = None,

    # AM/PM controls (passthrough to wrapper_graph_AM_vs_PM_trends.run_am_pm_pipeline)
    ampm_steps: Optional[Sequence[str]] = None,
    ampm_step_opts: Optional[Dict[str, Dict[str, Any]]] = None,
    ampm_resume: bool = True,
) -> Dict[str, Any]:
    """
    Run BOTH the 'daily' pipeline and the 'AM/PM' pipeline with one call.

    Parameters
    ----------
    decoded_database_json, creation_metadata_json, outdir : paths
        Standard inputs.
    run_daily, run_ampm : bool
        Toggle either half of the pipeline.
    daily_steps : list[str] | None
        Steps for wrapper_graph_daily_trends.run_pipeline.
        Default: ["daily_first_order_transitions", "phrase_duration_over_days",
                  "transition_entropy_daily", "syllable_heatmap"]
    daily_step_opts : dict[str, dict] | None
        Per-step kwargs for the daily half.
    daily_resume : bool
        Resume behavior for daily half (delegated).
    daily_manifest : str|Path|None
        Optional JSONL manifest path for the daily half.
    ampm_steps : list[str] | None
        Steps for wrapper_graph_AM_vs_PM_trends.run_am_pm_pipeline.
        Default: ["am_pm_syllable_heatmap", "am_pm_transition_entropy",
                  "am_pm_phrase_duration", "am_pm_transitions_movie"]
    ampm_step_opts : dict[str, dict] | None
        Per-step kwargs for the AM/PM half.
    ampm_resume : bool
        Resume behavior for AM/PM half.

    Returns
    -------
    dict:
        {
          "daily": {...results or "skipped"...},
          "am_pm": {...results or "skipped"...}
        }
    """
    outdir = _ensure_dir(outdir)
    decoded = Path(decoded_database_json)
    meta = Path(creation_metadata_json)

    results: Dict[str, Any] = {"daily": None, "am_pm": None}

    # Defaults
    if daily_steps is None:
        daily_steps = [
            "daily_first_order_transitions",
            "phrase_duration_over_days",
            "transition_entropy_daily",
            "syllable_heatmap",
        ]
    if ampm_steps is None:
        ampm_steps = [
            "am_pm_syllable_heatmap",
            "am_pm_transition_entropy",
            "am_pm_phrase_duration",
            "am_pm_transitions_movie",
        ]

    daily_step_opts = daily_step_opts or {}
    ampm_step_opts = ampm_step_opts or {}

    # Run DAILY pipeline
    if run_daily:
        if daily_manifest is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            daily_manifest = outdir / f"daily_pipeline_manifest_{ts}.jsonl"

        run_daily_pipeline(
            decoded_database_json=str(decoded),
            creation_metadata_json=str(meta),
            outdir=str(outdir),
            steps=list(daily_steps),
            step_opts=daily_step_opts,
            resume=daily_resume,
            manifest_path=str(daily_manifest),
        )
        results["daily"] = {
            "steps": list(daily_steps),
            "manifest": str(daily_manifest),
            "outdir": str(outdir),
            "status": "ok",
        }
    else:
        results["daily"] = {"status": "skipped"}

    # Run AM/PM pipeline
    if run_ampm:
        ampm_out = run_ampm_pipeline(
            decoded_database_json=str(decoded),
            creation_metadata_json=str(meta),
            outdir=str(outdir),
            steps=list(ampm_steps),
            step_opts=ampm_step_opts,
            resume=ampm_resume,
        )
        results["am_pm"] = {
            "steps": list(ampm_steps),
            "outputs": ampm_out,  # returns a dict of step results
            "outdir": str(outdir),
            "status": "ok",
        }
    else:
        results["am_pm"] = {"status": "skipped"}

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run both DAILY and AM/PM birdsong syntax pipelines."
    )
    p.add_argument("decoded", type=str, help="Path to *_decoded_database.json")
    p.add_argument("meta", type=str, help="Path to *_creation_data.json")
    p.add_argument("--outdir", type=str, default="./syntax_outputs", help="General output folder")

    # Which halves to run
    p.add_argument("--no-daily", action="store_true", help="Skip the DAILY pipeline")
    p.add_argument("--no-ampm", action="store_true", help="Skip the AM/PM pipeline")

    # Step lists (optional). If omitted, defaults will be used.
    p.add_argument("--daily-steps", nargs="*", default=None,
                   help="Daily steps to run (space-separated)")
    p.add_argument("--ampm-steps", nargs="*", default=None,
                   help="AM/PM steps to run (space-separated)")

    # Step options as JSON files or inline JSON
    p.add_argument("--daily-step-opts", type=str, default=None,
                   help="JSON string or path to JSON with kwargs for DAILY steps")
    p.add_argument("--ampm-step-opts", type=str, default=None,
                   help="JSON string or path to JSON with kwargs for AM/PM steps")

    # Resume + manifest
    p.add_argument("--daily-resume", action="store_true", help="Resume DAILY pipeline (skip completed)")
    p.add_argument("--ampm-resume", action="store_true", help="Resume AM/PM pipeline (skip existing)")
    p.add_argument("--daily-manifest", type=str, default=None, help="Path to DAILY manifest JSONL")
    return p.parse_args()


def main():
    ns = _parse_cli()
    daily_opts = _load_json_or_path(ns.daily_step_opts) or {}
    ampm_opts = _load_json_or_path(ns.ampm_step_opts) or {}

    res = run_daily_and_am_pm_pipeline(
        decoded_database_json=ns.decoded,
        creation_metadata_json=ns.meta,
        outdir=ns.outdir,
        run_daily=(not ns.no_daily),
        run_ampm=(not ns.no_ampm),
        daily_steps=ns.daily_steps,
        daily_step_opts=daily_opts,
        daily_resume=ns.daily_resume,
        daily_manifest=ns.daily_manifest,
        ampm_steps=ns.ampm_steps,
        ampm_step_opts=ampm_opts,
        ampm_resume=ns.ampm_resume,
    )
    print("\n[combined-wrapper] Summary:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

    
"""
from wrapper_daily_and_AM_vs_PM_trends import run_daily_and_am_pm_pipeline

decoded = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/data_inputs/Area_X_lesions_balanced_training_data/USA5288_creation_data.json"
outdir  = "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/figures"

animal_id = get_animal_id_from_decoded(decoded)

ampm_step_opts = {
    "am_pm_syllable_heatmap": {
        "normalize": "proportion",
        "log_scale": True,
        "save_path": f"{outdir}/am_pm/{animal_id}_am_pm.png",
        "show": False,
    },
    "am_pm_transition_entropy": {
        "only_song_present": False,
        "compute_durations": False,
        "surgery_date_override": None,
        "save_path": f"{outdir}/am_pm/{animal_id}_am_pm_entropy.png",
        "show": False,
    },
    "am_pm_phrase_duration": {
        "save_output_to_this_file_path": f"{outdir}/am_pm_phrase_duration",
        "only_song_present": False,
        "y_max_ms": 25000,
        "show_plots": False,
    },
    "am_pm_transitions_movie": {
        "output_dir": f"{outdir}/am_pm_transition_matrices",
        "save_movie_both_path": f"{outdir}/daily_transitions_am_pm/{animal_id}_transition_matrix_am_pm_per_day.gif",
        "save_movie_am_path": f"{outdir}/daily_transitions_am_pm/{animal_id}_transition_matrix_am_only.gif",
        "save_movie_pm_path": f"{outdir}/daily_transitions_am_pm/{animal_id}_transition_matrix_pm_only.gif",
        "movie_fps": 2,
        "enforce_consistent_order": True,
        "save_csv": False,
        "save_png": True,
        "show_plots": False,
    },
}

    

    
"""
