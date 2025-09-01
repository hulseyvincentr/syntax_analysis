# -*- coding: utf-8 -*-
# wrapper_daily_and_AM_vs_PM_trends.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import json
import argparse
from datetime import datetime
import copy

# Import your two existing wrappers
from wrapper_graph_daily_trends import run_pipeline as run_daily_pipeline
from wrapper_graph_AM_vs_PM_trends import run_am_pm_pipeline as run_ampm_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_animal_id_from_decoded(decoded_database_json: str | Path) -> str:
    """
    Extract animal_id from the decoded JSON if present; otherwise fallback heuristics:
      1) top-level keys: 'animal_id' or 'Animal ID'
      2) first results[*].file_name -> prefix before first underscore
      3) filename stem up to first underscore
    """
    decoded_path = Path(decoded_database_json)
    try:
        with decoded_path.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # direct top-level keys
            for k in ("animal_id", "Animal ID"):
                v = data.get(k, None)
                if isinstance(v, (str, int)) and str(v).strip():
                    return str(v)

            # parsed from first result file_name if available
            results = data.get("results", [])
            if isinstance(results, list) and results:
                fn = results[0].get("file_name") or results[0].get("filename") or results[0].get("path")
                if isinstance(fn, str) and "_" in fn:
                    return fn.split("_")[0]
    except Exception:
        pass

    # Fallback: filename stem up to first underscore
    stem = decoded_path.stem
    return stem.split("_")[0] if "_" in stem else stem


def _replace_animal_id_tokens(obj: Any, animal_id: str) -> Any:
    """
    Recursively replace '{animal_id}' in any str values within nested dict/list structures.
    Non-str values are returned unchanged.
    """
    if isinstance(obj, dict):
        return {k: _replace_animal_id_tokens(v, animal_id) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_animal_id_tokens(v, animal_id) for v in obj]
    if isinstance(obj, str):
        return obj.replace("{animal_id}", animal_id)
    return obj


def _load_json_or_path(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """Accepts either an inline JSON string or a path to a JSON file."""
    if not s:
        return None
    p = Path(s)
    if p.suffix.lower() in {".json", ".jsonl"} and p.exists():
        with p.open("r") as f:
            return json.load(f)
    return json.loads(s)


# ──────────────────────────────────────────────────────────────────────────────
# Big wrapper
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
    Orchestrate BOTH your 'daily' pipeline and your 'AM/PM' pipeline.

    - Any '{animal_id}' tokens inside daily_step_opts and ampm_step_opts are
      auto-replaced using the decoded JSON contents (fallback to filename stem).
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

    # Options dicts
    daily_step_opts = daily_step_opts or {}
    ampm_step_opts = ampm_step_opts or {}

    # Resolve animal_id and hydrate any {animal_id} tokens inside BOTH opts dicts
    animal_id = get_animal_id_from_decoded(decoded)
    hydrated_daily_opts = _replace_animal_id_tokens(copy.deepcopy(daily_step_opts), animal_id)
    hydrated_ampm_opts = _replace_animal_id_tokens(copy.deepcopy(ampm_step_opts), animal_id)

    # DAILY half
    if run_daily:
        if daily_manifest is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            daily_manifest = outdir / f"daily_pipeline_manifest_{ts}.jsonl"

        run_daily_pipeline(
            decoded_database_json=str(decoded),
            creation_metadata_json=str(meta),
            outdir=str(outdir),
            steps=list(daily_steps),
            step_opts=hydrated_daily_opts,
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

    # AM/PM half
    if run_ampm:
        ampm_out = run_ampm_pipeline(
            decoded_database_json=str(decoded),
            creation_metadata_json=str(meta),
            outdir=str(outdir),
            steps=list(ampm_steps),
            step_opts=hydrated_ampm_opts,
            resume=ampm_resume,
        )
        results["am_pm"] = {
            "steps": list(ampm_steps),
            "outputs": ampm_out,
            "outdir": str(outdir),
            "animal_id": animal_id,
            "status": "ok",
        }
    else:
        results["am_pm"] = {"status": "skipped", "animal_id": animal_id}

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run both DAILY and AM/PM birdsong syntax pipelines from one command."
    )
    p.add_argument("decoded", type=str, help="Path to *_decoded_database.json")
    p.add_argument("meta", type=str, help="Path to *_creation_data.json")
    p.add_argument("--outdir", type=str, default="./syntax_outputs", help="General output folder")

    # Which halves to run
    p.add_argument("--no-daily", action="store_true", help="Skip the DAILY pipeline")
    p.add_argument("--no-ampm", action="store_true", help="Skip the AM/PM pipeline")

    # Step lists (optional). If omitted, defaults will be used.
    p.add_argument("--daily-steps", nargs="*", default=None,
                   help="Daily steps to run, e.g. daily_first_order_transitions phrase_duration_over_days")
    p.add_argument("--ampm-steps", nargs="*", default=None,
                   help="AM/PM steps to run, e.g. am_pm_syllable_heatmap am_pm_transitions_movie")

    # Step options as JSON files or inline JSON
    p.add_argument("--daily-step-opts", type=str, default=None,
                   help="JSON string or path to JSON file with kwargs for DAILY steps")
    p.add_argument("--ampm-step-opts", type=str, default=None,
                   help="JSON string or path to JSON file with kwargs for AM/PM steps")

    # Resume controls
    p.add_argument("--daily-resume", action="store_true", help="Resume DAILY pipeline (skip completed)")
    p.add_argument("--ampm-resume", action="store_true", help="Resume AM/PM pipeline (skip existing)")

    # Optional manifest path for daily half
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

decoded = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5272/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5272_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5272/USA5272_metadata.json"
outdir  = "/Users/mirandahulsey-vincent/Desktop/analysis_results/USA5272/new_figures"

ampm_step_opts = {
    "am_pm_syllable_heatmap": {
        "normalize": "proportion",
        "log_scale": True,
        "save_path": f"{outdir}/am_pm/{{animal_id}}_am_pm.png",  # file
        "show": False,
    },
    "am_pm_transition_entropy": {
        "only_song_present": False,
        "compute_durations": False,
        "surgery_date_override": None,
        "save_path": f"{outdir}/am_pm/{{animal_id}}_am_pm_entropy.png",  # file
        "show": False,
    },
    "am_pm_phrase_duration": {
        "save_output_to_this_file_path": f"{outdir}/am_pm_phrase_duration",  # dir
        "only_song_present": False,
        "y_max_ms": 25000,
        "show_plots": False,
    },
    "am_pm_transitions_movie": {
        "output_dir": f"{outdir}/am_pm_transition_matrices",  # dir for per-day PNG/CSV if enabled
        "save_movie_both_path": f"{outdir}/daily_transitions_am_pm/{{animal_id}}_transition_matrix_am_pm_per_day.gif",
        "save_movie_am_path":   f"{outdir}/daily_transitions_am_pm/{{animal_id}}_transition_matrix_am_only.gif",
        "save_movie_pm_path":   f"{outdir}/daily_transitions_am_pm/{{animal_id}}_transition_matrix_pm_only.gif",
        "movie_fps": 1,
        "enforce_consistent_order": True,
        "save_csv": False,
        "save_png": True,       # turn on if you want per-day PNGs too
        "show_plots": False,
    },
}

daily_step_opts = {
    "daily_first_order_transitions": {
        "output_dir": f"{outdir}/daily_transitions",  # dir
        "save_movie_path": f"{outdir}/daily_transitions/first_order_daily.gif",  # file
        "movie_fps": 1,
        "enforce_consistent_order": True,
        "save_csv": False,
        "save_png": True,        # <= ensure PNGs are written if you want them
        "show_plots": False,
    },
    "phrase_duration_over_days": {
        "save_output_to_this_file_path": f"{outdir}/daily_phrase_duration",  # dir
        "only_song_present": False,
        "y_max_ms": 25000,
        "show_plots": False,
    },
    "transition_entropy_daily": {
        "only_song_present": False,
        "compute_durations": False,
        "surgery_date_override": None,
        "save_dir": outdir,  # dir -> {animal_id}_transition_entropy_daily.png
    },
    "syllable_heatmap": {
        "save_dir": outdir,   # dir -> {animal_id}_daily_syllable_heatmap.png
        "pseudocount": 1e-3,
        "cmap": "Greys",
        "show": False,
        "mark_treatment": True,
        "nearest_match": True,
        "max_days_off": 1,
        "line_position": "center",
    },
}

res = run_daily_and_am_pm_pipeline(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    outdir=outdir,
    run_daily=True,
    run_ampm=True,
    daily_step_opts=daily_step_opts,
    ampm_step_opts=ampm_step_opts,
    daily_resume=False,   # force-run even if a manifest exists
    ampm_resume=False,    # force-run AM/PM too
)

print(res)



"""