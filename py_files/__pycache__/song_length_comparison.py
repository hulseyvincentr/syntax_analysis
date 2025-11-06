# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Union, List, Dict, Any
import pandas as pd

def _ensure_list(obj):
    if obj is None:
        return None
    return obj if isinstance(obj, list) else [obj]

def build_meta_dataframe(json_input: Union[str, Path, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Create a DataFrame with columns:
      - filename (str)
      - song_present (bool)
      - spec_parameters (dict or None)
      - segments (list of dicts or None)

    json_input can be:
      - a Path/str to a .json file,
      - a JSON string (starts with '[' or '{'),
      - or a Python list of dicts (already-loaded JSON).
    """
    # Load the records
    if isinstance(json_input, (str, Path)):
        json_input = str(json_input)
        if json_input.strip().startswith("[") or json_input.strip().startswith("{"):
            records = json.loads(json_input)
        else:
            with open(json_input, "r") as f:
                records = json.load(f)
    elif isinstance(json_input, list):
        records = json_input
    else:
        raise TypeError("json_input must be a path, JSON string, or list of dicts")

    rows = []
    for rec in records:
        filename        = rec.get("filename")
        song_present    = bool(rec.get("song_present", False))
        spec_parameters = rec.get("spec_parameters")  # keep as dict or None
        segments        = rec.get("segments")         # keep as list or None

        # Normalize segments if a single dict sneaks in
        if segments is not None and not isinstance(segments, list):
            segments = _ensure_list(segments)

        rows.append({
            "filename": filename,
            "song_present": song_present,
            "spec_parameters": spec_parameters,
            "segments": segments
        })

    # Ensure the columns exist even if all values are None/missing
    df = pd.DataFrame(rows, columns=["filename", "song_present", "spec_parameters", "segments"])
    return df


"""
from pathlib import Path
from song_length_comparison import build_meta_dataframe

json_path = Path("/Volumes/my_own_ssd/2025_areax_lesion/R08_RC6_Comp2_song_detection.json")
df = build_meta_dataframe(json_path)
print(df[["filename", "song_present", "spec_parameters", "segments"]].head())
"""
