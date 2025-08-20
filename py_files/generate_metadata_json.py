# ────────────────────────────────────────────────────────────────
# USER-DEFINED VALUES
# ────────────────────────────────────────────────────────────────
main_folder        = '/Volumes/my_own_SSD/baseline_recordings/USA5507_RC4'
animal_id          = 'USA5507'                 # Used for ID checking and convenience only
expected_animal_id = animal_id                 # Warn if files don't match this ID; set to None to disable
treatment_date     = None                      # None → JSON null
treatment_type     = "None (baseline only)"    # or None → JSON null

# Where to save output JSON file. Use None to save in main_folder.
output_dir         = '/Users/mirandahulsey-vincent/Desktop/metadata_jsons'

# You can mix single entries and ranges.
# Ranges support '-' or ':' plus 'start'/'end' keywords:
#   "94-111", "94:111", "start-111", "94:end", "start:end"
folder_year_map = {
    2024: ["start:13"]
    , 2025: ["147:end"] 
}

# ────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Set, Union, Tuple

# ────────────────────────────────────────────────────────────────
# Helpers for folder-range expansion
# ────────────────────────────────────────────────────────────────
def _numeric_subfolders(folder_path: Path) -> List[int]:
    """
    Return a sorted list of subfolder names that are pure integers, as ints.
    """
    nums = []
    for p in folder_path.iterdir():
        if p.is_dir():
            name = p.name.strip()
            if name.isdigit():
                nums.append(int(name))
    nums.sort()
    return nums

def _parse_bound(bound: str, min_exist: int, max_exist: int) -> Union[int, None]:
    """
    Convert a textual bound ('start', 'end', '123') into an integer.
    Returns None if it cannot be parsed.
    """
    b = bound.strip().lower()
    if b == "start":
        return min_exist
    if b == "end":
        return max_exist
    if b.isdigit():
        return int(b)
    return None

def _split_range(entry: str) -> Union[None, Tuple[str, str]]:
    """
    Split a range entry by either '-' or ':'.
    Accepts 'A-B' or 'A:B'. Returns (A, B) as strings (unparsed).
    Returns None if no valid separator is found.
    """
    if "-" in entry:
        parts = entry.split("-", 1)
    elif ":" in entry:
        parts = entry.split(":", 1)
    else:
        return None
    return parts[0].strip(), parts[1].strip()

def _expand_entry(entry: str, existing_nums: List[int]) -> Set[int]:
    """
    Expand an entry into a set of integers, intersected with existing folders.
    Supports:
      • single numbers: '94'
      • ranges with '-' or ':': '94-111', '94:111'
      • 'start'/'end' keywords: 'start-111', '94:end', 'start:end'
    """
    entry = entry.strip()
    if not existing_nums:
        return set()

    existing_set = set(existing_nums)
    min_exist, max_exist = existing_nums[0], existing_nums[-1]

    # Single number
    if _split_range(entry) is None:
        if entry.isdigit():
            n = int(entry)
            return {n} if n in existing_set else set()
        # Not a digit and not a range → nothing to expand
        return set()

    # Range (A-B or A:B), with 'start'/'end' allowed
    start_str, end_str = _split_range(entry)
    start = _parse_bound(start_str, min_exist, max_exist)
    end   = _parse_bound(end_str,   min_exist, max_exist)
    if start is None or end is None:
        return set()

    # Normalize and clamp
    if start > end:
        start, end = end, start
    start = max(start, min_exist)
    end   = min(end,   max_exist)
    if start > end:
        return set()

    return set(n for n in range(start, end + 1) if n in existing_set)

def expand_folder_ranges(folder_path: Union[str, Path],
                         folder_year_map: Dict[int, Iterable[str]]) -> Dict[str, int]:
    """
    Convert a map like:
        {2023: ["start-end"], 2024: ["94:111", "210"]}
    into a flat { "94": 2024, ..., "111": 2024, ... }.

    Only existing numeric subfolders are included. If overlapping
    ranges are provided for different years, the *first* assignment wins.
    """
    folder_path = Path(folder_path)
    existing_nums = _numeric_subfolders(folder_path)
    flat: Dict[str, int] = {}

    if not existing_nums:
        print("⚠️ No numeric subfolders found — range expansion will be empty.")

    for year, entries in folder_year_map.items():
        for raw_entry in entries:
            entry = raw_entry.strip()
            expanded = _expand_entry(entry, existing_nums)
            if not expanded:
                print(f"⚠️ Entry '{entry}' expanded to nothing (no matching existing folders).")
                continue

            for n in sorted(expanded):
                key = str(n)
                if key not in flat:
                    flat[key] = year
                # If overlapping year specs occur, first assignment wins.

    return flat

# ────────────────────────────────────────────────────────────────
# Filename parsing helpers
# ────────────────────────────────────────────────────────────────
def extract_animal_id_from_filename(filename: str) -> str:
    """
    Assumes animal ID is the first underscore-separated token in the base name.
    e.g., 'USA5483_45355.32_3_4_9_0_38.wav' -> 'USA5483'
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    return parts[0] if parts else ""

def extract_date_time_from_filename(filename: str, year: int):
    """
    Expect filenames like: ..._<month>_<day>_<hour>_<minute>_<second>.wav
    Year is provided externally (from the folder-to-year map).
    """
    name, _ = os.path.splitext(filename)
    parts = name.split("_")

    if len(parts) < 7:
        return None

    try:
        month  = int(parts[-5])
        day    = int(parts[-4])
        hour   = int(parts[-3])
        minute = int(parts[-2])
        second = int(parts[-1])
        date = datetime(year, month, day)
        time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        return date.strftime("%Y-%m-%d"), time_str
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

# ────────────────────────────────────────────────────────────────
# Build JSON metadata structure (+ animal ID warnings)
# ────────────────────────────────────────────────────────────────
def build_json_structure(folder_path: Union[str, Path],
                         folder_to_year_map: Dict[str, int],
                         treatment_date,
                         treatment_type,
                         expected_animal_id: Union[str, None] = None):
    """
    Scans selected subfolders for .wav files, extracts unique creation dates
    from filenames (using provided year per subfolder), and returns a metadata dict.

    Prints warnings if animal IDs in filenames differ from expected_animal_id.
    """
    folder_path = Path(folder_path)

    output = {
        "treatment_date": treatment_date if treatment_date else None,
        "treatment_type": treatment_type if treatment_type else None,
        "subdirectories": {}
    }

    all_seen_ids: Set[str] = set()
    mismatched_examples: Dict[str, List[str]] = {}  # animal_id -> sample files
    MAX_EXAMPLES = 5

    for subfolder in sorted(folder_path.iterdir(), key=lambda p: p.name):
        if not subfolder.is_dir():
            continue

        subfolder_name = subfolder.name.strip()
        year = folder_to_year_map.get(subfolder_name)
        if year is None:
            continue  # not selected by range map

        creation_dates = set()

        for wav_file in subfolder.glob("*.wav"):
            file_animal_id = extract_animal_id_from_filename(wav_file.name)
            if file_animal_id:
                all_seen_ids.add(file_animal_id)

                if expected_animal_id and file_animal_id != expected_animal_id:
                    lst = mismatched_examples.setdefault(file_animal_id, [])
                    if len(lst) < MAX_EXAMPLES:
                        lst.append(str(wav_file))

            result = extract_date_time_from_filename(wav_file.name, year)
            if result:
                date_str, _ = result
                creation_dates.add(date_str)

        if creation_dates:
            earliest_date = sorted(creation_dates)[0]
            output["subdirectories"][subfolder_name] = {
                "subdirectory_creation_date": earliest_date,
                "unique_file_creation_dates": sorted(creation_dates)
            }

    # Animal ID summary / warnings
    if expected_animal_id:
        if not all_seen_ids:
            print("⚠️ No .wav files found — animal ID check skipped.")
        else:
            if all(id_ == expected_animal_id for id_ in all_seen_ids):
                print(f"✅ All files match expected animal ID: {expected_animal_id}")
            else:
                print("⚠️ Detected files from different animal IDs than expected:")
                print(f"   Expected: {expected_animal_id}")
                print(f"   Found IDs: {sorted(all_seen_ids)}")
                for bad_id, examples in mismatched_examples.items():
                    print(f"   • Mismatch ID '{bad_id}' — {len(examples)} example(s):")
                    for ex in examples:
                        print(f"       - {ex}")

    return output

# ────────────────────────────────────────────────────────────────
# RUN & SAVE
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Expand folder ranges (using existing numeric subfolders)
    flattened_map = expand_folder_ranges(main_folder, folder_year_map)

    # Build JSON metadata
    result_json = build_json_structure(
        main_folder,
        flattened_map,
        treatment_date,
        treatment_type,
        expected_animal_id=expected_animal_id
    )

    # Decide output location
    if output_dir:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)  # create if needed
    else:
        save_path = Path(main_folder)

    # Use the name of main_folder for output file (e.g., 'USA5177_Oct2023_metadata.json')
    folder_name = Path(main_folder).name
    output_file = save_path / f"{folder_name}_metadata.json"

    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(result_json, f, indent=4)

    print(f"\n✅ Saved metadata JSON to: {output_file}")
    print(f"   Subfolders mapped: {len(flattened_map)}")
    if not result_json.get("subdirectories"):
        print("⚠️ No subdirectories produced any dates — check your ranges and filename format.")
