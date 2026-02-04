
"""
make_synthetic_outlier_graphs_testdata.py

Generate a fully synthetic dataset to validate outlier_graphs.py.

Creates:
  1) usage_balanced_phrase_duration_stats.csv
       Columns: Animal ID, Syllable, Group, Variance_ms2, N_phrases

  2) Area_X_lesion_metadata_with_hit_types.xlsx
       Sheet: "metadata_with_hit_type"
       Columns: Animal ID, Treatment type, Area X visible in histology?,
                Lesion hit type, Medial Area X hit?, Lateral Area X hit?

  3) histology_volumes_dir/**/_final_volumes.json  (nested)
       Includes L/R lesion percent keys (and alternate key formats)
       NOTE: Sham animals are set to EXACTLY 0% lesioned.

Negative-control option:
  - You can force ONE hit type to be NOT significantly different from sham
    under one-tailed "greater" tests by making its post/pre ratios ~0 (or even slightly negative).
  - Default here: "Area X visible (single hit)" is made a "null" group (slightly negative drift)
    so the one-tailed "greater" test vs sham should FAIL (p large).

Run:
  python make_synthetic_outlier_graphs_testdata.py

Or import and call:
  from make_synthetic_outlier_graphs_testdata import make_synthetic_outlier_graphs_dataset
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Dict, Any

import numpy as np
import pandas as pd


def make_synthetic_outlier_graphs_dataset(
    out_root: Path | str,
    *,
    seed: int = 7,
    n_syllables: int = 18,
    # Choose which lesion hit-type should be a negative control vs sham
    null_hit_type: str = "Area X visible (single hit)",
    # How to implement the null effect:
    #   "equal"  -> post == pre exactly (delta_log10_ratio = 0)
    #   "noise"  -> post = pre * 10**Normal(null_mean, null_sd)
    null_mode: str = "noise",
    null_log10_ratio_mean: float = -0.03,  # slightly negative helps one-tailed "greater" fail robustly
    null_log10_ratio_sd: float = 0.02,
    # If True, delete out_root first (careful!)
    overwrite_dir: bool = False,
) -> Dict[str, Path]:
    """
    Create synthetic data compatible with outlier_graphs.py.

    Key design:
      - Sham has ~no change (log10(post/pre) mean ~ 0).
      - Lesion groups have positive changes EXCEPT the chosen null_hit_type.
      - Sham lesion% in volumes JSON is set to 0.0% (realistic).

    Returns a dict of important output paths.
    """
    rng = np.random.default_rng(seed)
    out_root = Path(out_root)

    if overwrite_dir and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # Define animals per hit-type category
    # Ensure enough animals per group for t-tests
    # ─────────────────────────────────────────────────────────────
    sham_animals = ["USA7001", "USA7002", "USA7003"]
    vis_single_animals = ["USA7101", "USA7102", "USA7103"]  # Area X visible (single hit)
    vis_ml_animals = ["USA7201", "USA7202"]                 # Area X visible (medial+lateral hit)
    not_vis_animals = ["USA7301", "USA7302"]                # large lesion Area X not visible
    miss_animals = ["USA7401"]                              # extra category

    all_animals = sham_animals + vis_single_animals + vis_ml_animals + not_vis_animals + miss_animals

    # Canonical hit-type strings used by your plotting/comparisons
    hit_type_map: Dict[str, str] = {}
    for a in sham_animals:
        hit_type_map[a] = "sham saline injection"
    for a in vis_single_animals:
        hit_type_map[a] = "Area X visible (single hit)"
    for a in vis_ml_animals:
        hit_type_map[a] = "Area X visible (medial+lateral hit)"
    for a in not_vis_animals:
        hit_type_map[a] = "large lesion Area X not visible"
    for a in miss_animals:
        hit_type_map[a] = "miss"

    # Treatment types (used by _infer_treatment_group)
    treatment_type_map: Dict[str, str] = {}
    for a in sham_animals:
        treatment_type_map[a] = "sham saline injection"
    for a in (vis_single_animals + vis_ml_animals + not_vis_animals + miss_animals):
        treatment_type_map[a] = "NMA lesion"

    # Visibility (used by _is_visible)
    visible_map: Dict[str, str] = {}
    for a in sham_animals:
        visible_map[a] = "no"  # doesn't matter much for sham, but realistic
    for a in vis_single_animals + vis_ml_animals:
        visible_map[a] = "yes"
    for a in not_vis_animals:
        visible_map[a] = "not visible"
    for a in miss_animals:
        visible_map[a] = "no"

    # Medial/Lateral hit columns (used if hit-type missing; still included)
    medial_map: Dict[str, str] = {a: "no" for a in all_animals}
    lateral_map: Dict[str, str] = {a: "no" for a in all_animals}
    for a in vis_single_animals:
        medial_map[a] = "yes"
        lateral_map[a] = "no"
    for a in vis_ml_animals + not_vis_animals:
        medial_map[a] = "yes"
        lateral_map[a] = "yes"

    # ─────────────────────────────────────────────────────────────
    # Synthetic variances
    # ─────────────────────────────────────────────────────────────
    syllables = [str(i) for i in range(n_syllables)]

    # Means for log10(post/pre) per hit-type (lesion > sham), EXCEPT null_hit_type (overridden below)
    log10_ratio_means = {
        "sham saline injection": 0.00,                  # post ~ pre
        "Area X visible (single hit)": 0.20,            # ~1.58x
        "Area X visible (medial+lateral hit)": 0.25,    # ~1.78x
        "large lesion Area X not visible": 0.30,        # ~2.00x
        "miss": 0.05,
    }
    log10_ratio_sd = 0.08

    rows = []
    for animal in all_animals:
        hit_type = hit_type_map[animal]
        mu = float(log10_ratio_means.get(hit_type, 0.05))

        # Pre variance distribution (positive, broad)
        pre_vals = 10 ** rng.normal(loc=-1.0, scale=0.6, size=n_syllables)

        # Post variances:
        # - for the null hit-type: intentionally NOT greater than sham
        # - for others: ratio centered at their mu
        if hit_type == null_hit_type:
            if null_mode.lower() == "equal":
                post_vals = pre_vals.copy()
            else:
                # Slightly negative drift makes one-tailed "greater" vs sham reliably non-significant
                log10_ratio = rng.normal(loc=null_log10_ratio_mean, scale=null_log10_ratio_sd, size=n_syllables)
                post_vals = pre_vals * (10 ** log10_ratio)
        else:
            log10_ratio = rng.normal(loc=mu, scale=log10_ratio_sd, size=n_syllables)
            post_vals = pre_vals * (10 ** log10_ratio)

        # N_phrases, include some below threshold to exercise filtering
        n_phrases = rng.integers(low=3, high=30, size=n_syllables)

        for syl, pre_v, post_v, nph in zip(syllables, pre_vals, post_vals, n_phrases):
            rows.append(
                {"Animal ID": animal, "Syllable": syl, "Group": "Late Pre",
                 "Variance_ms2": float(pre_v), "N_phrases": int(nph)}
            )
            rows.append(
                {"Animal ID": animal, "Syllable": syl, "Group": "Post",
                 "Variance_ms2": float(post_v), "N_phrases": int(nph)}
            )

    csv_df = pd.DataFrame(rows)

    # Inject a couple of non-positive variance rows to test log_scale filtering in your plotting code
    bad_idx = csv_df.sample(2, random_state=seed).index
    csv_df.loc[bad_idx, "Variance_ms2"] = 0.0

    csv_path = out_root / "usage_balanced_phrase_duration_stats.csv"
    csv_df.to_csv(csv_path, index=False)

    # ─────────────────────────────────────────────────────────────
    # Metadata Excel (sheet: metadata_with_hit_type)
    # ─────────────────────────────────────────────────────────────
    meta_rows = []
    for animal in all_animals:
        meta_rows.append({
            "Animal ID": animal,
            "Treatment type": treatment_type_map[animal],
            "Area X visible in histology?": visible_map[animal],
            "Lesion hit type": hit_type_map[animal],
            "Medial Area X hit?": medial_map[animal],
            "Lateral Area X hit?": lateral_map[animal],
        })
    meta_df = pd.DataFrame(meta_rows)

    xlsx_path = out_root / "Area_X_lesion_metadata_with_hit_types.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        meta_df.to_excel(w, sheet_name="metadata_with_hit_type", index=False)

    # ─────────────────────────────────────────────────────────────
    # Histology volumes dir with nested *_final_volumes.json
    # Sham animals: EXACTLY 0% lesion
    # Visible animals: finite lesion% so continuous plot can render
    # ─────────────────────────────────────────────────────────────
    volumes_dir = out_root / "histology_volumes_dir"
    nested_dir = volumes_dir / "nested" / "more_nested"
    nested_dir.mkdir(parents=True, exist_ok=True)

    def write_vol(animal: str, left: float, right: float, *, nested: bool) -> None:
        obj: Dict[str, Any] = {
            "AnimalID": animal,
            "Some": {"Nested": {
                "L_Percent_of_Area_X_Lesioned_pct": float(left),
                "R_Percent_of_Area_X_Lesioned_pct": float(right),
            }},
            # Alternate key formats to test heuristic matching:
            "Left Percent of Area X Lesioned (%)": float(left),
            "Right Percent of Area X Lesioned (%)": float(right),
        }
        p = (nested_dir if nested else volumes_dir) / f"{animal}_final_volumes.json"
        p.write_text(json.dumps(obj, indent=2))

    # Visible animals (including the null-hit-type group if it's visible single-hit)
    for a in vis_single_animals:
        write_vol(a, left=float(rng.uniform(10, 40)), right=float(rng.uniform(10, 40)), nested=True)
    for a in vis_ml_animals:
        write_vol(a, left=float(rng.uniform(25, 60)), right=float(rng.uniform(25, 60)), nested=True)

    # Not-visible lesions (still can have lesion% values in JSON)
    for a in not_vis_animals:
        write_vol(a, left=float(rng.uniform(40, 80)), right=float(rng.uniform(40, 80)), nested=False)

    # Sham: 0% lesion (realistic)
    for a in sham_animals:
        write_vol(a, left=0.0, right=0.0, nested=False)

    # Miss: small values (optional)
    for a in miss_animals:
        write_vol(a, left=float(rng.uniform(0, 10)), right=float(rng.uniform(0, 10)), nested=False)

    # Add an AppleDouble file that should be ignored by your module
    apple_double = volumes_dir / "._USA9999_final_volumes.json"
    apple_double.write_text('{"this":"should be ignored"}')

    return {
        "csv_path": csv_path,
        "metadata_xlsx": xlsx_path,
        "histology_volumes_dir": volumes_dir,
        "out_root": out_root,
    }


if __name__ == "__main__":
    paths = make_synthetic_outlier_graphs_dataset(
        "./synthetic_outlier_graphs_testdata",
        seed=7,
        n_syllables=18,
        null_hit_type="Area X visible (single hit)",
        null_mode="noise",               # "equal" or "noise"
        null_log10_ratio_mean=-0.03,     # slightly negative drift
        null_log10_ratio_sd=0.02,
        overwrite_dir=False,             # set True to delete/rebuild the folder
    )
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")



"""
from pathlib import Path
import importlib

import make_synthetic_outlier_graphs_testdata as synth
importlib.reload(synth)

paths = synth.make_synthetic_outlier_graphs_dataset(
    out_root=Path("/Volumes/my_own_SSD/synthetic_outlier_graphs_testdata2"),
    seed=7,
    n_syllables=18,
)
print(paths)


"""