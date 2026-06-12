#!/usr/bin/env python3
"""
Patch phrase_position_acoustic_crosscorr_v6_majority_vote_runs.py so it also computes
spectrogram cross-correlation to the Nth previous segmented syllable in the same phrase.

Default lag = 10, so segment 11 is compared to segment 1, segment 12 to segment 2, etc.
Segments with fewer than N previous occurrences get NaN.

This patch creates a new script; it does not overwrite the original unless you choose the same output path.
"""
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


def insert_once(text: str, needle: str, insertion: str, description: str) -> str:
    if insertion.strip() in text:
        print(f"[INFO] {description}: already present")
        return text
    if needle not in text:
        raise RuntimeError(f"Could not find insertion point for: {description}\nNeedle was:\n{needle}")
    print(f"[PATCH] {description}")
    return text.replace(needle, needle + insertion, 1)


def replace_once(text: str, old: str, new: str, description: str) -> str:
    if new.strip() in text:
        print(f"[INFO] {description}: already present")
        return text
    if old not in text:
        raise RuntimeError(f"Could not find replacement point for: {description}\nOld text was:\n{old}")
    print(f"[PATCH] {description}")
    return text.replace(old, new, 1)


def patch_script(text: str) -> str:
    # 1) Add feature labels for the default lag-10 columns.
    feature_needle = '    "distance_from_phrase_early_template": "Spectrogram distance from early-phrase template (1 - corr)",\n'
    feature_insert = (
        '    "corr_to_lag10_previous_syllable": "Spectrogram corr. to syllable 10 repeats earlier",\n'
        '    "distance_from_lag10_previous_syllable": "Spectrogram distance from syllable 10 repeats earlier (1 - corr)",\n'
    )
    text = insert_once(text, feature_needle, feature_insert, "add lag-10 feature labels")

    # 2) Add a command-line argument. It is comma-separated so you can later run 5,10,20 if useful.
    arg_needle = (
        '    parser.add_argument("--min-phrase-template-segments", type=int, default=2,\n'
        '                        help="Minimum number of early phrase segments needed to build a template.")\n'
    )
    arg_insert = (
        '    parser.add_argument("--lag-crosscorr-steps", default="10",\n'
        '                        help=("Comma-separated previous-syllable lags for intra-phrase cross-correlation. "\n'
        '                              "Default 10 means compare each syllable to the syllable 10 occurrences earlier "\n'
        '                              "within the same repeated phrase."))\n'
    )
    text = insert_once(text, arg_needle, arg_insert, "add --lag-crosscorr-steps argument")

    # 3) Parse the lag list after argparse.
    parse_needle = '    args = parser.parse_args()\n'
    parse_insert = (
        '    args.lag_crosscorr_steps = [int(x.strip()) for x in str(args.lag_crosscorr_steps).split(",") if x.strip()]\n'
        '    if len(args.lag_crosscorr_steps) == 0:\n'
        '        args.lag_crosscorr_steps = [10]\n'
        '    if any(x <= 0 for x in args.lag_crosscorr_steps):\n'
        '        raise ValueError("--lag-crosscorr-steps must contain positive integers, e.g. 10 or 5,10")\n'
    )
    text = insert_once(text, parse_needle, parse_insert, "parse lag-crosscorr list")

    # 4) Compute lag correlations inside the per-segment loop, using the already prepared corr_vec entries.
    compute_needle = (
        '                corr_to_template = corr_between_unit_vectors(info.get("corr_vec"), phrase_template_vec)\n'
        '                distance_from_template = 1.0 - corr_to_template if np.isfinite(corr_to_template) else np.nan\n'
    )
    compute_insert = (
        '                lag_corr_values = {}\n'
        '                for lag in getattr(args, "lag_crosscorr_steps", [10]):\n'
        '                    lag = int(lag)\n'
        '                    # seg_order is 1-based; list index is 0-based.\n'
        '                    lag_idx = seg_order - lag - 1\n'
        '                    prev_vec = seg_infos[lag_idx].get("corr_vec") if 0 <= lag_idx < len(seg_infos) else None\n'
        '                    lag_corr = corr_between_unit_vectors(info.get("corr_vec"), prev_vec)\n'
        '                    lag_corr_values[f"corr_to_lag{lag}_previous_syllable"] = lag_corr\n'
        '                    lag_corr_values[f"distance_from_lag{lag}_previous_syllable"] = (\n'
        '                        1.0 - lag_corr if np.isfinite(lag_corr) else np.nan\n'
        '                    )\n'
        '                    lag_corr_values[f"lag{lag}_reference_repeat_index_in_phrase"] = (\n'
        '                        lag_idx + 1 if prev_vec is not None else np.nan\n'
        '                    )\n'
    )
    text = insert_once(text, compute_needle, compute_insert, "compute lagged intra-phrase correlations")

    # 5) Add those dynamically generated columns to each output row.
    row_needle = '                    "distance_from_phrase_early_template": distance_from_template,\n'
    row_insert = '                    **lag_corr_values,\n'
    text = insert_once(text, row_needle, row_insert, "write lagged cross-correlation columns to CSV")

    # 6) Update the docstring/notes in a minimal way.
    doc_needle = '6) optionally computes spectrogram cross-correlation to an early-phrase template and tests whether similarity declines with repeated syllables.\n'
    doc_insert = '7) also computes intra-phrase cross-correlation to the Nth previous syllable, default N=10.\n'
    text = insert_once(text, doc_needle, doc_insert, "update module description")

    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a lag-10 version of the phrase-position cross-correlation script.")
    parser.add_argument("--input-script", default="phrase_position_acoustic_crosscorr_v6_majority_vote_runs.py",
                        help="Path to the existing v6 script to patch.")
    parser.add_argument("--output-script", default="phrase_position_acoustic_crosscorr_v10_lag10_previous.py",
                        help="Path for the patched output script.")
    args = parser.parse_args()

    in_path = Path(args.input_script).expanduser().resolve()
    out_path = Path(args.output_script).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input script not found: {in_path}")

    text = in_path.read_text()
    patched = patch_script(text)
    out_path.write_text(patched)
    py_compile.compile(str(out_path), doraise=True)
    print(f"[SAVED] {out_path}")
    print("[DONE] Patched script compiles successfully.")
    print("New columns include:")
    print("  corr_to_lag10_previous_syllable")
    print("  distance_from_lag10_previous_syllable")
    print("  lag10_reference_repeat_index_in_phrase")


if __name__ == "__main__":
    main()
