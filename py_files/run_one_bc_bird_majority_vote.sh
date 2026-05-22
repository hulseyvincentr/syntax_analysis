#!/usr/bin/env bash
set -u

NPZ="$1"
ANIMAL_ID="$(basename "$NPZ" .npz)"

DATA_ROOT="/Volumes/my_own_SSD/updated_AreaX_outputs"
OUT_ROOT="${DATA_ROOT}/bc_full_contiguous_MAIN_ALL_BIRDS_MAJORITY_VOTE_200bin_density20_grid99_title_spacing"
LOG_DIR="${OUT_ROOT}/_logs"

echo "[START] ${ANIMAL_ID}  NPZ=${NPZ}"

python bc_cluster_qc_and_summaries_v19_full_contiguous_majority_vote_smoothing.py \
  --v8-script export_equal_umap_cluster_spectrograms_v23_full_contiguous_majority_vote_smoothing_umap_title_spacing.py \
  --npz-path "$NPZ" \
  --metadata-excel-path "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --spectrogram-script "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files/pre_post_syllable_sample_spectrograms_long_rows_with_bouts_v7.py" \
  --phrase-csv "/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv" \
  --out-dir "$OUT_ROOT" \
  --animal-id "$ANIMAL_ID" \
  --period-mode early_late_pre_post \
  --early-late-split-method timebin_half \
  --bc-analysis-mode run_balanced_full_contiguous \
  --fill-noise-labels-from-nearest-nonnoise \
  --apply-majority-vote-label-smoothing \
  --majority-vote-window-bins 200 \
  --run-sample-mode random \
  --min-runs-per-group 20 \
  --max-runs-per-group 0 \
  --min-full-run-duration-ms 100 \
  --min-run-group-fraction 0.80 \
  --bc-grid-point-coverage 0.99 \
  --umap-density-bins 20 \
  --spectrogram-source-mode expanded_full_runs \
  --bins-per-spectrogram-row 2000 \
  --full-run-fixed-duration-s 5.4 \
  --fixed-panel-duration-s 5.4 \
  --min-balanced-duration-s 2.0 \
  --seed 0 \
  > "${LOG_DIR}/${ANIMAL_ID}.log" 2>&1

STATUS=$?
if [ "$STATUS" -eq 0 ]; then
  echo "[DONE] ${ANIMAL_ID}"
else
  echo "[FAILED] ${ANIMAL_ID} -- see log: ${LOG_DIR}/${ANIMAL_ID}.log"
fi

exit "$STATUS"
