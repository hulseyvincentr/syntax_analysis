#!/usr/bin/env bash
set -euo pipefail

PY_DIR="$HOME/Documents/allPythonCode/syntax_analysis/py_files"
ROOT="/Volumes/my_own_SSD/updated_AreaX_outputs"

cd "$PY_DIR"

WRAPPER="$PY_DIR/bc_cluster_qc_and_summaries_v19_full_contiguous_majority_vote_smoothing.py"
EXPORTER="$PY_DIR/export_equal_umap_cluster_spectrograms_v23_full_contiguous_majority_vote_smoothing_umap_title_spacing.py"
SPEC_SCRIPT="$PY_DIR/pre_post_syllable_sample_spectrograms_long_rows_with_bouts_v7.py"

METADATA="$ROOT/Area_X_lesion_metadata_with_hit_types.xlsx"
PHRASE_CSV="$ROOT/usage_balanced_phrase_duration_stats.csv"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$ROOT/bc_full_contiguous_PARALLEL_ALL21_MAJORITY_VOTE_200bin_density20_grid99_${RUN_TAG}"
LOG_DIR="$OUT_ROOT/_logs"

ORIGINAL="$ROOT/bc_full_contiguous_MAIN_ALL_BIRDS_MAJORITY_VOTE_200bin_density20_grid99_title_spacing"

if [[ "$OUT_ROOT" == "$ORIGINAL" ]]; then
    echo "[ABORT] New output path matches the original results folder."
    exit 1
fi

if [[ -e "$OUT_ROOT" ]]; then
    echo "[ABORT] Output folder already exists:"
    echo "$OUT_ROOT"
    exit 1
fi

for FILE in \
    "$WRAPPER" \
    "$EXPORTER" \
    "$SPEC_SCRIPT" \
    "$METADATA" \
    "$PHRASE_CSV"
do
    if [[ ! -f "$FILE" ]]; then
        echo "[ABORT] Required file is missing:"
        echo "$FILE"
        exit 1
    fi
done

mkdir -p "$LOG_DIR"

# Limit internal numerical-library thread pools. Parallelism is applied
# between birds instead.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Start with three concurrent birds. Change to 4 after checking that
# Activity Monitor shows green memory pressure.
N_JOBS="${N_JOBS:-3}"

BIRDS=(
    R08
    R09
    R10
    USA5271
    USA5283
    USA5288
    USA5325
    USA5326
    USA5336
    USA5337
    USA5347
    USA5371
    USA5443
    USA5468
    USA5483
    USA5494
    USA5499
    USA5505
    USA5506
    USA5509
    USA5510
)

run_bird() {
    local BIRD="$1"
    local NPZ="$ROOT/$BIRD/$BIRD.npz"
    local LOG="$LOG_DIR/$BIRD.log"

    echo
    echo "============================================================"
    echo "Running: $BIRD"
    echo "NPZ:     $NPZ"
    echo "Log:     $LOG"

    if [[ ! -f "$NPZ" ]]; then
        echo "[MISSING] $NPZ" | tee "$LOG"
        return 2
    fi

    if python "$WRAPPER" \
        --v8-script "$EXPORTER" \
        --npz-path "$NPZ" \
        --metadata-excel-path "$METADATA" \
        --spectrogram-script "$SPEC_SCRIPT" \
        --phrase-csv "$PHRASE_CSV" \
        --out-dir "$OUT_ROOT" \
        --animal-id "$BIRD" \
        --top-fraction 0.30 \
        --post-group-name "Post" \
        --top-min-n-phrases 100 \
        --period-mode early_late_pre_post \
        --treatment-day-assignment exclude \
        --early-late-split-method file_median \
        --bc-analysis-mode run_balanced_full_contiguous \
        --min-runs-per-group 20 \
        --max-runs-per-group 200 \
        --run-sample-mode random \
        --apply-majority-vote-label-smoothing \
        --majority-vote-window-bins 200 \
        --umap-density-bins 20 \
        --bc-grid-point-coverage 0.99 \
        --min-balanced-duration-s 2.0 \
        --bins-per-spectrogram-row 2000 \
        --spectrogram-source-mode expanded_full_runs \
        --full-run-fixed-duration-s 5.4 \
        --seconds-per-bin 0.0027 \
        --dpi 200 \
        --seed 0 \
        2>&1 | tee "$LOG"
    then
        echo "[DONE] $BIRD"
    else
        echo "[FAILED] $BIRD — inspect $LOG"
        return 1
    fi
}

export -f run_bird
export ROOT WRAPPER EXPORTER SPEC_SCRIPT
export METADATA PHRASE_CSV OUT_ROOT LOG_DIR

echo "============================================================"
echo "New output folder:"
echo "$OUT_ROOT"
echo
echo "Number of birds: ${#BIRDS[@]}"
echo "Parallel jobs:   $N_JOBS"
echo "============================================================"

printf '%s\n' "${BIRDS[@]}" |
    caffeinate -dimsu xargs -n 1 -P "$N_JOBS" bash -c 'run_bird "$1"' _

echo
echo "============================================================"
echo "Analysis finished."
echo "Results:"
echo "$OUT_ROOT"
echo "============================================================"

echo "$OUT_ROOT" > "$PY_DIR/LAST_FULL_CONTIGUOUS_BC_OUTPUT.txt"

echo
echo "Nonempty bird-level cluster summaries:"
find "$OUT_ROOT" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name "*_cluster_bc_summary.csv" \
    -size +0c |
    sort

echo
echo "Number of nonempty summaries:"
find "$OUT_ROOT" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name "*_cluster_bc_summary.csv" \
    -size +0c |
    wc -l
