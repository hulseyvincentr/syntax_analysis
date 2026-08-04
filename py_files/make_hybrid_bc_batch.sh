#!/usr/bin/env bash
set -euo pipefail

PY_DIR="$HOME/Documents/allPythonCode/syntax_analysis/py_files"
ROOT="/Volumes/my_own_SSD/updated_AreaX_outputs"

OLD_ROOT="$ROOT/bc_full_contiguous_MAIN_ALL_BIRDS_MAJORITY_VOTE_200bin_density20_grid99_title_spacing"

NEW_ROOT="$ROOT/bc_full_contiguous_PARALLEL_ALL21_MAJORITY_VOTE_200bin_density20_grid99_20260727_132604"

METADATA="$ROOT/Area_X_lesion_metadata_with_hit_types.xlsx"

BATCH_SCRIPT="$PY_DIR/bc_batch_lesion_group_summary_with_remaining_v8_dynamic_ylims_whisker_brackets_ML_combined.py"

COLOR_JSON="$PY_DIR/areax_lesion_group_colors_Fig4_purpleML.json"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"

HYBRID_ROOT="$ROOT/bc_HYBRID_original17_plus_R08_R09_R10_${RUN_TAG}"

BATCH_OUT="$HYBRID_ROOT/_batch_lesion_group_summaries_dynamic_ylims_ML_combined"

OLD_BIRDS=(
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
    USA5506
    USA5509
    USA5510
)

NEW_BIRDS=(
    R08
    R09
    R10
)

for FILE in "$BATCH_SCRIPT" "$METADATA"; do
    if [[ ! -f "$FILE" ]]; then
        echo "[ABORT] Missing required file:"
        echo "$FILE"
        exit 1
    fi
done

if [[ -e "$HYBRID_ROOT" ]]; then
    echo "[ABORT] Hybrid output already exists:"
    echo "$HYBRID_ROOT"
    exit 1
fi

mkdir -p "$HYBRID_ROOT"

printf "animal_id,source_root\n" > "$HYBRID_ROOT/hybrid_source_manifest.csv"

for BIRD in "${OLD_BIRDS[@]}"; do
    SOURCE="$OLD_ROOT/$BIRD/${BIRD}_cluster_bc_summary.csv"
    DEST_DIR="$HYBRID_ROOT/$BIRD"

    if [[ ! -s "$SOURCE" ]]; then
        echo "[ABORT] Missing or empty original summary:"
        echo "$SOURCE"
        exit 1
    fi

    mkdir -p "$DEST_DIR"
    cp -p "$SOURCE" "$DEST_DIR/"

    printf "%s,%s\n" "$BIRD" "$OLD_ROOT" \
        >> "$HYBRID_ROOT/hybrid_source_manifest.csv"
done

for BIRD in "${NEW_BIRDS[@]}"; do
    SOURCE="$NEW_ROOT/$BIRD/${BIRD}_cluster_bc_summary.csv"
    DEST_DIR="$HYBRID_ROOT/$BIRD"

    if [[ ! -s "$SOURCE" ]]; then
        echo "[ABORT] Missing or empty new lateral-only summary:"
        echo "$SOURCE"
        exit 1
    fi

    mkdir -p "$DEST_DIR"
    cp -p "$SOURCE" "$DEST_DIR/"

    printf "%s,%s\n" "$BIRD" "$NEW_ROOT" \
        >> "$HYBRID_ROOT/hybrid_source_manifest.csv"
done

if [[ ! -f "$COLOR_JSON" ]]; then
    cat > "$COLOR_JSON" <<'JSON'
{
  "colors": {
    "sham saline injection": "#5BB39A",
    "Lateral lesion only": "#B39DDB",
    "Medial and Lateral lesion": "#9575C8",
    "unknown": "#BDBDBD"
  }
}
JSON
fi

python "$BATCH_SCRIPT" \
    --bc-root "$HYBRID_ROOT" \
    --metadata-excel "$METADATA" \
    --out-dir "$BATCH_OUT" \
    --metadata-sheet "animal_hit_type_summary" \
    --metadata-animal-col "Animal ID" \
    --metadata-hit-type-col "Lesion hit type" \
    --bc-method selected_bins \
    --bird-aggregate median \
    --color-json "$COLOR_JSON" \
    --dpi 300

echo "$HYBRID_ROOT" > "$PY_DIR/LAST_HYBRID_BC_ROOT.txt"

echo
echo "============================================================"
echo "Hybrid input root:"
echo "$HYBRID_ROOT"
echo
echo "Batch output:"
echo "$BATCH_OUT"
echo
echo "Main all-cluster plot:"
find "$BATCH_OUT/bird_level" \
    -maxdepth 1 \
    -type f \
    -name "bird_level_selected_bins_all_clusters_pre_vs_post_by_lesion_boxplots_only.png" \
    -print
echo
echo "M+L high-versus-remaining plot:"
find "$BATCH_OUT/bird_level/medial_lateral_only_high_vs_remaining_prepost" \
    -type f \
    -name "*selected_bins*boxplots_only.png" \
    -print
echo "============================================================"
