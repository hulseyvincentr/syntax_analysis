#!/bin/bash

set -u

SOURCE_ROOT="/Volumes/my_own_SSD/updated_AreaX_outputs"
DEST_ROOT="$HOME/Desktop/AFP_lesion_jsons"

MEDIAL_AND_LATERAL_BIRDS=(
    "USA5288"
    "USA5325"
    "USA5326"
    "USA5337"
    "USA5371"
    "USA5443"
    "USA5468"
    "USA5505"
    "USA5509"
)

LATERAL_ONLY_BIRDS=(
    "R08"
    "R09"
    "R10"
    "USA5336"
    "USA5347"
    "USA5483"
    "USA5499"
    "USA5510"
)

SHAM_SALINE_BIRDS=(
    "USA5271"
    "USA5283"
    "USA5494"
    "USA5506"
)

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "ERROR: Source folder not found:"
    echo "  $SOURCE_ROOT"
    exit 1
fi

mkdir -p "$DEST_ROOT"

LOG_FILE="$DEST_ROOT/copy_log.txt"
: > "$LOG_FILE"

copied_count=0
missing_count=0

copy_bird_files() {
    lesion_folder="$1"
    bird="$2"

    source_dir="$SOURCE_ROOT/$bird"
    destination_dir="$DEST_ROOT/$lesion_folder/$bird"

    mkdir -p "$destination_dir"

    echo "Processing $bird [$lesion_folder]"
    echo "[$lesion_folder / $bird]" >> "$LOG_FILE"

    if [[ ! -d "$source_dir" ]]; then
        echo "  MISSING BIRD FOLDER: $source_dir"
        echo "  MISSING BIRD FOLDER: $source_dir" >> "$LOG_FILE"
        missing_count=$((missing_count + 2))
        echo >> "$LOG_FILE"
        return
    fi

    for suffix in "song_detection.json" "decoded_database.json"; do
        source_file="$source_dir/${bird}_${suffix}"
        destination_file="$destination_dir/${bird}_${suffix}"

        if [[ -f "$source_file" ]]; then
            cp -p "$source_file" "$destination_file"

            echo "  Copied: ${bird}_${suffix}"
            echo "  COPIED: $source_file -> $destination_file" >> "$LOG_FILE"

            copied_count=$((copied_count + 1))
        else
            echo "  MISSING: ${bird}_${suffix}"
            echo "  MISSING: $source_file" >> "$LOG_FILE"

            missing_count=$((missing_count + 1))
        fi
    done

    echo >> "$LOG_FILE"
}

echo "Source:      $SOURCE_ROOT"
echo "Destination: $DEST_ROOT"
echo

for bird in "${MEDIAL_AND_LATERAL_BIRDS[@]}"; do
    copy_bird_files "medial_and_lateral" "$bird"
done

for bird in "${LATERAL_ONLY_BIRDS[@]}"; do
    copy_bird_files "lateral_only" "$bird"
done

for bird in "${SHAM_SALINE_BIRDS[@]}"; do
    copy_bird_files "sham_saline" "$bird"
done

echo
echo "Finished."
echo "Files copied:           $copied_count"
echo "Missing expected files: $missing_count"
echo
echo "Destination:"
echo "  $DEST_ROOT"
echo
echo "Detailed log:"
echo "  $LOG_FILE"
