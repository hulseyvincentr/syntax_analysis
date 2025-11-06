from pathlib import Path
import importlib
import numpy as np

from organized_decoded_serialTS_segments import build_organized_segments_with_durations
import syllable_heatmap_linear as shlin
importlib.reload(shlin)  # pick up any recent edits

# --- Inputs ---
decoded = Path("/Users/mirandahulsey-vincent/Desktop/USA1234_VALIDATION_Data/USA1234_decoded_database.json")
tdate   = "2000-01-01"   # optional; set to None to skip the line

# --- Output dir ---
outdir = decoded.parent / "figures"
outdir.mkdir(parents=True, exist_ok=True)

# --- Build organized df ---
out = build_organized_segments_with_durations(
    decoded_database_json=decoded,
    only_song_present=True,
    compute_durations=False,
    add_recording_datetime=True,
)

# --- Build daily avg count table ---
count_table = shlin.build_daily_avg_count_table(
    out.organized_df,
    label_column="syllable_onsets_offsets_ms_dict",
    date_column="Date",
    syllable_labels=out.unique_syllable_labels,
)

# --- Determine file names ---
animal_id = shlin._infer_animal_id(out.organized_df, decoded) or "unknown_animal"
save_path_1 = outdir / f"{animal_id}_syllable_heatmap_linear_vmax1.00.png"
save_path_2 = outdir / f"{animal_id}_syllable_heatmap_linear_vmaxMAX.png"

# --- Compute vmax for Plot 2 (max avg count per song) ---
raw_max = float(np.nanmax(count_table.to_numpy()))
vmax_max = raw_max if np.isfinite(raw_max) and raw_max > 0 else 1.0  # safe fallback

# ─────────────────────────────────────────────────────────────
# Plot 1: vmin=0.00, vmax=1.00
# ─────────────────────────────────────────────────────────────
fig1, ax1 = shlin.plot_linear_scaled_syllable_counts(
    count_table,
    animal_id=animal_id,
    treatment_date=tdate,      # set to None to skip the red line
    save_path=save_path_1,
    show=True,                 # show the figure window
    cmap="Greys",
    vmin=0.0,
    vmax=1.0,
    nearest_match=True,
    max_days_off=1,
)

# ─────────────────────────────────────────────────────────────
# Plot 2: vmin=0.00, vmax=max average count per song
# ─────────────────────────────────────────────────────────────
fig2, ax2 = shlin.plot_linear_scaled_syllable_counts(
    count_table,
    animal_id=animal_id,
    treatment_date=tdate,
    save_path=save_path_2,
    show=True,
    cmap="Greys",
    vmin=0.0,
    vmax=vmax_max,
    nearest_match=True,
    max_days_off=1,
)

print("Saved:", save_path_1)
print("Saved:", save_path_2)
print("Max avg count per song (for Plot 2 vmax):", vmax_max)


"""
from pathlib import Path
import importlib
import numpy as np

from organized_decoded_serialTS_segments import build_organized_segments_with_durations
import syllable_heatmap_linear as shlin
importlib.reload(shlin)  # pick up any edits

# --- Inputs ---
decoded = Path("/Users/mirandahulsey-vincent/Desktop/USA1234_VALIDATION_Data/USA1234_decoded_database.json")
tdate   = "2000-01-01"   # optional; set to None to skip the treatment line

# --- Output dir (same for both files) ---
outdir = decoded.parent / "figures"
outdir.mkdir(parents=True, exist_ok=True)

# --- Build organized df ---
out = build_organized_segments_with_durations(
    decoded_database_json=decoded,
    only_song_present=True,
    compute_durations=False,
    add_recording_datetime=True,
)

# --- Build daily avg count table ---
count_table = shlin.build_daily_avg_count_table(
    out.organized_df,
    label_column="syllable_onsets_offsets_ms_dict",
    date_column="Date",
    syllable_labels=out.unique_syllable_labels,
)

# --- Infer ID and file paths ---
animal_id = shlin._infer_animal_id(out.organized_df, decoded) or "unknown_animal"
save_path_1 = outdir / f"{animal_id}_syllable_heatmap_linear_vmax1.00.png"
save_path_2 = outdir / f"{animal_id}_syllable_heatmap_linear_vmaxMAX.png"

# --- Compute vmax for Plot 2 (max avg count per song) ---
raw_max = float(np.nanmax(count_table.to_numpy())) if count_table.size else 1.0
vmax_max = raw_max if np.isfinite(raw_max) and raw_max > 0 else 1.0

# ─────────────────────────────────────────────────────────────
# Plot 1: vmin=0.00, vmax=1.00
# ─────────────────────────────────────────────────────────────
fig1, ax1 = shlin.plot_linear_scaled_syllable_counts(
    count_table,
    animal_id=animal_id,
    treatment_date=tdate,
    save_path=save_path_1,
    show=True,
    cmap="Greys",
    vmin=0.0,
    vmax=1.0,
    nearest_match=True,
    max_days_off=1,
)

# ─────────────────────────────────────────────────────────────
# Plot 2: vmin=0.00, vmax=max average count per song
# ─────────────────────────────────────────────────────────────
fig2, ax2 = shlin.plot_linear_scaled_syllable_counts(
    count_table,
    animal_id=animal_id,
    treatment_date=tdate,
    save_path=save_path_2,
    show=True,
    cmap="Greys",
    vmin=0.0,
    vmax=vmax_max,
    nearest_match=True,
    max_days_off=1,
)

print("Saved:", save_path_1)
print("Saved:", save_path_2)
print("Max avg count per song (vmax for Plot 2):", vmax_max)



"""