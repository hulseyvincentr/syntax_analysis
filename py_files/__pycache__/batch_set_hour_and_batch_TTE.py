## batch_set_hour_and_batch_TTE.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the per-bird routine you already have
from set_hour_and_batch_TTE import run_set_hour_and_batch_TTE_by_day

__all__ = ["run_batch_set_hour_and_batch_TTE"]

# ───────────────────────────────────────────────────────────────────────────────
# Helpers: discovery & labeling
# ───────────────────────────────────────────────────────────────────────────────
def _find_json_pair(dirpath: Path) -> Optional[Tuple[Path, Path]]:
    """
    Returns (decoded_json, metadata_json) if both are found in dirpath (or its immediate files).
    Heuristic: pick first match for "*_decoded_database.json" and "*_metadata.json".
    """
    if not dirpath.exists() or not dirpath.is_dir():
        return None
    decoded = sorted(dirpath.glob("*_decoded_database.json"))
    meta = sorted(dirpath.glob("*_metadata.json"))
    if decoded and meta:
        return decoded[0], meta[0]
    # Also try more generic patterns as fallback
    decoded_fallback = sorted(dirpath.glob("*decoded*.json"))
    meta_fallback = sorted(dirpath.glob("*meta*.json"))
    if decoded_fallback and meta_fallback:
        return decoded_fallback[0], meta_fallback[0]
    return None


def _short_bird_label(animal_id: str) -> str:
    """
    Make a short, readable label for plotting (e.g., 'USA5497' -> '5497', 'R07' stays 'R07').
    """
    if not animal_id:
        return "unknown"
    # If there's 'USA' followed by digits, use the digits
    m = re.match(r"USA(\d+)", animal_id, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return animal_id


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ───────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class BirdResult:
    animal_id: str
    summary_df: pd.DataFrame
    paired_fig_path: Optional[Path]
    seq_fig_path: Optional[Path]


def _compute_bird_level_am_pm(summary_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Returns (mean_am, std_am, mean_pm, std_pm) from per-day TTEs.
    """
    am = summary_df["tte_r1"].astype(float).dropna()
    pm = summary_df["tte_r2"].astype(float).dropna()
    mean_am = float(am.mean()) if len(am) else np.nan
    std_am = float(am.std(ddof=1)) if len(am) > 1 else 0.0
    mean_pm = float(pm.mean()) if len(pm) else np.nan
    std_pm = float(pm.std(ddof=1)) if len(pm) > 1 else 0.0
    return mean_am, std_am, mean_pm, std_pm


def _monthly_mean_composite(summary_df: pd.DataFrame) -> pd.Series:
    """
    Build a monthly time series for a bird: per day, average (AM, PM) TTE → then mean by month.
    - x: month index 1..12
    - y: monthly mean of ((tte_r1 + tte_r2)/2) across days in that month
    """
    df = summary_df.copy()
    df["month"] = pd.to_datetime(df["day"]).dt.month
    df["tte_composite"] = (df["tte_r1"].astype(float) + df["tte_r2"].astype(float)) / 2.0
    return df.groupby("month")["tte_composite"].mean().reindex(range(1, 13))


# ───────────────────────────────────────────────────────────────────────────────
# Plotters for the aggregate figures
# ───────────────────────────────────────────────────────────────────────────────
def _plot_am_vs_pm_by_bird(
    stats: List[Tuple[str, float, float, float, float]],
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
) -> Path:
    """
    stats: list of tuples (bird_label, mean_am, std_am, mean_pm, std_pm)
    """
    if not stats:
        return save_path  # nothing to do

    birds = [s[0] for s in stats]
    mean_am = np.array([s[1] for s in stats], dtype=float)
    std_am  = np.array([s[2] for s in stats], dtype=float)
    mean_pm = np.array([s[3] for s in stats], dtype=float)
    std_pm  = np.array([s[4] for s in stats], dtype=float)

    x = np.arange(len(birds))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(birds) * 0.6), 6))
    b1 = ax.bar(x - width/2, mean_am, width, yerr=std_am, capsize=3, label="AM (Range 1)")
    b2 = ax.bar(x + width/2, mean_pm, width, yerr=std_pm, capsize=3, label="PM (Range 2)")

    ax.set_xticks(x)
    ax.set_xticklabels(birds, rotation=45, ha="right")
    ax.set_ylabel("Total Transition Entropy (bits)")
    ax.set_title(
        f"AM vs PM TTE per Bird (mean ± SD)\n"
        f"Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def _plot_monthly_lines_by_bird(
    monthly_map: Dict[str, pd.Series],
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
) -> Path:
    """
    monthly_map: bird_label -> pd.Series indexed by month (1..12), values are monthly mean TTE_composite
    """
    if not monthly_map:
        return save_path

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(1, 13)

    for bird_label, series in monthly_map.items():
        y = series.values.astype(float)
        ax.plot(xs, y, marker="o", linewidth=1.8, alpha=0.9, label=bird_label)

    ax.set_xticks(xs)
    ax.set_xticklabels(months, rotation=0)
    ax.set_ylabel("Monthly Mean TTE (AM/PM averaged, bits)")
    ax.set_title(
        f"Monthly TTE by Bird (year ignored)\n"
        f"Per-day composite = mean(AM, PM); Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )
    ax.legend(title="Bird", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


# ───────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ───────────────────────────────────────────────────────────────────────────────
def run_batch_set_hour_and_batch_TTE(
    parent_dir: Union[str, Path],
    *,
    range1: str = "05:00-12:00",
    range2: str = "12:00-19:00",
    batch_size: int = 10,
    min_required_per_range: Optional[int] = None,     # None → defaults to batch_size
    only_song_present: bool = True,
    exclude_single_label: bool = True,
    compute_durations: bool = False,
    save_dir: Optional[Union[str, Path]] = None,      # None → save next to each decoded JSON
    fig_subdir: Optional[str] = "figures",            # can be None to avoid subfolder
    show: bool = False,
) -> Dict[str, BirdResult]:
    """
    Walk `parent_dir` for subfolders containing a decoded/metadata JSON pair.
    For each bird:
      - Runs run_set_hour_and_batch_TTE_by_day(...) to produce per-bird plots and summary_df.
    Then creates two aggregate figures across all birds:
      (A) Grouped AM vs PM bar chart (mean ± SD) per bird.
      (B) Monthly lines: for each bird, monthly mean of per-day composite TTE (mean of AM & PM per day),
          x-axis Jan..Dec, year ignored.

    Returns a dict: animal_id -> BirdResult
    """
    parent = Path(parent_dir)
    if not parent.exists() or not parent.is_dir():
        raise NotADirectoryError(f"{parent} is not a directory")

    # Discover candidate subdirectories
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    results: Dict[str, BirdResult] = {}

    # Process each subdir with a JSON pair
    for sub in sorted(subdirs):
        pair = _find_json_pair(sub)
        if not pair:
            # Also try one level deeper (some users keep JSONs directly in parent_dir)
            pair = _find_json_pair(parent)
            if not pair:
                continue
        decoded_json, meta_json = pair

        try:
            out = run_set_hour_and_batch_TTE_by_day(
                decoded_database_json=decoded_json,
                creation_metadata_json=meta_json,
                range1=range1,
                range2=range2,
                batch_size=batch_size,
                min_required_per_range=min_required_per_range,
                only_song_present=only_song_present,
                exclude_single_label=exclude_single_label,
                compute_durations=compute_durations,
                save_dir=save_dir if save_dir is not None else decoded_json.parent,
                fig_subdir=fig_subdir,
                show=show,
            )
        except Exception as e:
            print(f"[WARN] Skipping {sub} due to error: {e}")
            continue

        if out.summary_df is None or out.summary_df.empty:
            print(f"[INFO] No qualifying days for {sub.name}.")
            continue

        animal_id = out.animal_id or sub.name
        results[animal_id] = BirdResult(
            animal_id=animal_id,
            summary_df=out.summary_df.copy(),
            paired_fig_path=out.figure_path,
            seq_fig_path=out.trend_figure_path,
        )

    if not results:
        print("[INFO] No birds processed; nothing to aggregate.")
        return results

    # Decide where to save aggregate figs
    # If a global save_dir was provided, prefer that. Otherwise pick the parent_dir/fig_subdir
    base_dir = Path(save_dir) if save_dir is not None else parent
    if fig_subdir:
        base_dir = base_dir / fig_subdir
    _ensure_dir(base_dir)

    # ── Aggregate Fig A: AM vs PM per bird ─────────────────────────────────────
    am_pm_stats: List[Tuple[str, float, float, float, float]] = []
    for animal_id, br in results.items():
        bird_label = _short_bird_label(animal_id)
        mean_am, std_am, mean_pm, std_pm = _compute_bird_level_am_pm(br.summary_df)
        am_pm_stats.append((bird_label, mean_am, std_am, mean_pm, std_pm))

    ampm_path = base_dir / f"ALL_BIRDS_AM_vs_PM_TTE__R1_{range1.replace(':','').replace('-','_')}__R2_{range2.replace(':','').replace('-','_')}__N{batch_size}.png"
    _plot_am_vs_pm_by_bird(am_pm_stats, ampm_path, range1, range2, batch_size)
    print(f"[OK] Saved aggregate AM vs PM figure → {ampm_path}")

    # ── Aggregate Fig B: Monthly lines per bird (year ignored) ─────────────────
    monthly_map: Dict[str, pd.Series] = {}
    for animal_id, br in results.items():
        bird_label = _short_bird_label(animal_id)
        series = _monthly_mean_composite(br.summary_df)
        monthly_map[bird_label] = series

    months_path = base_dir / f"ALL_BIRDS_monthly_TTE_lines__R1_{range1.replace(':','').replace('-','_')}__R2_{range2.replace(':','').replace('-','_')}__N{batch_size}.png"
    _plot_monthly_lines_by_bird(monthly_map, months_path, range1, range2, batch_size)
    print(f"[OK] Saved aggregate monthly figure → {months_path}")

    return results


# ───────────────────────────────────────────────────────────────────────────────
# CLI quick test
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example:
    # python batch_set_hour_and_batch_TTE.py /path/to/parent_dir --range1 05:00-12:00 --range2 12:00-19:00 --batch-size 10
    import argparse

    parser = argparse.ArgumentParser(description="Batch TTE wrapper for multiple birds.")
    parser.add_argument("parent_dir", type=str, help="Directory whose subfolders contain decoded/metadata JSON pairs.")
    parser.add_argument("--range1", type=str, default="05:00-12:00")
    parser.add_argument("--range2", type=str, default="12:00-19:00")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--min-required-per-range", type=int, default=None)
    parser.add_argument("--only-song-present", action="store_true", default=True)
    parser.add_argument("--include-single-label", action="store_true", help="Include single-label files", default=False)
    parser.add_argument("--compute-durations", action="store_true", default=False)
    parser.add_argument("--save-dir", type=str, default=None, help="Base directory for outputs; default next to each decoded JSON.")
    parser.add_argument("--fig-subdir", type=str, default="figures")
    parser.add_argument("--show", action="store_true", default=False)

    args = parser.parse_args()

    run_batch_set_hour_and_batch_TTE(
        parent_dir=args.parent_dir,
        range1=args.range1,
        range2=args.range2,
        batch_size=args.batch_size,
        min_required_per_range=args.min_required_per_range,
        only_song_present=args.only_song_present,
        exclude_single_label=not args.include_single_label,
        compute_durations=args.compute_durations,
        save_dir=(Path(args.save_dir) if args.save_dir else None),
        fig_subdir=args.fig_subdir,
        show=args.show,
    )


"""
from batch_set_hour_and_batch_TTE import run_batch_set_hour_and_batch_TTE
from pathlib import Path

res = run_batch_set_hour_and_batch_TTE(
    parent_dir="/Users/mirandahulsey-vincent/Desktop/SfN_baseline_analysis",
    range1="05:00-12:00",
    range2="12:00-19:00",
    batch_size=10,
    min_required_per_range=10,
    only_song_present=True,
    exclude_single_label=False,
    save_dir=Path("./figures"),
    fig_subdir="batch_figures",
    show=True,
)

# res is a dict: animal_id -> BirdResult
for animal_id, bird in res.items():
    print(f"{animal_id} paired:", bird.paired_fig_path)
    print(f"{animal_id} trend :", bird.seq_fig_path)


"""