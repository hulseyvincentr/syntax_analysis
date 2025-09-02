# batch_set_hour_and_batch_TTE.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from set_hour_and_batch_TTE import run_set_hour_and_batch_TTE_by_day

__all__ = [
    "run_batch_set_hour_and_batch_TTE",
    "BirdResult",
    "BatchResults",
]

# ───────────────────────────────────────────────────────────────────────────────
# Data containers
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class BirdResult:
    animal_id: str
    summary_df: pd.DataFrame
    paired_fig_path: Optional[Path]
    seq_fig_path: Optional[Path]

@dataclass
class BatchResults:
    per_bird: Dict[str, BirdResult]
    am_pm_figure_path: Optional[Path]
    monthly_figure_path: Optional[Path]

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _find_json_pair(dirpath: Path) -> Optional[Tuple[Path, Path]]:
    """Return (decoded_json, metadata_json) if found under dirpath, else None."""
    if not dirpath.exists() or not dirpath.is_dir():
        return None
    decoded = sorted(dirpath.glob("*_decoded_database.json"))
    meta    = sorted(dirpath.glob("*_metadata.json"))
    if decoded and meta:
        return decoded[0], meta[0]
    # Fallbacks if naming differs
    dec_fb = sorted(dirpath.glob("*decoded*.json"))
    met_fb = sorted(dirpath.glob("*meta*.json"))
    if dec_fb and met_fb:
        return dec_fb[0], met_fb[0]
    return None

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _short_label(animal_id: str) -> str:
    """Optional shortening: 'USA5497' -> '5497'."""
    m = re.match(r"USA(\d+)", str(animal_id), flags=re.IGNORECASE)
    return m.group(1) if m else str(animal_id)

def _means_for_bird(summary_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Return mean±sd of AM (tte_r1) and PM (tte_r2) for a bird."""
    am = summary_df["tte_r1"].astype(float).dropna()
    pm = summary_df["tte_r2"].astype(float).dropna()
    mean_am = float(am.mean()) if len(am) else np.nan
    std_am  = float(am.std(ddof=1)) if len(am) > 1 else 0.0
    mean_pm = float(pm.mean()) if len(pm) else np.nan
    std_pm  = float(pm.std(ddof=1)) if len(pm) > 1 else 0.0
    return mean_am, std_am, mean_pm, std_pm

def _monthly_mean_composite(summary_df: pd.DataFrame) -> pd.Series:
    """
    Per-day composite = mean(AM, PM).
    Return monthly means (index 1..12), NaN where no data.
    """
    df = summary_df.copy()
    df["month"] = pd.to_datetime(df["day"]).dt.month
    df["tte_composite"] = (df["tte_r1"].astype(float) + df["tte_r2"].astype(float)) / 2.0
    return df.groupby("month")["tte_composite"].mean().reindex(range(1, 13))

# ───────────────────────────────────────────────────────────────────────────────
# Aggregate plotters
# ───────────────────────────────────────────────────────────────────────────────

def _plot_aggregate_am_pm(
    bird_summary_map: Dict[str, pd.DataFrame],
    *,
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
    style: str = "overlay",  # "overlay" (2 ticks, legend per animal) or "paired_xticks"
) -> Optional[Path]:
    """
    Aggregate AM↔PM figure, two styles:

    - overlay (default): two x-ticks (AM, PM). One line per animal; legend lists animal IDs.
    - paired_xticks: interleaved ticks per animal (e.g., 'R07 AM', 'R07 PM', ...).

    Returns the saved figure path or None if no data.
    """
    # collect means
    birds: List[str] = []
    am_means: List[float] = []
    pm_means: List[float] = []

    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        mu_am, _sd_am, mu_pm, _sd_pm = _means_for_bird(df)
        if np.isnan(mu_am) and np.isnan(mu_pm):
            continue
        birds.append(_short_label(animal_id))
        am_means.append(mu_am)
        pm_means.append(mu_pm)

    if not birds:
        print("[INFO] No birds with qualifying data to aggregate for AM/PM figure.")
        return None

    # stable order
    order = np.argsort(birds)
    birds    = [birds[i] for i in order]
    am_means = [am_means[i] for i in order]
    pm_means = [pm_means[i] for i in order]

    style = style.lower().strip()
    if style not in {"overlay", "paired_xticks"}:
        style = "overlay"

    if style == "overlay":
        # Two shared x positions
        fig, ax = plt.subplots(figsize=(12, 6))
        x_am, x_pm = 0, 1

        handles = []
        labels  = []
        for b, y0, y1 in zip(birds, am_means, pm_means):
            h, = ax.plot([x_am, x_pm], [y0, y1], marker="o", linewidth=1.8, alpha=0.95, label=b)
            handles.append(h); labels.append(b)

        ax.set_xlim(-0.25, 1.25)
        ax.set_xticks([x_am, x_pm])
        ax.set_xticklabels([f"{range1}\n(AM mean)", f"{range2}\n(PM mean)"])
        ax.set_ylabel("Total Transition Entropy (bits)")
        ax.set_title(
            "Aggregate AM vs PM TTE (overlay; one line per animal)\n"
            f"Range1={range1}, Range2={range2}, batch_size={batch_size}"
        )
        # Legend per animal
        n = len(birds)
        ncol = 1 if n <= 12 else 2 if n <= 24 else 3
        ax.legend(handles, labels, title="Animal", bbox_to_anchor=(1.02, 1),
                  loc="upper left", frameon=False, ncol=ncol, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        fig.tight_layout()

    else:  # "paired_xticks"
        # Interleaved positions per animal; ticks include animal_id
        x_positions, x_labels = [], []
        fig, ax = plt.subplots(figsize=(max(12, len(birds) * 0.8), 6))
        handles = []; labels = []
        for i, (b, y0, y1) in enumerate(zip(birds, am_means, pm_means)):
            x0, x1 = 2 * i, 2 * i + 1
            h, = ax.plot([x0, x1], [y0, y1], marker="o", linewidth=1.8, alpha=0.95, label=b)
            handles.append(h); labels.append(b)
            x_positions.extend([x0, x1])
            x_labels.extend([f"{b} AM", f"{b} PM"])

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Total Transition Entropy (bits)")
        ax.set_title(
            "Aggregate AM vs PM TTE (paired x-ticks per animal)\n"
            f"Range1={range1}, Range2={range2}, batch_size={batch_size}"
        )
        n = len(birds)
        ncol = 1 if n <= 12 else 2 if n <= 24 else 3
        ax.legend(handles, labels, title="Animal", bbox_to_anchor=(1.02, 1),
                  loc="upper left", frameon=False, ncol=ncol, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved aggregate AM/PM figure → {save_path}")
    return save_path


def _plot_monthly_lines_by_bird(
    bird_summary_map: Dict[str, pd.DataFrame],
    *,
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
) -> Optional[Path]:
    """
    Line plot: one line per bird; x-axis Jan..Dec (year ignored).
    y = monthly mean of per-day composite TTE (mean of AM & PM).
    """
    monthly_map: Dict[str, pd.Series] = {}
    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        monthly_map[_short_label(animal_id)] = _monthly_mean_composite(df)

    if not monthly_map:
        print("[INFO] No monthly data available for aggregate monthly figure.")
        return None

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    xs = np.arange(1, 13)

    fig, ax = plt.subplots(figsize=(12, 6))
    for bird_label, series in sorted(monthly_map.items(), key=lambda kv: kv[0]):
        y = series.values.astype(float)
        ax.plot(xs, y, marker="o", linewidth=1.8, alpha=0.9, label=bird_label)

    ax.set_xticks(xs)
    ax.set_xticklabels(months)
    ax.set_ylabel("Monthly Mean TTE (AM/PM averaged, bits)")
    ax.set_title(
        f"Monthly TTE by Bird (year ignored)\n"
        f"Per-day composite = mean(AM, PM); Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )
    ax.legend(title="Bird", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved aggregate monthly figure → {save_path}")
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
    min_required_per_range: Optional[int] = None,     # None → defaults to batch_size inside callee
    only_song_present: bool = True,
    exclude_single_label: bool = True,
    compute_durations: bool = False,
    save_dir: Optional[Union[str, Path]] = None,      # None → save next to each decoded JSON
    fig_subdir: Optional[str] = "figures",
    make_monthly_lines: bool = True,
    show: bool = False,
    aggregate_style: str = "overlay",                 # "overlay" or "paired_xticks"
) -> BatchResults:
    """
    1) Finds JSON pairs in subfolders of `parent_dir`.
    2) For each bird, calls run_set_hour_and_batch_TTE_by_day(...) to generate *individual* figures.
    3) Builds ONE aggregate AM vs PM figure:
         - aggregate_style="overlay"      → two ticks (AM, PM), one line per animal (legend per animal).
         - aggregate_style="paired_xticks"→ 'R07 AM','R07 PM','R13 AM','R13 PM',... on x-axis.
    4) Optionally builds a monthly lines figure (Jan..Dec; year ignored).
    """
    parent = Path(parent_dir)
    if not parent.exists() or not parent.is_dir():
        raise NotADirectoryError(f"{parent} is not a directory")

    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    per_bird: Dict[str, BirdResult] = {}

    # Per-bird processing
    for sub in sorted(subdirs):
        pair = _find_json_pair(sub)
        if not pair:
            continue
        decoded_json, meta_json = pair

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
        if out.summary_df is None or out.summary_df.empty:
            print(f"[INFO] No qualifying days for {sub.name}.")
            continue

        animal_id = out.animal_id or sub.name
        per_bird[animal_id] = BirdResult(
            animal_id=animal_id,
            summary_df=out.summary_df.copy(),
            paired_fig_path=out.figure_path,
            seq_fig_path=out.trend_figure_path,
        )

    if not per_bird:
        print("[INFO] No birds processed; nothing to aggregate.")
        return BatchResults(per_bird=per_bird, am_pm_figure_path=None, monthly_figure_path=None)

    # Decide where to save the aggregate figures
    base_dir = Path(save_dir) if save_dir is not None else parent
    if fig_subdir:
        base_dir = base_dir / fig_subdir
    _ensure_dir(base_dir)

    # Build aggregate inputs: animal_id -> summary_df
    bird_summary_map = {aid: br.summary_df for aid, br in per_bird.items()}

    # Aggregate AM vs PM (single figure)
    style_tag = "OVERLAY" if aggregate_style.lower() == "overlay" else "PAIRED"
    ampm_name = (
        f"ALL_BIRDS_AM_vs_PM_TTE_{style_tag}__"
        f"R1_{range1.replace(':','').replace('-','_')}__"
        f"R2_{range2.replace(':','').replace('-','_')}__N{batch_size}.png"
    )
    ampm_path = base_dir / ampm_name
    ampm_path = _plot_aggregate_am_pm(
        bird_summary_map,
        save_path=ampm_path,
        range1=range1,
        range2=range2,
        batch_size=batch_size,
        style=aggregate_style,
    )

    # Optional: monthly lines figure
    monthly_path = None
    if make_monthly_lines:
        monthly_name = (
            f"ALL_BIRDS_monthly_TTE_lines__"
            f"R1_{range1.replace(':','').replace('-','_')}__"
            f"R2_{range2.replace(':','').replace('-','_')}__N{batch_size}.png"
        )
        monthly_path = base_dir / monthly_name
        monthly_path = _plot_monthly_lines_by_bird(
            bird_summary_map,
            save_path=monthly_path,
            range1=range1,
            range2=range2,
            batch_size=batch_size,
        )

    return BatchResults(
        per_bird=per_bird,
        am_pm_figure_path=ampm_path,
        monthly_figure_path=monthly_path,
    )

# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch TTE wrapper for multiple birds (individual + aggregate figures)."
    )
    parser.add_argument("parent_dir", type=str,
                        help="Directory whose subfolders contain decoded/metadata JSON pairs.")
    parser.add_argument("--range1", type=str, default="05:00-12:00")
    parser.add_argument("--range2", type=str, default="12:00-19:00")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--min-required-per-range", type=int, default=None)
    parser.add_argument("--only-song-present", action="store_true", default=True)
    parser.add_argument("--include-single-label", action="store_true", default=False,
                        help="Include single-label files.")
    parser.add_argument("--compute-durations", action="store_true", default=False)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Base directory for outputs; default next to each decoded JSON.")
    parser.add_argument("--fig-subdir", type=str, default="figures")
    parser.add_argument("--no-monthly-lines", action="store_true",
                        help="Skip the monthly lines aggregate figure.")
    parser.add_argument("--aggregate-style", type=str, default="overlay",
                        choices=["overlay", "paired_xticks"],
                        help="overlay: 2 ticks (AM/PM) with one line per animal; "
                             "paired_xticks: interleaved 'ANIMAL AM/PM' ticks.")
    parser.add_argument("--show", action="store_true", default=False)

    args = parser.parse_args()

    out = run_batch_set_hour_and_batch_TTE(
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
        make_monthly_lines=not args.no_monthly_lines,
        show=args.show,
        aggregate_style=args.aggregate_style,
    )
    print("AM/PM aggregate:", out.am_pm_figure_path)
    print("Monthly figure :", out.monthly_figure_path)


"""
from pathlib import Path
import importlib, batch_set_hour_and_batch_TTE as batch_mod
importlib.reload(batch_mod)
from batch_set_hour_and_batch_TTE import run_batch_set_hour_and_batch_TTE

res = run_batch_set_hour_and_batch_TTE(
    parent_dir="/Users/mirandahulsey-vincent/Desktop/SfN_baseline_analysis",
    range1="05:00-12:00",
    range2="12:00-19:00",
    batch_size=10,
    min_required_per_range=10,
    only_song_present=True,
    exclude_single_label=False,
    fig_subdir="batch_figures",
    show=False,
    aggregate_style="paired_xticks",   # <<< this makes one figure with {ANIMAL AM, ANIMAL PM} ticks
)
print("AM/PM aggregate:", res["am_pm_figure_path"] if isinstance(res, dict) else res.am_pm_figure_path)
print("Monthly figure :", res["monthly_figure_path"] if isinstance(res, dict) else res.monthly_figure_path)



"""