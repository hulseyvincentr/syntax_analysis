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
    daily_points_figure_path: Optional[Path] = None  # NEW (optional)


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

def _first_last_dates(summary_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    dts = pd.to_datetime(summary_df["day"], errors="coerce").dropna()
    return (pd.NaT, pd.NaT) if dts.empty else (dts.min(), dts.max())

def _earliest_monthday_key(summary_df: pd.DataFrame) -> Tuple[int, int]:
    """Sort key ignoring year: (min month, min day) across that bird's data."""
    dts = pd.to_datetime(summary_df["day"], errors="coerce").dropna()
    if dts.empty:
        return (13, 32)  # put unknowns at the end
    m = int(dts.dt.month.min())
    d = int(dts[dts.dt.month == m].dt.day.min())
    return (m, d)

def _monthly_stats_composite(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-day composite = mean(AM, PM). Return DataFrame with:
    index=month (1..12), columns=['mean','std'].
    """
    df = summary_df.copy()
    df["month"] = pd.to_datetime(df["day"]).dt.month
    df["tte_composite"] = (df["tte_r1"].astype(float) + df["tte_r2"].astype(float)) / 2.0
    g = df.groupby("month")["tte_composite"].agg(["mean", "std"])
    g = g.reindex(range(1, 13))
    return g

def _daily_composite(summary_df: pd.DataFrame) -> pd.DataFrame:
    d = summary_df.copy()
    d["month"] = pd.to_datetime(d["day"]).dt.month
    d["tte_composite"] = (d["tte_r1"].astype(float) + d["tte_r2"].astype(float)) / 2.0
    return d[["day", "month", "tte_composite"]].dropna()

def _colors_for_labels(labels: List[str]) -> Dict[str, str]:
    """Deterministic color for each label using Matplotlib's prop cycle."""
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', list(plt.cm.tab10.colors))
    labels_sorted = sorted(labels)
    return {lab: base[i % len(base)] for i, lab in enumerate(labels_sorted)}


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
    style: str = "paired_xticks",   # 'paired_xticks' or 'overlay'
    show_std: bool = True,          # draw mean ± SD
    error_capsize: int = 4,
    error_alpha: float = 0.85,
) -> Optional[Path]:
    """
    Aggregate AM vs PM plot.

    style='paired_xticks' → x-axis shows {<ANIMAL>\\nAM, <ANIMAL>\\nPM, ...} with a
    short segment per animal (and SD error bars). Legend includes first→last dates.
    Birds are ordered by earliest month/day (year ignored).

    style='overlay'       → two x positions (AM, PM); all animals overlaid with
    slight horizontal jitter (and SD error bars).
    """
    rows: List[Tuple[str, float, float, float, float, pd.Timestamp, pd.Timestamp, Tuple[int,int]]] = []
    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        mu_am, sd_am, mu_pm, sd_pm = _means_for_bird(df)
        if np.isnan(mu_am) and np.isnan(mu_pm):
            continue
        first_dt, last_dt = _first_last_dates(df)
        sort_key = _earliest_monthday_key(df)
        rows.append((_short_label(animal_id), mu_am, sd_am, mu_pm, sd_pm, first_dt, last_dt, sort_key))

    if not rows:
        print("[INFO] No birds with qualifying data to aggregate for AM/PM figure.")
        return None

    # Stable order: earliest month/day, then label
    rows.sort(key=lambda r: (r[7], r[0]))

    labels_for_color = [r[0] for r in rows]
    color_map = _colors_for_labels(labels_for_color)

    if style not in {"paired_xticks", "overlay"}:
        print(f"[WARN] Unknown style '{style}', falling back to 'paired_xticks'.")
        style = "paired_xticks"

    if style == "paired_xticks":
        x_positions: List[int] = []
        x_labels: List[str] = []
        paired_vals: List[Tuple[int, int, float, float, float, float, str, str]] = []

        for i, (bird, mu_am, sd_am, mu_pm, sd_pm, first_dt, last_dt, _sk) in enumerate(rows):
            x_am = 2 * i
            x_pm = 2 * i + 1
            x_positions.extend([x_am, x_pm])
            # two-line tick: animal on first line, AM/PM on second
            x_labels.extend([f"{bird}\nAM", f"{bird}\nPM"])

            # legend label with first→last dates (month/day only)
            flabel = "n/a"
            if pd.notna(first_dt) and pd.notna(last_dt):
                flabel = f"{first_dt:%b %d} → {last_dt:%b %d}"

            paired_vals.append((x_am, x_pm, mu_am, sd_am, mu_pm, sd_pm, bird, flabel))

        fig, ax = plt.subplots(figsize=(max(12, len(x_labels) * 0.4), 6))

        for x_am, x_pm, mu_am, sd_am, mu_pm, sd_pm, bird, flabel in paired_vals:
            color = color_map[bird]
            (line,) = ax.plot([x_am, x_pm], [mu_am, mu_pm],
                              marker="o", linewidth=2.0, alpha=0.95,
                              label=f"{bird} — {flabel}", color=color, zorder=3)
            if show_std:
                if np.isfinite(sd_am) and sd_am > 0:
                    ax.errorbar(x_am, mu_am, yerr=sd_am, fmt="none",
                                ecolor=color, elinewidth=1.6,
                                capsize=error_capsize, alpha=error_alpha, zorder=2)
                if np.isfinite(sd_pm) and sd_pm > 0:
                    ax.errorbar(x_pm, mu_pm, yerr=sd_pm, fmt="none",
                                ecolor=color, elinewidth=1.6,
                                capsize=error_capsize, alpha=error_alpha, zorder=2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=0, ha="center")

    else:  # overlay
        x_am_base, x_pm_base = 0.0, 1.0
        jitters = np.linspace(-0.22, 0.22, len(rows))
        fig, ax = plt.subplots(figsize=(12, 6))

        for j, (bird, mu_am, sd_am, mu_pm, sd_pm, first_dt, last_dt, _sk) in zip(jitters, rows):
            x_am = x_am_base + j
            x_pm = x_pm_base + j
            color = color_map[bird]
            flabel = "n/a"
            if pd.notna(first_dt) and pd.notna(last_dt):
                flabel = f"{first_dt:%b %d} → {last_dt:%b %d}"

            (line,) = ax.plot([x_am, x_pm], [mu_am, mu_pm],
                              marker="o", linewidth=2.0, alpha=0.95,
                              label=f"{bird} — {flabel}", color=color, zorder=3)
            if show_std:
                if np.isfinite(sd_am) and sd_am > 0:
                    ax.errorbar(x_am, mu_am, yerr=sd_am, fmt="none",
                                ecolor=color, elinewidth=1.6,
                                capsize=error_capsize, alpha=error_alpha, zorder=2)
                if np.isfinite(sd_pm) and sd_pm > 0:
                    ax.errorbar(x_pm, mu_pm, yerr=sd_pm, fmt="none",
                                ecolor=color, elinewidth=1.6,
                                capsize=error_capsize, alpha=error_alpha, zorder=2)

        ax.set_xticks([x_am_base, x_pm_base])
        ax.set_xticklabels([f"AM ({range1})", f"PM ({range2})"])

    # Shared cosmetics & save
    ax.set_ylabel("Total Transition Entropy (bits)")
    title_mode = "paired x-ticks per animal" if style == "paired_xticks" else "overlay"
    ax.set_title(
        f"Aggregate AM vs PM TTE ({title_mode})\n"
        f"Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )
    ax.legend(title="Animal (first → last day recorded)", frameon=False,
              bbox_to_anchor=(1.02, 1), loc="upper left")
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
    capsize: int = 4,
) -> Optional[Path]:
    """
    Monthly line plot with error bars: one line per bird; x = Jan..Dec (year ignored).
    y = monthly mean of per-day composite TTE (mean of AM & PM); error = monthly SD.
    """
    months = np.arange(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # build stats per bird
    per_bird_stats: Dict[str, pd.DataFrame] = {}
    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        per_bird_stats[_short_label(animal_id)] = _monthly_stats_composite(df)

    if not per_bird_stats:
        print("[INFO] No monthly data available for aggregate monthly figure.")
        return None

    labels = sorted(per_bird_stats.keys())
    color_map = _colors_for_labels(labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    for label in labels:
        stats = per_bird_stats[label]
        y = stats["mean"].to_numpy(dtype=float)
        yerr = stats["std"].to_numpy(dtype=float)
        color = color_map[label]
        ax.errorbar(months, y, yerr=yerr, marker="o", linewidth=1.8,
                    elinewidth=1.2, capsize=capsize, alpha=0.95,
                    label=label, color=color)

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Monthly Mean TTE (AM/PM averaged, bits)")
    ax.set_title(
        f"Monthly TTE by Bird (mean ± SD; year ignored)\n"
        f"Per-day composite = mean(AM, PM); Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )
    ax.legend(title="Animal", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved aggregate monthly figure → {save_path}")
    return save_path


def _plot_monthly_daily_points(
    bird_summary_map: Dict[str, pd.DataFrame],
    *,
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
    jitter: float = 0.12,
    s: float = 28,
    alpha: float = 0.75,
) -> Optional[Path]:
    """
    Scatter of per-day composite TTE for all days, x = month (1..12) with slight jitter.
    Exactly ONE scatter call per animal → one consistent color and legend entry per animal.
    """
    # collect all day-points per bird label
    by_animal: Dict[str, List[pd.DataFrame]] = {}
    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        d = _daily_composite(df)
        if d.empty:
            continue
        by_animal.setdefault(_short_label(animal_id), []).append(d)

    if not by_animal:
        print("[INFO] No daily points found for monthly daily points figure.")
        return None

    labels = sorted(by_animal.keys())
    color_map = _colors_for_labels(labels)

    months = np.arange(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # one scatter per animal
    for label in labels:
        D = pd.concat(by_animal[label], ignore_index=True)
        x = D["month"].to_numpy(dtype=float)
        y = D["tte_composite"].to_numpy(dtype=float)
        # deterministic jitter per animal (no color cycling issues)
        rng = np.random.default_rng(abs(hash(label)) % (2**32))
        xj = x + rng.uniform(-jitter, jitter, size=len(x))
        ax.scatter(xj, y, s=s, alpha=alpha, color=color_map[label], label=label)

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, rotation=0)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylabel("Per-day composite TTE (AM/PM averaged, bits)")
    ax.set_title(
        f"Daily composite TTE by month (all animals)\n"
        f"Points = individual days; Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )
    ax.legend(title="Animal", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved monthly daily-points figure → {save_path}")
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
    min_required_per_range: Optional[int] = None,     # None → pass-through, callee handles default
    only_song_present: bool = True,
    exclude_single_label: bool = True,
    compute_durations: bool = False,
    save_dir: Optional[Union[str, Path]] = None,      # None → save next to each decoded JSON
    fig_subdir: Optional[str] = "figures",
    make_monthly_lines: bool = True,
    make_daily_points: bool = True,
    show: bool = False,
    aggregate_style: str = "paired_xticks",           # "overlay" or "paired_xticks"
) -> BatchResults:
    """
    1) Finds JSON pairs in subfolders of `parent_dir`.
    2) For each bird, calls run_set_hour_and_batch_TTE_by_day(...) to generate *individual* figures.
    3) Builds ONE aggregate AM vs PM figure (style selectable).
    4) Builds a monthly mean (±SD) figure and a per-day-points-by-month figure.
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

    # Optional: monthly means ± SD
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

    # Optional: daily points by month (one color per animal)
    daily_points_path = None
    if make_daily_points:
        daily_name = (
            f"ALL_BIRDS_monthly_daily_points__"
            f"R1_{range1.replace(':','').replace('-','_')}__"
            f"R2_{range2.replace(':','').replace('-','_')}__N{batch_size}.png"
        )
        daily_points_path = base_dir / daily_name
        daily_points_path = _plot_monthly_daily_points(
            bird_summary_map,
            save_path=daily_points_path,
            range1=range1,
            range2=range2,
            batch_size=batch_size,
        )

    return BatchResults(
        per_bird=per_bird,
        am_pm_figure_path=ampm_path,
        monthly_figure_path=monthly_path,
        daily_points_figure_path=daily_points_path,
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
                        help="Skip the monthly means ± SD figure.")
    parser.add_argument("--no-daily-points", action="store_true",
                        help="Skip the daily composite-by-month scatter.")
    parser.add_argument("--aggregate-style", type=str, default="paired_xticks",
                        choices=["overlay", "paired_xticks"],
                        help="overlay: 2 ticks (AM/PM) with one line per animal; "
                             "paired_xticks: interleaved '<ANIMAL>\\nAM/PM' ticks, sorted by earliest month/day.")
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
        make_daily_points=not args.no_daily_points,
        show=args.show,
        aggregate_style=args.aggregate_style,
    )
    print("AM/PM aggregate:", out.am_pm_figure_path)
    print("Monthly figure :", out.monthly_figure_path)
    print("Daily points  :", out.daily_points_figure_path)


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