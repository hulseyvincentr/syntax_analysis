# batch_set_hour_and_batch_TTE.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy (preferred for Wilcoxon); we fall back if missing
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

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
    daily_points_figure_path: Optional[Path]
    significance_diffs_figure_path: Optional[Path]
    significance_groups_figure_path: Optional[Path]

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

def _first_month(df: pd.DataFrame) -> int:
    """Earliest month number (1-12) in df['day'] (ignores year)."""
    d = pd.to_datetime(df["day"], errors="coerce").dropna()
    return int(d.dt.month.min()) if len(d) else 13

def _first_last_dates(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    d = pd.to_datetime(df["day"], errors="coerce").dropna().sort_values()
    if not len(d):
        return None, None
    return d.iloc[0], d.iloc[-1]

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

def _bootstrap_ci_mean(x: np.ndarray, alpha: float = 0.05, n_boot: int = 5000) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng()
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    samples = x[idx].mean(axis=1)
    lo = float(np.percentile(samples, 100 * (alpha / 2)))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi

def _p_to_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

def _paired_test(am: np.ndarray, pm: np.ndarray, method: str = "wilcoxon") -> Tuple[float, float]:
    """
    Return (stat, pvalue) for paired AM vs PM.
    Prefers SciPy Wilcoxon; falls back to sign-permutation test.
    """
    am = np.asarray(am, float)
    pm = np.asarray(pm, float)
    mask = np.isfinite(am) & np.isfinite(pm)
    am = am[mask]; pm = pm[mask]
    if len(am) == 0:
        return np.nan, np.nan

    if method == "wilcoxon" and _HAVE_SCIPY:
        try:
            res = _scipy_stats.wilcoxon(am, pm, alternative="two-sided", zero_method="wilcox")
            return float(res.statistic), float(res.pvalue)
        except Exception:
            pass  # fall through to permutation

    # Sign-permutation (flip signs of diffs)
    d = pm - am
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan, np.nan
    obs = np.mean(d)
    rng = np.random.default_rng()
    B = 10000 if n <= 40 else 5000
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    perm_means = (d[None, :] * signs).mean(axis=1)
    p = 2.0 * max(
        np.mean(perm_means >= abs(obs)),
        np.mean(perm_means <= -abs(obs))
    )
    p = float(min(1.0, p))
    return float(obs), p  # stat not comparable to Wilcoxon; report mean(diffs) as stat

# ───────────────────────────────────────────────────────────────────────────────
# Aggregate AM/PM plot (paired x-ticks, with optional error bars)
# ───────────────────────────────────────────────────────────────────────────────

def _plot_aggregate_am_pm(
    bird_summary_map: Dict[str, pd.DataFrame],
    *,
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
    show_std: bool = True,
) -> Optional[Path]:
    """
    Build ONE figure with x-axis like:
      5177\nAM, 5177\nPM, 5185\nAM, 5185\nPM, ...
    Each bird is a line segment connecting its AM mean to PM mean.
    """
    rows: List[Tuple[str, float, float, float, float, int]] = []
    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        mu_am, sd_am, mu_pm, sd_pm = _means_for_bird(df)
        first_m = _first_month(df)
        if np.isnan(mu_am) and np.isnan(mu_pm):
            continue
        rows.append((_short_label(animal_id), mu_am, sd_am, mu_pm, sd_pm, first_m))

    if not rows:
        print("[INFO] No birds with qualifying data to aggregate for AM/PM figure.")
        return None

    rows.sort(key=lambda x: (x[5], x[0]))  # by earliest month, then label

    x_positions: List[int] = []
    x_labels: List[str] = []
    paired_vals: List[Tuple[int, int, float, float, float, float, str]] = []

    for i, (bird, mu_am, sd_am, mu_pm, sd_pm, _m) in enumerate(rows):
        x_am = 2 * i
        x_pm = 2 * i + 1
        x_positions.extend([x_am, x_pm])
        x_labels.extend([f"{bird}\nAM", f"{bird}\nPM"])
        paired_vals.append((x_am, x_pm, mu_am, sd_am, mu_pm, sd_pm, bird))

    fig, ax = plt.subplots(figsize=(max(14, len(x_labels) * 0.75), 7))
    for x_am, x_pm, mu_am, sd_am, mu_pm, sd_pm, bird in paired_vals:
        (line,) = ax.plot([x_am, x_pm], [mu_am, mu_pm],
                          marker="o", linewidth=2.0, alpha=0.95,
                          label=bird, zorder=3)
        color = line.get_color()
        if show_std:
            if np.isfinite(sd_am) and sd_am > 0:
                ax.errorbar(x_am, mu_am, yerr=sd_am, fmt="none",
                            ecolor=color, elinewidth=1.6, capsize=4, alpha=0.85, zorder=2)
            if np.isfinite(sd_pm) and sd_pm > 0:
                ax.errorbar(x_pm, mu_pm, yerr=sd_pm, fmt="none",
                            ecolor=color, elinewidth=1.6, capsize=4, alpha=0.85, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel("Total Transition Entropy (bits)")
    ax.set_title(
        "Aggregate AM vs PM TTE (paired x-ticks per bird; mean ± SD)\n"
        f"Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )
    ax.legend(title="Bird", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved aggregate AM/PM figure → {save_path}")
    return save_path

# ───────────────────────────────────────────────────────────────────────────────
# Monthly line plot (per bird)
# ───────────────────────────────────────────────────────────────────────────────

def _plot_monthly_lines_by_bird(
    bird_summary_map: Dict[str, pd.DataFrame],
    *,
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
) -> Optional[Path]:
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
    for bird_label, series in monthly_map.items():
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
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved aggregate monthly figure → {save_path}")
    return save_path

# ───────────────────────────────────────────────────────────────────────────────
# Daily scatter: composite points (AM/PM averaged) colored by bird
# ───────────────────────────────────────────────────────────────────────────────

def _plot_daily_points_by_month_all_birds(
    bird_summary_map: Dict[str, pd.DataFrame],
    *,
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
) -> Optional[Path]:
    # Prepare points (month vs composite per-day)
    records = []
    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        bird = _short_label(animal_id)
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["day"], errors="coerce")
        tmp = tmp.dropna(subset=["date"])
        if tmp.empty:
            continue
        tmp["month"] = tmp["date"].dt.month
        tmp["tte_comp"] = (tmp["tte_r1"].astype(float) + tmp["tte_r2"].astype(float)) / 2.0
        for _i, r in tmp[["month", "tte_comp"]].iterrows():
            records.append((bird, int(r["month"]), float(r["tte_comp"])))

    if not records:
        print("[INFO] No per-day data for daily points by month.")
        return None

    df_all = pd.DataFrame(records, columns=["bird", "month", "tte_comp"])
    birds = sorted(df_all["bird"].unique().tolist())
    color_map = {b: plt.get_cmap("tab10")(i % 10) for i, b in enumerate(birds)}

    fig, ax = plt.subplots(figsize=(14, 7))
    for b in birds:
        d = df_all[df_all["bird"] == b]
        jitter = (np.random.rand(len(d)) - 0.5) * 0.28
        ax.scatter(
            d["month"].values + jitter,
            d["tte_comp"].values,
            s=28, alpha=0.9, color=color_map[b], edgecolors="none", label=b
        )

    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_ylabel("Per-day composite TTE (bits)")
    ax.set_title(
        "Daily TTE Points by Month (all birds)\n"
        f"Points = individual days; Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )
    ax.legend(title="Bird", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved daily points by month figure → {save_path}")
    return save_path

# ───────────────────────────────────────────────────────────────────────────────
# Significance plots
# ───────────────────────────────────────────────────────────────────────────────

def _plot_am_pm_significance_differences(
    bird_summary_map: Dict[str, pd.DataFrame],
    *,
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
) -> Optional[Path]:
    """
    One distribution *per bird* of paired differences: diff = PM − AM for each day.
    Shows jittered points, mean ± 95% bootstrap CI, and one-sample Wilcoxon/sign-permutation p vs 0.
    """
    rows = []
    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        tmp = df[["day", "tte_r1", "tte_r2"]].copy().dropna(subset=["tte_r1", "tte_r2"])
        if tmp.empty:
            continue
        tmp = tmp.sort_values("day")
        am = tmp["tte_r1"].astype(float).values
        pm = tmp["tte_r2"].astype(float).values
        diff = pm - am
        lo, hi = _bootstrap_ci_mean(diff)
        # one-sample Wilcoxon (or permutation) against 0
        if _HAVE_SCIPY:
            try:
                res = _scipy_stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                pval = float(res.pvalue)
            except Exception:
                # fall back to sign-permutation
                _stat, pval = _paired_test(am, pm, method="perm")
        else:
            _stat, pval = _paired_test(am, pm, method="perm")

        rows.append({
            "bird": _short_label(animal_id),
            "first_month": _first_month(df),
            "diff": diff,
            "mean": float(np.mean(diff)),
            "lo": lo, "hi": hi,
            "p": pval,
        })

    if not rows:
        print("[INFO] No paired diffs to plot.")
        return None

    rows.sort(key=lambda r: (r["first_month"], r["bird"]))
    n_birds = len(rows)
    xs = list(range(n_birds))
    xticklabels = [r["bird"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(12, 1.2 * n_birds), 7))
    cmap = plt.get_cmap("tab10")
    colors = {r["bird"]: cmap(i % 10) for i, r in enumerate(rows)}

    ymax = -np.inf
    ymin = np.inf
    for i, r in enumerate(rows):
        clr = colors[r["bird"]]
        d = np.asarray(r["diff"], float)
        j = (np.random.rand(len(d)) - 0.5) * 0.35
        ax.scatter(i + j, d, s=28, alpha=0.9, color=clr, edgecolors="none", zorder=2)
        # mean ± 95% bootstrap CI
        ax.errorbar(i, r["mean"],
                    yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                    fmt="o", color=clr, elinewidth=1.6, capsize=4, markersize=6, zorder=3)
        ymax = max(ymax, np.max(d), r["hi"])
        ymin = min(ymin, np.min(d), r["lo"])

        # bracket vs zero + stars
        y_brk = float(np.max(d)) + 0.04
        ax.plot([i - 0.12, i - 0.12, i + 0.12, i + 0.12],
                [y_brk-0.01, y_brk, y_brk, y_brk-0.01],
                color=clr, linewidth=1.2, alpha=0.9)
        ax.text(i, y_brk + 0.02, _p_to_stars(r["p"]),
                ha="center", va="bottom", fontsize=11, color=clr)

    ax.axhline(0, color="k", linewidth=1.0, alpha=0.5, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_ylabel("PM − AM  (bits)")
    ax.set_title(
        "AM vs PM TTE — paired differences per bird\n"
        f"Points=jittered per-day diffs; dot=mean ± 95% CI; "
        f"Wilcoxon/sign-permutation vs 0.  Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved significance (paired differences) figure → {save_path}")
    return save_path

def _plot_am_pm_two_distributions(
    bird_summary_map: Dict[str, pd.DataFrame],
    *,
    save_path: Path,
    range1: str,
    range2: str,
    batch_size: int,
) -> Optional[Path]:
    """
    For each bird, show *two* distributions: AM and PM per-day TTE.
    - Points jittered, with faint lines connecting paired days.
    - Mean ± 95% bootstrap CI shown for each group.
    - Paired Wilcoxon (or permutation) p-value annotated between groups.
    """
    rows = []
    for animal_id, df in bird_summary_map.items():
        if df is None or df.empty:
            continue
        tmp = df[["day", "tte_r1", "tte_r2"]].copy().dropna(subset=["tte_r1", "tte_r2"])
        if tmp.empty:
            continue
        tmp = tmp.sort_values("day")
        am = tmp["tte_r1"].astype(float).values
        pm = tmp["tte_r2"].astype(float).values
        stat, p = _paired_test(am, pm, method="wilcoxon")
        lo_am, hi_am = _bootstrap_ci_mean(am)
        lo_pm, hi_pm = _bootstrap_ci_mean(pm)
        rows.append({
            "bird": _short_label(animal_id),
            "first_month": _first_month(df),
            "am": am, "pm": pm, "p": p,
            "mean_am": float(np.mean(am)), "mean_pm": float(np.mean(pm)),
            "lo_am": lo_am, "hi_am": hi_am,
            "lo_pm": lo_pm, "hi_pm": hi_pm,
        })

    if not rows:
        print("[INFO] No per-day paired AM/PM data to plot.")
        return None

    rows.sort(key=lambda r: (r["first_month"], r["bird"]))
    n_birds = len(rows)
    xs_am = [2*i for i in range(n_birds)]
    xs_pm = [2*i + 1 for i in range(n_birds)]
    xticks = [v for pair in zip(xs_am, xs_pm) for v in pair]
    xticklabels = [lab for r in rows for lab in (f"{r['bird']}\nAM", f"{r['bird']}\nPM")]

    fig, ax = plt.subplots(figsize=(max(14, 1.8 * n_birds), 7))
    cmap = plt.get_cmap("tab10")
    colors = {r["bird"]: cmap(i % 10) for i, r in enumerate(rows)}

    ymax = -np.inf
    for i, r in enumerate(rows):
        clr = colors[r["bird"]]
        x_am = xs_am[i]; x_pm = xs_pm[i]
        am = r["am"];  pm = r["pm"]

        j_am = (np.random.rand(len(am)) - 0.5) * 0.2
        j_pm = (np.random.rand(len(pm)) - 0.5) * 0.2
        ax.scatter(x_am + j_am, am, s=28, alpha=0.85, color=clr, edgecolors="none", zorder=2)
        ax.scatter(x_pm + j_pm, pm, s=28, alpha=0.85, color=clr, edgecolors="none", zorder=2)

        n = min(len(am), len(pm))
        for k in range(n):
            ax.plot([x_am + j_am[k], x_pm + j_pm[k]], [am[k], pm[k]],
                    color=clr, alpha=0.25, linewidth=0.8, zorder=1)

        ax.errorbar(x_am, r["mean_am"],
                    yerr=[[r["mean_am"] - r["lo_am"]], [r["hi_am"] - r["mean_am"]]],
                    fmt="o", color=clr, elinewidth=1.6, capsize=4, markersize=6, zorder=3)
        ax.errorbar(x_pm, r["mean_pm"],
                    yerr=[[r["mean_pm"] - r["lo_pm"]], [r["hi_pm"] - r["mean_pm"]]],
                    fmt="o", color=clr, elinewidth=1.6, capsize=4, markersize=6, zorder=3)

        pair_max = max(np.max(am), np.max(pm))
        ymax = max(ymax, pair_max, r["hi_am"], r["hi_pm"])
        y_brk = pair_max + 0.06
        ax.plot([x_am, x_am, x_pm, x_pm], [y_brk-0.01, y_brk, y_brk, y_brk-0.01],
                color=clr, linewidth=1.2, alpha=0.9)
        ax.text((x_am + x_pm)/2.0, y_brk + 0.02, _p_to_stars(r["p"]),
                ha="center", va="bottom", fontsize=11, color=clr)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_ylabel("Total Transition Entropy (bits)")
    ax.set_title(
        "AM vs PM TTE Distributions per Bird (paired)\n"
        f"Points=jittered days with paired lines; dot=mean ± 95% CI; "
        f"Range1={range1}, Range2={range2}, batch_size={batch_size}"
    )

    handles = [plt.Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=colors[r["bird"]], markersize=6,
                          label=f"{r['bird']}  (n={min(len(r['am']), len(r['pm']))}, p={r['p']:.3g})")
               for r in rows]
    ax.legend(handles=handles, title="Bird (paired test results)",
              bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved AM/PM two-distributions significance figure → {save_path}")
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
    make_daily_points: bool = True,
    make_significance_plots: bool = True,             # NEW: make both significance plots
    show: bool = False,
) -> BatchResults:
    """
    1) Finds JSON pairs in subfolders of `parent_dir`.
    2) For each bird, calls run_set_hour_and_batch_TTE_by_day(...) to generate *individual* figures.
    3) Builds aggregate AM vs PM plot (paired x-ticks; mean ± SD).
    4) Optionally builds:
         - monthly lines plot (year ignored),
         - daily per-day composite scatter by month,
         - BOTH significance plots:
             a) distribution of PM−AM per bird (one distribution per bird),
             b) two distributions per bird (AM vs PM) with paired lines.
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
        return BatchResults(
            per_bird=per_bird,
            am_pm_figure_path=None,
            monthly_figure_path=None,
            daily_points_figure_path=None,
            significance_diffs_figure_path=None,
            significance_groups_figure_path=None,
        )

    # Decide where to save the aggregate figures
    base_dir = Path(save_dir) if save_dir is not None else parent
    if fig_subdir:
        base_dir = base_dir / fig_subdir
    _ensure_dir(base_dir)

    # Build aggregate inputs: animal_id -> summary_df
    bird_summary_map = {aid: br.summary_df for aid, br in per_bird.items()}

    # Aggregate AM vs PM (single figure with paired x-ticks and mean ± SD)
    ampm_name = (
        f"ALL_BIRDS_AM_vs_PM_TTE_PAIRED__"
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
        show_std=True,
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

    # Optional: daily points by month
    daily_points_path = None
    if make_daily_points:
        daily_name = (
            f"ALL_BIRDS_daily_TTE_points_by_month__"
            f"R1_{range1.replace(':','').replace('-','_')}__"
            f"R2_{range2.replace(':','').replace('-','_')}__N{batch_size}.png"
        )
        daily_points_path = base_dir / daily_name
        daily_points_path = _plot_daily_points_by_month_all_birds(
            bird_summary_map,
            save_path=daily_points_path,
            range1=range1,
            range2=range2,
            batch_size=batch_size,
        )

    # Optional: BOTH significance plots
    signif_diffs_path = None
    signif_groups_path = None
    if make_significance_plots:
        diffs_name = (
            f"ALL_BIRDS_AM_vs_PM_TTE_SIGNIFICANCE_DIFFS__"
            f"R1_{range1.replace(':','').replace('-','_')}__"
            f"R2_{range2.replace(':','').replace('-','_')}__N{batch_size}.png"
        )
        groups_name = (
            f"ALL_BIRDS_AM_vs_PM_TTE_SIGNIFICANCE_GROUPS__"
            f"R1_{range1.replace(':','').replace('-','_')}__"
            f"R2_{range2.replace(':','').replace('-','_')}__N{batch_size}.png"
        )
        signif_diffs_path = base_dir / diffs_name
        signif_groups_path = base_dir / groups_name

        signif_diffs_path = _plot_am_pm_significance_differences(
            bird_summary_map,
            save_path=signif_diffs_path,
            range1=range1,
            range2=range2,
            batch_size=batch_size,
        )
        signif_groups_path = _plot_am_pm_two_distributions(
            bird_summary_map,
            save_path=signif_groups_path,
            range1=range1,
            range2=range2,
            batch_size=batch_size,
        )

    return BatchResults(
        per_bird=per_bird,
        am_pm_figure_path=ampm_path,
        monthly_figure_path=monthly_path,
        daily_points_figure_path=daily_points_path,
        significance_diffs_figure_path=signif_diffs_path,
        significance_groups_figure_path=signif_groups_path,
    )

# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch TTE wrapper for multiple birds (individual + aggregate + significance figures)."
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
    parser.add_argument("--no-daily-points", action="store_true",
                        help="Skip the daily points by month figure.")
    parser.add_argument("--no-significance", action="store_true",
                        help="Skip both significance plots.")
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
        make_significance_plots=not args.no_significance,
        show=args.show,
    )
    print("AM/PM aggregate              :", out.am_pm_figure_path)
    print("Monthly figure               :", out.monthly_figure_path)
    print("Daily points by month figure :", out.daily_points_figure_path)
    print("Significance (diffs) figure  :", out.significance_diffs_figure_path)
    print("Significance (groups) figure :", out.significance_groups_figure_path)


"""
from pathlib import Path
import importlib, batch_set_hour_and_batch_TTE as batch_mod
importlib.reload(batch_mod)
from batch_set_hour_and_batch_TTE import run_batch_set_hour_and_batch_TTE

res = run_batch_set_hour_and_batch_TTE(
    parent_dir="/Users/mirandahulsey-vincent/Desktop/SfN_baseline_analysis",
    range1="05:00-12:00",
    range2="12:00-19:00",
    batch_size=5,
    min_required_per_range=5,
    only_song_present=True,
    exclude_single_label=False,
    fig_subdir="batch_figures",
    make_significance_plots=True,  # makes BOTH significance figures
    show=False,
)

print("Diffs plot :", res.significance_diffs_figure_path)
print("Groups plot:", res.significance_groups_figure_path)

"""