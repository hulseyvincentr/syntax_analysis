#!/usr/bin/env python3
"""
plot_peak_spcc_stuttered_vs_nonstuttered_results.py

Visualize outputs from batch_peak_spcc_stuttered_vs_nonstuttered_medial_lateral.py.

This script is intentionally tolerant of failed/partial runs:
- It always saves a run-status summary plot.
- It recomputes bird-level means from the all-birds summary CSV when needed.
- If a metric has no finite values, it writes that on the plot rather than making a blank 0-1 plot.

Example
-------
cd "/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files"

python plot_peak_spcc_stuttered_vs_nonstuttered_results.py \
  --results-dir "/Volumes/my_own_SSD/updated_AreaX_outputs/medial_lateral_peak_spcc_stuttered_vs_nonstuttered"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = {
    "stereotypy_delta_post_minus_pre": {
        "title": "Within-condition stereotypy change",
        "ylabel": "Δ median peak SPCC: post-post - pre-pre",
        "short": "stereotypy_delta",
        "interpretation": "negative = post-lesion renditions are less internally similar",
    },
    "identity_shift_delta_prepost_minus_prepre": {
        "title": "Pre/post identity shift",
        "ylabel": "Δ median peak SPCC: pre-post - pre-pre",
        "short": "identity_shift",
        "interpretation": "negative = post renditions are less similar to pre form",
    },
}

STATUS_COLORS = {
    "completed": "C0",
    "skipped_existing": "C1",
    "failed": "C3",
    "dry_run": "C2",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_or_none(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_status(s: object) -> str:
    if pd.isna(s):
        return "missing"
    return str(s)


def get_completed(summary: pd.DataFrame) -> pd.DataFrame:
    if "run_status" not in summary.columns:
        return summary.copy()
    return summary[summary["run_status"].isin(["completed", "skipped_existing"])].copy()


def recompute_bird_level(summary: pd.DataFrame) -> pd.DataFrame:
    ok = get_completed(summary)
    metric_cols = [m for m in METRICS if m in ok.columns]
    if ok.empty or not metric_cols:
        return pd.DataFrame(columns=["animal_id", "stutter_status", *METRICS.keys()])
    ok = coerce_numeric(ok, metric_cols)
    bird = ok.groupby(["animal_id", "stutter_status"], as_index=False)[metric_cols].mean()
    return bird


def finite_values(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def set_metric_ylim(ax, vals: np.ndarray, zero_line: bool = True):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ax.set_ylim(-0.1, 0.1)
        return
    ymin = float(np.nanmin(vals))
    ymax = float(np.nanmax(vals))
    if zero_line:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    if np.isclose(ymin, ymax):
        pad = max(0.02, abs(ymin) * 0.2)
    else:
        pad = max(0.02, 0.12 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)


def add_no_data_message(ax, message: str):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_run_status(summary: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if "run_status" not in summary.columns or summary.empty:
        add_no_data_message(ax, "No run_status column found")
    else:
        counts = summary["run_status"].map(clean_status).value_counts().sort_index()
        colors = [STATUS_COLORS.get(s, "C7") for s in counts.index]
        ax.bar(np.arange(len(counts)), counts.values, color=colors)
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels(counts.index, rotation=25, ha="right")
        ax.set_ylabel("Number of bird × label jobs")
        ax.set_title("Peak SPCC batch run status")
        for i, val in enumerate(counts.values):
            ax.text(i, val, str(int(val)), ha="center", va="bottom")
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_segment_counts(summary: pd.DataFrame, out_png: Path):
    cols = ["n_segments_combined_pre", "n_segments_combined_post"]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    if summary.empty or not all(c in summary.columns for c in cols):
        add_no_data_message(ax, "No segment-count columns found")
    else:
        ok = summary.copy()
        for c in cols:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")
        ok = ok[ok[cols].notna().any(axis=1)].copy()
        if ok.empty:
            add_no_data_message(ax, "No segment counts available yet")
        else:
            labels = [f"{r.animal_id}:{r.cluster_label}\n{r.stutter_status}" for r in ok.itertuples()]
            x = np.arange(len(ok))
            width = 0.4
            ax.bar(x - width/2, ok["n_segments_combined_pre"].fillna(0), width, label="pre usable segments")
            ax.bar(x + width/2, ok["n_segments_combined_post"].fillna(0), width, label="post usable segments")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
            ax.set_ylabel("Usable segmented syllables")
            ax.set_title("Usable segments by bird × syllable")
            ax.legend(frameon=True)
            ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_bird_metric(bird: pd.DataFrame, metric: str, out_png: Path):
    info = METRICS[metric]
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    if bird.empty or metric not in bird.columns:
        add_no_data_message(ax, f"No bird-level values for\n{metric}")
    else:
        b = bird.copy()
        b[metric] = pd.to_numeric(b[metric], errors="coerce")
        b = b[b[metric].notna()].copy()
        if b.empty:
            add_no_data_message(ax, f"No finite bird-level values for\n{metric}")
        else:
            xmap = {"nonstuttered": 0, "stuttered": 1}
            for animal, sub in b.groupby("animal_id", sort=True):
                sub = sub[sub["stutter_status"].isin(xmap)].copy()
                sub["x"] = sub["stutter_status"].map(xmap)
                sub = sub.sort_values("x")
                if len(sub) == 2:
                    ax.plot(sub["x"], sub[metric], alpha=0.45, lw=1.5)
                ax.scatter(sub["x"], sub[metric], s=60, alpha=0.85)
                for _, r in sub.iterrows():
                    ax.text(r["x"] + 0.035, r[metric], str(animal), fontsize=8, va="center")
            ax.axhline(0, ls="--", lw=1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Non-stuttered", "Stuttered"])
            ax.set_ylabel(info["ylabel"])
            ax.set_title(info["title"] + " by bird")
            ax.text(0.02, 0.02, info["interpretation"], transform=ax.transAxes, fontsize=9, va="bottom")
            set_metric_ylim(ax, b[metric].to_numpy())
            ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_syllable_metric_by_status(summary: pd.DataFrame, metric: str, out_png: Path):
    info = METRICS[metric]
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    if summary.empty or metric not in summary.columns:
        add_no_data_message(ax, f"No syllable-level values for\n{metric}")
    else:
        ok = get_completed(summary)
        ok[metric] = pd.to_numeric(ok[metric], errors="coerce")
        ok = ok[ok[metric].notna()].copy()
        if ok.empty:
            add_no_data_message(ax, f"No finite completed values for\n{metric}")
        else:
            status_order = ["nonstuttered", "stuttered"]
            rng = np.random.default_rng(0)
            all_vals = []
            for xi, status in enumerate(status_order):
                vals = ok.loc[ok["stutter_status"] == status, metric].dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                all_vals.extend(vals.tolist())
                ax.boxplot(vals, positions=[xi], widths=0.45, showfliers=False)
                jitter = rng.normal(0, 0.035, size=vals.size)
                ax.scatter(np.full(vals.size, xi) + jitter, vals, s=55, alpha=0.8)
                sub = ok.loc[ok["stutter_status"] == status].copy()
                for _, r in sub.iterrows():
                    y = r[metric]
                    if np.isfinite(y):
                        ax.text(xi + 0.06, y, f"{r['animal_id']}:{r['cluster_label']}", fontsize=7, va="center")
            ax.axhline(0, ls="--", lw=1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Non-stuttered", "Stuttered"])
            ax.set_ylabel(info["ylabel"])
            ax.set_title(info["title"] + " by syllable")
            set_metric_ylim(ax, np.asarray(all_vals))
            ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variance_vs_metric(summary: pd.DataFrame, metric: str, out_png: Path):
    info = METRICS[metric]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    if summary.empty or metric not in summary.columns or "phrase_variance_delta_ms2" not in summary.columns:
        add_no_data_message(ax, f"Missing {metric} or phrase_variance_delta_ms2")
    else:
        ok = get_completed(summary)
        ok[metric] = pd.to_numeric(ok[metric], errors="coerce")
        ok["phrase_variance_delta_ms2"] = pd.to_numeric(ok["phrase_variance_delta_ms2"], errors="coerce")
        ok = ok[np.isfinite(ok[metric]) & np.isfinite(ok["phrase_variance_delta_ms2"]) & (ok["phrase_variance_delta_ms2"] > 0)].copy()
        if ok.empty:
            add_no_data_message(ax, f"No finite completed values for\n{metric}")
        else:
            for status, sub in ok.groupby("stutter_status", sort=True):
                ax.scatter(sub["phrase_variance_delta_ms2"], sub[metric], s=65, alpha=0.8, label=status)
                for _, r in sub.iterrows():
                    ax.text(r["phrase_variance_delta_ms2"], r[metric], f"{r['animal_id']}:{r['cluster_label']}", fontsize=7)
            ax.set_xscale("log")
            ax.axhline(0, ls="--", lw=1)
            ax.set_xlabel("Post - pre phrase-duration variance change (ms²)")
            ax.set_ylabel(info["ylabel"])
            ax.set_title("Phrase-duration variance change vs " + info["title"].lower())
            ax.legend(frameon=True)
            ax.grid(alpha=0.25)
            set_metric_ylim(ax, ok[metric].to_numpy())
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_text_summary(summary: pd.DataFrame, bird: pd.DataFrame, tests: pd.DataFrame | None, out_txt: Path):
    lines = []
    lines.append("Peak SPCC visualization summary")
    lines.append("================================")
    lines.append("")
    lines.append(f"Total bird × label rows: {len(summary)}")
    if "run_status" in summary.columns:
        lines.append("Run status counts:")
        for status, n in summary["run_status"].map(clean_status).value_counts().items():
            lines.append(f"  {status}: {n}")
    lines.append("")
    if "error" in summary.columns and summary["error"].notna().any():
        lines.append("Errors:")
        err_counts = summary["error"].dropna().value_counts()
        for err, n in err_counts.items():
            lines.append(f"  {n} × {err}")
        lines.append("")
    for metric, info in METRICS.items():
        vals = finite_values(bird, metric) if bird is not None else np.array([])
        lines.append(metric)
        if vals.size:
            lines.append(f"  n bird-level values: {vals.size}")
            lines.append(f"  mean: {np.nanmean(vals):.4f}")
            lines.append(f"  median: {np.nanmedian(vals):.4f}")
        else:
            lines.append("  no finite bird-level values")
        lines.append(f"  interpretation: {info['interpretation']}")
        lines.append("")
    if tests is not None and not tests.empty:
        lines.append("Wilcoxon tests:")
        lines.append(tests.to_string(index=False))
    out_txt.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Visualize peak SPCC stuttered vs non-stuttered results.")
    parser.add_argument("--results-dir", required=True, help="Folder containing peak SPCC output CSVs.")
    parser.add_argument("--out-dir", default=None, help="Folder to save plots. Defaults to <results-dir>/visualization_plots.")
    parser.add_argument("--summary-csv", default="all_birds_stuttered_vs_nonstuttered_peak_spcc_summary.csv")
    parser.add_argument("--bird-level-csv", default="bird_level_stuttered_vs_nonstuttered_peak_spcc.csv")
    parser.add_argument("--tests-csv", default="group_level_peak_spcc_wilcoxon_tests.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = ensure_dir(args.out_dir or results_dir / "visualization_plots")

    summary_path = results_dir / args.summary_csv
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {summary_path}")
    summary = pd.read_csv(summary_path)

    bird_path = results_dir / args.bird_level_csv
    bird = read_csv_or_none(bird_path)
    if bird is None or bird.empty:
        bird = recompute_bird_level(summary)
    else:
        # If the existing bird-level file is all NaN because the run failed, this still lets plots say so clearly.
        pass

    tests = read_csv_or_none(results_dir / args.tests_csv)

    plot_run_status(summary, out_dir / "01_run_status_counts.png")
    plot_segment_counts(summary, out_dir / "02_usable_segment_counts.png")

    for metric, info in METRICS.items():
        plot_paired_bird_metric(bird, metric, out_dir / f"03_bird_level_paired_{info['short']}.png")
        plot_syllable_metric_by_status(summary, metric, out_dir / f"04_syllable_level_{info['short']}_by_status.png")
        plot_variance_vs_metric(summary, metric, out_dir / f"05_variance_delta_vs_{info['short']}.png")

    # Save cleaned/recomputed bird-level table so it is easy to inspect.
    bird.to_csv(out_dir / "recomputed_bird_level_peak_spcc.csv", index=False)
    write_text_summary(summary, bird, tests, out_dir / "visualization_summary.txt")

    print(f"[DONE] Saved plots to: {out_dir}")
    print(f"[SAVED] {out_dir / 'visualization_summary.txt'}")


if __name__ == "__main__":
    main()
