#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bc_cluster_qc_and_summaries_v9.py

Wrapper around export_equal_umap_cluster_spectrograms_v8_run_balanced_phrase_normalized.py
that adds:

1) per-cluster Bhattacharyya-coefficient summaries for one bird,
2) bird-level boxplots across all qualifying clusters,
3) bird-level boxplots across high-variance phrase-duration clusters only,
4) fixed-timescale representative spectrogram QC panels (e.g. all rows 0->2 s).

Typical usage:
  python bc_cluster_qc_and_summaries_v9.py \
    --npz-path /Volumes/my_own_SSD/updated_AreaX_outputs/USA5443/USA5443.npz \
    --metadata-excel-path /Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx \
    --spectrogram-script /Users/.../pre_post_syllable_sample_spectrograms_long_rows_with_bouts_v7.py \
    --phrase-csv /Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv \
    --out-dir /Volumes/my_own_SSD/updated_AreaX_outputs/bc_cluster_qc_outputs \
    --animal-id USA5443 \
    --period-mode early_late_pre_post \
    --bc-analysis-mode run_balanced_phrase_normalized \
    --min-balanced-duration-s 2.0 \
    --phase-bins-per-run 25 \
    --min-runs-per-group 20 \
    --max-runs-per-group 200
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import argparse
import importlib.util
import math
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

SCRIPT_VERSION = "bc_cluster_qc_and_summaries_v9"
DEFAULT_V8_SCRIPT = "/mnt/data/export_equal_umap_cluster_spectrograms_v8_run_balanced_phrase_normalized.py"


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_list_arg(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def _stars_from_p(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return float("nan"), "n.s."
    try:
        p = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except Exception:
        p = float("nan")
    return p, _stars_from_p(p)


def _compute_cluster_bcs(
    *,
    embedding_xy: np.ndarray,
    selected_by_group: Mapping[str, np.ndarray],
    density_bins: int,
) -> Dict[str, float]:
    xy = np.asarray(embedding_xy, dtype=float)
    x_min = float(np.nanmin(xy[:, 0]))
    x_max = float(np.nanmax(xy[:, 0]))
    y_min = float(np.nanmin(xy[:, 1]))
    y_max = float(np.nanmax(xy[:, 1]))
    pad_x = max(1e-6, (x_max - x_min) * 0.04)
    pad_y = max(1e-6, (y_max - y_min) * 0.04)
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, int(density_bins) + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, int(density_bins) + 1)

    out = {
        "bc_pre": np.nan,
        "bc_post": np.nan,
        "bc_prepost": np.nan,
    }
    keys = set(selected_by_group.keys())
    if not {"early_pre", "late_pre", "early_post", "late_post"}.issubset(keys):
        return out

    ep = np.asarray(selected_by_group["early_pre"], dtype=int)
    lp = np.asarray(selected_by_group["late_pre"], dtype=int)
    eo = np.asarray(selected_by_group["early_post"], dtype=int)
    lo = np.asarray(selected_by_group["late_post"], dtype=int)

    hist_ep = v8._compute_hist_density(xy, ep, x_edges=x_edges, y_edges=y_edges)
    hist_lp = v8._compute_hist_density(xy, lp, x_edges=x_edges, y_edges=y_edges)
    hist_eo = v8._compute_hist_density(xy, eo, x_edges=x_edges, y_edges=y_edges)
    hist_lo = v8._compute_hist_density(xy, lo, x_edges=x_edges, y_edges=y_edges)
    out["bc_pre"] = v8._hist_bc(hist_ep, hist_lp)
    out["bc_post"] = v8._hist_bc(hist_eo, hist_lo)

    hist_pre_all = v8._compute_hist_density(xy, np.concatenate([ep, lp]), x_edges=x_edges, y_edges=y_edges)
    hist_post_all = v8._compute_hist_density(xy, np.concatenate([eo, lo]), x_edges=x_edges, y_edges=y_edges)
    out["bc_prepost"] = v8._hist_bc(hist_pre_all, hist_post_all)
    return out


def _load_selected_bins(selected_csv: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(selected_csv)
    if "group" not in df.columns or "timebin_index" not in df.columns:
        raise KeyError(f"Expected columns 'group' and 'timebin_index' in {selected_csv}")
    out: Dict[str, np.ndarray] = {}
    for group_name, sub in df.groupby("group", sort=False):
        out[str(group_name)] = sub["timebin_index"].astype(int).to_numpy()
    return out


def plot_bc_boxplot_for_cluster_set(
    *,
    bc_df: pd.DataFrame,
    title: str,
    out_png: Path,
    subtitle: str = "",
    dpi: int = 200,
) -> None:
    out_png = Path(out_png)
    _safe_mkdir(out_png.parent)
    if bc_df.empty:
        raise ValueError("No rows available for boxplot.")

    pre = pd.to_numeric(bc_df["bc_pre"], errors="coerce").to_numpy(dtype=float)
    post = pd.to_numeric(bc_df["bc_post"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    if len(pre) == 0:
        raise ValueError("No finite pre/post BC values available for boxplot.")

    p, stars = _paired_wilcoxon(pre, post)

    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    bp = ax.boxplot(
        [pre, post],
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=0),
        boxprops=dict(facecolor="#f2f2f2", edgecolor="black", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.6),
        capprops=dict(color="black", linewidth=1.6),
    )
    # Colored median lines to mimic prior style.
    med_pre = float(np.nanmedian(pre))
    med_post = float(np.nanmedian(post))
    ax.hlines(med_pre, 1 - 0.275, 1 + 0.275, colors="red", linewidth=2.0, zorder=4)
    ax.hlines(med_post, 2 - 0.275, 2 + 0.275, colors="#5e2ca5", linewidth=2.0, zorder=4)

    # light paired lines
    for a, b in zip(pre, post):
        ax.plot([1, 2], [a, b], color="0.78", linewidth=0.9, zorder=1)
    ax.scatter(np.repeat(1.0, len(pre)), pre, s=18, color="red", alpha=0.55, zorder=3)
    ax.scatter(np.repeat(2.0, len(post)), post, s=18, color="#5e2ca5", alpha=0.55, zorder=3)

    y_top = max(np.nanmax(pre), np.nanmax(post))
    y_bottom = min(np.nanmin(pre), np.nanmin(post))
    yr = max(0.02, y_top - y_bottom)
    yb = y_top + 0.08 * yr
    ax.plot([1, 1, 2, 2], [yb - 0.02*yr, yb, yb, yb - 0.02*yr], color="black", linewidth=1.6)
    ax.text(1.5, yb + 0.015*yr, stars, ha="center", va="bottom", fontsize=18)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pre", "Post"], fontsize=16)
    ax.set_ylabel("Bhattacharyya coefficient\n(early vs. late, equal selected time bins)", fontsize=15)
    main = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(main, fontsize=18, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(0.35, 2.65)
    ax.set_ylim(max(0.0, y_bottom - 0.08*yr), yb + 0.12*yr)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)



def _fixed_duration_panel(
    *,
    S_FxT: np.ndarray,
    context_start: int,
    panel_n_bins: int,
) -> np.ndarray:
    panel = np.full((S_FxT.shape[0], int(panel_n_bins)), np.nan, dtype=float)
    context_start = int(context_start)
    end = min(S_FxT.shape[1], context_start + int(panel_n_bins))
    n = max(0, end - context_start)
    if n > 0:
        panel[:, :n] = S_FxT[:, context_start:end].astype(float)
    return panel


def plot_fixed_timescale_full_run_contexts(
    *,
    S_FxT: np.ndarray,
    run_df: pd.DataFrame,
    animal_id: str,
    cluster_id: int,
    group_name: str,
    out_png: Path,
    seconds_per_bin: float,
    fixed_panel_duration_s: float,
    pre_context_bins: int,
    max_examples: int,
    example_mode: str,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
    rng: np.random.Generator,
) -> None:
    out_png = Path(out_png)
    _safe_mkdir(out_png.parent)
    chosen = v8._choose_run_rows_for_context_examples(
        run_df,
        max_examples=int(max_examples),
        mode=str(example_mode),
        rng=rng,
    )
    if chosen.empty:
        raise ValueError(f"No rows available for fixed-timescale context plot: {group_name}")

    panel_n_bins = max(1, int(round(float(fixed_panel_duration_s) / float(seconds_per_bin))))
    n_rows = len(chosen)
    fig_h = max(2.0, 1.28 * n_rows + 0.9)
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, fig_h), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        run_start = int(row["run_start_bin"])
        run_end = int(row["run_end_bin_exclusive"])
        context_start = max(0, run_start - int(pre_context_bins))
        panel = _fixed_duration_panel(
            S_FxT=S_FxT,
            context_start=context_start,
            panel_n_bins=panel_n_bins,
        )
        ax.imshow(
            panel,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, float(fixed_panel_duration_s), 0, panel.shape[0]],
        )
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_xlim(0, float(fixed_panel_duration_s))

        # Blue = full run, red = selected segments, positioned relative to context_start.
        blue_start = max(run_start, context_start)
        blue_end = min(run_end, context_start + panel_n_bins)
        if blue_end > blue_start:
            ax.axvspan(
                (blue_start - context_start) * float(seconds_per_bin),
                (blue_end - context_start) * float(seconds_per_bin),
                color="tab:blue",
                alpha=0.16,
            )

        seg_starts = v8._parse_semicolon_ints(row.get("selected_segment_starts", ""))
        seg_ends = v8._parse_semicolon_ints(row.get("selected_segment_ends_exclusive", ""))
        for s, e in zip(seg_starts, seg_ends):
            s2 = max(int(s), context_start)
            e2 = min(int(e), context_start + panel_n_bins)
            if e2 > s2:
                ax.axvspan(
                    (s2 - context_start) * float(seconds_per_bin),
                    (e2 - context_start) * float(seconds_per_bin),
                    color="tab:red",
                    alpha=0.28,
                )

        ax.set_ylabel(
            f"run {int(row['run_id'])}\n{int(row['run_n_bins'])} bins\ncoverage={float(row['coverage_fraction']):.2f}",
            rotation=0,
            labelpad=58,
            va="center",
            fontsize=8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Original recording context (s)")
    fig.suptitle(
        f"{animal_id} — label {int(cluster_id):02d} — {group_name}\n"
        f"Representative full HDBSCAN phrase-label runs; blue = full run, red = equal UMAP-selected bins; fixed x-axis 0–{float(fixed_panel_duration_s):.1f} s",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.12)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)



def run_pipeline(
    *,
    npz_path: Path,
    metadata_excel_path: Path,
    spectrogram_script: Path,
    out_dir: Path,
    animal_id: Optional[str],
    phrase_csv: Optional[Path],
    top_fraction: float,
    post_group_name: str,
    top_min_n_phrases: int,
    include_noise: bool,
    only_cluster_ids: Optional[Sequence[int]],
    vocalization_only: bool,
    period_mode: str,
    treatment_day_assignment: str,
    early_late_split_method: str,
    bc_analysis_mode: str,
    min_points_per_group: int,
    max_points_per_group: Optional[int],
    sample_mode: str,
    min_runs_per_group: int,
    max_runs_per_group: Optional[int],
    phase_bins_per_run: int,
    min_full_run_duration_ms: float,
    min_run_group_fraction: float,
    run_sample_mode: str,
    allow_repeat_phase_bins: bool,
    save_phase_normalized_bc: bool,
    seed: int,
    bins_per_spectrogram_row: int,
    max_rows_per_spectrogram: Optional[int],
    spectrogram_source_mode: str,
    save_representative_full_run_contexts: bool,
    max_expanded_full_run_bins: Optional[int],
    full_run_context_bins: int,
    full_run_max_context_bins: int,
    full_run_max_examples: int,
    full_run_example_mode: str,
    seconds_per_bin: float,
    x_tick_step_s: float,
    figure_width: float,
    row_height: float,
    subplot_hspace: float,
    cmap: str,
    contrast_percentiles: Optional[Tuple[float, float]],
    dpi: int,
    umap_density_bins: int,
    dry_run: bool,
    min_balanced_duration_s: float,
    fixed_panel_duration_s: float,
    fixed_panel_pre_context_bins: int,
    fixed_panel_max_examples: int,
) -> Path:
    summary_csv = v8.export_equal_umap_cluster_spectrograms_for_one_bird(
        npz_path=npz_path,
        metadata_excel_path=metadata_excel_path,
        spectrogram_script=spectrogram_script,
        out_dir=out_dir,
        animal_id=animal_id,
        include_noise=include_noise,
        only_cluster_ids=only_cluster_ids,
        vocalization_only=vocalization_only,
        period_mode=period_mode,
        treatment_day_assignment=treatment_day_assignment,
        early_late_split_method=early_late_split_method,
        bc_analysis_mode=bc_analysis_mode,
        min_points_per_group=min_points_per_group,
        max_points_per_group=max_points_per_group,
        sample_mode=sample_mode,
        min_runs_per_group=min_runs_per_group,
        max_runs_per_group=max_runs_per_group,
        phase_bins_per_run=phase_bins_per_run,
        min_full_run_duration_ms=min_full_run_duration_ms,
        min_run_group_fraction=min_run_group_fraction,
        run_sample_mode=run_sample_mode,
        allow_repeat_phase_bins=allow_repeat_phase_bins,
        save_phase_normalized_bc=save_phase_normalized_bc,
        seed=seed,
        bins_per_spectrogram_row=bins_per_spectrogram_row,
        max_rows_per_spectrogram=max_rows_per_spectrogram,
        spectrogram_source_mode=spectrogram_source_mode,
        save_representative_full_run_contexts=save_representative_full_run_contexts,
        max_expanded_full_run_bins=max_expanded_full_run_bins,
        full_run_context_bins=full_run_context_bins,
        full_run_max_context_bins=full_run_max_context_bins,
        full_run_max_examples=full_run_max_examples,
        full_run_example_mode=full_run_example_mode,
        seconds_per_bin=seconds_per_bin,
        x_tick_step_s=x_tick_step_s,
        figure_width=figure_width,
        row_height=row_height,
        subplot_hspace=subplot_hspace,
        cmap=cmap,
        contrast_percentiles=contrast_percentiles,
        dpi=dpi,
        umap_density_bins=umap_density_bins,
        dry_run=dry_run,
        top_variance_csv=None,  # evaluate all clusters first
        top_fraction=top_fraction,
        post_group_name=post_group_name,
        top_min_n_phrases=top_min_n_phrases,
    )
    if dry_run:
        return Path(summary_csv)

    summary_df = pd.read_csv(summary_csv)
    animal_id_final = str(summary_df["animal_id"].dropna().astype(str).iloc[0]) if not summary_df.empty else (animal_id or npz_path.stem.split("_")[0])
    animal_out_dir = Path(out_dir).expanduser().resolve() / animal_id_final
    _safe_mkdir(animal_out_dir)

    arr = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(arr["hdbscan_labels"]).astype(int)
    embedding_xy = v8._get_embedding_xy(arr, "embedding_outputs")
    helper = _load_module_from_path(spectrogram_script, "spec_helper_v9")
    S_FxT = helper._orient_spectrogram_to_FxT(np.asarray(arr["s"]), labels)

    # Determine high-variance subset if available.
    high_variance_ids: Optional[set[int]] = None
    if phrase_csv is not None and Path(phrase_csv).exists():
        try:
            hv = v8.read_top_variance_cluster_ids(
                Path(phrase_csv),
                animal_id=animal_id_final,
                top_fraction=float(top_fraction),
                post_group_name=str(post_group_name),
                min_n_phrases=int(top_min_n_phrases),
            )
            high_variance_ids = set(int(x) for x in hv)
        except Exception as e:
            print(f"[WARN] Could not determine high-variance clusters from {phrase_csv}: {e}")
            traceback.print_exc()
            high_variance_ids = None

    cluster_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))

    done_df = summary_df[summary_df.get("status", pd.Series(dtype=str)).astype(str) == "done"].copy()
    for _, row in done_df.iterrows():
        try:
            cluster_id = int(row["cluster_id"])
            selected_csv = Path(str(row["selected_timebins_csv"]))
            if not selected_csv.exists():
                print(f"[WARN] Missing selected CSV for cluster {cluster_id}: {selected_csv}")
                continue
            selected_by_group = _load_selected_bins(selected_csv)
            bc_vals = _compute_cluster_bcs(
                embedding_xy=embedding_xy,
                selected_by_group=selected_by_group,
                density_bins=int(umap_density_bins),
            )
            n_equal = int(pd.to_numeric(row.get("n_equal_selected_per_group", np.nan), errors="coerce"))
            balanced_duration_s = float(n_equal) * float(seconds_per_bin)
            pass_min_duration = bool(balanced_duration_s >= float(min_balanced_duration_s))
            is_high_variance = bool(high_variance_ids is not None and int(cluster_id) in high_variance_ids)
            cluster_rows.append({
                "animal_id": animal_id_final,
                "cluster_id": int(cluster_id),
                "cluster_token": row.get("cluster_token", f"label{int(cluster_id):02d}"),
                "status": row.get("status", "done"),
                "n_equal_selected_per_group": n_equal,
                "n_equal_runs_per_group": pd.to_numeric(row.get("n_equal_runs_per_group", np.nan), errors="coerce"),
                "phase_bins_per_run": pd.to_numeric(row.get("phase_bins_per_run", np.nan), errors="coerce"),
                "balanced_duration_s_per_group": balanced_duration_s,
                "passes_min_balanced_duration": pass_min_duration,
                "is_high_variance_cluster": is_high_variance,
                "bc_pre": bc_vals["bc_pre"],
                "bc_post": bc_vals["bc_post"],
                "bc_prepost": bc_vals["bc_prepost"],
                "umap_png": row.get("umap_png", ""),
                "selected_timebins_csv": str(selected_csv),
                "full_run_mapping_csvs": row.get("full_run_mapping_csvs", ""),
                "representative_full_run_context_pngs": row.get("representative_full_run_context_pngs", ""),
            })

            # Create fixed-timescale QC spectrograms for each group-specific run summary CSV.
            csvs_text = str(row.get("full_run_mapping_csvs", "")).strip()
            group_csvs = [Path(x) for x in csvs_text.split(";") if str(x).strip()]
            vmin, vmax = v8._compute_shared_intensity_limits(S_FxT, selected_by_group, contrast_percentiles)
            for run_csv in group_csvs:
                if not run_csv.exists():
                    continue
                run_df = pd.read_csv(run_csv)
                if run_df.empty:
                    continue
                group_name = str(run_df.get("group_name", pd.Series([run_csv.stem])).iloc[0]) if "group_name" in run_df.columns else run_csv.stem
                out_png = run_csv.with_name(run_csv.stem.replace("_full_runs_from_equal_timebins", f"_fixed_{fixed_panel_duration_s:.1f}s_full_run_contexts") + ".png")
                try:
                    plot_fixed_timescale_full_run_contexts(
                        S_FxT=S_FxT,
                        run_df=run_df,
                        animal_id=animal_id_final,
                        cluster_id=cluster_id,
                        group_name=group_name,
                        out_png=out_png,
                        seconds_per_bin=float(seconds_per_bin),
                        fixed_panel_duration_s=float(fixed_panel_duration_s),
                        pre_context_bins=int(fixed_panel_pre_context_bins),
                        max_examples=int(fixed_panel_max_examples),
                        example_mode=str(full_run_example_mode),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        dpi=int(dpi),
                        rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                    )
                    print(f"[SAVE] {out_png}")
                except Exception as e:
                    print(f"[WARN] Could not save fixed-timescale QC spectrogram for {run_csv}: {e}")
                    traceback.print_exc()
        except Exception as e:
            print(f"[WARN] Could not process cluster summary row: {e}")
            traceback.print_exc()

    bc_df = pd.DataFrame(cluster_rows)
    bc_csv = animal_out_dir / f"{animal_id_final}_cluster_bc_summary.csv"
    bc_df.to_csv(bc_csv, index=False)
    print(f"[SAVE] {bc_csv}")

    # Aggregate summaries / boxplots.
    qual_df = bc_df[bc_df["passes_min_balanced_duration"] == True].copy()
    if not qual_df.empty:
        all_boxplot = animal_out_dir / f"{animal_id_final}_BC_boxplot_all_clusters_min{min_balanced_duration_s:.1f}s.png"
        subtitle = f"all qualifying clusters | minimum balanced duration per group = {float(min_balanced_duration_s):.1f} s | N clusters = {len(qual_df)}"
        plot_bc_boxplot_for_cluster_set(
            bc_df=qual_df,
            title=f"{animal_id_final}: BC across all clusters",
            subtitle=subtitle,
            out_png=all_boxplot,
            dpi=int(dpi),
        )
        print(f"[SAVE] {all_boxplot}")

        stats_rows = [{
            "set_name": "all_clusters",
            "n_clusters": int(len(qual_df)),
            "mean_bc_pre": float(np.nanmean(qual_df["bc_pre"])),
            "mean_bc_post": float(np.nanmean(qual_df["bc_post"])),
            "mean_bc_prepost": float(np.nanmean(qual_df["bc_prepost"])),
            "median_bc_pre": float(np.nanmedian(qual_df["bc_pre"])),
            "median_bc_post": float(np.nanmedian(qual_df["bc_post"])),
            "median_bc_prepost": float(np.nanmedian(qual_df["bc_prepost"])),
        }]

        if "is_high_variance_cluster" in qual_df.columns and qual_df["is_high_variance_cluster"].any():
            hv_df = qual_df[qual_df["is_high_variance_cluster"] == True].copy()
            if not hv_df.empty:
                hv_boxplot = animal_out_dir / f"{animal_id_final}_BC_boxplot_high_variance_clusters_min{min_balanced_duration_s:.1f}s.png"
                hv_subtitle = (
                    f"high-variance phrase-duration clusters only | minimum balanced duration per group = {float(min_balanced_duration_s):.1f} s | "
                    f"N clusters = {len(hv_df)}"
                )
                plot_bc_boxplot_for_cluster_set(
                    bc_df=hv_df,
                    title=f"{animal_id_final}: BC across high-variance clusters",
                    subtitle=hv_subtitle,
                    out_png=hv_boxplot,
                    dpi=int(dpi),
                )
                print(f"[SAVE] {hv_boxplot}")
                stats_rows.append({
                    "set_name": "high_variance_clusters",
                    "n_clusters": int(len(hv_df)),
                    "mean_bc_pre": float(np.nanmean(hv_df["bc_pre"])),
                    "mean_bc_post": float(np.nanmean(hv_df["bc_post"])),
                    "mean_bc_prepost": float(np.nanmean(hv_df["bc_prepost"])),
                    "median_bc_pre": float(np.nanmedian(hv_df["bc_pre"])),
                    "median_bc_post": float(np.nanmedian(hv_df["bc_post"])),
                    "median_bc_prepost": float(np.nanmedian(hv_df["bc_prepost"])),
                })

        stats_csv = animal_out_dir / f"{animal_id_final}_BC_aggregate_stats.csv"
        pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
        print(f"[SAVE] {stats_csv}")
    else:
        print("[WARN] No clusters passed the minimum balanced-duration threshold; no aggregate boxplots were created.")

    return bc_csv



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run one-bird cluster BC/QC exports, then create bird-level BC summaries, "
            "boxplots, and fixed-timescale representative spectrogram QC panels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz-path", required=True)
    p.add_argument("--metadata-excel-path", required=True)
    p.add_argument("--spectrogram-script", required=True)
    p.add_argument("--v8-script", default=DEFAULT_V8_SCRIPT, help="Path to the v8 run-balanced phrase-normalized export script.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--animal-id", default=None)
    p.add_argument("--phrase-csv", default=None, help="Optional phrase-duration CSV used to define high-variance clusters.")
    p.add_argument("--top-fraction", type=float, default=0.30)
    p.add_argument("--post-group-name", default="Post")
    p.add_argument("--top-min-n-phrases", type=int, default=100)

    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--only-cluster-ids", nargs="*", default=None)
    p.add_argument("--no-vocalization-only", action="store_true")

    p.add_argument("--period-mode", choices=["pre_post", "early_late_pre_post"], default="early_late_pre_post")
    p.add_argument("--treatment-day-assignment", choices=["exclude", "pre", "post"], default="exclude")
    p.add_argument("--early-late-split-method", choices=["file_median", "file_half", "timebin_half"], default="file_median")

    p.add_argument("--bc-analysis-mode", choices=["run_balanced_phrase_normalized", "frame_weighted"], default="run_balanced_phrase_normalized")
    p.add_argument("--min-runs-per-group", type=int, default=20)
    p.add_argument("--max-runs-per-group", type=int, default=200)
    p.add_argument("--phase-bins-per-run", type=int, default=25)
    p.add_argument("--min-full-run-duration-ms", type=float, default=100.0)
    p.add_argument("--min-run-group-fraction", type=float, default=0.80)
    p.add_argument("--run-sample-mode", choices=["random", "first", "longest"], default="random")
    p.add_argument("--allow-repeat-phase-bins", action="store_true")
    p.add_argument("--no-phase-normalized-bc", action="store_true")

    p.add_argument("--min-points-per-group", type=int, default=6000)
    p.add_argument("--max-points-per-group", type=int, default=6000)
    p.add_argument("--sample-mode", choices=["random", "first"], default="random")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--bins-per-spectrogram-row", type=int, default=2000)
    p.add_argument("--max-rows-per-spectrogram", type=int, default=None)
    p.add_argument("--spectrogram-source-mode", choices=["selected_timebins", "expanded_full_runs", "both"], default="expanded_full_runs")
    p.add_argument("--max-expanded-full-run-bins", type=int, default=10000)
    p.add_argument("--no-representative-full-run-contexts", action="store_true")
    p.add_argument("--full-run-context-bins", type=int, default=80)
    p.add_argument("--full-run-max-context-bins", type=int, default=1500)
    p.add_argument("--full-run-max-examples", type=int, default=12)
    p.add_argument("--full-run-example-mode", choices=["mixed", "longest", "shortest", "random"], default="mixed")
    p.add_argument("--seconds-per-bin", type=float, default=0.0027)
    p.add_argument("--x-tick-step-s", type=float, default=1.0)
    p.add_argument("--figure-width", type=float, default=24.0)
    p.add_argument("--row-height", type=float, default=2.2)
    p.add_argument("--subplot-hspace", type=float, default=0.005)
    p.add_argument("--cmap", default="gray_r")
    p.add_argument("--contrast-percentiles", nargs=2, type=float, default=[1, 99.5], metavar=("LOW", "HIGH"))
    p.add_argument("--no-contrast-percentiles", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--umap-density-bins", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--min-balanced-duration-s", type=float, default=2.0, help="Minimum balanced duration per group required for a cluster to contribute to aggregate BC summaries.")
    p.add_argument("--fixed-panel-duration-s", type=float, default=2.0, help="Fixed x-axis duration for representative QC spectrogram rows.")
    p.add_argument("--fixed-panel-pre-context-bins", type=int, default=40, help="How much pre-run context to show at the start of each fixed-duration representative QC row.")
    p.add_argument("--fixed-panel-max-examples", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global v8
    v8 = _load_module_from_path(Path(args.v8_script).expanduser().resolve(), "export_equal_umap_cluster_spectrograms_v8")

    only_cluster_ids = None
    if args.only_cluster_ids:
        only_cluster_ids = [int(x) for x in _parse_list_arg(args.only_cluster_ids)]

    max_points_per_group = None if int(args.max_points_per_group) <= 0 else int(args.max_points_per_group)
    max_runs_per_group = None if int(args.max_runs_per_group) <= 0 else int(args.max_runs_per_group)
    max_expanded_full_run_bins = None if int(args.max_expanded_full_run_bins) <= 0 else int(args.max_expanded_full_run_bins)
    contrast = None if bool(args.no_contrast_percentiles) else tuple(float(x) for x in args.contrast_percentiles)

    run_pipeline(
        npz_path=Path(args.npz_path).expanduser().resolve(),
        metadata_excel_path=Path(args.metadata_excel_path).expanduser().resolve(),
        spectrogram_script=Path(args.spectrogram_script).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        animal_id=args.animal_id,
        phrase_csv=(Path(args.phrase_csv).expanduser().resolve() if args.phrase_csv else None),
        top_fraction=float(args.top_fraction),
        post_group_name=str(args.post_group_name),
        top_min_n_phrases=int(args.top_min_n_phrases),
        include_noise=bool(args.include_noise),
        only_cluster_ids=only_cluster_ids,
        vocalization_only=not bool(args.no_vocalization_only),
        period_mode=str(args.period_mode),
        treatment_day_assignment=str(args.treatment_day_assignment),
        early_late_split_method=str(args.early_late_split_method),
        bc_analysis_mode=str(args.bc_analysis_mode),
        min_points_per_group=int(args.min_points_per_group),
        max_points_per_group=max_points_per_group,
        sample_mode=str(args.sample_mode),
        min_runs_per_group=int(args.min_runs_per_group),
        max_runs_per_group=max_runs_per_group,
        phase_bins_per_run=int(args.phase_bins_per_run),
        min_full_run_duration_ms=float(args.min_full_run_duration_ms),
        min_run_group_fraction=float(args.min_run_group_fraction),
        run_sample_mode=str(args.run_sample_mode),
        allow_repeat_phase_bins=bool(args.allow_repeat_phase_bins),
        save_phase_normalized_bc=not bool(args.no_phase_normalized_bc),
        seed=int(args.seed),
        bins_per_spectrogram_row=int(args.bins_per_spectrogram_row),
        max_rows_per_spectrogram=args.max_rows_per_spectrogram,
        spectrogram_source_mode=str(args.spectrogram_source_mode),
        save_representative_full_run_contexts=not bool(args.no_representative_full_run_contexts),
        max_expanded_full_run_bins=max_expanded_full_run_bins,
        full_run_context_bins=int(args.full_run_context_bins),
        full_run_max_context_bins=int(args.full_run_max_context_bins),
        full_run_max_examples=int(args.full_run_max_examples),
        full_run_example_mode=str(args.full_run_example_mode),
        seconds_per_bin=float(args.seconds_per_bin),
        x_tick_step_s=float(args.x_tick_step_s),
        figure_width=float(args.figure_width),
        row_height=float(args.row_height),
        subplot_hspace=float(args.subplot_hspace),
        cmap=str(args.cmap),
        contrast_percentiles=contrast,
        dpi=int(args.dpi),
        umap_density_bins=int(args.umap_density_bins),
        dry_run=bool(args.dry_run),
        min_balanced_duration_s=float(args.min_balanced_duration_s),
        fixed_panel_duration_s=float(args.fixed_panel_duration_s),
        fixed_panel_pre_context_bins=int(args.fixed_panel_pre_context_bins),
        fixed_panel_max_examples=int(args.fixed_panel_max_examples),
    )


if __name__ == "__main__":
    main()
