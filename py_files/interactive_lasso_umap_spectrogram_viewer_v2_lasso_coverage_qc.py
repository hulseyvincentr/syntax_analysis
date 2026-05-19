#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_lasso_umap_spectrogram_viewer.py

Interactive UMAP lasso tool for TweetyBERT/decoder outputs.

Purpose
-------
Open an interactive UMAP plot, lasso/circle a region, then save representative
spectrogram examples from the exact selected time bins.

This is meant for QC of HDBSCAN/UMAP labels: you can restrict the lasso to a
single HDBSCAN label, draw around a subregion of that label, and inspect whether
those UMAP points correspond to a full phrase/syllable-like unit or a sub-part
of the phrase label.

Expected NPZ keys by default
----------------------------
- embedding_outputs : (N, 2) UMAP coordinates
- hdbscan_labels    : (N,) cluster/phrase labels
- s                 : spectrogram array, either (N, F) or (F, N)
- file_indices      : (N,) optional recording/file identity per time bin
- vocalization      : (N,) optional vocalization mask

Outputs per lasso selection
---------------------------
<out-dir>/<animal_id>/lasso_selection_001/
    <animal>_lasso001_selected_timebins.csv
    <animal>_lasso001_umap_selection.png
    <animal>_lasso001_run_context_spectrograms.png
    <animal>_lasso001_stitched_selected_spectrogram.png
    <animal>_lasso001_overlapping_runs.csv
    <animal>_lasso001_run_coverage_summary.csv
    <animal>_lasso001_run_coverage_qc.png
    <animal>_lasso001_stitched_expanded_full_runs.png

How to use
----------
Run from a normal Mac Terminal, not inside a notebook. A Matplotlib window will
open. Drag with the mouse to draw a lasso around a UMAP region. Each completed
lasso selection saves a new output folder. Close the UMAP window when done.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import argparse
import math
import re
import sys

import numpy as np
import pandas as pd

SCRIPT_VERSION = "interactive_lasso_umap_spectrogram_viewer_v2_lasso_coverage_qc"


def _safe_mkdir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _clean_token(x: Any) -> str:
    s = str(x).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _cluster_token(cluster_id: Optional[int]) -> str:
    if cluster_id is None:
        return "all_labels"
    cluster_id = int(cluster_id)
    if cluster_id < 0:
        return "label_noise"
    return f"label{cluster_id:02d}"


def _parse_int_list(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    if not values:
        return None
    out: List[int] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out or None


def _get_optional_array(arr: Mapping[str, Any], key: str, expected_len: int) -> Optional[np.ndarray]:
    if key not in arr:
        return None
    x = np.asarray(arr[key])
    if x.shape[0] != expected_len:
        print(f"[WARN] Array {key!r} has length {x.shape[0]}, expected {expected_len}; ignoring it.")
        return None
    return x


def _orient_spectrogram_to_FxT(S: np.ndarray, n_timebins: int) -> np.ndarray:
    """Return spectrogram as frequency x time."""
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram array, got shape {S.shape}")
    if S.shape[1] == n_timebins:
        return S.astype(float)
    if S.shape[0] == n_timebins:
        return S.T.astype(float)
    raise ValueError(
        f"Could not orient spectrogram with shape {S.shape}; neither dimension matches N={n_timebins}."
    )


def _downsample_indices(idx: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    if max_points is None or max_points <= 0 or idx.size <= max_points:
        return idx
    return np.sort(rng.choice(idx, size=int(max_points), replace=False)).astype(int)


def _contiguous_runs_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open runs [start, end) for True spans in a 1D boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    parts = np.split(idx, breaks)
    return [(int(p[0]), int(p[-1]) + 1) for p in parts if p.size > 0]


def _find_label_runs_overlapping_selection(
    *,
    selected_indices: np.ndarray,
    labels: np.ndarray,
    cluster_id: Optional[int],
    max_examples: int,
    mode: str,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Find representative original-time runs to plot.

    If cluster_id is provided, this finds contiguous runs of that cluster label
    that overlap the lasso-selected time bins. Otherwise, it finds contiguous
    runs of the selected bins themselves.
    """
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_set = set(selected_indices.tolist())
    n = labels.shape[0]

    if cluster_id is not None:
        run_mask = np.asarray(labels).astype(int) == int(cluster_id)
    else:
        run_mask = np.zeros(n, dtype=bool)
        run_mask[selected_indices] = True

    candidate_runs: List[Dict[str, Any]] = []
    for start, end in _contiguous_runs_from_mask(run_mask):
        # Which selected lasso points fall inside this run?
        sel_inside = selected_indices[(selected_indices >= start) & (selected_indices < end)]
        if sel_inside.size == 0:
            continue

        local_selected_mask = np.zeros(int(end - start), dtype=bool)
        local_selected_mask[sel_inside - int(start)] = True
        selected_segments = _contiguous_runs_from_mask(local_selected_mask)
        selected_segment_lengths = [int(s1 - s0) for s0, s1 in selected_segments]
        selected_segment_ranges = [f"{int(s0 + start)}-{int(s1 + start)}" for s0, s1 in selected_segments]
        if len(selected_segments) > 1:
            gaps = [int(selected_segments[i + 1][0] - selected_segments[i][1]) for i in range(len(selected_segments) - 1)]
        else:
            gaps = []

        candidate_runs.append({
            "run_start": int(start),
            "run_end": int(end),
            "run_len": int(end - start),
            "n_selected_in_run": int(sel_inside.size),
            "selected_fraction_of_run": float(sel_inside.size / max(1, end - start)),
            "selected_min": int(sel_inside.min()),
            "selected_max": int(sel_inside.max()),
            "n_selected_segments": int(len(selected_segments)),
            "selected_segment_lengths_bins": ";".join(str(x) for x in selected_segment_lengths),
            "selected_segment_ranges": ";".join(selected_segment_ranges),
            "mean_selected_segment_len_bins": float(np.mean(selected_segment_lengths)) if selected_segment_lengths else 0.0,
            "max_selected_segment_len_bins": int(max(selected_segment_lengths)) if selected_segment_lengths else 0,
            "median_gap_between_selected_segments_bins": float(np.median(gaps)) if gaps else 0.0,
            "max_gap_between_selected_segments_bins": int(max(gaps)) if gaps else 0,
        })

    if not candidate_runs:
        return []

    if mode == "longest":
        candidate_runs.sort(key=lambda d: (d["run_len"], d["n_selected_in_run"]), reverse=True)
        return candidate_runs[:max_examples]

    if mode == "most_selected":
        candidate_runs.sort(key=lambda d: (d["n_selected_in_run"], d["run_len"]), reverse=True)
        return candidate_runs[:max_examples]

    if mode == "random":
        order = rng.permutation(len(candidate_runs))[:max_examples]
        return [candidate_runs[int(i)] for i in order]

    if mode == "mixed":
        # Use a mix of high-overlap, long, and random examples.
        out: List[Dict[str, Any]] = []
        seen: set = set()

        def add(items: Iterable[Dict[str, Any]]) -> None:
            for d in items:
                key = (d["run_start"], d["run_end"])
                if key not in seen and len(out) < max_examples:
                    out.append(d)
                    seen.add(key)

        by_overlap = sorted(candidate_runs, key=lambda d: (d["selected_fraction_of_run"], d["n_selected_in_run"]), reverse=True)
        by_long = sorted(candidate_runs, key=lambda d: d["run_len"], reverse=True)
        by_selected = sorted(candidate_runs, key=lambda d: d["n_selected_in_run"], reverse=True)
        add(by_overlap[: max(1, max_examples // 3)])
        add(by_long[: max(1, max_examples // 3)])
        add(by_selected[: max(1, max_examples // 3)])
        if len(out) < max_examples:
            remaining = [d for d in candidate_runs if (d["run_start"], d["run_end"]) not in seen]
            if remaining:
                order = rng.permutation(len(remaining))
                add([remaining[int(i)] for i in order])
        return out[:max_examples]

    raise ValueError("run_selection must be one of: mixed, longest, most_selected, random")


def _crop_window_for_run(
    *,
    run_start: int,
    run_end: int,
    selected_min: int,
    selected_max: int,
    n_timebins: int,
    context_bins: int,
    max_snippet_bins: int,
) -> Tuple[int, int]:
    """Choose a plot window around the selected region/run."""
    start = max(0, int(run_start) - int(context_bins))
    end = min(int(n_timebins), int(run_end) + int(context_bins))
    if end - start <= int(max_snippet_bins):
        return start, end

    # If the full run/context is too large, center the crop on the selected bins.
    center = int(round((int(selected_min) + int(selected_max)) / 2))
    half = int(max_snippet_bins) // 2
    start = max(0, center - half)
    end = min(int(n_timebins), start + int(max_snippet_bins))
    start = max(0, end - int(max_snippet_bins))
    return start, end


@dataclass
class ViewerConfig:
    npz_path: Path
    out_dir: Path
    animal_id: str
    embedding_key: str = "embedding_outputs"
    spectrogram_key: str = "s"
    label_key: str = "hdbscan_labels"
    file_key: str = "file_indices"
    vocalization_key: str = "vocalization"
    cluster_id: Optional[int] = None
    only_vocalization: bool = False
    include_noise: bool = False
    plot_max_background_points: int = 200000
    plot_max_candidate_points: int = 200000
    point_size_background: float = 1.0
    point_size_candidate: float = 3.0
    max_examples: int = 12
    run_selection: str = "mixed"
    context_bins: int = 80
    max_snippet_bins: int = 700
    stitched_bins_per_row: int = 500
    max_stitched_rows: int = 10
    max_expanded_run_examples: int = 12
    expanded_bins_per_row: int = 500
    max_expanded_total_bins: int = 5000
    seconds_per_bin: float = 0.0027
    cmap: str = "gray_r"
    contrast_percentiles: Optional[Tuple[float, float]] = (1, 99.5)
    dpi: int = 250
    seed: int = 0


class LassoUMAPSpectrogramViewer:
    def __init__(self, cfg: ViewerConfig, plt_module, widgets_module, path_class):
        self.cfg = cfg
        self.plt = plt_module
        self.LassoSelector = widgets_module.LassoSelector
        self.MplPath = path_class
        self.rng = np.random.default_rng(int(cfg.seed))
        self.selection_counter = 0

        arr = np.load(cfg.npz_path, allow_pickle=True)
        self.arr = arr
        self.embedding = np.asarray(arr[cfg.embedding_key], dtype=float)
        if self.embedding.ndim != 2 or self.embedding.shape[1] != 2:
            raise ValueError(f"{cfg.embedding_key!r} must have shape (N, 2); got {self.embedding.shape}")
        self.n_timebins = self.embedding.shape[0]

        self.labels = np.asarray(arr[cfg.label_key]).astype(int)
        if self.labels.shape[0] != self.n_timebins:
            raise ValueError(f"{cfg.label_key!r} length {self.labels.shape[0]} != embedding length {self.n_timebins}")

        self.S_FxT = _orient_spectrogram_to_FxT(np.asarray(arr[cfg.spectrogram_key]), self.n_timebins)
        self.file_indices = _get_optional_array(arr, cfg.file_key, self.n_timebins)
        self.vocalization = _get_optional_array(arr, cfg.vocalization_key, self.n_timebins)

        self.out_root = cfg.out_dir / _clean_token(cfg.animal_id) / _cluster_token(cfg.cluster_id)
        _safe_mkdir(self.out_root)

        self.candidate_mask = self._build_candidate_mask()
        self.candidate_idx = np.flatnonzero(self.candidate_mask).astype(int)
        if self.candidate_idx.size == 0:
            raise ValueError("No candidate points to lasso after applying cluster/vocalization/noise filters.")

        self.fig = None
        self.ax = None
        self.lasso = None

    def _build_candidate_mask(self) -> np.ndarray:
        mask = np.ones(self.n_timebins, dtype=bool)
        if self.cfg.cluster_id is not None:
            mask &= self.labels == int(self.cfg.cluster_id)
        elif not self.cfg.include_noise:
            mask &= self.labels >= 0
        if self.cfg.only_vocalization and self.vocalization is not None:
            mask &= np.asarray(self.vocalization, dtype=bool)
        return mask

    def _selection_dir(self) -> Path:
        self.selection_counter += 1
        out = self.out_root / f"lasso_selection_{self.selection_counter:03d}"
        _safe_mkdir(out)
        return out

    def launch(self) -> None:
        print("=" * 88)
        print(f"[INFO] Script version: {SCRIPT_VERSION}")
        print(f"[INFO] Animal: {self.cfg.animal_id}")
        print(f"[INFO] NPZ: {self.cfg.npz_path}")
        print(f"[INFO] Output root: {self.out_root}")
        print(f"[INFO] Candidate points: {self.candidate_idx.size:,}/{self.n_timebins:,}")
        if self.cfg.cluster_id is not None:
            print(f"[INFO] Lasso restricted to HDBSCAN label {self.cfg.cluster_id}")
        print("[INFO] Draw a lasso around a UMAP region. Each lasso selection saves outputs.")
        print("[INFO] Close the UMAP window when you are finished.")
        print("=" * 88)

        self.fig, self.ax = self.plt.subplots(figsize=(9, 7))

        all_idx = np.arange(self.n_timebins, dtype=int)
        bg_idx = all_idx
        if self.cfg.cluster_id is not None:
            # If focused on one cluster, show the rest as faint background.
            bg_idx = np.flatnonzero(~self.candidate_mask)
        bg_plot_idx = _downsample_indices(bg_idx, self.cfg.plot_max_background_points, self.rng)
        if bg_plot_idx.size > 0:
            self.ax.scatter(
                self.embedding[bg_plot_idx, 0],
                self.embedding[bg_plot_idx, 1],
                s=float(self.cfg.point_size_background),
                c="0.75",
                alpha=0.18,
                linewidths=0,
                label="background",
            )

        cand_plot_idx = _downsample_indices(self.candidate_idx, self.cfg.plot_max_candidate_points, self.rng)
        if cand_plot_idx.size > 0:
            if self.cfg.cluster_id is None:
                colors = self.labels[cand_plot_idx]
                self.ax.scatter(
                    self.embedding[cand_plot_idx, 0],
                    self.embedding[cand_plot_idx, 1],
                    s=float(self.cfg.point_size_candidate),
                    c=colors,
                    cmap="tab20",
                    alpha=0.55,
                    linewidths=0,
                    label="candidate points",
                )
            else:
                self.ax.scatter(
                    self.embedding[cand_plot_idx, 0],
                    self.embedding[cand_plot_idx, 1],
                    s=float(self.cfg.point_size_candidate),
                    c="#1f77b4",
                    alpha=0.65,
                    linewidths=0,
                    label=f"label {self.cfg.cluster_id}",
                )

        title = f"{self.cfg.animal_id} UMAP lasso"
        if self.cfg.cluster_id is not None:
            title += f" — label {self.cfg.cluster_id}"
        self.ax.set_title(title)
        self.ax.set_xlabel("UMAP 1")
        self.ax.set_ylabel("UMAP 2")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.legend(loc="best", frameon=False, markerscale=4)
        self.lasso = self.LassoSelector(self.ax, self.onselect)
        self.plt.tight_layout()
        self.plt.show()

    def onselect(self, verts) -> None:
        path = self.MplPath(verts)
        candidate_xy = self.embedding[self.candidate_idx]
        inside = path.contains_points(candidate_xy)
        selected_idx = self.candidate_idx[inside]
        selected_idx = np.asarray(selected_idx, dtype=int)
        if selected_idx.size == 0:
            print("[WARN] Lasso selected 0 candidate points.")
            return

        out_dir = self._selection_dir()
        print("-" * 88)
        print(f"[LASSO] Selection {self.selection_counter:03d}: {selected_idx.size:,} time bins")
        print(f"[SAVE] {out_dir}")

        selected_mask = np.zeros(self.n_timebins, dtype=bool)
        selected_mask[selected_idx] = True

        self._save_selected_csv(selected_idx, out_dir)
        self._save_umap_selection_png(selected_idx, out_dir)
        self._save_lasso_run_coverage_qc(selected_idx, out_dir)
        self._save_run_context_spectrograms(selected_idx, selected_mask, out_dir)
        self._save_stitched_selected_spectrogram(selected_idx, out_dir)
        self._save_stitched_expanded_full_runs_spectrogram(selected_idx, out_dir)
        print("-" * 88)

    def _save_selected_csv(self, selected_idx: np.ndarray, out_dir: Path) -> None:
        rows: List[Dict[str, Any]] = []
        for order, idx in enumerate(np.asarray(selected_idx, dtype=int)):
            row: Dict[str, Any] = {
                "selection_id": int(self.selection_counter),
                "within_selection_order": int(order),
                "timebin_index": int(idx),
                "umap_1": float(self.embedding[idx, 0]),
                "umap_2": float(self.embedding[idx, 1]),
                "hdbscan_label": int(self.labels[idx]),
            }
            if self.file_indices is not None:
                row["file_index"] = self.file_indices[idx]
            if self.vocalization is not None:
                row["vocalization"] = bool(self.vocalization[idx])
            rows.append(row)
        csv_path = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_selected_timebins.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[SAVE] {csv_path}")

    def _save_umap_selection_png(self, selected_idx: np.ndarray, out_dir: Path) -> None:
        fig, ax = self.plt.subplots(figsize=(8, 7))
        bg_idx = _downsample_indices(np.arange(self.n_timebins), self.cfg.plot_max_background_points, self.rng)
        ax.scatter(
            self.embedding[bg_idx, 0],
            self.embedding[bg_idx, 1],
            s=1,
            c="0.78",
            alpha=0.18,
            linewidths=0,
        )
        cand_plot_idx = _downsample_indices(self.candidate_idx, self.cfg.plot_max_candidate_points, self.rng)
        ax.scatter(
            self.embedding[cand_plot_idx, 0],
            self.embedding[cand_plot_idx, 1],
            s=2,
            c="#1f77b4",
            alpha=0.30,
            linewidths=0,
            label="lasso candidates",
        )
        ax.scatter(
            self.embedding[selected_idx, 0],
            self.embedding[selected_idx, 1],
            s=5,
            c="#d62728",
            alpha=0.85,
            linewidths=0,
            label=f"selected N={selected_idx.size:,}",
        )
        title = f"{self.cfg.animal_id} lasso selection {self.selection_counter:03d}"
        if self.cfg.cluster_id is not None:
            title += f" — label {self.cfg.cluster_id}"
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False, markerscale=3)
        fig.tight_layout()
        out_png = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_umap_selection.png"
        fig.savefig(out_png, dpi=int(self.cfg.dpi), bbox_inches="tight")
        self.plt.close(fig)
        print(f"[SAVE] {out_png}")

    def _get_all_overlapping_label_runs(self, selected_idx: np.ndarray) -> List[Dict[str, Any]]:
        """Return all full HDBSCAN label runs that contain at least one lasso-selected time bin."""
        return _find_label_runs_overlapping_selection(
            selected_indices=np.asarray(selected_idx, dtype=int),
            labels=self.labels,
            cluster_id=self.cfg.cluster_id,
            max_examples=1_000_000_000,
            mode="most_selected",
            rng=self.rng,
        )

    def _runs_dataframe_with_ms(self, runs: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(runs)
        if df.empty:
            return df
        sec = float(self.cfg.seconds_per_bin)
        df["run_duration_ms"] = df["run_len"].astype(float) * sec * 1000.0
        df["selected_duration_ms"] = df["n_selected_in_run"].astype(float) * sec * 1000.0
        df["mean_selected_segment_duration_ms"] = df["mean_selected_segment_len_bins"].astype(float) * sec * 1000.0
        df["max_selected_segment_duration_ms"] = df["max_selected_segment_len_bins"].astype(float) * sec * 1000.0
        df["median_gap_between_selected_segments_ms"] = df["median_gap_between_selected_segments_bins"].astype(float) * sec * 1000.0
        df["max_gap_between_selected_segments_ms"] = df["max_gap_between_selected_segments_bins"].astype(float) * sec * 1000.0
        df["selected_segments_per_second_of_full_run"] = df["n_selected_segments"].astype(float) / np.maximum(df["run_duration_ms"].astype(float) / 1000.0, 1e-12)
        return df

    def _save_lasso_run_coverage_qc(self, selected_idx: np.ndarray, out_dir: Path) -> None:
        """
        Quantify whether the lasso selected whole HDBSCAN phrase-label runs or only
        repeated subphases within those runs.
        """
        runs = self._get_all_overlapping_label_runs(selected_idx)
        if not runs:
            print("[WARN] No overlapping full label runs found for lasso coverage QC.")
            return

        df = self._runs_dataframe_with_ms(runs)
        runs_csv = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_overlapping_runs.csv"
        df.to_csv(runs_csv, index=False)
        print(f"[SAVE] {runs_csv}")

        total_full_bins = float(df["run_len"].sum())
        total_selected_bins = float(df["n_selected_in_run"].sum())
        summary = pd.DataFrame([{
            "selection_id": int(self.selection_counter),
            "animal_id": self.cfg.animal_id,
            "cluster_id": self.cfg.cluster_id,
            "n_lasso_selected_timebins": int(len(selected_idx)),
            "n_overlapping_full_label_runs": int(len(df)),
            "total_full_run_bins": int(total_full_bins),
            "total_selected_bins_inside_full_runs": int(total_selected_bins),
            "overall_lasso_coverage_fraction": float(total_selected_bins / max(total_full_bins, 1.0)),
            "median_run_coverage_fraction": float(df["selected_fraction_of_run"].median()),
            "mean_run_coverage_fraction": float(df["selected_fraction_of_run"].mean()),
            "median_selected_segments_per_run": float(df["n_selected_segments"].median()),
            "mean_selected_segments_per_run": float(df["n_selected_segments"].mean()),
            "fraction_runs_coverage_lt_0p25": float((df["selected_fraction_of_run"] < 0.25).mean()),
            "fraction_runs_coverage_lt_0p50": float((df["selected_fraction_of_run"] < 0.50).mean()),
            "fraction_runs_with_multiple_selected_segments": float((df["n_selected_segments"] > 1).mean()),
            "median_full_run_duration_ms": float(df["run_duration_ms"].median()),
            "median_lasso_selected_duration_per_run_ms": float(df["selected_duration_ms"].median()),
        }])
        summary_csv = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_run_coverage_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"[SAVE] {summary_csv}")

        # Visual QC plots.
        fig, axes = self.plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].hist(df["selected_fraction_of_run"].to_numpy(), bins=np.linspace(0, 1, 21), edgecolor="black")
        axes[0].set_xlabel("Lasso coverage of full label run")
        axes[0].set_ylabel("Number of full label runs")
        axes[0].set_title("Coverage fraction")

        sc = axes[1].scatter(
            df["run_duration_ms"],
            df["selected_fraction_of_run"],
            c=df["n_selected_segments"],
            s=18,
            alpha=0.75,
        )
        axes[1].set_xlabel("Full label-run duration (ms)")
        axes[1].set_ylabel("Lasso coverage fraction")
        axes[1].set_title("Does lasso capture whole runs?")
        cb = fig.colorbar(sc, ax=axes[1])
        cb.set_label("Selected segments per run")

        max_seg = int(max(1, min(50, df["n_selected_segments"].max())))
        axes[2].hist(df["n_selected_segments"].to_numpy(), bins=np.arange(1, max_seg + 2) - 0.5, edgecolor="black")
        axes[2].set_xlabel("Number of separate lasso-selected segments\ninside each full label run")
        axes[2].set_ylabel("Number of full label runs")
        axes[2].set_title("Repeated subphase selection")

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(
            f"{self.cfg.animal_id} lasso {self.selection_counter:03d}: lasso coverage within full HDBSCAN label runs",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        qc_png = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_run_coverage_qc.png"
        fig.savefig(qc_png, dpi=int(self.cfg.dpi), bbox_inches="tight")
        self.plt.close(fig)
        print(f"[SAVE] {qc_png}")

    def _save_run_context_spectrograms(self, selected_idx: np.ndarray, selected_mask: np.ndarray, out_dir: Path) -> None:
        runs = _find_label_runs_overlapping_selection(
            selected_indices=selected_idx,
            labels=self.labels,
            cluster_id=self.cfg.cluster_id,
            max_examples=int(self.cfg.max_examples),
            mode=str(self.cfg.run_selection),
            rng=self.rng,
        )
        if not runs:
            print("[WARN] No overlapping runs found for context spectrograms.")
            return

        n_examples = len(runs)
        fig_h = max(2.5, 2.0 * n_examples + 0.9)
        fig, axes = self.plt.subplots(n_examples, 1, figsize=(16, fig_h), sharex=False)
        if n_examples == 1:
            axes = [axes]

        # Contrast from all plotted snippets.
        snippets = []
        windows = []
        for d in runs:
            start, end = _crop_window_for_run(
                run_start=d["run_start"],
                run_end=d["run_end"],
                selected_min=d["selected_min"],
                selected_max=d["selected_max"],
                n_timebins=self.n_timebins,
                context_bins=int(self.cfg.context_bins),
                max_snippet_bins=int(self.cfg.max_snippet_bins),
            )
            windows.append((start, end))
            snippets.append(self.S_FxT[:, start:end].astype(float).ravel())
        combined = np.concatenate([x[np.isfinite(x)] for x in snippets if x.size > 0])
        if combined.size == 0 or self.cfg.contrast_percentiles is None:
            vmin = vmax = None
        else:
            lo, hi = self.cfg.contrast_percentiles
            vmin, vmax = np.nanpercentile(combined, [float(lo), float(hi)])
            vmin = float(vmin)
            vmax = float(vmax)

        for ax, d, (start, end) in zip(axes, runs, windows):
            S_panel = self.S_FxT[:, start:end].astype(float)
            n_bins = S_panel.shape[1]
            ax.imshow(
                S_panel,
                origin="lower",
                aspect="auto",
                cmap=self.cfg.cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[0, n_bins * self.cfg.seconds_per_bin, 0, S_panel.shape[0]],
            )
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)

            # Highlight selected lasso bins and the full cluster run as bars below the spectrogram.
            x0 = 0.0
            y0 = -0.15
            y1 = -0.28
            trans = ax.get_xaxis_transform()

            # Full HDBSCAN cluster run span inside this crop.
            run_start_rel = max(0, d["run_start"] - start) * self.cfg.seconds_per_bin
            run_end_rel = min(end, d["run_end"]) - start
            run_end_rel = run_end_rel * self.cfg.seconds_per_bin
            ax.plot([run_start_rel, run_end_rel], [y0, y0], transform=trans, color="#1f77b4", lw=5, solid_capstyle="butt", clip_on=False)

            # Lasso-selected bins inside this crop as red mini-spans.
            local_selected = np.flatnonzero(selected_mask[start:end])
            if local_selected.size > 0:
                selected_runs = _contiguous_runs_from_mask(selected_mask[start:end])
                for s0, s1 in selected_runs:
                    ax.plot(
                        [s0 * self.cfg.seconds_per_bin, s1 * self.cfg.seconds_per_bin],
                        [y1, y1],
                        transform=trans,
                        color="#d62728",
                        lw=4,
                        solid_capstyle="butt",
                        clip_on=False,
                    )

            ax.set_ylabel(
                f"run {d['run_start']}-{d['run_end']}\n"
                f"len={d['run_len']} bins\n"
                f"selected={d['n_selected_in_run']}",
                rotation=0,
                labelpad=68,
                va="center",
                fontsize=9,
            )
            ax.set_xlabel("Time in original recording context (s)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            f"{self.cfg.animal_id} lasso {self.selection_counter:03d}: representative original-context spectrograms\n"
            f"blue = full HDBSCAN label run; red = time bins inside lasso region",
            fontsize=15,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0)
        out_png = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_run_context_spectrograms.png"
        fig.savefig(out_png, dpi=int(self.cfg.dpi), bbox_inches="tight")
        self.plt.close(fig)
        print(f"[SAVE] {out_png}")

        context_csv = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_context_example_runs.csv"
        self._runs_dataframe_with_ms(runs).to_csv(context_csv, index=False)
        print(f"[SAVE] {context_csv}")

    def _save_stitched_selected_spectrogram(self, selected_idx: np.ndarray, out_dir: Path) -> None:
        selected_idx = np.asarray(selected_idx, dtype=int)
        if selected_idx.size == 0:
            return

        # Sort by original time so repeated/contiguous points stay together.
        selected_idx = np.sort(selected_idx)
        bins_per_row = int(self.cfg.stitched_bins_per_row)
        chunks = [selected_idx[i:i + bins_per_row] for i in range(0, selected_idx.size, bins_per_row)]
        if self.cfg.max_stitched_rows and self.cfg.max_stitched_rows > 0:
            chunks = chunks[: int(self.cfg.max_stitched_rows)]
        chunks = [c for c in chunks if c.size > 0]
        if not chunks:
            return

        arrays = [self.S_FxT[:, c].astype(float).ravel() for c in chunks]
        combined = np.concatenate([x[np.isfinite(x)] for x in arrays if x.size > 0])
        if combined.size == 0 or self.cfg.contrast_percentiles is None:
            vmin = vmax = None
        else:
            lo, hi = self.cfg.contrast_percentiles
            vmin, vmax = np.nanpercentile(combined, [float(lo), float(hi)])
            vmin = float(vmin)
            vmax = float(vmax)

        n_rows = len(chunks)
        fig_h = max(2.3, 1.7 * n_rows + 0.8)
        fig, axes = self.plt.subplots(n_rows, 1, figsize=(16, fig_h), sharex=True)
        if n_rows == 1:
            axes = [axes]

        max_time_s = bins_per_row * self.cfg.seconds_per_bin
        for row_i, (ax, chunk) in enumerate(zip(axes, chunks), start=1):
            S_panel = self.S_FxT[:, chunk].astype(float)
            actual_bins = S_panel.shape[1]
            actual_time_s = actual_bins * self.cfg.seconds_per_bin
            ax.imshow(
                S_panel,
                origin="lower",
                aspect="auto",
                cmap=self.cfg.cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[0, actual_time_s, 0, S_panel.shape[0]],
            )
            ax.set_xlim(0, max_time_s)
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_ylabel(f"row {row_i}\n{actual_bins} bins", rotation=0, labelpad=42, va="center", fontsize=9)

        axes[-1].set_xlabel("Stitched lasso-selected time bins (s)")
        fig.suptitle(
            f"{self.cfg.animal_id} lasso {self.selection_counter:03d}: stitched selected UMAP-region time bins\n"
            f"showing {sum(len(c) for c in chunks):,}/{selected_idx.size:,} selected bins",
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.25)
        out_png = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_stitched_selected_spectrogram.png"
        fig.savefig(out_png, dpi=int(self.cfg.dpi), bbox_inches="tight")
        self.plt.close(fig)
        print(f"[SAVE] {out_png}")

    def _save_stitched_expanded_full_runs_spectrogram(self, selected_idx: np.ndarray, out_dir: Path) -> None:
        """Save spectrogram rows after expanding selected lasso bins to their full HDBSCAN label runs."""
        runs = _find_label_runs_overlapping_selection(
            selected_indices=np.asarray(selected_idx, dtype=int),
            labels=self.labels,
            cluster_id=self.cfg.cluster_id,
            max_examples=int(self.cfg.max_expanded_run_examples),
            mode="most_selected",
            rng=self.rng,
        )
        if not runs:
            return

        expanded_indices_parts: List[np.ndarray] = []
        included_rows: List[Dict[str, Any]] = []
        total = 0
        max_total = int(self.cfg.max_expanded_total_bins)
        for d in runs:
            run_idx = np.arange(int(d["run_start"]), int(d["run_end"]), dtype=int)
            if max_total > 0 and total + run_idx.size > max_total:
                remaining = max_total - total
                if remaining <= 0:
                    break
                run_idx = run_idx[:remaining]
            if run_idx.size == 0:
                continue
            expanded_indices_parts.append(run_idx)
            row = dict(d)
            row["included_expanded_bins"] = int(run_idx.size)
            included_rows.append(row)
            total += int(run_idx.size)
            if max_total > 0 and total >= max_total:
                break

        if not expanded_indices_parts:
            return

        expanded_idx = np.concatenate(expanded_indices_parts).astype(int)
        bins_per_row = int(self.cfg.expanded_bins_per_row)
        chunks = [expanded_idx[i:i + bins_per_row] for i in range(0, expanded_idx.size, bins_per_row)]
        chunks = [c for c in chunks if c.size > 0]
        if not chunks:
            return

        arrays = [self.S_FxT[:, c].astype(float).ravel() for c in chunks]
        combined = np.concatenate([x[np.isfinite(x)] for x in arrays if x.size > 0])
        if combined.size == 0 or self.cfg.contrast_percentiles is None:
            vmin = vmax = None
        else:
            lo, hi = self.cfg.contrast_percentiles
            vmin, vmax = np.nanpercentile(combined, [float(lo), float(hi)])
            vmin = float(vmin)
            vmax = float(vmax)

        n_rows = len(chunks)
        fig_h = max(2.3, 1.7 * n_rows + 0.8)
        fig, axes = self.plt.subplots(n_rows, 1, figsize=(16, fig_h), sharex=True)
        if n_rows == 1:
            axes = [axes]

        max_time_s = bins_per_row * self.cfg.seconds_per_bin
        for row_i, (ax, chunk) in enumerate(zip(axes, chunks), start=1):
            S_panel = self.S_FxT[:, chunk].astype(float)
            actual_bins = S_panel.shape[1]
            actual_time_s = actual_bins * self.cfg.seconds_per_bin
            ax.imshow(
                S_panel,
                origin="lower",
                aspect="auto",
                cmap=self.cfg.cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[0, actual_time_s, 0, S_panel.shape[0]],
            )
            ax.set_xlim(0, max_time_s)
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_ylabel(f"row {row_i}\n{actual_bins} bins", rotation=0, labelpad=42, va="center", fontsize=9)

        axes[-1].set_xlabel("Stitched full label-run bins overlapping lasso (s)")
        fig.suptitle(
            f"{self.cfg.animal_id} lasso {self.selection_counter:03d}: lasso-expanded full HDBSCAN label runs\n"
            f"showing {expanded_idx.size:,} full-run bins from {len(included_rows):,} overlapping runs",
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.25)
        out_png = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_stitched_expanded_full_runs.png"
        fig.savefig(out_png, dpi=int(self.cfg.dpi), bbox_inches="tight")
        self.plt.close(fig)
        print(f"[SAVE] {out_png}")

        expanded_csv = out_dir / f"{self.cfg.animal_id}_lasso{self.selection_counter:03d}_expanded_full_runs_included.csv"
        self._runs_dataframe_with_ms(included_rows).to_csv(expanded_csv, index=False)
        print(f"[SAVE] {expanded_csv}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive lasso on UMAP, then save representative spectrograms from selected time bins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz-path", required=True, help="Path to one bird NPZ file.")
    p.add_argument("--out-dir", default="/Volumes/my_own_SSD/updated_AreaX_outputs/lasso_umap_spectrogram_qc")
    p.add_argument("--animal-id", default=None, help="If omitted, inferred from NPZ filename.")
    p.add_argument("--embedding-key", default="embedding_outputs")
    p.add_argument("--spectrogram-key", default="s")
    p.add_argument("--label-key", default="hdbscan_labels")
    p.add_argument("--file-key", default="file_indices")
    p.add_argument("--vocalization-key", default="vocalization")
    p.add_argument("--cluster-id", type=int, default=None, help="Restrict lasso selection to this HDBSCAN label/cluster.")
    p.add_argument("--only-vocalization", action="store_true", help="Restrict candidates to vocalization==True if the NPZ contains vocalization.")
    p.add_argument("--include-noise", action="store_true", help="If no --cluster-id is provided, include label -1/noise in candidates.")
    p.add_argument("--backend", default=None, help="Optional Matplotlib backend, e.g. TkAgg, QtAgg, MacOSX. Set before importing pyplot.")

    p.add_argument("--plot-max-background-points", type=int, default=200000)
    p.add_argument("--plot-max-candidate-points", type=int, default=200000)
    p.add_argument("--point-size-background", type=float, default=1.0)
    p.add_argument("--point-size-candidate", type=float, default=3.0)

    p.add_argument("--max-examples", type=int, default=12, help="Number of representative original-context spectrogram rows to save per lasso.")
    p.add_argument("--run-selection", choices=["mixed", "longest", "most_selected", "random"], default="mixed")
    p.add_argument("--context-bins", type=int, default=80, help="Extra bins before/after a selected label run in context examples.")
    p.add_argument("--max-snippet-bins", type=int, default=700, help="Maximum width of each original-context spectrogram example.")
    p.add_argument("--stitched-bins-per-row", type=int, default=500)
    p.add_argument("--max-stitched-rows", type=int, default=10)
    p.add_argument("--max-expanded-run-examples", type=int, default=12, help="How many overlapping full HDBSCAN runs to include in the expanded full-run spectrogram.")
    p.add_argument("--expanded-bins-per-row", type=int, default=500, help="Bins per row for the lasso-expanded full-run spectrogram.")
    p.add_argument("--max-expanded-total-bins", type=int, default=5000, help="Maximum full-run bins to show in the expanded full-run spectrogram. Use 0 for no cap.")
    p.add_argument("--seconds-per-bin", type=float, default=0.0027)
    p.add_argument("--cmap", default="gray_r")
    p.add_argument("--contrast-percentiles", nargs=2, type=float, default=[1, 99.5], metavar=("LOW", "HIGH"))
    p.add_argument("--no-contrast-percentiles", action="store_true")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Set interactive backend before importing pyplot.
    import matplotlib
    if args.backend:
        matplotlib.use(str(args.backend), force=True)
    import matplotlib.pyplot as plt
    from matplotlib import widgets
    from matplotlib.path import Path as MplPath

    npz_path = Path(args.npz_path).expanduser().resolve()
    animal_id = str(args.animal_id) if args.animal_id else npz_path.stem.split("_")[0]
    contrast = None if bool(args.no_contrast_percentiles) else tuple(float(x) for x in args.contrast_percentiles)

    cfg = ViewerConfig(
        npz_path=npz_path,
        out_dir=Path(args.out_dir).expanduser(),
        animal_id=animal_id,
        embedding_key=str(args.embedding_key),
        spectrogram_key=str(args.spectrogram_key),
        label_key=str(args.label_key),
        file_key=str(args.file_key),
        vocalization_key=str(args.vocalization_key),
        cluster_id=args.cluster_id,
        only_vocalization=bool(args.only_vocalization),
        include_noise=bool(args.include_noise),
        plot_max_background_points=int(args.plot_max_background_points),
        plot_max_candidate_points=int(args.plot_max_candidate_points),
        point_size_background=float(args.point_size_background),
        point_size_candidate=float(args.point_size_candidate),
        max_examples=int(args.max_examples),
        run_selection=str(args.run_selection),
        context_bins=int(args.context_bins),
        max_snippet_bins=int(args.max_snippet_bins),
        stitched_bins_per_row=int(args.stitched_bins_per_row),
        max_stitched_rows=int(args.max_stitched_rows),
        max_expanded_run_examples=int(args.max_expanded_run_examples),
        expanded_bins_per_row=int(args.expanded_bins_per_row),
        max_expanded_total_bins=int(args.max_expanded_total_bins),
        seconds_per_bin=float(args.seconds_per_bin),
        cmap=str(args.cmap),
        contrast_percentiles=contrast,
        dpi=int(args.dpi),
        seed=int(args.seed),
    )

    viewer = LassoUMAPSpectrogramViewer(cfg, plt, widgets, MplPath)
    viewer.launch()


if __name__ == "__main__":
    main()
