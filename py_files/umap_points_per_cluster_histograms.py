#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_points_per_cluster_histograms.py

Create per-bird and aggregate histograms of cluster sizes from NPZ files.

This version can compute:
- total raw points per cluster
- pre-lesion early raw points per cluster
- pre-lesion late raw points per cluster
- post-lesion early raw points per cluster
- post-lesion late raw points per cluster
- shared equalized points per group per cluster, where:
      n_shared_equal = min(pre_early, pre_late, post_early, post_late)

Early/late splitting is done at the file level within each period (pre or post),
using file order inferred from recording datetimes parsed from file names.
Treatment dates are read from the metadata workbook.

Default outputs
---------------
/Volumes/my_own_SSD/points_per_cluster_histograms/
  per_bird_csv/
    <animal_id>_cluster_point_counts.csv
  per_bird_plots/
    <animal_id>_cluster_point_count_histogram_<metric>.png
    <animal_id>_cluster_point_count_histogram_<metric>_zoom_0_to_5000.png
    <animal_id>_cluster_point_count_histogram_<metric>_zoom_0_to_2500.png
  aggregate/
    all_birds_cluster_point_counts.csv
    all_birds_cluster_point_count_histogram_<metric>.png
    all_birds_cluster_point_count_histogram_<metric>_zoom_0_to_5000.png
    all_birds_cluster_point_count_histogram_<metric>_zoom_0_to_2500.png
  run_manifest.csv

Example
-------
python umap_points_per_cluster_histograms.py \
  --root-dir "/Volumes/my_own_SSD/updated_AreaX_outputs" \
  --metadata-xlsx "/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx" \
  --out-dir "/Volumes/my_own_SSD/points_per_cluster_histograms" \
  --recursive
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]


def _find_npz_files(root_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(p for p in root_dir.rglob("*.npz") if p.is_file())
    return sorted(p for p in root_dir.glob("*.npz") if p.is_file())


def _normalize_cluster_value(x: Any) -> Any:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------


def _find_first_matching_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        c = cols.get(str(cand).strip().lower(), None)
        if c is not None:
            return c
    return None


def _parse_treatment_date_from_metadata_row(row: pd.Series, date_col: str) -> Optional[pd.Timestamp]:
    val = row.get(date_col, None)
    if pd.isna(val):
        return None
    try:
        ts = pd.to_datetime(val)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _lookup_treatment_date(
    metadata_xlsx: Path,
    animal_id: str,
    *,
    metadata_sheet: Optional[str],
    animal_id_col_candidates: Sequence[str],
    treatment_date_col_candidates: Sequence[str],
) -> pd.Timestamp:
    xlsx = pd.ExcelFile(metadata_xlsx)

    if metadata_sheet is not None and metadata_sheet in xlsx.sheet_names:
        candidate_sheets = [metadata_sheet]
    else:
        candidate_sheets = list(xlsx.sheet_names)

    animal_id_norm = str(animal_id).strip().lower()

    for sheet in candidate_sheets:
        try:
            df = pd.read_excel(metadata_xlsx, sheet_name=sheet)
        except Exception:
            continue

        if len(df.columns) == 0:
            continue

        animal_col = _find_first_matching_column(df.columns, animal_id_col_candidates)
        date_col = _find_first_matching_column(df.columns, treatment_date_col_candidates)
        if animal_col is None or date_col is None:
            continue

        work = df.copy()
        work["_animal_norm"] = work[animal_col].astype(str).str.strip().str.lower()
        hits = work[work["_animal_norm"] == animal_id_norm]
        if len(hits) == 0:
            continue

        for _, row in hits.iterrows():
            ts = _parse_treatment_date_from_metadata_row(row, date_col)
            if ts is not None:
                return ts

    raise ValueError(
        f"Could not find a treatment date for animal_id={animal_id!r} in metadata workbook: {metadata_xlsx}"
    )


# -----------------------------------------------------------------------------
# File-map parsing and date extraction
# -----------------------------------------------------------------------------


def _normalize_file_map(file_map_obj: Any) -> Dict[int, str]:
    mapping: Dict[int, str] = {}

    if file_map_obj is None:
        return mapping

    if isinstance(file_map_obj, np.ndarray):
        if file_map_obj.shape == () and file_map_obj.dtype == object:
            return _normalize_file_map(file_map_obj.item())
        if file_map_obj.dtype == object:
            try:
                obj = file_map_obj.tolist()
            except Exception:
                obj = list(file_map_obj)
            return _normalize_file_map(obj)
        for i, v in enumerate(file_map_obj):
            mapping[int(i)] = str(v)
        return mapping

    if isinstance(file_map_obj, dict):
        for k, v in file_map_obj.items():
            try:
                kk = int(k)
            except Exception:
                continue
            mapping[kk] = str(v)
        return mapping

    if isinstance(file_map_obj, (list, tuple)):
        if len(file_map_obj) > 0 and isinstance(file_map_obj[0], (list, tuple)) and len(file_map_obj[0]) >= 2:
            for item in file_map_obj:
                try:
                    kk = int(item[0])
                    vv = str(item[1])
                    mapping[kk] = vv
                except Exception:
                    pass
            return mapping
        for i, v in enumerate(file_map_obj):
            mapping[int(i)] = str(v)
        return mapping

    if isinstance(file_map_obj, pd.Series):
        for k, v in file_map_obj.items():
            try:
                mapping[int(k)] = str(v)
            except Exception:
                pass
        return mapping

    return mapping


_DATE_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
_DATETIME_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[^0-9]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})")


def _parse_datetime_from_filename(name: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    s = str(name)

    # Excel serial + explicit month/day/time
    m = re.search(
        r"_(\d{5}(?:\.\d+)?)_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})(?:_|$)",
        s,
    )
    if m is not None:
        serial_str, mo_str, d_str, hh_str, mm_str, ss_str = m.groups()
        try:
            serial = float(serial_str)
            dt = pd.Timestamp("1899-12-30") + pd.to_timedelta(serial, unit="D")
            mo = int(mo_str)
            d = int(d_str)
            hh = int(hh_str)
            mm = int(mm_str)
            ss = int(ss_str)
            if (
                dt.month != mo
                or dt.day != d
                or dt.hour != hh
                or dt.minute != mm
                or abs(dt.second - ss) > 1
            ):
                dt = pd.Timestamp(year=dt.year, month=mo, day=d, hour=hh, minute=mm, second=ss)
            return dt, dt.normalize()
        except Exception:
            pass

    # YYYY-MM-DD_HH-MM-SS or YYYY_MM_DD_HH_MM_SS
    m = re.search(
        r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})[T _-]?(\d{2})[:\-_]?(\d{2})[:\-_]?(\d{2})(?!\d)",
        s,
    )
    if m is not None:
        try:
            y, mo, d, hh, mm, ss = map(int, m.groups())
            dt = pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, second=ss)
            return dt, dt.normalize()
        except Exception:
            pass

    # YYYYMMDD_HHMMSS
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})[T _-]?(\d{2})(\d{2})(\d{2})(?!\d)", s)
    if m is not None:
        try:
            y, mo, d, hh, mm, ss = map(int, m.groups())
            dt = pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, second=ss)
            return dt, dt.normalize()
        except Exception:
            pass

    # YYYY-MM-DD / YYYY_MM_DD
    m = re.search(r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})(?!\d)", s)
    if m is not None:
        try:
            y, mo, d = map(int, m.groups())
            dt = pd.Timestamp(year=y, month=mo, day=d)
            return dt, dt.normalize()
        except Exception:
            pass

    # YYYYMMDD
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", s)
    if m is not None:
        try:
            y, mo, d = map(int, m.groups())
            dt = pd.Timestamp(year=y, month=mo, day=d)
            return dt, dt.normalize()
        except Exception:
            pass

    return None, None


# -----------------------------------------------------------------------------
# Point-table creation and early/late splitting
# -----------------------------------------------------------------------------


def _build_file_metadata(
    npz: np.lib.npyio.NpzFile,
    *,
    file_key: str,
    file_map_key: str,
) -> pd.DataFrame:
    file_map_obj = npz[file_map_key] if file_map_key in npz.files else None
    mapping = _normalize_file_map(file_map_obj)

    if len(mapping) == 0 and file_key in npz.files:
        file_indices = np.asarray(npz[file_key]).astype(int)
        uniq = np.unique(file_indices)
        mapping = {int(i): f"file_{int(i)}" for i in uniq}

    rows: List[Dict[str, Any]] = []
    for file_idx, file_name in sorted(mapping.items()):
        rec_dt, rec_date = _parse_datetime_from_filename(str(file_name))
        rows.append(
            {
                "file_index": int(file_idx),
                "file_name": str(file_name),
                "recording_datetime": rec_dt,
                "recording_date": rec_date,
            }
        )

    file_df = pd.DataFrame(rows)
    if len(file_df) == 0:
        raise ValueError("Could not construct file metadata from NPZ file_map / file indices.")

    file_df["sort_datetime"] = pd.to_datetime(file_df["recording_datetime"], errors="coerce")
    file_df["sort_datetime"] = file_df["sort_datetime"].fillna(pd.Timestamp("1900-01-01"))
    file_df = file_df.sort_values(["sort_datetime", "file_index"]).reset_index(drop=True)
    return file_df


def _load_points(
    npz_path: Path,
    *,
    cluster_key: str,
    file_key: str,
    file_map_key: str,
    vocalization_key: str,
    vocalization_only: bool,
    include_noise: bool,
    metadata_xlsx: Path,
    metadata_sheet: Optional[str],
    animal_id_col_candidates: Sequence[str],
    treatment_date_col_candidates: Sequence[str],
    exclude_treatment_day_from_post: bool,
) -> Tuple[pd.DataFrame, str, pd.Timestamp]:
    animal_id = _infer_animal_id(npz_path)

    with np.load(npz_path, allow_pickle=True) as npz:
        if cluster_key not in npz.files:
            raise KeyError(
                f"Cluster key '{cluster_key}' not found in {npz_path.name}. Available keys: {list(npz.files)}"
            )
        if file_key not in npz.files:
            raise KeyError(
                f"File key '{file_key}' not found in {npz_path.name}. Available keys: {list(npz.files)}"
            )

        clusters = np.asarray(npz[cluster_key]).ravel()
        file_indices = np.asarray(npz[file_key]).astype(int).ravel()
        if len(clusters) != len(file_indices):
            raise ValueError(
                f"Length mismatch in {npz_path.name}: clusters={len(clusters)} vs file_indices={len(file_indices)}"
            )

        if vocalization_only and (vocalization_key in npz.files):
            keep = np.asarray(npz[vocalization_key]).astype(bool).ravel()
            if len(keep) != len(clusters):
                raise ValueError(
                    f"Length mismatch in {npz_path.name}: vocalization={len(keep)} vs clusters={len(clusters)}"
                )
        else:
            keep = np.ones(len(clusters), dtype=bool)

        file_df = _build_file_metadata(npz, file_key=file_key, file_map_key=file_map_key)

    treatment_date = _lookup_treatment_date(
        metadata_xlsx,
        animal_id,
        metadata_sheet=metadata_sheet,
        animal_id_col_candidates=animal_id_col_candidates,
        treatment_date_col_candidates=treatment_date_col_candidates,
    )

    points = pd.DataFrame(
        {
            "point_index": np.arange(len(clusters), dtype=int),
            "cluster": [_normalize_cluster_value(x) for x in clusters],
            "file_index": file_indices,
            "keep": keep.astype(bool),
        }
    )

    points = points.merge(
        file_df[["file_index", "file_name", "recording_datetime", "recording_date"]],
        on="file_index",
        how="left",
        validate="many_to_one",
    )

    if points["recording_date"].isna().all():
        sample_names = points["file_name"].dropna().astype(str).head(10).tolist()
        raise ValueError(
            "Could not parse recording dates from file metadata. "
            f"Example file names: {sample_names}"
        )

    points = points.loc[points["keep"]].copy()
    if not include_noise:
        points = points.loc[points["cluster"] != -1].copy()

    points["recording_date"] = pd.to_datetime(points["recording_date"], errors="coerce").dt.normalize()
    points["recording_datetime"] = pd.to_datetime(points["recording_datetime"], errors="coerce")

    if exclude_treatment_day_from_post:
        points["period"] = np.where(points["recording_date"] < treatment_date, "pre", "other")
        points.loc[points["recording_date"] > treatment_date, "period"] = "post"
    else:
        points["period"] = np.where(points["recording_date"] < treatment_date, "pre", "post")

    points = points.loc[points["period"].isin(["pre", "post"])].copy()
    return points, animal_id, treatment_date


def _split_files_early_late(file_meta: pd.DataFrame, method: str) -> Tuple[np.ndarray, np.ndarray]:
    if len(file_meta) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    ordered = file_meta.sort_values(["recording_datetime", "recording_date", "file_index"]).reset_index(drop=True)
    files = ordered["file_index"].to_numpy(dtype=int)
    n = len(files)

    if n == 1:
        return files.copy(), np.array([], dtype=int)

    if method not in {"file_median", "file_half"}:
        raise ValueError(f"Unsupported early_late_split_method: {method}")

    if method == "file_half":
        cut = n // 2
    else:
        cut = int(math.ceil(n / 2.0))

    early = files[:cut]
    late = files[cut:]
    return early, late


# -----------------------------------------------------------------------------
# Counting per cluster
# -----------------------------------------------------------------------------


def _counts_for_one_cluster(cluster_points: pd.DataFrame, *, early_late_split_method: str) -> Dict[str, int]:
    out: Dict[str, int] = {
        "n_points_total_raw": int(len(cluster_points)),
        "n_points_pre_all_raw": 0,
        "n_points_pre_early_raw": 0,
        "n_points_pre_late_raw": 0,
        "n_points_post_all_raw": 0,
        "n_points_post_early_raw": 0,
        "n_points_post_late_raw": 0,
    }

    for period in ["pre", "post"]:
        sub = cluster_points.loc[cluster_points["period"] == period].copy()
        out[f"n_points_{period}_all_raw"] = int(len(sub))

        unique_files = (
            sub[["file_index", "recording_datetime", "recording_date"]]
            .drop_duplicates()
            .sort_values(["recording_datetime", "recording_date", "file_index"])
            .reset_index(drop=True)
        )
        early_files, late_files = _split_files_early_late(unique_files, method=early_late_split_method)

        out[f"n_points_{period}_early_raw"] = int(sub.loc[sub["file_index"].isin(early_files)].shape[0])
        out[f"n_points_{period}_late_raw"] = int(sub.loc[sub["file_index"].isin(late_files)].shape[0])

    pre_pair_equal = min(out["n_points_pre_early_raw"], out["n_points_pre_late_raw"])
    post_pair_equal = min(out["n_points_post_early_raw"], out["n_points_post_late_raw"])
    shared_equal = min(
        out["n_points_pre_early_raw"],
        out["n_points_pre_late_raw"],
        out["n_points_post_early_raw"],
        out["n_points_post_late_raw"],
    )

    out["n_points_pre_pair_equal_per_group"] = int(pre_pair_equal)
    out["n_points_post_pair_equal_per_group"] = int(post_pair_equal)
    out["n_points_shared_equal_per_group"] = int(shared_equal)
    out["n_points_shared_equal_total_four_groups"] = int(4 * shared_equal)
    out["has_complete_four_groups"] = int(shared_equal > 0)
    return out


def _cluster_count_table_from_points(points: pd.DataFrame, *, animal_id: str, npz_path: Path, early_late_split_method: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if len(points) == 0:
        return pd.DataFrame()

    grouped = points.groupby("cluster", sort=True, dropna=False)
    for cluster_value, cluster_points in grouped:
        row: Dict[str, Any] = {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "cluster": cluster_value,
        }
        row.update(_counts_for_one_cluster(cluster_points, early_late_split_method=early_late_split_method))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["rank_by_total_raw_desc"] = df["n_points_total_raw"].rank(method="first", ascending=False).astype(int)
    return df.sort_values(["rank_by_total_raw_desc", "cluster"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Histogram helpers
# -----------------------------------------------------------------------------


def _integer_bin_edges(vals: np.ndarray, target_bin_width: int, hard_max_bins: int = 140) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    vmin = float(np.floor(np.nanmin(vals)))
    vmax = float(np.ceil(np.nanmax(vals)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([0.0, 1.0], dtype=float)
    if vmax <= vmin:
        return np.array([vmin - 0.5, vmin + 0.5], dtype=float)

    width = max(1, int(target_bin_width))
    n_bins = int(np.ceil((vmax - vmin + 1) / width))
    if n_bins > int(hard_max_bins):
        width = max(1, int(np.ceil((vmax - vmin + 1) / int(hard_max_bins))))

    start = np.floor(vmin / width) * width
    stop = np.ceil((vmax + 1) / width) * width
    edges = np.arange(start, stop + width, width, dtype=float)
    if edges.size < 2:
        edges = np.array([start, start + width], dtype=float)
    return edges


def _auto_full_bin_width(vals: np.ndarray) -> int:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 100

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    spread = max(1.0, vmax - vmin)
    raw_width = spread / 60.0
    nice_choices = np.array(
        [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000],
        dtype=float,
    )
    return int(nice_choices[np.argmin(np.abs(nice_choices - raw_width))])


def _plot_histogram(
    values: Iterable[float],
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    dpi: int,
    target_bin_width: int,
    x_max: Optional[float] = None,
    add_rug: bool = True,
) -> None:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return

    if x_max is not None:
        vals = vals[vals <= float(x_max)]
        if vals.size == 0:
            return

    edges = _integer_bin_edges(vals, int(target_bin_width))

    plt.figure(figsize=(9.2, 5.8))
    counts, _, _ = plt.hist(vals, bins=edges, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Number of clusters")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)

    if add_rug and len(vals) <= 250:
        y_rug = np.full(len(vals), -max(float(np.max(counts)) * 0.02, 0.02))
        plt.scatter(vals, y_rug, marker="|", s=160, alpha=0.55)
        ymin, ymax = plt.ylim()
        plt.ylim(min(float(np.min(y_rug)) * 1.8, ymin), ymax)

    if x_max is not None:
        plt.xlim(left=0, right=float(x_max))

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Batch processing
# -----------------------------------------------------------------------------


METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("n_points_total_raw", "total_raw", "Total raw points per cluster"),
    ("n_points_pre_early_raw", "pre_early_raw", "Pre-lesion early raw points per cluster"),
    ("n_points_pre_late_raw", "pre_late_raw", "Pre-lesion late raw points per cluster"),
    ("n_points_post_early_raw", "post_early_raw", "Post-lesion early raw points per cluster"),
    ("n_points_post_late_raw", "post_late_raw", "Post-lesion late raw points per cluster"),
    (
        "n_points_shared_equal_per_group",
        "shared_equal_per_group",
        "Shared equalized points per group per cluster",
    ),
]


def _process_one_npz(
    npz_path: Path,
    *,
    cluster_key: str,
    file_key: str,
    file_map_key: str,
    vocalization_key: str,
    vocalization_only: bool,
    include_noise: bool,
    metadata_xlsx: Path,
    metadata_sheet: Optional[str],
    animal_id_col_candidates: Sequence[str],
    treatment_date_col_candidates: Sequence[str],
    exclude_treatment_day_from_post: bool,
    early_late_split_method: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    animal_id = _infer_animal_id(npz_path)

    try:
        points, animal_id, treatment_date = _load_points(
            npz_path,
            cluster_key=cluster_key,
            file_key=file_key,
            file_map_key=file_map_key,
            vocalization_key=vocalization_key,
            vocalization_only=vocalization_only,
            include_noise=include_noise,
            metadata_xlsx=metadata_xlsx,
            metadata_sheet=metadata_sheet,
            animal_id_col_candidates=animal_id_col_candidates,
            treatment_date_col_candidates=treatment_date_col_candidates,
            exclude_treatment_day_from_post=exclude_treatment_day_from_post,
        )
        counts_df = _cluster_count_table_from_points(
            points,
            animal_id=animal_id,
            npz_path=npz_path,
            early_late_split_method=early_late_split_method,
        )
        if counts_df.empty:
            return counts_df, {
                "animal_id": animal_id,
                "npz_path": str(npz_path),
                "treatment_date": str(treatment_date.date()),
                "ok": False,
                "message": "No valid clusters found after filtering.",
            }

        counts_df.insert(2, "treatment_date", pd.Timestamp(treatment_date).date().isoformat())
        return counts_df, {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "treatment_date": str(treatment_date.date()),
            "ok": True,
            "message": f"Found {len(counts_df)} clusters.",
        }
    except Exception as exc:
        return pd.DataFrame(), {
            "animal_id": animal_id,
            "npz_path": str(npz_path),
            "treatment_date": "",
            "ok": False,
            "message": f"{type(exc).__name__}: {exc}",
        }


def _save_metric_plots_for_df(
    df: pd.DataFrame,
    *,
    prefix: str,
    out_dir: Path,
    dpi: int,
    zoom_max: int,
    zoom_bin_width: int,
    focused_zoom_max: int,
    focused_zoom_bin_width: int,
) -> Dict[str, Path]:
    created: Dict[str, Path] = {}

    for col, short_name, label in METRIC_SPECS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue

        full_width = _auto_full_bin_width(vals)

        full_plot = out_dir / f"{prefix}_cluster_point_count_histogram_{short_name}.png"
        _plot_histogram(
            vals,
            out_path=full_plot,
            title=f"{prefix.replace('_', ' ')}: {label}",
            xlabel=label,
            dpi=dpi,
            target_bin_width=full_width,
        )
        created[f"{prefix}_{short_name}_full_plot"] = full_plot

        zoom_plot = out_dir / f"{prefix}_cluster_point_count_histogram_{short_name}_zoom_0_to_{int(zoom_max)}.png"
        _plot_histogram(
            vals,
            out_path=zoom_plot,
            title=f"{prefix.replace('_', ' ')}: {label} (zoom 0–{int(zoom_max)})",
            xlabel=label,
            dpi=dpi,
            target_bin_width=zoom_bin_width,
            x_max=float(zoom_max),
        )
        created[f"{prefix}_{short_name}_zoom_plot"] = zoom_plot

        focused_zoom_plot = out_dir / f"{prefix}_cluster_point_count_histogram_{short_name}_zoom_0_to_{int(focused_zoom_max)}.png"
        _plot_histogram(
            vals,
            out_path=focused_zoom_plot,
            title=f"{prefix.replace('_', ' ')}: {label} (zoom 0–{int(focused_zoom_max)})",
            xlabel=label,
            dpi=dpi,
            target_bin_width=focused_zoom_bin_width,
            x_max=float(focused_zoom_max),
        )
        created[f"{prefix}_{short_name}_focused_zoom_plot"] = focused_zoom_plot

    return created


def run_batch(
    *,
    root_dir: Path,
    metadata_xlsx: Path,
    out_dir: Path,
    recursive: bool,
    cluster_key: str,
    file_key: str,
    file_map_key: str,
    vocalization_key: str,
    vocalization_only: bool,
    include_noise: bool,
    metadata_sheet: Optional[str],
    animal_id_col_candidates: Sequence[str],
    treatment_date_col_candidates: Sequence[str],
    exclude_treatment_day_from_post: bool,
    early_late_split_method: str,
    zoom_max: int,
    zoom_bin_width: int,
    focused_zoom_max: int,
    focused_zoom_bin_width: int,
    dpi: int,
) -> Dict[str, Path]:
    root_dir = Path(root_dir)
    metadata_xlsx = Path(metadata_xlsx)
    out_dir = Path(out_dir)
    per_bird_csv_dir = out_dir / "per_bird_csv"
    per_bird_plots_dir = out_dir / "per_bird_plots"
    aggregate_dir = out_dir / "aggregate"
    _safe_mkdir(per_bird_csv_dir)
    _safe_mkdir(per_bird_plots_dir)
    _safe_mkdir(aggregate_dir)

    npz_files = _find_npz_files(root_dir, recursive=recursive)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under: {root_dir}")

    all_counts: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    created_paths: Dict[str, Path] = {}

    for npz_path in npz_files:
        counts_df, manifest = _process_one_npz(
            npz_path,
            cluster_key=cluster_key,
            file_key=file_key,
            file_map_key=file_map_key,
            vocalization_key=vocalization_key,
            vocalization_only=vocalization_only,
            include_noise=include_noise,
            metadata_xlsx=metadata_xlsx,
            metadata_sheet=metadata_sheet,
            animal_id_col_candidates=animal_id_col_candidates,
            treatment_date_col_candidates=treatment_date_col_candidates,
            exclude_treatment_day_from_post=exclude_treatment_day_from_post,
            early_late_split_method=early_late_split_method,
        )
        manifest_rows.append(manifest)
        animal_id = str(manifest["animal_id"])

        if counts_df.empty:
            continue

        counts_csv = per_bird_csv_dir / f"{animal_id}_cluster_point_counts.csv"
        counts_df.to_csv(counts_csv, index=False)
        created_paths[f"{animal_id}_csv"] = counts_csv

        created_paths.update(
            _save_metric_plots_for_df(
                counts_df,
                prefix=animal_id,
                out_dir=per_bird_plots_dir,
                dpi=dpi,
                zoom_max=zoom_max,
                zoom_bin_width=zoom_bin_width,
                focused_zoom_max=focused_zoom_max,
                focused_zoom_bin_width=focused_zoom_bin_width,
            )
        )

        all_counts.append(counts_df)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "run_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    created_paths["run_manifest"] = manifest_csv

    if not all_counts:
        return created_paths

    combined_df = pd.concat(all_counts, ignore_index=True)
    combined_csv = aggregate_dir / "all_birds_cluster_point_counts.csv"
    combined_df.to_csv(combined_csv, index=False)
    created_paths["aggregate_csv"] = combined_csv

    created_paths.update(
        _save_metric_plots_for_df(
            combined_df,
            prefix="all_birds",
            out_dir=aggregate_dir,
            dpi=dpi,
            zoom_max=zoom_max,
            zoom_bin_width=zoom_bin_width,
            focused_zoom_max=focused_zoom_max,
            focused_zoom_bin_width=focused_zoom_bin_width,
        )
    )

    return created_paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot per-bird and aggregate histograms of cluster sizes from NPZ files, "
            "including pre/post early/late splits and shared equalized counts."
        )
    )
    p.add_argument("--root-dir", required=True, help="Root folder containing NPZ files.")
    p.add_argument("--metadata-xlsx", required=True, help="Metadata workbook containing treatment dates.")
    p.add_argument(
        "--out-dir",
        default="/Volumes/my_own_SSD/points_per_cluster_histograms",
        help="Output folder for CSVs and plots.",
    )
    p.add_argument(
        "--cluster-key",
        default="hdbscan_labels",
        help="NPZ key containing cluster labels. Default: hdbscan_labels",
    )
    p.add_argument(
        "--file-key",
        default="file_indices",
        help="NPZ key containing file indices. Default: file_indices",
    )
    p.add_argument(
        "--file-map-key",
        default="file_map",
        help="NPZ key containing file_map. Default: file_map",
    )
    p.add_argument(
        "--vocalization-key",
        default="vocalization",
        help="NPZ key containing vocalization mask. Default: vocalization",
    )
    p.add_argument(
        "--no-vocalization-only",
        action="store_true",
        help="Do not restrict counts to vocalization==True when that key is present.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under root-dir for .npz files.",
    )
    p.add_argument(
        "--include-noise",
        action="store_true",
        help="Include negative/noise labels such as -1. Default excludes them.",
    )
    p.add_argument(
        "--metadata-sheet",
        default=None,
        help="Optional metadata sheet name. Default searches all sheets.",
    )
    p.add_argument(
        "--animal-id-col-candidates",
        default="Animal ID,animal_id,AnimalID,bird_id,Bird ID",
        help="Comma-separated candidate column names for animal ID in the metadata workbook.",
    )
    p.add_argument(
        "--treatment-date-col-candidates",
        default="Treatment date,treatment_date,Surgery date,Date",
        help="Comma-separated candidate column names for treatment date in the metadata workbook.",
    )
    p.add_argument(
        "--include-treatment-day-in-post",
        action="store_true",
        help="Include treatment day in the post period. Default excludes treatment day from both periods.",
    )
    p.add_argument(
        "--early-late-split-method",
        default="file_median",
        choices=["file_median", "file_half"],
        help="How to split files within pre/post into early and late. Default: file_median",
    )
    p.add_argument(
        "--zoom-max",
        type=int,
        default=5000,
        help="Upper x-limit for zoomed low-count histograms. Default: 5000",
    )
    p.add_argument(
        "--zoom-bin-width",
        type=int,
        default=100,
        help="Bin width for the 0-to-zoom-max histograms. Default: 100",
    )
    p.add_argument(
        "--focused-zoom-max",
        type=int,
        default=2500,
        help="Upper x-limit for extra focused low-count histograms. Default: 2500",
    )
    p.add_argument(
        "--focused-zoom-bin-width",
        type=int,
        default=50,
        help="Bin width for the extra focused 0-to-focused-zoom-max histograms. Default: 50",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution. Default: 200",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    animal_cols = [x.strip() for x in str(args.animal_id_col_candidates).split(",") if x.strip()]
    treatment_cols = [x.strip() for x in str(args.treatment_date_col_candidates).split(",") if x.strip()]

    created = run_batch(
        root_dir=Path(args.root_dir),
        metadata_xlsx=Path(args.metadata_xlsx),
        out_dir=Path(args.out_dir),
        recursive=bool(args.recursive),
        cluster_key=str(args.cluster_key),
        file_key=str(args.file_key),
        file_map_key=str(args.file_map_key),
        vocalization_key=str(args.vocalization_key),
        vocalization_only=not bool(args.no_vocalization_only),
        include_noise=bool(args.include_noise),
        metadata_sheet=args.metadata_sheet,
        animal_id_col_candidates=animal_cols,
        treatment_date_col_candidates=treatment_cols,
        exclude_treatment_day_from_post=not bool(args.include_treatment_day_in_post),
        early_late_split_method=str(args.early_late_split_method),
        zoom_max=int(args.zoom_max),
        zoom_bin_width=int(args.zoom_bin_width),
        focused_zoom_max=int(args.focused_zoom_max),
        focused_zoom_bin_width=int(args.focused_zoom_bin_width),
        dpi=int(args.dpi),
    )

    print("Done.")
    for key, path in created.items():
        print(f"[{key}] {path}")


if __name__ == "__main__":
    main()
