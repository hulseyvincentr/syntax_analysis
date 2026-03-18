
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umap_pre_post_early_late_cluster_variance_bc.py

Single-bird UMAP / variance / Bhattacharyya-coefficient pipeline.

What this version adds
----------------------
- Keeps the public interface expected by the batch runner:
    * UMAPEarlyLateConfig
    * run_one_bird(cfg)
- Saves BOTH:
    * raw/unbalanced 2x4 cluster figures
    * equal-sized 2x4 cluster figures
- Adds a NEW large panel on the far right of each cluster figure:
    * all pre-lesion points in purple
    * all post-lesion points in green
  so you can visually assess the overall pre/post cluster shift.

Figure layout
-------------
Each cluster figure is now a 2 x 4 layout:
  [pre early] [pre late] [pre early/late overlap] [          ]
  [post early][post late][post early/late overlap][ pre/post ]
  [                                              tall panel  ]

Outputs
-------
out_dir/
  clusters/
    <prefix>_cluster<cluster>_pre_post_early_late_umap_raw_groups.png
    <prefix>_cluster<cluster>_pre_post_early_late_umap_equal_groups.png
  <prefix>_cluster_variance_bc_summary.csv

Notes
-----
- This script tries to be robust to different NPZ file_map formats.
- If UMAP is not installed, it falls back to PCA for the 2D embedding.
- Treatment date is looked up from the metadata workbook by animal ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import math
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import umap.umap_ as umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False
    umap = None

from sklearn.decomposition import PCA


# =============================================================================
# Config
# =============================================================================

@dataclass
class UMAPEarlyLateConfig:
    npz_path: Path
    metadata_xlsx: Path

    # NPZ keys
    array_key: str = "predictions"
    cluster_key: str = "hdbscan_labels"
    file_key: str = "file_indices"
    vocalization_key: str = "vocalization"
    file_map_key: str = "file_map"

    # Filtering
    include_noise: bool = False
    only_cluster_id: Optional[int] = None
    vocalization_only: bool = True

    # Minimum / maximum points used for analysis
    min_points_per_period: int = 200
    max_points_per_period: Optional[int] = None
    max_points_per_period_for_plot: Optional[int] = None

    # Pre/post splitting
    exclude_treatment_day_from_post: bool = False
    early_late_split_method: str = "file_median"   # file_median or file_half

    # Embedding params
    random_seed: int = 0     # -1 means no fixed seed
    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "euclidean"

    # Output
    out_dir: Optional[Path] = None
    out_prefix: Optional[str] = None
    dpi: int = 200

    # Overlap / BC params
    bc_bins: int = 100
    overlap_density_bins: int = 180
    overlap_density_gamma: float = 0.55

    # Metadata lookup defaults
    metadata_sheet: Optional[str] = None
    animal_id_col_candidates: Tuple[str, ...] = ("Animal ID", "animal_id", "Bird", "bird", "ID", "Animal")
    treatment_date_col_candidates: Tuple[str, ...] = ("Treatment date", "treatment_date", "Surgery date", "Date")
    date_formats_hint: Tuple[str, ...] = ("%Y-%m-%d", "%Y_%m_%d")


# =============================================================================
# Utility helpers
# =============================================================================

PURPLE = (0.78, 0.00, 0.95)
GREEN = (0.00, 0.85, 0.00)
SCATTER_PURPLE = "#8A2BE2"
SCATTER_GREEN = "#66CC66"


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_animal_id(npz_path: Path) -> str:
    return npz_path.stem.split("_")[0]


def _coerce_path(value: Optional[Path | str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


def _normalize_cluster_value(x: Any) -> Any:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def _to_numpy_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _random_state_from_seed(seed: int) -> Optional[int]:
    return None if int(seed) == -1 else int(seed)


def _rng(seed: int) -> np.random.Generator:
    rs = _random_state_from_seed(seed)
    return np.random.default_rng(rs)


def _stack_nonempty_xy(*arrays: np.ndarray) -> np.ndarray:
    parts: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
            parts.append(arr)
    if not parts:
        return np.empty((0, 2), dtype=float)
    return np.vstack(parts)


def _subsample_indices(n: int, max_n: Optional[int], rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(n, dtype=int)
    if (max_n is None) or (n <= max_n):
        return idx
    return np.sort(rng.choice(idx, size=int(max_n), replace=False))


def _subsample_array(arr: np.ndarray, max_n: Optional[int], rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] == 0:
        return arr
    idx = _subsample_indices(arr.shape[0], max_n, rng)
    return arr[idx]


def _maybe_nan_ratio(num: float, den: float) -> float:
    if den is None or np.isnan(den) or float(den) == 0.0:
        return np.nan
    return float(num) / float(den)


def _cluster_change_label(pre_present_any: bool, post_present_any: bool) -> str:
    if pre_present_any and post_present_any:
        return "present_pre_and_post"
    if pre_present_any and not post_present_any:
        return "present_pre_only"
    if (not pre_present_any) and post_present_any:
        return "present_post_only"
    return "absent_pre_and_post"


def _format_dt_for_title(x: Optional[pd.Timestamp]) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def _title_range_text(dts: Sequence[pd.Timestamp]) -> str:
    if len(dts) == 0:
        return "NA to NA"
    vals = pd.to_datetime(pd.Series(dts), errors="coerce").dropna().sort_values()
    if len(vals) == 0:
        return "NA to NA"
    return f"{vals.iloc[0].strftime('%Y-%m-%d')} to {vals.iloc[-1].strftime('%Y-%m-%d')}"


# =============================================================================
# Metadata lookup
# =============================================================================

def _canonical_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_first_matching_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    canon_cols = {_canonical_name(c): c for c in columns}
    for cand in candidates:
        c = canon_cols.get(_canonical_name(cand))
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


def _lookup_treatment_date(metadata_xlsx: Path, animal_id: str, cfg: UMAPEarlyLateConfig) -> pd.Timestamp:
    xlsx = pd.ExcelFile(metadata_xlsx)

    candidate_sheets: List[str]
    if cfg.metadata_sheet is not None and cfg.metadata_sheet in xlsx.sheet_names:
        candidate_sheets = [cfg.metadata_sheet]
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

        animal_col = _find_first_matching_column(df.columns, cfg.animal_id_col_candidates)
        date_col = _find_first_matching_column(df.columns, cfg.treatment_date_col_candidates)
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


# =============================================================================
# NPZ / file-map parsing
# =============================================================================

def _normalize_file_map(file_map_obj: Any) -> Dict[int, str]:
    """
    Try to coerce a variety of file_map encodings into:
        {file_index: file_name}
    """
    mapping: Dict[int, str] = {}

    if file_map_obj is None:
        return mapping

    # object array with single dict inside
    if isinstance(file_map_obj, np.ndarray):
        if file_map_obj.shape == () and file_map_obj.dtype == object:
            return _normalize_file_map(file_map_obj.item())
        if file_map_obj.dtype == object:
            try:
                obj = file_map_obj.tolist()
            except Exception:
                obj = list(file_map_obj)
            return _normalize_file_map(obj)
        # non-object array: assume sequence of file names
        for i, v in enumerate(file_map_obj):
            mapping[int(i)] = str(v)
        return mapping

    # dict-like
    if isinstance(file_map_obj, dict):
        for k, v in file_map_obj.items():
            try:
                kk = int(k)
            except Exception:
                continue
            mapping[kk] = str(v)
        return mapping

    # list / tuple
    if isinstance(file_map_obj, (list, tuple)):
        # Sometimes encoded as list of pairs
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

    # pandas series / dataframe
    if isinstance(file_map_obj, pd.Series):
        for k, v in file_map_obj.items():
            try:
                mapping[int(k)] = str(v)
            except Exception:
                pass
        return mapping

    # fallback
    return mapping


_DATE_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
_DATETIME_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[^0-9]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})")


def _parse_datetime_from_filename(name: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Returns:
        (recording_datetime, recording_date)
    """
    s = str(name)

    mdt = _DATETIME_RE.search(s)
    if mdt is not None:
        y, mo, d, hh, mm, ss = map(int, mdt.groups())
        try:
            dt = pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, second=ss)
            return dt, dt.normalize()
        except Exception:
            pass

    md = _DATE_RE.search(s)
    if md is not None:
        y, mo, d = map(int, md.groups())
        try:
            dt = pd.Timestamp(year=y, month=mo, day=d)
            return dt, dt.normalize()
        except Exception:
            pass

    return None, None


def _build_file_metadata(npz: np.lib.npyio.NpzFile, cfg: UMAPEarlyLateConfig) -> pd.DataFrame:
    file_map_obj = npz[cfg.file_map_key] if cfg.file_map_key in npz.files else None
    mapping = _normalize_file_map(file_map_obj)

    if len(mapping) == 0 and cfg.file_key in npz.files:
        # If file_map is absent, infer file indices only.
        file_indices = np.asarray(npz[cfg.file_key]).astype(int)
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

    # If datetime missing, sort by file_index as stable fallback
    if "recording_datetime" not in file_df.columns:
        file_df["recording_datetime"] = pd.NaT
    file_df["sort_datetime"] = pd.to_datetime(file_df["recording_datetime"], errors="coerce")
    file_df["sort_datetime"] = file_df["sort_datetime"].fillna(pd.Timestamp("1900-01-01"))
    file_df = file_df.sort_values(["sort_datetime", "file_index"]).reset_index(drop=True)

    return file_df


# =============================================================================
# Point table creation
# =============================================================================

def _load_point_table(cfg: UMAPEarlyLateConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Timestamp, str]:
    npz_path = Path(cfg.npz_path)
    animal_id = _infer_animal_id(npz_path)

    with np.load(npz_path, allow_pickle=True) as npz:
        if cfg.array_key not in npz.files:
            raise KeyError(f"array_key={cfg.array_key!r} not found in NPZ. Available keys: {npz.files}")
        if cfg.cluster_key not in npz.files:
            raise KeyError(f"cluster_key={cfg.cluster_key!r} not found in NPZ. Available keys: {npz.files}")
        if cfg.file_key not in npz.files:
            raise KeyError(f"file_key={cfg.file_key!r} not found in NPZ. Available keys: {npz.files}")

        raw = _to_numpy_2d(np.asarray(npz[cfg.array_key]))
        clusters = np.asarray(npz[cfg.cluster_key])
        file_indices = np.asarray(npz[cfg.file_key]).astype(int)

        if len(raw) != len(clusters) or len(raw) != len(file_indices):
            raise ValueError(
                f"Length mismatch among raw array ({len(raw)}), clusters ({len(clusters)}), "
                f"and file indices ({len(file_indices)})."
            )

        if cfg.vocalization_only and (cfg.vocalization_key in npz.files):
            vocal = np.asarray(npz[cfg.vocalization_key]).astype(bool)
            if len(vocal) != len(raw):
                raise ValueError(
                    f"Vocalization key length mismatch: {len(vocal)} vs {len(raw)} points."
                )
        else:
            vocal = np.ones(len(raw), dtype=bool)

        file_df = _build_file_metadata(npz, cfg)

    treatment_date = _lookup_treatment_date(Path(cfg.metadata_xlsx), animal_id, cfg)

    points = pd.DataFrame(
        {
            "point_index": np.arange(len(raw), dtype=int),
            "cluster": [_normalize_cluster_value(x) for x in clusters],
            "file_index": file_indices,
            "keep": vocal.astype(bool),
        }
    )

    points = points.merge(
        file_df[["file_index", "file_name", "recording_datetime", "recording_date"]],
        on="file_index",
        how="left",
        validate="many_to_one",
    )

    if points["recording_date"].isna().all():
        raise ValueError(
            "Could not parse recording dates from file names. "
            "Expected file names to contain YYYY-MM-DD or YYYY_MM_DD."
        )

    points = points.loc[points["keep"]].copy()

    if not cfg.include_noise:
        points = points.loc[points["cluster"] != -1].copy()

    if cfg.only_cluster_id is not None:
        points = points.loc[points["cluster"] == cfg.only_cluster_id].copy()

    points["recording_date"] = pd.to_datetime(points["recording_date"], errors="coerce").dt.normalize()
    points["recording_datetime"] = pd.to_datetime(points["recording_datetime"], errors="coerce")

    if cfg.exclude_treatment_day_from_post:
        points["period"] = np.where(points["recording_date"] < treatment_date, "pre", "other")
        points.loc[points["recording_date"] > treatment_date, "period"] = "post"
    else:
        points["period"] = np.where(points["recording_date"] < treatment_date, "pre", "post")

    points = points.loc[points["period"].isin(["pre", "post"])].copy()

    return raw, clusters, points, treatment_date, animal_id


# =============================================================================
# File-based early/late splitting
# =============================================================================

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
        # file_median: give the extra file to the early group when odd
        cut = int(math.ceil(n / 2.0))

    early = files[:cut]
    late = files[cut:]
    return early, late


def _build_cluster_group_indices(cluster_points: pd.DataFrame, method: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    for period in ["pre", "post"]:
        sub = cluster_points.loc[cluster_points["period"] == period].copy()
        unique_files = (
            sub[["file_index", "recording_datetime", "recording_date"]]
            .drop_duplicates()
            .sort_values(["recording_datetime", "recording_date", "file_index"])
            .reset_index(drop=True)
        )
        early_files, late_files = _split_files_early_late(unique_files, method=method)

        out[f"{period}_all"] = sub["point_index"].to_numpy(dtype=int)
        out[f"{period}_early"] = sub.loc[sub["file_index"].isin(early_files), "point_index"].to_numpy(dtype=int)
        out[f"{period}_late"] = sub.loc[sub["file_index"].isin(late_files), "point_index"].to_numpy(dtype=int)

    return out


# =============================================================================
# Metrics
# =============================================================================

def _rms_radius(x: np.ndarray) -> float:
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    c = x.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((x - c) ** 2, axis=1))))


def _trace_cov(x: np.ndarray) -> float:
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan
    c = np.cov(x, rowvar=False)
    c = np.atleast_2d(c)
    return float(np.trace(c))


def _centroid_shift(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def _hist2d_prob(xy: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((len(yedges) - 1, len(xedges) - 1), dtype=float)
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xedges, yedges])
    H = H.T.astype(float)
    s = H.sum()
    if s > 0:
        H /= s
    return H


def _bhattacharyya_coefficient_2d(xy_a: np.ndarray, xy_b: np.ndarray, bins: int = 100) -> float:
    xy_a = np.asarray(xy_a)
    xy_b = np.asarray(xy_b)

    if len(xy_a) == 0 or len(xy_b) == 0:
        return np.nan

    both = _stack_nonempty_xy(xy_a, xy_b)
    xmin, ymin = both.min(axis=0)
    xmax, ymax = both.max(axis=0)

    xr = xmax - xmin
    yr = ymax - ymin
    if xr <= 0:
        xr = 1.0
    if yr <= 0:
        yr = 1.0

    xpad = 0.03 * xr
    ypad = 0.03 * yr

    xedges = np.linspace(xmin - xpad, xmax + xpad, int(bins) + 1)
    yedges = np.linspace(ymin - ypad, ymax + ypad, int(bins) + 1)

    p = _hist2d_prob(xy_a, xedges, yedges)
    q = _hist2d_prob(xy_b, xedges, yedges)

    return float(np.sum(np.sqrt(p * q)))


# =============================================================================
# Embedding
# =============================================================================

def _fit_2d_embedding(x: np.ndarray, cfg: UMAPEarlyLateConfig) -> np.ndarray:
    x = np.asarray(x)
    if len(x) == 0:
        return np.empty((0, 2), dtype=float)
    if len(x) == 1:
        return np.array([[0.0, 0.0]], dtype=float)

    if _HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(int(cfg.n_neighbors), max(2, len(x) - 1)),
            min_dist=float(cfg.min_dist),
            metric=str(cfg.metric),
            random_state=_random_state_from_seed(int(cfg.random_seed)),
        )
        return np.asarray(reducer.fit_transform(x), dtype=float)

    warnings.warn("UMAP is not available; falling back to PCA for 2D embedding.", RuntimeWarning)
    return np.asarray(PCA(n_components=2).fit_transform(x), dtype=float)


# =============================================================================
# Plot helpers
# =============================================================================

def _scatter_panel(ax, xy: np.ndarray, color: str, title: str) -> None:
    ax.scatter(xy[:, 0], xy[:, 1], s=8, c=color, alpha=0.55, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=13, color=color)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")


def _plot_overlap_density_rgb(
    ax,
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    *,
    bins: int = 180,
    gamma: float = 0.55,
    title: str = "",
    color_a: Tuple[float, float, float] = PURPLE,
    color_b: Tuple[float, float, float] = GREEN,
) -> None:
    xy_a = np.asarray(xy_a) if xy_a is not None else np.empty((0, 2))
    xy_b = np.asarray(xy_b) if xy_b is not None else np.empty((0, 2))

    ax.set_facecolor("black")

    if xy_a.size == 0 and xy_b.size == 0:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title + "\n(no points)", fontsize=11)
        return

    both = _stack_nonempty_xy(xy_a, xy_b)

    xmin, ymin = both.min(axis=0)
    xmax, ymax = both.max(axis=0)

    xr = xmax - xmin
    yr = ymax - ymin
    if xr <= 0:
        xr = 1.0
    if yr <= 0:
        yr = 1.0

    xpad = 0.03 * xr
    ypad = 0.03 * yr

    xedges = np.linspace(xmin - xpad, xmax + xpad, int(bins) + 1)
    yedges = np.linspace(ymin - ypad, ymax + ypad, int(bins) + 1)

    Ha = _hist2d_prob(xy_a, xedges, yedges)
    Hb = _hist2d_prob(xy_b, xedges, yedges)

    if Ha.max() > 0:
        Ha = (Ha / Ha.max()) ** float(gamma)
    if Hb.max() > 0:
        Hb = (Hb / Hb.max()) ** float(gamma)

    rgb = np.zeros((Ha.shape[0], Ha.shape[1], 3), dtype=float)
    rgb[..., 0] += color_a[0] * Ha
    rgb[..., 1] += color_a[1] * Ha
    rgb[..., 2] += color_a[2] * Ha

    rgb[..., 0] += color_b[0] * Hb
    rgb[..., 1] += color_b[1] * Hb
    rgb[..., 2] += color_b[2] * Hb
    rgb = np.clip(rgb, 0, 1)

    ax.imshow(
        rgb,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def _plot_cluster_figure(
    *,
    save_path: Path,
    animal_id: str,
    cluster_id: Any,
    treatment_date: pd.Timestamp,
    pre_early_xy: np.ndarray,
    pre_late_xy: np.ndarray,
    post_early_xy: np.ndarray,
    post_late_xy: np.ndarray,
    pre_early_dates: Sequence[pd.Timestamp],
    pre_late_dates: Sequence[pd.Timestamp],
    post_early_dates: Sequence[pd.Timestamp],
    post_late_dates: Sequence[pd.Timestamp],
    bc_pre_early_vs_late: float,
    bc_post_early_vs_late: float,
    bc_pre_vs_post: float,
    centroid_shift_value: float,
    post_over_pre_rms: float,
    post_over_pre_trace: float,
    cfg: UMAPEarlyLateConfig,
    label_suffix: str,
) -> None:
    fig = plt.figure(figsize=(19.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.15, 1.0, 1.15])

    ax_pre_early = fig.add_subplot(gs[0, 0])
    ax_pre_late = fig.add_subplot(gs[0, 1])
    ax_post_early = fig.add_subplot(gs[1, 0])
    ax_post_late = fig.add_subplot(gs[1, 1])

    ax_pre_overlap = fig.add_subplot(gs[0, 2])
    ax_post_overlap = fig.add_subplot(gs[1, 2])

    # NEW large panel spanning both rows
    ax_prepost_overlap = fig.add_subplot(gs[:, 3])

    _scatter_panel(
        ax_pre_early,
        pre_early_xy,
        SCATTER_PURPLE,
        (
            f"Pre-lesion early\n"
            f"{_title_range_text(pre_early_dates)}\n"
            f"N={len(pre_early_xy)}"
        ),
    )

    _scatter_panel(
        ax_pre_late,
        pre_late_xy,
        SCATTER_GREEN,
        (
            f"Pre-lesion late\n"
            f"{_title_range_text(pre_late_dates)}\n"
            f"N={len(pre_late_xy)}"
        ),
    )

    _scatter_panel(
        ax_post_early,
        post_early_xy,
        SCATTER_PURPLE,
        (
            f"Post-lesion early\n"
            f"{_title_range_text(post_early_dates)}\n"
            f"N={len(post_early_xy)}"
        ),
    )

    _scatter_panel(
        ax_post_late,
        post_late_xy,
        SCATTER_GREEN,
        (
            f"Post-lesion late\n"
            f"{_title_range_text(post_late_dates)}\n"
            f"N={len(post_late_xy)}"
        ),
    )

    _plot_overlap_density_rgb(
        ax_pre_overlap,
        pre_early_xy,
        pre_late_xy,
        bins=cfg.overlap_density_bins,
        gamma=cfg.overlap_density_gamma,
        title=f"Pre early vs late overlap\nBC={bc_pre_early_vs_late:.3f}" if not np.isnan(bc_pre_early_vs_late) else "Pre early vs late overlap\nBC=nan",
        color_a=PURPLE,
        color_b=GREEN,
    )

    _plot_overlap_density_rgb(
        ax_post_overlap,
        post_early_xy,
        post_late_xy,
        bins=cfg.overlap_density_bins,
        gamma=cfg.overlap_density_gamma,
        title=f"Post early vs late overlap\nBC={bc_post_early_vs_late:.3f}" if not np.isnan(bc_post_early_vs_late) else "Post early vs late overlap\nBC=nan",
        color_a=PURPLE,
        color_b=GREEN,
    )

    pre_all_xy = _stack_nonempty_xy(pre_early_xy, pre_late_xy)
    post_all_xy = _stack_nonempty_xy(post_early_xy, post_late_xy)

    _plot_overlap_density_rgb(
        ax_prepost_overlap,
        pre_all_xy,
        post_all_xy,
        bins=cfg.overlap_density_bins,
        gamma=cfg.overlap_density_gamma,
        title=(
            f"Pre vs post overlap\n"
            f"pre N={len(pre_all_xy)} | post N={len(post_all_xy)}\n"
            f"BC={bc_pre_vs_post:.3f}"
        ) if not np.isnan(bc_pre_vs_post) else (
            f"Pre vs post overlap\n"
            f"pre N={len(pre_all_xy)} | post N={len(post_all_xy)}\n"
            f"BC=nan"
        ),
        color_a=PURPLE,
        color_b=GREEN,
    )

    n_each = min(len(pre_early_xy), len(pre_late_xy), len(post_early_xy), len(post_late_xy))

    fig.suptitle(
        (
            f"{animal_id} cluster {cluster_id} early/late UMAPs ({label_suffix})\n"
            f"n_each={n_each} | pre BC={bc_pre_early_vs_late:.3f} | post BC={bc_post_early_vs_late:.3f} "
            f"| pre/post BC={bc_pre_vs_post:.3f} | treatment={_format_dt_for_title(treatment_date)}\n"
            f"centroid shift={centroid_shift_value:.3f} | post/pre rms={post_over_pre_rms:.3f} "
            f"| post/pre trace(cov)={post_over_pre_trace:.3f}"
        ),
        fontsize=15,
    )

    fig.savefig(save_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Per-cluster analysis
# =============================================================================

def _build_group_arrays(
    raw_all: np.ndarray,
    points_df: pd.DataFrame,
    group_point_indices: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, point_indices in group_point_indices.items():
        out[key] = raw_all[np.asarray(point_indices, dtype=int)]
    return out


def _extract_group_dates(points_df: pd.DataFrame, point_indices: np.ndarray) -> List[pd.Timestamp]:
    if len(point_indices) == 0:
        return []
    sub = points_df.loc[points_df["point_index"].isin(point_indices), ["recording_date"]].copy()
    vals = pd.to_datetime(sub["recording_date"], errors="coerce").dropna().sort_values().tolist()
    return [pd.Timestamp(v) for v in vals]


def _balanced_group_indices(
    group_arrays: Dict[str, np.ndarray],
    seed: int,
) -> Tuple[Optional[int], Dict[str, np.ndarray]]:
    rng = _rng(seed)
    sizes = {
        "pre_early": len(group_arrays["pre_early"]),
        "pre_late": len(group_arrays["pre_late"]),
        "post_early": len(group_arrays["post_early"]),
        "post_late": len(group_arrays["post_late"]),
    }
    n_each = min(sizes.values()) if len(sizes) else 0
    if n_each <= 0:
        return None, {k: np.array([], dtype=int) for k in sizes}

    out: Dict[str, np.ndarray] = {}
    for key in sizes:
        idx = np.arange(sizes[key], dtype=int)
        if len(idx) > n_each:
            idx = np.sort(rng.choice(idx, size=int(n_each), replace=False))
        out[key] = idx
    return int(n_each), out


def _compute_summary_for_pair(pre_x: np.ndarray, post_x: np.ndarray) -> Dict[str, float]:
    pre_rms = _rms_radius(pre_x)
    post_rms = _rms_radius(post_x)
    pre_trace = _trace_cov(pre_x)
    post_trace = _trace_cov(post_x)
    return {
        "centroid_shift": _centroid_shift(pre_x, post_x),
        "pre_rms_radius": pre_rms,
        "post_rms_radius": post_rms,
        "post_over_pre_rms_radius": _maybe_nan_ratio(post_rms, pre_rms),
        "pre_trace_cov": pre_trace,
        "post_trace_cov": post_trace,
        "post_over_pre_trace_cov": _maybe_nan_ratio(post_trace, pre_trace),
    }


def _analyze_one_cluster(
    *,
    cluster_id: Any,
    raw_all: np.ndarray,
    points_df: pd.DataFrame,
    treatment_date: pd.Timestamp,
    animal_id: str,
    cfg: UMAPEarlyLateConfig,
    clusters_out_dir: Path,
) -> Dict[str, Any]:
    cdf = points_df.loc[points_df["cluster"] == cluster_id].copy()

    group_point_indices = _build_cluster_group_indices(cdf, method=cfg.early_late_split_method)
    group_raw = _build_group_arrays(raw_all, cdf, group_point_indices)

    # Raw counts before any subsampling
    n_pre_total_raw = len(group_raw["pre_all"])
    n_post_total_raw = len(group_raw["post_all"])
    n_pre_early_raw = len(group_raw["pre_early"])
    n_pre_late_raw = len(group_raw["pre_late"])
    n_post_early_raw = len(group_raw["post_early"])
    n_post_late_raw = len(group_raw["post_late"])

    pre_present_any = n_pre_total_raw > 0
    post_present_any = n_post_total_raw > 0

    row: Dict[str, Any] = {
        "animal_id": animal_id,
        "cluster": cluster_id,
        "cluster_change": _cluster_change_label(pre_present_any, post_present_any),
        "pre_present_any": bool(pre_present_any),
        "post_present_any": bool(post_present_any),

        "n_pre_total_raw": int(n_pre_total_raw),
        "n_post_total_raw": int(n_post_total_raw),
        "n_pre_early_raw": int(n_pre_early_raw),
        "n_pre_late_raw": int(n_pre_late_raw),
        "n_post_early_raw": int(n_post_early_raw),
        "n_post_late_raw": int(n_post_late_raw),

        "bc_pre_early_vs_late": np.nan,
        "bc_post_early_vs_late": np.nan,
        "bc_pre_vs_post": np.nan,

        "centroid_shift_raw": np.nan,
        "pre_rms_radius_raw": np.nan,
        "post_rms_radius_raw": np.nan,
        "post_over_pre_rms_radius": np.nan,
        "pre_trace_cov_raw": np.nan,
        "post_trace_cov_raw": np.nan,
        "post_over_pre_trace_cov": np.nan,

        "centroid_shift_umap": np.nan,
        "pre_rms_radius_umap": np.nan,
        "post_rms_radius_umap": np.nan,
        "post_over_pre_rms_radius_umap": np.nan,

        "bc_pre_early_vs_late_equal_groups": np.nan,
        "bc_post_early_vs_late_equal_groups": np.nan,

        "centroid_shift_raw_equal_groups": np.nan,
        "pre_rms_radius_raw_equal_groups": np.nan,
        "post_rms_radius_raw_equal_groups": np.nan,
        "post_over_pre_rms_radius_equal_groups": np.nan,
        "pre_trace_cov_raw_equal_groups": np.nan,
        "post_trace_cov_raw_equal_groups": np.nan,
        "post_over_pre_trace_cov_equal_groups": np.nan,

        "centroid_shift_umap_equal_groups": np.nan,
        "pre_rms_radius_umap_equal_groups": np.nan,
        "post_rms_radius_umap_equal_groups": np.nan,
        "post_over_pre_rms_radius_umap_equal_groups": np.nan,
    }

    # Raw-space overall pre vs post metrics (all available raw points)
    raw_pre_all = _stack_raw(group_raw["pre_early"], group_raw["pre_late"])
    raw_post_all = _stack_raw(group_raw["post_early"], group_raw["post_late"])
    raw_pair = _compute_summary_for_pair(raw_pre_all, raw_post_all)
    row["centroid_shift_raw"] = raw_pair["centroid_shift"]
    row["pre_rms_radius_raw"] = raw_pair["pre_rms_radius"]
    row["post_rms_radius_raw"] = raw_pair["post_rms_radius"]
    row["post_over_pre_rms_radius"] = raw_pair["post_over_pre_rms_radius"]
    row["pre_trace_cov_raw"] = raw_pair["pre_trace_cov"]
    row["post_trace_cov_raw"] = raw_pair["post_trace_cov"]
    row["post_over_pre_trace_cov"] = raw_pair["post_over_pre_trace_cov"]

    # Subsample, if requested, before embedding / plotting
    local_rng = _rng(cfg.random_seed)

    group_raw_plot: Dict[str, np.ndarray] = {}
    for key in ["pre_early", "pre_late", "post_early", "post_late"]:
        group_raw_plot[key] = _subsample_array(group_raw[key], cfg.max_points_per_period, local_rng)

    # Require all four groups to be non-empty to make the 4-group figures
    all_groups_present = all(len(group_raw_plot[k]) > 0 for k in ["pre_early", "pre_late", "post_early", "post_late"])

    # UMAP/raw metrics on the subsampled (but still unbalanced) data
    if all_groups_present:
        raw_for_embedding = np.vstack(
            [
                group_raw_plot["pre_early"],
                group_raw_plot["pre_late"],
                group_raw_plot["post_early"],
                group_raw_plot["post_late"],
            ]
        )

        embeds = _fit_2d_embedding(raw_for_embedding, cfg)

        n1 = len(group_raw_plot["pre_early"])
        n2 = len(group_raw_plot["pre_late"])
        n3 = len(group_raw_plot["post_early"])
        n4 = len(group_raw_plot["post_late"])

        xy_pre_early = embeds[:n1]
        xy_pre_late = embeds[n1:n1 + n2]
        xy_post_early = embeds[n1 + n2:n1 + n2 + n3]
        xy_post_late = embeds[n1 + n2 + n3:n1 + n2 + n3 + n4]

        xy_pre_all = _stack_nonempty_xy(xy_pre_early, xy_pre_late)
        xy_post_all = _stack_nonempty_xy(xy_post_early, xy_post_late)

        # Unbalanced BCs
        row["bc_pre_early_vs_late"] = _bhattacharyya_coefficient_2d(
            xy_pre_early, xy_pre_late, bins=cfg.bc_bins
        )
        row["bc_post_early_vs_late"] = _bhattacharyya_coefficient_2d(
            xy_post_early, xy_post_late, bins=cfg.bc_bins
        )
        row["bc_pre_vs_post"] = _bhattacharyya_coefficient_2d(
            xy_pre_all, xy_post_all, bins=cfg.bc_bins
        )

        umap_pair = _compute_summary_for_pair(xy_pre_all, xy_post_all)
        row["centroid_shift_umap"] = umap_pair["centroid_shift"]
        row["pre_rms_radius_umap"] = umap_pair["pre_rms_radius"]
        row["post_rms_radius_umap"] = umap_pair["post_rms_radius"]
        row["post_over_pre_rms_radius_umap"] = umap_pair["post_over_pre_rms_radius"]

        # Save raw/unbalanced figure
        raw_fig_path = clusters_out_dir / f"{cfg.out_prefix}_cluster{cluster_id}_pre_post_early_late_umap_raw_groups.png"
        _plot_cluster_figure(
            save_path=raw_fig_path,
            animal_id=animal_id,
            cluster_id=cluster_id,
            treatment_date=treatment_date,
            pre_early_xy=xy_pre_early,
            pre_late_xy=xy_pre_late,
            post_early_xy=xy_post_early,
            post_late_xy=xy_post_late,
            pre_early_dates=_extract_group_dates(points_df, group_point_indices["pre_early"]),
            pre_late_dates=_extract_group_dates(points_df, group_point_indices["pre_late"]),
            post_early_dates=_extract_group_dates(points_df, group_point_indices["post_early"]),
            post_late_dates=_extract_group_dates(points_df, group_point_indices["post_late"]),
            bc_pre_early_vs_late=row["bc_pre_early_vs_late"],
            bc_post_early_vs_late=row["bc_post_early_vs_late"],
            bc_pre_vs_post=row["bc_pre_vs_post"],
            centroid_shift_value=row["centroid_shift_umap"],
            post_over_pre_rms=row["post_over_pre_rms_radius_umap"],
            post_over_pre_trace=row["post_over_pre_trace_cov"],
            cfg=cfg,
            label_suffix="raw groups",
        )

        # Equal-sized groups
        n_each, eq_idx = _balanced_group_indices(group_raw_plot, cfg.random_seed)
        if n_each is not None and n_each > 0:
            eq_pre_early_raw = group_raw_plot["pre_early"][eq_idx["pre_early"]]
            eq_pre_late_raw = group_raw_plot["pre_late"][eq_idx["pre_late"]]
            eq_post_early_raw = group_raw_plot["post_early"][eq_idx["post_early"]]
            eq_post_late_raw = group_raw_plot["post_late"][eq_idx["post_late"]]

            eq_pre_all_raw = _stack_raw(eq_pre_early_raw, eq_pre_late_raw)
            eq_post_all_raw = _stack_raw(eq_post_early_raw, eq_post_late_raw)

            raw_eq_pair = _compute_summary_for_pair(eq_pre_all_raw, eq_post_all_raw)
            row["centroid_shift_raw_equal_groups"] = raw_eq_pair["centroid_shift"]
            row["pre_rms_radius_raw_equal_groups"] = raw_eq_pair["pre_rms_radius"]
            row["post_rms_radius_raw_equal_groups"] = raw_eq_pair["post_rms_radius"]
            row["post_over_pre_rms_radius_equal_groups"] = raw_eq_pair["post_over_pre_rms_radius"]
            row["pre_trace_cov_raw_equal_groups"] = raw_eq_pair["pre_trace_cov"]
            row["post_trace_cov_raw_equal_groups"] = raw_eq_pair["post_trace_cov"]
            row["post_over_pre_trace_cov_equal_groups"] = raw_eq_pair["post_over_pre_trace_cov"]

            eq_raw_for_embedding = np.vstack(
                [eq_pre_early_raw, eq_pre_late_raw, eq_post_early_raw, eq_post_late_raw]
            )
            eq_embeds = _fit_2d_embedding(eq_raw_for_embedding, cfg)

            e1 = len(eq_pre_early_raw)
            e2 = len(eq_pre_late_raw)
            e3 = len(eq_post_early_raw)
            e4 = len(eq_post_late_raw)

            eq_xy_pre_early = eq_embeds[:e1]
            eq_xy_pre_late = eq_embeds[e1:e1 + e2]
            eq_xy_post_early = eq_embeds[e1 + e2:e1 + e2 + e3]
            eq_xy_post_late = eq_embeds[e1 + e2 + e3:e1 + e2 + e3 + e4]

            eq_xy_pre_all = _stack_nonempty_xy(eq_xy_pre_early, eq_xy_pre_late)
            eq_xy_post_all = _stack_nonempty_xy(eq_xy_post_early, eq_xy_post_late)

            row["bc_pre_early_vs_late_equal_groups"] = _bhattacharyya_coefficient_2d(
                eq_xy_pre_early, eq_xy_pre_late, bins=cfg.bc_bins
            )
            row["bc_post_early_vs_late_equal_groups"] = _bhattacharyya_coefficient_2d(
                eq_xy_post_early, eq_xy_post_late, bins=cfg.bc_bins
            )

            umap_eq_pair = _compute_summary_for_pair(eq_xy_pre_all, eq_xy_post_all)
            row["centroid_shift_umap_equal_groups"] = umap_eq_pair["centroid_shift"]
            row["pre_rms_radius_umap_equal_groups"] = umap_eq_pair["pre_rms_radius"]
            row["post_rms_radius_umap_equal_groups"] = umap_eq_pair["post_rms_radius"]
            row["post_over_pre_rms_radius_umap_equal_groups"] = umap_eq_pair["post_over_pre_rms_radius"]

            eq_fig_path = clusters_out_dir / f"{cfg.out_prefix}_cluster{cluster_id}_pre_post_early_late_umap_equal_groups.png"
            _plot_cluster_figure(
                save_path=eq_fig_path,
                animal_id=animal_id,
                cluster_id=cluster_id,
                treatment_date=treatment_date,
                pre_early_xy=eq_xy_pre_early,
                pre_late_xy=eq_xy_pre_late,
                post_early_xy=eq_xy_post_early,
                post_late_xy=eq_xy_post_late,
                pre_early_dates=_extract_group_dates(points_df, group_point_indices["pre_early"]),
                pre_late_dates=_extract_group_dates(points_df, group_point_indices["pre_late"]),
                post_early_dates=_extract_group_dates(points_df, group_point_indices["post_early"]),
                post_late_dates=_extract_group_dates(points_df, group_point_indices["post_late"]),
                bc_pre_early_vs_late=row["bc_pre_early_vs_late_equal_groups"],
                bc_post_early_vs_late=row["bc_post_early_vs_late_equal_groups"],
                bc_pre_vs_post=_bhattacharyya_coefficient_2d(eq_xy_pre_all, eq_xy_post_all, bins=cfg.bc_bins),
                centroid_shift_value=row["centroid_shift_umap_equal_groups"],
                post_over_pre_rms=row["post_over_pre_rms_radius_umap_equal_groups"],
                post_over_pre_trace=row["post_over_pre_trace_cov_equal_groups"],
                cfg=cfg,
                label_suffix="equal-sized four-group plot",
            )

    return row


def _stack_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 and len(b) == 0:
        if a.ndim == 2:
            return np.empty((0, a.shape[1]), dtype=float)
        if b.ndim == 2:
            return np.empty((0, b.shape[1]), dtype=float)
        return np.empty((0, 0), dtype=float)
    if len(a) == 0:
        return b.copy()
    if len(b) == 0:
        return a.copy()
    return np.vstack([a, b])


# =============================================================================
# Main public entry point
# =============================================================================

def run_one_bird(cfg: UMAPEarlyLateConfig) -> str:
    cfg.npz_path = Path(cfg.npz_path)
    cfg.metadata_xlsx = Path(cfg.metadata_xlsx)
    cfg.out_dir = _coerce_path(cfg.out_dir)
    if cfg.out_dir is None:
        cfg.out_dir = cfg.npz_path.parent / "umap_cluster_variance_bc"
    cfg.out_prefix = cfg.out_prefix or _infer_animal_id(cfg.npz_path)

    _safe_mkdir(cfg.out_dir)
    clusters_out_dir = cfg.out_dir / "clusters"
    _safe_mkdir(clusters_out_dir)

    raw_all, _clusters, points_df, treatment_date, animal_id = _load_point_table(cfg)

    if len(points_df) == 0:
        raise ValueError("No usable points remain after filtering.")

    cluster_values = sorted(points_df["cluster"].dropna().unique(), key=lambda x: (str(type(x)), x))

    rows: List[Dict[str, Any]] = []
    for cluster_id in cluster_values:
        print(f"[one-bird] {animal_id}: cluster {cluster_id}")
        row = _analyze_one_cluster(
            cluster_id=cluster_id,
            raw_all=raw_all,
            points_df=points_df,
            treatment_date=treatment_date,
            animal_id=animal_id,
            cfg=cfg,
            clusters_out_dir=clusters_out_dir,
        )
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["animal_id", "cluster"]).reset_index(drop=True)
    summary_csv = cfg.out_dir / f"{cfg.out_prefix}_cluster_variance_bc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[one-bird] saved summary: {summary_csv}")
    return str(summary_csv)


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umap_pre_post_early_late_cluster_variance_bc.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--npz-path", required=True, type=str)
    p.add_argument("--metadata-xlsx", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--out-prefix", default=None, type=str)

    p.add_argument("--array-key", default="predictions", type=str)
    p.add_argument("--cluster-key", default="hdbscan_labels", type=str)
    p.add_argument("--file-key", default="file_indices", type=str)
    p.add_argument("--vocalization-key", default="vocalization", type=str)
    p.add_argument("--file-map-key", default="file_map", type=str)

    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--only-cluster-id", default=None, type=int)
    p.add_argument("--no-vocalization-only", action="store_true")

    p.add_argument("--min-points-per-period", default=200, type=int)
    p.add_argument("--max-points-per-period", default=None, type=int)
    p.add_argument("--plot-max-points-per-period", default=None, type=int)

    p.add_argument("--exclude-treatment-day-from-post", action="store_true")
    p.add_argument("--early-late-split-method", default="file_median", choices=["file_median", "file_half"])

    p.add_argument("--seed", default=0, type=int, help="Use -1 for no seed / faster UMAP")
    p.add_argument("--n-neighbors", default=30, type=int)
    p.add_argument("--min-dist", default=0.1, type=float)
    p.add_argument("--metric", default="euclidean", type=str)

    p.add_argument("--dpi", default=200, type=int)
    p.add_argument("--bc-bins", default=100, type=int)
    p.add_argument("--overlap-density-bins", default=180, type=int)
    p.add_argument("--overlap-density-gamma", default=0.55, type=float)

    p.add_argument("--metadata-sheet", default=None, type=str)

    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = UMAPEarlyLateConfig(
        npz_path=Path(args.npz_path),
        metadata_xlsx=Path(args.metadata_xlsx),
        array_key=args.array_key,
        cluster_key=args.cluster_key,
        file_key=args.file_key,
        vocalization_key=args.vocalization_key,
        file_map_key=args.file_map_key,
        include_noise=bool(args.include_noise),
        only_cluster_id=args.only_cluster_id,
        vocalization_only=not bool(args.no_vocalization_only),
        min_points_per_period=int(args.min_points_per_period),
        max_points_per_period=(int(args.max_points_per_period) if args.max_points_per_period is not None else None),
        max_points_per_period_for_plot=(
            int(args.plot_max_points_per_period) if args.plot_max_points_per_period is not None else None
        ),
        exclude_treatment_day_from_post=bool(args.exclude_treatment_day_from_post),
        early_late_split_method=str(args.early_late_split_method),
        random_seed=int(args.seed),
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),
        out_dir=Path(args.out_dir),
        out_prefix=args.out_prefix,
        dpi=int(args.dpi),
        bc_bins=int(args.bc_bins),
        overlap_density_bins=int(args.overlap_density_bins),
        overlap_density_gamma=float(args.overlap_density_gamma),
        metadata_sheet=args.metadata_sheet,
    )

    run_one_bird(cfg)


if __name__ == "__main__":
    main()
