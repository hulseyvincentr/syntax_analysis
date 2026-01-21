# -*- coding: utf-8 -*-
"""
outlier_graphs.py

Utilities for visualizing high-variance syllables and pre/post variance comparisons
from a compiled phrase-duration stats CSV (e.g., usage_balanced_phrase_duration_stats.csv).

Main additions (Jan 2026):
  • plot_pre_post_variance_scatter(): one scatter across ALL birds
        x = pre-lesion variance, y = post-lesion variance

The CSV is expected to have (at minimum) columns like:
  - "Animal ID" (animal identifier)
  - "Group"     (e.g., "Early Pre", "Late Pre", "Post")
  - "Syllable"
  - "N_phrases"
  - "Variance_ms2" (variance for that row's group)

This script is designed to be drop-in friendly for Spyder/Jupyter and also runnable
from the command line.

Author: ChatGPT
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # optional nicer colorbar
except Exception:  # pragma: no cover
    make_axes_locatable = None  # type: ignore

import json

PathLike = Union[str, Path]
AggFunc = Literal["mean", "median", "max", "min"]
RankOn = Literal["pre", "post", "max"]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize(s: str) -> str:
    # filesystem-safe-ish
    return (
        str(s)
        .strip()
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
    )


def _pick_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _canonical_hit_type(cat_raw: Any) -> str:
    """
    Match (as best as possible) the canonical keys used in your plotting code.

    Notes
    -----
    If your Excel already has a "Hit type" / "Lesion hit type" column,
    we will use it directly (after canonicalization).
    Otherwise we infer a coarse hit-type from columns like:
      - Treatment type
      - Medial Area X hit?
      - Lateral Area X hit?
      - Area X visible in histology?
    """
    if cat_raw is None:
        return "unknown"
    s = str(cat_raw).strip()
    if s == "":
        return "unknown"
    low = s.lower()

    # Canonical shams
    if "sham" in low or ("saline" in low and "lesion" not in low):
        return "sham saline injection"

    # Canonical miss
    if "miss" in low:
        return "miss"

    # Canonical not-visible large lesions (keep your historic spelling as-is)
    if "large" in low and "lesion" in low and "not visible" in low:
        return "large lesion Area X not visible"

    return s  # keep original label if not recognized


def _parse_yes(raw: Any) -> bool:
    """Interpret common Y/N style cells."""
    if raw is None:
        return False
    s = str(raw).strip().lower()
    return s in {"y", "yes", "true", "1", "t"}


def _is_visible(raw: Any) -> bool:
    """
    Interpret visibility column (e.g., 'Area X visible in histology?').

    Accepts: Y/Yes/Visible, or strings containing 'not visible'.
    """
    if raw is None:
        return False
    s = str(raw).strip().lower()
    if "not visible" in s:
        return False
    if s.startswith("n"):
        return False
    if s.startswith("y"):
        return True
    if " visible" in s or s == "visible":
        return True
    return False


def load_hit_type_map_from_metadata_excel(
    metadata_excel: PathLike,
    *,
    sheet_name: Union[int, str] = 0,
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: Optional[str] = None,
    # fallback inference columns (if no explicit hit-type column is present)
    treatment_type_col: str = "Treatment type",
    visible_col: str = "Area X visible in histology?",
    medial_hit_col: str = "Medial Area X hit?",
    lateral_hit_col: str = "Lateral Area X hit?",
) -> Dict[str, str]:
    """
    Build {animal_id -> hit_type} from your Area X metadata Excel sheet.

    Strategy
    --------
    1) If a hit-type/category column exists, use it (after canonicalization).
    2) Otherwise, infer a COARSE hit-type per animal:
         - sham saline injection (if treatment type indicates sham/saline)
         - miss (if neither medial nor lateral hit anywhere for that animal)
         - large lesion Area X not visible (if hit but not visible)
         - lesion hit (fallback)
    """
    meta_df = pd.read_excel(metadata_excel, sheet_name=sheet_name)

    if meta_animal_col not in meta_df.columns:
        raise ValueError(
            f"metadata excel missing meta_animal_col='{meta_animal_col}'. "
            f"Columns: {list(meta_df.columns)}"
        )

    # Try to find a hit-type column if not provided
    if meta_hit_type_col is None or meta_hit_type_col not in meta_df.columns:
        meta_hit_type_col = _pick_existing_col(
            meta_df,
            [
                "Lesion hit type",
                "Hit type",
                "Hit Type",
                "hit_type",
                "Hit category",
                "Lesion category",
                "Category",
                "category",
            ],
        )

    hit_map: Dict[str, str] = {}

    # Group rows by animal (Excel often has 2+ rows per animal for L/R injections)
    for animal_id, g in meta_df.groupby(meta_animal_col, dropna=True):
        aid = str(animal_id).strip()
        if aid == "":
            continue

        # 1) Explicit hit-type column
        if meta_hit_type_col is not None and meta_hit_type_col in g.columns:
            raw = g[meta_hit_type_col].dropna()
            if not raw.empty:
                hit_map[aid] = _canonical_hit_type(raw.iloc[0])
                continue

        # 2) Infer coarse hit-type
        # sham?
        tcol = treatment_type_col if treatment_type_col in g.columns else None
        if tcol is not None:
            tt_vals = g[tcol].dropna().astype(str).tolist()
            tt_low = " ".join(tt_vals).lower()
            if ("sham" in tt_low) or ("saline" in tt_low and "lesion" not in tt_low):
                hit_map[aid] = "sham saline injection"
                continue

        medial_any = g[medial_hit_col].apply(_parse_yes).any() if medial_hit_col in g.columns else False
        lateral_any = g[lateral_hit_col].apply(_parse_yes).any() if lateral_hit_col in g.columns else False
        hit_score = int(bool(medial_any)) + int(bool(lateral_any))
        hit_any = hit_score > 0

        vis_any = g[visible_col].apply(_is_visible).any() if visible_col in g.columns else False

        if not hit_any:
            hit_map[aid] = "miss"
        else:
            if not vis_any:
                hit_map[aid] = "large lesion Area X not visible"
            else:
                # Provide 2 purple shades downstream by encoding hit_score in label
                if hit_score >= 2:
                    hit_map[aid] = "Area X visible (medial+lateral hit)"
                else:
                    hit_map[aid] = "Area X visible (single hit)"

    return hit_map


# ──────────────────────────────────────────────────────────────────────────────
# Metadata helpers: visibility + lesion % (optionally from histology volumes JSONs)
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _norm_key(k: str) -> str:
    return (
        str(k)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _walk_json(obj: Any):
    """Yield (key, value) pairs from nested dict/list JSON-like structures."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk_json(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_json(v)


def _extract_left_right_lesion_pct_from_json_obj(
    obj: Any,
    *,
    left_key: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_key: str = "R_Percent_of_Area_X_Lesioned_pct",
) -> tuple[float, float]:
    """Best-effort extraction of left/right % Area X lesioned from a JSON object."""
    lk = _norm_key(left_key)
    rk = _norm_key(right_key)

    left_vals: list[float] = []
    right_vals: list[float] = []

    # 1) direct keys (including nested)
    for k, v in _walk_json(obj):
        nk = _norm_key(k)

        # Exact matches
        if nk == lk:
            left_vals.append(_safe_float(v))
            continue
        if nk == rk:
            right_vals.append(_safe_float(v))
            continue

        # Fuzzy matches
        if "percent" in nk and "area_x" in nk and "lesion" in nk:
            fv = _safe_float(v)
            if not np.isfinite(fv):
                continue
            if nk.startswith("l_") or "left" in nk:
                left_vals.append(fv)
            elif nk.startswith("r_") or "right" in nk:
                right_vals.append(fv)

    def _pick(vals: list[float]) -> float:
        vals2 = [v for v in vals if np.isfinite(v)]
        if not vals2:
            return float("nan")
        # Use mean if multiple candidates exist
        return float(np.mean(vals2))

    return _pick(left_vals), _pick(right_vals)


def _find_volume_json_for_animal(volumes_dir: Path, animal_id: str) -> Optional[Path]:
    """Try a few common patterns for your histology volume JSON files."""
    patterns = [
        f"{animal_id}*_final_volumes.json",
        f"{animal_id}*final*volumes*.json",
        f"{animal_id}*volumes*.json",
        f"{animal_id}*.json",
    ]
    for pat in patterns:
        hits = sorted(volumes_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def load_areax_metadata_summary(
    metadata_excel: PathLike,
    *,
    sheet_name: Union[int, str] = 0,
    volumes_dir: Optional[PathLike] = None,
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: Optional[str] = None,
    treatment_type_col: str = "Treatment type",
    visible_col: str = "Area X visible in histology?",
    medial_hit_col: str = "Medial Area X hit?",
    lateral_hit_col: str = "Lateral Area X hit?",
    left_lesion_pct_col: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col: str = "R_Percent_of_Area_X_Lesioned_pct",
    lesion_pct_mode: Literal["left", "right", "avg"] = "avg",
) -> Dict[str, Dict[str, Any]]:
    """
    Build a compact per-animal metadata summary:

        meta[animal_id] = {
            'hit_type': str,
            'is_visible': bool,
            'lesion_pct_left': float,
            'lesion_pct_right': float,
            'lesion_pct_avg': float,
            'lesion_pct': float   # according to lesion_pct_mode
        }

    Lesion % handling
    -----------------
    - If the Excel already contains left/right % columns, we use them.
    - Else, if `volumes_dir` is provided, we try to load a JSON file like:
        <animal_id>*_final_volumes.json
      and extract left/right % from its keys.
    """
    meta_df = pd.read_excel(metadata_excel, sheet_name=sheet_name)

    if meta_animal_col not in meta_df.columns:
        raise ValueError(
            f"metadata excel missing meta_animal_col='{meta_animal_col}'. Columns: {list(meta_df.columns)}"
        )

    # Auto-detect hit-type column if not provided
    if meta_hit_type_col is None or meta_hit_type_col not in meta_df.columns:
        meta_hit_type_col = _pick_existing_col(
            meta_df,
            [
                "Lesion hit type",
                "Hit type",
                "Hit Type",
                "hit_type",
                "Hit category",
                "Lesion category",
                "Category",
                "category",
            ],
        )

    vdir = Path(volumes_dir) if volumes_dir is not None else None
    if vdir is not None and not vdir.exists():
        print(f"[WARN] volumes_dir does not exist: {vdir}")
        vdir = None

    out: Dict[str, Dict[str, Any]] = {}

    for animal_id, g in meta_df.groupby(meta_animal_col, dropna=True):
        aid = str(animal_id).strip()
        if not aid:
            continue

        # visibility (any row)
        is_vis = g[visible_col].apply(_is_visible).any() if visible_col in g.columns else False

        # hit type
        hit_type = "unknown"
        if meta_hit_type_col is not None and meta_hit_type_col in g.columns:
            raw = g[meta_hit_type_col].dropna()
            if not raw.empty:
                hit_type = _canonical_hit_type(raw.iloc[0])
        if hit_type == "unknown":
            # infer coarse hit-type (same as load_hit_type_map_from_metadata_excel)
            tcol = treatment_type_col if treatment_type_col in g.columns else None
            if tcol is not None:
                tt_vals = g[tcol].dropna().astype(str).tolist()
                tt_low = " ".join(tt_vals).lower()
                if ("sham" in tt_low) or ("saline" in tt_low and "lesion" not in tt_low):
                    hit_type = "sham saline injection"
            if hit_type == "unknown":
                medial_any = g[medial_hit_col].apply(_parse_yes).any() if medial_hit_col in g.columns else False
                lateral_any = g[lateral_hit_col].apply(_parse_yes).any() if lateral_hit_col in g.columns else False
                hit_score = int(bool(medial_any)) + int(bool(lateral_any))
                hit_any = hit_score > 0
                if not hit_any:
                    hit_type = "miss"
                else:
                    if not is_vis:
                        hit_type = "large lesion Area X not visible"
                    else:
                        hit_type = "Area X visible (medial+lateral hit)" if hit_score >= 2 else "Area X visible (single hit)"

        # lesion % from excel columns if present
        l_pct = float("nan")
        r_pct = float("nan")

        if left_lesion_pct_col in g.columns:
            vals = [_safe_float(v) for v in g[left_lesion_pct_col].tolist()]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                l_pct = float(np.mean(vals))

        if right_lesion_pct_col in g.columns:
            vals = [_safe_float(v) for v in g[right_lesion_pct_col].tolist()]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                r_pct = float(np.mean(vals))

        # if still missing, try volumes_dir JSON
        if vdir is not None and (not np.isfinite(l_pct) or not np.isfinite(r_pct)):
            jpath = _find_volume_json_for_animal(vdir, aid)
            if jpath is not None:
                try:
                    obj = json.loads(jpath.read_text())
                    jl, jr = _extract_left_right_lesion_pct_from_json_obj(
                        obj,
                        left_key=left_lesion_pct_col,
                        right_key=right_lesion_pct_col,
                    )
                    if np.isfinite(jl):
                        l_pct = jl
                    if np.isfinite(jr):
                        r_pct = jr
                except Exception as e:
                    print(f"[WARN] Failed reading lesion % from {jpath.name}: {e}")

        # avg and selected mode
        vals = [v for v in (l_pct, r_pct) if np.isfinite(v)]
        avg_pct = float(np.mean(vals)) if vals else float("nan")

        if lesion_pct_mode == "left":
            pct = l_pct
        elif lesion_pct_mode == "right":
            pct = r_pct
        else:
            pct = avg_pct

        out[aid] = {
            "hit_type": hit_type,
            "is_visible": bool(is_vis),
            "lesion_pct_left": l_pct,
            "lesion_pct_right": r_pct,
            "lesion_pct_avg": avg_pct,
            "lesion_pct": pct,
        }

    return out


def _default_areax_hit_type_color_map() -> Dict[str, Any]:
    """Default color scheme requested for Area X hit-types.

    - black  : large lesion Area X not visible
    - purple : Area X visible (2 shades: single hit vs medial+lateral hit)
    - RED    : sham saline injection  (CHANGED from yellow → red)
    - gray   : miss / unknown / other
    """
    purples = plt.get_cmap("Purples")
    return {
        "large lesion Area X not visible": (0.0, 0.0, 0.0, 1.0),
        "Area X visible (single hit)": purples(0.55),
        "Area X visible (medial+lateral hit)": purples(0.85),
        "sham saline injection": (1.0, 0.0, 0.0, 1.0),  # <-- RED
        "miss": (0.55, 0.55, 0.55, 1.0),
        "unknown": (0.55, 0.55, 0.55, 1.0),
    }


def _category_color_map(categories: list[str], *, cmap_name: str = "tab20") -> Dict[str, Any]:
    """
    Deterministic category -> color mapping using a Matplotlib colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    n = max(len(categories), 1)
    out: Dict[str, Any] = {}
    for i, cat in enumerate(categories):
        if cat == "unknown":
            out[cat] = (0.55, 0.55, 0.55, 1.0)
        else:
            out[cat] = cmap(i / max(n - 1, 1))
    return out


def _agg_series(x: pd.Series, agg: AggFunc) -> float:
    if agg == "mean":
        return float(np.nanmean(x.to_numpy(dtype=float)))
    if agg == "median":
        return float(np.nanmedian(x.to_numpy(dtype=float)))
    if agg == "max":
        return float(np.nanmax(x.to_numpy(dtype=float)))
    if agg == "min":
        return float(np.nanmin(x.to_numpy(dtype=float)))
    raise ValueError(f"Unknown agg='{agg}'")


# ──────────────────────────────────────────────────────────────────────────────
# (Legacy) per-animal bar plots for top-variance syllables
# ──────────────────────────────────────────────────────────────────────────────
def plot_top_variance_syllables(
    csv_path: PathLike,
    out_dir: PathLike,
    top_percentile: float = 90.0,
    variance_col: str = "Variance_ms2",
    group_col: str = "Group",
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    nphrases_col: str = "N_phrases",
    agg: AggFunc = "mean",
    min_n_phrases: int = 5,
    show: bool = False,
) -> Dict[str, Any]:
    """
    For each animal, aggregate variance per syllable and bar-plot syllables with
    variance in the top `top_percentile` within that animal.

    Notes
    -----
    This "collapses" across Group by default (agg across all rows per syllable).
    """
    csv_path = Path(csv_path)
    out_dir = _ensure_dir(Path(out_dir))

    df = pd.read_csv(csv_path)

    missing = [c for c in [animal_col, syllable_col, variance_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if nphrases_col in df.columns:
        df = df[df[nphrases_col].fillna(0) >= min_n_phrases].copy()

    # Aggregate per (animal, syllable)
    g = df.groupby([animal_col, syllable_col], dropna=False)[variance_col]
    agg_vals = g.apply(lambda s: _agg_series(s, agg)).reset_index(name="variance_agg")

    fig_paths: list[Path] = []
    selected_rows: list[pd.DataFrame] = []

    for animal_id, sub in agg_vals.groupby(animal_col, dropna=False):
        sub = sub.sort_values("variance_agg", ascending=False).reset_index(drop=True)
        if len(sub) == 0:
            continue

        thresh = float(np.nanpercentile(sub["variance_agg"].to_numpy(float), top_percentile))
        sel = sub[sub["variance_agg"] >= thresh].copy()
        if len(sel) == 0:
            continue

        selected_rows.append(sel.assign(**{animal_col: animal_id, "threshold": thresh}))

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(sel))))
        ax.barh(sel[syllable_col].astype(str), sel["variance_agg"].astype(float))
        ax.invert_yaxis()
        ax.set_xlabel(f"{variance_col} (aggregated: {agg})")
        ax.set_ylabel("Syllable")
        ax.set_title(f"{animal_id}: Top ≥ P{top_percentile:.0f} variance syllables (n={len(sel)})")
        fig.tight_layout()

        fpath = out_dir / f"{_sanitize(str(animal_id))}_topP{int(round(top_percentile))}_variance_bar.png"
        fig.savefig(fpath, dpi=200)
        fig_paths.append(fpath)
        if show:
            plt.show()
        plt.close(fig)

    selected_table = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    selected_path = out_dir / f"selected_topP{int(round(top_percentile))}_variance_syllables.csv"
    selected_table.to_csv(selected_path, index=False)

    return {
        "out_dir": out_dir,
        "figure_paths": fig_paths,
        "selected_table_path": selected_path,
        "selected_table": selected_table,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NEW: all-birds pre vs post variance scatter
# ──────────────────────────────────────────────────────────────────────────────

def plot_pre_post_variance_scatter(
    csv_path: PathLike,
    out_dir: PathLike,
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    variance_col: str = "Variance_ms2",
    group_col: str = "Group",
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    nphrases_col: str = "N_phrases",
    min_n_phrases: int = 5,
    # Optional: restrict to "high variance" syllables per animal (like your bar-plot step)
    top_percentile: Optional[float] = None,
    outlier_percentile: Optional[float] = None,
    rank_on: RankOn = "post",
    # Optional: metadata-based hit-type coloring + continuous lesion % coloring
    metadata_excel: Optional[PathLike] = None,
    meta_sheet_name: Union[int, str] = 0,
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: Optional[str] = None,
    histology_volumes_dir: Optional[PathLike] = None,
    lesion_pct_mode: Literal["left", "right", "avg"] = "avg",
    left_lesion_pct_col: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col: str = "R_Percent_of_Area_X_Lesioned_pct",
    visible_col: str = "Area X visible in histology?",
    color_by_hit_type: bool = False,
    hit_type_color_map: Optional[Dict[str, Any]] = None,
    make_continuous_lesion_pct_plot: bool = True,
    # Plot controls
    log_scale: bool = True,
    color_by_animal: bool = False,
    alpha: float = 0.7,
    marker_size: float = 28.0,
    show: bool = False,
) -> Dict[str, Any]:
    """
    Build ONE scatterplot across ALL birds:
        x-axis = pre-lesion variance (variance_col in pre_group rows)
        y-axis = post-lesion variance (variance_col in post_group rows)

    Each point is a (Animal ID, Syllable) pair.

    Percentile filtering
    --------------------
    If `top_percentile` (or `outlier_percentile`) is provided, keeps only syllables
    within each animal whose rank metric is ≥ that percentile.

    Example: top_percentile=90 keeps the top 10% highest-variance syllables per animal.

    Metadata coloring
    -----------------
    If `color_by_hit_type=True`, points are colored by an animal-level hit-type
    derived from the metadata Excel.

    If `make_continuous_lesion_pct_plot=True`, and lesion % is available, a SECOND
    plot is saved where:
      - Area X visible points are colored by % Area X lesioned (Purples colormap + colorbar)
      - Sham = RED  (CHANGED from yellow → red)
      - Large lesion Area X not visible = black

    Returns
    -------
    Dict with output paths + the merged pre/post table.
    """
    # Resolve percentile alias
    if outlier_percentile is not None and top_percentile is None:
        top_percentile = outlier_percentile

    csv_path = Path(csv_path)
    out_dir = _ensure_dir(Path(out_dir))

    df = pd.read_csv(csv_path)

    missing = [c for c in [animal_col, syllable_col, group_col, variance_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Keep only the two groups
    df2 = df[df[group_col].astype(str).isin([pre_group, post_group])].copy()
    if df2.empty:
        raise ValueError(
            f"No rows found for pre_group='{pre_group}' and post_group='{post_group}'. "
            f"Available groups: {sorted(df[group_col].astype(str).unique().tolist())}"
        )

    # Apply N_phrases filter if column exists
    if nphrases_col in df2.columns:
        df2 = df2[df2[nphrases_col].fillna(0) >= min_n_phrases].copy()

    # Pivot variance and N_phrases by group
    var_piv = df2.pivot_table(
        index=[animal_col, syllable_col],
        columns=group_col,
        values=variance_col,
        aggfunc="mean",
    )

    if nphrases_col in df2.columns:
        n_piv = df2.pivot_table(
            index=[animal_col, syllable_col],
            columns=group_col,
            values=nphrases_col,
            aggfunc="max",
        )
    else:
        n_piv = None

    if pre_group not in var_piv.columns or post_group not in var_piv.columns:
        raise ValueError(
            "Could not form both pre and post columns from Group pivot. "
            f"Have columns: {list(var_piv.columns)}"
        )

    merged = var_piv[[pre_group, post_group]].rename(
        columns={pre_group: "pre_variance", post_group: "post_variance"}
    ).reset_index()

    if n_piv is not None and pre_group in n_piv.columns and post_group in n_piv.columns:
        n_df = n_piv[[pre_group, post_group]].rename(
            columns={pre_group: "pre_n_phrases", post_group: "post_n_phrases"}
        ).reset_index()
        merged = merged.merge(n_df, on=[animal_col, syllable_col], how="left")

    # Drop NaNs (must have both)
    merged = merged.dropna(subset=["pre_variance", "post_variance"]).copy()

    # Metadata summary (optional)
    meta_summary: Optional[Dict[str, Dict[str, Any]]] = None
    if metadata_excel is not None:
        meta_summary = load_areax_metadata_summary(
            metadata_excel,
            sheet_name=meta_sheet_name,
            volumes_dir=histology_volumes_dir,
            meta_animal_col=meta_animal_col,
            meta_hit_type_col=meta_hit_type_col,
            visible_col=visible_col,
            left_lesion_pct_col=left_lesion_pct_col,
            right_lesion_pct_col=right_lesion_pct_col,
            lesion_pct_mode=lesion_pct_mode,
        )

        def _get_meta(a: Any, key: str, default: Any):
            d = meta_summary.get(str(a), {}) if meta_summary is not None else {}
            return d.get(key, default)

        merged["hit_type"] = merged[animal_col].astype(str).map(
            lambda a: _get_meta(a, "hit_type", "unknown")
        )
        merged["is_visible"] = merged[animal_col].astype(str).map(
            lambda a: bool(_get_meta(a, "is_visible", False))
        )
        merged["lesion_pct"] = merged[animal_col].astype(str).map(
            lambda a: _safe_float(_get_meta(a, "lesion_pct", float("nan")))
        )
        merged["lesion_pct_avg"] = merged[animal_col].astype(str).map(
            lambda a: _safe_float(_get_meta(a, "lesion_pct_avg", float("nan")))
        )
    else:
        merged["hit_type"] = "unknown"
        merged["is_visible"] = False
        merged["lesion_pct"] = float("nan")
        merged["lesion_pct_avg"] = float("nan")

    # Optional: within-animal percentile filter (top variance syllables)
    if top_percentile is not None:
        if not (0 < top_percentile < 100):
            raise ValueError("top_percentile must be between 0 and 100 (exclusive).")

        if rank_on == "pre":
            metric = merged["pre_variance"].astype(float)
        elif rank_on == "post":
            metric = merged["post_variance"].astype(float)
        elif rank_on == "max":
            metric = np.maximum(
                merged["pre_variance"].astype(float),
                merged["post_variance"].astype(float),
            )
        else:
            raise ValueError(f"rank_on must be one of: pre, post, max. Got: {rank_on}")

        merged = merged.assign(_rank_metric=metric)

        thresh = (
            merged.groupby(animal_col)["_rank_metric"]
            .quantile(top_percentile / 100.0)
            .rename("_threshold")
        )
        merged = merged.join(thresh, on=animal_col)
        merged = merged[merged["_rank_metric"] >= merged["_threshold"]].copy()
        merged.drop(columns=["_rank_metric", "_threshold"], inplace=True, errors="ignore")

    # For log axes, remove non-positive
    if log_scale:
        merged = merged[(merged["pre_variance"] > 0) & (merged["post_variance"] > 0)].copy()

    # Save merged table
    tag = f"pre_{_sanitize(pre_group)}__post_{_sanitize(post_group)}"
    if top_percentile is None:
        table_path = out_dir / f"pre_post_variance_table__{tag}.csv"
    else:
        table_path = out_dir / f"pre_post_variance_table__{tag}__topP{int(round(top_percentile))}_{rank_on}.csv"
    merged.to_csv(table_path, index=False)

    # Title prefix: user-requested wording
    if top_percentile is None:
        title_prefix = "Pre vs Post variance"
    else:
        title_prefix = f"Pre vs Post top {int(round(top_percentile))}% variance"

    # Build a single hit-type colormap to use everywhere (Plot 1 + Plot 2 + legend)
    hit_cmap: Dict[str, Any] = (
        dict(hit_type_color_map) if hit_type_color_map is not None else _default_areax_hit_type_color_map()
    )
    sham_color = hit_cmap.get("sham saline injection", (1.0, 0.0, 0.0, 1.0))  # RED fallback

    # ------------------------------------------------------------------
    # Plot 1: discrete hit-type (or other coloring modes)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7.5, 7.0))

    if color_by_hit_type:
        if metadata_excel is None:
            raise ValueError("color_by_hit_type=True requires metadata_excel.")
        cats = sorted(merged["hit_type"].astype(str).fillna("unknown").unique().tolist())

        missing_cats = [c for c in cats if c not in hit_cmap]
        if missing_cats:
            # Add deterministic colors for any unexpected categories
            hit_cmap.update(_category_color_map(missing_cats))

        for cat in cats:
            sub = merged[merged["hit_type"].astype(str) == cat]
            if sub.empty:
                continue
            ax1.scatter(
                sub["pre_variance"].astype(float).to_numpy(),
                sub["post_variance"].astype(float).to_numpy(),
                alpha=alpha,
                s=marker_size,
                label=str(cat),
                color=hit_cmap.get(cat, None),
                edgecolors="none",
            )
        ax1.legend(title="Hit type", loc="best", frameon=False)

    elif color_by_animal:
        animals = merged[animal_col].astype(str)
        codes, uniques = pd.factorize(animals, sort=True)
        ax1.scatter(
            merged["pre_variance"].astype(float).to_numpy(),
            merged["post_variance"].astype(float).to_numpy(),
            c=codes,
            alpha=alpha,
            s=marker_size,
            edgecolors="none",
        )
        if len(uniques) <= 12:
            handles = []
            for name in uniques:
                handles.append(
                    plt.Line2D([], [], linestyle="none", marker="o", markersize=6, label=str(name))
                )
            ax1.legend(handles=handles, title="Animal", loc="best", frameon=False)

    else:
        ax1.scatter(
            merged["pre_variance"].astype(float).to_numpy(),
            merged["post_variance"].astype(float).to_numpy(),
            alpha=alpha,
            s=marker_size,
            edgecolors="none",
        )

    # y=x reference line (red dashed, like your example)
    x1 = merged["pre_variance"].astype(float).to_numpy()
    y1 = merged["post_variance"].astype(float).to_numpy()
    if x1.size and y1.size:
        if log_scale:
            pos = np.concatenate([x1[x1 > 0], y1[y1 > 0]])
            lo = float(np.nanmin(pos)) / 1.3 if pos.size else 1e-3
            hi = float(np.nanmax(pos)) * 1.3 if pos.size else 1.0
        else:
            lo = float(np.nanmin(np.concatenate([x1, y1])))
            hi = float(np.nanmax(np.concatenate([x1, y1])))
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            lo = max(0.0, lo - pad)
            hi = hi + pad
        ax1.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="red")

    ax1.set_xlabel(f"{pre_group} variance (ms$^2$)")
    ax1.set_ylabel(f"{post_group} variance (ms$^2$)")
    ax1.set_title(f"{title_prefix}\n({pre_group} on x, {post_group} on y)")
    if log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.grid(False)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    fig1.tight_layout()

    if top_percentile is None:
        fig1_path = out_dir / f"pre_post_variance_scatter__{tag}__discrete.png"
    else:
        fig1_path = out_dir / f"pre_post_variance_scatter__{tag}__topP{int(round(top_percentile))}_{rank_on}__discrete.png"
    fig1.savefig(fig1_path, dpi=250, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Plot 2: continuous Purples by lesion % (Area X visible only), plus discrete sham/not-visible
    # ------------------------------------------------------------------
    fig2_path: Optional[Path] = None
    if (
        make_continuous_lesion_pct_plot
        and metadata_excel is not None
        and color_by_hit_type
        and ("lesion_pct" in merged.columns)
    ):
        # Only proceed if we actually have some finite lesion % values among visible points
        vis_mask = merged["is_visible"].astype(bool) & np.isfinite(merged["lesion_pct"].astype(float))
        if vis_mask.any():
            fig2, ax2 = plt.subplots(figsize=(7.5, 7.0))

            # Visible -> continuous
            x_vis = merged.loc[vis_mask, "pre_variance"].astype(float).to_numpy()
            y_vis = merged.loc[vis_mask, "post_variance"].astype(float).to_numpy()
            c_vis = merged.loc[vis_mask, "lesion_pct"].astype(float).to_numpy()

            sc = ax2.scatter(
                x_vis,
                y_vis,
                c=c_vis,
                cmap="Purples",
                s=marker_size,
                alpha=0.85,
                edgecolors="none",
            )

            # Discrete categories
            sham_mask = merged["hit_type"].astype(str).str.lower().str.contains("sham")
            notvis_mask = merged["hit_type"].astype(str) == "large lesion Area X not visible"
            other_mask = ~(vis_mask | sham_mask | notvis_mask)

            if sham_mask.any():
                ax2.scatter(
                    merged.loc[sham_mask, "pre_variance"].astype(float).to_numpy(),
                    merged.loc[sham_mask, "post_variance"].astype(float).to_numpy(),
                    s=marker_size,
                    alpha=0.9,
                    color=sham_color,  # <-- RED
                    edgecolors="none",
                )

            if notvis_mask.any():
                ax2.scatter(
                    merged.loc[notvis_mask, "pre_variance"].astype(float).to_numpy(),
                    merged.loc[notvis_mask, "post_variance"].astype(float).to_numpy(),
                    s=marker_size,
                    alpha=0.9,
                    color="black",
                    edgecolors="black",
                    linewidths=0.3,
                )

            if other_mask.any():
                ax2.scatter(
                    merged.loc[other_mask, "pre_variance"].astype(float).to_numpy(),
                    merged.loc[other_mask, "post_variance"].astype(float).to_numpy(),
                    s=marker_size,
                    alpha=0.6,
                    color="lightgray",
                    edgecolors="none",
                )

            # y=x line
            x2 = merged["pre_variance"].astype(float).to_numpy()
            y2 = merged["post_variance"].astype(float).to_numpy()
            if x2.size and y2.size:
                if log_scale:
                    pos = np.concatenate([x2[x2 > 0], y2[y2 > 0]])
                    lo = float(np.nanmin(pos)) / 1.3 if pos.size else 1e-3
                    hi = float(np.nanmax(pos)) * 1.3 if pos.size else 1.0
                else:
                    lo = float(np.nanmin(np.concatenate([x2, y2])))
                    hi = float(np.nanmax(np.concatenate([x2, y2])))
                    pad = 0.05 * (hi - lo) if hi > lo else 0.1
                    lo = max(0.0, lo - pad)
                    hi = hi + pad
                ax2.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="red")

            ax2.set_xlabel(f"{pre_group} variance (ms$^2$)")
            ax2.set_ylabel(f"{post_group} variance (ms$^2$)")
            ax2.set_title(
                f"{title_prefix}\n(Area X visible: Purples = % lesioned; sham = red; not visible = black)"
            )
            if log_scale:
                ax2.set_xscale("log")
                ax2.set_yscale("log")

            ax2.grid(False)
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)

            # Colorbar + legend (arranged like your example)
            if make_axes_locatable is not None:
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="4%", pad=0.15)
                cbar = fig2.colorbar(sc, cax=cax)
                cbar.set_label("% of Area X lesioned", fontsize=11)
                legend_anchor_x = 1.28
            else:
                cbar = fig2.colorbar(sc, ax=ax2, pad=0.02)
                cbar.set_label("% of Area X lesioned", fontsize=11)
                legend_anchor_x = 1.15

            handles: list[Any] = []
            handles.append(
                mlines.Line2D(
                    [], [],
                    marker="o",
                    linestyle="none",
                    markersize=9,
                    markerfacecolor="none",
                    markeredgecolor="blue",
                    markeredgewidth=1.2,
                    label="Area X visible (see colorbar)",
                )
            )
            handles.append(
                mlines.Line2D(
                    [], [],
                    marker="o",
                    linestyle="none",
                    markersize=9,
                    markerfacecolor=sham_color,  # <-- RED
                    markeredgecolor="none",
                    label="sham saline injection",
                )
            )
            handles.append(
                mlines.Line2D(
                    [], [],
                    marker="o",
                    linestyle="none",
                    markersize=9,
                    markerfacecolor="black",
                    markeredgecolor="black",
                    label="large lesion Area X not visible",
                )
            )
            handles.append(
                mlines.Line2D([], [], color="red", linestyle="--", label="y=x")
            )

            fig2.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(legend_anchor_x, 0.5),
                borderaxespad=0.0,
                frameon=True,
                facecolor="white",
                framealpha=0.85,
                fontsize=10,
            )

            fig2.tight_layout()
            if top_percentile is None:
                fig2_path = out_dir / f"pre_post_variance_scatter__{tag}__continuous_lesion_pct.png"
            else:
                fig2_path = out_dir / f"pre_post_variance_scatter__{tag}__topP{int(round(top_percentile))}_{rank_on}__continuous_lesion_pct.png"
            fig2.savefig(fig2_path, dpi=250, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig2)
        else:
            print(
                "[INFO] Continuous lesion-% plot skipped: no finite lesion_pct values for any Area X visible animals.\n"
                "       (If you expected these, pass histology_volumes_dir or add lesion-% columns to the metadata sheet.)"
            )

    return {
        "out_dir": out_dir,
        "figure_path_discrete": fig1_path,
        "figure_path_continuous": fig2_path,
        "table_path": table_path,
        "table": merged,
        "n_points": int(len(merged)),
        "pre_group": pre_group,
        "post_group": post_group,
        "top_percentile": top_percentile,
        "rank_on": rank_on,
        "log_scale": log_scale,
        "metadata_excel": metadata_excel,
        "meta_summary_loaded": bool(meta_summary is not None),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(
        description="Outlier / variance plotting utilities for usage_balanced_phrase_duration_stats.csv"
    )
    p.add_argument("--csv", required=True, help="Path to the compiled phrase-duration stats CSV.")
    p.add_argument("--out_dir", required=True, help="Directory to save outputs.")
    p.add_argument("--mode", choices=["scatter", "bar"], default="scatter",
                   help="scatter = pre/post variance scatter; bar = per-animal top-variance bars")

    # Shared-ish options
    p.add_argument("--variance_col", default="Variance_ms2", help="Variance column to plot.")
    p.add_argument("--animal_col", default="Animal ID", help="Animal ID column name.")
    p.add_argument("--syllable_col", default="Syllable", help="Syllable column name.")
    p.add_argument("--group_col", default="Group", help="Group column name.")
    p.add_argument("--nphrases_col", default="N_phrases", help="N_phrases column name.")
    p.add_argument("--min_n_phrases", type=int, default=5, help="Minimum N_phrases to include.")

    # Bar
    p.add_argument("--top_percentile", type=float, default=90.0,
                   help="(bar mode) percentile threshold within each animal (e.g., 90 means keep ≥90th percentile).")
    p.add_argument("--agg", choices=["mean", "median", "max", "min"], default="mean",
                   help="(bar mode) how to aggregate variance across rows per (animal, syllable).")

    # Scatter
    p.add_argument("--pre_group", default="Late Pre", help="(scatter mode) pre-lesion group label.")
    p.add_argument("--post_group", default="Post", help="(scatter mode) post-lesion group label.")
    p.add_argument("--scatter_top_percentile", type=float, default=None,
                   help="(scatter mode) optional percentile filter per animal (e.g., 90 keeps top 10%).")
    p.add_argument("--rank_on", choices=["pre", "post", "max"], default="post",
                   help="(scatter mode) metric used for percentile ranking if scatter_top_percentile is set.")
    p.add_argument("--linear", action="store_true", help="(scatter mode) use linear axes instead of log-log.")
    p.add_argument("--color_by_animal", action="store_true", help="(scatter mode) color points by animal.")

    # Metadata-based hit-type coloring (scatter mode)
    p.add_argument("--metadata_excel", default=None,
                   help="(scatter mode) Path to metadata Excel for lesion hit-type coloring.")
    p.add_argument("--meta_sheet_name", default=0,
                   help="(scatter mode) Sheet name or index in metadata Excel (default 0).")
    p.add_argument("--meta_animal_col", default="Animal ID",
                   help="(scatter mode) Animal ID column in metadata Excel.")
    p.add_argument("--meta_hit_type_col", default=None,
                   help="(scatter mode) Optional hit-type/category column name in metadata Excel.")
    p.add_argument("--color_by_hit_type", action="store_true",
                   help="(scatter mode) Color points by lesion hit-type (requires metadata_excel).")

    p.add_argument("--show", action="store_true", help="Display the figure interactively.")

    return p


def main():
    p = _build_argparser()
    args = p.parse_args()

    if args.mode == "bar":
        plot_top_variance_syllables(
            csv_path=args.csv,
            out_dir=args.out_dir,
            top_percentile=args.top_percentile,
            variance_col=args.variance_col,
            group_col=args.group_col,
            animal_col=args.animal_col,
            syllable_col=args.syllable_col,
            nphrases_col=args.nphrases_col,
            agg=args.agg,
            min_n_phrases=args.min_n_phrases,
            show=args.show,
        )
    else:
        meta_sheet = args.meta_sheet_name
        if isinstance(meta_sheet, str) and meta_sheet.strip().isdigit():
            meta_sheet = int(meta_sheet.strip())

        plot_pre_post_variance_scatter(
            csv_path=args.csv,
            out_dir=args.out_dir,
            pre_group=args.pre_group,
            post_group=args.post_group,
            variance_col=args.variance_col,
            group_col=args.group_col,
            animal_col=args.animal_col,
            syllable_col=args.syllable_col,
            nphrases_col=args.nphrases_col,
            min_n_phrases=args.min_n_phrases,
            top_percentile=args.scatter_top_percentile,
            rank_on=args.rank_on,
            metadata_excel=args.metadata_excel,
            meta_sheet_name=meta_sheet,
            meta_animal_col=args.meta_animal_col,
            meta_hit_type_col=args.meta_hit_type_col,
            color_by_hit_type=args.color_by_hit_type,
            log_scale=(not args.linear),
            color_by_animal=args.color_by_animal,
            show=args.show,
        )


if __name__ == "__main__":
    main()


"""
from pathlib import Path
import sys, importlib

# Folder that contains outlier_graphs.py
code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
sys.path.insert(0, str(code_dir))

import outlier_graphs_discrete_colorcoding as og
importlib.reload(og)

# Inputs
compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv")
metadata_xlsx = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata.xlsx")

# OPTIONAL but recommended for the 2nd (continuous) plot:
# directory that contains files like: <animal_id>*_final_volumes.json
histology_volumes_dir = Path("/Volumes/my_own_SSD/histology_files/lesion_quantification_csvs_jsons")

# Where to save outputs
out_dir = compiled_csv.parent / "outlier_variance_scatter"

res = og.plot_pre_post_variance_scatter(
    csv_path=compiled_csv,
    out_dir=out_dir,
    pre_group="Late Pre",
    post_group="Post",

    # your outlier cutoff (e.g., 90 => keep top 10% per animal)
    top_percentile=50,
    rank_on="post",

    # metadata coloring
    metadata_excel=metadata_xlsx,
    meta_sheet_name=0,
    color_by_hit_type=True,

    # enables the 2nd plot: visible points shaded by avg % Area X lesioned
    histology_volumes_dir=histology_volumes_dir,
    lesion_pct_mode="avg",
    make_continuous_lesion_pct_plot=True,

    # plot style
    log_scale=True,
    min_n_phrases=5,
    show=True,
)

print("Discrete plot:", res["figure_path_discrete"])
print("Continuous lesion-% plot:", res["figure_path_continuous"])
print("Saved table:", res["table_path"])
print(res["table"].head())


"""