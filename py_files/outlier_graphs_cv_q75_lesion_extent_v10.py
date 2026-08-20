# Version: v10 — CV + within-bird Q75 ΔCV selection matching Figure 3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outlier_graphs_cv_q75_lesion_extent_v10.py

Updated manuscript-format Area X lesion-extent figure using the same Q75 ΔCV
analysis as Figure 3.

Panel A
-------
Late-pre versus post-lesion phrase-duration CV for syllables at or above each
bird's Q75 of the full qualifying syllable-level ΔCV distribution, where
ΔCV = post CV - late-pre CV. The Q75 threshold is recomputed from ALL rows for
each bird using NumPy's sample quantile with linear interpolation by default.
This replaces the legacy top-30%-by-SD selection.

Panel B
-------
Effective Area X lesion percentage versus the bird-level Q75 ΔCV statistic.
This uses the Q75 itself, not the median of the >=Q75 selected syllables.
Spearman correlation is therefore performed on the same animal-level endpoint
used for the primary Figure 3 analysis.

Lesion-percentage metadata parsing and plotting conventions are inherited from
outlier_graphs_sd_paper_format_v9.py. Sham birds are assigned 0% effective
lesion extent, complete lesions 100%, and partial lesions retain the measured
Area X lesion percentage.

Expected behavioral input is the Figure 3 syllable-level quantile-selection CSV
(or the balanced pair-metrics CSV) containing one row per qualifying
bird x syllable with pre_cv, post_cv, and delta_cv columns.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except Exception:  # pragma: no cover
    make_axes_locatable = None

# Reuse your existing metadata helpers from outlier_graphs.py.
# Put this file in the same py_files directory as outlier_graphs.py.
import outlier_graphs as og

PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

def _ensure_dir(p: PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def _safe_float(x: Any) -> float:
    """Convert numeric-ish values robustly, including strings like '25.6%'."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return float(x)
        except Exception:
            return float("nan")
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return float("nan")
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        return float(s)
    except Exception:
        return float("nan")


def _finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _epoch_label(group: str) -> str:
    """Make group labels more manuscript-friendly."""
    g = str(group).strip()
    mapping = {
        "Early Pre": "Early pre-lesion",
        "Late Pre": "Late pre-lesion",
        "Post": "Post-lesion",
        "Early Post": "Early post-lesion",
        "Late Post": "Late post-lesion",
    }
    return mapping.get(g, g)


def _pretty_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _line_limits(x: np.ndarray, y: np.ndarray, *, log_scale: bool) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if log_scale:
        pos = np.concatenate([x[x > 0], y[y > 0]])
        if pos.size == 0:
            return 1e-3, 1.0
        return float(np.nanmin(pos)) / 1.3, float(np.nanmax(pos)) * 1.3

    both = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if both.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(both))
    hi = float(np.nanmax(both))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    return max(0.0, lo - pad), hi + pad


def _hit_type_contains(series: pd.Series, patterns: Sequence[str]) -> pd.Series:
    out = pd.Series(False, index=series.index)
    text = series.astype(str)
    for pat in patterns:
        out = out | text.str.contains(pat, case=False, na=False, regex=False)
    return out


# -----------------------------------------------------------------------------
# Statistics: animal-level correlation, not syllable-level pseudo-replication
# -----------------------------------------------------------------------------



def _norm_key(k: Any) -> str:
    return str(k).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _walk_json(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk_json(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_json(v)


def _guess_side_from_path(path_like: Any) -> Optional[str]:
    """Guess whether a volume JSON is left/right-specific from its filename/path."""
    s = str(path_like).lower()
    # Prefer full words, but also handle common _l_ / _r_ patterns.
    if re.search(r"(^|[_\-. /])(left|lhs|l)([_\-. /]|$)", s):
        return "left"
    if re.search(r"(^|[_\-. /])(right|rhs|r)([_\-. /]|$)", s):
        return "right"
    return None


def _is_area_x_lesion_pct_key(key: Any) -> bool:
    """Flexible match for percentage-of-Area-X-lesioned JSON keys."""
    raw = str(key)
    nk = _norm_key(raw)
    has_pct = ("percent" in nk) or ("pct" in nk) or ("%" in raw)
    has_area_x = ("area_x" in nk) or ("areax" in nk) or ("area" in nk and "_x" in nk)
    has_lesion = ("lesion" in nk) or ("lesioned" in nk)
    return bool(has_pct and has_area_x and has_lesion)


def _extract_left_right_general_lesion_pct_from_json_obj(
    obj: Any,
    *,
    left_key: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_key: str = "R_Percent_of_Area_X_Lesioned_pct",
    side_hint: Optional[str] = None,
) -> tuple[float, float, float, list[str]]:
    """
    Extract left/right/general % Area X lesioned from a JSON object.

    This is intentionally broader than the older parser because some final-volume
    JSONs store side-specific percentages in side-specific files rather than
    side-specific keys.
    """
    lk = _norm_key(left_key)
    rk = _norm_key(right_key)

    left_vals: list[float] = []
    right_vals: list[float] = []
    general_vals: list[float] = []
    matched_keys: list[str] = []

    for k, v in _walk_json(obj):
        nk = _norm_key(k)
        val = _safe_float(v)
        if not np.isfinite(val):
            continue

        if nk == lk:
            left_vals.append(val)
            matched_keys.append(str(k))
            continue
        if nk == rk:
            right_vals.append(val)
            matched_keys.append(str(k))
            continue

        if not _is_area_x_lesion_pct_key(k):
            continue

        matched_keys.append(str(k))
        if nk.startswith("l_") or "left" in nk:
            left_vals.append(val)
        elif nk.startswith("r_") or "right" in nk:
            right_vals.append(val)
        elif side_hint == "left":
            left_vals.append(val)
        elif side_hint == "right":
            right_vals.append(val)
        else:
            general_vals.append(val)

    def _mean(vals: list[float]) -> float:
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return _mean(left_vals), _mean(right_vals), _mean(general_vals), sorted(set(matched_keys))


def _extract_left_right_lesion_pct_from_json_obj(
    obj: Any,
    *,
    left_key: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_key: str = "R_Percent_of_Area_X_Lesioned_pct",
) -> tuple[float, float]:
    """Backward-compatible wrapper around the broader v3 JSON parser."""
    l, r, general, _keys = _extract_left_right_general_lesion_pct_from_json_obj(
        obj,
        left_key=left_key,
        right_key=right_key,
        side_hint=None,
    )
    if not np.isfinite(l) and not np.isfinite(r) and np.isfinite(general):
        # If a JSON has a single general Area X lesion percentage, use it for both
        # so lesion_pct_mode="avg" returns the general value.
        l = general
        r = general
    return l, r


def _select_lesion_pct(l_pct: float, r_pct: float, mode: str) -> tuple[float, float]:
    vals = [v for v in (l_pct, r_pct) if np.isfinite(v)]
    avg = float(np.mean(vals)) if vals else float("nan")
    if mode == "left":
        return l_pct, avg
    if mode == "right":
        return r_pct, avg
    return avg, avg


def _auto_find_pct_cols(df: pd.DataFrame, requested_left: str, requested_right: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return likely left, right, or general lesion-% columns from a metadata sheet."""
    cols = list(df.columns)
    norm_to_col = {_norm_key(c): c for c in cols}

    left_col = norm_to_col.get(_norm_key(requested_left))
    right_col = norm_to_col.get(_norm_key(requested_right))
    general_col = None

    for c in cols:
        n = _norm_key(c)
        if "percent" not in n and "pct" not in n and "%" not in str(c):
            continue
        if "area_x" not in n and "areax" not in n:
            continue
        if "lesion" not in n and "lesioned" not in n:
            continue
        if left_col is None and (n.startswith("l_") or "left" in n):
            left_col = c
        elif right_col is None and (n.startswith("r_") or "right" in n):
            right_col = c
        elif general_col is None:
            general_col = c

    return left_col, right_col, general_col


def _fill_lesion_pct_from_metadata_excel(
    merged: pd.DataFrame,
    *,
    metadata_excel: PathLike,
    sheet_name: Union[int, str],
    animal_col: str,
    meta_animal_col: str,
    left_lesion_pct_col: str,
    right_lesion_pct_col: str,
    lesion_pct_mode: str,
) -> pd.DataFrame:
    """Fill missing lesion_pct values from columns in the metadata Excel if present."""
    try:
        meta = pd.read_excel(metadata_excel, sheet_name=sheet_name)
    except Exception as e:
        print(f"[WARN] Could not read lesion %% from metadata Excel: {e}")
        return merged

    if meta_animal_col not in meta.columns:
        print(f"[WARN] Metadata Excel does not have animal column {meta_animal_col!r}; cannot use Excel lesion %% fallback.")
        return merged

    left_col, right_col, general_col = _auto_find_pct_cols(meta, left_lesion_pct_col, right_lesion_pct_col)
    if left_col is None and right_col is None and general_col is None:
        print("[INFO] No lesion-% columns found in metadata Excel for fallback.")
        return merged

    rows: dict[str, dict[str, float]] = {}
    for aid, g in meta.groupby(meta_animal_col, dropna=True):
        aid = str(aid).strip()
        if not aid:
            continue
        l_pct = float("nan")
        r_pct = float("nan")
        if left_col is not None:
            vals = pd.to_numeric(g[left_col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                l_pct = float(np.mean(vals))
        if right_col is not None:
            vals = pd.to_numeric(g[right_col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                r_pct = float(np.mean(vals))
        if general_col is not None and not (np.isfinite(l_pct) or np.isfinite(r_pct)):
            vals = pd.to_numeric(g[general_col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                l_pct = r_pct = float(np.mean(vals))
        pct, avg = _select_lesion_pct(l_pct, r_pct, lesion_pct_mode)
        if np.isfinite(pct) or np.isfinite(avg):
            rows[aid] = {"lesion_pct": pct, "lesion_pct_avg": avg}

    if not rows:
        print("[INFO] Metadata Excel lesion-% fallback found columns but no finite values.")
        return merged

    out = merged.copy()
    before = int(np.isfinite(pd.to_numeric(out["lesion_pct"], errors="coerce")).sum())
    for idx, aid in out[animal_col].astype(str).items():
        if aid not in rows:
            continue
        current = _safe_float(out.at[idx, "lesion_pct"])
        if not np.isfinite(current):
            out.at[idx, "lesion_pct"] = rows[aid]["lesion_pct"]
        current_avg = _safe_float(out.at[idx, "lesion_pct_avg"])
        if not np.isfinite(current_avg):
            out.at[idx, "lesion_pct_avg"] = rows[aid]["lesion_pct_avg"]
    after = int(np.isfinite(pd.to_numeric(out["lesion_pct"], errors="coerce")).sum())
    print(f"[INFO] Excel lesion-% fallback: finite rows before={before}, after={after}; columns used: left={left_col}, right={right_col}, general={general_col}")
    return out


def _fill_lesion_pct_from_volume_jsons(
    merged: pd.DataFrame,
    *,
    histology_volumes_dir: PathLike,
    animal_col: str,
    left_lesion_pct_col: str,
    right_lesion_pct_col: str,
    lesion_pct_mode: str,
) -> pd.DataFrame:
    """Fill missing lesion_pct values by recursively searching JSON files for animal IDs."""
    root = Path(histology_volumes_dir)
    if not root.exists():
        print(f"[WARN] histology_volumes_dir does not exist: {root}")
        return merged

    json_paths = [p for p in root.rglob("*.json") if p.is_file() and not p.name.startswith("._")]
    if not json_paths:
        print(f"[WARN] No JSON files found under histology_volumes_dir: {root}")
        return merged

    animal_ids = sorted(set(merged[animal_col].astype(str)))
    pct_by_animal: dict[str, dict[str, float]] = {}
    files_matched_by_name = 0
    files_with_area_x_pct_keys = 0
    debug_rows: list[str] = []

    for aid in animal_ids:
        # Match animal ID in the full path or file stem. This handles nested folders.
        matching = [p for p in json_paths if aid.lower() in str(p).lower()]
        if matching:
            files_matched_by_name += len(matching)

        left_vals: list[float] = []
        right_vals: list[float] = []
        general_vals: list[float] = []
        matched_keys_for_aid: set[str] = set()

        for jpath in matching:
            try:
                obj = json.loads(jpath.read_text())
            except Exception:
                continue

            side_hint = _guess_side_from_path(jpath)
            l_pct, r_pct, general_pct, matched_keys = _extract_left_right_general_lesion_pct_from_json_obj(
                obj,
                left_key=left_lesion_pct_col,
                right_key=right_lesion_pct_col,
                side_hint=side_hint,
            )

            if matched_keys:
                files_with_area_x_pct_keys += 1
                matched_keys_for_aid.update(matched_keys)

            if np.isfinite(l_pct):
                left_vals.append(l_pct)
            if np.isfinite(r_pct):
                right_vals.append(r_pct)
            if np.isfinite(general_pct):
                general_vals.append(general_pct)

        l = float(np.mean(left_vals)) if left_vals else float("nan")
        r = float(np.mean(right_vals)) if right_vals else float("nan")

        if not np.isfinite(l) and not np.isfinite(r) and general_vals:
            # Use general values only if no side-specific values were found.
            g = float(np.mean(general_vals))
            l = g
            r = g

        pct, avg = _select_lesion_pct(l, r, lesion_pct_mode)
        if np.isfinite(pct) or np.isfinite(avg):
            pct_by_animal[aid] = {"lesion_pct": pct, "lesion_pct_avg": avg}
            debug_rows.append(
                f"  {aid}: left={l:.4g}, right={r:.4g}, avg={avg:.4g}, files={len(matching)}, keys={sorted(matched_keys_for_aid)[:6]}"
            )

    out = merged.copy()
    before = int(np.isfinite(pd.to_numeric(out["lesion_pct"], errors="coerce")).sum())
    for idx, aid in out[animal_col].astype(str).items():
        if aid not in pct_by_animal:
            continue
        current = _safe_float(out.at[idx, "lesion_pct"])
        if not np.isfinite(current):
            out.at[idx, "lesion_pct"] = pct_by_animal[aid]["lesion_pct"]
        current_avg = _safe_float(out.at[idx, "lesion_pct_avg"])
        if not np.isfinite(current_avg):
            out.at[idx, "lesion_pct_avg"] = pct_by_animal[aid]["lesion_pct_avg"]

    after = int(np.isfinite(pd.to_numeric(out["lesion_pct"], errors="coerce")).sum())
    print(
        "[INFO] JSON lesion-% fallback: "
        f"finite rows before={before}, after={after}; "
        f"animals_with_pct={len(pct_by_animal)}; "
        f"json_files_seen={len(json_paths)}; "
        f"files_matched_by_animal_name={files_matched_by_name}; "
        f"files_with_area_x_pct_keys={files_with_area_x_pct_keys}"
    )
    if debug_rows:
        print("[INFO] Parsed lesion % by animal:")
        for row in debug_rows[:25]:
            print(row)
        if len(debug_rows) > 25:
            print(f"  ... {len(debug_rows) - 25} more")
    else:
        print("[INFO] No finite Area X lesion percentages parsed from matched JSON files.")
        # Print a few animal IDs and JSON names to make path/key mismatch obvious.
        print(f"[INFO] Animals in plotted table: {animal_ids[:20]}{' ...' if len(animal_ids) > 20 else ''}")
        print("[INFO] First JSON files seen:")
        for p in json_paths[:10]:
            print(f"  {p.name}")

    return out



def _attach_lesion_metadata(
    df: pd.DataFrame,
    *,
    animal_col: str,
    metadata_excel: Optional[PathLike],
    meta_sheet_name: Union[int, str],
    meta_animal_col: str,
    meta_hit_type_col: Optional[str],
    histology_volumes_dir: Optional[PathLike],
    lesion_pct_mode: str,
    left_lesion_pct_col: str,
    right_lesion_pct_col: str,
    visible_col: str,
) -> pd.DataFrame:
    """Attach hit type and measured Area X lesion percentage using the v9 logic."""
    out = df.copy()
    meta_summary: Optional[Dict[str, Dict[str, Any]]] = None

    if metadata_excel is not None:
        meta_summary = og.load_areax_metadata_summary(
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

        def _get_meta(a: Any, key: str, default: Any) -> Any:
            return meta_summary.get(str(a), {}).get(key, default) if meta_summary is not None else default

        out["hit_type"] = out[animal_col].astype(str).map(
            lambda a: str(_get_meta(a, "hit_type", "unknown"))
        )
        out["is_visible"] = out[animal_col].astype(str).map(
            lambda a: bool(_get_meta(a, "is_visible", False))
        )
        out["lesion_pct"] = out[animal_col].astype(str).map(
            lambda a: _safe_float(_get_meta(a, "lesion_pct", np.nan))
        )
        out["lesion_pct_avg"] = out[animal_col].astype(str).map(
            lambda a: _safe_float(_get_meta(a, "lesion_pct_avg", np.nan))
        )
        out["treatment_group"] = out[animal_col].astype(str).map(
            lambda a: str(_get_meta(a, "treatment_group", "other"))
        )
    else:
        out["hit_type"] = "unknown"
        out["is_visible"] = False
        out["lesion_pct"] = np.nan
        out["lesion_pct_avg"] = np.nan
        out["treatment_group"] = "other"

    # Preserve the robust v9 fallbacks for older metadata layouts.
    if metadata_excel is not None:
        out = _fill_lesion_pct_from_metadata_excel(
            out,
            metadata_excel=metadata_excel,
            sheet_name=meta_sheet_name,
            animal_col=animal_col,
            meta_animal_col=meta_animal_col,
            left_lesion_pct_col=left_lesion_pct_col,
            right_lesion_pct_col=right_lesion_pct_col,
            lesion_pct_mode=lesion_pct_mode,
        )
    if histology_volumes_dir is not None:
        out = _fill_lesion_pct_from_volume_jsons(
            out,
            histology_volumes_dir=histology_volumes_dir,
            animal_col=animal_col,
            left_lesion_pct_col=left_lesion_pct_col,
            right_lesion_pct_col=right_lesion_pct_col,
            lesion_pct_mode=lesion_pct_mode,
        )

    finite_rows = int(np.isfinite(pd.to_numeric(out["lesion_pct"], errors="coerce")).sum())
    finite_animals = (
        out.loc[np.isfinite(pd.to_numeric(out["lesion_pct"], errors="coerce")), animal_col]
        .astype(str)
        .nunique()
    )
    print(f"[INFO] Final finite lesion_pct: rows={finite_rows}, animals={finite_animals}")
    return out


def _build_q_selection(
    df: pd.DataFrame,
    *,
    animal_col: str,
    delta_cv_col: str,
    quantile: float,
    quantile_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the bird-specific quantile from ALL qualifying syllables, then mark
    syllables at or above that threshold.

    This intentionally mirrors Figure 3: the animal-level endpoint is the actual
    sample quantile, not a median calculated after selecting the upper quartile.
    """
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("--quantile must be between 0 and 1")

    d = df.copy()
    d[delta_cv_col] = _finite_numeric(d[delta_cv_col])
    d = d[np.isfinite(d[delta_cv_col])].copy()

    rows = []
    for bird, g in d.groupby(animal_col, sort=True):
        vals = g[delta_cv_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q_value = float(np.quantile(vals, quantile, method=quantile_method))
        rows.append({
            animal_col: bird,
            "n_qualifying_syllables": int(vals.size),
            "quantile": float(quantile),
            "quantile_label": f"Q{int(round(quantile * 100))}",
            "q_delta_cv": q_value,
            "quantile_method": quantile_method,
        })

    bird_q = pd.DataFrame(rows)
    if bird_q.empty:
        raise ValueError("No finite syllable-level ΔCV values were available.")

    d = d.merge(bird_q[[animal_col, "q_delta_cv"]], on=animal_col, how="inner")
    d["selected_at_or_above_q"] = d[delta_cv_col] >= d["q_delta_cv"]
    return d, bird_q


def _effective_lesion_columns(
    df: pd.DataFrame,
    *,
    complete_hit_type_patterns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    if "hit_type" not in out.columns:
        out["hit_type"] = "unknown"
    if "treatment_group" not in out.columns:
        out["treatment_group"] = "other"

    hit = out["hit_type"].astype(str)
    sham_mask = (hit == "sham saline injection") | (out["treatment_group"].astype(str) == "sham")
    complete_mask = _hit_type_contains(hit, complete_hit_type_patterns) & ~sham_mask

    out["effective_lesion_pct"] = _finite_numeric(out["lesion_pct"])
    out.loc[sham_mask, "effective_lesion_pct"] = 0.0
    out.loc[complete_mask, "effective_lesion_pct"] = 100.0
    out["lesion_group_for_plot"] = np.where(
        sham_mask,
        "sham (0%)",
        np.where(complete_mask, "complete (100%)", "partial / measured"),
    )
    return out


def save_panel_a_legend_png(
    out_path: PathLike,
    *,
    partial_edge_color: str = "#8C1E96",
    complete_ml_color: str = "#7A7A7A",
    sham_color: str = "#009E73",
    text_color: str = "0.35",
    dpi: int = 300,
) -> str:
    """Save a standalone transparent legend PNG for Panel A."""
    out_path = Path(out_path)
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor="none", markeredgecolor=partial_edge_color,
                      markeredgewidth=1.5, label="partial lesion (see colorbar)"),
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=complete_ml_color, markeredgecolor=complete_ml_color,
                      label="complete medial+lateral lesion"),
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=sham_color, markeredgecolor="none",
                      label="sham saline injection"),
        mlines.Line2D([], [], color=text_color, linestyle="--", label="y=x"),
    ]
    fig = plt.figure(figsize=(6.0, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=9,
              borderaxespad=0.0, handletextpad=0.8, labelspacing=0.7)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)
    return str(out_path)


def plot_cv_q_lesion_extent(
    selection_csv: PathLike,
    out_dir: PathLike,
    *,
    animal_col: str = "animal_id",
    syllable_col: str = "syllable",
    pre_cv_col: str = "pre_cv",
    post_cv_col: str = "post_cv",
    delta_cv_col: str = "delta_cv",
    quantile: float = 0.75,
    quantile_method: str = "linear",
    metadata_excel: Optional[PathLike] = None,
    meta_sheet_name: Union[int, str] = "metadata_with_hit_type",
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: Optional[str] = None,
    histology_volumes_dir: Optional[PathLike] = None,
    lesion_pct_mode: str = "avg",
    left_lesion_pct_col: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col: str = "R_Percent_of_Area_X_Lesioned_pct",
    visible_col: str = "Area X visible in histology?",
    complete_hit_type_patterns: Sequence[str] = ("large lesion Area X not visible", "complete"),
    partial_hit_type_patterns: Sequence[str] = ("Area X visible",),
    sham_color: str = "#009E73",
    complete_ml_color: str = "#7A7A7A",
    partial_cmap: str = "Purples",
    partial_color: str = "#6A51A3",
    marker_size: float = 30.0,
    alpha: float = 0.9,
    add_title: bool = False,
    exclude_sham_from_correlation: bool = False,
    dpi: int = 300,
    show: bool = False,
) -> Dict[str, Any]:
    out_dir = _ensure_dir(out_dir)
    selection_csv = Path(selection_csv)
    df = pd.read_csv(selection_csv)

    required = [animal_col, syllable_col, pre_cv_col, post_cv_col, delta_cv_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # One row per qualifying bird x syllable.
    d = df.drop_duplicates([animal_col, syllable_col]).copy()
    for col in (pre_cv_col, post_cv_col, delta_cv_col):
        d[col] = _finite_numeric(d[col])
    d = d[
        np.isfinite(d[pre_cv_col])
        & np.isfinite(d[post_cv_col])
        & np.isfinite(d[delta_cv_col])
    ].copy()

    # Recompute the within-bird Q cutoff from the full qualifying ΔCV distribution.
    d, bird_q = _build_q_selection(
        d,
        animal_col=animal_col,
        delta_cv_col=delta_cv_col,
        quantile=quantile,
        quantile_method=quantile_method,
    )
    q_label = f"Q{int(round(quantile * 100))}"

    # Optional audit against an existing Figure 3 selected_QXX column.
    existing_flag = f"selected_{q_label}"
    if existing_flag in df.columns:
        audit = df[[animal_col, syllable_col, existing_flag]].drop_duplicates([animal_col, syllable_col])
        audit[existing_flag] = audit[existing_flag].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        check = d.merge(audit, on=[animal_col, syllable_col], how="left")
        comparable = check[existing_flag].notna()
        if comparable.any():
            mismatches = int((check.loc[comparable, existing_flag] != check.loc[comparable, "selected_at_or_above_q"]).sum())
            print(f"[AUDIT] {existing_flag}: compared={int(comparable.sum())}, mismatches={mismatches}")

    d = _attach_lesion_metadata(
        d,
        animal_col=animal_col,
        metadata_excel=metadata_excel,
        meta_sheet_name=meta_sheet_name,
        meta_animal_col=meta_animal_col,
        meta_hit_type_col=meta_hit_type_col,
        histology_volumes_dir=histology_volumes_dir,
        lesion_pct_mode=lesion_pct_mode,
        left_lesion_pct_col=left_lesion_pct_col,
        right_lesion_pct_col=right_lesion_pct_col,
        visible_col=visible_col,
    )
    d = _effective_lesion_columns(d, complete_hit_type_patterns=complete_hit_type_patterns)

    # ------------------------------------------------------------------
    # Panel A: pre/post CV for >= within-bird Q cutoff syllables.
    # ------------------------------------------------------------------
    selected = d[d["selected_at_or_above_q"]].copy()
    selected_csv = out_dir / f"panelA_{q_label}_selected_syllables.csv"
    selected.to_csv(selected_csv, index=False)

    hit = selected["hit_type"].astype(str)
    finite_pct = np.isfinite(_finite_numeric(selected["lesion_pct"]))
    sham_mask = (hit == "sham saline injection") | (selected["treatment_group"].astype(str) == "sham")
    complete_mask = _hit_type_contains(hit, complete_hit_type_patterns) & ~sham_mask
    partial_mask = (
        (_hit_type_contains(hit, partial_hit_type_patterns) | selected["is_visible"].astype(bool))
        & finite_pct & ~sham_mask & ~complete_mask
    )
    other_mask = ~(partial_mask | sham_mask | complete_mask)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    sc = None
    if partial_mask.any():
        pvals = selected.loc[partial_mask, "lesion_pct"].astype(float).to_numpy()
        pvals = pvals[np.isfinite(pvals)]
        vmin = float(np.nanmin(pvals)) if pvals.size else None
        vmax = float(np.nanmax(pvals)) if pvals.size else None
        if vmin is not None and vmax is not None and vmax <= vmin:
            pad = max(1.0, abs(vmin) * 0.05)
            vmin -= pad
            vmax += pad
        sc = ax.scatter(
            selected.loc[partial_mask, pre_cv_col].astype(float),
            selected.loc[partial_mask, post_cv_col].astype(float),
            c=selected.loc[partial_mask, "lesion_pct"].astype(float),
            cmap=partial_cmap, vmin=vmin, vmax=vmax,
            s=marker_size, alpha=alpha, edgecolors="none", zorder=3,
        )

    if complete_mask.any():
        ax.scatter(
            selected.loc[complete_mask, pre_cv_col].astype(float),
            selected.loc[complete_mask, post_cv_col].astype(float),
            s=marker_size, alpha=0.95, color=complete_ml_color,
            edgecolors="black", linewidths=0.3, zorder=4,
        )
    if sham_mask.any():
        ax.scatter(
            selected.loc[sham_mask, pre_cv_col].astype(float),
            selected.loc[sham_mask, post_cv_col].astype(float),
            s=marker_size, alpha=0.95, color=sham_color,
            edgecolors="none", zorder=2,
        )
    if other_mask.any():
        ax.scatter(
            selected.loc[other_mask, pre_cv_col].astype(float),
            selected.loc[other_mask, post_cv_col].astype(float),
            s=marker_size, alpha=0.55, color="lightgray",
            edgecolors="none", zorder=1,
        )

    x = selected[pre_cv_col].to_numpy(dtype=float)
    y = selected[post_cv_col].to_numpy(dtype=float)
    lo, hi = _line_limits(x, y, log_scale=False)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="0.35", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Late pre-lesion phrase duration CV", fontsize=12)
    ax.set_ylabel("Post-lesion phrase duration CV", fontsize=12)
    if add_title:
        ax.set_title(f"Syllables at or above within-bird {q_label} ΔCV")
    _pretty_axes(ax)

    if sc is not None:
        if make_axes_locatable is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.12)
            cbar = fig.colorbar(sc, cax=cax)
        else:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Area X lesioned (%)", fontsize=11)

    fig.tight_layout()
    panel_a_path = out_dir / f"panelA_pre_post_CV_{q_label}_selected.png"
    legend_path = out_dir / f"panelA_pre_post_CV_{q_label}_legend.png"
    fig.savefig(panel_a_path, dpi=dpi, bbox_inches="tight")
    save_panel_a_legend_png(
        legend_path,
        partial_edge_color="#8C1E96",
        complete_ml_color=complete_ml_color,
        sham_color=sham_color,
        dpi=dpi,
    )
    if show:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # Panel B: effective lesion extent vs actual bird-level Q ΔCV.
    # ------------------------------------------------------------------
    bird_meta = (
        d.groupby(animal_col, dropna=False)
        .agg(
            lesion_pct=("lesion_pct", "median"),
            effective_lesion_pct=("effective_lesion_pct", "median"),
            lesion_group_for_plot=("lesion_group_for_plot", "first"),
            hit_type=("hit_type", "first"),
            treatment_group=("treatment_group", "first"),
        )
        .reset_index()
    )
    per_bird = bird_q.merge(bird_meta, on=animal_col, how="left")
    per_bird = per_bird[
        np.isfinite(_finite_numeric(per_bird["effective_lesion_pct"]))
        & np.isfinite(_finite_numeric(per_bird["q_delta_cv"]))
    ].copy()
    if exclude_sham_from_correlation:
        per_bird = per_bird[per_bird["lesion_group_for_plot"] != "sham (0%)"].copy()

    per_bird_csv = out_dir / f"panelB_effective_lesion_pct_vs_{q_label}_deltaCV_by_bird.csv"
    per_bird.to_csv(per_bird_csv, index=False)

    rho = np.nan
    p_spearman = np.nan
    if scipy_stats is not None and len(per_bird) >= 3:
        sp = scipy_stats.spearmanr(
            per_bird["effective_lesion_pct"].to_numpy(dtype=float),
            per_bird["q_delta_cv"].to_numpy(dtype=float),
        )
        rho = float(getattr(sp, "statistic", getattr(sp, "correlation", np.nan)))
        p_spearman = float(sp.pvalue)

    fig_b, ax_b = plt.subplots(figsize=(4.5, 3.8))
    point_colors = np.where(
        per_bird["lesion_group_for_plot"].astype(str) == "sham (0%)", sham_color,
        np.where(
            per_bird["lesion_group_for_plot"].astype(str) == "complete (100%)",
            complete_ml_color,
            partial_color,
        ),
    )
    ax_b.scatter(
        per_bird["effective_lesion_pct"].astype(float),
        per_bird["q_delta_cv"].astype(float),
        s=48, alpha=0.9, c=point_colors,
        edgecolors="black", linewidths=0.4, zorder=3,
    )

    if len(per_bird) >= 2:
        xb = per_bird["effective_lesion_pct"].to_numpy(dtype=float)
        yb = per_bird["q_delta_cv"].to_numpy(dtype=float)
        ok = np.isfinite(xb) & np.isfinite(yb)
        if ok.sum() >= 2 and np.unique(xb[ok]).size >= 2:
            slope, intercept = np.polyfit(xb[ok], yb[ok], 1)
            xx = np.linspace(float(np.nanmin(xb[ok])), float(np.nanmax(xb[ok])), 100)
            ax_b.plot(xx, intercept + slope * xx, linewidth=1.2, zorder=2)

    ax_b.axhline(0, linestyle="--", linewidth=1.0, zorder=1)
    ax_b.set_xlabel("Effective Area X lesion (%)")
    ax_b.set_ylabel(f"{q_label} Δ phrase duration CV\npost-lesion − late pre-lesion")
    ax_b.set_xlim(-5, 105)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_xticklabels(["Sham\n0", "25", "50", "75", "Complete\n100"])
    xt = ax_b.get_xticklabels()
    if len(xt) >= 5:
        xt[0].set_color(sham_color)
        xt[-1].set_color(complete_ml_color)
    if add_title:
        ax_b.set_title(f"Area X lesion extent vs bird-level {q_label} ΔCV")
    _pretty_axes(ax_b)
    fig_b.tight_layout()
    panel_b_path = out_dir / f"panelB_effective_lesion_pct_vs_{q_label}_deltaCV.png"
    fig_b.savefig(panel_b_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_b)

    report_path = out_dir / f"panelB_effective_lesion_pct_vs_{q_label}_deltaCV_stats.txt"
    report_lines = [
        f"Effective Area X lesion percentage vs bird-level {q_label} ΔCV",
        f"Quantile = {quantile}",
        f"Quantile method = {quantile_method}",
        "Animal-level endpoint = actual within-bird quantile of ALL qualifying syllable ΔCV values.",
        "Panel A selection = syllables with ΔCV >= that bird-specific quantile threshold.",
        "Sham = 0% effective lesion extent; complete lesion = 100%; partial = measured percentage.",
        f"Excluded sham from correlation = {exclude_sham_from_correlation}",
        f"n birds = {len(per_bird)}",
        f"Spearman rho = {rho:.6g}" if np.isfinite(rho) else "Spearman rho = NA",
        f"Spearman p = {p_spearman:.6g}" if np.isfinite(p_spearman) else "Spearman p = NA",
    ]
    report_path.write_text("\n".join(report_lines) + "\n")

    bird_q.to_csv(out_dir / f"bird_level_{q_label}_deltaCV_all_qualifying.csv", index=False)

    print(f"[SAVED] Panel A: {panel_a_path}")
    print(f"[SAVED] Panel A legend: {legend_path}")
    print(f"[SAVED] Panel A selected rows: {selected_csv}")
    print(f"[SAVED] Panel B: {panel_b_path}")
    print(f"[SAVED] Panel B bird table: {per_bird_csv}")
    print(f"[SAVED] Panel B stats: {report_path}")
    if np.isfinite(rho):
        print(f"[STATS] Spearman effective lesion % vs {q_label} ΔCV: rho={rho:.4f}, p={p_spearman:.6g}, n={len(per_bird)}")

    return {
        "panel_a_path": str(panel_a_path),
        "panel_b_path": str(panel_b_path),
        "legend_path": str(legend_path),
        "panel_a_selected_csv": str(selected_csv),
        "panel_b_per_bird_csv": str(per_bird_csv),
        "report_path": str(report_path),
        "n_panel_a_points": int(len(selected)),
        "n_panel_b_birds": int(len(per_bird)),
        "spearman_rho": rho,
        "spearman_p": p_spearman,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Make the lesion-extent supplemental panels using Figure 3 Q75 ΔCV selection."
    )
    p.add_argument("--selection_csv", required=True,
                   help="Figure 3 syllable-level quantile-selection or balanced pair-metrics CSV.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--animal_col", default="animal_id")
    p.add_argument("--syllable_col", default="syllable")
    p.add_argument("--pre_cv_col", default="pre_cv")
    p.add_argument("--post_cv_col", default="post_cv")
    p.add_argument("--delta_cv_col", default="delta_cv")
    p.add_argument("--quantile", type=float, default=0.75,
                   help="Within-bird ΔCV quantile. Default 0.75 = Q75, matching primary Figure 3.")
    p.add_argument(
        "--quantile_method", default="linear",
        choices=["linear", "lower", "higher", "midpoint", "nearest"],
        help="NumPy sample-quantile interpolation method. Figure 3 uses linear.",
    )
    p.add_argument("--metadata_excel", required=True)
    p.add_argument("--meta_sheet_name", default="metadata_with_hit_type")
    p.add_argument("--meta_animal_col", default="Animal ID")
    p.add_argument("--meta_hit_type_col", default=None)
    p.add_argument("--histology_volumes_dir", default=None)
    p.add_argument("--lesion_pct_mode", choices=["left", "right", "avg"], default="avg")
    p.add_argument("--left_lesion_pct_col", default="L_Percent_of_Area_X_Lesioned_pct")
    p.add_argument("--right_lesion_pct_col", default="R_Percent_of_Area_X_Lesioned_pct")
    p.add_argument("--exclude_sham_from_correlation", action="store_true")
    p.add_argument("--title", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    meta_sheet: Union[int, str] = args.meta_sheet_name
    if isinstance(meta_sheet, str) and meta_sheet.strip().isdigit():
        meta_sheet = int(meta_sheet.strip())

    plot_cv_q_lesion_extent(
        selection_csv=args.selection_csv,
        out_dir=args.out_dir,
        animal_col=args.animal_col,
        syllable_col=args.syllable_col,
        pre_cv_col=args.pre_cv_col,
        post_cv_col=args.post_cv_col,
        delta_cv_col=args.delta_cv_col,
        quantile=args.quantile,
        quantile_method=args.quantile_method,
        metadata_excel=args.metadata_excel,
        meta_sheet_name=meta_sheet,
        meta_animal_col=args.meta_animal_col,
        meta_hit_type_col=args.meta_hit_type_col,
        histology_volumes_dir=args.histology_volumes_dir,
        lesion_pct_mode=args.lesion_pct_mode,
        left_lesion_pct_col=args.left_lesion_pct_col,
        right_lesion_pct_col=args.right_lesion_pct_col,
        add_title=bool(args.title),
        exclude_sham_from_correlation=bool(args.exclude_sham_from_correlation),
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
