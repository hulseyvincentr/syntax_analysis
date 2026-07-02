# Version: v9 — bugfix: animal-level partial-lesion points are purple
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outlier_graphs_sd_paper_format.py

Paper-format version of the Area X lesion scatter plot.

Main plot:
    x = Late Pre phrase-duration standard deviation (ms)
    y = Post phrase-duration standard deviation (ms)

Key choices:
    • Uses the existing Std_ms column directly.
    • Pivots per Animal ID × Syllable × Group.
    • Sham saline = fixed color.
    • Complete medial+lateral lesions = fixed color.
    • Partial lesions = purple gradient by % Area X lesioned
      (darker purple = higher percentage lesioned).
    • Adds an animal-level correlation test:
          median syllable-level SD change vs % Area X lesioned

This script is designed to live next to outlier_graphs.py and reuse its metadata
loader, including recursive *_final_volumes.json search.
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


def compute_sd_change_vs_lesion_pct_correlation(
    merged: pd.DataFrame,
    *,
    animal_col: str,
    out_dir: PathLike,
    tag: str,
    lesion_pct_col: str = "lesion_pct",
    exclude_sham: bool = False,
    make_plot: bool = True,
    complete_hit_type_patterns: Sequence[str] = ("large lesion Area X not visible", "complete"),
    sham_color: str = "#009E73",
    complete_ml_color: str = "#7A7A7A",
    partial_color: str = "#6A51A3",
) -> Dict[str, Any]:
    """
    Test whether phrase-duration SD change correlates with % Area X lesioned.

    Two views are saved:
      1) one point per animal = animal-level median across retained syllables
      2) all retained syllable points, so you can inspect high-variance syllables directly

    For these plots/stats:
      - sham saline animals are assigned effective lesion % = 0
      - complete lesions are assigned effective lesion % = 100
      - partial lesions retain their measured lesion percentage
    """
    required = [animal_col, "pre_sd_ms", "post_sd_ms", lesion_pct_col]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        return {"ok": False, "reason": f"Missing columns: {missing}"}

    d = merged.copy()
    d["pre_sd_ms"] = _finite_numeric(d["pre_sd_ms"])
    d["post_sd_ms"] = _finite_numeric(d["post_sd_ms"])
    d[lesion_pct_col] = _finite_numeric(d[lesion_pct_col])
    d["delta_sd_ms"] = d["post_sd_ms"] - d["pre_sd_ms"]
    d["log2_post_pre_sd"] = np.log2(d["post_sd_ms"] / d["pre_sd_ms"])

    if "hit_type" not in d.columns:
        d["hit_type"] = "unknown"
    if "treatment_group" not in d.columns:
        d["treatment_group"] = "other"

    hit = d["hit_type"].astype(str)
    sham_mask = (hit == "sham saline injection") | (d["treatment_group"].astype(str) == "sham")
    complete_mask = _hit_type_contains(hit, complete_hit_type_patterns) & ~sham_mask

    d["effective_lesion_pct"] = d[lesion_pct_col].astype(float)
    d.loc[sham_mask, "effective_lesion_pct"] = 0.0
    d.loc[complete_mask, "effective_lesion_pct"] = 100.0

    d["lesion_group_for_plot"] = np.where(
        sham_mask, "sham (0%)",
        np.where(complete_mask, "complete (100%)", "partial / measured")
    )

    if exclude_sham:
        d = d[~sham_mask].copy()

    d = d[
        np.isfinite(d["effective_lesion_pct"])
        & np.isfinite(d["delta_sd_ms"])
        & np.isfinite(d["log2_post_pre_sd"])
    ].copy()

    out_dir = _ensure_dir(out_dir)
    result: Dict[str, Any] = {"ok": False, "reason": "not enough data"}

    if d.empty:
        return {"ok": False, "reason": "No finite effective_lesion_pct and SD-change rows after filtering."}

    all_points_csv = out_dir / f"sd_change_vs_lesion_pct__{_sanitize(tag)}__all_syllable_points.csv"
    d.to_csv(all_points_csv, index=False)

    per_animal = (
        d.groupby(animal_col, dropna=False)
         .agg(
             lesion_pct=(lesion_pct_col, "median"),
             effective_lesion_pct=("effective_lesion_pct", "median"),
             lesion_group_for_plot=("lesion_group_for_plot", "first"),
             n_syllables=("delta_sd_ms", "size"),
             median_pre_sd_ms=("pre_sd_ms", "median"),
             median_post_sd_ms=("post_sd_ms", "median"),
             median_delta_sd_ms=("delta_sd_ms", "median"),
             mean_delta_sd_ms=("delta_sd_ms", "mean"),
             median_log2_post_pre_sd=("log2_post_pre_sd", "median"),
             mean_log2_post_pre_sd=("log2_post_pre_sd", "mean"),
         )
         .reset_index()
    )

    per_animal = per_animal[
        np.isfinite(per_animal["effective_lesion_pct"])
        & np.isfinite(per_animal["median_delta_sd_ms"])
    ].copy()

    csv_path = out_dir / f"sd_change_vs_lesion_pct__{_sanitize(tag)}__per_animal.csv"
    per_animal.to_csv(csv_path, index=False)

    lines = [
        "Phrase-duration SD change vs % Area X lesioned",
        "Animal-level analysis: each animal contributes one median across retained syllables.",
        "Retained syllables are the same filtered set used in the scatter plot (e.g. topP70/post).",
        "For these plots/stats, sham animals are coded as 0% and complete lesions as 100%.",
        "Primary test: Spearman correlation using median_delta_sd_ms.",
        "",
        f"Excluded sham animals: {exclude_sham}",
        f"n animals = {len(per_animal)}",
        f"Per-animal CSV: {csv_path}",
        f"All-syllable CSV: {all_points_csv}",
        "",
    ]

    result = {
        "ok": True,
        "n_animals": int(len(per_animal)),
        "per_animal_csv": str(csv_path),
        "all_syllable_points_csv": str(all_points_csv),
    }

    if scipy_stats is not None and len(per_animal) >= 3:
        x = per_animal["effective_lesion_pct"].to_numpy(dtype=float)
        y_delta = per_animal["median_delta_sd_ms"].to_numpy(dtype=float)
        y_log2 = per_animal["median_log2_post_pre_sd"].to_numpy(dtype=float)

        sp_delta = scipy_stats.spearmanr(x, y_delta)
        pr_delta = scipy_stats.pearsonr(x, y_delta)
        sp_log2 = scipy_stats.spearmanr(x, y_log2)
        pr_log2 = scipy_stats.pearsonr(x, y_log2)

        result.update({
            "spearman_delta_sd": {
                "rho": float(getattr(sp_delta, "statistic", getattr(sp_delta, "correlation", np.nan))),
                "p_value": float(sp_delta.pvalue),
            },
            "pearson_delta_sd": {
                "r": float(getattr(pr_delta, "statistic", pr_delta[0] if hasattr(pr_delta, "__len__") else np.nan)),
                "p_value": float(getattr(pr_delta, "pvalue", pr_delta[1] if hasattr(pr_delta, "__len__") else np.nan)),
            },
            "spearman_log2_ratio": {
                "rho": float(getattr(sp_log2, "statistic", getattr(sp_log2, "correlation", np.nan))),
                "p_value": float(sp_log2.pvalue),
            },
            "pearson_log2_ratio": {
                "r": float(getattr(pr_log2, "statistic", pr_log2[0] if hasattr(pr_log2, "__len__") else np.nan)),
                "p_value": float(getattr(pr_log2, "pvalue", pr_log2[1] if hasattr(pr_log2, "__len__") else np.nan)),
            },
        })

        lines.extend([
            "Primary analysis:",
            f"Spearman effective_lesion_pct vs median_delta_sd_ms: rho={result['spearman_delta_sd']['rho']:.4f}, p={result['spearman_delta_sd']['p_value']:.6g}",
            "",
            "Secondary analyses:",
            f"Pearson effective_lesion_pct vs median_delta_sd_ms: r={result['pearson_delta_sd']['r']:.4f}, p={result['pearson_delta_sd']['p_value']:.6g}",
            f"Spearman effective_lesion_pct vs median_log2_post_pre_sd: rho={result['spearman_log2_ratio']['rho']:.4f}, p={result['spearman_log2_ratio']['p_value']:.6g}",
            f"Pearson effective_lesion_pct vs median_log2_post_pre_sd: r={result['pearson_log2_ratio']['r']:.4f}, p={result['pearson_log2_ratio']['p_value']:.6g}",
        ])
    else:
        lines.append("Too few animals or SciPy unavailable; correlation p-values were not computed.")

    report_path = out_dir / f"sd_change_vs_lesion_pct__{_sanitize(tag)}.txt"
    report_path.write_text("\n".join(lines))
    result["report_path"] = str(report_path)

    if make_plot and len(per_animal) > 0:
        # ---- Plot 1: one point per animal (median across retained syllables) ----
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        point_colors = np.where(
            per_animal["lesion_group_for_plot"].astype(str) == "sham (0%)", sham_color,
            np.where(per_animal["lesion_group_for_plot"].astype(str) == "complete (100%)", complete_ml_color, partial_color)
        )
        ax.scatter(
            per_animal["effective_lesion_pct"].astype(float),
            per_animal["median_delta_sd_ms"].astype(float),
            s=48,
            alpha=0.9,
            c=point_colors,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        if len(per_animal) >= 2:
            x = per_animal["effective_lesion_pct"].to_numpy(dtype=float)
            y = per_animal["median_delta_sd_ms"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 2 and np.unique(x[ok]).size >= 2:
                slope, intercept = np.polyfit(x[ok], y[ok], 1)
                xx = np.linspace(float(np.nanmin(x[ok])), float(np.nanmax(x[ok])), 100)
                ax.plot(xx, intercept + slope * xx, linewidth=1.2, zorder=2)
        ax.axhline(0, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_xlabel("Effective Area X lesion (%)")
        ax.set_ylabel("Median Δ phrase duration SD (ms)\npost-lesion − late pre-lesion")
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["Sham\n0", "25", "50", "75", "Complete\n100"])
        xtl = ax.get_xticklabels()
        if len(xtl) >= 5:
            xtl[0].set_color(sham_color)
            xtl[-1].set_color(complete_ml_color)
        _pretty_axes(ax)
        fig.tight_layout()
        plot_path = out_dir / f"sd_change_vs_lesion_pct__{_sanitize(tag)}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        result["plot_path"] = str(plot_path)

        # ---- Plot 2: all retained syllable points ----
        fig2, ax2 = plt.subplots(figsize=(4.8, 4.0))
        rng = np.random.default_rng(0)
        x_all = d["effective_lesion_pct"].to_numpy(dtype=float)
        y_all = d["delta_sd_ms"].to_numpy(dtype=float)
        jitter = rng.normal(loc=0.0, scale=1.0, size=len(d))
        x_plot = np.clip(x_all + jitter, -2, 102)

        colors_all = np.where(
            d["lesion_group_for_plot"].astype(str) == "sham (0%)", sham_color,
            np.where(d["lesion_group_for_plot"].astype(str) == "complete (100%)", complete_ml_color, partial_color)
        )
        ax2.scatter(
            x_plot,
            y_all,
            s=20,
            alpha=0.45,
            c=colors_all,
            edgecolors="none",
            zorder=2,
        )

        # Overlay animal medians for reference
        ax2.scatter(
            per_animal["effective_lesion_pct"].astype(float),
            per_animal["median_delta_sd_ms"].astype(float),
            s=58,
            alpha=1.0,
            facecolors="white",
            edgecolors="#1f1f1f",
            linewidths=0.9,
            zorder=4,
        )

        if len(per_animal) >= 2:
            x = per_animal["effective_lesion_pct"].to_numpy(dtype=float)
            y = per_animal["median_delta_sd_ms"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 2 and np.unique(x[ok]).size >= 2:
                slope, intercept = np.polyfit(x[ok], y[ok], 1)
                xx = np.linspace(float(np.nanmin(x[ok])), float(np.nanmax(x[ok])), 100)
                ax2.plot(xx, intercept + slope * xx, linewidth=1.2, zorder=3)

        ax2.axhline(0, linestyle="--", linewidth=1.0, zorder=1)
        ax2.set_xlabel("Effective Area X lesion (%)")
        ax2.set_ylabel("Δ phrase duration SD (ms)\npost-lesion − late pre-lesion")
        ax2.set_xlim(-5, 105)
        ax2.set_xticks([0, 25, 50, 75, 100])
        ax2.set_xticklabels(["Sham\n0", "25", "50", "75", "Complete\n100"])
        xtl2 = ax2.get_xticklabels()
        if len(xtl2) >= 5:
            xtl2[0].set_color(sham_color)
            xtl2[-1].set_color(complete_ml_color)
        _pretty_axes(ax2)

        handles = [
            mlines.Line2D([], [], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=sham_color, markeredgecolor="none", label="sham points (0%)"),
            mlines.Line2D([], [], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=partial_color, markeredgecolor="none", label="partial-lesion points"),
            mlines.Line2D([], [], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=complete_ml_color, markeredgecolor=complete_ml_color, label="complete-lesion points (100%)"),
            mlines.Line2D([], [], marker="o", linestyle="none", markersize=7,
                          markerfacecolor="white", markeredgecolor="#1f1f1f", label="animal median"),
        ]
        ax2.legend(handles=handles, loc="best", frameon=False, fontsize=8)
        fig2.tight_layout()
        all_points_plot_path = out_dir / f"sd_change_vs_lesion_pct__{_sanitize(tag)}__all_syllables.png"
        fig2.savefig(all_points_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        result["all_syllables_plot_path"] = str(all_points_plot_path)

    print("[CORRELATION] SD change vs lesion %")
    print(f"  Report: {report_path}")
    print(f"  Per-animal CSV: {csv_path}")
    print(f"  All-syllable CSV: {all_points_csv}")
    if "plot_path" in result:
        print(f"  Animal-median plot: {result['plot_path']}")
    if "all_syllables_plot_path" in result:
        print(f"  All-syllables plot: {result['all_syllables_plot_path']}")

    return result



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
    leg = ax.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
        handletextpad=0.8,
        labelspacing=0.7,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)
    return str(out_path)


# -----------------------------------------------------------------------------
# Main plot

# -----------------------------------------------------------------------------
# Main plot
# -----------------------------------------------------------------------------

def plot_pre_post_sd_scatter_paper_format(
    csv_path: PathLike,
    out_dir: PathLike,
    *,
    pre_group: str = "Late Pre",
    post_group: str = "Post",
    sd_col: str = "Std_ms",
    group_col: str = "Group",
    animal_col: str = "Animal ID",
    syllable_col: str = "Syllable",
    nphrases_col: str = "N_phrases",
    min_n_phrases: int = 5,
    top_percentile: Optional[float] = 70,
    rank_on: str = "post",  # pre, post, max
    metadata_excel: Optional[PathLike] = None,
    meta_sheet_name: Union[int, str] = "metadata_with_hit_type",
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: Optional[str] = None,
    histology_volumes_dir: Optional[PathLike] = None,
    lesion_pct_mode: str = "avg",  # left, right, avg
    left_lesion_pct_col: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col: str = "R_Percent_of_Area_X_Lesioned_pct",
    visible_col: str = "Area X visible in histology?",
    sham_color: str = "#009E73",  # teal, matches paper-style sham
    complete_ml_color: str = "#7A7A7A",  # grey, matches paper-style complete lesion
    partial_cmap: str = "Purples",
    partial_vmin: Optional[float] = None,
    partial_vmax: Optional[float] = None,
    complete_hit_type_patterns: Sequence[str] = ("large lesion Area X not visible", "complete"),
    partial_hit_type_patterns: Sequence[str] = ("Area X visible",),
    marker_size: float = 30.0,
    alpha: float = 0.9,
    log_scale: bool = True,
    add_title: bool = False,
    run_correlation: bool = True,
    show: bool = False,
) -> Dict[str, Any]:
    """
    Build the manuscript-style SD scatter plot using Std_ms directly.

    Each point is one Animal ID × Syllable pair.
    """
    csv_path = Path(csv_path)
    out_dir = _ensure_dir(out_dir)

    df = pd.read_csv(csv_path)
    required = [animal_col, syllable_col, group_col, sd_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df2 = df[df[group_col].astype(str).isin([pre_group, post_group])].copy()
    if nphrases_col in df2.columns:
        df2 = df2[df2[nphrases_col].fillna(0) >= min_n_phrases].copy()

    # The requested pivot: one SD value per Animal ID × Syllable × Group.
    # drop_duplicates avoids duplicated identical summary rows from affecting the pivot.
    df2 = df2.drop_duplicates([animal_col, syllable_col, group_col]).copy()
    df2[sd_col] = _finite_numeric(df2[sd_col])

    sd_piv = df2.pivot_table(
        index=[animal_col, syllable_col],
        columns=group_col,
        values=sd_col,
        aggfunc="mean",
    )

    if pre_group not in sd_piv.columns or post_group not in sd_piv.columns:
        raise ValueError(
            "Could not form both pre and post SD columns from Group pivot. "
            f"Have columns: {list(sd_piv.columns)}"
        )

    merged = sd_piv[[pre_group, post_group]].rename(
        columns={pre_group: "pre_sd_ms", post_group: "post_sd_ms"}
    ).reset_index()

    # Add N phrases for QC/reporting when available.
    if nphrases_col in df2.columns:
        n_piv = df2.pivot_table(
            index=[animal_col, syllable_col],
            columns=group_col,
            values=nphrases_col,
            aggfunc="max",
        )
        if pre_group in n_piv.columns and post_group in n_piv.columns:
            n_df = n_piv[[pre_group, post_group]].rename(
                columns={pre_group: "pre_n_phrases", post_group: "post_n_phrases"}
            ).reset_index()
            merged = merged.merge(n_df, on=[animal_col, syllable_col], how="left")

    merged = merged.dropna(subset=["pre_sd_ms", "post_sd_ms"]).copy()
    merged["delta_sd_ms"] = merged["post_sd_ms"] - merged["pre_sd_ms"]
    merged["log2_post_pre_sd"] = np.log2(merged["post_sd_ms"] / merged["pre_sd_ms"])

    if log_scale:
        merged = merged[(merged["pre_sd_ms"] > 0) & (merged["post_sd_ms"] > 0)].copy()

    # Attach metadata.
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

        merged["hit_type"] = merged[animal_col].astype(str).map(lambda a: str(_get_meta(a, "hit_type", "unknown")))
        merged["is_visible"] = merged[animal_col].astype(str).map(lambda a: bool(_get_meta(a, "is_visible", False)))
        merged["lesion_pct"] = merged[animal_col].astype(str).map(lambda a: _safe_float(_get_meta(a, "lesion_pct", np.nan)))
        merged["lesion_pct_avg"] = merged[animal_col].astype(str).map(lambda a: _safe_float(_get_meta(a, "lesion_pct_avg", np.nan)))
        merged["treatment_group"] = merged[animal_col].astype(str).map(lambda a: str(_get_meta(a, "treatment_group", "other")))
    else:
        merged["hit_type"] = "unknown"
        merged["is_visible"] = False
        merged["lesion_pct"] = np.nan
        merged["lesion_pct_avg"] = np.nan
        merged["treatment_group"] = "other"

    # Robust lesion-% fallback: some older outlier_graphs.py versions only read JSONs,
    # and some metadata files store % lesion directly in Excel columns.
    # Fill any missing lesion_pct values from Excel first, then from JSONs.
    if metadata_excel is not None:
        merged = _fill_lesion_pct_from_metadata_excel(
            merged,
            metadata_excel=metadata_excel,
            sheet_name=meta_sheet_name,
            animal_col=animal_col,
            meta_animal_col=meta_animal_col,
            left_lesion_pct_col=left_lesion_pct_col,
            right_lesion_pct_col=right_lesion_pct_col,
            lesion_pct_mode=lesion_pct_mode,
        )
    if histology_volumes_dir is not None:
        merged = _fill_lesion_pct_from_volume_jsons(
            merged,
            histology_volumes_dir=histology_volumes_dir,
            animal_col=animal_col,
            left_lesion_pct_col=left_lesion_pct_col,
            right_lesion_pct_col=right_lesion_pct_col,
            lesion_pct_mode=lesion_pct_mode,
        )

    finite_pct_animals = (
        merged.loc[np.isfinite(pd.to_numeric(merged["lesion_pct"], errors="coerce")), animal_col]
        .astype(str)
        .nunique()
    )
    finite_pct_rows = int(np.isfinite(pd.to_numeric(merged["lesion_pct"], errors="coerce")).sum())
    print(f"[INFO] Final finite lesion_pct: rows={finite_pct_rows}, animals={finite_pct_animals}")
    if finite_pct_rows == 0:
        print(
            "[WARN] No finite lesion_pct values were found. The partial-lesion gradient/colorbar cannot be drawn.\n"
            "       Check --histology_volumes_dir and/or lesion-percentage column names in metadata Excel."
        )

    # Optional within-animal top-percentile filter using SD values.
    if top_percentile is not None:
        if not (0 < float(top_percentile) < 100):
            raise ValueError("top_percentile must be between 0 and 100.")
        if rank_on == "pre":
            metric = merged["pre_sd_ms"].astype(float)
        elif rank_on == "post":
            metric = merged["post_sd_ms"].astype(float)
        elif rank_on == "max":
            metric = np.maximum(merged["pre_sd_ms"].astype(float), merged["post_sd_ms"].astype(float))
        else:
            raise ValueError("rank_on must be one of: pre, post, max")
        merged = merged.assign(_rank_metric=metric)
        thresh = merged.groupby(animal_col)["_rank_metric"].quantile(float(top_percentile) / 100.0).rename("_threshold")
        merged = merged.join(thresh, on=animal_col)
        merged = merged[merged["_rank_metric"] >= merged["_threshold"]].copy()
        merged.drop(columns=["_rank_metric", "_threshold"], inplace=True, errors="ignore")

    tag = f"pre_{_sanitize(pre_group)}__post_{_sanitize(post_group)}"
    if top_percentile is not None:
        tag = f"{tag}__topP{int(round(float(top_percentile)))}_{rank_on}"

    table_path = out_dir / f"pre_post_sd_table__{tag}.csv"
    merged.to_csv(table_path, index=False)

    # Plot masks.
    hit = merged["hit_type"].astype(str)
    finite_pct = np.isfinite(_finite_numeric(merged["lesion_pct"]))
    sham_mask = (hit == "sham saline injection") | (merged["treatment_group"].astype(str) == "sham")
    complete_mask = _hit_type_contains(hit, complete_hit_type_patterns) & ~sham_mask
    partial_mask = (_hit_type_contains(hit, partial_hit_type_patterns) | merged["is_visible"].astype(bool)) & finite_pct & ~sham_mask & ~complete_mask
    other_mask = ~(partial_mask | sham_mask | complete_mask)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))

    sc = None
    partial_color_vmin = partial_vmin
    partial_color_vmax = partial_vmax
    if partial_mask.any():
        partial_vals = merged.loc[partial_mask, "lesion_pct"].astype(float).to_numpy()
        finite_partial_vals = partial_vals[np.isfinite(partial_vals)]

        if finite_partial_vals.size > 0:
            if partial_color_vmin is None:
                partial_color_vmin = float(np.nanmin(finite_partial_vals))
            if partial_color_vmax is None:
                partial_color_vmax = float(np.nanmax(finite_partial_vals))

            # Avoid a degenerate color scale if all values are identical.
            if partial_color_vmax <= partial_color_vmin:
                pad = max(1.0, abs(partial_color_vmin) * 0.05)
                partial_color_vmin -= pad
                partial_color_vmax += pad

            print(
                f"[INFO] Partial-lesion color scale: "
                f"vmin={partial_color_vmin:.4g}, vmax={partial_color_vmax:.4g}"
            )

        sc = ax.scatter(
            merged.loc[partial_mask, "pre_sd_ms"].astype(float),
            merged.loc[partial_mask, "post_sd_ms"].astype(float),
            c=merged.loc[partial_mask, "lesion_pct"].astype(float),
            cmap=partial_cmap,
            vmin=partial_color_vmin,
            vmax=partial_color_vmax,
            s=marker_size,
            alpha=alpha,
            edgecolors="none",
            label="partial lesion",
            zorder=3,
        )

    if complete_mask.any():
        ax.scatter(
            merged.loc[complete_mask, "pre_sd_ms"].astype(float),
            merged.loc[complete_mask, "post_sd_ms"].astype(float),
            s=marker_size,
            alpha=0.95,
            color=complete_ml_color,
            edgecolors="black",
            linewidths=0.3,
            label="complete medial+lateral lesion",
            zorder=4,
        )

    if sham_mask.any():
        ax.scatter(
            merged.loc[sham_mask, "pre_sd_ms"].astype(float),
            merged.loc[sham_mask, "post_sd_ms"].astype(float),
            s=marker_size,
            alpha=0.95,
            color=sham_color,
            edgecolors="none",
            label="sham saline injection",
            zorder=2,
        )

    if other_mask.any():
        ax.scatter(
            merged.loc[other_mask, "pre_sd_ms"].astype(float),
            merged.loc[other_mask, "post_sd_ms"].astype(float),
            s=marker_size,
            alpha=0.55,
            color="lightgray",
            edgecolors="none",
            label="other / uncategorized",
            zorder=1,
        )

    x = merged["pre_sd_ms"].astype(float).to_numpy()
    y = merged["post_sd_ms"].astype(float).to_numpy()
    lo, hi = _line_limits(x, y, log_scale=log_scale)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="0.35", label="y=x", zorder=0)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(f"{_epoch_label(pre_group)} phrase duration SD (ms)", fontsize=12)
    ax.set_ylabel(f"{_epoch_label(post_group)} phrase duration SD (ms)", fontsize=12)
    if add_title:
        if top_percentile is None:
            ax.set_title("Pre vs post phrase-duration SD")
        else:
            ax.set_title(f"Top {int(round(float(top_percentile)))}% phrase-duration SD syllables")
    _pretty_axes(ax)

    # Colorbar directly beside axes. Legend is saved separately as a transparent PNG.
    if sc is not None:
        if make_axes_locatable is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.12)
            cbar = fig.colorbar(sc, cax=cax)
        else:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Area X lesioned (%)", fontsize=11)

    fig.tight_layout()
    fig_path = out_dir / f"pre_post_sd_scatter__{tag}__paper_colors.png"
    legend_path = out_dir / f"pre_post_sd_scatter__{tag}__legend.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    save_panel_a_legend_png(
        legend_path,
        partial_edge_color="#8C1E96",
        complete_ml_color=complete_ml_color,
        sham_color=sham_color,
    )
    if show:
        plt.show()
    plt.close(fig)

    corr_result: Dict[str, Any] = {"ok": False, "reason": "not computed"}
    if run_correlation:
        corr_result = compute_sd_change_vs_lesion_pct_correlation(
            merged,
            animal_col=animal_col,
            out_dir=out_dir,
            tag=tag,
            lesion_pct_col="lesion_pct",
            exclude_sham=False,
            make_plot=True,
            complete_hit_type_patterns=complete_hit_type_patterns,
            sham_color=sham_color,
            complete_ml_color=complete_ml_color,
            partial_color="#6A51A3",
        )

    print(f"[SAVED] SD scatter: {fig_path}")
    print(f"[SAVED] SD table:   {table_path}")

    return {
        "figure_path": fig_path,
        "legend_path": legend_path,
        "table_path": table_path,
        "table": merged,
        "correlation": corr_result,
        "n_points": int(len(merged)),
        "n_partial_points": int(partial_mask.sum()),
        "n_complete_points": int(complete_mask.sum()),
        "n_sham_points": int(sham_mask.sum()),
        "n_other_points": int(other_mask.sum()),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make paper-format pre/post phrase-duration SD scatter plot.")
    p.add_argument("--csv", required=True, help="Path to usage_balanced_phrase_duration_stats.csv")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--pre_group", default="Late Pre")
    p.add_argument("--post_group", default="Post")
    p.add_argument("--sd_col", default="Std_ms")
    p.add_argument("--animal_col", default="Animal ID")
    p.add_argument("--syllable_col", default="Syllable")
    p.add_argument("--group_col", default="Group")
    p.add_argument("--nphrases_col", default="N_phrases")
    p.add_argument("--min_n_phrases", type=int, default=5)
    p.add_argument("--scatter_top_percentile", type=float, default=70.0)
    p.add_argument("--rank_on", choices=["pre", "post", "max"], default="post")
    p.add_argument("--metadata_excel", required=True)
    p.add_argument("--meta_sheet_name", default="metadata_with_hit_type")
    p.add_argument("--meta_animal_col", default="Animal ID")
    p.add_argument("--meta_hit_type_col", default=None)
    p.add_argument("--histology_volumes_dir", default=None)
    p.add_argument("--lesion_pct_mode", choices=["left", "right", "avg"], default="avg")
    p.add_argument("--left_lesion_pct_col", default="L_Percent_of_Area_X_Lesioned_pct")
    p.add_argument("--right_lesion_pct_col", default="R_Percent_of_Area_X_Lesioned_pct")
    p.add_argument("--linear", action="store_true", help="Use linear axes instead of log-log.")
    p.add_argument("--title", action="store_true", help="Add title above plot.")
    p.add_argument("--no_correlation", action="store_true", help="Do not run SD-change vs lesion-percent correlation.")
    p.add_argument("--show", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    meta_sheet: Union[int, str] = args.meta_sheet_name
    if isinstance(meta_sheet, str) and meta_sheet.strip().isdigit():
        meta_sheet = int(meta_sheet.strip())

    plot_pre_post_sd_scatter_paper_format(
        csv_path=args.csv,
        out_dir=args.out_dir,
        pre_group=args.pre_group,
        post_group=args.post_group,
        sd_col=args.sd_col,
        animal_col=args.animal_col,
        syllable_col=args.syllable_col,
        group_col=args.group_col,
        nphrases_col=args.nphrases_col,
        min_n_phrases=args.min_n_phrases,
        top_percentile=args.scatter_top_percentile,
        rank_on=args.rank_on,
        metadata_excel=args.metadata_excel,
        meta_sheet_name=meta_sheet,
        meta_animal_col=args.meta_animal_col,
        meta_hit_type_col=args.meta_hit_type_col,
        histology_volumes_dir=args.histology_volumes_dir,
        lesion_pct_mode=args.lesion_pct_mode,
        log_scale=(not args.linear),
        add_title=bool(args.title),
        run_correlation=(not args.no_correlation),
        show=args.show,
    )


if __name__ == "__main__":
    main()
