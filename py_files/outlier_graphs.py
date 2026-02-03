# -*- coding: utf-8 -*-
"""
outlier_graphs.py

Utilities for visualizing high-variance syllables and pre/post variance comparisons
from a compiled phrase-duration stats CSV (e.g., usage_balanced_phrase_duration_stats.csv).

Key features
------------
• plot_pre_post_variance_scatter(): one scatter across ALL birds
      x = pre-lesion variance, y = post-lesion variance

• Metadata-driven coloring:
    - Discrete coloring by lesion hit type (e.g., sham / visible / not visible)
    - Optional second plot: Area X visible points colored continuously by % Area X lesioned

• One-tailed t-tests (printed to terminal + saved to .txt; NOT annotated on plots):
    A) NMA vs sham (per-animal medians across syllables; avoids pseudo-replication)
    B) Hit-type comparisons (ONE-TAILED; only the 3 requested comparisons):
        1) COMBINED (not visible + visible ML) vs sham saline injection (delta_log10_ratio)
        2) large lesion Area X not visible vs sham saline injection (delta_log10_ratio)
        3) Area X visible (single hit) vs sham saline injection (delta_log10_ratio)

Direction note (one-tailed)
---------------------------
One-tailed tests require a direction. This module uses:
    alternative="greater" by default  (tests mean(A) > mean(B))
You can switch to "less" via plot_pre_post_variance_scatter(ttest_alternative="less")
or via CLI (--ttest_alt less).

Compatibility notes
-------------------
- Your metadata Excel has sheet "metadata_with_hit_type" and contains:
    • Animal ID
    • Treatment type
    • Area X visible in histology?
    • Lesion hit type
  It may NOT include lesion% columns, so lesion% is sourced from *_final_volumes.json files.

- Histology volume JSONs may be inside nested subfolders. This module searches recursively.
- macOS AppleDouble files like "._USAxxxx...json" are ignored automatically.

Author: ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, Literal, Tuple

import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # optional nicer colorbar
except Exception:  # pragma: no cover
    make_axes_locatable = None  # type: ignore

try:
    from scipy import stats as _scipy_stats  # for t-test p-values
except Exception:  # pragma: no cover
    _scipy_stats = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

PathLike = Union[str, Path]
AggFunc = Literal["mean", "median", "max", "min"]
RankOn = Literal["pre", "post", "max"]
LesionPctMode = Literal["left", "right", "avg"]
TTestAlt = Literal["two-sided", "greater", "less"]


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize(s: str) -> str:
    return str(s).strip().replace("/", "-").replace("\\", "-").replace(" ", "_")


def _pick_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


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
    if s == "":
        return float("nan")
    # percent strings
    if s.endswith("%"):
        try:
            return float(s[:-1].strip())
        except Exception:
            return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _agg_series(x: pd.Series, agg: AggFunc) -> float:
    arr = x.to_numpy(dtype=float)
    if agg == "mean":
        return float(np.nanmean(arr))
    if agg == "median":
        return float(np.nanmedian(arr))
    if agg == "max":
        return float(np.nanmax(arr))
    if agg == "min":
        return float(np.nanmin(arr))
    raise ValueError(f"Unknown agg='{agg}'")


def _is_hidden_path(p: Path) -> bool:
    """True if any path component begins with '.' (including AppleDouble '._')."""
    for part in p.parts:
        if part.startswith("."):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Canonicalization helpers (for category strings)
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_hit_type(cat_raw: Any) -> str:
    """
    Normalize metadata hit-type strings onto the keys used by plotting and comparisons.
    """
    if cat_raw is None:
        return "unknown"
    s = str(cat_raw).strip()
    if s == "":
        return "unknown"
    low = s.lower()

    # sham
    if "sham" in low and "saline" in low:
        return "sham saline injection"

    # miss
    if low == "miss" or "miss" in low:
        return "miss"

    # not visible large lesion
    if "large" in low and "lesion" in low and ("not visible" in low):
        return "large lesion Area X not visible"

    # visible categories
    if "area x visible" in low:
        return " ".join(s.split())

    return " ".join(s.split())


def _parse_yes(raw: Any) -> bool:
    if raw is None:
        return False
    s = str(raw).strip().lower()
    return s in {"y", "yes", "true", "1", "t"}


def _is_visible(raw: Any) -> bool:
    """
    Interpret visibility cells (e.g., 'Area X visible in histology?').
    """
    if raw is None:
        return False
    s = str(raw).strip().lower()
    if not s:
        return False
    if "not visible" in s:
        return False
    if s.startswith("n"):
        return False
    if s.startswith("y"):
        return True
    if " visible" in s or s == "visible":
        return True
    return False


def _infer_treatment_group(treatment_type: Any, hit_type: str) -> str:
    """
    Return one of: 'sham', 'nma', 'other'
    Used for NMA vs sham t-tests.
    """
    ht = _canonical_hit_type(hit_type)
    if ht == "sham saline injection":
        return "sham"

    tt = str(treatment_type).strip().lower() if treatment_type is not None else ""
    is_nma = ("nma" in tt) and ("lesion" in tt)
    if is_nma:
        return "nma"
    return "other"


# ──────────────────────────────────────────────────────────────────────────────
# Robust metadata Excel reading (auto-select sheet)
# ──────────────────────────────────────────────────────────────────────────────

def _read_metadata_excel_best_sheet(
    metadata_excel: PathLike,
    sheet_name: Union[int, str] = 0,
    *,
    require_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, Union[int, str]]:
    """
    Read a metadata Excel sheet, but auto-switch to a better sheet if needed.
    Prefers 'metadata_with_hit_type' when present.
    """
    metadata_excel = Path(metadata_excel)
    require_cols = require_cols or []

    df_req = pd.read_excel(metadata_excel, sheet_name=sheet_name)
    if all(c in df_req.columns for c in require_cols):
        return df_req, sheet_name

    try:
        xls = pd.ExcelFile(metadata_excel)
        sheet_names = list(xls.sheet_names)
    except Exception:
        sheet_names = []

    preferred = ["metadata_with_hit_type", "Metadata_with_hit_type", "hit_type", "HitType"]
    for sname in preferred:
        if sname in sheet_names:
            df2 = pd.read_excel(metadata_excel, sheet_name=sname)
            if all(c in df2.columns for c in require_cols):
                return df2, sname

    return df_req, sheet_name


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for lesion % from volume JSONs (recursive, ignores AppleDouble)
# ──────────────────────────────────────────────────────────────────────────────

def _norm_key(k: str) -> str:
    return str(k).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _walk_json(obj: Any):
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
    """
    Extract lesion% values from a JSON object (searches recursively; flexible key matching).
    """
    lk = _norm_key(left_key)
    rk = _norm_key(right_key)
    left_vals: list[float] = []
    right_vals: list[float] = []

    for k, v in _walk_json(obj):
        nk = _norm_key(k)
        if nk == lk:
            left_vals.append(_safe_float(v))
            continue
        if nk == rk:
            right_vals.append(_safe_float(v))
            continue

        # fallback: heuristic match
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
        return float(np.mean(vals2))

    return _pick(left_vals), _pick(right_vals)


def _animal_id_from_filename(name: str) -> Optional[str]:
    """
    Extract an Animal ID from filenames like:
      USA5326_left_031524.02_final_volumes.json
    """
    m = re.match(r"^(USA\d+)", name)
    if m:
        return m.group(1)
    # fallback: first token before underscore if it looks like ID
    tok = name.split("_")[0]
    if tok.startswith("USA") and tok[3:].isdigit():
        return tok
    return None


@dataclass
class VolumeIndex:
    volumes_dir: Path
    by_animal: Dict[str, list[Path]]
    all_paths: list[Path]


def _build_volume_index(volumes_dir: Path) -> VolumeIndex:
    """
    Recursively collect *final*volumes*.json under volumes_dir, ignoring hidden/dotfiles.
    """
    volumes_dir = Path(volumes_dir)
    hits: list[Path] = []
    if volumes_dir.exists():
        for p in volumes_dir.rglob("*final*volumes*.json"):
            if not p.is_file():
                continue
            if p.name.startswith("._"):
                continue
            if _is_hidden_path(p):
                continue
            hits.append(p)

    hits = sorted(hits)
    by_animal: Dict[str, list[Path]] = {}
    for p in hits:
        aid = _animal_id_from_filename(p.name)
        if aid is None:
            continue
        by_animal.setdefault(aid, []).append(p)

    return VolumeIndex(volumes_dir=volumes_dir, by_animal=by_animal, all_paths=hits)


def _extract_pct_for_animal_from_index(
    vindex: VolumeIndex,
    animal_id: str,
    *,
    left_key: str,
    right_key: str,
) -> tuple[float, float]:
    """
    Read ALL candidate JSON files for this animal and average any found left/right values.
    """
    paths = vindex.by_animal.get(animal_id, [])
    if not paths:
        return float("nan"), float("nan")

    left_vals: list[float] = []
    right_vals: list[float] = []

    for jpath in paths:
        try:
            obj = json.loads(jpath.read_text())
        except Exception:
            continue
        l, r = _extract_left_right_lesion_pct_from_json_obj(obj, left_key=left_key, right_key=right_key)
        if np.isfinite(l):
            left_vals.append(l)
        if np.isfinite(r):
            right_vals.append(r)

    def _mean_or_nan(vals: list[float]) -> float:
        vals2 = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals2)) if vals2 else float("nan")

    return _mean_or_nan(left_vals), _mean_or_nan(right_vals)


# ──────────────────────────────────────────────────────────────────────────────
# Public utility: delete AppleDouble jsons (optional)
# ──────────────────────────────────────────────────────────────────────────────

def delete_appledouble_jsons(volumes_dir: PathLike, *, dry_run: bool = True) -> int:
    """
    Delete macOS AppleDouble jsons: ._*.json
    NOTE: This module already ignores these; deletion is optional.
    """
    volumes_dir = Path(volumes_dir)
    targets = [p for p in volumes_dir.rglob("._*.json") if p.is_file()]

    print(f"Found {len(targets)} AppleDouble ._*.json files under:\n  {volumes_dir}")
    for p in targets[:20]:
        print(" -", p)
    if len(targets) > 20:
        print(f" ... ({len(targets)-20} more)")

    if dry_run:
        print("\nDry run only (nothing deleted). Set dry_run=False to delete.")
        return len(targets)

    deleted = 0
    for p in targets:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"[WARN] Could not delete {p}: {e}")

    print(f"\nDeleted {deleted} files.")
    return deleted


# ──────────────────────────────────────────────────────────────────────────────
# Metadata summary loader
# ──────────────────────────────────────────────────────────────────────────────

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
    lesion_pct_mode: LesionPctMode = "avg",
) -> Dict[str, Dict[str, Any]]:
    """
    Build a compact per-animal metadata summary.
    """
    meta_df, chosen_sheet = _read_metadata_excel_best_sheet(
        metadata_excel,
        sheet_name=sheet_name,
        require_cols=[meta_animal_col],
    )

    if meta_animal_col not in meta_df.columns:
        raise ValueError(
            f"metadata excel missing meta_animal_col='{meta_animal_col}'. Columns: {list(meta_df.columns)}"
        )

    if meta_hit_type_col is None or meta_hit_type_col not in meta_df.columns:
        meta_hit_type_col = _pick_existing_col(
            meta_df,
            ["Lesion hit type", "Hit type", "Hit Type", "hit_type", "Category", "category",
             "Medial Area X hit type", "Lateral Area X hit type"],
        )

    # Build volume index once (recursive)
    vindex: Optional[VolumeIndex] = None
    if volumes_dir is not None:
        vdir = Path(volumes_dir)
        if vdir.exists():
            vindex = _build_volume_index(vdir)
        else:
            print(f"[WARN] volumes_dir does not exist: {vdir}")

    out: Dict[str, Dict[str, Any]] = {}

    for animal_id, g in meta_df.groupby(meta_animal_col, dropna=True):
        aid = str(animal_id).strip()
        if not aid:
            continue

        # Treatment type: first non-null
        ttype = ""
        if treatment_type_col in g.columns:
            vals = g[treatment_type_col].dropna().astype(str).tolist()
            ttype = vals[0] if vals else ""

        # visibility: any row True
        is_vis = g[visible_col].apply(_is_visible).any() if visible_col in g.columns else False

        # hit type (prefer explicit column)
        hit_type = "unknown"
        if meta_hit_type_col is not None and meta_hit_type_col in g.columns:
            raw = g[meta_hit_type_col].dropna()
            if not raw.empty:
                hit_type = _canonical_hit_type(raw.iloc[0])

        # If missing, infer coarse one
        if hit_type == "unknown":
            if ("sham" in str(ttype).lower()) or ("saline" in str(ttype).lower() and "lesion" not in str(ttype).lower()):
                hit_type = "sham saline injection"
            else:
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

        treatment_group = _infer_treatment_group(ttype, hit_type)

        # lesion %: from volumes JSONs
        l_pct = float("nan")
        r_pct = float("nan")
        if vindex is not None:
            jl, jr = _extract_pct_for_animal_from_index(
                vindex,
                aid,
                left_key=left_lesion_pct_col,
                right_key=right_lesion_pct_col,
            )
            if np.isfinite(jl):
                l_pct = jl
            if np.isfinite(jr):
                r_pct = jr

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
            "treatment_type": ttype,
            "treatment_group": treatment_group,
            "is_visible": bool(is_vis),
            "lesion_pct_left": l_pct,
            "lesion_pct_right": r_pct,
            "lesion_pct_avg": avg_pct,
            "lesion_pct": pct,
            "_metadata_sheet_used": str(chosen_sheet),
            "_n_volume_json_candidates": int(len(vindex.by_animal.get(aid, []))) if vindex is not None else 0,
        }

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Color maps
# ──────────────────────────────────────────────────────────────────────────────

def _default_areax_hit_type_color_map() -> Dict[str, Any]:
    purples = plt.get_cmap("Purples")
    return {
        "large lesion Area X not visible": (0.0, 0.0, 0.0, 1.0),
        "Area X visible (single hit)": purples(0.55),
        "Area X visible (medial+lateral hit)": purples(0.85),
        "sham saline injection": (1.0, 0.0, 0.0, 1.0),
        "miss": (0.55, 0.55, 0.55, 1.0),
        "unknown": (0.55, 0.55, 0.55, 1.0),
    }


def _category_color_map(categories: list[str], *, cmap_name: str = "tab20") -> Dict[str, Any]:
    cmap = plt.get_cmap(cmap_name)
    n = max(len(categories), 1)
    out: Dict[str, Any] = {}
    for i, cat in enumerate(categories):
        if cat == "unknown":
            out[cat] = (0.55, 0.55, 0.55, 1.0)
        else:
            out[cat] = cmap(i / max(n - 1, 1))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# T-tests
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PairwiseTTest:
    metric: str
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    t_stat: float
    p_value: float
    method: str  # "welch" or "student"


def _ttest_ind(
    a: np.ndarray,
    b: np.ndarray,
    *,
    equal_var: bool,
    alternative: TTestAlt = "greater",
) -> Tuple[float, float]:
    """
    Two-sample t-test.
    - alternative: "greater" tests mean(a) > mean(b)
                   "less"    tests mean(a) < mean(b)
                   "two-sided" tests mean(a) != mean(b)
    - Uses scipy if available; otherwise returns (t, nan) for p-value.
    - Backwards-compatible with older SciPy versions lacking `alternative=`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")

    if _scipy_stats is not None:
        # Try SciPy's built-in `alternative=` if available; otherwise compute two-sided and convert.
        try:
            import inspect
            sig = inspect.signature(_scipy_stats.ttest_ind)
            has_alt = "alternative" in sig.parameters
        except Exception:
            has_alt = False

        if has_alt:
            res = _scipy_stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit", alternative=alternative)
            return float(res.statistic), float(res.pvalue)

        # Older SciPy: do two-sided, then convert to one-tailed if requested
        res = _scipy_stats.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
        t = float(res.statistic)
        p2 = float(res.pvalue)

        if not np.isfinite(p2):
            return t, float("nan")

        if alternative == "two-sided":
            return t, p2

        # Convert two-sided p to one-sided p based on t sign
        if alternative == "greater":
            # H1: mean(a) > mean(b)
            p1 = (p2 / 2.0) if (t > 0) else (1.0 - p2 / 2.0)
            return t, float(p1)

        if alternative == "less":
            # H1: mean(a) < mean(b)
            p1 = (p2 / 2.0) if (t < 0) else (1.0 - p2 / 2.0)
            return t, float(p1)

        return t, p2  # fallback

    # fallback: compute t, but p needs df/cdf -> nan
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    na, nb = a.size, b.size
    se = (
        np.sqrt(va / na + vb / nb)
        if not equal_var
        else np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2) * (1 / na + 1 / nb))
    )
    if se <= 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    return float(t), float("nan")


def _per_animal_metrics(merged: pd.DataFrame, *, animal_col: str, group_col: str) -> pd.DataFrame:
    """
    Compute per-animal medians across syllables for:
      - log10_pre
      - log10_post
      - delta_log10_ratio = log10_post - log10_pre
    """
    df = merged.copy()
    df["log10_pre"] = np.log10(df["pre_variance"].astype(float).to_numpy())
    df["log10_post"] = np.log10(df["post_variance"].astype(float).to_numpy())
    df["delta_log10_ratio"] = df["log10_post"] - df["log10_pre"]

    per = (
        df.groupby([animal_col, group_col], dropna=False)[["log10_pre", "log10_post", "delta_log10_ratio"]]
        .median()
        .reset_index()
    )
    return per


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines))


def _compute_nma_vs_sham_tests(
    merged: pd.DataFrame,
    *,
    animal_col: str,
    treatment_group_col: str,
    out_dir: Path,
    tag: str,
    equal_var: bool = False,  # False = Welch (default)
    alternative: TTestAlt = "greater",
    print_to_terminal: bool = True,
) -> Dict[str, Any]:
    if treatment_group_col not in merged.columns:
        return {"ok": False, "reason": f"Missing column: {treatment_group_col}"}

    per = _per_animal_metrics(merged, animal_col=animal_col, group_col=treatment_group_col)
    nma = per[per[treatment_group_col] == "nma"]
    sham = per[per[treatment_group_col] == "sham"]

    if len(nma) < 2 or len(sham) < 2:
        return {"ok": False, "reason": f"Not enough animals (nma={len(nma)}, sham={len(sham)})"}

    method = "student" if equal_var else "welch"
    results: list[PairwiseTTest] = []
    for metric in ["log10_pre", "log10_post", "delta_log10_ratio"]:
        a = nma[metric].to_numpy(dtype=float)
        b = sham[metric].to_numpy(dtype=float)
        t, p = _ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
        results.append(
            PairwiseTTest(
                metric=metric,
                group_a="nma",
                group_b="sham",
                n_a=int(np.isfinite(a).sum()),
                n_b=int(np.isfinite(b).sum()),
                mean_a=float(np.nanmean(a)),
                mean_b=float(np.nanmean(b)),
                t_stat=float(t),
                p_value=float(p),
                method=method,
            )
        )

    out_dir = _ensure_dir(out_dir)
    report_path = out_dir / f"ttest_nma_vs_sham__{_sanitize(tag)}__{alternative}.txt"
    lines = [
        f"NMA vs sham two-sample t-tests (ONE-TAILED: alternative='{alternative}')",
        "Per-animal medians across syllables (avoids pseudo-replication).",
        f"Method: {method} (equal_var={equal_var})",
        "",
    ]
    for r in results:
        lines.append(
            f"{r.metric}: nma(n={r.n_a}) mean={r.mean_a:.4f} | sham(n={r.n_b}) mean={r.mean_b:.4f} "
            f"=> t={r.t_stat:.3f}, p={r.p_value:.6g}"
        )
    _write_report(report_path, lines)

    if print_to_terminal:
        print(f"[T-TEST] NMA vs sham (ONE-TAILED: alternative='{alternative}') — printed (not annotated).")
        print(f"  Report: {report_path}")
        for r in results:
            print(
                f"   - {r.metric}: t={r.t_stat:.3g}, p={r.p_value:.6g} "
                f"(nma n={r.n_a}, sham n={r.n_b}) [{r.method}]"
            )

    return {
        "ok": True,
        "method": method,
        "alternative": alternative,
        "report_path": str(report_path),
        "results": [r.__dict__ for r in results],
    }


def _compute_hit_type_comparisons(
    merged: pd.DataFrame,
    *,
    animal_col: str,
    hit_type_col: str,
    out_dir: Path,
    tag: str,
    equal_var: bool = False,  # False = Welch
    alternative: TTestAlt = "greater",
    metric: str = "delta_log10_ratio",
    print_to_terminal: bool = True,
) -> Dict[str, Any]:
    """
    ONE-TAILED requested comparisons (per-animal medians across syllables):
      1) COMBINED (not visible + visible ML) vs sham saline injection
      2) large lesion Area X not visible vs sham saline injection
      3) Area X visible (single hit) vs sham saline injection
    Metric: delta_log10_ratio by default
    """
    if hit_type_col not in merged.columns:
        return {"ok": False, "reason": f"Missing column: {hit_type_col}"}

    per = _per_animal_metrics(merged, animal_col=animal_col, group_col=hit_type_col)

    def vec(mask: np.ndarray) -> np.ndarray:
        return per.loc[mask, metric].to_numpy(dtype=float)

    ht = per[hit_type_col].astype(str)

    sham_mask = (ht == "sham saline injection")
    notvis_mask = (ht == "large lesion Area X not visible")
    visml_mask = (ht == "Area X visible (medial+lateral hit)")
    vissingle_mask = (ht == "Area X visible (single hit)")
    combined_mask = notvis_mask | visml_mask

    comparisons = [
        ("COMBINED (not visible + visible ML)", "sham saline injection", combined_mask, sham_mask),
        ("large lesion Area X not visible", "sham saline injection", notvis_mask, sham_mask),
        ("Area X visible (single hit)", "sham saline injection", vissingle_mask, sham_mask),
    ]

    method = "student" if equal_var else "welch"
    results: list[PairwiseTTest] = []

    for name_a, name_b, mask_a, mask_b in comparisons:
        a = vec(mask_a.to_numpy())
        b = vec(mask_b.to_numpy())
        t, p = _ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
        results.append(
            PairwiseTTest(
                metric=metric,
                group_a=name_a,
                group_b=name_b,
                n_a=int(np.isfinite(a).sum()),
                n_b=int(np.isfinite(b).sum()),
                mean_a=float(np.nanmean(a)) if np.isfinite(a).any() else float("nan"),
                mean_b=float(np.nanmean(b)) if np.isfinite(b).any() else float("nan"),
                t_stat=float(t),
                p_value=float(p),
                method=method,
            )
        )

    out_dir = _ensure_dir(out_dir)
    report_path = out_dir / f"ttest_hitType_requested_pairwise__{_sanitize(tag)}__{alternative}.txt"

    lines = [
        f"Hit-type requested pairwise t-tests (ONE-TAILED: alternative='{alternative}')",
        "Metric computed per animal as median across syllables (avoids pseudo-replication).",
        f"Metric: {metric}",
        f"Method: {method} (equal_var={equal_var})",
        "",
    ]
    for r in results:
        lines.append(
            f"{r.metric} | {r.group_a} vs {r.group_b}: "
            f"n_a={r.n_a} mean_a={r.mean_a:.4f} | n_b={r.n_b} mean_b={r.mean_b:.4f} "
            f"=> t={r.t_stat:.3f}, p={r.p_value:.6g}"
        )
    _write_report(report_path, lines)

    if print_to_terminal:
        print(f"[T-TEST] Hit-type requested pairwise tests (ONE-TAILED: alternative='{alternative}') — printed (not annotated).")
        print(f"  Report: {report_path}")
        for r in results:
            print(
                f"   - {r.group_a} vs {r.group_b} ({r.metric}): "
                f"t={r.t_stat:.3g}, p={r.p_value:.6g} (n_a={r.n_a}, n_b={r.n_b}) [{r.method}]"
            )

    return {
        "ok": True,
        "method": method,
        "alternative": alternative,
        "metric": metric,
        "report_path": str(report_path),
        "results": [r.__dict__ for r in results],
    }


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
    csv_path = Path(csv_path)
    out_dir = _ensure_dir(Path(out_dir))
    df = pd.read_csv(csv_path)

    missing = [c for c in [animal_col, syllable_col, variance_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if nphrases_col in df.columns:
        df = df[df[nphrases_col].fillna(0) >= min_n_phrases].copy()

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

        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(sel))))
        ax.barh(sel[syllable_col].astype(str), sel["variance_agg"].astype(float))
        ax.invert_yaxis()
        ax.set_xlabel(f"{variance_col} (aggregated: {agg})")
        ax.set_ylabel("Syllable")
        ax.set_title(f"{animal_id}: Top ≥ P{top_percentile:.0f} variance syllables (n={len(sel)})")
        fig.tight_layout()

        fpath = out_dir / f"{_sanitize(str(animal_id))}_topP{int(round(top_percentile))}_variance_bar.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
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
# Main scatter: all-birds pre vs post variance scatter
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
    top_percentile: Optional[float] = None,
    outlier_percentile: Optional[float] = None,
    rank_on: RankOn = "post",
    metadata_excel: Optional[PathLike] = None,
    meta_sheet_name: Union[int, str] = 0,
    meta_animal_col: str = "Animal ID",
    meta_hit_type_col: Optional[str] = None,
    histology_volumes_dir: Optional[PathLike] = None,
    lesion_pct_mode: LesionPctMode = "avg",
    left_lesion_pct_col: str = "L_Percent_of_Area_X_Lesioned_pct",
    right_lesion_pct_col: str = "R_Percent_of_Area_X_Lesioned_pct",
    visible_col: str = "Area X visible in histology?",
    color_by_hit_type: bool = False,
    hit_type_color_map: Optional[Dict[str, Any]] = None,
    make_continuous_lesion_pct_plot: bool = True,
    add_nma_vs_sham_ttests: bool = True,
    add_hit_type_pairwise_tests: bool = True,
    ttest_equal_var: bool = False,   # False = Welch; True = Student
    ttest_alternative: Literal["greater", "less"] = "greater",
    log_scale: bool = True,
    color_by_animal: bool = False,
    alpha: float = 0.7,
    marker_size: float = 28.0,
    show: bool = False,
) -> Dict[str, Any]:
    """
    Build ONE scatterplot across ALL birds:
      x-axis = pre-lesion variance
      y-axis = post-lesion variance
    """

    if outlier_percentile is not None and top_percentile is None:
        top_percentile = outlier_percentile

    csv_path = Path(csv_path)
    out_dir = _ensure_dir(Path(out_dir))

    df = pd.read_csv(csv_path)

    missing = [c for c in [animal_col, syllable_col, group_col, variance_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df2 = df[df[group_col].astype(str).isin([pre_group, post_group])].copy()
    if df2.empty:
        raise ValueError(
            f"No rows found for pre_group='{pre_group}' and post_group='{post_group}'. "
            f"Available groups: {sorted(df[group_col].astype(str).unique().tolist())}"
        )

    if nphrases_col in df2.columns:
        df2 = df2[df2[nphrases_col].fillna(0) >= min_n_phrases].copy()

    var_piv = df2.pivot_table(
        index=[animal_col, syllable_col],
        columns=group_col,
        values=variance_col,
        aggfunc="mean",
    )

    if pre_group not in var_piv.columns or post_group not in var_piv.columns:
        raise ValueError(
            "Could not form both pre and post columns from Group pivot. "
            f"Have columns: {list(var_piv.columns)}"
        )

    merged = var_piv[[pre_group, post_group]].rename(
        columns={pre_group: "pre_variance", post_group: "post_variance"}
    ).reset_index()

    merged = merged.dropna(subset=["pre_variance", "post_variance"]).copy()

    if log_scale:
        merged = merged[(merged["pre_variance"] > 0) & (merged["post_variance"] > 0)].copy()

    # Attach metadata
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
            lambda a: _canonical_hit_type(_get_meta(a, "hit_type", "unknown"))
        )
        merged["is_visible"] = merged[animal_col].astype(str).map(
            lambda a: bool(_get_meta(a, "is_visible", False))
        )
        merged["lesion_pct"] = merged[animal_col].astype(str).map(
            lambda a: _safe_float(_get_meta(a, "lesion_pct", float("nan")))
        )
        merged["treatment_group"] = merged[animal_col].astype(str).map(
            lambda a: str(_get_meta(a, "treatment_group", "other"))
        )
        merged["_metadata_sheet_used"] = merged[animal_col].astype(str).map(
            lambda a: str(_get_meta(a, "_metadata_sheet_used", ""))
        )
    else:
        merged["hit_type"] = "unknown"
        merged["is_visible"] = False
        merged["lesion_pct"] = float("nan")
        merged["treatment_group"] = "other"
        merged["_metadata_sheet_used"] = ""

    # Optional: within-animal percentile filter
    if top_percentile is not None:
        if not (0 < top_percentile < 100):
            raise ValueError("top_percentile must be between 0 and 100 (exclusive).")

        if rank_on == "pre":
            metric_rank = merged["pre_variance"].astype(float)
        elif rank_on == "post":
            metric_rank = merged["post_variance"].astype(float)
        elif rank_on == "max":
            metric_rank = np.maximum(
                merged["pre_variance"].astype(float),
                merged["post_variance"].astype(float),
            )
        else:
            raise ValueError(f"rank_on must be one of: pre, post, max. Got: {rank_on}")

        merged = merged.assign(_rank_metric=metric_rank)
        thresh = (
            merged.groupby(animal_col)["_rank_metric"]
            .quantile(top_percentile / 100.0)
            .rename("_threshold")
        )
        merged = merged.join(thresh, on=animal_col)
        merged = merged[merged["_rank_metric"] >= merged["_threshold"]].copy()
        merged.drop(columns=["_rank_metric", "_threshold"], inplace=True, errors="ignore")

    # Save merged table
    tag = f"pre_{_sanitize(pre_group)}__post_{_sanitize(post_group)}"
    if top_percentile is None:
        table_path = out_dir / f"pre_post_variance_table__{tag}.csv"
    else:
        table_path = out_dir / f"pre_post_variance_table__{tag}__topP{int(round(top_percentile))}_{rank_on}.csv"
    merged.to_csv(table_path, index=False)

    # Title
    if top_percentile is None:
        title_prefix = "Pre vs Post variance"
    else:
        title_prefix = f"Pre vs Post top {int(round(top_percentile))}% variance"

    # Hit-type colormap
    hit_cmap: Dict[str, Any] = (
        dict(hit_type_color_map) if hit_type_color_map is not None else _default_areax_hit_type_color_map()
    )
    sham_color = hit_cmap.get("sham saline injection", (1.0, 0.0, 0.0, 1.0))

    # ── T-tests (PRINT + REPORT; NOT ON PLOT) ────────────────────────────────
    ttest_nma: Dict[str, Any] = {"ok": False, "reason": "not computed"}
    ttest_hit: Dict[str, Any] = {"ok": False, "reason": "not computed"}

    ttag = tag if top_percentile is None else f"{tag}__topP{int(round(top_percentile))}_{rank_on}"

    if add_nma_vs_sham_ttests and metadata_excel is not None:
        ttest_nma = _compute_nma_vs_sham_tests(
            merged,
            animal_col=animal_col,
            treatment_group_col="treatment_group",
            out_dir=out_dir,
            tag=ttag,
            equal_var=ttest_equal_var,
            alternative=ttest_alternative,
            print_to_terminal=True,
        )

    if add_hit_type_pairwise_tests and metadata_excel is not None and color_by_hit_type:
        ttest_hit = _compute_hit_type_comparisons(
            merged,
            animal_col=animal_col,
            hit_type_col="hit_type",
            out_dir=out_dir,
            tag=ttag,
            equal_var=ttest_equal_var,
            alternative=ttest_alternative,
            metric="delta_log10_ratio",
            print_to_terminal=True,
        )

    # ── Plot 1: discrete hit-type (or other coloring modes) ──────────────────
    fig1, ax1 = plt.subplots(figsize=(7.5, 7.0))

    if color_by_hit_type:
        if metadata_excel is None:
            raise ValueError("color_by_hit_type=True requires metadata_excel.")

        cats = sorted(merged["hit_type"].astype(str).fillna("unknown").unique().tolist())

        missing_cats = [c for c in cats if c not in hit_cmap]
        if missing_cats:
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

        ax1.legend(
            title="Hit type",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
        )

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
            handles = [
                plt.Line2D([], [], linestyle="none", marker="o", markersize=6, label=str(name))
                for name in uniques
            ]
            ax1.legend(handles=handles, title="Animal", loc="center left",
                       bbox_to_anchor=(1.02, 0.5), frameon=False)

    else:
        ax1.scatter(
            merged["pre_variance"].astype(float).to_numpy(),
            merged["post_variance"].astype(float).to_numpy(),
            alpha=alpha,
            s=marker_size,
            edgecolors="none",
        )

    # y=x reference line
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

    # ── Plot 2: continuous lesion % (Area X visible only) ─────────────────────
    fig2_path: Optional[Path] = None
    if (
        make_continuous_lesion_pct_plot
        and metadata_excel is not None
        and color_by_hit_type
        and ("lesion_pct" in merged.columns)
    ):
        vis_mask = merged["is_visible"].astype(bool) & np.isfinite(merged["lesion_pct"].astype(float))
        if vis_mask.any():
            fig2, ax2 = plt.subplots(figsize=(7.5, 7.0))

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

            sham_mask2 = merged["hit_type"].astype(str) == "sham saline injection"
            notvis_mask2 = merged["hit_type"].astype(str) == "large lesion Area X not visible"
            other_mask = ~(vis_mask | sham_mask2 | notvis_mask2)

            if sham_mask2.any():
                ax2.scatter(
                    merged.loc[sham_mask2, "pre_variance"].astype(float).to_numpy(),
                    merged.loc[sham_mask2, "post_variance"].astype(float).to_numpy(),
                    s=marker_size,
                    alpha=0.9,
                    color=sham_color,
                    edgecolors="none",
                )

            if notvis_mask2.any():
                ax2.scatter(
                    merged.loc[notvis_mask2, "pre_variance"].astype(float).to_numpy(),
                    merged.loc[notvis_mask2, "post_variance"].astype(float).to_numpy(),
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
                    markerfacecolor=sham_color,
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
            handles.append(mlines.Line2D([], [], color="red", linestyle="--", label="y=x"))

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
            n_vis = int(merged["is_visible"].astype(bool).sum())
            n_finite = int(np.isfinite(merged["lesion_pct"].astype(float)).sum())
            print(
                "[INFO] Continuous lesion-% plot skipped: no finite lesion_pct values for any Area X visible animals.\n"
                f"       Visible points: {n_vis} | Finite lesion_pct points: {n_finite}\n"
                "       If you expected lesion %:\n"
                "         • Ensure histology_volumes_dir points to a folder containing (or above) *_final_volumes.json\n"
                "         • Ensure macOS '._*.json' files are removed/ignored (this module ignores them)\n"
                "         • Ensure JSON keys include left/right lesion% (this module searches flexibly)\n"
            )

    # Backward-compatible return keys to avoid KeyError in older snippets
    ret = {
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
        # new clearer keys
        "ttest_nma_vs_sham": ttest_nma,
        "ttest_hit_type_comparisons": ttest_hit,
        # legacy aliases (so your Spyder console code won’t crash)
        "nma_vs_sham_ttests": ttest_nma,
        "hit_type_ttests": ttest_hit,
    }
    return ret


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

    p.add_argument("--variance_col", default="Variance_ms2", help="Variance column to plot.")
    p.add_argument("--animal_col", default="Animal ID", help="Animal ID column name.")
    p.add_argument("--syllable_col", default="Syllable", help="Syllable column name.")
    p.add_argument("--group_col", default="Group", help="Group column name.")
    p.add_argument("--nphrases_col", default="N_phrases", help="N_phrases column name.")
    p.add_argument("--min_n_phrases", type=int, default=5, help="Minimum N_phrases to include.")

    p.add_argument("--top_percentile", type=float, default=90.0,
                   help="(bar mode) percentile threshold within each animal (e.g., 90 means keep ≥90th percentile).")
    p.add_argument("--agg", choices=["mean", "median", "max", "min"], default="mean",
                   help="(bar mode) how to aggregate variance across rows per (animal, syllable).")

    p.add_argument("--pre_group", default="Late Pre", help="(scatter mode) pre-lesion group label.")
    p.add_argument("--post_group", default="Post", help="(scatter mode) post-lesion group label.")
    p.add_argument("--scatter_top_percentile", type=float, default=None,
                   help="(scatter mode) optional percentile filter per animal (e.g., 90 keeps top 10%).")
    p.add_argument("--rank_on", choices=["pre", "post", "max"], default="post",
                   help="(scatter mode) metric used for percentile ranking if scatter_top_percentile is set.")
    p.add_argument("--linear", action="store_true", help="(scatter mode) use linear axes instead of log-log.")
    p.add_argument("--color_by_animal", action="store_true", help="(scatter mode) color points by animal.")

    p.add_argument("--metadata_excel", default=None,
                   help="(scatter mode) Path to metadata Excel for lesion hit-type coloring.")
    p.add_argument("--meta_sheet_name", default=0,
                   help="(scatter mode) Sheet name or index in metadata Excel (default 0).")
    p.add_argument("--meta_animal_col", default="Animal ID",
                   help="(scatter mode) Animal ID column in metadata Excel.")
    p.add_argument("--meta_hit_type_col", default=None,
                   help="(scatter mode) Optional hit-type/category column name in metadata Excel "
                        "(e.g., 'Lesion hit type').")
    p.add_argument("--color_by_hit_type", action="store_true",
                   help="(scatter mode) Color points by lesion hit-type (requires metadata_excel).")

    p.add_argument("--histology_volumes_dir", default=None,
                   help="(scatter mode) Directory ABOVE nested *_final_volumes.json files (recursive search).")
    p.add_argument("--lesion_pct_mode", choices=["left", "right", "avg"], default="avg",
                   help="(scatter mode) which lesion %% to use when available.")
    p.add_argument("--no_continuous_plot", action="store_true",
                   help="(scatter mode) disable the continuous lesion-%% plot.")

    p.add_argument("--no_ttests", action="store_true",
                   help="(scatter mode) disable NMA vs sham t-tests.")
    p.add_argument("--no_hit_type_tests", action="store_true",
                   help="(scatter mode) disable requested hit-type pairwise tests.")
    p.add_argument("--student_t", action="store_true",
                   help="Use Student t-test (equal_var=True) instead of Welch (default).")
    p.add_argument("--ttest_alt", choices=["greater", "less"], default="greater",
                   help="One-tailed alternative: 'greater' tests mean(A) > mean(B); 'less' tests mean(A) < mean(B).")

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
            histology_volumes_dir=args.histology_volumes_dir,
            lesion_pct_mode=args.lesion_pct_mode,
            make_continuous_lesion_pct_plot=(not args.no_continuous_plot),
            add_nma_vs_sham_ttests=(not args.no_ttests),
            add_hit_type_pairwise_tests=(not args.no_hit_type_tests),
            ttest_equal_var=bool(args.student_t),
            ttest_alternative=args.ttest_alt,
            log_scale=(not args.linear),
            color_by_animal=args.color_by_animal,
            show=args.show,
        )


if __name__ == "__main__":
    main()


"""
# Example Spyder console usage:

from pathlib import Path
import sys
import importlib
import numpy as np

code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/syntax_analysis/py_files")
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import outlier_graphs as og
importlib.reload(og)

compiled_csv = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/usage_balanced_phrase_duration_stats.csv")
metadata_xlsx = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx")
histology_volumes_dir = Path("/Volumes/my_own_SSD/histology_files/lesion_quantification_csvs_jsons")
out_dir = compiled_csv.parent / "outlier_variance_scatter"

res = og.plot_pre_post_variance_scatter(
    csv_path=compiled_csv,
    out_dir=out_dir,
    pre_group="Late Pre",
    post_group="Post",
    top_percentile=70,
    rank_on="post",
    metadata_excel=metadata_xlsx,
    meta_sheet_name="metadata_with_hit_type",
    color_by_hit_type=True,
    histology_volumes_dir=histology_volumes_dir,   # recursive search supported
    lesion_pct_mode="avg",
    make_continuous_lesion_pct_plot=True,
    add_nma_vs_sham_ttests=True,
    add_hit_type_pairwise_tests=True,
    ttest_equal_var=False,         # False = Welch (recommended). True = Student.
    ttest_alternative="greater",   # one-tailed direction
    log_scale=True,
    show=True,
)

print("Discrete plot:", res["figure_path_discrete"])
print("Continuous plot:", res["figure_path_continuous"])
print("Saved table:", res["table_path"])

print("NMA vs sham report:", res["nma_vs_sham_ttests"].get("report_path"))
print("Hit-type report:", res["hit_type_ttests"].get("report_path"))

m = res["table"]
print("\nFinite lesion_pct rows:", int(np.isfinite(m["lesion_pct"].astype(float)).sum()))
print("Visible + finite lesion_pct rows:",
      int((m["is_visible"].astype(bool) & np.isfinite(m["lesion_pct"].astype(float))).sum()))
"""
