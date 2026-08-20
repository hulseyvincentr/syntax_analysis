#!/usr/bin/env python3
"""Continuous within-bird analysis of phrase-duration change versus BC change.

Purpose
-------
For every BC-qualified syllable in pooled medial+lateral (M+L) lesion birds,
relate a continuous phrase-duration variability change metric to
Bhattacharyya coefficient change:

    x = phrase-duration variability change
    y = BC_post - BC_pre

A negative association means syllables with larger post-lesion increases in
phrase-duration variability tend to show larger post-lesion reductions in
acoustic overlap/stability.

Primary inferential statistic
-----------------------------
The primary statistic is the equal-bird mean of within-bird Spearman
correlations. Each bird contributes one correlation, regardless of repertoire
size. Its p-value is obtained by permuting DeltaBC labels among syllables
within each bird and rerunning the complete statistic.

Secondary effect-size statistic
-------------------------------
The script also reports an equal-bird weighted, bird-centered linear slope.
Every bird has total weight 1, so birds with more BC-qualified syllables do
not dominate. Bird-cluster bootstrap confidence intervals and within-bird
permutation p-values are provided.

No syllables are selected by an outcome threshold. This avoids the circularity
of choosing the most affected syllables and then testing the same observations.
The analysis is still associational and should be interpreted together with
sample sizes and per-bird plots.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires scipy. Install it in the active environment."
    ) from exc


POOLED_ML = "Medial and Lateral lesion"


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split()).lower()


def normalize_token(value: Any) -> str:
    """Make integer-like identifiers such as 3, 3.0, and '3.0' agree."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {norm_text(column): column for column in columns}
    for candidate in candidates:
        key = norm_text(candidate)
        if key in lookup:
            return lookup[key]
    return None


def canonical_group(value: Any) -> str:
    text = norm_text(value)
    if not text:
        return "unknown"
    if "sham" in text or "saline" in text:
        return "sham saline injection"
    if "lateral lesion only" in text or "lateral only" in text:
        return "Lateral lesion only"
    if (
        ("medial" in text and "lateral" in text)
        or "m+l" in text
        or "medial_and_lateral" in text
    ):
        return POOLED_ML
    return str(value).strip()


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite_array(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    return arr[np.isfinite(arr)]


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    values = np.asarray(list(p_values), dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    p = values[finite]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


# -----------------------------------------------------------------------------
# Input loading and identifier matching
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuous within-bird relationship between phrase-duration "
            "variability change and Bhattacharyya coefficient change."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variance-csv",
        required=True,
        help=(
            "Full Figure 3 animal-by-syllable metrics table, normally "
            "figure3_balanced_pair_metrics.csv."
        ),
    )
    parser.add_argument(
        "--bc-csv",
        required=True,
        help="Figure 4 bc_batch_cluster_level_long.csv.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--phrase-metric",
        choices=[
            "delta_cv",
            "log_cv_ratio",
            "delta_variance",
            "log_variance_ratio",
        ],
        default="delta_cv",
        help="Continuous phrase-duration variability change metric.",
    )
    parser.add_argument(
        "--bc-method",
        default="selected_bins",
        help="BC method retained when the BC table contains bc_method.",
    )
    parser.add_argument(
        "--set-name",
        default="all_clusters",
        help="BC set retained when the BC table contains set_name.",
    )
    parser.add_argument(
        "--min-syllables-primary",
        type=int,
        default=3,
        help=(
            "Minimum matched syllables required for a bird to enter the "
            "primary across-bird association test. Birds with two syllables "
            "are retained descriptively and in a sensitivity result."
        ),
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Monte Carlo within-bird permutations for aggregate tests.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Bird-cluster bootstrap replicates for confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_variance_table(path: str, phrase_metric: str) -> pd.DataFrame:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    table = pd.read_csv(csv_path)
    animal_col = first_present(
        table.columns, ["animal_id", "Animal ID", "bird_id", "bird"]
    )
    syllable_col = first_present(
        table.columns, ["syllable", "Syllable", "label", "cluster_id"]
    )
    pre_var_col = first_present(
        table.columns,
        ["pre_variance_s2", "pre_variance", "pre_variance_ms2"],
    )
    post_var_col = first_present(
        table.columns,
        ["post_variance_s2", "post_variance", "post_variance_ms2"],
    )
    pre_cv_col = first_present(table.columns, ["pre_cv", "cv_pre"])
    post_cv_col = first_present(table.columns, ["post_cv", "cv_post"])

    missing = []
    if animal_col is None:
        missing.append("animal ID")
    if syllable_col is None:
        missing.append("syllable ID")
    if phrase_metric in {"delta_cv", "log_cv_ratio"}:
        if pre_cv_col is None:
            missing.append("pre CV")
        if post_cv_col is None:
            missing.append("post CV")
    else:
        if pre_var_col is None:
            missing.append("pre variance")
        if post_var_col is None:
            missing.append("post variance")
    if missing:
        raise ValueError(
            f"Variance CSV is missing {missing}. Found columns: {list(table.columns)}"
        )

    out = pd.DataFrame(
        {
            "animal_id": table[animal_col].astype(str).str.strip(),
            "syllable": table[syllable_col].map(normalize_token),
        }
    )
    if pre_var_col is not None:
        out["pre_variance"] = pd.to_numeric(table[pre_var_col], errors="coerce")
    if post_var_col is not None:
        out["post_variance"] = pd.to_numeric(table[post_var_col], errors="coerce")
    if pre_cv_col is not None:
        out["pre_cv"] = pd.to_numeric(table[pre_cv_col], errors="coerce")
    if post_cv_col is not None:
        out["post_cv"] = pd.to_numeric(table[post_cv_col], errors="coerce")

    for optional in [
        "lesion_group_detailed",
        "lesion_group",
        "display_group",
        "n_balanced",
        "n_pre",
        "n_post",
    ]:
        if optional in table.columns:
            out[optional] = table[optional]

    if phrase_metric == "delta_cv":
        out["phrase_change"] = out["post_cv"] - out["pre_cv"]
        units = "Post CV - Pre CV"
    elif phrase_metric == "log_cv_ratio":
        valid = (out["pre_cv"] > 0) & (out["post_cv"] > 0)
        out["phrase_change"] = np.nan
        out.loc[valid, "phrase_change"] = np.log(
            out.loc[valid, "post_cv"] / out.loc[valid, "pre_cv"]
        )
        units = "log(Post CV / Pre CV)"
    elif phrase_metric == "delta_variance":
        out["phrase_change"] = out["post_variance"] - out["pre_variance"]
        units = "Post variance - Pre variance"
    elif phrase_metric == "log_variance_ratio":
        valid = (out["pre_variance"] > 0) & (out["post_variance"] > 0)
        out["phrase_change"] = np.nan
        out.loc[valid, "phrase_change"] = np.log(
            out.loc[valid, "post_variance"] / out.loc[valid, "pre_variance"]
        )
        units = "log(Post variance / Pre variance)"
    else:  # pragma: no cover
        raise AssertionError(phrase_metric)

    out["phrase_change_metric"] = phrase_metric
    out["phrase_change_units"] = units
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["animal_id", "syllable", "phrase_change"])
    out = out[(out["animal_id"] != "") & (out["syllable"] != "")]
    out = (
        out.sort_values(["animal_id", "syllable"])
        .drop_duplicates(["animal_id", "syllable"], keep="first")
        .reset_index(drop=True)
    )
    print(
        f"[INFO] Variance table: {len(out)} usable pairs across "
        f"{out['animal_id'].nunique()} birds"
    )
    return out


def choose_bc_identifier(
    table: pd.DataFrame,
    animal_col: str,
    variance: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    candidates = [
        "cluster_id",
        "cluster_token",
        "syllable",
        "Syllable",
        "label",
        "cluster_label",
    ]
    present = [column for column in candidates if column in table.columns]
    if not present:
        raise ValueError(
            "BC table has no cluster/syllable identifier column. "
            f"Found columns: {list(table.columns)}"
        )

    variance_keys = variance[["animal_id", "syllable"]].drop_duplicates()
    reports = []
    for column in present:
        candidate = pd.DataFrame(
            {
                "animal_id": table[animal_col].astype(str).str.strip(),
                "syllable": table[column].map(normalize_token),
            }
        ).drop_duplicates()
        candidate = candidate[
            (candidate["animal_id"] != "") & (candidate["syllable"] != "")
        ]
        matched = variance_keys.merge(
            candidate,
            on=["animal_id", "syllable"],
            how="inner",
        )
        reports.append(
            {
                "candidate_column": column,
                "matched_variance_pairs": len(matched),
                "matched_birds": matched["animal_id"].nunique(),
                "available_bc_pairs": len(candidate),
            }
        )

    report = pd.DataFrame(reports).sort_values(
        ["matched_variance_pairs", "matched_birds"],
        ascending=False,
        kind="stable",
    )
    for row in report.itertuples(index=False):
        print(
            f"[INFO] BC ID candidate {row.candidate_column!r}: "
            f"{row.matched_variance_pairs} variance pairs matched across "
            f"{row.matched_birds} birds"
        )
    best = report.iloc[0]
    if int(best["matched_variance_pairs"]) == 0:
        raise ValueError(
            "No BC identifier column matched the variance table. "
            "Inspect the identifier report and source tables."
        )
    chosen = str(best["candidate_column"])
    print(f"[OK] Using BC identifier column: {chosen}")
    return chosen, report.reset_index(drop=True)


def load_bc_table(
    path: str,
    variance: pd.DataFrame,
    bc_method: str,
    set_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    table = pd.read_csv(csv_path)
    print(f"[INFO] BC input columns: {list(table.columns)}")

    if "bc_method" in table.columns:
        mask = table["bc_method"].astype(str) == bc_method
        if not mask.any():
            raise ValueError(
                f"No BC rows have bc_method={bc_method!r}. "
                f"Available: {sorted(table['bc_method'].dropna().astype(str).unique())}"
            )
        table = table[mask].copy()
        print(f"[INFO] Retained bc_method={bc_method!r}: {len(table)} rows")

    if "set_name" in table.columns:
        mask = table["set_name"].astype(str) == set_name
        if not mask.any():
            raise ValueError(
                f"No BC rows have set_name={set_name!r}. "
                f"Available: {sorted(table['set_name'].dropna().astype(str).unique())}"
            )
        table = table[mask].copy()
        print(f"[INFO] Retained set_name={set_name!r}: {len(table)} rows")

    animal_col = first_present(
        table.columns, ["animal_id", "Animal ID", "bird_id", "bird"]
    )
    if animal_col is None:
        raise ValueError("BC table lacks an animal identifier column.")

    identifier_col, identifier_report = choose_bc_identifier(
        table, animal_col, variance
    )
    pre_col = first_present(table.columns, ["bc_pre", "pre_bc", "BC_pre"])
    post_col = first_present(table.columns, ["bc_post", "post_bc", "BC_post"])
    delta_col = first_present(
        table.columns,
        ["bc_delta_post_minus_pre", "delta_bc", "bc_delta", "post_minus_pre_bc"],
    )
    group_col = first_present(
        table.columns,
        [
            "lesion_hit_type",
            "lesion_group_detailed",
            "lesion_group",
            "display_group",
            "raw_lesion_hit_type",
        ],
    )
    if pre_col is None or post_col is None:
        raise ValueError(
            "BC table must contain pre and post BC columns. "
            f"Found: {list(table.columns)}"
        )
    if group_col is None:
        raise ValueError(
            "BC table must contain a lesion-group column to identify pooled M+L birds."
        )

    out = pd.DataFrame(
        {
            "animal_id": table[animal_col].astype(str).str.strip(),
            "syllable": table[identifier_col].map(normalize_token),
            "bc_pre": pd.to_numeric(table[pre_col], errors="coerce"),
            "bc_post": pd.to_numeric(table[post_col], errors="coerce"),
            "lesion_group_raw": table[group_col],
        }
    )
    if delta_col is not None:
        out["delta_bc_input"] = pd.to_numeric(table[delta_col], errors="coerce")
    out["delta_bc"] = out["bc_post"] - out["bc_pre"]
    out["lesion_group"] = out["lesion_group_raw"].map(canonical_group)
    out["bc_identifier_column"] = identifier_col
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["bc_pre", "bc_post", "delta_bc"])
    out = out[(out["animal_id"] != "") & (out["syllable"] != "")]
    out = out[out["lesion_group"] == POOLED_ML].copy()
    out = (
        out.sort_values(["animal_id", "syllable"])
        .drop_duplicates(["animal_id", "syllable"], keep="first")
        .reset_index(drop=True)
    )
    print(
        f"[INFO] BC table after filters: {len(out)} pooled M+L pairs across "
        f"{out['animal_id'].nunique()} birds"
    )
    return out, identifier_report


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    centered = x - np.mean(x)
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return np.nan
    return float(np.sum(centered * (y - np.mean(y))) / denominator)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(scipy_stats.spearmanr(x, y).statistic)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def one_sided_permutation_p(
    observed: float,
    null_values: np.ndarray,
    alternative: str,
) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or len(null_values) == 0:
        return np.nan
    if alternative == "less":
        count = int(np.sum(null_values <= observed + 1e-15))
    elif alternative == "greater":
        count = int(np.sum(null_values >= observed - 1e-15))
    else:
        count = int(np.sum(np.abs(null_values) >= abs(observed) - 1e-15))
    return (count + 1.0) / (len(null_values) + 1.0)


def per_bird_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    statistic: str,
    n_permutations: int,
    rng: np.random.Generator,
    alternative: str = "less",
) -> tuple[float, int, bool]:
    """Exact for n<=8; Monte Carlo otherwise."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    stat_fn = spearman_rho if statistic == "spearman" else linear_slope
    observed = stat_fn(x, y)
    if not np.isfinite(observed) or n < 3:
        return np.nan, 0, False

    if n <= 8:
        permutations = itertools.permutations(range(n))
        null = np.fromiter(
            (stat_fn(x, y[list(order)]) for order in permutations),
            dtype=float,
            count=math.factorial(n),
        )
        # Exact enumeration already includes the observed ordering.
        if alternative == "less":
            p = float(np.mean(null <= observed + 1e-15))
        elif alternative == "greater":
            p = float(np.mean(null >= observed - 1e-15))
        else:
            p = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
        return p, len(null), True

    null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        null[index] = stat_fn(x, rng.permutation(y))
    return (
        one_sided_permutation_p(observed, null, alternative),
        n_permutations,
        False,
    )


def equal_bird_mean_rho(data: pd.DataFrame) -> float:
    values = []
    for _, bird in data.groupby("animal_id", sort=False):
        rho = spearman_rho(
            bird["phrase_change"].to_numpy(float),
            bird["delta_bc"].to_numpy(float),
        )
        if np.isfinite(rho):
            values.append(rho)
    return float(np.mean(values)) if values else np.nan


def equal_bird_weighted_slope(data: pd.DataFrame) -> float:
    numerator = 0.0
    denominator = 0.0
    for _, bird in data.groupby("animal_id", sort=False):
        x = bird["phrase_change"].to_numpy(float)
        y = bird["delta_bc"].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        n = len(x)
        if n < 2:
            continue
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        if np.sum(x_centered**2) <= 0:
            continue
        weight = 1.0 / n
        numerator += float(weight * np.sum(x_centered * y_centered))
        denominator += float(weight * np.sum(x_centered**2))
    if denominator <= 0:
        return np.nan
    return numerator / denominator


def permute_within_birds(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pieces = []
    for _, bird in data.groupby("animal_id", sort=False):
        permuted = bird.copy()
        permuted["delta_bc"] = rng.permutation(
            permuted["delta_bc"].to_numpy(float)
        )
        pieces.append(permuted)
    return pd.concat(pieces, ignore_index=True)


def aggregate_permutation_test(
    data: pd.DataFrame,
    statistic_fn,
    n_permutations: int,
    rng: np.random.Generator,
    alternative: str = "less",
) -> tuple[float, float, np.ndarray]:
    observed = float(statistic_fn(data))
    null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        null[index] = statistic_fn(permute_within_birds(data, rng))
    p_value = one_sided_permutation_p(observed, null, alternative)
    return observed, p_value, null


def hierarchical_bootstrap(
    data: pd.DataFrame,
    statistic_fn,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    birds = list(data["animal_id"].drop_duplicates())
    if not birds:
        return np.array([], dtype=float)
    draws = np.empty(n_bootstrap, dtype=float)
    groups = {bird: data[data["animal_id"] == bird].copy() for bird in birds}

    for replicate in range(n_bootstrap):
        sampled_birds = rng.choice(birds, size=len(birds), replace=True)
        pieces = []
        for pseudo_index, bird_id in enumerate(sampled_birds):
            bird = groups[bird_id]
            sampled_rows = bird.sample(
                n=len(bird),
                replace=True,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).copy()
            sampled_rows["animal_id"] = f"bootstrap_{pseudo_index}"
            pieces.append(sampled_rows)
        bootstrap_data = pd.concat(pieces, ignore_index=True)
        draws[replicate] = statistic_fn(bootstrap_data)
    return draws


def summarize_per_bird(
    data: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for bird_index, (animal_id, bird) in enumerate(
        data.groupby("animal_id", sort=True)
    ):
        x = bird["phrase_change"].to_numpy(float)
        y = bird["delta_bc"].to_numpy(float)
        rho = spearman_rho(x, y)
        slope = linear_slope(x, y)
        rho_p, rho_nperm, rho_exact = per_bird_permutation_p(
            x,
            y,
            statistic="spearman",
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed + 1000 + bird_index),
            alternative="less",
        )
        slope_p, slope_nperm, slope_exact = per_bird_permutation_p(
            x,
            y,
            statistic="slope",
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed + 2000 + bird_index),
            alternative="less",
        )
        rows.append(
            {
                "animal_id": animal_id,
                "n_syllables": len(bird),
                "spearman_rho": rho,
                "spearman_permutation_p_one_sided_negative": rho_p,
                "spearman_n_permutations": rho_nperm,
                "spearman_exact_permutation": rho_exact,
                "linear_slope_deltaBC_per_phrase_change_unit": slope,
                "slope_permutation_p_one_sided_negative": slope_p,
                "slope_n_permutations": slope_nperm,
                "slope_exact_permutation": slope_exact,
                "pearson_r": pearson_r(x, y),
                "median_phrase_change": float(np.median(x)),
                "median_delta_bc": float(np.median(y)),
                "mean_phrase_change": float(np.mean(x)),
                "mean_delta_bc": float(np.mean(y)),
                "min_phrase_change": float(np.min(x)),
                "max_phrase_change": float(np.max(x)),
                "min_delta_bc": float(np.min(y)),
                "max_delta_bc": float(np.max(y)),
            }
        )

    result = pd.DataFrame(rows)
    result["spearman_bh_q"] = bh_fdr(
        result["spearman_permutation_p_one_sided_negative"]
    )
    result["slope_bh_q"] = bh_fdr(
        result["slope_permutation_p_one_sided_negative"]
    )
    return result


def exact_signflip_mean(values: np.ndarray, alternative: str = "less") -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan
    observed = float(np.mean(values))
    if n <= 20:
        null = np.empty(1 << n, dtype=float)
        for mask in range(1 << n):
            signs = np.array(
                [1.0 if ((mask >> index) & 1) else -1.0 for index in range(n)]
            )
            null[mask] = float(np.mean(values * signs))
        if alternative == "less":
            return float(np.mean(null <= observed + 1e-15))
        if alternative == "greater":
            return float(np.mean(null >= observed - 1e-15))
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    raise ValueError("Exact sign-flip is only implemented for <=20 birds.")


def run_overall_analysis(
    data: pd.DataFrame,
    min_syllables: int,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = data.groupby("animal_id").size()
    included_birds = counts[counts >= min_syllables].index
    analysis = data[data["animal_id"].isin(included_birds)].copy()
    excluded = counts[counts < min_syllables]

    if analysis["animal_id"].nunique() < 2:
        raise ValueError(
            f"Only {analysis['animal_id'].nunique()} bird(s) meet "
            f"min_syllables={min_syllables}; cannot run an across-bird test."
        )

    rho_observed, rho_p, _ = aggregate_permutation_test(
        analysis,
        equal_bird_mean_rho,
        n_permutations,
        np.random.default_rng(seed + 10),
        alternative="less",
    )
    slope_observed, slope_p, _ = aggregate_permutation_test(
        analysis,
        equal_bird_weighted_slope,
        n_permutations,
        np.random.default_rng(seed + 20),
        alternative="less",
    )

    rho_boot = hierarchical_bootstrap(
        analysis,
        equal_bird_mean_rho,
        n_bootstrap,
        np.random.default_rng(seed + 30),
    )
    slope_boot = hierarchical_bootstrap(
        analysis,
        equal_bird_weighted_slope,
        n_bootstrap,
        np.random.default_rng(seed + 40),
    )

    bird_stats = summarize_per_bird(analysis, n_permutations, seed + 50)
    rho_signflip = exact_signflip_mean(
        bird_stats["spearman_rho"].to_numpy(float), alternative="less"
    )
    slope_signflip = exact_signflip_mean(
        bird_stats["linear_slope_deltaBC_per_phrase_change_unit"].to_numpy(float),
        alternative="less",
    )

    overall = pd.DataFrame(
        [
            {
                "analysis": label,
                "minimum_syllables_per_bird": min_syllables,
                "n_birds": analysis["animal_id"].nunique(),
                "n_syllables": len(analysis),
                "included_birds": ";".join(sorted(included_birds.astype(str))),
                "excluded_birds_and_counts": ";".join(
                    f"{bird}:{int(count)}" for bird, count in excluded.items()
                ),
                "equal_bird_mean_spearman_rho": rho_observed,
                "rho_within_bird_permutation_p_one_sided_negative": rho_p,
                "rho_bird_cluster_bootstrap_ci_low": float(
                    np.nanpercentile(rho_boot, 2.5)
                ),
                "rho_bird_cluster_bootstrap_ci_high": float(
                    np.nanpercentile(rho_boot, 97.5)
                ),
                "mean_rho_exact_signflip_p_one_sided_negative": rho_signflip,
                "equal_bird_weighted_centered_slope": slope_observed,
                "slope_within_bird_permutation_p_one_sided_negative": slope_p,
                "slope_bird_cluster_bootstrap_ci_low": float(
                    np.nanpercentile(slope_boot, 2.5)
                ),
                "slope_bird_cluster_bootstrap_ci_high": float(
                    np.nanpercentile(slope_boot, 97.5)
                ),
                "mean_bird_slope_exact_signflip_p_one_sided_negative": slope_signflip,
                "n_permutations": n_permutations,
                "n_bootstrap": n_bootstrap,
            }
        ]
    )
    return overall, bird_stats


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def metric_axis_label(metric: str) -> str:
    labels = {
        "delta_cv": "Phrase-duration change: Post CV - Pre CV",
        "log_cv_ratio": "Phrase-duration change: log(Post CV / Pre CV)",
        "delta_variance": "Phrase-duration change: Post variance - Pre variance",
        "log_variance_ratio": (
            "Phrase-duration change: log(Post variance / Pre variance)"
        ),
    }
    return labels[metric]


def plot_per_bird_grid(
    data: pd.DataFrame,
    per_bird: pd.DataFrame,
    metric: str,
    out_path: Path,
    dpi: int,
    show: bool,
) -> None:
    birds = sorted(data["animal_id"].unique())
    n_cols = 3
    n_rows = int(math.ceil(len(birds) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    stats_lookup = per_bird.set_index("animal_id")
    for axis, bird_id in zip(axes.flat, birds):
        bird = data[data["animal_id"] == bird_id].copy()
        x = bird["phrase_change"].to_numpy(float)
        y = bird["delta_bc"].to_numpy(float)
        axis.scatter(x, y, s=45, alpha=0.9, edgecolors="white", linewidths=0.5)
        slope = linear_slope(x, y)
        if np.isfinite(slope) and np.unique(x).size >= 2:
            grid = np.linspace(np.min(x), np.max(x), 100)
            intercept = float(np.mean(y) - slope * np.mean(x))
            axis.plot(grid, intercept + slope * grid, linewidth=1.5)
        axis.axhline(0.0, linestyle="--", linewidth=0.9)
        axis.axvline(0.0, linestyle=":", linewidth=0.9)

        row = stats_lookup.loc[bird_id]
        title = (
            f"{bird_id}  n={int(row['n_syllables'])}\n"
            f"Spearman rho={row['spearman_rho']:.3f}, "
            f"perm. p={row['spearman_permutation_p_one_sided_negative']:.3g}"
            if np.isfinite(row["spearman_rho"])
            else f"{bird_id}  n={int(row['n_syllables'])}\nrho unavailable"
        )
        axis.set_title(title, fontsize=10.5)
        axis.tick_params(labelsize=9)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)

    for axis in axes.flat[len(birds) :]:
        axis.axis("off")

    fig.supxlabel(metric_axis_label(metric), fontsize=14)
    fig.supylabel("Delta BC (Post - Pre)", fontsize=14)
    fig.suptitle(
        "Within-bird association between phrase-duration change and BC change",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.975])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_separate_birds(
    data: pd.DataFrame,
    per_bird: pd.DataFrame,
    metric: str,
    out_dir: Path,
    dpi: int,
) -> None:
    ensure_dir(out_dir)
    stats_lookup = per_bird.set_index("animal_id")
    for bird_id, bird in data.groupby("animal_id", sort=True):
        x = bird["phrase_change"].to_numpy(float)
        y = bird["delta_bc"].to_numpy(float)
        fig, axis = plt.subplots(figsize=(5.2, 4.4))
        axis.scatter(x, y, s=55, alpha=0.9, edgecolors="white", linewidths=0.5)
        slope = linear_slope(x, y)
        if np.isfinite(slope) and np.unique(x).size >= 2:
            grid = np.linspace(np.min(x), np.max(x), 100)
            intercept = float(np.mean(y) - slope * np.mean(x))
            axis.plot(grid, intercept + slope * grid, linewidth=1.6)
        axis.axhline(0.0, linestyle="--", linewidth=0.9)
        axis.axvline(0.0, linestyle=":", linewidth=0.9)
        row = stats_lookup.loc[bird_id]
        if np.isfinite(row["spearman_rho"]):
            subtitle = (
                f"n={int(row['n_syllables'])}; Spearman rho="
                f"{row['spearman_rho']:.3f}; one-sided permutation p="
                f"{row['spearman_permutation_p_one_sided_negative']:.3g}"
            )
        else:
            subtitle = f"n={int(row['n_syllables'])}; association not estimable"
        axis.set_title(f"{bird_id}\n{subtitle}")
        axis.set_xlabel(metric_axis_label(metric))
        axis.set_ylabel("Delta BC (Post - Pre)")
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{safe_filename(str(bird_id))}_{metric}_vs_deltaBC.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_centered_all_birds(
    data: pd.DataFrame,
    overall: pd.DataFrame,
    metric: str,
    out_path: Path,
    dpi: int,
    show: bool,
) -> None:
    plot_data = data.copy()
    plot_data["phrase_change_centered"] = plot_data.groupby("animal_id")[
        "phrase_change"
    ].transform(lambda s: s - s.mean())
    plot_data["delta_bc_centered"] = plot_data.groupby("animal_id")[
        "delta_bc"
    ].transform(lambda s: s - s.mean())

    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for bird_id, bird in plot_data.groupby("animal_id", sort=True):
        axis.scatter(
            bird["phrase_change_centered"],
            bird["delta_bc_centered"],
            s=42,
            alpha=0.8,
            label=str(bird_id),
            edgecolors="white",
            linewidths=0.4,
        )

    slope = float(overall.iloc[0]["equal_bird_weighted_centered_slope"])
    x = plot_data["phrase_change_centered"].to_numpy(float)
    if np.isfinite(slope) and np.unique(x).size >= 2:
        grid = np.linspace(np.min(x), np.max(x), 100)
        axis.plot(grid, slope * grid, linewidth=2.0)

    axis.axhline(0.0, linestyle="--", linewidth=0.9)
    axis.axvline(0.0, linestyle=":", linewidth=0.9)
    row = overall.iloc[0]
    axis.set_title(
        "Bird-centered continuous association\n"
        f"equal-bird mean Spearman rho="
        f"{row['equal_bird_mean_spearman_rho']:.3f}, "
        f"permutation p="
        f"{row['rho_within_bird_permutation_p_one_sided_negative']:.3g}"
    )
    axis.set_xlabel(f"Within-bird centered {metric_axis_label(metric)}")
    axis.set_ylabel("Within-bird centered Delta BC")
    axis.legend(frameon=False, fontsize=8, ncol=2, title="Bird")
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bird_correlations(
    per_bird: pd.DataFrame,
    out_path: Path,
    dpi: int,
    show: bool,
) -> None:
    plot_data = per_bird.sort_values("spearman_rho").copy()
    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    positions = np.arange(len(plot_data))
    axis.scatter(positions, plot_data["spearman_rho"], s=55)
    axis.axhline(0.0, linestyle="--", linewidth=0.9)
    axis.set_xticks(positions)
    axis.set_xticklabels(plot_data["animal_id"], rotation=45, ha="right")
    axis.set_ylabel("Within-bird Spearman rho")
    axis.set_title("Phrase-duration change versus Delta BC by bird")
    axis.set_ylim(-1.05, 1.05)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def write_summary(
    path: Path,
    args: argparse.Namespace,
    merged: pd.DataFrame,
    per_bird_all: pd.DataFrame,
    overall_primary: pd.DataFrame,
    overall_sensitivity: Optional[pd.DataFrame],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Continuous phrase-duration change versus BC change\n")
        handle.write("================================================\n\n")
        handle.write(f"Variance input: {args.variance_csv}\n")
        handle.write(f"BC input: {args.bc_csv}\n")
        handle.write(f"Phrase metric: {args.phrase_metric}\n")
        handle.write("BC change: Post BC - Pre BC; negative values indicate reduced overlap\n")
        handle.write(f"Matched pooled M+L birds: {merged['animal_id'].nunique()}\n")
        handle.write(f"Matched pooled M+L syllables: {len(merged)}\n\n")

        handle.write("Primary across-bird analysis\n")
        handle.write("----------------------------\n")
        row = overall_primary.iloc[0]
        handle.write(
            f"Minimum syllables per bird: {int(row['minimum_syllables_per_bird'])}\n"
        )
        handle.write(f"Birds: {int(row['n_birds'])}\n")
        handle.write(f"Syllables: {int(row['n_syllables'])}\n")
        handle.write(
            f"Equal-bird mean within-bird Spearman rho: "
            f"{row['equal_bird_mean_spearman_rho']:.6g}\n"
        )
        handle.write(
            f"One-sided within-bird permutation p (negative association): "
            f"{row['rho_within_bird_permutation_p_one_sided_negative']:.6g}\n"
        )
        handle.write(
            f"Bird-cluster bootstrap 95% CI for mean rho: "
            f"[{row['rho_bird_cluster_bootstrap_ci_low']:.6g}, "
            f"{row['rho_bird_cluster_bootstrap_ci_high']:.6g}]\n"
        )
        handle.write(
            f"Equal-bird weighted centered slope: "
            f"{row['equal_bird_weighted_centered_slope']:.6g}\n"
        )
        handle.write(
            f"One-sided slope permutation p: "
            f"{row['slope_within_bird_permutation_p_one_sided_negative']:.6g}\n"
        )
        handle.write(
            f"Slope bootstrap 95% CI: "
            f"[{row['slope_bird_cluster_bootstrap_ci_low']:.6g}, "
            f"{row['slope_bird_cluster_bootstrap_ci_high']:.6g}]\n"
        )
        handle.write(f"Included birds: {row['included_birds']}\n")
        handle.write(
            f"Excluded for too few syllables: "
            f"{row['excluded_birds_and_counts'] or 'none'}\n\n"
        )

        if overall_sensitivity is not None:
            handle.write("Sensitivity including birds with at least two syllables\n")
            handle.write("----------------------------------------------------\n")
            row2 = overall_sensitivity.iloc[0]
            handle.write(f"Birds: {int(row2['n_birds'])}\n")
            handle.write(
                f"Equal-bird mean Spearman rho: "
                f"{row2['equal_bird_mean_spearman_rho']:.6g}\n"
            )
            handle.write(
                f"Permutation p: "
                f"{row2['rho_within_bird_permutation_p_one_sided_negative']:.6g}\n\n"
            )

        handle.write("Per-bird results\n")
        handle.write("----------------\n")
        for row in per_bird_all.sort_values("animal_id").itertuples(index=False):
            handle.write(
                f"{row.animal_id}: n={int(row.n_syllables)}, "
                f"rho={row.spearman_rho:.6g}, "
                f"one-sided permutation p="
                f"{row.spearman_permutation_p_one_sided_negative:.6g}, "
                f"slope={row.linear_slope_deltaBC_per_phrase_change_unit:.6g}\n"
            )
        handle.write("\nIndividual bird p-values are descriptive and BH q-values are in the CSV.\n")


def main() -> None:
    args = parse_args()
    if args.min_syllables_primary < 3:
        raise ValueError("--min-syllables-primary should be at least 3.")
    if args.n_permutations < 100:
        raise ValueError("--n-permutations must be at least 100.")
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100.")

    out_dir = ensure_dir(Path(args.out_dir).expanduser())
    variance = load_variance_table(args.variance_csv, args.phrase_metric)
    bc, identifier_report = load_bc_table(
        args.bc_csv,
        variance,
        args.bc_method,
        args.set_name,
    )
    identifier_report.to_csv(
        out_dir / "continuous_bc_identifier_match_report.csv", index=False
    )

    merged = bc.merge(
        variance,
        on=["animal_id", "syllable"],
        how="inner",
        suffixes=("_bc", "_variance"),
    )
    merged = (
        merged.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["phrase_change", "bc_pre", "bc_post", "delta_bc"])
        .sort_values(["animal_id", "syllable"])
        .drop_duplicates(["animal_id", "syllable"], keep="first")
        .reset_index(drop=True)
    )
    if merged.empty:
        raise ValueError("No pooled M+L animal-syllable pairs remained after merging.")

    counts = (
        merged.groupby("animal_id")
        .size()
        .rename("n_matched_syllables")
        .reset_index()
    )
    counts["eligible_primary_min_n"] = (
        counts["n_matched_syllables"] >= args.min_syllables_primary
    )
    counts["eligible_sensitivity_min_2"] = counts["n_matched_syllables"] >= 2
    counts.to_csv(out_dir / "continuous_bc_bird_eligibility.csv", index=False)
    merged.to_csv(out_dir / "continuous_bc_syllable_level.csv", index=False)

    print("[INFO] Matched pooled M+L syllables by bird:")
    print(counts.to_string(index=False))

    # Descriptive per-bird statistics for all birds with at least two syllables.
    descriptive_data = merged[
        merged["animal_id"].isin(
            counts.loc[
                counts["n_matched_syllables"] >= 2, "animal_id"
            ]
        )
    ].copy()
    per_bird_all = summarize_per_bird(
        descriptive_data,
        args.n_permutations,
        args.seed,
    )
    per_bird_all["included_in_primary"] = (
        per_bird_all["n_syllables"] >= args.min_syllables_primary
    )
    per_bird_all.to_csv(
        out_dir / "continuous_bc_per_bird_stats.csv", index=False
    )

    overall_primary, _ = run_overall_analysis(
        merged,
        args.min_syllables_primary,
        args.n_permutations,
        args.n_bootstrap,
        args.seed,
        label=f"primary_min_{args.min_syllables_primary}_syllables",
    )

    overall_sensitivity = None
    if (counts["n_matched_syllables"] == 2).any():
        overall_sensitivity, _ = run_overall_analysis(
            merged,
            2,
            args.n_permutations,
            args.n_bootstrap,
            args.seed + 10000,
            label="sensitivity_min_2_syllables",
        )

    overall_tables = [overall_primary]
    if overall_sensitivity is not None:
        overall_tables.append(overall_sensitivity)
    overall = pd.concat(overall_tables, ignore_index=True)
    overall.to_csv(out_dir / "continuous_bc_overall_stats.csv", index=False)

    primary_birds = set(
        overall_primary.iloc[0]["included_birds"].split(";")
    )
    primary_data = merged[merged["animal_id"].isin(primary_birds)].copy()
    primary_per_bird = per_bird_all[
        per_bird_all["animal_id"].isin(primary_birds)
    ].copy()

    metric_stub = safe_filename(args.phrase_metric)
    plot_per_bird_grid(
        descriptive_data,
        per_bird_all,
        args.phrase_metric,
        out_dir / f"continuous_bc_{metric_stub}_per_bird_grid.png",
        args.dpi,
        args.show,
    )
    plot_separate_birds(
        descriptive_data,
        per_bird_all,
        args.phrase_metric,
        out_dir / "per_bird_plots",
        args.dpi,
    )
    plot_centered_all_birds(
        primary_data,
        overall_primary,
        args.phrase_metric,
        out_dir / f"continuous_bc_{metric_stub}_bird_centered.png",
        args.dpi,
        args.show,
    )
    plot_bird_correlations(
        primary_per_bird,
        out_dir / f"continuous_bc_{metric_stub}_bird_rhos.png",
        args.dpi,
        args.show,
    )

    write_summary(
        out_dir / "continuous_bc_summary.txt",
        args,
        merged,
        per_bird_all,
        overall_primary,
        overall_sensitivity,
    )

    config = {
        "variance_csv": str(Path(args.variance_csv).expanduser()),
        "bc_csv": str(Path(args.bc_csv).expanduser()),
        "phrase_metric": args.phrase_metric,
        "bc_method": args.bc_method,
        "set_name": args.set_name,
        "min_syllables_primary": args.min_syllables_primary,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    with (out_dir / "continuous_bc_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"[OK] Wrote continuous BC analysis to: {out_dir}")
    for name in [
        "continuous_bc_summary.txt",
        "continuous_bc_overall_stats.csv",
        "continuous_bc_per_bird_stats.csv",
        "continuous_bc_syllable_level.csv",
        "continuous_bc_bird_eligibility.csv",
        "continuous_bc_identifier_match_report.csv",
        f"continuous_bc_{metric_stub}_per_bird_grid.png",
        f"continuous_bc_{metric_stub}_bird_centered.png",
        f"continuous_bc_{metric_stub}_bird_rhos.png",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
