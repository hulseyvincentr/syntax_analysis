#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan Bhattacharyya-coefficient (BC) pre/post results across syllable-percentage
cutoffs in pooled medial+lateral lesion birds.

For each requested cutoff and bird, the script:

1. Restricts the analysis to syllables with both phrase-duration metrics and
   valid Pre/Post BC values.
2. Ranks those BC-qualified syllables within bird.
3. Selects ceil(cutoff × n_eligible), capped so at least one syllable remains.
4. Calculates the median Pre BC, Post BC, and Post-minus-Pre BC separately for:
      - selected/high-ranked syllables
      - remaining syllables
5. Tests Pre versus Post BC across birds for each subset.
6. Directly tests whether ΔBC is more negative in selected than remaining
   syllables from the same birds.

The default ranking metric is post-lesion phrase-duration variance, matching
the original Figure 4D high-variance selection concept. The
log_variance_ratio option instead ranks syllables by proportional increase
from pre to post.

Because scanning cutoffs is exploratory, the script reports:
  - raw p-values
  - Holm and Benjamini-Hochberg corrections across UNIQUE selected sets
  - selection-equivalence groups, because multiple nominal cutoffs often
    select exactly the same syllables when birds have small repertoires

Use the cutoff scan to evaluate robustness, not to choose the cutoff with the
smallest p-value.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


POOLED_ML = "Complete and partial medial and lateral lesion"


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split()).lower()


def normalize_token(value: Any) -> str:
    """Normalize integer-like syllable labels, e.g. 3, 3.0, and '3.0'."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def first_present(
    columns: Iterable[str],
    candidates: list[str],
) -> Optional[str]:
    lookup = {norm_text(column): column for column in columns}
    for candidate in candidates:
        hit = lookup.get(norm_text(candidate))
        if hit is not None:
            return hit
    return None


def canonical_group(value: Any) -> str:
    text = norm_text(value)
    if not text:
        return "unknown"
    if "sham" in text or ("saline" in text and "lesion" not in text):
        return "sham saline injection"
    if "lateral lesion only" in text or "lateral only" in text or "single hit" in text:
        return "Lateral lesion only"
    if "complete and partial" in text and "medial" in text and "lateral" in text:
        return POOLED_ML
    if "complete" in text and "medial" in text and "lateral" in text:
        return POOLED_ML
    if "partial" in text and "medial" in text and "lateral" in text:
        return POOLED_ML
    if "area x not visible" in text or ("large" in text and "lesion" in text):
        return POOLED_ML
    if ("medial" in text and "lateral" in text) or "m+l" in text:
        return POOLED_ML
    return str(value)


def parse_cutoffs(text: str) -> list[float]:
    cutoffs: list[float] = []
    for token in re.split(r"[,;\s]+", text.strip()):
        if not token:
            continue
        value = float(token)
        if value > 1:
            value /= 100.0
        if not 0 < value < 1:
            raise ValueError(
                f"Cutoff {token!r} is invalid. Use fractions such as 0.15 "
                "or percentages such as 15."
            )
        cutoffs.append(value)
    if not cutoffs:
        raise ValueError("No valid cutoffs were supplied.")
    return sorted(set(cutoffs))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    result = np.full_like(p_values, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if len(finite) == 0:
        return result

    values = p_values[finite]
    order = np.argsort(values)
    ordered = values[order]
    adjusted_ordered = np.maximum.accumulate(
        (len(ordered) - np.arange(len(ordered))) * ordered
    )
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)

    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_ordered
    result[finite] = adjusted
    return result


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    result = np.full_like(p_values, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if len(finite) == 0:
        return result

    values = p_values[finite]
    order = np.argsort(values)
    ordered = values[order]
    m = len(ordered)
    adjusted_ordered = ordered * m / np.arange(1, m + 1)
    adjusted_ordered = np.minimum.accumulate(adjusted_ordered[::-1])[::-1]
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)

    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_ordered
    result[finite] = adjusted
    return result


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan selected-versus-remaining Pre/Post BC results across "
            "syllable-percentage cutoffs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variance-csv",
        required=True,
        help="Full animal-by-syllable phrase-duration metrics CSV.",
    )
    parser.add_argument(
        "--bc-csv",
        required=True,
        help="bc_batch_cluster_level_long.csv.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--cutoffs",
        default="10,15,20,25,30,35,40,50",
        help=(
            "Comma- or space-separated cutoffs. Values may be percentages "
            "(15) or fractions (0.15)."
        ),
    )
    parser.add_argument(
        "--rank-metric",
        choices=[
            "post_variance",
            "pooled_variance",
            "delta_variance",
            "log_variance_ratio",
            "delta_cv",
            "log_cv_ratio",
        ],
        default="post_variance",
        help=(
            "Metric used to rank syllables within each bird. post_variance "
            "most closely matches the original Figure 4D selection."
        ),
    )
    parser.add_argument(
        "--bc-method",
        default="selected_bins",
        help="BC method to retain.",
    )
    parser.add_argument(
        "--set-name",
        default="all_clusters",
        help="BC set to retain.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Plot resolution.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_variance_table(path: str, rank_metric: str) -> pd.DataFrame:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    table = pd.read_csv(csv_path)

    animal_col = first_present(
        table.columns,
        ["animal_id", "Animal ID", "bird_id", "bird"],
    )
    syllable_col = first_present(
        table.columns,
        ["syllable", "Syllable", "label", "cluster_id"],
    )
    pre_var_col = first_present(
        table.columns,
        ["pre_variance_s2", "pre_variance", "pre_variance_ms2"],
    )
    post_var_col = first_present(
        table.columns,
        ["post_variance_s2", "post_variance", "post_variance_ms2"],
    )
    pooled_var_col = first_present(
        table.columns,
        [
            "pooled_variance_s2",
            "pooled_variance",
            "mean_pre_post_variance_s2",
            "mean_pre_post_variance",
            "combined_variance_s2",
        ],
    )
    pre_cv_col = first_present(table.columns, ["pre_cv", "cv_pre"])
    post_cv_col = first_present(table.columns, ["post_cv", "cv_post"])

    if animal_col is None or syllable_col is None:
        raise ValueError(
            "Variance CSV must contain animal and syllable identifiers. "
            f"Found: {list(table.columns)}"
        )

    out = pd.DataFrame(
        {
            "animal_id": table[animal_col].astype(str).str.strip(),
            "syllable": table[syllable_col].map(normalize_token),
        }
    )

    if pre_var_col is not None:
        out["pre_variance"] = pd.to_numeric(
            table[pre_var_col], errors="coerce"
        )
    if post_var_col is not None:
        out["post_variance"] = pd.to_numeric(
            table[post_var_col], errors="coerce"
        )
    if pooled_var_col is not None:
        out["pooled_variance"] = pd.to_numeric(
            table[pooled_var_col], errors="coerce"
        )
    elif pre_var_col is not None and post_var_col is not None:
        # The updated Figure 3 pooled ranking averaged pre and post variances.
        out["pooled_variance"] = (
            out["pre_variance"] + out["post_variance"]
        ) / 2.0

    if pre_cv_col is not None:
        out["pre_cv"] = pd.to_numeric(table[pre_cv_col], errors="coerce")
    if post_cv_col is not None:
        out["post_cv"] = pd.to_numeric(table[post_cv_col], errors="coerce")

    if "pre_variance" in out and "post_variance" in out:
        out["delta_variance"] = (
            out["post_variance"] - out["pre_variance"]
        )
        valid = (out["pre_variance"] > 0) & (out["post_variance"] > 0)
        out["log_variance_ratio"] = np.nan
        out.loc[valid, "log_variance_ratio"] = np.log(
            out.loc[valid, "post_variance"]
            / out.loc[valid, "pre_variance"]
        )

    if "pre_cv" in out and "post_cv" in out:
        out["delta_cv"] = out["post_cv"] - out["pre_cv"]
        valid = (out["pre_cv"] > 0) & (out["post_cv"] > 0)
        out["log_cv_ratio"] = np.nan
        out.loc[valid, "log_cv_ratio"] = np.log(
            out.loc[valid, "post_cv"] / out.loc[valid, "pre_cv"]
        )

    if rank_metric not in out.columns:
        raise ValueError(
            f"Cannot calculate rank metric {rank_metric!r}. "
            f"Available derived columns: {list(out.columns)}. "
            f"Original columns: {list(table.columns)}"
        )

    out["rank_metric"] = pd.to_numeric(
        out[rank_metric], errors="coerce"
    )
    out["rank_metric_name"] = rank_metric

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["animal_id", "syllable", "rank_metric"])
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
    print(f"[INFO] Ranking metric: {rank_metric}")
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
    candidates = [column for column in candidates if column in table.columns]
    if not candidates:
        raise ValueError(
            "No BC cluster/syllable identifier column was found."
        )

    variance_keys = variance[["animal_id", "syllable"]].drop_duplicates()
    report_rows = []

    for column in candidates:
        candidate_keys = pd.DataFrame(
            {
                "animal_id": table[animal_col].astype(str).str.strip(),
                "syllable": table[column].map(normalize_token),
            }
        )
        candidate_keys = candidate_keys[
            (candidate_keys["animal_id"] != "")
            & (candidate_keys["syllable"] != "")
        ].drop_duplicates()

        matches = variance_keys.merge(
            candidate_keys,
            on=["animal_id", "syllable"],
            how="inner",
        )
        report_rows.append(
            {
                "candidate_column": column,
                "matched_variance_pairs": len(matches),
                "matched_birds": matches["animal_id"].nunique(),
                "available_bc_pairs": len(candidate_keys),
            }
        )

    report = pd.DataFrame(report_rows).sort_values(
        ["matched_variance_pairs", "matched_birds"],
        ascending=False,
        kind="stable",
    )
    for row in report.itertuples(index=False):
        print(
            f"[INFO] BC ID candidate {row.candidate_column!r}: "
            f"{row.matched_variance_pairs} matched pairs across "
            f"{row.matched_birds} birds"
        )

    best = report.iloc[0]
    if int(best["matched_variance_pairs"]) == 0:
        raise ValueError(
            "None of the BC identifier columns matched the variance table."
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
                f"Available: "
                f"{sorted(table['bc_method'].dropna().astype(str).unique())}"
            )
        table = table[mask].copy()
        print(f"[INFO] Retained bc_method={bc_method!r}: {len(table)} rows")

    if "set_name" in table.columns:
        mask = table["set_name"].astype(str) == set_name
        if not mask.any():
            raise ValueError(
                f"No BC rows have set_name={set_name!r}. "
                f"Available: "
                f"{sorted(table['set_name'].dropna().astype(str).unique())}"
            )
        table = table[mask].copy()
        print(f"[INFO] Retained set_name={set_name!r}: {len(table)} rows")

    animal_col = first_present(
        table.columns,
        ["animal_id", "Animal ID", "bird_id", "bird"],
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
    pre_col = first_present(table.columns, ["bc_pre", "pre_bc", "BC_pre"])
    post_col = first_present(
        table.columns, ["bc_post", "post_bc", "BC_post"]
    )

    if animal_col is None or group_col is None:
        raise ValueError(
            "BC CSV must contain animal and lesion-group columns."
        )
    if pre_col is None or post_col is None:
        raise ValueError("BC CSV must contain Pre and Post BC columns.")

    identifier_col, identifier_report = choose_bc_identifier(
        table, animal_col, variance
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
    out["delta_bc"] = out["bc_post"] - out["bc_pre"]
    out["lesion_group"] = out["lesion_group_raw"].map(canonical_group)
    out["bc_identifier_column"] = identifier_col
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["bc_pre", "bc_post", "delta_bc"])
    out = out[
        (out["animal_id"] != "")
        & (out["syllable"] != "")
        & (out["lesion_group"] == POOLED_ML)
    ]
    out = (
        out.sort_values(["animal_id", "syllable"])
        .drop_duplicates(["animal_id", "syllable"], keep="first")
        .reset_index(drop=True)
    )

    print(
        f"[INFO] BC table after filters: {len(out)} pooled M+L pairs "
        f"across {out['animal_id'].nunique()} birds"
    )
    return out, identifier_report


# ---------------------------------------------------------------------------
# Selection and statistics
# ---------------------------------------------------------------------------

def select_at_cutoff(
    merged: pd.DataFrame,
    cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    audit_rows = []

    for animal_id, bird in merged.groupby("animal_id", sort=True):
        bird = bird.copy()
        bird["_numeric_syllable"] = pd.to_numeric(
            bird["syllable"], errors="coerce"
        )
        bird["_numeric_missing"] = bird["_numeric_syllable"].isna()

        # Reproducible descending rank. Syllable label is only a tie-breaker.
        bird = bird.sort_values(
            [
                "rank_metric",
                "_numeric_missing",
                "_numeric_syllable",
                "syllable",
            ],
            ascending=[False, True, True, True],
            kind="stable",
        ).reset_index(drop=True)

        n_eligible = len(bird)
        if n_eligible < 2:
            audit_rows.append(
                {
                    "cutoff_fraction": cutoff,
                    "cutoff_percent": cutoff * 100,
                    "animal_id": animal_id,
                    "n_eligible": n_eligible,
                    "n_selected": 0,
                    "n_remaining": n_eligible,
                    "realized_fraction": 0.0,
                    "status": "excluded_fewer_than_2_syllables",
                    "boundary_tie": False,
                }
            )
            continue

        n_selected = max(1, int(math.ceil(cutoff * n_eligible)))
        n_selected = min(n_selected, n_eligible - 1)

        bird["selection_rank"] = np.arange(1, n_eligible + 1)
        bird["subset"] = np.where(
            bird["selection_rank"] <= n_selected,
            "selected",
            "remaining",
        )
        bird["cutoff_fraction"] = cutoff
        bird["cutoff_percent"] = cutoff * 100
        bird["n_eligible"] = n_eligible
        bird["n_selected"] = n_selected
        bird["realized_fraction"] = n_selected / n_eligible

        boundary_tie = False
        if n_selected < n_eligible:
            selected_boundary = bird.loc[
                n_selected - 1, "rank_metric"
            ]
            remaining_boundary = bird.loc[n_selected, "rank_metric"]
            boundary_tie = bool(
                np.isclose(
                    selected_boundary,
                    remaining_boundary,
                    rtol=1e-12,
                    atol=1e-15,
                )
            )

        selected_tokens = bird.loc[
            bird["subset"] == "selected", "syllable"
        ].astype(str).tolist()

        audit_rows.append(
            {
                "cutoff_fraction": cutoff,
                "cutoff_percent": cutoff * 100,
                "animal_id": animal_id,
                "n_eligible": n_eligible,
                "n_selected": n_selected,
                "n_remaining": n_eligible - n_selected,
                "realized_fraction": n_selected / n_eligible,
                "status": "included",
                "boundary_tie": boundary_tie,
                "selected_syllables": ";".join(selected_tokens),
            }
        )

        pieces.append(
            bird.drop(
                columns=["_numeric_syllable", "_numeric_missing"]
            )
        )

    selected = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame()
    )
    audit = pd.DataFrame(audit_rows)
    return selected, audit


def exact_signflip(
    values: np.ndarray,
    alternative: str = "two-sided",
) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan

    observed = float(np.mean(values))
    null = np.fromiter(
        (
            np.mean(values * np.asarray(signs, dtype=float))
            for signs in itertools.product([-1.0, 1.0], repeat=n)
        ),
        dtype=float,
        count=2**n,
    )

    tolerance = 1e-15
    if alternative == "less":
        return float(np.mean(null <= observed + tolerance))
    if alternative == "greater":
        return float(np.mean(null >= observed - tolerance))
    return float(
        np.mean(np.abs(null) >= abs(observed) - tolerance)
    )


def paired_wilcoxon(
    post: np.ndarray,
    pre: np.ndarray,
    alternative: str,
) -> float:
    post = np.asarray(post, dtype=float)
    pre = np.asarray(pre, dtype=float)
    mask = np.isfinite(post) & np.isfinite(pre)
    post = post[mask]
    pre = pre[mask]
    if len(post) == 0:
        return np.nan

    difference = post - pre
    if np.allclose(difference, 0):
        return 1.0

    result = stats.wilcoxon(
        post,
        pre,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return float(result.pvalue)


def one_sample_wilcoxon(
    values: np.ndarray,
    alternative: str,
) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if np.allclose(values, 0):
        return 1.0
    result = stats.wilcoxon(
        values,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return float(result.pvalue)


def summarize_cutoff(
    selected_rows: pd.DataFrame,
    cutoff: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    bird_level = (
        selected_rows.groupby(
            ["animal_id", "subset"],
            as_index=False,
        )
        .agg(
            n_syllables=("syllable", "nunique"),
            median_pre_bc=("bc_pre", "median"),
            median_post_bc=("bc_post", "median"),
            median_delta_bc=("delta_bc", "median"),
            n_eligible=("n_eligible", "first"),
            n_selected=("n_selected", "first"),
            realized_fraction=("realized_fraction", "first"),
        )
    )

    wide = bird_level.pivot(
        index="animal_id",
        columns="subset",
        values=[
            "n_syllables",
            "median_pre_bc",
            "median_post_bc",
            "median_delta_bc",
            "n_eligible",
            "n_selected",
            "realized_fraction",
        ],
    )
    wide.columns = [
        f"{metric}_{subset}" for metric, subset in wide.columns
    ]
    wide = wide.reset_index()

    required = [
        "median_pre_bc_selected",
        "median_post_bc_selected",
        "median_pre_bc_remaining",
        "median_post_bc_remaining",
    ]
    for column in required:
        if column not in wide.columns:
            wide[column] = np.nan

    wide = wide.dropna(subset=required).copy()
    wide["delta_bc_selected"] = (
        wide["median_post_bc_selected"]
        - wide["median_pre_bc_selected"]
    )
    wide["delta_bc_remaining"] = (
        wide["median_post_bc_remaining"]
        - wide["median_pre_bc_remaining"]
    )
    wide["delta_bc_selected_minus_remaining"] = (
        wide["delta_bc_selected"] - wide["delta_bc_remaining"]
    )
    wide["cutoff_fraction"] = cutoff
    wide["cutoff_percent"] = cutoff * 100

    selected_delta = wide["delta_bc_selected"].to_numpy(float)
    remaining_delta = wide["delta_bc_remaining"].to_numpy(float)
    contrast = wide[
        "delta_bc_selected_minus_remaining"
    ].to_numpy(float)

    result = {
        "cutoff_fraction": cutoff,
        "cutoff_percent": cutoff * 100,
        "n_birds": len(wide),
        "n_syllables_total": int(len(selected_rows)),
        "n_selected_total": int(
            (selected_rows["subset"] == "selected").sum()
        ),
        "n_remaining_total": int(
            (selected_rows["subset"] == "remaining").sum()
        ),
        "min_selected_per_bird": int(
            wide["n_syllables_selected"].min()
        ),
        "max_selected_per_bird": int(
            wide["n_syllables_selected"].max()
        ),
        "median_realized_fraction": float(
            wide["realized_fraction_selected"].median()
        ),

        "selected_median_pre_bc": float(
            np.median(wide["median_pre_bc_selected"])
        ),
        "selected_median_post_bc": float(
            np.median(wide["median_post_bc_selected"])
        ),
        "selected_median_paired_delta_bc": float(
            np.median(selected_delta)
        ),
        "selected_mean_paired_delta_bc": float(
            np.mean(selected_delta)
        ),
        "selected_wilcoxon_p_two_sided": paired_wilcoxon(
            wide["median_post_bc_selected"],
            wide["median_pre_bc_selected"],
            "two-sided",
        ),
        "selected_wilcoxon_p_one_sided_post_less_pre": paired_wilcoxon(
            wide["median_post_bc_selected"],
            wide["median_pre_bc_selected"],
            "less",
        ),
        "selected_signflip_p_two_sided": exact_signflip(
            selected_delta, "two-sided"
        ),
        "selected_signflip_p_one_sided_negative": exact_signflip(
            selected_delta, "less"
        ),

        "remaining_median_pre_bc": float(
            np.median(wide["median_pre_bc_remaining"])
        ),
        "remaining_median_post_bc": float(
            np.median(wide["median_post_bc_remaining"])
        ),
        "remaining_median_paired_delta_bc": float(
            np.median(remaining_delta)
        ),
        "remaining_mean_paired_delta_bc": float(
            np.mean(remaining_delta)
        ),
        "remaining_wilcoxon_p_two_sided": paired_wilcoxon(
            wide["median_post_bc_remaining"],
            wide["median_pre_bc_remaining"],
            "two-sided",
        ),
        "remaining_wilcoxon_p_one_sided_post_less_pre": paired_wilcoxon(
            wide["median_post_bc_remaining"],
            wide["median_pre_bc_remaining"],
            "less",
        ),
        "remaining_signflip_p_two_sided": exact_signflip(
            remaining_delta, "two-sided"
        ),
        "remaining_signflip_p_one_sided_negative": exact_signflip(
            remaining_delta, "less"
        ),

        "contrast_median_selected_minus_remaining_delta_bc": float(
            np.median(contrast)
        ),
        "contrast_mean_selected_minus_remaining_delta_bc": float(
            np.mean(contrast)
        ),
        "contrast_wilcoxon_p_two_sided": one_sample_wilcoxon(
            contrast, "two-sided"
        ),
        "contrast_wilcoxon_p_one_sided_selected_more_negative": (
            one_sample_wilcoxon(contrast, "less")
        ),
        "contrast_signflip_p_two_sided": exact_signflip(
            contrast, "two-sided"
        ),
        "contrast_signflip_p_one_sided_selected_more_negative": (
            exact_signflip(contrast, "less")
        ),
    }

    return result, wide


def membership_signature(selected_rows: pd.DataFrame) -> str:
    tokens = []
    for animal_id, bird in selected_rows.groupby("animal_id", sort=True):
        syllables = sorted(
            bird.loc[
                bird["subset"] == "selected", "syllable"
            ].astype(str)
        )
        tokens.append(f"{animal_id}:{','.join(syllables)}")
    signature = "|".join(tokens)
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]


def add_unique_selection_adjustments(
    results: pd.DataFrame,
) -> pd.DataFrame:
    unique = (
        results.sort_values("cutoff_fraction")
        .drop_duplicates("selection_signature", keep="first")
        .copy()
    )

    families = {
        "selected_wilcoxon_p_two_sided": "selected",
        "remaining_wilcoxon_p_two_sided": "remaining",
        "contrast_wilcoxon_p_two_sided": "contrast",
    }

    for p_column, prefix in families.items():
        unique[f"{prefix}_holm_p_unique_selections"] = holm_adjust(
            unique[p_column].to_numpy(float)
        )
        unique[f"{prefix}_bh_q_unique_selections"] = bh_adjust(
            unique[p_column].to_numpy(float)
        )

    keep = ["selection_signature"]
    for prefix in families.values():
        keep.extend(
            [
                f"{prefix}_holm_p_unique_selections",
                f"{prefix}_bh_q_unique_selections",
            ]
        )

    return results.merge(
        unique[keep],
        on="selection_signature",
        how="left",
    )


# ---------------------------------------------------------------------------
# Plots and summary
# ---------------------------------------------------------------------------

def make_pvalue_plot(
    results: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))

    axis.plot(
        results["cutoff_percent"],
        results["selected_wilcoxon_p_two_sided"],
        marker="o",
        label="Selected: Pre vs Post",
    )
    axis.plot(
        results["cutoff_percent"],
        results["remaining_wilcoxon_p_two_sided"],
        marker="s",
        label="Remaining: Pre vs Post",
    )
    axis.plot(
        results["cutoff_percent"],
        results["contrast_wilcoxon_p_two_sided"],
        marker="^",
        label="Selected vs remaining ΔBC",
    )
    axis.axhline(0.05, linestyle="--", linewidth=1)
    axis.set_xlabel("Target percentage selected within bird")
    axis.set_ylabel("Raw two-sided Wilcoxon p-value")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def make_effect_plot(
    results: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))

    axis.plot(
        results["cutoff_percent"],
        results["selected_median_paired_delta_bc"],
        marker="o",
        label="Selected",
    )
    axis.plot(
        results["cutoff_percent"],
        results["remaining_median_paired_delta_bc"],
        marker="s",
        label="Remaining",
    )
    axis.axhline(0, linestyle="--", linewidth=1)
    axis.set_xlabel("Target percentage selected within bird")
    axis.set_ylabel("Median bird-level ΔBC (Post − Pre)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    merged: pd.DataFrame,
    results: pd.DataFrame,
    equivalence: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("BC percentage-cutoff scan\n")
        handle.write("=========================\n\n")
        handle.write(f"Variance input: {args.variance_csv}\n")
        handle.write(f"BC input: {args.bc_csv}\n")
        handle.write(f"Ranking metric: {args.rank_metric}\n")
        handle.write(
            "Selection rule: ceil(cutoff × BC-qualified syllables), "
            "capped to retain at least one remaining syllable\n"
        )
        handle.write(
            f"Matched M+L birds: {merged['animal_id'].nunique()}\n"
        )
        handle.write(f"Matched M+L syllables: {len(merged)}\n")
        handle.write(
            f"Unique selected sets across requested cutoffs: "
            f"{results['selection_signature'].nunique()}\n\n"
        )

        handle.write(
            "Raw two-sided paired Wilcoxon p-values are shown below. "
            "Adjusted values in the CSV are calculated across unique "
            "selected sets, not duplicated cutoff labels.\n\n"
        )

        display_columns = [
            "cutoff_percent",
            "n_selected_total",
            "min_selected_per_bird",
            "max_selected_per_bird",
            "selected_median_paired_delta_bc",
            "selected_wilcoxon_p_two_sided",
            "selected_holm_p_unique_selections",
            "remaining_median_paired_delta_bc",
            "remaining_wilcoxon_p_two_sided",
            "remaining_holm_p_unique_selections",
            "contrast_median_selected_minus_remaining_delta_bc",
            "contrast_wilcoxon_p_two_sided",
            "contrast_holm_p_unique_selections",
            "selection_signature",
        ]
        handle.write(
            results[display_columns].to_string(index=False)
        )
        handle.write("\n\nEquivalent cutoff groups\n")
        handle.write("------------------------\n")
        handle.write(equivalence.to_string(index=False))
        handle.write(
            "\n\nInterpretation caution\n"
            "----------------------\n"
            "This is a sensitivity analysis. Do not define the final cutoff "
            "by selecting the smallest observed p-value. Emphasize whether "
            "effect direction and magnitude are stable across a prespecified "
            "range, and use the direct selected-versus-remaining ΔBC "
            "contrast to evaluate specificity.\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cutoffs = parse_cutoffs(args.cutoffs)

    output_dir = Path(args.out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    variance = load_variance_table(
        args.variance_csv,
        args.rank_metric,
    )
    bc, identifier_report = load_bc_table(
        args.bc_csv,
        variance,
        args.bc_method,
        args.set_name,
    )
    identifier_report.to_csv(
        output_dir / "bc_cutoff_scan_identifier_report.csv",
        index=False,
    )

    merged = bc.merge(
        variance,
        on=["animal_id", "syllable"],
        how="inner",
        suffixes=("_bc", "_variance"),
    )
    merged = (
        merged.sort_values(["animal_id", "syllable"])
        .drop_duplicates(["animal_id", "syllable"], keep="first")
        .reset_index(drop=True)
    )

    print(
        f"[INFO] Matched analysis pool: {len(merged)} syllables across "
        f"{merged['animal_id'].nunique()} pooled M+L birds"
    )

    all_results = []
    all_bird_rows = []
    all_membership = []
    all_audits = []

    for cutoff in cutoffs:
        selected_rows, audit = select_at_cutoff(merged, cutoff)
        if selected_rows.empty:
            print(
                f"[WARN] No birds were eligible at cutoff {cutoff:.3f}"
            )
            continue

        result, bird_level = summarize_cutoff(
            selected_rows, cutoff
        )
        signature = membership_signature(selected_rows)
        result["selection_signature"] = signature

        selected_rows["selection_signature"] = signature
        bird_level["selection_signature"] = signature
        audit["selection_signature"] = signature

        all_results.append(result)
        all_bird_rows.append(bird_level)
        all_membership.append(selected_rows)
        all_audits.append(audit)

        print(
            f"[INFO] {cutoff * 100:g}%: "
            f"selected p={result['selected_wilcoxon_p_two_sided']:.6g}, "
            f"remaining p={result['remaining_wilcoxon_p_two_sided']:.6g}, "
            f"direct contrast p="
            f"{result['contrast_wilcoxon_p_two_sided']:.6g}, "
            f"signature={signature}"
        )

    if not all_results:
        raise RuntimeError("No cutoff produced an analyzable result.")

    results = pd.DataFrame(all_results).sort_values(
        "cutoff_fraction"
    ).reset_index(drop=True)
    results = add_unique_selection_adjustments(results)

    bird_rows = pd.concat(all_bird_rows, ignore_index=True)
    membership = pd.concat(all_membership, ignore_index=True)
    audit = pd.concat(all_audits, ignore_index=True)

    equivalence = (
        results.groupby("selection_signature", as_index=False)
        .agg(
            cutoffs_percent=(
                "cutoff_percent",
                lambda values: ";".join(
                    f"{value:g}" for value in values
                ),
            ),
            n_equivalent_cutoffs=("cutoff_percent", "size"),
            first_cutoff_percent=("cutoff_percent", "min"),
            last_cutoff_percent=("cutoff_percent", "max"),
        )
        .sort_values("first_cutoff_percent")
        .reset_index(drop=True)
    )

    results.to_csv(
        output_dir / "bc_cutoff_scan_results.csv",
        index=False,
    )
    bird_rows.to_csv(
        output_dir / "bc_cutoff_scan_bird_level.csv",
        index=False,
    )
    membership.to_csv(
        output_dir / "bc_cutoff_scan_membership.csv",
        index=False,
    )
    audit.to_csv(
        output_dir / "bc_cutoff_scan_selection_audit.csv",
        index=False,
    )
    equivalence.to_csv(
        output_dir / "bc_cutoff_scan_selection_equivalence.csv",
        index=False,
    )
    merged.to_csv(
        output_dir / "bc_cutoff_scan_matched_input.csv",
        index=False,
    )

    make_pvalue_plot(
        results,
        output_dir / "bc_cutoff_scan_pvalues.png",
        args.dpi,
    )
    make_effect_plot(
        results,
        output_dir / "bc_cutoff_scan_effect_sizes.png",
        args.dpi,
    )
    write_summary(
        output_dir / "bc_cutoff_scan_summary.txt",
        args,
        merged,
        results,
        equivalence,
    )

    print(f"[OK] Wrote cutoff scan to: {output_dir}")
    for name in [
        "bc_cutoff_scan_summary.txt",
        "bc_cutoff_scan_results.csv",
        "bc_cutoff_scan_bird_level.csv",
        "bc_cutoff_scan_membership.csv",
        "bc_cutoff_scan_selection_audit.csv",
        "bc_cutoff_scan_selection_equivalence.csv",
        "bc_cutoff_scan_identifier_report.csv",
        "bc_cutoff_scan_pvalues.png",
        "bc_cutoff_scan_effect_sizes.png",
    ]:
        print(f"  {output_dir / name}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
