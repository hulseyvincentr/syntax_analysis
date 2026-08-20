#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 Panel C + Panel D
==========================

Panel C
-------
Bird-level Pre vs Post Bhattacharyya coefficient (BC) by lesion-hit type,
using ALL qualifying BC clusters for each bird.

Groups:
    - sham saline injection
    - lateral lesion only
    - pooled medial+lateral lesions

Within each bird and lesion group, the script takes the median Pre BC and
median Post BC across all valid BC clusters.

Panel D
-------
Bird-level Pre vs Post BC within pooled medial+lateral birds only, split by
Figure 3's Q75 delta-CV selection:
    - >= Q75 ΔCV syllables
    - < Q75 ΔCV syllables

The Panel D split is based on the exact Figure 3 qualifying-syllable table
(e.g. syllable_level_quantile_selection.csv), so the <Q75 subset contains only
syllables that were part of the Figure 3 eligible set.

Statistical annotations
-----------------------
Significance brackets in Panels C and D are based on Holm-adjusted,
two-sided paired Wilcoxon signed-rank p-values. Panel C applies Holm
correction across the three within-group Pre-vs-Post tests (sham,
lateral-only, pooled medial+lateral). Panel D applies Holm correction
across the two within-subset Pre-vs-Post tests (>=Q75 and <Q75). Raw
and adjusted p-values are both retained in the exported test CSVs.

Outputs
-------
This script writes:
    - figure4_panelC_all_qualifying_by_lesion_group.png
    - figure4_panelC_all_qualifying_by_lesion_group_boxesonly.png
    - figure4_panelD_pooled_ML_Q75_split.png
    - figure4_panelD_pooled_ML_Q75_split_boxesonly.png
    - figure4_panelCD_combined.png
    - figure4_panelCD_combined_boxesonly.png
    - panelC_all_qualifying_by_lesion_group_by_bird.csv
    - panelC_all_qualifying_by_lesion_group_tests.csv
    - panelD_pooled_ML_Q75_by_bird.csv
    - panelD_pooled_ML_Q75_tests.csv
    - bc_identifier_column_match_report.csv
    - panelD_bc_cluster_rows.csv
    - panelD_bc_merge_summary_by_bird.csv

Figure 3 palette:
    sham         #1B9E77
    lateral-only #A88BD9
    pooled M+L   #7A4FB7
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
COMPLETE_ML = "Complete Medial and Lateral lesion"
PARTIAL_ML = "Partial Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"
UNKNOWN = "unknown"

COLORS = {
    SHAM: "#1B9E77",
    LATERAL: "#A88BD9",
    POOLED_ML: "#7A4FB7",
}

GROUP_ORDER = [SHAM, LATERAL, POOLED_ML]
GROUP_DISPLAY = {
    SHAM: "sham saline\ninjection",
    LATERAL: "Lateral lesion\nonly",
    POOLED_ML: "Medial and\nLateral lesion",
}

FIGSIZE_C = (10.8, 5.3)
FIGSIZE_D = (8.3, 5.3)
FIGSIZE_COMBINED = (10.8, 10.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc-csv", required=True, help="Cluster-level BC CSV.")
    p.add_argument(
        "--quantile-selection-csv",
        required=True,
        help="Figure 3 syllable_level_quantile_selection.csv",
    )
    p.add_argument(
        "--selection-column",
        default="selected_Q75",
        help="Boolean Q-threshold selection column in the quantile-selection CSV.",
    )
    p.add_argument(
        "--metadata-excel",
        default=None,
        help="Optional lesion metadata workbook for lesion hit type mapping.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def norm_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).strip().split()).lower()


def normalize_syllable_token(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", s):
        try:
            return str(int(float(s)))
        except ValueError:
            pass
    return s


def first_present(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lookup = {norm_text(c): c for c in columns}
    for candidate in candidates:
        hit = lookup.get(norm_text(candidate))
        if hit is not None:
            return hit
    return None


def canonical_group(raw: Any) -> str:
    s = norm_text(raw)
    if not s:
        return UNKNOWN
    if "sham" in s or ("saline" in s and "lesion" not in s):
        return SHAM
    if "lateral lesion only" in s or "lateral only" in s or "single hit" in s:
        return LATERAL
    if "complete and partial" in s and "medial" in s and "lateral" in s:
        return POOLED_ML
    if "complete" in s and "medial" in s and "lateral" in s:
        return COMPLETE_ML
    if "partial" in s and "medial" in s and "lateral" in s:
        return PARTIAL_ML
    if "area x not visible" in s or ("large" in s and "lesion" in s):
        return COMPLETE_ML
    if ("medial" in s and "lateral" in s) or "m+l" in s:
        return PARTIAL_ML
    return str(raw)


def pooled_group(group: str) -> str:
    return POOLED_ML if group in {COMPLETE_ML, PARTIAL_ML, POOLED_ML} else group


def read_metadata_map(path: Optional[str]) -> dict[str, str]:
    if not path:
        return {}
    workbook = Path(path)
    if not workbook.exists():
        raise FileNotFoundError(workbook)

    animal_candidates = ["Animal ID", "animal_id", "bird_id", "bird"]
    hit_candidates = [
        "Lesion hit type",
        "lesion_hit_type",
        "Hit type",
        "hit_type",
        "raw_lesion_hit_type",
    ]
    xls = pd.ExcelFile(workbook)
    for sheet in xls.sheet_names:
        table = pd.read_excel(workbook, sheet_name=sheet)
        animal_col = first_present(table.columns, animal_candidates)
        hit_col = first_present(table.columns, hit_candidates)
        if animal_col and hit_col:
            return {
                str(animal).strip(): str(hit).strip()
                for animal, hit in zip(table[animal_col], table[hit_col])
                if pd.notna(animal) and pd.notna(hit)
            }
    raise ValueError(f"Could not find animal and lesion-hit-type columns in {workbook}")


def read_quantile_selection(path: str, selection_column: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    table = pd.read_csv(csv_path)
    animal_col = first_present(table.columns, ["animal_id", "Animal ID", "bird_id", "bird"])
    syllable_col = first_present(
        table.columns,
        [
            "syllable",
            "Syllable",
            "label",
            "cluster",
            "cluster_id",
            "cluster_token",
            "cluster_label",
            "syllable_cluster",
        ],
    )
    flag_col = first_present(table.columns, [selection_column])
    if animal_col is None or syllable_col is None or flag_col is None:
        raise ValueError(
            "Figure 3 quantile-selection CSV must contain animal, syllable/cluster, "
            f"and {selection_column!r} columns. Found: {list(table.columns)}"
        )

    out = table[[animal_col, syllable_col, flag_col]].copy()
    out.columns = ["animal_id", "syllable", "selection_flag_raw"]
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out["syllable"] = out["syllable"].map(normalize_syllable_token)

    raw = out["selection_flag_raw"]
    if pd.api.types.is_bool_dtype(raw):
        parsed = raw.astype(bool)
    else:
        normalized = raw.astype(str).str.strip().str.lower()
        true_values = {"true", "1", "yes", "y", "t"}
        false_values = {"false", "0", "no", "n", "f"}
        bad = ~normalized.isin(true_values | false_values)
        if bad.any():
            examples = raw[bad].drop_duplicates().head(12).tolist()
            raise ValueError(
                f"Could not parse {selection_column!r} as boolean. "
                f"Unrecognized values include: {examples}"
            )
        parsed = normalized.isin(true_values)

    out["is_q75"] = parsed.to_numpy(dtype=bool)
    out = out[(out["animal_id"] != "") & (out["syllable"] != "")].copy()

    conflicts = (
        out.groupby(["animal_id", "syllable"])["is_q75"]
        .nunique()
        .reset_index(name="n_unique_flags")
    )
    conflicts = conflicts[conflicts["n_unique_flags"] > 1]
    if not conflicts.empty:
        raise ValueError(
            "Conflicting Q75 selection flags were found for duplicated animal x syllable pairs."
        )

    out = out[["animal_id", "syllable", "is_q75"]].drop_duplicates().reset_index(drop=True)
    print(
        f"[INFO] Figure 3 qualifying pairs: {len(out)} across {out['animal_id'].nunique()} birds"
    )
    print(
        f"[INFO] {selection_column}=True: {int(out['is_q75'].sum())} pairs; "
        f"False: {int((~out['is_q75']).sum())} pairs"
    )
    return out


def choose_bc_syllable_column(
    table: pd.DataFrame,
    animal_col: str,
    selected_pairs: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    candidates = [
        "cluster_token",
        "syllable",
        "Syllable",
        "label",
        "cluster_label",
        "syllable_cluster",
        "cluster",
        "cluster_id",
    ]
    present = [c for c in candidates if c in table.columns]
    if not present:
        raise ValueError(
            "No syllable/cluster identifier column was found in the BC table. "
            f"Found: {list(table.columns)}"
        )

    selected_keys = selected_pairs[["animal_id", "syllable"]].drop_duplicates()
    reports = []
    for column in present:
        candidate = pd.DataFrame(
            {
                "animal_id": table[animal_col].astype(str).str.strip(),
                "syllable": table[column].map(normalize_syllable_token),
            }
        ).drop_duplicates()
        candidate = candidate[(candidate["animal_id"] != "") & (candidate["syllable"] != "")]
        matched = selected_keys.merge(candidate, on=["animal_id", "syllable"], how="inner")
        reports.append(
            {
                "candidate_column": column,
                "matched_selected_pairs": int(len(matched)),
                "matched_birds": int(matched["animal_id"].nunique()),
                "available_unique_pairs": int(len(candidate)),
            }
        )

    report = pd.DataFrame(reports).sort_values(
        ["matched_selected_pairs", "matched_birds"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    for row in report.itertuples(index=False):
        print(
            "[INFO] BC label candidate "
            f"{row.candidate_column!r}: {row.matched_selected_pairs}/{len(selected_keys)} "
            f"selected pairs matched across {row.matched_birds} birds"
        )

    best = report.iloc[0]
    if int(best["matched_selected_pairs"]) == 0:
        raise ValueError("None of the BC identifier columns matched the Figure 3 selection CSV.")

    chosen = str(best["candidate_column"])
    print(f"[OK] Using BC syllable identifier column: {chosen}")
    return chosen, report


def load_bc_table(
    path: str,
    metadata_excel: Optional[str],
    selected_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    table = pd.read_csv(csv_path)
    print(f"[INFO] BC input columns: {list(table.columns)}")

    animal_col = first_present(table.columns, ["animal_id", "Animal ID", "bird_id", "bird"])
    if animal_col is None:
        raise ValueError(
            "BC CSV must contain an animal/bird identifier column. "
            f"Found: {list(table.columns)}"
        )

    syllable_col, identifier_report = choose_bc_syllable_column(table, animal_col, selected_pairs)
    pre_col = first_present(
        table.columns,
        ["bc_pre", "pre_bc", "Pre BC", "median_pre_bc", "pre_bhattacharyya_coefficient"],
    )
    post_col = first_present(
        table.columns,
        ["bc_post", "post_bc", "Post BC", "median_post_bc", "post_bhattacharyya_coefficient"],
    )
    if pre_col is None or post_col is None:
        raise ValueError(
            "BC CSV must contain pre and post BC columns. "
            f"Found: {list(table.columns)}"
        )

    out = table.copy()
    out["animal_id"] = out[animal_col].astype(str).str.strip()
    out["syllable_source_column"] = syllable_col
    out["syllable_raw"] = out[syllable_col]
    out["syllable"] = out[syllable_col].map(normalize_syllable_token)
    out["pre_bc"] = pd.to_numeric(out[pre_col], errors="coerce")
    out["post_bc"] = pd.to_numeric(out[post_col], errors="coerce")
    out["delta_bc"] = out["post_bc"] - out["pre_bc"]

    group_col = first_present(
        out.columns,
        [
            "lesion_group_detailed",
            "display_group",
            "lesion_group",
            "lesion_hit_type",
            "raw_lesion_hit_type",
            "raw_hit_type",
            "group_label",
        ],
    )
    metadata_map = read_metadata_map(metadata_excel)
    if group_col is not None:
        out["detailed_group"] = out[group_col].map(canonical_group)
    else:
        out["detailed_group"] = UNKNOWN
    if metadata_map:
        from_workbook = out["animal_id"].map(metadata_map)
        out.loc[from_workbook.notna(), "detailed_group"] = from_workbook[from_workbook.notna()].map(canonical_group)

    out["pooled_group"] = out["detailed_group"].map(pooled_group)
    out = out.dropna(subset=["pre_bc", "post_bc"]).copy()
    out = out[(out["animal_id"] != "") & (out["syllable"] != "")].copy()
    return out.reset_index(drop=True), identifier_report


def exact_signflip_p(differences: np.ndarray, alternative: str) -> tuple[float, int]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n == 0:
        return np.nan, 0
    observed = float(np.mean(differences))
    if n <= 20:
        total = 1 << n
        null_values = np.empty(total, dtype=float)
        for mask in range(total):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            null_values[mask] = np.mean(differences * signs)
    else:
        total = 100_000
        rng = np.random.default_rng(123)
        signs = rng.choice([-1.0, 1.0], size=(total, n))
        null_values = np.mean(signs * differences[None, :], axis=1)
    if alternative == "less":
        p = np.mean(null_values <= observed + 1e-12)
    elif alternative == "greater":
        p = np.mean(null_values >= observed - 1e-12)
    else:
        p = np.mean(np.abs(null_values) >= abs(observed) - 1e-12)
    return float(p), int(total)


def paired_wilcoxon(pre: np.ndarray, post: np.ndarray, alternative: str = "two-sided") -> float:
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    if len(pre) == 0 or scipy_stats is None:
        return np.nan
    try:
        return float(
            scipy_stats.wilcoxon(
                post,
                pre,
                alternative=alternative,
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    except Exception:
        return np.nan


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return adjusted.tolist()
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(valid)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def p_label(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, height: float, text: str, fontsize: int = 16) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="0.15", linewidth=1.1)
    ax.text((x1 + x2) / 2, y + height * 1.15, text, ha="center", va="bottom", fontsize=fontsize)


def draw_box(ax: plt.Axes, values: np.ndarray, position: float, color: str, alpha: float, width: float = 0.68) -> None:
    result = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for patch in result["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(alpha)
        patch.set_linewidth(1.4)
    for artist in result["whiskers"] + result["caps"]:
        artist.set_color("0.25")
        artist.set_linewidth(1.2)
    for artist in result["medians"]:
        artist.set_color("black")
        artist.set_linewidth(1.8)


def build_panel_c_by_bird(bc: pd.DataFrame) -> pd.DataFrame:
    keep = bc[bc["pooled_group"].isin(GROUP_ORDER)].copy()
    grouped = (
        keep.groupby(["pooled_group", "animal_id"], dropna=False)
        .agg(
            n_clusters=("syllable", "nunique"),
            pre_bc=("pre_bc", "median"),
            post_bc=("post_bc", "median"),
        )
        .reset_index()
    )
    grouped["delta_bc"] = grouped["post_bc"] - grouped["pre_bc"]
    return grouped.sort_values(["pooled_group", "animal_id"]).reset_index(drop=True)


def run_panel_c_tests(by_bird: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        subset = by_bird[by_bird["pooled_group"] == group].copy()
        pre = subset["pre_bc"].to_numpy(dtype=float)
        post = subset["post_bc"].to_numpy(dtype=float)
        mask = np.isfinite(pre) & np.isfinite(post)
        pre = pre[mask]
        post = post[mask]
        diff = post - pre
        sf1, nperm = exact_signflip_p(diff, alternative="less")
        sf2, _ = exact_signflip_p(diff, alternative="two-sided")
        w1 = paired_wilcoxon(pre, post, alternative="less")
        w2 = paired_wilcoxon(pre, post, alternative="two-sided")
        rows.append(
            {
                "group": group,
                "n_birds": int(len(diff)),
                "median_pre_bc": float(np.median(pre)) if len(pre) else np.nan,
                "median_post_bc": float(np.median(post)) if len(post) else np.nan,
                "median_delta_bc_post_minus_pre": float(np.median(diff)) if len(diff) else np.nan,
                "mean_delta_bc_post_minus_pre": float(np.mean(diff)) if len(diff) else np.nan,
                "wilcoxon_two_sided_p": w2,
                "wilcoxon_one_sided_p_post_less_pre": w1,
                "signflip_two_sided_p": sf2,
                "signflip_one_sided_p_post_less_pre": sf1,
                "n_signflip_permutations": nperm,
            }
        )
    tests = pd.DataFrame(rows)
    tests["wilcoxon_two_sided_holm_p"] = holm_adjust(tests["wilcoxon_two_sided_p"].tolist())
    tests["wilcoxon_one_sided_holm_p"] = holm_adjust(tests["wilcoxon_one_sided_p_post_less_pre"].tolist())
    tests["signflip_two_sided_holm_p"] = holm_adjust(tests["signflip_two_sided_p"].tolist())
    tests["signflip_one_sided_holm_p"] = holm_adjust(tests["signflip_one_sided_p_post_less_pre"].tolist())
    return tests


def build_panel_d_tables(bc: pd.DataFrame, selected_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = bc.merge(selected_pairs, on=["animal_id", "syllable"], how="inner")
    merged["subset"] = np.where(merged["is_q75"], ">= Q75 ΔCV syllables", "< Q75 ΔCV syllables")

    ml = merged[merged["pooled_group"] == POOLED_ML].copy()
    if ml.empty:
        raise ValueError("No pooled medial+lateral rows remained after merging BC data with the Q75 selection CSV.")

    grouped = (
        ml.groupby(["animal_id", "subset"], dropna=False)
        .agg(
            n_syllables=("syllable", "nunique"),
            pre_bc=("pre_bc", "median"),
            post_bc=("post_bc", "median"),
        )
        .reset_index()
    )
    grouped["delta_bc"] = grouped["post_bc"] - grouped["pre_bc"]

    wide = grouped.pivot(index="animal_id", columns="subset")
    wide.columns = ["_".join(map(str, c)).strip("_") for c in wide.columns.to_flat_index()]
    wide = wide.reset_index().rename(
        columns={
            "n_syllables_>= Q75 ΔCV syllables": "q75_n_syllables",
            "pre_bc_>= Q75 ΔCV syllables": "q75_pre_bc",
            "post_bc_>= Q75 ΔCV syllables": "q75_post_bc",
            "delta_bc_>= Q75 ΔCV syllables": "q75_delta_bc",
            "n_syllables_< Q75 ΔCV syllables": "below_q75_n_syllables",
            "pre_bc_< Q75 ΔCV syllables": "below_q75_pre_bc",
            "post_bc_< Q75 ΔCV syllables": "below_q75_post_bc",
            "delta_bc_< Q75 ΔCV syllables": "below_q75_delta_bc",
        }
    )
    for col in [
        "q75_n_syllables", "q75_pre_bc", "q75_post_bc", "q75_delta_bc",
        "below_q75_n_syllables", "below_q75_pre_bc", "below_q75_post_bc", "below_q75_delta_bc",
    ]:
        if col not in wide.columns:
            wide[col] = np.nan
    wide["q75_delta_bc"] = wide["q75_post_bc"] - wide["q75_pre_bc"]
    wide["below_q75_delta_bc"] = wide["below_q75_post_bc"] - wide["below_q75_pre_bc"]
    wide["q75_minus_below_delta_bc"] = wide["q75_delta_bc"] - wide["below_q75_delta_bc"]
    return ml.reset_index(drop=True), grouped.reset_index(drop=True), wide.reset_index(drop=True)


def run_panel_d_tests(by_bird: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset_label, pre_col, post_col in [
        (">= Q75 ΔCV syllables", "q75_pre_bc", "q75_post_bc"),
        ("< Q75 ΔCV syllables", "below_q75_pre_bc", "below_q75_post_bc"),
    ]:
        pre_all = by_bird[pre_col].to_numpy(dtype=float)
        post_all = by_bird[post_col].to_numpy(dtype=float)
        mask = np.isfinite(pre_all) & np.isfinite(post_all)
        pre = pre_all[mask]
        post = post_all[mask]
        diff = post - pre
        sf1, nperm = exact_signflip_p(diff, alternative="less")
        sf2, _ = exact_signflip_p(diff, alternative="two-sided")
        w1 = paired_wilcoxon(pre, post, alternative="less")
        w2 = paired_wilcoxon(pre, post, alternative="two-sided")
        rows.append(
            {
                "comparison": f"{subset_label}: Pre vs Post",
                "subset": subset_label,
                "n_birds": int(len(diff)),
                "median_pre_bc": float(np.median(pre)) if len(pre) else np.nan,
                "median_post_bc": float(np.median(post)) if len(post) else np.nan,
                "median_delta_bc_post_minus_pre": float(np.median(diff)) if len(diff) else np.nan,
                "mean_delta_bc_post_minus_pre": float(np.mean(diff)) if len(diff) else np.nan,
                "wilcoxon_two_sided_p": w2,
                "wilcoxon_one_sided_p_post_less_pre": w1,
                "signflip_two_sided_p": sf2,
                "signflip_one_sided_p_post_less_pre": sf1,
                "n_signflip_permutations": nperm,
            }
        )
    tests = pd.DataFrame(rows)
    tests["wilcoxon_two_sided_holm_p"] = holm_adjust(tests["wilcoxon_two_sided_p"].tolist())
    tests["wilcoxon_one_sided_holm_p"] = holm_adjust(tests["wilcoxon_one_sided_p_post_less_pre"].tolist())
    tests["signflip_two_sided_holm_p"] = holm_adjust(tests["signflip_two_sided_p"].tolist())
    tests["signflip_one_sided_holm_p"] = holm_adjust(tests["signflip_one_sided_p_post_less_pre"].tolist())
    return tests


def _apply_jneurosci_axes_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel_c(ax: plt.Axes, by_bird: pd.DataFrame, tests: pd.DataFrame, show_points: bool = True) -> None:
    positions = {
        SHAM: (0.0, 1.0),
        LATERAL: (3.0, 4.0),
        POOLED_ML: (6.0, 7.0),
    }
    rng = np.random.default_rng(20260811)
    all_values = []

    for group in GROUP_ORDER:
        subset = by_bird[by_bird["pooled_group"] == group].copy()
        pre = subset["pre_bc"].to_numpy(dtype=float)
        post = subset["post_bc"].to_numpy(dtype=float)
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        all_values.extend([pre, post])
        xpre, xpost = positions[group]
        draw_box(ax, pre, xpre, COLORS[group], alpha=0.48)
        draw_box(ax, post, xpost, COLORS[group], alpha=0.72)
        if show_points:
            for x0, vals, alpha in [(xpre, pre, 0.82), (xpost, post, 0.95)]:
                x = np.full(len(vals), x0) + rng.uniform(-0.055, 0.055, len(vals))
                ax.scatter(x, vals, s=44, color=COLORS[group], alpha=alpha,
                           edgecolors="white", linewidths=0.55, zorder=3)

    ax.set_xticks([0, 1, 3, 4, 6, 7])
    ax.set_xticklabels(["Pre", "Post", "Pre", "Post", "Pre", "Post"], fontsize=16)
    ax.set_ylabel("Bhattacharyya coefficient", fontsize=19)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=8)
    trans = ax.get_xaxis_transform()
    for group in GROUP_ORDER:
        xmid = np.mean(positions[group])
        ax.text(xmid, -0.16, GROUP_DISPLAY[group], ha="center", va="top", transform=trans, fontsize=15)

    flat = np.concatenate([v for pair in all_values for v in ([pair] if pair.size else [])])
    data_min = float(np.nanmin(flat))
    data_max = float(np.nanmax(flat))
    span = max(data_max - data_min, 0.015)
    lower = max(0.0, data_min - 0.08 * span)
    bracket_base = data_max + 0.09 * span
    bracket_height = 0.035 * span
    # Figure brackets use the Holm-adjusted two-sided paired Wilcoxon p-values.
    for group in GROUP_ORDER:
        row = tests.loc[tests["group"] == group].iloc[0]
        p = float(row["wilcoxon_two_sided_holm_p"])
        x1, x2 = positions[group]
        add_bracket(ax, x1, x2, bracket_base, bracket_height, p_label(p), fontsize=15)
    ax.set_ylim(lower, bracket_base + 0.18 * span)
    ax.set_xlim(-0.8, 7.8)
    _apply_jneurosci_axes_style(ax)


def plot_panel_d(ax: plt.Axes, by_bird: pd.DataFrame, tests: pd.DataFrame, show_points: bool = True) -> None:
    positions = {"q75_pre": 0.0, "q75_post": 1.0, "below_pre": 3.1, "below_post": 4.1}
    q75_mask = np.isfinite(by_bird["q75_pre_bc"].to_numpy(float)) & np.isfinite(by_bird["q75_post_bc"].to_numpy(float))
    below_mask = np.isfinite(by_bird["below_q75_pre_bc"].to_numpy(float)) & np.isfinite(by_bird["below_q75_post_bc"].to_numpy(float))
    arrays = {
        "q75_pre": by_bird.loc[q75_mask, "q75_pre_bc"].to_numpy(float),
        "q75_post": by_bird.loc[q75_mask, "q75_post_bc"].to_numpy(float),
        "below_pre": by_bird.loc[below_mask, "below_q75_pre_bc"].to_numpy(float),
        "below_post": by_bird.loc[below_mask, "below_q75_post_bc"].to_numpy(float),
    }
    draw_box(ax, arrays["q75_pre"], positions["q75_pre"], COLORS[POOLED_ML], alpha=0.28)
    draw_box(ax, arrays["q75_post"], positions["q75_post"], COLORS[POOLED_ML], alpha=0.50)
    draw_box(ax, arrays["below_pre"], positions["below_pre"], COLORS[POOLED_ML], alpha=0.28)
    draw_box(ax, arrays["below_post"], positions["below_post"], COLORS[POOLED_ML], alpha=0.50)
    if show_points:
        rng = np.random.default_rng(20260806)
        for key, values in arrays.items():
            x = np.full(len(values), positions[key]) + rng.uniform(-0.055, 0.055, len(values))
            point_alpha = 0.82 if "pre" in key else 0.95
            ax.scatter(x, values, s=40, color=COLORS[POOLED_ML], alpha=point_alpha,
                       edgecolors="white", linewidths=0.55, zorder=3)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["Pre", "Post", "Pre", "Post"], fontsize=16)
    ax.set_ylabel("Bhattacharyya coefficient", fontsize=19)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=8)
    trans = ax.get_xaxis_transform()
    ax.text(0.5, -0.16, "≥ Q75 ΔCV\nsyllables", ha="center", va="top", transform=trans, fontsize=15)
    ax.text(3.6, -0.16, "< Q75 ΔCV\nsyllables", ha="center", va="top", transform=trans, fontsize=15)
    flat = np.concatenate(list(arrays.values()))
    data_min = float(np.nanmin(flat))
    data_max = float(np.nanmax(flat))
    span = max(data_max - data_min, 0.015)
    lower = max(0.0, data_min - 0.10 * span)
    bracket_base = data_max + 0.09 * span
    bracket_height = 0.035 * span
    # Figure brackets use the Holm-adjusted two-sided paired Wilcoxon p-values.
    q75_p = float(tests.loc[tests["subset"] == ">= Q75 ΔCV syllables", "wilcoxon_two_sided_holm_p"].iloc[0])
    below_p = float(tests.loc[tests["subset"] == "< Q75 ΔCV syllables", "wilcoxon_two_sided_holm_p"].iloc[0])
    add_bracket(ax, positions["q75_pre"], positions["q75_post"], bracket_base, bracket_height, p_label(q75_p), fontsize=16)
    add_bracket(ax, positions["below_pre"], positions["below_post"], bracket_base, bracket_height, p_label(below_p), fontsize=16)
    ax.set_ylim(lower, bracket_base + 0.17 * span)
    ax.set_xlim(-0.8, 4.9)
    _apply_jneurosci_axes_style(ax)


def save_panel_c(
    by_bird: pd.DataFrame,
    tests: pd.DataFrame,
    out_path: Path,
    dpi: int,
    show: bool,
    show_points: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_C)
    plot_panel_c(ax, by_bird, tests, show_points=show_points)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def save_panel_d(
    by_bird: pd.DataFrame,
    tests: pd.DataFrame,
    out_path: Path,
    dpi: int,
    show: bool,
    show_points: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_D)
    plot_panel_d(ax, by_bird, tests, show_points=show_points)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def save_combined(
    panel_c_by_bird: pd.DataFrame,
    panel_c_tests: pd.DataFrame,
    panel_d_by_bird: pd.DataFrame,
    panel_d_tests: pd.DataFrame,
    out_path: Path,
    dpi: int,
    show: bool,
    show_points: bool = True,
) -> None:
    fig = plt.figure(figsize=FIGSIZE_COMBINED)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.08, 0.92], hspace=0.35)
    axc = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[1, 0])
    plot_panel_c(axc, panel_c_by_bird, panel_c_tests, show_points=show_points)
    plot_panel_d(axd, panel_d_by_bird, panel_d_tests, show_points=show_points)
    axc.text(-0.10, 1.02, "C", transform=axc.transAxes, fontsize=22, fontweight="bold", va="top")
    axd.text(-0.10, 1.02, "D", transform=axd.transAxes, fontsize=22, fontweight="bold", va="top")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_pairs = read_quantile_selection(args.quantile_selection_csv, args.selection_column)
    bc, id_report = load_bc_table(args.bc_csv, args.metadata_excel, selected_pairs)
    id_report.to_csv(out_dir / "bc_identifier_column_match_report.csv", index=False)

    # Panel C: all qualifying clusters by lesion group.
    panel_c_by_bird = build_panel_c_by_bird(bc)
    panel_c_tests = run_panel_c_tests(panel_c_by_bird)
    panel_c_by_bird.to_csv(out_dir / "panelC_all_qualifying_by_lesion_group_by_bird.csv", index=False)
    panel_c_tests.to_csv(out_dir / "panelC_all_qualifying_by_lesion_group_tests.csv", index=False)

    # Panel D: pooled M+L only, split by Q75 selection.
    panel_d_cluster_rows, panel_d_long, panel_d_by_bird = build_panel_d_tables(bc, selected_pairs)
    panel_d_tests = run_panel_d_tests(panel_d_by_bird)
    panel_d_cluster_rows.to_csv(out_dir / "panelD_bc_cluster_rows.csv", index=False)
    panel_d_long.to_csv(out_dir / "panelD_bc_merge_summary_by_bird.csv", index=False)
    panel_d_by_bird.to_csv(out_dir / "panelD_pooled_ML_Q75_by_bird.csv", index=False)
    panel_d_tests.to_csv(out_dir / "panelD_pooled_ML_Q75_tests.csv", index=False)

    # Versions with bird-level points overlaid.
    save_panel_c(
        panel_c_by_bird, panel_c_tests,
        out_dir / "figure4_panelC_all_qualifying_by_lesion_group.png",
        args.dpi, args.show, show_points=True
    )
    save_panel_d(
        panel_d_by_bird, panel_d_tests,
        out_dir / "figure4_panelD_pooled_ML_Q75_split.png",
        args.dpi, args.show, show_points=True
    )
    save_combined(
        panel_c_by_bird, panel_c_tests, panel_d_by_bird, panel_d_tests,
        out_dir / "figure4_panelCD_combined.png",
        args.dpi, args.show, show_points=True
    )

    # Boxplot-only versions: identical statistics/boxes/brackets, but no
    # overlaid bird-level points.
    save_panel_c(
        panel_c_by_bird, panel_c_tests,
        out_dir / "figure4_panelC_all_qualifying_by_lesion_group_boxesonly.png",
        args.dpi, args.show, show_points=False
    )
    save_panel_d(
        panel_d_by_bird, panel_d_tests,
        out_dir / "figure4_panelD_pooled_ML_Q75_split_boxesonly.png",
        args.dpi, args.show, show_points=False
    )
    save_combined(
        panel_c_by_bird, panel_c_tests, panel_d_by_bird, panel_d_tests,
        out_dir / "figure4_panelCD_combined_boxesonly.png",
        args.dpi, args.show, show_points=False
    )

    print("\n[OK] Wrote:")
    for name in [
        "figure4_panelC_all_qualifying_by_lesion_group.png",
        "figure4_panelC_all_qualifying_by_lesion_group_boxesonly.png",
        "figure4_panelD_pooled_ML_Q75_split.png",
        "figure4_panelD_pooled_ML_Q75_split_boxesonly.png",
        "figure4_panelCD_combined.png",
        "figure4_panelCD_combined_boxesonly.png",
        "panelC_all_qualifying_by_lesion_group_by_bird.csv",
        "panelC_all_qualifying_by_lesion_group_tests.csv",
        "panelD_pooled_ML_Q75_by_bird.csv",
        "panelD_pooled_ML_Q75_tests.csv",
        "bc_identifier_column_match_report.csv",
        "panelD_bc_cluster_rows.csv",
        "panelD_bc_merge_summary_by_bird.csv",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
