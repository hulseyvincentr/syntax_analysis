#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 replacement panel: pooled M+L birds, Top 15% vs Remaining 85%,
with Pre and Post BC shown separately.

Selection is imported from the Figure 3 top-15% selected-pairs CSV.
Within each bird and subset, the script takes the median BC across qualifying
syllables, then compares Post vs Pre across birds.

Primary displayed tests:
  - Top 15%: raw two-sided paired Wilcoxon
  - Remaining 85%: raw two-sided paired Wilcoxon

This matches the statistical convention used for the original Figure 4D.
Additional one-sided and Holm-adjusted p-values are written to the output CSV.

For transparency, exact paired sign-flip p-values and the selected-vs-remaining
DeltaBC contrast are also written to the output tables.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
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

# Match the updated Figure 3 palette.
FIG3_POOLED_ML_PURPLE = "#7A4FB7"


def norm_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).strip().split()).lower()


def normalize_syllable_token(x: Any) -> str:
    """Normalize labels so numeric tokens such as 3, 3.0, and '3.0' match."""
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""

    # Spreadsheet/CSV parsing often turns integer labels into strings like
    # "3.0". Convert only values that are exactly integer-like numerics.
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


def read_selected_pairs(path: str) -> pd.DataFrame:
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
    if animal_col is None or syllable_col is None:
        raise ValueError(
            "Selected-pairs CSV must contain animal and syllable/cluster columns. "
            f"Found: {list(table.columns)}"
        )

    out = table[[animal_col, syllable_col]].copy()
    out.columns = ["animal_id", "syllable"]
    out["animal_id"] = out["animal_id"].astype(str).str.strip()
    out["syllable"] = out["syllable"].map(normalize_syllable_token)
    out = out[(out["animal_id"] != "") & (out["syllable"] != "")].copy()
    return out.drop_duplicates().reset_index(drop=True)


def choose_bc_syllable_column(
    table: pd.DataFrame,
    animal_col: str,
    selected_pairs: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Choose the BC label column that actually matches the selected-pairs CSV."""
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
        candidate = candidate[
            (candidate["animal_id"] != "") & (candidate["syllable"] != "")
        ]
        matched = selected_keys.merge(
            candidate,
            on=["animal_id", "syllable"],
            how="inner",
        )
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
    )
    for row in report.itertuples(index=False):
        print(
            "[INFO] BC label candidate "
            f"{row.candidate_column!r}: "
            f"{row.matched_selected_pairs}/{len(selected_keys)} selected pairs "
            f"matched across {row.matched_birds} birds"
        )

    best = report.iloc[0]
    if int(best["matched_selected_pairs"]) == 0:
        selected_example = selected_keys.head(12).to_dict("records")
        bc_examples = {}
        for column in present:
            values = (
                table[column]
                .map(normalize_syllable_token)
                .loc[lambda s: s != ""]
                .drop_duplicates()
                .head(12)
                .tolist()
            )
            bc_examples[column] = values
        raise ValueError(
            "None of the BC identifier columns matched the selected-pairs CSV, "
            "even after normalizing integer-like labels such as 3.0 to 3.\n"
            f"Selected-pair examples: {selected_example}\n"
            f"BC identifier examples: {bc_examples}"
        )

    chosen = str(best["candidate_column"])
    print(f"[OK] Using BC syllable identifier column: {chosen}")
    return chosen, report.reset_index(drop=True)


def read_bc_table(
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
    syllable_col, identifier_report = choose_bc_syllable_column(
        table,
        animal_col,
        selected_pairs,
    )
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
            "BC CSV must contain animal, syllable/cluster, pre-BC, and post-BC columns. "
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


def paired_wilcoxon(
    pre: np.ndarray,
    post: np.ndarray,
    alternative: str = "two-sided",
) -> float:
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre = pre[mask]
    post = post[mask]
    if len(pre) == 0 or scipy_stats is None:
        return np.nan
    try:
        # scipy compares the first array with the second. Thus
        # alternative="less" tests post < pre.
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


def bootstrap_mean_ci(values: np.ndarray, reps: int = 10_000, seed: int = 456) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    for i in range(reps):
        estimates[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


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


def make_bird_table(merged: pd.DataFrame) -> pd.DataFrame:
    ml = merged[merged["pooled_group"] == POOLED_ML].copy()
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
    wide.columns = ["_".join(map(str, col)).strip("_") for col in wide.columns.to_flat_index()]
    wide = wide.reset_index().rename(
        columns={
            "n_syllables_selected": "top15_n_syllables",
            "pre_bc_selected": "top15_pre_bc",
            "post_bc_selected": "top15_post_bc",
            "delta_bc_selected": "top15_delta_bc",
            "n_syllables_remaining": "remaining85_n_syllables",
            "pre_bc_remaining": "remaining85_pre_bc",
            "post_bc_remaining": "remaining85_post_bc",
            "delta_bc_remaining": "remaining85_delta_bc",
        }
    )

    required = [
        "top15_pre_bc",
        "top15_post_bc",
        "remaining85_pre_bc",
        "remaining85_post_bc",
    ]
    present = [c for c in required if c in wide.columns]
    if len(present) != len(required):
        raise ValueError(
            "At least one subset was absent after merging selected pairs with BC data. "
            f"Available columns: {list(wide.columns)}"
        )
    wide = wide.dropna(subset=required).copy()
    wide["top15_delta_bc"] = wide["top15_post_bc"] - wide["top15_pre_bc"]
    wide["remaining85_delta_bc"] = wide["remaining85_post_bc"] - wide["remaining85_pre_bc"]
    wide["top15_minus_remaining_delta_bc"] = wide["top15_delta_bc"] - wide["remaining85_delta_bc"]
    return wide


def run_tests(by_bird: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset, pre_col, post_col in [
        ("Top 15%", "top15_pre_bc", "top15_post_bc"),
        ("Remaining 85%", "remaining85_pre_bc", "remaining85_post_bc"),
    ]:
        pre = by_bird[pre_col].to_numpy(dtype=float)
        post = by_bird[post_col].to_numpy(dtype=float)
        diff = post - pre
        signflip_one_sided_p, n_permutations = exact_signflip_p(diff, alternative="less")
        signflip_two_sided_p, _ = exact_signflip_p(diff, alternative="two-sided")
        wilcoxon_one_sided_p = paired_wilcoxon(pre, post, alternative="less")
        wilcoxon_two_sided_p = paired_wilcoxon(pre, post, alternative="two-sided")
        ci_low, ci_high = bootstrap_mean_ci(diff)
        rows.append(
            {
                "comparison": f"{subset}: Pre vs Post",
                "subset": subset,
                "n_birds": len(diff),
                "median_pre_bc": float(np.median(pre)),
                "median_post_bc": float(np.median(post)),
                "median_delta_bc_post_minus_pre": float(np.median(diff)),
                "mean_delta_bc_post_minus_pre": float(np.mean(diff)),
                "mean_delta_bootstrap_ci_low": ci_low,
                "mean_delta_bootstrap_ci_high": ci_high,
                "wilcoxon_two_sided_p": wilcoxon_two_sided_p,
                "wilcoxon_one_sided_p_post_less_pre": wilcoxon_one_sided_p,
                "signflip_two_sided_p": signflip_two_sided_p,
                "signflip_one_sided_p_post_less_pre": signflip_one_sided_p,
                "n_signflip_permutations": n_permutations,
            }
        )

    tests = pd.DataFrame(rows)
    tests["wilcoxon_two_sided_holm_p"] = holm_adjust(
        tests["wilcoxon_two_sided_p"].tolist()
    )
    tests["wilcoxon_one_sided_holm_p"] = holm_adjust(
        tests["wilcoxon_one_sided_p_post_less_pre"].tolist()
    )
    tests["signflip_two_sided_holm_p"] = holm_adjust(
        tests["signflip_two_sided_p"].tolist()
    )
    tests["signflip_one_sided_holm_p"] = holm_adjust(
        tests["signflip_one_sided_p_post_less_pre"].tolist()
    )

    interaction = by_bird["top15_minus_remaining_delta_bc"].to_numpy(dtype=float)
    interaction_p, n_permutations = exact_signflip_p(interaction, alternative="less")
    ci_low, ci_high = bootstrap_mean_ci(interaction)
    interaction_row = pd.DataFrame(
        [
            {
                "comparison": "Top 15% DeltaBC < Remaining 85% DeltaBC",
                "subset": "DeltaBC contrast",
                "n_birds": len(interaction),
                "median_pre_bc": np.nan,
                "median_post_bc": np.nan,
                "median_delta_bc_post_minus_pre": float(np.median(interaction)),
                "mean_delta_bc_post_minus_pre": float(np.mean(interaction)),
                "mean_delta_bootstrap_ci_low": ci_low,
                "mean_delta_bootstrap_ci_high": ci_high,
                "wilcoxon_two_sided_p": paired_wilcoxon(
                    by_bird["remaining85_delta_bc"].to_numpy(dtype=float),
                    by_bird["top15_delta_bc"].to_numpy(dtype=float),
                    alternative="two-sided",
                ),
                "wilcoxon_one_sided_p_post_less_pre": paired_wilcoxon(
                    by_bird["remaining85_delta_bc"].to_numpy(dtype=float),
                    by_bird["top15_delta_bc"].to_numpy(dtype=float),
                    alternative="less",
                ),
                "signflip_two_sided_p": exact_signflip_p(
                    interaction, alternative="two-sided"
                )[0],
                "signflip_one_sided_p_post_less_pre": interaction_p,
                "n_signflip_permutations": n_permutations,
                "wilcoxon_two_sided_holm_p": np.nan,
                "wilcoxon_one_sided_holm_p": np.nan,
                "signflip_two_sided_holm_p": np.nan,
                "signflip_one_sided_holm_p": np.nan,
            }
        ]
    )
    return pd.concat([tests, interaction_row], ignore_index=True)


def draw_box(ax: plt.Axes, values: np.ndarray, position: float, alpha: float) -> None:
    color = FIG3_POOLED_ML_PURPLE
    result = ax.boxplot(
        [values],
        positions=[position],
        widths=0.65,
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


def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, height: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="0.15", linewidth=1.1)
    ax.text((x1 + x2) / 2, y + height * 1.2, text, ha="center", va="bottom", fontsize=13)


def plot_panel(by_bird: pd.DataFrame, tests: pd.DataFrame, output: Path, dpi: int, show: bool) -> None:
    positions = {
        "top15_pre": 0.0,
        "top15_post": 1.0,
        "remaining_pre": 3.1,
        "remaining_post": 4.1,
    }
    arrays = {
        "top15_pre": by_bird["top15_pre_bc"].to_numpy(dtype=float),
        "top15_post": by_bird["top15_post_bc"].to_numpy(dtype=float),
        "remaining_pre": by_bird["remaining85_pre_bc"].to_numpy(dtype=float),
        "remaining_post": by_bird["remaining85_post_bc"].to_numpy(dtype=float),
    }

    fig, ax = plt.subplots(figsize=(8.3, 5.3))

    # Use the same pooled-M+L purple as Figure 3. Pre is a lighter tint of the
    # same hue; Post is more saturated, while the labels remain the primary
    # pre/post coding.
    draw_box(ax, arrays["top15_pre"], positions["top15_pre"], alpha=0.28)
    draw_box(ax, arrays["top15_post"], positions["top15_post"], alpha=0.50)
    draw_box(ax, arrays["remaining_pre"], positions["remaining_pre"], alpha=0.28)
    draw_box(ax, arrays["remaining_post"], positions["remaining_post"], alpha=0.50)

    rng = np.random.default_rng(20260806)
    jitter_scale = 0.055
    for key, values in arrays.items():
        x = np.full(len(values), positions[key]) + rng.uniform(-jitter_scale, jitter_scale, len(values))
        point_alpha = 0.82 if "pre" in key else 0.95
        ax.scatter(
            x,
            values,
            s=34,
            color=FIG3_POOLED_ML_PURPLE,
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["Pre", "Post", "Pre", "Post"], fontsize=16)
    ax.set_ylabel("Bhattacharyya coefficient", fontsize=19)

    # Two-line subset labels below the Pre/Post labels, mirroring the former
    # Figure 4F layout.
    trans = ax.get_xaxis_transform()
    ax.text(
        0.5,
        -0.16,
        "Top 15%\nvariance syllables",
        ha="center",
        va="top",
        transform=trans,
        fontsize=14,
    )
    ax.text(
        3.6,
        -0.16,
        "Remaining 85%\nsyllables",
        ha="center",
        va="top",
        transform=trans,
        fontsize=14,
    )

    all_values = np.concatenate(list(arrays.values()))
    data_min = float(np.nanmin(all_values))
    data_max = float(np.nanmax(all_values))
    data_span = max(data_max - data_min, 0.015)
    lower = max(0.0, data_min - 0.10 * data_span)
    bracket_base = data_max + 0.09 * data_span
    bracket_height = 0.035 * data_span

    top_row = tests[tests["subset"] == "Top 15%"].iloc[0]
    remaining_row = tests[tests["subset"] == "Remaining 85%"].iloc[0]
    # Display Holm-adjusted paired Wilcoxon significance; raw and sign-flip
    # values remain in the output CSV/text summary.
    # Match the original Figure 4D convention: raw two-sided paired Wilcoxon.
    # Holm-adjusted values are still written to the output table.
    top_p = float(top_row["wilcoxon_two_sided_p"])
    remaining_p = float(remaining_row["wilcoxon_two_sided_p"])
    add_bracket(
        ax,
        positions["top15_pre"],
        positions["top15_post"],
        bracket_base,
        bracket_height,
        p_label(top_p),
    )
    add_bracket(
        ax,
        positions["remaining_pre"],
        positions["remaining_post"],
        bracket_base,
        bracket_height,
        p_label(remaining_p),
    )

    ax.set_ylim(lower, bracket_base + 0.17 * data_span)
    ax.set_xlim(-0.8, 4.9)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.30)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_text_summary(
    path: Path,
    bc_path: str,
    selected_path: str,
    by_bird: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Figure 4: Top 15% vs remaining 85% BC, Pre vs Post\n")
        handle.write("========================================================\n\n")
        handle.write(f"BC input: {bc_path}\n")
        handle.write(f"Selected-pairs input: {selected_path}\n")
        handle.write(f"Pooled M+L birds with both subsets: {len(by_bird)}\n")
        handle.write("Bird-level statistic: median BC across syllables within each subset and period\n")
        handle.write(
            "Displayed tests: raw two-sided paired Wilcoxon, matching the original Figure 4D convention\n"
        )
        handle.write(
            "Additional one-sided and Holm-adjusted p values are reported in the tests CSV\n\n"
        )

        for subset in ["Top 15%", "Remaining 85%"]:
            row = tests[tests["subset"] == subset].iloc[0]
            handle.write(f"{subset}\n")
            handle.write("-" * len(subset) + "\n")
            handle.write(f"n birds: {int(row['n_birds'])}\n")
            handle.write(f"median Pre BC: {row['median_pre_bc']:.8g}\n")
            handle.write(f"median Post BC: {row['median_post_bc']:.8g}\n")
            handle.write(f"median DeltaBC (Post - Pre): {row['median_delta_bc_post_minus_pre']:.8g}\n")
            handle.write(f"two-sided Wilcoxon raw p: {row['wilcoxon_two_sided_p']:.8g}\n")
            handle.write(f"two-sided Wilcoxon Holm p: {row['wilcoxon_two_sided_holm_p']:.8g}\n")
            handle.write(
                f"one-sided Wilcoxon raw p (Post < Pre): "
                f"{row['wilcoxon_one_sided_p_post_less_pre']:.8g}\n"
            )
            handle.write(
                f"one-sided Wilcoxon Holm p: {row['wilcoxon_one_sided_holm_p']:.8g}\n"
            )
            handle.write(f"two-sided exact sign-flip raw p: {row['signflip_two_sided_p']:.8g}\n")
            handle.write(
                f"one-sided exact sign-flip raw p (Post < Pre): "
                f"{row['signflip_one_sided_p_post_less_pre']:.8g}\n\n"
            )

        interaction = tests[tests["subset"] == "DeltaBC contrast"].iloc[0]
        handle.write("Selected-vs-remaining DeltaBC contrast\n")
        handle.write("--------------------------------------\n")
        handle.write(
            "Mean [Top 15% DeltaBC - Remaining 85% DeltaBC]: "
            f"{interaction['mean_delta_bc_post_minus_pre']:.8g}\n"
        )
        handle.write(
            "One-sided exact sign-flip p (Top 15% more negative): "
            f"{interaction['signflip_one_sided_p_post_less_pre']:.8g}\n"
        )
        handle.write(
            "One-sided paired Wilcoxon p (Top 15% more negative): "
            f"{interaction['wilcoxon_one_sided_p_post_less_pre']:.8g}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot and test Pre vs Post BC for the top 15% and remaining 85% "
            "syllables in pooled medial+lateral lesion birds."
        )
    )
    parser.add_argument("--bc-csv", required=True)
    parser.add_argument("--selected-pairs-csv", required=True)
    parser.add_argument("--metadata-excel", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = read_selected_pairs(args.selected_pairs_csv)
    bc, identifier_report = read_bc_table(
        args.bc_csv,
        args.metadata_excel,
        selected,
    )
    identifier_report.to_csv(
        out_dir / "figure4_bc_identifier_column_match_report.csv",
        index=False,
    )

    merged = bc.merge(
        selected.assign(is_top15=True),
        on=["animal_id", "syllable"],
        how="left",
    )
    merged["is_top15"] = merged["is_top15"].fillna(False).astype(bool)
    merged["subset"] = np.where(merged["is_top15"], "selected", "remaining")
    merged.to_csv(out_dir / "figure4_top15_prepost_bc_cluster_rows.csv", index=False)

    # Audit selected-token matching. This catches accidental cluster_id/token
    # mismatches and shows why a bird is or is not retained.
    requested = (
        selected.groupby("animal_id")["syllable"]
        .nunique()
        .rename("n_top15_requested")
    )
    matched = (
        merged.loc[merged["is_top15"]]
        .groupby("animal_id")["syllable"]
        .nunique()
        .rename("n_top15_matched_in_bc")
    )
    available = (
        merged.groupby("animal_id")["syllable"]
        .nunique()
        .rename("n_bc_syllables_available")
    )
    merge_audit = (
        pd.concat([requested, matched, available], axis=1)
        .fillna(0)
        .reset_index()
    )
    merge_audit["n_top15_unmatched"] = (
        merge_audit["n_top15_requested"] - merge_audit["n_top15_matched_in_bc"]
    )
    merge_audit.to_csv(
        out_dir / "figure4_top15_bc_token_merge_audit.csv", index=False
    )

    selected_matches = int(merged["is_top15"].sum())
    selected_birds = int(
        merged.loc[merged["is_top15"], "animal_id"].nunique()
    )
    print(
        f"[INFO] Matched {selected_matches} BC rows labeled Top 15% "
        f"across {selected_birds} birds"
    )

    by_bird = make_bird_table(merged)
    by_bird.to_csv(out_dir / "figure4_top15_prepost_bc_by_bird.csv", index=False)

    tests = run_tests(by_bird)
    tests.to_csv(out_dir / "figure4_top15_prepost_bc_tests.csv", index=False)

    plot_path = out_dir / "Figure4_top15_prepost_BC_ML_Fig3colors.png"
    plot_panel(by_bird, tests, plot_path, args.dpi, args.show)

    summary_path = out_dir / "figure4_top15_prepost_bc_summary.txt"
    write_text_summary(summary_path, args.bc_csv, args.selected_pairs_csv, by_bird, tests)

    print(f"[OK] Pooled M+L birds contributing: {len(by_bird)}")
    print(f"[OK] Wrote outputs to {out_dir}")
    for filename in [
        plot_path.name,
        "figure4_top15_prepost_bc_cluster_rows.csv",
        "figure4_bc_identifier_column_match_report.csv",
        "figure4_top15_prepost_bc_by_bird.csv",
        "figure4_top15_bc_token_merge_audit.csv",
        "figure4_top15_prepost_bc_tests.csv",
        summary_path.name,
    ]:
        print(" ", out_dir / filename)


if __name__ == "__main__":
    main()
