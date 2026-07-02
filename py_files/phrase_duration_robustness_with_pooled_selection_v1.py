#!/usr/bin/env python3
"""
phrase_duration_robustness_with_pooled_selection_v1.py

Purpose
-------
Run Figure 3 phrase-duration robustness analyses, including a pooled pre+post
variability syllable-selection check.

This script:
  1. Patches the Figure 3 script to support:
        --rank-on pooled
     where pooled rank_metric = mean(pre_variance, post_variance).

  2. Runs these robustness analyses:
        post-selected top 20%
        post-selected top 30%
        post-selected top 40%
        pooled pre+post-selected top 30%
        pre-selected top 30% validation

  3. Builds a summary table with partial M+L, complete M+L, and pooled M+L rows.

Default paths are set for Miranda Hulsey-Vincent's AFP_lesion_paper repo and SSD.
All paths can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_REPO_DIR = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/AFP_lesion_paper")
DEFAULT_ROOT = Path("/Volumes/my_own_SSD/updated_AreaX_outputs")
DEFAULT_OUT_NAME = "phrase_duration_robustness_2026-07-01"
DEFAULT_FIGURE3_SCRIPT = (
    "py_files/"
    "panel_C_D_sdscatter_boxplots_sdtimecourse_combined_FINAL_layouttuned_splitrow_NOTITLE_CLEANBOX_legendabove.py"
)

RUNS = [
    ("post_selected_top20", "Post-selected top 20%", "post variability", "top 20%", 80, "post"),
    ("post_selected_top30", "Post-selected top 30%", "post variability", "top 30%", 70, "post"),
    ("post_selected_top40", "Post-selected top 40%", "post variability", "top 40%", 60, "post"),
    (
        "pooled_prepost_selected_top30",
        "Pooled pre+post-selected top 30%",
        "mean pre+post variability",
        "top 30%",
        70,
        "pooled",
    ),
    ("pre_selected_top30_validation", "Pre-selected top 30% validation", "pre variability", "top 30%", 70, "pre"),
]

ML_GROUPS = [
    "Partial Medial and Lateral lesion",
    "Complete Medial and Lateral lesion",
]
POOLED_ML = "Complete and partial medial and lateral lesion"

GROUP_ORDER = [
    "sham saline injection",
    "Lateral lesion only",
    "Partial Medial and Lateral lesion",
    "Complete Medial and Lateral lesion",
    POOLED_ML,
]


def _print_header(msg: str) -> None:
    line = "=" * len(msg)
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def patch_figure3_script(script_path: Path) -> bool:
    """Patch Figure 3 script so --rank-on pooled is accepted.

    Returns True if the file was changed, False if it already appeared patched.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find Figure 3 script: {script_path}")

    text = script_path.read_text(encoding="utf-8")

    already_has_pooled_logic = (
        "rank_on in {\"pooled\", \"pooled_mean\", \"mean\"}" in text
        or "rank_on in {'pooled', 'pooled_mean', 'mean'}" in text
    )
    already_has_pooled_choice = (
        'choices=["pre", "post", "max", "pooled"]' in text
        or "choices=['pre', 'post', 'max', 'pooled']" in text
    )

    if already_has_pooled_logic and already_has_pooled_choice:
        print(f"[OK] Figure 3 script already supports --rank-on pooled:\n  {script_path}")
        return False

    backup = script_path.with_suffix(script_path.suffix + ".before_pooled_rank_patch")
    if not backup.exists():
        shutil.copy2(script_path, backup)
        print(f"[OK] Backup written:\n  {backup}")

    changed = False

    old_block = '''        elif rank_on == "max":
            wide["rank_metric"] = wide[["pre_variance", "post_variance"]].max(axis=1)
        else:
            raise ValueError("rank_on must be one of: pre, post, max")
'''
    new_block = '''        elif rank_on == "max":
            wide["rank_metric"] = wide[["pre_variance", "post_variance"]].max(axis=1)
        elif rank_on in {"pooled", "pooled_mean", "mean"}:
            # Symmetric robustness check: rank syllables by their average
            # variability across Late Pre and Post rather than by Post alone.
            wide["rank_metric"] = wide[["pre_variance", "post_variance"]].mean(axis=1)
        else:
            raise ValueError("rank_on must be one of: pre, post, max, pooled")
'''

    if not already_has_pooled_logic:
        if old_block in text:
            text = text.replace(old_block, new_block)
            changed = True
        else:
            pattern = (
                r'(?P<indent>[ \t]+)elif rank_on == ["\']max["\']:\n'
                r'(?P=indent)[ \t]+wide\["rank_metric"\]\s*=\s*'
                r'wide\[\["pre_variance",\s*"post_variance"\]\]\.max\(axis=1\)\n'
                r'(?P=indent)else:\n'
                r'(?P=indent)[ \t]+raise ValueError\(["\']rank_on must be one of: pre, post, max["\']\)\n'
            )

            def repl(match: re.Match[str]) -> str:
                indent = match.group("indent")
                return (
                    f'{indent}elif rank_on == "max":\n'
                    f'{indent}    wide["rank_metric"] = wide[["pre_variance", "post_variance"]].max(axis=1)\n'
                    f'{indent}elif rank_on in {{"pooled", "pooled_mean", "mean"}}:\n'
                    f'{indent}    # Symmetric robustness check: rank syllables by their average\n'
                    f'{indent}    # variability across Late Pre and Post rather than by Post alone.\n'
                    f'{indent}    wide["rank_metric"] = wide[["pre_variance", "post_variance"]].mean(axis=1)\n'
                    f'{indent}else:\n'
                    f'{indent}    raise ValueError("rank_on must be one of: pre, post, max, pooled")\n'
                )

            text2, n = re.subn(pattern, repl, text)
            if n:
                text = text2
                changed = True
            else:
                raise RuntimeError(
                    "Could not find the rank_on block to patch. "
                    "Search the Figure 3 script for 'rank_on == \"max\"' and patch manually."
                )

    if not already_has_pooled_choice:
        before = text
        text = text.replace(
            'choices=["pre", "post", "max"], default="post"',
            'choices=["pre", "post", "max", "pooled"], default="post"',
        )
        text = text.replace(
            "choices=['pre', 'post', 'max'], default='post'",
            "choices=['pre', 'post', 'max', 'pooled'], default='post'",
        )
        if text == before:
            pattern = r'choices\s*=\s*\[\s*["\']pre["\']\s*,\s*["\']post["\']\s*,\s*["\']max["\']\s*\]'
            text2, n = re.subn(pattern, 'choices=["pre", "post", "max", "pooled"]', text)
            if n:
                text = text2
            else:
                raise RuntimeError(
                    "Could not patch argparse choices. "
                    "Search the Figure 3 script for choices=[\"pre\", \"post\", \"max\"]."
                )
        changed = True

    if changed:
        script_path.write_text(text, encoding="utf-8")
        print(f"[OK] Patched Figure 3 script:\n  {script_path}")
        subprocess.run([sys.executable, "-m", "py_compile", str(script_path)], check=True)
        print("[OK] Figure 3 script passes py_compile after patch.")

    return changed


def run_one_analysis(
    python_executable: str,
    figure3_script: Path,
    scatter_csv: Path,
    metadata_excel: Path,
    out_dir: Path,
    top_percentile_cutoff: int,
    rank_on: str,
) -> None:
    cmd = [
        python_executable,
        str(figure3_script),
        "--scatter-csv",
        str(scatter_csv),
        "--metadata-excel",
        str(metadata_excel),
        "--out-dir",
        str(out_dir),
        "--scatter-top-percentile",
        str(top_percentile_cutoff),
        "--rank-on",
        rank_on,
        "--stats-level",
        "animal",
        "--stats-alternative",
        "greater",
        "--no-show",
    ]

    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def one_sided_p_from_t(t_stat: float, p_two_sided: float) -> float:
    """Convert a two-sided t-test p-value to the one-sided p-value for t > 0."""
    if not np.isfinite(t_stat) or not np.isfinite(p_two_sided):
        return np.nan
    return p_two_sided / 2.0 if t_stat >= 0 else 1.0 - (p_two_sided / 2.0)


def make_summary_table(root_out_dir: Path) -> pd.DataFrame:
    rows = []

    for folder, method_label, selection_basis, threshold_label, _, _ in RUNS:
        run_dir = root_out_dir / folder
        metrics_path = run_dir / "panel_C_change_metrics_animal_level.csv"

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing animal-level metrics for {method_label}:\n  {metrics_path}\n"
                "Run the robustness analyses first, or remove --summary-only."
            )

        metrics = pd.read_csv(metrics_path)

        required = {"display_group", "pre_sd", "post_sd", "sd_delta", "sd_log2_ratio"}
        missing = required.difference(metrics.columns)
        if missing:
            raise ValueError(f"{metrics_path} is missing required columns: {sorted(missing)}")

        pooled = metrics[metrics["display_group"].isin(ML_GROUPS)].copy()
        pooled["display_group"] = POOLED_ML
        metrics_plus = pd.concat([metrics, pooled], ignore_index=True)

        for group in GROUP_ORDER:
            g = metrics_plus[metrics_plus["display_group"].astype(str) == group].copy()
            if g.empty:
                continue

            pre = g["pre_sd"].to_numpy(dtype=float)
            post = g["post_sd"].to_numpy(dtype=float)
            delta = g["sd_delta"].to_numpy(dtype=float)

            finite_pair_mask = np.isfinite(pre) & np.isfinite(post)
            pre_pair = pre[finite_pair_mask]
            post_pair = post[finite_pair_mask]
            delta_finite = delta[np.isfinite(delta)]

            if "Animal ID" in g.columns:
                n_animals = int(g["Animal ID"].nunique())
            elif "animal_id" in g.columns:
                n_animals = int(g["animal_id"].nunique())
            else:
                n_animals = int(len(g))

            if len(pre_pair) >= 2:
                paired = stats.ttest_rel(post_pair, pre_pair, nan_policy="omit")
                paired_t = float(paired.statistic)
                paired_p_two = float(paired.pvalue)
                paired_p = one_sided_p_from_t(paired_t, paired_p_two)
                paired_df = int(len(pre_pair) - 1)
            else:
                paired_t = paired_p = paired_df = np.nan

            if group != "sham saline injection":
                sham = metrics_plus[
                    metrics_plus["display_group"].astype(str) == "sham saline injection"
                ]["sd_delta"].to_numpy(dtype=float)
                sham = sham[np.isfinite(sham)]

                if len(delta_finite) >= 2 and len(sham) >= 2:
                    welch = stats.ttest_ind(delta_finite, sham, equal_var=False, nan_policy="omit")
                    welch_t = float(welch.statistic)
                    welch_p_two = float(welch.pvalue)
                    welch_p = one_sided_p_from_t(welch_t, welch_p_two)

                    vx = np.var(delta_finite, ddof=1)
                    vy = np.var(sham, ddof=1)
                    nx = len(delta_finite)
                    ny = len(sham)
                    sx = vx / nx
                    sy = vy / ny
                    welch_df = float((sx + sy) ** 2 / ((sx * sx) / (nx - 1) + (sy * sy) / (ny - 1)))
                else:
                    welch_t = welch_p = welch_df = np.nan
            else:
                welch_t = welch_p = welch_df = np.nan

            median_delta = float(np.nanmedian(delta))
            if median_delta > 0 and np.isfinite(paired_p) and paired_p < 0.05:
                interpretation = "post SD increased"
            elif median_delta > 0:
                interpretation = "post SD increased directionally"
            elif median_delta < 0:
                interpretation = "post SD decreased"
            else:
                interpretation = "no median change"

            rows.append(
                {
                    "selection_method": method_label,
                    "selection_basis": selection_basis,
                    "threshold": threshold_label,
                    "lesion_group": group,
                    "n_animals": n_animals,
                    "pre_median_SD_ms": float(np.nanmedian(pre)),
                    "post_median_SD_ms": float(np.nanmedian(post)),
                    "median_SD_delta_ms": median_delta,
                    "mean_SD_delta_ms": float(np.nanmean(delta)),
                    "median_SD_log2_ratio": float(np.nanmedian(g["sd_log2_ratio"])),
                    "paired_t_post_gt_pre": paired_t,
                    "paired_df": paired_df,
                    "paired_p_post_gt_pre": paired_p,
                    "welch_t_delta_gt_sham": welch_t,
                    "welch_df": welch_df,
                    "welch_p_delta_gt_sham": welch_p,
                    "interpretation": interpretation,
                }
            )

    out = pd.DataFrame(rows)
    method_rank = {label: i for i, (_, label, _, _, _, _) in enumerate(RUNS)}
    group_rank = {group: i for i, group in enumerate(GROUP_ORDER)}

    out["_method_rank"] = out["selection_method"].map(method_rank)
    out["_group_rank"] = out["lesion_group"].map(group_rank)
    out = (
        out.sort_values(["_method_rank", "_group_rank"])
        .drop(columns=["_method_rank", "_group_rank"])
        .reset_index(drop=True)
    )
    return out


def write_summary_table(summary: pd.DataFrame, root_out_dir: Path) -> tuple[Path, Path]:
    csv_out = root_out_dir / "phrase_duration_robustness_summary_table_with_pooledML_and_pooled_selection.csv"
    xlsx_out = root_out_dir / "phrase_duration_robustness_summary_table_with_pooledML_and_pooled_selection.xlsx"

    summary.to_csv(csv_out, index=False)
    summary.to_excel(xlsx_out, index=False)

    return csv_out, xlsx_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phrase-duration robustness analyses with pooled pre+post syllable selection."
    )
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Default: ROOT/phrase_duration_robustness_2026-07-01",
    )
    parser.add_argument("--scatter-csv", type=Path, default=None)
    parser.add_argument("--metadata-excel", type=Path, default=None)
    parser.add_argument("--figure3-script", type=Path, default=None)
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable to use when calling the Figure 3 script. Default: current Python.",
    )
    parser.add_argument("--patch-only", action="store_true", help="Only patch the Figure 3 script, then stop.")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild the summary table from existing outputs.")
    parser.add_argument("--skip-patch", action="store_true", help="Do not patch the Figure 3 script.")
    parser.add_argument("--skip-runs", action="store_true", help="Skip running Figure 3 analyses and only build summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_dir = args.repo_dir.expanduser().resolve()
    root = args.root.expanduser().resolve()
    out_dir = (args.out_dir or (root / DEFAULT_OUT_NAME)).expanduser().resolve()

    scatter_csv = args.scatter_csv or (root / "usage_balanced_phrase_duration_stats.csv")
    metadata_excel = args.metadata_excel or (root / "Area_X_lesion_metadata_with_hit_types.xlsx")

    if args.figure3_script is None:
        figure3_script = repo_dir / DEFAULT_FIGURE3_SCRIPT
    else:
        figure3_script = args.figure3_script
        if not figure3_script.is_absolute():
            figure3_script = repo_dir / figure3_script
        figure3_script = figure3_script.expanduser().resolve()

    _print_header("Phrase-duration robustness with pooled pre+post selection")

    print(f"repo_dir:        {repo_dir}")
    print(f"root:            {root}")
    print(f"out_dir:         {out_dir}")
    print(f"scatter_csv:     {scatter_csv}")
    print(f"metadata_excel:  {metadata_excel}")
    print(f"figure3_script:  {figure3_script}")
    print(f"python:          {args.python_executable}")

    if not args.summary_only and not args.skip_patch:
        _print_header("Patch Figure 3 script")
        patch_figure3_script(figure3_script)

    if args.patch_only:
        print("\n[OK] Patch-only mode complete.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only and not args.skip_runs:
        if not scatter_csv.exists():
            raise FileNotFoundError(f"Could not find scatter CSV: {scatter_csv}")
        if not metadata_excel.exists():
            raise FileNotFoundError(f"Could not find metadata Excel: {metadata_excel}")

        _print_header("Run robustness analyses")
        for folder, method_label, _, _, percentile_cutoff, rank_on in RUNS:
            run_out = out_dir / folder
            print(f"\n--- {method_label} ---")
            run_one_analysis(
                python_executable=args.python_executable,
                figure3_script=figure3_script,
                scatter_csv=scatter_csv,
                metadata_excel=metadata_excel,
                out_dir=run_out,
                top_percentile_cutoff=percentile_cutoff,
                rank_on=rank_on,
            )

    _print_header("Build summary table")
    summary = make_summary_table(out_dir)
    csv_out, xlsx_out = write_summary_table(summary, out_dir)

    print("\n[OK] Wrote summary table:")
    print(f"  {csv_out}")
    print(f"  {xlsx_out}")
    print()
    print(summary.to_string(index=False, max_colwidth=48))


if __name__ == "__main__":
    main()
