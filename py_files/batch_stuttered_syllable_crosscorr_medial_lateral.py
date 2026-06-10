#!/usr/bin/env python3
"""
batch_stuttered_syllable_crosscorr_medial_lateral.py

Batch wrapper for cluster_spectrogram_crosscorr_canary_segmentation_v1.py.

Goal
----
For birds with Medial + Lateral lesion hit types:
1) Find the syllable/cluster labels with the largest post-lesion increase in
   phrase-duration variance. These are treated as the most "stuttered" labels.
2) Run the single-bird/single-label spectrogram cross-correlation script on those labels.
3) Compile template-correlation and within-condition pairwise-correlation stats
   across birds/labels.
4) Save summary CSVs and simple comparison plots.

Expected inputs
---------------
- Metadata Excel file with Animal ID, Treatment date, and lesion hit-type columns.
- Phrase-duration summary CSV with one row per animal x syllable x period, ideally
  including PostMinusPreVar_ms2 for Post rows.
- NPZ root containing one .npz per bird, usually named like USA5288.npz.
- The existing single-label cross-correlation script.

Notes
-----
The wrapper forwards unknown command-line options to the single-label cross-corr
script. That means you can pass options like --corr-time-bins, --corr-freq-min-khz,
--corr-freq-max-khz, --pairwise-max-segments-per-group, etc. after the wrapper args.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# General helper functions
# -----------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_col(s: object) -> str:
    """Normalize a column name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def label_to_str(x: object) -> str:
    """Convert labels like 17.0 to '17', while preserving string labels."""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)) and np.isfinite(x) and float(x).is_integer():
        return str(int(x))
    s = str(x).strip()
    # Handle CSV-loaded numeric labels like "17.0"
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def truthy_hit_value(x: object) -> bool:
    """Interpret metadata hit flag values such as Y/Yes/True/1."""
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"y", "yes", "true", "t", "1", "hit", "complete", "partial"}


def safe_name(s: object) -> str:
    """Make a string safe for paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def find_column(df: pd.DataFrame, requested: Optional[str], candidates: Iterable[str], required: bool = True) -> Optional[str]:
    """Find a column by exact name, normalized exact name, then candidate list."""
    cols = list(df.columns)
    if requested:
        if requested in cols:
            return requested
        requested_norm = norm_col(requested)
        for c in cols:
            if norm_col(c) == requested_norm:
                return c
        if required:
            raise ValueError(f"Requested column not found: {requested!r}. Available columns: {cols}")
        return None

    norm_to_col = {norm_col(c): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        nc = norm_col(cand)
        if nc in norm_to_col:
            return norm_to_col[nc]

    if required:
        raise ValueError(f"Could not find any of these columns: {list(candidates)}. Available columns: {cols}")
    return None


def auto_hit_type_columns(meta: pd.DataFrame) -> list[str]:
    """Find likely lesion hit-type columns in the metadata."""
    out = []
    for c in meta.columns:
        n = norm_col(c)
        # Catches columns like "Medial/Lateral hit type", "Medial hit type",
        # "Lateral hit type", "Lesion hit type", "Area X hit type".
        if ("hittype" in n) or (("hit" in n) and ("type" in n)) or (("lesion" in n) and ("type" in n)):
            out.append(c)
    return out


def read_metadata_animals(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """Read metadata and select animals matching Medial + Lateral hit type."""
    meta = pd.read_excel(args.metadata_excel_path, sheet_name=args.metadata_sheet)
    animal_col = find_column(meta, args.animal_id_col, ["Animal ID", "animal_id", "AnimalID"])

    if args.hit_type_cols:
        hit_type_cols = []
        for c in args.hit_type_cols:
            hit_type_cols.append(find_column(meta, c, [], required=True))
    else:
        hit_type_cols = auto_hit_type_columns(meta)

    selected_mask = pd.Series(False, index=meta.index)
    hit_regex = re.compile(args.hit_type_regex, flags=re.IGNORECASE)

    if hit_type_cols:
        hit_text = meta[hit_type_cols].astype(str).agg(" | ".join, axis=1)
        selected_mask = hit_text.str.contains(hit_regex, na=False)
        print(f"[INFO] Using hit-type columns for Medial + Lateral filter: {hit_type_cols}")
    else:
        print("[WARN] No hit-type columns auto-detected. Trying medial/lateral hit flag fallback.")

    # Optional/fallback: require both medial and lateral hit flags if those columns exist.
    medial_flag_col = find_column(
        meta,
        args.medial_hit_flag_col,
        ["Medial Area X hit?", "Medial hit?", "Medial hit", "Medial Area X hit"],
        required=False,
    )
    lateral_flag_col = find_column(
        meta,
        args.lateral_hit_flag_col,
        ["Lateral Area X hit?", "Lateral hit?", "Lateral hit", "Lateral Area X hit"],
        required=False,
    )
    if args.also_accept_both_hit_flags and medial_flag_col and lateral_flag_col:
        flag_mask = meta[medial_flag_col].map(truthy_hit_value) & meta[lateral_flag_col].map(truthy_hit_value)
        selected_mask = selected_mask | flag_mask
        print(f"[INFO] Also accepting rows with both hit flags: {medial_flag_col!r} and {lateral_flag_col!r}")

    selected = meta.loc[selected_mask].copy()
    selected[animal_col] = selected[animal_col].astype(str)
    selected_animals = sorted(selected[animal_col].dropna().astype(str).unique().tolist())
    if not selected_animals:
        raise ValueError(
            "No animals matched the Medial + Lateral filter. Try passing --hit-type-cols explicitly, "
            "or adjust --hit-type-regex."
        )
    print(f"[INFO] Selected {len(selected_animals)} Medial + Lateral animal(s): {', '.join(selected_animals)}")
    return selected, selected_animals


# -----------------------------
# Finding stuttered labels
# -----------------------------


def find_period_col(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    return find_column(
        df,
        requested,
        ["Group", "group", "Period", "period", "Epoch", "epoch", "Lesion group", "Timepoint"],
        required=False,
    )


def is_post_value(x: object) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    return s in {"post", "post lesion", "postlesion", "after", "after lesion"} or s.startswith("post")


def find_variance_delta_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    candidates = [
        "PostMinusPreVar_ms2",
        "PostMinusPreVariance_ms2",
        "PostMinusPreVariance",
        "PostMinusPreVar",
        "delta_variance_ms2",
        "DeltaVariance_ms2",
        "post_minus_pre_variance_ms2",
        "post_minus_pre_var_ms2",
        "Post - Pre Variance ms2",
        "PostMinusPreVar_ms^2",
    ]
    col = find_column(df, requested, candidates, required=False)
    if col:
        return col

    # Fuzzy fallback: any column containing post, pre, and var.
    for c in df.columns:
        n = norm_col(c)
        if "post" in n and "pre" in n and "var" in n:
            return c
    return None


def compute_variance_delta_if_needed(df: pd.DataFrame, animal_col: str, label_col: str, period_col: Optional[str]) -> pd.DataFrame:
    """
    Try to create PostMinusPreVar_ms2 if the phrase CSV does not already have it.
    Preferred fallback: Post Variance_ms2 - PrePooledVariance_ms2 on Post rows.
    Secondary fallback: pivot period rows and compute Post - Late Pre or Post - Pre.
    """
    out = df.copy()
    variance_col = find_column(
        out,
        None,
        ["Variance_ms2", "Variance_ms^2", "Variance", "Var_ms2", "variance_ms2", "variance"],
        required=False,
    )
    prepooled_col = find_column(
        out,
        None,
        ["PrePooledVariance_ms2", "PrePooledVar_ms2", "PrePooledVariance", "Pre pooled variance ms2"],
        required=False,
    )

    if variance_col and prepooled_col:
        out["PostMinusPreVar_ms2_auto"] = pd.to_numeric(out[variance_col], errors="coerce") - pd.to_numeric(out[prepooled_col], errors="coerce")
        return out

    if not (variance_col and period_col):
        raise ValueError(
            "Could not find a variance-delta column and could not compute it. "
            "Expected PostMinusPreVar_ms2, or Variance_ms2 plus PrePooledVariance_ms2, "
            "or period rows that can be pivoted."
        )

    tmp = out[[animal_col, label_col, period_col, variance_col]].copy()
    tmp["period_norm"] = tmp[period_col].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False).str.replace("-", " ", regex=False)
    tmp[variance_col] = pd.to_numeric(tmp[variance_col], errors="coerce")
    piv = tmp.pivot_table(index=[animal_col, label_col], columns="period_norm", values=variance_col, aggfunc="mean")

    post_candidates = [c for c in piv.columns if str(c).startswith("post")]
    pre_candidates_ordered = ["late pre", "late pre lesion", "pre", "pre lesion", "early pre"]
    pre_candidates = [c for c in pre_candidates_ordered if c in piv.columns]
    if not post_candidates or not pre_candidates:
        raise ValueError(
            f"Could not pivot variance delta. Found periods: {list(piv.columns)}. "
            "Need a Post period and a pre-like period."
        )

    post_c = post_candidates[0]
    pre_c = pre_candidates[0]
    delta = (piv[post_c] - piv[pre_c]).rename("PostMinusPreVar_ms2_auto").reset_index()
    out = out.merge(delta, on=[animal_col, label_col], how="left")
    print(f"[INFO] Computed variance delta as {post_c!r} - {pre_c!r} from {variance_col!r}.")
    return out


def read_stuttered_labels(args: argparse.Namespace, selected_animals: list[str]) -> pd.DataFrame:
    phrase = pd.read_csv(args.phrase_duration_csv)
    phrase_animal_col = find_column(
        phrase,
        args.phrase_animal_id_col,
        ["Animal ID", "animal_id", "AnimalID", "animal"],
    )
    label_col = find_column(
        phrase,
        args.label_col,
        ["label", "Label", "cluster_label", "Cluster label", "syllable", "Syllable", "syllable_label", "Syllable label"],
    )
    period_col = find_period_col(phrase, args.period_col)
    delta_col = find_variance_delta_column(phrase, args.variance_delta_col)

    phrase[phrase_animal_col] = phrase[phrase_animal_col].astype(str)
    phrase[label_col] = phrase[label_col].map(label_to_str)

    if delta_col is None:
        phrase = compute_variance_delta_if_needed(phrase, phrase_animal_col, label_col, period_col)
        delta_col = "PostMinusPreVar_ms2_auto"

    # Usually PostMinusPreVar_ms2 is filled on Post rows only. Restrict to Post rows when possible.
    if period_col:
        post_like = phrase[period_col].map(is_post_value)
        if post_like.any():
            phrase = phrase.loc[post_like].copy()

    phrase[delta_col] = pd.to_numeric(phrase[delta_col], errors="coerce")
    phrase = phrase[phrase[phrase_animal_col].isin(selected_animals)].copy()
    phrase = phrase[np.isfinite(phrase[delta_col])].copy()

    if not args.allow_nonpositive_variance_delta:
        phrase = phrase[phrase[delta_col] > args.min_variance_delta].copy()
    else:
        phrase = phrase[phrase[delta_col] >= args.min_variance_delta].copy()

    if phrase.empty:
        raise ValueError(
            "No phrase-duration rows remained after filtering selected animals and variance deltas. "
            "Check --phrase-duration-csv, --label-col, --variance-delta-col, and --min-variance-delta."
        )

    # Collapse duplicates to one row per animal x label using the largest variance increase.
    collapsed = (
        phrase.groupby([phrase_animal_col, label_col], as_index=False)[delta_col]
        .max()
        .rename(columns={phrase_animal_col: "animal_id", label_col: "cluster_label", delta_col: "phrase_variance_delta_ms2"})
    )

    collapsed["stutter_rank_within_bird"] = np.nan
    pieces = []
    for animal, sub in collapsed.groupby("animal_id", sort=True):
        sub = sub.sort_values("phrase_variance_delta_ms2", ascending=False).copy()
        sub["stutter_rank_within_bird"] = np.arange(1, len(sub) + 1)
        pieces.append(sub.head(args.top_n_stuttered))
    top = pd.concat(pieces, ignore_index=True)

    print(f"[INFO] Selected {len(top)} top stuttered animal/label pair(s) from phrase-duration variance.")
    return top


# -----------------------------
# NPZ and subprocess running
# -----------------------------


def find_npz_for_animal(npz_root: str | Path, animal_id: str, template: str) -> Path:
    root = Path(npz_root)
    if not root.exists():
        raise FileNotFoundError(f"NPZ root does not exist: {root}")

    pattern = template.format(animal_id=animal_id)
    candidates = sorted(root.rglob(pattern))
    if not candidates:
        candidates = sorted(root.rglob(f"*{animal_id}*.npz"))

    candidates = [p for p in candidates if p.is_file() and not p.name.startswith("._")]
    if not candidates:
        raise FileNotFoundError(f"Could not find an NPZ for {animal_id} under {root}")

    # Prefer exact basename match, then shorter path.
    exact = [p for p in candidates if p.name == f"{animal_id}.npz"]
    if exact:
        candidates = exact
    candidates = sorted(candidates, key=lambda p: (len(str(p)), str(p)))
    if len(candidates) > 1:
        print(f"[WARN] Multiple NPZ files found for {animal_id}. Using: {candidates[0]}")
    return candidates[0]


def stats_file_done(label_out: Path, animal_id: str, label: str) -> bool:
    return (
        label_out / f"{animal_id}_label{label}_template_correlation_stats.csv"
    ).exists() and (
        label_out / f"{animal_id}_label{label}_pairwise_correlation_stats.csv"
    ).exists()


def build_run_command(
    args: argparse.Namespace,
    unknown_args: list[str],
    animal_id: str,
    label: str,
    npz_path: Path,
    label_out: Path,
) -> list[str]:
    cmd = [
        args.python_executable,
        str(args.crosscorr_script),
        "--npz-path", str(npz_path),
        "--cluster-label", str(label),
        "--animal-id", str(animal_id),
        "--metadata-excel-path", str(args.metadata_excel_path),
        "--metadata-sheet", str(args.metadata_sheet),
        "--animal-id-col", str(args.animal_id_col),
        "--treatment-date-col", str(args.treatment_date_col),
        "--out-dir", str(label_out),
    ]
    cmd.extend(unknown_args)
    return cmd


def run_crosscorr_jobs(args: argparse.Namespace, unknown_args: list[str], plan: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for _, rec in plan.iterrows():
        animal_id = str(rec["animal_id"])
        label = label_to_str(rec["cluster_label"])
        animal_dir = ensure_dir(out_dir / animal_id)
        label_out = ensure_dir(animal_dir / f"label_{safe_name(label)}")
        log_path = label_out / "crosscorr_run.log"

        try:
            npz_path = find_npz_for_animal(args.npz_root, animal_id, args.npz_glob_template)
            cmd = build_run_command(args, unknown_args, animal_id, label, npz_path, label_out)
            cmd_text = " ".join(shlex.quote(x) for x in cmd)

            if args.skip_existing and stats_file_done(label_out, animal_id, label):
                status = "skipped_existing"
                print(f"[SKIP] {animal_id} label {label}: stats already exist")
            elif args.dry_run:
                status = "dry_run"
                print(f"[DRY RUN] {cmd_text}")
            else:
                print(f"[RUN] {animal_id} label {label}")
                with open(log_path, "w") as log:
                    log.write("COMMAND:\n" + cmd_text + "\n\n")
                    log.flush()
                    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
                if proc.returncode == 0:
                    status = "completed"
                    print(f"[DONE] {animal_id} label {label}")
                else:
                    status = f"failed_returncode_{proc.returncode}"
                    print(f"[ERROR] {animal_id} label {label} failed. See {log_path}")
                    if args.stop_on_error:
                        raise RuntimeError(f"Job failed: {animal_id} label {label}. See {log_path}")

            rows.append({
                "animal_id": animal_id,
                "cluster_label": label,
                "npz_path": str(npz_path),
                "out_dir": str(label_out),
                "log_path": str(log_path),
                "run_status": status,
                "command": cmd_text,
            })
        except Exception as exc:
            print(f"[ERROR] {animal_id} label {label}: {exc}")
            rows.append({
                "animal_id": animal_id,
                "cluster_label": label,
                "npz_path": "",
                "out_dir": str(label_out),
                "log_path": str(log_path),
                "run_status": "setup_failed",
                "error": str(exc),
                "command": "",
            })
            if args.stop_on_error:
                raise

    return pd.DataFrame(rows)


# -----------------------------
# Collecting stats and plotting
# -----------------------------


def read_one_stats_csv(path: Path, prefix: str) -> dict:
    if not path.exists():
        return {f"{prefix}_stats_found": False}
    df = pd.read_csv(path)
    if df.empty:
        return {f"{prefix}_stats_found": False}
    row = df.iloc[0].to_dict()
    out = {f"{prefix}_stats_found": True}
    for k, v in row.items():
        out[f"{prefix}_{k}"] = v
    return out


def collect_outputs(plan: pd.DataFrame, run_df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = plan.merge(run_df, on=["animal_id", "cluster_label"], how="left")
    rows = []
    for _, rec in merged.iterrows():
        animal_id = str(rec["animal_id"])
        label = label_to_str(rec["cluster_label"])
        label_out = Path(str(rec["out_dir"])) if pd.notna(rec.get("out_dir")) else out_dir / animal_id / f"label_{safe_name(label)}"

        template_stats = read_one_stats_csv(
            label_out / f"{animal_id}_label{label}_template_correlation_stats.csv",
            "template",
        )
        pairwise_stats = read_one_stats_csv(
            label_out / f"{animal_id}_label{label}_pairwise_correlation_stats.csv",
            "pairwise",
        )

        row = rec.to_dict()
        row.update(template_stats)
        row.update(pairwise_stats)

        # Friendly short names for the most important effect sizes.
        row["template_delta_median_post_minus_pre"] = row.get("template_delta_median_post_minus_pre", np.nan)
        if pd.isna(row["template_delta_median_post_minus_pre"]):
            row["template_delta_median_post_minus_pre"] = row.get("template_delta_median_post_minus_pre", row.get("template_delta_median_post_minus_pre", np.nan))
        # The single-label script stores this exact long name under each prefix.
        row["template_delta_median"] = row.get("template_delta_median_post_minus_pre", row.get("template_delta_median_post_minus_pre", np.nan))
        row["pairwise_delta_median"] = row.get("pairwise_delta_median_post_minus_pre", row.get("pairwise_delta_median_post_minus_pre", np.nan))

        # More robust extraction of the prefixed stat name.
        if "template_delta_median_post_minus_pre" not in row or pd.isna(row.get("template_delta_median")):
            row["template_delta_median"] = row.get("template_delta_median_post_minus_pre", np.nan)
        if "pairwise_delta_median_post_minus_pre" not in row or pd.isna(row.get("pairwise_delta_median")):
            row["pairwise_delta_median"] = row.get("pairwise_delta_median_post_minus_pre", np.nan)

        rows.append(row)

    summary = pd.DataFrame(rows)

    # The above names are correct if the input stats column was delta_median_post_minus_pre.
    # Keep a final fallback in case pandas/object dtype or future script names differ.
    for metric_prefix, short_col in [("template", "template_delta_median"), ("pairwise", "pairwise_delta_median")]:
        wanted = f"{metric_prefix}_delta_median_post_minus_pre"
        if wanted in summary.columns:
            summary[short_col] = pd.to_numeric(summary[wanted], errors="coerce")
        else:
            possible = [c for c in summary.columns if c.startswith(metric_prefix) and "delta" in c and "median" in c]
            if possible:
                summary[short_col] = pd.to_numeric(summary[possible[0]], errors="coerce")
            else:
                summary[short_col] = np.nan

    numeric_cols = [
        "phrase_variance_delta_ms2",
        "stutter_rank_within_bird",
        "template_delta_median",
        "pairwise_delta_median",
    ]
    for c in numeric_cols:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce")

    summary_csv = out_dir / "all_birds_medial_lateral_stuttered_syllable_crosscorr_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[SAVED] {summary_csv}")

    ok = summary[summary["run_status"].isin(["completed", "skipped_existing"])].copy()
    if not ok.empty:
        bird_summary = ok.groupby("animal_id", as_index=False).agg(
            n_labels_analyzed=("cluster_label", "count"),
            median_phrase_variance_delta_ms2=("phrase_variance_delta_ms2", "median"),
            median_template_delta_median=("template_delta_median", "median"),
            median_pairwise_delta_median=("pairwise_delta_median", "median"),
            mean_template_delta_median=("template_delta_median", "mean"),
            mean_pairwise_delta_median=("pairwise_delta_median", "mean"),
        )
    else:
        bird_summary = pd.DataFrame()

    bird_csv = out_dir / "bird_level_crosscorr_summary.csv"
    bird_summary.to_csv(bird_csv, index=False)
    print(f"[SAVED] {bird_csv}")
    return summary, bird_summary


def finite_xy(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    out = df.copy()
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    return out[np.isfinite(out[x_col]) & np.isfinite(out[y_col])].copy()


def plot_metric_by_bird(summary: pd.DataFrame, metric_col: str, ylabel: str, out_png: Path) -> None:
    df = summary[np.isfinite(pd.to_numeric(summary[metric_col], errors="coerce"))].copy()
    if df.empty:
        print(f"[WARN] No finite values for {metric_col}; skipping {out_png.name}")
        return
    animals = sorted(df["animal_id"].astype(str).unique())
    x_lookup = {a: i for i, a in enumerate(animals)}

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(animals) + 3), 5.5))
    for _, row in df.iterrows():
        x = x_lookup[str(row["animal_id"])]
        rank = row.get("stutter_rank_within_bird", np.nan)
        jitter = 0.0 if not np.isfinite(rank) else (float(rank) - 1.5) * 0.06
        ax.scatter(x + jitter, row[metric_col], s=45, alpha=0.8)
        ax.text(x + jitter, row[metric_col], str(row["cluster_label"]), fontsize=7, ha="center", va="bottom")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels(animals, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("Post - pre spectrogram similarity change for top variance-increase syllables")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_delta_vs_variance(summary: pd.DataFrame, metric_col: str, ylabel: str, out_png: Path) -> None:
    df = finite_xy(summary, "phrase_variance_delta_ms2", metric_col)
    if df.empty:
        print(f"[WARN] No finite values for {metric_col}; skipping {out_png.name}")
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.scatter(df["phrase_variance_delta_ms2"], df[metric_col], s=55, alpha=0.8)
    for _, row in df.iterrows():
        ax.text(
            row["phrase_variance_delta_ms2"],
            row[metric_col],
            f"{row['animal_id']}:{row['cluster_label']}",
            fontsize=7,
            ha="left",
            va="bottom",
        )
    if (df["phrase_variance_delta_ms2"] > 0).all():
        ax.set_xscale("log")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Post - pre phrase-duration variance increase (ms²)")
    ax.set_ylabel(ylabel)
    ax.set_title("Do the most stuttered syllables show reduced spectrogram similarity?")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_template_vs_pairwise(summary: pd.DataFrame, out_png: Path) -> None:
    df = finite_xy(summary, "template_delta_median", "pairwise_delta_median")
    if df.empty:
        print(f"[WARN] No finite values for template vs pairwise scatter; skipping {out_png.name}")
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(df["template_delta_median"], df["pairwise_delta_median"], s=55, alpha=0.8)
    for _, row in df.iterrows():
        ax.text(
            row["template_delta_median"],
            row["pairwise_delta_median"],
            f"{row['animal_id']}:{row['cluster_label']}",
            fontsize=7,
            ha="left",
            va="bottom",
        )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Δ median correlation to late-pre template\n(post - pre)")
    ax.set_ylabel("Δ median within-condition pairwise correlation\n(post - pre)")
    ax.set_title("Template shift vs within-condition stereotypy change")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_summary_plots(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    plot_metric_by_bird(
        summary,
        "template_delta_median",
        "Δ median correlation to late-pre template (post - pre)",
        out_dir / "template_corr_delta_by_bird_stuttered_syllables.png",
    )
    plot_metric_by_bird(
        summary,
        "pairwise_delta_median",
        "Δ median within-condition pairwise correlation (post - pre)",
        out_dir / "pairwise_corr_delta_by_bird_stuttered_syllables.png",
    )
    plot_delta_vs_variance(
        summary,
        "template_delta_median",
        "Δ median correlation to late-pre template (post - pre)",
        out_dir / "phrase_variance_delta_vs_template_corr_delta.png",
    )
    plot_delta_vs_variance(
        summary,
        "pairwise_delta_median",
        "Δ median within-condition pairwise correlation (post - pre)",
        out_dir / "phrase_variance_delta_vs_pairwise_corr_delta.png",
    )
    plot_template_vs_pairwise(summary, out_dir / "template_delta_vs_pairwise_delta_stuttered_syllables.png")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Run spectrogram cross-correlation across Medial + Lateral birds for top stuttered syllables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--crosscorr-script", required=True, type=Path,
                   help="Path to cluster_spectrogram_crosscorr_canary_segmentation_v1.py")
    p.add_argument("--metadata-excel-path", required=True, type=Path)
    p.add_argument("--phrase-duration-csv", required=True, type=Path,
                   help="CSV with phrase-duration variance summaries, ideally usage_balanced_phrase_duration_stats.csv")
    p.add_argument("--npz-root", required=True, type=Path,
                   help="Root folder containing bird NPZ files")
    p.add_argument("--out-dir", required=True, type=Path)

    p.add_argument("--metadata-sheet", default="metadata")
    p.add_argument("--animal-id-col", default="Animal ID")
    p.add_argument("--treatment-date-col", default="Treatment date")

    p.add_argument("--hit-type-cols", nargs="*", default=None,
                   help="Metadata column(s) to search for Medial + Lateral. If omitted, likely hit-type columns are auto-detected.")
    p.add_argument("--hit-type-regex", default=r"medial\s*\+\s*lateral|medial.*lateral|lateral.*medial",
                   help="Regex used on concatenated hit-type column text.")
    p.add_argument("--also-accept-both-hit-flags", action="store_true", default=True,
                   help="Also include rows where both medial and lateral hit flag columns are truthy, if found.")
    p.add_argument("--medial-hit-flag-col", default=None)
    p.add_argument("--lateral-hit-flag-col", default=None)

    p.add_argument("--phrase-animal-id-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--period-col", default=None)
    p.add_argument("--variance-delta-col", default=None,
                   help="Column giving Post - Pre variance, e.g. PostMinusPreVar_ms2. Auto-detected if omitted.")
    p.add_argument("--top-n-stuttered", type=int, default=3,
                   help="Number of top variance-increase labels to analyze per bird.")
    p.add_argument("--min-variance-delta", type=float, default=0.0)
    p.add_argument("--allow-nonpositive-variance-delta", action="store_true",
                   help="Allow zero/negative variance deltas when ranking labels. Usually leave off.")

    p.add_argument("--npz-glob-template", default="{animal_id}.npz",
                   help="Search pattern under npz-root. {animal_id} is replaced for each bird.")
    p.add_argument("--python-executable", default=sys.executable)

    p.add_argument("--skip-existing", action="store_true",
                   help="Skip animal/label jobs whose two stats CSVs already exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands and write the run plan, but do not run jobs.")
    p.add_argument("--stop-on-error", action="store_true")

    args, unknown_args = p.parse_known_args()
    return args, unknown_args


def main() -> None:
    args, unknown_args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    if unknown_args:
        print("[INFO] Forwarding these extra args to the single-label crosscorr script:")
        print("       " + " ".join(shlex.quote(x) for x in unknown_args))

    selected_meta, selected_animals = read_metadata_animals(args)
    selected_meta_csv = out_dir / "selected_medial_lateral_animals_from_metadata.csv"
    selected_meta.to_csv(selected_meta_csv, index=False)
    print(f"[SAVED] {selected_meta_csv}")

    plan = read_stuttered_labels(args, selected_animals)
    # Add NPZ path preview where possible.
    npz_paths = []
    for animal in plan["animal_id"].astype(str):
        try:
            npz_paths.append(str(find_npz_for_animal(args.npz_root, animal, args.npz_glob_template)))
        except Exception as exc:
            npz_paths.append(f"NOT_FOUND: {exc}")
    plan["npz_path_preview"] = npz_paths

    plan_csv = out_dir / "stuttered_syllables_to_analyze.csv"
    plan.to_csv(plan_csv, index=False)
    print(f"[SAVED] {plan_csv}")

    run_df = run_crosscorr_jobs(args, unknown_args, plan, out_dir)
    run_csv = out_dir / "crosscorr_batch_run_log.csv"
    run_df.to_csv(run_csv, index=False)
    print(f"[SAVED] {run_csv}")

    if not args.dry_run:
        summary, bird_summary = collect_outputs(plan, run_df, out_dir)
        make_summary_plots(summary, out_dir)

    print("[DONE]")


if __name__ == "__main__":
    main()
