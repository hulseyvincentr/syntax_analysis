#!/usr/bin/env python3
"""
phrase_duration_hierarchical_resampling_from_renditions_v1.py

True rendition-level hierarchical resampling for the AFP phrase-duration analysis.
Use this when you have a long-format table with one row per phrase rendition.

What this addresses
-------------------
- Bird is the independent experimental unit.
- Renditions are resampled only within bird x syllable x epoch cells.
- Point estimates average across repeated usage-balanced draws.
- Bootstrap CIs resample birds and renditions.
- Within-group p-values use bird-level sign flips.
- Between-group p-values use bird-level label shuffling.
- Independent selection can be run with pre-lesion selection or held-out post-lesion selection.

Expected long-format input columns, by default
---------------------------------------------
Animal ID       bird identifier
Syllable        syllable / cluster label
epoch           epoch label; use --pre-epoch and --post-epoch to choose values
duration_ms     phrase duration in ms
hit_type        lesion hit type, used to assign groups

If your columns have different names, use --animal-col, --syllable-col, --epoch-col,
--duration-col, --hit-type-col, and --treatment-col.

Examples
--------
Pre-selected independent validation:
python phrase_duration_hierarchical_resampling_from_renditions_v1.py \
  --phrases-csv phrase_duration_long.csv \
  --out-dir out/pre_selected_top30 \
  --selection-mode pre \
  --top-frac 0.30 \
  --pre-epoch "Late Pre" \
  --post-epoch "Post" \
  --n-balance 200 --n-bootstrap 5000 --n-permutations 10000

Held-out post-lesion validation:
python phrase_duration_hierarchical_resampling_from_renditions_v1.py \
  --phrases-csv phrase_duration_long.csv \
  --out-dir out/heldout_post_top30 \
  --selection-mode heldout_post \
  --top-frac 0.30 \
  --pre-epoch "Late Pre" \
  --post-epoch "Post" \
  --n-balance 200 --n-bootstrap 5000 --n-permutations 10000
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SHAM = "sham saline injection"
LATERAL = "Lateral lesion only"
PARTIAL_ML = "Partial Medial and Lateral lesion"
COMPLETE_ML = "Complete Medial and Lateral lesion"
POOLED_ML = "Complete and partial medial and lateral lesion"
CONTRASTS = [
    (POOLED_ML, LATERAL, "primary specificity: medial+lateral > lateral-only"),
    (POOLED_ML, SHAM, "secondary control: medial+lateral > sham"),
    (LATERAL, SHAM, "secondary control: lateral-only > sham"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rendition-level hierarchical resampling for phrase-duration ΔSD.")
    p.add_argument("--phrases-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--selection-mode", choices=["pre", "pooled", "post", "heldout_post"], default="pre")
    p.add_argument("--top-frac", type=float, default=0.30, help="Fraction of syllables selected per bird, e.g. 0.30 for top 30%.")
    p.add_argument("--pre-epoch", default="Late Pre")
    p.add_argument("--post-epoch", default="Post")
    p.add_argument("--animal-col", default="Animal ID")
    p.add_argument("--syllable-col", default="Syllable")
    p.add_argument("--epoch-col", default="epoch")
    p.add_argument("--duration-col", default="duration_ms")
    p.add_argument("--hit-type-col", default="hit_type")
    p.add_argument("--treatment-col", default="treatment_group")
    p.add_argument("--min-renditions-per-epoch", type=int, default=2)
    p.add_argument("--bird-syllable-stat", choices=["median", "mean"], default="median")
    p.add_argument("--group-stat", choices=["median", "mean"], default="median")
    p.add_argument("--n-balance", type=int, default=200)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--alternative", choices=["greater", "two-sided"], default="greater")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def center(values: Iterable[float], stat: str) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr) if stat == "mean" else np.median(arr))


def pvalue_from_null(obs: float, null: np.ndarray, alternative: str) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    if alternative == "greater":
        count = np.sum(null >= obs)
    else:
        count = np.sum(np.abs(null) >= abs(obs))
    return float((count + 1) / (null.size + 1))


def sig_label(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def normalize_group(hit_type: str, treatment: str = "") -> str:
    hit = str(hit_type).lower()
    treatment = str(treatment).lower()
    if "sham" in hit or treatment == "sham":
        return SHAM
    if "single hit" in hit:
        return LATERAL
    if "medial+lateral" in hit or "medial and lateral" in hit:
        return PARTIAL_ML
    if "not visible" in hit or "complete" in hit:
        return COMPLETE_ML
    if "lateral" in hit and "medial" not in hit:
        return LATERAL
    return "unclassified"


def load_long(args: argparse.Namespace) -> pd.DataFrame:
    raw = pd.read_csv(args.phrases_csv)
    needed = [args.animal_col, args.syllable_col, args.epoch_col, args.duration_col]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(raw.columns)}")

    hit_col = args.hit_type_col if args.hit_type_col in raw.columns else None
    treatment_col = args.treatment_col if args.treatment_col in raw.columns else None
    df = pd.DataFrame({
        "bird": raw[args.animal_col].astype(str),
        "syllable": raw[args.syllable_col].astype(str),
        "epoch_raw": raw[args.epoch_col].astype(str),
        "duration_ms": pd.to_numeric(raw[args.duration_col], errors="coerce"),
        "hit_type": raw[hit_col].astype(str) if hit_col else "",
        "treatment_group": raw[treatment_col].astype(str) if treatment_col else "",
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["duration_ms"])
    df = df[df["epoch_raw"].isin([args.pre_epoch, args.post_epoch])].copy()
    df["epoch"] = np.where(df["epoch_raw"] == args.pre_epoch, "pre", "post")
    df["lesion_group_base"] = [normalize_group(h, t) for h, t in zip(df["hit_type"], df["treatment_group"])]
    df = df[df["lesion_group_base"].isin([SHAM, LATERAL, PARTIAL_ML, COMPLETE_ML])].copy()
    if df.empty:
        raise ValueError("No valid rows after filtering epochs/groups. Check epoch labels and hit_type mapping.")
    return df


def split_for_heldout_post(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["post_split"] = "analysis"
    post_mask = df["epoch"] == "post"
    df.loc[post_mask, "post_split"] = "test"
    for (bird, syllable), idx in df[post_mask].groupby(["bird", "syllable"]).groups.items():
        idx = np.array(list(idx))
        if idx.size < 2:
            continue
        shuffled = rng.permutation(idx)
        cut = idx.size // 2
        df.loc[shuffled[:cut], "post_split"] = "select"
        df.loc[shuffled[cut:], "post_split"] = "test"
    return df


def syllable_scores(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "pre":
        score_df = df[df["epoch"] == "pre"]
    elif mode == "post":
        score_df = df[df["epoch"] == "post"]
    elif mode == "pooled":
        score_df = df
    elif mode == "heldout_post":
        score_df = df[(df["epoch"] == "post") & (df.get("post_split", "") == "select")]
    else:
        raise ValueError(mode)
    scores = (
        score_df.groupby(["bird", "syllable"], dropna=False)["duration_ms"]
        .agg(n="size", sd="std")
        .reset_index()
    )
    return scores


def select_top_syllables(df: pd.DataFrame, mode: str, top_frac: float, min_n: int) -> pd.DataFrame:
    scores = syllable_scores(df, mode)
    scores = scores[(scores["n"] >= min_n) & np.isfinite(scores["sd"])].copy()
    rows = []
    for bird, g in scores.groupby("bird", dropna=False):
        g = g.sort_values("sd", ascending=False).copy()
        k = max(1, int(math.ceil(len(g) * top_frac)))
        rows.append(g.head(k))
    if not rows:
        return pd.DataFrame(columns=["bird", "syllable", "n", "sd", "selected"])
    selected = pd.concat(rows, ignore_index=True)
    selected["selected"] = True
    return selected


def analysis_rows_for_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "heldout_post":
        # Pre rows plus the post-lesion half not used for selection.
        return df[(df["epoch"] == "pre") | ((df["epoch"] == "post") & (df["post_split"] == "test"))].copy()
    return df.copy()


def _epoch_sd(arr: np.ndarray, n: int, rng: np.random.Generator, replace: bool) -> float:
    if arr.size < 2 or n < 2:
        return np.nan
    idx = rng.choice(arr.size, size=n, replace=replace)
    return float(np.std(arr[idx], ddof=1))


def bird_delta_sd(
    df_bird: pd.DataFrame,
    selected_syllables: set[str],
    rng: np.random.Generator,
    n_balance: int,
    replace: bool,
    min_n: int,
    bird_syllable_stat: str,
) -> dict:
    pre_sds = []
    post_sds = []
    n_used = 0
    for syllable in selected_syllables:
        sub = df_bird[df_bird["syllable"] == syllable]
        pre = sub.loc[sub["epoch"] == "pre", "duration_ms"].to_numpy(dtype=float)
        post = sub.loc[sub["epoch"] == "post", "duration_ms"].to_numpy(dtype=float)
        n = min(pre.size, post.size)
        if n < min_n:
            continue
        pre_draws = []
        post_draws = []
        reps = max(1, n_balance)
        for _ in range(reps):
            pre_draws.append(_epoch_sd(pre, n, rng, replace=replace))
            post_draws.append(_epoch_sd(post, n, rng, replace=replace))
        pre_sds.append(np.nanmean(pre_draws))
        post_sds.append(np.nanmean(post_draws))
        n_used += 1
    if n_used == 0:
        return {"pre_sd": np.nan, "post_sd": np.nan, "delta_sd": np.nan, "n_syllables": 0}
    pre_center = center(pre_sds, stat=bird_syllable_stat)
    post_center = center(post_sds, stat=bird_syllable_stat)
    return {
        "pre_sd": pre_center,
        "post_sd": post_center,
        "delta_sd": post_center - pre_center,
        "n_syllables": n_used,
    }


def observed_bird_deltas(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    rng: np.random.Generator,
    n_balance: int,
    min_n: int,
    bird_syllable_stat: str,
) -> pd.DataFrame:
    selected_map = selected.groupby("bird")["syllable"].apply(lambda x: set(map(str, x))).to_dict()
    rows = []
    for bird, bdf in df.groupby("bird", dropna=False):
        if bird not in selected_map:
            continue
        metrics = bird_delta_sd(
            bdf, selected_map[bird], rng, n_balance=n_balance, replace=False,
            min_n=min_n, bird_syllable_stat=bird_syllable_stat,
        )
        if not np.isfinite(metrics["delta_sd"]):
            continue
        base_group = bdf["lesion_group_base"].iloc[0]
        rows.append({
            "bird": bird,
            "lesion_group_base": base_group,
            "pre_sd": metrics["pre_sd"],
            "post_sd": metrics["post_sd"],
            "delta_sd": metrics["delta_sd"],
            "n_selected_syllables_used": metrics["n_syllables"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pooled = out[out["lesion_group_base"].isin([PARTIAL_ML, COMPLETE_ML])].copy()
    pooled["lesion_group"] = POOLED_ML
    out["lesion_group"] = out["lesion_group_base"]
    return pd.concat([out, pooled], ignore_index=True)


def bootstrap_group_ci_from_renditions(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    group: str,
    rng: np.random.Generator,
    B: int,
    min_n: int,
    bird_syllable_stat: str,
    group_stat: str,
) -> tuple[float, float, float]:
    selected_map = selected.groupby("bird")["syllable"].apply(lambda x: set(map(str, x))).to_dict()
    # Determine eligible birds by group.
    bird_group = df.groupby("bird")["lesion_group_base"].first().to_dict()
    if group == POOLED_ML:
        birds = [b for b, g in bird_group.items() if g in [PARTIAL_ML, COMPLETE_ML] and b in selected_map]
    else:
        birds = [b for b, g in bird_group.items() if g == group and b in selected_map]
    if not birds:
        return np.nan, np.nan, np.nan
    # Observed with fewer draws? use the bootstrap setup's one draw with replacement for CIs only;
    # observed is better read from bird_deltas output. Here return bootstrap median and interval.
    boots = np.empty(B, dtype=float)
    for i in range(B):
        deltas = []
        sampled_birds = rng.choice(birds, size=len(birds), replace=True)
        for bird in sampled_birds:
            bdf = df[df["bird"] == bird]
            m = bird_delta_sd(
                bdf, selected_map[bird], rng, n_balance=1, replace=True,
                min_n=min_n, bird_syllable_stat=bird_syllable_stat,
            )
            if np.isfinite(m["delta_sd"]):
                deltas.append(m["delta_sd"])
        boots[i] = center(deltas, stat=group_stat)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(np.nanmedian(boots)), float(lo), float(hi)


def signflip_test(values: np.ndarray, rng: np.random.Generator, P: int, stat: str, alternative: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    obs = center(values, stat=stat)
    null = np.empty(P, dtype=float)
    for i in range(P):
        signs = rng.choice(np.array([-1.0, 1.0]), size=values.size, replace=True)
        null[i] = center(values * signs, stat=stat)
    return obs, pvalue_from_null(obs, null, alternative)


def labelshuffle_test(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, P: int, stat: str, alternative: str) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    obs = center(a, stat=stat) - center(b, stat=stat)
    pooled = np.concatenate([a, b])
    n_a = a.size
    null = np.empty(P, dtype=float)
    for i in range(P):
        perm = rng.permutation(pooled)
        null[i] = center(perm[:n_a], stat=stat) - center(perm[n_a:], stat=stat)
    return obs, pvalue_from_null(obs, null, alternative)


def bootstrap_diff_ci_from_bird_deltas(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, B: int, stat: str) -> tuple[float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    obs = center(a, stat=stat) - center(b, stat=stat)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        boots[i] = center(rng.choice(a, size=a.size, replace=True), stat=stat) - center(rng.choice(b, size=b.size, replace=True), stat=stat)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def summarize(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    bird_deltas: pd.DataFrame,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    for group, g in bird_deltas.groupby("lesion_group", dropna=False):
        vals = g["delta_sd"].to_numpy(dtype=float)
        obs = center(vals, stat=args.group_stat)
        _, lo, hi = bootstrap_group_ci_from_renditions(
            df, selected, group, rng, B=args.n_bootstrap, min_n=args.min_renditions_per_epoch,
            bird_syllable_stat=args.bird_syllable_stat, group_stat=args.group_stat,
        )
        _, p = signflip_test(vals, rng, P=args.n_permutations, stat=args.group_stat, alternative=args.alternative)
        group_rows.append({
            "selection_mode": args.selection_mode,
            "top_fraction": args.top_frac,
            "lesion_group": group,
            "n_birds": int(g["bird"].nunique()),
            "median_selected_syllables_used_per_bird": center(g["n_selected_syllables_used"], stat="median"),
            "median_late_pre_SD_ms": center(g["pre_sd"], stat=args.group_stat),
            "median_post_SD_ms": center(g["post_sd"], stat=args.group_stat),
            f"{args.group_stat}_delta_SD_ms": obs,
            "bootstrap_CI95_low_ms": lo,
            "bootstrap_CI95_high_ms": hi,
            f"signflip_p_delta_gt_0_{args.alternative}": p,
            "significance": sig_label(p),
            "resampling_level": "rendition-within-bird hierarchical bootstrap",
        })
    contrast_rows = []
    for group_a, group_b, label in CONTRASTS:
        a = bird_deltas.loc[bird_deltas["lesion_group"] == group_a, "delta_sd"].to_numpy(dtype=float)
        b = bird_deltas.loc[bird_deltas["lesion_group"] == group_b, "delta_sd"].to_numpy(dtype=float)
        if a.size == 0 or b.size == 0:
            continue
        diff, lo, hi = bootstrap_diff_ci_from_bird_deltas(a, b, rng, B=args.n_bootstrap, stat=args.group_stat)
        _, p = labelshuffle_test(a, b, rng, P=args.n_permutations, stat=args.group_stat, alternative=args.alternative)
        contrast_rows.append({
            "selection_mode": args.selection_mode,
            "top_fraction": args.top_frac,
            "contrast_type": label,
            "group_A": group_a,
            "group_B": group_b,
            "n_A_birds": int(np.isfinite(a).sum()),
            "n_B_birds": int(np.isfinite(b).sum()),
            f"observed_{args.group_stat}_delta_difference_ms_A_minus_B": diff,
            "bootstrap_CI95_low_ms": lo,
            "bootstrap_CI95_high_ms": hi,
            f"labelshuffle_p_A_gt_B_{args.alternative}": p,
            "significance": sig_label(p),
            "resampling_level": "bird-level label-shuffle; CIs from bird-delta bootstrap for contrasts",
        })
    return pd.DataFrame(group_rows), pd.DataFrame(contrast_rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_long(args)
    if args.selection_mode == "heldout_post":
        df = split_for_heldout_post(df, rng)

    selected = select_top_syllables(df, args.selection_mode, args.top_frac, args.min_renditions_per_epoch)
    analysis_df = analysis_rows_for_mode(df, args.selection_mode)
    bird_deltas = observed_bird_deltas(
        analysis_df, selected, rng, n_balance=args.n_balance,
        min_n=args.min_renditions_per_epoch, bird_syllable_stat=args.bird_syllable_stat,
    )
    group_summary, contrasts = summarize(analysis_df, selected, bird_deltas, rng, args)

    selected.to_csv(out_dir / "selected_syllables.csv", index=False)
    bird_deltas.to_csv(out_dir / "hierarchical_bird_deltas.csv", index=False)
    group_summary.round(6).to_csv(out_dir / "hierarchical_group_summary.csv", index=False)
    contrasts.round(6).to_csv(out_dir / "hierarchical_pairwise_contrasts.csv", index=False)

    print(f"Wrote outputs to: {out_dir}")
    print("\nGroup summary:")
    print(group_summary.round(4).to_string(index=False))
    print("\nContrasts:")
    print(contrasts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
