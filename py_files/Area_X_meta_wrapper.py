# -*- coding: utf-8 -*-
# Area_X_meta_wrapper.py  (single-sheet version)
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Area_X_analysis_wrapper import WrapperConfig, run_all

# Optional for per-animal TTE summaries
try:
    import importlib, tte_by_day
    importlib.reload(tte_by_day)
    from tte_by_day import TTE_by_day
    _HAVE_TTE = True
except Exception:
    _HAVE_TTE = False
    print("[warn] Could not import tte_by_day; ΔTTE summaries may be limited.", file=sys.stderr)


# ───────────────────────── helpers ─────────────────────────
def _parse_yes_no(x: object) -> Optional[bool]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"y", "yes", "true", "t", "1"}: return True
    if s in {"n", "no", "false", "f", "0"}: return False
    return None

def _safe_str(x: object) -> Optional[str]:
    if x is None: return None
    s = str(x).strip()
    return s or None

def resolve_decoded_path(decoded_glob_template: str, animal_id: str) -> Optional[Path]:
    pattern = decoded_glob_template.format(animal_id=animal_id)
    base = Path("/")
    candidates = list(base.glob(pattern.lstrip("/"))) or list(Path(".").resolve().glob(pattern))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: (len(str(p)), str(p)))
    return candidates[0]

def _bin_lesion_percent(x: Optional[float], edges: List[float]) -> Optional[str]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or x < 0 or x > 100)): return None
    for lo, hi in zip(edges[:-1], edges[1:]):
        if x >= lo and (x <= hi if hi == edges[-1] else x < hi):
            return f"{int(lo)}–{int(hi)}%"
    return None

def _stripna(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


# ───────────────────────── config ─────────────────────────
@dataclass
class BatchConfig:
    metadata_xlsx: Path                # single-sheet with name "metadata" (default)
    decoded_glob: str                  # must include {animal_id}
    output_root: Path
    sheet_name: str = "metadata"

    # forward toggles
    skip_daily: bool = False
    skip_heatmap: bool = False
    skip_phrase: bool = False
    skip_tte: bool = False
    show_plots: bool = False

    # daily-transitions
    min_row_total: int = 0
    movie_fps: int = 2
    movie_figsize: Tuple[float, float] = (8.0, 7.0)

    # phrase-duration
    grouping_mode: str = "auto_balance"
    early_group_size: int = 100
    late_group_size: int = 100
    post_group_size: int = 100
    y_max_ms: int = 40_000

    # TTE
    min_songs_per_day: int = 5
    treatment_in: str = "post"

    # grouping bins
    lesion_bin_edges: Tuple[float, float, float, float, float] = (0, 25, 50, 75, 100)


# ───────────────────────── batch run ─────────────────────────
def run_batch(cfg: BatchConfig) -> pd.DataFrame:
    req_cols = [
        "Animal ID", "Treatment date", "Treatment type",
        "Percentage of Area X lesioned", "Medial Area X lesioned (Y/N)",
        "Injection #", "Injection volume (uL)", "AP (mm)", "ML (mm)", "DV (mm)",
    ]
    df = pd.read_excel(cfg.metadata_xlsx, sheet_name=cfg.sheet_name)
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in '{cfg.sheet_name}': {missing}")

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    group_dir = cfg.output_root / "group_comparisons"
    group_dir.mkdir(parents=True, exist_ok=True)

    # Normalize animal-level fields once; allow multiple injection rows per animal
    df["Animal ID"] = df["Animal ID"].astype(str).str.strip()
    df["_tdate"] = df["Treatment date"].astype(str).str.strip()
    # Clean lesion %
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    df["_pct_lesion"] = df["Percentage of Area X lesioned"].apply(_to_float)
    df["_medial"] = df["Medial Area X lesioned (Y/N)"].apply(_parse_yes_no)
    df["_ttype"] = df["Treatment type"].astype(str).str.strip()

    # Per-animal meta (first non-null wins + warn on conflicts)
    meta_cols = ["_tdate", "_ttype", "_pct_lesion", "_medial"]
    animals = []
    for animal_id, sub in df.groupby("Animal ID"):
        meta = {c: _first_non_null(sub[c]) for c in meta_cols}
        # Warn if conflicting entries
        _warn_conflicts(sub, animal_id, meta)
        animals.append({
            "animal_id": animal_id,
            "treatment_date": meta["_tdate"] if meta["_tdate"] not in {"", "nan", "NaT"} else None,
            "treatment_type": meta["_ttype"] if meta["_ttype"] not in {"", "nan"} else None,
            "pct_lesion": meta["_pct_lesion"] if not (isinstance(meta["_pct_lesion"], float) and math.isnan(meta["_pct_lesion"])) else None,
            "medial_lesion": meta["_medial"],
            "injection_count": int(sub["Injection #"].dropna().shape[0]) if "Injection #" in sub else 0,
        })
    animals_df = pd.DataFrame(animals)

    summaries: List[Dict] = []
    for _, row in animals_df.iterrows():
        animal_id = row["animal_id"]
        tdate = row["treatment_date"]
        ttype = row["treatment_type"]
        pct_lesion = row["pct_lesion"]
        medial_yes = row["medial_lesion"]

        decoded_path = resolve_decoded_path(cfg.decoded_glob, animal_id)
        if not decoded_path or not decoded_path.exists():
            print(f"[warn] Decoded JSON not found for {animal_id} with pattern: {cfg.decoded_glob}")
            summaries.append(dict(
                animal_id=animal_id, treatment_date=tdate, treatment_type=ttype,
                pct_lesion=pct_lesion, medial_lesion=medial_yes, injection_count=row["injection_count"],
                decoded_found=False, tte_pre_mean=np.nan, tte_post_mean=np.nan, tte_delta=np.nan,
                tte_pvalue=np.nan, tte_test=None,
            ))
            continue

        # Save this animal's injection rows for reference
        animal_inj = df[df["Animal ID"] == animal_id][
            ["Injection #", "Injection volume (uL)", "AP (mm)", "ML (mm)", "DV (mm)"]
        ].dropna(how="all")
        animal_dir = (cfg.output_root / animal_id).resolve()
        animal_dir.mkdir(parents=True, exist_ok=True)
        inj_csv = animal_dir / f"{animal_id}_injections.csv"
        if animal_inj.shape[0] > 0:
            animal_inj.to_csv(inj_csv, index=False)

        # Run per-animal analyses
        sc = WrapperConfig(
            decoded=decoded_path,
            output_root=animal_dir,
            treatment_date=tdate,
            skip_daily=cfg.skip_daily,
            skip_heatmap=cfg.skip_heatmap,
            skip_phrase=cfg.skip_phrase or (tdate is None),
            skip_tte=cfg.skip_tte,
            restrict_to_labels=None,
            show_plots=cfg.show_plots,
            min_row_total=cfg.min_row_total,
            movie_fps=cfg.movie_fps,
            movie_figsize=cfg.movie_figsize,
            enforce_consistent_order=True,
            reload_modules=True,
            grouping_mode=cfg.grouping_mode,
            early_group_size=cfg.early_group_size,
            late_group_size=cfg.late_group_size,
            post_group_size=cfg.post_group_size,
            y_max_ms=cfg.y_max_ms,
            min_songs_per_day=cfg.min_songs_per_day,
            treatment_in=cfg.treatment_in,
        )
        run_all(sc)

        # Optional TTE summary (pre/post means, delta, p-value)
        tte_pre_mean = np.nan
        tte_post_mean = np.nan
        tte_delta = np.nan
        tte_pvalue = np.nan
        tte_test = None
        if _HAVE_TTE and tdate:
            try:
                res = TTE_by_day(
                    decoded_database_json=str(decoded_path),
                    creation_metadata_json=None,
                    only_song_present=True,
                    fig_dir=str(animal_dir / "tte_by_day"),
                    show=False,
                    min_songs_per_day=cfg.min_songs_per_day,
                    treatment_date=tdate,
                    treatment_in=cfg.treatment_in,
                )
                pre_vals = getattr(res, "pre_values", None)
                post_vals = getattr(res, "post_values", None)
                if pre_vals is not None and len(pre_vals) > 0:
                    tte_pre_mean = float(np.nanmean(pre_vals))
                if post_vals is not None and len(post_vals) > 0:
                    tte_post_mean = float(np.nanmean(post_vals))
                if not (np.isnan(tte_pre_mean) or np.isnan(tte_post_mean)):
                    tte_delta = float(tte_post_mean - tte_pre_mean)
                tte_pvalue = getattr(res, "prepost_p_value", np.nan)
                tte_test = getattr(res, "prepost_test", None)
            except Exception as e:
                print(f"[warn] TTE_by_day failed for {animal_id}: {e}")

        summaries.append(dict(
            animal_id=animal_id,
            treatment_date=tdate,
            treatment_type=ttype,
            pct_lesion=pct_lesion,
            medial_lesion=medial_yes,
            injection_count=row["injection_count"],
            decoded_found=True,
            tte_pre_mean=tte_pre_mean,
            tte_post_mean=tte_post_mean,
            tte_delta=tte_delta,
            tte_pvalue=tte_pvalue,
            tte_test=tte_test,
        ))

    # Save per-animal summary & group plots
    summary_df = pd.DataFrame(summaries)
    summary_csv = cfg.output_root / "summary_per_animal.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[summary] Wrote {summary_csv}")

    _make_group_plots(summary_df, group_dir, cfg)

    return summary_df


def _first_non_null(series: pd.Series):
    for v in series:
        if pd.notna(v) and str(v).strip() != "":
            return v
    return np.nan

def _warn_conflicts(sub: pd.DataFrame, animal_id: str, meta: Dict[str, object]) -> None:
    # If multiple distinct non-null values exist for a field, warn once
    for col, label in [("_tdate", "Treatment date"), ("_ttype", "Treatment type"),
                       ("_pct_lesion", "Percentage of Area X lesioned"), ("_medial", "Medial Area X lesioned (Y/N)")]:
        vals = set([str(v).strip() for v in sub[col].dropna().astype(str) if str(v).strip() != ""])
        if len(vals) > 1:
            print(f"[warn] Conflicting {label} for {animal_id}: {sorted(vals)}. Using first non-null: {meta[col]}.")


# ────────────────── group comparisons (ΔTTE) ──────────────────
def _box_scatter(ax, data_by_group: Dict[str, np.ndarray], title: str, ylabel: str) -> None:
    groups = list(data_by_group.keys())
    arrays = [data_by_group[g] for g in groups]
    bp = ax.boxplot(arrays, labels=groups, showfliers=False)
    for i, arr in enumerate(arrays, start=1):
        x = np.random.normal(loc=i, scale=0.06, size=len(arr))
        ax.scatter(x, arr, alpha=0.8, s=28)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)

def _make_group_plots(summary_df: pd.DataFrame, group_dir: Path, cfg: BatchConfig) -> None:
    if "tte_delta" not in summary_df.columns:
        print("[groups] No tte_delta in summary; skipping group plots.")
        return

    # 1) By Treatment type
    if "treatment_type" not in summary_df.columns:
        df = summary_df.rename(columns={"Treatment type": "treatment_type"})
    else:
        df = summary_df.copy()

    by_type: Dict[str, np.ndarray] = {}
    for name, sub in df.groupby(df.get("treatment_type", pd.Series(["Unknown"]*len(df)))):
        vals = _stripna(sub["tte_delta"])
        if len(vals) > 0:
            by_type[str(name)] = vals.to_numpy(dtype=float)
    if len(by_type) >= 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        _box_scatter(ax, by_type, "ΔTTE (post–pre) by Treatment type", "ΔTTE (post–pre)")
        out = group_dir / "delta_tte_by_treatment_type.png"
        fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
        print(f"[groups] Saved {out}")

    # 2) By Medial lesion (Y/N)
    if "medial_lesion" in df.columns:
        by_medial: Dict[str, np.ndarray] = {}
        for flag, sub in df.groupby(df["medial_lesion"].map({True: "Y", False: "N", None: "NA"})):
            vals = _stripna(sub["tte_delta"])
            if len(vals) > 0:
                by_medial[str(flag)] = vals.to_numpy(dtype=float)
        if len(by_medial) >= 1:
            fig, ax = plt.subplots(figsize=(6, 5))
            _box_scatter(ax, by_medial, "ΔTTE (post–pre) by Medial Area X lesioned", "ΔTTE (post–pre)")
            out = group_dir / "delta_tte_by_medial_lesion.png"
            fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
            print(f"[groups] Saved {out}")

    # 3) By lesion percentage bins
    if "pct_lesion" in df.columns and len(df["pct_lesion"].dropna()) > 0:
        edges = list(cfg.lesion_bin_edges)
        df["lesion_bin"] = df["pct_lesion"].apply(lambda v: _bin_lesion_percent(v, edges))
        by_bin: Dict[str, np.ndarray] = {}
        for b, sub in df.groupby(df["lesion_bin"].fillna("NA")):
            vals = _stripna(sub["tte_delta"])
            if len(vals) > 0:
                by_bin[str(b)] = vals.to_numpy(dtype=float)
        if len(by_bin) >= 1:
            fig, ax = plt.subplots(figsize=(7, 5))
            _box_scatter(ax, by_bin, "ΔTTE (post–pre) by Lesion % bin", "ΔTTE (post–pre)")
            out = group_dir / "delta_tte_by_lesion_bin.png"
            fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
            print(f"[groups] Saved {out}")


# ───────────────────────── CLI ─────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch wrapper (single-sheet metadata): run per-animal analyses and group comparisons."
    )
    p.add_argument("--metadata_xlsx", type=str, required=True,
                   help="Path to XLSX with a single 'metadata' sheet.")
    p.add_argument("--decoded_glob", type=str, required=True,
                   help="Glob with {animal_id}, e.g. '/Users/you/**/{animal_id}*decoded_database.json'")
    p.add_argument("--output_root", type=str, required=True,
                   help="Folder for per-animal outputs and group plots.")
    p.add_argument("--sheet_name", type=str, default="metadata")

    # forward
    p.add_argument("--skip_daily", action="store_true")
    p.add_argument("--skip_heatmap", action="store_true")
    p.add_argument("--skip_phrase", action="store_true")
    p.add_argument("--skip_tte", action="store_true")
    p.add_argument("--show_plots", action="store_true")

    # daily-transitions
    p.add_argument("--min_row_total", type=int, default=0)
    p.add_argument("--movie_fps", type=int, default=2)
    p.add_argument("--movie_figsize", type=float, nargs=2, default=[8.0, 7.0])

    # phrase-duration
    p.add_argument("--grouping_mode", type=str, choices=["auto_balance", "explicit"], default="auto_balance")
    p.add_argument("--early_group_size", type=int, default=100)
    p.add_argument("--late_group_size", type=int, default=100)
    p.add_argument("--post_group_size", type=int, default=100)
    p.add_argument("--y_max_ms", type=int, default=40000)

    # TTE
    p.add_argument("--min_songs_per_day", type=int, default=5)
    p.add_argument("--treatment_in", type=str, choices=["pre", "post"], default="post")

    # lesion bin edges
    p.add_argument("--lesion_bin_edges", type=float, nargs="+", default=[0, 25, 50, 75, 100])
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = BatchConfig(
        metadata_xlsx=Path(args.metadata_xlsx).expanduser().resolve(),
        decoded_glob=args.decoded_glob,
        output_root=Path(args.output_root).expanduser().resolve(),
        sheet_name=args.sheet_name,
        skip_daily=args.skip_daily,
        skip_heatmap=args.skip_heatmap,
        skip_phrase=args.skip_phrase,
        skip_tte=args.skip_tte,
        show_plots=args.show_plots,
        min_row_total=args.min_row_total,
        movie_fps=args.movie_fps,
        movie_figsize=(float(args.movie_figsize[0]), float(args.movie_figsize[1])),
        grouping_mode=args.grouping_mode,
        early_group_size=args.early_group_size,
        late_group_size=args.late_group_size,
        post_group_size=args.post_group_size,
        y_max_ms=args.y_max_ms,
        min_songs_per_day=args.min_songs_per_day,
        treatment_in=args.treatment_in,
        lesion_bin_edges=tuple(args.lesion_bin_edges),
    )
    print("============================================================")
    print("Area X Meta Batch Wrapper (single-sheet)")
    print("Metadata XLSX  :", cfg.metadata_xlsx)
    print("Sheet name     :", cfg.sheet_name)
    print("Decoded glob   :", cfg.decoded_glob)
    print("Output root    :", cfg.output_root)
    print("============================================================")
    _ = run_batch(cfg)
    print("============================================================")
    print("Done. See outputs in:", cfg.output_root)
    print(" - summary_per_animal.csv")
    print(" - group_comparisons/*.png")
    print("Per-animal outputs live under subfolders per Animal ID.")
    print("============================================================")


if __name__ == "__main__":
    main()

"""
from pathlib import Path
from Area_X_analysis_wrapper import run_all, WrapperConfig

decoded = Path("/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2024/USA5443_decoded_database.json")
outdir  = Path("/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2024/USA5443_figures")
tdate   = "2024-04-30"                 # or None

cfg = WrapperConfig(
    decoded=decoded,
    output_root=outdir,
    treatment_date=tdate,

    # toggle steps (False = run; True = skip)
    skip_daily=False,
    skip_heatmap=False,
    skip_phrase=False,
    skip_tte=False,

    # labels (None = use all)
    restrict_to_labels=None,  # e.g. ['0','1','2','3']

    # daily transitions movie
    movie_fps=2,
    movie_figsize=(8, 7),
    min_row_total=0,

    # phrase duration
    grouping_mode="auto_balance",  # or "explicit"
    early_group_size=100,
    late_group_size=100,
    post_group_size=100,
    y_max_ms=40000,

    # TTE
    min_songs_per_day=5,
    treatment_in="post",

    # visuals
    show_plots=True,
)

run_all(cfg)


"""
