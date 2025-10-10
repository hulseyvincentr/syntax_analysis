# -*- coding: utf-8 -*-
# phrase_duration_pre_vs_post_grouped.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import math
import json
import ast
import inspect
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
TITLE_FS = 18
LABEL_FS = 16
TICK_FS  = 13

def _pretty_axes(ax, x_rotation: int = 0):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.xaxis.label.set_size(LABEL_FS)
    ax.yaxis.label.set_size(LABEL_FS)
    if x_rotation:
        for lab in ax.get_xticklabels():
            lab.set_rotation(x_rotation)
            lab.set_horizontalalignment("right")

plt.rcParams.update({
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
})

# ──────────────────────────────────────────────────────────────────────────────
# Organizer import
# ──────────────────────────────────────────────────────────────────────────────
_ORGANIZE = None
try:
    from organized_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _ORGANIZE  # type: ignore
    )
except Exception:
    from organize_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _ORGANIZE  # type: ignore
    )

# ──────────────────────────────────────────────────────────────────────────────
# Dataclass result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GroupedPlotsResult:
    early_pre_path: Optional[Path]
    late_pre_path: Optional[Path]
    post_path: Optional[Path]
    aggregate_path: Optional[Path]                       # per-day box+scatter
    aggregate_three_box_path: Optional[Path]             # 3 boxes + per-syllable dots (uses y_max_ms if set)
    aggregate_three_box_auto_ylim_path: Optional[Path]   # same but auto y-limits
    syllable_labels: List[str]
    y_limits: Tuple[float, float]

# ──────────────────────────────────────────────────────────────────────────────
# Utilities: dates, parsing, table building
# ──────────────────────────────────────────────────────────────────────────────
def _parse_date_like(s: str | pd.Timestamp | None) -> pd.Timestamp | pd.NaT:
    if s is None:
        return pd.NaT
    if isinstance(s, pd.Timestamp):
        return s
    s2 = str(s).replace(".", "-")
    return pd.to_datetime(s2, errors="coerce")

def _choose_datetime_series(df: pd.DataFrame) -> pd.Series:
    dt = df.get("Recording DateTime", pd.Series([pd.NaT]*len(df), index=df.index)).copy()
    if "creation_date" in df.columns:
        need = dt.isna()
        if need.any():
            dt.loc[need] = pd.to_datetime(df.loc[need,"creation_date"], errors="coerce")
    if "Date" in df.columns and "Time" in df.columns:
        need = dt.isna()
        if need.any():
            combo = pd.to_datetime(
                df.loc[need,"Date"].astype(str).str.replace(".","-",regex=False)
                + " " + df.loc[need,"Time"].astype(str),
                errors="coerce"
            )
            dt.loc[need] = combo
    return dt

def _infer_animal_id(df: pd.DataFrame, decoded_path: Path) -> str:
    for col in ["animal_id","Animal","Animal ID"]:
        if col in df.columns and pd.notna(df[col]).any():
            val = str(df[col].dropna().iloc[0]).strip()
            if val:
                return val
    stem = decoded_path.stem
    tok  = stem.split("_")[0]
    return tok if tok else "unknown_animal"

def _maybe_parse_dict(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(obj)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return None

def _row_ms_per_bin(row: pd.Series) -> float | None:
    for k in ["time_bin_ms","bin_ms","ms_per_bin"]:
        if k in row and pd.notna(row[k]):
            try:
                return float(row[k])
            except Exception:
                pass
    return None

def _extract_durations_from_spans(spans, ms_per_bin: float | None) -> List[float]:
    out: List[float] = []
    if spans is None:
        return out
    if isinstance(spans, dict):
        spans = [spans]
    if not isinstance(spans,(list,tuple)):
        return out
    for item in spans:
        on = off = None; using_bins = False
        if isinstance(item, dict):
            if "onset_ms" in item or "on" in item:
                on=item.get("onset_ms",item.get("on"))
                off=item.get("offset_ms",item.get("off"))
            elif "onset_bin" in item or "on_bin" in item:
                on=item.get("onset_bin",item.get("on_bin"))
                off=item.get("offset_bin",item.get("off_bin"))
                using_bins=True
        elif isinstance(item,(list,tuple)) and len(item)>=2:
            on,off=item[:2]
        try:
            dur=float(off)-float(on)
            if using_bins and ms_per_bin:
                dur*=float(ms_per_bin)
            if dur>=0:
                out.append(dur)
        except Exception:
            continue
    return out

def _collect_phrase_durations_per_song(row: pd.Series, spans_col: str) -> Dict[str,List[float]]:
    out={}
    raw=row.get(spans_col,None)
    if raw is None or (isinstance(raw,float) and math.isnan(raw)):
        return out
    if spans_col=="phrase_durations_ms_dict":
        d=_maybe_parse_dict(raw)
        if isinstance(d,dict):
            for lbl,vals in d.items():
                if isinstance(vals,(list,tuple)):
                    clean=[float(v) for v in vals if isinstance(v,(int,float)) and v>=0]
                    if clean:
                        out[str(lbl)]=clean
        return out
    d=_maybe_parse_dict(raw)
    if not isinstance(d,dict):
        return out
    mpb=_row_ms_per_bin(row)
    for lbl,spans in d.items():
        vals=_extract_durations_from_spans(spans,mpb)
        if vals:
            out[str(lbl)]=vals
    return out

def _find_best_spans_column(df: pd.DataFrame)->str|None:
    for c in [
        "syllable_onsets_offsets_ms_dict",
        "syllable_onsets_offsets_dict",
        "onsets_offsets_ms_dict",
        "phrase_durations_ms_dict"
    ]:
        if c in df.columns:
            return c
    for c in df.columns:
        lc=c.lower()
        if ("onset" in lc or "start" in lc) and ("offset" in lc or "stop" in lc):
            return c
    for c in df.columns:
        if c.endswith("_dict") and df[c].notna().any():
            return c
    return None

def _build_durations_table(df: pd.DataFrame, labels: Sequence[str]) -> tuple[pd.DataFrame,str|None]:
    col=_find_best_spans_column(df)
    if col is None:
        return df.copy(),None
    per_song=df.apply(lambda r:_collect_phrase_durations_per_song(r,col),axis=1)
    out=df.copy()
    for lbl in labels:
        out[f"syllable_{lbl}_durations"]=per_song.apply(lambda d:d.get(str(lbl),[]))
    return out,col

def _explode_for_plot(dataset: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    parts=[]
    for lbl in labels:
        col=f"syllable_{lbl}_durations"
        if col not in dataset.columns:
            continue
        e=dataset[[col]].explode(col).dropna()
        if e.empty:
            continue
        e["Phrase Duration (ms)"]=pd.to_numeric(e[col],errors="coerce")
        e=e.dropna()
        if e.empty:
            continue
        try:
            s_val=int(lbl)
        except:
            s_val=lbl
        e["Syllable"]=s_val
        parts.append(e[["Syllable","Phrase Duration (ms)"]])
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["Syllable","Phrase Duration (ms)"])

def _calc_global_ylim(datasets,labels):
    vals=[]
    for ds in datasets:
        t=_explode_for_plot(ds,labels)
        if not t.empty:
            vals.extend(t["Phrase Duration (ms)"].to_list())
    if not vals:
        return (0.0,1.0)
    return (float(np.nanmin(vals)),float(np.nanmax(vals)))

# seaborn violin back-compat
def _violin_kwargs():
    params = inspect.signature(sns.violinplot).parameters
    if "density_norm" in params:   # seaborn >= 0.13
        return dict(inner="quartile", density_norm="width", color="lightgray")
    return dict(inner="quartile", scale="width", color="lightgray")

# plotting helpers
def _violin_plus_strip(ax,tall_df,y_limits):
    sns.set(style="white")
    if tall_df.empty:
        ax.text(0.5,0.5,"No phrase durations",ha="center",va="center",fontsize=TITLE_FS-2)
        _pretty_axes(ax); ax.set_axis_off(); return
    sns.violinplot(x="Syllable",y="Phrase Duration (ms)",data=tall_df,ax=ax,**_violin_kwargs())
    sns.stripplot(x="Syllable",y="Phrase Duration (ms)",data=tall_df,
                  jitter=True,size=5,color="#2E4845",alpha=0.6,ax=ax)
    ax.set_ylim(y_limits)
    ax.set_xlabel("Syllable Label")
    ax.set_ylabel("Phrase Duration (ms)")
    _pretty_axes(ax,0)

def _quick_duration_summary(df,labels,tag):
    c=Counter()
    for lbl in labels:
        col=f"syllable_{lbl}_durations"
        if col in df.columns:
            c[lbl]+=int(df[col].apply(lambda x:len(x) if isinstance(x,(list,tuple)) else 0).sum())
    print(f"[DEBUG] {tag} — durations per label:",dict(c)," (rows:",len(df),")")

def _build_daily_aggregate(df,labels,dt_series):
    cols=[f"syllable_{l}_durations" for l in labels if f"syllable_{l}_durations" in df.columns]
    if not cols:
        return pd.DataFrame(columns=["Date","Phrase Duration (ms)"])
    def _concat(row):
        out=[]
        for c in cols:
            v=row.get(c,[])
            if isinstance(v,(list,tuple)):
                out.extend([x for x in v if isinstance(x,(int,float))])
        return out
    tmp=df.copy()
    tmp["_all_durs"]=tmp.apply(_concat,axis=1)
    if tmp["_all_durs"].map(len).sum()==0:
        return pd.DataFrame(columns=["Date","Phrase Duration (ms)"])
    tmp["Date"]=pd.to_datetime(dt_series.dt.date)
    tall=tmp[["Date","_all_durs"]].explode("_all_durs").dropna()
    tall=tall.rename(columns={"_all_durs":"Phrase Duration (ms)"})
    tall["Phrase Duration (ms)"]=pd.to_numeric(tall["Phrase Duration (ms)"],errors="coerce")
    return tall.dropna(subset=["Phrase Duration (ms)"])

def _collect_all_durations(df_subset: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    arrs = []
    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        if col in df_subset.columns:
            vals = df_subset[col].explode().dropna()
            if not vals.empty:
                arrs.append(pd.to_numeric(vals, errors="coerce").dropna().to_numpy())
    if not arrs:
        return np.array([], dtype=float)
    return np.concatenate(arrs, axis=0)

def _per_syllable_medians(df_subset: pd.DataFrame, labels: Sequence[str]) -> Dict[str, float]:
    out = {}
    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        if col in df_subset.columns:
            s = pd.to_numeric(df_subset[col].explode(), errors="coerce").dropna()
            if not s.empty:
                out[str(lbl)] = float(s.median())
    return out

# ───── Stats helpers (Kruskal–Wallis omnibus + pairwise Mann–Whitney with BH FDR) ─────
def _try_mannwhitney(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        # Fallback: Welch t (approximate) if SciPy missing
        ma, mb = np.mean(x), np.mean(y)
        va, vb = np.var(x, ddof=1), np.var(y, ddof=1)
        na, nb = len(x), len(y)
        t = (ma - mb) / np.sqrt(va/na + vb/nb + 1e-12)
        # two-sided normal approx
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(t), float(p)

def _try_kruskal(groups: List[np.ndarray]) -> Tuple[float, float]:
    try:
        from scipy.stats import kruskal
        stat, p = kruskal(*groups)
        return float(stat), float(p)
    except Exception:
        # Very rough fallback: one-way ANOVA on ranks (approx)
        ranks = np.argsort(np.argsort(np.concatenate(groups)))
        sizes = [len(g) for g in groups]
        splits = np.cumsum([0] + sizes)
        parts = [ranks[splits[i]:splits[i+1]] for i in range(len(sizes))]
        means = [np.mean(p) for p in parts]
        grand = np.mean(ranks)
        ss_between = sum(n * (m - grand)**2 for n, m in zip(sizes, means))
        ss_within = sum(((p - np.mean(p))**2).sum() for p in parts)
        dfb = len(groups) - 1
        dfw = len(ranks) - len(groups)
        F = (ss_between / (dfb + 1e-12)) / (ss_within / (dfw + 1e-12) + 1e-12)
        # no exact p without SciPy; report pseudo-p via normal approx
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(F) / sqrt(2))))
        return float(F), float(p)

def _p_adjust_bh(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, int); ranks[order] = np.arange(1, n+1)
    adj = p * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty(n, float)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out.tolist()

def _stars(p: float) -> str:
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "n.s."

def _fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────
def run_phrase_duration_pre_vs_post_grouped(
    decoded_database_json, output_dir, treatment_date,
    grouping_mode="explicit", early_group_size=100, late_group_size=100, post_group_size=100,
    only_song_present=True, restrict_to_labels=None, y_max_ms=None,
    show_plots=True, make_aggregate_plot=True,
)->GroupedPlotsResult:

    decoded_path=Path(decoded_database_json)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)

    out=_ORGANIZE(decoded_database_json=decoded_path,
                  only_song_present=only_song_present,
                  compute_durations=False, add_recording_datetime=True)
    df=out.organized_df.copy()
    if only_song_present and "song_present" in df.columns:
        df=df[df["song_present"]==True].copy()

    dt=_choose_datetime_series(df)
    df=df.assign(_dt=dt).dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

    labels=[str(x) for x in (restrict_to_labels or out.unique_syllable_labels)]
    df,spans_col=_build_durations_table(df,labels)
    print("[INFO] Using spans column:",spans_col)

    t_date=_parse_date_like(treatment_date)
    pre,post=df[df["_dt"]<t_date].copy(),df[df["_dt"]>=t_date].copy()

    if grouping_mode=="explicit":
        early_n,late_n,post_n=map(int,[early_group_size,late_group_size,post_group_size])
        late_pre=pre.tail(late_n)
        early_pre=pre.iloc[max(0,len(pre)-late_n-early_n):max(0,len(pre)-late_n)]
        post_g=post.head(post_n)
        print(f"[INFO] explicit sizes → early={early_n}, late={late_n}, post={post_n}")
    else:
        n=max(len(pre)//2,len(post))
        if n<=0:
            late_pre=pre.iloc[0:0]; early_pre=pre.iloc[0:0]; post_g=post.iloc[0:0]
        else:
            late_pre=pre.tail(min(n,len(pre)))
            early_pre=pre.iloc[max(0,len(pre)-2*n):max(0,len(pre)-n)]
            post_g=post.head(min(n,len(post)))
        print(f"[INFO] auto_balance → n={n} (late_pre={len(late_pre)}, early_pre={len(early_pre)}, post={len(post_g)})")

    for nm,part in [("early_pre",early_pre),("late_pre",late_pre),("post",post_g)]:
        _quick_duration_summary(part,labels,nm)

    y_min,y_max=_calc_global_ylim([early_pre,late_pre,post_g],labels)
    if y_max_ms:
        y_max=float(y_max_ms)
    animal_id=_infer_animal_id(df,decoded_path)

    # — per-group violin+scatter
    def _make_plot(ds,title,n,fname):
        tall=_explode_for_plot(ds,labels)
        fig,ax=plt.subplots(figsize=(12,6))
        _violin_plus_strip(ax,tall,(y_min,y_max))
        ax.set_title(f"{animal_id} — {title} (N={n})",fontsize=TITLE_FS)
        fig.tight_layout()
        outp=output_dir/f"{animal_id}_{fname}_phrase_durations.png"
        fig.savefig(outp,dpi=300,transparent=True)
        if show_plots: plt.show()
        else: plt.close(fig)
        return outp

    ep=lp=po=None
    if len(early_pre): ep=_make_plot(early_pre,"Early Pre-Treatment",len(early_pre),"early_pre")
    if len(late_pre):  lp=_make_plot(late_pre,"Late Pre-Treatment",len(late_pre),"late_pre")
    if len(post_g):    po=_make_plot(post_g,"Post-Treatment",len(post_g),"post")

    # — per-day aggregate (no stats)
    agg=None
    if make_aggregate_plot:
        daily=_build_daily_aggregate(df,labels,df["_dt"])
        if not daily.empty:
            daily["DateStr"]=daily["Date"].dt.strftime("%Y-%m-%d")
            date_order=sorted(daily["DateStr"].unique())

            fig,ax=plt.subplots(figsize=(16,6))
            sns.set(style="white")
            sns.boxplot(x="DateStr",y="Phrase Duration (ms)",data=daily,
                        order=date_order,color="lightgray",fliersize=0,ax=ax)
            sns.stripplot(x="DateStr",y="Phrase Duration (ms)",data=daily,
                          order=date_order,jitter=0.25,size=3,alpha=0.6,ax=ax,color="#2E4845")

            t_str=t_date.strftime("%Y-%m-%d")
            if t_str in date_order:
                idx=date_order.index(t_str)
                ax.axvline(idx,color="red",ls="--",lw=1.2)

            ax.set_xlabel("Recording Date"); ax.set_ylabel("Phrase Duration (ms)")
            _pretty_axes(ax,90)
            if y_max_ms: ax.set_ylim(0,float(y_max_ms))
            ax.set_title(f"{animal_id} — Pre/Post Aggregate (per-day box + scatter)", fontsize=TITLE_FS)

            fig.tight_layout()
            agg=output_dir/f"{animal_id}_aggregate_pre_post_phrase_durations.png"
            fig.savefig(agg,dpi=300,transparent=True)
            if show_plots: plt.show()
            else: plt.close(fig)
        else:
            print("[WARN] Aggregate: no durations found to plot.")

    # — three-group aggregate with per-syllable median dots + stats
    agg3 = None
    agg3_auto = None
    three_groups = []
    group_defs = [(early_pre, "Early Pre"), (late_pre, "Late Pre"), (post_g, "Post")]
    for part, label in group_defs:
        if len(part):
            vals = _collect_all_durations(part, labels)
            if vals.size:
                three_groups.append(pd.DataFrame({"Group": label, "Phrase Duration (ms)": vals}))

    if three_groups:
        tall3 = pd.concat(three_groups, ignore_index=True)
        order = ["Early Pre", "Late Pre", "Post"]
        xpos = {g: i for i, g in enumerate(order)}

        # per-syllable medians for each group
        med_map = {glabel: _per_syllable_medians(gdf, labels) for gdf, glabel in group_defs}

        # arrays per group for stats
        group_arrays = {g: tall3.loc[tall3["Group"]==g, "Phrase Duration (ms)"].to_numpy() for g in order}

        # omnibus Kruskal–Wallis
        kw_stat, kw_p = _try_kruskal([group_arrays[g] for g in order])

        # pairwise Mann–Whitney + BH adjust
        pairs = list(combinations(order, 2))
        raw_p = []
        for a,b in pairs:
            _, p = _try_mannwhitney(group_arrays[a], group_arrays[b])
            raw_p.append(p)
        adj_p = _p_adjust_bh(raw_p)

        def _summary_text():
            lines = [f"Kruskal–Wallis: H={kw_stat:.3g}, p={_fmt_p(kw_p)}"]
            for (a,b), p_raw, p_adj in zip(pairs, raw_p, adj_p):
                lines.append(f"{a} vs {b}: p(BH)={_fmt_p(p_adj)}  [{_stars(p_adj)}]")
            return "\n".join(lines)

        def _draw_three_box(ax, set_ylim: bool, title: str):
            sns.boxplot(
                x="Group", y="Phrase Duration (ms)", data=tall3,
                order=order, color="lightgray",
                whis=(0, 100), showfliers=False, ax=ax
            )
            # overlay per-syllable dots and dashed connectors
            for lbl in [str(x) for x in labels]:
                xs, ys = [], []
                for g in order:
                    y = med_map.get(g, {}).get(lbl, None)
                    if y is not None:
                        xs.append(xpos[g])
                        ys.append(y)
                if xs:
                    ax.plot(xs, ys, linestyle="--", linewidth=1, color="0.4", alpha=0.7, zorder=3)
                    ax.scatter(xs, ys, s=28, edgecolor="white", linewidth=0.5, zorder=4)
            ax.set_xlabel("")
            ax.set_ylabel("Phrase Duration (ms)")
            _pretty_axes(ax, 0)
            if set_ylim and y_max_ms:
                ax.set_ylim(0, float(y_max_ms))
            ax.set_title(title, fontsize=TITLE_FS)

            # stats textbox (top-right)
            ax.text(
                0.99, 0.98, _summary_text(),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=LABEL_FS-2,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9)
            )

        # Figure 1: with ylim (if provided)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        _draw_three_box(ax1, set_ylim=True, title=f"{animal_id} — Early/Late Pre vs Post (Syllable medians + boxes)")
        fig1.tight_layout()
        agg3 = output_dir / f"{animal_id}_aggregate_three_group_phrase_durations.png"
        fig1.savefig(agg3, dpi=300, transparent=True)
        if show_plots: plt.show()
        else: plt.close(fig1)

        # Figure 2: auto y-limits
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        _draw_three_box(ax2, set_ylim=False, title=f"{animal_id} — Early/Late Pre vs Post (auto y-limits)")
        fig2.tight_layout()
        agg3_auto = output_dir / f"{animal_id}_aggregate_three_group_phrase_durations_auto_ylim.png"
        fig2.savefig(agg3_auto, dpi=300, transparent=True)
        if show_plots: plt.show()
        else: plt.close(fig2)
    else:
        print("[WARN] Three-group aggregate: no durations found to plot.")

    return GroupedPlotsResult(
        early_pre_path=ep,
        late_pre_path=lp,
        post_path=po,
        aggregate_path=agg,
        aggregate_three_box_path=agg3,
        aggregate_three_box_auto_ylim_path=agg3_auto,
        syllable_labels=[str(x) for x in labels],
        y_limits=(y_min,y_max),
    )

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pre/Post grouped phrase-duration plots per syllable + aggregate.")
    p.add_argument("decoded_database_json", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("treatment_date", type=str, help='e.g., "2025-03-04" or "2025.03.04"')
    p.add_argument("--grouping_mode", type=str, default="explicit",
                   choices=["explicit", "auto_balance"],
                   help="explicit: use sizes; auto_balance: n=max(len(pre)//2, len(post))")
    p.add_argument("--early_group_size", type=int, default=100)
    p.add_argument("--late_group_size", type=int, default=100)
    p.add_argument("--post_group_size", type=int, default=100)
    p.add_argument("--only_song_present", action="store_true", default=True)
    p.add_argument("--y_max_ms", type=float, default=None)
    p.add_argument("--labels", type=str, nargs="*", default=None, help="Restrict to these labels (e.g., 0 1 2 ...)")
    p.add_argument("--no_show", action="store_true", help="Disable plt.show()")
    p.add_argument("--no_aggregate", action="store_true", help="(legacy; ignored—aggregate stays on)")
    args = p.parse_args()

    _ = run_phrase_duration_pre_vs_post_grouped(
        decoded_database_json=args.decoded_database_json,
        output_dir=args.output_dir,
        treatment_date=args.treatment_date,
        grouping_mode=args.grouping_mode,
        early_group_size=args.early_group_size,
        late_group_size=args.late_group_size,
        post_group_size=args.post_group_size,
        only_song_present=args.only_song_present,
        restrict_to_labels=args.labels,
        y_max_ms=args.y_max_ms,
        show_plots=not args.no_show,
        make_aggregate_plot=True,
    )



"""
## Option 1: set batch sizes
from phrase_duration_pre_vs_post_grouped import run_phrase_duration_pre_vs_post_grouped

# Input paths
decoded = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08_RC6_Comp2_decoded_database.json"
outdir  = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08"
tdate   = "2025-05-22"

# Run the grouped phrase duration analysis
res = run_phrase_duration_pre_vs_post_grouped(
    decoded_database_json=decoded,
    output_dir=outdir,
    treatment_date=tdate,
    grouping_mode="explicit",       # use explicit group sizes
    early_group_size=100,           # number of early pre-lesion songs
    late_group_size=100,            # number of late pre-lesion songs
    post_group_size=100,            # number of post-lesion songs
    only_song_present=True,
    restrict_to_labels=None,        # or e.g. ['0','1','2','3','4']
    y_max_ms=40000,
    show_plots=True,                # display figures interactively
)

## Option 2: automatically set equal sized batches
from phrase_duration_pre_vs_post_grouped import run_phrase_duration_pre_vs_post_grouped

# Input paths
decoded = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08_RC6_Comp2_decoded_database.json"
outdir  = "/Users/mirandahulsey-vincent/Desktop/AreaX_lesion_2025/R08"
tdate   = "2025-05-22"

# Run the grouped phrase duration analysis (auto-balanced)
res = run_phrase_duration_pre_vs_post_grouped(
    decoded_database_json=decoded,
    output_dir=outdir,
    treatment_date=tdate,
    grouping_mode="auto_balance",   # automatically balance group sizes
    only_song_present=True,
    restrict_to_labels=None,        # or specify syllable labels
    y_max_ms=40000,
    show_plots=True,
)


"""

