
# phrase_duration_over_days_with_label_colors_and_variance_batch.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, Dict, List, Tuple, Any
from dataclasses import dataclass
import math
import json
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

_HAS_MERGE_BUILDER = False
try:
    from merge_annotations_from_split_songs import build_decoded_with_split_labels  # type: ignore
    _HAS_MERGE_BUILDER = True
except Exception:
    _HAS_MERGE_BUILDER = False

_USING_SERIAL_SEGMENTS = False
try:
    from organized_decoded_serialTS_segments import (
        build_organized_segments_with_durations as _build_organized,
        OrganizedDataset,
    )
    _USING_SERIAL_SEGMENTS = True
except ImportError:
    try:
        from organize_decoded_with_segments import (
            build_organized_segments_with_durations as _build_organized,
            OrganizedDataset,
        )
    except ImportError:
        from organize_decoded_with_durations import (
            build_organized_dataset_with_durations as _build_organized,
            OrganizedDataset,
        )


@dataclass
class BatchPhraseDurationOverDaysRecord:
    animal_id: str
    treatment_date: Optional[str]
    decoded_database_json: Optional[Path]
    song_detection_json: Optional[Path]
    output_dir: Optional[Path]
    status: str
    message: Optional[str] = None


LESION_GROUP_ORDER = [
    "sham saline injection",
    "lateral hit only",
    "combined large lesion and M+L hits",
]

LESION_GROUP_COLORS = {
    "sham saline injection": "#d62728",
    "lateral hit only": "#c4b5fd",
    "combined large lesion and M+L hits": "#6a3d9a",
}


def _normalize_lesion_hit_type(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if "sham" in s:
        return "sham saline injection"
    if "single hit" in s or "lateral hit only" in s:
        return "lateral hit only"
    if "medial+lateral" in s or "medial + lateral" in s or "medial lateral" in s:
        return "combined large lesion and M+L hits"
    if "large lesion" in s or "not visible" in s:
        return "combined large lesion and M+L hits"
    if "m+l" in s:
        return "combined large lesion and M+L hits"
    return None


def _load_lesion_group_map_from_excel(
    metadata_excel: Union[str, Path],
    *,
    sheet_name: Union[int, str] = "animal_hit_type_summary",
    id_col: str = "Animal ID",
    lesion_hit_type_col: str = "Lesion hit type",
) -> Dict[str, str]:
    metadata_excel = Path(metadata_excel)
    try:
        df = pd.read_excel(metadata_excel, sheet_name=sheet_name)
    except Exception:
        return {}
    df = df.rename(columns={str(c): str(c).strip() for c in df.columns})
    if id_col not in df.columns or lesion_hit_type_col not in df.columns:
        return {}
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        aid = str(row[id_col]).strip() if pd.notna(row[id_col]) else ""
        if not aid:
            continue
        grp = _normalize_lesion_hit_type(row[lesion_hit_type_col])
        if grp is not None:
            out[aid] = grp
    return out


def _violin_with_backcompat(*, data, x, y, order=None, color="lightgray", inner="quartile"):
    try:
        return sns.violinplot(
            data=data, x=x, y=y, order=order, inner=inner, density_norm="width", color=color
        )
    except TypeError:
        return sns.violinplot(
            data=data, x=x, y=y, order=order, inner=inner, scale="width", color=color
        )


def _get_tab60_palette() -> List[str]:
    tab20 = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    return [mcolors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]


def _generate_stable_label_colors(labels: Sequence[str]) -> Dict[str, str]:
    palette = _get_tab60_palette()
    out: Dict[str, str] = {"-1": "#7f7f7f"}
    for lbl in labels:
        s = str(lbl)
        if s == "-1":
            out[s] = "#7f7f7f"
            continue
        try:
            idx = int(s) % len(palette)
        except Exception:
            idx = abs(hash(s)) % len(palette)
        out[s] = palette[idx]
    return out


def _load_label_colors(
    labels: Sequence[str],
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    fallback = _generate_stable_label_colors(labels)
    if fixed_label_colors_json is None:
        return fallback

    p = Path(fixed_label_colors_json)
    with open(p, "r") as f:
        raw = json.load(f)

    loaded = {str(k): str(v) for k, v in raw.items()}
    for lbl, col in fallback.items():
        loaded.setdefault(lbl, col)
    return loaded


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
    for k in ["time_bin_ms", "timebin_ms", "bin_ms", "ms_per_bin"]:
        if k in row and pd.notna(row[k]):
            try:
                return float(row[k])
            except Exception:
                pass
    return None


def _is_timebin_col(colname: str) -> bool:
    c = (colname or "").lower()
    return "timebin" in c or c.endswith("_bins") or "bin" in c


def _extract_durations_from_spans(spans, *, ms_per_bin: float | None, treat_as_bins: bool) -> List[float]:
    out: List[float] = []
    if spans is None:
        return out

    if isinstance(spans, (list, tuple)) and len(spans) == 2 and all(isinstance(x, (int, float)) for x in spans):
        spans = [spans]
    if isinstance(spans, dict):
        spans = [spans]
    if not isinstance(spans, (list, tuple)):
        return out

    for item in spans:
        on = off = None
        using_bins = False

        if isinstance(item, dict):
            if "onset_ms" in item or "on" in item:
                on = item.get("onset_ms", item.get("on"))
                off = item.get("offset_ms", item.get("off"))
            elif "onset_bin" in item or "on_bin" in item:
                on = item.get("onset_bin", item.get("on_bin"))
                off = item.get("offset_bin", item.get("off_bin"))
                using_bins = True
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            on, off = item[:2]
            using_bins = False

        try:
            dur = float(off) - float(on)
            if dur < 0:
                continue
            if treat_as_bins or using_bins:
                if ms_per_bin:
                    dur *= float(ms_per_bin)
                else:
                    continue
            out.append(dur)
        except Exception:
            continue
    return out


def _find_best_spans_column(df: pd.DataFrame) -> str | None:
    for c in [
        "syllable_onsets_offsets_ms_dict",
        "syllable_onsets_offsets_ms",
        "onsets_offsets_ms_dict",
    ]:
        if c in df.columns and df[c].notna().any():
            return c
    for c in [
        "syllable_onsets_offsets_timebins",
        "syllable_onsets_offsets_timebins_dict",
    ]:
        if c in df.columns and df[c].notna().any():
            return c
    for c in df.columns:
        lc = str(c).lower()
        if lc.endswith("_dict") and df[c].notna().any():
            return c
    return None


def _collect_unique_labels_sorted(df: pd.DataFrame, dict_col: str) -> List[str]:
    labs: List[str] = []
    for d in df[dict_col].dropna():
        dd = _maybe_parse_dict(d) if not isinstance(d, dict) else d
        if isinstance(dd, dict):
            labs.extend(list(map(str, dd.keys())))
    labs = list(dict.fromkeys(labs))
    try:
        ints = [int(x) for x in labs]
        order = np.argsort(ints)
        return [labs[i] for i in order]
    except Exception:
        return sorted(labs)


def _build_durations_columns_from_dicts(df: pd.DataFrame, labels: Sequence[str]) -> Tuple[pd.DataFrame, str | None]:
    col = _find_best_spans_column(df)
    if col is None:
        return df.copy(), None

    def _per_song(row: pd.Series) -> Dict[str, List[float]]:
        raw = row.get(col, None)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return {}
        d = _maybe_parse_dict(raw) if not isinstance(raw, dict) else raw
        if not isinstance(d, dict):
            return {}
        mpb = _row_ms_per_bin(row)
        treat_as_bins = _is_timebin_col(col)
        out: Dict[str, List[float]] = {}
        for lbl, spans in d.items():
            vals = _extract_durations_from_spans(spans, ms_per_bin=mpb, treat_as_bins=treat_as_bins)
            if vals:
                out[str(lbl)] = vals
        return out

    per_song = df.apply(_per_song, axis=1)
    out = df.copy()
    for lbl in [str(x) for x in labels]:
        out[f"syllable_{lbl}_durations"] = per_song.apply(lambda d: d.get(lbl, []))
    return out, col


def _choose_dt_series(df: pd.DataFrame) -> pd.Series:
    for c in ["Recording DateTime", "recording_datetime"]:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                return dt

    if "Date" in df.columns and "Time" in df.columns:
        dt = pd.to_datetime(
            df["Date"].astype(str).str.replace(".", "-", regex=False) + " " + df["Time"].astype(str),
            errors="coerce",
        )
        if dt.notna().any():
            return dt

    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        if dt.notna().any():
            return dt

    return pd.Series([pd.NaT] * len(df), index=df.index)


def _infer_animal_id(df: pd.DataFrame, fallback_path: Optional[Path]) -> str:
    for col in ["animal_id", "Animal", "Animal ID"]:
        if col in df.columns and pd.notna(df[col]).any():
            val = str(df[col].dropna().iloc[0]).strip()
            if val:
                return val
    if fallback_path is not None:
        stem = fallback_path.stem
        if stem:
            return stem.split("_")[0] or "unknown_animal"
    return "unknown_animal"


def _coerce_treatment_date(user_treatment_date: Optional[str | pd.Timestamp], fallback_str: Optional[str]) -> Optional[pd.Timestamp]:
    if user_treatment_date is not None:
        if isinstance(user_treatment_date, pd.Timestamp):
            return user_treatment_date.normalize()
        dt = pd.to_datetime(user_treatment_date, errors="coerce")
        if pd.notna(dt):
            return dt.normalize()

    if fallback_str:
        dt = pd.to_datetime(fallback_str, format="%Y.%m.%d", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(fallback_str, errors="coerce")
        if pd.notna(dt):
            return dt.normalize()
    return None


def _drop_tz_if_present(dt: pd.Series) -> pd.Series:
    try:
        if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
            return dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt


def _plot_one_syllable_duration(
    exploded: pd.DataFrame,
    col: str,
    full_date_range: pd.DatetimeIndex,
    out_path: Path,
    *,
    violin_color: str = "lightgray",
    point_color: str = "#2E4845",
    y_max_ms: int = 25_000,
    jitter: float | bool = True,
    point_size: int = 5,
    point_alpha: float = 0.7,
    figsize: tuple[int, int] = (20, 11),
    font_size_labels: int = 30,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,
    treatment_date_dt: Optional[pd.Timestamp] = None,
    show_plots: bool = False,
    dpi: int = 300,
    transparent: bool = False,
):
    exploded = exploded.copy()

    dt = pd.to_datetime(exploded["Date"], errors="coerce")
    dt = _drop_tz_if_present(dt)
    exploded["DateDay"] = dt.dt.normalize()

    exploded[col] = pd.to_numeric(exploded[col], errors="coerce")
    exploded = exploded.dropna(subset=[col, "DateDay"])
    if exploded.empty:
        return

    date_order = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in full_date_range]
    exploded["DateStr"] = exploded["DateDay"].dt.strftime("%Y-%m-%d")
    exploded["DateCat"] = pd.Categorical(exploded["DateStr"], categories=date_order, ordered=True)

    sns.set(style="white")
    plt.figure(figsize=figsize)

    _violin_with_backcompat(
        data=exploded,
        x="DateCat",
        y=col,
        order=date_order,
        color=violin_color,
        inner="quartile",
    )
    sns.stripplot(
        data=exploded,
        x="DateCat",
        y=col,
        order=date_order,
        jitter=jitter,
        size=point_size,
        color=point_color,
        alpha=point_alpha,
    )

    ax = plt.gca()
    plt.ylim(0, y_max_ms)
    plt.xlabel("Recording Date", fontsize=font_size_labels)
    plt.ylabel("Phrase Duration (ms)", fontsize=font_size_labels)

    tick_idx = list(range(0, len(date_order), max(1, int(xtick_every))))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([date_order[i] for i in tick_idx], rotation=90, fontsize=xtick_fontsize)

    if treatment_date_dt is not None and not pd.isna(treatment_date_dt):
        t_str = pd.Timestamp(treatment_date_dt).strftime("%Y-%m-%d")
        if t_str in date_order:
            t_idx = date_order.index(t_str)
            plt.axvline(x=t_idx, color="red", linestyle="--", label="Treatment Date")
            plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=int(dpi), transparent=bool(transparent))
    if show_plots:
        plt.show()
    else:
        plt.close()


def _build_daily_variance_df(
    exploded: pd.DataFrame,
    col: str,
    full_date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    df = exploded.copy()
    dt = pd.to_datetime(df["Date"], errors="coerce")
    dt = _drop_tz_if_present(dt)
    df["DateDay"] = dt.dt.normalize()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col, "DateDay"])
    if df.empty:
        return pd.DataFrame(columns=["DateDay", "Variance (ms^2)", "N"])

    daily = (
        df.groupby("DateDay")[col]
        .agg(
            N="size",
            Variance_ms2=lambda s: float(np.var(pd.to_numeric(s, errors="coerce").dropna().to_numpy(), ddof=1))
            if pd.to_numeric(s, errors="coerce").dropna().size >= 2 else np.nan
        )
        .reset_index()
        .rename(columns={"Variance_ms2": "Variance (ms^2)"})
    )

    base = pd.DataFrame({"DateDay": pd.to_datetime(full_date_range).normalize()})
    daily = base.merge(daily, on="DateDay", how="left")
    return daily


def _plot_one_syllable_variance_line(
    daily_var_df: pd.DataFrame,
    full_date_range: pd.DatetimeIndex,
    out_path: Path,
    *,
    line_color: str = "#2E4845",
    marker_color: Optional[str] = None,
    y_max_variance_ms2: Optional[float] = None,
    figsize: tuple[int, int] = (20, 8),
    font_size_labels: int = 30,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,
    treatment_date_dt: Optional[pd.Timestamp] = None,
    show_plots: bool = False,
    dpi: int = 300,
    transparent: bool = False,
):
    if daily_var_df.empty:
        return

    date_order = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in full_date_range]
    x = np.arange(len(date_order), dtype=float)
    y = pd.to_numeric(daily_var_df["Variance (ms^2)"], errors="coerce").to_numpy(dtype=float)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.plot(
        x, y,
        color=line_color,
        linewidth=2.0,
        marker="o",
        markersize=5,
        markerfacecolor=(marker_color or line_color),
        markeredgecolor=(marker_color or line_color),
        alpha=0.9,
    )

    ax.set_xlabel("Recording Date", fontsize=font_size_labels)
    ax.set_ylabel("Variance of Phrase Duration (ms²)", fontsize=font_size_labels)
    tick_idx = list(range(0, len(date_order), max(1, int(xtick_every))))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([date_order[i] for i in tick_idx], rotation=90, fontsize=xtick_fontsize)

    if y_max_variance_ms2 is not None:
        ax.set_ylim(0, float(y_max_variance_ms2))
    else:
        finite = y[np.isfinite(y)]
        if finite.size:
            ymax = float(np.nanmax(finite))
            ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)

    if treatment_date_dt is not None and not pd.isna(treatment_date_dt):
        t_str = pd.Timestamp(treatment_date_dt).strftime("%Y-%m-%d")
        if t_str in date_order:
            t_idx = date_order.index(t_str)
            ax.axvline(x=t_idx, color="red", linestyle="--", label="Treatment Date")
            ax.legend()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=int(dpi), transparent=bool(transparent))
    if show_plots:
        plt.show()
    else:
        plt.close()


def _overall_variance_for_label(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col].explode(), errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size < 2:
        return np.nan
    return float(np.var(vals, ddof=1))


def _select_top_variance_labels(df: pd.DataFrame, labels: Sequence[str], top_fraction: float = 0.30) -> List[str]:
    rows: List[Tuple[str, float]] = []
    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        v = _overall_variance_for_label(df, col)
        if np.isfinite(v):
            rows.append((str(lbl), float(v)))
    if not rows:
        return []
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    n = max(1, int(math.ceil(len(rows) * float(top_fraction))))
    return [lbl for lbl, _ in rows[:n]]


def _plot_aggregate_variance_lines(
    daily_var_map: Dict[str, pd.DataFrame],
    labels_to_plot: Sequence[str],
    full_date_range: pd.DatetimeIndex,
    out_path: Path,
    *,
    label_colors: Optional[Dict[str, str]] = None,
    monochrome: bool = False,
    monochrome_color: str = "#2E4845",
    y_max_variance_ms2: Optional[float] = None,
    figsize: tuple[int, int] = (22, 10),
    font_size_labels: int = 24,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,
    treatment_date_dt: Optional[pd.Timestamp] = None,
    show_plots: bool = False,
    dpi: int = 300,
    transparent: bool = False,
    title: Optional[str] = None,
):
    labels_to_plot = [str(x) for x in labels_to_plot if str(x) in daily_var_map]
    if not labels_to_plot:
        return

    date_order = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in full_date_range]
    x = np.arange(len(date_order), dtype=float)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    ymax_values: List[float] = []
    for lbl in labels_to_plot:
        daily_var_df = daily_var_map[str(lbl)]
        y = pd.to_numeric(daily_var_df["Variance (ms^2)"], errors="coerce").to_numpy(dtype=float)
        finite = y[np.isfinite(y)]
        if finite.size:
            ymax_values.append(float(np.nanmax(finite)))
        color = monochrome_color if monochrome else (label_colors or {}).get(str(lbl), monochrome_color)
        ax.plot(
            x, y,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            alpha=0.85,
            label=str(lbl),
        )

    ax.set_xlabel("Recording Date", fontsize=font_size_labels)
    ax.set_ylabel("Variance of Phrase Duration (ms²)", fontsize=font_size_labels)

    tick_idx = list(range(0, len(date_order), max(1, int(xtick_every))))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([date_order[i] for i in tick_idx], rotation=90, fontsize=xtick_fontsize)

    if y_max_variance_ms2 is not None:
        ax.set_ylim(0, float(y_max_variance_ms2))
    elif ymax_values:
        ymax = float(np.nanmax(ymax_values))
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)

    if treatment_date_dt is not None and not pd.isna(treatment_date_dt):
        t_str = pd.Timestamp(treatment_date_dt).strftime("%Y-%m-%d")
        if t_str in date_order:
            t_idx = date_order.index(t_str)
            ax.axvline(x=t_idx, color="red", linestyle="--", linewidth=1.5)

    if title:
        ax.set_title(title)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    n_labels = len(labels_to_plot)
    ncols = 1 if n_labels <= 20 else 2 if n_labels <= 40 else 3
    ax.legend(
        title="Syllable",
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        ncol=ncols,
        fontsize=9,
        title_fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=int(dpi), transparent=bool(transparent), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()


def _plot_batch_aligned_variance_lines(
    aligned_df: pd.DataFrame,
    out_path: Path,
    *,
    label_colors: Optional[Dict[str, str]] = None,
    monochrome: bool = False,
    monochrome_color: str = "#2E4845",
    y_max_variance_ms2: Optional[float] = None,
    figsize: tuple[int, int] = (22, 10),
    font_size_labels: int = 24,
    xtick_fontsize: int = 10,
    show_plots: bool = False,
    dpi: int = 300,
    transparent: bool = False,
    title: Optional[str] = None,
):
    if aligned_df.empty:
        return

    df = aligned_df.copy()
    df["relative_day"] = pd.to_numeric(df["relative_day"], errors="coerce")
    df["Variance (ms^2)"] = pd.to_numeric(df["Variance (ms^2)"], errors="coerce")
    df = df.dropna(subset=["relative_day", "Variance (ms^2)"])
    if df.empty:
        return

    df["relative_day"] = df["relative_day"].astype(int)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    for (animal_id, syllable), g in df.groupby(["animal_id", "syllable"], sort=False):
        g = g.sort_values("relative_day")
        color = monochrome_color if monochrome else (label_colors or {}).get(str(syllable), monochrome_color)
        ax.plot(
            g["relative_day"].to_numpy(),
            g["Variance (ms^2)"].to_numpy(),
            color=color,
            linewidth=1.1,
            alpha=0.28,
            zorder=1,
        )

    summary = (
        df.groupby("relative_day")["Variance (ms^2)"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .sort_values("relative_day")
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))

    ax.plot(
        summary["relative_day"].to_numpy(),
        summary["mean"].to_numpy(),
        color="black",
        linewidth=3.0,
        marker="o",
        markersize=5,
        zorder=3,
        label="Mean across bird-syllable trajectories",
    )
    valid_sem = summary["sem"].notna()
    if valid_sem.any():
        ax.fill_between(
            summary.loc[valid_sem, "relative_day"].to_numpy(),
            (summary.loc[valid_sem, "mean"] - summary.loc[valid_sem, "sem"]).to_numpy(),
            (summary.loc[valid_sem, "mean"] + summary.loc[valid_sem, "sem"]).to_numpy(),
            color="black",
            alpha=0.15,
            zorder=2,
        )

    xmin = int(df["relative_day"].min())
    xmax = int(df["relative_day"].max())
    ax.set_xlim(xmin, xmax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.6, zorder=0)

    if y_max_variance_ms2 is not None:
        ax.set_ylim(0, float(y_max_variance_ms2))
    else:
        finite = df["Variance (ms^2)"].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ymax = float(np.nanmax(finite))
            ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)

    ax.set_xlabel("Days relative to lesion", fontsize=font_size_labels)
    ax.set_ylabel("Variance of Phrase Duration (ms²)", fontsize=font_size_labels)

    tick_positions = np.arange(xmin, xmax + 1, 1, dtype=int)
    if tick_positions.size > 18:
        step = int(np.ceil(tick_positions.size / 18.0))
        tick_positions = tick_positions[::step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(t)) for t in tick_positions], rotation=0, fontsize=xtick_fontsize)

    if title:
        ax.set_title(title)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if not monochrome:
        sylls = sorted(df["syllable"].astype(str).unique(), key=lambda s: (0, int(s)) if str(s).lstrip('-').isdigit() else (1, str(s)))
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=(label_colors or {}).get(s, monochrome_color), lw=2, label=s) for s in sylls]
        handles.append(Line2D([0], [0], color="black", lw=3, marker="o", label="Mean"))
        ncols = 1 if len(sylls) <= 20 else 2 if len(sylls) <= 40 else 3
        ax.legend(handles=handles, title="Syllable", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", ncol=ncols, fontsize=9, title_fontsize=10)
    else:
        ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=int(dpi), transparent=bool(transparent), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()


def _plot_batch_aligned_variance_by_lesion_group(
    aligned_df: pd.DataFrame,
    out_path: Path,
    *,
    split_panels: bool = False,
    y_max_variance_ms2: Optional[float] = None,
    figsize: tuple[int, int] = (22, 10),
    font_size_labels: int = 24,
    xtick_fontsize: int = 10,
    show_plots: bool = False,
    dpi: int = 300,
    transparent: bool = False,
    title: Optional[str] = None,
):
    if aligned_df.empty or "lesion_group" not in aligned_df.columns:
        return

    df = aligned_df.copy()
    df["relative_day"] = pd.to_numeric(df["relative_day"], errors="coerce")
    df["Variance (ms^2)"] = pd.to_numeric(df["Variance (ms^2)"], errors="coerce")
    df["lesion_group"] = df["lesion_group"].astype(str)
    df = df.dropna(subset=["relative_day", "Variance (ms^2)"])
    df = df[df["lesion_group"].isin(LESION_GROUP_ORDER)]
    if df.empty:
        return
    df["relative_day"] = df["relative_day"].astype(int)

    finite = df["Variance (ms^2)"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    auto_ymax = float(np.nanmax(finite)) * 1.05 if finite.size else 1.0
    ymax = float(y_max_variance_ms2) if y_max_variance_ms2 is not None else auto_ymax
    xmin = int(df["relative_day"].min())
    xmax = int(df["relative_day"].max())
    tick_positions = np.arange(xmin, xmax + 1, 1, dtype=int)
    if tick_positions.size > 18:
        step = int(np.ceil(tick_positions.size / 18.0))
        tick_positions = tick_positions[::step]

    def _draw_one(ax, subdf: pd.DataFrame, group_name: str):
        color = LESION_GROUP_COLORS[group_name]
        for (_, _), g in subdf.groupby(["animal_id", "syllable"], sort=False):
            g = g.sort_values("relative_day")
            ax.plot(
                g["relative_day"].to_numpy(),
                g["Variance (ms^2)"].to_numpy(),
                color=color,
                linewidth=1.0,
                alpha=0.18,
                zorder=1,
            )
        summary = (
            subdf.groupby("relative_day")["Variance (ms^2)"]
            .agg(["mean", "count", "std"])
            .reset_index()
            .sort_values("relative_day")
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
        ax.plot(
            summary["relative_day"].to_numpy(),
            summary["mean"].to_numpy(),
            color=color,
            linewidth=3.0,
            marker="o",
            markersize=5,
            zorder=3,
            label=group_name,
        )
        valid_sem = summary["sem"].notna()
        if valid_sem.any():
            ax.fill_between(
                summary.loc[valid_sem, "relative_day"].to_numpy(),
                (summary.loc[valid_sem, "mean"] - summary.loc[valid_sem, "sem"]).to_numpy(),
                (summary.loc[valid_sem, "mean"] + summary.loc[valid_sem, "sem"]).to_numpy(),
                color=color,
                alpha=0.16,
                zorder=2,
            )
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, zorder=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(t)) for t in tick_positions], rotation=0, fontsize=xtick_fontsize)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    if split_panels:
        fig, axes = plt.subplots(len(LESION_GROUP_ORDER), 1, figsize=(figsize[0], max(figsize[1], 14)), sharex=True, sharey=True)
        if len(LESION_GROUP_ORDER) == 1:
            axes = [axes]
        for ax, group_name in zip(axes, LESION_GROUP_ORDER):
            subdf = df[df["lesion_group"] == group_name].copy()
            if subdf.empty:
                ax.text(0.5, 0.5, f"No data for {group_name}", ha="center", va="center", transform=ax.transAxes)
                ax.axvline(0, color="red", linestyle="--", linewidth=1.5, zorder=0)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(0, ymax)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
            else:
                _draw_one(ax, subdf, group_name)
            ax.set_ylabel("Variance of Phrase Duration (ms²)", fontsize=max(16, font_size_labels - 4))
            ax.set_title(group_name, color=LESION_GROUP_COLORS[group_name], fontsize=max(16, font_size_labels - 2))
        axes[-1].set_xlabel("Days relative to lesion", fontsize=font_size_labels)
        if title:
            fig.suptitle(title, fontsize=font_size_labels + 4, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
        fig.savefig(out_path, format="png", dpi=int(dpi), transparent=bool(transparent), bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        return

    plt.figure(figsize=figsize)
    ax = plt.gca()
    for group_name in LESION_GROUP_ORDER:
        subdf = df[df["lesion_group"] == group_name].copy()
        if subdf.empty:
            continue
        _draw_one(ax, subdf, group_name)
    ax.set_xlabel("Days relative to lesion", fontsize=font_size_labels)
    ax.set_ylabel("Variance of Phrase Duration (ms²)", fontsize=font_size_labels)
    if title:
        ax.set_title(title)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=LESION_GROUP_COLORS[g], lw=3, marker="o", label=g) for g in LESION_GROUP_ORDER if (df["lesion_group"] == g).any()]
    ax.legend(handles=handles, title="Lesion hit type", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=int(dpi), transparent=bool(transparent), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()


def _prepare_plot_df_from_merged(
    *,
    premerged_annotations_df: Optional[pd.DataFrame] = None,
    premerged_annotations_path: Optional[Union[str, Path]] = None,
    decoded_database_json: Optional[Union[str, Path]] = None,
    song_detection_json: Optional[Union[str, Path]] = None,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    syllables_subset: Optional[Sequence[str]] = None,
    treatment_date: Optional[str | pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, List[str], pd.DatetimeIndex, Optional[pd.Timestamp], str]:
    if premerged_annotations_df is not None:
        df_merged = premerged_annotations_df.copy()
        merged_path_hint = Path(premerged_annotations_path) if premerged_annotations_path else None
    else:
        if not _HAS_MERGE_BUILDER:
            raise RuntimeError(
                "merge_annotations_from_split_songs.build_decoded_with_split_labels is not available; "
                "install/put it on PYTHONPATH to use internal merging."
            )
        decoded_database_json = Path(decoded_database_json)  # type: ignore[arg-type]
        song_detection_json = Path(song_detection_json)  # type: ignore[arg-type]

        ann = build_decoded_with_split_labels(
            decoded_database_json=decoded_database_json,
            song_detection_json=song_detection_json,
            only_song_present=True,
            compute_durations=True,
            add_recording_datetime=True,
            songs_only=True,
            flatten_spec_params=True,
            max_gap_between_song_segments=max_gap_between_song_segments,
            segment_index_offset=segment_index_offset,
            merge_repeated_syllables=merge_repeated_syllables,
            repeat_gap_ms=repeat_gap_ms,
            repeat_gap_inclusive=repeat_gap_inclusive,
        )
        df_merged = ann.annotations_appended_df.copy()
        merged_path_hint = Path(decoded_database_json)

    dt = _choose_dt_series(df_merged)
    dt = _drop_tz_if_present(pd.to_datetime(dt, errors="coerce"))
    df_merged = df_merged.assign(_dt=dt)
    df_merged = df_merged.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
    if df_merged.empty:
        raise ValueError("Merged annotations dataframe has no valid datetime rows.")
    df_merged["Date"] = df_merged["_dt"].dt.date

    spans_col = _find_best_spans_column(df_merged)
    if spans_col is None:
        raise ValueError("No *_dict spans column found in merged annotations.")

    labels = _collect_unique_labels_sorted(df_merged, spans_col)
    if syllables_subset is not None:
        keep = set(map(str, syllables_subset))
        labels = [str(l) for l in labels if str(l) in keep]
    if not labels:
        raise ValueError("No syllables to plot after filtering.")

    df_merged, _ = _build_durations_columns_from_dicts(df_merged, labels)

    earliest = pd.to_datetime(df_merged["_dt"].min()).normalize()
    latest = pd.to_datetime(df_merged["_dt"].max()).normalize()
    full_date_range = pd.date_range(start=earliest, end=latest, freq="D")

    fallback_treat = None
    if "treatment_date" in df_merged.columns and pd.notna(df_merged["treatment_date"]).any():
        try:
            fallback_treat = str(df_merged["treatment_date"].dropna().iloc[0])
        except Exception:
            fallback_treat = None
    t_dt = _coerce_treatment_date(treatment_date, fallback_treat)

    animal_id = _infer_animal_id(df_merged, merged_path_hint)
    return df_merged, labels, full_date_range, t_dt, animal_id


def _prepare_plot_df_from_organizer(
    *,
    decoded_database_json: Union[str, Path],
    creation_metadata_json: Optional[Union[str, Path]] = None,
    only_song_present: bool = False,
    syllables_subset: Optional[Sequence[str]] = None,
    treatment_date: Optional[str | pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, List[str], pd.DatetimeIndex, Optional[pd.Timestamp], str, Any]:
    out = _build_organized(
        decoded_database_json=decoded_database_json,
        creation_metadata_json=None if _USING_SERIAL_SEGMENTS else creation_metadata_json,
        only_song_present=only_song_present,
        compute_durations=True,
        add_recording_datetime=True if _USING_SERIAL_SEGMENTS else False,
    )

    df = out.organized_df.copy()
    df["Date"] = pd.to_datetime(df.get("Date", pd.Series([pd.NaT] * len(df))), errors="coerce")
    df["Date"] = _drop_tz_if_present(df["Date"])
    if not df["Date"].notna().any():
        raise ValueError("Organizer output has no valid dates.")

    earliest_date = df["Date"].min().normalize()
    latest_date = df["Date"].max().normalize()
    full_date_range = pd.date_range(start=earliest_date, end=latest_date, freq="D")

    t_dt = _coerce_treatment_date(treatment_date, getattr(out, "treatment_date", None))

    labels = out.unique_syllable_labels
    if syllables_subset is not None:
        keep = set(map(str, syllables_subset))
        labels = [lab for lab in labels if str(lab) in keep]
    if not labels:
        raise ValueError("No syllables to plot after filtering.")

    try:
        animal_id = str(df["Animal ID"].dropna().iloc[0])
    except Exception:
        animal_id = "unknown_animal"

    df["_DateOnly"] = df["Date"].dt.date
    df["Date"] = df["_DateOnly"]
    return df, [str(x) for x in labels], full_date_range, t_dt, animal_id, out


def _run_plots_on_prepared_df(
    *,
    df: pd.DataFrame,
    labels: Sequence[str],
    full_date_range: pd.DatetimeIndex,
    treatment_date_dt: Optional[pd.Timestamp],
    animal_id: str,
    save_dir: Path,
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    y_max_ms: int = 25_000,
    y_max_variance_ms2: Optional[float] = None,
    point_alpha: float = 0.7,
    point_size: int = 5,
    jitter: float | bool = True,
    dpi: int = 300,
    transparent: bool = False,
    figsize: tuple[int, int] = (20, 11),
    variance_figsize: tuple[int, int] = (20, 8),
    aggregate_variance_figsize: tuple[int, int] = (22, 10),
    font_size_labels: int = 30,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,
    show_plots: bool = False,
    make_green_plots: bool = True,
    make_colored_plots: bool = True,
    make_green_variance_plots: bool = True,
    make_colored_variance_plots: bool = True,
    make_green_aggregate_variance_plots: bool = True,
    make_colored_aggregate_variance_plots: bool = True,
) -> Dict[str, Any]:
    label_colors = _load_label_colors(labels, fixed_label_colors_json=fixed_label_colors_json)
    daily_var_map: Dict[str, pd.DataFrame] = {}

    for lbl in labels:
        col = f"syllable_{lbl}_durations"
        if col not in df.columns:
            continue

        exploded = df[["Date", col]].explode(col)
        if exploded[col].dropna().empty:
            continue

        plot_color = label_colors.get(str(lbl), "#2E4845")

        if make_green_plots:
            out_path_green = save_dir / f"{animal_id}_syllable_{lbl}_phrase_duration_plot_green.png"
            _plot_one_syllable_duration(
                exploded=exploded,
                col=col,
                full_date_range=full_date_range,
                out_path=out_path_green,
                violin_color="lightgray",
                point_color="#2E4845",
                y_max_ms=y_max_ms,
                jitter=jitter,
                point_size=point_size,
                point_alpha=point_alpha,
                figsize=figsize,
                font_size_labels=font_size_labels,
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                treatment_date_dt=treatment_date_dt,
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
            )

        if make_colored_plots:
            out_path_colored = save_dir / f"{animal_id}_syllable_{lbl}_phrase_duration_plot_colored.png"
            _plot_one_syllable_duration(
                exploded=exploded,
                col=col,
                full_date_range=full_date_range,
                out_path=out_path_colored,
                violin_color=plot_color,
                point_color=plot_color,
                y_max_ms=y_max_ms,
                jitter=jitter,
                point_size=point_size,
                point_alpha=point_alpha,
                figsize=figsize,
                font_size_labels=font_size_labels,
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                treatment_date_dt=treatment_date_dt,
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
            )

        daily_var_df = _build_daily_variance_df(exploded, col, full_date_range)
        if daily_var_df["Variance (ms^2)"].notna().sum() == 0:
            continue
        daily_var_map[str(lbl)] = daily_var_df

        if make_green_variance_plots:
            out_var_green = save_dir / f"{animal_id}_syllable_{lbl}_phrase_duration_variance_plot_green.png"
            _plot_one_syllable_variance_line(
                daily_var_df=daily_var_df,
                full_date_range=full_date_range,
                out_path=out_var_green,
                line_color="#2E4845",
                marker_color="#2E4845",
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=variance_figsize,
                font_size_labels=font_size_labels,
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                treatment_date_dt=treatment_date_dt,
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
            )

        if make_colored_variance_plots:
            out_var_colored = save_dir / f"{animal_id}_syllable_{lbl}_phrase_duration_variance_plot_colored.png"
            _plot_one_syllable_variance_line(
                daily_var_df=daily_var_df,
                full_date_range=full_date_range,
                out_path=out_var_colored,
                line_color=plot_color,
                marker_color=plot_color,
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=variance_figsize,
                font_size_labels=font_size_labels,
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                treatment_date_dt=treatment_date_dt,
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
            )

    if not daily_var_map:
        return {"daily_var_map": {}, "top30_labels": [], "label_colors": label_colors}

    top30_labels = _select_top_variance_labels(df, labels, top_fraction=0.30)

    if make_green_aggregate_variance_plots:
        _plot_aggregate_variance_lines(
            daily_var_map=daily_var_map,
            labels_to_plot=labels,
            full_date_range=full_date_range,
            out_path=save_dir / f"{animal_id}_aggregate_phrase_duration_variance_all_green.png",
            monochrome=True,
            monochrome_color="#2E4845",
            y_max_variance_ms2=y_max_variance_ms2,
            figsize=aggregate_variance_figsize,
            font_size_labels=max(16, font_size_labels - 4),
            xtick_fontsize=xtick_fontsize,
            xtick_every=xtick_every,
            treatment_date_dt=treatment_date_dt,
            show_plots=show_plots,
            dpi=dpi,
            transparent=transparent,
            title=f"{animal_id} — Daily phrase-duration variance (all syllables)",
        )
        if top30_labels:
            _plot_aggregate_variance_lines(
                daily_var_map=daily_var_map,
                labels_to_plot=top30_labels,
                full_date_range=full_date_range,
                out_path=save_dir / f"{animal_id}_aggregate_phrase_duration_variance_top30_green.png",
                monochrome=True,
                monochrome_color="#2E4845",
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=aggregate_variance_figsize,
                font_size_labels=max(16, font_size_labels - 4),
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                treatment_date_dt=treatment_date_dt,
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
                title=f"{animal_id} — Daily phrase-duration variance (top 30% variance syllables)",
            )

    if make_colored_aggregate_variance_plots:
        _plot_aggregate_variance_lines(
            daily_var_map=daily_var_map,
            labels_to_plot=labels,
            full_date_range=full_date_range,
            out_path=save_dir / f"{animal_id}_aggregate_phrase_duration_variance_all_colored.png",
            label_colors=label_colors,
            monochrome=False,
            y_max_variance_ms2=y_max_variance_ms2,
            figsize=aggregate_variance_figsize,
            font_size_labels=max(16, font_size_labels - 4),
            xtick_fontsize=xtick_fontsize,
            xtick_every=xtick_every,
            treatment_date_dt=treatment_date_dt,
            show_plots=show_plots,
            dpi=dpi,
            transparent=transparent,
            title=f"{animal_id} — Daily phrase-duration variance (all syllables)",
        )
        if top30_labels:
            _plot_aggregate_variance_lines(
                daily_var_map=daily_var_map,
                labels_to_plot=top30_labels,
                full_date_range=full_date_range,
                out_path=save_dir / f"{animal_id}_aggregate_phrase_duration_variance_top30_colored.png",
                label_colors=label_colors,
                monochrome=False,
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=aggregate_variance_figsize,
                font_size_labels=max(16, font_size_labels - 4),
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                treatment_date_dt=treatment_date_dt,
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
                title=f"{animal_id} — Daily phrase-duration variance (top 30% variance syllables)",
            )

    return {"daily_var_map": daily_var_map, "top30_labels": top30_labels, "label_colors": label_colors}


def graph_phrase_duration_over_days(
    decoded_database_json: str | Path | None = None,
    creation_metadata_json: str | Path | None = None,
    save_output_to_this_file_path: str | Path = ".",
    *,
    premerged_annotations_df: Optional[pd.DataFrame] = None,
    premerged_annotations_path: Optional[Union[str, Path]] = None,
    song_detection_json: Optional[Union[str, Path]] = None,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    only_song_present: bool = False,
    y_max_ms: int = 25_000,
    y_max_variance_ms2: Optional[float] = None,
    point_alpha: float = 0.7,
    point_size: int = 5,
    jitter: float | bool = True,
    dpi: int = 300,
    transparent: bool = False,
    figsize: tuple[int, int] = (20, 11),
    variance_figsize: tuple[int, int] = (20, 8),
    aggregate_variance_figsize: tuple[int, int] = (22, 10),
    font_size_labels: int = 30,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,
    show_plots: bool = False,
    syllables_subset: Optional[Sequence[str]] = None,
    treatment_date: Optional[str | pd.Timestamp] = None,
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    make_green_plots: bool = True,
    make_colored_plots: bool = True,
    make_green_variance_plots: bool = True,
    make_colored_variance_plots: bool = True,
    make_green_aggregate_variance_plots: bool = True,
    make_colored_aggregate_variance_plots: bool = True,
) -> "OrganizedDataset | None":
    save_dir = Path(save_output_to_this_file_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if premerged_annotations_df is not None or (decoded_database_json and song_detection_json):
        df, labels, full_date_range, t_dt, animal_id = _prepare_plot_df_from_merged(
            premerged_annotations_df=premerged_annotations_df,
            premerged_annotations_path=premerged_annotations_path,
            decoded_database_json=decoded_database_json,
            song_detection_json=song_detection_json,
            max_gap_between_song_segments=max_gap_between_song_segments,
            segment_index_offset=segment_index_offset,
            merge_repeated_syllables=merge_repeated_syllables,
            repeat_gap_ms=repeat_gap_ms,
            repeat_gap_inclusive=repeat_gap_inclusive,
            syllables_subset=syllables_subset,
            treatment_date=treatment_date,
        )
        _run_plots_on_prepared_df(
            df=df,
            labels=labels,
            full_date_range=full_date_range,
            treatment_date_dt=t_dt,
            animal_id=animal_id,
            save_dir=save_dir,
            fixed_label_colors_json=fixed_label_colors_json,
            y_max_ms=y_max_ms,
            y_max_variance_ms2=y_max_variance_ms2,
            point_alpha=point_alpha,
            point_size=point_size,
            jitter=jitter,
            dpi=dpi,
            transparent=transparent,
            figsize=figsize,
            variance_figsize=variance_figsize,
            aggregate_variance_figsize=aggregate_variance_figsize,
            font_size_labels=font_size_labels,
            xtick_fontsize=xtick_fontsize,
            xtick_every=xtick_every,
            show_plots=show_plots,
            make_green_plots=make_green_plots,
            make_colored_plots=make_colored_plots,
            make_green_variance_plots=make_green_variance_plots,
            make_colored_variance_plots=make_colored_variance_plots,
            make_green_aggregate_variance_plots=make_green_aggregate_variance_plots,
            make_colored_aggregate_variance_plots=make_colored_aggregate_variance_plots,
        )
        return None

    df, labels, full_date_range, t_dt, animal_id, out = _prepare_plot_df_from_organizer(
        decoded_database_json=decoded_database_json,  # type: ignore[arg-type]
        creation_metadata_json=creation_metadata_json,
        only_song_present=only_song_present,
        syllables_subset=syllables_subset,
        treatment_date=treatment_date,
    )
    _run_plots_on_prepared_df(
        df=df,
        labels=labels,
        full_date_range=full_date_range,
        treatment_date_dt=t_dt,
        animal_id=animal_id,
        save_dir=save_dir,
        fixed_label_colors_json=fixed_label_colors_json,
        y_max_ms=y_max_ms,
        y_max_variance_ms2=y_max_variance_ms2,
        point_alpha=point_alpha,
        point_size=point_size,
        jitter=jitter,
        dpi=dpi,
        transparent=transparent,
        figsize=figsize,
        variance_figsize=variance_figsize,
        aggregate_variance_figsize=aggregate_variance_figsize,
        font_size_labels=font_size_labels,
        xtick_fontsize=xtick_fontsize,
        xtick_every=xtick_every,
        show_plots=show_plots,
        make_green_plots=make_green_plots,
        make_colored_plots=make_colored_plots,
        make_green_variance_plots=make_green_variance_plots,
        make_colored_variance_plots=make_colored_variance_plots,
        make_green_aggregate_variance_plots=make_green_aggregate_variance_plots,
        make_colored_aggregate_variance_plots=make_colored_aggregate_variance_plots,
    )
    return out


def _find_json_for_animal(
    root: Path,
    animal_id: str,
    decoded_suffix: str = "decoded_database.json",
    detect_suffix: str = "song_detection.json",
) -> Tuple[Optional[Path], Optional[Path]]:
    decoded_candidates = [
        p for p in root.rglob(f"*{animal_id}*{decoded_suffix}")
        if not p.name.startswith("._")
    ]
    detect_candidates = [
        p for p in root.rglob(f"*{animal_id}*{detect_suffix}")
        if not p.name.startswith("._")
    ]

    key = lambda p: (len(str(p)), str(p))
    decoded_candidates = sorted(decoded_candidates, key=key)
    detect_candidates = sorted(detect_candidates, key=key)

    decoded_path = decoded_candidates[0] if decoded_candidates else None
    detect_path = detect_candidates[0] if detect_candidates else None
    return decoded_path, detect_path


def run_batch_phrase_duration_over_days_from_excel(
    metadata_excel: Union[str, Path],
    json_root: Union[str, Path],
    *,
    sheet_name: Union[int, str] = "metadata",
    id_col: str = "Animal ID",
    treatment_date_col: str = "Treatment date",
    hit_type_sheet_name: Union[int, str] = "animal_hit_type_summary",
    hit_type_id_col: str = "Animal ID",
    hit_type_col: str = "Lesion hit type",
    output_root: Optional[Union[str, Path]] = None,
    decoded_suffix: str = "decoded_database.json",
    detect_suffix: str = "song_detection.json",
    syllables_subset: Optional[Sequence[str]] = None,
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
    max_gap_between_song_segments: int = 500,
    segment_index_offset: int = 0,
    merge_repeated_syllables: bool = True,
    repeat_gap_ms: float = 10.0,
    repeat_gap_inclusive: bool = False,
    y_max_ms: int = 25_000,
    y_max_variance_ms2: Optional[float] = None,
    point_alpha: float = 0.7,
    point_size: int = 5,
    jitter: float | bool = True,
    dpi: int = 300,
    transparent: bool = False,
    figsize: tuple[int, int] = (20, 11),
    variance_figsize: tuple[int, int] = (20, 8),
    aggregate_variance_figsize: tuple[int, int] = (22, 10),
    font_size_labels: int = 30,
    xtick_fontsize: int = 8,
    xtick_every: int = 1,
    show_plots: bool = False,
    make_green_plots: bool = True,
    make_colored_plots: bool = True,
    make_green_variance_plots: bool = True,
    make_colored_variance_plots: bool = True,
    make_green_aggregate_variance_plots: bool = True,
    make_colored_aggregate_variance_plots: bool = True,
) -> Dict[str, BatchPhraseDurationOverDaysRecord]:
    metadata_excel = Path(metadata_excel)
    json_root = Path(json_root)
    output_root = Path(output_root) if output_root is not None else None

    meta_df = pd.read_excel(metadata_excel, sheet_name=sheet_name)
    meta_df = meta_df.rename(columns={str(c): str(c).strip() for c in meta_df.columns})

    if id_col not in meta_df.columns:
        raise ValueError(f"Column {id_col!r} not found in {metadata_excel}")
    if treatment_date_col not in meta_df.columns:
        raise ValueError(f"Column {treatment_date_col!r} not found in {metadata_excel}")

    lesion_group_map = _load_lesion_group_map_from_excel(
        metadata_excel,
        sheet_name=hit_type_sheet_name,
        id_col=hit_type_id_col,
        lesion_hit_type_col=hit_type_col,
    )

    records: Dict[str, BatchPhraseDurationOverDaysRecord] = {}
    batch_aligned_rows_all: List[Dict[str, Any]] = []
    batch_aligned_rows_top30: List[Dict[str, Any]] = []

    summary_root = (output_root if output_root is not None else (json_root / "phrase_duration_batch_outputs"))
    summary_dir = Path(summary_root) / "_batch_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for animal_id, group in meta_df.groupby(id_col):
        animal_id = str(animal_id).strip()
        if not animal_id:
            continue

        tvals = pd.to_datetime(group[treatment_date_col], errors="coerce").dropna().unique()
        treatment_date = pd.Timestamp(tvals[0]).strftime("%Y-%m-%d") if len(tvals) > 0 else None
        lesion_group = lesion_group_map.get(animal_id)

        decoded_path, detect_path = _find_json_for_animal(
            json_root,
            animal_id,
            decoded_suffix=decoded_suffix,
            detect_suffix=detect_suffix,
        )

        if decoded_path is None or detect_path is None:
            msg = f"Missing JSON(s) for {animal_id}: decoded={decoded_path}, detect={detect_path}"
            print(f"[WARN] {msg}")
            records[animal_id] = BatchPhraseDurationOverDaysRecord(
                animal_id=animal_id,
                treatment_date=treatment_date,
                decoded_database_json=decoded_path,
                song_detection_json=detect_path,
                output_dir=None,
                status="skipped",
                message=msg,
            )
            continue

        if treatment_date is None:
            msg = f"No valid treatment date for {animal_id} in {metadata_excel.name}"
            print(f"[WARN] {msg}")
            records[animal_id] = BatchPhraseDurationOverDaysRecord(
                animal_id=animal_id,
                treatment_date=None,
                decoded_database_json=decoded_path,
                song_detection_json=detect_path,
                output_dir=None,
                status="skipped",
                message=msg,
            )
            continue

        if output_root is None:
            outdir = decoded_path.parent / "figures" / "phrase_duration_over_days"
        else:
            outdir = output_root / animal_id / "phrase_duration_over_days"
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[RUN] {animal_id} | treatment_date={treatment_date} | decoded={decoded_path.name} | detect={detect_path.name}")
            df, labels, full_date_range, t_dt, prepared_animal_id = _prepare_plot_df_from_merged(
                decoded_database_json=decoded_path,
                song_detection_json=detect_path,
                max_gap_between_song_segments=max_gap_between_song_segments,
                segment_index_offset=segment_index_offset,
                merge_repeated_syllables=merge_repeated_syllables,
                repeat_gap_ms=repeat_gap_ms,
                repeat_gap_inclusive=repeat_gap_inclusive,
                syllables_subset=syllables_subset,
                treatment_date=treatment_date,
            )
            plot_info = _run_plots_on_prepared_df(
                df=df,
                labels=labels,
                full_date_range=full_date_range,
                treatment_date_dt=t_dt,
                animal_id=prepared_animal_id,
                save_dir=outdir,
                fixed_label_colors_json=fixed_label_colors_json,
                y_max_ms=y_max_ms,
                y_max_variance_ms2=y_max_variance_ms2,
                point_alpha=point_alpha,
                point_size=point_size,
                jitter=jitter,
                dpi=dpi,
                transparent=transparent,
                figsize=figsize,
                variance_figsize=variance_figsize,
                aggregate_variance_figsize=aggregate_variance_figsize,
                font_size_labels=font_size_labels,
                xtick_fontsize=xtick_fontsize,
                xtick_every=xtick_every,
                show_plots=show_plots,
                make_green_plots=make_green_plots,
                make_colored_plots=make_colored_plots,
                make_green_variance_plots=make_green_variance_plots,
                make_colored_variance_plots=make_colored_variance_plots,
                make_green_aggregate_variance_plots=make_green_aggregate_variance_plots,
                make_colored_aggregate_variance_plots=make_colored_aggregate_variance_plots,
            )

            if t_dt is not None and plot_info.get("daily_var_map"):
                top30_set = set(map(str, plot_info.get("top30_labels", [])))
                for lbl, daily_df in plot_info["daily_var_map"].items():
                    tmp = daily_df.copy()
                    tmp["DateDay"] = pd.to_datetime(tmp["DateDay"], errors="coerce").dt.normalize()
                    tmp = tmp.dropna(subset=["DateDay", "Variance (ms^2)"])
                    if tmp.empty:
                        continue
                    tmp["relative_day"] = (tmp["DateDay"] - pd.Timestamp(t_dt).normalize()).dt.days
                    for _, row in tmp.iterrows():
                        rec = {
                            "animal_id": animal_id,
                            "syllable": str(lbl),
                            "relative_day": int(row["relative_day"]),
                            "Variance (ms^2)": float(row["Variance (ms^2)"]),
                            "lesion_group": lesion_group,
                        }
                        batch_aligned_rows_all.append(rec)
                        if str(lbl) in top30_set:
                            batch_aligned_rows_top30.append(dict(rec))

            records[animal_id] = BatchPhraseDurationOverDaysRecord(
                animal_id=animal_id,
                treatment_date=treatment_date,
                decoded_database_json=decoded_path,
                song_detection_json=detect_path,
                output_dir=outdir,
                status="ok",
                message=None,
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[ERROR] {animal_id}: {msg}")
            records[animal_id] = BatchPhraseDurationOverDaysRecord(
                animal_id=animal_id,
                treatment_date=treatment_date,
                decoded_database_json=decoded_path,
                song_detection_json=detect_path,
                output_dir=outdir,
                status="error",
                message=msg,
            )

    all_df = pd.DataFrame(batch_aligned_rows_all)
    top30_df = pd.DataFrame(batch_aligned_rows_top30)

    if not all_df.empty:
        if make_green_aggregate_variance_plots:
            _plot_batch_aligned_variance_lines(
                aligned_df=all_df,
                out_path=summary_dir / "batch_aligned_phrase_duration_variance_all_green.png",
                monochrome=True,
                monochrome_color="#2E4845",
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=aggregate_variance_figsize,
                font_size_labels=max(16, font_size_labels - 4),
                xtick_fontsize=max(9, xtick_fontsize),
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
                title="All syllables — phrase-duration variance aligned to lesion day",
            )
        if make_colored_aggregate_variance_plots:
            batch_label_colors = _load_label_colors(sorted(all_df["syllable"].astype(str).unique()), fixed_label_colors_json=fixed_label_colors_json)
            _plot_batch_aligned_variance_lines(
                aligned_df=all_df,
                out_path=summary_dir / "batch_aligned_phrase_duration_variance_all_colored.png",
                label_colors=batch_label_colors,
                monochrome=False,
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=aggregate_variance_figsize,
                font_size_labels=max(16, font_size_labels - 4),
                xtick_fontsize=max(9, xtick_fontsize),
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
                title="All syllables — phrase-duration variance aligned to lesion day",
            )

    if not top30_df.empty:
        if make_green_aggregate_variance_plots:
            _plot_batch_aligned_variance_lines(
                aligned_df=top30_df,
                out_path=summary_dir / "batch_aligned_phrase_duration_variance_top30_green.png",
                monochrome=True,
                monochrome_color="#2E4845",
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=aggregate_variance_figsize,
                font_size_labels=max(16, font_size_labels - 4),
                xtick_fontsize=max(9, xtick_fontsize),
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
                title="Top 30% variance syllables — phrase-duration variance aligned to lesion day",
            )
        if make_colored_aggregate_variance_plots:
            batch_top_label_colors = _load_label_colors(sorted(top30_df["syllable"].astype(str).unique()), fixed_label_colors_json=fixed_label_colors_json)
            _plot_batch_aligned_variance_lines(
                aligned_df=top30_df,
                out_path=summary_dir / "batch_aligned_phrase_duration_variance_top30_colored.png",
                label_colors=batch_top_label_colors,
                monochrome=False,
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=aggregate_variance_figsize,
                font_size_labels=max(16, font_size_labels - 4),
                xtick_fontsize=max(9, xtick_fontsize),
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
                title="Top 30% variance syllables — phrase-duration variance aligned to lesion day",
            )

        top30_lesion_df = top30_df.dropna(subset=["lesion_group"]).copy()
        if not top30_lesion_df.empty:
            _plot_batch_aligned_variance_by_lesion_group(
                aligned_df=top30_lesion_df,
                out_path=summary_dir / "batch_aligned_phrase_duration_variance_top30_by_lesion_type_combined.png",
                split_panels=False,
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=aggregate_variance_figsize,
                font_size_labels=max(16, font_size_labels - 4),
                xtick_fontsize=max(9, xtick_fontsize),
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
                title="Top 30% variance syllables — aligned to lesion day, colored by lesion hit type",
            )
            _plot_batch_aligned_variance_by_lesion_group(
                aligned_df=top30_lesion_df,
                out_path=summary_dir / "batch_aligned_phrase_duration_variance_top30_by_lesion_type_panels.png",
                split_panels=True,
                y_max_variance_ms2=y_max_variance_ms2,
                figsize=aggregate_variance_figsize,
                font_size_labels=max(16, font_size_labels - 4),
                xtick_fontsize=max(9, xtick_fontsize),
                show_plots=show_plots,
                dpi=dpi,
                transparent=transparent,
                title="Top 30% variance syllables — aligned to lesion day by lesion hit type",
            )

    return records


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Plot per-syllable phrase duration distributions and daily variance across calendar days. "
                    "Can run on one bird or batch across a root directory using an Excel metadata file."
    )

    ap.add_argument("--decoded", type=str, default=None, help="Path to *_decoded_database.json")
    ap.add_argument("--creation-meta", type=str, default=None, help="Path to *_creation_data.json (unused with serial organizer)")
    ap.add_argument("--save-dir", type=str, default=None, help="Directory to save PNGs for single-bird mode")
    ap.add_argument("--song-detection", type=str, default=None, help="Path to *_song_detection.json (enables internal merge)")

    ap.add_argument("--metadata-excel", type=str, default=None, help="Excel metadata path for batch mode")
    ap.add_argument("--json-root", type=str, default=None, help="Root directory containing per-bird JSONs for batch mode")
    ap.add_argument("--sheet-name", type=str, default="metadata")
    ap.add_argument("--id-col", type=str, default="Animal ID")
    ap.add_argument("--treatment-date-col", type=str, default="Treatment date")
    ap.add_argument("--hit-type-sheet-name", type=str, default="animal_hit_type_summary")
    ap.add_argument("--hit-type-id-col", type=str, default="Animal ID")
    ap.add_argument("--hit-type-col", type=str, default="Lesion hit type")
    ap.add_argument("--output-root", type=str, default=None, help="Optional central output root for batch mode")

    ap.add_argument("--ann-gap-ms", type=int, default=500)
    ap.add_argument("--seg-offset", type=int, default=0)
    ap.add_argument("--merge-repeats", action="store_true")
    ap.add_argument("--repeat-gap-ms", type=float, default=10.0)
    ap.add_argument("--repeat-gap-inclusive", action="store_true")
    ap.add_argument("--only-song-present", action="store_true")
    ap.add_argument("--y-max-ms", type=int, default=25000)
    ap.add_argument("--y-max-variance-ms2", type=float, default=None)
    ap.add_argument("--xtick-every", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--syllables", type=str, nargs="*", default=None, help="Subset of syllable labels to plot")
    ap.add_argument("--treatment-date", type=str, default=None, help="e.g., '2025-03-04' or '2025.03.04' (single-bird mode)")
    ap.add_argument("--fixed-label-colors-json", type=str, default=None)

    ap.add_argument("--no-green-plots", action="store_true")
    ap.add_argument("--no-colored-plots", action="store_true")
    ap.add_argument("--no-green-variance-plots", action="store_true")
    ap.add_argument("--no-colored-variance-plots", action="store_true")
    ap.add_argument("--no-green-aggregate-variance-plots", action="store_true")
    ap.add_argument("--no-colored-aggregate-variance-plots", action="store_true")

    args = ap.parse_args()

    if args.metadata_excel and args.json_root:
        results = run_batch_phrase_duration_over_days_from_excel(
            metadata_excel=args.metadata_excel,
            json_root=args.json_root,
            sheet_name=args.sheet_name,
            id_col=args.id_col,
            treatment_date_col=args.treatment_date_col,
            hit_type_sheet_name=args.hit_type_sheet_name,
            hit_type_id_col=args.hit_type_id_col,
            hit_type_col=args.hit_type_col,
            output_root=args.output_root,
            syllables_subset=args.syllables,
            fixed_label_colors_json=args.fixed_label_colors_json,
            max_gap_between_song_segments=args.ann_gap_ms,
            segment_index_offset=args.seg_offset,
            merge_repeated_syllables=args.merge_repeats,
            repeat_gap_ms=args.repeat_gap_ms,
            repeat_gap_inclusive=args.repeat_gap_inclusive,
            y_max_ms=args.y_max_ms,
            y_max_variance_ms2=args.y_max_variance_ms2,
            dpi=args.dpi,
            transparent=args.transparent,
            xtick_every=args.xtick_every,
            show_plots=args.show,
            make_green_plots=not args.no_green_plots,
            make_colored_plots=not args.no_colored_plots,
            make_green_variance_plots=not args.no_green_variance_plots,
            make_colored_variance_plots=not args.no_colored_variance_plots,
            make_green_aggregate_variance_plots=not args.no_green_aggregate_variance_plots,
            make_colored_aggregate_variance_plots=not args.no_colored_aggregate_variance_plots,
        )
        ok = sum(1 for r in results.values() if r.status == "ok")
        skipped = sum(1 for r in results.values() if r.status == "skipped")
        errors = sum(1 for r in results.values() if r.status == "error")
        print(f"\n[SUMMARY] ok={ok}, skipped={skipped}, errors={errors}")
    else:
        if args.save_dir is None:
            raise SystemExit("--save-dir is required in single-bird mode")

        graph_phrase_duration_over_days(
            decoded_database_json=args.decoded,
            creation_metadata_json=args.creation_meta,
            save_output_to_this_file_path=args.save_dir,
            song_detection_json=args.song_detection,
            max_gap_between_song_segments=args.ann_gap_ms,
            segment_index_offset=args.seg_offset,
            merge_repeated_syllables=args.merge_repeats,
            repeat_gap_ms=args.repeat_gap_ms,
            repeat_gap_inclusive=args.repeat_gap_inclusive,
            only_song_present=args.only_song_present,
            y_max_ms=args.y_max_ms,
            y_max_variance_ms2=args.y_max_variance_ms2,
            xtick_every=args.xtick_every,
            dpi=args.dpi,
            transparent=args.transparent,
            show_plots=args.show,
            syllables_subset=args.syllables,
            treatment_date=args.treatment_date,
            fixed_label_colors_json=args.fixed_label_colors_json,
            make_green_plots=not args.no_green_plots,
            make_colored_plots=not args.no_colored_plots,
            make_green_variance_plots=not args.no_green_variance_plots,
            make_colored_variance_plots=not args.no_colored_variance_plots,
            make_green_aggregate_variance_plots=not args.no_green_aggregate_variance_plots,
            make_colored_aggregate_variance_plots=not args.no_colored_aggregate_variance_plots,
        )


"""
Example Spyder usage (batch):

from pathlib import Path
import importlib
import phrase_duration_over_days_with_label_colors_and_variance_batch as pdod

importlib.reload(pdod)

results = pdod.run_batch_phrase_duration_over_days_from_excel(
    metadata_excel=Path("/Volumes/my_own_SSD/updated_AreaX_outputs/Area_X_lesion_metadata_with_hit_types.xlsx"),
    json_root=Path("/Volumes/my_own_SSD/updated_AreaX_outputs"),
    sheet_name="metadata",
    id_col="Animal ID",
    treatment_date_col="Treatment date",
    hit_type_sheet_name="animal_hit_type_summary",
    hit_type_id_col="Animal ID",
    hit_type_col="Lesion hit type",
    fixed_label_colors_json=Path("/Volumes/my_own_SSD/updated_AreaX_outputs/fixed_label_colors_50.json"),
    show_plots=False,
)

for animal_id, rec in results.items():
    print(animal_id, rec.status, rec.output_dir)
"""
