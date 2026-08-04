#!/usr/bin/env python3
"""
Inventory and compare Bhattacharyya coefficient (BC) output versions.

This script recursively scans one or more analysis-output roots and optional
code roots, identifies BC/Bhattacharyya-related files, extracts statistical
results into standardized tables, inventories bird inclusion, and summarizes
likely analysis-version differences.

Designed for the AFP lesion / canary song analysis directories, but written
to tolerate multiple historical output formats.

Example
-------
python inventory_compare_bc_outputs.py \
  --roots "/Volumes/my_own_SSD/updated_AreaX_outputs" \
  --code-roots "$HOME/Documents/allPythonCode" \
  --out-dir "$HOME/Desktop/bc_output_audit"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "This script requires pandas. Install it with:\n"
        "  python -m pip install pandas\n"
    ) from exc


BC_TERMS = re.compile(
    r"(?:^|[_\-/ ])(?:bc|bhattacharyya|equal[_ -]?time|equal[_ -]?groups|"
    r"time[_ -]?balanced|frame[_ -]?weighted|early[_ -]?late)(?:$|[_\-/ ])",
    re.IGNORECASE,
)
RUN_DIR_TERMS = re.compile(
    r"(bc|bhattach|time[_ -]?balanced|frame[_ -]?weighted|equal[_ -]?only|"
    r"umap_graph_batch|figure4|lesion_group_summaries)",
    re.IGNORECASE,
)
STATS_FILE_TERMS = re.compile(
    r"(stat|summary|result|test|comparison|bird_level|group)", re.IGNORECASE
)
BC_HEADER_TERMS = re.compile(
    r"(bhattacharyya|\bbc\b|bc_|_bc|early_late_equal|selected_bins)",
    re.IGNORECASE,
)

TEXT_EXTENSIONS = {".csv", ".tsv", ".txt", ".log", ".json"}
INVENTORY_EXTENSIONS = TEXT_EXTENSIONS | {".py", ".png", ".jpg", ".jpeg", ".pdf", ".svg"}
SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".ipynb_checkpoints", "node_modules",
    ".Trash", ".Trashes", "$RECYCLE.BIN",
}
GENERIC_DIR_NAMES = {
    "stats", "bird_level", "cluster_level", "clusters", "figures", "plots",
    "outputs", "output", "source_data", "tables",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and compare all BC/Bhattacharyya output versions."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Analysis-output roots to scan recursively.",
    )
    parser.add_argument(
        "--code-roots",
        nargs="*",
        default=[],
        help="Optional code roots to scan for BC-related Python scripts.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory in which audit CSVs and the Markdown report are written.",
    )
    parser.add_argument(
        "--max-tabular-bytes",
        type=int,
        default=25_000_000,
        help="Skip parsing candidate tables larger than this many bytes (default 25 MB).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Maximum rows to read from a single candidate table.",
    )
    return parser.parse_args()


def norm_path(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def safe_stat(path: Path) -> tuple[int | None, str | None]:
    try:
        stat = path.stat()
        return stat.st_size, datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None, None


def sha1_prefix(path: Path, max_bytes: int = 1_000_000) -> str:
    h = hashlib.sha1()
    try:
        with path.open("rb") as handle:
            h.update(handle.read(max_bytes))
        return h.hexdigest()[:12]
    except OSError:
        return ""


def looks_like_bc_path(path: Path) -> bool:
    return bool(BC_TERMS.search(str(path)))


def text_header_contains_bc(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            sample = "".join(handle.readline() for _ in range(5))
        return bool(BC_HEADER_TERMS.search(sample))
    except OSError:
        return False


def candidate_reason(path: Path) -> str:
    reasons: list[str] = []
    if looks_like_bc_path(path):
        reasons.append("path/name contains BC-analysis term")
    if path.suffix.lower() in TEXT_EXTENSIONS and text_header_contains_bc(path):
        reasons.append("file header contains BC field")
    return "; ".join(reasons)


def iter_files(root: Path, extensions: set[str] | None = None) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIR_NAMES and not d.startswith("._")
        ]
        base = Path(dirpath)
        for filename in filenames:
            if filename.startswith("._"):
                continue
            path = base / filename
            if extensions is None or path.suffix.lower() in extensions:
                yield path


def infer_version_root(path: Path, scan_root: Path) -> Path:
    """
    Infer the analysis-run directory most useful for grouping outputs.

    Prefer the deepest non-generic ancestor with a BC/run keyword. If no such
    ancestor exists, use the first directory below the scan root.
    """
    try:
        relative = path.relative_to(scan_root)
    except ValueError:
        return path.parent

    ancestors = list(relative.parents)
    # relative.parents includes "." at the end; inspect from deepest to shallowest.
    for rel_parent in ancestors:
        if str(rel_parent) == ".":
            continue
        name = rel_parent.name
        if name.lower() in GENERIC_DIR_NAMES:
            continue
        if RUN_DIR_TERMS.search(name):
            return scan_root / rel_parent

    if len(relative.parts) > 1:
        return scan_root / relative.parts[0]
    return scan_root


def make_inventory(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            rows.append({
                "scan_root": str(root),
                "version_root": "",
                "path": "",
                "relative_path": "",
                "extension": "",
                "size_bytes": None,
                "modified_time": None,
                "sha1_prefix": "",
                "candidate_reason": "ROOT DOES NOT EXIST",
            })
            continue

        for path in iter_files(root, INVENTORY_EXTENSIONS):
            reason = candidate_reason(path)
            if not reason:
                continue
            size, mtime = safe_stat(path)
            version_root = infer_version_root(path, root)
            rows.append({
                "scan_root": str(root),
                "version_root": str(version_root),
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "extension": path.suffix.lower(),
                "size_bytes": size,
                "modified_time": mtime,
                "sha1_prefix": sha1_prefix(path) if path.suffix.lower() in TEXT_EXTENSIONS else "",
                "candidate_reason": reason,
            })
    return pd.DataFrame(rows)


def sniff_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            sample = handle.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except Exception:
        return ","


def read_table(path: Path, max_rows: int) -> pd.DataFrame | None:
    delimiter = sniff_delimiter(path)
    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            engine="python",
            nrows=max_rows,
            on_bad_lines="skip",
        )
        # A one-column parse of a comma-delimited file is probably wrong.
        if len(df.columns) == 1 and delimiter != ",":
            df = pd.read_csv(
                path,
                sep=",",
                engine="python",
                nrows=max_rows,
                on_bad_lines="skip",
            )
        return df
    except Exception:
        return None


def is_number(value: Any) -> bool:
    try:
        value_float = float(value)
        return math.isfinite(value_float)
    except (TypeError, ValueError):
        return False


def standard_col_map(columns: Iterable[Any]) -> dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}


def first_existing(colmap: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if name in colmap:
            return colmap[name]
    return None


def find_columns(colmap: dict[str, str], pattern: str) -> list[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [original for normalized, original in colmap.items() if regex.search(normalized)]


def value_from(row: pd.Series, column: str | None) -> Any:
    if column is None:
        return None
    value = row.get(column)
    if pd.isna(value):
        return None
    return value


def parse_headered_stats(
    df: pd.DataFrame,
    path: Path,
    version_root: Path,
) -> list[dict[str, Any]]:
    colmap = standard_col_map(df.columns)
    p_cols = find_columns(
        colmap,
        r"(^p$|p_value|pvalue|wilcoxon.*p|mann.*p|kruskal.*p|paired.*p|_p$)",
    )
    if not p_cols:
        return []

    group_col = first_existing(
        colmap,
        ["lesion_hit_type", "hit_type", "group", "group_name", "lesion_group", "raw_lesion_hit_type"],
    )
    set_col = first_existing(colmap, ["set_name", "subset", "cluster_set", "selection"])
    set_label_col = first_existing(colmap, ["set_label", "subset_label"])
    level_col = first_existing(colmap, ["analysis_level", "level"])
    method_col = first_existing(colmap, ["bc_method", "method", "metric"])
    method_label_col = first_existing(colmap, ["bc_method_label", "method_label"])
    aggregate_col = first_existing(colmap, ["bird_aggregate", "aggregate", "aggregation"])

    n_col = first_existing(
        colmap,
        [
            "n_animals", "n_birds", "n_birds_with_pre_post", "n_observations",
            "n_pairs", "n",
        ],
    )

    pre_cols = find_columns(colmap, r"(mean|median)?_?bc_?pre$|bc_pre_(mean|median)$")
    post_cols = find_columns(colmap, r"(mean|median)?_?bc_?post$|bc_post_(mean|median)$")
    delta_cols = find_columns(colmap, r"(mean|median).*delta.*pre|delta.*post.*pre|bc_delta")
    statistic_cols = find_columns(colmap, r"(statistic|^stat$|test_stat)")

    output: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        p_values = [(col, value_from(row, col)) for col in p_cols]
        p_values = [(col, val) for col, val in p_values if is_number(val)]
        if not p_values:
            continue

        for p_col, p_value in p_values:
            lower_p = p_col.lower()
            if "formatted" in lower_p or "adjust" in lower_p or "correct" in lower_p:
                p_kind = "formatted_or_adjusted"
            else:
                p_kind = "raw_or_primary"

            result = {
                "version_root": str(version_root),
                "source_file": str(path),
                "source_format": "headered_table",
                "set_name": value_from(row, set_col),
                "set_label": value_from(row, set_label_col),
                "analysis_level": value_from(row, level_col),
                "lesion_group": value_from(row, group_col),
                "bc_method": value_from(row, method_col),
                "bc_method_label": value_from(row, method_label_col),
                "bird_aggregate": value_from(row, aggregate_col),
                "n": value_from(row, n_col),
                "p_column": p_col,
                "p_kind": p_kind,
                "p_value": float(p_value),
                "test_name": "",
                "test_statistic": None,
                "mean_bc_pre": value_from(row, colmap.get("mean_bc_pre")),
                "mean_bc_post": value_from(row, colmap.get("mean_bc_post")),
                "median_bc_pre": value_from(row, colmap.get("median_bc_pre")),
                "median_bc_post": value_from(row, colmap.get("median_bc_post")),
                "mean_delta_post_minus_pre": value_from(
                    row, colmap.get("mean_delta_post_minus_pre")
                ),
                "median_delta_post_minus_pre": value_from(
                    row, colmap.get("median_delta_post_minus_pre")
                ),
                "review_note": "",
            }
            if statistic_cols:
                result["test_statistic"] = value_from(row, statistic_cols[0])

            # Preserve alternate pre/post names if standard names are absent.
            if result["mean_bc_pre"] is None and pre_cols:
                result["mean_bc_pre"] = value_from(row, pre_cols[0])
            if result["mean_bc_post"] is None and post_cols:
                result["mean_bc_post"] = value_from(row, post_cols[0])
            if result["mean_delta_post_minus_pre"] is None and delta_cols:
                result["mean_delta_post_minus_pre"] = value_from(row, delta_cols[0])

            output.append(result)
    return output


def parse_no_header_paired_stats(
    path: Path,
    version_root: Path,
    max_rows: int,
) -> list[dict[str, Any]]:
    """
    Parse historical rows such as:
    paired_bird_level_mean,bc_early_late_equal_groups,GROUP,wilcoxon,
    8,2.0,0.0234375,,0.0703125,n.s.
    """
    try:
        raw = pd.read_csv(
            path,
            header=None,
            nrows=max_rows,
            engine="python",
            on_bad_lines="skip",
        )
    except Exception:
        return []

    output: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        vals = row.tolist()
        if len(vals) < 7:
            continue
        first = str(vals[0]).lower()
        metric = str(vals[1]).lower() if len(vals) > 1 else ""
        test_name = str(vals[3]).lower() if len(vals) > 3 else ""
        if not (
            ("bird_level" in first or "cluster_level" in first)
            and ("bc" in metric or "bhattach" in metric)
            and ("wilcoxon" in test_name or "mann" in test_name)
        ):
            continue

        raw_p = vals[6] if len(vals) > 6 else None
        adjusted_p = vals[8] if len(vals) > 8 else None
        common = {
            "version_root": str(version_root),
            "source_file": str(path),
            "source_format": "no_header_paired_stats",
            "set_name": metric,
            "set_label": "",
            "analysis_level": vals[0],
            "lesion_group": vals[2],
            "bc_method": metric,
            "bc_method_label": "",
            "bird_aggregate": "mean" if "mean" in first else ("median" if "median" in first else ""),
            "n": vals[4],
            "test_name": vals[3],
            "test_statistic": vals[5],
            "mean_bc_pre": None,
            "mean_bc_post": None,
            "median_bc_pre": None,
            "median_bc_post": None,
            "mean_delta_post_minus_pre": None,
            "median_delta_post_minus_pre": None,
            "review_note": "Historical no-header paired-test format.",
        }
        if is_number(raw_p):
            output.append({
                **common,
                "p_column": "raw_p",
                "p_kind": "raw_or_primary",
                "p_value": float(raw_p),
            })
        if is_number(adjusted_p):
            output.append({
                **common,
                "p_column": "adjusted_p",
                "p_kind": "formatted_or_adjusted",
                "p_value": float(adjusted_p),
            })
    return output


def parse_text_stats(path: Path, version_root: Path) -> list[dict[str, Any]]:
    """
    Conservative text parser. It only emits lines containing both a BC-related
    term and an explicit p-value.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    p_regex = re.compile(
        r"\bp(?:[_ -]?value)?\s*[=:]\s*"
        r"(?P<p>[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",
        re.IGNORECASE,
    )
    n_regex = re.compile(r"\bn(?:[_ -]?(?:birds|animals|pairs))?\s*[=:]\s*(\d+)", re.IGNORECASE)

    output: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not BC_HEADER_TERMS.search(line):
            continue
        match = p_regex.search(line)
        if not match:
            continue
        n_match = n_regex.search(line)
        output.append({
            "version_root": str(version_root),
            "source_file": str(path),
            "source_format": "text_line",
            "set_name": "",
            "set_label": "",
            "analysis_level": "",
            "lesion_group": "",
            "bc_method": "",
            "bc_method_label": "",
            "bird_aggregate": "",
            "n": int(n_match.group(1)) if n_match else None,
            "p_column": f"line_{line_no}",
            "p_kind": "raw_or_primary",
            "p_value": float(match.group("p")),
            "test_name": "",
            "test_statistic": None,
            "mean_bc_pre": None,
            "mean_bc_post": None,
            "median_bc_pre": None,
            "median_bc_post": None,
            "mean_delta_post_minus_pre": None,
            "median_delta_post_minus_pre": None,
            "review_note": line.strip()[:500],
        })
    return output


def extract_stats(
    inventory: pd.DataFrame,
    roots: list[Path],
    max_tabular_bytes: int,
    max_rows: int,
) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()

    root_lookup = sorted(roots, key=lambda p: len(str(p)), reverse=True)
    output: list[dict[str, Any]] = []

    for record in inventory.to_dict("records"):
        path_str = record.get("path")
        if not path_str:
            continue
        path = Path(path_str)
        suffix = path.suffix.lower()
        size = record.get("size_bytes")
        if size is not None and size > max_tabular_bytes:
            continue
        if suffix not in TEXT_EXTENSIONS:
            continue
        if not (
            STATS_FILE_TERMS.search(path.name)
            or "stats" in [part.lower() for part in path.parts]
        ):
            continue

        scan_root = next((r for r in root_lookup if str(path).startswith(str(r))), path.parent)
        version_root = Path(record.get("version_root") or infer_version_root(path, scan_root))

        if suffix in {".csv", ".tsv"}:
            # Try historical no-header files first when the name strongly suggests it.
            if path.name == "paired_bird_level_mean_pre_post_stats.csv":
                output.extend(parse_no_header_paired_stats(path, version_root, max_rows))

            df = read_table(path, max_rows)
            if df is not None:
                output.extend(parse_headered_stats(df, path, version_root))
        else:
            output.extend(parse_text_stats(path, version_root))

    result = pd.DataFrame(output)
    if not result.empty:
        dedupe_cols = [
            "source_file", "set_name", "analysis_level", "lesion_group",
            "p_column", "p_value",
        ]
        result = result.drop_duplicates(subset=[c for c in dedupe_cols if c in result.columns])
    return result


def extract_bird_inclusion(
    inventory: pd.DataFrame,
    max_tabular_bytes: int,
    max_rows: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    if inventory.empty:
        return pd.DataFrame()

    for record in inventory.to_dict("records"):
        path_str = record.get("path")
        if not path_str:
            continue
        path = Path(path_str)
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        size = record.get("size_bytes")
        if size is not None and size > max_tabular_bytes:
            continue
        if not re.search(r"bird.*summary|summary.*bird|bird_level", path.name, re.IGNORECASE):
            continue

        df = read_table(path, max_rows)
        if df is None or df.empty:
            continue

        colmap = standard_col_map(df.columns)
        animal_col = first_existing(colmap, ["animal_id", "bird_id", "animal", "bird"])
        if animal_col is None:
            continue
        bc_pre_col = first_existing(
            colmap,
            ["bc_pre", "median_bc_pre", "mean_bc_pre", "bc_pre_early_vs_late_equal_groups"],
        )
        bc_post_col = first_existing(
            colmap,
            ["bc_post", "median_bc_post", "mean_bc_post", "bc_post_early_vs_late_equal_groups"],
        )
        if bc_pre_col is None and bc_post_col is None:
            continue

        group_col = first_existing(
            colmap,
            ["lesion_hit_type", "hit_type", "group", "lesion_group", "raw_lesion_hit_type"],
        )
        raw_group_col = first_existing(colmap, ["raw_lesion_hit_type"])
        set_col = first_existing(colmap, ["set_name", "subset"])
        set_label_col = first_existing(colmap, ["set_label", "subset_label"])
        n_clusters_col = first_existing(colmap, ["n_clusters", "cluster_count"])
        aggregate_col = first_existing(colmap, ["bird_aggregate", "aggregate", "aggregation"])
        delta_col = first_existing(
            colmap, ["bc_delta_post_minus_pre", "delta_post_minus_pre", "bc_delta"]
        )

        for _, row in df.iterrows():
            animal = value_from(row, animal_col)
            if animal is None:
                continue
            pre = value_from(row, bc_pre_col)
            post = value_from(row, bc_post_col)
            delta = value_from(row, delta_col)
            if delta is None and is_number(pre) and is_number(post):
                delta = float(post) - float(pre)
            output.append({
                "version_root": record.get("version_root"),
                "source_file": str(path),
                "set_name": value_from(row, set_col),
                "set_label": value_from(row, set_label_col),
                "animal_id": animal,
                "lesion_group": value_from(row, group_col),
                "raw_lesion_group": value_from(row, raw_group_col),
                "n_clusters": value_from(row, n_clusters_col),
                "bc_pre": pre,
                "bc_post": post,
                "bc_delta_post_minus_pre": delta,
                "bird_aggregate": value_from(row, aggregate_col),
            })

    result = pd.DataFrame(output)
    if not result.empty:
        result = result.drop_duplicates()
    return result


def script_summary(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""

    tests = []
    for term in ["wilcoxon", "mannwhitneyu", "kruskal", "permutation", "bootstrap"]:
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            tests.append(term)

    aggregations = []
    if re.search(r"groupby[\s\S]{0,300}\.median\(|aggregate.*median|bird_aggregate.*median", text, re.IGNORECASE):
        aggregations.append("median")
    if re.search(r"groupby[\s\S]{0,300}\.mean\(|aggregate.*mean|bird_aggregate.*mean", text, re.IGNORECASE):
        aggregations.append("mean")

    analysis_hints = []
    for label, pattern in [
        ("selected equal time bins", r"selected_bins|selected equal time bins"),
        ("equal groups", r"equal_groups|equal groups"),
        ("time balanced", r"time[_ -]?balanced"),
        ("frame weighted", r"frame[_ -]?weighted"),
        ("top 30%", r"top.?30|high_variance"),
        ("remaining clusters", r"remaining_non_high|remaining.*cluster"),
        ("density 100", r"density.?100"),
        ("density 20", r"density.?20"),
        ("grid 99", r"grid.?99"),
        ("5.4 s", r"5p4s|5\.4"),
    ]:
        if re.search(pattern, text, re.IGNORECASE):
            analysis_hints.append(label)

    output_paths = sorted(set(re.findall(
        r"""["']([^"']*(?:stats|summary|figure|output)[^"']*\.(?:csv|txt|png|pdf))["']""",
        text,
        re.IGNORECASE,
    )))[:30]

    size, mtime = safe_stat(path)
    return {
        "path": str(path),
        "size_bytes": size,
        "modified_time": mtime,
        "tests_detected": "; ".join(tests),
        "bird_aggregation_detected": "; ".join(aggregations),
        "analysis_hints": "; ".join(analysis_hints),
        "referenced_output_files": "; ".join(output_paths),
        "sha1_prefix": sha1_prefix(path),
    }


def inventory_scripts(code_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in code_roots:
        if not root.exists():
            rows.append({
                "path": str(root),
                "size_bytes": None,
                "modified_time": None,
                "tests_detected": "",
                "bird_aggregation_detected": "",
                "analysis_hints": "",
                "referenced_output_files": "",
                "sha1_prefix": "",
                "note": "CODE ROOT DOES NOT EXIST",
            })
            continue

        for path in iter_files(root, {".py"}):
            include = looks_like_bc_path(path)
            if not include:
                try:
                    sample = path.read_text(encoding="utf-8", errors="replace")[:250_000]
                    include = bool(re.search(r"Bhattacharyya|bhattacharyya|\bbc_", sample))
                except OSError:
                    include = False
            if include:
                row = script_summary(path)
                row["note"] = ""
                rows.append(row)
    return pd.DataFrame(rows)


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        pd.DataFrame({"note": ["No rows found."]}).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if is_number(value):
        return f"{float(value):.{digits}g}"
    return str(value)


def build_report(
    inventory: pd.DataFrame,
    stats: pd.DataFrame,
    birds: pd.DataFrame,
    scripts: pd.DataFrame,
    roots: list[Path],
) -> str:
    lines: list[str] = []
    lines.append("# Bhattacharyya coefficient output audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Scan roots")
    for root in roots:
        lines.append(f"- `{root}` — {'found' if root.exists() else 'NOT FOUND'}")
    lines.append("")

    candidate_count = 0 if inventory.empty else len(inventory)
    version_count = 0 if inventory.empty or "version_root" not in inventory else inventory["version_root"].replace("", pd.NA).dropna().nunique()
    lines.append("## Inventory summary")
    lines.append(f"- Candidate BC-related files: **{candidate_count}**")
    lines.append(f"- Inferred output-version roots: **{version_count}**")
    lines.append(f"- Standardized statistical rows: **{0 if stats.empty else len(stats)}**")
    lines.append(f"- Bird-level inclusion/value rows: **{0 if birds.empty else len(birds)}**")
    lines.append(f"- Candidate BC scripts: **{0 if scripts.empty else len(scripts)}**")
    lines.append("")

    if not stats.empty:
        primary = stats.copy()
        if "p_kind" in primary:
            raw = primary[primary["p_kind"] == "raw_or_primary"]
            if not raw.empty:
                primary = raw

        # Keep likely bird-level rows first.
        level = primary["analysis_level"].astype(str).str.lower()
        primary = primary.assign(
            _bird_priority=level.str.contains("bird").astype(int),
            _mtime="",
        ).sort_values(
            ["_bird_priority", "version_root", "source_file"],
            ascending=[False, True, True],
        )

        lines.append("## Extracted BC test results")
        lines.append("")
        lines.append(
            "| Version root | Set | Level | Group | Aggregate | n | Pre | Post | Δ post−pre | p | Source |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        shown = 0
        for _, row in primary.iterrows():
            if shown >= 120:
                break
            pre = row.get("median_bc_pre")
            post = row.get("median_bc_post")
            delta = row.get("median_delta_post_minus_pre")
            aggregate = row.get("bird_aggregate")
            if pre is None or (isinstance(pre, float) and math.isnan(pre)):
                pre = row.get("mean_bc_pre")
            if post is None or (isinstance(post, float) and math.isnan(post)):
                post = row.get("mean_bc_post")
            if delta is None or (isinstance(delta, float) and math.isnan(delta)):
                delta = row.get("mean_delta_post_minus_pre")
            if not aggregate:
                source_lower = str(row.get("analysis_level", "")).lower()
                aggregate = "mean" if "mean" in source_lower else ("median" if "median" in source_lower else "")

            source = Path(str(row.get("source_file", ""))).name
            version = Path(str(row.get("version_root", ""))).name
            set_name = row.get("set_label") or row.get("set_name") or ""
            lines.append(
                "| " + " | ".join([
                    str(version),
                    str(set_name),
                    str(row.get("analysis_level") or ""),
                    str(row.get("lesion_group") or ""),
                    str(aggregate or ""),
                    fmt(row.get("n")),
                    fmt(pre),
                    fmt(post),
                    fmt(delta),
                    fmt(row.get("p_value")),
                    str(source),
                ]) + " |"
            )
            shown += 1
        if len(primary) > shown:
            lines.append("")
            lines.append(f"_Table truncated to {shown} rows; see `bc_stats_comparison.csv` for all rows._")
        lines.append("")

    if not birds.empty:
        lines.append("## Bird inclusion differences")
        lines.append("")
        grouped = birds.groupby(
            ["version_root", "set_name", "lesion_group"],
            dropna=False,
        )["animal_id"].agg(lambda s: sorted({str(v) for v in s if pd.notna(v)})).reset_index()
        for _, row in grouped.iterrows():
            version = Path(str(row["version_root"])).name
            set_name = row["set_name"] if pd.notna(row["set_name"]) else ""
            group = row["lesion_group"] if pd.notna(row["lesion_group"]) else ""
            animals = row["animal_id"]
            lines.append(
                f"- **{version}** | `{set_name}` | {group}: "
                f"n={len(animals)} — {', '.join(animals)}"
            )
        lines.append("")

    if not scripts.empty:
        lines.append("## Candidate generating scripts")
        lines.append("")
        lines.append("| Script | Modified | Tests | Bird aggregation | Analysis hints |")
        lines.append("|---|---|---|---|---|")
        for _, row in scripts.sort_values("modified_time", ascending=False).head(100).iterrows():
            lines.append(
                "| " + " | ".join([
                    Path(str(row.get("path", ""))).name,
                    str(row.get("modified_time") or ""),
                    str(row.get("tests_detected") or ""),
                    str(row.get("bird_aggregation_detected") or ""),
                    str(row.get("analysis_hints") or ""),
                ]) + " |"
            )
        if len(scripts) > 100:
            lines.append("")
            lines.append("_Table truncated; see `bc_script_inventory.csv`._")
        lines.append("")

    lines.append("## Interpretation checklist")
    lines.append("")
    lines.append(
        "Before selecting a manuscript result, compare versions on these fields:"
    )
    lines.append("")
    lines.append("1. **BC definition and sampling:** all bins, equal groups, selected equal-time bins, time-balanced, or frame-weighted.")
    lines.append("2. **Density/grid settings:** for example density 20 versus density 100 and grid cropping.")
    lines.append("3. **Unit of analysis:** cluster-level tests versus bird-level tests.")
    lines.append("4. **Bird aggregation:** mean versus median across qualifying clusters.")
    lines.append("5. **Syllable subset:** all clusters, top-30% high-variance clusters, or remaining clusters.")
    lines.append("6. **Bird inclusion:** especially USA5505 versus USA5509 and lateral-only n=5 versus n=8.")
    lines.append("7. **Statistical test and correction:** paired Wilcoxon, Mann–Whitney, and raw versus adjusted p-values.")
    lines.append("8. **File modification time and generating script:** use these to identify the latest intentional Figure 4 run.")
    lines.append("")
    lines.append(
        "Files with identical `sha1_prefix` values are likely duplicate copies of the same content."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    roots = [norm_path(p) for p in args.roots]
    code_roots = [norm_path(p) for p in args.code_roots]
    out_dir = norm_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning BC outputs...")
    inventory = make_inventory(roots)
    stats = extract_stats(
        inventory,
        roots,
        max_tabular_bytes=args.max_tabular_bytes,
        max_rows=args.max_rows,
    )
    birds = extract_bird_inclusion(
        inventory,
        max_tabular_bytes=args.max_tabular_bytes,
        max_rows=args.max_rows,
    )

    print("Scanning BC-related scripts...")
    scripts = inventory_scripts(code_roots)

    safe_write_csv(inventory, out_dir / "bc_output_inventory.csv")
    safe_write_csv(stats, out_dir / "bc_stats_comparison.csv")
    safe_write_csv(birds, out_dir / "bc_bird_inclusion_comparison.csv")
    safe_write_csv(scripts, out_dir / "bc_script_inventory.csv")

    report = build_report(inventory, stats, birds, scripts, roots)
    report_path = out_dir / "bc_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("")
    print("Audit complete.")
    print(f"Output directory: {out_dir}")
    print(f"Report: {report_path}")
    print("")
    print("Files written:")
    for name in [
        "bc_output_inventory.csv",
        "bc_stats_comparison.csv",
        "bc_bird_inclusion_comparison.csv",
        "bc_script_inventory.csv",
        "bc_comparison_report.md",
    ]:
        print(f"  - {out_dir / name}")

    if not stats.empty:
        raw = stats[stats["p_kind"] == "raw_or_primary"] if "p_kind" in stats else stats
        bird_rows = raw[
            raw["analysis_level"].astype(str).str.contains("bird", case=False, na=False)
        ]
        print("")
        print(f"Extracted primary p-value rows: {len(raw)}")
        print(f"Likely bird-level primary rows: {len(bird_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
