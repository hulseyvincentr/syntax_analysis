#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any, Iterable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

SHAM = 'sham saline injection'
LAT = 'Lateral lesion only'
COMPLETE = 'Complete Medial and Lateral lesion'
PARTIAL = 'Partial Medial and Lateral lesion'
POOLED_ML = 'Complete and partial medial and lateral lesion'
UNKNOWN = 'unknown'

DEFAULT_COLORS = {
    SHAM: '#1B9E77',
    LAT: '#A88BD9',
    COMPLETE: '#3F007D',
    PARTIAL: '#7A4FB7',
    POOLED_ML: '#7A4FB7',
    UNKNOWN: '#4D4D4D',
}


def lower(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ''
    return ' '.join(str(x).strip().split()).lower()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def first_present(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    colmap = {lower(c): c for c in columns}
    for cand in candidates:
        if lower(cand) in colmap:
            return colmap[lower(cand)]
    return None


def canonical_group(raw: Any) -> str:
    s = lower(raw)
    if not s:
        return UNKNOWN
    if 'sham' in s or ('saline' in s and 'lesion' not in s):
        return SHAM
    if 'lateral lesion only' in s or 'lateral only' in s or 'single hit' in s or 'lateral hit only' in s:
        return LAT
    if 'complete' in s and 'medial' in s and 'lateral' in s:
        return COMPLETE
    if 'partial' in s and 'medial' in s and 'lateral' in s:
        return PARTIAL
    if 'area x not visible' in s or ('large' in s and 'lesion' in s):
        return COMPLETE
    if ('medial' in s and 'lateral' in s) or 'm+l' in s:
        return PARTIAL
    if 'complete and partial medial and lateral lesion' in s:
        return POOLED_ML
    return str(raw)


def pooled_group(canon: str) -> str:
    if canon in {COMPLETE, PARTIAL, POOLED_ML}:
        return POOLED_ML
    return canon


def load_selected_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    animal_col = first_present(df.columns, ['animal_id', 'Animal ID', 'bird_id', 'bird'])
    syll_col = first_present(df.columns, ['syllable', 'Syllable', 'label'])
    if animal_col is None or syll_col is None:
        raise ValueError(f'Selected-pairs CSV missing animal/syllable columns. Found: {list(df.columns)}')
    out = df[[animal_col, syll_col]].copy()
    out.columns = ['animal_id', 'syllable']
    out['animal_id'] = out['animal_id'].astype(str)
    out['syllable'] = out['syllable'].astype(str)
    return out.drop_duplicates().reset_index(drop=True)


def maybe_load_metadata(metadata_excel: Optional[str]) -> dict[str, str]:
    if not metadata_excel:
        return {}
    p = Path(metadata_excel)
    if not p.exists():
        raise FileNotFoundError(p)
    xls = pd.ExcelFile(p)
    required_animal = ['Animal ID', 'animal_id', 'bird_id']
    hit_candidates = ['Lesion hit type', 'Hit type', 'hit_type']
    for sheet in xls.sheet_names:
        df = pd.read_excel(p, sheet_name=sheet)
        animal_col = first_present(df.columns, required_animal)
        hit_col = first_present(df.columns, hit_candidates)
        if animal_col and hit_col:
            return {str(a).strip(): str(h).strip() for a, h in zip(df[animal_col], df[hit_col]) if pd.notna(a) and pd.notna(h)}
    raise ValueError(f'Could not find animal and hit-type columns in workbook: {p}')


def load_bc_table(path: str, metadata_excel: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    animal_col = first_present(df.columns, ['animal_id', 'Animal ID', 'bird_id', 'bird'])
    syll_col = first_present(df.columns, ['syllable', 'Syllable', 'label'])
    if animal_col is None or syll_col is None:
        raise ValueError(f'BC CSV missing animal/syllable columns. Found: {list(df.columns)}')

    # Wide format first
    pre_col = first_present(df.columns, ['pre_bc', 'bc_pre', 'pre_BC', 'Pre BC', 'late_pre_bc', 'median_pre_bc'])
    post_col = first_present(df.columns, ['post_bc', 'bc_post', 'post_BC', 'Post BC', 'post_period_bc', 'median_post_bc'])
    delta_col = first_present(df.columns, ['delta_bc', 'bc_delta', 'post_minus_pre_bc', 'change_in_bc'])

    if pre_col and post_col:
        out = df.copy()
        out['pre_bc'] = pd.to_numeric(out[pre_col], errors='coerce')
        out['post_bc'] = pd.to_numeric(out[post_col], errors='coerce')
    else:
        period_col = first_present(df.columns, ['period', 'Period', 'epoch', 'Epoch', 'group', 'Group'])
        value_col = first_present(df.columns, ['bc', 'BC', 'bhattacharyya_coefficient', 'Bhattacharyya coefficient', 'median_bc', 'value'])
        if period_col is None or value_col is None:
            raise ValueError('Could not identify BC pre/post columns or a pivotable period/value structure.')
        work = df[[animal_col, syll_col, period_col, value_col] + [c for c in df.columns if c not in {animal_col, syll_col, period_col, value_col}]].copy()
        work[period_col] = work[period_col].astype(str)
        def classify_period(x: str) -> Optional[str]:
            s = lower(x)
            if 'pre' in s:
                return 'pre_bc'
            if 'post' in s:
                return 'post_bc'
            return None
        work['_period_key'] = work[period_col].map(classify_period)
        work = work[work['_period_key'].notna()].copy()
        piv = work.pivot_table(index=[animal_col, syll_col], columns='_period_key', values=value_col, aggfunc='mean').reset_index()
        out = piv.copy()
        if 'pre_bc' not in out.columns or 'post_bc' not in out.columns:
            raise ValueError('Could not recover both pre and post BC values from long-format table.')
        # merge one representative group col if available
        for cand in ['display_group', 'lesion_group_detailed', 'lesion_group', 'raw_hit_type', 'group_label']:
            if cand in df.columns:
                m = df.groupby([animal_col, syll_col])[cand].first().reset_index()
                out = out.merge(m, on=[animal_col, syll_col], how='left')
                break

    out['pre_bc'] = pd.to_numeric(out['pre_bc'], errors='coerce')
    out['post_bc'] = pd.to_numeric(out['post_bc'], errors='coerce')
    if delta_col and delta_col in df.columns and 'delta_bc' not in out.columns:
        tmp = df[[animal_col, syll_col, delta_col]].copy()
        tmp.columns = [animal_col, syll_col, 'delta_bc']
        out = out.merge(tmp, on=[animal_col, syll_col], how='left')
    out['delta_bc'] = out['post_bc'] - out['pre_bc']

    out = out.rename(columns={animal_col: 'animal_id', syll_col: 'syllable'})
    out['animal_id'] = out['animal_id'].astype(str)
    out['syllable'] = out['syllable'].astype(str)

    group_col = first_present(out.columns, ['lesion_group_detailed', 'display_group', 'lesion_group', 'raw_hit_type', 'group_label'])
    meta_map = maybe_load_metadata(metadata_excel)
    if group_col is not None:
        out['detailed_group'] = out[group_col].apply(canonical_group)
    else:
        out['detailed_group'] = out['animal_id'].map(meta_map).apply(canonical_group)
    # Prefer workbook info if it exists
    if meta_map:
        out['detailed_group'] = out['animal_id'].map(meta_map).fillna(out['detailed_group']).apply(canonical_group)
    out['pooled_group'] = out['detailed_group'].map(pooled_group)
    out = out.dropna(subset=['pre_bc', 'post_bc', 'delta_bc'])
    return out.reset_index(drop=True)


def exact_signflip_p(diffs: np.ndarray, alternative: str = 'less') -> tuple[float, int]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan, 0
    observed = float(np.mean(diffs))
    if n <= 20:
        total = 1 << n
        values = np.empty(total, dtype=float)
        for mask in range(total):
            signs = np.where([(mask >> i) & 1 for i in range(n)], 1.0, -1.0)
            values[mask] = np.mean(diffs * signs)
        if alternative == 'less':
            p = float(np.mean(values <= observed + 1e-12))
        elif alternative == 'greater':
            p = float(np.mean(values >= observed - 1e-12))
        else:
            p = float(np.mean(np.abs(values) >= abs(observed) - 1e-12))
        return p, total
    rng = np.random.default_rng(123)
    reps = 100000
    values = np.empty(reps, dtype=float)
    for i in range(reps):
        signs = rng.choice([-1.0, 1.0], size=n)
        values[i] = np.mean(diffs * signs)
    if alternative == 'less':
        p = float(np.mean(values <= observed + 1e-12))
    elif alternative == 'greater':
        p = float(np.mean(values >= observed - 1e-12))
    else:
        p = float(np.mean(np.abs(values) >= abs(observed) - 1e-12))
    return p, reps


def paired_bootstrap_ci(diffs: np.ndarray, reps: int = 10000) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(456)
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        samp = rng.choice(diffs, size=n, replace=True)
        draws[i] = np.mean(samp)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return 'n.s.'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def plot_panel(by_bird: pd.DataFrame, out_path: Path, *, dpi: int = 300, show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    x1, x2 = 0, 1
    sel = by_bird['selected_delta_bc'].to_numpy(dtype=float)
    rem = by_bird['remaining_delta_bc'].to_numpy(dtype=float)

    color = DEFAULT_COLORS[POOLED_ML]
    # paired lines
    for a, b in zip(sel, rem):
        ax.plot([x1, x2], [a, b], color='0.7', linewidth=0.8, zorder=1)
    # boxes
    for xpos, vals, fill in [(x1, sel, 0.45), (x2, rem, 0.15)]:
        vals = vals[np.isfinite(vals)]
        box = ax.boxplot([vals], positions=[xpos], widths=0.50, patch_artist=True, showfliers=False)
        for patch in box['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(fill)
            patch.set_linewidth(1.5)
        for item in box['whiskers'] + box['caps']:
            item.set_color(color)
            item.set_linewidth(1.3)
        for med in box['medians']:
            med.set_color('0.2')
            med.set_linewidth(1.6)

    rng = np.random.default_rng(0)
    ax.scatter(np.full(len(sel), x1) + rng.uniform(-0.05, 0.05, len(sel)), sel,
               s=36, color=color, alpha=0.95, edgecolors='white', linewidths=0.4, zorder=3)
    ax.scatter(np.full(len(rem), x2) + rng.uniform(-0.05, 0.05, len(rem)), rem,
               s=36, color=color, alpha=0.60, edgecolors='white', linewidths=0.4, zorder=3)
    ax.axhline(0, color='0.45', linestyle='--', linewidth=1.0, zorder=0)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(['Top 15%', 'Remaining 85%'], fontsize=12.5)
    ax.set_ylabel('Δ Bhattacharyya coefficient\n(Post − Pre)', fontsize=16)
    for spine in ('top','right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)

    yvals = np.concatenate([sel[np.isfinite(sel)], rem[np.isfinite(rem)]]) if len(sel) and len(rem) else np.array([0.0])
    ymin = float(np.min(yvals))
    ymax = float(np.max(yvals))
    span = max(ymax - ymin, 0.05)
    ax.set_ylim(ymin - 0.10*span, ymax + 0.28*span)
    # annotation populated by caller later if desired; here leave space
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.18, top=0.97)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.08)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def add_annotation(fig_path: Path, by_bird: pd.DataFrame, p_holm: float, dpi: int = 300):
    # redraw with annotation to keep code simple
    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    x1, x2 = 0, 1
    sel = by_bird['selected_delta_bc'].to_numpy(dtype=float)
    rem = by_bird['remaining_delta_bc'].to_numpy(dtype=float)
    color = DEFAULT_COLORS[POOLED_ML]
    for a, b in zip(sel, rem):
        ax.plot([x1, x2], [a, b], color='0.7', linewidth=0.8, zorder=1)
    for xpos, vals, fill in [(x1, sel, 0.45), (x2, rem, 0.15)]:
        vals = vals[np.isfinite(vals)]
        box = ax.boxplot([vals], positions=[xpos], widths=0.50, patch_artist=True, showfliers=False)
        for patch in box['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(fill)
            patch.set_linewidth(1.5)
        for item in box['whiskers'] + box['caps']:
            item.set_color(color)
            item.set_linewidth(1.3)
        for med in box['medians']:
            med.set_color('0.2')
            med.set_linewidth(1.6)
    rng = np.random.default_rng(0)
    ax.scatter(np.full(len(sel), x1) + rng.uniform(-0.05, 0.05, len(sel)), sel,
               s=36, color=color, alpha=0.95, edgecolors='white', linewidths=0.4, zorder=3)
    ax.scatter(np.full(len(rem), x2) + rng.uniform(-0.05, 0.05, len(rem)), rem,
               s=36, color=color, alpha=0.60, edgecolors='white', linewidths=0.4, zorder=3)
    ax.axhline(0, color='0.45', linestyle='--', linewidth=1.0, zorder=0)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(['Top 15%', 'Remaining 85%'], fontsize=12.5)
    ax.set_ylabel('Δ Bhattacharyya coefficient\n(Post − Pre)', fontsize=16)
    for spine in ('top','right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)
    yvals = np.concatenate([sel[np.isfinite(sel)], rem[np.isfinite(rem)]]) if len(sel) and len(rem) else np.array([0.0])
    ymin = float(np.min(yvals))
    ymax = float(np.max(yvals))
    span = max(ymax - ymin, 0.05)
    bracket_y = ymax + 0.10*span
    tick = 0.03 * span
    ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y+tick, bracket_y+tick, bracket_y], color='0.15', linewidth=1.0)
    label = p_to_stars(p_holm) if np.isfinite(p_holm) else 'n.s.'
    ax.text((x1+x2)/2, bracket_y + tick + 0.02*span, f'{label}\nHolm p={p_holm:.3g}' if np.isfinite(p_holm) else label,
            ha='center', va='bottom', fontsize=11)
    ax.set_ylim(ymin - 0.10*span, ymax + 0.36*span)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.18, top=0.97)
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Figure 4 top-15% vs remaining-85% BC analysis for pooled M+L birds.')
    ap.add_argument('--bc-csv', required=True, help='BC metrics CSV containing pre/post BC values or a pivotable long form.')
    ap.add_argument('--selected-pairs-csv', required=True, help='top15_selected_animal_syllable_pairs.csv from the Figure 3 top-15% analysis.')
    ap.add_argument('--metadata-excel', default=None, help='Optional metadata workbook to recover detailed lesion hit types.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    bc = load_bc_table(args.bc_csv, metadata_excel=args.metadata_excel)
    sel = load_selected_pairs(args.selected_pairs_csv)
    merged = bc.merge(sel.assign(is_selected=True), on=['animal_id','syllable'], how='left')
    merged['is_selected'] = merged['is_selected'].fillna(False)
    merged['subset'] = np.where(merged['is_selected'], 'selected', 'remaining')
    merged['pooled_group'] = merged['detailed_group'].map(pooled_group)

    # Save all merged rows
    merged.to_csv(out_dir / 'figure4_top15_selected_vs_remaining_bc_rows.csv', index=False)

    ml = merged[merged['pooled_group'] == POOLED_ML].copy()
    per_bird = (
        ml.groupby(['animal_id','subset'], dropna=False)
          .agg(
              n_syllables=('syllable', 'nunique'),
              median_pre_bc=('pre_bc', 'median'),
              median_post_bc=('post_bc', 'median'),
              median_delta_bc=('delta_bc', 'median'),
          )
          .reset_index()
    )
    wide = per_bird.pivot(index='animal_id', columns='subset')
    wide.columns = ['_'.join(map(str,c)).strip('_') for c in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    # Rename for easier downstream use
    rename_map = {
        'n_syllables_selected': 'selected_n_syllables',
        'median_pre_bc_selected': 'selected_pre_bc',
        'median_post_bc_selected': 'selected_post_bc',
        'median_delta_bc_selected': 'selected_delta_bc',
        'n_syllables_remaining': 'remaining_n_syllables',
        'median_pre_bc_remaining': 'remaining_pre_bc',
        'median_post_bc_remaining': 'remaining_post_bc',
        'median_delta_bc_remaining': 'remaining_delta_bc',
    }
    wide = wide.rename(columns=rename_map)
    need = ['selected_delta_bc', 'remaining_delta_bc']
    wide = wide.dropna(subset=[c for c in need if c in wide.columns]).copy()
    wide['delta_selected_minus_remaining'] = wide['selected_delta_bc'] - wide['remaining_delta_bc']
    wide.to_csv(out_dir / 'figure4_top15_selected_vs_remaining_by_bird.csv', index=False)

    diffs = wide['delta_selected_minus_remaining'].to_numpy(dtype=float)
    p_one, nperm = exact_signflip_p(diffs, alternative='less')  # selected more negative than remaining
    ci_low, ci_high = paired_bootstrap_ci(diffs)
    p_two, _ = exact_signflip_p(diffs, alternative='two-sided')

    wilcoxon_p = np.nan
    if scipy_stats is not None and len(wide) > 0:
        try:
            res = scipy_stats.wilcoxon(wide['selected_delta_bc'], wide['remaining_delta_bc'], alternative='less', zero_method='wilcox')
            wilcoxon_p = float(res.pvalue)
        except Exception:
            wilcoxon_p = np.nan

    summary = pd.DataFrame([{
        'group': POOLED_ML,
        'n_birds': len(wide),
        'selected_mean_delta_bc': float(np.mean(wide['selected_delta_bc'])) if len(wide) else np.nan,
        'selected_median_delta_bc': float(np.median(wide['selected_delta_bc'])) if len(wide) else np.nan,
        'remaining_mean_delta_bc': float(np.mean(wide['remaining_delta_bc'])) if len(wide) else np.nan,
        'remaining_median_delta_bc': float(np.median(wide['remaining_delta_bc'])) if len(wide) else np.nan,
        'mean_difference_selected_minus_remaining': float(np.mean(diffs)) if len(diffs) else np.nan,
        'bootstrap_ci_low': ci_low,
        'bootstrap_ci_high': ci_high,
        'signflip_p_one_sided_selected_less': p_one,
        'signflip_p_two_sided': p_two,
        'wilcoxon_p_one_sided_selected_less': wilcoxon_p,
        'n_signflip_permutations': nperm,
    }])
    summary.to_csv(out_dir / 'figure4_top15_selected_vs_remaining_summary.csv', index=False)

    # plotting
    plot_path = out_dir / 'Figure4_top15_vs_remaining_BC_ML.png'
    add_annotation(plot_path, wide, p_one, dpi=300)

    with (out_dir / 'figure4_top15_selected_vs_remaining_summary.txt').open('w', encoding='utf-8') as f:
        f.write('Figure 4 top-15% vs remaining-85% BC analysis\n')
        f.write('================================================\n\n')
        f.write(f'BC input: {args.bc_csv}\n')
        f.write(f'Selected pairs input: {args.selected_pairs_csv}\n')
        f.write(f'Pooled M+L birds contributing: {len(wide)}\n\n')
        if len(wide):
            f.write('Per-bird selected/remaining counts and median ΔBC (Post - Pre)\n')
            f.write('------------------------------------------------------------\n')
            for _, row in wide.iterrows():
                f.write(
                    f"{row['animal_id']}: selected n={int(row.get('selected_n_syllables', np.nan)) if pd.notna(row.get('selected_n_syllables', np.nan)) else 'NA'}, "
                    f"remaining n={int(row.get('remaining_n_syllables', np.nan)) if pd.notna(row.get('remaining_n_syllables', np.nan)) else 'NA'}, "
                    f"selected ΔBC={row['selected_delta_bc']:.6g}, remaining ΔBC={row['remaining_delta_bc']:.6g}, "
                    f"selected-minus-remaining={row['delta_selected_minus_remaining']:.6g}\n"
                )
            f.write('\n')
        f.write('Summary\n')
        f.write('-------\n')
        if len(summary):
            r = summary.iloc[0]
            f.write(f"Selected mean ΔBC: {r['selected_mean_delta_bc']:.6g}\n")
            f.write(f"Remaining mean ΔBC: {r['remaining_mean_delta_bc']:.6g}\n")
            f.write(f"Mean(selected - remaining): {r['mean_difference_selected_minus_remaining']:.6g}\n")
            f.write(f"95% bootstrap CI: [{r['bootstrap_ci_low']:.6g}, {r['bootstrap_ci_high']:.6g}]\n")
            f.write(f"One-sided exact sign-flip p (selected < remaining): {r['signflip_p_one_sided_selected_less']:.6g}\n")
            f.write(f"Two-sided exact sign-flip p: {r['signflip_p_two_sided']:.6g}\n")
            f.write(f"One-sided Wilcoxon p (selected < remaining): {r['wilcoxon_p_one_sided_selected_less']:.6g}\n")
        f.write(f"\nSaved plot: {plot_path}\n")

    print('[OK] Wrote outputs to', out_dir)
    for name in [
        'Figure4_top15_vs_remaining_BC_ML.png',
        'figure4_top15_selected_vs_remaining_bc_rows.csv',
        'figure4_top15_selected_vs_remaining_by_bird.csv',
        'figure4_top15_selected_vs_remaining_summary.csv',
        'figure4_top15_selected_vs_remaining_summary.txt',
    ]:
        print(' ', out_dir / name)


if __name__ == '__main__':
    main()
