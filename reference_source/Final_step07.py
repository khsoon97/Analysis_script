# %%
# ==============================================================================
# [Final Step 07] T-cell Anchored HLA-DR B-cell Motif Analysis
# ▷ Input: protected Step07 saved Figure-source CSVs in tables/mv1
# ▷ Purpose: re-export canonical Step07-01–05 Figure-only PNG/PPTX files
# ▷ Excluded: motif/KDTree/statistical/TIFF calculations and Step07-06, whose saved plot sources are incomplete
# ==============================================================================
import os
import sys

# First guard against recursion-limit or numba cache-locator failures during the
# anndata/scanpy/numba stages of importing Final_utils in SpaceFlow Python 3.13.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
os.environ.setdefault('FINAL_LIGHT_IMPORTS', '1')
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
os.environ['FINAL_FIGURE_ONLY_WRITE_GUARD'] = '1'
from Final_utils import *

from itertools import combinations
import matplotlib.colors as mcolors

# ==============================================================================
# [Step 07-0] Shared Figure-only settings
# ==============================================================================
STEP07_FIGURE_ONLY = True

if not STEP07_FIGURE_ONLY:
    raise RuntimeError(
        "Final_step07 is currently Figure-only. "
        "Use separate analysis code to permit writing analysis artifacts."
    )
STEP06_FDR_ALPHA = 0.05
STEP07_WRITE_INDIVIDUAL_PPTX = True
STEP07_FIGURE_ONLY_OUTPUTS = {
    'Final_MV1_Step07A_TcellAnchored_DRB_Motif_Composition.png',
    'Final_MV1_Step07B_TcellAnchored_Tsubtype_Composition_ByMotif.png',
    'Final_MV1_Step07C_TcellAnchored_FunctionalScores_ByMotif.png',
    'Final_MV1_Step07C2_TcellAnchored_MotifPairwiseStats_ByTSubtype.png',
    'Final_MV1_Step07D_TcellAnchored_FunctionalScore_Heatmap_BySubtype.png',
}
STEP07_PRESERVED_OUTPUTS_WITHOUT_COMPLETE_PLOT_SOURCE = {
    'Final_MV1_Step07G_TcellAnchored_DRB_Motif_TIFF_Multiscale_Map.png',
}
STEP06_DISPLAY_SAVED_PNG = not FINAL_SUPPRESS_INTERACTIVE_SHOW
STEP06_T_CELL_LABEL_ORDER = ['CD4_CTL', 'CD8T', 'Tfh', 'Th', 'Tpr', 'Treg']
STEP06_T_SUBTYPE_COMPOSITION_COLORS = {
    'CD4_CTL': '#7E57C2',
    'CD8T': '#1E88E5',
    'Tfh': '#F9A825',
    'Th': '#43A047',
    'Tpr': '#E64A19',
    'Treg': '#6D4C41',
}


def _step07_display_t_subset(label):
    """Return publication display text without changing stored subtype keys."""
    return {'CD4_CTL': 'CD4 CTL', 'CD8T': 'CD8 T'}.get(str(label), str(label))


def _save_step06_matplotlib(fig, filename, label, *, tight=True):
    """Save one member of the complete active Step07 Figure-only set."""
    if filename not in STEP07_FIGURE_ONLY_OUTPUTS:
        plt.close(fig)
        print(f"  ⏩ {label} output excluded: {filename}")
        return None
    out_path = os.path.join(save_dir, filename)
    pptx_path = os.path.splitext(out_path)[0] + '.pptx'
    saved_png_path, saved_pptx_path = save_final_figure_pair(
        fig, out_path, pptx_path,
        dpi=300, facecolor='white',
        auxiliary_tight=False,
        validate=True,
    )
    plt.close(fig)
    if saved_png_path is None:
        return None
    print(f"  ✅ {label} PNG saved: {saved_png_path}")
    if STEP07_WRITE_INDIVIDUAL_PPTX:
        print(f"  ✅ {label} PPTX saved: {saved_pptx_path}")
    if STEP06_DISPLAY_SAVED_PNG:
        IPython.display.display(IPython.display.Image(filename=saved_png_path))
    return saved_png_path


STEP07_MAIN_FIGSIZES = {
    'Final_MV1_Step07A_TcellAnchored_DRB_Motif_Composition.png': (6.20, 6.35),
    'Final_MV1_Step07D_TcellAnchored_FunctionalScore_Heatmap_BySubtype.png': (8.30, 7.55),
}
STEP07_SUPPLEMENT_FIGSIZES = {
    # A4 is only the final typesetting boundary; save each file on its own canvas with the aspect ratio below.
    'Final_MV1_Step07B_TcellAnchored_Tsubtype_Composition_ByMotif.png': (6.80, 6.35),
    'Final_MV1_Step07C_TcellAnchored_FunctionalScores_ByMotif.png': (8.80, 5.35),
    'Final_MV1_Step07C2_TcellAnchored_MotifPairwiseStats_ByTSubtype.png': (10.40, 4.55),
}

def _bh_fdr(p_values):
    """Apply Benjamini-Hochberg FDR correction, preserving NaNs as NaNs."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked
    q[valid] = q_valid
    return q

def _sig_from_q(q):
    """Convert FDR q-values to asterisks for figure annotations."""
    if not np.isfinite(q):
        return 'ns'
    if q < 0.001:
        return '***'
    if q < 0.01:
        return '**'
    if q < STEP06_FDR_ALPHA:
        return '*'
    return 'ns'


def _step07_anchor_stats_within_family_bh(frame):
    """Derive Figure-only BH q values within each score × T-cell stratum family."""
    required = {
        'Score', 'T_Subset_Group', 'Group_A', 'Group_B', 'p_value',
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f'[Step 07] anchor-level stats lack BH source columns: {missing}')
    out = frame.copy()
    family_columns = ['Score', 'T_Subset_Group']
    family_sizes = out.groupby(family_columns, sort=False, dropna=False).size()
    invalid = family_sizes[family_sizes.ne(6)]
    if not invalid.empty:
        raise RuntimeError(
            '[Step 07] expected exactly six motif contrasts per score × stratum; '
            f'found {invalid.to_dict()}'
        )
    q_column = 'q_BH_within_score_stratum_6'
    out[q_column] = np.nan
    for _, row_indices in out.groupby(family_columns, sort=False, dropna=False).groups.items():
        row_indices = list(row_indices)
        out.loc[row_indices, q_column] = _bh_fdr(
            pd.to_numeric(out.loc[row_indices, 'p_value'], errors='coerce').to_numpy(dtype=float)
        )
    out['sig_BH_within_score_stratum_6'] = out[q_column].map(_sig_from_q)
    return out


def _step07_add_pairwise_significance(axis, score_stats, motif_order, *, data_ylim=None):
    """Draw all six motif comparisons as compact line brackets with BH-FDR stars."""
    if len(motif_order) != 4:
        raise RuntimeError(f'[Step 07] pairwise bracket contract requires four motifs: {motif_order}')
    expected_pairs = set(combinations(motif_order, 2))
    observed_pairs = set(zip(
        score_stats['Group_A'].astype(str), score_stats['Group_B'].astype(str),
    ))
    if observed_pairs != expected_pairs:
        raise RuntimeError(
            '[Step 07] pairwise bracket contrast mismatch: '
            f'missing={sorted(expected_pairs - observed_pairs)}, '
            f'extra={sorted(observed_pairs - expected_pairs)}'
        )
    x_index = {group: position for position, group in enumerate(motif_order)}
    levels = {
        (0, 1): 0,
        (2, 3): 0,
        (1, 2): 1,
        (0, 2): 2,
        (1, 3): 3,
        (0, 3): 4,
    }
    y_low, y_high = data_ylim if data_ylim is not None else axis.get_ylim()
    y_range = max(y_high - y_low, 1e-6)
    base = y_high + 0.025 * y_range
    step = 0.060 * y_range
    bracket_height = 0.012 * y_range
    for _, row in score_stats.iterrows():
        left = x_index[str(row['Group_A'])]
        right = x_index[str(row['Group_B'])]
        if left > right:
            left, right = right, left
        y_value = base + levels[(left, right)] * step
        axis.plot(
            [left, left, right, right],
            [y_value, y_value + bracket_height, y_value + bracket_height, y_value],
            color='#4A4A4A', linewidth=0.85, clip_on=False,
        )
        axis.text(
            (left + right) / 2,
            y_value + bracket_height,
            str(row['sig_BH_within_score_stratum_6']),
            ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#333333',
        )
    axis.set_ylim(y_low, base + 4.80 * step)
    axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, prune='upper'))


# Initialize shared variables
save_dir = FIG_DIR
os.makedirs(save_dir, exist_ok=True)

print(f"📁 Output directory: {save_dir}")
print('  ▶ Generating Figure-only outputs from protected Step07 saved tables.')


def _step07_require_saved_table(filename, required_columns, *, usecols=None):
    """Load one protected Step07 main-result table for Figure rendering."""
    path = os.path.join(FINAL_TABLE_DIR, filename)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise FileNotFoundError(f'[Step 07] saved Figure source is missing: {path}')
    if usecols is not None:
        usecols = list(dict.fromkeys(usecols))
        omitted = sorted(set(required_columns) - set(usecols))
        if omitted:
            raise ValueError(f'[Step 07] usecols omit required columns {omitted}: {path}')
    frame = pd.read_csv(path, usecols=usecols)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f'[Step 07] saved Figure source lacks columns {missing}: {path}')
    return frame


def _step07_render_saved_motif_figures():
    """Render table-backed Step07 panels without GMM, KDTree, motif, or tests."""
    motif_order = [
        'DR+ B-dominant motif', 'Mixed B motif',
        'DR- B-dominant motif', 'B-poor motif',
    ]
    motif_colors = {
        'DR+ B-dominant motif': '#C43D3D', 'Mixed B motif': '#7F8C8D',
        'DR- B-dominant motif': '#2F79A8', 'B-poor motif': '#BDBDBD',
    }
    saved_reports = []

    name_a = 'Final_MV1_Step07A_TcellAnchored_DRB_Motif_Composition.png'
    if name_a in STEP07_FIGURE_ONLY_OUTPUTS:
        composition = _step07_require_saved_table(
            'MV1_Step07_TcellAnchored_DRB_Motif_CompositionSummary.csv',
            ['Motif_Group', 'T_anchor_n'],
        )
        feature_columns = [
            column for column in composition.columns
            if column.startswith('Mean_N_neighbor__') and column != 'Mean_N_neighbor__Other'
        ]
        if not feature_columns:
            raise RuntimeError('[Step 07A] saved composition table has no neighbor features')
        frame = (
            composition.set_index('Motif_Group')[feature_columns]
            .reindex([group for group in motif_order if group in set(composition['Motif_Group'].astype(str))])
            .T
        )
        frame.index = [
            label.replace('Mean_N_neighbor__', '').replace('plus', '+').replace('_', ' ')
            for label in frame.index
        ]
        fig = plt.figure(figsize=STEP07_MAIN_FIGSIZES[name_a])
        axis = fig.add_axes([0.245, 0.210, 0.610, 0.755])
        colorbar_axis = fig.add_axes([0.918, 0.355, 0.018, 0.360])
        sns.heatmap(
            frame, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.6,
            linecolor='white', cbar_ax=colorbar_axis,
            annot_kws={'fontsize': 8.6, 'fontweight': 'bold'}, ax=axis,
        )
        axis.set_xlabel('HLA-DR B-cell motif group', fontsize=11.2, fontweight='bold')
        axis.set_ylabel('Neighbor cell state', fontsize=11.2, fontweight='bold')
        plt.setp(axis.get_xticklabels(), rotation=33, ha='right', fontsize=8.8, fontweight='bold')
        plt.setp(axis.get_yticklabels(), rotation=0, fontsize=8.6, fontweight='bold')
        colorbar_axis.set_title('Mean neighbor\ncount', fontsize=8.2, fontweight='bold', pad=4)
        saved_reports.append(_save_step06_matplotlib(
            fig, name_a, 'Step07-01 saved motif composition', tight=False,
        ))

    name_b = 'Final_MV1_Step07B_TcellAnchored_Tsubtype_Composition_ByMotif.png'
    if name_b in STEP07_FIGURE_ONLY_OUTPUTS:
        subtype = _step07_require_saved_table(
            'MV1_Step07_TcellAnchored_DRB_Motif_TcellSubtypeComposition.csv',
            ['Motif_Group', 'T_Subset', 'Percent_of_T_anchors_in_motif', 'T_subset_count_in_motif'],
        )
        order = [group for group in motif_order if group in set(subtype['Motif_Group'].astype(str))]
        subtype_order = [
            label for label in STEP06_T_CELL_LABEL_ORDER
            if label in set(subtype['T_Subset'].astype(str))
        ]
        percent = subtype.pivot(
            index='Motif_Group', columns='T_Subset', values='Percent_of_T_anchors_in_motif',
        ).reindex(index=order, columns=subtype_order).fillna(0)
        fig, axis = plt.subplots(figsize=STEP07_SUPPLEMENT_FIGSIZES[name_b])
        bottom = np.zeros(len(percent), dtype=float)
        x_positions = np.arange(len(percent))
        for label in subtype_order:
            values = percent[label].to_numpy(dtype=float)
            axis.bar(
                x_positions, values, bottom=bottom,
                color=STEP06_T_SUBTYPE_COMPOSITION_COLORS.get(label, '#999999'),
                edgecolor='white', linewidth=0.7, width=0.54,
                label=_step07_display_t_subset(label),
            )
            bottom += values
        axis.set_xticks(x_positions)
        axis.set_xticklabels([group.replace(' motif', '') for group in order], rotation=32, ha='right')
        axis.set_ylim(0, 100)
        axis.set_ylabel('Anchor T-cell composition (%)', fontsize=11.0, fontweight='bold')
        axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=8.5)
        axis.grid(axis='y', linestyle='--', alpha=0.3)
        fig.subplots_adjust(left=0.15, right=0.985, top=0.80, bottom=0.25)
        saved_reports.append(_save_step06_matplotlib(
            fig, name_b, 'Step07-02 saved T-cell subtype composition', tight=False,
        ))

    name_c = 'Final_MV1_Step07C_TcellAnchored_FunctionalScores_ByMotif.png'
    name_c2 = 'Final_MV1_Step07C2_TcellAnchored_MotifPairwiseStats_ByTSubtype.png'
    name_d = 'Final_MV1_Step07D_TcellAnchored_FunctionalScore_Heatmap_BySubtype.png'
    if name_c in STEP07_FIGURE_ONLY_OUTPUTS or name_c2 in STEP07_FIGURE_ONLY_OUTPUTS:
        anchor_stats = _step07_require_saved_table(
            'MV1_Step07_TcellAnchored_DRB_Motif_AnchorLevelScoreStats.csv',
            [
                'Test', 'Score', 'T_Subset_Group', 'Group_A', 'Group_B',
                'Median_Delta_A_minus_B', 'p_value',
            ],
            usecols=[
                'Test', 'Score', 'T_Subset_Group', 'Group_A', 'Group_B',
                'Median_Delta_A_minus_B', 'p_value',
            ],
        )
        if not anchor_stats['Test'].astype(str).eq('Mann_Whitney_U_motif_anchor_level').all():
            raise RuntimeError('[Step 07] saved anchor-stat source has an unexpected test contract')
        anchor_stats = _step07_anchor_stats_within_family_bh(anchor_stats)
        score_order = [
            score for score in ('Activation_Score', 'Exhaustion_Score')
            if score in set(anchor_stats['Score'].astype(str))
        ]
        if not score_order:
            raise RuntimeError('[Step 07] saved anchor-stat table has no supported scores')

        motif_short = {
            'DR+ B-dominant motif': 'DR+ B-dominant',
            'Mixed B motif': 'Mixed B',
            'DR- B-dominant motif': 'DR− B-dominant',
            'B-poor motif': 'B-poor',
        }
        motif_box_labels = {
            'DR+ B-dominant motif': 'DR+\nB-dominant',
            'Mixed B motif': 'Mixed B',
            'DR- B-dominant motif': 'DR−\nB-dominant',
            'B-poor motif': 'B-poor',
        }
        motif_tiny = {
            'DR+ B-dominant motif': 'DR+',
            'Mixed B motif': 'Mix',
            'DR- B-dominant motif': 'DR−',
            'B-poor motif': 'Poor',
        }
        score_labels = {
            'Activation_Score': 'Activation score',
            'Exhaustion_Score': 'Exhaustion score',
        }

        if name_c in STEP07_FIGURE_ONLY_OUTPUTS:
            anchor_scores = _step07_require_saved_table(
                'MV1_Step07_TcellAnchored_DRB_Motif_AnchorTable.csv',
                ['Motif_Group', 'Activation_Score', 'Exhaustion_Score'],
                usecols=['Motif_Group', 'Activation_Score', 'Exhaustion_Score'],
            )
            order = [
                group for group in motif_order
                if group in set(anchor_scores['Motif_Group'].astype(str))
            ]
            if order != motif_order:
                raise RuntimeError(f'[Step 07-03] saved anchor table lacks canonical motif order: {order}')
            fig, axes = plt.subplots(
                1, len(score_order), figsize=STEP07_SUPPLEMENT_FIGSIZES[name_c],
                squeeze=False, sharey=True,
            )
            for panel_index, (axis, score) in enumerate(zip(axes.ravel(), score_order)):
                arrays = [
                    pd.to_numeric(
                        anchor_scores.loc[anchor_scores['Motif_Group'].astype(str).eq(group), score],
                        errors='coerce',
                    ).dropna().to_numpy(dtype=float)
                    for group in order
                ]
                axis.axhline(0, color='#8A8A8A', linewidth=0.8, zorder=0)
                box = axis.boxplot(
                    arrays, positions=np.arange(len(order)), widths=0.48,
                    patch_artist=True, showfliers=False,
                    medianprops={'color': '#222222', 'linewidth': 1.65},
                    whiskerprops={'color': '#444444', 'linewidth': 1.05},
                    capprops={'color': '#444444', 'linewidth': 1.05},
                )
                for patch, group in zip(box['boxes'], order):
                    patch.set_facecolor(motif_colors[group])
                    patch.set_edgecolor('#333333')
                    patch.set_linewidth(1.0)
                    patch.set_alpha(0.84)
                axis.set_xticks(np.arange(len(order)))
                axis.set_xticklabels([motif_box_labels[group] for group in order])
                axis.set_title(
                    score_labels.get(score, score), fontsize=14.5, fontweight='bold', pad=8,
                )
                axis.set_xlabel('')
                axis.set_ylabel(
                    'Anchor T-cell score' if panel_index == 0 else '',
                    fontsize=11.5, fontweight='bold',
                )
                axis.tick_params(axis='x', rotation=0, labelsize=9.1, pad=6)
                axis.tick_params(axis='y', labelsize=9.5)
                for tick_label in axis.get_xticklabels():
                    tick_label.set_fontweight('bold')
                    tick_label.set_linespacing(1.15)
                axis.set_axisbelow(True)
                axis.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.28)
                axis.grid(axis='x', visible=False)
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
            y_low, y_high = axes.ravel()[0].get_ylim()
            for axis, score in zip(axes.ravel(), score_order):
                score_stats = anchor_stats.loc[
                    anchor_stats['Score'].astype(str).eq(score)
                    & anchor_stats['T_Subset_Group'].astype(str).eq('All T cells')
                ].copy()
                _step07_add_pairwise_significance(
                    axis, score_stats, order, data_ylim=(y_low, y_high),
                )
            fig.supxlabel(
                'HLA-DR B-cell motif group', fontsize=12.2, fontweight='bold', y=0.060,
            )
            fig.subplots_adjust(left=0.075, right=0.990, top=0.915, bottom=0.205, wspace=0.115)
            saved_reports.append(_save_step06_matplotlib(
                fig, name_c, 'Step07-03 anchor-cell functional scores', tight=False,
            ))

        if name_c2 in STEP07_FIGURE_ONLY_OUTPUTS:
            row_order = [
                label for label in ['All T cells'] + STEP06_T_CELL_LABEL_ORDER
                if label in set(anchor_stats['T_Subset_Group'].astype(str))
            ]
            row_display = {
                label: _step07_display_t_subset(label) for label in row_order
            }
            pair_order = list(combinations(motif_order, 2))
            comparison_order = [
                f"{motif_tiny[group_a]} vs {motif_tiny[group_b]}"
                for group_a, group_b in pair_order
            ]
            plot_stats = anchor_stats.copy()
            plot_stats['Comparison_Display'] = [
                f"{motif_tiny[str(group_a)]} vs {motif_tiny[str(group_b)]}"
                for group_a, group_b in zip(plot_stats['Group_A'], plot_stats['Group_B'])
            ]
            finite_values = pd.to_numeric(plot_stats['Median_Delta_A_minus_B'], errors='coerce')
            vmax = (
                max(0.35, float(np.nanpercentile(np.abs(finite_values.dropna()), 95)))
                if finite_values.notna().any() else 1.0
            )
            fig = plt.figure(figsize=STEP07_SUPPLEMENT_FIGSIZES[name_c2])
            grid = fig.add_gridspec(
                1, 3, width_ratios=[1, 1, 0.035],
                left=0.105, right=0.955, top=0.89, bottom=0.23, wspace=0.09,
            )
            axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
            colorbar_slot = fig.add_subplot(grid[0, 2])
            colorbar_slot.set_axis_off()
            colorbar_slot_bbox = colorbar_slot.get_position()
            colorbar_height = colorbar_slot_bbox.height * 0.58
            colorbar_axis = fig.add_axes([
                colorbar_slot_bbox.x0 - 0.012,
                colorbar_slot_bbox.y0 + (colorbar_slot_bbox.height - colorbar_height) / 2,
                colorbar_slot_bbox.width,
                colorbar_height,
            ])
            cmap = sns.color_palette('vlag', as_cmap=True)
            for panel_index, (axis, score) in enumerate(zip(axes, score_order)):
                subset = plot_stats.loc[plot_stats['Score'].astype(str).eq(score)].copy()
                values = subset.pivot_table(
                    index='T_Subset_Group', columns='Comparison_Display',
                    values='Median_Delta_A_minus_B', aggfunc='first',
                ).reindex(index=row_order, columns=comparison_order)
                annotations = values.apply(
                    lambda column: column.map(
                        lambda value: '' if not np.isfinite(value) else (
                            '0.00' if abs(float(value)) < 0.005 else f'{float(value):+.2f}'
                        )
                    )
                )
                sns.heatmap(
                    values.rename(index=row_display),
                    annot=annotations.rename(index=row_display), fmt='',
                    cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                    linewidths=0.45, linecolor='white', cbar=False,
                    annot_kws={
                        'fontsize': 10.2, 'fontweight': 'bold',
                        'ha': 'center', 'va': 'center',
                    },
                    ax=axis,
                )
                axis.set_title(score_labels.get(score, score), fontsize=15.0, fontweight='bold', pad=8)
                axis.set_xlabel('')
                axis.set_ylabel('')
                axis.tick_params(axis='x', labelsize=8.3)
                axis.tick_params(axis='y', labelsize=8.7)
                plt.setp(axis.get_xticklabels(), rotation=36, ha='right')
                plt.setp(axis.get_yticklabels(), rotation=0)
                if panel_index:
                    axis.set_yticklabels([])
                    axis.tick_params(axis='y', left=False)
            norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
            scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar.set_array([])
            colorbar = fig.colorbar(scalar, cax=colorbar_axis)
            colorbar.ax.tick_params(labelsize=9.0)
            colorbar.ax.set_title(
                'Median\nscore Δ\n(left − right)', fontsize=11.2, fontweight='bold', pad=8,
            )
            fig.supylabel('Anchor T-cell group', fontsize=11.7, fontweight='bold', x=0.018)
            fig.supxlabel('Motif comparison (left − right)', fontsize=11.5, fontweight='bold', y=0.035)
            saved_reports.append(_save_step06_matplotlib(
                fig, name_c2, 'Step07-04 anchor-cell median differences', tight=False,
            ))

    if name_d in STEP07_FIGURE_ONLY_OUTPUTS:
        # Figure 6b: pooled individual-anchor means, not tissue-summary means.
        anchor_scores = _step07_require_saved_table(
            'MV1_Step07_TcellAnchored_DRB_Motif_AnchorTable.csv',
            ['Motif_Group', 'T_Subset', 'Activation_Score', 'Exhaustion_Score'],
            usecols=['Motif_Group', 'T_Subset', 'Activation_Score', 'Exhaustion_Score'],
        )
        score_order = ['Activation_Score', 'Exhaustion_Score']
        subset_order = [
            label for label in STEP06_T_CELL_LABEL_ORDER
            if label in set(anchor_scores['T_Subset'].astype(str))
        ]
        panels = [
            anchor_scores.pivot_table(
                index='Motif_Group', columns='T_Subset', values=score,
                aggfunc='mean', observed=True,
            ).reindex(index=motif_order, columns=subset_order)
            for score in score_order
        ]
        motif_display = {
            'DR+ B-dominant motif': 'DR+ B-dominant',
            'Mixed B motif': 'Mixed B',
            'DR- B-dominant motif': 'DR- B-dominant',
            'B-poor motif': 'B-poor',
        }
        fig = plt.figure(figsize=STEP07_MAIN_FIGSIZES[name_d])
        for index, (score, panel) in enumerate(zip(score_order, panels)):
            bottom = 0.578 if index == 0 else 0.106
            axis = fig.add_axes([0.205, bottom, 0.697, 0.360])
            colorbar_axis = fig.add_axes([0.925, bottom + 0.096, 0.024, 0.168])
            finite = panel.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            vmax = max(0.5, float(np.nanpercentile(np.abs(finite), 95))) if finite.size else 1.0
            score_label = score.replace('_', ' ').replace('Score', 'score')
            sns.heatmap(
                panel.rename(index=motif_display, columns=_step07_display_t_subset),
                annot=True, fmt='.2f', cmap='vlag', center=0,
                vmin=-vmax, vmax=vmax, cbar_ax=colorbar_axis,
                linewidths=0.6, linecolor='white', ax=axis,
                annot_kws={'fontsize': 14.0, 'fontweight': 'bold'},
            )
            axis.set_title(score_label, fontsize=18.0, fontweight='bold', pad=8)
            axis.set_xlabel('Anchor T-cell subtype' if index else '',
                            fontsize=12.0, fontweight='bold', labelpad=5)
            axis.set_ylabel('')
            axis.tick_params(axis='x', rotation=25, labelsize=10.0, pad=4)
            axis.tick_params(axis='y', rotation=0, labelsize=10.0, pad=3)
            plt.setp(axis.get_xticklabels(), ha='right', fontweight='bold')
            plt.setp(axis.get_yticklabels(), fontweight='bold')
            colorbar_axis.set_title(
                'Mean\n' + score_label.lower().replace(' ', '\n'),
                fontsize=9.5, fontweight='bold', pad=5, x=1.0,
            )
            colorbar_axis.tick_params(labelsize=9.0)
        fig.supylabel('Motif group', x=0.018, fontsize=19.0, fontweight='bold')
        saved_reports.append(_save_step06_matplotlib(
            fig, name_d, 'Step07-05 pooled anchor-mean functional-score heatmap', tight=False,
        ))

    return saved_reports


if STEP07_FIGURE_ONLY:
    _step07_saved_reports = _step07_render_saved_motif_figures()
    _step07_saved_count = sum(
        report is not None for report in _step07_saved_reports
    )
    if _step07_saved_count != len(STEP07_FIGURE_ONLY_OUTPUTS):
        raise RuntimeError(
            'Step07 active Figure family was not saved completely: '
            f'{_step07_saved_count}/{len(STEP07_FIGURE_ONLY_OUTPUTS)} pairs'
        )
    print(
        '✅ Step07 Figure-only saved-result workflow: '
        f'saved_pairs={_step07_saved_count}'
    )
    print(
        '⏩ Step07-06 SKIPPED: complete saved plot source unavailable; '
        'no motif/KDTree/TIFF fallback was run.'
    )
    raise SystemExit(0)

# %%
# Canonical Step07 retains only the Figure-only renderer based on protected saved tables.
# This file does not run motif, KDTree, statistical, or TIFF analyses.
