# %%
# ==============================================================================
# [Final Step 06] Spatial Immunology and Functional T-B Mapping
# ▷ Original source: Step 06 block in mv4_all_V8.py
# ▷ Default inputs: Final-local Step06 checkpoint and registered protected tables
# ▷ Default outputs: figures/v2/Final_mv1_Step06-* Figure pairs selected by target
# ▷ Recalculation path: start from the Step05 checkpoint only with separate authorization.
# ==============================================================================
import os
import sys

# Block checkpoint/table writes during Figure-only execution.
os.environ['FINAL_FIGURE_ONLY_WRITE_GUARD'] = '1'
# Guard before Step06 imports because anndata/scanpy/numba imports in SpaceFlow
# Python 3.13 may fail due to recursion limits or numba cache-locator issues.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
os.environ.setdefault('FINAL_LIGHT_IMPORTS', '1')
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
from Final_utils import *
import Final_utils as _final_utils

_final_utils.FINAL_FIGURE_ONLY_WRITE_GUARD = True
_final_figure_stem_registered = _final_utils._final_figure_stem_registered

# The preserved recalculation block references this variable assuming Step05 is complete.
step05_skip = False

# [Step 06-1] Load checkpoints and initialize shared settings
# ▷ Figure-only defaults to the Final Step06 checkpoint; only explicit recalculation reads Step05.
# ▷ Includes restoration of required variables such as unique_labels and save_dir
# ==============================================================================

# 1. Additional Step06-specific imports
# Final_utils already supplies most shared aliases (pd/np/sc/sp/px/go/plt/tqdm, etc.).
# Explicitly add only statistical functions not provided by Final_utils here.
import re
from itertools import combinations
from scipy.stats import wilcoxon, chi2_contingency, fisher_exact, mannwhitneyu

# 2. Define checkpoint paths
CHECKPOINT_STEP05 = os.path.join(CKPT_DIR, 'mv1_step05_cn.h5ad')
CHECKPOINT_STEP06 = os.path.join(FINAL_CKPT_WRITE_DIR, 'mv1_step06.h5ad')
STEP06_FIGURE_ONLY = True
STEP06_FORCE_RELOAD_INPUT = True  # Restart from Step05 to avoid stale adata when rebuilding the checkpoint.


def _step06_write_table_if_enabled(frame, output_path, **kwargs):
    """Write analysis tables only during an explicitly authorized analysis run."""
    if STEP06_FIGURE_ONLY:
        raise PermissionError(
            f'Step06 Figure-only blocked protected analysis-table write: {output_path}'
        )
    frame.to_csv(output_path, **kwargs)
    return True

# 3. Checkpoint loading logic
if STEP06_FORCE_RELOAD_INPUT and 'adata' in globals():
    adata = None

if 'adata' not in dir() or adata is None:
    if STEP06_FIGURE_ONLY and os.path.exists(CHECKPOINT_STEP06):
        adata = sc.read_h5ad(CHECKPOINT_STEP06)
        print(f"🔹 [Step 06-1] Figure-only mode → load completed Final Step06 results: {CHECKPOINT_STEP06}")
    elif not STEP06_FIGURE_ONLY and os.path.exists(CHECKPOINT_STEP05):
        adata = sc.read_h5ad(CHECKPOINT_STEP05)
        print(f"✅ [Step 06-1] Step 05 checkpoint loaded: {os.path.basename(CHECKPOINT_STEP05)}")
    else:
        raise FileNotFoundError(
            '[Step 06-1] Figure-only rendering requires the Final Step06 checkpoint; '
            'V8/Legacy and Step05 calculation fallbacks are disabled.\n'
            f'  required: {CHECKPOINT_STEP06}'
        )
else:
    print("✅ [Step 06-1] adata is in memory. Using results from the previous step.")

# Incorporate meeting notes: from Step06 onward, retain FDR-adjusted q-values alongside p-values in figures.
STEP06_HLA_DR_MARKER_CANDIDATES = ['HLA-DR', 'HLA-DRA', 'HLA-DRB1']
STEP06_FDR_ALPHA = 0.05
STEP06_STATS_CSV = 'MV1_Step06_Spatial_Immunology_Stats.csv'
STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM = 50.0
# Roider CODEX annotation x/y coordinates are confirmed to use the original stitched-image pixel coordinate system.
# Public TIFFs are downsampled twofold relative to annotation coordinates; for distance analyses,
# convert annotation coordinates to physical distances by multiplying by 0.37744 um/unit.
STEP06_COORD_UM_PER_UNIT = 0.37744
STEP06_PUBLIC_TIFF_DOWNSAMPLE_FACTOR = 2.0
STEP06_COORD_UNIT_SOURCE = (
    "annotation_x_y_original_stitched_pixel; "
    "20x_Keyence_PlanApo_inferred_0.37744um_per_pixel; "
    "public_tiff_downsample_factor_2"
)
STEP06_NEAREST_B_STATS_CSV = 'MV1_Step065_PhenoImagerStyle_NearestB_Stats.csv'
STEP06_NEAREST_B_DISTANCE_SUMMARY_CSV = 'MV1_Step065_PhenoImagerStyle_Distance_Summary.csv'
STEP06_PAIRED_B_DISTANCE_STATS_CSV = 'MV1_Step065_HLADR_PairedDistance_By_TcellSubtype.csv'
STEP06_NEAREST_ONLY_ASSIGNMENT_CSV = 'MV1_Step065C_NearestOnly_Tcell_To_Bstate_Assignment.csv'
STEP06_NEAREST_ONLY_COMPOSITION_CSV = 'MV1_Step065C_NearestOnly_Tcell_Composition_By_Bstate.csv'
STEP06_WRITE_INDIVIDUAL_PNG = True    # Save individual manuscript-review PNGs with Matplotlib/Seaborn.
STEP06_EXPORT_STATIC_FIGURES = {
    'Final_MV1_Step062B_TtoB_Tcell_Distance_to_Bstate.png',
    'Final_MV1_Step062D_TtoB_Functional_Gradients.png',
    'Final_MV1_Step065B_Tcell_Bstate_Median_Distance_Heatmap.png',
    'Final_MV1_Step065C_NearestOnly_Tcell_To_Bstate_Composition.png',
}
STEP06_DISPLAY_SAVED_PNG = not FINAL_SUPPRESS_INTERACTIVE_SHOW  # Display in cells only in explicit interactive mode.
STEP06_DISPLAY_INTERACTIVE_FIGURES = False  # Inspect only static PNGs during default execution.
STEP06_SAVE_CHECKPOINT = False       # Preserve the existing Step06 checkpoint in Figure-only mode.
STEP06_EXPORT_TTOB_ENRICHMENT_PANEL = False  # 2A: exclude from default manuscript outputs because the expected baseline is ambiguous.
STEP06_EXPORT_TTOB_ENGAGEMENT_PANEL = False  # 2C: exclude from default manuscript outputs due to B-cell abundance / overlapping hot zones.
STEP06_T_CELL_LABEL_ORDER = ['CD4_CTL', 'CD8T', 'Tfh', 'Th', 'Tpr', 'Treg']
STEP06_DLBCL_PATIENT_ORDER = ['DLBCL1', 'DLBCL2', 'DLBCL3', 'DLBCL4']
STEP06_B_CELL_LABEL_ORDER = ['B cell']
STEP06_2D_DISTANCE_BINS_UM = [0, 20, 50, 100, 150, 200, 250]
STEP06_B_STATE_POS = 'HLA-DR+ B cell'
STEP06_B_STATE_NEG = 'HLA-DR- B cell'
STEP06_B_STATE_COLORS = {STEP06_B_STATE_POS: '#d62728', STEP06_B_STATE_NEG: '#1f77b4'}
STEP06_T_SUBTYPE_COMPOSITION_COLORS = {
    'CD4_CTL': '#7E57C2',
    'CD8T': '#1E88E5',
    'Tfh': '#F9A825',
    'Th': '#43A047',
    'Tpr': '#E64A19',
    'Treg': '#6D4C41',
}
STEP06_T_SUBTYPE_GRADIENT_COLORS = {
    'CD4_CTL': '#1F77B4',
    'CD8T': '#FF7F0E',
    'Tfh': '#2CA02C',
    'Th': '#D62728',
    'Tpr': '#9467BD',
    'Treg': '#8C564B',
}

def _safe_step06_filename(label):
    """Sanitize sample/panel names for safe use in filenames."""
    safe = str(label).replace('+', 'plus').replace('/', '_').replace('\\', '_')
    safe = ''.join(ch if ch.isalnum() or ch in ('_', '-', '.') else '_' for ch in safe)
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_')[:120] or 'panel'

def _plain_step06_label(label):
    """Convert Plotly labels to plain strings suitable for Matplotlib axis labels."""
    text = str(label)
    replacements = {
        '<br>': '\n', '<b>': '', '</b>': '',
        '&lt;': '<', '&gt;': '>', '&amp;': '&'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'<[^>]+>', '', text)
    return text

def _step06_display_t_subset(label):
    """Return publication display text without changing stored subtype keys."""
    return {'CD4_CTL': 'CD4 CTL', 'CD8T': 'CD8 T'}.get(str(label), str(label))

def _step06_display_plot_label(label):
    """Apply subtype display text to the first line of an annotated tick label."""
    parts = _plain_step06_label(label).split('\n')
    if parts:
        parts[0] = _step06_display_t_subset(parts[0])
    return '\n'.join(parts)

def _ordered_existing_labels(observed_labels, preferred_order):
    """Return observed labels used in manuscript figures in the predefined order."""
    observed_set = {str(label).strip() for label in observed_labels if pd.notna(label)}
    return [label for label in preferred_order if label in observed_set]

def _get_step06_b_cell_labels(observed_labels):
    """Include only B cells as Step06 reference cells for HLA-DR stratification."""
    observed = [str(label).strip() for label in observed_labels if pd.notna(label)]
    ordered = _ordered_existing_labels(observed, STEP06_B_CELL_LABEL_ORDER)
    extra_b = sorted([
        label for label in observed
        if label not in ordered and (label.startswith('B ') or label.startswith('B-'))
    ])
    return ordered + extra_b

def _get_step06_t_cell_labels(observed_labels):
    """Restrict Step06 evaluated cells to the T-cell subtype whitelist."""
    labels = _ordered_existing_labels(observed_labels, STEP06_T_CELL_LABEL_ORDER)
    if not labels:
        raise ValueError(
            "[Step 06] No evaluable T-cell subset labels are available. "
            f"Allowed labels: {STEP06_T_CELL_LABEL_ORDER}"
        )
    return labels

def _step06_t_subset_from_plot_label(label):
    """Extract the actual T-cell subtype name from a plot label with a q-value annotation."""
    plain = _plain_step06_label(label)
    return plain.split('\n')[0].strip()

def _step06_plot_order_from_labels(plot_labels):
    """Order current plot labels that match the T-cell subset whitelist."""
    label_map = {
        _step06_t_subset_from_plot_label(label): _plain_step06_label(label)
        for label in plot_labels
    }
    return [label_map[label] for label in t_cell_labels if label in label_map]

def _filter_step06_t_subset_df(df, subset_col='T_Subset'):
    """Retain only T-cell subsets in figures even if old kernel variables contain non-T labels."""
    out = df.copy()
    out[subset_col] = out[subset_col].astype(str).str.strip()
    return out[out[subset_col].isin(t_cell_labels)].copy()

STEP06_FIGURE5_SIZES = {
    'Final_MV1_Step062B_TtoB_Tcell_Distance_to_Bstate.png': (7.20, 4.00),
    'Final_MV1_Step062D_TtoB_Functional_Gradients.png': (647.920 / 72, 254.000 / 72),
    'Final_MV1_Step062E_TtoB_FunctionalScores_NearVsFar_Boxplots.png': (13.20, 5.35),
    # Preserve the selected supplement panel aspect ratio directly in the generated canvas.
    'Final_MV1_Step065B_Tcell_Bstate_Median_Distance_Heatmap.png': (7.40, 4.55),
    'Final_MV1_Step065C_NearestOnly_Tcell_To_Bstate_Composition.png': (6.60, 4.54),
}


def _save_step06_matplotlib(fig, filename, label):
    """Save a Matplotlib Figure as a PNG/PPTX pair on the same canvas."""
    if filename not in STEP06_EXPORT_STATIC_FIGURES:
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
    print(f"  ✅ {label} PPTX saved: {saved_pptx_path}")
    if STEP06_DISPLAY_SAVED_PNG:
        IPython.display.display(IPython.display.Image(filename=saved_png_path))
    return saved_png_path

def _display_step06_figure(fig):
    """Wrap individual notebook figure displays without affecting script execution syntax."""
    if STEP06_DISPLAY_INTERACTIVE_FIGURES:
        IPython.display.display(fig)

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


def _step06_q_label(q):
    """Build q-value text for figure titles and axis labels."""
    if pd.isna(q) or not np.isfinite(q):
        return 'q=na'
    if q < 0.001:
        return 'q<0.001'
    return f'q={q:.3f}'


# 4. Initialize shared variables
save_dir = FIG_DIR  # Reuse the path defined in Step 00-3
os.makedirs(save_dir, exist_ok=True)

# 5. Restore unique_labels (defined in Step 05, in case it is absent from memory)
if 'unique_labels' not in dir():
    mask_dlbcl = adata.obs['Disease'].str.contains('DLBCL', na=False)
    unique_labels = sorted(adata.obs.loc[mask_dlbcl, 'Niche_Label'].unique().tolist())
    unique_labels = [l for l in unique_labels if l not in ['Unknown', 'na']]
    print(f"  📊 Subtype list restored: {unique_labels}")

print(f"📁 Output directory: {save_dir}")
print("✅ [Step 06-1] Initialization complete!")


def _step06_load_saved_table_if_complete(filename, required_columns):
    """Load a protected saved source, or warn and return None for fallback."""
    path = os.path.join(FINAL_TABLE_DIR, filename)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        warnings.warn(
            f'[Step 06] saved Figure source is missing or empty; using in-memory fallback: {path}',
            RuntimeWarning,
        )
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(
            f'[Step 06] saved Figure source could not be read; using in-memory fallback: '
            f'{path}: {exc}',
            RuntimeWarning,
        )
        return None
    if frame.empty:
        warnings.warn(
            f'[Step 06] saved Figure source has no rows; using in-memory fallback: {path}',
            RuntimeWarning,
        )
        return None
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        warnings.warn(
            f'[Step 06] saved Figure source lacks columns {missing}; '
            f'using in-memory fallback: {path}',
            RuntimeWarning,
        )
        return None
    return frame


def _step06_warn_incomplete_saved_source(filename, reason):
    """Report a semantic saved-source validation failure before fallback."""
    warnings.warn(
        f'[Step 06] saved Figure source is incomplete ({reason}); '
        f'using in-memory fallback: {os.path.join(FINAL_TABLE_DIR, filename)}',
        RuntimeWarning,
    )


def _step06_checkpoint_t_cell_frame(required_columns, figure_label):
    """Copy only protected checkpoint columns needed by one Figure fallback."""
    required = {'Disease', 'Niche_Label', *required_columns}
    missing = sorted(required - set(adata.obs.columns))
    if missing:
        raise RuntimeError(
            f'[{figure_label}] frozen Step06 checkpoint lacks Figure columns: {missing}'
        )
    disease = adata.obs['Disease'].astype(str)
    subtype = adata.obs['Niche_Label'].astype(str).str.strip()
    mask = disease.str.contains('DLBCL', na=False) & subtype.isin(STEP06_T_CELL_LABEL_ORDER)
    frame = adata.obs.loc[mask, ['Niche_Label', *required_columns]].copy()
    frame['T_Subset'] = frame.pop('Niche_Label').astype(str).str.strip()
    if frame.empty:
        raise RuntimeError(f'[{figure_label}] frozen checkpoint has no evaluable DLBCL T cells')
    return frame


def _step06_load_assignment_distance_source(figure_label):
    """Load the saved per-T-cell nearest-B assignment as a distance plot source."""
    required = [
        'Cell_Index', 'T_Subset',
        'Distance_to_Nearest_B_HLA_DR_pos',
        'Distance_to_Nearest_B_HLA_DR_neg',
    ]
    saved = _step06_load_saved_table_if_complete(
        STEP06_NEAREST_ONLY_ASSIGNMENT_CSV, required,
    )
    if saved is None:
        return None
    if saved['Cell_Index'].astype(str).duplicated().any():
        _step06_warn_incomplete_saved_source(
            STEP06_NEAREST_ONLY_ASSIGNMENT_CSV, 'duplicated Cell_Index values',
        )
        return None
    saved = saved.copy()
    saved['T_Subset'] = saved['T_Subset'].astype(str).str.strip()
    saved = saved[saved['T_Subset'].isin(STEP06_T_CELL_LABEL_ORDER)].copy()
    numeric_columns = [
        'Distance_to_Nearest_B_HLA_DR_pos',
        'Distance_to_Nearest_B_HLA_DR_neg',
    ]
    saved[numeric_columns] = saved[numeric_columns].apply(pd.to_numeric, errors='coerce')
    if saved.empty or saved[numeric_columns].isna().any().any():
        _step06_warn_incomplete_saved_source(
            STEP06_NEAREST_ONLY_ASSIGNMENT_CSV,
            f'missing/non-numeric {figure_label} distance values',
        )
        return None
    observed = set(saved['T_Subset'])
    missing_subtypes = sorted(set(STEP06_T_CELL_LABEL_ORDER) - observed)
    if missing_subtypes:
        _step06_warn_incomplete_saved_source(
            STEP06_NEAREST_ONLY_ASSIGNMENT_CSV,
            f'missing T-cell subtypes {missing_subtypes}',
        )
        return None
    return saved.rename(columns={
        'Distance_to_Nearest_B_HLA_DR_pos': 'Dist_to_Pos',
        'Distance_to_Nearest_B_HLA_DR_neg': 'Dist_to_Neg',
    })


def _step06_checkpoint_distance_source(figure_label):
    """Build the standard T-subtype/positive/negative distance frame in memory."""
    frame = _step06_checkpoint_t_cell_frame(
        ['Dist_to_Pos', 'Dist_to_Neg'], figure_label,
    )
    frame[['Dist_to_Pos', 'Dist_to_Neg']] = frame[
        ['Dist_to_Pos', 'Dist_to_Neg']
    ].apply(pd.to_numeric, errors='coerce')
    frame[['Dist_to_Pos', 'Dist_to_Neg']] = frame[
        ['Dist_to_Pos', 'Dist_to_Neg']
    ].replace([np.inf, -np.inf], np.nan)
    if frame[['Dist_to_Pos', 'Dist_to_Neg']].dropna(how='all').empty:
        raise RuntimeError(f'[{figure_label}] frozen checkpoint has no finite distance values')
    return frame


def _step06_complete_pair_distance_frame(distance_source, figure_label):
    """Retain finite, non-tied same-cell DR+/- distances when both are <=50 µm."""
    paired = distance_source.copy()
    paired[['Dist_to_Pos', 'Dist_to_Neg']] = paired[
        ['Dist_to_Pos', 'Dist_to_Neg']
    ].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    paired = paired[
        paired['Dist_to_Pos'].le(STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM)
        & paired['Dist_to_Neg'].le(STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM)
        & np.isfinite(paired['Dist_to_Pos'].to_numpy(dtype=float))
        & np.isfinite(paired['Dist_to_Neg'].to_numpy(dtype=float))
    ].copy()
    paired = paired[paired['Dist_to_Pos'].ne(paired['Dist_to_Neg'])].copy()
    if paired.empty:
        raise RuntimeError(
            f'[{figure_label}] no non-tied T cells have finite HLA-DR+ and '
            f'HLA-DR- B-cell distances within '
            f'{STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM:g} µm'
        )
    return paired


def _step06_paired_proximity_stats(distance_source, figure_label='Step 06-01'):
    """Calculate within-cell DR+/- Wilcoxon tests and six-subtype BH-FDR."""
    paired = _step06_complete_pair_distance_frame(distance_source, figure_label)
    rows = []
    for subtype in STEP06_T_CELL_LABEL_ORDER:
        subset = paired[paired['T_Subset'].eq(subtype)]
        d_pos = subset['Dist_to_Pos'].to_numpy(dtype=float)
        d_neg = subset['Dist_to_Neg'].to_numpy(dtype=float)
        delta = d_pos - d_neg
        if len(subset) > 5 and np.any(delta != 0):
            try:
                p_value = wilcoxon(
                    d_pos, d_neg, alternative='two-sided', zero_method='wilcox',
                ).pvalue
            except ValueError:
                p_value = np.nan
        elif len(subset) > 5:
            p_value = 1.0
        else:
            p_value = np.nan
        rows.append({
            'T_Subset': subtype,
            'Paired_T_cells_within_50um_n': int(len(subset)),
            'Median_distance_to_HLADRpos_B_um': float(np.median(d_pos)) if len(d_pos) else np.nan,
            'Median_distance_to_HLADRneg_B_um': float(np.median(d_neg)) if len(d_neg) else np.nan,
            'Median_delta_HLADRpos_minus_HLADRneg_um': float(np.median(delta)) if len(delta) else np.nan,
            'Proximity_p (paired Wilcoxon)': p_value,
        })
    stats = pd.DataFrame(rows)
    stats['Proximity_q (BH-FDR)'] = _bh_fdr(
        stats['Proximity_p (paired Wilcoxon)'].to_numpy(dtype=float)
    )
    return paired, stats


def _render_step06_saved_distance_panel():
    """Render Step06-01 from saved per-cell distances with protected fallback."""
    filename = 'Final_MV1_Step062B_TtoB_Tcell_Distance_to_Bstate.png'
    if not _final_figure_stem_registered(filename):
        raise RuntimeError(f'Step06 canonical Figure stem is not registered: {filename}')

    distance_source = _step06_load_assignment_distance_source('Step 06-01')
    if distance_source is None:
        distance_source = _step06_checkpoint_distance_source('Step 06-01')
    paired_source, paired_stats = _step06_paired_proximity_stats(
        distance_source, 'Step 06-01',
    )
    labels = [
        label for label in STEP06_T_CELL_LABEL_ORDER
        if label in set(paired_source['T_Subset'])
    ]
    distance_rows = []
    for label in labels:
        subset = paired_source.loc[
            paired_source['T_Subset'].eq(label), ['Dist_to_Pos', 'Dist_to_Neg'],
        ]
        for column, state in (
            ('Dist_to_Pos', 'HLA-DR+ B cells'),
            ('Dist_to_Neg', 'HLA-DR- B cells'),
        ):
            values = subset[column]
            distance_rows.append(pd.DataFrame({
                'T_Subset': label,
                'B_State': state,
                'Distance_um': values.to_numpy(dtype=float),
            }))
    plot_frame = pd.concat(distance_rows, ignore_index=True) if distance_rows else pd.DataFrame()
    if plot_frame.empty:
        raise RuntimeError('[Step 06-01] frozen Step06 distances contain no plottable values')

    q_map = paired_stats.set_index('T_Subset')['Proximity_q (BH-FDR)'].to_dict()
    display_order = [
        f'{_step06_display_t_subset(label)}\n({_step06_q_label(float(q_map.get(label, np.nan)))})'
        for label in labels
    ]
    display_map = dict(zip(labels, display_order))
    plot_frame['T_Subset_Display'] = plot_frame['T_Subset'].map(display_map)
    fig, axis = plt.subplots(figsize=STEP06_FIGURE5_SIZES[filename])
    sns.boxplot(
        data=plot_frame, x='T_Subset_Display', y='Distance_um', hue='B_State',
        order=display_order, showfliers=False, width=0.58, linewidth=1.25,
        palette={'HLA-DR+ B cells': '#d62728', 'HLA-DR- B cells': '#1f77b4'},
        boxprops={'alpha': 0.94},
        whiskerprops={'linewidth': 1.25},
        capprops={'linewidth': 1.25},
        medianprops={'linewidth': 1.65, 'color': '#333333'},
        ax=axis,
    )
    axis.set_xlabel('')
    axis.set_ylabel(
        'Distance to nearest B cell (µm)',
        fontsize=13.2, fontweight='bold', labelpad=14,
    )
    axis.set_ylim(0, STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM)
    axis.tick_params(axis='x', rotation=0, labelsize=10.7, pad=7, width=1.1, length=4)
    axis.tick_params(axis='y', labelsize=11.2, width=1.1, length=4)
    plt.setp(axis.get_xticklabels(), fontweight='bold', ha='center')
    for tick_label in axis.get_xticklabels():
        tick_label.set_multialignment('center')
    plt.setp(axis.get_yticklabels(), fontweight='bold')
    legend_handles, legend_labels = axis.get_legend_handles_labels()
    axis.get_legend().remove()
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center', bbox_to_anchor=(0.555, 0.018), ncol=2,
        frameon=False, fontsize=10.6,
        handlelength=1.55, handleheight=0.90, columnspacing=1.45,
    )
    axis.set_axisbelow(True)
    axis.grid(axis='y', linestyle='--', color='#A7A7A7', linewidth=0.75, alpha=0.45)
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.265, top=0.980)
    return _save_step06_matplotlib(
        fig, filename, 'Step06-01 saved T-to-B distance panel',
    )


def _step06_paired_distance_source():
    """Return same-cell positive/negative B distances from saved source or checkpoint."""
    source = _step06_load_assignment_distance_source('Step 06-03')
    if source is None:
        derived_columns = {
            'Distance_to_Nearest_B',
            'Distance_to_Nearest_B_HLA_DR_pos',
            'Distance_to_Nearest_B_HLA_DR_neg',
        }
        if derived_columns.issubset(adata.obs.columns):
            source = _step06_checkpoint_t_cell_frame(
                sorted(derived_columns), 'Step 06-03',
            ).rename(columns={
                'Distance_to_Nearest_B_HLA_DR_pos': 'Dist_to_Pos',
                'Distance_to_Nearest_B_HLA_DR_neg': 'Dist_to_Neg',
            })
        else:
            source = _step06_checkpoint_distance_source('Step 06-03')

    source = _step06_complete_pair_distance_frame(source, 'Step 06-03')
    if 'Distance_to_Nearest_B' not in source.columns:
        source['Distance_to_Nearest_B'] = source[['Dist_to_Pos', 'Dist_to_Neg']].min(
            axis=1, skipna=True,
        )
    else:
        source['Distance_to_Nearest_B'] = pd.to_numeric(
            source['Distance_to_Nearest_B'], errors='coerce',
        )
    return source


def _render_step06_saved_paired_distance_heatmap():
    """Render Step06-03 as the protected full-style 3-by-subtype heatmap."""
    filename = 'Final_MV1_Step065B_Tcell_Bstate_Median_Distance_Heatmap.png'
    if not _final_figure_stem_registered(filename):
        raise RuntimeError(f'Step06 canonical Figure stem is not registered: {filename}')

    paired = _step06_paired_distance_source()
    order = [
        label for label in STEP06_T_CELL_LABEL_ORDER
        if label in set(paired['T_Subset'].astype(str))
    ]
    heatmap = pd.DataFrame({
        'B cell (Any)': paired.groupby('T_Subset', observed=True)[
            'Distance_to_Nearest_B'
        ].median(),
        'HLA-DR+ B cell': paired.groupby('T_Subset', observed=True)['Dist_to_Pos'].median(),
        'HLA-DR- B cell': paired.groupby('T_Subset', observed=True)['Dist_to_Neg'].median(),
    }).reindex(order).T
    heatmap = heatmap.rename(index={
        'B cell (Any)': 'B cells (Any)',
        'HLA-DR+ B cell': 'HLA-DR+ B cells',
        'HLA-DR- B cell': 'HLA-DR- B cells',
    }).rename(columns={label: _step06_display_t_subset(label) for label in order})
    finite = heatmap.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError('[Step 06-03] paired-distance source has no finite medians')
    if float(finite.min()) == float(finite.max()):
        vmin = max(0.0, float(finite.min()) - 1.0)
        vmax = float(finite.max()) + 1.0
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    fig, axis = plt.subplots(figsize=STEP06_FIGURE5_SIZES[filename])
    colorbar_axis = fig.add_axes([0.900, 0.405, 0.010, 0.245])
    sns.heatmap(
        heatmap, annot=True, fmt='.2f',
        cmap=mcolors.LinearSegmentedColormap.from_list(
            'step065_distance', ['#551A8B', '#B65F2A', '#FFD700'],
        ),
        vmin=vmin, vmax=vmax, linewidths=0.6,
        linecolor='white', cbar_ax=colorbar_axis,
        annot_kws={'fontsize': 11.2, 'fontweight': 'bold'}, ax=axis,
    )
    axis.set_xlabel('')
    axis.set_ylabel('')
    plt.setp(
        axis.get_xticklabels(), rotation=30, ha='right',
        fontsize=10.8, fontweight='bold',
    )
    plt.setp(
        axis.get_yticklabels(), rotation=0, fontsize=11.4, fontweight='bold',
    )
    colorbar_axis.tick_params(labelsize=9.5, length=2)
    colorbar_axis.set_title(
        'Median distance\n(µm)', fontsize=9.8,
        fontweight='bold', pad=7, loc='center', linespacing=0.95,
    )
    fig.subplots_adjust(left=0.205, right=0.875, bottom=0.205, top=0.975)
    return _save_step06_matplotlib(
        fig, filename, 'Step06-03 saved paired-distance heatmap',
    )


def _render_step06_saved_functional_gradients():
    """Render Step06-02 as descriptive distance-bin means from the frozen checkpoint."""
    filename = 'Final_MV1_Step062D_TtoB_Functional_Gradients.png'
    if not _final_figure_stem_registered(filename):
        raise RuntimeError(f'Step06 canonical Figure stem is not registered: {filename}')

    required_obs = {
        'Disease', 'Niche_Label', 'Dist_to_Pos', 'Dist_to_Neg',
        'Exhaustion_Score', 'Activation_Score',
    }
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    if missing_obs:
        raise RuntimeError(
            f'[Step 06-02] frozen Step06 checkpoint lacks Figure columns: {missing_obs}'
        )

    disease = adata.obs['Disease'].astype(str)
    subtype = adata.obs['Niche_Label'].astype(str).str.strip()
    figure_mask = disease.str.contains('DLBCL', na=False) & subtype.isin(STEP06_T_CELL_LABEL_ORDER)
    plot_frame = adata.obs.loc[
        figure_mask,
        ['Niche_Label', 'Dist_to_Pos', 'Dist_to_Neg', 'Exhaustion_Score', 'Activation_Score'],
    ].copy()
    plot_frame['T_Subset'] = plot_frame.pop('Niche_Label').astype(str).str.strip()
    numeric_columns = ['Dist_to_Pos', 'Dist_to_Neg', 'Exhaustion_Score', 'Activation_Score']
    plot_frame[numeric_columns] = plot_frame[numeric_columns].apply(pd.to_numeric, errors='coerce')
    plot_frame[numeric_columns] = plot_frame[numeric_columns].replace([np.inf, -np.inf], np.nan)

    plot_order = [
        label for label in STEP06_T_CELL_LABEL_ORDER
        if label in set(plot_frame['T_Subset'])
    ]
    if not plot_order:
        raise RuntimeError('[Step 06-02] frozen Step06 checkpoint has no evaluable DLBCL T cells')

    interact_cutoff = float(
        adata.uns.get('INTERACT_CUTOFF', STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM)
    )
    if not np.isfinite(interact_cutoff) or interact_cutoff <= 0:
        raise RuntimeError(f'[Step 06-02] invalid saved INTERACT_CUTOFF: {interact_cutoff!r}')
    distance_bins = sorted(set(STEP06_2D_DISTANCE_BINS_UM + [interact_cutoff]))
    bin_labels = [
        f'{distance_bins[index]:g}-{distance_bins[index + 1]:g}µm'
        for index in range(len(distance_bins) - 1)
    ]
    bin_midpoints = {
        label: (distance_bins[index] + distance_bins[index + 1]) / 2.0
        for index, label in enumerate(bin_labels)
    }
    plot_frame['Bin_Pos'] = pd.cut(
        plot_frame['Dist_to_Pos'], bins=distance_bins, labels=bin_labels
    )
    plot_frame['Bin_Neg'] = pd.cut(
        plot_frame['Dist_to_Neg'], bins=distance_bins, labels=bin_labels
    )

    panel_specs = [
        ('Exhaustion_Score', 'Exhaustion Score\n(PD1, TIM3, LAG3, CTLA4)'),
        ('Activation_Score', 'Activation Score\n(CD69, ICOS, CD134)'),
    ]
    fig, axes = plt.subplots(
        1, len(panel_specs), figsize=STEP06_FIGURE5_SIZES[filename], squeeze=False
    )
    axes = axes[0]
    for panel_index, (score_column, panel_title) in enumerate(panel_specs):
        axis = axes[panel_index]
        axis.axvspan(0, min(20, interact_cutoff), color='#FFE082', alpha=0.30, zorder=0)
        if interact_cutoff > 20:
            axis.axvspan(20, min(50, interact_cutoff), color='#AED6F1', alpha=0.24, zorder=0)
        axis.axvline(20, color='#6C757D', linestyle='--', linewidth=1.0, alpha=0.85, zorder=1)
        axis.axvline(
            interact_cutoff, color='#8A6D3B', linestyle=':', linewidth=1.0, alpha=0.85, zorder=1
        )

        for subtype_label in plot_order:
            subtype_frame = plot_frame[plot_frame['T_Subset'].eq(subtype_label)]
            color = STEP06_T_SUBTYPE_GRADIENT_COLORS[subtype_label]
            for bin_column, linestyle, linewidth, marker_size in (
                ('Bin_Pos', '-', 2.8, 4.5),
                ('Bin_Neg', ':', 2.1, 3.8),
            ):
                grouped = (
                    subtype_frame.groupby(bin_column, observed=True)[score_column]
                    .mean()
                    .reindex(bin_labels)
                    .dropna()
                )
                if grouped.empty:
                    continue
                axis.plot(
                    [bin_midpoints[label] for label in grouped.index],
                    grouped.to_numpy(dtype=float),
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    marker='o',
                    markersize=marker_size,
                    alpha=0.96 if bin_column == 'Bin_Pos' else 0.88,
                    label=(
                        _step06_display_t_subset(subtype_label)
                        if panel_index == 0 and bin_column == 'Bin_Pos'
                        else None
                    ),
                )

        axis.set_xlim(0, max(distance_bins))
        axis.set_xticks([
            value for value in STEP06_2D_DISTANCE_BINS_UM
            if value <= max(distance_bins)
        ])
        axis.set_title(panel_title, fontsize=13.2, fontweight='bold', pad=6)
        axis.set_xlabel('Distance to nearest B cell (µm)', fontsize=10.5, fontweight='bold')
        axis.set_ylabel('Mean T-cell score (z-score)', fontsize=10.5, fontweight='bold')
        axis.tick_params(axis='x', labelsize=10.5)
        axis.tick_params(axis='y', labelsize=10.5)
        axis.grid(True, axis='y', linestyle='--', alpha=0.28)
        axis.grid(True, axis='x', linestyle=':', alpha=0.16)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    subtype_handles = [
        mlines.Line2D(
            [], [], color=STEP06_T_SUBTYPE_GRADIENT_COLORS[label],
            linewidth=2.5, marker='o', markersize=4.0,
            label=_step06_display_t_subset(label),
        )
        for label in plot_order
    ]
    state_handles = [
        mlines.Line2D([], [], color='black', linewidth=2.2, linestyle='-',
                      label='nearest HLA-DR+ B cell'),
        mlines.Line2D([], [], color='black', linewidth=2.2, linestyle=':',
                      label='nearest HLA-DR− B cell'),
    ]
    legend = fig.legend(
        subtype_handles + state_handles,
        [handle.get_label() for handle in subtype_handles + state_handles],
        loc='lower center', bbox_to_anchor=(0.5, 0.012), ncol=4,
        fontsize=8.2, frameon=False,
        title='Points indicate mean scores within distance bins',
        title_fontsize=8.4,
    )
    legend.get_title().set_fontweight('normal')
    fig.tight_layout(rect=[0, 0.21, 1.0, 0.98])
    return _save_step06_matplotlib(
        fig, filename, 'Step06-02 saved functional-score gradients',
    )


def _step06_checkpoint_functional_source(figure_label):
    """Return protected DLBCL T-cell distances, scores, and tissue labels."""
    columns = [
        'Run_Tissue_name', 'Dist_to_Pos', 'Dist_to_Neg',
        'Exhaustion_Score', 'Activation_Score',
    ]
    frame = _step06_checkpoint_t_cell_frame(columns, figure_label)
    numeric = ['Dist_to_Pos', 'Dist_to_Neg', 'Exhaustion_Score', 'Activation_Score']
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='coerce')
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    if frame[numeric].dropna(how='all').empty:
        raise RuntimeError(f'[{figure_label}] frozen checkpoint has no functional-score rows')
    return frame


def _render_step06_saved_near_far_functional_scores():
    """Render Step06-02b from protected checkpoint rows without persisting statistics."""
    filename = 'Final_MV1_Step062E_TtoB_FunctionalScores_NearVsFar_Boxplots.png'
    if not _final_figure_stem_registered(filename):
        raise RuntimeError(f'Step06 Figure stem is not registered: {filename}')

    source = _step06_checkpoint_functional_source('Step 06-02b')
    source['Nearest_B_Distance_um'] = source[['Dist_to_Pos', 'Dist_to_Neg']].min(
        axis=1, skipna=True,
    )
    source = source.dropna(subset=['Run_Tissue_name', 'Nearest_B_Distance_um']).copy()
    distance_order = ['0–<20 µm', '20–<50 µm', '≥50 µm']
    source['Distance_Group'] = pd.cut(
        source['Nearest_B_Distance_um'],
        bins=[0.0, 20.0, 50.0, np.inf],
        labels=distance_order,
        right=False,
        include_lowest=True,
    )
    source = source.dropna(subset=['Distance_Group']).copy()
    source['Distance_Group'] = pd.Categorical(
        source['Distance_Group'], categories=distance_order, ordered=True,
    )
    source['T_Subset_Display'] = source['T_Subset'].map(_step06_display_t_subset)
    plot_order = [
        subtype for subtype in STEP06_T_CELL_LABEL_ORDER
        if subtype in set(source['T_Subset'])
    ]
    score_columns = ['Exhaustion_Score', 'Activation_Score']

    stats_rows = []
    tissue_rows = []
    for score in score_columns:
        for subtype in plot_order:
            subset = source[source['T_Subset'].eq(subtype)].dropna(subset=[score])
            tissue_summary = (
                subset.groupby(['Run_Tissue_name', 'Distance_Group'], observed=True)[score]
                .agg(Median='median', Cell_n='size')
                .reset_index()
            )
            tissue_summary = tissue_summary[tissue_summary['Cell_n'].ge(5)]
            for row in tissue_summary.itertuples(index=False):
                tissue_rows.append({
                    'Score': score,
                    'T_Subset': subtype,
                    'T_Subset_Display': _step06_display_t_subset(subtype),
                    'Tissue': row.Run_Tissue_name,
                    'Distance_Group': str(row.Distance_Group),
                    'Tissue_Median': float(row.Median),
                    'Cell_n': int(row.Cell_n),
                })
            pivot = tissue_summary.pivot(
                index='Run_Tissue_name', columns='Distance_Group', values='Median',
            )
            for group_a, group_b in combinations(distance_order, 2):
                paired = (
                    pivot[[group_a, group_b]].dropna()
                    if group_a in pivot.columns and group_b in pivot.columns
                    else pd.DataFrame()
                )
                n_tissues = int(len(paired))
                if n_tissues >= 3 and np.any(
                    np.abs(
                        paired[group_a].to_numpy(dtype=float)
                        - paired[group_b].to_numpy(dtype=float)
                    ) > 0
                ):
                    try:
                        p_value = wilcoxon(
                            paired[group_a], paired[group_b], alternative='two-sided',
                        ).pvalue
                    except ValueError:
                        p_value = np.nan
                else:
                    p_value = np.nan
                stats_rows.append({
                    'Score': score,
                    'T_Subset': subtype,
                    'Group_A': group_a,
                    'Group_B': group_b,
                    'Paired_tissues_n': n_tissues,
                    'p': p_value,
                })

    stats = pd.DataFrame(stats_rows)
    stats['q'] = _bh_fdr(stats['p'].to_numpy(dtype=float))
    stats['sig'] = stats['q'].map(_sig_from_q)
    tissue_plot = pd.DataFrame(tissue_rows)
    if tissue_plot.empty:
        raise RuntimeError('[Step 06-02b] no tissue medians meet the five-cell threshold')

    fig, axes = plt.subplots(
        1, len(score_columns),
        figsize=STEP06_FIGURE5_SIZES[filename],
        squeeze=False,
    )
    palette = {
        '0–<20 µm': '#2F6B9A',
        '20–<50 µm': '#6FA7C7',
        '≥50 µm': '#A7A7A7',
    }
    display_order = [_step06_display_t_subset(value) for value in plot_order]
    group_offsets = dict(zip(distance_order, [-0.24, 0.0, 0.24]))
    legend_handles, legend_labels = [], []
    for panel_index, score in enumerate(score_columns):
        axis = axes[0, panel_index]
        score_tissues = tissue_plot[tissue_plot['Score'].eq(score)]
        score_cells = source.dropna(subset=[score]).copy()
        sns.boxplot(
            data=score_cells,
            x='T_Subset_Display', y=score, hue='Distance_Group',
            order=display_order, hue_order=distance_order,
            palette=palette, showfliers=False, width=0.68,
            linewidth=0.85, saturation=0.82, ax=axis,
        )
        panel_handles, panel_labels = axis.get_legend_handles_labels()
        if not legend_handles:
            legend_handles, legend_labels = panel_handles[:3], panel_labels[:3]
        if axis.get_legend() is not None:
            axis.get_legend().remove()
        x_lookup = {label: index for index, label in enumerate(display_order)}
        for (subtype_display, tissue), line_frame in score_tissues.groupby(
            ['T_Subset_Display', 'Tissue'], observed=True,
        ):
            x_center = x_lookup.get(subtype_display)
            if x_center is None:
                continue
            line_frame = line_frame.copy()
            line_frame['Distance_Group'] = pd.Categorical(
                line_frame['Distance_Group'], categories=distance_order, ordered=True,
            )
            line_frame = line_frame.sort_values('Distance_Group')
            tissue_order = sorted(
                score_tissues.loc[
                    score_tissues['T_Subset_Display'].eq(subtype_display), 'Tissue',
                ].astype(str).unique()
            )
            jitter_lookup = dict(zip(
                tissue_order,
                np.linspace(-0.028, 0.028, max(len(tissue_order), 1)),
            ))
            x_values = [
                x_center + group_offsets[str(group)] + jitter_lookup.get(str(tissue), 0.0)
                for group in line_frame['Distance_Group']
            ]
            y_values = line_frame['Tissue_Median'].to_numpy(dtype=float)
            if len(line_frame) >= 2:
                axis.plot(
                    x_values, y_values,
                    color='#5F5F5F', linewidth=0.68, alpha=0.34, zorder=4,
                )
            for x_point, y_point, group in zip(
                x_values, y_values, line_frame['Distance_Group'],
            ):
                axis.scatter(
                    [x_point], [y_point], s=16, c=palette[str(group)],
                    edgecolors='white', linewidths=0.40, alpha=0.95, zorder=5,
                )
        axis.set_title(score.replace('_', ' '), fontsize=14.0, fontweight='bold', pad=8)
        axis.set_xlabel('T-cell subtype', fontsize=11.4, fontweight='bold')
        axis.set_ylabel('T-cell score (cell-level z-score)', fontsize=11.4, fontweight='bold')
        axis.tick_params(axis='x', rotation=23, labelsize=9.5)
        axis.tick_params(axis='y', labelsize=9.7)
        plt.setp(axis.get_xticklabels(), ha='right', fontweight='bold')
        axis.grid(axis='y', linestyle='--', alpha=0.28)

        panel_stats = stats[stats['Score'].eq(score)]
        y_min, y_max = axis.get_ylim()
        y_span = max(y_max - y_min, 0.2)
        axis.set_ylim(y_min, y_max + 0.46 * y_span)
        for x_index, subtype in enumerate(plot_order):
            subtype_stats = panel_stats[panel_stats['T_Subset'].eq(subtype)]
            for pair_index, (group_a, group_b) in enumerate(combinations(distance_order, 2)):
                row = subtype_stats[
                    subtype_stats['Group_A'].eq(group_a)
                    & subtype_stats['Group_B'].eq(group_b)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                q_value = row['q']
                q_text = 'q=n/a' if not np.isfinite(q_value) else (
                    'q<0.001' if q_value < 0.001 else f'q={q_value:.3f}'
                )
                sig_text = 'n/a' if not np.isfinite(q_value) else str(row['sig'])
                bracket_y = y_max + (0.045 + 0.115 * pair_index) * y_span
                bracket_h = 0.018 * y_span
                x_a = x_index + group_offsets[group_a]
                x_b = x_index + group_offsets[group_b]
                axis.plot(
                    [x_a, x_a, x_b, x_b],
                    [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                    color='black', linewidth=0.72, clip_on=False,
                )
                axis.text(
                    (x_a + x_b) / 2, bracket_y + 1.15 * bracket_h,
                    f"{sig_text} {q_text}; n={int(row['Paired_tissues_n'])}",
                    ha='center', va='bottom', fontsize=6.1,
                    fontweight='bold', clip_on=False,
                )

    fig.legend(
        legend_handles, legend_labels,
        loc='lower center', bbox_to_anchor=(0.5, 0.025),
        ncol=3, frameon=False, fontsize=9.8,
        title='Distance to nearest B cell', title_fontsize=10.3,
    )
    fig.suptitle(
        'Functional scores by distance to the nearest B cell',
        fontsize=15.0, fontweight='bold', y=0.985,
    )
    fig.text(
        0.5, 0.905,
        'Boxes = individual T cells; dots/lines = tissue medians; '
        'paired Wilcoxon tests with BH-FDR',
        ha='center', va='center', fontsize=9.2,
    )
    fig.subplots_adjust(left=0.060, right=0.985, bottom=0.225, top=0.76, wspace=0.16)
    return _save_step06_matplotlib(
        fig, filename, 'Step06-02b tissue-level functional-score near-versus-far boxplots',
    )


def _step06_nearest_only_counts_from_distances(frame, figure_label):
    """Build tie-excluded 2-by-6 nearest-state counts from same-cell distances."""
    required = {'T_Subset', 'Dist_to_Pos', 'Dist_to_Neg'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f'[{figure_label}] nearest-only source lacks columns: {missing}')
    rows = frame.copy()
    rows['T_Subset'] = rows['T_Subset'].astype(str).str.strip()
    rows[['Dist_to_Pos', 'Dist_to_Neg']] = rows[
        ['Dist_to_Pos', 'Dist_to_Neg']
    ].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    pos = rows['Dist_to_Pos'].to_numpy(dtype=float)
    neg = rows['Dist_to_Neg'].to_numpy(dtype=float)
    pos_finite = np.isfinite(pos)
    neg_finite = np.isfinite(neg)
    nearest = np.fmin(pos, neg)
    nearest[~(pos_finite | neg_finite)] = np.nan
    within = np.isfinite(nearest) & (
        nearest <= STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM
    )
    exact_tie = within & pos_finite & neg_finite & (pos == neg)
    rows['Nearest_B_Type'] = np.select(
        [
            within & ~exact_tie & pos_finite & (~neg_finite | (pos < neg)),
            within & ~exact_tie & neg_finite & (~pos_finite | (neg < pos)),
        ],
        ['HLA-DR+ B Cell', 'HLA-DR- B Cell'],
        default='Excluded',
    )
    state_order = ['HLA-DR+ B Cell', 'HLA-DR- B Cell']
    rows = rows[
        rows['T_Subset'].isin(STEP06_T_CELL_LABEL_ORDER)
        & rows['Nearest_B_Type'].isin(state_order)
    ]
    counts = pd.crosstab(rows['Nearest_B_Type'], rows['T_Subset']).reindex(
        index=state_order, columns=STEP06_T_CELL_LABEL_ORDER, fill_value=0,
    )
    if counts.to_numpy(dtype=float).sum() <= 0 or (counts.sum(axis=1) <= 0).any():
        raise RuntimeError(f'[{figure_label}] nearest-only source has no valid state counts')
    return counts.astype(int), int(exact_tie.sum())


def _step06_saved_nearest_only_counts():
    """Load and validate the protected 2-state-by-6-subtype composition summary."""
    required = [
        'Nearest_B_State', 'T_Subset',
        'Assigned_T_cells', 'Assigned_Tcell_pct_within_B_state',
    ]
    saved = _step06_load_saved_table_if_complete(
        STEP06_NEAREST_ONLY_COMPOSITION_CSV, required,
    )
    if saved is None:
        return None
    saved = saved.copy()
    saved['Nearest_B_State'] = saved['Nearest_B_State'].astype(str).str.strip()
    saved['T_Subset'] = saved['T_Subset'].astype(str).str.strip()
    keys = ['Nearest_B_State', 'T_Subset']
    state_order = ['HLA-DR+ B Cell', 'HLA-DR- B Cell']
    expected_keys = {
        (state, subtype)
        for state in state_order for subtype in STEP06_T_CELL_LABEL_ORDER
    }
    observed_keys = set(map(tuple, saved[keys].to_numpy()))
    if saved.duplicated(keys).any() or observed_keys != expected_keys:
        _step06_warn_incomplete_saved_source(
            STEP06_NEAREST_ONLY_COMPOSITION_CSV,
            'state-by-subtype keys are not the exact 2-by-6 Cartesian set',
        )
        return None
    saved['Assigned_T_cells'] = pd.to_numeric(saved['Assigned_T_cells'], errors='coerce')
    saved['Assigned_Tcell_pct_within_B_state'] = pd.to_numeric(
        saved['Assigned_Tcell_pct_within_B_state'], errors='coerce',
    )
    if (
        saved[['Assigned_T_cells', 'Assigned_Tcell_pct_within_B_state']].isna().any().any()
        or (saved['Assigned_T_cells'] < 0).any()
        or not np.allclose(saved['Assigned_T_cells'], np.round(saved['Assigned_T_cells']))
    ):
        _step06_warn_incomplete_saved_source(
            STEP06_NEAREST_ONLY_COMPOSITION_CSV, 'invalid count or percentage values',
        )
        return None
    counts = saved.pivot(
        index='Nearest_B_State', columns='T_Subset', values='Assigned_T_cells',
    ).reindex(index=state_order, columns=STEP06_T_CELL_LABEL_ORDER)
    saved_pct = saved.pivot(
        index='Nearest_B_State',
        columns='T_Subset',
        values='Assigned_Tcell_pct_within_B_state',
    ).reindex(index=state_order, columns=STEP06_T_CELL_LABEL_ORDER)
    derived_pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    if (
        counts.isna().any().any()
        or (counts.sum(axis=1) <= 0).any()
        or not np.allclose(saved_pct, derived_pct, rtol=1e-7, atol=1e-7)
    ):
        _step06_warn_incomplete_saved_source(
            STEP06_NEAREST_ONLY_COMPOSITION_CSV,
            'saved percentages do not match the state-wise count denominators',
        )
        return None
    return counts.astype(int)


def _step06_assignment_nearest_only_counts():
    """Rebuild tie-excluded 06-04 counts from protected per-cell distances."""
    saved = _step06_load_assignment_distance_source('Step 06-04')
    if saved is None:
        return None
    try:
        return _step06_nearest_only_counts_from_distances(saved, 'Step 06-04')
    except RuntimeError as exc:
        _step06_warn_incomplete_saved_source(
            STEP06_NEAREST_ONLY_ASSIGNMENT_CSV, str(exc),
        )
        return None


def _step06_checkpoint_nearest_only_counts():
    """Reconstruct tie-excluded counts from protected checkpoint distances."""
    frame = _step06_checkpoint_distance_source('Step 06-04')
    return _step06_nearest_only_counts_from_distances(frame, 'Step 06-04')


def _step06_patient_nearest_only_distance_source():
    """Attach frozen checkpoint tissue/patient labels to the protected assignment source."""
    source = _step06_load_assignment_distance_source('Step 06-04 patient composition')
    if source is None:
        source = _step06_checkpoint_distance_source('Step 06-04 patient composition')
        source = source.copy()
        source.insert(0, 'Cell_Index', source.index.astype(str))
    else:
        source = source.copy()

    source['Cell_Index'] = source['Cell_Index'].astype(str)
    if source['Cell_Index'].duplicated().any():
        raise RuntimeError('[Step 06-04] patient source contains duplicated Cell_Index values')
    if not adata.obs.index.is_unique:
        raise RuntimeError('[Step 06-04] frozen checkpoint obs index is not unique')
    if 'Run_Tissue_name' not in adata.obs.columns:
        raise RuntimeError('[Step 06-04] frozen checkpoint lacks Run_Tissue_name')

    source_ids = pd.Index(source['Cell_Index'].to_numpy(dtype=str))
    run_tissue = adata.obs['Run_Tissue_name'].reindex(source_ids)
    if len(run_tissue) != len(source) or run_tissue.isna().any():
        missing_n = int(run_tissue.isna().sum())
        raise RuntimeError(
            '[Step 06-04] assignment Cell_Index values do not map 1:1 to the '
            f'frozen checkpoint (missing={missing_n:,})'
        )
    source['Run_Tissue_name'] = run_tissue.astype(str).to_numpy()
    source['Patient_Code'] = source['Run_Tissue_name'].str.extract(
        r'^(DLBCL\d+)_\d+$', expand=False,
    )
    if source['Patient_Code'].isna().any():
        invalid = sorted(source.loc[source['Patient_Code'].isna(), 'Run_Tissue_name'].unique())
        raise RuntimeError(
            '[Step 06-04] Run_Tissue_name does not follow the DLBCL patient/tissue '
            f'contract: {invalid}'
        )

    observed_patients = set(source['Patient_Code'].unique())
    expected_patients = set(STEP06_DLBCL_PATIENT_ORDER)
    if observed_patients != expected_patients:
        raise RuntimeError(
            '[Step 06-04] patient set differs from the frozen four-patient contract: '
            f'observed={sorted(observed_patients)}, expected={STEP06_DLBCL_PATIENT_ORDER}'
        )
    tissue_counts = source.groupby('Patient_Code', observed=True)['Run_Tissue_name'].nunique()
    invalid_tissue_counts = tissue_counts.reindex(STEP06_DLBCL_PATIENT_ORDER)
    if invalid_tissue_counts.isna().any() or not invalid_tissue_counts.eq(2).all():
        raise RuntimeError(
            '[Step 06-04] each patient must map to exactly two tissues: '
            f'{invalid_tissue_counts.to_dict()}'
        )
    return source


def _render_step06_saved_nearest_only_composition():
    """Render descriptive patient-stratified Step06-04 composition after tie exclusion."""
    filename = 'Final_MV1_Step065C_NearestOnly_Tcell_To_Bstate_Composition.png'
    if not _final_figure_stem_registered(filename):
        raise RuntimeError(f'Step06 canonical Figure stem is not registered: {filename}')

    source = _step06_patient_nearest_only_distance_source()
    state_order = ['HLA-DR+ B Cell', 'HLA-DR- B Cell']
    patient_percentages = []
    patient_diagnostics = []
    for patient_code in STEP06_DLBCL_PATIENT_ORDER:
        patient_source = source[source['Patient_Code'] == patient_code].copy()
        counts, excluded_tie_n = _step06_nearest_only_counts_from_distances(
            patient_source, f'Step 06-04 {patient_code}',
        )
        patient_percentages.append(
            counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        )
        patient_diagnostics.append({
            'Patient_Code': patient_code,
            'Tissues_n': int(patient_source['Run_Tissue_name'].nunique()),
            'Assignment_rows_n': int(len(patient_source)),
            'HLA_DR_pos_assigned_n': int(counts.loc['HLA-DR+ B Cell'].sum()),
            'HLA_DR_neg_assigned_n': int(counts.loc['HLA-DR- B Cell'].sum()),
            'Exact_ties_excluded_n': int(excluded_tie_n),
        })

    group_centers = np.arange(len(STEP06_DLBCL_PATIENT_ORDER), dtype=float) * 0.95
    state_offsets = np.array([-0.18, 0.18], dtype=float)
    bar_positions = np.concatenate([center + state_offsets for center in group_centers])
    fig, axis = plt.subplots(figsize=STEP06_FIGURE5_SIZES[filename])
    bottom = np.zeros(len(bar_positions), dtype=float)
    for subtype in STEP06_T_CELL_LABEL_ORDER:
        values = np.concatenate([
            percentages[subtype].reindex(state_order).fillna(0).to_numpy(dtype=float)
            for percentages in patient_percentages
        ])
        axis.bar(
            bar_positions, values, bottom=bottom,
            color=STEP06_T_SUBTYPE_COMPOSITION_COLORS.get(subtype, '#999999'),
            edgecolor='white', linewidth=0.7, width=0.30,
            label=_step06_display_t_subset(subtype),
        )
        bottom += values

    axis.set_xticks(bar_positions)
    axis.set_xticklabels(
        ['DR+', 'DR−'] * len(STEP06_DLBCL_PATIENT_ORDER),
        fontsize=10.0, fontweight='bold',
    )
    axis.set_xlim(group_centers[0] - 0.43, group_centers[-1] + 0.43)
    axis.set_ylim(0, 100)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.set_ylabel('Nearest-assigned T cells (%)', fontsize=13.0, fontweight='bold')
    axis.set_xlabel(
        'Nearest HLA-DR B-cell state (≤50 µm)',
        fontsize=12.0, fontweight='bold', labelpad=6,
    )
    axis.tick_params(axis='y', labelsize=11.0)
    plt.setp(axis.get_yticklabels(), fontweight='bold')
    axis.grid(axis='y', linestyle='--', alpha=0.35)
    for center, patient_code in zip(group_centers, STEP06_DLBCL_PATIENT_ORDER):
        axis.text(
            center, 1.025, patient_code,
            transform=axis.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=12.5,
            fontweight='bold', clip_on=False,
        )

    handles, labels = axis.get_legend_handles_labels()
    fig.suptitle(
        'Nearest-only T-cell composition by patient',
        fontsize=15.5, fontweight='bold', y=0.970,
    )
    fig.legend(
        handles, labels, loc='lower center', bbox_to_anchor=(0.50, 0.015),
        ncol=6, fontsize=10.2, frameon=False, borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.110, right=0.990, bottom=0.215, top=0.845,
    )
    for diagnostic in patient_diagnostics:
        print(f'[Step 06-04] patient composition: {diagnostic}')
    return _save_step06_matplotlib(
        fig, filename, 'Step06-04 patient-stratified nearest-only T-cell composition',
    )


# Retain the Step06-02b renderer above for provenance, but exclude it from active
# Figure-only dispatch and the static-output whitelist.
STEP06_SAVED_FIGURE_RENDERERS = (
    _render_step06_saved_distance_panel,
    _render_step06_saved_functional_gradients,
    _render_step06_saved_paired_distance_heatmap,
    _render_step06_saved_nearest_only_composition,
)


def _run_step06_figure_only():
    """Render the complete active Step06 set from protected saved sources."""
    saved_reports = [renderer() for renderer in STEP06_SAVED_FIGURE_RENDERERS]
    saved_count = sum(report is not None for report in saved_reports)
    if saved_count != len(STEP06_SAVED_FIGURE_RENDERERS):
        raise RuntimeError(
            f'Step06 active Figure family was not saved completely: '
            f'{saved_count}/{len(STEP06_SAVED_FIGURE_RENDERERS)} pairs'
        )
    print(
        '✅ Step06 Figure-only saved-result workflow: '
        f'saved_pairs={saved_count}'
    )


if STEP06_FIGURE_ONLY:
    _run_step06_figure_only()
    raise SystemExit(0)


# The current Final entry point exits above after protected-source rendering.
# The calculation blocks below remain as the preserved full-analysis reference
# path and must not run without separate authorization to write new artifacts.

# %%
# ==============================================================================
# [Step 06-2] T-to-B Spatial Distance and T-cell Functional Gradients
# ▷ Analyze distances from each T cell to the nearest HLA-DR+/- B cells and
#    the distance dependence of T-cell activation/exhaustion scores.
# ▷ GMM-based HLA-DR smart cutoff → spatial T-B KD-Tree distance calculation
# ▷ Explore candidate pooled T-cell proximity radii using hypergeometric tests
# ==============================================================================

# (Library imports are completed in Step 06-1.)

print("🔹 [Step 06-2] Analyzing T-cell-to-HLA-DR+/- B-cell proximity and functional-score gradients...")

# =====================================================================
# 📌 1. Define target markers and scoring markers
# =====================================================================
checkpoints = ['CD69', 'PD1', 'TIM3', 'LAG3', 'ICOS', 'CD134', 'CTLA4']
checkpoints = [m for m in checkpoints if m in adata.var_names]

exh_markers = ['PD1', 'TIM3', 'LAG3', 'CTLA4']
act_markers = ['CD69', 'ICOS', 'CD134']

exh_valid = [m for m in exh_markers if m in adata.var_names]
act_valid = [m for m in act_markers if m in adata.var_names]

all_needed_markers = list(set(checkpoints + exh_valid + act_valid))

b_cell_labels = _get_step06_b_cell_labels(unique_labels)
t_cell_labels = _get_step06_t_cell_labels(unique_labels)
if not b_cell_labels:
    raise ValueError("[Step 06] No B-cell labels are available for the HLA-DR reference.")
hla_dr_marker = next((m for m in STEP06_HLA_DR_MARKER_CANDIDATES if m in adata.var_names), None)
if hla_dr_marker is None:
    raise KeyError(f"Markers required for Step06 HLA-DR B-cell polarity analysis are missing: {STEP06_HLA_DR_MARKER_CANDIDATES}")
print(f"  ▶ HLA-DR polarity marker: {hla_dr_marker}")
print(f"  ▶ Reference B cell labels: {b_cell_labels}")
print(f"  ▶ Evaluated T cell labels: {t_cell_labels}")

# =====================================================================
# 📌 2. Classify B-cell polarity (GMM-based HLA-DR smart cutoff)
# =====================================================================
b_mask = adata.obs['Disease'].str.contains('DLBCL', na=False) & adata.obs['Niche_Label'].isin(b_cell_labels)
b_idx = adata.obs[b_mask].index

if sp.issparse(adata.X): b_expr_vals = adata[b_idx, adata.var_names == hla_dr_marker].X.toarray().flatten()
else: b_expr_vals = adata[b_idx, adata.var_names == hla_dr_marker].X.flatten()

b_expr = pd.Series(b_expr_vals, index=b_idx)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
b_expr_reshaped = b_expr_vals.reshape(-1, 1)
try:
    gmm.fit(b_expr_reshaped)
    pred_labels = gmm.predict(b_expr_reshaped)
    means = gmm.means_.flatten()
    pos_comp_idx = 0 if means[0] > means[1] else 1
    hla_pos_idx = b_idx[pred_labels == pos_comp_idx]
    hla_neg_idx = b_idx[pred_labels != pos_comp_idx]
    if b_expr[hla_pos_idx].mean() < b_expr[hla_neg_idx].mean():
         hla_pos_idx, hla_neg_idx = hla_neg_idx, hla_pos_idx
    gmm_cutoff = b_expr[hla_pos_idx].min()
    print(f"  ▶ [Complete] GMM-based HLA-DR threshold: {gmm_cutoff:.3f}")
except Exception as e:
    hla_pos_idx = b_idx[b_expr > 0.0]; hla_neg_idx = b_idx[b_expr <= 0.0]
    print(f"  ▶ GMM failed; applying the default (0.0): {e}")

STEP06_CUTOFF_COORD_UNITS = STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM / STEP06_COORD_UM_PER_UNIT
adata.uns['STEP06_COORD_UM_PER_UNIT'] = STEP06_COORD_UM_PER_UNIT
adata.uns['STEP06_PUBLIC_TIFF_DOWNSAMPLE_FACTOR'] = STEP06_PUBLIC_TIFF_DOWNSAMPLE_FACTOR
adata.uns['STEP06_COORD_UNIT_SOURCE'] = STEP06_COORD_UNIT_SOURCE
adata.uns['STEP06_DISTANCE_UNIT'] = 'um'
adata.uns['STEP06_PHENOIMAGER_DISTANCE_CUTOFF_COORD_UNITS'] = STEP06_CUTOFF_COORD_UNITS
print(
    f"  ▶ Distance unit conversion: 1 annotation coordinate unit = {STEP06_COORD_UM_PER_UNIT:.5f} um; "
    f"{STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM:.0f} um = {STEP06_CUTOFF_COORD_UNITS:.1f} coordinate units"
)

# =====================================================================
# 📌 3. Calculate spatial KD-Tree distances and combine expression / scores
# =====================================================================
t_mask = adata.obs['Disease'].str.contains('DLBCL', na=False) & adata.obs['Niche_Label'].isin(t_cell_labels)
dlbcl_tissues = adata.obs.loc[t_mask, 'Run_Tissue_name'].unique()
t_dists_list, all_dists_list = [], []

for tissue in tqdm(dlbcl_tissues, desc="KD-Tree tissue distance calculation"):
    ts_mask = (adata.obs['Run_Tissue_name'] == tissue)
    b_p_coords = adata.obs.loc[hla_pos_idx.intersection(adata.obs[ts_mask].index), ['X_centroid', 'Y_centroid']].values
    b_n_coords = adata.obs.loc[hla_neg_idx.intersection(adata.obs[ts_mask].index), ['X_centroid', 'Y_centroid']].values
    t_cells_in_tissue = adata.obs.loc[t_mask & ts_mask].index
    t_coords = adata.obs.loc[t_cells_in_tissue, ['X_centroid', 'Y_centroid']].values
    all_cells_in_tissue = adata.obs.loc[ts_mask & adata.obs['Disease'].str.contains('DLBCL', na=False)].index
    all_coords = adata.obs.loc[all_cells_in_tissue, ['X_centroid', 'Y_centroid']].values
    
    if len(t_coords) == 0:
        continue
    if len(b_p_coords) > 0:
        tree_pos = cKDTree(b_p_coords)
        d_pos_coord, _ = tree_pos.query(t_coords)
        d_pos_all_coord, _ = tree_pos.query(all_coords)
        d_pos = d_pos_coord * STEP06_COORD_UM_PER_UNIT
        d_pos_all = d_pos_all_coord * STEP06_COORD_UM_PER_UNIT
    else:
        d_pos_coord = np.full(len(t_coords), np.nan)
        d_pos_all_coord = np.full(len(all_coords), np.nan)
        d_pos = np.full(len(t_coords), np.nan)
        d_pos_all = np.full(len(all_coords), np.nan)
    if len(b_n_coords) > 0:
        tree_neg = cKDTree(b_n_coords)
        d_neg_coord, _ = tree_neg.query(t_coords)
        d_neg_all_coord, _ = tree_neg.query(all_coords)
        d_neg = d_neg_coord * STEP06_COORD_UM_PER_UNIT
        d_neg_all = d_neg_all_coord * STEP06_COORD_UM_PER_UNIT
    else:
        d_neg_coord = np.full(len(t_coords), np.nan)
        d_neg_all_coord = np.full(len(all_coords), np.nan)
        d_neg = np.full(len(t_coords), np.nan)
        d_neg_all = np.full(len(all_coords), np.nan)
    
    t_dists_list.append(pd.DataFrame({
        'T_Subset': adata.obs.loc[t_cells_in_tissue, 'Niche_Label'].values,
        'Dist_to_Pos': d_pos,
        'Dist_to_Neg': d_neg,
        'Dist_to_Pos_coord': d_pos_coord,
        'Dist_to_Neg_coord': d_neg_coord,
    }, index=t_cells_in_tissue))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_dists_list.append(np.nanmin(np.column_stack([d_pos_all, d_neg_all]), axis=1))

if not t_dists_list:
    raise ValueError("[Step 06-2] No T-cell distance results are available. Check T-cell labels or coordinate columns.")

dist_df = pd.concat(t_dists_list)
dist_df['T_Subset'] = dist_df['T_Subset'].astype(str).str.strip()
dist_df = dist_df[dist_df['T_Subset'].isin(t_cell_labels)].copy()
plot_t_cell_labels = [label for label in t_cell_labels if label in set(dist_df['T_Subset'])]
if not plot_t_cell_labels:
    raise ValueError("[Step 06-2] No distance results remain after retaining only whitelisted T-cell subsets.")
t_idx = dist_df.index
# 💡 [Change] Explicitly enforce column order to prevent value/name mapping errors
valid_markers = [m for m in all_needed_markers if m in adata.var_names]
if sp.issparse(adata.X): 
    t_expr = pd.DataFrame(adata[t_idx][:, valid_markers].X.toarray(), index=t_idx, columns=valid_markers)
else: 
    t_expr = pd.DataFrame(adata[t_idx][:, valid_markers].X, index=t_idx, columns=valid_markers)

dist_df = pd.concat([dist_df, t_expr], axis=1)
if len(exh_valid) > 0: dist_df['Exhaustion_Score'] = dist_df[exh_valid].apply(zscore, nan_policy='omit').mean(axis=1)
if len(act_valid) > 0: dist_df['Activation_Score'] = dist_df[act_valid].apply(zscore, nan_policy='omit').mean(axis=1)

new_scores = [s for s in ['Exhaustion_Score', 'Activation_Score'] if s in dist_df.columns]
checkpoints = new_scores + checkpoints 

# 💡 [Important] Persist calculated distances and scores in adata.obs (for downstream steps such as Step 08)
for col in ['Dist_to_Pos', 'Dist_to_Neg', 'Dist_to_Pos_coord', 'Dist_to_Neg_coord', 'Exhaustion_Score', 'Activation_Score']:
    if col in dist_df.columns:
        adata.obs.loc[dist_df.index, col] = dist_df[col]

# =====================================================================
# 📌 4. Exploratory proximity cutoff and statistical tests
# =====================================================================
if not all_dists_list:
    raise ValueError("[Step 06-2] No all-cell distance results are available. Check B-cell HLA-DR(+/-) coordinates.")
all_dists_array = np.concatenate(all_dists_list)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    t_dists_array = np.nanmin(dist_df[['Dist_to_Pos', 'Dist_to_Neg']].values, axis=1)

test_radii = np.arange(10, 101, 5); valid_radii = []
N_tot, n_T = np.sum(~np.isnan(all_dists_array)), np.sum(~np.isnan(t_dists_array))
for r in test_radii:
    K_tot, k_obs = np.sum(all_dists_array <= r), np.sum(t_dists_array <= r)
    if K_tot == 0 or k_obs == 0: continue
    if (1 - hypergeom.cdf(k_obs - 1, N_tot, K_tot, n_T)) < 0.05: valid_radii.append(r)

INTERACT_CUTOFF = max(valid_radii) if valid_radii else 50
# Define a global variable for downstream use, including Step 08
adata.uns['INTERACT_CUTOFF'] = INTERACT_CUTOFF
print(f"  ▶ [Derived] Exploratory T-cell-to-B-cell proximity cutoff: {INTERACT_CUTOFF} um\n")

t_totals = dist_df.groupby('T_Subset', observed=True).size().reindex(plot_t_cell_labels).fillna(0)
pos_engaged = dist_df[dist_df['Dist_to_Pos'] <= INTERACT_CUTOFF].groupby('T_Subset', observed=True).size().reindex(plot_t_cell_labels).fillna(0)
neg_engaged = dist_df[dist_df['Dist_to_Neg'] <= INTERACT_CUTOFF].groupby('T_Subset', observed=True).size().reindex(plot_t_cell_labels).fillna(0)
global_pos_ratio = len(hla_pos_idx) / (len(hla_pos_idx) + len(hla_neg_idx))

stats_rows_raw = []
paired_dist_df_062 = _step06_complete_pair_distance_frame(
    dist_df, 'Step 06-01 full-analysis',
)
for t_sub in plot_t_cell_labels:
    d_sub = paired_dist_df_062[paired_dist_df_062['T_Subset'] == t_sub]
    p_cnt, n_cnt = pos_engaged.get(t_sub, 0), neg_engaged.get(t_sub, 0)
    tot_eng = p_cnt + n_cnt
    exp_p, exp_n = tot_eng * global_pos_ratio, tot_eng * (1 - global_pos_ratio)
    lfc_p = np.log2((p_cnt + 0.5) / (exp_p + 0.5)) if tot_eng > 0 else 0
    lfc_n = np.log2((n_cnt + 0.5) / (exp_n + 0.5)) if tot_eng > 0 else 0
    p_chi = chisquare([p_cnt, n_cnt], f_exp=[exp_p, exp_n])[1] if tot_eng > 5 and exp_p > 0 and exp_n > 0 else np.nan
    
    d_pos_v = d_sub['Dist_to_Pos'].to_numpy(dtype=float)
    d_neg_v = d_sub['Dist_to_Neg'].to_numpy(dtype=float)
    delta_v = d_pos_v - d_neg_v
    if len(d_sub) > 5 and np.any(delta_v != 0):
        try:
            p_paired = wilcoxon(
                d_pos_v, d_neg_v, alternative='two-sided', zero_method='wilcox',
            ).pvalue
        except ValueError:
            p_paired = np.nan
    elif len(d_sub) > 5:
        p_paired = 1.0
    else:
        p_paired = np.nan
    stats_rows_raw.append({
        'T_Subset': t_sub,
        'DR_pos_engaged_n': int(p_cnt),
        'DR_neg_engaged_n': int(n_cnt),
        'Affinity_Log2FC_DRpos': float(lfc_p),
        'Affinity_Log2FC_DRneg': float(lfc_n),
        'Affinity_p (Chi2)': p_chi,
        'Paired_T_cells_within_50um_n': int(len(d_sub)),
        'Proximity_p (paired Wilcoxon)': p_paired,
    })

stats_df = pd.DataFrame(stats_rows_raw)
stats_df['Affinity_q (BH-FDR)'] = _bh_fdr(stats_df['Affinity_p (Chi2)'].values)
stats_df['Proximity_q (BH-FDR)'] = _bh_fdr(
    stats_df['Proximity_p (paired Wilcoxon)'].values
)
_step06_write_table_if_enabled(stats_df, os.path.join(FINAL_TABLE_DIR, STEP06_STATS_CSV), index=False)
print(f"  ✅ Step06 statistics table saved: {os.path.join(FINAL_TABLE_DIR, STEP06_STATS_CSV)}")

stats_dict_plot, lfc_list = {}, []
for _, row in stats_df.iterrows():
    t_sub = row['T_Subset']
    sig_A = _sig_from_q(row['Affinity_q (BH-FDR)'])
    lfc_list.append({'T_Subset_Plot': f"{t_sub}<br>(q:{sig_A})", 'Target': STEP06_B_STATE_POS, 'Log2FC': row['Affinity_Log2FC_DRpos']})
    lfc_list.append({'T_Subset_Plot': f"{t_sub}<br>(q:{sig_A})", 'Target': STEP06_B_STATE_NEG, 'Log2FC': row['Affinity_Log2FC_DRneg']})
    stats_dict_plot[t_sub] = {
        'label_B': f"{t_sub}<br>({_step06_q_label(row['Proximity_q (BH-FDR)'])})"
    }
# =====================================================================
# 📌 PART ❶. Visualization (A. Affinity, B. Proximity Depth, C. Subset Distribution)
# =====================================================================
fig_freq = px.bar(pd.DataFrame(lfc_list), x="T_Subset_Plot", y="Log2FC", color="Target", barmode='group',
    title=f"<b>[Step06-2A | T-to-B] T-cell subset enrichment around HLA-DR+/- B cells (hot-zone: {INTERACT_CUTOFF}µm; BH-FDR q)</b>",
    labels={'Log2FC': 'Observed/expected T cells within B-cell hot-zone (log2FC)', 'Target': 'Reference B-cell state'},
    color_discrete_map=STEP06_B_STATE_COLORS)
fig_freq.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')
fig_freq.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside', showgrid=True, gridcolor='#f0f0f0')
fig_freq.update_layout(plot_bgcolor='white', width=1300, height=450)

dist_interact_only = paired_dist_df_062.melt(
    id_vars=['T_Subset'],
    value_vars=['Dist_to_Pos', 'Dist_to_Neg'],
    var_name='B_Cell_Type',
    value_name='Distance',
)
dist_interact_only['B_Cell_Type'] = dist_interact_only['B_Cell_Type'].replace({'Dist_to_Pos': STEP06_B_STATE_POS, 'Dist_to_Neg': STEP06_B_STATE_NEG})
dist_interact_only['T_Subset_Plot'] = dist_interact_only['T_Subset'].map(lambda x: stats_dict_plot.get(x, {}).get('label_B', str(x)))

fig_dist = px.box(dist_interact_only, x="T_Subset_Plot", y="Distance", color="B_Cell_Type",
    title="<b>[Step06-2B | T-to-B] Complete-pair distance to nearest HLA-DR+/- B cells "
          "(paired Wilcoxon; BH-FDR q)</b>",
    labels={'Distance': 'Distance from T cell to nearest HLA-DR+/- B cell (µm)', 'B_Cell_Type': 'Reference B-cell state'},
    color_discrete_map=STEP06_B_STATE_COLORS, points=False)
fig_dist.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')
fig_dist.update_yaxes(
    showline=True, linewidth=2, linecolor='black', mirror=True,
    ticks='outside', showgrid=True, gridcolor='#f0f0f0',
    range=[0, STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM],
)
fig_dist.update_layout(plot_bgcolor='white', width=1300, height=450, boxmode="group")

# 💡 [Added] 1-C. Compare T-cell subset abundances by HLA-DR(+/-) status
subset_dist_df = pd.DataFrame({
    f'Near {STEP06_B_STATE_POS} (%)': (pos_engaged / t_totals * 100),
    f'Near {STEP06_B_STATE_NEG} (%)': (neg_engaged / t_totals * 100)
}).fillna(0).reset_index().rename(columns={'index':'T_Subset'})
# Transform data for visualization (Melt)
subset_melt = subset_dist_df.melt(id_vars='T_Subset', var_name='Niche', value_name='Percentage')
# Create the plot
fig_subset = px.bar(subset_melt, x='T_Subset', y='Percentage', color='Niche', barmode='group',
    title=f"<b>[Step06-2C | T-to-B] T-cell subset fraction inside HLA-DR+/- B-cell hot-zones (<{INTERACT_CUTOFF}µm)</b>",
    labels={'Percentage': 'Fraction of each T-cell subset within B-cell hot-zone (%)', 'Niche': 'Reference B-cell state'},
    color_discrete_map={f'Near {STEP06_B_STATE_POS} (%)': '#d62728', f'Near {STEP06_B_STATE_NEG} (%)': '#1f77b4'})
# Detailed design settings
fig_subset.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')
fig_subset.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside', 
                         showgrid=True, gridcolor='#f0f0f0', title_text="Fraction of each T-cell subset within B-cell hot-zone (%)")
fig_subset.update_layout(plot_bgcolor='white', width=1300, height=450,
                         legend_title_text="Niche Distribution")

# =====================================================================
# 📌 PART ❷. Spatial Gradient (restore axes and lines)
# =====================================================================
def get_marker_title(m):
    if m == 'Exhaustion_Score': return f"<b>Exhaustion Score</b><br>({', '.join(exh_valid)})"
    elif m == 'Activation_Score': return f"<b>Activation Score</b><br>({', '.join(act_valid)})"
    else: return f"<b>{m} Spatial Gradient</b>"

colors = px.colors.qualitative.D3 + px.colors.qualitative.Bold
bins = sorted(list(set(STEP06_2D_DISTANCE_BINS_UM + [float(INTERACT_CUTOFF)])))
bin_labels = [f"{bins[k]:g}-{bins[k+1]:g}µm" for k in range(len(bins)-1)]
bin_midpoints = {lbl: (bins[k] + bins[k+1]) / 2.0 for k, lbl in enumerate(bin_labels)}
dist_df['Bin_Pos'] = pd.cut(dist_df['Dist_to_Pos'], bins=bins, labels=bin_labels)
dist_df['Bin_Neg'] = pd.cut(dist_df['Dist_to_Neg'], bins=bins, labels=bin_labels)

fig_expr = make_subplots(rows=math.ceil(len(checkpoints)/2), cols=2, 
                         subplot_titles=[get_marker_title(m) for m in checkpoints],
                         vertical_spacing=0.08, horizontal_spacing=0.07)

for i, marker in enumerate(checkpoints):
    r, c = (i // 2) + 1, (i % 2) + 1
    fig_expr.add_vrect(x0=0, x1=min(20, INTERACT_CUTOFF), fillcolor="rgba(255, 224, 130, 0.35)", layer="below", line_width=0, row=r, col=c)
    if INTERACT_CUTOFF > 20:
        fig_expr.add_vrect(x0=20, x1=min(50, INTERACT_CUTOFF), fillcolor="rgba(174, 214, 241, 0.28)", layer="below", line_width=0, row=r, col=c)
    fig_expr.add_vline(x=20, line_width=1, line_dash='dash', line_color='rgba(80,80,80,0.65)', row=r, col=c)
    grp_p = dist_df.groupby(['T_Subset', 'Bin_Pos'], observed=True)[marker].mean().reset_index()
    grp_n = dist_df.groupby(['T_Subset', 'Bin_Neg'], observed=True)[marker].mean().reset_index()
    for j, t_sub in enumerate(plot_t_cell_labels):
        lc = colors[j % len(colors)]
        df_p = grp_p[grp_p['T_Subset'] == t_sub].dropna()
        df_n = grp_n[grp_n['T_Subset'] == t_sub].dropna()
        if len(df_p) > 0:
            fig_expr.add_trace(go.Scatter(x=df_p['Bin_Pos'].map(bin_midpoints), y=df_p[marker], name=f"<b>{t_sub}</b> near HLA-DR+ B", line=dict(color=lc, width=3.5), legendgroup=t_sub, showlegend=(i==0)), row=r, col=c)
        if len(df_n) > 0:
            fig_expr.add_trace(go.Scatter(x=df_n['Bin_Neg'].map(bin_midpoints), y=df_n[marker], name=f"{t_sub} near HLA-DR- B", line=dict(color=lc, width=2.0, dash='dot'), legendgroup=t_sub, showlegend=(i==0)), row=r, col=c)

fig_expr.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', showgrid=True, gridcolor='#f0f0f0')
fig_expr.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', showgrid=True, gridcolor='#f0f0f0')
fig_expr.update_layout(
    title="<b>[Step06-2D] Functional marker gradients by HLA-DR B-cell proximity</b><br><span style='font-size:12px'>Solid line = nearest HLA-DR+ B cell; dotted line = nearest HLA-DR- B cell; shaded zones = 0-20 and 20-50 um</span>",
    height=510 * math.ceil(len(checkpoints)/2),
    width=1700,
    plot_bgcolor='white',
    hovermode="x unified",
    margin=dict(l=80, r=260, t=110, b=95)
)
fig_expr.add_annotation(
    text="<b>Distance direction:</b> T cell -> nearest HLA-DR+/- B cell",
    x=0.5,
    y=-0.08,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=13, color="#333333")
)

# =====================================================================
# 📌 PART 3. Individual PNG outputs
# =====================================================================
def _save_step062_static_pngs():
    """Save the same data as the Step06-2 Plotly panels as Matplotlib/Seaborn PNGs."""
    lfc_plot_df = pd.DataFrame(lfc_list).copy()
    lfc_plot_df['T_Subset_Name'] = lfc_plot_df['T_Subset_Plot'].map(_step06_t_subset_from_plot_label)
    lfc_plot_df = lfc_plot_df[lfc_plot_df['T_Subset_Name'].isin(plot_t_cell_labels)].copy()
    lfc_plot_df['T_Subset_Plot'] = lfc_plot_df['T_Subset_Plot'].map(_plain_step06_label)
    lfc_order = _step06_plot_order_from_labels(lfc_plot_df['T_Subset_Plot'].unique())
    if STEP06_EXPORT_TTOB_ENRICHMENT_PANEL:
        fig, ax = plt.subplots(figsize=(11.6, 4.2))
        sns.barplot(
            data=lfc_plot_df,
            x='T_Subset_Plot',
            y='Log2FC',
            hue='Target',
            order=lfc_order,
            palette=STEP06_B_STATE_COLORS,
            edgecolor='black',
            linewidth=0.6,
            ax=ax
        )
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(
            f"T-to-B enrichment of T-cell subsets around HLA-DR+/- B cells\n"
            f"log2(observed / expected) within nearest B-cell hot-zone ({INTERACT_CUTOFF:g} um)",
            fontsize=13.2,
            fontweight='bold',
            pad=8
        )
        ax.set_xlabel('')
        ax.set_ylabel('Enrichment (log2 O/E)', fontsize=12, fontweight='bold', labelpad=12)
        ax.tick_params(axis='x', rotation=30, labelsize=9.2)
        ax.tick_params(axis='y', labelsize=10.5)
        ax.legend(title='Reference B-cell state', frameon=True, fontsize=9.5, title_fontsize=10.2)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.24, top=0.78)
        _save_step06_matplotlib(fig, "Final_MV1_Step062A_TtoB_Tcell_Subset_Enrichment.png", "Step06-2A T-to-B T-cell subset enrichment")

    dist_plot_df = _filter_step06_t_subset_df(dist_interact_only)
    dist_plot_df['T_Subset_Plot'] = dist_plot_df['T_Subset_Plot'].map(_plain_step06_label)
    dist_order_raw = _step06_plot_order_from_labels(dist_plot_df['T_Subset_Plot'].unique())
    dist_plot_df['T_Subset_Plot'] = dist_plot_df['T_Subset_Plot'].map(_step06_display_plot_label)
    dist_order = [_step06_display_plot_label(v) for v in dist_order_raw]
    _b_state_display_062 = {
        STEP06_B_STATE_POS: 'HLA-DR+ B cells',
        STEP06_B_STATE_NEG: 'HLA-DR- B cells',
    }
    dist_plot_df['B_Cell_Type'] = dist_plot_df['B_Cell_Type'].replace(_b_state_display_062)
    _step062b_name = "Final_MV1_Step062B_TtoB_Tcell_Distance_to_Bstate.png"
    fig, ax = plt.subplots(figsize=STEP06_FIGURE5_SIZES[_step062b_name])
    sns.boxplot(
        data=dist_plot_df,
        x='T_Subset_Plot',
        y='Distance',
        hue='B_Cell_Type',
        order=dist_order,
        palette={
            'HLA-DR+ B cells': STEP06_B_STATE_COLORS[STEP06_B_STATE_POS],
            'HLA-DR- B cells': STEP06_B_STATE_COLORS[STEP06_B_STATE_NEG],
        },
        width=0.48,
        showfliers=False,
        linewidth=0.8,
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_ylim(0, STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM)
    ax.tick_params(axis='x', rotation=22, labelsize=10.4)
    ax.tick_params(axis='y', labelsize=10.8)
    plt.setp(ax.get_xticklabels(), fontweight='bold', ha='right')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    ax.legend(
        loc='upper right',
        frameon=False,
        fontsize=10.2,
        title='Nearest B-cell state',
        title_fontsize=11.2
    )
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.235, top=0.965)
    fig.text(
        0.055, 0.59, 'Distance to nearest B cell (µm)',
        rotation=90, ha='center', va='center',
        fontsize=13.2, fontweight='bold',
    )
    _save_step06_matplotlib(fig, _step062b_name, "Step06-2B T-to-B distance to B-cell state")

    subset_plot_df = _filter_step06_t_subset_df(subset_melt)
    if STEP06_EXPORT_TTOB_ENGAGEMENT_PANEL:
        fig, ax = plt.subplots(figsize=(11.6, 4.2))
        sns.barplot(
            data=subset_plot_df,
            x='T_Subset',
            y='Percentage',
            hue='Niche',
            order=plot_t_cell_labels,
            palette={f'Near {STEP06_B_STATE_POS} (%)': '#d62728', f'Near {STEP06_B_STATE_NEG} (%)': '#1f77b4'},
            edgecolor='black',
            linewidth=0.6,
            ax=ax
        )
        ax.set_title(
            f"T-cell subset fraction inside HLA-DR+/- B-cell hot-zones\n"
            f"T cells mapped to B-cell niches ({INTERACT_CUTOFF:g} um)",
            fontsize=13.2,
            fontweight='bold',
            pad=8
        )
        ax.set_xlabel('')
        ax.set_ylabel('T-cell subset in hot-zone (%)', fontsize=12, fontweight='bold', labelpad=12)
        ax.tick_params(axis='x', rotation=30, labelsize=9.2)
        ax.tick_params(axis='y', labelsize=10.5)
        ax.legend(title='Reference B-cell state', frameon=True, fontsize=9.5, title_fontsize=10.2)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        fig.subplots_adjust(left=0.095, right=0.985, bottom=0.24, top=0.78)
        _save_step06_matplotlib(fig, "Final_MV1_Step062C_TtoB_Tcell_Subset_Engagement.png", "Step06-2C T-to-B T-cell subset engagement")

    plot_dist_df = _filter_step06_t_subset_df(dist_df)

    def _draw_step062_gradient_panels(marker_list_062, title_062, filename_062, label_062, n_cols_062=2, main_panel_062=False):
        """Save Step06-2D gradients as separate score-summary and individual-marker panels."""
        marker_list_062 = [m for m in marker_list_062 if m in plot_dist_df.columns]
        if not marker_list_062:
            return
        n_cols_local_062 = min(n_cols_062, len(marker_list_062))
        n_rows_local_062 = math.ceil(len(marker_list_062) / n_cols_local_062)
        if main_panel_062 and filename_062 in STEP06_FIGURE5_SIZES:
            fig_width_062, fig_height_062 = STEP06_FIGURE5_SIZES[filename_062]
        else:
            fig_width_062 = 4.3 * n_cols_local_062
            fig_height_062 = 4.0 * n_rows_local_062
        fig, axes = plt.subplots(
            n_rows_local_062,
            n_cols_local_062,
            figsize=(fig_width_062, fig_height_062),
            squeeze=False
        )
        xticks_062 = [x for x in STEP06_2D_DISTANCE_BINS_UM if x <= max(bins)]

        def _format_step062_gradient_axis(ax):
            """Display the 0–20 um and 20–50 um hot zones separately."""
            ax.axvspan(0, min(20, INTERACT_CUTOFF), color='#FFE082', alpha=0.30, zorder=0)
            if INTERACT_CUTOFF > 20:
                ax.axvspan(20, min(50, INTERACT_CUTOFF), color='#AED6F1', alpha=0.24, zorder=0)
            ax.axvline(20, color='#6c757d', linestyle='--', linewidth=1.0, alpha=0.85, zorder=1)
            ax.axvline(INTERACT_CUTOFF, color='#8a6d3b', linestyle=':', linewidth=1.0, alpha=0.85, zorder=1)
            ax.set_xlim(0, max(bins))
            ax.set_xticks(xticks_062)
            ax.tick_params(axis='x', labelsize=10.5 if main_panel_062 else 9.5)
            ax.tick_params(axis='y', labelsize=10.5 if main_panel_062 else 9.5)
            ax.grid(True, axis='y', linestyle='--', alpha=0.28)
            ax.grid(True, axis='x', linestyle=':', alpha=0.16)
            for spine_062 in ax.spines.values():
                spine_062.set_linewidth(1.0)

        for marker_idx_062, marker_062 in enumerate(marker_list_062):
            ax = axes[marker_idx_062 // n_cols_local_062, marker_idx_062 % n_cols_local_062]
            _format_step062_gradient_axis(ax)
            grp_p = plot_dist_df.groupby(['T_Subset', 'Bin_Pos'], observed=True)[marker_062].mean().reset_index()
            grp_n = plot_dist_df.groupby(['T_Subset', 'Bin_Neg'], observed=True)[marker_062].mean().reset_index()
            for j, t_sub in enumerate(plot_t_cell_labels):
                lc = colors[j % len(colors)]
                df_p = grp_p[grp_p['T_Subset'] == t_sub].dropna()
                df_n = grp_n[grp_n['T_Subset'] == t_sub].dropna()
                if len(df_p) > 0:
                    ax.plot(
                        df_p['Bin_Pos'].map(bin_midpoints),
                        df_p[marker_062],
                        color=lc,
                        linewidth=2.8 if main_panel_062 else 2.1,
                        marker='o',
                        markersize=4.5 if main_panel_062 else 3.3,
                        alpha=0.96,
                        label=_step06_display_t_subset(t_sub) if marker_idx_062 == 0 else None
                    )
                if len(df_n) > 0:
                    ax.plot(
                        df_n['Bin_Neg'].map(bin_midpoints),
                        df_n[marker_062],
                        color=lc,
                        linewidth=2.1 if main_panel_062 else 1.6,
                        linestyle=':',
                        marker='o',
                        markersize=3.8 if main_panel_062 else 2.9,
                        alpha=0.88,
                        label=None
                    )
            ax.set_title(_plain_step06_label(get_marker_title(marker_062)), fontsize=13.2 if main_panel_062 else 10.8, fontweight='bold', pad=6)
            ax.set_xlabel('Distance to nearest HLA-DR+/- B cell (µm)', fontsize=10.5 if main_panel_062 else 9.2, fontweight='bold')
            ylabel_062 = 'Mean marker z-score' if marker_062.endswith('_Score') else 'Mean marker expression'
            ax.set_ylabel(ylabel_062, fontsize=10.5 if main_panel_062 else 9.2, fontweight='bold')
        for empty_idx_062 in range(len(marker_list_062), n_rows_local_062 * n_cols_local_062):
            axes[empty_idx_062 // n_cols_local_062, empty_idx_062 % n_cols_local_062].axis('off')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        line_style_handles_062 = [
            mlines.Line2D([], [], color='black', linewidth=2.2, linestyle='-', label='nearest HLA-DR+ B'),
            mlines.Line2D([], [], color='black', linewidth=2.2, linestyle=':', label='nearest HLA-DR- B'),
        ]
        legend_handles_062 = handles + line_style_handles_062
        legend_labels_062 = labels + [h.get_label() for h in line_style_handles_062]
        if legend_handles_062:
            fig.legend(
                legend_handles_062,
                legend_labels_062,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.020),
                ncol=4 if main_panel_062 else min(3, max(len(legend_labels_062), 1)),
                fontsize=8.2 if main_panel_062 else 7.7,
                frameon=not main_panel_062,
                title=None if main_panel_062 else 'T-cell subset / reference B-state'
            )
        if not main_panel_062:
            fig.suptitle(title_062, fontsize=13.5, fontweight='bold', y=0.965)
        fig.tight_layout(rect=[0, 0.16, 1.0, 0.98 if main_panel_062 else 0.94])
        _save_step06_matplotlib(fig, filename_062, label_062)

    score_markers_062 = [s for s in ['Exhaustion_Score', 'Activation_Score'] if s in checkpoints]
    individual_markers_062 = [m for m in checkpoints if m not in score_markers_062]
    _draw_step062_gradient_panels(
        score_markers_062,
        "Activation/exhaustion score gradients by HLA-DR B-cell proximity",
        "Final_MV1_Step062D_TtoB_Functional_Gradients.png",
        "Step06-2D T-to-B activation/exhaustion score gradients",
        n_cols_062=2,
        main_panel_062=True
    )
    _draw_step062_gradient_panels(
        individual_markers_062,
        "Individual T-cell marker gradients by HLA-DR B-cell proximity",
        "Final_MV1_Step062D_TtoB_IndividualMarker_Gradients.png",
        "Step06-2D individual T-cell marker gradients",
        n_cols_062=2,
        main_panel_062=False
    )

    # Absolute-distance companion panel. Boxes summarize individual T-cell
    # scores, while each overlaid point is one tissue median. Statistical tests
    # remain paired at the tissue level to avoid treating cells as independent
    # biological replicates.
    _functional_scores_062 = [
        score for score in ['Exhaustion_Score', 'Activation_Score']
        if score in plot_dist_df.columns
    ]
    if _functional_scores_062:
        _near_far_df_062 = plot_dist_df[
            ['T_Subset', 'Dist_to_Pos', 'Dist_to_Neg'] + _functional_scores_062
        ].copy()
        _near_far_df_062['Tissue'] = adata.obs.reindex(_near_far_df_062.index)['Run_Tissue_name']
        _near_far_df_062['Nearest_B_Distance_um'] = _near_far_df_062[
            ['Dist_to_Pos', 'Dist_to_Neg']
        ].min(axis=1, skipna=True)
        _near_far_df_062 = _near_far_df_062.dropna(subset=['Tissue', 'Nearest_B_Distance_um'])
        _distance_group_order_062 = ['0–<20 µm', '20–<50 µm', '≥50 µm']
        _near_far_df_062['Distance_Group'] = pd.cut(
            _near_far_df_062['Nearest_B_Distance_um'],
            bins=[0.0, 20.0, 50.0, np.inf],
            labels=_distance_group_order_062,
            right=False,
            include_lowest=True,
        )
        _near_far_df_062 = _near_far_df_062.dropna(subset=['Distance_Group']).copy()
        _near_far_df_062['Distance_Group'] = pd.Categorical(
            _near_far_df_062['Distance_Group'],
            categories=_distance_group_order_062,
            ordered=True,
        )
        _near_far_df_062['T_Subset_Display'] = _near_far_df_062['T_Subset'].map(
            _step06_display_t_subset
        )

        _near_far_stats_062 = []
        _near_far_plot_rows_062 = []
        for _score_062 in _functional_scores_062:
            for _subtype_062 in plot_t_cell_labels:
                _sub_df_062 = _near_far_df_062[
                    _near_far_df_062['T_Subset'].eq(_subtype_062)
                ].dropna(subset=[_score_062])
                _tissue_summary_062 = (
                    _sub_df_062.groupby(['Tissue', 'Distance_Group'], observed=True)[_score_062]
                    .agg(Median='median', Cell_n='size')
                    .reset_index()
                )
                _tissue_summary_062 = _tissue_summary_062[_tissue_summary_062['Cell_n'].ge(5)]
                for _summary_row_062 in _tissue_summary_062.itertuples(index=False):
                    _near_far_plot_rows_062.append({
                        'Score': _score_062,
                        'T_Subset': _subtype_062,
                        'T_Subset_Display': _step06_display_t_subset(_subtype_062),
                        'Tissue': _summary_row_062.Tissue,
                        'Distance_Group': str(_summary_row_062.Distance_Group),
                        'Tissue_Median': float(_summary_row_062.Median),
                        'Cell_n': int(_summary_row_062.Cell_n),
                    })
                _summary_pivot_062 = _tissue_summary_062.pivot(
                    index='Tissue', columns='Distance_Group', values='Median'
                )
                for _group_a_062, _group_b_062 in combinations(_distance_group_order_062, 2):
                    if all(group in _summary_pivot_062.columns for group in [_group_a_062, _group_b_062]):
                        _paired_062 = _summary_pivot_062[[_group_a_062, _group_b_062]].dropna()
                    else:
                        _paired_062 = pd.DataFrame()
                    _n_tissues_062 = int(len(_paired_062))
                    if _n_tissues_062 >= 3 and np.any(
                        np.abs(
                            _paired_062[_group_a_062].to_numpy(dtype=float)
                            - _paired_062[_group_b_062].to_numpy(dtype=float)
                        ) > 0
                    ):
                        try:
                            _p_062 = wilcoxon(
                                _paired_062[_group_a_062],
                                _paired_062[_group_b_062],
                                alternative='two-sided'
                            ).pvalue
                        except ValueError:
                            _p_062 = np.nan
                    else:
                        _p_062 = np.nan
                    _near_far_stats_062.append({
                        'Score': _score_062,
                        'T_Subset': _subtype_062,
                        'Group_A': _group_a_062,
                        'Group_B': _group_b_062,
                        'Paired_tissues_n': _n_tissues_062,
                        'p': _p_062,
                    })

        _near_far_stats_062 = pd.DataFrame(_near_far_stats_062)
        _near_far_stats_062['q'] = _bh_fdr(_near_far_stats_062['p'].to_numpy(dtype=float))
        _near_far_stats_062['sig'] = _near_far_stats_062['q'].map(_sig_from_q)
        _near_far_plot_062 = pd.DataFrame(_near_far_plot_rows_062)

        if not _near_far_plot_062.empty:
            _step062e_name = 'Final_MV1_Step062E_TtoB_FunctionalScores_NearVsFar_Boxplots.png'
            fig, axes = plt.subplots(
                1, len(_functional_scores_062),
                figsize=STEP06_FIGURE5_SIZES[_step062e_name],
                squeeze=False
            )
            _near_far_palette_062 = {
                '0–<20 µm': '#2F6B9A',
                '20–<50 µm': '#6FA7C7',
                '≥50 µm': '#A7A7A7',
            }
            _display_order_062 = [_step06_display_t_subset(v) for v in plot_t_cell_labels]
            _group_offsets_062 = dict(zip(_distance_group_order_062, [-0.24, 0.0, 0.24]))
            for _panel_idx_062, _score_062 in enumerate(_functional_scores_062):
                ax = axes[0, _panel_idx_062]
                _score_plot_062 = _near_far_plot_062[_near_far_plot_062['Score'].eq(_score_062)]
                _score_cells_062 = _near_far_df_062.dropna(subset=[_score_062]).copy()
                sns.boxplot(
                    data=_score_cells_062,
                    x='T_Subset_Display', y=_score_062, hue='Distance_Group',
                    order=_display_order_062,
                    hue_order=_distance_group_order_062,
                    palette=_near_far_palette_062, showfliers=False, width=0.68,
                    linewidth=0.85, saturation=0.82, ax=ax
                )
                _handles_062, _labels_062 = ax.get_legend_handles_labels()
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                _x_lookup_062 = {label: idx for idx, label in enumerate(_display_order_062)}
                for (_sub_display_062, _tissue_062), _line_df_062 in _score_plot_062.groupby(
                    ['T_Subset_Display', 'Tissue'], observed=True
                ):
                    _x_center_062 = _x_lookup_062.get(_sub_display_062)
                    if _x_center_062 is None:
                        continue
                    _line_df_062 = _line_df_062.copy()
                    _line_df_062['Distance_Group'] = pd.Categorical(
                        _line_df_062['Distance_Group'],
                        categories=_distance_group_order_062,
                        ordered=True,
                    )
                    _line_df_062 = _line_df_062.sort_values('Distance_Group')
                    _tissue_order_062 = sorted(
                        _score_plot_062.loc[
                            _score_plot_062['T_Subset_Display'].eq(_sub_display_062),
                            'Tissue'
                        ].astype(str).unique()
                    )
                    _jitter_lookup_062 = {
                        tissue: jitter for tissue, jitter in zip(
                            _tissue_order_062,
                            np.linspace(-0.028, 0.028, max(len(_tissue_order_062), 1))
                        )
                    }
                    _x_values_062 = [
                        _x_center_062 + _group_offsets_062[str(group)] + _jitter_lookup_062.get(str(_tissue_062), 0.0)
                        for group in _line_df_062['Distance_Group']
                    ]
                    _y_values_062 = _line_df_062['Tissue_Median'].to_numpy(dtype=float)
                    if len(_line_df_062) >= 2:
                        ax.plot(
                            _x_values_062, _y_values_062,
                            color='#5F5F5F', linewidth=0.68, alpha=0.34, zorder=4
                        )
                    for _x_point_062, _y_point_062, _group_062 in zip(
                        _x_values_062, _y_values_062, _line_df_062['Distance_Group']
                    ):
                        ax.scatter(
                            [_x_point_062], [_y_point_062],
                            s=16, c=_near_far_palette_062[str(_group_062)],
                            edgecolors='white', linewidths=0.40, alpha=0.95, zorder=5
                        )
                ax.set_title(
                    _score_062.replace('_', ' '),
                    fontsize=14.0, fontweight='bold', pad=8
                )
                ax.set_xlabel('T-cell subtype', fontsize=11.4, fontweight='bold')
                ax.set_ylabel('T-cell score (cell-level z-score)', fontsize=11.4, fontweight='bold')
                ax.tick_params(axis='x', rotation=23, labelsize=9.5)
                ax.tick_params(axis='y', labelsize=9.7)
                plt.setp(ax.get_xticklabels(), ha='right', fontweight='bold')
                ax.grid(axis='y', linestyle='--', alpha=0.28)

                _panel_stats_062 = _near_far_stats_062[_near_far_stats_062['Score'].eq(_score_062)]
                _y_min_062, _y_max_062 = ax.get_ylim()
                _y_span_062 = max(_y_max_062 - _y_min_062, 0.2)
                ax.set_ylim(_y_min_062, _y_max_062 + 0.46 * _y_span_062)
                for _x_idx_062, _subtype_062 in enumerate(plot_t_cell_labels):
                    _sub_stats_062 = _panel_stats_062[
                        _panel_stats_062['T_Subset'].eq(_subtype_062)
                    ].copy()
                    for _pair_idx_062, (_group_a_062, _group_b_062) in enumerate(
                        combinations(_distance_group_order_062, 2)
                    ):
                        _stat_row_062 = _sub_stats_062[
                            _sub_stats_062['Group_A'].eq(_group_a_062)
                            & _sub_stats_062['Group_B'].eq(_group_b_062)
                        ]
                        if _stat_row_062.empty:
                            continue
                        _stat_row_062 = _stat_row_062.iloc[0]
                        _q_062 = _stat_row_062['q']
                        _q_text_062 = 'q=n/a' if not np.isfinite(_q_062) else (
                            'q<0.001' if _q_062 < 0.001 else f'q={_q_062:.3f}'
                        )
                        _sig_text_062 = (
                            'n/a' if not np.isfinite(_q_062) else str(_stat_row_062['sig'])
                        )
                        _bracket_y_062 = _y_max_062 + (0.045 + 0.115 * _pair_idx_062) * _y_span_062
                        _bracket_h_062 = 0.018 * _y_span_062
                        _x_a_062 = _x_idx_062 + _group_offsets_062[_group_a_062]
                        _x_b_062 = _x_idx_062 + _group_offsets_062[_group_b_062]
                        ax.plot(
                            [_x_a_062, _x_a_062, _x_b_062, _x_b_062],
                            [_bracket_y_062, _bracket_y_062 + _bracket_h_062,
                             _bracket_y_062 + _bracket_h_062, _bracket_y_062],
                            color='black', linewidth=0.72, clip_on=False
                        )
                        ax.text(
                            (_x_a_062 + _x_b_062) / 2,
                            _bracket_y_062 + 1.15 * _bracket_h_062,
                            f"{_sig_text_062} {_q_text_062}; n={int(_stat_row_062['Paired_tissues_n'])}",
                            ha='center', va='bottom', fontsize=6.1,
                            fontweight='bold', clip_on=False
                        )

            fig.legend(
                _handles_062[:3], _labels_062[:3],
                loc='lower center', bbox_to_anchor=(0.5, 0.025),
                ncol=3, frameon=False, fontsize=9.8,
                title='Distance to nearest B cell', title_fontsize=10.3
            )
            fig.suptitle(
                'Functional scores by distance to the nearest B cell',
                fontsize=15.0, fontweight='bold', y=0.985
            )
            fig.text(
                0.5, 0.905,
                'Boxes = individual T cells; dots/lines = tissue medians; paired Wilcoxon tests with BH-FDR',
                ha='center', va='center', fontsize=9.2
            )
            # Keep the final subtype's statistical label inside the designed
            # canvas; the annotation is centered on the last x position and
            # therefore needs a small right gutter beyond the axes.
            fig.subplots_adjust(left=0.060, right=0.985, bottom=0.225, top=0.76, wspace=0.16)
            _save_step06_matplotlib(
                fig, _step062e_name,
                'Step06-2E tissue-level functional-score near-versus-far boxplots'
            )

if STEP06_WRITE_INDIVIDUAL_PNG:
    _save_step062_static_pngs()

if STEP06_DISPLAY_INTERACTIVE_FIGURES:
    if STEP06_EXPORT_TTOB_ENRICHMENT_PANEL:
        _display_step06_figure(fig_freq)
    _display_step06_figure(fig_dist)
    if STEP06_EXPORT_TTOB_ENGAGEMENT_PANEL:
        _display_step06_figure(fig_subset)
    _display_step06_figure(fig_expr)



# %%
# ==============================================================================
# [Step 06-3] T cell Activation & Exhaustion Spatial Mapping (Subplot Visualization)
# ▷ Generate B-cell density contours and background texture to visually emphasize B-T cell distances
# ▷ Provide tissue-sample-specific side-by-side views (Activation vs Exhaustion)
# ▷ Replace point rasterization with Scattergl to fully support legend click toggling
# ==============================================================================

if not step05_skip:
    print("🔹 [Step 06-3] Generating sample-specific side-by-side functional spatial maps (linked dropdown & legend)...")
    
    x_col = next((col for col in ['X_centroid','CenterX','X','x','x_centroid'] if col in adata.obs.columns), None)
    y_col = next((col for col in ['Y_centroid','CenterY','Y','y','y_centroid'] if col in adata.obs.columns), None)
    
    all_obs_06 = adata.obs[adata.obs['Disease'].str.contains('DLBCL', na=False)].copy()
    if 'DLBCL1_02' in all_obs_06['Run_Tissue_name'].values:
        all_obs_06 = all_obs_06[all_obs_06['Run_Tissue_name'] != 'DLBCL1_02']
        
    unique_samples = sorted(all_obs_06['Run_Tissue_name'].unique())
    
    # 📌 1. Extract cells and map scores
    _b_labels = _get_step06_b_cell_labels(all_obs_06['Niche_Label'].unique())
    b_df = all_obs_06[all_obs_06['Niche_Label'].isin(_b_labels)].copy()
    
    # [Added] Prefer HLA-DR status (hla_pos_idx) already calculated in 06-2 if it is in memory
    b_df['B_State'] = 'B cell'
    if 'hla_pos_idx' in globals() or 'hla_pos_idx' in locals():
        b_df.loc[b_df.index.intersection(hla_pos_idx), 'B_State'] = 'B cell: HLA-DR (+)'
        b_df.loc[b_df.index.intersection(hla_neg_idx), 'B_State'] = 'B cell: HLA-DR (-)'
    else:
        # Fall back to local calculation only if the global variable is absent in a particular environment
        hla_marker = next((m for m in ['HLA-DRA', 'HLA-DRB1', 'HLA-DR'] if m in adata.var_names), None)
        if hla_marker:
            _X = adata[b_df.index, hla_marker].X
            b_expr = np.asarray(_X.todense()).flatten() if hasattr(_X, 'todense') else np.asarray(_X).flatten()
            b_df['HLA_DR_Expr'] = b_expr
            gmm = GaussianMixture(n_components=2, random_state=42).fit(b_expr.reshape(-1, 1))
            high_cluster = np.argmax(gmm.means_)
            b_df['B_State'] = np.where(gmm.predict(b_expr.reshape(-1, 1)) == high_cluster, 'B cell: HLA-DR (+)', 'B cell: HLA-DR (-)')
    
    t_idx = dist_df.dropna(subset=['Exhaustion_Score', 'Activation_Score']).index.intersection(all_obs_06.index)
    t_cells = all_obs_06.loc[t_idx].copy()
    t_cells['Act_Score'] = dist_df.loc[t_idx, 'Activation_Score']
    t_cells['Exh_Score'] = dist_df.loc[t_idx, 'Exhaustion_Score']
    
    # 📌 2. Create the subplot frame for rendering
    fig_score = make_subplots(rows=1, cols=2,
                              subplot_titles=("<b>[Left] T-cell Activation Map</b>", "<b>[Right] T-cell Exhaustion Map</b>"),
                              horizontal_spacing=0.05, shared_yaxes=True, shared_xaxes=True)
    
    act_cmin, act_cmax = -1.5, 1.5
    exh_cmin, exh_cmax = -1.5, 1.5
    
    fig_score.update_layout(
        coloraxis=dict(colorscale='Blues', cmin=act_cmin, cmax=act_cmax, colorbar=dict(title='Activation<br>Z-Score', x=-0.08, len=0.8)),
        coloraxis2=dict(colorscale='Reds', cmin=exh_cmin, cmax=exh_cmax, colorbar=dict(title='Exhaustion<br>Z-Score', x=1.03, len=0.8)),
        legend=dict(itemsizing='constant', title="<b>INDEX (Click to Toggle)</b>", yanchor="top", y=1, xanchor="left", x=1.13)
    )
    
    visibility_map = {s: [] for s in unique_samples}
    global_trace_idx = 0
    t_cell_subsets = [label for label in t_cell_labels if label in set(t_cells['Niche_Label'].dropna().astype(str))]

    # 📌 3. Render sample-specific traces
    for i, s in enumerate(tqdm(unique_samples, desc="Generate sample-specific spatial maps")):
        b_df_s = b_df[b_df['Run_Tissue_name'] == s]
        t_df_s = t_cells[t_cells['Run_Tissue_name'] == s]
        is_first = (i == 0)
        
        # 🟢 [Layer 1] B-cell background (meaningful color mapping based on HLA-DR expression)
        # Show tumor regions clearly with green HLA-DR(+) and gray-white (sand-colored) HLA-DR(-) cells instead of contours.
        for b_state in sorted(b_df_s['B_State'].unique()):
            sub_b = b_df_s[b_df_s['B_State'] == b_state]
            if len(sub_b) == 0: continue
            
            b_x, b_y = sub_b[x_col].round(2).values, sub_b[y_col].round(2).values
            
            # [Change] Separate colors and improve background visibility with complementary colors distinct from blue/red T cells
            # HLA-DR(+) = gold/yellow, HLA-DR(-) = pale pastel purple
            b_color = '#F1C40F' if '+' in b_state else '#D2B4DE' 
            lg_name = f"{s}_{b_state}" # Prevent state-crossing bugs
            
            # [Change 1] Increase size to 5.0 to preserve visible background-cell shapes when zooming in
            b_state_display = b_state.replace('B cell:', 'B cells:')
            tr_b_L = go.Scattergl(x=b_x, y=b_y, mode='markers',
                                  marker=dict(color=b_color, size=5.0, opacity=0.45, symbol='square'), 
                                  name=b_state_display, legendgroup=lg_name, showlegend=True, visible=is_first, hoverinfo='skip')
            fig_score.add_trace(tr_b_L, row=1, col=1)
            visibility_map[s].append(global_trace_idx); global_trace_idx += 1
            
            tr_b_R = go.Scattergl(x=b_x, y=b_y, mode='markers',
                                  marker=dict(color=b_color, size=5.0, opacity=0.45, symbol='square'), 
                                  name=b_state_display, legendgroup=lg_name, showlegend=False, visible=is_first, hoverinfo='skip')
            fig_score.add_trace(tr_b_R, row=1, col=2)
            visibility_map[s].append(global_trace_idx); global_trace_idx += 1

        # 🟢 [Layer 2] Color mapping for specific T-cell types
        for subset in t_cell_subsets:
            sub_df = t_df_s[t_df_s['Niche_Label'] == subset]
            if len(sub_df) == 0: continue
            
            sx, sy = sub_df[x_col].round(2).values, sub_df[y_col].round(2).values
            s_act = sub_df['Act_Score'].round(3).values
            s_exh = sub_df['Exh_Score'].round(3).values
            
            hover_act = [f"<b>{subset}</b><br>Act Score: {a:.2f}" for a in s_act]
            hover_exh = [f"<b>{subset}</b><br>Exh Score: {e:.2f}" for e in s_exh]
            lg_name = f"{s}_{subset}" 
            
            # [Change 2] Match outline colors exactly to the Step 06-2 distribution (Gradient) plot colors
            _step062_colors = px.colors.qualitative.D3 + px.colors.qualitative.Bold
            _ref_labels = t_cell_labels if 't_cell_labels' in globals() or 't_cell_labels' in locals() else t_cell_subsets
            try:
                _c_idx = _ref_labels.index(subset)
            except ValueError:
                _c_idx = t_cell_subsets.index(subset)
            subset_color = _step062_colors[_c_idx % len(_step062_colors)]
            
            # Left: Activation Map
            tr_t_act = go.Scattergl(x=sx, y=sy, mode='markers', 
                                    marker=dict(color=s_act, coloraxis='coloraxis', size=7.5, symbol='circle', opacity=0.9, line=dict(width=1.5, color=subset_color)), 
                                    text=hover_act, hoverinfo='text', name=f"T cells: {subset}", legendgroup=lg_name, showlegend=True, visible=is_first)
            fig_score.add_trace(tr_t_act, row=1, col=1)
            visibility_map[s].append(global_trace_idx); global_trace_idx += 1
            
            # Right: Exhaustion Map
            tr_t_exh = go.Scattergl(x=sx, y=sy, mode='markers', 
                                    marker=dict(color=s_exh, coloraxis='coloraxis2', size=7.5, symbol='circle', opacity=0.9, line=dict(width=1.5, color=subset_color)), 
                                    text=hover_exh, hoverinfo='text', name=f"T cells: {subset}", legendgroup=lg_name, showlegend=False, visible=is_first)
            fig_score.add_trace(tr_t_exh, row=1, col=2)
            visibility_map[s].append(global_trace_idx); global_trace_idx += 1

    # 📌 4. Dropdown and [Change 3] linked horizontal/vertical XY axes (matches)
    dropdown_buttons = []
    for s in unique_samples:
        vis_array = [False] * global_trace_idx
        for idx in visibility_map[s]: vis_array[idx] = True
        
        x_vals = all_obs_06.loc[all_obs_06['Run_Tissue_name'] == s, x_col]
        y_vals = all_obs_06.loc[all_obs_06['Run_Tissue_name'] == s, y_col]
        x_range = [x_vals.min() - 50, x_vals.max() + 50]
        y_range = [y_vals.max() + 50, y_vals.min() - 50] # Reverse the Y-axis to match the tissue image
        
        dropdown_buttons.append(dict(
            label=f"Sample: {s}",
            method="update",
            args=[{"visible": vis_array}, 
                  {"xaxis.range": x_range, "yaxis.range": y_range, 
                   "xaxis2.range": x_range, "yaxis2.range": y_range,
                   "title.text": f"<b>[Step 06-3] T-cell/B-cell Spatial Interactions ({s})</b>"
                  }]
        ))

    init_s = unique_samples[0]
    init_x = all_obs_06.loc[all_obs_06['Run_Tissue_name'] == init_s, x_col]
    init_y = all_obs_06.loc[all_obs_06['Run_Tissue_name'] == init_s, y_col]
    score_dropdown_menus = [dict(active=0, buttons=dropdown_buttons, x=0.01, y=1.12, xanchor="left", yanchor="bottom")]
    
    fig_score.update_layout(
        updatemenus=score_dropdown_menus,
        margin=dict(t=120), # Add top margin to prevent title overlap
        width=1650, height=800, plot_bgcolor='white', hovermode='closest',
        title=dict(
            text=f"<b>[Step 06-3] T-cell/B-cell Spatial Interactions ({init_s})</b>",
            y=0.98, x=0.5, xanchor='center', yanchor='top'
        )
    )
    
    # Apply matches to synchronize X/Y drag zooming fully across the left and right subplots
    fig_score.update_xaxes(showgrid=False, zeroline=False, mirror=True, linecolor='black', matches='x')
    fig_score.update_yaxes(showgrid=False, zeroline=False, mirror=True, linecolor='black', matches='y', scaleanchor="x", scaleratio=1)

    def _apply_step063_sample_view(sample_name):
        """Apply sample-specific trace visibility and axis ranges for PNG saving instead of dropdown viewing."""
        sample_vis = [False] * global_trace_idx
        for trace_idx in visibility_map[sample_name]:
            sample_vis[trace_idx] = True
        for trace_idx, is_visible in enumerate(sample_vis):
            fig_score.data[trace_idx].visible = is_visible

        x_vals = all_obs_06.loc[all_obs_06['Run_Tissue_name'] == sample_name, x_col]
        y_vals = all_obs_06.loc[all_obs_06['Run_Tissue_name'] == sample_name, y_col]
        x_range = [x_vals.min() - 50, x_vals.max() + 50]
        y_range = [y_vals.max() + 50, y_vals.min() - 50]
        fig_score.update_layout(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            xaxis2=dict(range=x_range),
            yaxis2=dict(range=y_range),
            title=dict(
                text=f"<b>[Step 06-3] T-cell/B-cell Spatial Interactions ({sample_name})</b>",
                y=0.98, x=0.5, xanchor='center', yanchor='top'
            )
        )

    def _save_step063_sample_static_png(sample_name):
        """Save sample-specific Activation/Exhaustion spatial maps as Matplotlib PNGs."""
        b_df_s = b_df[b_df['Run_Tissue_name'] == sample_name]
        t_df_s = t_cells[t_cells['Run_Tissue_name'] == sample_name]
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), sharex=True, sharey=True)
        panel_specs = [
            (axes[0], 'Act_Score', 'Activation score', 'Blues', act_cmin, act_cmax),
            (axes[1], 'Exh_Score', 'Exhaustion score', 'Reds', exh_cmin, exh_cmax),
        ]
        subset_palette = {
            subset: (px.colors.qualitative.D3 + px.colors.qualitative.Bold)[idx % len(px.colors.qualitative.D3 + px.colors.qualitative.Bold)]
            for idx, subset in enumerate(t_cell_subsets)
        }

        scatter_for_cbar = {}
        for ax, score_col, score_title, cmap_name, cmin, cmax in panel_specs:
            for b_state in sorted(b_df_s['B_State'].unique()):
                sub_b = b_df_s[b_df_s['B_State'] == b_state]
                if len(sub_b) == 0:
                    continue
                b_color = '#F1C40F' if '+' in b_state else '#D2B4DE'
                ax.scatter(
                    sub_b[x_col], sub_b[y_col],
                    s=7, c=b_color, marker='s', alpha=0.35,
                    linewidths=0, rasterized=True,
                    label=b_state.replace('B cell:', 'B cells:')
                )
            if len(t_df_s) > 0:
                edge_cols = [subset_palette.get(v, '#555555') for v in t_df_s['Niche_Label']]
                sc_plot = ax.scatter(
                    t_df_s[x_col], t_df_s[y_col],
                    c=t_df_s[score_col], cmap=cmap_name, vmin=cmin, vmax=cmax,
                    s=16, marker='o', alpha=0.9,
                    edgecolors=edge_cols, linewidths=0.35,
                    rasterized=True
                )
                scatter_for_cbar[score_col] = sc_plot
            ax.set_title(score_title, fontsize=13.5, fontweight='bold', pad=7)
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()
            ax.set_xlabel('X centroid', fontsize=11)
            ax.grid(False)
        axes[0].set_ylabel('Y centroid', fontsize=11)

        if 'Act_Score' in scatter_for_cbar:
            cbar_act = fig.colorbar(scatter_for_cbar['Act_Score'], ax=axes[0], fraction=0.046, pad=0.02)
            cbar_act.set_label('Activation Z-score', fontsize=10, fontweight='bold')
        if 'Exh_Score' in scatter_for_cbar:
            cbar_exh = fig.colorbar(scatter_for_cbar['Exh_Score'], ax=axes[1], fraction=0.046, pad=0.02)
            cbar_exh.set_label('Exhaustion Z-score', fontsize=10, fontweight='bold')

        legend_handles = [
            mlines.Line2D([], [], marker='s', linestyle='', color='#F1C40F', alpha=0.6, label='B cells: HLA-DR (+)'),
            mlines.Line2D([], [], marker='s', linestyle='', color='#D2B4DE', alpha=0.6, label='B cells: HLA-DR (-)'),
        ]
        legend_handles.extend([
            mlines.Line2D([], [], marker='o', linestyle='', markerfacecolor='white',
                          markeredgecolor=subset_palette.get(subset, '#555555'), markeredgewidth=1.5,
                          label=f'T edge: {subset}')
            for subset in t_cell_subsets
        ])
        axes[1].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.12, 0.5),
                       fontsize=7.4, frameon=True, title='Index', title_fontsize=8.4)
        fig.suptitle(f"T-cell/B-cell spatial interactions ({sample_name})", fontsize=15.0, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 0.86, 0.92])
        sample_safe = _safe_step06_filename(sample_name)
        _save_step06_matplotlib(
            fig,
            f"Final_MV1_Step063_Functional_Maps_{sample_safe}.png",
            f"Step06-3 functional map ({sample_name})"
        )

    if STEP06_WRITE_INDIVIDUAL_PNG:
        for sample_name in unique_samples:
            _save_step063_sample_static_png(sample_name)

    _apply_step063_sample_view(init_s)
    _display_step06_figure(fig_score)


# %%
# ==============================================================================
# [Step 06-4] Checkpoint Save Deferred
# ▷ Check STEP06_SAVE_CHECKPOINT only at the end of the file after incorporating Step06-5/6.
# ==============================================================================
print("🔹 [Step 06-4] Skip intermediate checkpoint saving and defer to the final Step06 save condition.")


# %%
# ==============================================================================
# [Step 06-5] PhenoImager-style Nearest B-cell Assignment
# ▷ Purpose: assign each T cell to a single nearest HLA-DR(+/-) B-cell type, as in the existing PhenoImager analysis
# ▷ Inputs: Dist_to_Pos and Dist_to_Neg calculated in Step 06-2
# ▷ Preserved recalculation outputs: Figures in a separately authorized path and derived Step06 checkpoint columns
# ==============================================================================
print("[Step 06-5] Running PhenoImager-style nearest-B-cell assignment analysis code...")

required_065_cols = ['Dist_to_Pos', 'Dist_to_Neg']
missing_065_cols = [c for c in required_065_cols if c not in adata.obs.columns]
if missing_065_cols:
    raise KeyError(f"[Step 06-5] Required distance columns are missing: {missing_065_cols}. Run Step 06-2 first.")

# Match PhenoImager Step3/3B: nearest-B assignment panels use a fixed 50um cutoff instead of the dynamic hot zone.
NEAREST_B_CUTOFF = float(STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM)
adata.uns['STEP06_PHENOIMAGER_DISTANCE_CUTOFF_UM'] = NEAREST_B_CUTOFF
if 'INTERACT_CUTOFF' in globals() or 'INTERACT_CUTOFF' in adata.uns:
    _step06_hotzone = float(globals().get('INTERACT_CUTOFF', adata.uns.get('INTERACT_CUTOFF', np.nan)))
    print(f"[Step 06-5] Retain the Step06 hot zone ({_step06_hotzone:.0f}um) and calculate PhenoImager-style nearest-B panels using {NEAREST_B_CUTOFF:.0f}um.")

if 'dist_df' in globals() and all(c in dist_df.columns for c in ['T_Subset', 'Dist_to_Pos', 'Dist_to_Neg']):
    nearest_b_df = dist_df[['T_Subset', 'Dist_to_Pos', 'Dist_to_Neg']].copy()
else:
    if 't_cell_labels' not in globals():
        t_cell_labels = _get_step06_t_cell_labels(adata.obs['Niche_Label'].dropna().unique())
    t_mask_065 = adata.obs['Niche_Label'].isin(t_cell_labels) & adata.obs['Dist_to_Pos'].notna()
    nearest_b_df = adata.obs.loc[t_mask_065, ['Niche_Label', 'Dist_to_Pos', 'Dist_to_Neg']].copy()
    nearest_b_df = nearest_b_df.rename(columns={'Niche_Label': 'T_Subset'})

nearest_b_df = nearest_b_df[nearest_b_df['T_Subset'].isin(t_cell_labels)].copy()
if nearest_b_df.empty:
    raise ValueError("[Step 06-5] No T-cell subset rows are available for nearest-B-cell analysis.")

nearest_b_df['Distance_to_Nearest_B_HLA_DR_pos'] = nearest_b_df['Dist_to_Pos']
nearest_b_df['Distance_to_Nearest_B_HLA_DR_neg'] = nearest_b_df['Dist_to_Neg']
nearest_b_df['Distance_to_Nearest_B'] = nearest_b_df[
    ['Distance_to_Nearest_B_HLA_DR_pos', 'Distance_to_Nearest_B_HLA_DR_neg']
].min(axis=1, skipna=True)
pos_dist_065 = nearest_b_df['Dist_to_Pos'].to_numpy(dtype=float)
neg_dist_065 = nearest_b_df['Dist_to_Neg'].to_numpy(dtype=float)
pos_finite_065 = np.isfinite(pos_dist_065)
neg_finite_065 = np.isfinite(neg_dist_065)
nearest_b_df['Distance_delta_HLA_DRpos_minus_DRneg_um'] = pos_dist_065 - neg_dist_065

# Select only the closer B-cell state. Record exact distance ties as Tie without
# arbitrary assignment, and exclude them from nearest-only composition and tests.
exact_tie_065 = pos_finite_065 & neg_finite_065 & (pos_dist_065 == neg_dist_065)
nearest_b_df['Nearest_B_Type'] = np.select(
    [
        exact_tie_065,
        ~exact_tie_065 & pos_finite_065 & (~neg_finite_065 | (pos_dist_065 < neg_dist_065)),
        ~exact_tie_065 & neg_finite_065 & (~pos_finite_065 | (neg_dist_065 < pos_dist_065)),
    ],
    ['Tie', 'HLA-DR+ B Cell', 'HLA-DR- B Cell'],
    default='Unavailable'
)
nearest_b_df['Nearest_B_Within_Cutoff'] = (
    nearest_b_df['Distance_to_Nearest_B'].notna()
    & np.isfinite(nearest_b_df['Distance_to_Nearest_B'])
    & (nearest_b_df['Distance_to_Nearest_B'] <= NEAREST_B_CUTOFF)
)

# Save derived columns in obs using the same names as PhenoImager for direct comparison in later steps.
adata.obs['Distance_to_Nearest_B'] = np.nan
adata.obs['Distance_to_Nearest_B_HLA_DR_pos'] = np.nan
adata.obs['Distance_to_Nearest_B_HLA_DR_neg'] = np.nan
adata.obs['Distance_delta_HLA_DRpos_minus_DRneg_um'] = np.nan
adata.obs['Nearest_B_Type'] = 'Not_T_cell'
adata.obs['Nearest_B_Within_Cutoff'] = False
adata.obs.loc[nearest_b_df.index, 'Distance_to_Nearest_B'] = nearest_b_df['Distance_to_Nearest_B']
adata.obs.loc[nearest_b_df.index, 'Distance_to_Nearest_B_HLA_DR_pos'] = nearest_b_df['Distance_to_Nearest_B_HLA_DR_pos']
adata.obs.loc[nearest_b_df.index, 'Distance_to_Nearest_B_HLA_DR_neg'] = nearest_b_df['Distance_to_Nearest_B_HLA_DR_neg']
adata.obs.loc[nearest_b_df.index, 'Distance_delta_HLA_DRpos_minus_DRneg_um'] = nearest_b_df['Distance_delta_HLA_DRpos_minus_DRneg_um']
adata.obs.loc[nearest_b_df.index, 'Nearest_B_Type'] = nearest_b_df['Nearest_B_Type']
adata.obs.loc[nearest_b_df.index, 'Nearest_B_Within_Cutoff'] = nearest_b_df['Nearest_B_Within_Cutoff'].astype(bool)

near_t_065 = nearest_b_df[
    nearest_b_df['Nearest_B_Within_Cutoff']
    & nearest_b_df['Nearest_B_Type'].isin(['HLA-DR+ B Cell', 'HLA-DR- B Cell'])
].copy()

if near_t_065.empty:
    print(f"[Step 06-5] No nearest-B-cell T cells are within {NEAREST_B_CUTOFF:.0f}µm; skip figure generation.")
else:
    subtype_order_065 = [s for s in t_cell_labels if s in nearest_b_df['T_Subset'].unique()]
    if not subtype_order_065:
        raise ValueError("[Step 06-5] Cannot create a T-cell subset order for the figure.")

    total_by_subset_065 = nearest_b_df.groupby('T_Subset', observed=True).size()
    near_counts_065 = (
        near_t_065.groupby(['T_Subset', 'Nearest_B_Type'], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for col in ['HLA-DR+ B Cell', 'HLA-DR- B Cell']:
        if col not in near_counts_065.columns:
            near_counts_065[col] = 0

    stats_rows_065 = []
    distance_summary_rows_065 = []
    for t_sub in subtype_order_065:
        sub_all = nearest_b_df[nearest_b_df['T_Subset'] == t_sub]
        sub_near = near_t_065[near_t_065['T_Subset'] == t_sub]
        d_pos_nearest = sub_near.loc[sub_near['Nearest_B_Type'] == 'HLA-DR+ B Cell', 'Distance_to_Nearest_B'].dropna()
        d_neg_nearest = sub_near.loc[sub_near['Nearest_B_Type'] == 'HLA-DR- B Cell', 'Distance_to_Nearest_B'].dropna()
        p_mwu = (
            mannwhitneyu(d_pos_nearest, d_neg_nearest, alternative='two-sided')[1]
            if len(d_pos_nearest) > 5 and len(d_neg_nearest) > 5 else np.nan
        )
        median_pos_065 = d_pos_nearest.median() if len(d_pos_nearest) else np.nan
        median_neg_065 = d_neg_nearest.median() if len(d_neg_nearest) else np.nan
        mean_pos_065 = d_pos_nearest.mean() if len(d_pos_nearest) else np.nan
        mean_neg_065 = d_neg_nearest.mean() if len(d_neg_nearest) else np.nan

        stats_rows_065.append({
            'T_Subset': t_sub,
            'Total_T_cells': int(len(sub_all)),
            'Nearest_B_within_cutoff_n': int(len(sub_near)),
            'HLA_DRpos_nearest_n': int(len(d_pos_nearest)),
            'HLA_DRneg_nearest_n': int(len(d_neg_nearest)),
            'HLA_DRpos_nearest_pct_total': float(len(d_pos_nearest) / max(len(sub_all), 1) * 100),
            'HLA_DRneg_nearest_pct_total': float(len(d_neg_nearest) / max(len(sub_all), 1) * 100),
            'Median_distance_to_nearest_DRpos_B_um': float(median_pos_065) if np.isfinite(median_pos_065) else np.nan,
            'Median_distance_to_nearest_DRneg_B_um': float(median_neg_065) if np.isfinite(median_neg_065) else np.nan,
            'Median_distance_delta_DRpos_minus_DRneg_um': float(median_pos_065 - median_neg_065) if np.isfinite(median_pos_065) and np.isfinite(median_neg_065) else np.nan,
            'Mean_distance_to_nearest_DRpos_B_um': float(mean_pos_065) if np.isfinite(mean_pos_065) else np.nan,
            'Mean_distance_to_nearest_DRneg_B_um': float(mean_neg_065) if np.isfinite(mean_neg_065) else np.nan,
            'Mean_distance_delta_DRpos_minus_DRneg_um': float(mean_pos_065 - mean_neg_065) if np.isfinite(mean_pos_065) and np.isfinite(mean_neg_065) else np.nan,
            'Distance_p_MWU': p_mwu
        })

        distance_sets_065 = {
            'Nearest B (any)': sub_all.loc[sub_all['Distance_to_Nearest_B'] <= NEAREST_B_CUTOFF, 'Distance_to_Nearest_B'].dropna(),
            'HLA-DR+ B': sub_all.loc[sub_all['Distance_to_Nearest_B_HLA_DR_pos'] <= NEAREST_B_CUTOFF, 'Distance_to_Nearest_B_HLA_DR_pos'].dropna(),
            'HLA-DR- B': sub_all.loc[sub_all['Distance_to_Nearest_B_HLA_DR_neg'] <= NEAREST_B_CUTOFF, 'Distance_to_Nearest_B_HLA_DR_neg'].dropna()
        }
        for target_name_065, dist_values_065 in distance_sets_065.items():
            distance_summary_rows_065.append({
                'T_Subset': t_sub,
                'B_cell_distance_target': target_name_065,
                'Cells_within_50um_n': int(len(dist_values_065)),
                'Mean_distance_um': float(dist_values_065.mean()) if len(dist_values_065) else np.nan,
                'Median_distance_um': float(dist_values_065.median()) if len(dist_values_065) else np.nan
            })

    nearest_stats_065 = pd.DataFrame(stats_rows_065)
    nearest_stats_065['Distance_q_BH_FDR'] = _bh_fdr(nearest_stats_065['Distance_p_MWU'].values)
    nearest_stats_065['Distance_sig_BH_FDR'] = nearest_stats_065['Distance_q_BH_FDR'].map(_sig_from_q)
    nearest_stats_path_065 = os.path.join(FINAL_TABLE_DIR, STEP06_NEAREST_B_STATS_CSV)
    _step06_write_table_if_enabled(nearest_stats_065, nearest_stats_path_065, index=False)
    print(f"[Step 06-5] Nearest-B-cell statistics table saved: {nearest_stats_path_065}")

    distance_summary_065 = pd.DataFrame(distance_summary_rows_065)
    distance_summary_path_065 = os.path.join(FINAL_TABLE_DIR, STEP06_NEAREST_B_DISTANCE_SUMMARY_CSV)
    _step06_write_table_if_enabled(distance_summary_065, distance_summary_path_065, index=False)
    print(f"[Step 06-5] PhenoImager-style distance summary table saved: {distance_summary_path_065}")

    fig_step065c_nearest_only_065 = None

    # Supplementary analysis: nearest-only sensitivity
    # Assign each T cell to only one nearest B-cell state to avoid counting the same
    # T cell repeatedly around multiple B cells in tissues with abundant B-cell anchors.
    nearest_only_state_order_065 = ['HLA-DR+ B Cell', 'HLA-DR- B Cell']
    nearest_only_state_labels_065 = {
        'HLA-DR+ B Cell': 'Nearest HLA-DR+ B cell\n≤50 µm',
        'HLA-DR- B Cell': 'Nearest HLA-DR- B cell\n≤50 µm'
    }
    nearest_only_assignment_065 = nearest_b_df.copy()
    nearest_only_assignment_065['NearestOnly_Assignment_Group'] = 'No nearest B within 50um'
    tie_within_cutoff_065 = (
        nearest_only_assignment_065['Nearest_B_Within_Cutoff']
        & nearest_only_assignment_065['Nearest_B_Type'].eq('Tie')
    )
    nearest_only_assignment_065.loc[
        tie_within_cutoff_065, 'NearestOnly_Assignment_Group'
    ] = 'Exact distance tie excluded'
    valid_nearest_only_065 = (
        nearest_only_assignment_065['Nearest_B_Within_Cutoff']
        & nearest_only_assignment_065['Nearest_B_Type'].isin(nearest_only_state_order_065)
    )
    nearest_only_assignment_065.loc[valid_nearest_only_065, 'NearestOnly_Assignment_Group'] = (
        nearest_only_assignment_065.loc[valid_nearest_only_065, 'Nearest_B_Type'].map({
            'HLA-DR+ B Cell': 'Nearest HLA-DR+ B <=50um',
            'HLA-DR- B Cell': 'Nearest HLA-DR- B <=50um'
        })
    )
    nearest_only_export_065 = nearest_only_assignment_065.reset_index()
    if 'Cell_Index' not in nearest_only_export_065.columns:
        nearest_only_export_065 = nearest_only_export_065.rename(columns={nearest_only_export_065.columns[0]: 'Cell_Index'})
    nearest_only_export_cols_065 = [
        'Cell_Index', 'T_Subset', 'Nearest_B_Type', 'Nearest_B_Within_Cutoff',
        'NearestOnly_Assignment_Group', 'Distance_to_Nearest_B',
        'Distance_to_Nearest_B_HLA_DR_pos', 'Distance_to_Nearest_B_HLA_DR_neg',
        'Distance_delta_HLA_DRpos_minus_DRneg_um'
    ]
    nearest_only_assignment_path_065 = os.path.join(FINAL_TABLE_DIR, STEP06_NEAREST_ONLY_ASSIGNMENT_CSV)
    _step06_write_table_if_enabled(
        nearest_only_export_065[nearest_only_export_cols_065],
        nearest_only_assignment_path_065,
        index=False,
    )
    print(f"[Step 06-5C] Nearest-only T-cell assignment table saved: {nearest_only_assignment_path_065}")

    nearest_only_core_065 = nearest_only_assignment_065[
        nearest_only_assignment_065['Nearest_B_Within_Cutoff']
        & nearest_only_assignment_065['Nearest_B_Type'].isin(nearest_only_state_order_065)
    ].copy()
    if nearest_only_core_065.empty:
        print("[Step 06-5C] No nearest-only assigned T cells are within 50um; skip the supplementary figure.")
    else:
        nearest_only_counts_065 = (
            pd.crosstab(nearest_only_core_065['Nearest_B_Type'], nearest_only_core_065['T_Subset'])
            .reindex(index=nearest_only_state_order_065, columns=subtype_order_065, fill_value=0)
        )
        nearest_only_pct_065 = nearest_only_counts_065.div(
            nearest_only_counts_065.sum(axis=1).replace(0, np.nan), axis=0
        ) * 100

        b_state_total_map_065 = {
            'HLA-DR+ B Cell': int(len(hla_pos_idx)) if 'hla_pos_idx' in globals() else np.nan,
            'HLA-DR- B Cell': int(len(hla_neg_idx)) if 'hla_neg_idx' in globals() else np.nan
        }
        b_state_total_series_065 = pd.Series(b_state_total_map_065, dtype=float).replace(0, np.nan)
        nearest_only_per_1000_b_065 = nearest_only_counts_065.div(b_state_total_series_065, axis=0) * 1000

        valid_chi_cols_065 = nearest_only_counts_065.columns[nearest_only_counts_065.sum(axis=0) > 0]
        if len(valid_chi_cols_065) > 1 and (nearest_only_counts_065[valid_chi_cols_065].sum(axis=1) > 0).all():
            chi2_nearest_only_065, p_nearest_only_065, dof_nearest_only_065, _ = chi2_contingency(
                nearest_only_counts_065[valid_chi_cols_065].values
            )
        else:
            chi2_nearest_only_065, p_nearest_only_065, dof_nearest_only_065 = np.nan, np.nan, np.nan

        # Cell-level subtype-specific Fisher tests complement the omnibus
        # state-by-subtype chi-square test. These are labeled explicitly in the
        # Figure so they are not interpreted as tissue-level inference.
        _fisher_rows_065 = []
        for t_sub_065 in subtype_order_065:
            _pos_sub_065 = int(nearest_only_counts_065.loc['HLA-DR+ B Cell', t_sub_065])
            _neg_sub_065 = int(nearest_only_counts_065.loc['HLA-DR- B Cell', t_sub_065])
            _pos_other_065 = int(nearest_only_counts_065.loc['HLA-DR+ B Cell'].sum() - _pos_sub_065)
            _neg_other_065 = int(nearest_only_counts_065.loc['HLA-DR- B Cell'].sum() - _neg_sub_065)
            _fisher_table_065 = np.array([
                [_pos_sub_065, _pos_other_065],
                [_neg_sub_065, _neg_other_065],
            ], dtype=int)
            if (_fisher_table_065.sum(axis=1) > 0).all() and (_fisher_table_065.sum(axis=0) > 0).all():
                _odds_065, _fisher_p_065 = fisher_exact(_fisher_table_065, alternative='two-sided')
            else:
                _odds_065, _fisher_p_065 = np.nan, np.nan
            _fisher_rows_065.append({
                'T_Subset': t_sub_065,
                'Odds_ratio': _odds_065,
                'p': _fisher_p_065,
            })
        _fisher_stats_065 = pd.DataFrame(_fisher_rows_065)
        _fisher_stats_065['q'] = _bh_fdr(_fisher_stats_065['p'].to_numpy(dtype=float))
        _fisher_stats_065['sig'] = _fisher_stats_065['q'].map(_sig_from_q)
        _fisher_q_map_065 = _fisher_stats_065.set_index('T_Subset')['q'].to_dict()
        _fisher_sig_map_065 = _fisher_stats_065.set_index('T_Subset')['sig'].to_dict()

        nearest_only_summary_rows_065 = []
        for b_state_065 in nearest_only_state_order_065:
            for t_sub_065 in subtype_order_065:
                nearest_only_summary_rows_065.append({
                    'Nearest_B_State': b_state_065,
                    'T_Subset': t_sub_065,
                    'Assigned_T_cells': int(nearest_only_counts_065.loc[b_state_065, t_sub_065]),
                    'Assigned_Tcell_pct_within_B_state': float(nearest_only_pct_065.loc[b_state_065, t_sub_065])
                    if pd.notna(nearest_only_pct_065.loc[b_state_065, t_sub_065]) else np.nan,
                    'B_state_total_n': b_state_total_map_065[b_state_065],
                    'Tcells_within_50um_per_1000_Bcells': float(nearest_only_per_1000_b_065.loc[b_state_065, t_sub_065])
                    if pd.notna(nearest_only_per_1000_b_065.loc[b_state_065, t_sub_065]) else np.nan,
                    'Assigned_Tcells_per_1000_Bcells': float(nearest_only_per_1000_b_065.loc[b_state_065, t_sub_065])
                    if pd.notna(nearest_only_per_1000_b_065.loc[b_state_065, t_sub_065]) else np.nan,
                    'Global_chi2_p_for_state_by_subset_table': p_nearest_only_065,
                    'Global_chi2_dof': dof_nearest_only_065
                })
        nearest_only_summary_065 = pd.DataFrame(nearest_only_summary_rows_065)
        nearest_only_summary_path_065 = os.path.join(FINAL_TABLE_DIR, STEP06_NEAREST_ONLY_COMPOSITION_CSV)
        _step06_write_table_if_enabled(nearest_only_summary_065, nearest_only_summary_path_065, index=False)
        print(f"[Step 06-5C] Adjusted nearest-only composition table saved: {nearest_only_summary_path_065}")

        _step065c_name = "Final_MV1_Step065C_NearestOnly_Tcell_To_Bstate_Composition.png"
        fig = plt.figure(figsize=STEP06_FIGURE5_SIZES[_step065c_name])
        # Place the plot and legend in separate rows within the generated canvas so
        # two-line X-tick labels and the legend do not share the same region.
        ax_comp_065 = fig.add_axes([0.120, 0.320, 0.865, 0.620])
        x_pos_065 = np.linspace(-0.28, 0.28, len(nearest_only_state_order_065))
        bottom_065 = np.zeros(len(nearest_only_state_order_065), dtype=float)
        for t_sub_065 in subtype_order_065:
            vals_065 = nearest_only_pct_065[t_sub_065].reindex(nearest_only_state_order_065).fillna(0).to_numpy(dtype=float)
            _fisher_q_065 = _fisher_q_map_065.get(t_sub_065, np.nan)
            _fisher_q_text_065 = 'q=n/a' if not np.isfinite(_fisher_q_065) else (
                'q<0.001' if _fisher_q_065 < 0.001 else f'q={_fisher_q_065:.3f}'
            )
            _legend_label_065 = (
                f"{_step06_display_t_subset(t_sub_065)}  "
                f"{_fisher_sig_map_065.get(t_sub_065, 'ns')} ({_fisher_q_text_065})"
            )
            ax_comp_065.bar(
                x_pos_065, vals_065, bottom=bottom_065,
                color=STEP06_T_SUBTYPE_COMPOSITION_COLORS.get(t_sub_065, '#999999'),
                edgecolor='white', linewidth=0.8, width=0.30, label=_legend_label_065
            )
            bottom_065 += vals_065
        ax_comp_065.set_xlim(-0.52, 0.52)
        ax_comp_065.set_xticks(x_pos_065)
        ax_comp_065.set_xticklabels(
            [nearest_only_state_labels_065[s] for s in nearest_only_state_order_065],
            fontsize=10.2, fontweight='bold'
        )
        ax_comp_065.set_ylabel('Nearest-assigned T cells (%)', fontsize=12.0, fontweight='bold')
        ax_comp_065.set_ylim(0, 100)
        ax_comp_065.set_title('Nearest-only T-cell composition', fontsize=14.0, fontweight='bold', pad=4)
        _legend_handles_065, _legend_labels_065 = ax_comp_065.get_legend_handles_labels()
        fig.legend(
            _legend_handles_065,
            _legend_labels_065,
            loc='lower center',
            bbox_to_anchor=(0.50, 0.110),
            ncol=3,
            fontsize=9.0,
            frameon=False,
            borderaxespad=0.0
        )
        ax_comp_065.grid(axis='y', linestyle='--', alpha=0.35)
        ax_comp_065.tick_params(axis='y', labelsize=10.4)
        plt.setp(ax_comp_065.get_yticklabels(), fontweight='bold')
        _omnibus_text_065 = (
            'Omnibus 2×6 chi-square p=n/a'
            if not np.isfinite(p_nearest_only_065)
            else (
                'Omnibus 2×6 chi-square p<0.001'
                if p_nearest_only_065 < 0.001
                else f'Omnibus 2×6 chi-square p={p_nearest_only_065:.3f}'
            )
        )
        fig.text(
            0.50, 0.014,
            f'Exact distance ties excluded (n={int(tie_within_cutoff_065.sum()):,}); '
            f'two-sided Fisher tests with BH-FDR; {_omnibus_text_065}',
            ha='center', va='bottom', fontsize=8.2, color='#444444'
        )
        fig_step065c_nearest_only_065 = fig

    # Main question: how do distances to HLA-DR(+) and HLA-DR(-) B cells differ for the same T cell?
    # Conservatively restrict the potential interaction range by comparing only T cells within 50um of both B-cell states.
    paired_distance_065 = nearest_b_df[
        nearest_b_df['Distance_to_Nearest_B_HLA_DR_pos'].le(NEAREST_B_CUTOFF)
        & nearest_b_df['Distance_to_Nearest_B_HLA_DR_neg'].le(NEAREST_B_CUTOFF)
        & np.isfinite(nearest_b_df['Distance_to_Nearest_B_HLA_DR_pos'].to_numpy(dtype=float))
        & np.isfinite(nearest_b_df['Distance_to_Nearest_B_HLA_DR_neg'].to_numpy(dtype=float))
    ].copy()
    paired_distance_065 = paired_distance_065[
        paired_distance_065['Distance_to_Nearest_B_HLA_DR_pos'].ne(
            paired_distance_065['Distance_to_Nearest_B_HLA_DR_neg']
        )
    ].copy()

    paired_stats_rows_065 = []
    for t_sub in subtype_order_065:
        sub_pair_065 = paired_distance_065[paired_distance_065['T_Subset'] == t_sub]
        d_pos_pair_065 = sub_pair_065['Distance_to_Nearest_B_HLA_DR_pos'].astype(float)
        d_neg_pair_065 = sub_pair_065['Distance_to_Nearest_B_HLA_DR_neg'].astype(float)
        delta_pair_065 = sub_pair_065['Distance_delta_HLA_DRpos_minus_DRneg_um'].astype(float)
        if len(delta_pair_065) > 5 and np.any(np.abs(delta_pair_065.to_numpy()) > 0):
            try:
                paired_p_065 = wilcoxon(d_pos_pair_065, d_neg_pair_065, alternative='two-sided')[1]
            except ValueError:
                paired_p_065 = np.nan
        else:
            paired_p_065 = np.nan
        paired_stats_rows_065.append({
            'T_Subset': t_sub,
            'Paired_T_cells_within_50um_n': int(len(sub_pair_065)),
            'Median_distance_to_HLADRpos_B_um': float(d_pos_pair_065.median()) if len(d_pos_pair_065) else np.nan,
            'Median_distance_to_HLADRneg_B_um': float(d_neg_pair_065.median()) if len(d_neg_pair_065) else np.nan,
            'Median_delta_HLADRpos_minus_HLADRneg_um': float(delta_pair_065.median()) if len(delta_pair_065) else np.nan,
            'Mean_delta_HLADRpos_minus_HLADRneg_um': float(delta_pair_065.mean()) if len(delta_pair_065) else np.nan,
            'Paired_Wilcoxon_p': paired_p_065
        })

    paired_stats_065 = pd.DataFrame(paired_stats_rows_065)
    paired_stats_065['Paired_Wilcoxon_q_BH_FDR'] = _bh_fdr(paired_stats_065['Paired_Wilcoxon_p'].values)
    paired_stats_065['Paired_Wilcoxon_sig_BH_FDR'] = paired_stats_065['Paired_Wilcoxon_q_BH_FDR'].map(_sig_from_q)
    paired_stats_path_065 = os.path.join(FINAL_TABLE_DIR, STEP06_PAIRED_B_DISTANCE_STATS_CSV)
    _step06_write_table_if_enabled(paired_stats_065, paired_stats_path_065, index=False)
    print(f"[Step 06-5] HLA-DR(+/-) paired-distance statistics table saved: {paired_stats_path_065}")

    # Step06-2B/2C already show nearest-B distances and hot-zone engagement; in Step06-5,
    # retain only paired panels directly comparing distances to HLA-DR+ and HLA-DR- B cells within the same T cell.
    print("[Step 06-5] Skip nearest-B box/rate panels that duplicate Step06-2B/2C.")

    if paired_distance_065.empty:
        print("[Step 06-5] No paired T cells are within 50um of both HLA-DR+/- B-cell states; skip the paired PNG panel.")
    else:
        paired_sig_label_map_065 = {
            row['T_Subset']: f"{row['T_Subset']}<br>(q:{row['Paired_Wilcoxon_sig_BH_FDR']})"
            for _, row in paired_stats_065.iterrows()
        }
        paired_plot_order_065 = [paired_sig_label_map_065.get(s, s) for s in subtype_order_065]
        paired_distance_065['T_Subset_Plot'] = paired_distance_065['T_Subset'].map(
            lambda x: paired_sig_label_map_065.get(x, str(x))
        )
        paired_long_065 = paired_distance_065.melt(
            id_vars=['T_Subset', 'T_Subset_Plot'],
            value_vars=['Distance_to_Nearest_B_HLA_DR_pos', 'Distance_to_Nearest_B_HLA_DR_neg'],
            var_name='B_Cell_HLA_DR_State',
            value_name='Distance_to_B_um'
        )
        paired_long_065['B_Cell_HLA_DR_State'] = paired_long_065['B_Cell_HLA_DR_State'].replace({
            'Distance_to_Nearest_B_HLA_DR_pos': 'Distance to HLA-DR+ B cell',
            'Distance_to_Nearest_B_HLA_DR_neg': 'Distance to HLA-DR- B cell'
        })

        fig_paired_distance_065 = go.Figure()
        for _state, _side, _color in [
            ('Distance to HLA-DR+ B cell', 'negative', '#d62728'),
            ('Distance to HLA-DR- B cell', 'positive', '#1f77b4')
        ]:
            _state_df = paired_long_065[paired_long_065['B_Cell_HLA_DR_State'] == _state]
            for _plot_label in paired_plot_order_065:
                _vals = _state_df.loc[_state_df['T_Subset_Plot'] == _plot_label, 'Distance_to_B_um'].dropna()
                if _vals.empty:
                    continue
                fig_paired_distance_065.add_trace(go.Violin(
                    x=[_plot_label] * len(_vals),
                    y=_vals,
                    name=_state,
                    legendgroup=_state,
                    scalegroup=_plot_label,
                    side=_side,
                    line_color=_color,
                    fillcolor=_color,
                    opacity=0.72,
                    box_visible=True,
                    meanline_visible=False,
                    points=False,
                    spanmode='hard',
                    showlegend=(_plot_label == paired_plot_order_065[0])
                ))
        fig_paired_distance_065.update_layout(
            title="<b>[Step 06-5A] Same T cells: distance to HLA-DR+ vs HLA-DR- B cells</b>",
            violinmode='overlay',
            violingap=0,
            plot_bgcolor='white',
            width=1350,
            height=560,
            legend_title_text='B-cell HLA-DR state'
        )
        fig_paired_distance_065.update_yaxes(
            title='Distance to B cell (um)',
            range=[0, 60],
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside'
        )
        fig_paired_distance_065.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')

        hm_median_065 = pd.DataFrame({
            'B cell (Any)': paired_distance_065.groupby('T_Subset', observed=True)['Distance_to_Nearest_B'].median(),
            'HLA-DR+ B cell': paired_distance_065.groupby('T_Subset', observed=True)['Distance_to_Nearest_B_HLA_DR_pos'].median(),
            'HLA-DR- B cell': paired_distance_065.groupby('T_Subset', observed=True)['Distance_to_Nearest_B_HLA_DR_neg'].median()
        }).reindex(subtype_order_065).T
        hm_median_display_065 = hm_median_065.rename(index={
            'B cell (Any)': 'B cells (Any)',
            'HLA-DR+ B cell': 'HLA-DR+ B cells',
            'HLA-DR- B cell': 'HLA-DR- B cells',
        }).rename(columns={
            'CD4_CTL': 'CD4 CTL',
            'CD8T': 'CD8 T',
        })
        hm_values_065 = hm_median_065.to_numpy(dtype=float)
        hm_finite_065 = hm_values_065[np.isfinite(hm_values_065)]
        if len(hm_finite_065) == 0:
            hm_zmin_065, hm_zmax_065 = 0.0, NEAREST_B_CUTOFF
        elif np.nanmin(hm_finite_065) == np.nanmax(hm_finite_065):
            hm_zmin_065 = max(0.0, float(np.nanmin(hm_finite_065)) - 1.0)
            hm_zmax_065 = float(np.nanmax(hm_finite_065)) + 1.0
        else:
            hm_zmin_065 = float(np.nanmin(hm_finite_065))
            hm_zmax_065 = float(np.nanmax(hm_finite_065))
        fig_distance_heatmap_065 = go.Figure(data=go.Heatmap(
            z=hm_median_065.values,
            x=hm_median_065.columns,
            y=hm_median_display_065.index,
            colorscale=[[0.0, '#551A8B'], [0.5, '#B65F2A'], [1.0, '#FFD700']],
            zmin=hm_zmin_065,
            zmax=hm_zmax_065,
            text=np.round(hm_median_065.values, 2),
            texttemplate='%{text:.2f}',
            colorbar=dict(title='Median<br>distance<br>(um)'),
            hovertemplate='T subset=%{x}<br>Target=%{y}<br>Median distance=%{z:.2f}um<extra></extra>'
        ))
        fig_distance_heatmap_065.update_layout(
            title="<b>[Step 06-5B] Median T-cell distance by B-cell HLA-DR state</b>",
            width=1050,
            height=450,
            plot_bgcolor='white'
        )

    def _save_step065_static_pngs():
        """Save only the Step06-5 paired HLA-DR+/- B-cell distance panel as PNG."""
        if paired_distance_065.empty:
            return

        paired_plot_df = paired_long_065.copy()
        paired_plot_df['T_Subset_Plot'] = paired_plot_df['T_Subset_Plot'].map(_plain_step06_label)
        paired_order_plain = [_plain_step06_label(v) for v in paired_plot_order_065]
        fig, ax = plt.subplots(figsize=(10.8, 4.4))
        sns.violinplot(
            data=paired_plot_df,
            x='T_Subset_Plot',
            y='Distance_to_B_um',
            hue='B_Cell_HLA_DR_State',
            order=paired_order_plain,
            hue_order=['Distance to HLA-DR+ B cell', 'Distance to HLA-DR- B cell'],
            palette={'Distance to HLA-DR+ B cell': '#d62728', 'Distance to HLA-DR- B cell': '#1f77b4'},
            split=True,
            inner='box',
            cut=0,
            linewidth=0.8,
            ax=ax
        )
        ax.set_title("Same T cells: distance to HLA-DR+ vs HLA-DR- B cells", fontsize=13.2, fontweight='bold', pad=7)
        ax.set_xlabel('')
        ax.set_ylabel('T-cell-to-B-cell distance (um)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 60)
        ax.tick_params(axis='x', rotation=30, labelsize=8.6)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(title='B-cell HLA-DR state', frameon=True, fontsize=9.2, title_fontsize=10.0)
        _save_step06_matplotlib(fig, "Final_MV1_Step065A_HLADR_Paired_Distance.png", "Step06-5A paired HLA-DR+/- B distance")

        _step065b_name = "Final_MV1_Step065B_Tcell_Bstate_Median_Distance_Heatmap.png"
        fig, ax = plt.subplots(figsize=STEP06_FIGURE5_SIZES[_step065b_name])
        cbar_ax_065 = fig.add_axes([0.900, 0.405, 0.010, 0.245])
        sns.heatmap(
            hm_median_display_065,
            annot=True,
            fmt='.2f',
            cmap=mcolors.LinearSegmentedColormap.from_list('step065_distance', ['#551A8B', '#B65F2A', '#FFD700']),
            vmin=hm_zmin_065,
            vmax=hm_zmax_065,
            linewidths=0.6,
            linecolor='white',
            cbar_ax=cbar_ax_065,
            cbar_kws={},
            annot_kws={'fontsize': 11.2, 'fontweight': 'bold'},
            ax=ax
        )
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10.8, fontweight='bold')
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11.4, fontweight='bold')
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9.5, length=2)
        cbar.ax.set_title('Median distance\n(µm)', fontsize=9.8,
                          fontweight='bold', pad=7, loc='center', linespacing=0.95)
        # Rotated x labels and the HLA-DR state labels must fit inside the
        # publication canvas before save; do not repair this with tight crop.
        fig.subplots_adjust(left=0.205, right=0.875, bottom=0.205, top=0.975)
        _save_step06_matplotlib(
            fig,
            _step065b_name,
            "Step06-5B T cell B-state median distance heatmap"
        )

    if STEP06_WRITE_INDIVIDUAL_PNG:
        _save_step065_static_pngs()
        if fig_step065c_nearest_only_065 is not None:
            _save_step06_matplotlib(
                fig_step065c_nearest_only_065,
                _step065c_name,
                "Step06-5C nearest-only T-cell assignment sensitivity"
            )

    if STEP06_DISPLAY_INTERACTIVE_FIGURES:
        if not paired_distance_065.empty:
            _display_step06_figure(fig_paired_distance_065)
            _display_step06_figure(fig_distance_heatmap_065)



# %%
# ==============================================================================
# [Step 06-6] Motif Analysis Moved to Final Step 07
# ==============================================================================
print("🔹 [Step 06-6] Motif analysis has moved to Final_step07.py. Step06 performs only distance-focused analyses.")


if STEP06_SAVE_CHECKPOINT and not STEP06_FIGURE_ONLY:
    adata.write_h5ad(CHECKPOINT_STEP06)
    print(f"✅ [Step 06] Step06 derived columns saved to the checkpoint: {CHECKPOINT_STEP06}")
else:
    print(f"🔹 [Step 06] Figure-only or STEP06_SAVE_CHECKPOINT=False → skip checkpoint saving: {CHECKPOINT_STEP06}")
