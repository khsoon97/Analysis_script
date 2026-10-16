# %%
from Final_utils import *
from Final_utils import _final_figure_stem_registered
from itertools import combinations as _combs
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from sklearn.metrics import balanced_accuracy_score, f1_score


def _to_numpy_array(values):
    """Safely convert CuPy/cuDF-family results to a NumPy array."""
    if hasattr(values, 'get'):
        values = values.get()
    return np.asarray(values)


def _align_probability_columns(probs, model_classes, n_classes):
    """
    Align each model's predict_proba output to the full set of classes.

    If a rare class is absent from a training fold in a group split, some models may
    return fewer probability columns; fill missing-class columns with 0.
    """
    probs = _to_numpy_array(probs)
    if probs.ndim == 1:
        if n_classes == 2:
            probs = np.vstack([1 - probs, probs]).T
        else:
            probs = probs.reshape(-1, 1)

    if probs.shape[1] == n_classes:
        return probs.astype(np.float32, copy=False)

    aligned = np.zeros((probs.shape[0], n_classes), dtype=np.float32)
    if model_classes is None:
        n_cols = min(probs.shape[1], n_classes)
        aligned[:, :n_cols] = probs[:, :n_cols]
        return aligned

    classes = _to_numpy_array(model_classes).astype(int).ravel()
    for src_idx, cls_idx in enumerate(classes[:probs.shape[1]]):
        if 0 <= cls_idx < n_classes:
            aligned[:, cls_idx] = probs[:, src_idx]
    return aligned


# ==============================================================================
# [Step 04B-1] Pan-Cellular Subtype Annotation: check the checkpoint
# ==============================================================================
CHECKPOINT_STEP02 = os.path.join(CKPT_DIR, 'mv1_step02.h5ad')
MODELS_STEP02_PATH = os.path.join(CKPT_DIR, 'mv1_models_broad.pkl')
CHECKPOINT_STEP04 = os.path.join(CKPT_DIR, 'mv1_step04B_pan_cell.h5ad')
PROBS_STEP04_PATH = os.path.join(CKPT_DIR, 'mv1_probs_subtype_04B.pkl')
TRAIN_LOG_STEP04_PATH = os.path.join(CKPT_DIR, 'mv1_step04B_training_log.csv')
PER_SUBTYPE_METRIC_STEP04_PATH = os.path.join(CKPT_DIR, 'mv1_step04B_per_subtype_metric_log.csv')
V9_TRANSFER_ROOT = os.path.join(ANALYSIS_DIR, 'checkpoints', 'transfer_models')
V9_EXPORT_FROZEN_MODELS = False  # Final v2 Figure-only: read V9 frozen models only.
V9_TRANSFER_OVERWRITE_EXISTING = False

# Execution modes
# - STEP04_FIGURE_ONLY=True : load the saved completed Step04B data and regenerate figures only.
# - RUN_CD4CTL_GMM_PREFLIGHT_ONLY=True: run only CD4_CTL GMM cutoff diagnostics without lengthy training.
# - RUN_STEP04_TRAINING=True: use Step02 results as input to run new lengthy subtype ML training.
# - To train after preflight review, set RUN_CD4CTL_GMM_PREFLIGHT_ONLY=False and RUN_STEP04_TRAINING=True.
STEP04_FIGURE_ONLY = True
RUN_CD4CTL_GMM_PREFLIGHT_ONLY = False
RUN_STEP04_TRAINING = False
RUN_STEP04B_REVIEWED_ALE_RECALCULATION = (
    os.environ.get('FINAL_STEP04B_ALE_RECALCULATE', '0') == '1'
)
OOB_EVAL_EVERY_STEP04 = 1
OOB_EVAL_EXTRA_ITERS_STEP04 = set()

# Figure-only execution reads the saved Step04B checkpoint and resaves figures only.
# Remove prior training/prediction variables from the same kernel first, as they could weaken the image-only guarantee.
STEP04B_FIGURE_ONLY_OUTPUTS = bool(
    STEP04_FIGURE_ONLY and
    not RUN_CD4CTL_GMM_PREFLIGHT_ONLY and
    not RUN_STEP04_TRAINING and
    not RUN_STEP04B_REVIEWED_ALE_RECALCULATION
)


def _install_step04b_strict_figure_only_write_firewall():
    """Delegate protected writes to the shared exact-path Final firewall."""
    import builtins as _builtins

    _guarded_open = _builtins.open
    _guarded_df_to_csv = pd.DataFrame.to_csv
    _guarded_series_to_csv = pd.Series.to_csv
    _guarded_joblib_dump = joblib.dump
    _guarded_write_h5ad = ad.AnnData.write_h5ad

    def _strict_open(file, mode='r', *args, **kwargs):
        return _guarded_open(file, mode, *args, **kwargs)

    def _strict_df_to_csv(self, path_or_buf=None, *args, **kwargs):
        return _guarded_df_to_csv(self, path_or_buf, *args, **kwargs)

    def _strict_series_to_csv(self, path_or_buf=None, *args, **kwargs):
        return _guarded_series_to_csv(self, path_or_buf, *args, **kwargs)

    def _strict_joblib_dump(value, filename, *args, **kwargs):
        return _guarded_joblib_dump(value, filename, *args, **kwargs)

    def _strict_write_h5ad(self, filename=None, *args, **kwargs):
        return _guarded_write_h5ad(self, filename, *args, **kwargs)

    _builtins.open = _strict_open
    pd.DataFrame.to_csv = _strict_df_to_csv
    pd.Series.to_csv = _strict_series_to_csv
    joblib.dump = _strict_joblib_dump
    ad.AnnData.write_h5ad = _strict_write_h5ad


if STEP04B_FIGURE_ONLY_OUTPUTS:
    _install_step04b_strict_figure_only_write_firewall()


def _step04b_write_table_if_enabled(frame, output_path, **kwargs):
    """Write computed tables only during an explicitly authorized analysis run."""
    if STEP04B_FIGURE_ONLY_OUTPUTS:
        raise PermissionError(
            f'Step04B Figure-only blocked protected table write: {output_path}'
        )
    frame.to_csv(output_path, **kwargs)
    return True


def _step04b_write_checkpoint_if_enabled(adata_obj, output_path):
    """Write an AnnData checkpoint only outside the figure-only contract."""
    if STEP04B_FIGURE_ONLY_OUTPUTS:
        raise PermissionError(
            f'Step04B Figure-only blocked protected checkpoint write: {output_path}'
        )
    adata_obj.write_h5ad(output_path)
    return True


def _step04b_dump_if_enabled(value, output_path):
    """Write model/cache payloads only outside the figure-only contract."""
    if STEP04B_FIGURE_ONLY_OUTPUTS:
        raise PermissionError(
            f'Step04B Figure-only blocked protected model/cache write: {output_path}'
        )
    joblib.dump(value, output_path)
    return True

STEP04B_RUNTIME_STATE_VARS = [
    'history_sub', 'all_sub_probs_global', 'fi_log_subtype',
    'per_subtype_metric_log', 'sub_bkp', 'val_records', 'df_val',
    'test_results', 'results_df', 'target_combo_sub', '_idx_dlbcl'
]


def _clear_step04b_runtime_state(reason):
    """Remove residual Step04B global variables from the same kernel."""
    removed = []
    for _name in STEP04B_RUNTIME_STATE_VARS:
        if _name in globals():
            globals().pop(_name, None)
            removed.append(_name)
    if removed:
        print(f"  ℹ️ {reason}: removed variables from the previous run → {', '.join(removed)}")


# Visualization options that also apply to Figure-only regeneration.
# Automatically expand B-Acc/F1 axes using recorded values to avoid clipping early-iteration spikes.
STEP04_DYNAMIC_METRIC_AXIS = True
# Candidate markers for the T-cell marker-purity heatmap include literature anchor markers and statistically significant subtype markers.
STEP04_TCELL_MARKER_PURITY_CURATED_ONLY = False
# Also display negative z-scores in signature marker purity to assess relative depletion.
STEP04_MARKER_PURITY_CLIP_NEGATIVE = False
# Preserve the staircase marker order in the T-cell purity panel while also displaying off-target positive scores.
STEP04_TCELL_MARKER_PURITY_DISPLAY_MODE = 'positive_zscore'  # 'target_only_positive' or 'positive_zscore'
STEP04_MARKER_PURITY_DYNAMIC_COLOR_RANGE = True
STEP04_MARKER_PURITY_ANNOT_MIN = 0.05
# Display 0.0 values as numbers rather than empty cells.
STEP04_MARKER_PURITY_SHOW_ZERO_ANNOT = True
# Select marker candidates after Benjamini-Hochberg correction of Wilcoxon rank-sum results.
STEP04_MARKER_PURITY_PADJ_ALPHA = 0.05
STEP04_MARKER_PURITY_PADJ_METHOD = 'fdr_bh'
# Strengthen the T-cell heatmap staircase by retaining only markers sufficiently high in the target subtype and higher than in other subtypes.
STEP04_TCELL_MARKER_PURITY_ENFORCE_SPECIFICITY = True
STEP04_TCELL_MARKER_PURITY_MIN_TARGET_Z = 0.25
STEP04_TCELL_MARKER_PURITY_MIN_MARGIN = 0.10
STEP04_TCELL_MARKER_PURITY_TOP_N_PER_SUBTYPE = 4
# Move selected markers broadly elevated across multiple subtypes to the bottom of the y-axis so they do not obscure the main staircase signal.
STEP04_TCELL_MARKER_PURITY_BOTTOM_MARKERS = {'GATA3'}

# Because CD4_CTL lacks expert GT, first define it using a GRZB GMM-based
# pseudo-GT from reference CD4 candidates, then decide whether to include it in the main T-cell subtype classifier.
STEP04B_TPR_GMM_MODE = 'cd4_gt_cd8'  # 'cd4_gt_cd8', 'exclude', 'keep'
STEP04B_TPR_GMM_CD4_MINUS_CD8_MIN = 0.0
STEP04B_TPR_GMM_ELIGIBLE_COL = 'STEP04B_TPR_GMM_Eligible'
CD4CTL_CANDIDATE_SUBTYPES = ['Th', 'Tfh', 'Treg', 'Tpr']
if STEP04B_TPR_GMM_MODE == 'exclude':
    CD4CTL_CANDIDATE_SUBTYPES = [s for s in CD4CTL_CANDIDATE_SUBTYPES if s != 'Tpr']
CD4CTL_GMM_COMPONENTS_TO_COMPARE = [2, 3, 4]
CD4CTL_GMM_N_COMPONENTS = 3
CD4CTL_GMM_POSTERIOR_THRESHOLDS = [0.5, 0.7, 0.8, 0.9]
CD4CTL_GMM_POSTERIOR_THRESHOLD = 0.7
CD4CTL_GMM_CUTOFF_DISPLAY_DECIMALS = 3
STEP04B03_POINT_JITTER_SEED = 42
STEP04B03_PANEL_OUTPUT_STEMS = {
    'A': 'Final_mv1_Step04B-03A_CD4CTL_GMM_GRZB_Distribution',
    'B': 'Final_mv1_Step04B-03B_CD4CTL_GMM_High_Component_Posterior',
    'C': 'Final_mv1_Step04B-03C_CD4CTL_GMM_Cumulative_Composition',
    'D': 'Final_mv1_Step04B-03D_CD4CTL_GMM_Marker_Profile',
}
INCLUDE_CD4CTL_PSEUDOLABEL_IN_MAIN_MODEL = True
EXCLUDE_CD4CTL_FROM_MAIN_TCELL_MODEL = not INCLUDE_CD4CTL_PSEUDOLABEL_IN_MAIN_MODEL
SAVE_STEP04B03_CD4CTL_GMM_INDIVIDUAL = True
SAVE_STEP04B03_CD4CTL_GMM_COMPOSITE = False
SAVE_STEP04B05B_IMPORTANCE_ALPHA_PVALUE = False
SAVE_STEP04B08B_NORMALIZED_SCORE_HEATMAPS = False
SAVE_STEP04B07_SUBTYPE_VALIDATION = True
SAVE_STEP04B08_DLBCL_BENCHMARK = True
SAVE_STEP04B08_ALL17_HEATMAPS = False
SAVE_STEP04B09_PURITY_COLOR_SYNCED = True
# Exclude individual lineage reliability figures from the Final list and output only the composite.
SAVE_STEP04B09_MARKER_RELIABILITY = False
SAVE_STEP04B09_MARKER_RELIABILITY_COMPOSITE = True
SAVE_STEP04B10_WILCOXON_HISTOGRAM = False
SAVE_STEP04B11_MARKER_COMPARISON = False
SAVE_STEP04B12_NO_CD4CTL_CONFUSION = False
SAVE_STEP04B12_GT_ML_CONFUSION = True
SAVE_STEP04B13_ERROR_PROFILE = False
SAVE_STEP04B045_COMBINED_COMPOSITE = True
STEP04B_BENCHMARK_ONLY = os.environ.get('FINAL_STEP04B_BENCHMARK_ONLY', '0') == '1'
if STEP04B_BENCHMARK_ONLY:
    SAVE_STEP04B03_CD4CTL_GMM_INDIVIDUAL = False
    SAVE_STEP04B07_SUBTYPE_VALIDATION = False
    SAVE_STEP04B045_COMBINED_COMPOSITE = False
    SAVE_STEP04B09_PURITY_COLOR_SYNCED = False
    SAVE_STEP04B09_MARKER_RELIABILITY_COMPOSITE = False
    SAVE_STEP04B12_GT_ML_CONFUSION = False
SHOW_STEP04B045_IMPORTANCE_PANEL = False
STEP04B11_MARKER_COMPARISON_OUTPUT_LINEAGES = ['T cells']
STEP04B03_PANEL_PAD_INCHES = 0.05
STEP04B03_PANEL_SIZES = {
    # Match each native canvas to its manuscript-slide picture slot so the
    # exported figure can be placed without stretching or post-save cropping.
    # A uses a compact plot-only canvas. B fits one half of an A4-landscape
    # two-up row while reserving vertical room for the below-axis legend and
    # the three GMM component-mean markers.
    'A': (5.45, 3.05),
    'B': (5.35, 4.30),
    'C': (12.40, 4.90),
    'D': (10.80, 5.00),
}
STEP04B_SUPPLEMENT_FIGSIZES = {
    'combined_performance': (7.40, 10.80),
    'subtype_validation': (7.40, 3.58),
    'gt_ml_confusion': (4.60, 4.00),
}
STEP04B_SUPPLEMENT_PURITY_FIGSIZES = {
    'Myeloid cells': (10.85, 6.58),
    'Stromal cells': (10.85, 6.58),
}
STEP04_MAIN_EXCLUDED_SUBTYPES = {
    'T cells': ['CD4_CTL'] if EXCLUDE_CD4CTL_FROM_MAIN_TCELL_MODEL else []
}


def _format_cd4ctl_gmm_cutoff(value):
    """Display the stored effective cutoff without implying full grid precision."""
    return f'{float(value):.{CD4CTL_GMM_CUTOFF_DISPLAY_DECIMALS}f}'


def _cd4ctl_gmm_cutoff_display(value, posterior_threshold=CD4CTL_GMM_POSTERIOR_THRESHOLD):
    """Canonical manuscript-facing cutoff notation shared by Step04B Figures."""
    return (
        f'GRZB Z ≥ {_format_cd4ctl_gmm_cutoff(value)} '
        f'(P(high) ≥ {float(posterior_threshold):g})'
    )


def _step04_excluded_subtypes(lineage):
    return STEP04_MAIN_EXCLUDED_SUBTYPES.get(str(lineage), [])


step04_loaded_from_completed = False
apply_step04_predictions = True

if STEP04_FIGURE_ONLY:
    if not os.path.exists(CHECKPOINT_STEP04):
        raise FileNotFoundError(f"STEP04_FIGURE_ONLY=True but the Step04B checkpoint is missing: {CHECKPOINT_STEP04}")
    _clear_step04b_runtime_state('Step04B figure-only')
    print("🔹 [Step 04B] Figure-only mode → loading the saved completed Step04B data.")
    adata = sc.read_h5ad(CHECKPOINT_STEP04, backed='r')
    step04_skip = True
    step04_loaded_from_completed = True
    apply_step04_predictions = False
    print(f"  ✅ image-only confirmed: h5ad/CSV/pkl save branches are disabled; only figures will be resaved. FIG_DIR={FIG_DIR}")
elif RUN_CD4CTL_GMM_PREFLIGHT_ONLY and not RUN_STEP04_TRAINING:
    if 'adata' not in dir() and 'adata' not in globals():
        adata = sc.read_h5ad(CHECKPOINT_STEP02)
        print(f"🔹 Loading Step 02 checkpoint: {os.path.basename(CHECKPOINT_STEP02)}")
    step04_skip = True
    apply_step04_predictions = False
    print("🔹 [Step 04B] CD4_CTL GMM preflight-only mode → running cutoff diagnostics without training.")
elif RUN_STEP04_TRAINING:
    # In training mode, Step04B always starts afresh from the Step02 checkpoint
    # to avoid mixing prior Step04 adata/training logs from the same kernel.
    adata = sc.read_h5ad(CHECKPOINT_STEP02)
    print(f"🔹 Loading Step 02 checkpoint: {os.path.basename(CHECKPOINT_STEP02)}")
    for _stale_name in [
        'history_sub', 'all_sub_probs_global', 'fi_log_subtype',
        'per_subtype_metric_log', 'sub_bkp', 'val_records'
    ]:
        globals().pop(_stale_name, None)
    step04_skip = False
    print("🔹 [Step 04B] Analysis mode → loading Step02 results and running new Subtype ML training.")
elif os.path.exists(PROBS_STEP04_PATH):
    print("🔹 [Step 04B] Analysis mode → loading Step02 results and reusing existing subtype training logs.")
    if 'adata' not in dir() and 'adata' not in globals():
        adata = sc.read_h5ad(CHECKPOINT_STEP02)
        print(f"🔹 Loading Step 02 checkpoint: {os.path.basename(CHECKPOINT_STEP02)}")
    step04_skip = True
    apply_step04_predictions = True
elif os.path.exists(CHECKPOINT_STEP04):
    print("✅ Completed Step 04B data (.h5ad) exists → skipping training")
    adata = sc.read_h5ad(CHECKPOINT_STEP04)
    step04_skip = True
    step04_loaded_from_completed = True
    apply_step04_predictions = False
else:
    raise FileNotFoundError(
        "Step04B training logs are missing. To run the lengthy subtype training, "
        "set RUN_STEP04_TRAINING=True and run it directly in the notebook."
    )


# %%
# ==============================================================================
# [Step 04B-2] Apply Subtype mapping
# ==============================================================================
if not step04_loaded_from_completed:
    print("🔹 [Step 04B-2] Applying fine-grained (Subtype) mapping...")
    # Always create GT_Subtype if absent (regardless of whether Modeling_Broad exists)
    if 'GT_Subtype' not in adata.obs.columns:
        adata.obs['GT_Subtype'] = adata.obs['Merged_final'].apply(map_subtype)
    if 'Modeling_Broad' not in adata.obs.columns:
        adata.obs['Modeling_Broad'] = adata.obs['GT_Subtype'].apply(refine_broad_for_subtyping)

    print("\n📊 [Subtype application results]")
    print(adata.obs['GT_Subtype'].value_counts())


_STEP04B_MARKER_COLUMN_CACHE = {}


def _extract_marker_vector(adata_obj, row_idx, marker_name):
    """Safely extract an expression vector for one marker from AnnData."""
    marker_idx = adata_obj.var_names.get_loc(marker_name)
    if STEP04B_FIGURE_ONLY_OUTPUTS and getattr(adata_obj, 'isbacked', False):
        if marker_name not in _STEP04B_MARKER_COLUMN_CACHE:
            full_vals = adata_obj.X[:, marker_idx]
            if hasattr(full_vals, 'toarray'):
                full_vals = full_vals.toarray()
            _STEP04B_MARKER_COLUMN_CACHE[marker_name] = (
                np.asarray(full_vals).reshape(-1).astype(float)
            )
        return _STEP04B_MARKER_COLUMN_CACHE[marker_name][np.asarray(row_idx)]
    vals = adata_obj.X[np.asarray(row_idx), marker_idx]
    if hasattr(vals, 'toarray'):
        vals = vals.toarray()
    return np.asarray(vals).reshape(-1).astype(float)


def _extract_marker_matrix(adata_obj, row_idx, marker_names):
    """Safely extract an expression matrix for multiple markers from AnnData."""
    marker_idx = [adata_obj.var_names.get_loc(m) for m in marker_names]
    vals = adata_obj.X[np.asarray(row_idx), :]
    if hasattr(vals, 'toarray'):
        vals = vals.toarray()
    return np.asarray(vals, dtype=float)[:, marker_idx]


_STEP04B_GLOBAL_SCALED_MATRIX = None
_STEP04B_GLOBAL_SCALED_PATH = None


def _step04b_get_global_scaled_matrix(adata_obj, chunk_size=100_000):
    """Disk-backed global z-score matrix for Figure-only marker panels."""
    global _STEP04B_GLOBAL_SCALED_MATRIX, _STEP04B_GLOBAL_SCALED_PATH
    if _STEP04B_GLOBAL_SCALED_MATRIX is not None:
        return _STEP04B_GLOBAL_SCALED_MATRIX

    if not STEP04B_FIGURE_ONLY_OUTPUTS:
        full = adata_obj.X.toarray() if hasattr(adata_obj.X, 'toarray') else np.asarray(adata_obj.X)
        _STEP04B_GLOBAL_SCALED_MATRIX = StandardScaler().fit_transform(full)
        return _STEP04B_GLOBAL_SCALED_MATRIX

    import atexit
    import tempfile

    n_obs, n_vars = adata_obj.shape
    sums = np.zeros(n_vars, dtype=np.float64)
    square_sums = np.zeros(n_vars, dtype=np.float64)
    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        chunk = adata_obj.X[start:stop]
        if hasattr(chunk, 'toarray'):
            chunk = chunk.toarray()
        chunk = np.asarray(chunk, dtype=np.float64)
        sums += chunk.sum(axis=0)
        square_sums += np.square(chunk).sum(axis=0)
    means = sums / max(n_obs, 1)
    variances = np.maximum(square_sums / max(n_obs, 1) - np.square(means), 0.0)
    scales = np.sqrt(variances)
    scales[scales == 0] = 1.0

    temp_handle = tempfile.NamedTemporaryFile(
        prefix='roider_step04b_global_z_',
        suffix='.dat',
        dir='__TEMP_ROOT__',
        delete=False,
    )
    temp_handle.close()
    _STEP04B_GLOBAL_SCALED_PATH = temp_handle.name
    scaled = np.memmap(
        _STEP04B_GLOBAL_SCALED_PATH,
        mode='w+',
        dtype=np.float32,
        shape=(n_obs, n_vars),
    )
    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        chunk = adata_obj.X[start:stop]
        if hasattr(chunk, 'toarray'):
            chunk = chunk.toarray()
        scaled[start:stop] = (
            np.asarray(chunk, dtype=np.float32) - means
        ) / scales
    scaled.flush()
    _STEP04B_GLOBAL_SCALED_MATRIX = scaled

    def _cleanup_step04b_memmap():
        global _STEP04B_GLOBAL_SCALED_MATRIX
        _STEP04B_GLOBAL_SCALED_MATRIX = None
        try:
            os.unlink(_STEP04B_GLOBAL_SCALED_PATH)
        except (FileNotFoundError, PermissionError, OSError):
            pass

    atexit.register(_cleanup_step04b_memmap)
    return _STEP04B_GLOBAL_SCALED_MATRIX


def _save_step04b_publication_figure(fig_obj, png_path, pptx_path=None, label='figure',
                                     show=True, close=True, pad_inches=None):
    """Save an individual Step04B panel to PNG/PPTX using its original creation canvas."""
    if pptx_path is None:
        pptx_path = os.path.splitext(png_path)[0] + '.pptx'
    if pad_inches is None:
        pad_inches = STEP04B03_PANEL_PAD_INCHES
    png_path, pptx_path = save_final_figure_pair(
        fig_obj, png_path, pptx_path,
        dpi=300, facecolor='white', validate=True,
        show=False, close=False,
    )
    print(f"  ✅ {label} PNG saved: {png_path}")
    print(f"  ✅ {label} PPTX saved: {pptx_path}")
    if show:
        plt.figure(fig_obj.number)
        plt.show()
    if close:
        plt.close(fig_obj)
    return png_path, pptx_path


def _save_step04b_canvas_figure(fig_obj, png_path, label='figure', show=True, close=True,
                                dpi=300, pptx_path=None):
    """Save PNG/PPTX with the figure canvas ratio itself, without bbox cropping."""
    if pptx_path is None:
        pptx_path = os.path.splitext(png_path)[0] + '.pptx'
    png_path, pptx_path = save_final_figure_pair(
        fig_obj, png_path, pptx_path,
        dpi=dpi, facecolor='white', validate=True,
        show=False, close=False,
    )
    print(f"  ✅ {label} PNG saved: {png_path}")
    print(f"  ✅ {label} PPTX saved: {pptx_path}")
    if show:
        plt.figure(fig_obj.number)
        plt.show()
    if close:
        plt.close(fig_obj)
    return png_path, pptx_path


# Preserve the required 1 x 4 benchmark order while enlarging the canvas and
# the physical plotting area allocated to every lineage panel.
STEP04B_BENCHMARK_FIGSIZE = (19.20, 6.50)
# Keep the canvas/layout fixed while scaling every benchmark visual element
# together. Marker areas use the squared factor so their diameters follow the
# same scale as text, strokes, and legend symbols.
STEP04B_BENCHMARK_ELEMENT_SCALE = 1.18
STEP04B_MARKER_RELIABILITY_COMPOSITE_FIGSIZE = (17.20, 12.40)
# Keep the two main purity panels on the same physical canvas so their titles,
# heatmaps, benchmark rows, and colorbars align when placed side by side.
STEP04B_MAIN_PURITY_FIGSIZES = {
    'B cells': (12.40, 6.78),
    'T cells': (12.40, 6.78),
}


def _build_cd4ctl_cumulative_stack_table(
    grzb_vals,
    subtype_vals,
    selected_cutoff,
    subtype_order=None,
    cutoff_points=None
):
    """Compute cumulative subtype composition among cells at or above the GRZB cutoff."""
    grzb_vals = np.asarray(grzb_vals, dtype=float)
    subtype_vals = np.asarray(subtype_vals, dtype=str)
    finite_mask = np.isfinite(grzb_vals) & pd.Series(subtype_vals).notna().to_numpy()
    grzb_vals = grzb_vals[finite_mask]
    subtype_vals = subtype_vals[finite_mask]
    if len(grzb_vals) == 0:
        raise RuntimeError('No GRZB values are available for the CD4_CTL cumulative composition figure.')

    present_subtypes = list(pd.Series(subtype_vals).value_counts().index)
    if subtype_order is None:
        subtype_order = ['CD4_CTL'] + [s for s in CD4CTL_CANDIDATE_SUBTYPES if s in present_subtypes]
    subtype_order = [s for s in subtype_order if s in present_subtypes]
    subtype_order += [s for s in present_subtypes if s not in subtype_order]
    if 'CD4_CTL' not in subtype_order:
        subtype_order = ['CD4_CTL'] + subtype_order

    if cutoff_points is None:
        if np.isfinite(selected_cutoff):
            raw_points = [
                selected_cutoff - 1.50,
                selected_cutoff - 1.00,
                selected_cutoff - 0.50,
                selected_cutoff - 0.25,
                selected_cutoff,
                selected_cutoff + 0.25,
                selected_cutoff + 0.50,
                selected_cutoff + 1.00,
                selected_cutoff + 1.50,
            ]
        else:
            raw_points = np.linspace(
                float(np.nanpercentile(grzb_vals, 25.0)),
                float(np.nanpercentile(grzb_vals, 95.0)),
                8
            )
        low_bound = float(np.nanpercentile(grzb_vals, 5.0))
        high_bound = float(np.nanpercentile(grzb_vals, 99.2))
        cutoff_points = [
            x for x in raw_points
            if np.isfinite(x) and (low_bound <= float(x) <= high_bound)
        ]
        if np.isfinite(selected_cutoff):
            cutoff_points.append(float(selected_cutoff))

    cutoff_points = {round(float(x), 2) for x in cutoff_points if np.isfinite(x)}
    if np.isfinite(selected_cutoff):
        # Preserve the exact stored selected cutoff for the selected composition.
        # Only the surrounding descriptive grid points remain rounded for display.
        cutoff_points = {
            point for point in cutoff_points
            if not np.isclose(point, float(selected_cutoff), atol=0.015)
        }
        cutoff_points.add(float(selected_cutoff))
    cutoff_points = sorted(cutoff_points)
    if len(cutoff_points) == 0:
        raise RuntimeError('Cannot create CD4_CTL cumulative composition cutoff points.')

    rows = []
    summary_rows = []
    for cutoff in cutoff_points:
        selected = grzb_vals >= float(cutoff)
        total_n = int(np.sum(selected))
        subtype_counts = pd.Series(subtype_vals[selected]).value_counts()
        cd4ctl_n = int(subtype_counts.get('CD4_CTL', 0))
        cd4ctl_pct = float(cd4ctl_n / total_n * 100.0) if total_n > 0 else np.nan
        summary_rows.append({
            'cutoff': float(cutoff),
            'total_n': total_n,
            'CD4_CTL_n': cd4ctl_n,
            'CD4_CTL_pct': cd4ctl_pct,
            'is_selected_cutoff': bool(np.isfinite(selected_cutoff) and np.isclose(float(cutoff), float(selected_cutoff), atol=0.015))
        })
        for subtype in subtype_order:
            subtype_n = int(subtype_counts.get(subtype, 0))
            rows.append({
                'cutoff': float(cutoff),
                'Subtype': subtype,
                'n': subtype_n,
                'pct': float(subtype_n / total_n * 100.0) if total_n > 0 else 0.0,
                'total_n': total_n,
                'CD4_CTL_pct': cd4ctl_pct
            })

    return pd.DataFrame(rows), pd.DataFrame(summary_rows), subtype_order


def _get_cd4ctl_cumulative_stack_inputs(
    adata_obj,
    fallback_grzb_vals,
    fallback_selected_cd4ctl,
    fallback_subtype_vals
):
    """Select GRZB/subtype values for the 04B-03 C panel."""
    cd4_subtypes_for_stack = [s for s in CD4CTL_CANDIDATE_SUBTYPES] + ['CD4_CTL']
    required_cols = {'ML_Lineage_Raw', 'ML_Subtype_Raw', 'Disease'}
    if required_cols.issubset(set(adata_obj.obs.columns)) and 'GRZB' in adata_obj.var_names:
        mask_stack = (
            (adata_obj.obs['ML_Lineage_Raw'].astype(str) == 'T cells') &
            (adata_obj.obs['Disease'].astype(str).str.contains('DLBCL', na=False)) &
            (adata_obj.obs['ML_Subtype_Raw'].astype(str).isin(cd4_subtypes_for_stack))
        )
        idx_stack = np.where(mask_stack.to_numpy())[0]
        if len(idx_stack) >= 30 and (adata_obj.obs['ML_Subtype_Raw'].iloc[idx_stack].astype(str) == 'CD4_CTL').any():
            return (
                _extract_marker_vector(adata_obj, idx_stack, 'GRZB'),
                adata_obj.obs['ML_Subtype_Raw'].iloc[idx_stack].astype(str).values,
                'DLBCL T-cell ML subtype calls'
            )

    return (
        np.asarray(fallback_grzb_vals, dtype=float),
        np.where(np.asarray(fallback_selected_cd4ctl, dtype=bool), 'CD4_CTL', np.asarray(fallback_subtype_vals, dtype=str)),
        'Reference CD4-lineage pseudo-label set'
    )


def _plot_cd4ctl_cumulative_stack_panel(
    ax_bar,
    grzb_vals,
    subtype_vals,
    selected_cutoff,
    selected_threshold=CD4CTL_GMM_POSTERIOR_THRESHOLD,
    scope_label='DLBCL T-cell ML subtype calls',
    title='Cumulative composition above each GRZB cutoff',
    title_pad=64,
    title_fontsize=13.2,
    label_fontsize=12,
    tick_fontsize=10,
    count_fontsize=8.0,
    legend_fontsize=8.0,
    legend_y=1.01,
    show_selected_annotation=True,
    xlabel='Cumulative lower bound: cells with GRZB ≥ cutoff',
    ylabel='Subtype composition among retained cells (%)'
):
    """Show enrichment of the CD4_CTL-like population at the selected GRZB cutoff in one panel."""
    composition_df, summary_df, subtype_order = _build_cd4ctl_cumulative_stack_table(
        grzb_vals,
        subtype_vals,
        selected_cutoff,
        subtype_order=['CD4_CTL'] + CD4CTL_CANDIDATE_SUBTYPES
    )
    if composition_df.empty or summary_df.empty:
        raise RuntimeError('The summary table for the CD4_CTL cumulative composition figure is empty.')

    plot_df = pd.DataFrame({
        'GRZB': np.asarray(grzb_vals, dtype=float),
        'Subtype': np.asarray(subtype_vals, dtype=str)
    }).replace({'Subtype': {'nan': np.nan, 'None': np.nan}})
    plot_df = plot_df[np.isfinite(plot_df['GRZB']) & plot_df['Subtype'].notna()].copy()
    plot_df = plot_df[plot_df['Subtype'].isin(subtype_order)].copy()

    palette = {s: CELL_COLORS.get(s, '#808080') for s in subtype_order}
    palette['CD4_CTL'] = CELL_COLORS.get('CD4_CTL', '#c0392b')
    selected_row = summary_df.loc[summary_df['is_selected_cutoff']]
    if selected_row.empty and np.isfinite(selected_cutoff):
        selected_idx = int(np.argmin(np.abs(summary_df['cutoff'].to_numpy(dtype=float) - float(selected_cutoff))))
        selected_row = summary_df.iloc[[selected_idx]]
    elif selected_row.empty:
        selected_idx = 0
        selected_row = summary_df.iloc[[0]]
    else:
        selected_idx = int(selected_row.index[0])
    selected_row = selected_row.iloc[0]
    selected_cutoff_display = float(selected_row['cutoff'])
    selected_cd4ctl_pct = float(selected_row['CD4_CTL_pct'])
    selected_total_n = int(selected_row['total_n'])

    cutoff_values = summary_df['cutoff'].to_numpy(dtype=float)
    x_pos = np.arange(len(cutoff_values))
    bottom = np.zeros(len(cutoff_values), dtype=float)
    for subtype in subtype_order:
        vals = (
            composition_df[composition_df['Subtype'] == subtype]
            .set_index('cutoff')
            .reindex(cutoff_values)['pct']
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        ax_bar.bar(
            x_pos,
            vals,
            bottom=bottom,
            color=palette.get(subtype, '#808080'),
            edgecolor='white',
            linewidth=0.9,
            label='CD4 CTL' if subtype == 'CD4_CTL' else subtype,
            alpha=1.0
        )
        bottom += vals

    if 0 <= selected_idx < len(x_pos):
        ax_bar.axvspan(
            selected_idx - 0.47,
            selected_idx + 0.47,
            facecolor=(0.12, 0.18, 0.24, 0.04),
            edgecolor='#1f2d3d',
            lw=2.6
        )
    ax_bar.plot(
        x_pos,
        summary_df['CD4_CTL_pct'].to_numpy(dtype=float),
        color='#111111',
        marker='o',
        lw=2.6,
        ms=6.2,
        label='CD4 CTL %'
    )
    ax_bar.axhline(50, color='#111111', lw=1.4, ls=':', alpha=0.75)
    ax_bar.set_ylim(0, 112)
    ax_bar.set_xticks(x_pos)
    xticklabels = []
    for cutoff in cutoff_values:
        if np.isclose(cutoff, selected_cutoff_display, atol=0.015):
            xticklabels.append(
                f'{_format_cd4ctl_gmm_cutoff(cutoff)}\n'
                f'P(high) ≥ {float(selected_threshold):g}'
            )
        else:
            xticklabels.append(f'{cutoff:.2f}')
    ax_bar.set_xticklabels(xticklabels, fontsize=tick_fontsize)
    ax_bar.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='bold')
    ax_bar.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold')
    majority_text = 'majority' if selected_cd4ctl_pct >= 50.0 else 'fraction'
    ax_bar.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=title_pad)
    if show_selected_annotation:
        ax_bar.text(
            0.99,
            0.975,
            f'{_cd4ctl_gmm_cutoff_display(selected_cutoff_display, selected_threshold)} | '
            f'CD4 CTL {majority_text} {selected_cd4ctl_pct:.1f}% (n={selected_total_n:,})',
            transform=ax_bar.transAxes,
            ha='right',
            va='top',
            fontsize=max(count_fontsize, tick_fontsize - 0.5),
            fontweight='bold',
            color='#222222',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.86, pad=1.8),
            zorder=6,
        )
    ax_bar.grid(axis='y', ls='--', alpha=0.22)
    ax_bar.tick_params(axis='y', labelsize=tick_fontsize)
    for tick_label in ax_bar.get_xticklabels() + ax_bar.get_yticklabels():
        tick_label.set_fontweight('bold')
    for i, row in summary_df.iterrows():
        ax_bar.text(
            i,
            101.2,
            f"n = {int(row['total_n']):,}",
            ha='center',
            va='bottom',
            fontsize=max(count_fontsize - 1.0, 9.0),
            color='#333333',
            fontweight='bold' if int(i) == selected_idx else 'normal'
        )
    ax_bar.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        fontsize=legend_fontsize,
        ncol=min(6, len(subtype_order) + 1),
        borderaxespad=0.0,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=0.9
    )
    return composition_df, summary_df


def _plot_cd4ctl_gmm_point_layers(
    ax_dist,
    preflight_df,
    subtype_order_preflight,
    *,
    background_size=1.1,
    background_alpha=0.08,
    candidate_size=2.0,
    candidate_alpha=0.48,
):
    """Plot every saved cell with deterministic jitter and rasterized dense layers."""
    subtype_codes = pd.Categorical(
        preflight_df['Original_Subtype'],
        categories=subtype_order_preflight,
        ordered=True,
    ).codes
    grzb_values = pd.to_numeric(preflight_df['GRZB'], errors='coerce').to_numpy(dtype=float)
    candidate_mask = preflight_df['CD4_CTL_candidate'].astype(bool).to_numpy()
    valid_points = (subtype_codes >= 0) & np.isfinite(grzb_values)
    jitter = np.random.default_rng(STEP04B03_POINT_JITTER_SEED).uniform(
        -0.25, 0.25, size=len(preflight_df)
    )
    x_values = subtype_codes.astype(float) + jitter

    # Keep every cell in the Figure while rasterizing only the dense point
    # layers so the normal Figure-only export remains practical and compact.
    background_points = valid_points & ~candidate_mask
    candidate_points = valid_points & candidate_mask
    ax_dist.scatter(
        x_values[background_points],
        grzb_values[background_points],
        color='black',
        s=background_size ** 2,
        alpha=background_alpha,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax_dist.scatter(
        x_values[candidate_points],
        grzb_values[candidate_points],
        color='#c0392b',
        s=candidate_size ** 2,
        alpha=candidate_alpha,
        linewidths=0,
        rasterized=True,
        zorder=4,
    )


def _plot_cd4ctl_gmm_distribution_panel(
    ax_dist,
    preflight_df,
    subtype_order_preflight,
    subtype_palette_preflight,
    selected_cutoff,
    selected_cd4ctl,
    display_min,
    display_max
):
    """A panel: show the reference CD4-lineage GRZB distribution and selected cutoff."""
    sns.boxplot(
        data=preflight_df,
        x='Original_Subtype',
        y='GRZB',
        order=subtype_order_preflight,
        palette=subtype_palette_preflight,
        showfliers=False,
        width=0.56,
        linewidth=1.6,
        ax=ax_dist,
        boxprops=dict(edgecolor='black', alpha=0.88),
        zorder=3
    )
    _plot_cd4ctl_gmm_point_layers(
        ax_dist,
        preflight_df,
        subtype_order_preflight,
    )
    ax_dist.axhline(selected_cutoff, color='#c0392b', ls='--', lw=2.0, zorder=5)
    ax_dist.set_ylim(display_min, display_max)
    ax_dist.set_title('Reference CD4-lineage GRZB distribution', fontsize=13.5, fontweight='bold', pad=7)
    ax_dist.set_xlabel('Original CD4-lineage subtype', fontsize=11.5)
    ax_dist.set_ylabel('GRZB expression', fontsize=11.5)
    ax_dist.tick_params(axis='x', rotation=0, labelsize=10.5)
    ax_dist.tick_params(axis='y', labelsize=10.5)
    ax_dist.grid(axis='y', ls='--', alpha=0.25)
    ax_dist.text(
        0.98,
        0.965,
        _cd4ctl_gmm_cutoff_display(selected_cutoff),
        transform=ax_dist.transAxes,
        ha='right',
        va='top',
        fontsize=9.4,
        fontweight='bold',
        color='#222222',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.86, pad=1.8),
        zorder=6,
    )


def _plot_cd4ctl_gmm_posterior_panel(
    ax_post,
    preflight_df,
    selected_cutoff,
    display_min,
    display_max,
    means_text='',
    gmm=None,
    high_comp=None
):
    """B panel: show the high-GRZB component posterior curve."""
    from matplotlib.ticker import MaxNLocator
    if gmm is not None and high_comp is not None:
        grid = np.linspace(float(np.nanmin(preflight_df['GRZB'])), float(np.nanmax(preflight_df['GRZB'])), 2000)
        high_prob_grid = gmm.predict_proba(grid.reshape(-1, 1))[:, high_comp]
        curve_x = np.asarray(grid, dtype=float)
        curve_y = np.asarray(high_prob_grid, dtype=float)
        ax_post.plot(grid, high_prob_grid, color='#c0392b', lw=2.3, label='P(high GRZB component)')
    else:
        curve_df = preflight_df[['GRZB', 'High_Component_Prob']].copy()
        curve_df = curve_df[(curve_df['GRZB'] >= display_min) & (curve_df['GRZB'] <= display_max)]
        bins = np.linspace(display_min, display_max, 180)
        curve_df['GRZB_bin'] = pd.cut(curve_df['GRZB'], bins=bins, include_lowest=True)
        curve_summary = (
            curve_df.groupby('GRZB_bin', observed=True)
            .agg(GRZB=('GRZB', 'mean'), High_Component_Prob=('High_Component_Prob', 'median'))
            .dropna()
        )
        curve_x = curve_summary['GRZB'].to_numpy(dtype=float)
        curve_y = curve_summary['High_Component_Prob'].to_numpy(dtype=float)
        ax_post.plot(
            curve_summary['GRZB'],
            curve_summary['High_Component_Prob'],
            color='#c0392b',
            lw=2.3,
            label='Median saved posterior'
        )
    ax_post.axhline(CD4CTL_GMM_POSTERIOR_THRESHOLD, color='#2c3e50', ls=':', lw=1.9,
                    label=f'Posterior threshold = {CD4CTL_GMM_POSTERIOR_THRESHOLD}')
    ax_post.axvline(selected_cutoff, color='#c0392b', ls='--', lw=1.9,
                    label=_cd4ctl_gmm_cutoff_display(selected_cutoff))

    try:
        if isinstance(means_text, str):
            component_means = np.asarray(
                [float(x) for x in means_text.split(',') if str(x).strip()],
                dtype=float,
            )
        else:
            component_means = np.asarray(means_text, dtype=float).reshape(-1)
        component_means = np.sort(component_means[np.isfinite(component_means)])
    except Exception:
        component_means = np.asarray([], dtype=float)

    if component_means.size:
        if gmm is not None and high_comp is not None:
            component_probs = gmm.predict_proba(component_means.reshape(-1, 1))[:, high_comp]
        elif curve_x.size:
            component_probs = np.interp(component_means, curve_x, curve_y)
        else:
            component_probs = np.full(component_means.shape, np.nan, dtype=float)

        if component_means.size == 3:
            component_names = ['Low', 'Intermediate', 'High']
        else:
            component_names = [f'Component {idx + 1}' for idx in range(component_means.size)]
        component_colors = ['#4C78A8', '#F2A65A', '#8E63B7']
        component_markers = ['o', 's', 'D']
        for idx, (mean_value, mean_prob, component_name) in enumerate(zip(
            component_means, component_probs, component_names
        )):
            if not np.isfinite(mean_prob):
                continue
            ax_post.scatter(
                [mean_value],
                [mean_prob],
                s=66,
                marker=component_markers[idx % len(component_markers)],
                color=component_colors[idx % len(component_colors)],
                edgecolor='white',
                linewidth=0.9,
                zorder=6,
                label=f'{component_name} component mean = {mean_value:.2f}',
            )
    ax_post.set_xlim(display_min, display_max)
    # Adjust the locator to keep endpoint tick labels within the compact canvas boundaries.
    ax_post.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax_post.set_ylim(-0.03, 1.03)
    ax_post.set_title('GMM high-component posterior', fontsize=13.5, fontweight='bold', pad=7)
    ax_post.set_xlabel('GRZB expression', fontsize=11.5)
    ax_post.set_ylabel('Posterior probability', fontsize=11.5)
    ax_post.tick_params(axis='both', labelsize=10.5)
    ax_post.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        borderaxespad=0.0,
        fontsize=7.6,
        frameon=True,
        ncol=2,
        columnspacing=0.75,
        handlelength=1.8,
        handletextpad=0.45,
    )
    ax_post.grid(axis='y', ls='--', alpha=0.25)


def _plot_cd4ctl_gmm_marker_heatmap_panel(ax_heat, marker_heatmap_summary, marker_panel, cbar_ax=None):
    """D panel: show the marker-profile heatmap after applying the cutoff."""
    median_marker_cols = [f'{m}_median' for m in marker_panel]
    if (not marker_heatmap_summary.empty) and median_marker_cols:
        heat_df = marker_heatmap_summary.set_index('Tcell_Group')[median_marker_cols].copy()
        heat_df.columns = marker_panel
        _group_display = {
            'CD4_CTL candidate': 'CD4 CTL candidate',
            'CD8T reference': 'CD8 T reference',
        }
        heat_df.index = [
            f"{_group_display.get(row['Tcell_Group'], row['Tcell_Group'])}\n(n={int(row['n']):,})"
            for _, row in marker_heatmap_summary.iterrows()
        ]
        sns.heatmap(
            heat_df,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            linewidths=0.5,
            cbar_ax=cbar_ax,
            cbar_kws={
                'label': '',
                'shrink': 0.50,
                'pad': 0.008,
                'aspect': 30,
                'fraction': 0.022,
            },
            ax=ax_heat,
            annot_kws={'fontsize': 13.5, 'fontweight': 'bold'}
        )
        ax_heat.set_title('Marker profile after cutoff', fontsize=17.0, fontweight='bold', pad=10)
        ax_heat.set_xlabel('Marker', fontsize=14.0, fontweight='bold', labelpad=7)
        ax_heat.set_ylabel('T-cell subtype / group', fontsize=14.0, fontweight='bold', labelpad=3)
        ax_heat.tick_params(axis='x', rotation=35, labelsize=12.5)
        ax_heat.tick_params(axis='y', rotation=0, labelsize=12.0)
        for tick_label in ax_heat.get_xticklabels() + ax_heat.get_yticklabels():
            tick_label.set_fontweight('bold')
        if ax_heat.collections and ax_heat.collections[0].colorbar is not None:
            cbar = ax_heat.collections[0].colorbar
            cbar.ax.set_ylabel('')
            cbar.ax.set_title(
                'Median\nexpression', fontsize=11.4, fontweight='bold', pad=7,
                loc='center'
            )
            cbar.ax.tick_params(labelsize=11.0, width=1.0)
            for tick_label in cbar.ax.get_yticklabels():
                tick_label.set_fontweight('bold')
    else:
        ax_heat.text(0.5, 0.5, 'No marker heatmap available', ha='center', va='center')
        ax_heat.set_axis_off()


def _save_cd4ctl_gmm_individual_panels(
    preflight_df,
    subtype_order_preflight,
    subtype_palette_preflight,
    selected_cutoff,
    selected_cd4ctl,
    display_min,
    display_max,
    marker_heatmap_summary,
    marker_panel,
    stack_grzb_vals,
    stack_subtype_vals,
    stack_scope_label,
    means_text='',
    gmm=None,
    high_comp=None,
    show=True
):
    """Save the Step04B-03 CD4_CTL GMM summary as individual A-D panel files."""
    if not SAVE_STEP04B03_CD4CTL_GMM_INDIVIDUAL:
        return {}

    output_specs = STEP04B03_PANEL_OUTPUT_STEMS
    output_paths = {}

    fig_a, ax_a = plt.subplots(figsize=STEP04B03_PANEL_SIZES['A'])
    _plot_cd4ctl_gmm_distribution_panel(
        ax_a,
        preflight_df,
        subtype_order_preflight,
        subtype_palette_preflight,
        selected_cutoff,
        selected_cd4ctl,
        display_min,
        display_max
    )
    fig_a.subplots_adjust(left=0.135, right=0.985, top=0.85, bottom=0.22)
    output_paths['A'] = _save_step04b_publication_figure(
        fig_a,
        os.path.join(FIG_DIR, f"{output_specs['A']}.png"),
        label='Step04B-03A CD4_CTL GMM distribution',
        show=show
    )

    fig_b, ax_b = plt.subplots(figsize=STEP04B03_PANEL_SIZES['B'])
    _plot_cd4ctl_gmm_posterior_panel(
        ax_b,
        preflight_df,
        selected_cutoff,
        display_min,
        display_max,
        means_text=means_text,
        gmm=gmm,
        high_comp=high_comp
    )
    fig_b.subplots_adjust(left=0.145, right=0.985, top=0.86, bottom=0.34)
    output_paths['B'] = _save_step04b_publication_figure(
        fig_b,
        os.path.join(FIG_DIR, f"{output_specs['B']}.png"),
        label='Step04B-03B CD4_CTL GMM posterior',
        show=show
    )

    fig_c, ax_c = plt.subplots(figsize=STEP04B03_PANEL_SIZES['C'])
    _, summary_c = _plot_cd4ctl_cumulative_stack_panel(
        ax_c,
        stack_grzb_vals,
        stack_subtype_vals,
        selected_cutoff,
        selected_threshold=CD4CTL_GMM_POSTERIOR_THRESHOLD,
        scope_label=stack_scope_label,
        title='',
        title_pad=0,
        title_fontsize=19.0,
        label_fontsize=15.0,
        tick_fontsize=13.5,
        count_fontsize=14.0,
        legend_fontsize=13.5,
        legend_y=1.060,
        show_selected_annotation=False,
        xlabel='GRZB cutoff (retained cells ≥ cutoff)',
        ylabel='Retained-cell composition (%)'
    )
    selected_c = summary_c.loc[summary_c['is_selected_cutoff'].astype(bool)]
    if selected_c.empty:
        selected_c = summary_c.iloc[[int(np.argmin(np.abs(
            summary_c['cutoff'].to_numpy(dtype=float) - float(selected_cutoff)
        )))]]
    selected_c = selected_c.iloc[0]
    selected_c_pct = float(selected_c['CD4_CTL_pct'])
    fig_c.suptitle(
        'Composition above GRZB cutoff',
        fontsize=19.0,
        fontweight='bold',
        y=0.982,
    )
    fig_c.text(
        0.5,
        0.900,
        f"{_cd4ctl_gmm_cutoff_display(selected_c['cutoff'])} | "
        f"CD4 CTL {selected_c_pct:.1f}% | "
        f"n={int(selected_c['total_n']):,}",
        ha='center',
        va='center',
        fontsize=14.0,
        fontweight='bold',
        color='#222222',
    )
    fig_c.subplots_adjust(left=0.055, right=0.994, top=0.770, bottom=0.17)
    output_paths['C'] = _save_step04b_publication_figure(
        fig_c,
        os.path.join(FIG_DIR, f"{output_specs['C']}.png"),
        label='Step04B-03C CD4_CTL GMM cumulative composition',
        show=show
    )

    fig_d, ax_d = plt.subplots(figsize=STEP04B03_PANEL_SIZES['D'])
    cbar_d = fig_d.add_axes([0.910, 0.35, 0.008, 0.29])
    _plot_cd4ctl_gmm_marker_heatmap_panel(
        ax_d, marker_heatmap_summary, marker_panel, cbar_ax=cbar_d
    )
    # Direct compact canvas: keep labels and colorbar visible without post-save crop/resize.
    fig_d.subplots_adjust(left=0.185, right=0.890, top=0.88, bottom=0.20)
    output_paths['D'] = _save_step04b_publication_figure(
        fig_d,
        os.path.join(FIG_DIR, f"{output_specs['D']}.png"),
        label='Step04B-03D CD4_CTL GMM marker profile',
        show=show
    )

    return output_paths


def _mark_step04b_tpr_gmm_eligibility(adata_obj):
    """
    Mark only Tpr cells to include among the CD4_CTL GMM candidates.

    This function does not change GT_Subtype. Thus, the existing Tpr labels remain
    in overall subtype-model training, and the Tpr candidate set is restricted
    only during CD4_CTL GMM cutoff estimation/application.
    """
    valid_modes = {'keep', 'exclude', 'cd4_gt_cd8'}
    if STEP04B_TPR_GMM_MODE not in valid_modes:
        raise ValueError(f"STEP04B_TPR_GMM_MODE must be one of {valid_modes}: {STEP04B_TPR_GMM_MODE}")

    subtype_vals = adata_obj.obs['GT_Subtype'].astype(str).values
    tpr_mask = subtype_vals == 'Tpr'
    tpr_idx = np.where(tpr_mask)[0]
    eligible_tpr = np.ones(len(tpr_idx), dtype=bool)

    if STEP04B_TPR_GMM_MODE == 'exclude':
        eligible_tpr[:] = False
    elif STEP04B_TPR_GMM_MODE == 'cd4_gt_cd8':
        missing_markers = [m for m in ['CD4', 'CD8'] if m not in adata_obj.var_names]
        if missing_markers:
            raise KeyError(f"Markers required to restrict Step04B Tpr GMM candidates are missing: {missing_markers}")
        if len(tpr_idx) > 0:
            cd4_vals = _extract_marker_vector(adata_obj, tpr_idx, 'CD4')
            cd8_vals = _extract_marker_vector(adata_obj, tpr_idx, 'CD8')
            eligible_tpr = (cd4_vals - cd8_vals) > float(STEP04B_TPR_GMM_CD4_MINUS_CD8_MIN)

    adata_obj.obs[STEP04B_TPR_GMM_ELIGIBLE_COL] = True
    if len(tpr_idx) > 0:
        adata_obj.obs.iloc[
            tpr_idx,
            adata_obj.obs.columns.get_loc(STEP04B_TPR_GMM_ELIGIBLE_COL)
        ] = eligible_tpr

    adata_obj.uns['step04B_tpr_gmm_policy'] = {
        'mode': STEP04B_TPR_GMM_MODE,
        'cd4_minus_cd8_min': float(STEP04B_TPR_GMM_CD4_MINUS_CD8_MIN),
        'n_tpr_total': int(len(tpr_idx)),
        'n_tpr_gmm_eligible': int(eligible_tpr.sum()) if len(tpr_idx) > 0 else 0,
        'n_tpr_gmm_excluded': int((~eligible_tpr).sum()) if len(tpr_idx) > 0 else 0,
        'eligible_column': STEP04B_TPR_GMM_ELIGIBLE_COL,
        'scope': 'CD4_CTL_GMM_candidate_only'
    }
    print(
        f"  ℹ️ Step04B Tpr GMM candidate restriction: mode={STEP04B_TPR_GMM_MODE}, "
        f"GMM-eligible={int(eligible_tpr.sum()) if len(tpr_idx) > 0 else 0:,}/"
        f"{len(tpr_idx):,}, GMM-excluded={int((~eligible_tpr).sum()) if len(tpr_idx) > 0 else 0:,}; "
        "All Tpr training labels are retained."
    )


def _gmm_component_cutoff(gmm, high_comp, posterior_threshold, values):
    """
    Estimate the first GRZB value at which the high-component posterior exceeds the specified threshold.
    If no grid-based estimate is available, use the minimum GRZB value among actually selected cells.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    grid = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), 2000).reshape(-1, 1)
    high_prob_grid = gmm.predict_proba(grid)[:, high_comp]
    passed = np.where(high_prob_grid >= posterior_threshold)[0]
    if len(passed) > 0:
        return float(grid[passed[0], 0])

    high_prob = gmm.predict_proba(values.reshape(-1, 1))[:, high_comp]
    selected = values[high_prob >= posterior_threshold]
    return float(np.nanmin(selected)) if len(selected) > 0 else np.nan


# ── Reviewed DLBCL ALE marker-response integration ────────────────────────────
# Default execution reads the pinned completed review tables/manifests.  The
# frozen-model ALE calculation remains available only behind an explicit opt-in
# and an unused output directory under Final/recalculations.
from pathlib import Path as _Path
import hashlib as _hashlib
import json as _json
import uuid as _uuid

STEP04B_REVIEWED_ALE_FIGURE_ONLY = (
    STEP04B_FIGURE_ONLY_OUTPUTS and not STEP04B_BENCHMARK_ONLY
)
CONTINUE_STEP04B_LEGACY_FIGURES = (
    os.environ.get('FINAL_STEP04B_CONTINUE_LEGACY_FIGURES', '0') == '1'
)
STEP04B_REVIEWED_ALE_NEW_OUTPUT_ROOT = os.environ.get(
    'FINAL_STEP04B_ALE_NEW_OUTPUT_ROOT', ''
).strip()
STEP04B_REVIEWED_ALE_ROOT = _Path(
    FINAL_TABLE_DIR, 'step04b_dlbcl_ale_marker_response'
)
STEP04B_REVIEWED_ALE_SOURCE_RUN = _Path(
    STEP04B_REVIEWED_ALE_ROOT, 'source_attribution'
)
STEP04B_REVIEWED_ALE_HEATMAP_ROOT = _Path(
    STEP04B_REVIEWED_ALE_ROOT, 'heatmaps'
)
STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS = {
    'bubble_selection': STEP04B_REVIEWED_ALE_HEATMAP_ROOT / 'bubble_top15_selection.csv',
    'global_z_stats': STEP04B_REVIEWED_ALE_HEATMAP_ROOT / 'global_z_feature_stats.csv',
    'heatmap_selection': STEP04B_REVIEWED_ALE_HEATMAP_ROOT / 'heatmap_selected_rows.csv',
    'heatmap_profiles': STEP04B_REVIEWED_ALE_HEATMAP_ROOT / 'heatmap_profile_values.csv',
    'manifest': STEP04B_REVIEWED_ALE_HEATMAP_ROOT / 'render_manifest.json',
}
STEP04B_REVIEWED_DISPLAY_MARKER_INCLUDE = tuple(
    marker.strip() for marker in os.environ.get(
        'FINAL_STEP04B_DISPLAY_MARKERS', ''
    ).split(',') if marker.strip()
)
STEP04B_REVIEWED_DISPLAY_MARKER_EXCLUDE = tuple(
    marker.strip() for marker in os.environ.get(
        'FINAL_STEP04B_EXCLUDE_MARKERS', ''
    ).split(',') if marker.strip()
)
STEP04B_REVIEWED_BUBBLE_TOP_N_PER_CLASS = 15
STEP04B_REVIEWED_HEATMAP_CANDIDATE_TOP_N_PER_CLASS = 15
STEP04B_REVIEWED_HEATMAP_MAX_POSITIVE_PER_CLASS = 4
STEP04B_REVIEWED_HEATMAP_MAX_NEGATIVE_PER_CLASS = 2
STEP04B_REVIEWED_HEATMAP_TARGET_MARGIN_MIN = 0.10
STEP04B_REVIEWED_PROFILE_CHUNK_SIZE = 100_000
STEP04B_REVIEWED_DISPLAY_RENDER_CONTRACT = (
    'final_local_ale_top15_dynamic_global_z_profiles_v2'
)
STEP04B_REVIEWED_ALE_BUBBLE_STEM = 'Final_mv1_Step04-08_MarkerReliability_Composite'
STEP04B_REVIEWED_ALE_BUBBLE_CANVAS_CM = (17.800, 16.182)
STEP04B_REVIEWED_ALE_FIGURE_DIR = _Path(FIG_DIR)
STEP04B_REVIEWED_ARIAL_REGULAR = _Path('__FONT_ROOT__/arial.ttf')
STEP04B_REVIEWED_ARIAL_BOLD = _Path('__FONT_ROOT__/arialbd.ttf')
STEP04B_REVIEWED_ALE_HEATMAP_STEMS = {
    'B cells': 'Final_mv1_Step04-09_Purity_ColorSynced_B_cells',
    'Myeloid cells': 'Final_mv1_Step04-10_Purity_ColorSynced_Myeloid_cells',
    'Stromal cells': 'Final_mv1_Step04-11_Purity_ColorSynced_Stromal_cells',
    'T cells': 'Final_mv1_Step04-12_Purity_ColorSynced_T_cells',
}
STEP04B_ACTIVE_REVIEWED_FIGURE_STEMS = {
    STEP04B_REVIEWED_ALE_BUBBLE_STEM,
    *STEP04B_REVIEWED_ALE_HEATMAP_STEMS.values(),
}
# Direct physical canvases for the planned A4-landscape assembly.  The 13.6 cm
# width leaves room for page margins and a gutter in a two-column layout; B/T
# share the requested 6.07 cm row height.  Supplementary Myeloid/Stromal panels
# keep the same half-page width and use row-count-appropriate native heights.
STEP04B_REVIEWED_ALE_HEATMAP_LAYOUTS = {
    'B cells': {
        'canvas_cm': (7.4744, 4.6858), 'plot_bottom': 0.175,
        'plot_left': 0.190, 'plot_right': 0.995, 'plot_top': 0.880,
        'plot_wspace': 0.045, 'method_title_fontsize': 5.8,
        'method_title_y': 1.090,
        'ylabel_fontsize': 6.0,
        'colorbar_rect': (0.815, 0.050, 0.170, 0.035),
        'colorbar_label_fontsize': 4.4, 'colorbar_tick_fontsize': 3.8,
        'x_rotation': 0, 'annotation_fontsize': 5.2,
        'xtick_fontsize': 4.8, 'ytick_fontsize': 5.2,
        'x_tick_pad': 1.0,
        'wrap_xtick_labels': True,
    },
    'T cells': {
        'canvas_cm': (11.3584, 4.6858), 'plot_bottom': 0.190,
        'plot_left': 0.130, 'plot_right': 0.999, 'plot_top': 0.870,
        'plot_wspace': 0.020, 'method_title_fontsize': 7.0,
        'method_title_y': 1.105,
        'ylabel_fontsize': 6.2,
        'colorbar_rect': (0.870, 0.050, 0.115, 0.035),
        'colorbar_label_fontsize': 4.4, 'colorbar_tick_fontsize': 3.8,
        'x_rotation': 0, 'annotation_fontsize': 4.6,
        'xtick_fontsize': 5.0, 'ytick_fontsize': 5.3,
        'xtick_fontsize_by_class': {'CD8T': 4.4},
        'x_tick_pad': 1.0,
        'wrap_tcell_xtick_labels': True,
    },
    'Myeloid cells': {
        'canvas_cm': (7.95, 6.82), 'plot_bottom': 0.236,
        'plot_left': 0.205, 'plot_right': 0.995, 'plot_top': 0.873,
        'plot_wspace': 0.070, 'method_title_fontsize': 6.0,
        'method_title_y': 1.105,
        'ylabel_fontsize': 6.8,
        'colorbar_rect': (0.745, 0.045, 0.230, 0.037),
        'colorbar_label_fontsize': 5.6, 'colorbar_tick_fontsize': 4.8,
        'x_rotation': 45, 'annotation_fontsize': 5.8,
        'xtick_fontsize': 5.2, 'ytick_fontsize': 6.0,
        'x_tick_pad': 1.0,
    },
    'Stromal cells': {
        'canvas_cm': (7.95, 6.82), 'plot_bottom': 0.151,
        'plot_left': 0.205, 'plot_right': 0.995, 'plot_top': 0.851,
        'plot_wspace': 0.070, 'method_title_fontsize': 6.0,
        'method_title_y': 1.105,
        'ylabel_fontsize': 7.0,
        'colorbar_rect': (0.745, 0.045, 0.230, 0.037),
        'colorbar_label_fontsize': 5.6, 'colorbar_tick_fontsize': 4.8,
        'x_rotation': 0, 'annotation_fontsize': 6.2,
        'xtick_fontsize': 6.0, 'ytick_fontsize': 6.4,
        'x_tick_pad': 1.0,
    },
}
STEP04B_REVIEWED_ALE_CLASS_DISPLAY = {
    'CD8T': 'CD8 T',
    'CD4_CTL': 'CD4 CTL',
    'B cell': 'B cells',
    'Plasma cell': 'Plasma cells',
    'Myeloid': 'Myeloid cells',
    'Macrophage': 'Macrophages',
    'Stromal': 'Stromal cells',
}
STEP04B_REVIEWED_ALE_STAGE_CONTRACTS = (
    ('T_cells', 'T cells', ('Th', 'Tfh', 'Treg', 'CD8T', 'CD4_CTL', 'Tpr')),
    ('B_cells', 'B cells', ('B cell', 'Plasma cell')),
    ('Myeloid_cells', 'Myeloid cells', ('Myeloid', 'Macrophage', 'DC')),
    ('Stromal_cells', 'Stromal cells', ('Stromal', 'FDC')),
)
STEP04B_REVIEWED_ALE_CLASS_COLORS = {
    'Th': '#AD0000', 'Tfh': '#FDC38D', 'Treg': '#CE2417', 'CD4_CTL': '#E7533A',
    'CD8T': '#FCA26D', 'Tpr': '#F67B51', 'B cell': '#1764AB',
    'Plasma cell': '#4A98C9', 'Myeloid': '#5B3495', 'Macrophage': '#9E9BC8',
    'DC': '#796EB2', 'Stromal': '#157F3B', 'FDC': '#4BB062',
}
STEP04B_REVIEWED_DISPLAY_SELECTION_COLUMNS = (
    'stage', 'lineage', 'class', 'feature', 'feature_position',
    'mean_ale_q10_to_q90_probability_change', 'pair_sign_consistency',
    'support_sign_concordance', 'direction_consistency',
    'direction_consistency_threshold', 'direction_symbol', 'figure_alpha',
    'class_normalized_importance_percent',
    'positive_top_k_attribution_frequency_percent', 'source_join_complete',
    'response_metric',
)


def _step04b_sha256(path):
    digest = _hashlib.sha256()
    with _Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _step04b_require_arial():
    """Register and resolve the exact user-approved Arial files before plotting."""
    families = []
    for font_path in (STEP04B_REVIEWED_ARIAL_REGULAR, STEP04B_REVIEWED_ARIAL_BOLD):
        if not font_path.is_file() or font_path.stat().st_size <= 0:
            raise FileNotFoundError(f'Step04B reviewed Arial font file is missing: {font_path}')
        font_manager.fontManager.addfont(str(font_path))
        families.append(font_manager.FontProperties(fname=str(font_path)).get_name())
    if families != ['Arial', 'Arial']:
        raise RuntimeError(
            'Step04B reviewed Figure requires exact Arial Regular/Bold files; '
            f'resolved {families}'
        )
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
    })
    return tuple(map(str, (STEP04B_REVIEWED_ARIAL_REGULAR, STEP04B_REVIEWED_ARIAL_BOLD)))


def _step04b_assert_arial_text(fig_obj):
    fig_obj.canvas.draw()
    resolved = {
        text.get_fontproperties().get_name()
        for text in fig_obj.findobj(
            match=lambda artist: isinstance(artist, matplotlib.text.Text)
        )
        if text.get_visible() and text.get_text().strip()
    }
    if not resolved or resolved != {'Arial'}:
        raise RuntimeError(f'Step04B reviewed Figure font contract failed: {sorted(resolved)}')
    return sorted(resolved)


def _step04b_save_reviewed_pair(fig_obj, png_path, pptx_path, *, dpi=300, show=False):
    """Save one reviewed pair through the common transactional pair helper."""
    png_path = _Path(png_path)
    pptx_path = _Path(pptx_path)
    expected_parent = STEP04B_REVIEWED_ALE_FIGURE_DIR.resolve()
    if png_path.resolve().parent != expected_parent or pptx_path.resolve().parent != expected_parent:
        raise RuntimeError('Step04B reviewed Figures must stay directly in figures/v2')
    png_path.parent.mkdir(parents=True, exist_ok=True)
    saved_png_path, saved_pptx_path = save_final_figure_pair(
        fig_obj, png_path, pptx_path,
        dpi=int(dpi), facecolor='white', auxiliary_tight=False, validate=True,
        show=show, close=True,
    )
    return saved_png_path, saved_pptx_path


def _step04b_validate_completed_ale_bundle():
    """Validate immutable Final-local ALE/importance inputs, excluding derived displays."""
    manifest_path = STEP04B_REVIEWED_ALE_ROOT / 'manifests' / 'run_manifest.json'
    copy_manifest_path = STEP04B_REVIEWED_ALE_ROOT / 'manifests' / 'final_copy_manifest.json'
    required_tables = (
        'ale_bin_support.csv', 'baseline_saved_annotation_agreement_audit.csv',
        'class_feature_marker_response_summary.csv', 'paired_iteration_ale_curve.csv.gz',
        'paired_iteration_class_feature_response.csv.gz',
        'paired_iteration_patient_bin_response.csv.gz',
        'selected_read_only_input_audit.csv', 'source_importance_frequency_join_audit.csv',
    )
    required = [
        manifest_path, copy_manifest_path,
        *(STEP04B_REVIEWED_ALE_ROOT / 'tables' / name for name in required_tables),
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError('Missing Final-local Step04B ALE artifacts: ' + ', '.join(missing))
    run_manifest = _json.loads(manifest_path.read_text(encoding='utf-8'))
    if (
        run_manifest.get('status') != 'complete'
        or run_manifest.get('input_unchanged') is not True
        or run_manifest.get('retraining') is not False
        or run_manifest.get('source_models_read_only') is not True
        or run_manifest.get('source_checkpoints_read_only') is not True
        or run_manifest.get('direction_contract', {}).get('metric')
        != 'patient-equal first-order ALE q10-to-q90 contrast'
    ):
        raise RuntimeError('Final-local Step04B ALE completion contract failed')
    copy_manifest = _json.loads(copy_manifest_path.read_text(encoding='utf-8'))
    if copy_manifest.get('status') != 'complete':
        raise RuntimeError('Final-local Step04B ALE copy manifest is incomplete')
    copied_files = copy_manifest.get('files', {})
    expected_copy_entries = {
        'manifests/run_manifest.json',
        *(f'tables/{name}' for name in required_tables),
        'source_attribution/manifests/run_manifest.json',
        'source_attribution/tables/class_feature_joint_importance_summary.csv',
        'source_attribution/tables/paired_iteration_top20_attribution_frequency.csv',
        'source_attribution/tables/selected_read_only_input_audit.csv',
    }
    if not expected_copy_entries.issubset(copied_files):
        raise RuntimeError(
            'Final-local Step04B ALE copy manifest lacks required entries: '
            f'{sorted(expected_copy_entries - set(copied_files))}'
        )
    derived_copy_entries = {
        f'heatmaps/{path.name}'
        for path in STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS.values()
    }
    for relative, audit in copied_files.items():
        if relative in derived_copy_entries:
            continue
        copied = STEP04B_REVIEWED_ALE_ROOT / relative
        if not copied.is_file() or copied.stat().st_size != int(audit['bytes']):
            raise RuntimeError(f'Final-local Step04B ALE copy size mismatch: {copied}')
        if _step04b_sha256(copied) != audit['sha256']:
            raise RuntimeError(f'Final-local Step04B ALE copy hash mismatch: {copied}')
    return {
        'status': 'complete', 'run_root': str(STEP04B_REVIEWED_ALE_ROOT),
        'required_tables': list(required_tables), 'copy_hashes_verified': True,
    }


def _step04b_new_ale_output_root():
    if not RUN_STEP04B_REVIEWED_ALE_RECALCULATION:
        raise RuntimeError('Set FINAL_STEP04B_ALE_RECALCULATE=1 for an explicit new run')
    if not STEP04B_REVIEWED_ALE_NEW_OUTPUT_ROOT:
        raise RuntimeError(
            'FINAL_STEP04B_ALE_NEW_OUTPUT_ROOT must name a new unique output directory'
        )
    output_root = _Path(STEP04B_REVIEWED_ALE_NEW_OUTPUT_ROOT).expanduser().resolve()
    allowed_parent = _Path(
        FINAL_WORKSPACE_DIR, 'recalculations', 'step04b_dlbcl_ale_marker_response'
    ).resolve()
    if output_root.parent != allowed_parent:
        raise RuntimeError(
            f'New Step04B ALE output must be one direct child of {allowed_parent}: {output_root}'
        )
    if output_root.exists() or output_root.with_name(f'.{output_root.name}.staging').exists():
        raise FileExistsError(f'New Step04B ALE output/staging path already exists: {output_root}')
    return output_root


def _step04b_build_reviewed_ale_calculation_config(output_root, execution_profile='fast'):
    """Build the four-lineage frozen XGB+MLP ALE calculation; no fitting occurs."""
    from frozen_dlbcl_marker_response_utils import FrozenResponseStage, ResponseRunConfig

    stages = tuple(
        FrozenResponseStage(
            stage_key=stage_key,
            lineage=lineage,
            checkpoint_path=_Path(CHECKPOINT_STEP04),
            model_root=_Path(FINAL_CHECKPOINT_ROOT, 'transfer_models', 'subtype', stage_key),
            source_run_root=STEP04B_REVIEWED_ALE_SOURCE_RUN,
            output_root=_Path(output_root),
            route_column='ML_Lineage_Raw',
            route_value=lineage,
            saved_prediction_column='ML_Subtype_Raw',
            expected_classes=classes,
        )
        for stage_key, lineage, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS
    )
    return ResponseRunConfig(
        workflow='Step04B frozen XGB+MLP DLBCL subtype marker response',
        stages=stages,
        execution_profile=execution_profile,
    )


def _step04b_run_new_reviewed_ale_calculation():
    """Explicit future-only ALE path; completed sources can never be overwritten."""
    output_root = _step04b_new_ale_output_root()
    from frozen_dlbcl_marker_response_utils import run

    profile = os.environ.get('FINAL_STEP04B_ALE_EXECUTION_PROFILE', 'fast')
    config = _step04b_build_reviewed_ale_calculation_config(output_root, profile)
    caller_path = _Path(globals().get(
        '__file__', _Path(FINAL_WORKSPACE_DIR, 'scripts', 'Final_step04B.py')
    )).resolve()
    return run(config, caller_paths=(caller_path,))


def _step04b_as_bool_series(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return (
        values.fillna('').astype(str).str.strip().str.lower()
        .isin({'true', '1', 'yes'})
    )


def _step04b_load_reviewed_display_source():
    """Load only the frozen Final-local ALE/importance columns used for display."""
    summary_path = STEP04B_REVIEWED_ALE_ROOT / 'tables' / 'class_feature_marker_response_summary.csv'
    table = pd.read_csv(summary_path)
    missing = sorted(set(STEP04B_REVIEWED_DISPLAY_SELECTION_COLUMNS) - set(table.columns))
    if missing:
        raise RuntimeError(f'Step04B display source lacks required columns: {missing}')
    expected_rows = sum(
        len(classes) for _, _, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS
    ) * int(adata.n_vars)
    if len(table) != expected_rows or table.duplicated(['stage', 'class', 'feature']).any():
        raise RuntimeError(
            'Step04B display source must contain one row per saved class/feature: '
            f'expected={expected_rows}, observed={len(table)}'
        )
    if set(table['response_metric'].astype(str)) != {
        'patient-equal first-order ALE q10-to-q90 contrast'
    }:
        raise RuntimeError('Step04B display source uses a different ALE response metric')

    table = table.loc[:, STEP04B_REVIEWED_DISPLAY_SELECTION_COLUMNS].copy()
    numeric_columns = (
        'feature_position', 'mean_ale_q10_to_q90_probability_change',
        'pair_sign_consistency', 'support_sign_concordance',
        'direction_consistency', 'direction_consistency_threshold', 'figure_alpha',
        'class_normalized_importance_percent',
        'positive_top_k_attribution_frequency_percent',
    )
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors='coerce')
    table['source_join_complete'] = _step04b_as_bool_series(table['source_join_complete'])
    table['absolute_ale_q10_to_q90'] = table[
        'mean_ale_q10_to_q90_probability_change'
    ].abs()

    available = set(table['feature'].astype(str))
    include = set(STEP04B_REVIEWED_DISPLAY_MARKER_INCLUDE)
    exclude = set(STEP04B_REVIEWED_DISPLAY_MARKER_EXCLUDE)
    overlap = sorted(include & exclude)
    unknown = sorted((include | exclude) - available)
    if overlap:
        raise ValueError(f'Step04B display marker include/exclude overlap: {overlap}')
    if unknown:
        raise ValueError(f'Step04B configured display markers are absent: {unknown}')
    if include:
        table = table.loc[table['feature'].astype(str).isin(include)].copy()
    if exclude:
        table = table.loc[~table['feature'].astype(str).isin(exclude)].copy()
    if table.empty:
        raise RuntimeError('Step04B configured display marker filter removed every source row')
    return summary_path, table.reset_index(drop=True)


def _step04b_select_reviewed_bubble_rows(source):
    """Apply the documented top-15 Bubble display ranking to the current marker filter."""
    importance = pd.to_numeric(
        source['class_normalized_importance_percent'], errors='coerce'
    )
    ranked = source.loc[np.isfinite(importance) & importance.gt(0)].copy()
    ranked = ranked.sort_values(
        ['stage', 'class', 'class_normalized_importance_percent',
         'feature_position', 'feature'],
        ascending=[True, True, False, True, True], kind='mergesort',
    )
    ranked['display_rank_within_class'] = (
        ranked.groupby(['stage', 'class'], sort=False).cumcount() + 1
    )
    selected = ranked.loc[
        ranked['display_rank_within_class'].le(
            STEP04B_REVIEWED_BUBBLE_TOP_N_PER_CLASS
        )
    ].copy()
    missing_classes = [
        (stage, class_name)
        for stage, _, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS
        for class_name in classes
        if not (
            selected['stage'].astype(str).eq(stage)
            & selected['class'].astype(str).eq(class_name)
        ).any()
    ]
    if missing_classes:
        raise RuntimeError(
            f'Step04B Bubble marker filter leaves empty class blocks: {missing_classes}'
        )
    return selected.reset_index(drop=True)


def _step04b_select_reviewed_heatmap_candidates(source):
    """Keep stable signed candidates and rank the current top-N pool per subtype."""
    eligible = source.loc[
        source['source_join_complete']
        & source['direction_symbol'].astype(str).isin({'+', '−'})
        & np.isfinite(source['class_normalized_importance_percent'])
        & source['class_normalized_importance_percent'].gt(0)
        & source['direction_consistency'].ge(source['direction_consistency_threshold'])
    ].copy()
    if eligible.empty:
        raise RuntimeError('No stable signed Step04B Heatmap candidates passed the display gate')
    ranked = eligible.sort_values(
        ['stage', 'class', 'class_normalized_importance_percent',
         'positive_top_k_attribution_frequency_percent', 'direction_consistency',
         'absolute_ale_q10_to_q90', 'feature_position', 'feature'],
        ascending=[True, True, False, False, False, False, True, True],
        kind='mergesort',
    ).copy()
    ranked['importance_candidate_rank'] = (
        ranked.groupby(['stage', 'class'], sort=False).cumcount() + 1
    )
    pool = ranked.loc[
        ranked['importance_candidate_rank'].le(
            STEP04B_REVIEWED_HEATMAP_CANDIDATE_TOP_N_PER_CLASS
        )
    ].reset_index(drop=True)
    return eligible.reset_index(drop=True), pool


def _step04b_dense_float64(values):
    if hasattr(values, 'toarray'):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64)


def _step04b_compute_reviewed_global_z_profiles(candidate_pool):
    """Stream frozen expression once; aggregate only saved-label Figure profiles."""
    required_obs = {
        'Disease', 'ML_Lineage_Raw',
        'ML_Subtype_Raw', 'Pred_SC_Centroid', 'Pred_Clustering',
    }
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    feature_names = list(map(str, adata.var_names))
    feature_lookup = {name: index for index, name in enumerate(feature_names)}
    missing_features = sorted(
        set(candidate_pool['feature'].astype(str)) - set(feature_lookup)
    )
    if missing_obs or missing_features:
        raise RuntimeError(
            'Step04B frozen profile input contract failed: '
            f'missing_obs={missing_obs}, missing_features={missing_features}'
        )

    method_specs = (
        ('ML_Subtype_Raw', 'Proposed ML (Ours: XGB+MLP)'),
        ('Pred_SC_Centroid', 'SC-Centroid'),
        ('Pred_Clustering', 'Clustering (Leiden)'),
    )
    n_vars = int(adata.n_vars)
    global_sum = np.zeros(n_vars, dtype=np.float64)
    global_square_sum = np.zeros(n_vars, dtype=np.float64)
    group_sums = {}
    group_counts = {}
    for stage, _, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS:
        for method_column, _ in method_specs:
            for class_name in classes:
                key = (stage, method_column, class_name)
                group_sums[key] = np.zeros(n_vars, dtype=np.float64)
                group_counts[key] = 0

    chunk_size = int(STEP04B_REVIEWED_PROFILE_CHUNK_SIZE)
    if chunk_size < 1:
        raise ValueError('STEP04B_REVIEWED_PROFILE_CHUNK_SIZE must be positive')
    for start in range(0, int(adata.n_obs), chunk_size):
        stop = min(start + chunk_size, int(adata.n_obs))
        matrix = _step04b_dense_float64(adata.X[start:stop, :])
        global_sum += matrix.sum(axis=0, dtype=np.float64)
        global_square_sum += np.square(matrix, dtype=np.float64).sum(
            axis=0, dtype=np.float64
        )
        obs = adata.obs.iloc[start:stop]
        disease_mask = (
            obs['Disease'].astype(str).str.contains('DLBCL', case=False, na=False)
            .to_numpy()
        )
        route_values = obs['ML_Lineage_Raw'].astype(str).to_numpy()
        for stage, lineage, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS:
            lineage_mask = disease_mask & (route_values == lineage)
            if not bool(lineage_mask.any()):
                continue
            for method_column, _ in method_specs:
                predictions = obs[method_column].astype(str).to_numpy()
                for class_name in classes:
                    mask = lineage_mask & (predictions == class_name)
                    count = int(mask.sum())
                    if count:
                        key = (stage, method_column, class_name)
                        group_sums[key] += matrix[mask].sum(axis=0, dtype=np.float64)
                        group_counts[key] += count

    denominator = int(adata.n_obs)
    if denominator < 1:
        raise RuntimeError('Step04B frozen checkpoint has no cells')
    global_mean = global_sum / denominator
    global_variance = np.maximum(
        global_square_sum / denominator - np.square(global_mean), 0.0
    )
    global_scale = np.sqrt(global_variance)
    global_scale[global_scale == 0] = 1.0
    global_stats = pd.DataFrame({
        'feature': feature_names,
        'global_mean': global_mean,
        'global_variance': global_variance,
        'global_scale': global_scale,
        'global_n_obs': denominator,
    })

    profile_rows = []
    for stage, lineage, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS:
        stage_features = (
            candidate_pool.loc[candidate_pool['stage'].astype(str).eq(stage), 'feature']
            .astype(str).drop_duplicates().tolist()
        )
        for method_column, method_title in method_specs:
            for class_name in classes:
                key = (stage, method_column, class_name)
                count = group_counts[key]
                mean_z = (
                    (group_sums[key] / count - global_mean) / global_scale
                    if count else np.full(n_vars, np.nan, dtype=np.float64)
                )
                for feature in stage_features:
                    profile_rows.append({
                        'stage': stage, 'lineage': lineage,
                        'method_column': method_column, 'method_title': method_title,
                        'predicted_subtype': class_name, 'feature': feature,
                        'mean_global_z': float(mean_z[feature_lookup[feature]]),
                        'group_cell_count': int(count),
                    })
    profiles = pd.DataFrame(profile_rows)
    if profiles.duplicated(
        ['stage', 'method_column', 'predicted_subtype', 'feature']
    ).any():
        raise RuntimeError('Step04B derived profile table contains duplicate group/marker rows')
    group_report = [
        {
            'stage': stage,
            'lineage': next(
                lineage for stage_key, lineage, _ in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS
                if stage_key == stage
            ),
            'method_column': method_column,
            'predicted_subtype': class_name,
            'cells': int(count),
        }
        for (stage, method_column, class_name), count in sorted(group_counts.items())
    ]
    return global_stats, profiles, group_report


def _step04b_select_reviewed_heatmap_rows(candidate_pool, profiles):
    """Apply the configured sign, specificity, cap, dedup, and row-order contract."""
    anchor = profiles.loc[
        profiles['method_column'].astype(str).eq('ML_Subtype_Raw')
    ].copy()
    if anchor.duplicated(['stage', 'feature', 'predicted_subtype']).any():
        raise RuntimeError('Step04B DLBCL selection anchor contains duplicate profiles')
    profile_lookup = {
        (str(row.stage), str(row.feature), str(row.predicted_subtype)):
            float(row.mean_global_z)
        for row in anchor.itertuples(index=False)
    }
    stage_classes = {
        stage: tuple(classes)
        for stage, _, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS
    }
    scored_rows = []
    for candidate in candidate_pool.to_dict(orient='records'):
        stage = str(candidate['stage'])
        class_name = str(candidate['class'])
        feature = str(candidate['feature'])
        target_z = profile_lookup.get((stage, feature, class_name), np.nan)
        other_values = [
            profile_lookup.get((stage, feature, other), np.nan)
            for other in stage_classes[stage] if other != class_name
        ]
        finite_other = [value for value in other_values if np.isfinite(value)]
        max_other = max(finite_other) if finite_other else np.nan
        min_other = min(finite_other) if finite_other else np.nan
        row = dict(candidate)
        row.update({
            'dlbcl_selection_anchor': 'ML_Subtype_Raw',
            'dlbcl_target_mean_global_z': float(target_z),
            'dlbcl_max_other_mean_global_z': float(max_other),
            'dlbcl_min_other_mean_global_z': float(min_other),
            'dlbcl_target_margin': float(target_z - max_other),
            'dlbcl_negative_specificity': float(min_other - target_z),
        })
        scored_rows.append(row)
    scored = pd.DataFrame(scored_rows)
    margin = float(STEP04B_REVIEWED_HEATMAP_TARGET_MARGIN_MIN)
    class_order = {
        (stage, class_name): class_index
        for stage, _, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS
        for class_index, class_name in enumerate(classes)
    }

    positive = scored.loc[
        scored['direction_symbol'].astype(str).eq('+')
        & np.isfinite(scored['dlbcl_target_margin'])
        & scored['dlbcl_target_margin'].ge(margin)
    ].copy()
    positive = (
        positive.sort_values(
            ['stage', 'dlbcl_target_margin', 'dlbcl_target_mean_global_z',
             'class_normalized_importance_percent',
             'positive_top_k_attribution_frequency_percent', 'direction_consistency',
             'feature_position', 'class'],
            ascending=[True, False, False, False, False, False, True, True],
            kind='mergesort',
        ).drop_duplicates(['stage', 'feature'], keep='first').copy()
    )
    positive['class_order'] = [
        class_order[(str(stage), str(class_name))]
        for stage, class_name in zip(positive['stage'], positive['class'])
    ]
    positive = positive.sort_values(
        ['stage', 'class_order', 'dlbcl_target_mean_global_z',
         'dlbcl_target_margin', 'class_normalized_importance_percent',
         'positive_top_k_attribution_frequency_percent', 'direction_consistency',
         'feature_position', 'feature'],
        ascending=[True, True, False, False, False, False, False, True, True],
        kind='mergesort',
    ).groupby(['stage', 'class'], sort=False, group_keys=False).head(
        STEP04B_REVIEWED_HEATMAP_MAX_POSITIVE_PER_CLASS
    ).copy()
    positive['selection_direction'] = 'positive'
    selected_keys = set(zip(positive['stage'].astype(str), positive['feature'].astype(str)))

    negative = scored.loc[
        scored['direction_symbol'].astype(str).eq('−')
        & np.isfinite(scored['dlbcl_negative_specificity'])
        & scored['dlbcl_negative_specificity'].ge(margin)
    ].copy()
    negative = negative.loc[
        [(str(row.stage), str(row.feature)) not in selected_keys
         for row in negative.itertuples(index=False)]
    ].copy()
    negative = (
        negative.sort_values(
            ['stage', 'dlbcl_negative_specificity', 'dlbcl_target_mean_global_z',
             'class_normalized_importance_percent',
             'positive_top_k_attribution_frequency_percent', 'direction_consistency',
             'feature_position', 'class'],
            ascending=[True, False, True, False, False, False, True, True],
            kind='mergesort',
        ).drop_duplicates(['stage', 'feature'], keep='first').copy()
    )
    negative['class_order'] = [
        class_order[(str(stage), str(class_name))]
        for stage, class_name in zip(negative['stage'], negative['class'])
    ]
    negative = negative.sort_values(
        ['stage', 'class_order', 'dlbcl_negative_specificity',
         'dlbcl_target_mean_global_z', 'class_normalized_importance_percent',
         'positive_top_k_attribution_frequency_percent', 'direction_consistency',
         'feature_position', 'feature'],
        ascending=[True, True, False, True, False, False, False, True, True],
        kind='mergesort',
    ).groupby(['stage', 'class'], sort=False, group_keys=False).head(
        STEP04B_REVIEWED_HEATMAP_MAX_NEGATIVE_PER_CLASS
    ).copy()
    negative['selection_direction'] = 'negative'

    selected = pd.concat([positive, negative], ignore_index=True, sort=False)
    selected['direction_block_order'] = selected['selection_direction'].map(
        {'positive': 0, 'negative': 1}
    )
    selected['dlbcl_directional_specificity'] = np.where(
        selected['selection_direction'].eq('positive'),
        selected['dlbcl_target_margin'], selected['dlbcl_negative_specificity'],
    )
    selected['dlbcl_directional_target_magnitude'] = np.where(
        selected['selection_direction'].eq('positive'),
        selected['dlbcl_target_mean_global_z'], -selected['dlbcl_target_mean_global_z'],
    )
    selected = selected.sort_values(
        ['stage', 'class_order', 'direction_block_order',
         'dlbcl_directional_target_magnitude', 'dlbcl_directional_specificity',
         'class_normalized_importance_percent',
         'positive_top_k_attribution_frequency_percent', 'feature_position', 'feature'],
        ascending=[True, True, True, False, False, False, False, True, True],
        kind='mergesort', na_position='last',
    ).reset_index(drop=True)
    if selected.empty or selected.duplicated(['stage', 'feature']).any():
        raise RuntimeError('Step04B Heatmap selection is empty or marker ownership is not unique')
    selected['selection_contract'] = (
        'stable signed Final-local ALE plus positive importance; current marker filter; '
        'top-N importance candidates; Proposed ML DLBCL sign-matched margin; positive '
        'stage-feature rows own duplicates; configured per-class caps; no model inference'
    )
    selected['row_order_contract'] = (
        'saved subtype order; positive then negative; directional target magnitude, '
        'specificity, importance, attribution frequency, feature position, feature'
    )
    return selected.drop(columns=['class_order', 'direction_block_order'])


def _step04b_reviewed_display_selection_report(eligible, pool, selected):
    counts = []
    for stage, lineage, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS:
        for class_name in classes:
            rows = selected.loc[
                selected['stage'].astype(str).eq(stage)
                & selected['class'].astype(str).eq(class_name)
            ]
            counts.append({
                'stage': stage, 'lineage': lineage, 'class': class_name,
                'selected_markers': int(len(rows)),
                'selected_positive_markers': int(
                    rows['selection_direction'].astype(str).eq('positive').sum()
                ),
                'selected_negative_markers': int(
                    rows['selection_direction'].astype(str).eq('negative').sum()
                ),
            })
    return {
        'display_selection_population': 'DLBCL cells routed by saved ML_Lineage_Raw',
        'display_selection_anchor': 'ML_Subtype_Raw',
        'marker_include': list(STEP04B_REVIEWED_DISPLAY_MARKER_INCLUDE),
        'marker_exclude': list(STEP04B_REVIEWED_DISPLAY_MARKER_EXCLUDE),
        'candidate_top_n_per_class': STEP04B_REVIEWED_HEATMAP_CANDIDATE_TOP_N_PER_CLASS,
        'positive_target_margin_min': STEP04B_REVIEWED_HEATMAP_TARGET_MARGIN_MIN,
        'negative_target_margin_min': STEP04B_REVIEWED_HEATMAP_TARGET_MARGIN_MIN,
        'max_positive_per_class': STEP04B_REVIEWED_HEATMAP_MAX_POSITIVE_PER_CLASS,
        'max_negative_per_class': STEP04B_REVIEWED_HEATMAP_MAX_NEGATIVE_PER_CLASS,
        'ale_stable_signed_candidates': int(len(eligible)),
        'profiled_top_importance_candidates': int(len(pool)),
        'selected_rows': int(len(selected)),
        'counts': counts,
        'empty_subtype_blocks': [
            {'stage': row['stage'], 'class': row['class']}
            for row in counts if row['selected_markers'] == 0
        ],
    }


def _step04b_checkpoint_scan_snapshot():
    checkpoint = _Path(CHECKPOINT_STEP04).resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f'Step04B frozen checkpoint is missing: {checkpoint}')
    stat = checkpoint.stat()
    return {
        'checkpoint': str(checkpoint),
        'size_bytes': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def _step04b_refresh_reviewed_display_derivatives():
    """Regenerate only approved display-derived tables and promote manifest last."""
    _step04b_validate_completed_ale_bundle()
    summary_path, source = _step04b_load_reviewed_display_source()
    bubble = _step04b_select_reviewed_bubble_rows(source)
    eligible, candidate_pool = _step04b_select_reviewed_heatmap_candidates(source)
    checkpoint_before_scan = _step04b_checkpoint_scan_snapshot()
    global_stats, profiles, group_report = _step04b_compute_reviewed_global_z_profiles(
        candidate_pool
    )
    checkpoint_after_scan = _step04b_checkpoint_scan_snapshot()
    checkpoint_size_mtime_unchanged = (
        checkpoint_before_scan == checkpoint_after_scan
    )
    if not checkpoint_size_mtime_unchanged:
        raise RuntimeError(
            'Step04B frozen checkpoint size/mtime changed during the profile scan: '
            f'before={checkpoint_before_scan}, after={checkpoint_after_scan}'
        )
    selected = _step04b_select_reviewed_heatmap_rows(candidate_pool, profiles)
    selected_keys = selected[['stage', 'feature']].drop_duplicates()
    selected_profiles = profiles.merge(
        selected_keys, on=['stage', 'feature'], how='inner', validate='many_to_one'
    )
    finite = np.abs(
        selected_profiles.loc[
            np.isfinite(selected_profiles['mean_global_z']), 'mean_global_z'
        ].to_numpy(dtype=float)
    )
    if finite.size == 0:
        raise RuntimeError('Step04B selected marker profiles contain no finite values')
    vlim = float(np.clip(np.nanpercentile(finite, 95.0), 1.0, 5.0))

    generation_id = _uuid.uuid4().hex
    for frame in (bubble, global_stats, selected, selected_profiles):
        frame.insert(0, 'derived_generation_id', generation_id)
    root = STEP04B_REVIEWED_ALE_HEATMAP_ROOT.resolve()
    if root.parent != STEP04B_REVIEWED_ALE_ROOT.resolve():
        raise RuntimeError(f'Step04B derived Heatmap root escaped canonical ALE root: {root}')
    root.mkdir(parents=True, exist_ok=True)
    staging_root = root / f'.display-derived.staging.{generation_id}'
    if staging_root.exists():
        raise FileExistsError(f'Step04B derived staging path already exists: {staging_root}')
    staging_paths = {
        key: staging_root / target.name
        for key, target in STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS.items()
    }
    backup_paths = {
        key: staging_root / f'{target.name}.backup'
        for key, target in STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS.items()
    }
    exact_paths = [
        *map(str, staging_paths.values()),
        *map(str, STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS.values()),
        *map(str, backup_paths.values()),
    ]
    frames = {
        'bubble_selection': bubble,
        'global_z_stats': global_stats,
        'heatmap_selection': selected,
        'heatmap_profiles': selected_profiles,
    }
    promotion_order = (*frames.keys(), 'manifest')
    initial_target_exists = {
        key: STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS[key].exists()
        for key in promotion_order
    }
    with allow_exact_final_derived_writes(
        exact_paths, owner='Final_step04B.display_derivatives'
    ):
        staging_root.mkdir(parents=False, exist_ok=False)
        backed_up = []
        promoted = []
        try:
            for key, frame in frames.items():
                frame.to_csv(staging_paths[key], index=False, float_format='%.17g')
            table_hashes = {
                key: _step04b_sha256(path)
                for key, path in staging_paths.items() if key != 'manifest'
            }
            manifest = {
                'status': 'complete',
                'derived_generation_id': generation_id,
                'workflow': 'Step04B Final-local dynamic display derivatives',
                'render_contract': STEP04B_REVIEWED_DISPLAY_RENDER_CONTRACT,
                'source_ale_summary': str(summary_path.resolve()),
                'source_ale_summary_sha256': _step04b_sha256(summary_path),
                'source_checkpoint_read_only': {
                    'checkpoint': checkpoint_before_scan['checkpoint'],
                    'before_profile_scan': checkpoint_before_scan,
                    'after_profile_scan': checkpoint_after_scan,
                    'size_mtime_unchanged': checkpoint_size_mtime_unchanged,
                    'n_obs': int(adata.n_obs), 'n_vars': int(adata.n_vars),
                    'grouping_columns': [
                        'ML_Subtype_Raw', 'Pred_SC_Centroid', 'Pred_Clustering'
                    ],
                },
                'bubble_selection': {
                    'marker_include': list(STEP04B_REVIEWED_DISPLAY_MARKER_INCLUDE),
                    'marker_exclude': list(STEP04B_REVIEWED_DISPLAY_MARKER_EXCLUDE),
                    'rank': 'importance percent desc, feature position asc, feature asc',
                    'top_n_per_class': STEP04B_REVIEWED_BUBBLE_TOP_N_PER_CLASS,
                    'rows': int(len(bubble)),
                    'table': str(
                        STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS['bubble_selection'].resolve()
                    ),
                },
                'selection': _step04b_reviewed_display_selection_report(
                    eligible, candidate_pool, selected
                ),
                'profile_contract': {
                    'global_z': 'population mean and variance across full saved Step04B matrix',
                    'groups': (
                        'Disease contains DLBCL; saved ML_Lineage_Raw route; '
                        'grouped by each saved prediction column'
                    ),
                    'chunk_size': STEP04B_REVIEWED_PROFILE_CHUNK_SIZE,
                    'group_counts': group_report,
                },
                'cell_values': (
                    'mean global marker z-score within routed DLBCL cells by saved prediction'
                ),
                'shared_color_norm': {'vmin': -vlim, 'center': 0.0, 'vmax': vlim},
                'tables': {
                    key: {
                        'path': str(STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS[key].resolve()),
                        'sha256': table_hashes[key], 'rows': int(len(frame)),
                    }
                    for key, frame in frames.items()
                },
                'model_inference': False, 'model_retraining': False,
                'ale_recalculation': False, 'importance_recalculation': False,
                'figure_profile_aggregation_executed': True,
                'heavy_calculation_executed': False,
                'promotion_contract': {
                    'unique_generation_backup': True,
                    'manifest_promoted_last': True,
                    'rollback_restores_initial_presence': True,
                    'initial_target_exists': initial_target_exists,
                },
            }
            with staging_paths['manifest'].open('w', encoding='utf-8') as handle:
                _json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write('\n')

            for key, frame in frames.items():
                staged = pd.read_csv(staging_paths[key])
                if len(staged) != len(frame):
                    raise RuntimeError(f'Step04B staged derived row count mismatch: {key}')
                if set(staged['derived_generation_id'].astype(str)) != {generation_id}:
                    raise RuntimeError(f'Step04B staged generation mismatch: {key}')
                if _step04b_sha256(staging_paths[key]) != manifest['tables'][key]['sha256']:
                    raise RuntimeError(f'Step04B staged derived hash mismatch: {key}')
            staged_manifest = _json.loads(
                staging_paths['manifest'].read_text(encoding='utf-8')
            )
            if (
                staged_manifest.get('status') != 'complete'
                or staged_manifest.get('derived_generation_id') != generation_id
            ):
                raise RuntimeError('Step04B staged derived manifest validation failed')

            for key in promotion_order:
                target = STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS[key]
                if target.exists() != initial_target_exists[key]:
                    raise RuntimeError(
                        'Step04B derived target presence changed before backup: '
                        f'{target}'
                    )
                if initial_target_exists[key]:
                    os.replace(target, backup_paths[key])
                    backed_up.append(key)
            for key in promotion_order:
                target = STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS[key]
                if target.exists():
                    raise RuntimeError(
                        f'Step04B derived target reappeared after backup: {target}'
                    )
                os.replace(
                    staging_paths[key],
                    target,
                )
                promoted.append(key)
        except Exception as promotion_error:
            rollback_errors = []
            for key in reversed(promoted):
                target = STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS[key]
                try:
                    if target.exists():
                        target.unlink()
                except Exception as error:
                    rollback_errors.append(f'remove promoted {target}: {error}')
            restored = set()
            for key in reversed(backed_up):
                target = STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS[key]
                try:
                    if target.exists():
                        target.unlink()
                    os.replace(backup_paths[key], target)
                    restored.add(key)
                except Exception as error:
                    rollback_errors.append(f'restore backup {target}: {error}')
            for key in promotion_order:
                target = STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS[key]
                if target.exists() != initial_target_exists[key]:
                    rollback_errors.append(
                        f'presence mismatch after rollback {target}: '
                        f'expected={initial_target_exists[key]}, observed={target.exists()}'
                    )
            for path in staging_paths.values():
                try:
                    if path.exists():
                        path.unlink()
                except Exception as error:
                    rollback_errors.append(f'cleanup staged {path}: {error}')
            for key, path in backup_paths.items():
                try:
                    if path.exists() and (key not in backed_up or key in restored):
                        path.unlink()
                except Exception as error:
                    rollback_errors.append(f'cleanup backup {path}: {error}')
            if not rollback_errors and staging_root.exists():
                staging_root.rmdir()
            if rollback_errors:
                raise RuntimeError(
                    'Step04B derived promotion failed and rollback was incomplete: '
                    + '; '.join(rollback_errors)
                ) from promotion_error
            raise

        for path in backup_paths.values():
            if path.exists():
                path.unlink()
        for path in staging_paths.values():
            if path.exists():
                raise RuntimeError(
                    f'Step04B committed promotion left a staged file: {path}'
                )
        staging_root.rmdir()
    return manifest


def _step04b_reviewed_display_payload():
    """Load and hash-validate one promoted derived display generation."""
    paths = STEP04B_REVIEWED_DISPLAY_DERIVED_PATHS
    missing = [
        str(path) for path in paths.values()
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise FileNotFoundError('Missing Step04B display derivatives: ' + ', '.join(missing))
    manifest = _json.loads(paths['manifest'].read_text(encoding='utf-8'))
    if (
        manifest.get('status') != 'complete'
        or manifest.get('render_contract') != STEP04B_REVIEWED_DISPLAY_RENDER_CONTRACT
        or manifest.get('source_checkpoint_read_only', {}).get(
            'size_mtime_unchanged'
        ) is not True
    ):
        raise RuntimeError('Step04B derived display manifest contract failed')
    summary_path = STEP04B_REVIEWED_ALE_ROOT / 'tables' / 'class_feature_marker_response_summary.csv'
    if manifest.get('source_ale_summary_sha256') != _step04b_sha256(summary_path):
        raise RuntimeError('Step04B derived display source-summary hash changed')
    generation_id = str(manifest.get('derived_generation_id', ''))
    payload = {'manifest': manifest}
    for key in ('bubble_selection', 'global_z_stats', 'heatmap_selection', 'heatmap_profiles'):
        audit = manifest.get('tables', {}).get(key, {})
        if audit.get('sha256') != _step04b_sha256(paths[key]):
            raise RuntimeError(f'Step04B derived display table hash mismatch: {key}')
        frame = pd.read_csv(paths[key])
        if len(frame) != int(audit.get('rows', -1)):
            raise RuntimeError(f'Step04B derived display table row mismatch: {key}')
        if set(frame['derived_generation_id'].astype(str)) != {generation_id}:
            raise RuntimeError(f'Step04B derived display generation mismatch: {key}')
        payload[key] = frame.drop(columns=['derived_generation_id'])
    selected = payload['heatmap_selection']
    if selected.empty or selected.duplicated(['stage', 'feature']).any():
        raise RuntimeError('Step04B derived Heatmap marker ownership validation failed')
    return payload


def _step04b_reviewed_bubble_size(percent):
    return 0.115 * (28.0 + 3.2 * max(0.0, float(percent)))


def _step04b_reviewed_rgba(color, alpha):
    red, green, blue = matplotlib.colors.to_rgb(color)
    return red, green, blue, float(alpha)


def _render_step04b_reviewed_ale_bubble(show=False):
    """Render the Bubble from the current promoted top-15 display selection."""
    if not _final_figure_stem_registered(STEP04B_REVIEWED_ALE_BUBBLE_STEM):
        raise RuntimeError(
            'Step04B canonical Bubble stem is not registered: '
            f'{STEP04B_REVIEWED_ALE_BUBBLE_STEM}'
        )
    _step04b_require_arial()
    table = _step04b_reviewed_display_payload()['bubble_selection']
    required = {
        'stage', 'class', 'feature', 'feature_position', 'direction_symbol',
        'figure_alpha', 'class_normalized_importance_percent',
        'positive_top_k_attribution_frequency_percent', 'response_metric',
        'display_rank_within_class',
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f'Step04B reviewed ALE summary lacks Figure columns: {missing}')
    if set(table['response_metric'].astype(str)) != {
        'patient-equal first-order ALE q10-to-q90 contrast'
    }:
        raise RuntimeError('Step04B reviewed ALE direction metric changed')

    selected = {}
    for stage, _, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS:
        selected[stage] = {}
        for class_name in classes:
            rows = table.loc[
                table['stage'].astype(str).eq(stage)
                & table['class'].astype(str).eq(class_name)
            ].copy()
            rows = rows.sort_values(
                ['display_rank_within_class'], ascending=[True], kind='mergesort',
            )
            if rows.empty:
                raise RuntimeError(f'No reviewed Step04B ALE rows for {stage}/{class_name}')
            selected[stage][class_name] = rows.reset_index(drop=True)

    panel_specs = (
        ('T cells', 'T_cells', ('Th', 'Tfh')),
        ('T cells', 'T_cells', ('Treg', 'CD4_CTL')),
        ('T cells', 'T_cells', ('CD8T', 'Tpr')),
        ('B cells', 'B_cells', ('B cell', 'Plasma cell')),
        ('Myeloid cells', 'Myeloid_cells', ('Myeloid', 'Macrophage', 'DC')),
        ('Stromal cells', 'Stromal_cells', ('Stromal', 'FDC')),
    )
    lineage_title_colors = {
        'T_cells': STEP04B_REVIEWED_ALE_CLASS_COLORS['Treg'],
        'B_cells': STEP04B_REVIEWED_ALE_CLASS_COLORS['B cell'],
        'Myeloid_cells': STEP04B_REVIEWED_ALE_CLASS_COLORS['Myeloid'],
        'Stromal_cells': STEP04B_REVIEWED_ALE_CLASS_COLORS['Stromal'],
    }

    def add_panel_title(axis, lineage_title, stage, classes):
        title_props = {'fontsize': 6.8, 'fontweight': 'bold', 'fontfamily': 'Arial'}
        lineage_box = TextArea(
            lineage_title,
            textprops={**title_props, 'color': lineage_title_colors[stage]},
        )
        subtype_parts = []
        for class_index, class_name in enumerate(classes):
            if class_index:
                subtype_parts.append(TextArea(
                    ' / ', textprops={**title_props, 'color': '#555555'},
                ))
            subtype_parts.append(TextArea(
                STEP04B_REVIEWED_ALE_CLASS_DISPLAY.get(class_name, class_name),
                textprops={
                    **title_props,
                    'color': STEP04B_REVIEWED_ALE_CLASS_COLORS[class_name],
                },
            ))
        subtype_box = HPacker(children=subtype_parts, align='center', pad=0, sep=0)
        title_box = VPacker(
            children=[lineage_box, subtype_box], align='center', pad=0, sep=1.5,
        )
        axis.add_artist(AnchoredOffsetbox(
            loc='lower center', child=title_box, frameon=False,
            bbox_to_anchor=(0.5, 1.004), bbox_transform=axis.transAxes,
            borderpad=0, pad=0,
        ))

    fig = plt.figure(
        figsize=tuple(value / 2.54 for value in STEP04B_REVIEWED_ALE_BUBBLE_CANVAS_CM),
        facecolor='white',
    )
    grid = fig.add_gridspec(
        3, 3, left=0.085, right=0.990, top=0.960, bottom=0.018,
        height_ratios=[1.0, 1.45, 0.18], wspace=0.20, hspace=0.25,
    )
    for panel_index, (lineage_title, stage, classes) in enumerate(panel_specs):
        plot_row, plot_column = divmod(panel_index, 3)
        axis = fig.add_subplot(grid[plot_row, plot_column])
        blocks, boundaries, offset = [], [], 0
        for class_index, class_name in enumerate(classes):
            block = selected[stage][class_name].copy()
            block['panel_class'] = class_name
            block['panel_y'] = np.arange(offset, offset + len(block), dtype=float)
            blocks.append(block)
            offset += len(block)
            if class_index < len(classes) - 1:
                boundaries.append(offset - 0.5)
        rows = pd.concat(blocks, ignore_index=True)
        y = rows['panel_y'].to_numpy(dtype=float)
        for row_y in y:
            axis.axhline(row_y, color='#D3D3D3', linewidth=0.20, alpha=0.48, zorder=0)
        for boundary in boundaries:
            axis.axhline(boundary, color='#888888', linewidth=0.35,
                        linestyle='--', alpha=0.62, zorder=1)
        axis.scatter(
            rows['class_normalized_importance_percent'], y,
            s=[_step04b_reviewed_bubble_size(v) for v in rows['positive_top_k_attribution_frequency_percent']],
            c=[_step04b_reviewed_rgba(STEP04B_REVIEWED_ALE_CLASS_COLORS[c], a)
               for c, a in zip(rows['panel_class'], rows['figure_alpha'])],
            edgecolors='#111111', linewidths=0.34, zorder=3,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(
            [f"{m} {s}" for m, s in zip(rows['feature'], rows['direction_symbol'])],
            fontsize=4.6 if len(rows) <= 30 else 3.9, fontweight='bold',
        )
        for tick, class_name in zip(axis.get_yticklabels(), rows['panel_class']):
            tick.set_color(STEP04B_REVIEWED_ALE_CLASS_COLORS[class_name])
        axis.set_ylim(len(rows) - 0.35, -1.10)
        panel_max = float(rows['class_normalized_importance_percent'].max())
        panel_right = max(5.0, panel_max * 1.10)
        axis.set_xlim(-0.045 * panel_right, panel_right)
        locator = matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        axis.set_xticks([
            value for value in locator.tick_values(0.0, panel_right)
            if 0.0 <= value <= panel_right
        ])
        axis.axvline(0.0, color='#777777', linewidth=0.45, linestyle='--', zorder=1)
        add_panel_title(axis, lineage_title, stage, classes)
        if plot_row == 1:
            axis.set_xlabel('Permutation importance (%)', fontsize=6.2, fontweight='bold')
        if plot_column == 0:
            axis.set_ylabel('Subtype / marker', fontsize=7.8, fontweight='bold', labelpad=2)
        for spine in ('top', 'right'):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis='x', labelsize=5.2, pad=1.0)
        axis.tick_params(axis='y', pad=1.0)

    legend_axis = fig.add_subplot(grid[2, :])
    legend_axis.set_axis_off()
    legend_axis.set_xlim(0.0, 1.0)
    legend_axis.set_ylim(0.0, 1.0)
    legend_axis.set_xticks([])
    legend_axis.set_yticks([])
    legend_axis.text(0.29, 0.82, 'Attribution frequency', ha='center', va='center',
                     fontsize=7.2, fontweight='bold')
    for xpos, value in zip(np.linspace(0.19, 0.39, 4), (25, 50, 75, 100)):
        legend_axis.scatter(xpos, 0.45, s=1.60 * _step04b_reviewed_bubble_size(value),
                            facecolor='#A7A7A7', edgecolor='#4F4F4F', linewidth=0.35)
        legend_axis.text(xpos, 0.08, f'{value}%', ha='center', va='center', fontsize=5.8)
    legend_axis.text(0.72, 0.82, 'Direction consistency', ha='center', va='center',
                     fontsize=7.2, fontweight='bold')
    for xpos, (label, alpha) in zip(np.linspace(0.65, 0.79, 3), ((80, .45), (90, .70), (100, .95))):
        legend_axis.scatter(xpos, 0.45, s=1.60 * _step04b_reviewed_bubble_size(65),
                            facecolor=_step04b_reviewed_rgba('#777777', alpha),
                            edgecolor='#3F3F3F', linewidth=0.35)
        legend_axis.text(xpos, 0.08, f'{label}%', ha='center', va='center', fontsize=5.8)
    _step04b_assert_arial_text(fig)

    png_path = STEP04B_REVIEWED_ALE_FIGURE_DIR / (STEP04B_REVIEWED_ALE_BUBBLE_STEM + '.png')
    pptx_path = STEP04B_REVIEWED_ALE_FIGURE_DIR / (STEP04B_REVIEWED_ALE_BUBBLE_STEM + '.pptx')
    return _step04b_save_reviewed_pair(
        fig, png_path, pptx_path, dpi=600, show=show,
    )


def _step04b_reviewed_heatmap_payload():
    """Load current hash-linked selection/profile tables from one promoted generation."""
    payload = _step04b_reviewed_display_payload()
    return (
        payload['manifest'], payload['heatmap_selection'], payload['heatmap_profiles']
    )


def _step04b_reviewed_luminance(rgba):
    def linear(channel):
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
    red, green, blue = (linear(float(channel)) for channel in rgba[:3])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _render_step04b_reviewed_ale_heatmaps(show=False):
    """Render four reviewed Heatmaps from the promoted derived generation."""
    _step04b_require_arial()
    manifest, selected, profiles = _step04b_reviewed_heatmap_payload()
    class_map = {
        stage: classes for stage, _, classes in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS
    }
    lineage_stage = {
        lineage: stage for stage, lineage, _ in STEP04B_REVIEWED_ALE_STAGE_CONTRACTS
    }
    method_specs = (
        ('ML_Subtype_Raw', 'Proposed ML\n(Ours: XGB+MLP)'),
        ('Pred_SC_Centroid', 'SC-Centroid'),
        ('Pred_Clustering', 'Clustering\n(Leiden)'),
    )
    norm_contract = manifest.get('shared_color_norm', {})
    vlim = max(abs(float(norm_contract['vmin'])), abs(float(norm_contract['vmax'])))
    reports = []

    for lineage in ('T cells', 'B cells', 'Myeloid cells', 'Stromal cells'):
        stage = lineage_stage[lineage]
        stem = STEP04B_REVIEWED_ALE_HEATMAP_STEMS[lineage]
        if not _final_figure_stem_registered(stem):
            raise RuntimeError(f'Step04B canonical heatmap stem is not registered: {stem}')
        classes = class_map[stage]
        rows = selected.loc[selected['stage'].astype(str).eq(stage)].copy()
        markers = rows['feature'].astype(str).tolist()
        owners = dict(zip(markers, rows['class'].astype(str)))
        signs = dict(zip(markers, rows['direction_symbol'].astype(str)))
        boundaries, cumulative = [], 0
        for class_name in classes[:-1]:
            cumulative += int(rows['class'].astype(str).eq(class_name).sum())
            if cumulative:
                boundaries.append(cumulative)

        layout = STEP04B_REVIEWED_ALE_HEATMAP_LAYOUTS[lineage]
        canvas_inches = tuple(float(value) / 2.54 for value in layout['canvas_cm'])
        fig = plt.figure(figsize=canvas_inches, facecolor='white')
        # Center the complete visible composition (y label + marker labels +
        # three method panels) on the native canvas rather than centering only
        # the heatmap rectangles.
        grid = fig.add_gridspec(
            1, 3,
            left=float(layout.get('plot_left', 0.125)),
            right=float(layout.get('plot_right', 0.985)),
            top=float(layout.get('plot_top', 0.895)),
            bottom=float(layout['plot_bottom']),
            wspace=float(layout.get('plot_wspace', 0.050)),
        )
        cmap = matplotlib.colormaps['RdBu_r'].copy()
        cmap.set_bad('#D9D9D9')
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        mesh = None
        for panel_index, (method_column, method_title) in enumerate(method_specs):
            axis = fig.add_subplot(grid[panel_index])
            frame = (
                profiles.loc[
                    profiles['stage'].astype(str).eq(stage)
                    & profiles['method_column'].astype(str).eq(method_column)
                ]
                .pivot(index='feature', columns='predicted_subtype', values='mean_global_z')
                .reindex(index=markers, columns=classes)
            )
            values = frame.to_numpy(dtype=float)
            mesh = axis.pcolormesh(
                np.arange(len(classes) + 1), np.arange(len(markers) + 1),
                np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                edgecolors='white', linewidth=0.36, shading='flat',
            )
            for row_index in range(len(markers)):
                for column_index in range(len(classes)):
                    value = values[row_index, column_index]
                    if not np.isfinite(value):
                        continue
                    rgba = cmap(norm(float(value)))
                    text_color = '#262626' if _step04b_reviewed_luminance(rgba) > 0.408 else 'white'
                    text_value = '0.0' if abs(float(value)) < 0.05 else f'{float(value):.1f}'
                    axis.text(
                        column_index + 0.5, row_index + 0.5, text_value,
                        ha='center', va='center',
                        fontsize=float(layout['annotation_fontsize']),
                        fontweight='bold', color=text_color,
                    )
            for boundary in boundaries:
                axis.axhline(boundary, color='#555555', linewidth=0.68, alpha=0.78)
            axis.set_xlim(0, len(classes))
            axis.set_ylim(len(markers), 0)
            axis.set_xticks(np.arange(len(classes)) + 0.5)
            display_classes = [
                STEP04B_REVIEWED_ALE_CLASS_DISPLAY.get(class_name, class_name)
                for class_name in classes
            ]
            if layout.get('wrap_xtick_labels', False):
                display_classes = [
                    'Plasma\ncells' if label == 'Plasma cells' else label
                    for label in display_classes
                ]
            if layout.get('wrap_tcell_xtick_labels', False):
                display_classes = [
                    {'CD8 T': 'CD8T', 'CD4 CTL': 'CD4\nCTL'}.get(label, label)
                    for label in display_classes
                ]
            x_rotation = float(layout['x_rotation'])
            axis.set_xticklabels(
                display_classes,
                rotation=x_rotation,
                ha='right' if x_rotation else 'center',
                rotation_mode='anchor',
            )
            for tick, class_name in zip(axis.get_xticklabels(), classes):
                tick.set_color(STEP04B_REVIEWED_ALE_CLASS_COLORS[class_name])
                tick.set_fontweight('bold')
                tick.set_fontsize(float(
                    layout.get('xtick_fontsize_by_class', {}).get(
                        class_name, layout['xtick_fontsize']
                    )
                ))
            axis.tick_params(
                axis='x', pad=float(layout.get('x_tick_pad', 1.5)), length=0,
            )
            axis.set_yticks(np.arange(len(markers)) + 0.5)
            if panel_index == 0:
                axis.set_yticklabels([f'{marker} ({signs[marker]})' for marker in markers])
                for tick, marker in zip(axis.get_yticklabels(), markers):
                    tick.set_color(STEP04B_REVIEWED_ALE_CLASS_COLORS[owners[marker]])
                    tick.set_fontweight('bold')
                    tick.set_fontsize(float(layout['ytick_fontsize']))
                axis.set_ylabel('Signature markers', fontsize=float(layout.get('ylabel_fontsize', 7.3)),
                                fontweight='bold', labelpad=2)
            else:
                axis.set_yticklabels([])
            axis.tick_params(axis='y', length=0, pad=1.5)
            # Center one-line and two-line method headings on the same vertical
            # anchor so their blocks align rather than sharing only a baseline.
            axis.text(
                0.5, float(layout.get('method_title_y', 1.060)), method_title,
                transform=axis.transAxes, ha='center', va='center',
                fontsize=float(layout.get('method_title_fontsize', 7.6)),
                fontweight='bold', linespacing=0.90,
                clip_on=False,
            )
            for spine in axis.spines.values():
                spine.set_visible(False)
        if mesh is None:
            plt.close(fig)
            raise RuntimeError(f'No reviewed Heatmap mesh rendered for {lineage}')
        # Minimal lower-right shared colorbar: shorter, visibly thicker, and
        # limited to the endpoints plus the zero center.
        colorbar_axis = fig.add_axes(layout.get(
            'colorbar_rect', (0.765, 0.050, 0.220, 0.040)
        ))
        colorbar = fig.colorbar(mesh, cax=colorbar_axis, orientation='horizontal')
        colorbar.set_ticks([-vlim, 0.0, vlim])
        colorbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        colorbar.ax.text(
            -0.18, 0.5, 'Z-score', transform=colorbar.ax.transAxes,
            ha='right', va='center',
            fontsize=float(layout.get('colorbar_label_fontsize', 6.2)),
            fontweight='bold',
            clip_on=False,
        )
        colorbar.ax.tick_params(
            labelsize=float(layout.get('colorbar_tick_fontsize', 5.4)),
            length=2, pad=1.2,
        )
        colorbar.outline.set_linewidth(0.55)
        _step04b_assert_arial_text(fig)

        png_path = STEP04B_REVIEWED_ALE_FIGURE_DIR / (stem + '.png')
        pptx_path = STEP04B_REVIEWED_ALE_FIGURE_DIR / (stem + '.pptx')
        reports.append(_step04b_save_reviewed_pair(
            fig, png_path, pptx_path, show=show,
        ))
    return reports


if RUN_STEP04B_REVIEWED_ALE_RECALCULATION:
    raise RuntimeError(
        'Step04B ALE is a protected main calculation and cannot run from the Final '
        'canonical workflow. Keep the retained recalculation code for a separately '
        'reviewed, explicitly authorized execution path.'
    )
elif STEP04B_REVIEWED_ALE_FIGURE_ONLY:
    if not STEP04_FIGURE_ONLY or RUN_STEP04_TRAINING or RUN_CD4CTL_GMM_PREFLIGHT_ONLY:
        raise RuntimeError('Reviewed Step04B ALE Figures require the Figure-only mode')
    _step04b_reviewed_derived_manifest = _step04b_refresh_reviewed_display_derivatives()
    _step04b_reviewed_reports = [
        _render_step04b_reviewed_ale_bubble(show=False),
        *_render_step04b_reviewed_ale_heatmaps(show=False),
    ]
    _step04b_reviewed_saved_count = sum(
        1 for report in _step04b_reviewed_reports if report and all(report)
    )
    if _step04b_reviewed_saved_count != len(STEP04B_ACTIVE_REVIEWED_FIGURE_STEMS):
        raise RuntimeError(
            'Step04B active reviewed Figure family was not saved completely: '
            f'{_step04b_reviewed_saved_count}/'
            f'{len(STEP04B_ACTIVE_REVIEWED_FIGURE_STEMS)} pairs'
        )
    print(
        '✅ Step04B reviewed display-derived workflow: '
        f'generation={_step04b_reviewed_derived_manifest["derived_generation_id"]}, '
        f'saved_pairs={_step04b_reviewed_saved_count}'
    )
    if not CONTINUE_STEP04B_LEGACY_FIGURES:
        print('🔹 Step04B legacy Figure blocks were not run.')
        raise SystemExit(0)
    # The reviewed Bubble and Heatmaps replace these independently saved legacy
    # outputs.  Keep the rest of the explicitly requested legacy continuation,
    # but do not emit duplicate subtype-reliability/profile Figures.
    SAVE_STEP04B09_PURITY_COLOR_SYNCED = False
    SAVE_STEP04B09_MARKER_RELIABILITY_COMPOSITE = False


if not step04_loaded_from_completed:
    _mark_step04b_tpr_gmm_eligibility(adata)


# %%
# ==============================================================================
# [Step 04B-3] CD4_CTL GMM Preflight and pseudo-GT labeling
# ==============================================================================
if not step04_loaded_from_completed:
    from sklearn.mixture import GaussianMixture
    print('🔹 [Step 04B-3] GRZB GMM cutoff preflight and CD4_CTL pseudo-GT generation in Reference CD4 candidates...')

    if 'GRZB' not in adata.var_names:
        raise KeyError("The GRZB marker required for CD4_CTL GMM preflight is missing from adata.var_names.")

    if 'GT_Subtype_Before_STEP04B_CD4CTL_GMM' not in adata.obs.columns:
        adata.obs['GT_Subtype_Before_STEP04B_CD4CTL_GMM'] = adata.obs['GT_Subtype'].astype(str).values

    base_subtype_vals = adata.obs['GT_Subtype_Before_STEP04B_CD4CTL_GMM'].astype(str).values
    disease_is_dlbcl = adata.obs['Disease'].str.contains('DLBCL', na=False).values
    modeling_broad_vals = adata.obs['Modeling_Broad'].astype(str).values

    mask_ref_cd4 = (
        (modeling_broad_vals == 'T cells') &
        (~disease_is_dlbcl) &
        np.isin(base_subtype_vals, CD4CTL_CANDIDATE_SUBTYPES)
    )
    if STEP04B_TPR_GMM_MODE != 'keep' and STEP04B_TPR_GMM_ELIGIBLE_COL in adata.obs.columns:
        tpr_gmm_eligible = adata.obs[STEP04B_TPR_GMM_ELIGIBLE_COL].astype(bool).values
        mask_ref_cd4 &= (
            (base_subtype_vals != 'Tpr') |
            ((base_subtype_vals == 'Tpr') & tpr_gmm_eligible)
        )
    idx_ref_cd4 = np.where(mask_ref_cd4)[0]

    if len(idx_ref_cd4) < 30:
        raise ValueError(
            f"Too few CD4_CTL GMM candidate cells: {len(idx_ref_cd4):,} cells. "
            f"Candidate subtypes={CD4CTL_CANDIDATE_SUBTYPES}"
        )

    grzb_vals = _extract_marker_vector(adata, idx_ref_cd4, 'GRZB')
    finite_mask = np.isfinite(grzb_vals)
    idx_ref_cd4 = idx_ref_cd4[finite_mask]
    grzb_vals = grzb_vals[finite_mask]
    subtype_for_gmm = base_subtype_vals[idx_ref_cd4]

    print(f"  📌 GMM targets: {len(idx_ref_cd4):,} Reference CD4 candidates "
          f"({', '.join(CD4CTL_CANDIDATE_SUBTYPES)})")
    print(pd.Series(subtype_for_gmm).value_counts().to_string())

    model_rows = []
    fitted_gmms = {}
    for n_comp in CD4CTL_GMM_COMPONENTS_TO_COMPARE:
        gmm_tmp = GaussianMixture(
            n_components=n_comp,
            covariance_type='full',
            n_init=10,
            random_state=RANDOM_SEED
        )
        gmm_tmp.fit(grzb_vals.reshape(-1, 1))
        means_tmp = gmm_tmp.means_.flatten()
        order_tmp = np.argsort(means_tmp)
        fitted_gmms[n_comp] = gmm_tmp
        model_rows.append({
            'n_components': int(n_comp),
            'aic': float(gmm_tmp.aic(grzb_vals.reshape(-1, 1))),
            'bic': float(gmm_tmp.bic(grzb_vals.reshape(-1, 1))),
            'means_sorted': ','.join([f'{means_tmp[i]:.4f}' for i in order_tmp]),
            'weights_sorted': ','.join([f'{gmm_tmp.weights_[i]:.4f}' for i in order_tmp])
        })

    gmm_model_summary = pd.DataFrame(model_rows)
    gmm_model_summary_path = os.path.join(CKPT_DIR, 'mv1_step04B_cd4ctl_gmm_model_selection.csv')
    _step04b_write_table_if_enabled(gmm_model_summary, gmm_model_summary_path, index=False)

    gmm = fitted_gmms[CD4CTL_GMM_N_COMPONENTS]
    means = gmm.means_.flatten()
    high_comp = int(np.argmax(means))
    high_prob = gmm.predict_proba(grzb_vals.reshape(-1, 1))[:, high_comp]

    source_subtype_counts = pd.Series(subtype_for_gmm).value_counts()
    threshold_rows = []
    threshold_subtype_rows = []
    for thr in CD4CTL_GMM_POSTERIOR_THRESHOLDS:
        selected = high_prob >= thr
        selected_n = int(selected.sum())
        selected_subtype_counts = pd.Series(subtype_for_gmm[selected]).value_counts()
        row = {
            'posterior_threshold': float(thr),
            'is_selected_threshold': bool(np.isclose(thr, CD4CTL_GMM_POSTERIOR_THRESHOLD)),
            'candidate_n': selected_n,
            'candidate_pct': float(selected.mean() * 100.0),
            'effective_grzb_cutoff': _gmm_component_cutoff(gmm, high_comp, thr, grzb_vals),
            'mean_grzb_selected': float(np.nanmean(grzb_vals[selected])) if selected.any() else np.nan,
            'median_grzb_selected': float(np.nanmedian(grzb_vals[selected])) if selected.any() else np.nan,
            'mean_posterior_selected': float(np.nanmean(high_prob[selected])) if selected.any() else np.nan,
            'median_posterior_selected': float(np.nanmedian(high_prob[selected])) if selected.any() else np.nan
        }
        for subtype in CD4CTL_CANDIDATE_SUBTYPES:
            source_n = int(source_subtype_counts.get(subtype, 0))
            subtype_selected_n = int(selected_subtype_counts.get(subtype, 0))
            subtype_selected_pct = (
                float(subtype_selected_n / source_n * 100.0)
                if source_n > 0 else np.nan
            )
            subtype_candidate_pct = (
                float(subtype_selected_n / selected_n * 100.0)
                if selected_n > 0 else np.nan
            )
            row[f'{subtype}_selected_n'] = subtype_selected_n
            row[f'{subtype}_selected_pct_within_subtype'] = subtype_selected_pct
            row[f'{subtype}_pct_of_candidates'] = subtype_candidate_pct
            threshold_subtype_rows.append({
                'posterior_threshold': float(thr),
                'Original_Subtype': subtype,
                'source_n': source_n,
                'selected_n': subtype_selected_n,
                'selected_pct_within_subtype': subtype_selected_pct,
                'pct_of_candidates': subtype_candidate_pct
            })
        threshold_rows.append(row)
    threshold_summary = pd.DataFrame(threshold_rows)
    if not threshold_summary.empty:
        baseline_n = float(threshold_summary['candidate_n'].iloc[0])
        prev_counts = threshold_summary['candidate_n'].shift(1)
        threshold_summary['retained_vs_lowest_threshold_pct'] = (
            threshold_summary['candidate_n'] / baseline_n * 100.0
            if baseline_n > 0 else np.nan
        )
        threshold_summary['candidate_n_drop_from_previous'] = (
            prev_counts - threshold_summary['candidate_n']
        ).fillna(0).astype(int)
        threshold_summary['candidate_n_drop_pct_from_previous'] = np.where(
            prev_counts > 0,
            (prev_counts - threshold_summary['candidate_n']) / prev_counts * 100.0,
            0.0
        )
    threshold_subtype_summary = pd.DataFrame(threshold_subtype_rows)
    threshold_summary_path = os.path.join(CKPT_DIR, 'mv1_step04B_cd4ctl_gmm_threshold_summary.csv')
    _step04b_write_table_if_enabled(threshold_summary, threshold_summary_path, index=False)
    threshold_subtype_path = os.path.join(CKPT_DIR, 'mv1_step04B_cd4ctl_gmm_threshold_by_subtype.csv')
    _step04b_write_table_if_enabled(threshold_subtype_summary, threshold_subtype_path, index=False)

    selected_cd4ctl = high_prob >= CD4CTL_GMM_POSTERIOR_THRESHOLD
    selected_cutoff = _gmm_component_cutoff(gmm, high_comp, CD4CTL_GMM_POSTERIOR_THRESHOLD, grzb_vals)
    idx_cd4ctl = idx_ref_cd4[selected_cd4ctl]

    marker_panel = [m for m in ['CD3', 'CD4', 'CD8', 'GRZB', 'FOXP3', 'CXCR5', 'PD1', 'Ki67'] if m in adata.var_names]
    marker_mat = None
    profile_rows = []
    if marker_panel:
        marker_mat = _extract_marker_matrix(adata, idx_ref_cd4, marker_panel)
        profile_df = pd.DataFrame(marker_mat, columns=marker_panel)
        profile_df['GMM_Group'] = np.where(selected_cd4ctl, 'CD4_CTL_candidate', 'CD4_reference_non_CTL')
        profile_df['Original_Subtype'] = subtype_for_gmm
        for group_name, group_df in profile_df.groupby('GMM_Group'):
            row = {'group': group_name, 'n': int(len(group_df))}
            for marker in marker_panel:
                row[f'{marker}_mean'] = float(group_df[marker].mean())
                row[f'{marker}_median'] = float(group_df[marker].median())
            profile_rows.append(row)
    marker_profile_summary = pd.DataFrame(profile_rows)
    marker_profile_path = os.path.join(CKPT_DIR, 'mv1_step04B_cd4ctl_gmm_marker_profile.csv')
    _step04b_write_table_if_enabled(marker_profile_summary, marker_profile_path, index=False)

    preflight_df = pd.DataFrame({
        'GRZB': grzb_vals,
        'Original_Subtype': subtype_for_gmm,
        'High_Component_Prob': high_prob,
        'CD4_CTL_candidate': selected_cd4ctl
    })
    preflight_df['GMM_Label'] = np.where(
        preflight_df['CD4_CTL_candidate'],
        'CD4_CTL candidate',
        'CD4 reference non-CTL'
    )
    fig_path_cd4ctl_gmm = os.path.join(FIG_DIR, 'Final_mv1_Step04B-03_CD4CTL_GMM_Summary.png')

    subtype_order_preflight = [s for s in CD4CTL_CANDIDATE_SUBTYPES if s in set(subtype_for_gmm)]
    subtype_palette_preflight = {
        s: CELL_COLORS.get(s, '#808080')
        for s in subtype_order_preflight
    }
    preflight_df['Comparison_Group'] = np.where(
        preflight_df['CD4_CTL_candidate'],
        'CD4_CTL_candidate',
        preflight_df['Original_Subtype']
    )
    comparison_group_order = []
    if selected_cd4ctl.any():
        comparison_group_order.append('CD4_CTL_candidate')
    comparison_group_order.extend([
        s for s in subtype_order_preflight
        if ((preflight_df['Original_Subtype'] == s) & (~preflight_df['CD4_CTL_candidate'])).any()
    ])
    comparison_label_map = {}
    for group_name in comparison_group_order:
        if group_name == 'CD4_CTL_candidate':
            n_group = int((preflight_df['Comparison_Group'] == group_name).sum())
            comparison_label_map[group_name] = f'CD4_CTL\ncandidate\n(n={n_group:,})'
        else:
            n_group = int((preflight_df['Comparison_Group'] == group_name).sum())
            comparison_label_map[group_name] = f'{group_name}\nnon-CTL\n(n={n_group:,})'
    preflight_df['Comparison_Label'] = preflight_df['Comparison_Group'].map(comparison_label_map)
    comparison_order_labels = [comparison_label_map[g] for g in comparison_group_order]
    comparison_palette = {
        comparison_label_map[g]: (
            CELL_COLORS.get('CD4_CTL', '#c0392b')
            if g == 'CD4_CTL_candidate'
            else CELL_COLORS.get(g, '#808080')
        )
        for g in comparison_group_order
    }

    marker_heatmap_path = os.path.join(CKPT_DIR, 'mv1_step04B_cd4ctl_gmm_tcell_marker_heatmap.csv')
    marker_heatmap_rows = []
    if marker_panel and marker_mat is not None:
        def _append_marker_heatmap_row(group_id, group_label, group_values, is_cd4ctl_candidate=False):
            if group_values is None or len(group_values) == 0:
                return
            group_values = np.asarray(group_values, dtype=float)
            row = {
                'group_id': group_id,
                'Tcell_Group': group_label,
                'n': int(group_values.shape[0]),
                'is_cd4ctl_candidate': bool(is_cd4ctl_candidate)
            }
            for marker_idx, marker in enumerate(marker_panel):
                row[f'{marker}_median'] = float(np.nanmedian(group_values[:, marker_idx]))
                row[f'{marker}_mean'] = float(np.nanmean(group_values[:, marker_idx]))
            marker_heatmap_rows.append(row)

        _append_marker_heatmap_row(
            'CD4_CTL_candidate',
            'CD4_CTL candidate',
            marker_mat[selected_cd4ctl],
            is_cd4ctl_candidate=True
        )
        for subtype in subtype_order_preflight:
            residual_mask = (subtype_for_gmm == subtype) & (~selected_cd4ctl)
            _append_marker_heatmap_row(
                f'{subtype}_non_CTL',
                f'{subtype} non-CTL',
                marker_mat[residual_mask]
            )

        # CD8T is not used to fit the GMM cutoff; include it only in the heatmap as a T-cell reference contrasting with CD4+GRZB+ candidates.
        mask_ref_cd8t = (
            (modeling_broad_vals == 'T cells') &
            (~disease_is_dlbcl) &
            (base_subtype_vals == 'CD8T')
        )
        idx_ref_cd8t = np.where(mask_ref_cd8t)[0]
        if len(idx_ref_cd8t) > 0:
            _append_marker_heatmap_row(
                'CD8T_reference',
                'CD8T reference',
                _extract_marker_matrix(adata, idx_ref_cd8t, marker_panel)
            )

    marker_heatmap_summary = pd.DataFrame(marker_heatmap_rows)
    _step04b_write_table_if_enabled(marker_heatmap_summary, marker_heatmap_path, index=False)

    from scipy.stats import mannwhitneyu
    candidate_grzb = preflight_df.loc[preflight_df['CD4_CTL_candidate'], 'GRZB']
    grzb_comparison_rows = []
    for subtype in subtype_order_preflight:
        residual_mask = (
            (preflight_df['Original_Subtype'] == subtype) &
            (~preflight_df['CD4_CTL_candidate'])
        )
        residual_grzb = preflight_df.loc[residual_mask, 'GRZB']
        selected_from_subtype = (
            (preflight_df['Original_Subtype'] == subtype) &
            preflight_df['CD4_CTL_candidate']
        )
        if len(candidate_grzb) > 0 and len(residual_grzb) > 0:
            _, p_greater = mannwhitneyu(candidate_grzb, residual_grzb, alternative='greater')
            _, p_two_sided = mannwhitneyu(candidate_grzb, residual_grzb, alternative='two-sided')
        else:
            p_greater = np.nan
            p_two_sided = np.nan
        grzb_comparison_rows.append({
            'comparison': f'CD4_CTL_candidate_vs_{subtype}_non_CTL',
            'candidate_n_total': int(len(candidate_grzb)),
            'candidate_n_from_subtype': int(selected_from_subtype.sum()),
            'residual_n': int(len(residual_grzb)),
            'candidate_grzb_mean': float(candidate_grzb.mean()) if len(candidate_grzb) > 0 else np.nan,
            'candidate_grzb_median': float(candidate_grzb.median()) if len(candidate_grzb) > 0 else np.nan,
            'residual_grzb_mean': float(residual_grzb.mean()) if len(residual_grzb) > 0 else np.nan,
            'residual_grzb_median': float(residual_grzb.median()) if len(residual_grzb) > 0 else np.nan,
            'median_delta_candidate_minus_residual': (
                float(candidate_grzb.median() - residual_grzb.median())
                if len(candidate_grzb) > 0 and len(residual_grzb) > 0 else np.nan
            ),
            'mannwhitneyu_p_greater': float(p_greater) if np.isfinite(p_greater) else np.nan,
            'mannwhitneyu_p_two_sided': float(p_two_sided) if np.isfinite(p_two_sided) else np.nan
        })
    grzb_comparison_stats = pd.DataFrame(grzb_comparison_rows)
    grzb_comparison_path = os.path.join(CKPT_DIR, 'mv1_step04B_cd4ctl_gmm_grzb_candidate_vs_subtype.csv')
    _step04b_write_table_if_enabled(grzb_comparison_stats, grzb_comparison_path, index=False)

    selected_hi = (
        float(np.nanpercentile(grzb_vals[selected_cd4ctl], 95))
        if selected_cd4ctl.any()
        else float(np.nanpercentile(grzb_vals, 99.0))
    )
    display_min = min(-1.0, float(np.nanpercentile(grzb_vals, 0.5)))
    display_max = max(
        float(np.nanpercentile(grzb_vals, 99.5)),
        selected_hi,
        float(selected_cutoff) + 0.5 if np.isfinite(selected_cutoff) else selected_hi
    )

    stack_grzb_vals, stack_subtype_vals, stack_scope_label = _get_cd4ctl_cumulative_stack_inputs(
        adata,
        grzb_vals,
        selected_cd4ctl,
        subtype_for_gmm
    )
    cd4ctl_gmm_panel_paths = _save_cd4ctl_gmm_individual_panels(
        preflight_df,
        subtype_order_preflight,
        subtype_palette_preflight,
        selected_cutoff,
        selected_cd4ctl,
        display_min,
        display_max,
        marker_heatmap_summary,
        marker_panel,
        stack_grzb_vals,
        stack_subtype_vals,
        stack_scope_label,
        means_text=means,
        gmm=gmm,
        high_comp=high_comp,
        show=True
    )

    # Manuscript summary figure combining 03A/03B.
    # Remove the weak stacked-histogram/source-composition panel,
    # retaining only key panels explaining the cutoff definition and biological marker profiles.
    fig = plt.figure(figsize=(18.5, 13.2))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.2, 1.65],
        height_ratios=[1.25, 1.40],
        wspace=0.24,
        hspace=0.50
    )
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_post = fig.add_subplot(gs[0, 1])
    ax_trade = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    sns.boxplot(
        data=preflight_df,
        x='Original_Subtype',
        y='GRZB',
        order=subtype_order_preflight,
        palette=subtype_palette_preflight,
        showfliers=False,
        width=0.58,
        linewidth=1.8,
        ax=ax_dist,
        boxprops=dict(edgecolor='black', alpha=0.88),
        zorder=3
    )
    _plot_cd4ctl_gmm_point_layers(
        ax_dist,
        preflight_df,
        subtype_order_preflight,
        background_size=1.35,
        background_alpha=0.10,
        candidate_size=2.5,
        candidate_alpha=0.52,
    )
    ax_dist.axhline(selected_cutoff, color='#c0392b', ls='--', lw=2.3, zorder=5)
    ax_dist.set_ylim(display_min, display_max)
    ax_dist.set_title('Reference CD4-lineage GRZB distribution', fontsize=14, fontweight='bold')
    ax_dist.set_xlabel('Original CD4-lineage subtype')
    ax_dist.set_ylabel('GRZB expression')
    ax_dist.tick_params(axis='x', rotation=0)
    ax_dist.text(
        0.02, 0.96,
        f'{_cd4ctl_gmm_cutoff_display(selected_cutoff)}\nCD4_CTL candidates = {int(selected_cd4ctl.sum()):,} ({selected_cd4ctl.mean()*100:.2f}%)',
        transform=ax_dist.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='lightgray', alpha=0.92)
    )

    grid = np.linspace(float(np.nanmin(grzb_vals)), float(np.nanmax(grzb_vals)), 2000)
    high_prob_grid = gmm.predict_proba(grid.reshape(-1, 1))[:, high_comp]
    ax_post.plot(grid, high_prob_grid, color='#c0392b', lw=2.5, label='P(high GRZB component)')
    ax_post.axhline(CD4CTL_GMM_POSTERIOR_THRESHOLD, color='#2c3e50', ls=':', lw=2.0,
                    label=f'Posterior threshold = {CD4CTL_GMM_POSTERIOR_THRESHOLD}')
    ax_post.axvline(selected_cutoff, color='#c0392b', ls='--', lw=2.0,
                    label=_cd4ctl_gmm_cutoff_display(selected_cutoff))
    ax_post.set_xlim(display_min, display_max)
    ax_post.set_ylim(-0.03, 1.03)
    ax_post.set_title('GMM high-component posterior', fontsize=14, fontweight='bold')
    ax_post.set_xlabel('GRZB expression')
    ax_post.set_ylabel('Posterior probability')
    means_text = ', '.join([f'{x:.2f}' for x in sorted(means)])
    ax_post.text(
        0.04, 0.96,
        f'{CD4CTL_GMM_N_COMPONENTS}-component GMM\nmeans: {means_text}',
        transform=ax_post.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='lightgray', alpha=0.92)
    )
    ax_post.legend(loc='lower right', fontsize=9, frameon=True)

    median_marker_cols = [f'{m}_median' for m in marker_panel]
    if (not marker_heatmap_summary.empty) and median_marker_cols:
        heat_df = marker_heatmap_summary.set_index('Tcell_Group')[median_marker_cols].copy()
        heat_df.columns = marker_panel
        heat_df.index = [
            f"{row['Tcell_Group']}\n(n={int(row['n']):,})"
            for _, row in marker_heatmap_summary.iterrows()
        ]
        sns.heatmap(
            heat_df,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Median expression', 'shrink': 0.82},
            ax=ax_heat,
            annot_kws={'fontsize': 8}
        )
        ax_heat.set_title('Marker profile after selected cutoff', fontsize=14, fontweight='bold')
        ax_heat.set_xlabel('Marker')
        ax_heat.set_ylabel('T-cell subtype / group')
        ax_heat.tick_params(axis='x', rotation=45, labelsize=8)
        ax_heat.tick_params(axis='y', rotation=0, labelsize=7)
    else:
        ax_heat.text(0.5, 0.5, 'No marker heatmap available', ha='center', va='center')
        ax_heat.set_axis_off()

    stack_grzb_vals, stack_subtype_vals, stack_scope_label = _get_cd4ctl_cumulative_stack_inputs(
        adata,
        grzb_vals,
        selected_cd4ctl,
        subtype_for_gmm
    )
    _plot_cd4ctl_cumulative_stack_panel(
        ax_trade,
        stack_grzb_vals,
        stack_subtype_vals,
        selected_cutoff,
        selected_threshold=CD4CTL_GMM_POSTERIOR_THRESHOLD,
        scope_label=stack_scope_label,
        title='Cumulative composition above each GRZB cutoff'
    )

    for ax in [ax_dist, ax_post, ax_trade]:
        ax.grid(axis='y', ls='--', alpha=0.25)

    fig.suptitle(
        'CD4_CTL GMM summary: GRZB-high CD4-lineage pseudo-label',
        fontsize=17,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    if SAVE_STEP04B03_CD4CTL_GMM_COMPOSITE:
        _save_step04b_canvas_figure(
            fig,
            fig_path_cd4ctl_gmm,
            label='Step04B-03 CD4_CTL GMM composite',
            show=True,
            close=True
        )
    else:
        plt.close(fig)

    adata.obs['CD4CTL_GMM_Prob'] = np.nan
    adata.obs['CD4CTL_GMM_Candidate'] = False
    adata.obs['CD4CTL_GMM_SourceSubtype'] = 'na'
    adata.obs.loc[adata.obs.index[idx_ref_cd4], 'CD4CTL_GMM_Prob'] = high_prob
    adata.obs.loc[adata.obs.index[idx_ref_cd4], 'CD4CTL_GMM_Candidate'] = selected_cd4ctl
    adata.obs.loc[adata.obs.index[idx_ref_cd4], 'CD4CTL_GMM_SourceSubtype'] = subtype_for_gmm

    new_subtype_vals = base_subtype_vals.copy().astype(object)
    new_subtype_vals[idx_cd4ctl] = 'CD4_CTL'
    adata.obs['GT_Subtype'] = pd.Series(new_subtype_vals, index=adata.obs.index, dtype='object')
    adata.obs['Modeling_Broad'] = adata.obs['GT_Subtype'].apply(refine_broad_for_subtyping)

    adata.uns['step04B_cd4ctl_gmm_preflight'] = {
        'candidate_subtypes': ','.join(CD4CTL_CANDIDATE_SUBTYPES),
        'n_candidates': int(len(idx_ref_cd4)),
        'n_components_selected': int(CD4CTL_GMM_N_COMPONENTS),
        'posterior_threshold_selected': float(CD4CTL_GMM_POSTERIOR_THRESHOLD),
        'high_component': int(high_comp),
        'component_means': ','.join([f'{x:.6f}' for x in means]),
        'component_weights': ','.join([f'{x:.6f}' for x in gmm.weights_]),
        'selected_threshold_cutoff': float(selected_cutoff) if np.isfinite(selected_cutoff) else np.nan,
        'n_cd4ctl_pseudolabel': int(len(idx_cd4ctl)),
        'tpr_gmm_policy_mode': STEP04B_TPR_GMM_MODE,
        'tpr_gmm_policy_cd4_minus_cd8_min': float(STEP04B_TPR_GMM_CD4_MINUS_CD8_MIN),
        'tpr_gmm_policy_n_total': int(adata.uns.get('step04B_tpr_gmm_policy', {}).get('n_tpr_total', 0)),
        'tpr_gmm_policy_n_eligible': int(adata.uns.get('step04B_tpr_gmm_policy', {}).get('n_tpr_gmm_eligible', 0)),
        'tpr_gmm_policy_n_excluded': int(adata.uns.get('step04B_tpr_gmm_policy', {}).get('n_tpr_gmm_excluded', 0)),
        'model_selection_csv': gmm_model_summary_path,
        'threshold_summary_csv': threshold_summary_path,
        'threshold_by_subtype_csv': threshold_subtype_path,
        'marker_profile_csv': marker_profile_path,
        'tcell_marker_heatmap_csv': marker_heatmap_path,
        'grzb_candidate_vs_subtype_csv': grzb_comparison_path,
        'summary_figure': fig_path_cd4ctl_gmm if SAVE_STEP04B03_CD4CTL_GMM_COMPOSITE else '',
        'individual_figures': {
            panel: paths[0]
            for panel, paths in cd4ctl_gmm_panel_paths.items()
        }
    }

    print("\n  ✅ CD4_CTL GMM preflight saved")
    print(f"    - Model comparison: {gmm_model_summary_path}")
    print(f"    - Threshold summary: {threshold_summary_path}")
    print(f"    - Source subtype summary by threshold: {threshold_subtype_path}")
    print(f"    - Marker profiles: {marker_profile_path}")
    print(f"    - T-cell marker heatmap summary: {marker_heatmap_path}")
    print(f"    - GRZB comparison after applying cutoff: {grzb_comparison_path}")
    print("    - Individual diagnostic Figures:")
    for _panel, _paths in cd4ctl_gmm_panel_paths.items():
        print(f"      · {_panel}: {_paths[0]}")
    if SAVE_STEP04B03_CD4CTL_GMM_COMPOSITE:
        print(f"    - 2x2 composite Figure: {fig_path_cd4ctl_gmm}")
    print(f"  📊 Selection criteria: {CD4CTL_GMM_N_COMPONENTS}-component GMM, "
          f"P(high) ≥ {CD4CTL_GMM_POSTERIOR_THRESHOLD}, "
          f"GRZB Z ≥ {_format_cd4ctl_gmm_cutoff(selected_cutoff)}")
    print(f"  ✅ CD4_CTL pseudo-GT labeling completed: {len(idx_cd4ctl):,}/{len(idx_ref_cd4):,} cells "
          f"({len(idx_cd4ctl) / max(len(idx_ref_cd4), 1) * 100:.2f}%)")
    if EXCLUDE_CD4CTL_FROM_MAIN_TCELL_MODEL:
        print("  ℹ️ Current setting: CD4_CTL is excluded from main T-cell subtype training.")
    else:
        print("  ℹ️ Current setting: include CD4_CTL pseudo-GT as a main T-cell subtype training class.")

    if RUN_CD4CTL_GMM_PREFLIGHT_ONLY and not RUN_STEP04_TRAINING:
        raise SystemExit(
            "Normal stop: CD4_CTL GMM preflight-only completed. Review the CSV/Figures; "
            "to train, set RUN_CD4CTL_GMM_PREFLIGHT_ONLY=False and RUN_STEP04_TRAINING=True."
        )

elif STEP04_FIGURE_ONLY and not STEP04B_BENCHMARK_ONLY:
    print('🔹 [Step 04B-3] Figure-only mode → regenerating the single CD4_CTL GMM summary figure from saved Step04B checkpoint/CSV files.')
    fig_path_cd4ctl_gmm = os.path.join(FIG_DIR, 'Final_mv1_Step04B-03_CD4CTL_GMM_Summary.png')
    required_cols = ['CD4CTL_GMM_Prob', 'CD4CTL_GMM_Candidate', 'CD4CTL_GMM_SourceSubtype']
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise RuntimeError(f'Required columns for Step04B-03 figure-only regeneration are missing: {missing_cols}')
    if 'GRZB' not in adata.var_names:
        raise KeyError('The GRZB marker required for Step04B-03 figure-only regeneration is missing.')

    prob_vals = pd.to_numeric(adata.obs['CD4CTL_GMM_Prob'], errors='coerce').to_numpy(dtype=float)
    source_vals = adata.obs['CD4CTL_GMM_SourceSubtype'].astype(str).to_numpy()
    cand_vals = adata.obs['CD4CTL_GMM_Candidate'].astype(bool).to_numpy()
    valid_mask = np.isfinite(prob_vals) & (source_vals != 'na')
    idx_ref_cd4 = np.where(valid_mask)[0]
    if len(idx_ref_cd4) < 30:
        raise RuntimeError(f'Too few CD4 candidates for Step04B-03 figure-only regeneration: {len(idx_ref_cd4):,} cells')

    grzb_vals = _extract_marker_vector(adata, idx_ref_cd4, 'GRZB')
    finite_mask = np.isfinite(grzb_vals)
    idx_ref_cd4 = idx_ref_cd4[finite_mask]
    grzb_vals = grzb_vals[finite_mask]
    high_prob = prob_vals[idx_ref_cd4]
    selected_cd4ctl = cand_vals[idx_ref_cd4]
    subtype_for_gmm = source_vals[idx_ref_cd4]

    preflight_df = pd.DataFrame({
        'GRZB': grzb_vals,
        'Original_Subtype': subtype_for_gmm,
        'High_Component_Prob': high_prob,
        'CD4_CTL_candidate': selected_cd4ctl
    })
    subtype_order_preflight = [s for s in CD4CTL_CANDIDATE_SUBTYPES if s in set(subtype_for_gmm)]
    subtype_palette_preflight = {s: CELL_COLORS.get(s, '#808080') for s in subtype_order_preflight}

    threshold_summary_path = os.path.join(CKPT_DIR, 'mv1_step04B_cd4ctl_gmm_threshold_summary.csv')
    marker_heatmap_path = os.path.join(CKPT_DIR, 'mv1_step04B_cd4ctl_gmm_tcell_marker_heatmap.csv')
    if not os.path.exists(threshold_summary_path):
        raise FileNotFoundError(f'Step04B-03 threshold summary CSV is missing: {threshold_summary_path}')
    if not os.path.exists(marker_heatmap_path):
        raise FileNotFoundError(f'Step04B-03 marker heatmap CSV is missing: {marker_heatmap_path}')
    threshold_summary = pd.read_csv(threshold_summary_path)
    marker_heatmap_summary = pd.read_csv(marker_heatmap_path)

    selected_info = adata.uns.get('step04B_cd4ctl_gmm_preflight', {})
    selected_cutoff = float(selected_info.get('selected_threshold_cutoff', np.nan))
    if not np.isfinite(selected_cutoff):
        selected_row = threshold_summary.loc[threshold_summary['is_selected_threshold'].astype(bool)]
        if selected_row.empty:
            selected_row = threshold_summary.iloc[[0]]
        selected_cutoff = float(selected_row['effective_grzb_cutoff'].iloc[0])

    selected_hi = (
        float(np.nanpercentile(grzb_vals[selected_cd4ctl], 95))
        if selected_cd4ctl.any()
        else float(np.nanpercentile(grzb_vals, 99.0))
    )
    display_min = min(-1.0, float(np.nanpercentile(grzb_vals, 0.5)))
    display_max = max(
        float(np.nanpercentile(grzb_vals, 99.5)),
        selected_hi,
        float(selected_cutoff) + 0.5 if np.isfinite(selected_cutoff) else selected_hi
    )

    marker_panel = []
    for col in marker_heatmap_summary.columns:
        if col.endswith('_median'):
            marker_panel.append(col.replace('_median', ''))
    marker_panel = [
        m for m in ['CD3', 'CD4', 'CD8', 'GRZB', 'FOXP3', 'CXCR5', 'PD1', 'Ki67']
        if m in marker_panel and m in adata.var_names
    ]
    stack_grzb_vals, stack_subtype_vals, stack_scope_label = _get_cd4ctl_cumulative_stack_inputs(
        adata,
        grzb_vals,
        selected_cd4ctl,
        subtype_for_gmm
    )
    cd4ctl_gmm_panel_paths = _save_cd4ctl_gmm_individual_panels(
        preflight_df,
        subtype_order_preflight,
        subtype_palette_preflight,
        selected_cutoff,
        selected_cd4ctl,
        display_min,
        display_max,
        marker_heatmap_summary,
        marker_panel,
        stack_grzb_vals,
        stack_subtype_vals,
        stack_scope_label,
        means_text=selected_info.get('component_means', ''),
        show=True
    )

    fig = plt.figure(figsize=(18.5, 13.2))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.2, 1.65],
        height_ratios=[1.25, 1.40],
        wspace=0.24,
        hspace=0.50
    )
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_post = fig.add_subplot(gs[0, 1])
    ax_trade = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    sns.boxplot(
        data=preflight_df,
        x='Original_Subtype',
        y='GRZB',
        order=subtype_order_preflight,
        palette=subtype_palette_preflight,
        showfliers=False,
        width=0.58,
        linewidth=1.8,
        ax=ax_dist,
        boxprops=dict(edgecolor='black', alpha=0.88),
        zorder=3
    )
    _plot_cd4ctl_gmm_point_layers(
        ax_dist,
        preflight_df,
        subtype_order_preflight,
        background_size=1.35,
        background_alpha=0.10,
        candidate_size=2.5,
        candidate_alpha=0.52,
    )
    ax_dist.axhline(selected_cutoff, color='#c0392b', ls='--', lw=2.3, zorder=5)
    ax_dist.set_ylim(display_min, display_max)
    ax_dist.set_title('Reference CD4-lineage GRZB distribution', fontsize=14, fontweight='bold')
    ax_dist.set_xlabel('Original CD4-lineage subtype')
    ax_dist.set_ylabel('GRZB expression')
    ax_dist.tick_params(axis='x', rotation=0)
    ax_dist.text(
        0.02, 0.96,
        f'{_cd4ctl_gmm_cutoff_display(selected_cutoff)}\nCD4_CTL candidates = {int(selected_cd4ctl.sum()):,} ({selected_cd4ctl.mean()*100:.2f}%)',
        transform=ax_dist.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='lightgray', alpha=0.92)
    )

    curve_df = preflight_df[['GRZB', 'High_Component_Prob']].copy()
    curve_df = curve_df[(curve_df['GRZB'] >= display_min) & (curve_df['GRZB'] <= display_max)]
    bins = np.linspace(display_min, display_max, 180)
    curve_df['GRZB_bin'] = pd.cut(curve_df['GRZB'], bins=bins, include_lowest=True)
    curve_summary = (
        curve_df.groupby('GRZB_bin', observed=True)
        .agg(GRZB=('GRZB', 'mean'), High_Component_Prob=('High_Component_Prob', 'median'))
        .dropna()
    )
    ax_post.plot(curve_summary['GRZB'], curve_summary['High_Component_Prob'], color='#c0392b', lw=2.5, label='Median saved posterior')
    ax_post.axhline(CD4CTL_GMM_POSTERIOR_THRESHOLD, color='#2c3e50', ls=':', lw=2.0,
                    label=f'Posterior threshold = {CD4CTL_GMM_POSTERIOR_THRESHOLD}')
    ax_post.axvline(selected_cutoff, color='#c0392b', ls='--', lw=2.0,
                    label=_cd4ctl_gmm_cutoff_display(selected_cutoff))
    ax_post.set_xlim(display_min, display_max)
    ax_post.set_ylim(-0.03, 1.03)
    ax_post.set_title('GMM high-component posterior', fontsize=14, fontweight='bold')
    ax_post.set_xlabel('GRZB expression')
    ax_post.set_ylabel('Posterior probability')
    means_text = selected_info.get('component_means', '')
    if means_text:
        try:
            means_text = ', '.join([f'{x:.2f}' for x in sorted(float(x) for x in str(means_text).split(','))])
        except Exception:
            means_text = str(means_text)
    ax_post.text(
        0.04, 0.96,
        f'{CD4CTL_GMM_N_COMPONENTS}-component GMM\nmeans: {means_text}',
        transform=ax_post.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='lightgray', alpha=0.92)
    )
    ax_post.legend(loc='lower right', fontsize=9, frameon=True)

    median_marker_cols = [f'{m}_median' for m in marker_panel]
    if (not marker_heatmap_summary.empty) and median_marker_cols:
        heat_df = marker_heatmap_summary.set_index('Tcell_Group')[median_marker_cols].copy()
        heat_df.columns = marker_panel
        heat_df.index = [
            f"{row['Tcell_Group']}\n(n={int(row['n']):,})"
            for _, row in marker_heatmap_summary.iterrows()
        ]
        sns.heatmap(
            heat_df,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Median expression', 'shrink': 0.82},
            ax=ax_heat,
            annot_kws={'fontsize': 8}
        )
        ax_heat.set_title('Marker profile after selected cutoff', fontsize=14, fontweight='bold')
        ax_heat.set_xlabel('Marker')
        ax_heat.set_ylabel('T-cell subtype / group')
        ax_heat.tick_params(axis='x', rotation=45, labelsize=8)
        ax_heat.tick_params(axis='y', rotation=0, labelsize=7)
    else:
        ax_heat.text(0.5, 0.5, 'No marker heatmap available', ha='center', va='center')
        ax_heat.set_axis_off()

    stack_grzb_vals, stack_subtype_vals, stack_scope_label = _get_cd4ctl_cumulative_stack_inputs(
        adata,
        grzb_vals,
        selected_cd4ctl,
        subtype_for_gmm
    )
    _plot_cd4ctl_cumulative_stack_panel(
        ax_trade,
        stack_grzb_vals,
        stack_subtype_vals,
        selected_cutoff,
        selected_threshold=CD4CTL_GMM_POSTERIOR_THRESHOLD,
        scope_label=stack_scope_label,
        title='Cumulative composition above each GRZB cutoff'
    )

    for ax in [ax_dist, ax_post, ax_trade]:
        ax.grid(axis='y', ls='--', alpha=0.25)

    fig.suptitle(
        'CD4_CTL GMM summary: GRZB-high CD4-lineage pseudo-label',
        fontsize=17,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    if SAVE_STEP04B03_CD4CTL_GMM_COMPOSITE:
        _save_step04b_canvas_figure(
            fig,
            fig_path_cd4ctl_gmm,
            label='Step04B-03 CD4_CTL GMM composite',
            show=True,
            close=True
        )
        print(f'  🖼️ [Step 04B-03] 2x2 CD4_CTL GMM composite figure saved: {fig_path_cd4ctl_gmm}')
    else:
        plt.close(fig)
    print('  🖼️ [Step 04B-03] CD4_CTL GMM individual panel figures saved:')
    for _panel, _paths in cd4ctl_gmm_panel_paths.items():
        print(f'    - {_panel}: {_paths[0]}')

def append_per_subtype_metric_rows(rows, lineage, iter_idx, eval_type, name, y_true, y_pred,
                                   class_names, rank=np.nan, oob_n=None, excluded_subtypes=None):
    """Record one-vs-rest performance per subtype in the same format as Step02 per-class logs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    excluded_subtypes = excluded_subtypes or []

    for cls_idx, cls_name in enumerate(class_names):
        true_bin = (y_true == cls_idx)
        pred_bin = (y_pred == cls_idx)
        tn, fp, fn, tp = confusion_matrix(true_bin, pred_bin, labels=[False, True]).ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        f1_cls = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else 0.0
        )
        bacc_cls = np.nanmean([recall, specificity])

        rows.append({
            'lineage': str(lineage),
            'iter': int(iter_idx),
            'eval_type': eval_type,
            'name': name,
            'rank': rank,
            'subtype_class': str(cls_name),
            'bacc_ovr': float(bacc_cls) if np.isfinite(bacc_cls) else np.nan,
            'f1_ovr': float(f1_cls) if np.isfinite(f1_cls) else np.nan,
            'precision': float(precision) if np.isfinite(precision) else np.nan,
            'recall': float(recall) if np.isfinite(recall) else np.nan,
            'specificity': float(specificity) if np.isfinite(specificity) else np.nan,
            'support': int(tp + fn),
            'predicted_n': int(tp + fp),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'oob_n': int(oob_n) if oob_n is not None else int(len(y_true)),
            'excluded_subtypes': ','.join(excluded_subtypes),
            'timestamp': pd.Timestamp.now().isoformat()
        })

# %%
# ==============================================================================
# [Step 04B-4] Lineage-wise (Pan-Cellular) ML ensemble training and prediction
# 4 models: RF(GPU) + XGB(GPU) + ENet(CPU) + MLP(GPU)
# Full 100-iteration cumulative loop, 7:3 Patient ID split, Feature Importance
# ==============================================================================
if not step04_skip:
    print("🔹 [Step 04B-4] Starting sequential ensemble subtype prediction...", flush=True)
    print(f"  ⏱️ Start time: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    adata.obs['ML_Subtype_Raw']  = adata.obs['GT_Subtype'].copy()
    adata.obs['ML_Subtype_Pure'] = adata.obs['GT_Subtype'].copy()
    adata.obs['ML_Subtype_Conf'] = 1.0

    dlbcl_mask_4 = adata.obs['Disease'].str.contains('DLBCL', na=False)
    ref_mask_4   = ~dlbcl_mask_4
    X_all4       = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    lineages     = ['T cells', 'B cells', 'NK cells', 'Myeloid cells', 'Stromal cells']
    feature_names_sub = adata.var_names.tolist()

    val_records = []
    training_log_rows = []
    per_subtype_metric_rows = []
    subtype_training_exclusions = []
    fi_log_subtype = {}  # Feature Importance per lineage
    subtype_transfer_manifests = {}

    if V9_EXPORT_FROZEN_MODELS:
        from V9_transfer_utils import (
            finalize_transfer_stage,
            mlp_iteration_path,
            prepare_transfer_stage,
            register_mlp_iteration,
            save_iteration_scaler,
            save_joblib_model_iteration,
            save_xgb_iteration,
        )

    if 'history_sub' not in globals():
        global history_sub
        history_sub = {}
    if 'all_sub_probs_global' not in globals():
        global all_sub_probs_global
        all_sub_probs_global = {}

    for lin_idx_loop, lin in enumerate(lineages):
        print(f"\n{'='*60}\n🚀 [{lin_idx_loop+1}/{len(lineages)}] Starting lineage: {lin}  ⏱️ {pd.Timestamp.now().strftime('%H:%M:%S')}", flush=True)

        cp.get_default_memory_pool().free_all_blocks(); gc.collect()

        idx_ref   = np.where(ref_mask_4 & (adata.obs['Modeling_Broad'] == lin))[0]
        idx_dlbcl = np.where(dlbcl_mask_4 & (adata.obs['ML_Lineage_Raw'] == lin))[0]

        excluded_subtypes = _step04_excluded_subtypes(lin)
        if excluded_subtypes and len(idx_ref) > 0:
            ref_subtype_values = adata.obs['GT_Subtype'].iloc[idx_ref].astype(str).values
            keep_ref = ~np.isin(ref_subtype_values, excluded_subtypes)
            removed_n = int((~keep_ref).sum())
            idx_ref = idx_ref[keep_ref]
            subtype_training_exclusions.append({
                'lineage': lin,
                'excluded_subtypes': ','.join(excluded_subtypes),
                'removed_reference_cells': removed_n
            })
            print(f"  ℹ️ Subtypes excluded from main training: {excluded_subtypes} ({removed_n:,} Reference cells excluded)")

        if len(idx_ref) < 10 or len(idx_dlbcl) == 0:
            print(f"  ⚠️ {lin}: insufficient data → skipping"); continue

        y_ref_str        = adata.obs['GT_Subtype'].iloc[idx_ref].astype(str).values
        X_ref_lin        = np.ascontiguousarray(X_all4[idx_ref], dtype=np.float32)
        X_dlbcl_lin      = np.ascontiguousarray(X_all4[idx_dlbcl], dtype=np.float32)
        groups_lin       = adata.obs['PatientID'].iloc[idx_ref].values

        le_sub = LabelEncoder()
        y_enc_lin = le_sub.fit_transform(y_ref_str)
        n_cls_sub = len(le_sub.classes_)

        if n_cls_sub <= 1:
            sc_lbl = le_sub.classes_[0]
            _mask_single = np.zeros(len(adata), dtype=bool)
            _mask_single[idx_dlbcl] = True
            adata.obs.loc[_mask_single, 'ML_Subtype_Raw'] = sc_lbl
            adata.obs.loc[_mask_single, 'ML_Subtype_Pure'] = sc_lbl
            print(f"  Single class ({sc_lbl}) → passing through"); continue

        print(f"  📊 [{lin}] Classes: {le_sub.classes_}, training: {len(idx_ref):,}, prediction: {len(idx_dlbcl):,}")

        subtype_transfer_stage = None
        if V9_EXPORT_FROZEN_MODELS:
            subtype_transfer_stage = prepare_transfer_stage(
                V9_TRANSFER_ROOT,
                stage='subtype',
                lineage=lin,
                feature_names=feature_names_sub,
                classes=le_sub.classes_,
                expected_iterations=N_ITER,
                final_combo=None,
                overwrite=V9_TRANSFER_OVERWRITE_EXISTING,
                metadata={
                    'source_code': 'Final_step04B.py',
                    'random_seed': RANDOM_SEED,
                    'cv_test_size': CV_TEST_SIZE,
                    'excluded_subtypes': excluded_subtypes,
                    'model_selection': 'Final manual combo and iteration selection',
                },
            )
            print(f"  💾 [{lin}] frozen-model save location: {subtype_transfer_stage}")

        fi_log_subtype[lin] = {k: [] for k in MODEL_KEYS}
        history_sub[lin] = {
            'iters': [],
            'best': {'bacc': [], 'f1': [], 'avg': [], 'combo': []},
            **{k: {'bacc': [], 'f1': []} for k in MODEL_KEYS}
        }

        sub_probs = {k: [] for k in MODEL_KEYS}
        oob_probs_sub  = {k: np.zeros((len(idx_ref), n_cls_sub), dtype=np.float32) for k in MODEL_KEYS}
        oob_counts_sub = {k: np.zeros(len(idx_ref), dtype=np.int32) for k in MODEL_KEYS}

        sub_best_val_score = -1.0
        sub_best_val_combo = ''
        sub_actual_iters = 0

        _sub_es_combos = []
        for _r in range(1, len(MODEL_KEYS) + 1):
            for _c in _combs(MODEL_KEYS, _r):
                _sub_es_combos.append(_c)
        _sub_es_combo_names = ['+'.join(c) for c in _sub_es_combos]

        for i in tqdm(range(N_ITER), desc=f"{lin} Ensemble"):
            sub_actual_iters = i + 1

            # As in Step02, split 7:3 by Patient ID at each iteration and fit the scaler only on the training fold.
            gss_sub = GroupShuffleSplit(n_splits=1, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED + i)
            for ti, vi in gss_sub.split(X_ref_lin, y_enc_lin, groups=groups_lin):
                pass

            if len(np.unique(y_enc_lin[ti])) < n_cls_sub:
                print(f"\n      Iter {i+1}/{N_ITER} ({lin}) skipped: not all subtypes are present in the training fold", flush=True)
                continue

            scaler_iter = StandardScaler()
            X_tr_sub    = np.ascontiguousarray(scaler_iter.fit_transform(X_ref_lin[ti]), dtype=np.float32)
            X_val_sub   = np.ascontiguousarray(scaler_iter.transform(X_ref_lin[vi]), dtype=np.float32)
            X_dlbcl_sc  = np.ascontiguousarray(scaler_iter.transform(X_dlbcl_lin), dtype=np.float32)
            y_tr_sub    = y_enc_lin[ti]
            y_val_sub   = y_enc_lin[vi]

            if subtype_transfer_stage is not None:
                try:
                    save_iteration_scaler(
                        subtype_transfer_stage,
                        i + 1,
                        scaler_iter,
                        random_seed=RANDOM_SEED + i,
                        train_n=len(ti),
                        validation_n=len(vi),
                        train_patient_n=len(np.unique(groups_lin[ti])),
                        validation_patient_n=len(np.unique(groups_lin[vi])),
                    )
                except Exception as e:
                    print(f"        ⚠️ Failed to save the Iter {i+1} scaler (training continues): {e}")

            _vram = cp.get_default_memory_pool().used_bytes() / (1024**2)
            print(f"\n      Iter {i+1}/{N_ITER} ({lin}) | Train: {len(ti):,} | Val: {len(vi):,} | "
                  f"DLBCL: {len(idx_dlbcl):,} | VRAM: {_vram:.0f}MB | ⏱️ {pd.Timestamp.now().strftime('%H:%M:%S')}", flush=True)
            print(f"        Class distribution (Train): {dict(zip(*np.unique(y_tr_sub, return_counts=True)))}", flush=True)

            print("        🌲 RF...", end=' ', flush=True)
            try:
                X_tr_gpu = cp.asarray(X_tr_sub)
                y_tr_gpu = cp.asarray(y_tr_sub, dtype=cp.int32)
                X_dlbcl_gpu = cp.asarray(X_dlbcl_sc)
                X_val_gpu = cp.asarray(X_val_sub)

                rf_sub = cuRF(random_state=RANDOM_SEED + i)
                rf_sub.fit(X_tr_gpu, y_tr_gpu)

                p_d = _align_probability_columns(rf_sub.predict_proba(X_dlbcl_gpu), getattr(rf_sub, 'classes_', None), n_cls_sub)
                p_v = _align_probability_columns(rf_sub.predict_proba(X_val_gpu), getattr(rf_sub, 'classes_', None), n_cls_sub)

                if hasattr(rf_sub, 'feature_importances_'):
                    fi_v = rf_sub.feature_importances_
                    if hasattr(fi_v, 'get'): fi_v = fi_v.get()
                    record_feature_importance(fi_log_subtype[lin], 'RF', feature_names_sub, fi_v)

                sub_probs['RF'].append(p_d)
                oob_probs_sub['RF'][vi] += p_v
                oob_counts_sub['RF'][vi] += 1
                if subtype_transfer_stage is not None:
                    try:
                        save_joblib_model_iteration(
                            subtype_transfer_stage, i + 1, 'RF', rf_sub
                        )
                    except Exception as e:
                        print(f"⚠️ Failed to save RF model (predictions retained): {e}", end=' ')
                print("✅", end=' ', flush=True)
                del rf_sub, X_tr_gpu, y_tr_gpu, X_dlbcl_gpu, X_val_gpu
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"❌ {e}", flush=True)

            print("🌳 XGB...", end=' ', flush=True)
            try:
                xgb_sub = xgb.XGBClassifier(
                    tree_method='hist', device='cuda', random_state=RANDOM_SEED + i, verbosity=0
                )
                xgb_sub.fit(X_tr_sub, y_tr_sub.astype(np.int32))

                p_d = _align_probability_columns(xgb_sub.predict_proba(X_dlbcl_sc), getattr(xgb_sub, 'classes_', None), n_cls_sub)
                p_v = _align_probability_columns(xgb_sub.predict_proba(X_val_sub), getattr(xgb_sub, 'classes_', None), n_cls_sub)

                record_feature_importance(fi_log_subtype[lin], 'XGB', feature_names_sub, xgb_sub.feature_importances_)
                sub_probs['XGB'].append(p_d)
                oob_probs_sub['XGB'][vi] += p_v
                oob_counts_sub['XGB'][vi] += 1
                if subtype_transfer_stage is not None:
                    try:
                        save_xgb_iteration(subtype_transfer_stage, i + 1, xgb_sub)
                    except Exception as e:
                        print(f"⚠️ Failed to save XGB model (predictions retained): {e}", end=' ')
                print("✅", end=' ', flush=True)
                del xgb_sub
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"❌ {e}", flush=True)

            print("📏 ENet...", end=' ', flush=True)
            try:
                enet_sub = SGDClassifier(
                    loss='log_loss', penalty='elasticnet', l1_ratio=0.5,
                    random_state=RANDOM_SEED + i
                )
                enet_sub.fit(X_tr_sub, y_tr_sub)
                p_d = _align_probability_columns(enet_sub.predict_proba(X_dlbcl_sc), getattr(enet_sub, 'classes_', None), n_cls_sub)
                p_v = _align_probability_columns(enet_sub.predict_proba(X_val_sub), getattr(enet_sub, 'classes_', None), n_cls_sub)

                sub_probs['ENet'].append(p_d)
                oob_probs_sub['ENet'][vi] += p_v
                oob_counts_sub['ENet'][vi] += 1

                fi_enet = np.abs(enet_sub.coef_).mean(axis=0) if enet_sub.coef_.ndim > 1 else np.abs(enet_sub.coef_)
                record_feature_importance(fi_log_subtype[lin], 'ENet', feature_names_sub, fi_enet)
                if subtype_transfer_stage is not None:
                    try:
                        save_joblib_model_iteration(
                            subtype_transfer_stage, i + 1, 'ENet', enet_sub
                        )
                    except Exception as e:
                        print(f"⚠️ Failed to save ENet model (predictions retained): {e}", end=' ')
                print("✅", end=' ', flush=True)
                del enet_sub
            except Exception as e:
                print(f"❌ {e}", flush=True)

            print("🧠 MLP...", end=' ', flush=True)
            try:
                _mlp_pred_n_dlbcl = X_dlbcl_sc.shape[0]
                X_mlp_pred = np.ascontiguousarray(np.vstack([X_dlbcl_sc, X_val_sub]), dtype=np.float32)
                _mlp_model_output = (
                    mlp_iteration_path(subtype_transfer_stage, i + 1)
                    if subtype_transfer_stage is not None else None
                )
                p_mlp_all = spawn_mlp_predict_batched(
                    np.ascontiguousarray(X_tr_sub, dtype=np.float32),
                    np.ascontiguousarray(y_tr_sub, dtype=np.int64),
                    X_mlp_pred, n_cls_sub,
                    model_output_path=_mlp_model_output,
                    random_seed=RANDOM_SEED + i
                )
                p_mlp_d = _align_probability_columns(p_mlp_all[:_mlp_pred_n_dlbcl], None, n_cls_sub)
                p_mlp_v = _align_probability_columns(p_mlp_all[_mlp_pred_n_dlbcl:], None, n_cls_sub)
                sub_probs['MLP'].append(p_mlp_d)
                oob_probs_sub['MLP'][vi] += p_mlp_v
                oob_counts_sub['MLP'][vi] += 1
                if subtype_transfer_stage is not None:
                    try:
                        register_mlp_iteration(
                            subtype_transfer_stage, i + 1, _mlp_model_output
                        )
                    except Exception as e:
                        print(f"⚠️ Failed to register MLP model (predictions retained): {e}", end=' ')
                print("✅", flush=True)
                del X_mlp_pred, p_mlp_all
            except Exception as e:
                print(f"❌ {e}", flush=True)

            should_eval_oob = (
                ((i + 1) % OOB_EVAL_EVERY_STEP04 == 0)
                or ((i + 1) == N_ITER)
                or ((i + 1) in OOB_EVAL_EXTRA_ITERS_STEP04)
            )
            if not should_eval_oob:
                print(f"        🔍 [Iter {i+1}] Skipping OOB performance evaluation (OOB_EVAL_EVERY_STEP04={OOB_EVAL_EVERY_STEP04})", flush=True)
                del X_tr_sub, X_val_sub, X_dlbcl_sc, y_tr_sub, y_val_sub
                gc.collect()
                continue

            valid_mask = np.zeros(len(idx_ref), dtype=bool)
            for _mk in MODEL_KEYS:
                valid_mask |= oob_counts_sub[_mk] > 0

            if np.sum(valid_mask) > 0:
                combo_results_sub = []
                for combo, cname in zip(_sub_es_combos, _sub_es_combo_names):
                    combo_valid_mask = valid_mask.copy()
                    for k in combo:
                        combo_valid_mask &= oob_counts_sub[k] > 0
                    if np.sum(combo_valid_mask) == 0:
                        continue

                    y_combo_true = y_enc_lin[combo_valid_mask]
                    combo_probs_list = [
                        oob_probs_sub[k][combo_valid_mask] / oob_counts_sub[k][combo_valid_mask, np.newaxis]
                        for k in combo
                    ]
                    combo_val_avg = np.mean(combo_probs_list, axis=0)
                    combo_val_pred = np.argmax(combo_val_avg, axis=1)
                    combo_ba = balanced_accuracy_score(y_combo_true, combo_val_pred)
                    combo_f1 = f1_score(y_combo_true, combo_val_pred, average='macro', zero_division=0)
                    combo_avg = (combo_ba + combo_f1) / 2
                    combo_results_sub.append({
                        'Combo': cname, 'Avg': combo_avg, 'B-Acc': combo_ba,
                        'F1': combo_f1, 'OOB_N': int(np.sum(combo_valid_mask))
                    })

                combo_results_sub = sorted(combo_results_sub, key=lambda x: x['Avg'], reverse=True)
                if combo_results_sub:
                    top_res = combo_results_sub[0]
                    if top_res['Avg'] > sub_best_val_score:
                        sub_best_val_score = top_res['Avg']
                        sub_best_val_combo = top_res['Combo']
                        print(f"        🏆 [{lin}] Iter {i+1} new best score: Best={sub_best_val_combo} "
                              f"B-Acc={top_res['B-Acc']*100:.2f}% | F1={top_res['F1']*100:.2f}% | "
                              f"Avg={sub_best_val_score*100:.2f}%", flush=True)

                    print(f"        🏆 [Iter {i+1} cumulative ensemble Top 5] (OOB n={np.sum(valid_mask):,})")
                    for rank, res in enumerate(combo_results_sub[:5], 1):
                        prefix = "🌟" if rank == 1 else "  "
                        print(f"          {prefix} Rank {rank}: {res['Combo']:<20} | Avg: {res['Avg']*100:.2f}% "
                              f"(B-Acc: {res['B-Acc']*100:.1f}%, F1: {res['F1']*100:.1f}%, OOB n={res['OOB_N']:,})")

                    history_sub[lin]['iters'].append(i + 1)
                    history_sub[lin]['best']['bacc'].append(top_res['B-Acc'])
                    history_sub[lin]['best']['f1'].append(top_res['F1'])
                    history_sub[lin]['best']['avg'].append(top_res['Avg'])
                    history_sub[lin]['best']['combo'].append(top_res['Combo'])

                    for _mk in MODEL_KEYS:
                        _mk_mask = oob_counts_sub[_mk] > 0
                        if np.sum(_mk_mask) > 0:
                            _mk_probs = oob_probs_sub[_mk][_mk_mask] / oob_counts_sub[_mk][_mk_mask, np.newaxis]
                            _mk_pred = np.argmax(_mk_probs, axis=1)
                            _mk_true = y_enc_lin[_mk_mask]
                            _mk_bacc = balanced_accuracy_score(_mk_true, _mk_pred)
                            _mk_f1 = f1_score(_mk_true, _mk_pred, average='macro', zero_division=0)
                            _mk_oob_n = int(np.sum(_mk_mask))
                        else:
                            _mk_bacc, _mk_f1, _mk_oob_n = 0.0, 0.0, 0
                        history_sub[lin][_mk]['bacc'].append(_mk_bacc)
                        history_sub[lin][_mk]['f1'].append(_mk_f1)
                        if _mk_oob_n > 0:
                            append_per_subtype_metric_rows(
                                per_subtype_metric_rows, lin, i + 1, 'single_model', _mk,
                                _mk_true, _mk_pred, le_sub.classes_,
                                oob_n=_mk_oob_n, excluded_subtypes=excluded_subtypes
                            )
                        training_log_rows.append({
                            'lineage': lin,
                            'iter': i + 1,
                            'eval_type': 'single_model',
                            'name': _mk,
                            'rank': np.nan,
                            'bacc': _mk_bacc,
                            'f1': _mk_f1,
                            'avg': (_mk_bacc + _mk_f1) / 2,
                            'oob_n': _mk_oob_n,
                            'best_at_iter': False,
                            'best_overall_so_far': False,
                            'excluded_subtypes': ','.join(excluded_subtypes),
                            'timestamp': pd.Timestamp.now().isoformat()
                        })

                    for rank, res in enumerate(combo_results_sub, 1):
                        training_log_rows.append({
                            'lineage': lin,
                            'iter': i + 1,
                            'eval_type': 'ensemble_combo',
                            'name': res['Combo'],
                            'rank': rank,
                            'bacc': res['B-Acc'],
                            'f1': res['F1'],
                            'avg': res['Avg'],
                            'oob_n': res['OOB_N'],
                            'best_at_iter': rank == 1,
                            'best_overall_so_far': res['Combo'] == sub_best_val_combo and np.isclose(res['Avg'], sub_best_val_score),
                            'excluded_subtypes': ','.join(excluded_subtypes),
                            'timestamp': pd.Timestamp.now().isoformat()
                        })
                        combo_keys = res['Combo'].split('+')
                        combo_valid_mask = valid_mask.copy()
                        for k in combo_keys:
                            combo_valid_mask &= oob_counts_sub[k] > 0
                        if np.sum(combo_valid_mask) > 0:
                            y_combo_true = y_enc_lin[combo_valid_mask]
                            combo_probs_for_log = [
                                oob_probs_sub[k][combo_valid_mask] / oob_counts_sub[k][combo_valid_mask, np.newaxis]
                                for k in combo_keys
                            ]
                            combo_pred = np.argmax(np.mean(combo_probs_for_log, axis=0), axis=1)
                            append_per_subtype_metric_rows(
                                per_subtype_metric_rows, lin, i + 1, 'ensemble_combo', res['Combo'],
                                y_combo_true, combo_pred, le_sub.classes_,
                                rank=rank, oob_n=int(np.sum(combo_valid_mask)),
                                excluded_subtypes=excluded_subtypes
                            )

                    _step04b_write_table_if_enabled(
                        pd.DataFrame(training_log_rows), TRAIN_LOG_STEP04_PATH, index=False
                    )
                    if per_subtype_metric_rows:
                        _step04b_write_table_if_enabled(
                            pd.DataFrame(per_subtype_metric_rows),
                            PER_SUBTYPE_METRIC_STEP04_PATH,
                            index=False,
                        )
                    print(f"        💾 [Iter {i+1}] Training log saved: {TRAIN_LOG_STEP04_PATH}", flush=True)

                    if (i + 1) == N_ITER:
                        print(f"\n        🎯 [{lin}] Final predictions completed - ranking of all 15 model combinations")
                        for rank, res in enumerate(combo_results_sub, 1):
                            print(f"          Rank {rank:2d}: {res['Combo']:<20} | Avg: {res['Avg']*100:.2f}%")

                        val_records.append({'Lineage': lin, 'Combo': top_res['Combo'], 'Metric': 'Balanced Acc', 'Score': top_res['B-Acc']})
                        val_records.append({'Lineage': lin, 'Combo': top_res['Combo'], 'Metric': 'F1-Score', 'Score': top_res['F1']})

            del X_tr_sub, X_val_sub, X_dlbcl_sc, y_tr_sub, y_val_sub
            gc.collect()

        print(f"  ✅ [{lin}] Prediction collection completed: {sub_actual_iters}/{N_ITER} iterations "
              f"(Best cumulative Combo: {sub_best_val_combo}, Avg: {sub_best_val_score*100:.2f}%)", flush=True)

        all_sub_probs_global[lin] = {
            'probs': sub_probs,
            'best_combo': sub_best_val_combo,
            'classes': le_sub.classes_,
            'excluded_subtypes': excluded_subtypes
        }

        summarize_feature_importance(fi_log_subtype[lin], feature_names_sub)

        if subtype_transfer_stage is not None:
            subtype_transfer_manifests[lin] = finalize_transfer_stage(
                subtype_transfer_stage,
                extra_metadata={
                    'best_oob_combo': sub_best_val_combo,
                    'best_oob_score': sub_best_val_score,
                },
            )
            print(
                f"  ✅ [{lin}] frozen 4-model package saved: "
                f"{len(subtype_transfer_manifests[lin]['all_model_complete_iterations'])} "
                "all-model-complete iterations"
            )

        del sub_probs, X_ref_lin, X_dlbcl_lin
        gc.collect()

    sub_bkp = {
        'all_sub_probs_global': all_sub_probs_global,
        'history_sub': history_sub,
        'training_log': pd.DataFrame(training_log_rows),
        'training_log_path': TRAIN_LOG_STEP04_PATH,
        'per_subtype_metric_log': pd.DataFrame(per_subtype_metric_rows),
        'per_subtype_metric_log_path': PER_SUBTYPE_METRIC_STEP04_PATH,
        'subtype_training_exclusions': pd.DataFrame(subtype_training_exclusions),
        'fi_log_subtype': fi_log_subtype,
        'feature_names': feature_names_sub,
        'frozen_transfer_manifests': subtype_transfer_manifests,
        'frozen_transfer_root': V9_TRANSFER_ROOT,
        'params': {
            'N_ITER': N_ITER,
            'MODEL_KEYS': MODEL_KEYS,
            'CV_TEST_SIZE': CV_TEST_SIZE,
            'RANDOM_SEED': RANDOM_SEED,
            'MANUAL_SUBTYPE_COMBO': 'XGB+MLP',
            'EXCLUDE_CD4CTL_FROM_MAIN_TCELL_MODEL': EXCLUDE_CD4CTL_FROM_MAIN_TCELL_MODEL,
            'STEP04_MAIN_EXCLUDED_SUBTYPES': STEP04_MAIN_EXCLUDED_SUBTYPES,
            'STEP04B_TPR_GMM_POLICY': adata.uns.get('step04B_tpr_gmm_policy', {})
        }
    }
    _step04b_dump_if_enabled(sub_bkp, PROBS_STEP04_PATH)
    print(f"✅ Lineage-wise OOB probabilities, Feature Importance, and logs saved: {PROBS_STEP04_PATH}")
    print(f"  ⏱️ Training end time: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    print("\n✅ [Step 04B-4] Sequential ensemble probability collection completed for all lineages!")

# %%
# ==============================================================================
# [Step 04B-4.5] Unified manual ensemble combination/iteration selection and performance plots
# ==============================================================================
# Shared combination and iteration count selected by the user after inspecting logs. (An empty '' automatically applies each lineage's highest score.)
# ※ Step 04B subtype prediction uses only four models (RF, XGB, ENet, MLP).
# Do not include 'Rescue' here: it is specific to Step 02 Broad Lineage spatial correction.
MANUAL_SUBTYPE_COMBO = 'XGB+MLP'    # Examples: 'XGB+MLP', 'RF+XGB+MLP'
MANUAL_SUBTYPE_ITER  = N_ITER # Examples: 50, 100
# Finalize the manual combo first so the Figure-only benchmark uses the same saved-selection
# contract even when it skips the large probability payload.
target_combo_sub = MANUAL_SUBTYPE_COMBO if MANUAL_SUBTYPE_COMBO.strip() else ''

if (
    not STEP04B_BENCHMARK_ONLY
    and 'all_sub_probs_global' not in globals()
    and os.path.exists(PROBS_STEP04_PATH)
):
    print("🔹 [Step 04B-4.5] Restoring existing Subtype ensemble logs from disk...")
    sub_bkp = joblib.load(PROBS_STEP04_PATH)
    all_sub_probs_global = sub_bkp['all_sub_probs_global']
    history_sub = sub_bkp['history_sub']
    if 'fi_log_subtype' in sub_bkp:
        fi_log_subtype = sub_bkp['fi_log_subtype']
    if 'per_subtype_metric_log' in sub_bkp:
        per_subtype_metric_log = sub_bkp['per_subtype_metric_log']
    if 'params' in sub_bkp and 'STEP04_MAIN_EXCLUDED_SUBTYPES' in sub_bkp['params']:
        STEP04_MAIN_EXCLUDED_SUBTYPES = sub_bkp['params']['STEP04_MAIN_EXCLUDED_SUBTYPES']
    print(f"  ✅ Restored (number of lineages: {len(all_sub_probs_global)})")

if 'all_sub_probs_global' in globals():
    print(f"🔹 [Step 04B-4.5] Applying manual ensemble (combination: {MANUAL_SUBTYPE_COMBO or 'Auto(Best)'}, Iter: {MANUAL_SUBTYPE_ITER})", flush=True)

    if 'lineages' not in globals():
        lineages = ['T cells', 'B cells', 'NK cells', 'Myeloid cells', 'Stromal cells']
        
    # ── Training performance over time + Feature Importance Bubble (combined Figure, Step 04B-9 approach) ──
    if 'history_sub' in globals():
        _MODEL_COLORS = {'RF': '#27AE60', 'XGB': '#E67E22', 'ENet': '#3498DB', 'MLP': '#9B59B6'}
        _STEP04B045_LINEAGE_ORDER = ['B cells', 'Myeloid cells', 'Stromal cells', 'T cells']
        valid_lins = [
            l for l in _STEP04B045_LINEAGE_ORDER
            if l in history_sub and len(history_sub[l]['iters']) > 0
        ]
        
        _YLIM_MAP = {
            'T cells': (0.725, 0.95), 'B cells': (0.825, 0.95),
            'Myeloid cells': (0.9, 1.0), 'Stromal cells': (0.94, 0.98),
            'NK cells': (0.7, 1.0)
        }
        _SUBTYPE_MAP = {
            'T cells': ['Th', 'Tfh', 'Treg', 'CD8T', 'CD4_CTL', 'Tpr'],
            'B cells': ['B cell', 'Plasma cell'],
            'Myeloid cells': ['Macrophage', 'DC', 'Myeloid'],
            'Stromal cells': ['Stromal', 'FDC'],
            'NK cells': ['NK cell']
        }
        for _lin_name, _excluded in STEP04_MAIN_EXCLUDED_SUBTYPES.items():
            if _excluded and _lin_name in _SUBTYPE_MAP:
                _SUBTYPE_MAP[_lin_name] = [s for s in _SUBTYPE_MAP[_lin_name] if s not in _excluded]

        def _collect_history_metric_values(history_entry, metric):
            vals = []
            if 'best' in history_entry:
                for model_key in MODEL_KEYS:
                    vals.extend(history_entry.get(model_key, {}).get(metric, []))
                if metric == 'bacc':
                    vals.extend(history_entry.get('best', {}).get('bacc', history_entry.get('best', {}).get('avg', [])))
                elif metric == 'f1':
                    vals.extend(history_entry.get('best', {}).get('f1', []))
            else:
                vals.extend(history_entry.get(metric, []))
                if metric == 'bacc':
                    vals.extend(history_entry.get('avg', []))
            arr = np.asarray(vals, dtype=float)
            return arr[np.isfinite(arr)]

        def _dynamic_history_ylim(history_entry, metric, base_ylim):
            if not STEP04_DYNAMIC_METRIC_AXIS:
                return base_ylim
            vals = _collect_history_metric_values(history_entry, metric)
            if vals.size == 0:
                return base_ylim
            ymin, ymax = float(np.min(vals)), float(np.max(vals))
            spread = max(ymax - ymin, 1e-4)
            pad = max(0.015, spread * 0.18)
            lo = max(0.0, min(base_ylim[0], ymin - pad))
            hi = min(1.02, max(base_ylim[1], ymax + pad))
            if (hi - lo) < 0.06:
                mid = (hi + lo) / 2
                lo = max(0.0, mid - 0.03)
                hi = min(1.02, mid + 0.03)
            return lo, hi

        def _step04b_markevery_5(iter_values):
            """Keep performance curves as lines but display markers only every 5 iterations."""
            iter_arr = np.asarray(iter_values, dtype=float)
            idx = [
                i for i, val in enumerate(iter_arr)
                if i == 0 or i == len(iter_arr) - 1 or (np.isfinite(val) and int(round(val)) % 5 == 0)
            ]
            return sorted(set(idx))

        def _step04b_set_iteration_ticks(ax, iter_values, label_step=10, tick_step=5):
            iter_arr = np.asarray(iter_values, dtype=float)
            iter_arr = iter_arr[np.isfinite(iter_arr)]
            if iter_arr.size == 0:
                return
            tick_step = max(1, int(tick_step))
            tick_start = int(np.floor(iter_arr.min() / tick_step) * tick_step)
            tick_end = int(np.ceil(iter_arr.max() / tick_step) * tick_step)
            ticks = np.arange(max(0, tick_start), tick_end + 1, tick_step)
            if ticks.size:
                ax.set_xticks(ticks)
                ax.set_xticklabels([
                    str(int(round(tick))) if int(round(tick)) % int(label_step) == 0 else ''
                    for tick in ticks
                ])

        def _step04b_plot_history_pair(
            ax_b,
            ax_f,
            history_entry,
            is_new_history,
            b_ylim,
            f_ylim,
            *,
            b_title='B-Acc',
            f_title='F1',
            show_ylabels=True,
            title_fontsize=11.0,
            label_fontsize=9.0,
            tick_fontsize=7.8,
            marker_size=3.8,
            best_marker_size=5.0,
            line_width=1.05,
            best_line_width=1.9,
            iteration_label_step=10,
            iteration_tick_step=5,
            right_xlim_pad=0.0,
            ensemble_entry=None,
            ensemble_label='Best Ensemble',
        ):
            from matplotlib.ticker import MaxNLocator

            markevery_local = _step04b_markevery_5(history_entry['iters'])
            ensemble_history = ensemble_entry if ensemble_entry is not None else history_entry.get('best', {})

            if is_new_history:
                for model_key in MODEL_KEYS:
                    if model_key in history_entry and history_entry[model_key]['bacc']:
                        ax_b.plot(
                            history_entry['iters'],
                            history_entry[model_key]['bacc'],
                            marker='o',
                            ms=marker_size,
                            markevery=markevery_local,
                            label=model_key,
                            color=_MODEL_COLORS.get(model_key),
                            lw=line_width,
                            alpha=0.82,
                        )
                best_bacc_local = ensemble_history.get('bacc') if ensemble_history.get('bacc') else ensemble_history.get('avg', [])
                ax_b.plot(
                    history_entry['iters'],
                    best_bacc_local,
                    marker='^',
                    ms=best_marker_size,
                    markevery=markevery_local,
                    label=ensemble_label,
                    color='#E74C3C',
                    lw=best_line_width,
                )
            else:
                ax_b.plot(history_entry['iters'], history_entry['bacc'], marker='o', markevery=markevery_local, label='B-Acc', color='#5DADE2')
                ax_b.plot(history_entry['iters'], history_entry['avg'], marker='^', markevery=markevery_local, label='Best Avg', color='#E74C3C', lw=best_line_width)
            ax_b.axvline(x=MANUAL_SUBTYPE_ITER, color='k', ls='--', lw=1.4, label=f'Cut-off ({MANUAL_SUBTYPE_ITER})')
            ax_b.set_title(b_title, fontsize=title_fontsize, fontweight='bold', pad=4)
            ax_b.set_ylabel("B-Acc" if show_ylabels else "", fontsize=label_fontsize, fontweight='bold')
            ax_b.set_ylim(b_ylim)
            ax_b.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
            ax_b.grid(axis='y', ls='--', alpha=0.48)
            ax_b.grid(axis='x', which='major', ls='--', alpha=0.48)
            _step04b_set_iteration_ticks(
                ax_b,
                history_entry['iters'],
                label_step=iteration_label_step,
                tick_step=iteration_tick_step,
            )
            ax_b.tick_params(labelsize=tick_fontsize)
            ax_b.tick_params(axis='x', labelbottom=False)

            if is_new_history:
                for model_key in MODEL_KEYS:
                    if model_key in history_entry and history_entry[model_key]['f1']:
                        ax_f.plot(
                            history_entry['iters'],
                            history_entry[model_key]['f1'],
                            marker='s',
                            ms=marker_size,
                            markevery=markevery_local,
                            color=_MODEL_COLORS.get(model_key),
                            lw=line_width,
                            alpha=0.82,
                        )
                ax_f.plot(
                    history_entry['iters'],
                    ensemble_history['f1'],
                    marker='^',
                    ms=best_marker_size,
                    markevery=markevery_local,
                    color='#E74C3C',
                    lw=best_line_width,
                )
            else:
                ax_f.plot(history_entry['iters'], history_entry['f1'], marker='s', markevery=markevery_local, color='#F1C40F')
            ax_f.axvline(x=MANUAL_SUBTYPE_ITER, color='k', ls='--', lw=1.4)
            ax_f.set_title(f_title, fontsize=title_fontsize, fontweight='bold', pad=4)
            ax_f.set_xlabel("Iteration", fontsize=label_fontsize, fontweight='bold', labelpad=4)
            ax_f.set_ylabel("Macro F1" if show_ylabels else "", fontsize=label_fontsize, fontweight='bold')
            ax_f.set_ylim(f_ylim)
            ax_f.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
            ax_f.grid(axis='y', ls='--', alpha=0.48)
            ax_f.grid(axis='x', which='major', ls='--', alpha=0.48)
            _step04b_set_iteration_ticks(
                ax_f,
                history_entry['iters'],
                label_step=iteration_label_step,
                tick_step=iteration_tick_step,
            )
            ax_f.tick_params(labelsize=tick_fontsize)

            finite_iters = np.asarray(history_entry['iters'], dtype=float)
            finite_iters = finite_iters[np.isfinite(finite_iters)]
            if finite_iters.size:
                x_left = min(0.0, float(finite_iters.min()))
                x_right = float(finite_iters.max()) + float(right_xlim_pad)
                ax_b.set_xlim(x_left, x_right)
                ax_f.set_xlim(x_left, x_right)

            if not show_ylabels:
                ax_b.tick_params(axis='y', labelleft=False)
                ax_f.tick_params(axis='y', labelleft=False)
            for tick_label in ax_b.get_xticklabels() + ax_b.get_yticklabels() + ax_f.get_xticklabels() + ax_f.get_yticklabels():
                tick_label.set_fontweight('bold')

            return ax_b.get_legend_handles_labels()
        
        _selected_combo_curves_45 = {}
        if os.path.exists(TRAIN_LOG_STEP04_PATH):
            try:
                _combo_log_45 = pd.read_csv(
                    TRAIN_LOG_STEP04_PATH,
                    usecols=['lineage', 'iter', 'eval_type', 'name', 'bacc', 'f1', 'avg'],
                )
                _combo_log_45 = _combo_log_45[
                    (_combo_log_45['eval_type'] == 'ensemble_combo')
                    & (_combo_log_45['name'] == MANUAL_SUBTYPE_COMBO)
                ]
                for _lin_combo_45 in valid_lins:
                    _iters_combo_45 = np.asarray(history_sub[_lin_combo_45]['iters'], dtype=int)
                    _rows_combo_45 = (
                        _combo_log_45[_combo_log_45['lineage'] == _lin_combo_45]
                        .drop_duplicates('iter', keep='last')
                        .set_index('iter')
                        .reindex(_iters_combo_45)
                    )
                    if len(_rows_combo_45) == len(_iters_combo_45) and not _rows_combo_45[['bacc', 'f1']].isna().any().any():
                        _selected_combo_curves_45[_lin_combo_45] = {
                            'bacc': _rows_combo_45['bacc'].astype(float).tolist(),
                            'f1': _rows_combo_45['f1'].astype(float).tolist(),
                            'avg': _rows_combo_45['avg'].astype(float).tolist(),
                        }
            except Exception as exc:
                print(f'  ⚠️ Failed to restore the Selected ensemble curve: {exc}')

        if len(valid_lins) > 0:
            import matplotlib.gridspec as gridspec
            import matplotlib.lines as mlines
            import scipy.stats as stats
            import pickle
            from sklearn.preprocessing import StandardScaler
            
            # Same FI data and Global Scaled data as Step 04B-9
            _fi_path_45 = os.path.join(CKPT_DIR, 'mv1_feature_importance_04B.pkl')
            _gfi_45 = {}
            if os.path.exists(_fi_path_45):
                try:
                    with open(_fi_path_45, 'rb') as f: _gfi_45 = pickle.load(f).get('subtype', {})
                except: pass
            
            _Xz = _step04b_get_global_scaled_matrix(adata)
            _ref_mask = (~adata.obs['Disease'].str.contains('DLBCL', na=False)) & \
                        (~np.isin(adata.obs['GT_Subtype'], ['na', 'Unknown', 'nan', '']))
            _combined_performance_payloads_45 = {}
            for lin in valid_lins:
                h = history_sub[lin]
                _is_new = 'best' in h
                yl, yh = _YLIM_MAP.get(lin, (0.3, 1.02))
                b_yl, b_yh = _dynamic_history_ylim(h, 'bacc', (yl, yh))
                f_yl, f_yh = _dynamic_history_ylim(h, 'f1', (yl, yh))
                _combined_performance_payloads_45[lin] = {
                    'history': h,
                    'is_new': _is_new,
                    'b_ylim': (b_yl, b_yh),
                    'f_ylim': (f_yl, f_yh),
                    'selected_ensemble': _selected_combo_curves_45.get(lin),
                }
                vsubs = [s for s in _SUBTYPE_MAP.get(lin, []) if (adata.obs['GT_Subtype'] == s).any()]
                sub_colors = {s: CELL_COLORS.get(s, '#AAA') for s in vsubs}
                
                # ── Marker extraction using the Step 04B-9 approach ──
                _ref_lin = _ref_mask & (adata.obs['Modeling_Broad'] == lin)
                recs_xgb = _gfi_45.get(lin, {}).get('XGB', [])
                n_it = len(recs_xgb)
                _cnt, _imp_sum = defaultdict(int), defaultdict(float)
                for rec in recs_xgb:
                    sf = sorted(rec.items(), key=lambda x: abs(x[1]), reverse=True)
                    for fn, imp in rec.items(): _imp_sum[fn] += abs(imp)
                    for fn, _ in sf[:20]: _cnt[fn] += 1
                _mean_imp = {k: v / max(n_it, 1) for k, v in _imp_sum.items()}
                
                sub_top = {}
                for s in vsubs:
                    is_ = np.where((adata.obs['GT_Subtype'] == s) & _ref_lin)[0]
                    io_ = np.where((adata.obs['GT_Subtype'] != s) & _ref_lin)[0]
                    if len(is_) > 0 and len(io_) > 0:
                        ms = _Xz[is_].mean(0); mo = _Xz[io_].mean(0)
                        vi = np.where((ms > 0.1) & ((ms - mo) > 0.1))[0]
                        sig = []
                        for idx in vi:
                            st, pv = stats.ranksums(_Xz[is_, idx], _Xz[io_, idx])
                            if st > 0 and pv < 0.05: sig.append((adata.var_names[idx], pv))
                        sub_top[s] = sorted(sig, key=lambda x: (_cnt[x[0]], _mean_imp.get(x[0], 0)), reverse=True)[:15]
                
                mbest = {}
                for s in vsubs:
                    is_ = np.where((adata.obs['GT_Subtype'] == s) & _ref_lin)[0]
                    if len(is_) > 0:
                        ms = _Xz[is_].mean(0)
                        for mn, _ in sub_top.get(s, []):
                            mi = adata.var_names.get_loc(mn)
                            if mn not in mbest or ms[mi] > mbest[mn][1]: mbest[mn] = (s, ms[mi])
                
                _dl_lin = (adata.obs['ML_Lineage_Raw'] == lin) & adata.obs['Disease'].str.contains('DLBCL', na=False)
                dl_means = {}
                for s in vsubs:
                    ids = np.where((adata.obs['GT_Subtype'] == s) & _dl_lin)[0]
                    dl_means[s] = _Xz[ids].mean(0) if len(ids) > 0 else np.zeros(_Xz.shape[1])
                
                ordered_markers, mk2sub = [], {}
                for s in vsubs:
                    sp = [m for m, _ in sub_top.get(s, []) if m in mbest and mbest[m][0] == s]
                    sp.sort(key=lambda m: dl_means[s][adata.var_names.get_loc(m)], reverse=True)
                    for m in sp:
                        if m not in ordered_markers: ordered_markers.append(m); mk2sub[m] = s
                
                has_fi = len(ordered_markers) > 0
                
                if has_fi:
                    # FI per model
                    _fi_ms = [m for m in MODEL_KEYS if m in _gfi_45.get(lin, {})]
                    if not _fi_ms and 'fi_log_subtype' in globals() and lin in fi_log_subtype:
                        _fi_ms = [m for m in MODEL_KEYS if m in fi_log_subtype[lin] and fi_log_subtype[lin][m]]
                    if not _fi_ms: _fi_ms = MODEL_KEYS
                    # [Revision] Exclude MLP from the plot because its Feature Selection is less clear.
                    _fi_ms = [m for m in _fi_ms if m != 'MLP']
                    
                    mfi = {}
                    for mn in _fi_ms:
                        rs = _gfi_45.get(lin, {}).get(mn, [])
                        if not rs and 'fi_log_subtype' in globals() and lin in fi_log_subtype:
                            rs = fi_log_subtype[lin].get(mn, [])
                        avg = defaultdict(float)
                        for r in rs:
                            for fn, v in r.items(): avg[fn] += abs(v)
                        if rs: avg = {k: v/len(rs) for k, v in avg.items()}
                        tot = sum(avg.values())
                        if tot > 1.5: avg = {k: v/tot for k, v in avg.items()}
                        mfi[mn] = avg
                    
                    mk_pv = {}
                    for mk in ordered_markers:
                        asub = mk2sub.get(mk)
                        if asub:
                            is_ = np.where((adata.obs['GT_Subtype'] == asub) & _ref_lin)[0]
                            io_ = np.where((adata.obs['GT_Subtype'] != asub) & _ref_lin)[0]
                            if len(is_) > 2 and len(io_) > 2:
                                mi = adata.var_names.get_loc(mk)
                                ts, pv = stats.ranksums(_Xz[is_, mi], _Xz[io_, mi])
                                mk_pv[mk] = pv/2 if ts > 0 else 1.0
                            else: mk_pv[mk] = 1.0
                        else: mk_pv[mk] = 1.0
                    
                    pd_rows = []
                    for mk in ordered_markers:
                        for mn in _fi_ms:
                            pd_rows.append({'Model': mn, 'Marker': mk, 'Importance': mfi.get(mn, {}).get(mk, 0),
                                            'Subtype': mk2sub.get(mk, '?'), 'P_value': mk_pv.get(mk, 1.0)})
                    df_plot = pd.DataFrame(pd_rows)
                    xlab = _fi_ms
                    xpd = {m: i for i, m in enumerate(xlab)}
                    ylab = ordered_markers
                    ypd = {m: i for i, m in enumerate(ylab)}
                    df_plot['X_Pos'] = df_plot['Model'].map(xpd)
                    df_plot['Y_Pos'] = df_plot['Marker'].map(ypd)
                    mx = max(float(df_plot['Importance'].max()), 1e-12)
                    df_plot['Rel_Importance'] = (df_plot['Importance'] / mx).clip(0, 1)
                    lpv = -np.log10(np.clip(df_plot['P_value'], 1e-50, 1.0))
                    df_plot['NegLog10_P'] = lpv
                    lpv_max = max(float(np.nanmax(lpv)) if len(lpv) else 0.0, 1.0)
                    size_scale = 380 if len(ylab) >= 24 else 470
                    size_base = 22
                    ec = [sub_colors.get(st, 'gray') for st in df_plot['Subtype']]
                
                # ── Combined Figure ──
                # Keep only performance curves in Combined output; retain the FI bubble panel only in separate 05B code.
                show_fi_panel = has_fi and SHOW_STEP04B045_IMPORTANCE_PANEL
                _marker_n = len(ordered_markers)
                if show_fi_panel:
                    _fig_w = 16.8
                    _fig_h = max(8.8, min(13.8, 5.4 + 0.30 * max(_marker_n, 1)))
                else:
                    # In individual figures, arrange B-Acc/F1 in two vertically stacked rows,
                    # using a narrow portrait ratio suited for external placement of four figures side by side.
                    _fig_w = 4.15
                    _fig_h = 5.85
                fig = plt.figure(figsize=(_fig_w, _fig_h))
                if show_fi_panel:
                    gs = gridspec.GridSpec(
                        3, 3,
                        width_ratios=[1.12, 1.00, 0.48],
                        height_ratios=[1, 1, 0.15],
                        wspace=0.34,
                        hspace=0.34
                    )
                else:
                    gs = gridspec.GridSpec(
                        3, 1,
                        height_ratios=[1.0, 1.0, 0.34],
                        hspace=0.44
                    )
                
                if show_fi_panel:
                    ax_b = fig.add_subplot(gs[0, 0])
                    ax_f = fig.add_subplot(gs[1, 0], sharex=ax_b)
                    _bacc_title = f"{lin}: B-Acc"
                    _f1_title = f"{lin}: F1"
                else:
                    ax_b = fig.add_subplot(gs[0, 0])
                    ax_f = fig.add_subplot(gs[1, 0], sharex=ax_b)
                    _bacc_title = "B-Acc"
                    _f1_title = "F1"
                markevery_5 = _step04b_markevery_5(h['iters'])
                
                if _is_new:
                    for mk in MODEL_KEYS:
                        if mk in h and h[mk]['bacc']:
                            ax_b.plot(h['iters'], h[mk]['bacc'], marker='o', ms=4.2, markevery=markevery_5, label=mk, color=_MODEL_COLORS.get(mk), lw=1.15, alpha=0.8)
                    best_bacc_values = h['best'].get('bacc') if h['best'].get('bacc') else h['best'].get('avg', [])
                    ax_b.plot(h['iters'], best_bacc_values, marker='^', ms=5.8, markevery=markevery_5, label='Best Ensemble', color='#E74C3C', lw=2.2)
                else:
                    ax_b.plot(h['iters'], h['bacc'], marker='o', markevery=markevery_5, label='B-Acc', color='#5DADE2')
                    ax_b.plot(h['iters'], h['avg'], marker='^', markevery=markevery_5, label='Best Avg', color='#E74C3C', lw=2)
                ax_b.axvline(x=MANUAL_SUBTYPE_ITER, color='k', ls='--', lw=1.7, label=f'Cut-off ({MANUAL_SUBTYPE_ITER})')
                ax_b.set_title(_bacc_title, fontsize=11.6, fontweight='bold', pad=5); ax_b.set_ylabel("B-Acc", fontsize=9.6, fontweight='bold')
                ax_b.set_ylim(b_yl, b_yh); ax_b.grid(True, ls='--', alpha=0.5)
                _step04b_set_iteration_ticks(ax_b, h['iters'])
                ax_b.tick_params(labelsize=8.2); [l.set_fontweight('bold') for l in ax_b.get_xticklabels()+ax_b.get_yticklabels()]
                if not show_fi_panel:
                    ax_b.tick_params(axis='x', labelbottom=False)
                
                if _is_new:
                    for mk in MODEL_KEYS:
                        if mk in h and h[mk]['f1']:
                            ax_f.plot(h['iters'], h[mk]['f1'], marker='s', ms=4.2, markevery=markevery_5, color=_MODEL_COLORS.get(mk), lw=1.15, alpha=0.8)
                    ax_f.plot(h['iters'], h['best']['f1'], marker='^', ms=5.8, markevery=markevery_5, color='#E74C3C', lw=2.2)
                else:
                    ax_f.plot(h['iters'], h['f1'], marker='s', markevery=markevery_5, color='#F1C40F')
                ax_f.axvline(x=MANUAL_SUBTYPE_ITER, color='k', ls='--', lw=1.7)
                ax_f.set_title(_f1_title, fontsize=11.6, fontweight='bold', pad=5); ax_f.set_xlabel("Iteration", fontsize=9.6, fontweight='bold', labelpad=5)
                ax_f.set_ylabel("F1", fontsize=9.6, fontweight='bold'); ax_f.set_ylim(f_yl, f_yh); ax_f.grid(True, ls='--', alpha=0.5)
                _step04b_set_iteration_ticks(ax_f, h['iters'])
                ax_f.tick_params(labelsize=8.2); [l.set_fontweight('bold') for l in ax_f.get_xticklabels()+ax_f.get_yticklabels()]
                
                ax_lg = fig.add_subplot(gs[2, 0]); ax_lg.axis('off')
                hl, ll = ax_b.get_legend_handles_labels()
                ax_lg.legend(hl, ll, loc='center', ncol=2, fontsize=7.4, frameon=False)
                
                if show_fi_panel:
                    ax_bb = fig.add_subplot(gs[:, 1])
                    ax_idx = fig.add_subplot(gs[:, 2])
                    ax_idx.axis('off')
                    sz = df_plot['Rel_Importance'] * size_scale + size_base
                    
                    sc = ax_bb.scatter(
                        df_plot['X_Pos'],
                        df_plot['Y_Pos'],
                        s=sz,
                        c=lpv,
                        cmap='Greys',
                        vmin=0,
                        vmax=lpv_max,
                        edgecolors=ec,
                        linewidths=1.9,
                        alpha=0.72
                    )
                    _marker_font = max(6.2, min(9.2, 145.0 / max(len(ylab), 1)))
                    ax_bb.set_xticks(range(len(xlab))); ax_bb.set_xticklabels(xlab, fontsize=9.5, fontweight='bold')
                    ax_bb.set_yticks(range(len(ylab))); ax_bb.set_yticklabels(ylab, fontsize=_marker_font, fontweight='bold')
                    for marker_name, tick_label in zip(ylab, ax_bb.get_yticklabels()):
                        tick_label.set_color(sub_colors.get(mk2sub.get(marker_name), 'black'))
                        tick_label.set_fontweight('bold')
                    
                    bds, csb = [], mk2sub.get(ordered_markers[0]) if ordered_markers else None
                    for bi, mk in enumerate(ordered_markers):
                        if mk2sub.get(mk) != csb: bds.append(bi - 0.5); csb = mk2sub.get(mk)
                    for b in bds: ax_bb.axhline(y=b, color='gray', ls='--', alpha=0.5)
                    
                    ax_bb.set_xlim(-0.5, len(xlab)-0.5)
                    ax_bb.set_ylim(len(ylab) - 0.15, -0.85)
                    ax_bb.grid(True, ls=':', alpha=0.6, axis='both')
                    ax_bb.set_title(f"Feature importance: {lin}", fontsize=12.8, fontweight='bold', pad=8)
                    cb = plt.colorbar(sc, ax=ax_bb, shrink=0.42, pad=0.02)
                    cb.set_label('-log10(P-value)', fontweight='bold', fontsize=8.6)
                    cb.ax.tick_params(labelsize=7.8)
                    
                    le = [mlines.Line2D([], [], color='w', marker='', label='[Relative Importance]')]
                    for v in [0.25, 0.5, 0.75, 1.0]:
                        le.append(mlines.Line2D([], [], color='w', marker='o', markerfacecolor='gray', markeredgecolor='none', alpha=0.72, markersize=np.sqrt(v*size_scale+size_base), label=f'{int(v*100)}%'))
                    le.append(mlines.Line2D([], [], color='w', marker='', label='\n[Subtype]'))
                    for st in vsubs:
                        le.append(mlines.Line2D([], [], color='w', marker='o', markerfacecolor='w', markeredgecolor=sub_colors[st], markeredgewidth=2.0, markersize=8.0, label=st))
                    lg = ax_idx.legend(handles=le, loc='center', title="Index", title_fontsize=9.8, fontsize=8.1, frameon=True, shadow=True, labelspacing=0.80, borderpad=0.55, handletextpad=0.45)
                    lg.get_title().set_fontweight('bold')
                    for _txt in lg.get_texts():
                        _label = _txt.get_text().strip()
                        if _label in sub_colors:
                            _txt.set_color(sub_colors[_label])
                            _txt.set_fontweight('bold')
                        elif _label.startswith('['):
                            _txt.set_fontweight('bold')
                
                fig.suptitle(
                    f"{lin} ensemble performance",
                    fontsize=12.2,
                    fontweight='bold',
                    y=0.965
                )
                if show_fi_panel:
                    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.075, wspace=0.34, hspace=0.34)
                else:
                    fig.subplots_adjust(left=0.150, right=0.985, top=0.865, bottom=0.080, hspace=0.44)
                _combined_path = os.path.join(FIG_DIR, f'Final_mv1_Step04B-04-5_Combined_{lin.replace(" ", "_")}.png')
                _save_step04b_canvas_figure(
                    fig,
                    _combined_path,
                    label=f'Step04B-04-5 combined {lin}',
                    show=True,
                    close=True
                )

                if has_fi:
                    # 04B-5B: represent Importance with opacity (alpha) and P-value with dot size.
                    p_neglog_values = df_plot['NegLog10_P'].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
                    p_level_low, p_level_high = float(np.nanmin(p_neglog_values)), float(np.nanmax(p_neglog_values))
                    if np.isclose(p_level_low, p_level_high):
                        p_level_edges = np.array([p_level_low, p_level_low, p_level_low, p_level_low, p_level_low], dtype=float)
                        p_size_levels = np.full(len(p_neglog_values), 2, dtype=int)
                    else:
                        p_level_edges = np.linspace(p_level_low, p_level_high, 5)
                        p_size_levels = np.digitize(p_neglog_values, p_level_edges[1:-1], right=True) + 1
                        p_size_levels = np.clip(p_size_levels, 1, 4).astype(int)
                    p_size_map = {1: 38.0, 2: 95.0, 3: 175.0, 4: 285.0}
                    df_plot['P_Size_Level'] = p_size_levels
                    p_sizes = np.array([p_size_map[level] for level in p_size_levels], dtype=float)
                    imp_alpha = 0.18 + df_plot['Rel_Importance'].clip(0, 1) * 0.82
                    rgba_colors = [
                        list(to_rgba('#555555', alpha=float(a)))
                        for a in imp_alpha
                    ]

                    fig_b, ax_b5b = plt.subplots(figsize=(max(8.0, 1.6 * len(xlab) + 5.5), max(9.0, 0.52 * len(ylab) + 3.2)))
                    ax_b5b.scatter(
                        df_plot['X_Pos'],
                        df_plot['Y_Pos'],
                        s=p_sizes,
                        c=rgba_colors,
                        edgecolors='black',
                        linewidths=1.05
                    )
                    ax_b5b.set_xticks(range(len(xlab)))
                    ax_b5b.set_xticklabels(xlab, fontsize=13, fontweight='bold')
                    ax_b5b.set_yticks(range(len(ylab)))
                    ax_b5b.set_yticklabels(ylab, fontsize=10.8, fontweight='bold')
                    for marker_name, tick_label in zip(ylab, ax_b5b.get_yticklabels()):
                        tick_label.set_color(sub_colors.get(mk2sub.get(marker_name), 'black'))
                        tick_label.set_fontweight('bold')
                    bds_5b = []
                    csb_5b = mk2sub.get(ordered_markers[0]) if ordered_markers else None
                    for bi, marker_name in enumerate(ordered_markers):
                        if mk2sub.get(marker_name) != csb_5b:
                            bds_5b.append(bi - 0.5)
                            csb_5b = mk2sub.get(marker_name)
                    for b in bds_5b:
                        ax_b5b.axhline(y=b, color='gray', ls='--', alpha=0.45)
                    ax_b5b.set_xlim(-0.5, len(xlab) - 0.5)
                    ax_b5b.set_ylim(-0.5, len(ylab) - 0.5)
                    ax_b5b.invert_yaxis()
                    ax_b5b.grid(True, ls=':', alpha=0.55, axis='both')
                    ax_b5b.set_xlabel('Subtype model', fontsize=13, fontweight='bold')
                    ax_b5b.set_title(
                        f"Feature marker support: {lin}\n"
                        "Dot size = -log10(P-value) bin, alpha = relative ML importance, marker label color = assigned subtype",
                        fontsize=16,
                        fontweight='bold',
                        pad=16
                    )

                    size_handles = []
                    for level in range(1, 5):
                        if np.isclose(p_level_low, p_level_high):
                            label = f'Level {level}: {p_level_low:.2f}'
                        elif level == 1:
                            label = f'L{level}: <= {p_level_edges[level]:.2f}'
                        elif level == 4:
                            label = f'L{level}: >= {p_level_edges[level - 1]:.2f}'
                        else:
                            label = f'L{level}: {p_level_edges[level - 1]:.2f}-{p_level_edges[level]:.2f}'
                        size_handles.append(
                            mlines.Line2D(
                                [], [], color='w', marker='o',
                                markerfacecolor='lightgray', markeredgecolor='black',
                                markersize=np.sqrt(p_size_map[level]),
                                label=label
                            )
                        )
                    alpha_handles = [
                        mlines.Line2D(
                            [], [], color='w', marker='o',
                            markerfacecolor=(0.35, 0.35, 0.35, alpha),
                            markeredgecolor='black',
                            markersize=10,
                            label=label
                        )
                        for alpha, label in [(0.25, 'low'), (0.60, 'mid'), (1.00, 'high')]
                    ]
                    subtype_handles = [
                        mlines.Line2D(
                            [], [], linestyle='None', marker='',
                            color=sub_colors[st], label=st
                        )
                        for st in vsubs
                    ]
                    leg_size = ax_b5b.legend(
                        handles=size_handles, title='Dot size\n-log10(P-value) bin',
                        loc='upper left', bbox_to_anchor=(1.02, 1.00),
                        frameon=True, fontsize=10.0, title_fontsize=11.2,
                        borderaxespad=0.0, labelspacing=1.15
                    )
                    ax_b5b.add_artist(leg_size)
                    leg_alpha = ax_b5b.legend(
                        handles=alpha_handles, title='Alpha\nML importance',
                        loc='upper left', bbox_to_anchor=(1.02, 0.68),
                        frameon=True, fontsize=10.5, title_fontsize=11.5,
                        borderaxespad=0.0, labelspacing=1.0
                    )
                    ax_b5b.add_artist(leg_alpha)
                    ax_b5b.legend(
                        handles=subtype_handles, title='Marker label color\nSubtype',
                        loc='upper left', bbox_to_anchor=(1.02, 0.40),
                        frameon=True, fontsize=10.5, title_fontsize=11.5,
                        borderaxespad=0.0, labelspacing=0.8,
                        handlelength=0.0, handletextpad=0.2
                    )
                    for _txt in ax_b5b.get_legend().get_texts():
                        _label = _txt.get_text()
                        if _label in sub_colors:
                            _txt.set_color(sub_colors[_label])
                            _txt.set_fontweight('bold')
                    fig_b.subplots_adjust(left=0.28, right=0.76, top=0.88, bottom=0.10)
                    fig_b_path = os.path.join(FIG_DIR, f'Final_mv1_Step04B-05B_ImportanceAlpha_PvalueSize_{lin.replace(" ", "_")}.png')
                    if SAVE_STEP04B05B_IMPORTANCE_ALPHA_PVALUE:
                        _save_step04b_canvas_figure(
                            fig_b,
                            fig_b_path,
                            label=f'Step04B-05B importance-alpha/P-value-size {lin}',
                            show=True,
                            close=True
                        )
                        print(f"  🖼️ [Step 04B-5B] Importance-alpha/P-value-size dotplot saved: {fig_b_path}")
                    else:
                        plt.close(fig_b)

            if SAVE_STEP04B045_COMBINED_COMPOSITE and _combined_performance_payloads_45:
                _preferred_order_45 = ['B cells', 'Myeloid cells', 'Stromal cells', 'T cells']
                _composite_lins_45 = [
                    lin_name for lin_name in _preferred_order_45
                    if lin_name in _combined_performance_payloads_45
                ]
                _composite_lins_45 += [
                    lin_name for lin_name in valid_lins
                    if lin_name in _combined_performance_payloads_45 and lin_name not in _composite_lins_45
                ]
                _nrows_45 = len(_composite_lins_45)
                if _nrows_45 > 0:
                    fig_comp_45 = plt.figure(figsize=STEP04B_SUPPLEMENT_FIGSIZES['combined_performance'])
                    gs_comp_45 = gridspec.GridSpec(
                        _nrows_45 + 1,
                        2,
                        height_ratios=[1.0] * _nrows_45 + [0.30],
                        hspace=0.62,
                        wspace=0.30,
                    )
                    _legend_handles_45, _legend_labels_45 = [], []
                    _row_axes_45 = []

                    for _ri_45, _lin_45 in enumerate(_composite_lins_45):
                        _payload_45 = _combined_performance_payloads_45[_lin_45]
                        ax_b_comp_45 = fig_comp_45.add_subplot(gs_comp_45[_ri_45, 0])
                        ax_f_comp_45 = fig_comp_45.add_subplot(gs_comp_45[_ri_45, 1], sharex=ax_b_comp_45)
                        _handles_45, _labels_45 = _step04b_plot_history_pair(
                            ax_b_comp_45,
                            ax_f_comp_45,
                            _payload_45['history'],
                            _payload_45['is_new'],
                            _payload_45['b_ylim'],
                            _payload_45['f_ylim'],
                            b_title='',
                            f_title='',
                            show_ylabels=False,
                            title_fontsize=10.4,
                            label_fontsize=8.8,
                            tick_fontsize=7.6,
                            marker_size=3.5,
                            best_marker_size=4.7,
                            line_width=1.0,
                            best_line_width=1.8,
                            iteration_label_step=20,
                            iteration_tick_step=20,
                            right_xlim_pad=3,
                            ensemble_entry=_payload_45['selected_ensemble'],
                            ensemble_label=(
                                f'Selected: {MANUAL_SUBTYPE_COMBO}'
                                if _payload_45['selected_ensemble'] is not None
                                else 'Best ensemble (varies by iteration)'
                            ),
                        )
                        ax_b_comp_45.tick_params(axis='y', labelleft=True)
                        ax_f_comp_45.tick_params(axis='y', labelleft=True)
                        ax_b_comp_45.tick_params(axis='x', labelbottom=True)
                        ax_b_comp_45.set_xlabel('Iteration', fontsize=8.8, fontweight='bold', labelpad=4)
                        _row_axes_45.append((_lin_45, ax_b_comp_45, ax_f_comp_45))
                        if not _legend_handles_45:
                            _legend_handles_45, _legend_labels_45 = _handles_45, _labels_45

                    ax_lg_comp_45 = fig_comp_45.add_subplot(gs_comp_45[_nrows_45, :])
                    ax_lg_comp_45.axis('off')
                    ax_lg_comp_45.legend(
                        _legend_handles_45,
                        _legend_labels_45,
                        loc='lower center',
                        bbox_to_anchor=(0.5, 0.04),
                        ncol=min(6, max(1, len(_legend_labels_45))),
                        fontsize=9.4,
                        frameon=False,
                        columnspacing=1.05,
                        handlelength=1.7,
                        handletextpad=0.55,
                        borderaxespad=0.0,
                    )
                    # Individual lineage headings already clarify panel meaning; hide the overall title
                    # in manuscript-layout output and allocate the space above the axes to panel headings.
                    # fig_comp_45.suptitle("Subtype ensemble performance", ...)
                    fig_comp_45.subplots_adjust(
                        left=0.125,
                        right=0.985,
                        top=0.945,
                        bottom=0.020,
                        hspace=0.62,
                        wspace=0.30,
                    )
                    fig_comp_45.text(
                        0.335, 0.978, 'Balanced accuracy (B-Acc)',
                        ha='center', va='top', fontsize=12.0, fontweight='bold',
                    )
                    fig_comp_45.text(
                        0.775, 0.978, 'Macro F1',
                        ha='center', va='top', fontsize=12.0, fontweight='bold',
                    )
                    for _lin_row_45, _ax_b_row_45, _ax_f_row_45 in _row_axes_45:
                        _row_box_45 = _ax_b_row_45.get_position()
                        fig_comp_45.text(
                            0.018,
                            (_row_box_45.y0 + _row_box_45.y1) / 2,
                            _lin_row_45,
                            ha='center', va='center', rotation=90,
                            fontsize=11.0, fontweight='bold',
                        )
                    _combined_composite_path_45 = os.path.join(
                        FIG_DIR,
                        'Final_mv1_Step04B-04-5_Combined_Composite.png'
                    )
                    _save_step04b_canvas_figure(
                        fig_comp_45,
                        _combined_composite_path_45,
                        label='Step04B-04-5 combined composite',
                        show=True,
                        close=True,
                    )

    # ── Run prediction ensembles (apply a unified model to all lineages) ──
    # As in Step 02-3.5, uniformly apply one user-specified combination across all lineages
    target_combo_sub = MANUAL_SUBTYPE_COMBO if MANUAL_SUBTYPE_COMBO.strip() != '' else ''
    if target_combo_sub == '':
        # Automatic selection: use the most frequent best_combo across lineages as the unified model
        _best_combos = [all_sub_probs_global[l]['best_combo'] for l in lineages if l in all_sub_probs_global]
        if _best_combos:
            from collections import Counter
            target_combo_sub = Counter(_best_combos).most_common(1)[0][0]
        else:
            target_combo_sub = '+'.join(MODEL_KEYS)
    
    print(f"  ⭐ Unified Subtype model across all lineages: {target_combo_sub} (Iter: {MANUAL_SUBTYPE_ITER})")
    _subtype_transfer_root = os.path.join(V9_TRANSFER_ROOT, 'subtype')
    if STEP04_FIGURE_ONLY:
        print("  ℹ️ Figure-only mode → skipping the frozen subtype-package selection manifest update.")
    elif os.path.isdir(_subtype_transfer_root):
        from V9_transfer_utils import set_transfer_selection, transfer_stage_dir
        for _lin_transfer in all_sub_probs_global:
            _stage_dir_transfer = transfer_stage_dir(
                V9_TRANSFER_ROOT, 'subtype', _lin_transfer
            )
            if os.path.exists(os.path.join(_stage_dir_transfer, 'manifest.json')):
                set_transfer_selection(
                    _stage_dir_transfer,
                    target_combo_sub,
                    MANUAL_SUBTYPE_ITER,
                )
        print(
            "  ✅ Final frozen subtype-package selection recorded: "
            f"{target_combo_sub}, Iter={MANUAL_SUBTYPE_ITER}"
        )
    if not apply_step04_predictions:
        print("  ℹ️ apply_step04_predictions=False → retaining existing Step04B labels and skipping prediction reapplication.")
    
    for lin in (lineages if apply_step04_predictions else []):
        if lin not in all_sub_probs_global:
            continue
            
        probsData = all_sub_probs_global[lin]['probs']
        classes   = all_sub_probs_global[lin]['classes']
        
        print(f"  👉 [{lin}] Applied combination: {target_combo_sub} (Iter: {MANUAL_SUBTYPE_ITER})")
        
        # Safely recreate dlbcl_mask_4 because it may not exist
        if 'dlbcl_mask_4' not in globals() and 'dlbcl_mask_4' not in locals():
            dlbcl_mask_4 = adata.obs['Disease'].str.contains('DLBCL', na=False)
        
        idx_dlbcl = np.where(dlbcl_mask_4 & (adata.obs['ML_Lineage_Raw'] == lin))[0]
        if len(idx_dlbcl) == 0:
            continue

        le_tmp = LabelEncoder()
        le_tmp.classes_ = classes

        combo_keys = [k.strip() for k in target_combo_sub.split('+')] if target_combo_sub else list(MODEL_KEYS)
        # Automatically remove any keys absent from MODEL_KEYS (e.g., 'Rescue')
        combo_keys = [k for k in combo_keys if k in MODEL_KEYS]
        if not combo_keys:
            print(f"  ⚠️ [{lin}] No valid model keys; falling back to all models ({MODEL_KEYS}).")
            combo_keys = list(MODEL_KEYS)
        
        combined_probs = []
        for k in combo_keys:
            combined_probs.extend(probsData[k][:MANUAL_SUBTYPE_ITER])
            
        if combined_probs:
            avg_sub = np.mean(combined_probs, axis=0)
            pred_sub = le_tmp.inverse_transform(np.argmax(avg_sub, axis=1))
            conf_sub = np.max(avg_sub, axis=1)

            for c in ['ML_Subtype_Raw', 'ML_Subtype_Pure', 'ML_Subtype_Conf']:
                if c not in adata.obs.columns:
                    adata.obs[c] = 'na'

            # Safely map using .loc
            _mask_sub = np.zeros(len(adata), dtype=bool)
            _mask_sub[idx_dlbcl] = True
            adata.obs.loc[_mask_sub, 'ML_Subtype_Raw']  = pred_sub.astype(str)
            adata.obs.loc[_mask_sub, 'ML_Subtype_Pure'] = pred_sub.astype(str)
            adata.obs.loc[_mask_sub, 'ML_Subtype_Conf'] = conf_sub
            
    if apply_step04_predictions and 'ML_Subtype_Conf' in adata.obs.columns:
        adata.obs['ML_Subtype_Conf'] = pd.to_numeric(adata.obs['ML_Subtype_Conf'], errors='coerce')
    
    if apply_step04_predictions:
        print(f"\n✅ Lineage-wise manual ensemble application completed! (unified model: {target_combo_sub})")
        print(adata.obs['ML_Subtype_Raw'].value_counts())

# %%
# ==============================================================================
# [Step 04B-6] Save the final checkpoint (before visualization)
# ==============================================================================
# Explicitly skip checkpoint/FI saving in Figure-only mode.
# Save adata and feature-importance records only in training/prediction-application mode.
if STEP04_FIGURE_ONLY:
    print("⏩ [Step 04B-6] Figure-only mode → skipping checkpoint/FI saving")
elif 'all_sub_probs_global' in globals() and apply_step04_predictions:
    _best_combo_by_lineage = {
        lin: all_sub_probs_global[lin].get('best_combo', 'na')
        for lin in all_sub_probs_global
    }
    _best_score_by_lineage = {}
    _actual_iters_by_lineage = {}
    if 'history_sub' in globals():
        for lin, h in history_sub.items():
            _best_vals = h.get('best', {}).get('avg', [])
            _iter_vals = h.get('iters', [])
            _best_score_by_lineage[lin] = float(max(_best_vals)) if _best_vals else np.nan
            _actual_iters_by_lineage[lin] = int(max(_iter_vals)) if _iter_vals else 0

    step04_selection_manifest = {
        'step': 'Step04B',
        'selection_mode': 'manual' if MANUAL_SUBTYPE_COMBO.strip() != '' else 'lineage_best_mode',
        'manual_combo': MANUAL_SUBTYPE_COMBO,
        'applied_combo': target_combo_sub if 'target_combo_sub' in locals() else 'na',
        'manual_iter': int(MANUAL_SUBTYPE_ITER) if 'MANUAL_SUBTYPE_ITER' in locals() else -1,
        'best_val_combo_by_lineage': _best_combo_by_lineage,
        'best_val_score_avg_by_lineage': _best_score_by_lineage,
        'actual_iters_by_lineage': _actual_iters_by_lineage,
        'model_keys': MODEL_KEYS,
        'input_checkpoint': CHECKPOINT_STEP02,
        'probability_log': PROBS_STEP04_PATH,
        'training_log_path': TRAIN_LOG_STEP04_PATH,
        'per_subtype_metric_log_path': PER_SUBTYPE_METRIC_STEP04_PATH,
        'oob_eval_every': int(OOB_EVAL_EVERY_STEP04),
        'oob_eval_extra_iters': sorted([int(x) for x in OOB_EVAL_EXTRA_ITERS_STEP04]),
        'exclude_cd4ctl_from_main_tcell_model': bool(EXCLUDE_CD4CTL_FROM_MAIN_TCELL_MODEL),
        'main_excluded_subtypes': {k: ','.join(v) for k, v in STEP04_MAIN_EXCLUDED_SUBTYPES.items()},
        'tpr_gmm_policy_mode': STEP04B_TPR_GMM_MODE,
        'tpr_gmm_policy_cd4_minus_cd8_min': float(STEP04B_TPR_GMM_CD4_MINUS_CD8_MIN),
        'tpr_gmm_policy': adata.uns.get('step04B_tpr_gmm_policy', {}),
        'frozen_transfer_root': V9_TRANSFER_ROOT
    }
    adata.uns['step04B_model_selection'] = step04_selection_manifest
    _step04b_write_checkpoint_if_enabled(adata, CHECKPOINT_STEP04)
    print(f"✅ [Step 04B-6] Final checkpoint saved: {CHECKPOINT_STEP04}")

    # Save the full Feature Importance records
    fi_save_path = os.path.join(CKPT_DIR, 'mv1_feature_importance_04B.pkl')
    fi_save = {
        'broad': fi_log_broad if 'fi_log_broad' in globals() else {},
        'subtype': fi_log_subtype if 'fi_log_subtype' in globals() else {}
    }
    _step04b_dump_if_enabled(fi_save, fi_save_path)
    print(f"✅ Full Feature Importance records saved: {fi_save_path}")

# %%
# ==============================================================================
# [Step 04B-7] Visualize lineage-wise Subtype model performance validation
# ==============================================================================
def _build_step04b07_validation_df(
    training_log_path,
    lineage_order,
    selected_combo,
    selected_iter,
):
    """Restore the OOB performance table for the selected subtype ensemble/iteration from the exact log rows."""
    selected_combo = str(selected_combo).strip()
    if not selected_combo:
        raise ValueError('Step04B-07 selected_combo must be non-empty.')
    try:
        selected_iter = int(selected_iter)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid Step04B-07 selected_iter: {selected_iter!r}') from exc

    required_cols = ['lineage', 'iter', 'eval_type', 'name', 'bacc', 'f1']
    if not os.path.exists(training_log_path):
        raise FileNotFoundError(f'Step04B-07 training log not found: {training_log_path}')
    try:
        log_df = pd.read_csv(training_log_path, usecols=required_cols)
    except ValueError as exc:
        raise RuntimeError(
            f'Step04B-07 training log is missing required columns {required_cols}: '
            f'{training_log_path}'
        ) from exc

    log_df['iter'] = pd.to_numeric(log_df['iter'], errors='coerce')
    selected_df = log_df[
        log_df['eval_type'].eq('ensemble_combo')
        & log_df['name'].eq(selected_combo)
        & log_df['iter'].eq(float(selected_iter))
    ].copy()

    row_counts = {
        lin: int(selected_df['lineage'].eq(lin).sum())
        for lin in lineage_order
    }
    invalid_counts = {lin: count for lin, count in row_counts.items() if count != 1}
    if invalid_counts:
        raise RuntimeError(
            'Step04B-07 expected exactly one selected ensemble row per lineage for '
            f'combo={selected_combo!r}, iter={selected_iter}; counts={invalid_counts}; '
            f'log={training_log_path}'
        )

    rows = []
    for lin in lineage_order:
        selected_row = selected_df.loc[selected_df['lineage'].eq(lin)].iloc[0]
        bacc = pd.to_numeric(selected_row['bacc'], errors='coerce')
        f1_val = pd.to_numeric(selected_row['f1'], errors='coerce')
        if not (np.isfinite(bacc) and np.isfinite(f1_val)):
            raise RuntimeError(
                'Step04B-07 selected metric is non-finite for '
                f'lineage={lin!r}, combo={selected_combo!r}, iter={selected_iter}'
            )
        rows.append({
            'Lineage': lin,
            'Combo': selected_combo,
            'Metric': 'Balanced Acc',
            'Score': float(bacc),
        })
        rows.append({
            'Lineage': lin,
            'Combo': selected_combo,
            'Metric': 'Macro F1',
            'Score': float(f1_val),
        })

    return pd.DataFrame(rows, columns=['Lineage', 'Combo', 'Metric', 'Score'])


_STEP04B07_LINEAGE_ORDER = ['T cells', 'B cells', 'Myeloid cells', 'Stromal cells']
_step04b07_combo = (
    target_combo_sub if ('target_combo_sub' in globals() or 'target_combo_sub' in locals())
    else MANUAL_SUBTYPE_COMBO
)
_step04b07_combo = str(_step04b07_combo).strip() or str(MANUAL_SUBTYPE_COMBO).strip()
_step04b07_iter = int(MANUAL_SUBTYPE_ITER)
df_val = _build_step04b07_validation_df(
    TRAIN_LOG_STEP04_PATH,
    _STEP04B07_LINEAGE_ORDER,
    selected_combo=_step04b07_combo,
    selected_iter=_step04b07_iter,
)
_has_step04b07_df = True
print(
    '  ℹ️ [Step 04B-7] selected ensemble OOB log restoration: '
    f'{_step04b07_combo}, iter={_step04b07_iter}, rows={len(df_val)}'
)

if _has_step04b07_df:
    print("🔹 [Step 04B-7] Visualizing lineage-wise subtype model performance validation...")

    if not df_val.empty:
        df_plot = df_val.copy()

        fig_val = plt.figure(figsize=STEP04B_SUPPLEMENT_FIGSIZES['subtype_validation'])
        ax = fig_val.add_axes([0.080, 0.205, 0.900, 0.635])
        unique_lineages = df_plot['Lineage'].unique()
        n_lin = len(unique_lineages)

        ax = sns.barplot(data=df_plot, x='Lineage', y='Score', hue='Metric',
                         hue_order=['Balanced Acc', 'Macro F1'],
                         palette={'Balanced Acc': 'white', 'Macro F1': 'white'}, edgecolor='black',
                         ax=ax)
        _auto_legend = ax.get_legend()
        if _auto_legend is not None:
            _auto_legend.remove()

        for idx_b, bar in enumerate(ax.patches):
            lin_idx = idx_b % n_lin
            met_idx = idx_b // n_lin
            lin_name = unique_lineages[lin_idx]
            base_col = LINEAGE_COLORS.get(lin_name, '#808080')

            if met_idx == 0:
                bar.set_facecolor(base_col); bar.set_alpha(0.9)
            else:
                bar.set_facecolor(base_col); bar.set_alpha(0.4); bar.set_hatch('///')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#555555', alpha=0.9, edgecolor='black', label='Balanced Acc'),
            Patch(facecolor='#555555', alpha=0.4, hatch='///', edgecolor='black', label='Macro F1')
        ]
        fig_val.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.965),
            ncol=2,
            fontsize=10.2,
            frameon=False,
            borderpad=0.30,
            handlelength=1.7,
            labelspacing=0.35,
            columnspacing=1.25
        )

        ax.set_ylim(0.5, 1.045)
        ax.set_ylabel("Reference OOB score", fontsize=11.5, fontweight='bold')
        ax.set_xlabel(
            f"Lineage and selected ensemble ({_step04b07_iter} iterations)",
            fontsize=11.5,
            fontweight='bold',
        )
        ax.tick_params(axis='both', labelsize=10.5)
        for _tick_label in ax.get_yticklabels():
            _tick_label.set_fontweight('bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Show the unified model combination (Combo) in X-axis labels
        current_labels = [label.get_text() for label in ax.get_xticklabels()]
        new_labels = []
        for l_name in current_labels:
            if 'Combo' in df_plot.columns and len(df_plot[df_plot['Lineage'] == l_name]) > 0:
                c_name = df_plot[df_plot['Lineage'] == l_name]['Combo'].iloc[0]
                new_labels.append(f"{l_name}\n[{c_name}]")
            else:
                new_labels.append(l_name)
        ax.set_xticklabels(new_labels, fontweight='bold')

        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', xytext=(0, 3), textcoords='offset points',
                            fontsize=8.5, fontweight='bold')

        fig_path_val = os.path.join(FIG_DIR, 'Final_mv1_Step04B-07_Subtype_Validation_Performance.png')
        if SAVE_STEP04B07_SUBTYPE_VALIDATION:
            _save_step04b_canvas_figure(
                fig_val,
                fig_path_val,
                label='Step04B-07 subtype validation performance',
                show=True,
                close=True
            )
        else:
            plt.close(fig_val)
    else:
        print("  ⚠️ [Step 04B-7] Selected ensemble OOB performance records are empty; no figure generated.")
else:
    print("  ⚠️ [Step 04B-7] Could not restore the selected ensemble OOB performance table.")

# %%
# ==============================================================================
# [Step 04B-8] DLBCL Test Set: generate and benchmark Subtype predictions by model (all lineages)
# Evaluation targets: DLBCL patient data (Test Set) — fully separate from Reference training data
# ==============================================================================
# Always run (if predictions already exist in adata.obs, compute scores and visualize only)
print('🔹 [Step 04B-08] Generating and benchmarking model-specific predictions for the DLBCL Test Set...')
import rapids_singlecell as rsc  # type: ignore
import scanpy as sc
import cupy as cp
from itertools import combinations as _combs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm.auto import tqdm

if not adata.obs_names.is_unique:
    adata.obs_names_make_unique()

# ── Generate the list of 15 ML combinations (matching the latest version without 'Cat') ──
_keys = MODEL_KEYS  # ['RF', 'XGB', 'ENet', 'MLP']
all_combos = []
for r in range(1, len(_keys) + 1):
    for combo in _combs(_keys, r):
        all_combos.append(combo)
combo_names = ['+'.join(c) for c in all_combos]  # 15 combinations

# Define benchmark columns (2 Baseline + 15 ML)
benchmark_cols = ['Pred_Clustering', 'Pred_SC_Centroid'] + [f'Pred_ML_Sub_{i}' for i in range(1, 16)]
benchmark_labels = ['Clustering', 'SC-Centroid'] + [f'ML ({n})' for n in combo_names]

# Check whether all predictions already exist.
# In Figure-only mode, do not compute missing columns. Update the checkpoint first if any are missing.
missing_benchmark_cols = [col for col in benchmark_cols if col not in adata.obs.columns]
skip_prediction = len(missing_benchmark_cols) == 0
if STEP04_FIGURE_ONLY and not skip_prediction:
    raise RuntimeError(
        "STEP04_FIGURE_ONLY=True does not compute new DLBCL subtype benchmark predictions. "
        f"Missing columns: {missing_benchmark_cols}. Update the checkpoint in Step04B analysis mode first."
    )
if STEP04_FIGURE_ONLY:
    print("  ✅ Figure-only: benchmark columns verified in the checkpoint → generating figures without recomputing predictions.")

if not skip_prediction:
    for col in benchmark_cols:
        if col not in adata.obs.columns:
            adata.obs[col] = 'na'
        else:
            adata.obs[col] = adata.obs[col].astype(str)

target_lineages = ['T cells', 'B cells', 'Myeloid cells', 'Stromal cells', 'NK cells']
X_full_08 = (
    None
    if skip_prediction
    else (adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X))
)
test_results = []

for lin in tqdm(target_lineages, desc="Benchmarking Lineages"):
    print(f"\n▶️ [{lin}] Running the prediction pipeline...")
    
    mask_ref = (adata.obs['Modeling_Broad'] == lin) & \
               (~adata.obs['Disease'].str.contains('DLBCL', na=False)) & \
               (~np.isin(adata.obs['GT_Subtype'], ['na', 'Unknown', 'nan', '']))
    excluded_eval_subtypes = _step04_excluded_subtypes(lin)
    if excluded_eval_subtypes:
        mask_ref &= ~adata.obs['GT_Subtype'].astype(str).isin(excluded_eval_subtypes)
    idx_ref = np.where(mask_ref)[0]
    
    mask_dlbcl = (adata.obs['ML_Lineage_Raw'] == lin) & (adata.obs['Disease'].str.contains('DLBCL', na=False))
    idx_dlbcl = np.where(mask_dlbcl)[0]

    if len(idx_ref) == 0 or len(idx_dlbcl) == 0:
        continue

    classes = np.unique(adata.obs['GT_Subtype'].iloc[idx_ref].values)
    if len(classes) <= 1:
        if len(classes) == 1 and not skip_prediction:
            target_names = adata.obs_names[idx_dlbcl]
            for col in benchmark_cols:
                adata.obs.loc[target_names, col] = classes[0]
        continue

    if not skip_prediction:
        # ─── A. Clustering (Leiden)-based prediction ───
        adata_dlbcl_lin = adata[idx_dlbcl].copy()
        try:
            rsc.get.anndata_to_GPU(adata_dlbcl_lin)
            rsc.pp.neighbors(adata_dlbcl_lin, n_neighbors=15)
            rsc.tl.leiden(adata_dlbcl_lin, resolution=2.0, key_added='leiden_dlbcl')
            rsc.get.anndata_to_CPU(adata_dlbcl_lin)
        except Exception:
            sc.pp.neighbors(adata_dlbcl_lin, n_neighbors=15)
            sc.tl.leiden(adata_dlbcl_lin, resolution=2.0, key_added='leiden_dlbcl')

        scaler = StandardScaler()
        X_ref_sc = scaler.fit_transform(X_full_08[idx_ref])
        X_dlbcl_sc = scaler.transform(X_full_08[idx_dlbcl])

        centroids_ref = {c: cp.array(X_ref_sc[adata.obs['GT_Subtype'].iloc[idx_ref].values == c].mean(axis=0)) 
                         for c in classes if (adata.obs['GT_Subtype'].iloc[idx_ref].values == c).any()}
                
        cmap_dlbcl = {}
        for c in adata_dlbcl_lin.obs['leiden_dlbcl'].unique():
            c_mask = (adata_dlbcl_lin.obs['leiden_dlbcl'].values == c)
            c_cent = cp.array(X_dlbcl_sc[c_mask].mean(axis=0))
            sims = [cp.dot(c_cent, centroids_ref[k]) / (cp.linalg.norm(c_cent) * cp.linalg.norm(centroids_ref[k]) + 1e-8) for k in classes]
            cmap_dlbcl[c] = classes[cp.argmax(cp.array(sims)).get()]
            
        target_names = adata.obs_names[idx_dlbcl]
        adata.obs.loc[target_names, 'Pred_Clustering'] = adata_dlbcl_lin.obs['leiden_dlbcl'].map(cmap_dlbcl).values

        # ─── B. SC-Centroid-based prediction ───
        sc_labels_list = list(centroids_ref.keys())
        sc_mat_norm = np.array([centroids_ref[l].get() for l in sc_labels_list])
        sc_mat_norm /= (np.linalg.norm(sc_mat_norm, axis=1, keepdims=True) + 1e-8)
        X_dl_norm = X_dlbcl_sc / (np.linalg.norm(X_dlbcl_sc, axis=1, keepdims=True) + 1e-8)
        sc_preds = [sc_labels_list[i] for i in (X_dl_norm @ sc_mat_norm.T).argmax(axis=1)]
        adata.obs.loc[target_names, 'Pred_SC_Centroid'] = sc_preds
        
        # ─── C. Derive ensembles for 15 ML combinations (combine Probabilities) ───
        if 'all_sub_probs_global' in globals() and lin in all_sub_probs_global:  # type: ignore
            probsData = all_sub_probs_global[lin]['probs']  # type: ignore
            le_tmp = LabelEncoder()
            le_tmp.classes_ = all_sub_probs_global[lin]['classes']  # type: ignore
            
            for m_idx, combo in enumerate(all_combos, 1):
                combined_probs = []
                for k in combo:
                    combined_probs.extend(probsData[k][:MANUAL_SUBTYPE_ITER])
                if combined_probs:
                    avg_sub = np.mean(combined_probs, axis=0)
                    pred_sub = le_tmp.inverse_transform(np.argmax(avg_sub, axis=1))
                    adata.obs.loc[target_names, f'Pred_ML_Sub_{m_idx}'] = pred_sub
        cp.get_default_memory_pool().free_all_blocks()
        
    # ── Performance evaluation (DLBCL Test Set) ──
    # Restrict evaluation to cells truly belonging to the lineage according to GT (Modeling_Broad == lin)
    # so subtype performance is not distorted by misclassification in the preceding Broad Lineage step
    mask_eval = mask_dlbcl & (~np.isin(adata.obs['GT_Subtype'], ['na', 'Unknown', 'nan', ''])) & (adata.obs['Modeling_Broad'] == lin)
    if excluded_eval_subtypes:
        mask_eval &= ~adata.obs['GT_Subtype'].astype(str).isin(excluded_eval_subtypes)
    idx_eval = np.where(mask_eval)[0]
    if len(idx_eval) > 0:
        y_true_eval = adata.obs['GT_Subtype'].iloc[idx_eval].astype(str).values
        existing_labels = np.unique(y_true_eval)

        for m_col, m_name in zip(benchmark_cols, benchmark_labels):
            y_pred_eval = adata.obs[m_col].iloc[idx_eval].astype(str).values
            test_results.append({
                'Lineage': lin, 'Model': m_name,
                'Balanced Acc': balanced_accuracy_score(y_true_eval, y_pred_eval),
                'Macro F1': f1_score(y_true_eval, y_pred_eval, average='macro', zero_division=0, labels=existing_labels),
                'Weighted F1': f1_score(y_true_eval, y_pred_eval, average='weighted', zero_division=0)
            })

        # 🌟 Print diagnostic logs for all lineages (expanded from B cells only)
        print(f"\n  📊 [{lin} diagnostics] GT class distribution:")
        unique_gt, counts_gt = np.unique(y_true_eval, return_counts=True)
        for c, n in zip(unique_gt, counts_gt):
            print(f"     GT '{c}': {n:,} cells")
            
        # Also inspect the prediction distribution of the final ensemble model (No. 15) using the unified model
        y_pred_final = adata.obs['Pred_ML_Sub_15'].iloc[idx_eval].astype(str).values
        unique_pred, counts_pred = np.unique(y_pred_final, return_counts=True)
        print(f"  📊 [{lin} diagnostics] ML (Proposed) prediction distribution:")
        for c, n in zip(unique_pred, counts_pred):
            print(f"     Pred '{c}': {n:,} cells")

    # ==============================================================================
# Visualization (compact manuscript format — representative models including the best model) and model-ranking analysis
# ==============================================================================
df_perf = pd.DataFrame(test_results)
if not df_perf.empty:
    # Add the evaluation-criterion variable (mean of 3 metrics)
    df_perf['Avg_Score'] = df_perf[['Balanced Acc', 'Macro F1', 'Weighted F1']].mean(axis=1)

    # ── Panel 1: print the table of all 17 models (for console inspection) ──
    print("\n📊 [Step 04B-08] All benchmark results (DLBCL Test Set):")
    with pd.option_context('display.max_rows', None, 'display.float_format', '{:.3f}'.format):
        pivot = df_perf.pivot_table(index='Model', columns='Lineage', 
                                    values=['Balanced Acc', 'Macro F1', 'Weighted F1'])
        print(pivot.to_string())

    # ── Panel 2: Top 10 model rankings per lineage ──
    print("\n🏆 [Top 10 model rankings per lineage (criterion: mean of 3 performance metrics)]")
    for lin in df_perf['Lineage'].unique():
        df_lin = df_perf[df_perf['Lineage'] == lin].sort_values(by='Avg_Score', ascending=False).head(10)
        print(f"\n🔹 {lin} Top 10:")
        for rank, (_, row) in enumerate(df_lin.iterrows(), 1):
            print(f"  Rank {rank:2d}: {row['Model']:<30} | Avg: {row['Avg_Score']*100:.1f}% (B-Acc: {row['Balanced Acc']*100:.1f}%, M-F1: {row['Macro F1']*100:.1f}%)")

    # ── Panel 3: overall Top 10 model rankings ──
    print("\n🥇 [Overall Top 10 model rankings (mean across all lineages)]")
    df_overall = df_perf.groupby('Model')[['Balanced Acc', 'Macro F1', 'Weighted F1']].mean().reset_index()
    df_overall['Avg_Score'] = df_overall[['Balanced Acc', 'Macro F1', 'Weighted F1']].mean(axis=1)
    df_overall = df_overall.sort_values(by='Avg_Score', ascending=False).head(10)
        
    for rank, (_, row) in enumerate(df_overall.iterrows(), 1):
        print(f"  Rank {rank:2d}: {row['Model']:<30} | Avg: {row['Avg_Score']*100:.1f}% (B-Acc: {row['Balanced Acc']*100:.1f}%, M-F1: {row['Macro F1']*100:.1f}%)")

    # ── Panel 4: compact manuscript bar chart (representative models + user-selected Best ensemble) ──
    # Identify the model name selected by the user in the preceding Step
    best_manual_name = f"ML ({target_combo_sub})" if 'target_combo_sub' in locals() or 'target_combo_sub' in globals() else benchmark_labels[-1]
        
    # List models to display in the plot (always include the Best model)
    repr_models = [
        'Clustering', 'SC-Centroid', 
        'ML (RF)', 'ML (XGB)', 'ML (ENet)', 'ML (MLP)',
        best_manual_name
    ]
    # Remove duplicates and finalize the model list
    repr_models = list(dict.fromkeys(repr_models))
    df_repr = df_perf[df_perf['Model'].isin(repr_models)].copy()
        
    # Convert performance metrics to percentages (0~1 -> 0~100)
    met_cols = ['Balanced Acc', 'Macro F1', 'Weighted F1']
    for col in met_cols:
        df_repr[col] = df_repr[col] * 100
        
    if not df_repr.empty:
        # Highlight the best model in Dark Red; use the default palette for the others
        comp_palette = {
            'Clustering': '#BDC3C7', 'SC-Centroid': '#5DADE2',
            'ML (RF)': '#F1C40F', 'ML (XGB)': '#E67E22',
            'ML (ENet)': '#E74C3C', 'ML (MLP)': '#F1948A'
        }
        if best_manual_name not in comp_palette:
            comp_palette[best_manual_name] = '#943126' # Highlight color
            
        # A coordinate view replaces the dense 84-value heatmap. Each lineage
        # gets one panel: x = balanced accuracy, y = macro F1, and marker area
        # = weighted F1. The selected ensemble keeps its exact three scores.
        import matplotlib.lines as mlines
        from matplotlib.legend_handler import HandlerTuple
        from matplotlib.ticker import MaxNLocator

        benchmark_element_scale = float(STEP04B_BENCHMARK_ELEMENT_SCALE)
        benchmark_marker_area_scale = benchmark_element_scale ** 2
        fig = plt.figure(figsize=STEP04B_BENCHMARK_FIGSIZE)
        gs_bench = fig.add_gridspec(
            1, 4,
            left=0.065, right=0.985, bottom=0.300, top=0.780,
            wspace=0.22,
        )
        axes = [fig.add_subplot(gs_bench[0, i]) for i in range(4)]
        lineage_order_bench = [
            lin_name for lin_name in ['T cells', 'B cells', 'Myeloid cells', 'Stromal cells']
            if lin_name in set(df_repr['Lineage'])
        ]

        def _benchmark_marker_area(weighted_f1):
            # Exponential zoom in the high-score range makes even a 0.1-point
            # Weighted F1 difference visible while retaining a compact floor
            # for lower-performing baselines.
            weighted = float(np.clip(weighted_f1, 80.0, 100.0))
            base_area = 118.0 + 2.05 * ((weighted - 80.0) ** 1.72)
            return float(base_area * benchmark_marker_area_scale)

        marker_for_model = {model_name: 'o' for model_name in repr_models}

        for panel_idx, (ax, lineage_name) in enumerate(zip(axes, lineage_order_bench)):
            lineage_df = (
                df_repr[df_repr['Lineage'] == lineage_name]
                .set_index('Model')
                .reindex(repr_models)
                .dropna(subset=met_cols)
                .reset_index()
            )
            x_values = lineage_df['Balanced Acc'].to_numpy(dtype=float)
            y_values = lineage_df['Macro F1'].to_numpy(dtype=float)
            axis_min_x = max(30.0, np.floor((np.nanmin(x_values) - 5.0) / 5.0) * 5.0)
            axis_min_y = max(30.0, np.floor((np.nanmin(y_values) - 5.0) / 5.0) * 5.0)
            axis_min = min(axis_min_x, axis_min_y)
            axis_max = min(100.8, max(np.nanmax(x_values), np.nanmax(y_values)) + 4.0)

            ax.plot(
                [axis_min, axis_max], [axis_min, axis_max],
                color='#A8A8A8', linestyle='--',
                linewidth=1.45 * benchmark_element_scale, zorder=0,
            )
            for _, score_row in lineage_df.iterrows():
                model_name = score_row['Model']
                is_selected = model_name == best_manual_name
                ax.scatter(
                    score_row['Balanced Acc'], score_row['Macro F1'],
                    s=_benchmark_marker_area(score_row['Weighted F1']),
                    marker=marker_for_model[model_name],
                    color=comp_palette[model_name],
                    edgecolor='#111111',
                    linewidth=(2.4 if is_selected else 1.25) * benchmark_element_scale,
                    alpha=0.96,
                    zorder=4 if is_selected else 3,
                )
                if is_selected:
                    # Selected ensemble remains a circle for direct size
                    # comparison; the internal star provides selection status.
                    ax.scatter(
                        score_row['Balanced Acc'], score_row['Macro F1'],
                        s=max(
                            64.0 * benchmark_marker_area_scale,
                            _benchmark_marker_area(score_row['Weighted F1']) * 0.31,
                        ),
                        marker='*', facecolor='white', edgecolor='#111111',
                        linewidth=0.80 * benchmark_element_scale, zorder=5,
                    )

            ax.set_xlim(axis_min_x, axis_max)
            ax.set_ylim(axis_min_y, axis_max)
            ax.set_title(
                lineage_name,
                fontsize=19.0 * benchmark_element_scale,
                fontweight='bold', pad=9,
            )
            ax.set_xlabel(
                'Balanced accuracy (%)',
                fontsize=15.5 * benchmark_element_scale,
                fontweight='bold', labelpad=8,
            )
            ax.set_ylabel(
                'Macro F1-score (%)' if panel_idx == 0 else '',
                fontsize=15.5 * benchmark_element_scale,
                fontweight='bold', labelpad=9,
            )
            x_tick_max = min(100.0, np.floor(axis_max))
            y_tick_max = min(100.0, np.floor(axis_max))
            ax.set_xticks(np.unique(np.rint(np.linspace(axis_min_x, x_tick_max, 4)).astype(int)))
            ax.set_yticks(np.unique(np.rint(np.linspace(axis_min_y, y_tick_max, 4)).astype(int)))
            ax.tick_params(
                axis='both',
                labelsize=13.7 * benchmark_element_scale,
                width=1.25 * benchmark_element_scale,
                length=4.5 * benchmark_element_scale,
            )
            for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
                tick_label.set_fontweight('bold')
            ax.grid(
                True, linestyle='--',
                linewidth=0.9 * benchmark_element_scale, alpha=0.34,
            )
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.25 * benchmark_element_scale)

        fig.suptitle(
            'DLBCL subtype benchmark',
            x=0.52, y=0.950,
            fontsize=26.0 * benchmark_element_scale,
            fontweight='bold',
        )

        method_handles = []
        for model_name in repr_models:
            outer = mlines.Line2D(
                [], [], linestyle='none', marker=marker_for_model[model_name],
                markersize=(12.6 if model_name == best_manual_name else 10.8) * benchmark_element_scale,
                markerfacecolor=comp_palette[model_name], markeredgecolor='#111111',
                markeredgewidth=(1.2 if model_name == best_manual_name else 0.8) * benchmark_element_scale,
            )
            if model_name == best_manual_name:
                inner = mlines.Line2D(
                    [], [], linestyle='none', marker='*',
                    markersize=8.2 * benchmark_element_scale,
                    markerfacecolor='white', markeredgecolor='#111111',
                    markeredgewidth=0.45 * benchmark_element_scale,
                )
                method_handles.append((outer, inner))
            else:
                method_handles.append(outer)

        weighted_examples = [92.5, 95.0, 97.5, 100.0]
        weighted_handles = [
            plt.scatter([], [], s=_benchmark_marker_area(value), marker='o',
                        facecolor='#B8B8B8', edgecolor='#333333',
                        linewidth=0.8 * benchmark_element_scale,
                        label=f'{value}%')
            for value in weighted_examples
        ]
        legend_labels = [
            model_name
            for model_name in repr_models
        ]
        equality_handle = mlines.Line2D(
            [], [], color='#A8A8A8', linestyle='--',
            linewidth=1.6 * benchmark_element_scale,
            label='BAcc = Macro F1'
        )
        method_legend_handles = method_handles + [equality_handle]
        method_legend_labels = legend_labels + ['BAcc = Macro F1']
        fig.legend(
            handles=method_legend_handles, labels=method_legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
            loc='lower center', bbox_to_anchor=(0.50, 0.125),
            ncol=len(method_legend_handles), frameon=False,
            fontsize=13.0 * benchmark_element_scale,
            handletextpad=0.52, columnspacing=1.00,
        )
        fig.legend(
            handles=weighted_handles,
            labels=[f'{value:g}%' for value in weighted_examples],
            title='Marker area = Weighted F1',
            loc='lower center', bbox_to_anchor=(0.50, 0.020),
            ncol=len(weighted_handles), frameon=False,
            fontsize=12.5 * benchmark_element_scale,
            title_fontsize=13.0 * benchmark_element_scale,
            handletextpad=0.50, columnspacing=1.15,
        )
            
        fig_path = os.path.join(FIG_DIR, 'Final_mv1_Step04B-08_DLBCL_Benchmark_Final.png')
        if SAVE_STEP04B08_DLBCL_BENCHMARK:
            _save_step04b_canvas_figure(
                fig,
                fig_path,
                label='Step04B-08 DLBCL benchmark final',
                show=True,
                close=True
            )
            print(f"\n🖼️ [Step 04B-08] Optimized DLBCL benchmark results saved: {fig_path}")
            if STEP04B_BENCHMARK_ONLY:
                print('🔹 [Step 04B-08] Benchmark-only export complete; later Figure sections are skipped.')
                raise SystemExit(0)
        else:
            plt.close(fig)


    # ── Panel 5: heatmap of all 17 models (Supplementary) ──
    for met_col in ['Balanced Acc', 'Macro F1', 'Weighted F1']:
        pivot_met = df_perf.pivot_table(index='Model', columns='Lineage', values=met_col) * 100
            
        # [Revision] Apply model-rank-based scaling within each Lineage (column) to clearly distinguish differences as small as 0.1%
        pivot_rank = pivot_met.rank(ascending=True)
        pivot_minmax = (pivot_rank - pivot_rank.min()) / (pivot_rank.max() - pivot_rank.min() + 1e-8)
            
        fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
        # Use Min-Max values (0~1) for data (colors) to maximize contrast; retain original percentages (%) as displayed text
        sns.heatmap(pivot_minmax, annot=pivot_met, fmt='.1f', cmap='YlOrRd', vmin=0, vmax=1,
                    linewidths=0.5, cbar_kws={'label': 'Relative Scaled Score (Min=0, Max=1)'}, ax=ax_hm,
                    annot_kws={'size': 9, 'weight': 'bold'})
        ax_hm.set_title(f"{met_col}: all 17 models", fontsize=14, fontweight='bold')
        ax_hm.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax_hm.set_xlabel('Lineage', fontsize=12, fontweight='bold')
        plt.setp(ax_hm.get_yticklabels(), fontsize=9, fontweight='bold')
        fig_hm.subplots_adjust(left=0.23, right=0.94, top=0.91, bottom=0.11)
        fig_path_hm = os.path.join(FIG_DIR, f'Final_mv1_Step04B-08_All17_{met_col.replace(" ","_")}_Heatmap.png')
        if SAVE_STEP04B08_ALL17_HEATMAPS:
            _save_step04b_canvas_figure(
                fig_hm,
                fig_path_hm,
                label=f'Step04B-08 all17 {met_col} heatmap',
                show=True,
                close=True
            )
        else:
            plt.close(fig_hm)

        # 04B-08B: encode actual performance values (0-1 normalized scores) as colors rather than rank-based relative scaling.
        pivot_norm = (pivot_met / 100.0).clip(lower=0, upper=1)
        fig_hm_norm, ax_hm_norm = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot_norm,
            annot=pivot_met,
            fmt='.1f',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar_kws={'label': 'Normalized Score (0=0%, 1=100%)'},
            ax=ax_hm_norm,
            annot_kws={'size': 9, 'weight': 'bold'}
        )
        ax_hm_norm.set_title(f"{met_col}: normalized score", fontsize=14, fontweight='bold')
        ax_hm_norm.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax_hm_norm.set_xlabel('Lineage', fontsize=12, fontweight='bold')
        plt.setp(ax_hm_norm.get_yticklabels(), fontsize=9, fontweight='bold')
        fig_hm_norm.subplots_adjust(left=0.23, right=0.94, top=0.91, bottom=0.11)
        fig_path_hm_norm = os.path.join(FIG_DIR, f'Final_mv1_Step04B-08B_All17_{met_col.replace(" ","_")}_NormalizedScore_Heatmap.png')
        if SAVE_STEP04B08B_NORMALIZED_SCORE_HEATMAPS:
            _save_step04b_canvas_figure(
                fig_hm_norm,
                fig_path_hm_norm,
                label=f'Step04B-08B all17 {met_col} normalized heatmap',
                show=True,
                close=True
            )
        else:
            plt.close(fig_hm_norm)
else:
    print("⚠️ Insufficient evaluable data; skipping visualization.")

# %%
# ==============================================================================
# [Step 04B-9] PUBLICATION QUALITY: Integrated Recall & Purity Heatmap (Global Scaled)
# Evaluation targets: DLBCL Test Set performance and global marker-expression purity comparison
# ==============================================================================
if True:
    print('\n🎨 [Step 04B-09] Visualizing integrated lineage-wise performance and marker Purity by method for the manuscript (Feature Freq & Large Fonts)...')
    import os
    import pickle
    import math
    from collections import defaultdict
    import matplotlib.colors as mcolors
    from sklearn.metrics import recall_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    sns.set_style("white")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Fallback handling for variable independence (in case earlier steps were skipped)
    if 'target_combo_sub' not in globals() and 'target_combo_sub' not in locals():
        target_combo_sub = 'XGB+MLP' # Default when the previous Step was skipped
    
    expert_subtypes_dict = {
        'T cells': ['Th', 'Tfh', 'Treg', 'CD8T', 'CD4_CTL', 'Tpr'],
        'B cells': ['B cell', 'Plasma cell'],
        'Myeloid cells': ['Myeloid', 'Macrophage', 'DC'],
        'Stromal cells': ['Stromal', 'FDC']
    }
    for _lin_name, _excluded in STEP04_MAIN_EXCLUDED_SUBTYPES.items():
        if _excluded and _lin_name in expert_subtypes_dict:
            expert_subtypes_dict[_lin_name] = [s for s in expert_subtypes_dict[_lin_name] if s not in _excluded]

    marker_reliability_payloads_09 = {}

    def _step04b_marker_reliability_alpha(p_value):
        return np.clip((-np.log10(max(float(p_value), 1e-50)) - 1.0) / 4.0, 0.1, 1.0)

    def _step04b_marker_reliability_size(freq_count, n_iters, scale=4.2, base=18):
        freq_pct = 100.0 * float(freq_count) / max(float(n_iters), 1.0)
        return freq_pct * scale + base

    def _plot_step04b_marker_reliability_payload(
        fig_obj,
        outer_spec,
        payload,
        title,
        n_cols=None,
        marker_fontsize=8,
        title_fontsize=13,
        axis_fontsize=8,
        bubble_scale=4.2,
        bubble_base=18,
        title_height=0.075,
        inner_wspace=0.28,
        inner_hspace=0.30
    ):
        valid_subtypes = payload['valid_subtypes']
        subtype_top_markers = payload['subtype_top_markers']
        mean_importance = payload['mean_importance']
        counts = payload['counts']
        distinct_colors = payload['distinct_colors']
        n_iters = payload['n_iters']
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        def _format_importance_tick(x_value, _pos=None):
            if abs(x_value) < 1e-12:
                return '0'
            if abs(x_value) < 0.01:
                return f'{x_value:.3f}'.rstrip('0').rstrip('.')
            return f'{x_value:.2f}'.rstrip('0').rstrip('.')

        n_subtypes = len(valid_subtypes)
        n_cols = min(3, max(1, n_subtypes)) if n_cols is None else min(n_cols, max(1, n_subtypes))
        n_rows = int(math.ceil(n_subtypes / n_cols))
        inner_gs = outer_spec.subgridspec(
            n_rows + 1,
            n_cols,
            height_ratios=[title_height] + [1.0] * n_rows,
            wspace=inner_wspace,
            hspace=inner_hspace
        )

        ax_title = fig_obj.add_subplot(inner_gs[0, :])
        ax_title.axis('off')
        ax_title.text(
            0.5,
            0.58,
            title,
            ha='center',
            va='center',
            fontsize=title_fontsize,
            fontweight='bold'
        )

        axes_rel = []
        for grid_idx in range(n_rows * n_cols):
            row_idx = grid_idx // n_cols
            col_idx = grid_idx % n_cols
            ax = fig_obj.add_subplot(inner_gs[row_idx + 1, col_idx])
            if grid_idx < n_subtypes:
                axes_rel.append(ax)
            else:
                ax.axis('off')

        for ax_idx, subtype_name in enumerate(valid_subtypes):
            ax = axes_rel[ax_idx]
            subtype_color = distinct_colors.get(subtype_name, CELL_COLORS.get(subtype_name, '#2C3E50'))
            rgba_base = mcolors.to_rgba(subtype_color)
            top_m_info = subtype_top_markers.get(subtype_name, [])

            if not top_m_info:
                ax.set_title(subtype_name, color=subtype_color, fontweight='bold', fontsize=axis_fontsize + 2)
                ax.text(
                    0.5,
                    0.5,
                    'No reliable markers',
                    ha='center',
                    va='center',
                    fontsize=axis_fontsize,
                    fontweight='bold',
                    color='gray',
                    transform=ax.transAxes
                )
                ax.axis('off')
                continue

            top_m_sorted = sorted(top_m_info, key=lambda x: mean_importance.get(x[0], 0))
            marker_names = [x[0] for x in top_m_sorted]
            marker_pvals = [x[1] for x in top_m_sorted]
            freq_counts = [counts.get(marker_name, 0) for marker_name in marker_names]
            importance_values = [mean_importance.get(marker_name, 0) for marker_name in marker_names]
            sizes = [
                _step04b_marker_reliability_size(freq, n_iters, scale=bubble_scale, base=bubble_base)
                for freq in freq_counts
            ]
            alphas = [_step04b_marker_reliability_alpha(p_val) for p_val in marker_pvals]
            color_array = np.zeros((len(marker_names), 4))
            for idx_alpha, alpha_val in enumerate(alphas):
                color_array[idx_alpha] = (
                    rgba_base[0],
                    rgba_base[1],
                    rgba_base[2],
                    alpha_val
                )

            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
            for ypos in range(len(marker_names)):
                ax.axhline(y=ypos, color='gray', linestyle='-', linewidth=0.25, alpha=0.45, zorder=1)

            ax.scatter(
                importance_values,
                np.arange(len(marker_names)),
                s=sizes,
                c=color_array,
                edgecolors='black',
                linewidth=0.8,
                zorder=2
            )

            ax.set_yticks(np.arange(len(marker_names)))
            ax.set_yticklabels(marker_names, fontweight='bold', fontsize=marker_fontsize)
            max_imp = max(importance_values) if importance_values else 0.1
            ax.set_xlim(left=-max_imp * 0.05, right=max_imp * 1.12)
            ax.margins(y=0.08)

            if ax_idx % n_cols == 0:
                ax.set_ylabel('Significant markers', fontsize=axis_fontsize, fontweight='bold')
            else:
                ax.set_ylabel('')
            if ax_idx // n_cols == n_rows - 1:
                ax.set_xlabel('Mean ML importance', fontsize=axis_fontsize, fontweight='bold')
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

            ax.set_title(subtype_name, color=subtype_color, fontweight='bold', fontsize=axis_fontsize + 2, pad=7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
            ax.xaxis.set_major_formatter(FuncFormatter(_format_importance_tick))
            ax.tick_params(axis='x', labelsize=max(axis_fontsize - 1, 7), pad=2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    def _plot_step04b_marker_reliability_shared_legend(
        ax_leg,
        bubble_scale=4.2,
        bubble_base=18
    ):
        ax_leg.axis('off')
        ax_leg.set_xlim(0, 1)
        ax_leg.set_ylim(0, 1)
        ax_leg.text(0.04, 0.96, 'Index', fontweight='bold', fontsize=17, va='top')

        ax_leg.text(0.04, 0.84, 'Extraction\nfrequency', fontweight='bold', fontsize=13, va='top')
        for idx, freq_pct in enumerate([25, 50, 75, 100]):
            y_pos = 0.73 - idx * 0.075
            size_val = freq_pct * bubble_scale + bubble_base
            ax_leg.scatter([0.16], [y_pos], s=size_val, color='gray', edgecolors='black', alpha=0.60)
            ax_leg.text(0.40, y_pos, f'{freq_pct}%', va='center', fontsize=11.5, fontweight='bold')

        ax_leg.text(0.04, 0.40, 'Alpha:\nsignificance', fontweight='bold', fontsize=13, va='top')
        ax_leg.text(0.04, 0.325, '(Wilcoxon adjusted P)', fontsize=10.5, color='gray', va='top')
        p_vals_leg = [0.05, 0.01, 0.001, 1e-15]
        p_labels = ['P < 0.05', 'P < 0.01', 'P < 0.001', 'P < 1e-15']
        for idx, p_val in enumerate(p_vals_leg):
            alpha_val = _step04b_marker_reliability_alpha(p_val)
            rgba_val = (0.3, 0.3, 0.3, alpha_val)
            y_pos = 0.25 - idx * 0.064
            ax_leg.scatter([0.16], [y_pos], s=145, color=rgba_val, edgecolors='black', linewidth=1.0)
            ax_leg.text(0.40, y_pos, p_labels[idx], va='center', fontsize=10.8)

    def _plot_step04b_marker_reliability_stacked_payload(
        ax,
        payload,
        title,
        *,
        marker_fontsize=8.0,
        subtype_fontsize=9.5,
        title_fontsize=13.0,
        axis_fontsize=9.2,
        bubble_scale=4.2,
        bubble_base=18,
        show_ylabel=True,
        show_xlabel=True,
        show_subtype_labels=False,
    ):
        valid_subtypes = payload['valid_subtypes']
        subtype_top_markers = payload['subtype_top_markers']
        mean_importance = payload['mean_importance']
        counts = payload['counts']
        distinct_colors = payload['distinct_colors']
        n_iters = payload['n_iters']
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        from matplotlib import transforms as mtransforms

        def _format_importance_tick(x_value, _pos=None):
            if abs(x_value) < 1e-12:
                return '0'
            if abs(x_value) < 0.01:
                return f'{x_value:.3f}'.rstrip('0').rstrip('.')
            return f'{x_value:.2f}'.rstrip('0').rstrip('.')

        rows = []
        subtype_ranges = []
        current_y = 0
        for subtype_name in valid_subtypes:
            top_m_info = subtype_top_markers.get(subtype_name, [])
            if not top_m_info:
                continue
            subtype_color = distinct_colors.get(subtype_name, CELL_COLORS.get(subtype_name, '#2C3E50'))
            top_m_sorted = sorted(
                top_m_info,
                key=lambda x: mean_importance.get(x[0], 0),
                reverse=True
            )
            start_y = current_y
            for marker_name, p_value in top_m_sorted:
                rows.append({
                    'subtype': subtype_name,
                    'marker': marker_name,
                    'p_value': p_value,
                    'importance': mean_importance.get(marker_name, 0),
                    'freq': counts.get(marker_name, 0),
                    'color': subtype_color,
                    'y': current_y,
                })
                current_y += 1
            subtype_ranges.append((subtype_name, start_y, current_y - 1, subtype_color))

        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=7)
        if not rows:
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                'No reliable markers',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=axis_fontsize,
                fontweight='bold',
                color='gray'
            )
            return

        y_values = np.array([row['y'] for row in rows], dtype=float)
        x_values = np.array([row['importance'] for row in rows], dtype=float)
        sizes = [
            _step04b_marker_reliability_size(row['freq'], n_iters, scale=bubble_scale, base=bubble_base)
            for row in rows
        ]
        color_array = np.zeros((len(rows), 4))
        for row_idx, row in enumerate(rows):
            rgba_base = mcolors.to_rgba(row['color'])
            alpha_val = _step04b_marker_reliability_alpha(row['p_value'])
            color_array[row_idx] = (rgba_base[0], rgba_base[1], rgba_base[2], alpha_val)

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
        for y_pos in y_values:
            ax.axhline(y=y_pos, color='gray', linestyle='-', linewidth=0.22, alpha=0.36, zorder=1)
        for subtype_name, start_y, end_y, subtype_color in subtype_ranges:
            if start_y > 0:
                ax.axhline(y=start_y - 0.5, color='gray', linestyle='--', linewidth=0.75, alpha=0.58, zorder=1)
            if show_subtype_labels:
                y_mid = (start_y + end_y) / 2
                trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
                ax.text(
                    0.985,
                    y_mid,
                    subtype_name,
                    ha='right',
                    va='center',
                    transform=trans,
                    fontsize=subtype_fontsize,
                    fontweight='bold',
                    color=subtype_color,
                    bbox=dict(facecolor='white', edgecolor=subtype_color, alpha=0.86, boxstyle='round,pad=0.18'),
                    clip_on=True,
                    zorder=4,
                )

        ax.scatter(
            x_values,
            y_values,
            s=sizes,
            c=color_array,
            edgecolors='black',
            linewidth=0.72,
            zorder=2
        )

        max_imp = max(float(np.nanmax(x_values)) if len(x_values) else 0.1, 0.01)
        ax.set_xlim(left=-max_imp * 0.05, right=max_imp * 1.10)
        ax.set_ylim(len(rows) + 0.05, -1.05)
        ax.set_yticks(y_values)
        ax.set_yticklabels([row['marker'] for row in rows], fontsize=marker_fontsize, fontweight='bold')
        for row, tick_label in zip(rows, ax.get_yticklabels()):
            tick_label.set_color(row['color'])
            tick_label.set_fontweight('bold')
        ax.set_xlabel('Mean ML importance' if show_xlabel else '', fontsize=axis_fontsize, fontweight='bold', labelpad=4)
        ax.set_ylabel('Subtype / marker' if show_ylabel else '', fontsize=axis_fontsize, fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_importance_tick))
        ax.tick_params(axis='x', labelsize=max(axis_fontsize - 1, 7), pad=2)
        ax.tick_params(axis='y', pad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _subset_step04b_marker_reliability_payload(payload, subtype_subset):
        subtype_subset = [
            subtype_name for subtype_name in subtype_subset
            if subtype_name in payload['valid_subtypes']
        ]
        return {
            **payload,
            'valid_subtypes': subtype_subset,
            'subtype_top_markers': {
                subtype_name: payload['subtype_top_markers'].get(subtype_name, [])
                for subtype_name in subtype_subset
            },
        }

    def _plot_step04b_marker_reliability_bottom_legend(
        ax_leg,
        bubble_scale=4.2,
        bubble_base=18
    ):
        ax_leg.axis('off')
        ax_leg.set_xlim(0, 1)
        ax_leg.set_ylim(0, 1)
        ax_leg.text(0.28, 0.78, 'Extraction frequency', fontweight='bold', fontsize=18.0, va='center', ha='center')
        for idx, freq_pct in enumerate([25, 50, 75, 100]):
            x_pos = 0.16 + idx * 0.08
            size_val = freq_pct * bubble_scale + bubble_base
            ax_leg.scatter([x_pos], [0.34], s=size_val, color='gray', edgecolors='black', alpha=0.60)
            ax_leg.text(x_pos, 0.03, f'{freq_pct}%', ha='center', va='center', fontsize=15.0, fontweight='bold')

        ax_leg.text(0.72, 0.78, 'Significance', fontweight='bold', fontsize=18.0, va='center', ha='center')
        ax_leg.text(0.72, 0.56, '(Wilcoxon adjusted P)', fontsize=13.0, color='gray', va='center', ha='center')
        p_vals_leg = [0.05, 0.01, 0.001, 1e-15]
        p_labels = ['P < 0.05', 'P < 0.01', 'P < 0.001', 'P < 1e-15']
        for idx, p_val in enumerate(p_vals_leg):
            alpha_val = _step04b_marker_reliability_alpha(p_val)
            rgba_val = (0.3, 0.3, 0.3, alpha_val)
            x_pos = 0.57 + idx * 0.10
            ax_leg.scatter([x_pos], [0.30], s=190, color=rgba_val, edgecolors='black', linewidth=1.15)
            ax_leg.text(x_pos, 0.03, p_labels[idx], ha='center', va='center', fontsize=13.5)
    
    print("  📊 Applying Global StandardScaler to all data...")
    X_global_scaled = _step04b_get_global_scaled_matrix(adata)

    _mask_ref_total = (~adata.obs['Disease'].str.contains('DLBCL', na=False)) & \
                      (~np.isin(adata.obs['GT_Subtype'], ['na', 'Unknown', 'nan', '']))

    # Because DLBCL lacks expert CD4_CTL GT, apply the Step04B-3 reference cutoff
    # only to evaluation target labels. Do not modify the original GT_Subtype column.
    eval_subtype_09 = adata.obs['GT_Subtype'].astype(str).copy()
    preflight_info_09 = adata.uns.get('step04B_cd4ctl_gmm_preflight', {})
    cd4ctl_cutoff_09 = preflight_info_09.get('selected_threshold_cutoff', np.nan)
    if np.isfinite(cd4ctl_cutoff_09) and 'GRZB' in adata.var_names:
        disease_is_dlbcl_09 = adata.obs['Disease'].str.contains('DLBCL', na=False).values
        ml_lineage_vals_09 = adata.obs['ML_Lineage_Raw'].astype(str).values
        base_subtype_vals_09 = adata.obs['GT_Subtype'].astype(str).values
        mask_dlbcl_cd4_candidate_09 = (
            disease_is_dlbcl_09 &
            (ml_lineage_vals_09 == 'T cells') &
            np.isin(base_subtype_vals_09, CD4CTL_CANDIDATE_SUBTYPES)
        )
        if STEP04B_TPR_GMM_MODE != 'keep' and STEP04B_TPR_GMM_ELIGIBLE_COL in adata.obs.columns:
            tpr_gmm_eligible_09 = adata.obs[STEP04B_TPR_GMM_ELIGIBLE_COL].astype(bool).values
            mask_dlbcl_cd4_candidate_09 &= (
                (base_subtype_vals_09 != 'Tpr') |
                ((base_subtype_vals_09 == 'Tpr') & tpr_gmm_eligible_09)
            )
        idx_dlbcl_cd4_candidate_09 = np.where(mask_dlbcl_cd4_candidate_09)[0]
        if len(idx_dlbcl_cd4_candidate_09) > 0:
            grzb_dlbcl_cd4_09 = _extract_marker_vector(adata, idx_dlbcl_cd4_candidate_09, 'GRZB')
            idx_dlbcl_cd4ctl_09 = idx_dlbcl_cd4_candidate_09[grzb_dlbcl_cd4_09 >= float(cd4ctl_cutoff_09)]
            eval_subtype_09.iloc[idx_dlbcl_cd4ctl_09] = 'CD4_CTL'
            print(
                f"  ℹ️ Applying the CD4_CTL proxy for Step04B-9 evaluation: "
                f"GRZB Z ≥ {_format_cd4ctl_gmm_cutoff(cd4ctl_cutoff_09)}, "
                f"{len(idx_dlbcl_cd4ctl_09):,}/{len(idx_dlbcl_cd4_candidate_09):,} cells"
            )

    # 🌟 Preload global Feature Importance data
    fi_save_path = os.path.join(CKPT_DIR, 'mv1_feature_importance_04B.pkl')
    global_fi_data = {}
    if os.path.exists(fi_save_path):
        try:
            with open(fi_save_path, 'rb') as f:
                global_fi_data = pickle.load(f).get('subtype', {})
        except Exception as e:
            print(f"  ⚠️ Failed to load the FI file: {e}")

    for lin, target_subtypes in expert_subtypes_dict.items():
        _mask_dlbcl_lin = (adata.obs['ML_Lineage_Raw'] == lin) & \
                          (adata.obs['Disease'].str.contains('DLBCL', na=False))
                          
        exist_subtypes = eval_subtype_09.loc[_mask_dlbcl_lin].dropna().unique()
        valid_subtypes = [s for s in target_subtypes if s in exist_subtypes]
        
        if len(valid_subtypes) <= 1: continue
            
        print(f"\n▶️ [{lin}] Extracting markers and computing frequencies...")
        
        # ── 1. Within-lineage DEG analysis and marker-subtype color mapping (integrated marker extraction) ──
        _mask_ref_lin = (_mask_ref_total) & (adata.obs['Modeling_Broad'] == lin)
        excluded_marker_subtypes = _step04_excluded_subtypes(lin)
        if excluded_marker_subtypes:
            _mask_ref_lin &= ~adata.obs['GT_Subtype'].astype(str).isin(excluded_marker_subtypes)
        
        # Scan XGBoost Feature Importance records
        records = global_fi_data.get(lin, {}).get('XGB', [])
        n_iters = len(records)
        counts = defaultdict(int)
        importance_sums = defaultdict(float)
        if n_iters > 0:
            for rec in records:
                sorted_f = sorted(rec.items(), key=lambda x: abs(x[1]), reverse=True)
                for fname, imp in rec.items():
                    importance_sums[fname] += abs(imp)
                for fname, _ in sorted_f[:20]:
                    counts[fname] += 1
        mean_importance = {k: v / max(n_iters, 1) for k, v in importance_sums.items()}
        
        # Extract statistics for all significant markers per subtype (using the Wilcoxon Rank-Sum Test)
        def _adjust_pvalues_for_marker_screening(p_values):
            """Adjust marker-screening p-values for multiple testing."""
            p_values = np.asarray(p_values, dtype=float)
            if p_values.size == 0:
                return p_values
            try:
                from statsmodels.stats.multitest import multipletests
                return multipletests(
                    p_values,
                    alpha=STEP04_MARKER_PURITY_PADJ_ALPHA,
                    method=STEP04_MARKER_PURITY_PADJ_METHOD
                )[1]
            except Exception:
                # Benjamini-Hochberg fallback used when statsmodels is unavailable.
                n_tests = len(p_values)
                order = np.argsort(p_values)
                ranked = p_values[order] * n_tests / np.arange(1, n_tests + 1)
                ranked = np.minimum.accumulate(ranked[::-1])[::-1]
                adjusted = np.empty_like(ranked)
                adjusted[order] = np.clip(ranked, 0, 1)
                return adjusted

        subtype_top_markers = {}
        import scipy.stats as stats
        for s in valid_subtypes:
            idx_s = np.where((adata.obs['GT_Subtype'] == s) & (_mask_ref_lin))[0]
            idx_others = np.where((adata.obs['GT_Subtype'] != s) & (_mask_ref_lin))[0]
            if len(idx_s) > 0 and len(idx_others) > 0:
                mean_s = X_global_scaled[idx_s].mean(axis=0)
                mean_others = X_global_scaled[idx_others].mean(axis=0)
                contrast = mean_s - mean_others
                
                valid_mask = (mean_s > 0.1) & (contrast > 0.1)
                valid_indices = np.where(valid_mask)[0]
                
                candidate_marker_tests = []
                for idx in valid_indices:
                    stat, pval = stats.ranksums(X_global_scaled[idx_s, idx], X_global_scaled[idx_others, idx])
                    if stat > 0:
                        candidate_marker_tests.append((idx, pval))

                significant_markers = []
                if candidate_marker_tests:
                    adjusted_pvals = _adjust_pvalues_for_marker_screening([p for _, p in candidate_marker_tests])
                    for (idx, raw_pval), adjusted_pval in zip(candidate_marker_tests, adjusted_pvals):
                        if adjusted_pval < STEP04_MARKER_PURITY_PADJ_ALPHA:
                            m_name = adata.var_names[idx]
                            significant_markers.append((m_name, adjusted_pval))
                
                # Sort jointly by frequency (counts) and importance (mean_importance)
                sorted_by_freq = sorted(significant_markers, key=lambda x: (counts[x[0]], mean_importance.get(x[0], 0)), reverse=True)
                # If there are too many, limit to the Top 15 for readability (keep the Heatmap uncluttered)
                subtype_top_markers[s] = sorted_by_freq[:15]

        # Construct ordered_markers for the Heatmap (ensure a perfect staircase diagonal pattern)
        # Check which subtype has the highest expression of each marker (Specificity)
        # Basic algorithm: identify the subtype with the highest expression of each marker
        marker_best_subtype = {}
        for s in valid_subtypes:
            idx_s = np.where((adata.obs['GT_Subtype'] == s) & (_mask_ref_lin))[0]
            if len(idx_s) > 0:
                mean_s = X_global_scaled[idx_s].mean(axis=0)
                for m_name, _ in subtype_top_markers.get(s, []):
                    m_idx = adata.var_names.get_loc(m_name)
                    val = mean_s[m_idx]
                    if m_name not in marker_best_subtype or val > marker_best_subtype[m_name][1]:
                        marker_best_subtype[m_name] = (s, val)

        ordered_markers = []
        marker_to_color = {}
        marker_to_subtype = {}
        
        # Adjust subtype-specific palettes to maximize visual distinction (retain the same color family and maximize differences)
        import seaborn as sns
        distinct_colors = {}
        # Get each lineage's Base color spectrum (T cells -> OrRd(orange~red), B cells -> Blues ...)
        cmap_dict = {
            'T cells': 'OrRd_r', 'B cells': 'Blues_r', 'Myeloid cells': 'Purples_r',
            'NK cells': 'Oranges_r', 'Stromal cells': 'Greens_r'
        }
        pal_name = cmap_dict.get(lin, 'Greys_r')
        # Add +2 to avoid overly pale colors (close to white)
        pal = sns.color_palette(pal_name, len(valid_subtypes) + 2)
        
        # Shuffle palette order to maximize color differences between adjacent subtypes (alternate dark/light)
        sub_pal = pal[:len(valid_subtypes)]
        mixed_pal = []
        left, right = 0, len(sub_pal) - 1
        while left <= right:
            mixed_pal.append(sub_pal[left])
            if left != right:
                mixed_pal.append(sub_pal[right])
            left += 1
            right -= 1

        for i, s in enumerate(valid_subtypes):
            distinct_colors[s] = mcolors.to_hex(mixed_pal[i])

        # To create a staircase heatmap, first place biologically interpretable anchor markers
        # in subtype order, then supplement gaps with statistically selected markers.
        signature_marker_blocks = {
            'T cells': {
                'Th': ['CD5', 'CD62L', 'CD45RO'],
                'Tfh': ['CXCL13', 'CXCR5', 'PD1', 'ICOS', 'CD134'],
                'Treg': ['FOXP3', 'CD25', 'CTLA4'],
                'CD8T': ['CD8', 'CD57', 'Tbet'],
                # Exclude CD3/CD4 from the purity heatmap because they are pan-T or CD4-lineage markers;
                # assess CD4+GRZB+ identity separately in the Step04B-03 marker profile.
                'CD4_CTL': ['GRZB'],
                'Tpr': ['Ki67', 'TIM3', 'LAG3', 'CD38']
            }
        }
        marker_blocks_for_lineage = signature_marker_blocks.get(lin, {})
        curated_marker_only = (
            STEP04_TCELL_MARKER_PURITY_CURATED_ONLY
            and lin == 'T cells'
            and len(marker_blocks_for_lineage) > 0
        )

        # Calculate DLBCL means to exactly match the ordering to the data rendered in the Heatmap (DLBCL test set)
        dlbcl_gt_means = {}
        mask_dlbcl_lin_arr = _mask_dlbcl_lin.values if hasattr(_mask_dlbcl_lin, 'values') else _mask_dlbcl_lin
        for s in valid_subtypes:
            idx_s_dlbcl = np.where((eval_subtype_09.values == s) & mask_dlbcl_lin_arr)[0]
            if len(idx_s_dlbcl) > 0:
                dlbcl_gt_means[s] = X_global_scaled[idx_s_dlbcl].mean(axis=0)
            else:
                dlbcl_gt_means[s] = np.zeros(X_global_scaled.shape[1])

        for s in valid_subtypes:
            s_color = distinct_colors[s]

            for m_name in marker_blocks_for_lineage.get(s, []):
                marker_passes_adjusted_p = (
                    lin != 'T cells'
                    or (
                        m_name in marker_best_subtype
                        and marker_best_subtype[m_name][0] == s
                    )
                )
                if marker_passes_adjusted_p and m_name in adata.var_names and m_name not in ordered_markers:
                    ordered_markers.append(m_name)
                    marker_to_color[m_name] = s_color
                    marker_to_subtype[m_name] = s

            if curated_marker_only:
                continue
            
            # Filter each marker to belong only to the subtype block with its highest expression
            s_specific_markers = []
            for top_m, _ in subtype_top_markers.get(s, []):
                if top_m in marker_best_subtype and marker_best_subtype[top_m][0] == s:
                    s_specific_markers.append(top_m)
            
            # Arrange the Heatmap Y-axis in descending order of actual mean expression in the rendered DLBCL test set
            s_specific_markers.sort(key=lambda m: dlbcl_gt_means[s][adata.var_names.get_loc(m)], reverse=True)
            
            for m_name in s_specific_markers:
                if m_name not in ordered_markers:
                    ordered_markers.append(m_name)
                    marker_to_color[m_name] = s_color
                    marker_to_subtype[m_name] = s

        if not ordered_markers: continue

        # ── Compute FI frequency (proportion of appearances in the XGB Top-15) ──
        # ── 2. Compute Recall on DLBCL Test data ──
        _idx_dlbcl_gt = np.where(_mask_dlbcl_lin & (eval_subtype_09.isin(valid_subtypes)))[0]
        if len(_idx_dlbcl_gt) == 0: continue
        
        _y_true = eval_subtype_09.iloc[_idx_dlbcl_gt].values
        methods = [
            ('Pred_Clustering', 'Clustering (Leiden)'), 
            ('Pred_SC_Centroid', 'SC-Centroid'), 
            ('ML_Subtype_Pure', f'Proposed ML (Ours: {target_combo_sub})')
        ]
        benchmark_recall_by_method = {}
        for m_col, m_name in methods:
            if m_col not in adata.obs.columns: continue
            _y_pred = adata.obs[m_col].iloc[_idx_dlbcl_gt].values
            recalls = recall_score(
                _y_true.astype(str),
                _y_pred.astype(str),
                labels=valid_subtypes,
                average=None,
                zero_division=0
            )
            benchmark_recall_by_method[m_name] = pd.Series(recalls * 100, index=valid_subtypes)

        # ── 3. Visualization (integrate benchmarks as bottom heatmap rows instead of separate panels) ──
        _idx_dlbcl_lin = np.where(_mask_dlbcl_lin)[0]
        method_preds_lin = {'Expert Ground Truth': eval_subtype_09.iloc[_idx_dlbcl_lin].values}
        for m_col, m_name in methods:
            if m_col in adata.obs.columns: method_preds_lin[m_name] = adata.obs[m_col].iloc[_idx_dlbcl_lin].values
            
        profiles = {}
        for m_key in [m_name for _, m_name in methods][::-1]:
            if m_key not in method_preds_lin: continue
            preds = method_preds_lin[m_key]
            heatmap_data = np.zeros((len(ordered_markers), len(valid_subtypes)))
            for j, s in enumerate(valid_subtypes):
                mask_s = (preds == s)
                if mask_s.sum() > 0:
                    mean_z = X_global_scaled[_idx_dlbcl_lin[mask_s]].mean(axis=0)
                    for i, m in enumerate(ordered_markers): heatmap_data[i, j] = mean_z[adata.var_names.get_loc(m)]
            
            # Preserve the original profile; STEP04_MARKER_PURITY_CLIP_NEGATIVE determines the actual display mode.
            profiles[m_key] = pd.DataFrame(heatmap_data, index=ordered_markers, columns=valid_subtypes)

        # Follow the meeting-note principle: "redder toward the top, less overlap with other subtypes toward the top".
        # Prioritize the Proposed ML profile, retain only subtype-specific markers, and reorder within each block.
        if lin == 'T cells' and profiles:
            preferred_profile_key = next(
                (k for k in profiles.keys() if str(k).startswith('Proposed ML')),
                next(iter(profiles.keys()))
            )
            sort_profile = profiles[preferred_profile_key].clip(lower=0)
            reordered_markers = []
            for subtype_name in valid_subtypes:
                block_markers = [
                    m_name for m_name in ordered_markers
                    if marker_to_subtype.get(m_name) == subtype_name
                ]
                other_subtypes = [s for s in valid_subtypes if s != subtype_name]

                def _marker_step_score(marker_name):
                    target_val = 0.0
                    if marker_name in sort_profile.index and subtype_name in sort_profile.columns:
                        target_val = float(sort_profile.loc[marker_name, subtype_name])
                    off_target_val = 0.0
                    if marker_name in sort_profile.index and other_subtypes:
                        off_target_vals = sort_profile.loc[marker_name, other_subtypes].astype(float)
                        if len(off_target_vals) > 0:
                            off_target_val = float(off_target_vals.max())
                    specificity_val = target_val - off_target_val
                    return target_val, specificity_val, off_target_val

                scored_markers = []
                fallback_markers = []
                for marker_name in block_markers:
                    target_val, specificity_val, off_target_val = _marker_step_score(marker_name)
                    sort_key = (target_val, specificity_val, -off_target_val)
                    if target_val >= STEP04_TCELL_MARKER_PURITY_MIN_TARGET_Z:
                        fallback_markers.append((sort_key, marker_name))
                    if (
                        not STEP04_TCELL_MARKER_PURITY_ENFORCE_SPECIFICITY
                        or (
                            target_val >= STEP04_TCELL_MARKER_PURITY_MIN_TARGET_Z
                            and specificity_val >= STEP04_TCELL_MARKER_PURITY_MIN_MARGIN
                        )
                    ):
                        scored_markers.append((sort_key, marker_name))

                # If no marker passes the strict conditions, retain only the best marker for that subtype.
                # This avoids omitting subtypes such as Th that lack completely independent markers within the panel.
                if not scored_markers and fallback_markers:
                    fallback_markers.sort(reverse=True)
                    scored_markers = fallback_markers[:1]

                scored_markers.sort(reverse=True)
                if STEP04_TCELL_MARKER_PURITY_TOP_N_PER_SUBTYPE is not None:
                    scored_markers = scored_markers[:STEP04_TCELL_MARKER_PURITY_TOP_N_PER_SUBTYPE]
                if STEP04_TCELL_MARKER_PURITY_BOTTOM_MARKERS:
                    _bottom_markers = set(STEP04_TCELL_MARKER_PURITY_BOTTOM_MARKERS)
                    scored_markers = (
                        [item for item in scored_markers if item[1] not in _bottom_markers]
                        + [item for item in scored_markers if item[1] in _bottom_markers]
                    )
                reordered_markers.extend([marker_name for _, marker_name in scored_markers])

            if reordered_markers:
                ordered_markers = reordered_markers
                profiles = {
                    profile_name: profile_df.loc[ordered_markers]
                    for profile_name, profile_df in profiles.items()
                }

        if STEP04_MARKER_PURITY_DYNAMIC_COLOR_RANGE and profiles:
            _profile_scale_values = []
            for _profile_df in profiles.values():
                if STEP04_MARKER_PURITY_CLIP_NEGATIVE:
                    _profile_scale_values.extend(_profile_df.clip(lower=0).to_numpy().ravel())
                else:
                    _profile_scale_values.extend(np.abs(_profile_df.to_numpy().ravel()))
            _profile_scale_values = np.asarray(_profile_scale_values, dtype=float)
            _profile_scale_values = _profile_scale_values[np.isfinite(_profile_scale_values)]
            if _profile_scale_values.size > 0:
                v_limit = float(np.nanpercentile(_profile_scale_values, 95))
                v_limit = min(max(v_limit, 1.0), 5.0)
            else:
                v_limit = 1.0
        else:
            v_limit = 1.0
        marker_block_boundaries = []
        if ordered_markers:
            _prev_marker_subtype = marker_to_subtype.get(ordered_markers[0])
            for _i_marker, _marker_name in enumerate(ordered_markers[1:], start=1):
                _cur_marker_subtype = marker_to_subtype.get(_marker_name)
                if _cur_marker_subtype != _prev_marker_subtype:
                    marker_block_boundaries.append(_i_marker)
                    _prev_marker_subtype = _cur_marker_subtype
        subtype_boundaries = list(range(1, len(valid_subtypes)))
        num_panels = len(profiles)
        if num_panels == 0:
            continue
        # For B/T purity used in the Main Figure, directly use the edited version's canvas ratio.
        # For Supplementary lineages, retain existing dynamic sizing based on content volume.
        _panel_width = 3.05
        _panel_height = 6.15
        _left_margin_in = 1.95
        _right_margin_in = 1.38
        _bottom_margin_in = 1.65
        _top_margin_in = 0.70
        _between_panel_in = 1.18
        _panel_wspace = _between_panel_in / _panel_width
        fig_width = (
            _left_margin_in + _right_margin_in +
            (_panel_width * num_panels) +
            (_between_panel_in * max(num_panels - 1, 0))
        )
        fig_height = _top_margin_in + _bottom_margin_in + _panel_height
        _direct_purity_size = (
            STEP04B_MAIN_PURITY_FIGSIZES.get(lin)
            or STEP04B_SUPPLEMENT_PURITY_FIGSIZES.get(lin)
        )
        if _direct_purity_size is not None:
            fig_width, fig_height = _direct_purity_size
            _left_margin_in = 1.25
            _right_margin_in = 0.25
            _bottom_margin_in = 1.04
            _top_margin_in = 0.66
            _between_panel_in = 0.30
            _panel_width = (
                fig_width - _left_margin_in - _right_margin_in
                - (_between_panel_in * max(num_panels - 1, 0))
            ) / num_panels
            _panel_height = fig_height - _top_margin_in - _bottom_margin_in
            _panel_wspace = _between_panel_in / _panel_width
        fig = plt.figure(figsize=(fig_width, fig_height))
        _purity_gs = fig.add_gridspec(
            1,
            num_panels,
            left=_left_margin_in / fig_width,
            right=1 - (_right_margin_in / fig_width),
            bottom=_bottom_margin_in / fig_height,
            top=1 - (_top_margin_in / fig_height),
            wspace=_panel_wspace,
        )
        axes = np.asarray([
            fig.add_subplot(_purity_gs[0, _panel_idx])
            for _panel_idx in range(num_panels)
        ])
        _purity_cbar_width_in = min(1.42, fig_width * 0.15)
        _purity_cbar_height_in = 0.075
        _purity_cbar_left_in = fig_width - _purity_cbar_width_in - 0.55
        _purity_cbar_bottom_in = 0.28
        purity_cbar_ax = fig.add_axes([
            _purity_cbar_left_in / fig_width,
            _purity_cbar_bottom_in / fig_height,
            _purity_cbar_width_in / fig_width,
            _purity_cbar_height_in / fig_height,
        ])

        def _format_purity_subtype_tick(label):
            """Use compact horizontal labels without changing subtype values."""
            return {
                'B cell': 'B cells',
                'Plasma cell': 'Plasma\ncells',
                'Myeloid': 'Myeloid\ncells',
                'Macrophage': 'Macro-\nphages',
                'Stromal': 'Stromal\ncells',
                'CD8T': 'CD8 T',
                'CD4_CTL': 'CD4\nCTL',
            }.get(str(label), str(label))

        # 🎨 Function to apply X/Y-axis label colors (with fixed rotation added)
        def apply_axis_colors(ax):
            _original_xlabels = [t.get_text() for t in ax.get_xticklabels()]
            ax.set_xticklabels(
                [_format_purity_subtype_tick(label) for label in _original_xlabels],
                rotation=0,
                ha='center',
            )
            for t, s_name in zip(ax.get_xticklabels(), _original_xlabels):
                if s_name in distinct_colors: t.set_color(distinct_colors[s_name])
                elif s_name in CELL_COLORS: t.set_color(CELL_COLORS[s_name])
                t.set_fontweight('bold')
                t.set_fontsize(11.8 if lin == 'T cells' else 14.2)
                t.set_rotation(0)
                t.set_ha('center')
            for t in ax.get_yticklabels():
                m_label = t.get_text()
                m_orig = m_label.split(" (")[0]
                if m_orig in marker_to_color: t.set_color(marker_to_color[m_orig])
                t.set_fontweight('bold')
                t.set_fontsize(12.6 if lin == 'T cells' else (13.8 if lin == 'B cells' else 13.8))
                if m_label.startswith('Bench.'):
                    t.set_fontsize(12.5)
                t.set_rotation(0) # Force horizontal labels to prevent overlap

        def add_step_block_separators(ax):
            for _boundary in marker_block_boundaries:
                ax.axhline(_boundary, color='black', lw=1.25, alpha=0.55)
            for _boundary in subtype_boundaries:
                ax.axvline(_boundary, color='black', lw=0.8, alpha=0.20)
            ax.axhline(len(ordered_markers), color='black', lw=1.6, alpha=0.75)

        def _format_purity_panel_title(method_name):
            """Split only the long Proposed ML title over two lines to stabilize panel width."""
            if str(method_name).startswith('Proposed ML'):
                combo_label = str(method_name).replace('Proposed ML', '').strip()
                return f'Proposed ML\n{combo_label}' if combo_label else 'Proposed ML'
            return method_name

        def _format_heatmap_annotations(plot_df):
            def _format_marker_value(v):
                if not np.isfinite(v):
                    return ''
                if abs(v) < STEP04_MARKER_PURITY_ANNOT_MIN:
                    return '0.0' if STEP04_MARKER_PURITY_SHOW_ZERO_ANNOT else ''
                return f'{v:.1f}'
            return plot_df.applymap(_format_marker_value)

        def _purity_profile_for_display(profile_df):
            if lin == 'T cells' and STEP04_TCELL_MARKER_PURITY_DISPLAY_MODE == 'target_only_positive':
                display_df = pd.DataFrame(0.0, index=profile_df.index, columns=profile_df.columns)
                for _marker_name in profile_df.index:
                    _target_subtype = marker_to_subtype.get(_marker_name)
                    if _target_subtype in display_df.columns:
                        display_df.loc[_marker_name, _target_subtype] = max(float(profile_df.loc[_marker_name, _target_subtype]), 0.0)
                return display_df
            if STEP04_MARKER_PURITY_CLIP_NEGATIVE:
                return profile_df.clip(lower=0)
            return profile_df
        
        def _append_benchmark_row(profile_df, annotation_df, method_name):
            """Add per-subtype benchmark recall rows below the Purity heatmap."""
            benchmark_label = 'Bench. (%)'
            plot_with_benchmark = profile_df.copy()
            plot_with_benchmark.loc[benchmark_label] = 0.0

            annot_with_benchmark = annotation_df.copy()
            recall_series = benchmark_recall_by_method.get(method_name, pd.Series(dtype=float))
            annot_with_benchmark.loc[benchmark_label] = [
                f'{float(recall_series.get(subtype_name, np.nan)):.1f}'
                if np.isfinite(recall_series.get(subtype_name, np.nan)) else ''
                for subtype_name in profile_df.columns
            ]
            return plot_with_benchmark, annot_with_benchmark

        # Expression-purity profile (Purity) panels by Method
        for ax_idx, m_key in enumerate(profiles.keys()):
            ax = axes[ax_idx]
            _profile_to_plot = _purity_profile_for_display(profiles[m_key])
            _annot_to_plot = _format_heatmap_annotations(_profile_to_plot)
            _profile_to_plot, _annot_to_plot = _append_benchmark_row(_profile_to_plot, _annot_to_plot, m_key)
            if STEP04_MARKER_PURITY_CLIP_NEGATIVE:
                _heatmap_cmap = 'Reds'
                _heatmap_vmin, _heatmap_vmax, _heatmap_center = 0, v_limit, None
                _cbar_label = 'Z-score'
            else:
                _heatmap_cmap = 'RdBu_r'
                _heatmap_vmin, _heatmap_vmax, _heatmap_center = -v_limit, v_limit, 0
                _cbar_label = 'Z-score'

            _purity_cbar_kws = {
                'orientation': 'horizontal',
            }

            _heatmap_kwargs = dict(
                data=_profile_to_plot, annot=_annot_to_plot, fmt='', cmap=_heatmap_cmap,
                vmin=_heatmap_vmin, vmax=_heatmap_vmax, linewidths=0.7, linecolor='white',
                ax=ax, cbar=(ax_idx == len(profiles)-1),
                cbar_ax=purity_cbar_ax if ax_idx == len(profiles)-1 else None,
                cbar_kws=_purity_cbar_kws if ax_idx == len(profiles)-1 else None,
                annot_kws={
                    "size": 13.2 if lin == 'T cells' else (14.8 if lin == 'B cells' else 15.5),
                    "weight": "bold",
                },
                square=False
            )
            if _heatmap_center is not None:
                _heatmap_kwargs['center'] = _heatmap_center
            hm = sns.heatmap(**_heatmap_kwargs)
            if ax_idx == len(profiles) - 1 and hm.collections:
                cbar = hm.collections[0].colorbar
                if cbar is not None:
                    cbar.ax.set_ylabel('')
                    cbar.ax.set_title(_cbar_label, fontsize=11.2, fontweight='bold', pad=3, loc='center')
                    cbar.ax.tick_params(axis='x', labelsize=10.5, width=1.0, pad=1)
                    for _tick_label in cbar.ax.get_xticklabels():
                        _tick_label.set_fontweight('bold')
            ax.set_aspect('auto')
            _purity_title_size = 20.0 if lin in {'B cells', 'T cells'} else 19.5
            ax.set_title(
                _format_purity_panel_title(m_key),
                fontsize=_purity_title_size,
                fontweight='bold',
                pad=7,
            )
            # Display Y-axis labels only on the first Purity panel for a clean appearance
            if ax_idx == 0:
                ax.set_ylabel('Signature markers', fontsize=15.0, fontweight='bold', labelpad=1)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)
            apply_axis_colors(ax)
            ax.tick_params(axis='x', pad=8)
            add_step_block_separators(ax)

        fig_path_purity = os.path.join(FIG_DIR, f'Final_mv1_Step04B-09_Purity_ColorSynced_{lin.replace(" ", "_")}.png')
        if SAVE_STEP04B09_PURITY_COLOR_SYNCED:
            _save_step04b_canvas_figure(
                fig,
                fig_path_purity,
                label=f'Step04B-09 purity color synced {lin}',
                show=True,
                close=True
            )
        else:
            plt.close(fig)

        # ── 4. Separate Figure: plot significant Marker Reliability for each Subtype ──
        print(f"  📊 [{lin}] Generating separate Marker Reliability (Bubble plot)...")
        if n_iters > 0:
            marker_reliability_payloads_09[lin] = {
                'valid_subtypes': list(valid_subtypes),
                'subtype_top_markers': {
                    subtype_name: list(marker_entries)
                    for subtype_name, marker_entries in subtype_top_markers.items()
                },
                'mean_importance': dict(mean_importance),
                'counts': dict(counts),
                'distinct_colors': dict(distinct_colors),
                'n_iters': int(n_iters)
            }

            # Set height dynamically (based on the Subtype with the most markers)
            max_markers = max([len(v) for v in subtype_top_markers.values()]) if subtype_top_markers else 5
            n_subtypes_rel = len(valid_subtypes)
            n_cols_rel = min(3, max(1, n_subtypes_rel))
            n_rows_rel = int(math.ceil(n_subtypes_rel / n_cols_rel))
            row_height_rel = max(4.8, max_markers * 0.42 + 1.6)
            fig_height_rel = max(7.8, n_rows_rel * row_height_rel)
            fig_width_rel = max(15.5, 4.7 * n_cols_rel + 3.0)

            # Visualization (Subtype-specific Bubble Plot panels in a grid with at most 2 rows + dedicated legend panel on the right)
            fig_rel = plt.figure(figsize=(fig_width_rel, fig_height_rel))
            gs_rel = fig_rel.add_gridspec(
                n_rows_rel,
                n_cols_rel + 1,
                width_ratios=[1.0] * n_cols_rel + [0.62],
                wspace=0.42,
                hspace=0.62
            )
            axes_rel = []
            for _grid_idx in range(n_rows_rel * n_cols_rel):
                _row_idx = _grid_idx // n_cols_rel
                _col_idx = _grid_idx % n_cols_rel
                _ax_rel = fig_rel.add_subplot(gs_rel[_row_idx, _col_idx])
                if _grid_idx < n_subtypes_rel:
                    axes_rel.append(_ax_rel)
                else:
                    _ax_rel.axis('off')

            from matplotlib.colors import to_rgba

            for ax_idx, s in enumerate(valid_subtypes):
                ax = axes_rel[ax_idx]
                s_color = distinct_colors.get(s, CELL_COLORS.get(s, '#2C3E50'))
                s_rgba_base = to_rgba(s_color)
                top_m_info = subtype_top_markers.get(s, [])
                
                if not top_m_info:
                    ax.set_title(s, color=s_color, fontweight='bold', fontsize=16)
                    ax.text(0.5, 0.5, 'No reliable markers', ha='center', va='center', fontweight='bold', color='gray')
                    ax.axis('off'); continue
                    
                # Sort in ascending ML Importance (importance increases from bottom to top)
                top_m_sorted = sorted(top_m_info, key=lambda x: mean_importance.get(x[0], 0))
                m_names = [x[0] for x in top_m_sorted]
                m_pvals = [x[1] for x in top_m_sorted]
                
                freqs_raw = [counts[m] for m in m_names] # Actual appearance counts
                importance_values = [mean_importance.get(m, 0) for m in m_names]
                
                # Bubble Sizes: proportional to appearance count (frequency)
                sizes = [f * 6 + 20 for f in freqs_raw]
                
                # Bubble Alphas: proportional to P-value (map opacity using -log10(p) for more pronounced differences)
                # Map more steeply across 0.1 (transparent) ~ 1.0 (fully opaque)
                alphas = [np.clip((-np.log10(max(p, 1e-50)) - 1.0) / 4.0, 0.1, 1.0) for p in m_pvals]
                color_array = np.zeros((len(m_names), 4))
                for i, a in enumerate(alphas):
                    color_array[i] = (s_rgba_base[0], s_rgba_base[1], s_rgba_base[2], a)
                
                # Guidelines (grid) and reference lines
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, zorder=0)
                for ypos in range(len(m_names)):
                    ax.axhline(y=ypos, color='gray', linestyle='-', linewidth=0.3, alpha=0.5, zorder=1)
                
                # Add y-axis padding to prevent clipping circles at the top and bottom
                ax.margins(y=0.08)
                
                # Bubble Scatter Plot
                scatter = ax.scatter(importance_values, np.arange(len(m_names)), s=sizes, c=color_array, edgecolors='black', linewidth=1.2, zorder=2)

                ax.set_yticks(np.arange(len(m_names)))
                ax.set_yticklabels(m_names, fontweight='bold', fontsize=13)
                
                # Adjust the x-axis scale
                max_imp = max(importance_values) if importance_values else 0.1
                ax.set_xlim(left=-max_imp*0.05, right=max_imp * 1.1)

                if ax_idx % n_cols_rel == 0:
                    ax.set_ylabel('Significant Markers', fontsize=14, fontweight='bold')
                else:
                    ax.set_ylabel('')
                
                ax.set_xlabel('Mean ML Importance', fontsize=12, fontweight='bold')
                ax.set_title(s, color=s_color, fontweight='bold', fontsize=18, pad=15)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            # --- Dedicated legend (Index) panel ---
            ax_leg = fig_rel.add_subplot(gs_rel[:, -1])
            ax_leg.axis('off')
            ax_leg.set_xlim(0, 1)
            ax_leg.set_ylim(0, 1)
            
            # 1. Size Legend (Frequency)
            ax_leg.text(0.05, 0.95, 'Extraction Frequency', fontweight='bold', fontsize=14, va='top')
            sizes_leg = [25, 50, 75, 100]
            for idx, s_val in enumerate(sizes_leg):
                s_plot = (s_val / 100.0 * n_iters) * 6 + 20
                y_pos = 0.85 - idx * 0.08
                ax_leg.scatter([0.15], [y_pos], s=s_plot, color='gray', edgecolors='black', alpha=0.6)
                ax_leg.text(0.35, y_pos, f'{s_val}%', va='center', fontsize=13, fontweight='bold')
            
            # 2. Alpha Legend (P-value)
            ax_leg.text(0.05, 0.45, 'Alpha: Significance', fontweight='bold', fontsize=14, va='top')
            ax_leg.text(0.05, 0.40, '(Wilcoxon P-value)', fontsize=11, color='gray', va='top')
            p_vals_leg = [0.05, 0.01, 0.001, 1e-15]
            p_labels = ['P < 0.05', 'P < 0.01', 'P < 0.001', 'P < 1e-15']
            for idx, p_val in enumerate(p_vals_leg):
                alpha_val = np.clip((-np.log10(max(p_val, 1e-50)) - 1.0) / 4.0, 0.1, 1.0)
                rgba_val = (0.3, 0.3, 0.3, alpha_val) # Neutral dark gray
                y_pos = 0.30 - idx * 0.08
                ax_leg.scatter([0.15], [y_pos], s=150, color=rgba_val, edgecolors='black', linewidth=1.2)
                ax_leg.text(0.35, y_pos, p_labels[idx], va='center', fontsize=12)

            fig_rel.suptitle(f"Significant Marker Reliability per Subtype ({lin})", fontsize=20, fontweight='bold', y=0.96)
            fig_rel.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.88)
            fig_path_rel = os.path.join(FIG_DIR, f'Final_mv1_Step04B-09_MarkerReliability_{lin.replace(" ", "_")}.png')
            if SAVE_STEP04B09_MARKER_RELIABILITY:
                _save_step04b_canvas_figure(
                    fig_rel,
                    fig_path_rel,
                    label=f'Step04B-09 marker reliability {lin}',
                    show=True,
                    close=True
                )
            else:
                plt.close(fig_rel)

    if SAVE_STEP04B09_MARKER_RELIABILITY_COMPOSITE and marker_reliability_payloads_09:
        composite_order_09 = ['T cells', 'B cells', 'Myeloid cells', 'Stromal cells']
        available_composite_lineages_09 = [
            lineage_name for lineage_name in composite_order_09
            if lineage_name in marker_reliability_payloads_09
        ]
        if available_composite_lineages_09:
            print("  📊 Generating the Step04B-09 Marker Reliability combined figure...")
            fig_rel_comp = plt.figure(figsize=STEP04B_MARKER_RELIABILITY_COMPOSITE_FIGSIZE)
            gs_rel_outer = fig_rel_comp.add_gridspec(
                2,
                1,
                height_ratios=[1.0, 0.16],
                hspace=0.075
            )
            gs_rel_comp = gs_rel_outer[0].subgridspec(
                2,
                3,
                wspace=0.30,
                hspace=0.20
            )

            if 'T cells' in marker_reliability_payloads_09:
                _t_payload_09 = marker_reliability_payloads_09['T cells']
                _t_group_specs_09 = [
                    ('T cells: Th / Tfh', ['Th', 'Tfh']),
                    ('T cells: Treg / CD4 CTL', ['Treg', 'CD4_CTL']),
                    ('T cells: CD8 T / Tpr', ['CD8T', 'Tpr']),
                ]
                for _tg_idx_09, (_tg_title_09, _tg_subtypes_09) in enumerate(_t_group_specs_09):
                    ax_t_rel_comp = fig_rel_comp.add_subplot(gs_rel_comp[0, _tg_idx_09])
                    _tg_payload_09 = _subset_step04b_marker_reliability_payload(_t_payload_09, _tg_subtypes_09)
                    _plot_step04b_marker_reliability_stacked_payload(
                        ax_t_rel_comp,
                        _tg_payload_09,
                        _tg_title_09,
                        marker_fontsize=10.8,
                        subtype_fontsize=8.2,
                        title_fontsize=16.5,
                        axis_fontsize=13.6,
                        bubble_scale=2.8,
                        bubble_base=12,
                        show_ylabel=(_tg_idx_09 == 0),
                        show_xlabel=False,
                        show_subtype_labels=False,
                    )
            else:
                fig_rel_comp.add_subplot(gs_rel_comp[0, :]).axis('off')

            bottom_lineage_specs_09 = [
                ('B cells', gs_rel_comp[1, 0]),
                ('Myeloid cells', gs_rel_comp[1, 1]),
                ('Stromal cells', gs_rel_comp[1, 2])
            ]
            for bottom_lineage_09, bottom_spec_09 in bottom_lineage_specs_09:
                if bottom_lineage_09 in marker_reliability_payloads_09:
                    ax_bottom_rel_comp = fig_rel_comp.add_subplot(bottom_spec_09)
                    _bottom_payload_09 = marker_reliability_payloads_09[bottom_lineage_09]
                    _bottom_subtitle_09 = ' / '.join(_bottom_payload_09['valid_subtypes'])
                    _bottom_title_09 = f"{bottom_lineage_09}\n{_bottom_subtitle_09}" if _bottom_subtitle_09 else bottom_lineage_09
                    _plot_step04b_marker_reliability_stacked_payload(
                        ax_bottom_rel_comp,
                        _bottom_payload_09,
                        _bottom_title_09,
                        marker_fontsize=11.0,
                        subtype_fontsize=8.8,
                        title_fontsize=16.3,
                        axis_fontsize=13.6,
                        bubble_scale=2.8,
                        bubble_base=12,
                        show_ylabel=(bottom_lineage_09 == 'B cells'),
                        show_xlabel=True,
                        show_subtype_labels=False,
                    )
                else:
                    fig_rel_comp.add_subplot(bottom_spec_09).axis('off')

            ax_rel_comp_leg = fig_rel_comp.add_subplot(gs_rel_outer[1])
            _plot_step04b_marker_reliability_bottom_legend(
                ax_rel_comp_leg,
                bubble_scale=4.8,
                bubble_base=24
            )

            # Reserve enough canvas space for the right-column tick labels.  The
            # last formatted tick can extend beyond the axes even when the axes
            # itself is inside the canvas, so keep a small explicit right gutter.
            fig_rel_comp.subplots_adjust(left=0.095, right=0.948, bottom=0.025, top=0.958)
            fig_path_rel_comp = os.path.join(FIG_DIR, 'Final_mv1_Step04B-09_MarkerReliability_Composite.png')
            _save_step04b_canvas_figure(
                fig_rel_comp,
                fig_path_rel_comp,
                label='Step04B-09 marker reliability composite',
                show=True,
                close=True
            )
# %%
# ==============================================================================
# [Step 04B-10] REFINED STATISTICAL LINES: GMM Cutoff & Marginal Histogram
# ==============================================================================
if True:
    print('\n📏 [Step 04B-10] Creating an integrated visualization of GMM Cutoff and marker-expression density (Histogram)...')
    from scipy.stats import mannwhitneyu
    from matplotlib.gridspec import GridSpec

    # 1. Data preparation (CD4 candidates within T cells)
    cd4_subtypes = [s for s in CD4CTL_CANDIDATE_SUBTYPES] + ['CD4_CTL']
    _mask = (adata.obs['ML_Lineage_Raw'] == 'T cells') & \
            (adata.obs['Disease'].str.contains('DLBCL', na=False)) & \
            (adata.obs['ML_Subtype_Raw'].isin(cd4_subtypes))
    _idx = np.where(_mask)[0]
    
    if len(_idx) > 0:
        _df = pd.DataFrame({
            'GRZB': _extract_marker_vector(adata, _idx, 'GRZB'),
            'Subtype': adata.obs['ML_Subtype_Raw'].iloc[_idx].values
        })

        # Prefer the cutoff determined in Reference preflight; otherwise use the minimum CD4_CTL value.
        cd4_ctl_grzb = _df[_df['Subtype'] == 'CD4_CTL']['GRZB']
        _preflight_info = adata.uns.get('step04B_cd4ctl_gmm_preflight', {})
        _ref_cutoff = _preflight_info.get('selected_threshold_cutoff', np.nan)
        gmm_cutoff = float(_ref_cutoff) if np.isfinite(_ref_cutoff) else (
            cd4_ctl_grzb.min() if not cd4_ctl_grzb.empty else 1.5
        )

        # Create labels including sample counts (n=counts)
        stats_labels = {s: f"{s}\n(n={len(_df[_df['Subtype']==s]):,})" for s in cd4_subtypes}
        _df['Stats_X'] = _df['Subtype'].map(stats_labels)
        order_labels = [stats_labels[s] for s in cd4_subtypes]
        
        # Apply colors (using mv1 global CELL_COLORS)
        palette = {stats_labels[s]: CELL_COLORS.get(s, '#808080') for s in cd4_subtypes}
        hist_palette = {s: CELL_COLORS.get(s, '#808080') for s in cd4_subtypes}

        # 2. Set the visualization layout
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(1, 2, width_ratios=[3.5, 1.5], wspace=0.03)
        ax_main = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax_main)

        # ── [Left panel]: Boxplot + Stripplot ──
        sns.boxplot(data=_df, x='Stats_X', y='GRZB', order=order_labels, 
                    palette=palette, showfliers=False, width=0.5,
                    linewidth=2.0, ax=ax_main, boxprops=dict(edgecolor='black', alpha=0.9), zorder=3)
        
        sns.stripplot(data=_df, x='Stats_X', y='GRZB', order=order_labels,
                      color='black', size=2.0, alpha=0.15, ax=ax_main, jitter=0.25, zorder=2)

        # ── [Right panel]: cumulative histogram ──
        sns.histplot(data=_df, y='GRZB', hue='Subtype', hue_order=cd4_subtypes,
                     palette=hist_palette, multiple='stack', bins=80, 
                     ax=ax_hist, edgecolor='black', linewidth=0.2, legend=False)

        ax_hist.set_xscale('symlog', linthresh=10)
        
        # 3. Label the GMM Cutoff
        for ax in [ax_main, ax_hist]:
            ax.axhline(gmm_cutoff, color='#e74c3c', lw=2.5, ls='--', zorder=4)
            
        ax_hist.text(ax_hist.get_xlim()[1]*0.1, gmm_cutoff + 0.5, _cd4ctl_gmm_cutoff_display(gmm_cutoff), 
                     color='#c0392b', fontsize=12, fontweight='bold', ha='center', va='bottom')

        ax_hist.tick_params(left=False, labelleft=False)
        ax_hist.set_ylabel('')
        ax_hist.set_xlabel('Cell Count (Log Scale)', fontsize=12)
        ax_hist.spines['left'].set_visible(False)
        
        ax_main.set_ylabel('GRZB Expression Level (Z-score)', fontsize=14, fontweight='bold')
        ax_main.set_xlabel('')
        max_grzb = _df['GRZB'].max()
        ax_main.set_ylim(-1.5, max_grzb * 1.05 if not np.isnan(max_grzb) else 5.0)

        # 4. Wilcoxon rank-sum test (Mann-Whitney U) and statistical annotation lines
        targets = [(s, pos) for pos, s in enumerate(cd4_subtypes) if s != 'CD4_CTL']
        cd4ctl_pos = cd4_subtypes.index('CD4_CTL')
        base_y = max_grzb * 0.70 if not np.isnan(max_grzb) else 3.0
        step_h = max_grzb * 0.08 if not np.isnan(max_grzb) else 0.5
        
        for i, (name, x_pos) in enumerate(targets):
            sample_data = _df[_df['Subtype']==name]['GRZB']
            if not sample_data.empty and not cd4_ctl_grzb.empty:
                _, p = mannwhitneyu(cd4_ctl_grzb, sample_data, alternative='two-sided')
            else:
                p = 1.0
            
            y = base_y + (i * step_h)
            h = step_h * 0.35
            
            ax_main.plot([x_pos, x_pos, cd4ctl_pos, cd4ctl_pos], [y, y+h, y+h, y], lw=2.0, c='#2c3e50')
            
            sig = 'p < 0.001 ***' if p < 0.001 else f'p = {p:.3f}'
            ax_main.text((x_pos + cd4ctl_pos) * 0.5, y + h + 0.1, sig, 
                      ha='center', va='bottom', color='#2c3e50', fontsize=12, fontweight='bold')

        fig.suptitle('Validation of CD4+ Cytotoxic T-cell Isolation (Wilcoxon Test)',
                     fontsize=18, fontweight='bold', y=0.96)
        
        sns.despine(ax=ax_main)
        sns.despine(ax=ax_hist, left=True)
        
        fig_path_wilcoxon = os.path.join(FIG_DIR, 'mv1_Step04B-10_Wilcoxon_CD4CTL_Histogram.png')
        if SAVE_STEP04B10_WILCOXON_HISTOGRAM:
            _save_step04b_canvas_figure(
                fig,
                fig_path_wilcoxon,
                label='Step04B-10 Wilcoxon CD4CTL histogram',
                show=True,
                close=True,
                dpi=400
            )
        else:
            plt.close(fig)
    else:
        print("  ⚠️ No CD4+ cells in DLBCL are available for statistical processing; skipping chart generation.")

# %%
# ==============================================================================
# [Step 04B-11] Compare marker-expression patterns (Reference GT vs DLBCL results from all models)
# ==============================================================================
if True:
    print('🔹 [Step 04B-11] Starting lineage-wise model marker-expression pattern comparison...')

    # 🌟 [User settings] Enter the markers to inspect for each lineage here.
    lineage_markers_to_plot = {
        'T cells': ['CD3', 'CD4', 'CD8', 'FOXP3', 'GRZB', 'PD1', 'Ki67', 'CXCR5'],
        'B cells': ['CD20', 'CD21','BCL6', 'PAX5'], # Enter manually (e.g., ['CD20', 'CD19', 'BCL6', ...])
        'Myeloid cells': ['CD68', 'CD11c'], # Enter manually (e.g., ['CD68', 'CD163', 'CD11c', ...])
        'Stromal cells': ['PDPN'] # Enter manually (e.g., ['PDPN', 'Vimentin', 'SMA', ...])
    }

    # Target lineages (subtype sorting order)
    subtype_order_dict = {
        'T cells': ['Th', 'Tfh', 'Treg', 'CD8T', 'Tpr', 'CD4_CTL'],
        'B cells': ['B cell', 'Plasma cell'],
        'Myeloid cells': ['Myeloid', 'Macrophage', 'DC'],
        'Stromal cells': ['Stromal', 'FDC']
    }
    for _lin_name, _excluded in STEP04_MAIN_EXCLUDED_SUBTYPES.items():
        if _excluded and _lin_name in subtype_order_dict:
            subtype_order_dict[_lin_name] = [s for s in subtype_order_dict[_lin_name] if s not in _excluded]

    def get_marker_matrix(subset_adata, label_col, subtypes, markers):
        if len(markers) == 0: return None
        matrix = []
        for s in subtypes:
            mask = (subset_adata.obs[label_col] == s)
            if mask.any():
                # Compute the mean values of these markers
                m_indices = [subset_adata.var_names.get_loc(m) for m in markers if m in subset_adata.var_names]
                if not m_indices: 
                    means = [np.nan] * len(markers)
                else:
                    means = np.asarray(subset_adata.X[mask][:, m_indices].mean(axis=0)).flatten()
            else:
                means = [np.nan] * len(markers)
            matrix.append(means)
        return pd.DataFrame(matrix, index=subtypes, columns=markers)

    # ── Start the lineage-wise loop ──
    for lin in lineage_markers_to_plot.keys():
        markers = lineage_markers_to_plot[lin]
        subtypes = subtype_order_dict.get(lin, [])
        excluded_marker_subtypes = _step04_excluded_subtypes(lin)
        
        if len(markers) == 0:
            print(f"  ⏩ [{lin}]: marker list is empty; skipping.")
            continue
            
        print(f"  ▶️ [{lin}] Generating marker heatmap...")

        # 🌟 Recompute indices at this point (prevent loss)
        ref_lin_mask = (adata.obs['Modeling_Broad'] == lin) & \
                       (~adata.obs['Disease'].str.contains('DLBCL', na=False))
        if excluded_marker_subtypes:
            ref_lin_mask &= ~adata.obs['GT_Subtype'].astype(str).isin(excluded_marker_subtypes)
        idx_ref_lin = np.where(ref_lin_mask)[0]
        idx_dlbcl_lin = np.where((adata.obs['ML_Lineage_Raw'] == lin) & 
                                 (adata.obs['Disease'].str.contains('DLBCL', na=False)))[0]

        if len(idx_ref_lin) == 0 or len(idx_dlbcl_lin) == 0:
            print(f"  ⚠️ [{lin}]: insufficient data (Ref/DLBCL) for comparison.")
            continue

        # 1. Reference (GT)
        df_ref = get_marker_matrix(adata[idx_ref_lin], 'GT_Subtype', subtypes, markers)
        # 2. DLBCL (Proposed ML)
        df_ours = get_marker_matrix(adata[idx_dlbcl_lin], 'ML_Subtype_Raw', subtypes, markers)
        # 3. DLBCL (Clustering)
        df_cl = get_marker_matrix(adata[idx_dlbcl_lin], 'Pred_Clustering', subtypes, markers)
        # 4. DLBCL (SC-Centroid)
        df_sc = get_marker_matrix(adata[idx_dlbcl_lin], 'Pred_SC_Centroid', subtypes, markers)

        panel_specs = [
            (df_ref, 'Reference GT', 'darkblue'),
            (df_ours, 'Proposed ML', 'darkred'),
            (df_cl, 'Leiden clustering', 'black'),
            (df_sc, 'SC-Centroid', 'black')
        ]
        _panel_values = np.concatenate([_df.values.ravel() for _df, _, _ in panel_specs])
        _finite_panel_values = _panel_values[np.isfinite(_panel_values)]
        if _finite_panel_values.size > 0:
            _vlim_panel = max(1.0, np.nanpercentile(np.abs(_finite_panel_values), 98))
        else:
            _vlim_panel = 1.0

        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
        fig.suptitle(f'{lin} marker expression patterns', fontsize=15.5, fontweight='bold', y=0.98)

        for _panel_idx, (ax, (_plot_df, _plot_title, _title_color)) in enumerate(zip(axes.flat, panel_specs)):
            sns.heatmap(
                _plot_df,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-_vlim_panel,
                vmax=_vlim_panel,
                ax=ax,
                linewidths=0.55,
                linecolor='white',
                cbar=(_panel_idx == 3),
                cbar_kws={'label': 'Mean global Z-score', 'shrink': 0.74, 'pad': 0.025} if _panel_idx == 3 else None,
                annot_kws={'size': 8.4, 'weight': 'bold'}
            )
            ax.set_title(_plot_title, fontsize=11.5, fontweight='bold', color=_title_color, pad=8)
            ax.set_ylabel('Subtype', fontweight='bold', fontsize=9.8)
            ax.set_xlabel('Marker', fontweight='bold', fontsize=9.8)
            plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8.6, fontweight='bold')
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8.8, fontweight='bold')

        fig.subplots_adjust(left=0.085, right=0.925, bottom=0.105, top=0.91, wspace=0.22, hspace=0.32)
        
        # Save
        save_name = f'Final_mv1_Step04B-11_{lin.replace(" ", "_")}_Marker_Comparison.png'
        fig_path_marker_comp = os.path.join(FIG_DIR, save_name)
        if SAVE_STEP04B11_MARKER_COMPARISON and lin in STEP04B11_MARKER_COMPARISON_OUTPUT_LINEAGES:
            _save_step04b_canvas_figure(
                fig,
                fig_path_marker_comp,
                label=f'Step04B-11 marker comparison {lin}',
                show=True,
                close=True
            )
        else:
            plt.close(fig)

    print('✅ [Step 04B-11] Marker comparison completed for all lineages.')

# %%
# ==============================================================================
# [Step 04B-12] FINAL VALIDATION: Target GT vs. Proposed ML Prediction (DLBCL)
# ==============================================================================
if True:
    print('🎯 [Step 04B-12] Comparing DLBCL target ground truth (GT) vs ML prediction model...')
    from sklearn.metrics import confusion_matrix
    from sklearn.mixture import GaussianMixture

    _mask_dlbcl = (adata.obs['ML_Lineage_Raw'] == 'T cells') & \
                  (adata.obs['Disease'].str.contains('DLBCL', na=False))
    _idx_dlbcl = np.where(_mask_dlbcl)[0]

    if len(_idx_dlbcl) > 0:
        y_target_gt = adata.obs['GT_Subtype'].iloc[_idx_dlbcl].astype(str).values
        y_pred = adata.obs['ML_Subtype_Raw'].iloc[_idx_dlbcl].astype(str).values
        
        # ── [Additional request] 1. Original evaluation before applying CD4_CTL (Original GT) ──
        subtypes_orig = ['Th', 'Tfh', 'Treg', 'CD8T', 'Tpr']
        cm_orig = confusion_matrix(y_target_gt, y_pred, labels=subtypes_orig)
        cm_norm_orig = cm_orig.astype('float') / (cm_orig.sum(axis=1)[:, np.newaxis] + 1e-8) * 100

        df_cm_norm_orig = pd.DataFrame(cm_norm_orig, index=subtypes_orig, columns=subtypes_orig)
        df_cm_count_orig = pd.DataFrame(cm_orig, index=subtypes_orig, columns=subtypes_orig)

        sns.set_style("white")
        fig_orig = plt.figure(figsize=(6.4, 5.5))
        ax_orig = fig_orig.add_axes([0.22, 0.13, 0.62, 0.722])
        cbar_ax_orig = fig_orig.add_axes([0.88, 0.285, 0.035, 0.44])
        sns.heatmap(df_cm_norm_orig, annot=True, fmt='.1f', cmap='Blues',
                    linewidths=1.1, linecolor='white', square=True,
                    cbar_ax=cbar_ax_orig,
                    cbar_kws={'label': 'Agreement (%)'},
                    annot_kws={"size": 10.0, "weight": "bold"}, ax=ax_orig)
        for i in range(len(subtypes_orig)):
            for j in range(len(subtypes_orig)):
                count = df_cm_count_orig.iloc[i, j]
                if count > 0:
                    ax_orig.text(j+0.5, i+0.78, f"({count:,})", ha='center', va='center', fontsize=7.6)

        ax_orig.set_title('GT vs ML prediction without CD4_CTL',
                          fontsize=12.2, fontweight='bold', pad=9)
        ax_orig.set_xlabel('ML prediction', fontsize=10.5, fontweight='bold', labelpad=7)
        ax_orig.set_ylabel('Ground truth', fontsize=10.5, fontweight='bold', labelpad=7)
        plt.setp(ax_orig.get_xticklabels(), fontsize=9.4, fontweight='bold', rotation=0, ha='center')
        plt.setp(ax_orig.get_yticklabels(), fontsize=9.4, fontweight='bold', rotation=0)
        for _, spine in ax_orig.spines.items(): spine.set_visible(True); spine.set_linewidth(1)
        fig_path_orig = os.path.join(FIG_DIR, 'Final_mv1_Step04B-12_GT_vs_ML_Confusion_Heatmap_No_CD4CTL.png')
        if SAVE_STEP04B12_NO_CD4CTL_CONFUSION:
            _save_step04b_canvas_figure(
                fig_orig,
                fig_path_orig,
                label='Step04B-12 GT vs ML confusion heatmap without CD4_CTL',
                show=True,
                close=True,
                dpi=600
            )
            print(f'  🖼️ Original (excluding CD4_CTL) GT vs ML confusion matrix saved: {fig_path_orig}')
        else:
            plt.close(fig_orig)

        total_acc_orig = np.diag(cm_orig).sum() / cm_orig.sum() * 100
        print(f"\n📊 DLBCL original agreement analysis (excluding CD4_CTL):")
        print(f"  - Overall Agreement: {total_acc_orig:.1f}%\n")

        # ── 2. Final evaluation after applying the CD4_CTL GMM proxy ──
        # Prefer the Reference CD4-candidate preflight cutoff to align training/evaluation criteria.
        _cd4_lineage = list(CD4CTL_CANDIDATE_SUBTYPES)
        _mask_dlbcl_cd4 = _mask_dlbcl & adata.obs['GT_Subtype'].isin(_cd4_lineage)
        if STEP04B_TPR_GMM_MODE != 'keep' and STEP04B_TPR_GMM_ELIGIBLE_COL in adata.obs.columns:
            _base_subtype_for_eval = adata.obs['GT_Subtype'].astype(str)
            _tpr_gmm_eligible_eval = adata.obs[STEP04B_TPR_GMM_ELIGIBLE_COL].astype(bool)
            _mask_dlbcl_cd4 &= (
                (_base_subtype_for_eval != 'Tpr') |
                ((_base_subtype_for_eval == 'Tpr') & _tpr_gmm_eligible_eval)
            )
        _idx_cd4 = np.where(_mask_dlbcl_cd4)[0]
        
        if len(_idx_cd4) > 10:
            gr_vals = _extract_marker_vector(adata, _idx_cd4, 'GRZB')
            _preflight_info = adata.uns.get('step04B_cd4ctl_gmm_preflight', {})
            _ref_cutoff = _preflight_info.get('selected_threshold_cutoff', np.nan)
            if np.isfinite(_ref_cutoff):
                cd4ctl_proxy_labels = gr_vals >= float(_ref_cutoff)
                print(
                    '  ▶ CD4_CTL evaluation proxy: applying the reference GMM cutoff '
                    f'(GRZB Z ≥ {_format_cd4ctl_gmm_cutoff(_ref_cutoff)})'
                )
            else:
                gr_vals_2d = gr_vals.reshape(-1, 1)
                gmm_dlbcl = GaussianMixture(n_components=3, random_state=42).fit(gr_vals_2d)
                gmm_labels = gmm_dlbcl.predict(gr_vals_2d)
                high_comp = np.argmax(gmm_dlbcl.means_.flatten())
                cd4ctl_proxy_labels = gmm_labels == high_comp
                print("  ▶ CD4_CTL evaluation proxy: no reference cutoff → applying DLBCL CD4-candidate GMM fallback")
            
            for i, idx in enumerate(_idx_cd4):
                if cd4ctl_proxy_labels[i]:
                    rel_pos = np.where(_idx_dlbcl == idx)[0][0]
                    y_target_gt[rel_pos] = 'CD4_CTL'

        subtypes = ['Th', 'Tfh', 'Treg', 'CD8T', 'Tpr', 'CD4_CTL']

        cm = confusion_matrix(y_target_gt, y_pred, labels=subtypes)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100

        df_cm_norm = pd.DataFrame(cm_norm, index=subtypes, columns=subtypes)
        df_cm_count = pd.DataFrame(cm, index=subtypes, columns=subtypes)
        _subtype_display = {'CD4_CTL': 'CD4 CTL', 'CD8T': 'CD8 T'}
        df_cm_norm_display = df_cm_norm.rename(index=_subtype_display, columns=_subtype_display)

        fig_cm = plt.figure(figsize=STEP04B_SUPPLEMENT_FIGSIZES['gt_ml_confusion'])
        ax = fig_cm.add_axes([0.160, 0.17, 0.665, 0.76])
        cbar_ax = fig_cm.add_axes([0.862, 0.37, 0.014, 0.30])
        sns.set_style("white")
        
        sns.heatmap(df_cm_norm_display, annot=True, fmt='.1f', cmap='Blues',
                    linewidths=1.1, linecolor='white', square=True,
                    cbar_ax=cbar_ax,
                    cbar_kws={},
                    annot_kws={"size": 10.0, "weight": "bold"}, ax=ax)
        
        for i in range(len(subtypes)):
            for j in range(len(subtypes)):
                count = df_cm_count.iloc[i, j]
                if count > 0:
                    ax.text(j+0.5, i+0.78, f"({count:,})", ha='center', va='center', fontsize=7.5)

        ax.set_xlabel('ML prediction', fontsize=10.5, fontweight='bold', labelpad=7)
        ax.set_ylabel('Ground truth + GMM proxy', fontsize=9.2, fontweight='bold', labelpad=-6)

        plt.setp(ax.get_xticklabels(), fontsize=9.3, fontweight='bold', rotation=0, ha='center')
        plt.setp(ax.get_yticklabels(), fontsize=9.3, fontweight='bold', rotation=0)
        cbar_ax.set_title('')
        fig_cm.text(
            0.840, 0.690, 'Agreement (%)',
            ha='left', va='bottom', fontsize=7.0, fontweight='bold',
        )
        cbar_ax.tick_params(labelsize=8.5, length=2)
        
        for _, spine in ax.spines.items(): spine.set_visible(True); spine.set_linewidth(1)
        fig_path = os.path.join(FIG_DIR, 'Final_mv1_Step04B-12_GT_vs_ML_Confusion_Heatmap.png')
        if SAVE_STEP04B12_GT_ML_CONFUSION:
            _save_step04b_canvas_figure(
                fig_cm,
                fig_path,
                label='Step04B-12 GT vs ML confusion heatmap',
                show=True,
                close=True,
                dpi=600
            )
            print(f'  🖼️ Final GT vs ML confusion matrix saved: {fig_path}')
        else:
            plt.close(fig_cm)

        total_acc = np.diag(cm).sum() / cm.sum() * 100
        print(f"\n📊 DLBCL final agreement analysis:")
        print(f"  - Overall Agreement: {total_acc:.1f}%")
        if 'CD4_CTL' in subtypes:
            ctl_idx = subtypes.index('CD4_CTL')
            ctl_recall = cm_norm[ctl_idx, ctl_idx]
            print(f"  - CD4_CTL detection accuracy (Recall against GMM Proxy): {ctl_recall:.1f}%")

# %%
# ==============================================================================
# [Step 04B-13] Error Analysis: marker-profile analysis of misclassified CD4_CTL cells
# ==============================================================================
if '_idx_dlbcl' in dir() and len(_idx_dlbcl) > 0:
    print('🔬 [Step 04B-13] Analyzing marker-expression profiles of misclassified CD4_CTL cells...')

    # 1. Extract target indices (cells with CD4_CTL GT)
    _ctl_mask_in_dlbcl = (y_target_gt == 'CD4_CTL')
    _ctl_indices = _idx_dlbcl[_ctl_mask_in_dlbcl]
    
    # Retrieve ML predictions and actual expression values for these cells
    _error_df = pd.DataFrame(
        adata.X[_ctl_indices], 
        columns=adata.var_names
    )
    _error_df['ML_Prediction'] = y_pred[_ctl_mask_in_dlbcl]

    # 2. Group by prediction result and compute mean expression
    # (Match: CD4_CTL / Mismatch: Th, Tfh, Treg, CD8T, etc.)
    _profile_summary = _error_df.groupby('ML_Prediction').mean()

    # Count cells per prediction result for visualization readability
    _counts = _error_df['ML_Prediction'].value_counts()
    _new_index = [f"{idx} (n={_counts[idx]:,})" for idx in _profile_summary.index]
    _profile_summary.index = _new_index

    # 3. Heatmap visualization
    fig_error = plt.figure(figsize=(20, 8))
    sns.set_style("white")
    
    # Z-score-style limits (vmax/vmin) can be applied to maximize expression differences
    ax = sns.heatmap(_profile_summary, annot=False, cmap='RdBu_r', center=0, 
                    linewidths=0.5, cbar_kws={'label': 'Mean Expression (Z-score)'})

    plt.title('Marker Profiles of Ground Truth CD4_CTL: Grouped by ML Predictions', 
              fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('Markers', fontsize=12, fontweight='bold')
    plt.ylabel('ML Predicted Subtypes (for GT CD4_CTL)', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    fig_error.subplots_adjust(left=0.11, right=0.985, bottom=0.18, top=0.86)
    
    fig_path_error = os.path.join(FIG_DIR, 'mv1_Step04B-13_CD4CTL_Mismatch_Profile_Heatmap.png')
    if SAVE_STEP04B13_ERROR_PROFILE:
        _save_step04b_canvas_figure(
            fig_error,
            fig_path_error,
            label='Step04B-13 CD4CTL mismatch profile heatmap',
            show=True,
            close=True,
            dpi=400
        )
    else:
        plt.close(fig_error)

    print(f"  💡 Analysis points:")
    print(f"  - 'CD4_CTL' group: the typical cytotoxic profile we expect (GRZB high, etc.)")
    print(f"  - 'Th' or 'Treg' group: GRZB is high according to GT, but other baseline markers are likely to strongly retain Th or Treg characteristics.")


# %%
# ==============================================================================
# [Step 04B-14] Save the final checkpoint
# ==============================================================================
if STEP04_FIGURE_ONLY:
    print("⏩ [Step 04B-14] Figure-only mode → skipping final checkpoint saving")
elif not step04_skip:
    _step04b_write_checkpoint_if_enabled(adata, CHECKPOINT_STEP04)
    print(f"✅ [Step 04B-14] Final checkpoint saved: {CHECKPOINT_STEP04}")
