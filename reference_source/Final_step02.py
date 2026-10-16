# %%
from Final_utils import *
from itertools import combinations as _combs

# ==============================================================================
# [Step 02-1] Check the Step 02 checkpoint and prepare data
# ==============================================================================
CHECKPOINT_STEP01 = os.path.join(CKPT_DIR, 'mv1_step01.h5ad')
CHECKPOINT_STEP02 = os.path.join(CKPT_DIR, 'mv1_step02.h5ad')
MODELS_STEP02_PATH = os.path.join(CKPT_DIR, 'mv1_models_broad.pkl')
PROBS_STEP02_PATH  = os.path.join(CKPT_DIR, 'mv1_probs_broad.pkl')
TRAIN_LOG_STEP02_PATH = os.path.join(CKPT_DIR, 'mv1_step02_training_log.csv')
RESCUE_QC_STEP02_PATH = os.path.join(CKPT_DIR, 'mv1_step02_rescue_qc_cells.csv')
RESCUE_GUARDRAIL_STEP02_PATH = os.path.join(CKPT_DIR, 'mv1_step02_rescue_guardrail_summary.csv')
V9_TRANSFER_ROOT = os.path.join(ANALYSIS_DIR, 'checkpoints', 'transfer_models')
V9_EXPORT_FROZEN_MODELS = False  # Final v2 Figure-only: read V9 frozen models only.
V9_TRANSFER_OVERWRITE_EXISTING = False

# Execution modes
# - STEP02_FIGURE_ONLY=False: use Step01 results as input for Step02 analysis/correction.
# - STEP02_FIGURE_ONLY=True : load the saved completed Step02 data and regenerate figures only.
STEP02_FIGURE_ONLY = True
RUN_STEP02_TRAINING = False  # Use only saved training results in Figure-only mode.
RUN_STEP02_CV = False        # Step02-2 preflight CV. Skip by default because final Step02 predictions do not use it.
OOB_EVAL_EVERY = 1           # Step02-3 cumulative OOB B-Acc/F1 evaluation interval. Record every iteration when set to 1.
OOB_EVAL_EXTRA_ITERS = set() # When OOB_EVAL_EVERY > 1, e.g., {57} additionally prints/saves the iteration 57 OOB results.

skip_training = False
skip_rescue = False
apply_step02_predictions = True

if STEP02_FIGURE_ONLY:
    if not os.path.exists(CHECKPOINT_STEP02):
        raise FileNotFoundError(f"STEP02_FIGURE_ONLY=True but the Step02 checkpoint is missing: {CHECKPOINT_STEP02}")
    print("🔹 [Step 02] Figure-only mode → loading the saved completed Step02 data.")
    skip_training = True
    skip_rescue = True
    apply_step02_predictions = False
    _target_ckpt = CHECKPOINT_STEP02
elif RUN_STEP02_TRAINING:
    print("🔹 [Step 02] Analysis mode → loading Step01 results and running new ML training.")
    skip_training = False
    skip_rescue = False
    apply_step02_predictions = True
    _target_ckpt = CHECKPOINT_STEP01
elif os.path.exists(PROBS_STEP02_PATH):
    print("🔹 [Step 02] Analysis mode → loading Step01 results and reusing existing training logs.")
    skip_training = True
    skip_rescue = False
    apply_step02_predictions = True
    _target_ckpt = CHECKPOINT_STEP01
else:
    raise FileNotFoundError(
        "Step02 training logs are missing. To run the lengthy training, "
        "set RUN_STEP02_TRAINING=True and run it directly in the notebook."
    )

# Load data (always load if not already in memory)
if 'adata' not in dir() and 'adata' not in globals():
    adata = sc.read_h5ad(
        _target_ckpt,
        backed='r' if STEP02_FIGURE_ONLY else None,
    )
    print(f"🔹 Loading checkpoint: {os.path.basename(_target_ckpt)}")

required_obs_cols = ['Disease', 'PatientID', 'GT_Lineage_Broad', 'Run_Tissue_name', 'X_centroid', 'Y_centroid']
missing_obs_cols = [c for c in required_obs_cols if c not in adata.obs.columns]
if missing_obs_cols:
    raise KeyError(f"Required Step02 obs columns are missing: {missing_obs_cols}")

def _clean_obs_string(series):
    return series.astype('string').str.strip()

def make_exact_disease_mask(adata_obj, disease_name):
    disease_clean = _clean_obs_string(adata_obj.obs['Disease'])
    return disease_clean.eq(disease_name).fillna(False).to_numpy(dtype=bool)

def make_valid_label_mask(values):
    label_clean = pd.Series(values).astype('string').str.strip()
    invalid_mask = (
        label_clean.isna()
        | label_clean.eq('')
        | label_clean.str.fullmatch(r'(unknown|na|nan)', case=False, na=False)
    )
    return (~invalid_mask).to_numpy(dtype=bool), label_clean

STEP02_A4_LANDSCAPE = (11.69, 8.27)
STEP02_A4_PORTRAIT = (8.27, 11.69)
STEP02_A4_HALF_LANDSCAPE = (11.69, 5.85)
STEP02_CM_TO_IN = 1 / 2.54
STEP02_SAVE_PAD = 0.06
STEP02_FIGURE2AB_SIZE = (751.625 / 72, 565.737 / 72)
STEP02_FIGURE2C_A_SIZE = (650.284 / 72, 380.862 / 72)
STEP02_FIGURE2C_B_SIZE = (606.688 / 72, 422.850 / 72)
STEP02_RESCUE_EXAMPLE_SIZE_CM = (578.324 / 72 * 2.54, 227.981 / 72 * 2.54)
STEP02_OOB_PER_LINEAGE_SIZE = (760.567 / 72, 270.114 / 72)
# Preserve the individual panel ratio (3:2) without creating a fixed page canvas larger than A4.
STEP02_RESCUE_EXAMPLES_COMBINED_SIZE = (10.8, 7.8)
STEP02_RESCUE_CM_SIZE = (8.2, 4.6)

def _step02_save_publication_figure(fig_obj, png_path, pptx_path=None, label='figure',
                                    show=True, close=True, fixed_canvas=True,
                                    target_tight_width_cm=None):
    """Save a Step02 figure to PNG/PPTX using its original creation canvas."""
    if pptx_path is None:
        pptx_path = os.path.splitext(png_path)[0] + '.pptx'
    if target_tight_width_cm is not None:
        raise ValueError('target_tight_width_cm post-hoc scaling is prohibited. Set the design directly with figsize.')
    png_path, pptx_path = save_final_figure_pair(
        fig_obj, png_path, pptx_path, dpi=300,
        auxiliary_tight=False,
        validate=True,
    )
    print(f"  ✅ {label} PNG saved: {png_path}")
    print(f"  ✅ {label} PPTX saved: {pptx_path}")
    if show:
        plt.figure(fig_obj.number)
        plt.show()
    if close:
        plt.close(fig_obj)
    return png_path, pptx_path

def _step02_style_ticklabels(ax, x_size=11, y_size=11, x_rotation=0, y_rotation=0):
    """Standardize axis-label styles for Step02 figures."""
    plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha='right' if x_rotation else 'center',
             fontsize=x_size, fontweight='bold')
    plt.setp(ax.get_yticklabels(), rotation=y_rotation, fontsize=y_size, fontweight='bold')

def _step02_short_lineage_label(label):
    """Abbreviate only axis labels to prevent clipping of long broad-lineage labels in figures."""
    return str(label).replace(' cells', '').replace(' cell', '')


# ── Reviewed DLBCL ALE marker-response integration ────────────────────────────
# The retained review bundle is the only default source.  The calculation entry
# point below is deliberately independent from RUN_STEP02_TRAINING and can run
# only after two explicit environment opt-ins select a new, unused result root.
from pathlib import Path as _Path
import hashlib as _hashlib
import json as _json

STEP02_REVIEWED_ALE_FIGURE_ONLY = True
CONTINUE_STEP02_LEGACY_FIGURES = (
    os.environ.get('FINAL_STEP02_CONTINUE_LEGACY_FIGURES', '0') == '1'
)
RUN_STEP02_REVIEWED_ALE_RECALCULATION = (
    os.environ.get('FINAL_STEP02_ALE_RECALCULATE', '0') == '1'
)
STEP02_REVIEWED_ALE_NEW_OUTPUT_ROOT = os.environ.get(
    'FINAL_STEP02_ALE_NEW_OUTPUT_ROOT', ''
).strip()
STEP02_REVIEWED_ALE_ROOT = _Path(
    FINAL_TABLE_DIR, 'step02_dlbcl_ale_marker_response'
)
STEP02_REVIEWED_ALE_SOURCE_RUN = _Path(
    STEP02_REVIEWED_ALE_ROOT, 'source_attribution'
)
STEP02_REVIEWED_ALE_STEM = 'Final_mv1_Step02-01B_DLBCL_Broad_Marker_Response'
STEP02_REVIEWED_ALE_FIGURE_DIR = _Path(FIG_DIR)
STEP02_REVIEWED_ARIAL_REGULAR = _Path('__FONT_ROOT__/arial.ttf')
STEP02_REVIEWED_ARIAL_BOLD = _Path('__FONT_ROOT__/arialbd.ttf')


def _step02_require_arial():
    """Register and resolve the exact user-approved Arial files before plotting."""
    families = []
    for font_path in (STEP02_REVIEWED_ARIAL_REGULAR, STEP02_REVIEWED_ARIAL_BOLD):
        if not font_path.is_file() or font_path.stat().st_size <= 0:
            raise FileNotFoundError(f'Step02 reviewed Arial font file is missing: {font_path}')
        font_manager.fontManager.addfont(str(font_path))
        families.append(font_manager.FontProperties(fname=str(font_path)).get_name())
    if families != ['Arial', 'Arial']:
        raise RuntimeError(
            'Step02 reviewed Figure requires exact Arial Regular/Bold files; '
            f'resolved {families}'
        )
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
    })
    return tuple(map(str, (STEP02_REVIEWED_ARIAL_REGULAR, STEP02_REVIEWED_ARIAL_BOLD)))


def _step02_validate_completed_ale_bundle():
    """Validate the Final-local immutable copy without importing review source."""
    manifest_path = STEP02_REVIEWED_ALE_ROOT / 'manifests' / 'run_manifest.json'
    recovery_path = STEP02_REVIEWED_ALE_ROOT / 'manifests' / 'finalization_recovery_manifest.json'
    copy_manifest_path = STEP02_REVIEWED_ALE_ROOT / 'manifests' / 'final_copy_manifest.json'
    required_tables = (
        'ale_bin_support.csv', 'baseline_saved_annotation_agreement_audit.csv',
        'class_feature_marker_response_summary.csv', 'paired_iteration_ale_curve.csv.gz',
        'paired_iteration_class_feature_response.csv.gz',
        'paired_iteration_patient_bin_response.csv.gz',
        'selected_read_only_input_audit.csv', 'source_importance_frequency_join_audit.csv',
    )
    required = [
        manifest_path, recovery_path, copy_manifest_path,
        *(STEP02_REVIEWED_ALE_ROOT / 'tables' / name for name in required_tables),
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError('Missing Final-local Step02 ALE artifacts: ' + ', '.join(missing))
    run_manifest = _json.loads(manifest_path.read_text(encoding='utf-8'))
    recovery = _json.loads(recovery_path.read_text(encoding='utf-8'))
    if (
        run_manifest.get('retraining') is not False
        or run_manifest.get('source_models_read_only') is not True
        or run_manifest.get('source_checkpoints_read_only') is not True
        or run_manifest.get('direction_contract', {}).get('metric')
        != 'patient-equal first-order ALE q10-to-q90 contrast'
        or recovery.get('status') != 'complete'
        or recovery.get('calculation_status') != 'complete_100_of_100_pairs'
        or recovery.get('calculation_reexecuted') is not False
        or recovery.get('input_unchanged') is not True
        or recovery.get('validation', {}).get('all_pass') is not True
    ):
        raise RuntimeError('Final-local Step02 ALE completion contract failed')
    copy_manifest = _json.loads(copy_manifest_path.read_text(encoding='utf-8'))
    if copy_manifest.get('status') != 'complete':
        raise RuntimeError('Final-local Step02 ALE copy manifest is incomplete')
    copied_files = copy_manifest.get('files', {})
    expected_copy_entries = {
        'manifests/run_manifest.json',
        'manifests/finalization_recovery_manifest.json',
        *(f'tables/{name}' for name in required_tables),
        'source_attribution/manifests/run_manifest.json',
        'source_attribution/tables/class_feature_joint_importance_summary.csv',
        'source_attribution/tables/paired_iteration_top20_attribution_frequency.csv',
        'source_attribution/tables/selected_read_only_input_audit.csv',
    }
    if not expected_copy_entries.issubset(copied_files):
        raise RuntimeError(
            'Final-local Step02 ALE copy manifest lacks required entries: '
            f'{sorted(expected_copy_entries - set(copied_files))}'
        )
    for relative, audit in copied_files.items():
        copied = STEP02_REVIEWED_ALE_ROOT / relative
        if not copied.is_file() or copied.stat().st_size != int(audit['bytes']):
            raise RuntimeError(f'Final-local Step02 ALE copy size mismatch: {copied}')
        digest = _hashlib.sha256(copied.read_bytes()).hexdigest()
        if digest != audit['sha256']:
            raise RuntimeError(f'Final-local Step02 ALE copy hash mismatch: {copied}')
    return {
        'status': 'complete', 'run_root': str(STEP02_REVIEWED_ALE_ROOT),
        'required_tables': list(required_tables), 'copy_hashes_verified': True,
    }


def _step02_new_ale_output_root():
    """Require an unused, explicitly named Final/recalculations result directory."""
    if not RUN_STEP02_REVIEWED_ALE_RECALCULATION:
        raise RuntimeError('Set FINAL_STEP02_ALE_RECALCULATE=1 for an explicit new run')
    if not STEP02_REVIEWED_ALE_NEW_OUTPUT_ROOT:
        raise RuntimeError(
            'FINAL_STEP02_ALE_NEW_OUTPUT_ROOT must name a new unique output directory'
        )
    output_root = _Path(STEP02_REVIEWED_ALE_NEW_OUTPUT_ROOT).expanduser().resolve()
    allowed_parent = _Path(
        FINAL_WORKSPACE_DIR, 'recalculations', 'step02_dlbcl_ale_marker_response'
    ).resolve()
    if output_root.parent != allowed_parent:
        raise RuntimeError(
            f'New Step02 ALE output must be one direct child of {allowed_parent}: {output_root}'
        )
    if output_root.exists() or output_root.with_name(f'.{output_root.name}.staging').exists():
        raise FileExistsError(f'New Step02 ALE output/staging path already exists: {output_root}')
    return output_root


def _step02_build_reviewed_ale_calculation_config(output_root, execution_profile='fast'):
    """Build the frozen XGB+MLP ALE calculation without fitting any model."""
    from frozen_dlbcl_marker_response_utils import FrozenResponseStage, ResponseRunConfig

    stage = FrozenResponseStage(
        stage_key='broad',
        lineage='Broad lineage',
        checkpoint_path=_Path(CHECKPOINT_STEP02),
        model_root=_Path(FINAL_CHECKPOINT_ROOT, 'transfer_models', 'broad'),
        source_run_root=STEP02_REVIEWED_ALE_SOURCE_RUN,
        output_root=_Path(output_root),
        route_column=None,
        route_value=None,
        saved_prediction_column='ML_Lineage_Pure',
        expected_classes=('B cells', 'Myeloid cells', 'NK cells', 'Stromal cells', 'T cells'),
    )
    return ResponseRunConfig(
        workflow='Step02 frozen XGB+MLP DLBCL broad marker response',
        stages=(stage,),
        execution_profile=execution_profile,
    )


def _step02_run_new_reviewed_ale_calculation():
    """Explicit future-only ALE path; never reuses or overwrites a completed root."""
    output_root = _step02_new_ale_output_root()
    from frozen_dlbcl_marker_response_utils import run

    profile = os.environ.get('FINAL_STEP02_ALE_EXECUTION_PROFILE', 'fast')
    config = _step02_build_reviewed_ale_calculation_config(output_root, profile)
    caller_path = _Path(globals().get(
        '__file__', _Path(FINAL_WORKSPACE_DIR, 'scripts', 'Final_step02.py')
    )).resolve()
    return run(config, caller_paths=(caller_path,))


def _step02_reviewed_bubble_size(percent):
    return 28.0 + 3.2 * max(0.0, float(percent))


def _step02_reviewed_rgba(color, alpha):
    red, green, blue = matplotlib.colors.to_rgb(color)
    return red, green, blue, float(alpha)


def _step02_assert_arial_text(fig_obj):
    fig_obj.canvas.draw()
    resolved = {
        text.get_fontproperties().get_name()
        for axis in fig_obj.axes
        for text in [*axis.texts, *axis.get_xticklabels(), *axis.get_yticklabels()]
        if text.get_visible() and text.get_text().strip()
    }
    if not resolved or resolved != {'Arial'}:
        raise RuntimeError(f'Step02 reviewed Figure font contract failed: {sorted(resolved)}')
    return sorted(resolved)


def _step02_save_reviewed_pair(fig_obj, png_path, pptx_path, *, show=False):
    """Save the reviewed pair through the common transactional pair helper."""
    png_path = _Path(png_path)
    pptx_path = _Path(pptx_path)
    expected_parent = STEP02_REVIEWED_ALE_FIGURE_DIR.resolve()
    if png_path.resolve().parent != expected_parent or pptx_path.resolve().parent != expected_parent:
        raise RuntimeError('Step02 reviewed Figures must stay directly in figures/v2')
    png_path.parent.mkdir(parents=True, exist_ok=True)
    saved_png_path, saved_pptx_path = save_final_figure_pair(
        fig_obj, png_path, pptx_path,
        dpi=300, facecolor='white', auxiliary_tight=False, validate=True,
        show=show, close=True,
    )
    return saved_png_path, saved_pptx_path


def _render_step02_reviewed_ale_bubble(show=True):
    """Render the reviewed broad ALE Bubble from the completed table only."""
    if not STEP02_REVIEWED_ALE_FIGURE_ONLY:
        return None, None
    if not STEP02_FIGURE_ONLY or RUN_STEP02_TRAINING:
        raise RuntimeError('Reviewed Step02 ALE Figure requires the Figure-only mode')
    if not _final_figure_stem_registered(STEP02_REVIEWED_ALE_STEM):
        raise RuntimeError(
            f'Step02 canonical Figure stem is not registered: {STEP02_REVIEWED_ALE_STEM}'
        )
    _step02_validate_completed_ale_bundle()
    _step02_require_arial()

    summary_path = STEP02_REVIEWED_ALE_ROOT / 'tables' / 'class_feature_marker_response_summary.csv'
    table = pd.read_csv(summary_path)
    required = {
        'stage', 'class', 'feature', 'feature_position', 'direction_symbol',
        'figure_alpha', 'class_normalized_importance_percent',
        'positive_top_k_attribution_frequency_percent', 'response_metric',
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f'Step02 reviewed ALE summary lacks Figure columns: {missing}')
    if set(table['response_metric'].astype(str)) != {
        'patient-equal first-order ALE q10-to-q90 contrast'
    }:
        raise RuntimeError('Step02 reviewed ALE direction metric changed')

    class_order = ('B cells', 'T cells', 'Myeloid cells', 'Stromal cells', 'NK cells')
    colors = {
        'B cells': '#2471A3', 'T cells': '#C0392B', 'Myeloid cells': '#6C3483',
        'Stromal cells': '#27AE60', 'NK cells': '#D4AC0D',
    }
    selected = {}
    for class_name in class_order:
        rows = table.loc[
            table['stage'].astype(str).eq('broad')
            & table['class'].astype(str).eq(class_name)
            & np.isfinite(pd.to_numeric(table['class_normalized_importance_percent'], errors='coerce'))
            & pd.to_numeric(table['class_normalized_importance_percent'], errors='coerce').gt(0)
        ].copy()
        rows = rows.sort_values(
            ['class_normalized_importance_percent', 'feature_position'],
            ascending=[False, True], kind='mergesort',
        ).head(15)
        if rows.empty:
            raise RuntimeError(f'No reviewed Step02 ALE rows for {class_name}')
        selected[class_name] = rows.reset_index(drop=True)

    fig = plt.figure(figsize=(17.2, 12.4), facecolor='white')
    grid = fig.add_gridspec(
        3, 3, left=0.070, right=0.985, top=0.965, bottom=0.070,
        height_ratios=[1.0, 1.0, 0.14], wspace=0.30, hspace=0.22,
    )
    for panel_index, class_name in enumerate(class_order):
        plot_row, plot_column = divmod(panel_index, 3)
        axis = fig.add_subplot(grid[plot_row, plot_column])
        rows = selected[class_name]
        y = np.arange(len(rows), dtype=float)
        for row_y in y:
            axis.axhline(row_y, color='#D3D3D3', linewidth=0.28, alpha=0.48, zorder=0)
        axis.scatter(
            rows['class_normalized_importance_percent'], y,
            s=[_step02_reviewed_bubble_size(v) for v in rows['positive_top_k_attribution_frequency_percent']],
            c=[_step02_reviewed_rgba(colors[class_name], a) for a in rows['figure_alpha']],
            edgecolors='#111111', linewidths=0.72, zorder=3,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(
            [f"{m} {s}" for m, s in zip(rows['feature'], rows['direction_symbol'])],
            fontsize=9.2, fontweight='bold', color=colors[class_name],
        )
        axis.set_ylim(len(rows) - 0.35, -0.65)
        panel_max = float(rows['class_normalized_importance_percent'].max())
        panel_right = max(5.0, panel_max * 1.10)
        axis.set_xlim(-0.045 * panel_right, panel_right)
        locator = matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        axis.set_xticks([
            value for value in locator.tick_values(0.0, panel_right)
            if 0.0 <= value <= panel_right
        ])
        axis.axvline(0.0, color='#777777', linewidth=0.9, linestyle='--', zorder=1)
        axis.set_title(class_name, fontsize=15.5, fontweight='bold', pad=9)
        if plot_row == 1:
            axis.set_xlabel('Permutation importance (%)', fontsize=12.5, fontweight='bold')
        if plot_column == 0:
            axis.set_ylabel('Lineage / marker', fontsize=15.0, fontweight='bold')
        for spine in ('top', 'right'):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis='x', labelsize=9.5)

    empty_axis = fig.add_subplot(grid[1, 2])
    empty_axis.set_axis_off()
    legend_axis = fig.add_subplot(grid[2, :])
    legend_axis.set_axis_off()
    legend_axis.set_xlim(0.0, 1.0)
    legend_axis.set_ylim(0.0, 1.0)
    legend_axis.text(0.29, 0.82, 'Attribution frequency', ha='center', va='center',
                     fontsize=12.5, fontweight='bold')
    for xpos, value in zip(np.linspace(0.19, 0.39, 4), (25, 50, 75, 100)):
        legend_axis.scatter(xpos, 0.50, s=_step02_reviewed_bubble_size(value),
                            facecolor='#A7A7A7', edgecolor='#4F4F4F', linewidth=0.75)
        legend_axis.text(xpos, 0.20, f'{value}%', ha='center', va='center', fontsize=9.5)
    legend_axis.text(0.72, 0.82, 'Direction consistency', ha='center', va='center',
                     fontsize=12.5, fontweight='bold')
    for xpos, (label, alpha) in zip(np.linspace(0.65, 0.79, 3), ((80, .45), (90, .70), (100, .95))):
        legend_axis.scatter(xpos, 0.50, s=_step02_reviewed_bubble_size(65),
                            facecolor=_step02_reviewed_rgba('#777777', alpha),
                            edgecolor='#3F3F3F', linewidth=0.75)
        legend_axis.text(xpos, 0.20, f'{label}%', ha='center', va='center', fontsize=9.5)
    legend_axis.text(
        0.50, -0.12,
        '+ higher marker level increases class probability; − decreases; '
        '± nonmonotonic or unstable.',
        ha='center', va='center', fontsize=9.0,
    )
    _step02_assert_arial_text(fig)

    png_path = STEP02_REVIEWED_ALE_FIGURE_DIR / (STEP02_REVIEWED_ALE_STEM + '.png')
    pptx_path = STEP02_REVIEWED_ALE_FIGURE_DIR / (STEP02_REVIEWED_ALE_STEM + '.pptx')
    return _step02_save_reviewed_pair(fig, png_path, pptx_path, show=show)


if RUN_STEP02_REVIEWED_ALE_RECALCULATION:
    raise RuntimeError(
        'Step02 ALE is a protected main calculation and cannot run from the Final '
        'canonical workflow. Keep the retained recalculation code for a separately '
        'reviewed, explicitly authorized execution path.'
    )
elif STEP02_REVIEWED_ALE_FIGURE_ONLY:
    _step02_reviewed_pair = _render_step02_reviewed_ale_bubble(show=False)
    _step02_reviewed_saved_count = int(all(_step02_reviewed_pair))
    print(
        '✅ Step02 reviewed ALE Figure-only workflow: '
        f'saved_pairs={_step02_reviewed_saved_count}'
    )
    if not CONTINUE_STEP02_LEGACY_FIGURES:
        print('🔹 Step02 legacy Figure blocks were not run.')
        raise SystemExit(0)
    # The legacy broad-importance panel and the unique OOB performance panel
    # share one Figure 2A-B canvas and one save call, so suppressing only the
    # duplicate importance panel is not safely separable.  The composite is
    # therefore retained exclusively behind the explicit legacy opt-in.
    print(
        '  ⚠️ Step02 legacy Figure 2A-B remains opt-in: its broad feature-importance '
        'panel cannot be suppressed without also suppressing the OOB performance panel.'
    )

# ── Data preparation (shared by training and manual application) ──
dlbcl_mask  = make_exact_disease_mask(adata, 'DLBCL')
ref_mask    = ~dlbcl_mask
broad_ref_raw  = adata.obs['GT_Lineage_Broad'][ref_mask].values
valid_ref_mask, broad_ref_clean = make_valid_label_mask(broad_ref_raw)
y_ref_v = broad_ref_clean[valid_ref_mask].astype(str).to_numpy()
le_broad = LabelEncoder()
y_enc_v  = le_broad.fit_transform(y_ref_v)
groups_ref = adata.obs['PatientID'][ref_mask][valid_ref_mask].values
if STEP02_FIGURE_ONLY:
    # Figure-only panels use saved logs/tables plus small marker slices. Avoid
    # materializing the complete multi-million-cell expression matrix.
    X_full = X_ref = X_ref_v = X_dlbcl = None
else:
    X_full = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.astype(np.float32)
    X_ref = X_full[ref_mask]
    X_dlbcl = X_full[dlbcl_mask]
    X_ref_v = X_ref[valid_ref_mask]

if 'ML_Lineage_Raw' not in adata.obs.columns:
    adata.obs['ML_Lineage_Raw'] = adata.obs['GT_Lineage_Broad'].copy()

print(f"  → [Ready] Reference: {int(valid_ref_mask.sum()):,} training samples | DLBCL: {int(dlbcl_mask.sum()):,} prediction targets")
print(f"  → Classes: {le_broad.classes_}")
print(f"  → Number of patients: {len(np.unique(groups_ref))}")

def append_per_class_metric_rows(rows, iter_idx, eval_type, name, y_true, y_pred, class_names, rank=np.nan, oob_n=None):
    """Record one-vs-rest performance for each broad lineage at each iteration."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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
            'iter': int(iter_idx),
            'eval_type': eval_type,
            'name': name,
            'rank': rank,
            'lineage_class': str(cls_name),
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
            'timestamp': pd.Timestamp.now().isoformat()
        })

def record_class_feature_importance(log_dict, rows, iter_idx, model_name, class_names, feature_names, importances_by_class):
    """Save class-specific feature importance in raw and normalized forms."""
    arr = np.asarray(importances_by_class, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(class_names):
        return

    for cls_idx, cls_name in enumerate(class_names):
        vals = np.nan_to_num(arr[cls_idx], nan=0.0, posinf=0.0, neginf=0.0)
        abs_vals = np.abs(vals)
        total_abs = float(abs_vals.sum())
        record = {fname: float(val) for fname, val in zip(feature_names, vals)}
        log_dict.setdefault(model_name, {}).setdefault(str(cls_name), []).append(record)

        ranked = sorted(zip(feature_names, vals, abs_vals), key=lambda x: x[2], reverse=True)
        for rank, (fname, raw_val, abs_val) in enumerate(ranked, 1):
            rows.append({
                'iter': int(iter_idx),
                'model': model_name,
                'lineage_class': str(cls_name),
                'feature': fname,
                'importance': float(raw_val),
                'abs_importance': float(abs_val),
                'normalized_importance': float(abs_val / total_abs) if total_abs > 0 else 0.0,
                'rank': int(rank),
                'timestamp': pd.Timestamp.now().isoformat()
            })

def extract_xgb_class_feature_importance(xgb_model, feature_names, n_classes):
    """
    Compute gain-based importance by dividing XGBoost multiclass trees into class-specific tree groups.
    In the default one_output_per_tree structure, tree index % n_classes corresponds to the class index.
    """
    out = np.zeros((n_classes, len(feature_names)), dtype=np.float32)
    feature_to_idx = {f'f{i}': i for i in range(len(feature_names))}
    feature_to_idx.update({name: i for i, name in enumerate(feature_names)})
    try:
        booster_df = xgb_model.get_booster().trees_to_dataframe()
        if booster_df.empty or 'Feature' not in booster_df.columns:
            return out
        split_df = booster_df[booster_df['Feature'] != 'Leaf']
        if split_df.empty:
            return out
        for _, row in split_df.iterrows():
            feat = row['Feature']
            if feat not in feature_to_idx:
                continue
            cls_idx = int(row['Tree']) % n_classes
            gain = float(row['Gain']) if pd.notna(row.get('Gain', np.nan)) else 0.0
            out[cls_idx, feature_to_idx[feat]] += gain
    except Exception as e:
        print(f"  ⚠️ Failed to extract XGB class-specific importance: {e}")
    return out

def summarize_class_feature_importance(class_fi_rows):
    """Convert class-specific importance row logs into a mean summary table."""
    if not class_fi_rows:
        return pd.DataFrame()
    class_fi_df = pd.DataFrame(class_fi_rows)
    summary = (
        class_fi_df
        .groupby(['model', 'lineage_class', 'feature'], as_index=False)
        .agg(
            mean_importance=('importance', 'mean'),
            mean_abs_importance=('abs_importance', 'mean'),
            mean_normalized_importance=('normalized_importance', 'mean'),
            n_records=('iter', 'nunique')
        )
    )
    summary['rank_in_model_class'] = (
        summary
        .groupby(['model', 'lineage_class'])['mean_abs_importance']
        .rank(method='dense', ascending=False)
        .astype(int)
    )
    return summary

# %%
# ==============================================================================
# [Step 02-2] CV validation (4-model ensemble, GroupShuffleSplit 7:3)
# ==============================================================================
if not skip_training and RUN_STEP02_CV:
    print("🔹 [Step 02-2] Starting CV validation (4 models, GroupShuffleSplit 7:3)...")

    # Sample data for CV (300,000 per class)
    sidx = stratified_sample(X_ref_v, y_enc_v, n_samples_per_class=CELLS_PER_CLASS, random_state=RANDOM_SEED)
    X_cv = X_ref_v[sidx]
    y_cv = y_enc_v[sidx]
    groups_cv = groups_ref[sidx]
    n_cls = len(np.unique(y_cv))

    gss = GroupShuffleSplit(n_splits=10, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED)
    cv_ba_scores = []

    pbar = tqdm(enumerate(gss.split(X_cv, y_cv, groups=groups_cv)), total=10, desc="CV Iterations (GSS 7:3)")

    for fold_idx, (tr, te) in pbar:
        print(f"\n{'='*50}\n🚀 Starting FOLD {fold_idx+1}/10", flush=True)

        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        # Scale data
        scaler = StandardScaler()
        X_tr_cpu = np.ascontiguousarray(scaler.fit_transform(X_cv[tr]), dtype=np.float32)
        X_te_cpu = np.ascontiguousarray(scaler.transform(X_cv[te]), dtype=np.float32)
        y_tr_cpu = np.ascontiguousarray(y_cv[tr])

        # Load onto the GPU
        X_tr_gpu = cp.asarray(X_tr_cpu)
        X_te_gpu = cp.asarray(X_te_cpu)

        # ── 1. RF (cuML GPU) ──
        pbar.set_description(f"Fold {fold_idx+1}/10 | 🌲 RF (GPU)")
        rf_cv = cuRF(random_state=RANDOM_SEED)  # Default parameters
        rf_cv.fit(X_tr_gpu, cp.asarray(y_tr_cpu, dtype=cp.int32))
        p_rf = rf_cv.predict_proba(X_te_gpu)
        if hasattr(p_rf, 'get'): p_rf = p_rf.get()
        elif isinstance(p_rf, cp.ndarray): p_rf = cp.asnumpy(p_rf)
        del rf_cv; cp.get_default_memory_pool().free_all_blocks()

        # ── 2. XGBoost (GPU) ──
        pbar.set_description(f"Fold {fold_idx+1}/10 | 🌳 XGB (GPU)")
        xgb_cv = xgb.XGBClassifier(
            tree_method='hist', device='cuda', random_state=RANDOM_SEED, verbosity=0
        )
        xgb_cv.fit(X_tr_cpu, y_tr_cpu.astype(np.int32))
        p_xgb = xgb_cv.predict_proba(X_te_cpu)
        del xgb_cv; cp.get_default_memory_pool().free_all_blocks()

        # ── 3. ElasticNet (CPU SGD) ──
        pbar.set_description(f"Fold {fold_idx+1}/10 | 📏 ENet (CPU)")
        enet_cv = SGDClassifier(
            loss='log_loss', penalty='elasticnet', l1_ratio=0.5, random_state=RANDOM_SEED
        )
        enet_cv.fit(X_tr_cpu, y_tr_cpu)
        p_enet = enet_cv.predict_proba(X_te_cpu)
        del enet_cv; gc.collect()

        # ── 4. MLP (GPU subprocess) ──
        pbar.set_description(f"Fold {fold_idx+1}/10 | 🧠 MLP (GPU)")
        p_mlp = spawn_mlp(
            np.ascontiguousarray(X_tr_cpu),
            np.ascontiguousarray(y_tr_cpu, dtype=np.int64),
            X_te_cpu, n_cls,
            random_seed=RANDOM_SEED
        )

        # ── 5. Aggregate results ──
        avg_probs = np.mean([p_rf, p_xgb, p_enet, p_mlp], axis=0)
        ba = balanced_accuracy_score(y_cv[te], np.argmax(avg_probs, axis=1))
        cv_ba_scores.append(ba)

        print(f"  ✅ Fold {fold_idx+1}/10 completed | B-Acc: {ba*100:.2f}%", flush=True)
        pbar.set_postfix(B_Acc=f"{ba*100:.2f}%")

        del p_rf, p_xgb, p_enet, p_mlp, avg_probs
        del X_tr_cpu, X_te_cpu, y_tr_cpu, X_tr_gpu, X_te_gpu
        gc.collect()

    print(f"\n📊 [CV results] Mean B-Acc: {np.mean(cv_ba_scores)*100:.2f}% ± {np.std(cv_ba_scores)*100:.2f}%")
elif not skip_training:
    print("🔹 [Step 02-2] Skipping CV validation: RUN_STEP02_CV=False (no effect on final Step02 predictions)")

# %%
# ==============================================================================
# [Step 02-3] Main training: full 100-iteration loop of the cumulative 4-model ensemble + Feature Importance
# ==============================================================================
if not skip_training:
    print(f"\n🔹 [Step 02-3] Starting main training | N_ITER={N_ITER}, models={MODEL_KEYS}")
    print(f"  ⏱️ Start time: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    n_cls = len(np.unique(y_enc_v))
    feature_names_list = adata.var_names.tolist()

    broad_transfer_stage = None
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
        broad_transfer_stage = prepare_transfer_stage(
            V9_TRANSFER_ROOT,
            stage='broad',
            feature_names=feature_names_list,
            classes=le_broad.classes_,
            expected_iterations=N_ITER,
            final_combo=None,
            overwrite=V9_TRANSFER_OVERWRITE_EXISTING,
            metadata={
                'source_code': 'Final_step02.py',
                'random_seed': RANDOM_SEED,
                'cv_test_size': CV_TEST_SIZE,
                'model_selection': 'V8/Final manual combo and iteration selection',
            },
        )
        print(f"  💾 Broad frozen-model save location: {broad_transfer_stage}")

    # Global Feature Importance store
    fi_log_broad = {k: [] for k in MODEL_KEYS}
    # Store XGB/ENet class-specific importance and per-class OOB performance only within the pkl.
    class_fi_log_broad = {k: {str(cls): [] for cls in le_broad.classes_} for k in ['XGB', 'ENet']}
    class_fi_rows = []

    # Accumulate model DLBCL probability predictions (for final Soft Voting)
    all_probs = {k: [] for k in MODEL_KEYS}
    
    # Store cumulative OOB (Out-Of-Bag) predicted probabilities and counts
    oob_probs  = {k: np.zeros((X_ref_v.shape[0], n_cls), dtype=np.float32) for k in MODEL_KEYS}
    oob_counts = {k: np.zeros(X_ref_v.shape[0], dtype=np.int32) for k in MODEL_KEYS}

    # Score-tracking variables
    best_val_score   = -1.0     
    best_val_combo   = ''       
    actual_iters     = 0
    
    # Ensemble performance visualization history (used in Step 02-3.5)
    # Record B-Acc, F1, and Avg for each model (RF, XGB, ENet, MLP) and the Best combination
    if 'history_broad' not in globals():
        global history_broad
    history_broad = {
        'iters': [],
        'best': {'bacc': [], 'f1': [], 'avg': [], 'combo': []},
        **{k: {'bacc': [], 'f1': []} for k in MODEL_KEYS}
    }
    training_log_rows = []
    per_class_metric_rows = []

    # Precompute the 15 combinations
    _es_combos = []
    for _r in range(1, len(MODEL_KEYS) + 1):
        for _c in _combs(MODEL_KEYS, _r):
            _es_combos.append(_c)
    _es_combo_names = ['+'.join(c) for c in _es_combos]

    # Preload DLBCL prediction data onto the GPU
    X_dlbcl_contig = np.ascontiguousarray(X_dlbcl, dtype=np.float32)

    train_pbar = tqdm(range(N_ITER), desc="Broad Ensemble Training")
    for i in train_pbar:
        actual_iters = i + 1

        # 7:3 data split (by Patient ID, with a different seed at each iteration)
        gss_train = GroupShuffleSplit(n_splits=1, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED + i)
        for tr_idx, val_idx in gss_train.split(X_ref_v, y_enc_v, groups=groups_ref):
            pass  # Split only once

        # Scaling (fit on train, then transform val/dlbcl)
        scaler_iter = StandardScaler()
        X_tr      = np.ascontiguousarray(scaler_iter.fit_transform(X_ref_v[tr_idx]), dtype=np.float32)
        X_val     = np.ascontiguousarray(scaler_iter.transform(X_ref_v[val_idx]),    dtype=np.float32)
        X_dlbcl_sc = np.ascontiguousarray(scaler_iter.transform(X_dlbcl_contig),    dtype=np.float32)
        y_tr  = y_enc_v[tr_idx]
        y_val = y_enc_v[val_idx]

        if broad_transfer_stage is not None:
            try:
                save_iteration_scaler(
                    broad_transfer_stage,
                    i + 1,
                    scaler_iter,
                    random_seed=RANDOM_SEED + i,
                    train_n=len(tr_idx),
                    validation_n=len(val_idx),
                    train_patient_n=len(np.unique(groups_ref[tr_idx])),
                    validation_patient_n=len(np.unique(groups_ref[val_idx])),
                )
            except Exception as e:
                print(f"  ⚠️ Failed to save the Iter {i+1} scaler (training continues): {e}")

        # ── Debug log: print training progress at each iteration ──
        _vram = cp.get_default_memory_pool().used_bytes() / (1024**2)
        print(f"\n  {'='*55}")
        print(f"  📊 Iter {i+1}/{N_ITER} | Train: {len(tr_idx):,} | Val: {len(val_idx):,} | "
              f"DLBCL: {len(X_dlbcl_contig):,} | VRAM: {_vram:.0f}MB | ⏱️ {pd.Timestamp.now().strftime('%H:%M:%S')}")
        print(f"     Patients(Train): {len(np.unique(groups_ref[tr_idx]))} | "
              f"Patients(Val): {len(np.unique(groups_ref[val_idx]))} | "
              f"Class distribution (Train): {dict(zip(*np.unique(y_tr, return_counts=True)))}")

        # ── 🌲 RF (cuML GPU) ──
        print(f"    🌲 Training RF...", end=' ', flush=True)
        try:
            X_tr_gpu    = cp.asarray(X_tr)
            y_tr_gpu    = cp.asarray(y_tr, dtype=cp.int32)
            X_dlbcl_gpu = cp.asarray(X_dlbcl_sc)
            X_val_gpu   = cp.asarray(X_val)

            rf_clf = cuRF(random_state=RANDOM_SEED + i)
            rf_clf.fit(X_tr_gpu, y_tr_gpu)

            p_rf = rf_clf.predict_proba(X_dlbcl_gpu)
            if hasattr(p_rf, 'get'): p_rf = p_rf.get()
            p_rf_val = rf_clf.predict_proba(X_val_gpu)
            if hasattr(p_rf_val, 'get'): p_rf_val = p_rf_val.get()

            # Record Feature Importance
            if hasattr(rf_clf, 'feature_importances_'):
                fi_vals = rf_clf.feature_importances_
                if hasattr(fi_vals, 'get'): fi_vals = fi_vals.get()
                record_feature_importance(fi_log_broad, 'RF', feature_names_list, fi_vals)

            all_probs['RF'].append(p_rf)
            oob_probs['RF'][val_idx] += p_rf_val
            oob_counts['RF'][val_idx] += 1
            if broad_transfer_stage is not None:
                try:
                    save_joblib_model_iteration(
                        broad_transfer_stage, i + 1, 'RF', rf_clf
                    )
                except Exception as e:
                    print(f"⚠️ Failed to save RF model (predictions retained): {e}", end=' ')
            print("✅", flush=True)
            del rf_clf, X_tr_gpu, y_tr_gpu, X_dlbcl_gpu, X_val_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"  ⚠️ RF iteration {i+1} failed: {e}")

        # ── 🌳 XGBoost (GPU, default hyperparameters) ──
        print(f"    🌳 Training XGB...", end=' ', flush=True)
        try:
            xgb_clf = xgb.XGBClassifier(
                tree_method='hist', device='cuda', random_state=RANDOM_SEED + i, verbosity=0
            )
            xgb_clf.fit(X_tr, y_tr.astype(np.int32))

            p_xgb     = xgb_clf.predict_proba(X_dlbcl_sc)
            p_xgb_val = xgb_clf.predict_proba(X_val)

            # Record Feature Importance
            record_feature_importance(fi_log_broad, 'XGB', feature_names_list, xgb_clf.feature_importances_)
            xgb_class_fi = extract_xgb_class_feature_importance(xgb_clf, feature_names_list, n_cls)
            record_class_feature_importance(
                class_fi_log_broad, class_fi_rows, i + 1, 'XGB',
                le_broad.classes_, feature_names_list, xgb_class_fi
            )

            all_probs['XGB'].append(p_xgb)
            oob_probs['XGB'][val_idx] += p_xgb_val
            oob_counts['XGB'][val_idx] += 1
            if broad_transfer_stage is not None:
                try:
                    save_xgb_iteration(broad_transfer_stage, i + 1, xgb_clf)
                except Exception as e:
                    print(f"⚠️ Failed to save XGB model (predictions retained): {e}", end=' ')
            print("✅", flush=True)
            del xgb_clf
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"  ⚠️ XGB iteration {i+1} failed: {e}")

        # ── 📏 ElasticNet (CPU SGD) ──
        print(f"    📏 Training ENet...", end=' ', flush=True)
        try:
            enet_clf = SGDClassifier(
                loss='log_loss', penalty='elasticnet', l1_ratio=0.5, random_state=RANDOM_SEED + i
            )
            enet_clf.fit(X_tr, y_tr)
            p_enet     = enet_clf.predict_proba(X_dlbcl_sc)
            p_enet_val = enet_clf.predict_proba(X_val)

            # Record Feature Importance (mean absolute coef_)
            fi_enet = np.abs(enet_clf.coef_).mean(axis=0)
            record_feature_importance(fi_log_broad, 'ENet', feature_names_list, fi_enet)
            enet_class_fi = np.abs(enet_clf.coef_)
            record_class_feature_importance(
                class_fi_log_broad, class_fi_rows, i + 1, 'ENet',
                le_broad.classes_, feature_names_list, enet_class_fi
            )

            all_probs['ENet'].append(p_enet)
            oob_probs['ENet'][val_idx] += p_enet_val
            oob_counts['ENet'][val_idx] += 1
            if broad_transfer_stage is not None:
                try:
                    save_joblib_model_iteration(
                        broad_transfer_stage, i + 1, 'ENet', enet_clf
                    )
                except Exception as e:
                    print(f"⚠️ Failed to save ENet model (predictions retained): {e}", end=' ')
            print("✅", flush=True)
            del enet_clf; gc.collect()
        except Exception as e:
            print(f"  ⚠️ ENet iteration {i+1} failed: {e}")

        # ── 🧠 MLP (GPU subprocess) ──
        print(f"    🧠 Training MLP...", end=' ', flush=True)
        try:
            # Like RF/XGB/ENet, use the same trained MLP model to predict both DLBCL and validation data.
            _mlp_pred_n_dlbcl = X_dlbcl_sc.shape[0]
            X_mlp_pred = np.ascontiguousarray(
                np.vstack([X_dlbcl_sc, X_val]),
                dtype=np.float32
            )
            _mlp_model_output = (
                mlp_iteration_path(broad_transfer_stage, i + 1)
                if broad_transfer_stage is not None else None
            )
            p_mlp_all = spawn_mlp_predict_batched(
                np.ascontiguousarray(X_tr, dtype=np.float32),
                np.ascontiguousarray(y_tr, dtype=np.int64),
                X_mlp_pred, n_cls,
                model_output_path=_mlp_model_output,
                random_seed=RANDOM_SEED + i
            )
            p_mlp = p_mlp_all[:_mlp_pred_n_dlbcl]
            p_mlp_val = p_mlp_all[_mlp_pred_n_dlbcl:]
            all_probs['MLP'].append(p_mlp)
            oob_probs['MLP'][val_idx] += p_mlp_val
            oob_counts['MLP'][val_idx] += 1
            if broad_transfer_stage is not None:
                try:
                    register_mlp_iteration(
                        broad_transfer_stage, i + 1, _mlp_model_output
                    )
                except Exception as e:
                    print(f"⚠️ Failed to register MLP model (predictions retained): {e}", end=' ')
            print("✅", flush=True)
            del X_mlp_pred, p_mlp_all
            gc.collect()
        except Exception as e:
            print(f"  ⚠️ MLP iteration {i+1} failed: {e}")

        # ── Evaluate cumulative OOB at the specified interval (no early stopping) ──
        should_eval_oob = (
            ((i + 1) % OOB_EVAL_EVERY == 0)
            or ((i + 1) == N_ITER)
            or ((i + 1) in OOB_EVAL_EXTRA_ITERS)
        )
        if not should_eval_oob:
            print(f"    🔍 [Iter {i+1}] Skipping OOB performance evaluation (OOB_EVAL_EVERY={OOB_EVAL_EVERY})", flush=True)
            # Final predicted probabilities (all_probs) and cumulative OOB arrays are already updated, so results are unaffected
            del X_tr, X_val, X_dlbcl_sc, y_tr, y_val
            gc.collect()
            continue

        print(f"    🔍 [Iter {i+1}] Evaluating cumulative ensemble performance through iteration {i+1}...", flush=True)
            
        # Extract only samples used as Val data at least once by at least one model
        valid_mask = np.zeros(X_ref_v.shape[0], dtype=bool)
        for _mk in MODEL_KEYS:
            valid_mask |= oob_counts[_mk] > 0
        if np.sum(valid_mask) > 0:
            combo_results = []
            for combo, cname in zip(_es_combos, _es_combo_names):
                combo_valid_mask = valid_mask.copy()
                for k in combo:
                    combo_valid_mask &= oob_counts[k] > 0
                if np.sum(combo_valid_mask) == 0:
                    continue

                y_combo_true = y_enc_v[combo_valid_mask]
                combo_probs_list = []
                for k in combo:
                    # Cumulative probability / cumulative count = mean ensemble probability
                    p_k_oob = oob_probs[k][combo_valid_mask] / oob_counts[k][combo_valid_mask, np.newaxis]
                    combo_probs_list.append(p_k_oob)
                
                combo_val_avg  = np.mean(combo_probs_list, axis=0)
                combo_val_pred = np.argmax(combo_val_avg, axis=1)
                
                combo_ba = balanced_accuracy_score(y_combo_true, combo_val_pred)
                combo_f1 = f1_score(y_combo_true, combo_val_pred, average='macro', zero_division=0)
                combo_avg = (combo_ba + combo_f1) / 2
                
                combo_results.append({
                    'Combo': cname, 'Avg': combo_avg, 'B-Acc': combo_ba, 'F1': combo_f1,
                    'OOB_N': int(np.sum(combo_valid_mask))
                })
            if not combo_results:
                print(f"  ⚠️ [Iter {i+1}] No OOB combinations are available for evaluation.")
                del X_tr, X_val, X_dlbcl_sc, y_tr, y_val
                gc.collect()
                continue
            
            # Sort rankings in descending order
            combo_results = sorted(combo_results, key=lambda x: x['Avg'], reverse=True)
            
            # Print the Top 5
            print(f"  🏆 [Iter {i+1} cumulative ensemble Top 5] (evaluable cumulative OOB samples: {np.sum(valid_mask):,})")
            for rank, res in enumerate(combo_results[:5], 1):
                prefix = "🌟" if rank == 1 else "  "
                print(f"    {prefix} Rank {rank}: {res['Combo']:<20} | Avg: {res['Avg']*100:.2f}% "
                      f"(B-Acc: {res['B-Acc']*100:.1f}%, F1: {res['F1']*100:.1f}%, OOB n={res['OOB_N']:,})")
            
            # Update the highest-score ranking (for application of final results)
            top_res = combo_results[0]
            if top_res['Avg'] > best_val_score:
                best_val_score = top_res['Avg']
                best_val_combo = top_res['Combo']
                
            # Record cumulative history (for Step 02-3.5 plots)
            history_broad['iters'].append(i+1)
            # Best ensemble combination performance
            history_broad['best']['bacc'].append(top_res['B-Acc'])
            history_broad['best']['f1'].append(top_res['F1'])
            history_broad['best']['avg'].append(top_res['Avg'])
            history_broad['best']['combo'].append(top_res['Combo'])
            
            single_model_scores = []
            for _mk in MODEL_KEYS:
                _mk_mask = oob_counts[_mk] > 0
                if np.sum(_mk_mask) > 0:
                    _mk_probs = oob_probs[_mk][_mk_mask] / oob_counts[_mk][_mk_mask, np.newaxis]
                    _mk_pred = np.argmax(_mk_probs, axis=1)
                    _mk_true = y_enc_v[_mk_mask]
                    _mk_bacc = balanced_accuracy_score(_mk_true, _mk_pred)
                    _mk_f1   = f1_score(_mk_true, _mk_pred, average='macro', zero_division=0)
                    _mk_oob_n = int(np.sum(_mk_mask))
                else:
                    _mk_bacc, _mk_f1, _mk_oob_n = 0.0, 0.0, 0
                history_broad[_mk]['bacc'].append(_mk_bacc)
                history_broad[_mk]['f1'].append(_mk_f1)
                single_model_scores.append((_mk, _mk_bacc, _mk_f1, _mk_oob_n))
                if _mk_oob_n > 0:
                    append_per_class_metric_rows(
                        per_class_metric_rows, i + 1, 'single_model', _mk,
                        _mk_true, _mk_pred, le_broad.classes_, oob_n=_mk_oob_n
                    )
                training_log_rows.append({
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
                    'timestamp': pd.Timestamp.now().isoformat()
                })

            single_summary = " | ".join(
                [f"{mk}: B-Acc {bacc*100:.1f}%, F1 {f1*100:.1f}%" for mk, bacc, f1, _ in single_model_scores]
            )
            print(f"  📈 [Iter {i+1} individual-model OOB] {single_summary}")

            for rank, res in enumerate(combo_results, 1):
                training_log_rows.append({
                    'iter': i + 1,
                    'eval_type': 'ensemble_combo',
                    'name': res['Combo'],
                    'rank': rank,
                    'bacc': res['B-Acc'],
                    'f1': res['F1'],
                    'avg': res['Avg'],
                    'oob_n': res['OOB_N'],
                    'best_at_iter': rank == 1,
                    'best_overall_so_far': res['Combo'] == best_val_combo and np.isclose(res['Avg'], best_val_score),
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                combo_keys = res['Combo'].split('+')
                combo_valid_mask = valid_mask.copy()
                for k in combo_keys:
                    combo_valid_mask &= oob_counts[k] > 0
                if np.sum(combo_valid_mask) > 0:
                    y_combo_true = y_enc_v[combo_valid_mask]
                    combo_probs_list = [
                        oob_probs[k][combo_valid_mask] / oob_counts[k][combo_valid_mask, np.newaxis]
                        for k in combo_keys
                    ]
                    combo_pred = np.argmax(np.mean(combo_probs_list, axis=0), axis=1)
                    append_per_class_metric_rows(
                        per_class_metric_rows, i + 1, 'ensemble_combo', res['Combo'],
                        y_combo_true, combo_pred, le_broad.classes_, rank=rank, oob_n=int(np.sum(combo_valid_mask))
                    )

            # Update the CSV at each iteration to retain logs if lengthy training is interrupted
            pd.DataFrame(training_log_rows).to_csv(TRAIN_LOG_STEP02_PATH, index=False)
            print(f"  💾 [Iter {i+1}] Training log saved: {TRAIN_LOG_STEP02_PATH}", flush=True)
            train_pbar.set_postfix(Best=top_res['Combo'], Avg=f"{top_res['Avg']*100:.2f}%")
                
            # Print rankings for all 15 combinations at the final loop iteration
            if (i + 1) == N_ITER:
                print(f"\n  🎯 [Final training completed - ranking of all 15 model combinations]")
                for rank, res in enumerate(combo_results, 1):
                    print(f"    Rank {rank:2d}: {res['Combo']:<20} | Avg: {res['Avg']*100:.2f}%")

        # Clean up memory
        del X_tr, X_val, X_dlbcl_sc, y_tr, y_val
        gc.collect()

    print(f"\n✅ Main-training cumulative evaluation finished (completed iterations: {actual_iters} / {N_ITER}, "
          f"Best Combo: {best_val_combo}, "
          f"Best Avg(B-Acc,F1): {best_val_score*100:.2f}%)")

    # Print the Feature Importance summary
    summarize_feature_importance(fi_log_broad, feature_names_list)

    broad_transfer_manifest = None
    if broad_transfer_stage is not None:
        broad_transfer_manifest = finalize_transfer_stage(
            broad_transfer_stage,
            extra_metadata={
                'best_oob_combo': best_val_combo,
                'best_oob_score': best_val_score,
            },
        )
        print(
            "✅ Broad 4-model frozen package saved: "
            f"{len(broad_transfer_manifest['all_model_complete_iterations'])} "
            "all-model-complete iterations"
        )

    # ── Save backup logging data to disk (for kernel restarts) ──
    broad_bkp = {
        'all_probs': all_probs,
        'history_broad': history_broad,
        'training_log': pd.DataFrame(training_log_rows),
        'training_log_path': TRAIN_LOG_STEP02_PATH,
        'per_class_metric_log': pd.DataFrame(per_class_metric_rows),
        'class_fi_log_broad': class_fi_log_broad,
        'class_fi_table': pd.DataFrame(class_fi_rows),
        'class_fi_summary': summarize_class_feature_importance(class_fi_rows),
        'best_val_combo': best_val_combo,
        'fi_log_broad': fi_log_broad,
        'feature_names': feature_names_list,
        'frozen_transfer_manifest': broad_transfer_manifest,
        'frozen_transfer_root': V9_TRANSFER_ROOT,
    }
    joblib.dump(broad_bkp, PROBS_STEP02_PATH)
    print(f"✅ OOB ensemble probabilities, Feature Importance, and logs saved: {PROBS_STEP02_PATH}")
    print(f"  ⏱️ Training end time: {pd.Timestamp.now().strftime('%H:%M:%S')}")

else:
    if os.path.exists(PROBS_STEP02_PATH):
        print(f"\n🔹 [Step 02-3] Skipping training: printing the evaluation summary from the existing log file ({os.path.basename(PROBS_STEP02_PATH)}).")
        try:
            _bkp = joblib.load(PROBS_STEP02_PATH)
            _best_combo = _bkp.get('best_val_combo', 'Unknown')
            _history = _bkp.get('history_broad', {})
            _best_avg = _history.get('best', {}).get('avg', [0])[-1] if 'best' in _history and _history['best'].get('avg') else 0
            
            print(f"  ✅ Best cumulative OOB evaluation combination: {_best_combo}")
            print(f"  ✅ Best OOB mean Score (B-Acc & F1): {_best_avg*100:.2f}%")
            
            if 'fi_log_broad' in _bkp and 'feature_names' in _bkp:
                summarize_feature_importance(_bkp['fi_log_broad'], _bkp['feature_names'])
        except Exception as e:
            print(f"  ⚠️ Error while printing the log summary: {e}")

# %%
# ==============================================================================
# [Step 02-3.5] Manually select the ensemble combination and iteration cutoff
# ==============================================================================
# Users can inspect the logs and select the best-performing combination name (combo) and stopping iteration (iter).
# If an empty string ('') is entered, the best-performing ensemble setting found above (best_val_combo) is automatically applied at N_ITER.
MANUAL_BROAD_COMBO = 'XGB+MLP'    # Examples: 'XGB+MLP', 'RF+MLP', ''
MANUAL_BROAD_ITER  = N_ITER # Specify the cumulative number of iterations to average (e.g., 50, 100)

# Attempt to restore logs from disk (also works after a kernel restart)
if 'all_probs' not in globals() and os.path.exists(PROBS_STEP02_PATH):
    print("🔹 [Step 02-3.5] Restoring existing Broad ensemble logs from disk...")
    broad_bkp = joblib.load(PROBS_STEP02_PATH)
    all_probs = broad_bkp['all_probs']
    history_broad = broad_bkp['history_broad']
    best_val_combo = broad_bkp['best_val_combo']
    # Restore Feature Importance
    if 'fi_log_broad' in broad_bkp:
        fi_log_broad = broad_bkp['fi_log_broad']
    if 'feature_names' in broad_bkp:
        feature_names_list = broad_bkp['feature_names']
    if 'class_fi_log_broad' in broad_bkp:
        class_fi_log_broad = broad_bkp['class_fi_log_broad']
    if 'class_fi_table' in broad_bkp:
        class_fi_table = broad_bkp['class_fi_table']
    if 'class_fi_summary' in broad_bkp:
        class_fi_summary = broad_bkp['class_fi_summary']
    if 'per_class_metric_log' in broad_bkp:
        per_class_metric_log = broad_bkp['per_class_metric_log']
    print(f"  ✅ Restored (Best Combo: {best_val_combo}, Iterations: {len(all_probs[MODEL_KEYS[0]])})")

if 'all_probs' in globals() and 'best_val_combo' in globals():
    target_combo = MANUAL_BROAD_COMBO if MANUAL_BROAD_COMBO.strip() != '' else best_val_combo
    print(f"\n🔹 [Step 02-3.5] Creating ensemble (combination: {target_combo}, Iter: {MANUAL_BROAD_ITER})")

    _broad_transfer_manifest_path = os.path.join(
        V9_TRANSFER_ROOT, 'broad', 'manifest.json'
    )
    if STEP02_FIGURE_ONLY:
        print("  ℹ️ Figure-only mode → skipping the frozen-package selection manifest update.")
    elif os.path.exists(_broad_transfer_manifest_path):
        from V9_transfer_utils import set_transfer_selection, transfer_stage_dir
        set_transfer_selection(
            transfer_stage_dir(V9_TRANSFER_ROOT, 'broad'),
            target_combo,
            MANUAL_BROAD_ITER,
        )
        print(
            "  ✅ Final frozen-package selection recorded: "
            f"{target_combo}, Iter={MANUAL_BROAD_ITER}"
        )
    
    # ── Visualize training performance over time (individual models + Best ensemble) ──
    if 'history_broad' in globals() and history_broad['iters']:
        _MODEL_COLORS = {'RF': '#27AE60', 'XGB': '#E67E22', 'ENet': '#3498DB', 'MLP': '#9B59B6'}
        
        # ── [Data preparation] Feature Importance ──
        has_fi = False
        if 'fi_log_broad' in globals() and fi_log_broad:
            # Use all models to display their results in the visualization (Bubble plot)
            _fi_models = [m for m in MODEL_KEYS if m in fi_log_broad and fi_log_broad[m]]
            if _fi_models:
                has_fi = True
                import scipy.stats as stats
                import matplotlib.patches as mpatches
                import matplotlib.lines as mlines
                
                global_avg_fi = defaultdict(float)
                model_fi = {m: defaultdict(float) for m in _fi_models}
                count_models = 0
                for m_name in _fi_models:
                    records = fi_log_broad[m_name]
                    if not records: continue
                    for rec in records:
                        for fname, val in rec.items():
                            model_fi[m_name][fname] += abs(val)
                    for k, v in model_fi[m_name].items():
                        model_fi[m_name][k] /= len(records)
                    
                    if m_name == 'ENet' or sum(model_fi[m_name].values()) > 1.5:
                        total_sum = sum(model_fi[m_name].values())
                        if total_sum > 0:
                            for k in model_fi[m_name].keys():
                                model_fi[m_name][k] /= total_sum
                                
                    for k, v in model_fi[m_name].items():
                        global_avg_fi[k] += v
                    count_models += 1
                
                if count_models > 0:
                    global_avg_fi = {k: v / count_models for k, v in global_avg_fi.items()}
                
                top_markers = [k for k, v in sorted(global_avg_fi.items(), key=lambda x: x[1], reverse=True)[:30]]
                lineage_list = ['B cells', 'T cells', 'Myeloid cells', 'Stromal cells', 'NK cells']
                masks_lin = {lin: (adata.obs['GT_Lineage_Broad'] == lin).values for lin in lineage_list}
                
                valid_markers = [m for m in top_markers if m in adata.var_names]
                if valid_markers:
                    expr_matrix = adata[:, valid_markers].X
                    if hasattr(expr_matrix, "toarray"):
                        expr_matrix = expr_matrix.toarray()
                else:
                    expr_matrix = np.array([[]])
                
                marker_meta = {}
                for idx, marker in enumerate(valid_markers):
                    expr = expr_matrix[:, idx]
                    best_lineage = None
                    best_mean = -np.inf
                    best_pval = 1.0
                    for lin in lineage_list:
                        mask_lin = masks_lin[lin]
                        mask_rest = ~mask_lin
                        expr_lin = expr[mask_lin]
                        expr_rest = expr[mask_rest]
                        mean_lin = np.mean(expr_lin) if len(expr_lin) > 0 else 0
                        if mean_lin > best_mean:
                            best_mean = mean_lin
                            best_lineage = lin
                            if len(expr_lin) > 2 and len(expr_rest) > 2:
                                t_stat, p_val = stats.ttest_ind(expr_lin, expr_rest, equal_var=False)
                                best_pval = p_val / 2 if t_stat > 0 else 1.0
                            else:
                                best_pval = 1.0
                    marker_meta[marker] = {'Assigned_Lineage': best_lineage, 'P_value': best_pval}
                
                lineage_order_dict = {l: i for i, l in enumerate(lineage_list)}
                marker_df = pd.DataFrame([{'Marker': m, 'Lineage': marker_meta[m]['Assigned_Lineage'], 'Global_Imp': global_avg_fi[m]} for m in valid_markers])
                marker_df['Lin_Idx'] = marker_df['Lineage'].map(lineage_order_dict)
                # Sort (by Lineage, then by importance within each Lineage)
                marker_df = marker_df.sort_values(['Lin_Idx', 'Global_Imp'], ascending=[True, False])
                ordered_markers = marker_df['Marker'].tolist()
                
                # ── [Key] Extract markers for actual Rescue using the mean over only the models in the user-specified target_combo ──
                target_models = [m for m in target_combo.split('+') if m in model_fi]
                if not target_models: target_models = _fi_models  # fallback
                target_avg_fi = defaultdict(float)
                for m_name in target_models:
                    for k, v in model_fi[m_name].items():
                        target_avg_fi[k] += v
                if len(target_models) > 0:
                    target_avg_fi = {k: v / len(target_models) for k, v in target_avg_fi.items()}
                
                marker_df_target = pd.DataFrame([{'Marker': m, 'Lineage': marker_meta[m]['Assigned_Lineage'], 'Target_Imp': target_avg_fi[m]} for m in valid_markers])
                marker_df_target = marker_df_target.sort_values(['Lineage', 'Target_Imp'], ascending=[True, False])
                
                global DYNAMIC_MARKER_MAP
                DYNAMIC_MARKER_MAP = {}
                # ── Multiple-marker guardrail: use all markers reaching at least 50% cumulative importance; otherwise Top-3 ──
                for lin in lineage_list:
                    lin_df = marker_df_target[marker_df_target['Lineage'] == lin].sort_values('Target_Imp', ascending=False)
                    if lin_df.empty:
                        continue
                    total_imp = lin_df['Target_Imp'].sum()
                    if total_imp == 0:
                        DYNAMIC_MARKER_MAP[lin] = [lin_df.iloc[0]['Marker']]
                        continue
                    # Select markers reaching at least 50% cumulative importance
                    cumsum = lin_df['Target_Imp'].cumsum() / total_imp
                    cutoff_mask = cumsum <= 0.5
                    # Include the first marker exceeding 50%
                    cutoff_idx = cutoff_mask.sum()  # Number of markers at or below 0.5
                    selected = lin_df.iloc[:max(cutoff_idx + 1, 1)]['Marker'].tolist()
                    # If no marker exceeds 50% (if the first marker already exceeds 50%), use Top-3
                    if len(selected) < 2 and len(lin_df) >= 3:
                        selected = lin_df.iloc[:3]['Marker'].tolist()
                    DYNAMIC_MARKER_MAP[lin] = selected
                print(f"    🌟 [{target_combo} Multi-Marker Guardrail Extracted]:")
                for lin, mks in DYNAMIC_MARKER_MAP.items():
                    print(f"       {lin}: {mks}")
                
                plot_data = []
                for marker in ordered_markers:
                    for m_idx, m_name in enumerate(_fi_models):
                        imp = model_fi[m_name].get(marker, 0)
                        plot_data.append({'Model_Idx': m_idx, 'Model': m_name, 'Marker': marker, 'Importance': imp, 'Lineage': marker_meta[marker]['Assigned_Lineage'], 'P_value': marker_meta[marker]['P_value']})
                df_plot = pd.DataFrame(plot_data)

        # ── [Figure 2A-B] Combined broad-lineage ensemble performance + feature importance figure ──
        import matplotlib.gridspec as gridspec
        _fig2ab_scale = 0.80
        _fig2ab_alpha = 0.80
        _fig2ab_dot_scale = 0.25

        if has_fi:
            fig_2ab = plt.figure(figsize=STEP02_FIGURE2AB_SIZE)
            gs_2ab = gridspec.GridSpec(
                3, 4,
                # Order: left performance panel, spacer, feature importance, index panel.
                # Set the effective panel ratio to approximately 1.5 : 1.0 : 0.5.
                width_ratios=[1.50, 0.32, 1.00, 0.50],
                height_ratios=[1, 1, 0.30],
                wspace=0.08,
                hspace=0.30
            )
            ax_bacc = fig_2ab.add_subplot(gs_2ab[0, 0])
            ax_f1 = fig_2ab.add_subplot(gs_2ab[1, 0], sharex=ax_bacc)
            ax_leg = fig_2ab.add_subplot(gs_2ab[2, 0])
            ax_spacer = fig_2ab.add_subplot(gs_2ab[:, 1])
            ax_spacer.axis('off')
            ax_bub = fig_2ab.add_subplot(gs_2ab[:, 2])
            ax_index = fig_2ab.add_subplot(gs_2ab[:, 3])
            ax_index.axis('off')
            ax_index.set_xlim(0, 1)
            ax_index.set_ylim(0, 1)
        else:
            print("    ⚠️ No Feature Importance logs; generating only the performance panel for Figure 2A-B.")
            fig_2ab = plt.figure(figsize=(14, 10))
            gs_2ab = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.16], hspace=0.30)
            ax_bacc = fig_2ab.add_subplot(gs_2ab[0, 0])
            ax_f1 = fig_2ab.add_subplot(gs_2ab[1, 0], sharex=ax_bacc)
            ax_leg = fig_2ab.add_subplot(gs_2ab[2, 0])
        
        # [Figure 2A-1] B-Acc trend
        _is_new_fmt = 'best' in history_broad
        if _is_new_fmt:
            for mk in MODEL_KEYS:
                if mk in history_broad and history_broad[mk]['bacc']:
                    ax_bacc.plot(history_broad['iters'], history_broad[mk]['bacc'], marker='o', markersize=6*_fig2ab_scale, label=f'{mk}', color=_MODEL_COLORS.get(mk, '#888'), linewidth=1.5*_fig2ab_scale, alpha=0.8)
            ax_bacc.plot(history_broad['iters'], history_broad['best']['bacc'], marker='^', markersize=8*_fig2ab_scale, label='Best Ensemble', color='#E74C3C', linewidth=2.5*_fig2ab_scale)
        else:
            ax_bacc.plot(history_broad['iters'], history_broad['bacc'], marker='o', markersize=6*_fig2ab_scale, label='B-Acc', color='#5DADE2')
            ax_bacc.plot(history_broad['iters'], history_broad['avg'], marker='^', markersize=8*_fig2ab_scale, label='Best Ensemble', color='#E74C3C', linewidth=2*_fig2ab_scale)
        
        ax_bacc.axvline(x=MANUAL_BROAD_ITER, color='k', linestyle='--', linewidth=2*_fig2ab_scale, label=f'Cut-off ({MANUAL_BROAD_ITER})')
        ax_bacc.set_ylabel("OOB B-Acc", fontsize=17*_fig2ab_scale, fontweight='bold', labelpad=7)
        ax_bacc.set_ylim(0.675, 0.875)
        ax_bacc.grid(True, linestyle='--', alpha=0.5)
        
        ax_bacc.tick_params(axis='both', labelsize=13*_fig2ab_scale)
        for label in ax_bacc.get_xticklabels() + ax_bacc.get_yticklabels():
            label.set_fontweight('bold')
        
        # [Figure 2A-2] F1 trend
        if _is_new_fmt:
            for mk in MODEL_KEYS:
                if mk in history_broad and history_broad[mk]['f1']:
                    ax_f1.plot(history_broad['iters'], history_broad[mk]['f1'], marker='s', markersize=6*_fig2ab_scale, color=_MODEL_COLORS.get(mk, '#888'), linewidth=1.5*_fig2ab_scale, alpha=0.8)
            ax_f1.plot(history_broad['iters'], history_broad['best']['f1'], marker='^', markersize=8*_fig2ab_scale, color='#E74C3C', linewidth=2.5*_fig2ab_scale)
        else:
            ax_f1.plot(history_broad['iters'], history_broad['f1'], marker='s', markersize=6*_fig2ab_scale, color='#F1C40F')
            
        ax_f1.axvline(x=MANUAL_BROAD_ITER, color='k', linestyle='--', linewidth=2*_fig2ab_scale)
        ax_f1.set_xlabel("Iteration", fontsize=17*_fig2ab_scale, fontweight='bold', labelpad=5)
        ax_f1.set_ylabel("OOB Macro F1", fontsize=17*_fig2ab_scale, fontweight='bold', labelpad=7)
        ax_f1.set_ylim(0.725, 0.875)
        ax_f1.grid(True, linestyle='--', alpha=0.5)
        
        ax_f1.tick_params(axis='both', labelsize=13*_fig2ab_scale)
        for label in ax_f1.get_xticklabels() + ax_f1.get_yticklabels():
            label.set_fontweight('bold')
            
        # Place the legend on a separate axis to preserve the plotting areas of both performance curves.
        ax_leg.axis('off')
        handles, labels = ax_bacc.get_legend_handles_labels()
        ax_leg.legend(
            handles, labels,
            loc='center', ncol=3,
            fontsize=14*_fig2ab_scale,
            frameon=False,
            markerscale=1.25,
            handlelength=2.35,
            handletextpad=0.62,
            columnspacing=1.30,
        )

        # [Figure 2B] Broad lineage feature importance bubble plot
        if has_fi:
            x_labels = _fi_models
            x_pos_dict = {m: i for i, m in enumerate(x_labels)}
            df_plot['X_Pos'] = df_plot['Model'].map(x_pos_dict)
            
            y_labels = ordered_markers
            y_pos_dict = {m: i for i, m in enumerate(y_labels)}
            df_plot['Y_Pos'] = df_plot['Marker'].map(y_pos_dict)
            
            # Bubble size: normalize within each model × lineage combination for intuitive comparisons, e.g., CD7 importance in NK
            def normalize_within_group(df, group_cols, value_col):
                """Normalize each value to its relative proportion within its group."""
                df = df.copy()
                group_max = df.groupby(group_cols)[value_col].transform('max')
                df[value_col + '_norm'] = df[value_col] / group_max.replace(0, 1)
                return df
            
            df_plot = normalize_within_group(df_plot, ['Model', 'Lineage'], 'Importance')
            sizes = (df_plot['Importance_norm'] * 800 + 30) * _fig2ab_dot_scale
            
            log_pvals = -np.log10(np.clip(df_plot['P_value'], 1e-50, 1.0))
            edge_colors = [LINEAGE_COLORS.get(l, 'gray') for l in df_plot['Lineage']]
            # Fill color represents statistical significance (-log10 P-value); edge color represents lineage only.
            # Use neutral slate fills because red fills visually blend with T-cell edges.
            significance_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'significance_slate_clear',
                ['#f7f8fa', '#d7dee5', '#93a4b2', '#3f4c57']
            )
            
            scatter = ax_bub.scatter(
                df_plot['X_Pos'], df_plot['Y_Pos'], 
                s=sizes, 
                c=log_pvals, cmap=significance_cmap, vmin=0, vmax=70,
                edgecolors=edge_colors, linewidths=2.7*_fig2ab_scale, alpha=_fig2ab_alpha, zorder=3
            )
            
            ax_bub.set_xticks(range(len(x_labels)))
            ax_bub.set_xticklabels(x_labels, fontsize=17*_fig2ab_scale, fontweight='bold')
            
            ax_bub.set_yticks(range(len(y_labels)))
            ax_bub.set_yticklabels(y_labels, fontsize=12*_fig2ab_scale, fontweight='bold')
            ax_bub.tick_params(axis='y', pad=1)
            marker_lineage_lookup = marker_df.set_index('Marker')['Lineage'].to_dict()
            for tick_label in ax_bub.get_yticklabels():
                marker_name = tick_label.get_text()
                marker_lineage = marker_lineage_lookup.get(marker_name)
                tick_label.set_color(LINEAGE_COLORS.get(marker_lineage, '#222222'))
            
            # Lineage separators (horizontal because markers are now on the Y-axis)
            boundaries = []
            current_lin = marker_df['Lineage'].iloc[0]
            for i, lin in enumerate(marker_df['Lineage']):
                if lin != current_lin:
                    boundaries.append(i - 0.5)
                    current_lin = lin
            for b in boundaries:
                ax_bub.axhline(y=b, color='gray', linestyle='--', linewidth=1.4*_fig2ab_scale, alpha=0.5)
            
            ax_bub.set_xlim(-0.5, len(x_labels) - 0.5)
            ax_bub.set_ylim(-0.5, len(y_labels) - 0.5)
            
            # Invert the Y-axis to draw markers from top to bottom starting with B cells
            ax_bub.invert_yaxis()
            
            ax_bub.grid(True, linestyle=':', alpha=0.35, axis='both', zorder=0)
            
            # As in the edited version, show only the key without a separate Index title or outer box.
            ax_index.text(0.50, 0.795, "-log10(P-value)", transform=ax_index.transAxes,
                          ha='center', va='center', fontsize=12*_fig2ab_scale,
                          fontweight='bold')
            ax_cbar_inset = ax_index.inset_axes([0.20, 0.755, 0.60, 0.030])
            cbar = fig_2ab.colorbar(scatter, cax=ax_cbar_inset, orientation='horizontal')
            cbar.set_ticks([0, 35, 70])
            cbar.ax.tick_params(labelsize=11*_fig2ab_scale, length=2, pad=1)

            ax_index.text(0.50, 0.660, "Relative\nImportance", transform=ax_index.transAxes,
                          ha='center', va='center', fontsize=14*_fig2ab_scale,
                          fontweight='bold', linespacing=0.9)
            importance_y = [0.580, 0.528, 0.476, 0.424]
            for val, y_pos in zip([0.25, 0.5, 0.75, 1.0], importance_y):
                legend_size = (val * 800 + 30) * _fig2ab_dot_scale
                ax_index.scatter([0.35], [y_pos], s=legend_size, c='gray',
                                 edgecolors='none', alpha=_fig2ab_alpha,
                                 transform=ax_index.transAxes, zorder=2)
                ax_index.text(0.52, y_pos, f"{int(val*100)}%", transform=ax_index.transAxes,
                              ha='left', va='center', fontsize=13*_fig2ab_scale)

            ax_index.text(0.50, 0.320, "Lineage", transform=ax_index.transAxes,
                          ha='center', va='center', fontsize=14*_fig2ab_scale,
                          fontweight='bold')
            lineage_y = np.linspace(0.275, 0.145, len(lineage_list))
            for lin, y_pos in zip(lineage_list, lineage_y):
                c = LINEAGE_COLORS.get(lin, 'gray')
                ax_index.scatter([0.20], [y_pos], s=82*_fig2ab_scale,
                                 facecolors='white', edgecolors=c,
                                 linewidths=2.3*_fig2ab_scale,
                                 transform=ax_index.transAxes, zorder=2)
                ax_index.text(0.31, y_pos, lin, transform=ax_index.transAxes,
                              ha='left', va='center', fontsize=12*_fig2ab_scale)

        fig_2ab.subplots_adjust(left=0.10, right=0.985, top=0.975, bottom=0.055)
        fig2ab_png = os.path.join(FIG_DIR, 'Final_Figure2A-B_Broad_Lineage_Ensemble_Performance_and_Feature_Importance.png')
        fig2ab_pptx = os.path.join(FIG_DIR, 'Final_Figure2A-B_Broad_Lineage_Ensemble_Performance_and_Feature_Importance.pptx')
        _step02_save_publication_figure(
            fig_2ab, fig2ab_png, fig2ab_pptx,
            label='Figure 2A-B combined',
            fixed_canvas=True
        )

        # ── [Supplementary] Lineage-wise performance heatmap for the OOB validation set ──
        # This figure uses per_class_metric_log saved during Step02 training.
        # Without new training or prediction, it provides supplementary reviewer-response material
        # to assess which models perform consistently for each broad lineage in OOB validation.
        PLOT_STEP02_OOB_PER_LINEAGE_HEATMAP = True

        if PLOT_STEP02_OOB_PER_LINEAGE_HEATMAP:
            if 'per_class_metric_log' not in globals() or per_class_metric_log is None:
                print("    ⚠️ No per_class_metric_log; the OOB lineage-wise heatmap was not generated.")
            else:
                oob_class_df = pd.DataFrame(per_class_metric_log).copy()
                required_oob_cols = {
                    'iter', 'eval_type', 'name', 'lineage_class',
                    'bacc_ovr', 'f1_ovr', 'support', 'oob_n'
                }
                missing_oob_cols = sorted(required_oob_cols - set(oob_class_df.columns))
                if missing_oob_cols:
                    print(f"    ⚠️ Required columns for the OOB lineage-wise heatmap are missing: {missing_oob_cols}")
                elif oob_class_df.empty:
                    print("    ⚠️ per_class_metric_log is empty; the OOB lineage-wise heatmap was not generated.")
                else:
                    oob_class_df['iter'] = pd.to_numeric(oob_class_df['iter'], errors='coerce')
                    valid_iters = sorted(oob_class_df['iter'].dropna().astype(int).unique())
                    if not valid_iters:
                        print("    ⚠️ No valid OOB iteration logs; the OOB lineage-wise heatmap was not generated.")
                    else:
                        oob_iter = MANUAL_BROAD_ITER if MANUAL_BROAD_ITER in valid_iters else valid_iters[-1]
                        if oob_iter != MANUAL_BROAD_ITER:
                            print(f"    ⚠️ No log for MANUAL_BROAD_ITER={MANUAL_BROAD_ITER}; using OOB results from Iter {oob_iter}.")

                        oob_iter_df = oob_class_df[oob_class_df['iter'].eq(float(oob_iter))].copy()
                        oob_model_order = [m for m in MODEL_KEYS if m in oob_iter_df['name'].values]
                        if target_combo and target_combo in oob_iter_df['name'].values and target_combo not in oob_model_order:
                            oob_model_order.append(target_combo)
                        elif 'rank' in oob_iter_df.columns:
                            rank_df = oob_iter_df[
                                (oob_iter_df['eval_type'] == 'ensemble_combo')
                                & pd.to_numeric(oob_iter_df['rank'], errors='coerce').eq(1)
                            ]
                            if not rank_df.empty:
                                best_oob_combo = str(rank_df.iloc[0]['name'])
                                if best_oob_combo not in oob_model_order:
                                    oob_model_order.append(best_oob_combo)

                        if not oob_model_order:
                            print("    ⚠️ No OOB model logs to display; the OOB lineage-wise heatmap was not generated.")
                        else:
                            def _display_oob_model_name_step02(model_name):
                                if model_name in MODEL_KEYS:
                                    return model_name
                                return f"Ensemble\n({model_name})"

                            oob_iter_df = oob_iter_df[oob_iter_df['name'].isin(oob_model_order)].copy()
                            oob_iter_df['Model_Display'] = oob_iter_df['name'].map(_display_oob_model_name_step02)
                            oob_iter_df['lineage_class'] = oob_iter_df['lineage_class'].astype(str)
                            oob_lineage_order = ['B cells', 'Myeloid cells', 'NK cells', 'Stromal cells', 'T cells']
                            oob_lineage_order = [
                                c for c in oob_lineage_order
                                if c in set(oob_iter_df['lineage_class'])
                            ]
                            display_order = [_display_oob_model_name_step02(m) for m in oob_model_order]

                            # ════════════════════════════════════════════════════
                            # Figure ①: BAcc heatmap + F1 heatmap
                            # ════════════════════════════════════════════════════
                            fig_oob, (ax_oob_bacc, ax_oob_f1) = plt.subplots(
                                1, 2,
                                figsize=STEP02_OOB_PER_LINEAGE_SIZE,
                                gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.12}
                            )
                            # Keep the vertical color scale, but reserve a separate
                            # title band above its top tick for compact PPT placement.
                            cbar_oob_ax = fig_oob.add_axes([0.938, 0.335, 0.0075, 0.255])
                            # Store heatmap data (reused in the bar chart)
                            _mat_store = {}

                            for panel_idx, (ax_oob, metric_col) in enumerate([
                                (ax_oob_bacc, 'bacc_ovr'),
                                (ax_oob_f1,   'f1_ovr'),
                            ]):
                                mat_oob = oob_iter_df.pivot_table(
                                    index='lineage_class',
                                    columns='Model_Display',
                                    values=metric_col,
                                    aggfunc='mean'
                                )
                                mat_oob = mat_oob.reindex(index=oob_lineage_order, columns=display_order)
                                _mat_store[metric_col] = mat_oob.copy()

                                sns.heatmap(
                                    mat_oob * 100,
                                    annot=True,
                                    fmt='.1f',
                                    cmap='YlGnBu',
                                    vmin=0,
                                    vmax=100,
                                    linewidths=0.8,
                                    linecolor='white',
                                    cbar=(panel_idx == 1),
                                    cbar_ax=cbar_oob_ax if panel_idx == 1 else None,
                                    annot_kws={'fontsize': 9.8, 'fontweight': 'bold'},
                                    ax=ax_oob,
                                )
                                ax_oob.set_xlabel('')
                                ax_oob.set_ylabel('')
                                if panel_idx != 0:
                                    ax_oob.set_yticklabels([])
                                    ax_oob.tick_params(axis='y', left=False)
                                plt.setp(ax_oob.get_xticklabels(), rotation=0, ha='center',
                                         fontsize=9.4, fontweight='bold')
                                plt.setp(ax_oob.get_yticklabels(), rotation=0, fontsize=9.8, fontweight='bold')
                                if ax_oob.collections and ax_oob.collections[0].colorbar is not None:
                                    cbar_oob = ax_oob.collections[0].colorbar
                                    cbar_oob.ax.tick_params(labelsize=9.4, pad=2)
                                    cbar_oob.set_label('')

                            cbar_oob_ax.set_title(
                                'Score (%)',
                                fontsize=9.4,
                                fontweight='bold',
                                pad=6,
                                loc='center',
                            )

                            fig_oob.suptitle(
                                'OOB lineage performance | BAcc (left), one-vs-rest binary F1 (right)',
                                fontsize=12.8,
                                fontweight='bold',
                                y=0.975,
                            )
                            fig_oob.supxlabel('Model', fontsize=10.6, fontweight='bold', y=0.030)
                            fig_oob.supylabel(
                                'Ground truth lineage',
                                fontsize=10.8,
                                fontweight='bold',
                                x=0.018,
                            )
                            # Reserve a dedicated bottom band for the shared x label below
                            # the two-line Ensemble tick labels.
                            fig_oob.subplots_adjust(left=0.142, right=0.915, top=0.885, bottom=0.205, wspace=0.08)

                            oob_png = os.path.join(FIG_DIR, 'Final_mv1_Step02-Supp_OOB_PerLineage_BAcc_F1.png')
                            oob_pptx = os.path.join(FIG_DIR, 'Final_mv1_Step02-Supp_OOB_PerLineage_BAcc_F1.pptx')
                            _step02_save_publication_figure(
                                fig_oob, oob_png, oob_pptx,
                                label='OOB lineage-wise heatmap',
                                fixed_canvas=True
                            )

                            # ════════════════════════════════════════════════════
                            # Figure ②: lineage-wise bar chart (Step04B-07 style, standalone)
                            # ════════════════════════════════════════════════════
                            _ens_col = display_order[-1]
                            _bar_records = []
                            for _lin in oob_lineage_order:
                                _bv = _mat_store['bacc_ovr'].loc[_lin, _ens_col] if _lin in _mat_store['bacc_ovr'].index else np.nan
                                _fv = _mat_store['f1_ovr'].loc[_lin, _ens_col]   if _lin in _mat_store['f1_ovr'].index   else np.nan
                                if np.isfinite(_bv):
                                    _bar_records.append({'Lineage': _lin, 'Metric': 'Balanced Acc', 'Score': _bv})
                                if np.isfinite(_fv):
                                    _bar_records.append({'Lineage': _lin, 'Metric': 'F1-Score', 'Score': _fv})

                            if _bar_records:
                                _df_bar  = pd.DataFrame(_bar_records)
                                _lin_list = oob_lineage_order

                                fig_bar, ax_bar = plt.subplots(figsize=(12, 7))

                                sns.barplot(
                                    data=_df_bar, x='Lineage', y='Score', hue='Metric',
                                    hue_order=['Balanced Acc', 'F1-Score'],
                                    order=_lin_list,
                                    palette={'Balanced Acc': 'white', 'F1-Score': 'white'},
                                    edgecolor='black', ax=ax_bar
                                )
                                # Apply lineage colors: Solid(BAcc) / Hatch(F1)
                                _n_lin = len([l for l in _lin_list if l in _df_bar['Lineage'].values])
                                for _bi, _br in enumerate(ax_bar.patches):
                                    _li = _bi % _n_lin
                                    _mi = _bi // _n_lin
                                    _lname = _lin_list[_li] if _li < len(_lin_list) else ''
                                    _col = LINEAGE_COLORS.get(_lname, '#808080')
                                    if _mi == 0:
                                        _br.set_facecolor(_col); _br.set_alpha(0.9)
                                    else:
                                        _br.set_facecolor(_col); _br.set_alpha(0.4); _br.set_hatch('///')

                                # Numeric annotations
                                for _p in ax_bar.patches:
                                    _h = _p.get_height()
                                    if _h > 0:
                                        ax_bar.annotate(
                                            f"{_h*100:.1f}%",
                                            (_p.get_x() + _p.get_width() / 2., _h),
                                            ha='center', va='bottom',
                                            xytext=(0, 5), textcoords='offset points',
                                            fontsize=9, fontweight='bold'
                                        )

                                # Legend (Step04B-07 style)
                                from matplotlib.patches import Patch as _Patch
                                _leg_elems = [
                                    _Patch(facecolor='#555555', alpha=0.9, edgecolor='black', label='Balanced Acc'),
                                    _Patch(facecolor='#555555', alpha=0.4, hatch='///', edgecolor='black', label='F1-Score'),
                                ]
                                ax_bar.legend(
                                    handles=_leg_elems,
                                    title='Performance Metric',
                                    loc='upper right'
                                )

                                ax_bar.set_ylim(0.5, 1.08)
                                ax_bar.set_title(
                                    f'Lineage-Specific Broad Lineage Prediction Performance\n'
                                    f'(Reference OOB Validation)',
                                    fontsize=15, fontweight='bold', pad=20
                                )
                                ax_bar.set_xlabel('Cell Lineage & Best Model', fontsize=12, fontweight='bold')
                                ax_bar.set_ylabel('Score', fontsize=12, fontweight='bold')
                                ax_bar.grid(axis='y', linestyle='--', alpha=0.3)
                                
                                # Show the unified model combination (Combo) in X-axis labels
                                _current_labels = [label.get_text() for label in ax_bar.get_xticklabels()]
                                _new_labels = []
                                for _l_name in _current_labels:
                                    _new_labels.append(f"{_l_name}\n[{_ens_col}]")
                                ax_bar.set_xticklabels(_new_labels, fontweight='bold')
                                plt.setp(ax_bar.get_xticklabels(), rotation=0, ha='center', fontsize=11, fontweight='bold')
                                plt.setp(ax_bar.get_yticklabels(), fontsize=11, fontweight='bold')

                                fig_bar.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.22)

                                bar_png = os.path.join(FIG_DIR, 'Final_mv1_Step02-Supp_OOB_PerLineage_Bar.png')
                                bar_pptx = os.path.join(FIG_DIR, 'Final_mv1_Step02-Supp_OOB_PerLineage_Bar.pptx')
                                _step02_save_publication_figure(
                                    fig_bar, bar_png, bar_pptx,
                                    label='OOB lineage-wise bar',
                                    fixed_canvas=False
                                )



    if apply_step02_predictions:
        # ── Accumulate only through the specified iteration to create Soft Voting for 15 combinations ──
        _broad_model_probs = {k: all_probs[k][:MANUAL_BROAD_ITER] for k in MODEL_KEYS}

        broad_combos = []
        for r in range(1, len(MODEL_KEYS) + 1):
            for combo in _combs(MODEL_KEYS, r):
                broad_combos.append(combo)
        combo_names = ['+'.join(c) for c in broad_combos]

        preds_mapping = []
        for combo in broad_combos:
            combined = []
            for key in combo:
                combined.extend(_broad_model_probs[key])
            preds_mapping.append(np.mean(combined, axis=0))

        # Recreate dlbcl_mask independently of the current kernel state for safe mapping
        dlbcl_mask_step02 = make_exact_disease_mask(adata, 'DLBCL')

        # Safely map predictions for the 15 combinations using .loc
        for m in range(1, N_COMBOS + 1):
            col_name = f'Pred_ML_Lineage_{m}'
            if col_name not in adata.obs.columns:
                adata.obs[col_name] = 'na'
            mapped_labels = le_broad.inverse_transform(np.argmax(preds_mapping[m-1], axis=1))
            adata.obs.loc[dlbcl_mask_step02, col_name] = mapped_labels.astype(str)

        # Soft Voting for the target combination to apply finally
        if target_combo in combo_names:
            best_combo_idx = combo_names.index(target_combo)
            print(f"\n⭐ Final Broad model ensemble selected: {target_combo} (combination {best_combo_idx+1})")
            p_ml_final = preds_mapping[best_combo_idx]
        else:
            print("\n⚠️ Failed to track the optimal model; using the default (all-model ensemble)")
            p_ml_final = preds_mapping[N_COMBOS - 1]

        final_preds = le_broad.inverse_transform(np.argmax(p_ml_final, axis=1))
        final_conf  = np.max(p_ml_final, axis=1)

        for c in ['ML_Lineage_Raw', 'ML_Lineage_Pure', 'ML_Broad_Confidence']:
            if c not in adata.obs.columns:
                adata.obs[c] = 'na'

        # Safely map with .loc (prevent iloc ValueError)
        adata.obs.loc[dlbcl_mask_step02, 'ML_Lineage_Raw']  = final_preds.astype(str)
        adata.obs['ML_Lineage_Pure'] = adata.obs['ML_Lineage_Raw'].copy()
        adata.obs.loc[dlbcl_mask_step02, 'ML_Broad_Confidence'] = final_conf
        adata.obs['ML_Broad_Confidence'] = pd.to_numeric(adata.obs['ML_Broad_Confidence'], errors='coerce')

        print(f"\n✅ Manual Soft Voting completed! (selected {target_combo}, accumulated over {MANUAL_BROAD_ITER} iterations)")
        print(f"  Mean prediction confidence: {final_conf.mean():.4f}")
    else:
        print("\n🔹 Figure-only mode: retaining saved Step02 predictions/Rescue results and skipping Soft Voting reapplication.")
# %%
# ==============================================================================
# [Step 02-4] KNN Rescue & Marker-Aware Spatial Smoothing
# ==============================================================================
if not skip_rescue:
    print("🔹 [Step 02-4] Running KNN spatial correction (Rescue)...")
    if 'p_ml_final' not in globals():
        raise RuntimeError(
            "KNN Rescue requires p_ml_final. Reuse existing training logs or "
            "set RUN_STEP02_TRAINING=True to generate Soft Voting results first."
        )

    def spillover_aware_prob_rescue(orig_label, cell_probs, le_broad, neighbor_labels, cell_expr_z, marker_col_idx):
        if len(neighbor_labels) == 0:
            return (False, "No Neighbors", orig_label, 0.0, 0.0, "not_evaluated", np.nan, np.nan)
            
        lbl_counts = pd.Series(neighbor_labels).value_counts()
        dom_env = lbl_counts.index[0]
        dom_ratio = lbl_counts.iloc[0] / len(neighbor_labels)

        classes = list(le_broad.classes_)

        orig_idx = classes.index(orig_label) if orig_label in classes else -1
        orig_p = cell_probs[orig_idx] if orig_idx != -1 else 0.0
        if orig_idx == -1:
            return (False, "Blocked: current label not in model classes", orig_label, 0.0, 0.0, "not_evaluated", np.nan, np.nan)
        if dom_env not in classes:
            return (False, "Blocked: dominant neighbor label not in model classes", orig_label, orig_p, orig_p, "not_evaluated", np.nan, np.nan)

        # Rescue aims to reverse cases where the surrounding dominant lineage overwrote low-confidence cells.
        # Thus, retain a current predicted label that differs from the surrounding dominant lineage, as it is not a spillover candidate.
        if orig_label != dom_env:
            return (False, f"Kept Original / Not dominated by neighbor {dom_env}", orig_label, orig_p, orig_p, "not_evaluated", np.nan, np.nan)

        adj_probs = cell_probs.copy()
        env_idx = classes.index(dom_env)
        n_cls = len(classes)
        p_safe = np.clip(cell_probs, 1e-9, 1.0)
        H = -np.sum(p_safe * np.log(p_safe))
        H_norm = H / np.log(n_cls)

        sorted_p = np.sort(cell_probs)[::-1]
        margin = float(sorted_p[0] - sorted_p[1])

        # Lower the dominant-lineage probability only for peripheral spillover candidates (orig_label == dom_env).
        # Apply stronger correction when the probability distribution is ambiguous (high H_norm), surrounding dominance is strong (high dom_ratio),
        # and the margin between the top two classes is smaller.
        spillover_penalty = dom_ratio * H_norm * (1.0 - margin)
        adj_probs[env_idx] = max(0.0, adj_probs[env_idx] - spillover_penalty)

        new_best_idx = np.argmax(adj_probs)
        best_lin = classes[new_best_idx]
        best_p = adj_probs[new_best_idx]

        if best_lin != orig_label:
            # The candidate label for restoration must have at least 40% confidence
            if best_p > 0.4: 
                if 'DYNAMIC_MARKER_MAP' in globals() and DYNAMIC_MARKER_MAP:
                    marker_map = DYNAMIC_MARKER_MAP
                else:
                    marker_map = {'T cells':['CD3'], 'B cells':['CD20'], 'NK cells':['CD56'], 'Myeloid cells':['CD68'], 'Stromal cells':['PDPN']}

                new_markers = marker_map.get(best_lin, [])
                orig_markers = marker_map.get(orig_label, [])
                if isinstance(new_markers, str): new_markers = [new_markers]
                if isinstance(orig_markers, str): orig_markers = [orig_markers]

                new_z_vals = [float(cell_expr_z[marker_col_idx[m]]) for m in new_markers if m in marker_col_idx]
                orig_z_vals = [float(cell_expr_z[marker_col_idx[m]]) for m in orig_markers if m in marker_col_idx]

                if new_z_vals and orig_z_vals:
                    avg_new_z = np.mean(new_z_vals)
                    avg_orig_z = np.mean(orig_z_vals)
                    
                    # Approve only when marker expression for the candidate restoration label is sufficiently higher than for the surrounding dominant label.
                    if avg_new_z < 0 or avg_new_z <= (avg_orig_z + 0.2):
                        blocked_mk_str = '+'.join(new_markers[:2])
                        orig_mk_str = '+'.join(orig_markers[:2])
                        return (
                            False,
                            f"Blocked: {blocked_mk_str}({avg_new_z:.2f}) not sufficiently higher than {orig_mk_str}({avg_orig_z:.2f})",
                            orig_label,
                            orig_p,
                            best_p,
                            "blocked",
                            float(avg_new_z),
                            float(avg_orig_z)
                        )
                else:
                    avg_new_z = np.nan
                    avg_orig_z = np.nan
                    marker_status = "not_evaluable_missing_marker"
                    reason = f"Spillover-Rescued: {orig_label} -> {best_lin} (neighbor={dom_env}; marker not evaluable)"
                    return (True, reason, best_lin, orig_p, best_p, marker_status, avg_new_z, avg_orig_z)

                reason = f"Spillover-Rescued: {orig_label} -> {best_lin} (neighbor={dom_env})"
                return (True, reason, best_lin, orig_p, best_p, "passed", float(avg_new_z), float(avg_orig_z))

        return (False, "Kept Original / Spillover penalty insufficient", orig_label, orig_p, best_p, "not_evaluated", np.nan, np.nan)

    # --- Main execution logic ---
    max_conf    = adata.obs['ML_Broad_Confidence'].values
    
    # Create a dictionary for p_ml_final mapping (full index -> dlbcl_mask index)
    dlbcl_indices = np.where(dlbcl_mask)[0]
    full_to_dlbcl = {full_idx: i for i, full_idx in enumerate(dlbcl_indices)}
    # Raise the confidence threshold to below 0.6 to broaden rescue targets (actively rescue spillover)
    noise_cells = np.where((max_conf < 0.6) & dlbcl_mask)[0]
    rescued_count, blocked_count = 0, 0
    reason_counts = {}
    blocked_reasons = {}
    rescued_log = []  # Record store for Step 02-4.5 visualization
    rescue_qc_rows = []  # Audit log of all Rescue attempts/blocks/approvals

    X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)

    scaler_guard = StandardScaler()
    X_z_scaled = scaler_guard.fit_transform(X_raw)

    marker_col_idx = {m: idx for idx, m in enumerate(adata.var_names)}

    if len(noise_cells) > 0:
        for tissue in tqdm(adata.obs['Run_Tissue_name'][dlbcl_mask].unique(),
                          desc="Spatial Smoothing (Marker-Guard)"):
            t_idx   = np.where((adata.obs['Run_Tissue_name'] == tissue) & dlbcl_mask)[0]
            t_noise = np.intersect1d(t_idx, noise_cells)
            if len(t_idx) < 10 or len(t_noise) == 0: continue

            X_spat  = adata.obs[['X_centroid', 'Y_centroid']].iloc[t_idx].values.astype(np.float32)
            labels  = adata.obs['ML_Lineage_Raw'].iloc[t_idx].values
            n_loc = np.where(np.isin(t_idx, t_noise))[0]
            c_loc = np.where(~np.isin(t_idx, t_noise))[0]

            if len(c_loc) < 10: continue

            knn_model = cuNN(n_neighbors=min(10, len(c_loc))).fit(X_spat[c_loc])
            dists, nn_idx = knn_model.kneighbors(X_spat[n_loc])
            dists_np = dists.get() if hasattr(dists, 'get') else dists
            nn_idx_np = nn_idx.get() if hasattr(nn_idx, 'get') else nn_idx

            for j, n_row, d_row in zip(n_loc, nn_idx_np, dists_np):
                full_idx = t_idx[j]
                orig = adata.obs['ML_Lineage_Raw'].iloc[full_idx]
                original_confidence = pd.to_numeric(pd.Series([max_conf[full_idx]]), errors='coerce').iloc[0]
                original_confidence = float(original_confidence) if pd.notna(original_confidence) else np.nan
                prob_idx = full_to_dlbcl[full_idx]
                cell_probs = p_ml_final[prob_idx]

                valid_nb_mask = d_row <= SPILLOVER_RADIUS_PX
                if not np.any(valid_nb_mask):
                    blocked_count += 1
                    if "No Neighbors within 20px" not in blocked_reasons:
                        blocked_reasons["No Neighbors within 20px"] = 0
                    blocked_reasons["No Neighbors within 20px"] += 1
                    rescue_qc_rows.append({
                        'cell_idx': int(full_idx),
                        'tissue': tissue,
                        'before_lineage': orig,
                        'after_lineage': orig,
                        'dominant_neighbor_lineage': 'none',
                        'dominant_neighbor_ratio': 0.0,
                        'n_neighbors_within_radius': 0,
                        'original_confidence': original_confidence,
                        'orig_label_probability': np.nan,
                        'candidate_probability_after_penalty': np.nan,
                        'rescue_status': 'blocked',
                        'marker_guard_status': 'not_evaluated',
                        'candidate_marker_z': np.nan,
                        'original_marker_z': np.nan,
                        'reason': "No Neighbors within 20px"
                    })
                    continue
                
                ls = labels[c_loc][n_row][valid_nb_mask]
                nb_counts = pd.Series(ls).value_counts()
                dominant_neighbor_lineage = str(nb_counts.index[0]) if len(nb_counts) > 0 else 'none'
                dominant_neighbor_ratio = float(nb_counts.iloc[0] / len(ls)) if len(nb_counts) > 0 else 0.0

                is_approved, reason, new_label, orig_p, best_p, marker_guard_status, avg_new_z, avg_orig_z = spillover_aware_prob_rescue(
                    orig_label=orig, 
                    cell_probs=cell_probs, 
                    le_broad=le_broad, 
                    neighbor_labels=ls,
                    cell_expr_z=X_z_scaled[t_idx[j]], 
                    marker_col_idx=marker_col_idx
                )

                rescue_status = 'rescued' if is_approved else 'kept_or_blocked'
                rescue_qc_rows.append({
                    'cell_idx': int(full_idx),
                    'tissue': tissue,
                    'before_lineage': orig,
                    'after_lineage': new_label if is_approved else orig,
                    'dominant_neighbor_lineage': dominant_neighbor_lineage,
                    'dominant_neighbor_ratio': dominant_neighbor_ratio,
                    'n_neighbors_within_radius': int(np.sum(valid_nb_mask)),
                    'original_confidence': original_confidence,
                    'orig_label_probability': float(orig_p),
                    'candidate_probability_after_penalty': float(best_p),
                    'rescue_status': rescue_status,
                    'marker_guard_status': marker_guard_status,
                    'candidate_marker_z': avg_new_z,
                    'original_marker_z': avg_orig_z,
                    'reason': reason
                })

                if not is_approved:
                    blocked_count += 1
                    if reason not in blocked_reasons:
                        blocked_reasons[reason] = 0
                    blocked_reasons[reason] += 1
                    continue

                adata.obs.iloc[t_idx[j], adata.obs.columns.get_loc('ML_Lineage_Raw')] = new_label
                rescued_count += 1
                
                if reason not in reason_counts:
                    reason_counts[reason] = 0
                reason_counts[reason] += 1

                rescued_log.append({
                    'cell_idx': full_idx,
                    'orig': orig,
                    'new': new_label,
                    'tissue': tissue,
                    'neighbors': t_idx[c_loc][n_row],
                    'reason': reason,
                    'orig_z': orig_p,
                    'win_z': best_p,
                    'original_confidence': original_confidence,
                    'dominant_neighbor_lineage': dominant_neighbor_lineage,
                    'dominant_neighbor_ratio': dominant_neighbor_ratio,
                    'marker_guard_status': marker_guard_status,
                    'candidate_marker_z': avg_new_z,
                    'original_marker_z': avg_orig_z
                })

    print(f"\n📊 [Spillover Rescue & Guardrail Summary]")
    print(f"  ➜ Stage 1: low-confidence cells compared against the surrounding dominant label (total {rescued_count + blocked_count:,} cases)")
    print(f"  ➜ Stage 2: original prediction retained or blocked by marker guardrails (Kept/Blocked): {blocked_count:,} cases")
    if blocked_count > 0:
        print("     [Detailed blocking reasons]")
        for k, v in sorted(blocked_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"       - {k}: {v:,} cases")
    print(f"  ➜ Stage 2: passed guardrails and finally corrected (Rescued): {rescued_count:,} cases")
    print("     [Detailed approval reasons]")
    for k, v in reason_counts.items():
        if v > 0: print(f"       - {k}: {v:,} cases")

    # Save logs to enable visualization after future kernel restarts or skipped steps
    with open(os.path.join(CKPT_DIR, 'mv1_rescued_log.pkl'), 'wb') as f:
        pickle.dump(rescued_log, f)
    rescue_qc_df = pd.DataFrame(rescue_qc_rows)
    rescue_qc_df.to_csv(RESCUE_QC_STEP02_PATH, index=False)
    print(f"  ✅ Rescue QC cell log saved: {RESCUE_QC_STEP02_PATH}")

# %%
# ==============================================================================
# [Step 02-4.5] Visualize Rescue (spatial correction) results (neighbor-cell mapping & marker expression)
# ==============================================================================
RESCUE_LOG_PATH = os.path.join(CKPT_DIR, 'mv1_rescued_log.pkl')

# Control Step 02 correction/figure outputs only in this block.
# By default, save only manuscript main candidates 2C-a/b, before/after correction matrices, and representative examples.
STEP02_RESCUE_OUTPUTS = {
    'rescue_examples': True,
    'rescue_eval_standalone': True,
    'rescue_changed_cm': True,
    'broad_cm_before_after': True,
    'lineage_composition_standalone': False,
    'figure2c_ab_single_panels': True,
    'figure2c_combined_qc': False,
    'figure2c_extra_single_panels': False,
    'figure2c_d_with_eval': True,
}
SAVE_STEP02_RESCUE_EXAMPLES = STEP02_RESCUE_OUTPUTS['rescue_examples']
SAVE_STEP02_RESCUE_EVAL_STANDALONE = STEP02_RESCUE_OUTPUTS['rescue_eval_standalone']
SAVE_STEP02_RESCUE_CHANGED_CM = STEP02_RESCUE_OUTPUTS['rescue_changed_cm']
SAVE_STEP02_BROAD_CM_BEFORE_AFTER = STEP02_RESCUE_OUTPUTS['broad_cm_before_after']
SAVE_STEP02_LINEAGE_COMPOSITION_STANDALONE = STEP02_RESCUE_OUTPUTS['lineage_composition_standalone']
SAVE_FIGURE2C_AB_SINGLE_PANELS = STEP02_RESCUE_OUTPUTS['figure2c_ab_single_panels']
SAVE_STEP02_RESCUE_QC_COMBINED = STEP02_RESCUE_OUTPUTS['figure2c_combined_qc']
SAVE_FIGURE2C_EXTRA_SINGLE_PANELS = STEP02_RESCUE_OUTPUTS['figure2c_extra_single_panels']
SAVE_FIGURE2C_D_WITH_EVAL = STEP02_RESCUE_OUTPUTS['figure2c_d_with_eval']


def _step02_save_figure(fig_obj, png_path, pptx_path=None, label='figure', show=True, close=True):
    """Handle PNG/PPTX saving and notebook display in one place."""
    return _step02_save_publication_figure(
        fig_obj, png_path, pptx_path,
        label=label, show=show, close=close,
        fixed_canvas=False
    )


def _step02_save_stem(fig_obj, file_stem, label='figure', show=True, close=True):
    """Save PNG/PPTX together using a file stem relative to FIG_DIR."""
    return _step02_save_figure(
        fig_obj,
        os.path.join(FIG_DIR, f'{file_stem}.png'),
        os.path.join(FIG_DIR, f'{file_stem}.pptx'),
        label=label,
        show=show,
        close=close,
    )

if skip_rescue and 'rescued_log' not in locals():
    if os.path.exists(RESCUE_LOG_PATH):
        import pickle
        with open(RESCUE_LOG_PATH, 'rb') as f:
            rescued_log = pickle.load(f)
            print(f"🔹 [Step 02-4.5] Restored Rescue logs ({len(rescued_log)} cases) from disk.")
    elif 'ML_Lineage_Pure' in adata.obs and 'ML_Lineage_Raw' in adata.obs:
        dlbcl_mask_for_plot = make_exact_disease_mask(adata, 'DLBCL')
        diff_mask = (adata.obs['ML_Lineage_Pure'] != adata.obs['ML_Lineage_Raw']) & dlbcl_mask_for_plot
        rescued_cells = np.where(diff_mask)[0]
        if len(rescued_cells) > 0:
            print(f"🔹 [Step 02-4.5] No log file; automatically restoring {len(rescued_cells)} corrected cells from adata...")
            rescued_log = []
            for idx in rescued_cells:
                orig = adata.obs['ML_Lineage_Pure'].iloc[idx]
                new = adata.obs['ML_Lineage_Raw'].iloc[idx]
                tissue = adata.obs['Run_Tissue_name'].iloc[idx]
                
                t_mask = (adata.obs['Run_Tissue_name'] == tissue) & dlbcl_mask_for_plot
                t_idx = np.where(t_mask)[0]
                c_loc = np.where(~np.isin(t_idx, rescued_cells))[0]
                
                if len(c_loc) >= 10:
                    X_spat = adata.obs[['X_centroid', 'Y_centroid']].iloc[t_idx].values.astype(np.float32)
                    knn = cuNN(n_neighbors=10).fit(X_spat[c_loc])
                    dists, nn_idx = knn.kneighbors(X_spat[t_idx == idx])
                    nn_idx_np = nn_idx[0].get() if hasattr(nn_idx[0], 'get') else nn_idx[0]
                    nn_indices = t_idx[c_loc][nn_idx_np]
                else:
                    nn_indices = []
                    
                rescued_log.append({
                    'cell_idx': idx,
                    'orig': orig,
                    'new': new,
                    'tissue': tissue,
                    'neighbors': nn_indices
                })
        else:
            print("🔹 [Step 02-4.5] No corrected (Rescued) cell records in adata.")

if 'rescued_log' in locals() and len(rescued_log) > 0:
    print("\n🔹 [Step 02-4.5] Generating visualizations of cells subjected to spatial correction (Rescue)...")
    
    if 'DYNAMIC_MARKER_MAP' in globals() and DYNAMIC_MARKER_MAP:
        marker_map = DYNAMIC_MARKER_MAP
    else:
        marker_map = {'T cells': ['CD3'], 'B cells': ['CD20'], 'NK cells': ['CD56'], 'Myeloid cells': ['CD68'], 'Stromal cells': ['PDPN']}
    
    # ── [Fix] Define marker order by lineage (B -> T -> Myeloid -> Stromal -> NK) ──
    lineage_order = ['B cells', 'T cells', 'Myeloid cells', 'Stromal cells', 'NK cells']
    ordered_markers = []
    for lin in lineage_order:
        m_list = marker_map.get(lin, [])
        if isinstance(m_list, str): m_list = [m_list]
        for m in m_list:
            if m not in ordered_markers:
                ordered_markers.append(m)
    
    valid_viz_markers = [m for m in ordered_markers if m in adata.var_names]
    marker_lineage_lookup = {}
    for lin in lineage_order:
        m_list = marker_map.get(lin, [])
        if isinstance(m_list, str):
            m_list = [m_list]
        for m in m_list:
            marker_lineage_lookup.setdefault(m, lin)
    
    # Extract the Top 3 most frequent transition types (e.g., 'B cells -> T cells')
    transitions = [f"{r['orig']} -> {r['new']}" for r in rescued_log]
    from collections import Counter
    top_transitions = [t[0] for t in Counter(transitions).most_common(3)]
    
    # Randomly sample and visualize one example per Top 3 transition type
    np.random.seed(RANDOM_SEED)
    viz_samples = []
    for t in top_transitions:
        t_logs = [r for r in rescued_log if f"{r['orig']} -> {r['new']}" == t]
        if t_logs:
            viz_samples.append(np.random.choice(t_logs))
            
    if viz_samples:
        def _safe_rescue_example_stem(value):
            safe = str(value).replace(' -> ', '_to_').replace('/', '_').replace('\\', '_')
            safe = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in safe)
            while '__' in safe:
                safe = safe.replace('__', '_')
            return safe.strip('_')[:80] or 'rescue_example'

        def _plot_single_rescue(sample, ax_sp, ax_ex, idx):
            is_primary_example = (idx == 1)
            title_fs = 9.8 if is_primary_example else 11.4
            axis_fs = 8.6 if is_primary_example else 10.4
            tick_fs = 7.8 if is_primary_example else 9.2
            marker_tick_fs = 8.6 if is_primary_example else 10.2
            bar_label_fs = 7.4 if is_primary_example else 8.6
            nearby_dot_size = 32 if is_primary_example else 50
            voting_dot_size = 112 if is_primary_example else 150
            target_dot_size = 380 if is_primary_example else 500
            c_idx = sample['cell_idx']
            orig = sample['orig']
            new = sample['new']
            tissue = sample['tissue']
            nn_indices = sample['neighbors']
            
            cx = adata.obs['X_centroid'].iloc[c_idx]
            cy = adata.obs['Y_centroid'].iloc[c_idx]
            
            t_mask = (adata.obs['Run_Tissue_name'] == tissue)
            t_data = adata.obs[t_mask]
            nearby_mask = ((t_data['X_centroid'] - cx)**2 + (t_data['Y_centroid'] - cy)**2) <= 100**2
            nearby_cells = t_data[nearby_mask]
            
            for lin in nearby_cells['ML_Lineage_Raw'].unique():
                lin_mask = nearby_cells['ML_Lineage_Raw'] == lin
                ax_sp.scatter(nearby_cells[lin_mask]['X_centroid'], nearby_cells[lin_mask]['Y_centroid'], 
                              color=get_color(lin), alpha=0.3, s=nearby_dot_size, label=lin)
                
            nn_df = adata.obs.iloc[nn_indices]
            ax_sp.scatter(nn_df['X_centroid'], nn_df['Y_centroid'], 
                          facecolors='none', edgecolors='black', s=voting_dot_size, linewidth=2, label='10-NN Voting Cells')
            
            for _, nn_row in nn_df.iterrows():
                nx, ny = nn_row['X_centroid'], nn_row['Y_centroid']
                dist = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                l_style = '-' if dist <= SPILLOVER_RADIUS_PX else '--'
                ax_sp.plot([cx, nx], [cy, ny], color='black', linestyle=l_style, alpha=0.5, linewidth=1, zorder=1)
                
            spillover_circle = patches.Circle((cx, cy), SPILLOVER_RADIUS_PX, color='red', fill=False, 
                                              linestyle='--', linewidth=2, label=f'Spillover Zone ({SPILLOVER_RADIUS_PX}px)')
            ax_sp.add_patch(spillover_circle)
                
            ax_sp.scatter([cx], [cy], color=get_color(new), marker='*', s=target_dot_size, edgecolors='gold', linewidth=2, zorder=5, label='Target Cell')
            
            title_prefix = '' if is_primary_example else f'Example {idx}: '
            ax_sp.set_title(
                f"{title_prefix}{orig} -> {new}\n{tissue} | KNN context",
                fontsize=title_fs, fontweight='bold', pad=3
            )
            ax_sp.set_xlabel("X (px)", fontsize=axis_fs, fontweight='bold', labelpad=3)
            ax_sp.set_ylabel("Y (px)", fontsize=axis_fs, fontweight='bold', labelpad=3)
            ax_sp.tick_params(axis='both', labelsize=tick_fs)
            ax_sp.grid(True, linestyle=':', alpha=0.5)
            spatial_window_px = 125 if is_primary_example else 105
            ax_sp.set_xlim(cx - spatial_window_px, cx + spatial_window_px)
            ax_sp.set_ylim(cy - spatial_window_px, cy + spatial_window_px)
            if is_primary_example:
                # Because a single representative example requires matching left/right panel heights,
                # fix the aspect using datalim so Matplotlib does not shrink the axes boxes again.
                ax_sp.set_aspect('equal', adjustable='datalim')
            else:
                ax_sp.set_aspect('equal', adjustable='box')
                if hasattr(ax_sp, 'set_box_aspect'):
                    ax_sp.set_box_aspect(1)
            ax_sp.set_anchor('E')
            spatial_handles, spatial_labels = ax_sp.get_legend_handles_labels()
            
            if valid_viz_markers:
                marker_indices = [
                    int(adata.var_names.get_loc(marker))
                    for marker in valid_viz_markers
                ]
                # Backed AnnData/HDF5 permits one fancy index per read. Load
                # the small row subset first, then select marker columns in RAM.
                expr_raw = adata[c_idx, :].X
                if hasattr(expr_raw, "toarray"): expr_raw = expr_raw.toarray()
                expr_raw = np.asarray(expr_raw).reshape(-1)[marker_indices]
                
                tissue_indices = np.flatnonzero(np.asarray(t_mask, dtype=bool))
                t_expr = adata[tissue_indices, :].X
                if hasattr(t_expr, "toarray"): t_expr = t_expr.toarray()
                t_expr = np.asarray(t_expr)[:, marker_indices]
                t_mean = np.mean(t_expr, axis=0)
                t_std  = np.std(t_expr, axis=0) + 1e-8
                expr_z = (expr_raw - t_mean) / t_std
                
                x_pos = np.arange(len(valid_viz_markers))
                target_markers = marker_map.get(new, [])
                if isinstance(target_markers, str): target_markers = [target_markers]
                
                bars = ax_ex.bar(x_pos, expr_z, color=[get_color(new) if m in target_markers else '#95A5A6' for m in valid_viz_markers])
                
                ax_ex.set_xticks(x_pos)
                marker_rotation = 36 if is_primary_example else 45
                ax_ex.set_xticklabels(valid_viz_markers, fontweight='bold', fontsize=marker_tick_fs, rotation=marker_rotation, ha='right', rotation_mode='anchor')
                for tick_label, marker_name in zip(ax_ex.get_xticklabels(), valid_viz_markers):
                    tick_label.set_color(get_color(marker_lineage_lookup.get(marker_name, new)))
                    tick_label.set_fontweight('bold')
                ax_ex.set_ylabel("Z-score", fontsize=axis_fs, fontweight='bold', labelpad=1)
                ax_ex.set_title("Rescued-cell marker expression", fontsize=title_fs, fontweight='bold', pad=3)
                ax_ex.grid(True, axis='y', linestyle='--', alpha=0.5)
                ax_ex.tick_params(axis='y', labelsize=tick_fs)
                ax_ex.set_anchor('W')
                
                bar_heights = np.array([bar.get_height() for bar in bars], dtype=float)
                max_val = float(np.nanmax(np.r_[bar_heights, 0]))
                min_val = float(np.nanmin(np.r_[bar_heights, 0]))
                value_span = max(max_val - min_val, 1.0)
                label_pad = value_span * 0.08
                ylim_bottom = min_val - value_span * 0.40
                ylim_top = max_val + value_span * 0.48
                if is_primary_example:
                    ylim_top = 4.5
                ax_ex.set_ylim(bottom=ylim_bottom, top=ylim_top)
                
                for bar in bars:
                    yval = float(bar.get_height())
                    if yval >= 0:
                        y_text = min(yval + label_pad, ylim_top - label_pad * 1.8)
                        va = 'bottom'
                    else:
                        y_text = max(yval - label_pad, ylim_bottom + label_pad * 1.8)
                        va = 'top'
                    ax_ex.text(bar.get_x() + bar.get_width() / 2, y_text, f'{yval:.1f}', ha='center', va=va, fontweight='bold', fontsize=bar_label_fs, clip_on=True)
            
            return spatial_handles, spatial_labels, orig, new, tissue

        def _add_axes_in_inches(fig_obj, left_in, bottom_in, width_in, height_in):
            """Place Axes directly on the final canvas in inches."""
            fig_w, fig_h = fig_obj.get_size_inches()
            return fig_obj.add_axes([
                left_in / fig_w,
                bottom_in / fig_h,
                width_in / fig_w,
                height_in / fig_h,
            ])

        def _add_rescue_legends(
            fig_obj, handles, labels, *, legend_y, fontsize, markerscale=1.00
        ):
            """Arrange Lineage and KNN-context symbols in a single-row legend."""
            context_labels = {'10-NN Voting Cells', f'Spillover Zone ({SPILLOVER_RADIUS_PX}px)', 'Target Cell'}
            lineage_map = {}
            for handle, label in zip(handles, labels):
                if label not in context_labels and label not in lineage_map:
                    lineage_map[label] = handle
            context_handles = [
                mlines.Line2D([], [], marker='o', linestyle='None', markerfacecolor='none',
                              markeredgecolor='black', markeredgewidth=1.5, markersize=6.8),
                mlines.Line2D([], [], color='red', linestyle='--', linewidth=2.0),
                mlines.Line2D([], [], marker='*', linestyle='None', markerfacecolor='#F1C40F',
                              markeredgecolor='black', markeredgewidth=1.0, markersize=9.2),
            ]
            context_text = ['10-NN voting cells', f'Spillover zone ({SPILLOVER_RADIUS_PX}px)', 'Target cell']
            legend_handles = list(lineage_map.values()) + context_handles
            legend_labels = list(lineage_map.keys()) + context_text
            fig_obj.legend(
                legend_handles, legend_labels,
                loc='lower center', bbox_to_anchor=(0.5, legend_y),
                ncol=len(legend_labels), fontsize=fontsize, frameon=False,
                markerscale=markerscale, handlelength=1.55,
                handletextpad=0.55, columnspacing=1.08,
                borderaxespad=0.0
            )

        if len(viz_samples) > 0:
            sample = viz_samples[0]
            _fig_w_cm, _fig_h_cm = STEP02_RESCUE_EXAMPLE_SIZE_CM
            fig_r = plt.figure(figsize=(_fig_w_cm * STEP02_CM_TO_IN, _fig_h_cm * STEP02_CM_TO_IN))
            # [User adjustments: Example 1 standalone figure, units: cm]
            # _left_cm: overall left margin. Smaller values shift left, but overly small values clip the Y-axis label.
            # _gap_cm: gap between the spatial map and marker bars.
            # _bottom_cm: panel bottom position. Larger values move away from the legend; smaller values move closer.
            # _panel_h_cm: vertical height containing both panels.
            # _sp_w_cm: spatial-map panel width. Set equal to _panel_h_cm for the existing square ratio.
            # _ex_w_cm: marker-bar panel width. Using all remaining width makes the bar plot appear excessively large.
            _left_cm = 1.45
            _gap_cm = 1.35
            _bottom_cm = 2.05
            _panel_h_cm = 5.10
            _sp_w_cm = _panel_h_cm
            _ex_w_cm = 12.15
            _left_in = _left_cm * STEP02_CM_TO_IN
            _gap_in = _gap_cm * STEP02_CM_TO_IN
            _bottom_in = _bottom_cm * STEP02_CM_TO_IN
            _panel_h_in = _panel_h_cm * STEP02_CM_TO_IN
            _sp_w_in = _sp_w_cm * STEP02_CM_TO_IN
            _ex_w_in = _ex_w_cm * STEP02_CM_TO_IN
            ax_sp = _add_axes_in_inches(fig_r, _left_in, _bottom_in, _sp_w_in, _panel_h_in)
            ax_ex = _add_axes_in_inches(fig_r, _left_in + _sp_w_in + _gap_in, _bottom_in, _ex_w_in, _panel_h_in)
            _sp_bounds = ax_sp.get_position().bounds
            _ex_bounds = ax_ex.get_position().bounds
            handles, labels, orig, new, tissue = _plot_single_rescue(sample, ax_sp, ax_ex, 1)
            # Reapply original positions after aspect/equal handling to keep the left/right plot-frame heights matched.
            ax_sp.set_position(_sp_bounds)
            ax_ex.set_position(_ex_bounds)
            
            transition_stem = _safe_rescue_example_stem(f"{orig}_to_{new}")
            tissue_stem = _safe_rescue_example_stem(tissue)
            viz_path = os.path.join(FIG_DIR, f'Final_mv1_Step02-04_Rescue_Example_01_{transition_stem}_{tissue_stem}.png')
            viz_pptx_path = os.path.join(FIG_DIR, f'Final_mv1_Step02-04_Rescue_Example_01_{transition_stem}_{tissue_stem}.pptx')
            _add_rescue_legends(
                fig_r, handles, labels,
                legend_y=0.035, fontsize=9.0, markerscale=1.15
            )
            _step02_save_publication_figure(
                fig_r, viz_path, viz_pptx_path,
                label='Representative correction example 1',
                fixed_canvas=True
            )
            
        if len(viz_samples) > 1:
            samples_to_plot = viz_samples[1:]
            n_rows = len(samples_to_plot)
            combined_fig_w, combined_fig_h = STEP02_RESCUE_EXAMPLES_COMBINED_SIZE
            fig_r2 = plt.figure(
                figsize=(combined_fig_w, combined_fig_h if n_rows > 1 else 4.6)
            )
            # [User adjustments: Examples 2-3 combined figure, units: inches]
            # row_bottoms_in: bottom position of each row. Adjust inter-row spacing with these differences and row_height_in.
            # row_height_in: panel height for each row. Larger values enlarge panels and reduce inter-row spacing.
            # _left_in: combined figure left margin. Recommended: 0.80~1.05; overly small values clip the Y-axis label.
            # _gap_in: gap between the left spatial map and right marker bars.
            # _sp_w_in: spatial-map panel width. Set equal to row_height_in for a square ratio.
            # _ex_w_in: marker-bar width. Automatically calculated by subtracting the other values from the total width.
            if n_rows > 1:
                # Reserve inter-row spacing within the creation canvas so the top row x-label
                # does not overlap the bottom row title.
                row_bottoms_in = [4.72, 1.35]
                row_height_in = 2.50
            else:
                row_bottoms_in = [1.05]
                row_height_in = 3.05
            _left_in = 0.80
            _gap_in = 0.78
            _right_margin_in = 0.25
            _sp_w_in = row_height_in
            _ex_w_in = combined_fig_w - _left_in - _sp_w_in - _gap_in - _right_margin_in
            
            all_handles, all_labels = [], []
            for row_idx, sample in enumerate(samples_to_plot):
                row_bottom = row_bottoms_in[row_idx]
                ax_sp = _add_axes_in_inches(fig_r2, _left_in, row_bottom, _sp_w_in, row_height_in)
                ax_ex = _add_axes_in_inches(fig_r2, _left_in + _sp_w_in + _gap_in, row_bottom, _ex_w_in, row_height_in)
                h, l, o, n, t = _plot_single_rescue(sample, ax_sp, ax_ex, row_idx + 2)
                all_handles.extend(h)
                all_labels.extend(l)
                
            _add_rescue_legends(
                fig_r2, all_handles, all_labels,
                legend_y=0.020, fontsize=11.2, markerscale=1.25
            )
            
            viz_path2 = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Rescue_Example_Combined_02_03.png')
            viz_pptx_path2 = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Rescue_Example_Combined_02_03.pptx')
            _step02_save_publication_figure(
                fig_r2, viz_path2, viz_pptx_path2,
                label='Representative correction examples 2 and 3 combined',
                fixed_canvas=True
            )
    else:
        print("  ⚠️ No transition samples are available for visualization.")

    # ── [Additional validation] Evaluate accuracy before vs after correction ──
    print("\n🔹 [Step 02-4.5] Evaluating accuracy before vs after Rescue...")
    dlbcl_mask_eval = make_exact_disease_mask(adata, 'DLBCL')
    valid_gt_mask, _ = make_valid_label_mask(adata.obs['GT_Lineage_Broad'].values)
    bench_mask = dlbcl_mask_eval & valid_gt_mask
    adata_bench = adata[bench_mask]
    
    if len(adata_bench) > 0:
        y_true = adata_bench.obs['GT_Lineage_Broad'].astype(str)
        y_before = adata_bench.obs['ML_Lineage_Pure'].astype(str)
        y_after  = adata_bench.obs['ML_Lineage_Raw'].astype(str)
        
        bacc_before = balanced_accuracy_score(y_true, y_before)
        bacc_after  = balanced_accuracy_score(y_true, y_after)
        f1_before = f1_score(y_true, y_before, average='macro')
        f1_after  = f1_score(y_true, y_after, average='macro')
        
        print(f"  [Before Rescue (Pure ML)] B-Acc: {bacc_before*100:.2f}% | Macro F1: {f1_before*100:.2f}%")
        print(f"  [After Rescue  (Raw ML)] B-Acc: {bacc_after*100:.2f}% | Macro F1: {f1_after*100:.2f}%")
        
        diff_bacc = bacc_after - bacc_before
        diff_f1   = f1_after - f1_before
        print(f"  ➜ Performance change: B-Acc {'+' if diff_bacc>=0 else ''}{diff_bacc*100:.2f}%p, F1 {'+' if diff_f1>=0 else ''}{diff_f1*100:.2f}%p")
        
        # Display a simple bar chart
        df_eval = pd.DataFrame({
            'Stage': ['Before (Pure ML)', 'After (Rescue)', 'Before (Pure ML)', 'After (Rescue)'],
            'Metric': ['Balanced Acc', 'Balanced Acc', 'Macro F1', 'Macro F1'],
            'Score': [bacc_before, bacc_after, f1_before, f1_after]
        })
        # Design the canvas directly as a compact panel for comparing both metrics at a glance.
        _eval_fig_w = 8.2 * STEP02_CM_TO_IN
        _eval_fig_h = 6.3 * STEP02_CM_TO_IN
        fig_eval = plt.figure(figsize=(_eval_fig_w, _eval_fig_h))
        ax_eval = fig_eval.add_subplot(1, 1, 1)
        metric_order = ['Balanced Acc', 'Macro F1']
        stage_order = ['Before (Pure ML)', 'After (Rescue)']
        stage_colors = {'Before (Pure ML)': '#95A5A6', 'After (Rescue)': '#E74C3C'}
        metric_x = np.array([0.0, 1.0])
        bar_width = 0.26
        bar_offsets = {'Before (Pure ML)': -0.15, 'After (Rescue)': 0.15}
        eval_bar_patches = []
        for stage in stage_order:
            stage_scores = [
                float(df_eval[(df_eval['Metric'] == metric) & (df_eval['Stage'] == stage)]['Score'].iloc[0])
                for metric in metric_order
            ]
            bars_eval = ax_eval.bar(
                metric_x + bar_offsets[stage],
                stage_scores,
                width=bar_width,
                color=stage_colors[stage],
                edgecolor='black',
                label=stage
            )
            eval_bar_patches.extend(bars_eval)
        ax_eval.set_ylim(0, 1.02)
        ax_eval.set_yticks(np.linspace(0.0, 1.0, 6))
        ax_eval.set_xlim(-0.55, 1.55)
        ax_eval.set_xticks(metric_x)
        ax_eval.set_xticklabels(metric_order)
        ax_eval.set_xlabel("")
        ax_eval.set_ylabel("Score", fontsize=9.0, fontweight='bold', labelpad=3)
        plt.setp(ax_eval.get_xticklabels(), fontsize=8.3, fontweight='bold')
        plt.setp(ax_eval.get_yticklabels(), fontsize=8.0, fontweight='bold')
        ax_eval.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.32)
        ax_eval.spines['top'].set_visible(False)
        ax_eval.spines['right'].set_visible(False)
        for p in eval_bar_patches:
            h = p.get_height()
            if h > 0:
                ax_eval.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', xytext=(0, 3), textcoords='offset points',
                                fontsize=8.0, fontweight='bold')
        handles_eval, labels_eval = ax_eval.get_legend_handles_labels()
        fig_eval.legend(
            handles_eval, labels_eval,
            loc='lower center', bbox_to_anchor=(0.5, 0.025),
            ncol=2, fontsize=7.4, title=None, frameon=False,
            borderaxespad=0.0, columnspacing=1.0, handlelength=1.1
        )
        fig_eval.subplots_adjust(left=0.145, right=0.985, top=0.94, bottom=0.27)
        eval_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Rescue_Eval.png')
        eval_pptx_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Rescue_Eval.pptx')
        if SAVE_STEP02_RESCUE_EVAL_STANDALONE:
            _step02_save_publication_figure(
                fig_eval, eval_path, eval_pptx_path,
                label="Rescue evaluation figure",
                fixed_canvas=True
            )
        else:
            plt.close(fig_eval)
        
        # ── [Detailed analysis heatmap of changed cells] ──
        changed_mask = (y_before != y_after)
        if changed_mask.sum() > 0:
            y_true_c = y_true[changed_mask]
            y_bef_c  = y_before[changed_mask]
            y_aft_c  = y_after[changed_mask]
            
            # Combine all existing labels (GT, Before, After) to create consistent axes
            ordered_classes = ['B cells', 'Myeloid cells', 'NK cells', 'Stromal cells', 'T cells']
            existing_classes = set(y_true_c.unique()) | set(y_bef_c.unique()) | set(y_aft_c.unique())
            ordered_classes = [c for c in ordered_classes if c in existing_classes]
            # Append any labels beyond the five default lineages above
            for c in existing_classes:
                if c not in ordered_classes: ordered_classes.append(c)
                
            fig, axes = plt.subplots(1, 2, figsize=STEP02_RESCUE_CM_SIZE, gridspec_kw={'wspace': 0.12})
            
            # Left: Before Rescue
            cm_bef = pd.crosstab(y_true_c, y_bef_c)
            cm_bef = cm_bef.reindex(index=ordered_classes, columns=ordered_classes, fill_value=0)
            cm_bef.index = [c.replace(' cells', '') for c in cm_bef.index]
            cm_bef.columns = [c.replace(' cells', '') for c in cm_bef.columns]
            sns.heatmap(cm_bef, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, linewidths=.5, square=True, annot_kws={"size": 13, "weight": "bold"})
            axes[0].set_title("Before Rescue", fontsize=13.5, fontweight='bold', pad=4)
            axes[0].set_ylabel("GT lineage", fontsize=12.5, fontweight='bold', labelpad=5)
            axes[0].set_xlabel("Predicted (Pure ML)", fontsize=11.5, fontweight='bold', labelpad=5)
            plt.setp(axes[0].get_xticklabels(), fontweight='bold', rotation=35, ha='right', rotation_mode='anchor', fontsize=10.5)
            plt.setp(axes[0].get_yticklabels(), fontweight='bold', rotation=0, fontsize=10.5)
            
            # Right: After Rescue
            cm_aft = pd.crosstab(y_true_c, y_aft_c)
            cm_aft = cm_aft.reindex(index=ordered_classes, columns=ordered_classes, fill_value=0)
            cm_aft.index = [c.replace(' cells', '') for c in cm_aft.index]
            cm_aft.columns = [c.replace(' cells', '') for c in cm_aft.columns]
            sns.heatmap(cm_aft, annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar=False, linewidths=.5, square=True, annot_kws={"size": 13, "weight": "bold"})
            axes[1].set_title("After Rescue", fontsize=13.5, fontweight='bold', pad=4)
            axes[1].set_ylabel("")
            axes[1].set_xlabel("Predicted (Raw ML)", fontsize=11.5, fontweight='bold', labelpad=5)
            axes[1].set_yticklabels([])
            axes[1].tick_params(axis='y', left=False)
            plt.setp(axes[1].get_xticklabels(), fontweight='bold', rotation=35, ha='right', rotation_mode='anchor', fontsize=10.5)
            plt.setp(axes[1].get_yticklabels(), fontweight='bold', rotation=0, fontsize=10.5)
            
            for _ax in axes:
                if hasattr(_ax, 'set_box_aspect'):
                    _ax.set_box_aspect(1)
            fig.subplots_adjust(left=0.115, right=0.985, top=0.915, bottom=0.175, wspace=0.12)
            cm_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Rescue_Changed_CM.png')
            cm_pptx_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Rescue_Changed_CM.pptx')
            if SAVE_STEP02_RESCUE_CHANGED_CM:
                _step02_save_publication_figure(
                    fig, cm_path, cm_pptx_path,
                    label='Rescue changed-cell CM',
                    fixed_canvas=True
                )
            else:
                plt.close(fig)
        else:
            print("  ⚠️ No GT-labeled cells had their labels changed by correction (Rescue).")
            
    else:
        print("  ⚠️ No DLBCL cells have GT (ground-truth) labels; cannot benchmark before vs after correction.")
        
else:
    print("🔹 [Step 02-4.5] No corrected (Rescued) cell logs; skipping visualization.")

# %%
# ==============================================================================
# [Step 02-4.6] Rescue QC: confusion matrix, confidence, guardrail, composition
# ==============================================================================
# Control output selection only through STEP02_RESCUE_OUTPUTS at the start of Step 02-4.5.

print("\n🔹 [Step 02-4.6] Computing Rescue QC...")

dlbcl_mask_qc = make_exact_disease_mask(adata, 'DLBCL')
valid_gt_mask_qc, _ = make_valid_label_mask(adata.obs['GT_Lineage_Broad'].values)
qc_classes_default = ['B cells', 'Myeloid cells', 'NK cells', 'Stromal cells', 'T cells']

if {'ML_Lineage_Pure', 'ML_Lineage_Raw', 'GT_Lineage_Broad'}.issubset(adata.obs.columns):
    qc_bench_mask = dlbcl_mask_qc & valid_gt_mask_qc
    qc_obs = adata.obs.loc[qc_bench_mask].copy()

    if len(qc_obs) > 0:
        y_true_all = qc_obs['GT_Lineage_Broad'].astype(str)
        y_before_all = qc_obs['ML_Lineage_Pure'].astype(str)
        y_after_all = qc_obs['ML_Lineage_Raw'].astype(str)
        cm_classes = [c for c in qc_classes_default if c in set(y_true_all) | set(y_before_all) | set(y_after_all)]
        for c in sorted((set(y_true_all) | set(y_before_all) | set(y_after_all)) - set(cm_classes)):
            cm_classes.append(c)

        cm_before_all = pd.crosstab(y_true_all, y_before_all).reindex(index=cm_classes, columns=cm_classes, fill_value=0)
        cm_after_all = pd.crosstab(y_true_all, y_after_all).reindex(index=cm_classes, columns=cm_classes, fill_value=0)
        cm_before_norm = cm_before_all.div(cm_before_all.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        cm_after_norm = cm_after_all.div(cm_after_all.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

        cm_before_path = os.path.join(CKPT_DIR, 'mv1_step02_broad_cm_before_rescue.csv')
        cm_after_path = os.path.join(CKPT_DIR, 'mv1_step02_broad_cm_after_rescue.csv')
        cm_before_norm_path = os.path.join(CKPT_DIR, 'mv1_step02_broad_cm_before_rescue_row_norm.csv')
        cm_after_norm_path = os.path.join(CKPT_DIR, 'mv1_step02_broad_cm_after_rescue_row_norm.csv')
        cm_before_all.to_csv(cm_before_path)
        cm_after_all.to_csv(cm_after_path)
        cm_before_norm.to_csv(cm_before_norm_path)
        cm_after_norm.to_csv(cm_after_norm_path)

        fig_cm, axes_cm = plt.subplots(1, 2, figsize=STEP02_RESCUE_CM_SIZE, gridspec_kw={'wspace': 0.12})
        cm_before_all.index = [c.replace(' cells', '') for c in cm_before_all.index]
        cm_before_all.columns = [c.replace(' cells', '') for c in cm_before_all.columns]
        cm_after_all.index = [c.replace(' cells', '') for c in cm_after_all.index]
        cm_after_all.columns = [c.replace(' cells', '') for c in cm_after_all.columns]
        
        sns.heatmap(cm_before_all, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0], cbar=False,
                    linewidths=.5, square=True, annot_kws={"size": 13, "weight": "bold"})
        axes_cm[0].set_title("Before Rescue", fontsize=13.5, fontweight='bold', pad=4)
        axes_cm[0].set_xlabel("Predicted (Pure ML)", fontsize=11.5, fontweight='bold', labelpad=5)
        axes_cm[0].set_ylabel("GT lineage", fontsize=12.5, fontweight='bold', labelpad=5)
        sns.heatmap(cm_after_all, annot=True, fmt='d', cmap='Reds', ax=axes_cm[1], cbar=False,
                    linewidths=.5, square=True, annot_kws={"size": 13, "weight": "bold"})
        axes_cm[1].set_title("After Rescue", fontsize=13.5, fontweight='bold', pad=4)
        axes_cm[1].set_xlabel("Predicted (Raw ML)", fontsize=11.5, fontweight='bold', labelpad=5)
        axes_cm[1].set_ylabel("")
        axes_cm[1].set_yticklabels([])
        axes_cm[1].tick_params(axis='y', left=False)
        for ax in axes_cm:
            plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor', fontweight='bold', fontsize=10.5)
            plt.setp(ax.get_yticklabels(), rotation=0, fontweight='bold', fontsize=10.5)
        for _ax in axes_cm:
            if hasattr(_ax, 'set_box_aspect'):
                _ax.set_box_aspect(1)
        fig_cm.subplots_adjust(left=0.115, right=0.985, top=0.915, bottom=0.175, wspace=0.12)
        cm_fig_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Broad_CM_Before_After.png')
        cm_fig_pptx_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Broad_CM_Before_After.pptx')
        if SAVE_STEP02_BROAD_CM_BEFORE_AFTER:
            _step02_save_publication_figure(
                fig_cm, cm_fig_path, cm_fig_pptx_path,
                label='All-DLBCL broad confusion matrix',
                fixed_canvas=True
            )
        else:
            plt.close(fig_cm)
    else:
        print("  ⚠️ No DLBCL cells with GT labels; skipping the overall confusion matrix.")

    comp_rows = []
    for stage, col in [('Before Rescue', 'ML_Lineage_Pure'), ('After Rescue', 'ML_Lineage_Raw')]:
        stage_labels = adata.obs.loc[dlbcl_mask_qc, col].astype(str)
        stage_counts = stage_labels.value_counts()
        total_count = int(stage_counts.sum())
        for lineage, count in stage_counts.items():
            comp_rows.append({
                'stage': stage,
                'lineage': lineage,
                'count': int(count),
                'proportion': float(count / total_count) if total_count > 0 else np.nan
            })

    lineage_comp_df = pd.DataFrame(comp_rows)
    comp_path = os.path.join(CKPT_DIR, 'mv1_step02_lineage_composition_before_after.csv')
    lineage_comp_df.to_csv(comp_path, index=False)
    if not lineage_comp_df.empty:
        comp_delta = (
            lineage_comp_df
            .pivot(index='lineage', columns='stage', values=['count', 'proportion'])
            .fillna(0)
        )
        comp_delta.columns = [f"{metric}_{stage.replace(' ', '_').lower()}" for metric, stage in comp_delta.columns]
        comp_delta = comp_delta.reset_index()
        if {'count_before_rescue', 'count_after_rescue'}.issubset(comp_delta.columns):
            comp_delta['delta_count'] = comp_delta['count_after_rescue'] - comp_delta['count_before_rescue']
        if {'proportion_before_rescue', 'proportion_after_rescue'}.issubset(comp_delta.columns):
            comp_delta['delta_proportion'] = comp_delta['proportion_after_rescue'] - comp_delta['proportion_before_rescue']
        comp_delta_path = os.path.join(CKPT_DIR, 'mv1_step02_lineage_composition_delta.csv')
        comp_delta.to_csv(comp_delta_path, index=False)

        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        order_lineages = [c for c in qc_classes_default if c in lineage_comp_df['lineage'].unique()]
        for c in sorted(set(lineage_comp_df['lineage']) - set(order_lineages)):
            order_lineages.append(c)
        sns.barplot(data=lineage_comp_df, x='lineage', y='proportion', hue='stage',
                    order=order_lineages, palette=['#95A5A6', '#E74C3C'], edgecolor='black', ax=ax_comp)
        ax_comp.set_xlabel("Predicted broad lineage", fontsize=12, fontweight='bold')
        ax_comp.set_ylabel("Proportion among DLBCL cells", fontsize=12, fontweight='bold')
        ax_comp.set_ylim(0, max(0.05, lineage_comp_df['proportion'].max() * 1.20))
        plt.setp(ax_comp.get_xticklabels(), rotation=25, ha='right', fontweight='bold')
        fig_comp.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.28)
        comp_fig_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Lineage_Composition_Before_After.png')
        comp_fig_pptx_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Lineage_Composition_Before_After.pptx')
        if SAVE_STEP02_LINEAGE_COMPOSITION_STANDALONE:
            _step02_save_publication_figure(
                fig_comp, comp_fig_path, comp_fig_pptx_path,
                label='Lineage composition before/after',
                fixed_canvas=False
            )
        else:
            plt.close(fig_comp)
else:
    print("  ⚠️ ML_Lineage_Pure/Raw or GT_Lineage_Broad columns are missing; skipping before/after QC.")

if 'rescue_qc_df' not in locals():
    if os.path.exists(RESCUE_QC_STEP02_PATH):
        rescue_qc_df = pd.read_csv(RESCUE_QC_STEP02_PATH)
        print(f"  🔹 Restoring existing Rescue QC cell log: {RESCUE_QC_STEP02_PATH}")
    else:
        rescue_qc_df = pd.DataFrame()

if not rescue_qc_df.empty:
    rescue_qc_df.to_csv(RESCUE_QC_STEP02_PATH, index=False)
    guardrail_summary = (
        rescue_qc_df
        .groupby(['rescue_status', 'marker_guard_status'], dropna=False)
        .size()
        .reset_index(name='count')
    )
    guardrail_summary['fraction_of_attempted'] = guardrail_summary['count'] / len(rescue_qc_df)
    guardrail_summary.to_csv(RESCUE_GUARDRAIL_STEP02_PATH, index=False)

    rescued_qc_df = rescue_qc_df[rescue_qc_df['rescue_status'] == 'rescued'].copy()
    transition_summary = (
        rescued_qc_df
        .groupby(['before_lineage', 'after_lineage', 'dominant_neighbor_lineage'], dropna=False)
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    transition_path = os.path.join(CKPT_DIR, 'mv1_step02_rescue_transition_summary.csv')
    transition_summary.to_csv(transition_path, index=False)

    # Manuscript figure highlighting KNN Rescue:
    # 1) Distribution of low-confidence candidates, 2) actual prediction transitions before/after rescue,
    # 3) rescue-induced changes in the GT-based confusion matrix, 4) marker-guardrail approval/blocking proportions.
    fig_qc, axes_qc = plt.subplots(2, 2, figsize=(18, 12))
    conf_plot_df = rescue_qc_df.dropna(subset=['original_confidence'])
    if not conf_plot_df.empty:
        rescue_status_display_map = {
            'kept_or_blocked': 'Not changed cells',
            'blocked': 'No nearby cells',
            'rescued': 'Rescued'
        }
        conf_plot_df = conf_plot_df.copy()
        conf_plot_df['rescue_decision'] = (
            conf_plot_df['rescue_status']
            .astype(str)
            .map(rescue_status_display_map)
            .fillna(conf_plot_df['rescue_status'].astype(str))
        )
        sns.histplot(data=conf_plot_df, x='original_confidence', hue='rescue_decision',
                     bins=30, stat='density', common_norm=False, element='step', ax=axes_qc[0, 0])
        axes_qc[0, 0].axvline(0.6, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        axes_qc[0, 0].set_xlabel("ML Broad Confidence before rescue", fontsize=12, fontweight='bold')
        axes_qc[0, 0].set_ylabel("Density", fontsize=12, fontweight='bold')
    else:
        axes_qc[0, 0].text(0.5, 0.5, "No confidence values", ha='center', va='center', fontweight='bold')
        axes_qc[0, 0].set_axis_off()

    trans_cm = None
    delta_gt_cm = None
    max_delta = None

    if not rescued_qc_df.empty:
        ordered_lineages = ['B cells', 'Myeloid cells', 'NK cells', 'Stromal cells', 'T cells']

        trans_cm = pd.crosstab(rescued_qc_df['before_lineage'], rescued_qc_df['after_lineage'])
        trans_cm = trans_cm.reindex(index=ordered_lineages, columns=ordered_lineages, fill_value=0)
        sns.heatmap(trans_cm, annot=True, fmt='d', cmap='Purples', ax=axes_qc[0, 1], cbar=False,
                    linewidths=.5, annot_kws={"size": 11, "weight": "bold"})
        axes_qc[0, 1].set_xlabel("After rescue prediction", fontsize=12, fontweight='bold')
        axes_qc[0, 1].set_ylabel("Before rescue prediction", fontsize=12, fontweight='bold')

        rescued_qc_gt_df = rescued_qc_df.copy()
        rescued_qc_gt_df['cell_idx'] = pd.to_numeric(rescued_qc_gt_df['cell_idx'], errors='coerce')
        rescued_qc_gt_df = rescued_qc_gt_df.dropna(subset=['cell_idx']).copy()
        rescued_qc_gt_df['cell_idx'] = rescued_qc_gt_df['cell_idx'].astype(int)
        valid_idx_mask = (rescued_qc_gt_df['cell_idx'] >= 0) & (rescued_qc_gt_df['cell_idx'] < adata.n_obs)
        rescued_qc_gt_df = rescued_qc_gt_df.loc[valid_idx_mask].copy()

        if not rescued_qc_gt_df.empty and 'GT_Lineage_Broad' in adata.obs.columns:
            rescued_qc_gt_df['GT_Lineage_Broad'] = (
                adata.obs['GT_Lineage_Broad']
                .iloc[rescued_qc_gt_df['cell_idx'].to_numpy()]
                .astype(str)
                .to_numpy()
            )
            before_gt_cm = pd.crosstab(rescued_qc_gt_df['GT_Lineage_Broad'], rescued_qc_gt_df['before_lineage'])
            after_gt_cm = pd.crosstab(rescued_qc_gt_df['GT_Lineage_Broad'], rescued_qc_gt_df['after_lineage'])
            before_gt_cm = before_gt_cm.reindex(index=ordered_lineages, columns=ordered_lineages, fill_value=0)
            after_gt_cm = after_gt_cm.reindex(index=ordered_lineages, columns=ordered_lineages, fill_value=0)
            delta_gt_cm = after_gt_cm - before_gt_cm
            max_delta = max(1, int(np.nanmax(np.abs(delta_gt_cm.to_numpy()))))
            sns.heatmap(delta_gt_cm, annot=True, fmt='+d', cmap='RdBu_r', center=0,
                        vmin=-max_delta, vmax=max_delta, ax=axes_qc[1, 0], cbar=True,
                        linewidths=.5, annot_kws={"size": 11, "weight": "bold"})
            axes_qc[1, 0].set_xlabel("Predicted broad lineage", fontsize=12, fontweight='bold')
            axes_qc[1, 0].set_ylabel("Ground truth broad lineage", fontsize=12, fontweight='bold')
        else:
            axes_qc[1, 0].text(0.5, 0.5, "GT labels unavailable", ha='center', va='center',
                               fontweight='bold')
            axes_qc[1, 0].set_axis_off()
    else:
        axes_qc[0, 1].text(0.5, 0.5, "No rescued cells", ha='center', va='center', fontweight='bold')
        axes_qc[0, 1].set_axis_off()
        axes_qc[1, 0].text(0.5, 0.5, "No rescued cells", ha='center', va='center', fontweight='bold')
        axes_qc[1, 0].set_axis_off()

    guard_plot = guardrail_summary.copy()
    status_label_map = {
        ('blocked', 'not_evaluated'): 'No nearby\ncells',
        ('kept_or_blocked', 'not_evaluated'): 'Not\nchanged',
        ('kept_or_blocked', 'blocked'): 'Marker\nmismatch',
        ('rescued', 'passed'): 'Rescued',
        ('rescued', 'not_evaluable_missing_marker'): 'Rescued\nmarker missing'
    }
    guard_plot['label'] = [
        status_label_map.get((str(r.rescue_status), str(r.marker_guard_status)),
                             f"{r.rescue_status}\n{r.marker_guard_status}")
        for r in guard_plot.itertuples(index=False)
    ]
    guard_plot = guard_plot.sort_values('fraction_of_attempted', ascending=False)
    sns.barplot(data=guard_plot, x='label', y='fraction_of_attempted', color='#4C9AC7',
                edgecolor='black', ax=axes_qc[1, 1])
    axes_qc[1, 1].set_xlabel("")
    axes_qc[1, 1].set_ylabel("Fraction of attempted cells", fontsize=12, fontweight='bold')
    axes_qc[1, 1].set_ylim(0, max(0.05, guard_plot['fraction_of_attempted'].max() * 1.18))
    for p, (_, row) in zip(axes_qc[1, 1].patches, guard_plot.iterrows()):
        h = p.get_height()
        axes_qc[1, 1].annotate(f"{row['count']:,}\n({h*100:.1f}%)",
                               (p.get_x() + p.get_width() / 2, h),
                               ha='center', va='bottom', xytext=(0, 4),
                               textcoords='offset points', fontsize=9, fontweight='bold')
    plt.setp(axes_qc[1, 1].get_xticklabels(), rotation=20, ha='right', fontweight='bold')

    for ax in axes_qc.ravel():
        ax.tick_params(axis='both', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    fig_qc.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.12, wspace=0.30, hspace=0.42)
    rescue_qc_fig_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Rescue_QC.png')
    rescue_qc_pptx_path = os.path.join(FIG_DIR, 'Final_mv1_Step02-04_Rescue_QC.pptx')
    figure2c_png_path = os.path.join(FIG_DIR, 'Final_Figure2C_KNN_Rescue_Guardrail.png')
    figure2c_pptx_path = os.path.join(FIG_DIR, 'Final_Figure2C_KNN_Rescue_Guardrail.pptx')
    if SAVE_STEP02_RESCUE_QC_COMBINED:
        _step02_save_publication_figure(
            fig_qc, rescue_qc_fig_path, rescue_qc_pptx_path,
            label='Rescue QC figure',
            show=False,
            close=False,
            fixed_canvas=False
        )
        _step02_save_publication_figure(
            fig_qc, figure2c_png_path, figure2c_pptx_path,
            label='Figure 2C combined QC',
            fixed_canvas=False
        )
    else:
        plt.close(fig_qc)

    # ── Save individual Figure 2C panels for easier panel sizing in PPT/manuscript layouts.
    def _style_single_panel_axis(ax):
        ax.tick_params(axis='both', labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    def _save_single_panel(fig_obj, file_stem, fixed_canvas=False, target_tight_width_cm=None):
        png_path = os.path.join(FIG_DIR, f'{file_stem}.png')
        pptx_path = os.path.join(FIG_DIR, f'{file_stem}.pptx')
        _step02_save_publication_figure(
            fig_obj, png_path, pptx_path,
            label=f'{file_stem} individual panel',
            fixed_canvas=fixed_canvas,
            target_tight_width_cm=target_tight_width_cm
        )

    fig_2c_a, ax_2c_a = plt.subplots(figsize=STEP02_FIGURE2C_A_SIZE)
    if not conf_plot_df.empty:
        sns.histplot(data=conf_plot_df, x='original_confidence', hue='rescue_decision',
                     bins=30, stat='density', common_norm=False, element='step', ax=ax_2c_a)
        from matplotlib.ticker import MaxNLocator
        ax_2c_a.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
        ax_2c_a.axvline(0.6, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_2c_a.set_xlabel("ML Broad Confidence before rescue", fontsize=18, fontweight='bold')
        ax_2c_a.set_ylabel("Density", fontsize=18, fontweight='bold')
        legend_2c_a = ax_2c_a.get_legend()
        if legend_2c_a is not None:
            legend_2c_a.set_title("Rescue decision", prop={'size': 18, 'weight': 'bold'})
            for text in legend_2c_a.get_texts():
                text.set_fontsize(17)
    else:
        ax_2c_a.text(0.5, 0.5, "No confidence values", ha='center', va='center', fontweight='bold')
        ax_2c_a.set_axis_off()
    _style_single_panel_axis(ax_2c_a)
    # Inset the axes boundaries so tick labels also fit within the creation canvas.
    # Do not correct by bbox cropping at the save stage.
    fig_2c_a.subplots_adjust(left=0.11, right=0.955, top=0.925, bottom=0.18)
    _save_single_panel(fig_2c_a, 'Final_Figure2C_A_Candidate_Confidence', fixed_canvas=True)

    # Reserve an internal left margin to prevent overlap between the y-axis label and tick labels,
    # and apply the edited version's ratio directly to the creation canvas itself.
    fig_2c_b, ax_2c_b = plt.subplots(figsize=STEP02_FIGURE2C_B_SIZE)
    if trans_cm is not None:
        trans_cm.columns = [c.replace(' cells', '') for c in trans_cm.columns]
        trans_cm.index = [i.replace(' cells', '') for i in trans_cm.index]
        _n_labels_2c_b = len(trans_cm.columns)
        _annot_sz_2c_b = max(17, 23 - _n_labels_2c_b)
        _xfont_2c_b    = max(14, 17 - _n_labels_2c_b)
        sns.heatmap(trans_cm, annot=True, fmt='d', cmap='Purples', ax=ax_2c_b, cbar=False,
                    linewidths=.5, annot_kws={"size": _annot_sz_2c_b, "weight": "bold"})
        
        # Keep category labels horizontal because vertical Y ticks overlap the y-axis title.
        plt.setp(ax_2c_b.get_xticklabels(), rotation=0, ha='center', fontsize=_xfont_2c_b, fontweight='bold')
        plt.setp(ax_2c_b.get_yticklabels(), rotation=0, ha='right', va='center', fontsize=_xfont_2c_b, fontweight='bold')

        # Display title (Prediction transition (n=actual value))
        _total_cells = int(trans_cm.values.sum())
        ax_2c_b.set_title(f"Prediction transition (n={_total_cells:,})", fontsize=18, fontweight='bold', pad=8)

        ax_2c_b.set_xlabel("After rescue prediction", fontsize=16.5, fontweight='bold', labelpad=8)
        ax_2c_b.set_ylabel("Before rescue prediction", fontsize=16.5, fontweight='bold', labelpad=14)
    else:
        ax_2c_b.text(0.5, 0.5, "No rescued cells", ha='center', va='center', fontweight='bold')
        ax_2c_b.set_axis_off()
    _style_single_panel_axis(ax_2c_b)

    fig_2c_b.subplots_adjust(left=0.185, right=0.990, top=0.900, bottom=0.125)
    _save_single_panel(fig_2c_b, 'Final_Figure2C_B_Prediction_Transition', fixed_canvas=True)

    fig_2c_c, ax_2c_c = plt.subplots(figsize=(8.2, 5.6))
    if delta_gt_cm is not None:
        sns.heatmap(delta_gt_cm, annot=True, fmt='+d', cmap='RdBu_r', center=0,
                    vmin=-max_delta, vmax=max_delta, ax=ax_2c_c, cbar=True,
                    linewidths=.5, annot_kws={"size": 11, "weight": "bold"})
        ax_2c_c.set_xlabel("Predicted broad lineage", fontsize=12, fontweight='bold')
        ax_2c_c.set_ylabel("Ground truth broad lineage", fontsize=12, fontweight='bold')
    else:
        ax_2c_c.text(0.5, 0.5, "GT labels unavailable", ha='center', va='center', fontweight='bold')
        ax_2c_c.set_axis_off()
    _style_single_panel_axis(ax_2c_c)
    fig_2c_c.tight_layout()
    if SAVE_FIGURE2C_EXTRA_SINGLE_PANELS:
        _save_single_panel(fig_2c_c, 'Final_Figure2C_C_GT_Stratified_Delta')
    else:
        plt.close(fig_2c_c)

    # Enlarge categories for readability while preserving the compact panel ratio.
    _guard_fig_w = 8.5 * STEP02_CM_TO_IN
    _guard_fig_h = 6.5 * STEP02_CM_TO_IN
    fig_2c_d, ax_2c_d = plt.subplots(figsize=(_guard_fig_w, _guard_fig_h))
    guard_x = np.arange(len(guard_plot)) * 0.46
    guard_bars = ax_2c_d.bar(
        guard_x,
        guard_plot['fraction_of_attempted'].to_numpy(dtype=float),
        width=0.27,
        color='#4C9AC7',
        edgecolor='black'
    )
    ax_2c_d.set_xlabel("")
    ax_2c_d.set_title("KNN rescue decision", fontsize=10.0, fontweight='bold', pad=5)
    ax_2c_d.set_ylabel("Fraction of attempted cells", fontsize=8.2, fontweight='bold', labelpad=3)
    ax_2c_d.set_ylim(0, max(0.05, guard_plot['fraction_of_attempted'].max() * 1.45))
    ax_2c_d.set_xlim(guard_x[0] - 0.22, guard_x[-1] + 0.22)
    ax_2c_d.set_xticks(guard_x)
    import textwrap as _step02_textwrap
    _guard_labels = [
        '\n'.join(_step02_textwrap.wrap(str(label), width=10))
        for label in guard_plot['label'].tolist()
    ]
    ax_2c_d.set_xticklabels(_guard_labels)
    for p, (_, row) in zip(guard_bars, guard_plot.iterrows()):
        h = p.get_height()
        ax_2c_d.annotate(f"{row['count']:,}\n({h*100:.1f}%)",
                         (p.get_x() + p.get_width() / 2, h),
                         ha='center', va='bottom', xytext=(0, 3),
                         textcoords='offset points', fontsize=8.0, fontweight='bold')
    _style_single_panel_axis(ax_2c_d)
    plt.setp(ax_2c_d.get_xticklabels(), rotation=0, ha='center', fontweight='bold', fontsize=7.4)
    plt.setp(ax_2c_d.get_yticklabels(), fontweight='bold', fontsize=7.6)
    ax_2c_d.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.28)
    ax_2c_d.spines['top'].set_visible(False)
    ax_2c_d.spines['right'].set_visible(False)
    fig_2c_d.subplots_adjust(left=0.17, right=0.985, top=0.88, bottom=0.22)
    if SAVE_FIGURE2C_D_WITH_EVAL:
        _save_single_panel(fig_2c_d, 'Final_Figure2C_D_Rescue_Decision', fixed_canvas=True)
    else:
        plt.close(fig_2c_d)

    if SAVE_STEP02_RESCUE_QC_COMBINED:
        print(f"  ✅ Rescue QC figure saved: {rescue_qc_fig_path}")
        print(f"  ✅ Rescue QC figure PPTX saved: {rescue_qc_pptx_path}")
        print(f"  ✅ Figure 2C saved: {figure2c_png_path}")
        print(f"  ✅ Figure 2C PPTX saved: {figure2c_pptx_path}")
    else:
        print("  🔹 Figure 2C combined QC output is disabled. Saving individual panels A/B only.")
    print(f"  ✅ Rescue QC table saved: {RESCUE_QC_STEP02_PATH}")
    print(f"  ✅ Guardrail summary saved: {RESCUE_GUARDRAIL_STEP02_PATH}")
else:
    print("  ⚠️ No Rescue QC cell log; skipping confidence/guardrail QC.")

# %%
# ==============================================================================
# [Step 02-5] Save Step 02 (checkpoint + models)
# ==============================================================================
if not skip_rescue:
    _best_score_for_manifest = np.nan
    if 'best_val_score' in locals():
        _best_score_for_manifest = float(best_val_score)
    elif 'history_broad' in locals() and 'best' in history_broad and history_broad['best'].get('avg'):
        _best_score_for_manifest = float(max(history_broad['best']['avg']))

    selection_manifest = {
        'step': 'Step02',
        'selection_mode': 'manual' if MANUAL_BROAD_COMBO.strip() != '' else 'best_val_combo',
        'manual_combo': MANUAL_BROAD_COMBO,
        'applied_combo': target_combo if 'target_combo' in locals() else 'na',
        'manual_iter': int(MANUAL_BROAD_ITER) if 'MANUAL_BROAD_ITER' in locals() else -1,
        'best_val_combo': best_val_combo if 'best_val_combo' in locals() else 'na',
        'best_val_score_avg': _best_score_for_manifest,
        'model_keys': MODEL_KEYS,
        'input_checkpoint': CHECKPOINT_STEP01,
        'probability_log': PROBS_STEP02_PATH,
        'training_log_path': TRAIN_LOG_STEP02_PATH,
        'run_step02_cv': bool(RUN_STEP02_CV),
        'oob_eval_every': int(OOB_EVAL_EVERY),
        'oob_eval_extra_iters': sorted([int(x) for x in OOB_EVAL_EXTRA_ITERS]),
        'rescue_confidence_threshold': 0.6,
        'spillover_radius_px': SPILLOVER_RADIUS_PX,
        'frozen_transfer_manifest': os.path.join(
            V9_TRANSFER_ROOT, 'broad', 'manifest.json'
        )
    }
    adata.uns['step02_model_selection'] = selection_manifest
    adata.write_h5ad(CHECKPOINT_STEP02)
    print(f"✅ Step 02 h5ad saved: {CHECKPOINT_STEP02}")

    # Save model-related objects (Figures can be generated without retraining)
    model_save_data = {
        'le_broad': le_broad,
        'fi_log_broad': fi_log_broad if 'fi_log_broad' in locals() else {},
        'class_fi_log_broad': class_fi_log_broad if 'class_fi_log_broad' in locals() else {},
        'class_fi_table': (
            pd.DataFrame(class_fi_rows) if 'class_fi_rows' in locals()
            else locals().get('class_fi_table', pd.DataFrame())
        ),
        'class_fi_summary': (
            summarize_class_feature_importance(class_fi_rows) if 'class_fi_rows' in locals()
            else locals().get('class_fi_summary', pd.DataFrame())
        ),
        'per_class_metric_log': (
            pd.DataFrame(per_class_metric_rows) if 'per_class_metric_rows' in locals()
            else locals().get('per_class_metric_log', pd.DataFrame())
        ),
        'preds_mapping': preds_mapping if 'preds_mapping' in locals() else [],
        'combo_names': combo_names if 'combo_names' in locals() else [],
        'broad_combos': broad_combos if 'broad_combos' in locals() else [],
        'actual_iters': locals().get('actual_iters', N_ITER),
        'feature_names': feature_names_list if 'feature_names_list' in locals() else [],
        'target_combo': target_combo if 'target_combo' in locals() else None,
        'selection_manifest': selection_manifest,
        'training_log_path': TRAIN_LOG_STEP02_PATH,
        'frozen_transfer_manifest': os.path.join(
            V9_TRANSFER_ROOT, 'broad', 'manifest.json'
        ),
    }
    joblib.dump(model_save_data, MODELS_STEP02_PATH)
    print(f"✅ Step 02 model data saved: {MODELS_STEP02_PATH}")
