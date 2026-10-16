# Canonical script; synchronized to the paired notebook via scripts/sync_notebooks.py.
# %%
# ==============================================================================
# [Final Step 08] Receptor-Ligand and Activation/Exhaustion Correlation Analysis
# ▷ Purpose: analyze only two axes using distances/immune scores calculated in Step 06.
#    1) CODEX marker-constrained spatial receptor-ligand proximity enrichment
#    2) T-cell activation/exhaustion score correlation by T-cell subtype
# ▷ Input: checkpoints/mv1/mv1_step06.h5ad
# ▷ Outputs: figures/v2/Final_mv1_Step08-01–03 (active canonical generation)
# ==============================================================================
import hashlib
import json
import os
import re
import time
from pathlib import Path

from matplotlib import colors as mcolors, font_manager
from matplotlib.textpath import TextPath
from matplotlib.transforms import Bbox


def _step08_env_flag(name, default=False):
    raw = os.environ.get(name, '1' if default else '0').strip()
    if raw not in {'0', '1'}:
        raise ValueError(f'{name} must be 0 or 1, got {raw!r}')
    return raw == '1'


STEP08_MODE = os.environ.get('STEP08_MODE', 'figure_only').strip()
if STEP08_MODE not in {'figure_only', 'full_run'}:
    raise ValueError(f"STEP08_MODE must be 'figure_only' or 'full_run': {STEP08_MODE!r}")
STEP08_FIGURE_ONLY = STEP08_MODE == 'figure_only'
STEP08_RUN_ID = os.environ.get('STEP08_RUN_ID', '').strip()
STEP08_PREVIEW_ID = os.environ.get('STEP08_PREVIEW_ID', '').strip()
STEP08_ACTIVE_RUN_ID = 'step08_full8_tpr_noallt_seed42_perm1000_20260821_01'
STEP08_COHORT = os.environ.get('STEP08_COHORT', 'dlbcl_8_tissues').strip()
STEP08_INCLUDE_TPR = _step08_env_flag('STEP08_INCLUDE_TPR', True)
STEP08_DROP_ALL_T = _step08_env_flag('STEP08_DROP_ALL_T', True)
STEP08_REQUIRE_GPU = _step08_env_flag('STEP08_REQUIRE_GPU', True)
STEP08_RESUME_PARTIAL_RUN = _step08_env_flag('STEP08_RESUME_PARTIAL_RUN', False)
STEP08_RUN_STARTED = time.time()
STEP08_EXACT_DLBCL_TISSUES = (
    'DLBCL1_01', 'DLBCL1_02',
    'DLBCL2_01', 'DLBCL2_02',
    'DLBCL3_01', 'DLBCL3_02',
    'DLBCL4_01', 'DLBCL4_02',
)
if STEP08_MODE == 'full_run':
    if STEP08_COHORT != 'dlbcl_8_tissues':
        raise ValueError("STEP08 full_run requires STEP08_COHORT='dlbcl_8_tissues'")
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', STEP08_RUN_ID):
        raise ValueError(f'STEP08_RUN_ID is missing or unsafe: {STEP08_RUN_ID!r}')
elif (
    STEP08_RUN_ID or STEP08_COHORT != 'dlbcl_8_tissues' or not STEP08_INCLUDE_TPR
    or not STEP08_DROP_ALL_T or STEP08_RESUME_PARTIAL_RUN
):
    raise ValueError(
        'Step08 figure_only is fixed to the active 8-tissue, Tpr-included, '
        'All-T-dropped canonical generation'
    )
if STEP08_PREVIEW_ID:
    if not STEP08_FIGURE_ONLY:
        raise ValueError('STEP08_PREVIEW_ID is available only in figure_only mode')
    if not re.fullmatch(r'_tmp_[A-Za-z0-9][A-Za-z0-9_-]{0,119}', STEP08_PREVIEW_ID):
        raise ValueError(
            'STEP08_PREVIEW_ID must be a safe _tmp_-prefixed directory name: '
            f'{STEP08_PREVIEW_ID!r}'
        )
os.environ['FINAL_FIGURE_ONLY_WRITE_GUARD'] = '1' if STEP08_FIGURE_ONLY else '0'
os.environ.setdefault('FINAL_LIGHT_IMPORTS', '1')  # Load shared utilities lightly and selectively load only GPU modules required by Step08 below.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')  # Reduce h5ad read-lock conflicts on WSL/NTFS.
from Final_utils import *
from Final_utils import _final_figure_stem_registered
from editable_pptx import _validate_native_shape_slide
from collections import Counter

if STEP08_PREVIEW_ID:
    STEP08_PREVIEW_FIG_DIR = os.path.realpath(os.path.join(FIG_DIR, STEP08_PREVIEW_ID))
    if os.path.dirname(STEP08_PREVIEW_FIG_DIR) != os.path.realpath(FIG_DIR):
        raise PermissionError(f'Step08 preview directory escaped FIG_DIR: {STEP08_PREVIEW_FIG_DIR}')
    if os.path.exists(STEP08_PREVIEW_FIG_DIR):
        raise FileExistsError(
            f'Refusing to overwrite an existing Step08 preview directory: {STEP08_PREVIEW_FIG_DIR}'
        )
else:
    STEP08_PREVIEW_FIG_DIR = None

if STEP08_MODE == 'full_run':
    STEP08_RUN_TABLE_DIR = os.path.join(FINAL_TABLE_DIR, 'Step08_runs', STEP08_RUN_ID)
    STEP08_RUN_FIG_DIR = os.path.join(FIG_DIR, 'Step08_runs', STEP08_RUN_ID)
    if STEP08_RESUME_PARTIAL_RUN:
        if not os.path.isdir(STEP08_RUN_TABLE_DIR) or not os.path.isdir(STEP08_RUN_FIG_DIR):
            raise FileNotFoundError(
                'STEP08_RESUME_PARTIAL_RUN requires both exact run directories: '
                f'{STEP08_RUN_TABLE_DIR}, {STEP08_RUN_FIG_DIR}'
            )
    elif os.path.exists(STEP08_RUN_TABLE_DIR) or os.path.exists(STEP08_RUN_FIG_DIR):
        raise FileExistsError(
            'Refusing to overwrite an existing Step08 run: '
            f'{STEP08_RUN_TABLE_DIR}, {STEP08_RUN_FIG_DIR}'
        )
else:
    STEP08_RUN_TABLE_DIR = None
    STEP08_RUN_FIG_DIR = None

# Prefer the GPU for Step08-1 R-L permutations, the most time-consuming step.
# Even when Final_utils sets cp/cuNN to None in light-import mode, reload only
# the required GPU modules here and automatically fall back to the CPU on failure.
STEP08_USE_GPU = False if STEP08_FIGURE_ONLY else bool(globals().get('STEP08_USE_GPU', True))
STEP08_GPU_NEAREST_NEIGHBOR = bool(globals().get('STEP08_GPU_NEAREST_NEIGHBOR', True))
STEP08_GPU_LABEL_SHUFFLE = bool(globals().get('STEP08_GPU_LABEL_SHUFFLE', True))
STEP08_N_PERM = int(os.environ.get('STEP08_N_PERM', globals().get('STEP08_N_PERM', 1000)))
STEP08_MIN_NEIGHBOR_PAIRS = int(os.environ.get(
    'STEP08_MIN_NEIGHBOR_PAIRS', globals().get('STEP08_MIN_NEIGHBOR_PAIRS', 5)
))
STEP08_GPU_AVAILABLE = False
STEP08_GPU_DISABLED_REASON = ''

if STEP08_USE_GPU:
    try:
        if globals().get('cp', None) is None:
            import cupy as cp
        if globals().get('cuNN', None) is None:
            from cuml.neighbors import NearestNeighbors as cuNN
        _ = cp.asarray([0], dtype=cp.float32).sum()
        STEP08_GPU_AVAILABLE = True
        print("🚀 [Step 08] GPU acceleration available: CuPy/cuML verified")
    except Exception as e:
        STEP08_GPU_AVAILABLE = False
        STEP08_GPU_DISABLED_REASON = f"{type(e).__name__}: {e}"
        print(f"💻 [Step 08] GPU acceleration disabled → use CPU fallback ({STEP08_GPU_DISABLED_REASON})")
else:
    STEP08_GPU_DISABLED_REASON = 'STEP08_USE_GPU=False'
    print("💻 [Step 08] STEP08_USE_GPU=False → run in CPU mode")

if STEP08_MODE == 'full_run' and STEP08_REQUIRE_GPU and not STEP08_GPU_AVAILABLE:
    raise RuntimeError(
        'Step08 full_run requires a working GPU; CPU fallback is disabled for the '
        f'1,000-permutation R-L calculation ({STEP08_GPU_DISABLED_REASON})'
    )
if STEP08_MODE == 'full_run' and not STEP08_RESUME_PARTIAL_RUN:
    os.makedirs(STEP08_RUN_TABLE_DIR, exist_ok=False)
    os.makedirs(STEP08_RUN_FIG_DIR, exist_ok=False)

# ==============================================================================
# [Step 08-0] Load checkpoints and configure shared settings
# ▷ Import shared libraries and global paths from Final_utils.
# ==============================================================================

CHECKPOINT_STEP06 = os.path.join(FINAL_CKPT_WRITE_DIR, 'mv1_step06.h5ad')
CHECKPOINT_STEP08 = os.path.join(FINAL_CKPT_WRITE_DIR, 'mv1_step08.h5ad')
STEP08_USE_SAVED_CHECKPOINT = False  # Redraw figures using Step06 calculations after updating distance conversion.
STEP08_WRITE_CHECKPOINT = False      # Do not create a new h5ad checkpoint in Figure-only mode.

if STEP08_USE_SAVED_CHECKPOINT and os.path.exists(CHECKPOINT_STEP08):
    adata = sc.read_h5ad(CHECKPOINT_STEP08)
    print(f"🔹 [Step 08] Load saved completed V9 Step 08 results: {CHECKPOINT_STEP08}")
elif os.path.exists(CHECKPOINT_STEP06):
    adata = sc.read_h5ad(CHECKPOINT_STEP06)
    print(f"✅ [Step 08] Load the Final Step06 checkpoint: {CHECKPOINT_STEP06}")
else:
    raise FileNotFoundError(
        "[Step 08] Figure rendering requires the Final Step06 checkpoint; "
        "V8/Legacy fallback is disabled.\n"
        f"  Required file: {CHECKPOINT_STEP06}\n"
        "  protected distance calculations are not run by Final_step08."
    )

_required_cols = ['Dist_to_Pos', 'Dist_to_Neg', 'Activation_Score', 'Exhaustion_Score']
_missing_cols = [c for c in _required_cols if c not in adata.obs.columns]
if _missing_cols:
    raise KeyError(f"[Step 08] Required columns calculated in Step 06 are missing: {_missing_cols}. Run Step 06-2 in Final_step06.py first.")

STEP08_COORD_UM_PER_UNIT = float(adata.uns.get('STEP06_COORD_UM_PER_UNIT', 0.37744))
STEP08_PUBLIC_TIFF_DOWNSAMPLE_FACTOR = float(adata.uns.get('STEP06_PUBLIC_TIFF_DOWNSAMPLE_FACTOR', 2.0))
STEP08_PUBLIC_TIFF_UM_PER_PIXEL = STEP08_COORD_UM_PER_UNIT * STEP08_PUBLIC_TIFF_DOWNSAMPLE_FACTOR
STEP08_COORD_UNIT_SOURCE = str(adata.uns.get(
    'STEP06_COORD_UNIT_SOURCE',
    'PhenoImager/CODEX annotation centroid coordinates; Step06 distance conversion fallback.'
))
if not np.isfinite(STEP08_COORD_UM_PER_UNIT) or STEP08_COORD_UM_PER_UNIT <= 0:
    raise ValueError(f"[Step 08] Invalid coordinate-to-um conversion value: {STEP08_COORD_UM_PER_UNIT}")
adata.uns['STEP08_DISTANCE_SOURCE'] = (
    'Step08 uses Step06-derived Dist_to_Pos/Dist_to_Neg in um; direct KDTree coordinate radii are converted '
    f'with 1 annotation coordinate unit = {STEP08_COORD_UM_PER_UNIT:.5f} um.'
)
print(
    f"📏 [Step 08] Distance unit check: Dist_to_Pos/Dist_to_Neg use values saved in um by Step06; "
    f"coordinate-based KDTree radii are converted using 1 coordinate unit = {STEP08_COORD_UM_PER_UNIT:.5f} um."
)

STEP08_T_CELL_LABEL_ORDER = ['CD4_CTL', 'CD8T', 'Tfh', 'Th', 'Tpr', 'Treg']
STEP08_B_CELL_LABEL_ORDER = ['B cell']
STEP08_HLA_DR_MARKER_CANDIDATES = ['HLA-DR', 'HLA-DRA', 'HLA-DRB1']
STEP08_CELL_CATEGORY_DISPLAY = {
    'B cell': 'B cells',
}


def _step08_display_cell_category(label):
    """Preserve computational cell labels and pluralize only Figure category labels."""
    label_text = str(label)
    return STEP08_CELL_CATEGORY_DISPLAY.get(label_text, label_text)


def _step08_display_cell_pair(pair_label):
    """Convert a `Sender → Receiver` computation key to a manuscript display label."""
    pair_text = str(pair_label)
    for separator in (' → ', ' -> '):
        if separator in pair_text:
            sender, receiver = pair_text.split(separator, 1)
            return (
                f'{_step08_display_cell_category(sender)} → '
                f'{_step08_display_cell_category(receiver)}'
            )
    return _step08_display_cell_category(pair_text)


STEP08_RECOMPUTE_TB_DISTANCE = _step08_env_flag('STEP08_RECOMPUTE_TB_DISTANCE', False)
if STEP08_MODE == 'full_run' and STEP08_RECOMPUTE_TB_DISTANCE:
    raise ValueError(
        'Step08 full_run must reuse the protected Step06 distance columns; '
        'STEP08_RECOMPUTE_TB_DISTANCE must remain 0'
    )


def _step08_ordered_existing_labels(observed_labels, preferred_order):
    """Return only observed labels included in the preferred order."""
    observed_set = {str(label).strip() for label in observed_labels if pd.notna(label)}
    return [label for label in preferred_order if label in observed_set]


def _step08_get_b_cell_labels(observed_labels):
    """Determine reference B-cell labels for HLA-DR stratification."""
    observed = [str(label).strip() for label in observed_labels if pd.notna(label)]
    ordered = _step08_ordered_existing_labels(observed, STEP08_B_CELL_LABEL_ORDER)
    extra_b = sorted([
        label for label in observed
        if label not in ordered and (label.startswith('B ') or label.startswith('B-'))
    ])
    return ordered + extra_b


def _step08_get_t_cell_labels(observed_labels):
    """Determine T-cell subset labels for distance recalculation."""
    labels = _step08_ordered_existing_labels(observed_labels, STEP08_T_CELL_LABEL_ORDER)
    if not labels:
        raise ValueError(f"[Step 08] No T-cell subset labels are available for distance recalculation: {STEP08_T_CELL_LABEL_ORDER}")
    return labels


def _step08_array_to_numpy(arr):
    """Convert CuPy/cuML results or NumPy arrays to arrays for downstream pandas/NumPy processing."""
    if globals().get('cp', None) is not None and hasattr(arr, 'get'):
        return arr.get()
    if globals().get('cp', None) is not None and hasattr(cp, 'asnumpy'):
        try:
            return cp.asnumpy(arr)
        except Exception:
            pass
    return np.asarray(arr)


def _step08_query_nearest_distance(reference_coords, query_coords, context_label='nearest query'):
    """
    Calculate nearest distances from reference coordinates to query coordinates.
    Prefer cuML NearestNeighbors when a GPU is available; automatically fall back to cKDTree on failure.
    """
    ref = np.asarray(reference_coords, dtype=np.float32)
    qry = np.asarray(query_coords, dtype=np.float32)
    if ref.shape[0] == 0 or qry.shape[0] == 0:
        return np.full(qry.shape[0], np.nan), 'empty'

    if STEP08_GPU_AVAILABLE and STEP08_GPU_NEAREST_NEIGHBOR and globals().get('cuNN', None) is not None:
        try:
            nn_model = cuNN(n_neighbors=1)
            nn_model.fit(ref)
            dists, _ = nn_model.kneighbors(qry)
            dists_np = _step08_array_to_numpy(dists).reshape(-1)
            return dists_np.astype(float, copy=False), 'gpu'
        except Exception as e:
            print(f"  ⚠️ [Step 08] {context_label} GPU nearest failed → cKDTree CPU fallback: {type(e).__name__}: {e}")

    dists_np, _ = cKDTree(ref).query(qry)
    return np.asarray(dists_np, dtype=float), 'cpu'


def _step08_recompute_tb_distance_columns():
    """
    Recalculate T-to-B distance columns in case the Step06 checkpoint uses older distance units.
    Calculate KDTree distances in raw annotation coordinates, then multiply by 0.37744 um/coordinate unit.
    """
    obs_labels = adata.obs['Niche_Label'].astype(str).str.strip()
    disease_mask = adata.obs['Disease'].astype(str).str.contains('DLBCL', na=False)
    b_cell_labels = _step08_get_b_cell_labels(obs_labels.unique())
    t_cell_labels = _step08_get_t_cell_labels(obs_labels.unique())
    b_mask = disease_mask & obs_labels.isin(b_cell_labels)
    t_mask_recalc = disease_mask & obs_labels.isin(t_cell_labels)

    if int(b_mask.sum()) == 0:
        raise ValueError(f"[Step 08] HLA-DR reference B cells were not found. Candidate labels: {b_cell_labels}")
    if int(t_mask_recalc.sum()) == 0:
        raise ValueError(f"[Step 08] T cells for distance recalculation were not found. Candidate labels: {t_cell_labels}")

    hla_marker = next((m for m in STEP08_HLA_DR_MARKER_CANDIDATES if m in adata.var_names), None)
    if hla_marker is None:
        raise KeyError(f"[Step 08] Markers required for HLA-DR B-cell polarity analysis are missing: {STEP08_HLA_DR_MARKER_CANDIDATES}")

    b_idx = adata.obs.index[b_mask]
    if sp.issparse(adata[b_idx, [hla_marker]].X):
        b_expr_vals = adata[b_idx, [hla_marker]].X.toarray().ravel()
    else:
        b_expr_vals = np.asarray(adata[b_idx, [hla_marker]].X).ravel()
    b_expr = pd.Series(b_expr_vals, index=b_idx)

    try:
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(b_expr_vals.reshape(-1, 1))
        pred_labels = gmm.predict(b_expr_vals.reshape(-1, 1))
        means = gmm.means_.ravel()
        pos_comp_idx = int(np.argmax(means))
        hla_pos_idx = b_idx[pred_labels == pos_comp_idx]
        hla_neg_idx = b_idx[pred_labels != pos_comp_idx]
        if b_expr.loc[hla_pos_idx].mean() < b_expr.loc[hla_neg_idx].mean():
            hla_pos_idx, hla_neg_idx = hla_neg_idx, hla_pos_idx
        print(f"  ▶ [Step 08] HLA-DR GMM recalculation complete: marker={hla_marker}, cutoff~{b_expr.loc[hla_pos_idx].min():.3f}")
    except Exception as e:
        hla_pos_idx = b_idx[b_expr > 0.0]
        hla_neg_idx = b_idx[b_expr <= 0.0]
        print(f"  ⚠️ [Step 08] HLA-DR GMM failed → use a greater-than-zero criterion: {e}")

    recalc_rows = []
    nearest_backend_counts = Counter()
    tissue_values = adata.obs.loc[t_mask_recalc, 'Run_Tissue_name'].astype(str).unique()
    for tissue in tqdm(tissue_values, desc='Step08 T-to-B distance recalculation'):
        tissue_mask = adata.obs['Run_Tissue_name'].astype(str).eq(str(tissue))
        coord_valid = adata.obs[['X_centroid', 'Y_centroid']].notna().all(axis=1)
        t_idx = adata.obs.index[t_mask_recalc & tissue_mask & coord_valid]
        if len(t_idx) == 0:
            continue
        t_coords = adata.obs.loc[t_idx, ['X_centroid', 'Y_centroid']].to_numpy(dtype=float)
        b_pos_idx_t = hla_pos_idx.intersection(adata.obs.index[tissue_mask & coord_valid])
        b_neg_idx_t = hla_neg_idx.intersection(adata.obs.index[tissue_mask & coord_valid])

        if len(b_pos_idx_t) > 0:
            pos_coords = adata.obs.loc[b_pos_idx_t, ['X_centroid', 'Y_centroid']].to_numpy(dtype=float)
            d_pos_coord, backend_pos = _step08_query_nearest_distance(
                pos_coords,
                t_coords,
                f'{tissue} HLA-DR+ nearest B'
            )
            nearest_backend_counts[backend_pos] += 1
            d_pos = d_pos_coord * STEP08_COORD_UM_PER_UNIT
        else:
            d_pos_coord = np.full(len(t_idx), np.nan)
            d_pos = np.full(len(t_idx), np.nan)
            nearest_backend_counts['empty_pos'] += 1

        if len(b_neg_idx_t) > 0:
            neg_coords = adata.obs.loc[b_neg_idx_t, ['X_centroid', 'Y_centroid']].to_numpy(dtype=float)
            d_neg_coord, backend_neg = _step08_query_nearest_distance(
                neg_coords,
                t_coords,
                f'{tissue} HLA-DR- nearest B'
            )
            nearest_backend_counts[backend_neg] += 1
            d_neg = d_neg_coord * STEP08_COORD_UM_PER_UNIT
        else:
            d_neg_coord = np.full(len(t_idx), np.nan)
            d_neg = np.full(len(t_idx), np.nan)
            nearest_backend_counts['empty_neg'] += 1

        recalc_rows.append(pd.DataFrame({
            'Dist_to_Pos': d_pos,
            'Dist_to_Neg': d_neg,
            'Dist_to_Pos_coord': d_pos_coord,
            'Dist_to_Neg_coord': d_neg_coord,
        }, index=t_idx))

    if not recalc_rows:
        raise ValueError('[Step 08] No recalculated T-to-B distances are available. Check coordinates and tissue labels.')

    recalc_df = pd.concat(recalc_rows, axis=0)
    for col in ['Dist_to_Pos', 'Dist_to_Neg', 'Dist_to_Pos_coord', 'Dist_to_Neg_coord']:
        adata.obs[col] = np.nan
        adata.obs.loc[recalc_df.index, col] = recalc_df[col]
    adata.uns['STEP06_COORD_UM_PER_UNIT'] = STEP08_COORD_UM_PER_UNIT
    adata.uns['STEP06_PUBLIC_TIFF_DOWNSAMPLE_FACTOR'] = STEP08_PUBLIC_TIFF_DOWNSAMPLE_FACTOR
    adata.uns['STEP06_DISTANCE_UNIT'] = 'um'
    adata.uns['STEP08_TB_DISTANCE_RECOMPUTED'] = True
    print(f"  ✅ [Step 08] T-to-B distance columns recalculated: {len(recalc_df):,} T cells, unit=um")
    print(f"  📌 [Step 08] Nearest-distance computation backend: {dict(nearest_backend_counts)}")


if STEP08_RECOMPUTE_TB_DISTANCE:
    print(
        "🔹 [Step 08] Step06 distance metadata or coord columns are missing; "
        "recalculate T-to-B distance columns using the current conversion."
    )
    _step08_recompute_tb_distance_columns()
else:
    print("🔹 [Step 08] Step06 distance metadata and coord columns verified → use the existing distance columns.")

save_dir = (
    STEP08_RUN_FIG_DIR if STEP08_MODE == 'full_run'
    else STEP08_PREVIEW_FIG_DIR if STEP08_PREVIEW_FIG_DIR is not None
    else FIG_DIR
)
os.makedirs(save_dir, exist_ok=True)

STEP08_FDR_ALPHA = 0.05
STEP08_LEGACY_TABLE_DIR = os.path.join(FINAL_TABLE_DIR, 'Step08_CSV')
STEP08_ACTIVE_TABLE_DIR = os.path.join(
    FINAL_TABLE_DIR, 'Step08_runs', STEP08_ACTIVE_RUN_ID
)
STEP08_TABLE_DIR = (
    STEP08_RUN_TABLE_DIR if STEP08_MODE == 'full_run' else STEP08_ACTIVE_TABLE_DIR
)
if STEP08_FIGURE_ONLY and not os.path.isdir(STEP08_TABLE_DIR):
    raise FileNotFoundError(
        f'Step08 active 8-tissue generation is missing: {STEP08_TABLE_DIR}'
    )
STEP08_RL_SUMMARY_CSV = 'MV1_Step081_RL_Interaction_Summary.csv'
STEP08_RL_LONGFORM_CSV = 'MV1_Step081_RL_Interaction_LongForm.csv'
STEP08_RL_HEATMAP_FIG = 'MV1_Step081_RL_Interaction_Heatmap.png'
STEP08_RL_TOP_FIG = 'MV1_Step081b_RL_Grouped_By_CellCellPair_AllSignificant.png'
STEP08_RL_HEATMAP_CANVAS = (6765925 / 914400, 1847850 / 914400)
STEP08_RL_DOTPLOT_CANVAS = (7.40, 3.30)
STEP08_MAIN_FIGSIZES = {
    # Author-approved Slide 10 ratio, encoded directly in the Figure canvas.
    STEP08_RL_HEATMAP_FIG: STEP08_RL_HEATMAP_CANVAS,
    STEP08_RL_TOP_FIG: STEP08_RL_DOTPLOT_CANVAS,
    # Lay out the Supplement 6 2x3 correlation panels using this independent canvas aspect ratio.
    'MV1_Step082_Activation_Exhaustion_Correlation_ByTcellType.png': (4.50, 5.35),
}
STEP08_RL_AXIS_ORDER = (
    'HLA-DR → LAG3',
    'CXCL13 → CXCR5',
    'PDPN → CD44',
    'CD34 → CD62L',
)
STEP08_RL_AXIS_PALETTE = {
    'HLA-DR → LAG3': '#1f77b4',
    'CXCL13 → CXCR5': '#ff7f0e',
    'PDPN → CD44': '#2ca02c',
    'CD34 → CD62L': '#d62728',
}
STEP08_RL_PANEL_POSITIONS = {
    'HLA-DR → LAG3': (0.017, 0.515, 0.420, 0.405),
    'CXCL13 → CXCR5': (0.439, 0.515, 0.420, 0.405),
    'PDPN → CD44': (0.017, 0.075, 0.420, 0.405),
    'CD34 → CD62L': (0.439, 0.075, 0.420, 0.405),
}
STEP08_RL_LEGEND_POSITIONS = {
    'score': (0.870, 0.635, 0.120, 0.245),
    'fdr': (0.870, 0.375, 0.120, 0.205),
    'settings': (0.870, 0.185, 0.120, 0.135),
}
STEP08_RL_PANEL_CONTENT_WIDTH = 0.416
STEP08_RL_SENDER_LABEL_WIDTHS = {
    'HLA-DR → LAG3': 0.078,
    'CXCL13 → CXCR5': 0.046,
    'PDPN → CD44': 0.078,
    'CD34 → CD62L': 0.046,
}
STEP08_RL_PANEL_CONTENT_GAP = 0.002
STEP08_RL_MATRIX_ROW_HEIGHT = 0.080
STEP08_RL_LEGEND_CENTER_X = 0.930
STEP08_RL_EXPECTED_SIGNIFICANT_CELLS = 38
STEP08_RL_DOTPLOT_PANEL_WIDTH_RATIOS = (1.35, 0.78, 1.10, 1.20)
STEP08_RL_EXPECTED_SIGNIFICANT_COUNTS = {
    'HLA-DR → LAG3': 15,
    'CXCL13 → CXCR5': 4,
    'PDPN → CD44': 12,
    'CD34 → CD62L': 7,
}
STEP08_RL_INTERACTION_RADIUS_UM = 50.0
STEP08_RL_N_PERMUTATIONS = 1000
STEP08_SAVE_PAD_INCHES = 0.015
STEP08_EXPORT_STATIC_FIGURES = {
    STEP08_RL_HEATMAP_FIG,
    STEP08_RL_TOP_FIG,
    'MV1_Step082_Activation_Exhaustion_Correlation_ByTcellType.png',
}
STEP08_FIGURE_ONLY_SAVED_REPORTS = []
STEP08_ACTIVE_ANALYSIS_ORDER = ['Step08-1 R-L proximity enrichment', 'Step08-2 activation/exhaustion correlation']
print('📌 [Step 08] Active configuration: 1) R-L analysis → 2) Activation/Exhaustion score correlation')


def _step08_sha256(path, block_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


def _step08_validate_active_generation():
    """Fail closed unless Figure-only uses the promoted exact 8-tissue generation."""
    manifest_path = os.path.join(STEP08_ACTIVE_TABLE_DIR, 'Step08_Run_Manifest.json')
    hash_path = os.path.join(STEP08_ACTIVE_TABLE_DIR, 'Step08_Output_Hashes.csv')
    promotion_path = os.path.join(
        FINAL_TABLE_DIR,
        'Step08_promotions',
        STEP08_ACTIVE_RUN_ID,
        'canonical_figure_promotion_manifest.json',
    )
    for required_path in (manifest_path, hash_path, promotion_path):
        if not os.path.isfile(required_path) or os.path.getsize(required_path) <= 0:
            raise FileNotFoundError(
                f'Step08 active-generation provenance is missing: {required_path}'
            )

    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    expected_contract = {
        'status': 'complete',
        'run_id': STEP08_ACTIVE_RUN_ID,
        'cohort': 'dlbcl_8_tissues',
        'include_DLBCL1_02': True,
        'include_Tpr': True,
        'drop_All_T_cells': True,
    }
    observed_contract = {key: manifest.get(key) for key in expected_contract}
    if observed_contract != expected_contract:
        raise RuntimeError(
            'Step08 active-generation contract mismatch: '
            f'observed={observed_contract}, expected={expected_contract}'
        )
    if tuple(manifest.get('exact_tissues', ())) != STEP08_EXACT_DLBCL_TISSUES:
        raise RuntimeError(
            'Step08 active-generation tissue order/content mismatch: '
            f"{manifest.get('exact_tissues')}"
        )
    expected_parameters = {
        'spatial_radius_um': 50.0,
        'coordinate_um_per_unit': 0.37744,
        'n_permutations': 1000,
        'minimum_neighbor_pairs': 5,
        'random_seed': 42,
        'recompute_T_to_B_distance': False,
    }
    observed_parameters = {
        key: manifest.get('parameters', {}).get(key) for key in expected_parameters
    }
    if observed_parameters != expected_parameters:
        raise RuntimeError(
            'Step08 active-generation parameter mismatch: '
            f'observed={observed_parameters}, expected={expected_parameters}'
        )
    expected_input_hashes = {
        'checkpoints/mv1/mv1_step06.h5ad': (
            '4fb7c10f2a622712be8b1947c6f06920e859c390a6beb9d9c8a67ee972126cba'
        ),
    }
    observed_input_hashes = {
        key: manifest.get('input_hashes', {}).get(key)
        for key in expected_input_hashes
    }
    if observed_input_hashes != expected_input_hashes:
        raise RuntimeError(
            'Step08 active-generation input-hash contract mismatch: '
            f'observed={observed_input_hashes}, expected={expected_input_hashes}'
        )

    hash_frame = pd.read_csv(hash_path)
    required_hash_columns = {'Relative_Path', 'Size_Bytes', 'SHA256'}
    if not required_hash_columns.issubset(hash_frame.columns):
        raise RuntimeError(
            f'Step08 hash table lacks {sorted(required_hash_columns)}: {hash_path}'
        )
    hash_frame['Basename'] = hash_frame['Relative_Path'].map(
        lambda value: os.path.basename(os.fspath(value))
    )
    required_tables = {
        STEP08_RL_LONGFORM_CSV,
        STEP08_RL_SUMMARY_CSV,
        'MV1_Step082_Activation_Exhaustion_Correlation_ByTcellType.csv',
    }
    for filename in sorted(required_tables):
        rows = hash_frame.loc[hash_frame['Basename'].eq(filename)]
        if len(rows) != 1:
            raise RuntimeError(
                f'Step08 active hash entry must be unique for {filename}: {len(rows)}'
            )
        target = os.path.join(STEP08_ACTIVE_TABLE_DIR, filename)
        row = rows.iloc[0]
        if not os.path.isfile(target):
            raise FileNotFoundError(f'Step08 active table is missing: {target}')
        if int(os.path.getsize(target)) != int(row['Size_Bytes']):
            raise RuntimeError(f'Step08 active table size mismatch: {target}')
        if _step08_sha256(target) != str(row['SHA256']):
            raise RuntimeError(f'Step08 active table hash mismatch: {target}')

    with open(promotion_path, encoding='utf-8') as handle:
        promotion = json.load(handle)
    if (
        promotion.get('status') != 'promoted'
        or promotion.get('source_run_id') != STEP08_ACTIVE_RUN_ID
        or promotion.get('active_figure_steps') != ['Step08-01', 'Step08-02', 'Step08-03']
    ):
        raise RuntimeError(
            f'Step08 canonical promotion manifest is invalid: {promotion_path}'
        )
    promoted_figures = promotion.get('canonical_figures', [])
    if len(promoted_figures) != 6:
        raise RuntimeError(
            f'Step08 promotion must contain exactly six canonical Figure files: '
            f'{len(promoted_figures)}'
        )
    for figure_record in promoted_figures:
        relative_path = os.fspath(figure_record.get('canonical_relative_path', ''))
        target = os.path.realpath(os.path.join(FINAL_WORKSPACE_DIR, relative_path))
        if os.path.commonpath([target, os.path.realpath(FIG_DIR)]) != os.path.realpath(FIG_DIR):
            raise PermissionError(f'Step08 promoted Figure escaped FIG_DIR: {target}')
        if not os.path.isfile(target) or os.path.getsize(target) <= 0:
            raise FileNotFoundError(f'Step08 promoted Figure is missing: {target}')
        # A canonical Figure-only run replaces the complete active Step08 set
        # from the validated tables, so historical Figure hashes are not a
        # precondition for that re-render. A temporary preview does not replace
        # canonical outputs and therefore continues to audit their hashes.
        replacing_canonical_set = STEP08_PREVIEW_FIG_DIR is None
        if replacing_canonical_set:
            continue
        observed_size = int(os.path.getsize(target))
        observed_hash = _step08_sha256(target)
        if (
            observed_size != int(figure_record.get('size_bytes', -1))
            or observed_hash != str(figure_record.get('sha256', ''))
        ):
            print(
                '⚠️ Step08 canonical Figure differs from the historical manifest '
                f'and is retained read-only during this preview: {target}'
            )


if STEP08_FIGURE_ONLY:
    _step08_validate_active_generation()


STEP08_RUN_OUTPUT_PATHS = []
STEP08_PROTECTED_BASELINE = {}
if STEP08_MODE == 'full_run':
    protected_paths = [
        CHECKPOINT_STEP06,
    ]
    for protected_table_dir in (STEP08_LEGACY_TABLE_DIR, STEP08_ACTIVE_TABLE_DIR):
        protected_paths.extend(
            sorted(str(path) for path in Path(protected_table_dir).glob('*'))
        )
    protected_paths.extend(sorted(str(path) for path in Path(FIG_DIR).glob('Final_mv1_Step08-*')))
    STEP08_PROTECTED_BASELINE = {
        os.path.realpath(path): _step08_sha256(path)
        for path in protected_paths
        if os.path.isfile(path)
    }


def _step08_write_csv(frame, path, *, index=False):
    """Write one exact run-local CSV while keeping the global Final guard enabled."""
    target = os.path.realpath(os.path.abspath(os.fspath(path)))
    if STEP08_MODE != 'full_run':
        raise PermissionError(
            f'Step08 figure-only mode cannot write analysis tables: {target}'
        )
    if os.path.dirname(target) != os.path.realpath(STEP08_TABLE_DIR):
        raise PermissionError(f'Step08 run CSV escaped the exact table directory: {target}')
    if os.path.exists(target):
        if STEP08_RESUME_PARTIAL_RUN:
            existing = pd.read_csv(target)
            candidate = frame.reset_index() if index else frame.reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(
                    existing.reset_index(drop=True),
                    candidate.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-14,
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f'Step08 partial-run CSV differs from the recomputed result: {target}'
                ) from exc
            STEP08_RUN_OUTPUT_PATHS.append(target)
            print(f'♻️ Step08 existing run-local CSV retained after equality check: {target}')
            return target
        raise FileExistsError(f'Refusing to overwrite Step08 run output: {target}')
    with allow_exact_final_derived_writes([target], owner=f'Step08 full run {STEP08_RUN_ID}'):
        frame.to_csv(target, index=index)
    STEP08_RUN_OUTPUT_PATHS.append(target)
    return target


def _step08_write_json(payload, path):
    """Write one exact run-local JSON while keeping the global Final guard enabled."""
    target = os.path.realpath(os.path.abspath(os.fspath(path)))
    if STEP08_MODE != 'full_run':
        raise RuntimeError('Run manifest JSON is available only in STEP08_MODE=full_run')
    if os.path.dirname(target) != os.path.realpath(STEP08_TABLE_DIR):
        raise PermissionError(f'Step08 run JSON escaped the exact table directory: {target}')
    if os.path.exists(target):
        raise FileExistsError(f'Refusing to overwrite Step08 run output: {target}')
    with allow_exact_final_derived_writes([target], owner=f'Step08 full run {STEP08_RUN_ID}'):
        Path(target).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    STEP08_RUN_OUTPUT_PATHS.append(target)
    return target

def _step08_table_path(filename):
    """Save Step 08 CSV/table files together in one folder."""
    return os.path.join(STEP08_TABLE_DIR, filename)

def _resolve_step08_rl_summary_path():
    """Return the selected canonical or isolated run-local R-L summary path."""
    return _step08_table_path(STEP08_RL_SUMMARY_CSV)

def _save_step08_matplotlib(fig, filename, label, dpi=300, *, bbox_inches=None, pad_inches=STEP08_SAVE_PAD_INCHES):
    """Save Step08 manuscript figures on the canvas as designed.

    The bbox_inches/pad_inches arguments support existing calls and do not affect the saved aspect ratio.
    """
    if filename not in STEP08_EXPORT_STATIC_FIGURES:
        plt.close(fig)
        print(f"  ⏩ {label} output excluded: {filename}")
        return None

    out_path = os.path.join(save_dir, filename)
    pptx_path = os.path.splitext(out_path)[0] + '.pptx'
    if STEP08_MODE == 'full_run' and STEP08_RESUME_PARTIAL_RUN:
        source_stem = Path(out_path).stem
        prefixed_stem = source_stem if source_stem.startswith('Final_') else f'Final_{source_stem}'
        normalized_stem = FINAL_FIGURE_STEM_ALIASES.get(prefixed_stem, prefixed_stem)
        normalized_png_path = os.path.join(save_dir, f'{normalized_stem}.png')
        normalized_pptx_path = os.path.join(save_dir, f'{normalized_stem}.pptx')
        existing_pair = (
            os.path.isfile(normalized_png_path),
            os.path.isfile(normalized_pptx_path),
        )
        if any(existing_pair):
            if not all(existing_pair):
                raise RuntimeError(
                    f'Step08 partial-run Figure pair is incomplete: '
                    f'{normalized_png_path}, {normalized_pptx_path}'
                )
            if os.path.getsize(normalized_png_path) <= 0 or os.path.getsize(normalized_pptx_path) <= 0:
                raise RuntimeError(
                    f'Step08 partial-run Figure pair is empty: '
                    f'{normalized_png_path}, {normalized_pptx_path}'
                )
            STEP08_RUN_OUTPUT_PATHS.extend([normalized_png_path, normalized_pptx_path])
            plt.close(fig)
            print(f'♻️ {label} existing run-local PNG/PPTX pair retained')
            return normalized_png_path
    saved_png_path, saved_pptx_path = save_final_figure_pair(
        fig, out_path, pptx_path,
        dpi=dpi, facecolor='white',
        auxiliary_tight=False,
        validate=True,
        preserve_target_directory=(
            STEP08_MODE == 'full_run' or STEP08_PREVIEW_FIG_DIR is not None
        ),
    )
    if filename in {STEP08_RL_HEATMAP_FIG, STEP08_RL_TOP_FIG}:
        native_validation = _validate_native_shape_slide(
            Path(saved_pptx_path), require_outlined_text=True,
        )
        forbidden_objects = {
            key: int(native_validation[key])
            for key in ('picture', 'svg_picture', 'text_node')
            if int(native_validation[key]) != 0
        }
        if forbidden_objects:
            raise RuntimeError(
                'Step08 R-L PPTX must be fully vectorized native geometry: '
                f'{forbidden_objects}'
            )
        print(
            f'✅ {Path(saved_pptx_path).stem} PPTX full-vector validation: '
            f"native_shapes={native_validation['native_shape']}, "
            'pictures=0, svg_pictures=0, text_nodes=0'
        )
    print(f"✅ {label} direct-canvas PNG/PPTX saved: {saved_png_path}, {saved_pptx_path}")
    if STEP08_MODE == 'full_run':
        STEP08_RUN_OUTPUT_PATHS.extend([saved_png_path, saved_pptx_path])
    return saved_png_path


def _step08_render_rl_heatmap_from_longform():
    """Regenerate Step08-01 with the approved 2x2 direct-canvas layout."""
    if not _final_figure_stem_registered(STEP08_RL_HEATMAP_FIG):
        raise RuntimeError(
            f'Step08 canonical heatmap stem is not registered: {STEP08_RL_HEATMAP_FIG}'
        )
    long_path = _step08_table_path(STEP08_RL_LONGFORM_CSV)
    if not os.path.exists(long_path):
        raise FileNotFoundError(
            "[Step 08-1] Heatmap input CSV is missing: "
            f"{long_path}"
        )

    rl_long_df = pd.read_csv(long_path)
    required_cols = {
        'R-L Axis', 'Sender', 'Receiver', 'LR_Log2_Enrichment',
        'Q-value (BH-FDR)', 'N_neighbor_pairs', 'Interaction_Radius_um', 'N_PERM',
    }
    missing_cols = sorted(required_cols.difference(rl_long_df.columns))
    if missing_cols:
        raise KeyError(f"[Step 08-1] Required long-form CSV columns are missing: {missing_cols}")

    source_pair_order = list(dict.fromkeys(rl_long_df['R-L Axis'].astype(str)))
    observed_axes = set(source_pair_order)
    if observed_axes != set(STEP08_RL_AXIS_ORDER):
        raise RuntimeError(
            '[Step 08-1] R-L axis set drifted: '
            f'observed={source_pair_order}, expected={list(STEP08_RL_AXIS_ORDER)}'
        )
    pair_order = list(STEP08_RL_AXIS_ORDER)
    receiver_order = sorted(set(rl_long_df['Receiver'].astype(str)))
    y_labels = []
    row_pair_names = []
    z_rows = []
    text_rows = []

    for pair_name in pair_order:
        pair_df = rl_long_df.loc[rl_long_df['R-L Axis'].astype(str).eq(pair_name)]
        sender_order = list(dict.fromkeys(pair_df['Sender'].astype(str)))
        for sender_name in sender_order:
            sender_df = pair_df.loc[pair_df['Sender'].astype(str).eq(sender_name)].copy()
            sender_df['N_neighbor_pairs'] = pd.to_numeric(
                sender_df['N_neighbor_pairs'], errors='coerce'
            )
            sender_df = sender_df.loc[
                sender_df['N_neighbor_pairs'].ge(STEP08_MIN_NEIGHBOR_PAIRS)
            ]
            value_by_receiver = {
                str(row['Receiver']): float(pd.to_numeric(row['LR_Log2_Enrichment'], errors='coerce'))
                for _, row in sender_df.iterrows()
            }
            q_by_receiver = {
                str(row['Receiver']): float(pd.to_numeric(row['Q-value (BH-FDR)'], errors='coerce'))
                for _, row in sender_df.iterrows()
            }
            row_values = np.array(
                [value_by_receiver.get(receiver, np.nan) for receiver in receiver_order],
                dtype=float,
            )
            if not np.any(np.isfinite(row_values)):
                continue
            row_text = []
            for receiver, value in zip(receiver_order, row_values):
                q_value = q_by_receiver.get(receiver, np.nan)
                if np.isfinite(q_value) and q_value < STEP08_FDR_ALPHA and value > 0:
                    row_text.append('***' if q_value < 0.001 else '**' if q_value < 0.01 else '*')
                else:
                    row_text.append('')
            y_labels.append({
                'B cell': 'B cells',
                'CD4_CTL': 'CD4 CTL',
                'CD8T': 'CD8 T',
            }.get(str(sender_name), str(sender_name)))
            row_pair_names.append(pair_name)
            z_rows.append(row_values)
            text_rows.append(row_text)

    if not z_rows:
        raise ValueError("[Step 08-1] The saved long-form CSV contains no values to display in a heatmap.")

    z_matrix_full = np.asarray(z_rows, dtype=float)
    active_col_idx = [
        idx for idx in range(len(receiver_order))
        if np.any(np.isfinite(z_matrix_full[:, idx]))
    ]
    active_receivers = [receiver_order[idx] for idx in active_col_idx]
    z_matrix = z_matrix_full[:, active_col_idx]
    text_matrix = np.asarray(
        [[row[idx] for idx in active_col_idx] for row in text_rows],
        dtype=object,
    )

    finite_vals = z_matrix[np.isfinite(z_matrix)]
    z_cap = float(np.nanpercentile(np.abs(finite_vals), 95)) if finite_vals.size else 1.0
    z_cap = min(max(z_cap, 0.5), 3.0)

    pair_groups = []
    group_start = 0
    for row_idx in range(1, len(row_pair_names)):
        if row_pair_names[row_idx] != row_pair_names[row_idx - 1]:
            pair_groups.append((row_pair_names[row_idx - 1], group_start, row_idx))
            group_start = row_idx
    pair_groups.append((row_pair_names[-1], group_start, len(row_pair_names)))
    if int(np.count_nonzero(text_matrix != '')) != STEP08_RL_EXPECTED_SIGNIFICANT_CELLS:
        raise RuntimeError(
            '[Step 08-1] significant-cell count drifted: '
            f"{int(np.count_nonzero(text_matrix != ''))}"
        )

    def _matrix_width(axis_name):
        return (
            STEP08_RL_PANEL_CONTENT_WIDTH
            - STEP08_RL_SENDER_LABEL_WIDTHS[axis_name]
            - STEP08_RL_PANEL_CONTENT_GAP
        )

    def _matrix_center_x(axis_name):
        card_x, _, card_w, _ = STEP08_RL_PANEL_POSITIONS[axis_name]
        content_left = card_x + (card_w - STEP08_RL_PANEL_CONTENT_WIDTH) / 2.0
        matrix_left = (
            content_left + STEP08_RL_SENDER_LABEL_WIDTHS[axis_name]
            + STEP08_RL_PANEL_CONTENT_GAP
        )
        return matrix_left + _matrix_width(axis_name) / 2.0

    def _display_bbox(fig, position):
        x, y, width, height = position
        points = fig.transFigure.transform([[x, y], [x + width, y + height]])
        return Bbox.from_extents(
            points[0, 0], points[0, 1], points[1, 0], points[1, 1]
        )

    def _validate_containment(fig, region_name, region, artists, axes=()):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        region_box = _display_bbox(fig, region)
        offenders = []
        for artist in artists:
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            box = artist.get_window_extent(renderer=renderer)
            if (
                box.x0 < region_box.x0 - 1.5 or box.y0 < region_box.y0 - 1.5
                or box.x1 > region_box.x1 + 1.5 or box.y1 > region_box.y1 + 1.5
            ):
                offenders.append(artist.get_text())
        for axis in axes:
            box = axis.get_window_extent(renderer=renderer)
            if (
                box.x0 < region_box.x0 - 1.5 or box.y0 < region_box.y0 - 1.5
                or box.x1 > region_box.x1 + 1.5 or box.y1 > region_box.y1 + 1.5
            ):
                offenders.append(f'axes:{axis.get_label() or "<unnamed>"}')
        if offenders:
            raise RuntimeError(
                f'[Step 08-1] {region_name} containment failed: {offenders[:8]}'
            )

    def _validate_cross_panel_text(fig, groups):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overlaps = []
        for left_index, left_group in enumerate(groups):
            for right_index in range(left_index + 1, len(groups)):
                for left_artist in left_group:
                    if not left_artist.get_visible() or not left_artist.get_text().strip():
                        continue
                    left_box = left_artist.get_window_extent(renderer=renderer)
                    for right_artist in groups[right_index]:
                        if not right_artist.get_visible() or not right_artist.get_text().strip():
                            continue
                        if left_box.overlaps(
                            right_artist.get_window_extent(renderer=renderer)
                        ):
                            overlaps.append(
                                f'{left_artist.get_text()!r} vs {right_artist.get_text()!r}'
                            )
        if overlaps:
            raise RuntimeError(
                f'[Step 08-1] cross-panel text overlap: {overlaps[:8]}'
            )

    matrix_area_center_x = (
        _matrix_center_x('HLA-DR → LAG3')
        + _matrix_center_x('CXCL13 → CXCR5')
    ) / 2.0
    fig_rl = plt.figure(figsize=STEP08_MAIN_FIGSIZES[STEP08_RL_HEATMAP_FIG])
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#EFEFEF')
    norm = mcolors.Normalize(vmin=-z_cap, vmax=z_cap)
    title = fig_rl.text(
        matrix_area_center_x, 0.985,
        'Ligand → Receptor enrichment by R–L axis',
        ha='center', va='top', fontsize=10.2, fontweight='bold',
    )
    y_axis_label = fig_rl.text(
        0.011, 0.4975, 'Sender cell type', rotation=90,
        ha='center', va='center', fontsize=8.4, fontweight='bold',
    )
    shared_x_label = fig_rl.text(
        matrix_area_center_x, 0.008,
        'Receiver cell type (receptor marker)',
        ha='center', va='bottom', fontsize=7.5, fontweight='bold',
    )

    group_slices = {
        name: (start, end) for name, start, end in pair_groups
    }
    panel_text_groups = []
    all_sender_texts = []
    matrix_horizontal_bounds = {}
    star_font_size = 6.2
    star_font = font_manager.FontProperties(
        family='Arial', weight='bold', size=star_font_size,
    )
    star_ink_center_points = {
        mark: float(np.mean(TextPath((0, 0), mark, prop=star_font).get_extents().intervaly))
        for mark in ('*', '**', '***')
    }
    max_star_center_offset = 0.0

    for axis_name in STEP08_RL_AXIS_ORDER:
        start, end = group_slices[axis_name]
        row_count = end - start
        card_x, card_y, card_w, _ = STEP08_RL_PANEL_POSITIONS[axis_name]
        sender_width = STEP08_RL_SENDER_LABEL_WIDTHS[axis_name]
        matrix_width = _matrix_width(axis_name)
        content_left = (
            card_x + (card_w - STEP08_RL_PANEL_CONTENT_WIDTH) / 2.0
        )
        matrix_left = (
            content_left + sender_width + STEP08_RL_PANEL_CONTENT_GAP
        )
        matrix_height = row_count * STEP08_RL_MATRIX_ROW_HEIGHT
        matrix_center_y = 0.690 if card_y > 0.45 else 0.323
        matrix_bottom = matrix_center_y - matrix_height / 2.0
        matrix_center_x = matrix_left + matrix_width / 2.0
        matrix_horizontal_bounds[axis_name] = (
            matrix_left, matrix_left + matrix_width,
        )
        panel_title = fig_rl.text(
            matrix_center_x, card_y + 0.360, axis_name,
            ha='center', va='center', fontsize=8.0, fontweight='bold',
            color=STEP08_RL_AXIS_PALETTE[axis_name],
        )
        sender_ax = fig_rl.add_axes(
            [content_left, matrix_bottom, sender_width, matrix_height],
            frameon=False,
        )
        heat_ax = fig_rl.add_axes(
            [matrix_left, matrix_bottom, matrix_width, matrix_height],
            facecolor='white',
        )
        values = z_matrix[start:end, :]
        x_edges = np.arange(values.shape[1] + 1, dtype=float) - 0.5
        y_edges = np.arange(row_count + 1, dtype=float) - 0.5
        heat_ax.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(values),
            cmap=cmap, norm=norm, shading='flat', edgecolors='none',
            rasterized=False,
        )
        heat_ax.set_xlim(-0.5, values.shape[1] - 0.5)
        heat_ax.set_ylim(row_count - 0.5, -0.5)
        heat_ax.set_xticks(np.arange(values.shape[1]))
        heat_ax.set_xticklabels(
            [
                {'B cell': 'B cells', 'CD4_CTL': 'CD4 CTL', 'CD8T': 'CD8 T'}
                .get(str(value), str(value))
                for value in active_receivers
            ] if card_y < 0.45 else [],
            rotation=42, ha='right', va='top', fontsize=6.0,
            fontweight='bold',
        )
        heat_ax.tick_params(axis='x', length=1.6, width=0.65, pad=0.7)
        heat_ax.set_yticks([])
        heat_ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
        heat_ax.set_yticks(np.arange(-0.5, row_count, 1), minor=True)
        heat_ax.grid(which='minor', color='white', linewidth=0.55)
        heat_ax.tick_params(which='minor', bottom=False, left=False)
        for spine in heat_ax.spines.values():
            spine.set_color(STEP08_RL_AXIS_PALETTE[axis_name])
            spine.set_linewidth(0.75)

        sender_ax.set_xlim(0, 1)
        sender_ax.set_ylim(row_count - 0.5, -0.5)
        sender_ax.set_xticks([])
        sender_ax.set_yticks([])
        sender_texts = [
            sender_ax.text(
                0.98, local_row, label, ha='right', va='center',
                fontsize=6.5, fontweight='bold',
            )
            for local_row, label in enumerate(y_labels[start:end])
        ]
        all_sender_texts.extend(sender_texts)

        annotation_texts = []
        for local_row in range(row_count):
            for column in range(values.shape[1]):
                mark = str(text_matrix[start + local_row, column])
                value = values[local_row, column]
                if not mark or not np.isfinite(value):
                    continue
                red, green, blue, _ = cmap(norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                cell_center_display = heat_ax.transData.transform((column, local_row))
                baseline_display_y = (
                    cell_center_display[1]
                    - star_ink_center_points[mark] * fig_rl.dpi / 72.0
                )
                baseline_data_y = heat_ax.transData.inverted().transform(
                    (cell_center_display[0], baseline_display_y)
                )[1]
                artist = heat_ax.text(
                    column, baseline_data_y, mark, ha='center', va='baseline',
                    fontsize=star_font_size, fontweight='bold',
                    color='white' if luminance < 0.52 else '#202020',
                )
                annotation_texts.append(artist)
                rendered_center_y = (
                    heat_ax.transData.transform(artist.get_position())[1]
                    + star_ink_center_points[mark] * fig_rl.dpi / 72.0
                )
                max_star_center_offset = max(
                    max_star_center_offset,
                    abs(float(rendered_center_y - cell_center_display[1])),
                )

        panel_artists = [
            panel_title, *sender_texts, *heat_ax.get_xticklabels(),
            *annotation_texts,
        ]
        panel_text_groups.append(panel_artists)
        _validate_containment(
            fig_rl, axis_name, STEP08_RL_PANEL_POSITIONS[axis_name],
            panel_artists, axes=(sender_ax, heat_ax),
        )

    if max_star_center_offset >= 1e-6:
        raise RuntimeError(
            '[Step 08-1] significance stars are not optically centered: '
            f'{max_star_center_offset}'
        )

    legend_groups = []
    score_texts = [
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.835, 'L–R score',
            ha='center', va='center', fontsize=6.6, fontweight='bold',
        ),
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.790,
            'log2(observed / null mean)',
            ha='center', va='center', fontsize=5.2, fontweight='bold',
        ),
    ]
    cbar_ax = fig_rl.add_axes([
        STEP08_RL_LEGEND_CENTER_X - 0.0525, 0.700, 0.105, 0.027,
    ])
    colorbar_edges = np.linspace(float(norm.vmin), float(norm.vmax), 65)
    colorbar_values = (
        (colorbar_edges[:-1] + colorbar_edges[1:]) / 2.0
    )[None, :]
    cbar_ax.pcolormesh(
        colorbar_edges, np.array([0.0, 1.0]), colorbar_values,
        cmap=cmap, norm=norm, shading='flat', edgecolors='none',
        rasterized=False,
    )
    cbar_ax.set_xlim(float(norm.vmin), float(norm.vmax))
    cbar_ax.set_ylim(0.0, 1.0)
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([-2.0, 0.0, 2.0])
    cbar_ax.tick_params(
        axis='x', labelsize=5.0, length=1.3, width=0.55, pad=0.7,
    )
    for tick in cbar_ax.get_xticklabels():
        tick.set_fontweight('bold')
    for spine in cbar_ax.spines.values():
        spine.set_linewidth(0.65)
    score_group = [*score_texts, *cbar_ax.get_xticklabels()]
    legend_groups.append(score_group)
    _validate_containment(
        fig_rl, 'L-R score', STEP08_RL_LEGEND_POSITIONS['score'],
        score_group, axes=(cbar_ax,),
    )

    fdr_group = [
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.545, 'BH-FDR',
            ha='center', va='center', fontsize=6.5, fontweight='bold',
        ),
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.485, '*   q < 0.05',
            ha='center', va='center', fontsize=5.3, fontweight='bold',
        ),
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.443, '**  q < 0.01',
            ha='center', va='center', fontsize=5.3, fontweight='bold',
        ),
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.403, '*** q < 0.001',
            ha='center', va='center', fontsize=5.3, fontweight='bold',
        ),
    ]
    legend_groups.append(fdr_group)
    _validate_containment(
        fig_rl, 'BH-FDR', STEP08_RL_LEGEND_POSITIONS['fdr'], fdr_group,
    )

    radius_um = float(pd.to_numeric(
        rl_long_df['Interaction_Radius_um'], errors='raise',
    ).iloc[0])
    n_perm = int(pd.to_numeric(
        rl_long_df['N_PERM'], errors='raise',
    ).iloc[0])
    settings_group = [
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.292, 'Analysis settings',
            ha='center', va='center', fontsize=6.1, fontweight='bold',
        ),
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.247, f'≤{radius_um:g} µm',
            ha='center', va='center', fontsize=5.0, fontweight='bold',
        ),
        fig_rl.text(
            STEP08_RL_LEGEND_CENTER_X, 0.216, f'{n_perm:,} permutations',
            ha='center', va='center', fontsize=5.0, fontweight='bold',
        ),
    ]
    legend_groups.append(settings_group)
    _validate_containment(
        fig_rl, 'Analysis settings',
        STEP08_RL_LEGEND_POSITIONS['settings'], settings_group,
    )

    fig_rl.canvas.draw()
    renderer = fig_rl.canvas.get_renderer()
    y_axis_box = y_axis_label.get_window_extent(renderer=renderer)
    sender_boxes = [
        artist.get_window_extent(renderer=renderer)
        for artist in all_sender_texts
    ]
    y_axis_left_clearance = float(y_axis_box.x0)
    y_axis_sender_gap = float(
        min(box.x0 for box in sender_boxes) - y_axis_box.x1
    )
    if y_axis_left_clearance < 3.0 or y_axis_sender_gap < 3.0:
        raise RuntimeError(
            '[Step 08-1] Y-axis title lacks clearance: '
            f'left={y_axis_left_clearance:.2f}, sender_gap={y_axis_sender_gap:.2f}'
        )
    matrix_gaps = {
        'top': (
            matrix_horizontal_bounds['CXCL13 → CXCR5'][0]
            - matrix_horizontal_bounds['HLA-DR → LAG3'][1]
        ),
        'bottom': (
            matrix_horizontal_bounds['CD34 → CD62L'][0]
            - matrix_horizontal_bounds['PDPN → CD44'][1]
        ),
    }
    if min(gap * fig_rl.bbox.width for gap in matrix_gaps.values()) < 3.0:
        raise RuntimeError(
            f'[Step 08-1] left/right heatmaps overlap: {matrix_gaps}'
        )
    lower_panel_bottom = min(
        _display_bbox(fig_rl, STEP08_RL_PANEL_POSITIONS[name]).y0
        for name in ('PDPN → CD44', 'CD34 → CD62L')
    )
    if lower_panel_bottom - shared_x_label.get_window_extent(renderer=renderer).y1 < 3.0:
        raise RuntimeError('[Step 08-1] shared X-axis title overlaps lower panels')
    _validate_cross_panel_text(fig_rl, [*panel_text_groups, *legend_groups])
    validate_final_figure_canvas(fig_rl)

    saved_path = _save_step08_matplotlib(
        fig_rl, STEP08_RL_HEATMAP_FIG,
        'Step08-1 R-L enrichment heatmap', bbox_inches=None,
    )
    if saved_path is not None:
        plt.show()
    return saved_path

# The Final canonical pipeline does not automatically move/delete existing tables/Figures/HTML.
# Migrate/clean up historical outputs only with a separate command using a reviewed exact allowlist;
# do not include destructive cleanup in Step08 analysis or Figure execution paths.
print("🔹 [Step 08] No automatic moving/deletion of existing outputs → retain all legacy artifacts as read-only.")

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

def _fdr_matrix(pvalue_matrix, valid_mask):
    """Apply BH-FDR correction only to valid_mask entries in a 2D p-value matrix."""
    q_matrix = np.full(np.asarray(pvalue_matrix).shape, np.nan, dtype=float)
    if np.any(valid_mask):
        q_matrix[valid_mask] = _bh_fdr(np.asarray(pvalue_matrix)[valid_mask])
    return q_matrix

# %%
# ==============================================================================
# [Step 08-1] CODEX marker-constrained spatial R-L proximity enrichment
# ▷ Test R-L proximity using a neighborhood-enrichment structure commonly used in spatial studies
# ▷ Fix the cKDTree radius graph → shuffle labels within Run_Tissue_name → observed/null enrichment
# ▷ Score = mean(ligand_sender × receptor_receiver) across neighboring directed sender→receiver pairs
# ▷ Interpretation: evidence of CODEX marker-based spatial R-L proximity, rather than direct proof of signaling
# ==============================================================================

if STEP08_FIGURE_ONLY:
    print("🔹 [Step 08-1] Figure-only mode → skip permutation analysis and use the saved summary CSV.")
    STEP08_FIGURE_ONLY_SAVED_REPORTS.append(
        _step08_render_rl_heatmap_from_longform()
    )
elif STEP08_RESUME_PARTIAL_RUN:
    _resume_long_path_082 = _step08_table_path(STEP08_RL_LONGFORM_CSV)
    _resume_summary_path_082 = _step08_table_path(STEP08_RL_SUMMARY_CSV)
    _resume_required_082 = [_resume_long_path_082, _resume_summary_path_082]
    _resume_missing_082 = [
        path for path in _resume_required_082
        if not os.path.isfile(path) or os.path.getsize(path) <= 0
    ]
    if _resume_missing_082:
        raise RuntimeError(
            f'Step08 partial-run R-L tables are missing or empty: {_resume_missing_082}'
        )
    _resume_disease_082 = adata.obs['Disease'].astype(str)
    _resume_tissue_082 = adata.obs['Run_Tissue_name'].astype(str)
    _resume_x_col_082 = next(
        (c for c in ['X_centroid', 'CenterX', 'X', 'x'] if c in adata.obs.columns),
        None,
    )
    _resume_y_col_082 = next(
        (c for c in ['Y_centroid', 'CenterY', 'Y', 'y'] if c in adata.obs.columns),
        None,
    )
    if _resume_x_col_082 is None or _resume_y_col_082 is None:
        raise KeyError('[Step 08-1] resume requires the same X/Y centroid columns as the source run')
    _resume_rl_mask_082 = (
        _resume_disease_082.str.contains('DLBCL', na=False)
        & _resume_tissue_082.isin(STEP08_EXACT_DLBCL_TISSUES)
        & adata.obs[[_resume_x_col_082, _resume_y_col_082, 'Run_Tissue_name', 'Niche_Label']]
        .notna().all(axis=1)
    )
    STEP08_RL_TISSUE_COUNTS = (
        _resume_tissue_082.loc[_resume_rl_mask_082]
        .value_counts().sort_index().astype(int).to_dict()
    )
    if tuple(STEP08_RL_TISSUE_COUNTS) != tuple(sorted(STEP08_EXACT_DLBCL_TISSUES)):
        raise RuntimeError(
            f'Step08 partial-run R-L tissue contract mismatch: {STEP08_RL_TISSUE_COUNTS}'
        )
    rl_long_df = pd.read_csv(_resume_long_path_082)
    summary_df = pd.read_csv(_resume_summary_path_082)
    _resume_heatmap_pair_082 = [
        os.path.join(STEP08_RUN_FIG_DIR, f'Final_mv1_Step08-01_RL_Interaction_Heatmap.{suffix}')
        for suffix in ('png', 'pptx')
    ]
    if any(not os.path.isfile(path) or os.path.getsize(path) <= 0 for path in _resume_heatmap_pair_082):
        raise RuntimeError(
            f'Step08 partial-run R-L heatmap pair is missing or empty: {_resume_heatmap_pair_082}'
        )
    STEP08_RUN_OUTPUT_PATHS.extend(_resume_heatmap_pair_082)
    print(
        '♻️ [Step 08-1] verified partial-run R-L tables and Figure pair retained; '
        'permutation calculation was not repeated.'
    )
else:
    print("🔹 [Step 08-1] Starting CODEX marker-based R-L spatial proximity enrichment analysis...")


    # =====================================================================
    # 📌 1. Define R-L pairs and restrict cell types by biologically plausible expression
    # =====================================================================
    # bio_senders: cell types that can express the ligand (based on biological evidence)
    # bio_receivers: cell types that can express the receptor (based on biological evidence)
    # Exclude other Sender↔Receiver combinations from analysis by masking their scores to zero
    RL_PAIRS = [
        {
            'name': 'CXCL13 → CXCR5',
            'ligand': 'CXCL13',
            'receptor': 'CXCR5',
            'description': 'Follicular positioning-compatible spatial context',
            # Spatial proximity does not establish a chemokine gradient or recruitment.
            'bio_senders':   ['Tfh', 'FDC'],
            'bio_receivers': ['B cell', 'Tfh']
        },
        {
            'name': 'HLA-DR → LAG3',
            'ligand': 'HLA-DR',
            'receptor': 'LAG3',
            'description': 'LAG3–MHC-II interaction-compatible spatial context',
            # HLA-DR (MHC-II) = expressed by B cells, DCs, macrophages, and FDCs
            # LAG3 = T-cell lineages (inhibitory receptor)
            'bio_senders':   ['B cell', 'DC', 'Macrophage', 'FDC'],
            'bio_receivers': ['CD8T', 'Tfh', 'Th', 'Treg', 'CD4_CTL', 'Tpr']
        },
        {
            'name': 'CD34 → CD62L',
            'ligand': 'CD34',
            'receptor': 'CD62L',
            'description': 'HEV-like vascular homing-compatible spatial context',
            # CD34 = HEV endothelial cells (Stromal), stem-cell marker
            # CD62L (L-selectin) = expressed by naive/CM T cells and NK cells
            'bio_senders':   ['Stromal'],
            'bio_receivers': ['CD8T', 'Tfh', 'Th', 'Treg', 'CD4_CTL', 'Tpr', 'NK cell']
        },
        {
            'name': 'PDPN → CD44',
            'ligand': 'PDPN',
            'receptor': 'CD44',
            'description': 'Non-canonical stromal/membrane organization co-context',
            # PDPN = expressed by FRCs and stromal cells
            # CD44 = expressed by activated lymphocytes (including B cells)
            'bio_senders':   ['Stromal', 'FDC'],
            'bio_receivers': ['B cell', 'CD8T', 'Tfh', 'Th', 'Treg', 'CD4_CTL', 'Tpr']
        }
    ]

    # Check that markers exist in the data, then retain only valid pairs
    valid_pairs = [p for p in RL_PAIRS if p['ligand'] in adata.var_names and p['receptor'] in adata.var_names]
    results = {}
    rl_long_rows = []
    print(f"  ✅ Valid R-L pairs: {len(valid_pairs)}/{len(RL_PAIRS)}")
    for p in valid_pairs:
        print(f"     • {p['name']} — {p['description']}")

    if not valid_pairs:
        print("  ⚠️ No analyzable R-L pairs are available; skip Step 08-1.")
    else:
        # =====================================================================
        # 📌 2. Extract target DLBCL data, coordinates, and the expression matrix
        # =====================================================================
        required_cols_082 = ['Disease', 'Run_Tissue_name', 'Niche_Label']
        missing_cols_082 = [c for c in required_cols_082 if c not in adata.obs.columns]
        if missing_cols_082:
            raise KeyError(f"[Step 08-1] Required obs columns are missing: {missing_cols_082}")

        disease_series = adata.obs['Disease'].astype(str)
        tissue_series_all = adata.obs['Run_Tissue_name'].astype(str)
        dlbcl_mask = (
            disease_series.str.contains('DLBCL', na=False)
            & tissue_series_all.isin(STEP08_EXACT_DLBCL_TISSUES)
        )

        obs_082 = adata.obs.loc[dlbcl_mask].copy()
        x_col = next((c for c in ['X_centroid', 'CenterX', 'X', 'x'] if c in obs_082.columns), None)
        y_col = next((c for c in ['Y_centroid', 'CenterY', 'Y', 'y'] if c in obs_082.columns), None)
        if x_col is None or y_col is None:
            raise KeyError("[Step 08-1] Spatial coordinate columns were not found. Check X/Y centroid column names.")

        obs_082[x_col] = pd.to_numeric(obs_082[x_col], errors='coerce')
        obs_082[y_col] = pd.to_numeric(obs_082[y_col], errors='coerce')
        valid_obs_mask = (
            obs_082[[x_col, y_col, 'Run_Tissue_name', 'Niche_Label']]
            .notna()
            .all(axis=1)
        )
        n_drop = int((~valid_obs_mask).sum())
        if n_drop > 0:
            print(f"  ⚠️ Exclude {n_drop:,} cells lacking coordinate/label/tissue information")
        obs_082 = obs_082.loc[valid_obs_mask].copy()
        obs_082['Niche_Label'] = obs_082['Niche_Label'].astype(str)
        obs_082['Run_Tissue_name'] = obs_082['Run_Tissue_name'].astype(str)
        if len(obs_082) == 0:
            raise ValueError("[Step 08-1] No valid DLBCL cells are available.")
        if STEP08_MODE == 'full_run':
            observed_tissues_082 = tuple(sorted(obs_082['Run_Tissue_name'].unique()))
            expected_tissues_082 = tuple(sorted(STEP08_EXACT_DLBCL_TISSUES))
            if observed_tissues_082 != expected_tissues_082:
                raise RuntimeError(
                    '[Step 08-1] full-run tissue contract mismatch: '
                    f'observed={observed_tissues_082}, expected={expected_tissues_082}'
                )
            STEP08_RL_TISSUE_COUNTS = (
                obs_082['Run_Tissue_name'].value_counts().sort_index().astype(int).to_dict()
            )
            print(
                f"  ✅ [Step 08-1] exact 8-tissue cohort: {len(obs_082):,} valid cells; "
                f"tissues={observed_tissues_082}"
            )

        all_rl_markers = sorted(set([p['ligand'] for p in valid_pairs] + [p['receptor'] for p in valid_pairs]))

        def _extract_marker_frame(source_adata, obs_names, markers, prefer_raw=False):
            """Extract a marker-expression matrix from AnnData or raw as a float32 DataFrame."""
            matrix_source = source_adata.raw if prefer_raw and source_adata.raw is not None else source_adata
            _X = matrix_source[obs_names, markers].X
            if sp.issparse(_X):
                arr = _X.toarray()
            elif hasattr(_X, 'todense'):
                arr = np.asarray(_X.todense())
            else:
                arr = np.asarray(_X)
            return pd.DataFrame(arr.astype(np.float32, copy=False), index=obs_names, columns=markers)

        expr_rl = _extract_marker_frame(adata, obs_082.index, all_rl_markers, prefer_raw=False)
        if adata.raw is not None:
            try:
                expr_raw = _extract_marker_frame(adata, obs_082.index, all_rl_markers, prefer_raw=True)
                expr_source_note = 'adata.raw'
            except Exception as e:
                print(f"  ⚠️ Raw-data extraction failed: {e} → use current adata.X values")
                expr_raw = expr_rl.copy()
                expr_source_note = 'adata.X'
        else:
            print("  ⚠️ adata.raw is absent → use current adata.X values")
            expr_raw = expr_rl.copy()
            expr_source_note = 'adata.X'

        # Use only nonnegative signals for R-L scores to avoid overinterpreting products of negative z-scores.
        expr_signal = expr_raw.clip(lower=0)

        cell_types = sorted(obs_082['Niche_Label'].unique())
        n_types = len(cell_types)
        ct_to_int = {ct: i for i, ct in enumerate(cell_types)}
        labels_int_original = obs_082['Niche_Label'].map(ct_to_int).to_numpy(dtype=np.int32)

        INTERACT_R = float(adata.uns.get('INTERACT_CUTOFF', 50))
        INTERACT_R_COORD_UNITS = INTERACT_R / STEP08_COORD_UM_PER_UNIT
        N_PERM = STEP08_N_PERM
        MIN_NEIGHBOR_PAIRS = STEP08_MIN_NEIGHBOR_PAIRS
        EPS = 1e-9
        rng_seed_base = int(globals().get('RANDOM_SEED', 42))

        tissue_labels = obs_082['Run_Tissue_name'].to_numpy()
        tissues_082 = np.array(pd.unique(obs_082['Run_Tissue_name']))
        coords_all = obs_082[[x_col, y_col]].to_numpy(dtype=np.float32)

        print(
            f"  📏 Spatial-neighbor search radius: {INTERACT_R:g} µm = {INTERACT_R_COORD_UNITS:.2f} annotation coordinate units "
            f"(1 coordinate unit = {STEP08_COORD_UM_PER_UNIT:.5f} µm)"
        )
        print(f"  🧫 Cells analyzed: {len(obs_082):,} / {len(tissues_082)} tissues")
        print(f"  🧬 R-L score expression values: extracted from {expr_source_note}, then negative values set to zero")
        print(f"  🔀 Null model: {N_PERM:,} within-tissue label shuffles")

        # =====================================================================
        # 📌 3. Generate tissue-specific spatial-neighbor graphs
        # =====================================================================
        print("  🔄 Calculating tissue-specific spatial-neighbor graphs...")
        neighbor_pairs = []
        tissue_pos_groups = []

        for tissue in tqdm(tissues_082, desc="Tissue-specific KD-tree neighbor search"):
            tissue_pos = np.flatnonzero(tissue_labels == tissue)
            if len(tissue_pos) > 1:
                tissue_pos_groups.append(tissue_pos)
            if len(tissue_pos) < 10:
                continue

            coords = coords_all[tissue_pos]
            tree = cKDTree(coords)
            pairs = tree.query_pairs(r=INTERACT_R_COORD_UNITS, output_type='ndarray')
            if len(pairs) == 0:
                continue

            pos_pairs = np.column_stack([tissue_pos[pairs[:, 0]], tissue_pos[pairs[:, 1]]])
            neighbor_pairs.append(pos_pairs.astype(np.int64, copy=False))

        if not neighbor_pairs:
            print("  ⚠️ No neighbor pairs fall within the radius; no R-L proximity analysis was generated.")
        else:
            all_neighbor_pairs = np.concatenate(neighbor_pairs, axis=0)
            all_neighbor_pairs_bi = np.concatenate([all_neighbor_pairs, all_neighbor_pairs[:, ::-1]], axis=0)
            sender_pos_np = all_neighbor_pairs_bi[:, 0].astype(np.int64, copy=False)
            receiver_pos_np = all_neighbor_pairs_bi[:, 1].astype(np.int64, copy=False)
            print(f"  ✅ Total directed neighbor pairs: {len(all_neighbor_pairs_bi):,}")

            # =====================================================================
            # 📌 4. Detect GPU/CPU automatically and calculate vectorized R-L scores
            # =====================================================================
            USE_GPU = bool(STEP08_GPU_AVAILABLE)
            xp = cp if USE_GPU else np
            if USE_GPU:
                print("  🚀 R-L score/permutation: run in GPU vectorized mode.")
            else:
                print(
                    "  💻 R-L score/permutation: run in CPU vectorized mode. "
                    f"(Reason GPU is disabled: {STEP08_GPU_DISABLED_REASON or 'unknown'})"
                )

            sender_pos = xp.asarray(sender_pos_np, dtype=xp.int64)
            receiver_pos = xp.asarray(receiver_pos_np, dtype=xp.int64)
            labels_int_xp = xp.asarray(labels_int_original, dtype=xp.int32)
            expr_arrays = {
                m: xp.asarray(expr_signal[m].to_numpy(dtype=np.float32), dtype=xp.float32)
                for m in all_rl_markers
            }
            flat_size = n_types * n_types

            def _to_numpy(arr):
                return cp.asnumpy(arr) if USE_GPU else np.asarray(arr)

            tissue_label_blocks_xp = []
            if USE_GPU and STEP08_GPU_LABEL_SHUFFLE:
                for pos in tissue_pos_groups:
                    if len(pos) <= 1:
                        continue
                    pos_xp = cp.asarray(pos, dtype=cp.int64)
                    tissue_label_blocks_xp.append((pos_xp, labels_int_xp[pos_xp]))
                print(f"  🚀 GPU tissue-stratified label shuffle enabled: tissue blocks={len(tissue_label_blocks_xp)}")

            def _make_tissue_stratified_labels_cpu(perm_idx, pair_idx):
                """Shuffle Niche_Label only within each tissue to preserve sample composition."""
                rng = np.random.default_rng(rng_seed_base + pair_idx * 100_003 + perm_idx)
                shuffled = labels_int_original.copy()
                for pos in tissue_pos_groups:
                    shuffled[pos] = rng.permutation(shuffled[pos])
                return shuffled

            def _make_tissue_stratified_labels_xp(perm_idx, pair_idx):
                """Shuffle labels on the GPU when in use; fall back to CPU shuffling on failure."""
                if USE_GPU and STEP08_GPU_LABEL_SHUFFLE and tissue_label_blocks_xp:
                    try:
                        cp.random.seed(rng_seed_base + pair_idx * 100_003 + perm_idx)
                        shuffled_xp = labels_int_xp.copy()
                        for pos_xp, labels_block_xp in tissue_label_blocks_xp:
                            shuffled_xp[pos_xp] = labels_block_xp[cp.random.permutation(labels_block_xp.size)]
                        return shuffled_xp
                    except Exception as e:
                        if perm_idx == 0:
                            print(f"  ⚠️ GPU label shuffle failed → CPU label shuffle fallback: {type(e).__name__}: {e}")
                return xp.asarray(_make_tissue_stratified_labels_cpu(perm_idx, pair_idx), dtype=xp.int32)

            def compute_scores_vectorized(label_arr, lig_marker, rec_marker):
                """
                Calculate mean R-L product scores and neighbor counts together
                using sender/receiver labels of directed neighbor pairs.
                """
                s_types = label_arr[sender_pos]
                r_types = label_arr[receiver_pos]
                flat_idx = s_types * n_types + r_types

                lig_vals = expr_arrays[lig_marker][sender_pos]
                rec_vals = expr_arrays[rec_marker][receiver_pos]
                product_vals = lig_vals * rec_vals

                count_flat = xp.bincount(flat_idx, minlength=flat_size).astype(xp.float32)
                product_sum_flat = xp.bincount(flat_idx, weights=product_vals, minlength=flat_size).astype(xp.float32)

                counts = count_flat.reshape(n_types, n_types)
                product_sums = product_sum_flat.reshape(n_types, n_types)
                valid = counts >= MIN_NEIGHBOR_PAIRS
                scores = xp.where(valid, product_sums / xp.maximum(counts, 1), 0.0)
                return scores, counts

            # =====================================================================
            # 📌 5. Calculate observed values and tissue-stratified permutation nulls
            # =====================================================================
            for pair_idx, pair in enumerate(valid_pairs):
                print(f"\n  🔬 [{pair['name']}] Running tissue-stratified permutation analysis...")
                lig_m, rec_m = pair['ligand'], pair['receptor']

                observed, observed_counts = compute_scores_vectorized(labels_int_xp, lig_m, rec_m)
                perm_scores = xp.zeros((N_PERM, n_types, n_types), dtype=xp.float32)
                perm_counts = xp.zeros((N_PERM, n_types, n_types), dtype=xp.float32)

                for k in tqdm(range(N_PERM), desc=f"  Within-tissue label shuffle ({pair['name']})", leave=False):
                    shuffled_xp = _make_tissue_stratified_labels_xp(k, pair_idx)
                    score_k, count_k = compute_scores_vectorized(shuffled_xp, lig_m, rec_m)
                    perm_scores[k] = score_k
                    perm_counts[k] = count_k

                observed_np = _to_numpy(observed).astype(float)
                observed_counts_np = _to_numpy(observed_counts).astype(float)
                perm_np = _to_numpy(perm_scores).astype(float)
                perm_counts_np = _to_numpy(perm_counts).astype(float)

                null_mean = perm_np.mean(axis=0)
                null_sd = perm_np.std(axis=0, ddof=1)
                null_count_mean = perm_counts_np.mean(axis=0)
                null_count_sd = perm_counts_np.std(axis=0, ddof=1)

                lr_zscore = np.divide(
                    observed_np - null_mean,
                    null_sd,
                    out=np.full_like(observed_np, np.nan, dtype=float),
                    where=null_sd > 0
                )
                count_zscore = np.divide(
                    observed_counts_np - null_count_mean,
                    null_count_sd,
                    out=np.full_like(observed_counts_np, np.nan, dtype=float),
                    where=null_count_sd > 0
                )
                lr_log2_enrichment = np.log2((observed_np + EPS) / (null_mean + EPS))
                count_log2_enrichment = np.log2((observed_counts_np + EPS) / (null_count_mean + EPS))

                score_pvalues = np.full((n_types, n_types), np.nan, dtype=float)
                count_pvalues = np.full((n_types, n_types), np.nan, dtype=float)
                bio_s = set(pair.get('bio_senders', cell_types))
                bio_r = set(pair.get('bio_receivers', cell_types))
                bio_mask = np.zeros((n_types, n_types), dtype=bool)
                testable_mask = np.zeros((n_types, n_types), dtype=bool)

                for i, ct_s in enumerate(cell_types):
                    for j, ct_r in enumerate(cell_types):
                        if ct_s not in bio_s or ct_r not in bio_r:
                            continue
                        bio_mask[i, j] = True
                        if observed_counts_np[i, j] < MIN_NEIGHBOR_PAIRS:
                            continue
                        testable_mask[i, j] = True
                        score_pvalues[i, j] = (np.sum(perm_np[:, i, j] >= observed_np[i, j]) + 1) / (N_PERM + 1)
                        count_pvalues[i, j] = (np.sum(perm_counts_np[:, i, j] >= observed_counts_np[i, j]) + 1) / (N_PERM + 1)

                qvalues = _fdr_matrix(score_pvalues, testable_mask & np.isfinite(score_pvalues))
                count_qvalues = _fdr_matrix(count_pvalues, testable_mask & np.isfinite(count_pvalues))

                results[pair['name']] = {
                    'observed': observed_np,
                    'observed_counts': observed_counts_np,
                    'null_mean': null_mean,
                    'null_sd': null_sd,
                    'null_count_mean': null_count_mean,
                    'lr_zscore': lr_zscore,
                    'count_zscore': count_zscore,
                    'lr_log2_enrichment': lr_log2_enrichment,
                    'count_log2_enrichment': count_log2_enrichment,
                    'pvalues': score_pvalues,
                    'qvalues': qvalues,
                    'count_pvalues': count_pvalues,
                    'count_qvalues': count_qvalues,
                    'bio_mask': bio_mask,
                    'testable_mask': testable_mask,
                    'description': pair['description'],
                    'ligand': pair['ligand'],
                    'receptor': pair['receptor']
                }
                if USE_GPU:
                    cp.get_default_memory_pool().free_all_blocks()

                for i, ct_s in enumerate(cell_types):
                    for j, ct_r in enumerate(cell_types):
                        if not bio_mask[i, j]:
                            continue
                        qv = qvalues[i, j]
                        lr_log2 = lr_log2_enrichment[i, j]
                        significant = bool(np.isfinite(qv) and qv < STEP08_FDR_ALPHA and lr_log2 > 0)
                        rl_long_rows.append({
                            'R-L Axis': pair['name'],
                            'Ligand': lig_m,
                            'Receptor': rec_m,
                            'Description': pair['description'],
                            'Sender': ct_s,
                            'Receiver': ct_r,
                            'N_neighbor_pairs': int(observed_counts_np[i, j]),
                            'Expected_neighbor_pairs': float(null_count_mean[i, j]),
                            'Neighbor_Log2_Enrichment': float(count_log2_enrichment[i, j]),
                            'Neighbor_Zscore': float(count_zscore[i, j]) if np.isfinite(count_zscore[i, j]) else np.nan,
                            'Neighbor_P-value': float(count_pvalues[i, j]) if np.isfinite(count_pvalues[i, j]) else np.nan,
                            'Neighbor_Q-value (BH-FDR)': float(count_qvalues[i, j]) if np.isfinite(count_qvalues[i, j]) else np.nan,
                            'Observed_LR_Score': float(observed_np[i, j]),
                            'Null_Mean_LR_Score': float(null_mean[i, j]),
                            'LR_Log2_Enrichment': float(lr_log2),
                            'LR_Zscore': float(lr_zscore[i, j]) if np.isfinite(lr_zscore[i, j]) else np.nan,
                            'P-value': float(score_pvalues[i, j]) if np.isfinite(score_pvalues[i, j]) else np.nan,
                            'Q-value (BH-FDR)': float(qv) if np.isfinite(qv) else np.nan,
                            'Significant_Enriched': significant,
                            'Permutation_Strategy': 'label shuffle within Run_Tissue_name',
                            'Interaction_Radius_um': INTERACT_R,
                            'N_PERM': N_PERM
                        })

                n_sig = int(np.nansum((qvalues < STEP08_FDR_ALPHA) & (lr_log2_enrichment > 0)))
                print(f"    ✅ Complete — enriched R-L proximity: {n_sig} pairs")

            # =====================================================================
            # 📌 6. Save long-form results and generate summary figures
            # =====================================================================
            long_cols = [
                'R-L Axis', 'Ligand', 'Receptor', 'Description', 'Sender', 'Receiver',
                'N_neighbor_pairs', 'Expected_neighbor_pairs', 'Neighbor_Log2_Enrichment',
                'Neighbor_Zscore', 'Neighbor_P-value', 'Neighbor_Q-value (BH-FDR)',
                'Observed_LR_Score', 'Null_Mean_LR_Score', 'LR_Log2_Enrichment',
                'LR_Zscore', 'P-value', 'Q-value (BH-FDR)', 'Significant_Enriched',
                'Permutation_Strategy', 'Interaction_Radius_um', 'N_PERM'
            ]
            rl_long_df = pd.DataFrame(rl_long_rows, columns=long_cols)
            for p_col, out_col in [
                ('P-value', 'NegLog10_P_value'),
                ('Q-value (BH-FDR)', 'NegLog10_Q_value')
            ]:
                p_vals = pd.to_numeric(rl_long_df[p_col], errors='coerce')
                safe_p = p_vals.clip(lower=np.finfo(float).tiny, upper=1.0)
                rl_long_df[out_col] = -np.log10(safe_p)
            long_path = _step08_table_path(STEP08_RL_LONGFORM_CSV)
            _step08_write_csv(rl_long_df, long_path, index=False)
            print(f"\n  💾 Long-form R-L results saved: {long_path}")

            summary_cols = [
                'R-L Axis', 'Ligand', 'Receptor', 'Sender', 'Receiver',
                'N_neighbor_pairs', 'Expected_neighbor_pairs',
                'Neighbor_Log2_Enrichment', 'Observed_LR_Score',
                'Null_Mean_LR_Score', 'LR_Log2_Enrichment', 'LR_Zscore',
                'P-value', 'Q-value (BH-FDR)',
                'NegLog10_P_value', 'NegLog10_Q_value'
            ]
            if len(rl_long_df) > 0:
                summary_df = (
                    rl_long_df.loc[rl_long_df['Significant_Enriched'], summary_cols]
                    .sort_values(['Q-value (BH-FDR)', 'LR_Zscore'], ascending=[True, False])
                )
            else:
                summary_df = pd.DataFrame(columns=summary_cols)
            summary_path = _step08_table_path(STEP08_RL_SUMMARY_CSV)
            _step08_write_csv(summary_df, summary_path, index=False)

            print(f"\n📊 [Step 08-1] Significant R-L proximity summary (BH-FDR q < {STEP08_FDR_ALPHA}, enriched only):")
            print("=" * 100)
            if len(summary_df) > 0:
                print(summary_df.to_string(index=False))
                print(f"\n  Saved: {summary_path}")
                print(f"  A total of {len(summary_df)} enriched R-L proximity pairs were found.")
            else:
                print("  ⚠️ No significantly enriched R-L proximity pairs were found.")
                print(f"  Empty summary CSV saved: {summary_path}")

            print("\n🎨 [Step 08-1] Running the approved enrichment heatmap renderer...")
            _step08_render_rl_heatmap_from_longform()

            print("  ℹ️ Generate the Top R-L dot plot only once in Step 08-1b using the summary CSV.")

            adata.uns['step08_rl_analysis'] = {
                'method': 'CODEX marker-constrained spatial R-L proximity enrichment',
                'score': 'mean(ligand_sender * receptor_receiver) over directed within-radius neighbor edges',
                'null_model': 'shuffle Niche_Label within Run_Tissue_name while keeping coordinates and expression fixed',
                'interaction_radius_um': INTERACT_R,
                'n_permutations': int(N_PERM),
                'min_neighbor_pairs': int(MIN_NEIGHBOR_PAIRS),
                'fdr_alpha': float(STEP08_FDR_ALPHA),
                'expression_source': expr_source_note,
                'n_cells': int(len(obs_082)),
                'n_tissues': int(len(tissues_082)),
                'n_directed_neighbor_pairs': int(len(all_neighbor_pairs_bi)),
                'valid_pairs': [p['name'] for p in valid_pairs],
                'outputs': [
                    STEP08_RL_HEATMAP_FIG,
                    STEP08_RL_LONGFORM_CSV,
                    STEP08_RL_SUMMARY_CSV
                ],
                'dot_plot_redraw_cell': 'Step 08-1b writes PNG from STEP08_RL_SUMMARY_CSV'
            }

    print("\n✅ [Step 08-1] CODEX marker-based R-L proximity enrichment analysis complete!")

# %%
# ==============================================================================
# [Step 08-1b] Re-export the R-L dot plot (using the existing Step 08-1 CSV)
# ▷ Save only the dot plot again without rerunning all Step 08-1 permutations.
# ▷ Input: tables/mv1/Step08_CSV/MV1_Step081_RL_Interaction_Summary.csv
# ==============================================================================
def _step08_render_rl_dotplot_one_row(summary_frame):
    """Save Step08-02 directly on the canvas using the approved horizontal 4-panel layout."""
    required = {
        'R-L Axis', 'Sender', 'Receiver', 'LR_Log2_Enrichment',
        'Q-value (BH-FDR)', 'N_neighbor_pairs', 'Expected_neighbor_pairs',
    }
    missing = sorted(required.difference(summary_frame.columns))
    if missing:
        raise KeyError(f'[Step 08-1b] summary table lacks {missing}')

    frame = summary_frame.copy()
    for column in (
        'LR_Log2_Enrichment', 'Q-value (BH-FDR)',
        'N_neighbor_pairs', 'Expected_neighbor_pairs',
    ):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    significant = frame.loc[
        frame['Q-value (BH-FDR)'].lt(STEP08_FDR_ALPHA)
        & frame['LR_Log2_Enrichment'].gt(0)
    ].copy()
    if significant.empty:
        raise RuntimeError('[Step 08-1b] no positive R-L pairs pass BH-FDR')

    significant['CellCell_Pair'] = (
        significant['Sender'].astype(str) + ' → '
        + significant['Receiver'].astype(str)
    )
    observed_axes = list(dict.fromkeys(significant['R-L Axis'].astype(str)))
    axis_order = [axis for axis in STEP08_RL_AXIS_ORDER if axis in observed_axes]
    axis_order.extend(axis for axis in observed_axes if axis not in axis_order)
    if axis_order != list(STEP08_RL_AXIS_ORDER):
        raise RuntimeError(f'[Step 08-1b] R-L panel order drifted: {axis_order}')

    def _display_cell(value):
        return {
            'B cell': 'B cells',
            'CD4_CTL': 'CD4 CTL',
            'CD8T': 'CD8 T',
        }.get(str(value), str(value))

    def _display_pair(value):
        sender, receiver = str(value).split(' → ', maxsplit=1)
        return f'{_display_cell(sender)} → {_display_cell(receiver)}'

    def _q_category(q_value):
        if not np.isfinite(q_value):
            return 'q unknown'
        if q_value < 0.001:
            return 'q < 0.001'
        if q_value < 0.01:
            return 'q < 0.01'
        return 'q < 0.05'

    alpha_map = {
        'q < 0.001': 1.00,
        'q < 0.01': 0.58,
        'q < 0.05': 0.25,
        'q unknown': 0.16,
    }
    significant['Q_Category'] = significant['Q-value (BH-FDR)'].map(_q_category)
    significant['Dot_Alpha'] = significant['Q_Category'].map(alpha_map).fillna(0.35)
    significant['Neighbor_Pair_Support'] = (
        significant['N_neighbor_pairs']
        / significant['Expected_neighbor_pairs'].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-6)
    log_support = np.log2(
        significant['Neighbor_Pair_Support'].to_numpy(dtype=float)
    )
    log_min = float(log_support.min())
    log_max = float(log_support.max())

    def _support_to_size(values):
        support = (
            pd.to_numeric(pd.Series(values), errors='coerce')
            .fillna(1.0).clip(lower=1e-6)
        )
        logged = np.log2(support.to_numpy(dtype=float))
        if np.isclose(log_max, log_min):
            scaled = np.full_like(logged, 0.5)
        else:
            scaled = (logged - log_min) / (log_max - log_min)
        # Increase marker-area contrast nonlinearly to clearly distinguish 1.0x and 1.5x
        # support even in the reduced layout on Slide 10.
        scaled = np.clip(scaled, 0.0, 1.0)
        return 8.0 + np.power(scaled, 2.5) * 120.0

    significant['Dot_Size'] = _support_to_size(
        significant['Neighbor_Pair_Support']
    )
    pair_orders = {}
    counts = {}
    for axis_name in axis_order:
        ordered = (
            significant.loc[significant['R-L Axis'].eq(axis_name)]
            .groupby('CellCell_Pair', observed=True)
            .agg(
                Max_LR_Log2_Enrichment=('LR_Log2_Enrichment', 'max'),
                Min_Q=('Q-value (BH-FDR)', 'min'),
            )
            .reset_index()
            .sort_values(
                ['Max_LR_Log2_Enrichment', 'Min_Q'],
                ascending=[True, False],
            )
        )
        pair_orders[axis_name] = ordered['CellCell_Pair'].tolist()
        counts[axis_name] = int(
            len(significant.loc[significant['R-L Axis'].eq(axis_name)])
        )
    if counts != STEP08_RL_EXPECTED_SIGNIFICANT_COUNTS:
        raise RuntimeError(
            f'[Step 08-1b] significant pair count drifted: {counts}'
        )

    # Reserve shared left padding sufficient for the largest marker radius so low-score
    # points in the PDPN panel are not clipped at the Y-axis boundary.
    x_min = min(-0.75, float(significant['LR_Log2_Enrichment'].min()) - 0.45)
    x_max = float(significant['LR_Log2_Enrichment'].max()) + 0.45
    fig = plt.figure(figsize=STEP08_MAIN_FIGSIZES[STEP08_RL_TOP_FIG])
    outer = fig.add_gridspec(
        1, 4,
        width_ratios=STEP08_RL_DOTPLOT_PANEL_WIDTH_RATIOS,
        left=0.038, right=0.997, bottom=0.250, top=0.925, wspace=0.065,
    )
    label_fractions = (0.66, 0.71, 0.69, 0.69)
    label_axes = []
    data_axes = []
    panel_text_groups = [[] for _ in axis_order]
    pair_label_groups = [[] for _ in axis_order]

    for panel_index, axis_name in enumerate(axis_order):
        fraction = label_fractions[panel_index]
        subgrid = outer[0, panel_index].subgridspec(
            1, 2, width_ratios=(fraction, 1.0 - fraction), wspace=0.015,
        )
        label_ax = fig.add_subplot(subgrid[0, 0], frameon=False)
        data_ax = fig.add_subplot(subgrid[0, 1])
        label_axes.append(label_ax)
        data_axes.append(data_ax)

        panel = significant.loc[
            significant['R-L Axis'].eq(axis_name)
        ].copy()
        pair_order = pair_orders[axis_name]
        y_lookup = {pair: position for position, pair in enumerate(pair_order)}
        panel['Y_Pos'] = panel['CellCell_Pair'].map(y_lookup).astype(float)
        color = STEP08_RL_AXIS_PALETTE[axis_name]
        point_colors = [
            mcolors.to_rgba(color, alpha=float(alpha))
            for alpha in panel['Dot_Alpha']
        ]
        data_ax.scatter(
            panel['LR_Log2_Enrichment'], panel['Y_Pos'],
            s=panel['Dot_Size'], c=point_colors,
            edgecolor='#333333', linewidth=0.50, zorder=3,
            rasterized=False,
        )
        for pair in pair_order:
            data_ax.axhline(
                y_lookup[pair], color='black', lw=0.35, alpha=0.10, zorder=0,
            )
        data_ax.axvline(0, color='black', lw=0.95, linestyle='--')
        data_ax.set_ylim(-0.60, len(pair_order) - 0.40)
        data_ax.set_xlim(x_min, x_max)
        if x_max >= 1:
            data_ax.set_xticks(np.arange(0, np.ceil(x_max) + 0.1, 2.0))
        data_ax.tick_params(
            axis='x', labelsize=8.2, width=0.8, length=2.6, pad=1,
        )
        data_ax.tick_params(
            axis='y', left=False, right=False, labelleft=False, labelright=False,
        )
        data_ax.grid(
            axis='x', linestyle='--', color='#A7A7A7',
            linewidth=0.50, alpha=0.38,
        )
        data_ax.set_axisbelow(True)
        for spine in data_ax.spines.values():
            spine.set_linewidth(0.90)
        for tick in data_ax.get_xticklabels():
            tick.set_fontweight('bold')
        n_artist = data_ax.text(
            0.97, 0.025, f'n={len(panel)}', transform=data_ax.transAxes,
            ha='right', va='bottom', fontsize=8.0,
            fontweight='bold', color=color,
        )
        panel_text_groups[panel_index].append(n_artist)

        label_ax.set_xlim(0, 1)
        label_ax.set_ylim(data_ax.get_ylim())
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        label_fontsize = (
            8.1 if len(pair_order) >= 14
            else 8.4 if len(pair_order) >= 10
            else 8.6
        )
        for pair in pair_order:
            pair_artist = label_ax.text(
                0.98, y_lookup[pair], _display_pair(pair),
                ha='right', va='center', fontsize=label_fontsize,
                fontweight='bold',
            )
            panel_text_groups[panel_index].append(pair_artist)
            pair_label_groups[panel_index].append(pair_artist)

    fig.canvas.draw()
    for panel_index, (axis_name, label_ax, data_ax) in enumerate(
        zip(axis_order, label_axes, data_axes)
    ):
        left_box = label_ax.get_position()
        right_box = data_ax.get_position()
        title_artist = fig.text(
            (left_box.x0 + right_box.x1) / 2.0, 0.955, axis_name,
            ha='center', va='bottom', fontsize=10.5,
            fontweight='bold', color=STEP08_RL_AXIS_PALETTE[axis_name],
        )
        panel_text_groups[panel_index].append(title_artist)
        panel_text_groups[panel_index].extend(data_ax.get_xticklabels())

    shared_x_label = fig.text(
        0.50, 0.165, 'L–R score: log2(observed / null mean)',
        ha='center', va='center', fontsize=9.8, fontweight='bold',
    )
    fig.text(
        0.012, 0.588, 'Cell-cell pair', rotation=90,
        ha='center', va='center', fontsize=10.2, fontweight='bold',
    )

    support_min = float(significant['Neighbor_Pair_Support'].min())
    support_max = float(significant['Neighbor_Pair_Support'].max())
    support_values = [
        value for value in (0.5, 1.0, 1.5, 2.0)
        if support_min <= value <= support_max
    ]
    if not support_values:
        support_values = np.quantile(
            significant['Neighbor_Pair_Support'], [0.25, 0.5, 0.75],
        ).tolist()
    support_values = sorted({
        round(float(value), 2) for value in support_values if value > 0
    })
    support_handles = [
        mlines.Line2D(
            [], [], marker='o', linestyle='None', markerfacecolor='white',
            markeredgecolor='#333333',
            markersize=max(
                4.2, np.sqrt(float(_support_to_size([value])[0])) * 1.00,
            ),
            label=f'{value:.1f}x',
        )
        for value in support_values
    ]
    support_legend = fig.legend(
        handles=support_handles,
        title='Dot size = observed / expected neighbor pairs',
        loc='lower center', bbox_to_anchor=(0.285, 0.015),
        ncol=len(support_handles), frameon=False,
        fontsize=8.0, title_fontsize=8.4,
        handletextpad=0.35, columnspacing=0.85, borderaxespad=0,
    )
    support_legend.get_title().set_fontweight('bold')
    fig.add_artist(support_legend)

    alpha_handles = [
        mlines.Line2D(
            [], [], marker='o', linestyle='None',
            markerfacecolor=mcolors.to_rgba('#555555', alpha=alpha_map[category]),
            markeredgecolor='#333333', markersize=6.6, label=category,
        )
        for category in ('q < 0.05', 'q < 0.01', 'q < 0.001')
    ]
    alpha_legend = fig.legend(
        handles=alpha_handles, title='Fill alpha = BH-FDR',
        loc='lower center', bbox_to_anchor=(0.755, 0.015),
        ncol=3, frameon=False, fontsize=8.0, title_fontsize=8.4,
        handletextpad=0.35, columnspacing=0.85, borderaxespad=0,
    )
    alpha_legend.get_title().set_fontweight('bold')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = []
    for left_index, left_group in enumerate(panel_text_groups):
        for right_index in range(left_index + 1, len(panel_text_groups)):
            for left_artist in left_group:
                if not left_artist.get_visible() or not left_artist.get_text().strip():
                    continue
                left_bbox = left_artist.get_window_extent(renderer=renderer)
                for right_artist in panel_text_groups[right_index]:
                    if not right_artist.get_visible() or not right_artist.get_text().strip():
                        continue
                    if left_bbox.overlaps(
                        right_artist.get_window_extent(renderer=renderer)
                    ):
                        overlaps.append(
                            f'{left_artist.get_text()!r} vs {right_artist.get_text()!r}'
                        )
    if overlaps:
        raise RuntimeError(
            f'[Step 08-1b] cross-panel text overlap: {overlaps[:8]}'
        )
    escaped_labels = []
    for label_ax, text_group in zip(label_axes, pair_label_groups):
        label_bbox = label_ax.get_window_extent(renderer=renderer)
        for text_artist in text_group:
            text_bbox = text_artist.get_window_extent(renderer=renderer)
            if (
                text_bbox.x0 < label_bbox.x0 - 1.5
                or text_bbox.x1 > label_bbox.x1 + 1.5
            ):
                escaped_labels.append(text_artist.get_text())
    if escaped_labels:
        raise RuntimeError(
            f'[Step 08-1b] labels escaped dedicated columns: {escaped_labels[:8]}'
        )
    shared_x_bbox = shared_x_label.get_window_extent(renderer=renderer)
    shared_x_conflicts = []
    for data_ax in data_axes:
        for tick_artist in data_ax.get_xticklabels():
            if shared_x_bbox.overlaps(
                tick_artist.get_window_extent(renderer=renderer)
            ):
                shared_x_conflicts.append(tick_artist.get_text())
    for legend_name, legend_artist in (
        ('support legend', support_legend),
        ('BH-FDR legend', alpha_legend),
    ):
        if shared_x_bbox.overlaps(
            legend_artist.get_window_extent(renderer=renderer)
        ):
            shared_x_conflicts.append(legend_name)
    if shared_x_conflicts:
        raise RuntimeError(
            '[Step 08-1b] shared x-axis title overlap: '
            f'{shared_x_conflicts[:8]}'
        )
    validate_final_figure_canvas(fig)
    saved_path = _save_step08_matplotlib(
        fig, STEP08_RL_TOP_FIG,
        'Step08-1b cell-cell grouped R-L dot plot', bbox_inches=None,
    )
    if saved_path is not None:
        plt.show()
        print(
            f'✅ [Step 08-1b] One-row R-L dot plot saved: {saved_path} '
            f'(counts={counts})'
        )
    return saved_path


print("🔹 [Step 08-1b] Re-exporting the dot plot using the existing R-L summary CSV...")

rl_summary_path = _resolve_step08_rl_summary_path()
if not os.path.exists(rl_summary_path):
    raise FileNotFoundError(
        f'[Step 08-1b] canonical saved R-L summary is missing: {rl_summary_path}'
    )
else:
    top_df = pd.read_csv(rl_summary_path)
    required_cols_082b = [
        'R-L Axis', 'Sender', 'Receiver', 'LR_Log2_Enrichment', 'LR_Zscore',
        'P-value', 'Q-value (BH-FDR)', 'Observed_LR_Score', 'Null_Mean_LR_Score',
        'N_neighbor_pairs', 'Expected_neighbor_pairs'
    ]
    missing_cols_082b = [c for c in required_cols_082b if c not in top_df.columns]
    if missing_cols_082b:
        raise KeyError(f"[Step 08-1b] Required R-L summary CSV columns are missing: {missing_cols_082b}")

    for p_col, out_col in [
        ('P-value', 'NegLog10_P_value'),
        ('Q-value (BH-FDR)', 'NegLog10_Q_value')
    ]:
        p_vals = pd.to_numeric(top_df[p_col], errors='coerce')
        safe_p = p_vals.clip(lower=np.finfo(float).tiny, upper=1.0)
        top_df[out_col] = -np.log10(safe_p)

    STEP08_FIGURE_ONLY_SAVED_REPORTS.append(
        _step08_render_rl_dotplot_one_row(top_df)
    )

# %%
# ==============================================================================
# [Step 08-2] Activation/Exhaustion score correlation by T cell type
# ▷ Execution order: run this cell after Step 08-0.
# ▷ Outputs: Activation/Exhaustion correlations for all T cells and for each T-cell type
# ==============================================================================
print("🔹 [Step 08-2] Starting Activation/Exhaustion score correlation analysis...")

STEP08_SCORE_CORRELATION_CSV = 'MV1_Step082_Activation_Exhaustion_Correlation_ByTcellType.csv'
STEP08_SCORE_CORRELATION_BY_CELLTYPE_FIG = 'MV1_Step082_Activation_Exhaustion_Correlation_ByTcellType.png'
STEP08_SCORE_CORR_MAX_CELLS_PER_TYPE = 25_000
STEP08_T_SUBTYPE_TITLE_COLORS = {
    'CD4_CTL': '#7E57C2',
    'CD8T': '#1E88E5',
    'Tfh': '#F9A825',
    'Th': '#43A047',
    'Tpr': '#E64A19',
    'Treg': '#6D4C41',
}
STEP08_SCORE_CORR_B_STATE_COLORS = {
    'Near DR(+) B cell': '#C43C39',
    'Near DR(-) B cell': '#2F7EAD',
}
STEP08_SCORE_CORR_B_STATE_DISPLAY = {
    'Near DR(+) B cell': 'Near DR(+) B cells',
    'Near DR(-) B cell': 'Near DR(-) B cells',
}

if 'adata' not in globals():
    raise RuntimeError('[Step 08-2] adata is missing. Run the Step 08-0 cell first to load the Step 06 checkpoint.')

_required_cols_084 = ['Niche_Label', 'Activation_Score', 'Exhaustion_Score']
_missing_cols_084 = [c for c in _required_cols_084 if c not in adata.obs.columns]
if _missing_cols_084:
    raise KeyError(f'[Step 08-2] Required obs columns are missing: {_missing_cols_084}. Check Step 06-2 in Final_step06.py first.')

save_dir = (
    STEP08_RUN_FIG_DIR if STEP08_MODE == 'full_run'
    else STEP08_PREVIEW_FIG_DIR if STEP08_PREVIEW_FIG_DIR is not None
    else FIG_DIR
)
os.makedirs(save_dir, exist_ok=True)

_niche_label_084 = adata.obs['Niche_Label'].astype(str)
_missing_scope_cols_084 = [
    col for col in ['Disease', 'Run_Tissue_name']
    if col not in adata.obs.columns
]
if _missing_scope_cols_084:
    raise KeyError(
        f'[Step 08-2] cohort columns are missing: {_missing_scope_cols_084}'
    )
_analysis_labels_084 = list(STEP08_T_CELL_LABEL_ORDER)
if not STEP08_INCLUDE_TPR:
    _analysis_labels_084.remove('Tpr')
t_mask_084 = (
    adata.obs['Disease'].astype(str).str.contains('DLBCL', na=False)
    & adata.obs['Run_Tissue_name'].astype(str).isin(STEP08_EXACT_DLBCL_TISSUES)
    & _niche_label_084.isin(_analysis_labels_084)
)
if t_mask_084.sum() == 0:
    raise ValueError('[Step 08-2] No target T-cell data were found. Check the Step 06-2 execution status.')

_score_cols_084 = ['Niche_Label', 'Activation_Score', 'Exhaustion_Score']
if 'Run_Tissue_name' in adata.obs.columns:
    _score_cols_084.append('Run_Tissue_name')
_distance_cols_084 = [c for c in ['Dist_to_Pos', 'Dist_to_Neg'] if c in adata.obs.columns]
score_pair_df = adata.obs.loc[t_mask_084, _score_cols_084 + _distance_cols_084].copy()
score_pair_df['T_Subset'] = score_pair_df['Niche_Label'].astype(str)
score_pair_df['Activation_Score'] = pd.to_numeric(score_pair_df['Activation_Score'], errors='coerce')
score_pair_df['Exhaustion_Score'] = pd.to_numeric(score_pair_df['Exhaustion_Score'], errors='coerce')
if 'Run_Tissue_name' in score_pair_df.columns:
    score_pair_df['Run_Tissue_name'] = score_pair_df['Run_Tissue_name'].astype(str)
else:
    score_pair_df['Run_Tissue_name'] = 'unknown'
if {'Dist_to_Pos', 'Dist_to_Neg'}.issubset(score_pair_df.columns):
    score_pair_df['Dist_to_Pos'] = pd.to_numeric(score_pair_df['Dist_to_Pos'], errors='coerce')
    score_pair_df['Dist_to_Neg'] = pd.to_numeric(score_pair_df['Dist_to_Neg'], errors='coerce')
    dist_pos_084 = score_pair_df['Dist_to_Pos'].to_numpy(dtype=float)
    dist_neg_084 = score_pair_df['Dist_to_Neg'].to_numpy(dtype=float)
    finite_pos_084 = np.isfinite(dist_pos_084)
    finite_neg_084 = np.isfinite(dist_neg_084)
    nearest_state_084 = np.full(len(score_pair_df), 'Unassigned', dtype=object)
    nearest_state_084[finite_pos_084 & (~finite_neg_084 | (dist_pos_084 <= dist_neg_084))] = 'Near DR(+) B cell'
    nearest_state_084[finite_neg_084 & (~finite_pos_084 | (dist_neg_084 < dist_pos_084))] = 'Near DR(-) B cell'
    score_pair_df['Nearest_B_State'] = nearest_state_084
else:
    score_pair_df['Nearest_B_State'] = 'Unassigned'
score_pair_df = score_pair_df.dropna(subset=['Activation_Score', 'Exhaustion_Score']).copy()
if len(score_pair_df) == 0:
    raise ValueError('[Step 08-2] No T cells have valid values for both Activation and Exhaustion scores.')
_score_tissues_084 = tuple(sorted(score_pair_df['Run_Tissue_name'].unique()))
_expected_tissues_084 = tuple(sorted(STEP08_EXACT_DLBCL_TISSUES))
if _score_tissues_084 != _expected_tissues_084:
    raise RuntimeError(
        '[Step 08-2] exact tissue contract mismatch: '
        f'observed={_score_tissues_084}, expected={_expected_tissues_084}'
    )
_score_labels_084 = tuple(sorted(score_pair_df['T_Subset'].unique()))
_expected_labels_084 = tuple(sorted(_analysis_labels_084))
if _score_labels_084 != _expected_labels_084:
    raise RuntimeError(
        '[Step 08-2] exact T-cell label contract mismatch: '
        f'observed={_score_labels_084}, expected={_expected_labels_084}'
    )
if STEP08_MODE == 'full_run':
    STEP08_SCORE_TISSUE_COUNTS = (
        score_pair_df['Run_Tissue_name'].value_counts().sort_index().astype(int).to_dict()
    )
    STEP08_SCORE_SUBTYPE_COUNTS = (
        score_pair_df['T_Subset'].value_counts().reindex(_analysis_labels_084).astype(int).to_dict()
    )
    print(
        f'  ✅ [Step 08-2] exact 8-tissue T-cell cohort: {len(score_pair_df):,} cells; '
        f'subtypes={STEP08_SCORE_SUBTYPE_COUNTS}'
    )


def _safe_corr_084(x, y, method):
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = np.asarray(x[valid], dtype=float)
    y_valid = np.asarray(y[valid], dtype=float)
    if len(x_valid) < 3 or np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return np.nan, np.nan
    try:
        if method == 'pearson':
            r, p = stats.pearsonr(x_valid, y_valid)
        else:
            r, p = stats.spearmanr(x_valid, y_valid)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def _step084_plot_regression_line(ax, df_line, color, label=None, linewidth=2.4):
    """Safely add a linear regression line to an Activation/Exhaustion scatter plot."""
    valid_line = df_line[['Activation_Score', 'Exhaustion_Score']].dropna()
    if len(valid_line) < 3 or valid_line['Activation_Score'].nunique() < 2:
        return False
    slope, intercept, _, _, _ = stats.linregress(
        valid_line['Activation_Score'],
        valid_line['Exhaustion_Score']
    )
    x_line = np.linspace(valid_line['Activation_Score'].min(), valid_line['Activation_Score'].max(), 140)
    ax.plot(x_line, slope * x_line + intercept, color=color, lw=linewidth, label=label)
    return True


def _step084_make_corr_record(label, df_corr, extra=None):
    """Calculate Activation/Exhaustion correlations using the same schema."""
    pearson_r, pearson_p = _safe_corr_084(
        df_corr['Activation_Score'].to_numpy(),
        df_corr['Exhaustion_Score'].to_numpy(),
        method='pearson'
    )
    spearman_r, spearman_p = _safe_corr_084(
        df_corr['Activation_Score'].to_numpy(),
        df_corr['Exhaustion_Score'].to_numpy(),
        method='spearman'
    )
    record = {
        'T_Subset': label,
        'N_cells': int(len(df_corr)),
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_rho': spearman_r,
        'Spearman_p': spearman_p
    }
    if extra:
        record.update(extra)
    return record


def _step084_analysis_groups(df_corr):
    """Return subtype groups, optionally preceded by the aggregate All-T group."""
    observed = set(df_corr['T_Subset'].astype(str))
    ordered_labels = [label for label in STEP08_T_CELL_LABEL_ORDER if label in observed]
    ordered_labels.extend(sorted(observed.difference(ordered_labels)))
    groups = [
        (label, df_corr.loc[df_corr['T_Subset'].astype(str).eq(label)])
        for label in ordered_labels
    ]
    if not STEP08_DROP_ALL_T:
        groups.insert(0, ('All_T_cells', df_corr))
    return groups


def _step084_hide_outside_x_tick_labels(axis):
    """Hide locator ticks outside the plotted x limits so layout can use the full canvas."""
    x_min, x_max = axis.get_xlim()
    tolerance = max(abs(x_max - x_min), 1.0) * 1e-9
    for tick_value, tick_label in zip(axis.get_xticks(), axis.get_xticklabels()):
        tick_label.set_visible(
            x_min - tolerance <= float(tick_value) <= x_max + tolerance
        )


def _step08_render_saved_score_correlation():
    """Render Step08-03 from frozen scores and the saved correlation result table."""
    if not _final_figure_stem_registered(STEP08_SCORE_CORRELATION_BY_CELLTYPE_FIG):
        raise RuntimeError(
            'Step08 canonical correlation stem is not registered: '
            f'{STEP08_SCORE_CORRELATION_BY_CELLTYPE_FIG}'
        )
    corr_path = _step08_table_path(STEP08_SCORE_CORRELATION_CSV)
    if not os.path.isfile(corr_path) or os.path.getsize(corr_path) <= 0:
        raise FileNotFoundError(
            f'[Step 08-03] canonical saved correlation table is missing: {corr_path}'
        )
    corr_saved = pd.read_csv(corr_path)
    required = {'T_Subset', 'N_cells', 'Spearman_rho', 'Spearman_q_BH_FDR'}
    missing = sorted(required - set(corr_saved.columns))
    if missing:
        raise RuntimeError(
            f'[Step 08-03] saved correlation table lacks columns {missing}: {corr_path}'
        )
    corr_saved['T_Subset'] = corr_saved['T_Subset'].astype(str)
    available = set(corr_saved['T_Subset'])
    subtype_order = [
        label for label in STEP08_T_CELL_LABEL_ORDER
        if label in available and label in set(score_pair_df['T_Subset'].astype(str))
    ]
    panel_groups = [
        (
            {'CD4_CTL': 'CD4 CTL', 'CD8T': 'CD8 T'}.get(label, label),
            label,
            score_pair_df.loc[score_pair_df['T_Subset'].astype(str).eq(label)],
        )
        for label in subtype_order
    ]
    if not STEP08_DROP_ALL_T:
        panel_groups.insert(0, ('All T cells', 'All_T_cells', score_pair_df))
    required_result_labels = {result_label for _, result_label, _ in panel_groups}
    missing_results = sorted(required_result_labels.difference(available))
    if missing_results:
        raise RuntimeError(
            f'[Step 08-03] saved correlation table lacks results: {missing_results}'
        )
    n_columns = min(2, len(panel_groups))
    n_rows = math.ceil(len(panel_groups) / n_columns)
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=STEP08_MAIN_FIGSIZES[STEP08_SCORE_CORRELATION_BY_CELLTYPE_FIG],
        squeeze=False,
    )
    axes_flat = axes.flatten()
    random_seed = RANDOM_SEED if 'RANDOM_SEED' in globals() else 42
    for axis_index, (display_label, result_label, panel_frame) in enumerate(panel_groups):
        axis = axes_flat[axis_index]
        if len(panel_frame) > STEP08_SCORE_CORR_MAX_CELLS_PER_TYPE:
            plot_frame = panel_frame.sample(
                n=STEP08_SCORE_CORR_MAX_CELLS_PER_TYPE, random_state=random_seed,
            )
        else:
            plot_frame = panel_frame
        plotted = pd.Series(False, index=plot_frame.index)
        for state_label, state_color in STEP08_SCORE_CORR_B_STATE_COLORS.items():
            state_frame = plot_frame.loc[
                plot_frame['Nearest_B_State'].astype(str).eq(state_label)
            ]
            if state_frame.empty:
                continue
            plotted.loc[state_frame.index] = True
            axis.scatter(
                state_frame['Activation_Score'], state_frame['Exhaustion_Score'],
                s=1.7, alpha=0.12, color=state_color, rasterized=True,
            )
        unassigned = plot_frame.loc[~plotted]
        if not unassigned.empty:
            axis.scatter(
                unassigned['Activation_Score'], unassigned['Exhaustion_Score'],
                s=1.5, alpha=0.08, color='#7f8c8d', rasterized=True,
            )
        for state_label, state_color in STEP08_SCORE_CORR_B_STATE_COLORS.items():
            state_frame = panel_frame.loc[
                panel_frame['Nearest_B_State'].astype(str).eq(state_label)
            ]
            _step084_plot_regression_line(
                axis, state_frame, state_color,
                label=state_label, linewidth=1.35,
            )
        result = corr_saved.loc[corr_saved['T_Subset'].eq(result_label)]
        if len(result) != 1:
            raise RuntimeError(
                f'[Step 08-03] expected one saved correlation row for {result_label}; '
                f'found {len(result)}'
            )
        result = result.iloc[0]
        title_color = STEP08_T_SUBTYPE_TITLE_COLORS.get(result_label, '#222222')
        axis.set_title(
            f"{display_label} (N={int(result['N_cells']):,})\n"
            f"q={float(result['Spearman_q_BH_FDR']):.1e}",
            fontsize=8.4, fontweight='bold', color=title_color, pad=5.0,
        )
        axis.text(
            0.035, 0.955,
            f"ρ={float(result['Spearman_rho']):.2f}",
            transform=axis.transAxes,
            ha='left', va='top', fontsize=8.2, fontweight='bold',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 1.2},
            zorder=10,
        )
        axis.set_xlabel('')
        axis.set_ylabel('')
        axis.tick_params(axis='both', labelsize=7.2)
        axis.grid(True, linestyle='--', alpha=0.25)
        _step084_hide_outside_x_tick_labels(axis)
    for axis in axes_flat[len(panel_groups):]:
        axis.axis('off')
    legend_handles = [
        mlines.Line2D(
            [], [], color=color, linewidth=1.3, marker='o', markersize=3.5,
            markerfacecolor=color, markeredgecolor='none',
            label=STEP08_SCORE_CORR_B_STATE_DISPLAY.get(label, label),
        )
        for label, color in STEP08_SCORE_CORR_B_STATE_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles, title='Nearest B-cell state', loc='lower center',
        bbox_to_anchor=(0.5, 0.002), ncol=2, frameon=False,
        fontsize=7.5, title_fontsize=7.9,
    )
    fig.supxlabel(
        'Activation score', x=0.5, y=0.080,
        fontsize=8.4, fontweight='bold',
    )
    fig.supylabel(
        'Exhaustion score', x=0.018, y=0.510,
        fontsize=8.4, fontweight='bold',
    )
    fig.subplots_adjust(
        left=0.105, right=0.965, bottom=0.135, top=0.910,
        wspace=0.25, hspace=0.48,
    )
    return _save_step08_matplotlib(
        fig, STEP08_SCORE_CORRELATION_BY_CELLTYPE_FIG,
        'Step08-03 saved activation/exhaustion correlation', bbox_inches=None,
    )


if STEP08_FIGURE_ONLY:
    _step08_saved_corr_report = _step08_render_saved_score_correlation()
    STEP08_FIGURE_ONLY_SAVED_REPORTS.append(_step08_saved_corr_report)
    _step08_saved_count = sum(
        report is not None for report in STEP08_FIGURE_ONLY_SAVED_REPORTS
    )
    if _step08_saved_count != len(STEP08_EXPORT_STATIC_FIGURES):
        raise RuntimeError(
            'Step08 active Figure family was not saved completely: '
            f'{_step08_saved_count}/{len(STEP08_EXPORT_STATIC_FIGURES)} pairs'
        )
    print(
        '✅ Step08 Figure-only saved-result workflow: '
        f'saved_pairs={_step08_saved_count}'
    )
    raise SystemExit(0)


corr_rows = []
corr_groups = _step084_analysis_groups(score_pair_df)
for label, df_corr in corr_groups:
    corr_rows.append(_step084_make_corr_record(label, df_corr))

corr_df = pd.DataFrame(corr_rows)
corr_df['Pearson_q_BH_FDR'] = _bh_fdr(corr_df['Pearson_p'].to_numpy())
corr_df['Spearman_q_BH_FDR'] = _bh_fdr(corr_df['Spearman_p'].to_numpy())
corr_path = _step08_table_path(STEP08_SCORE_CORRELATION_CSV)
_step08_write_csv(corr_df, corr_path, index=False)
print(f"  ✅ T-cell-type-specific Activation/Exhaustion correlation CSV saved: {corr_path}")

subtype_counts = score_pair_df['T_Subset'].value_counts()
subtype_order = [
    label for label in STEP08_T_CELL_LABEL_ORDER
    if label in set(score_pair_df['T_Subset'].astype(str))
]
subtype_order.extend([
    label for label in sorted(set(score_pair_df['T_Subset'].astype(str)))
    if label not in subtype_order
])
panel_groups_084 = [
    (
        {'CD4_CTL': 'CD4 CTL', 'CD8T': 'CD8 T'}.get(str(t_sub), str(t_sub)),
        str(t_sub),
        score_pair_df[score_pair_df['T_Subset'].astype(str).eq(str(t_sub))],
    )
    for t_sub in subtype_order
]
if not STEP08_DROP_ALL_T:
    panel_groups_084.insert(0, ('All T cells', 'All_T_cells', score_pair_df))
if panel_groups_084:
    n_cols = min(2, len(panel_groups_084))
    n_rows = math.ceil(len(panel_groups_084) / n_cols)
    fig_ct, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=STEP08_MAIN_FIGSIZES[STEP08_SCORE_CORRELATION_BY_CELLTYPE_FIG],
        squeeze=False
    )
    axes_flat = axes.flatten()
    rng_seed = RANDOM_SEED if 'RANDOM_SEED' in globals() else 42

    for ax_idx, (display_label, corr_label, panel_df) in enumerate(panel_groups_084):
        ax = axes_flat[ax_idx]
        if len(panel_df) > STEP08_SCORE_CORR_MAX_CELLS_PER_TYPE:
            sub_plot = panel_df.sample(n=STEP08_SCORE_CORR_MAX_CELLS_PER_TYPE, random_state=rng_seed)
        else:
            sub_plot = panel_df
        plotted_state_mask = pd.Series(False, index=sub_plot.index)
        for state_label, state_color in STEP08_SCORE_CORR_B_STATE_COLORS.items():
            state_plot = sub_plot[sub_plot['Nearest_B_State'].astype(str).eq(state_label)]
            if len(state_plot) == 0:
                continue
            plotted_state_mask.loc[state_plot.index] = True
            ax.scatter(
                state_plot['Activation_Score'],
                state_plot['Exhaustion_Score'],
                s=1.7,
                alpha=0.12,
                color=state_color,
                rasterized=True
            )
        unassigned_plot = sub_plot.loc[~plotted_state_mask]
        if len(unassigned_plot) > 0:
            ax.scatter(
                unassigned_plot['Activation_Score'],
                unassigned_plot['Exhaustion_Score'],
                s=1.5,
                alpha=0.08,
                color='#7f8c8d',
                rasterized=True
            )
        for state_label, state_color in STEP08_SCORE_CORR_B_STATE_COLORS.items():
            state_df = panel_df[panel_df['Nearest_B_State'].astype(str) == state_label]
            _step084_plot_regression_line(ax, state_df, state_color, label=state_label, linewidth=1.35)
        corr_rec = corr_df[corr_df['T_Subset'] == corr_label].iloc[0]
        title_color_084 = STEP08_T_SUBTYPE_TITLE_COLORS.get(corr_label, '#222222')
        ax.set_title(
            f"{display_label} (N={int(corr_rec['N_cells']):,})\n"
            f"q={corr_rec['Spearman_q_BH_FDR']:.1e}",
            fontsize=8.0,
            fontweight='bold',
            color=title_color_084,
            pad=5.0,
        )
        ax.text(
            0.035, 0.955,
            f"ρ={corr_rec['Spearman_rho']:.2f}",
            transform=ax.transAxes,
            ha='left', va='top', fontsize=8.2, fontweight='bold',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 1.2},
            zorder=10,
        )
        ax.set_xlabel('')
        ax.set_ylabel('')
        if ax_idx % n_cols != 0:
            ax.tick_params(axis='y', labelleft=True, pad=2)
        ax.tick_params(axis='both', labelsize=7.3)
        ax.grid(True, linestyle='--', alpha=0.25)
        _step084_hide_outside_x_tick_labels(ax)

    for ax in axes_flat[len(panel_groups_084):]:
        ax.axis('off')

    line_handles_084 = [
        mlines.Line2D(
            [],
            [],
            color=color,
            lw=1.3,
            marker='o',
            markersize=3.0,
            markerfacecolor=color,
            markeredgecolor='none',
            label=STEP08_SCORE_CORR_B_STATE_DISPLAY.get(label, label)
        )
        for label, color in STEP08_SCORE_CORR_B_STATE_COLORS.items()
    ]
    fig_ct.legend(
        handles=line_handles_084,
        title='Nearest B-cell state',
        loc='lower center',
        bbox_to_anchor=(0.5, 0.002),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        title_fontsize=7.9,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.4,
        columnspacing=0.8
    )
    fig_ct.supxlabel(
        'Activation score', x=0.5, y=0.080,
        fontsize=8.4, fontweight='bold',
    )
    fig_ct.supylabel(
        'Exhaustion score', x=0.018, y=0.510,
        fontsize=8.4, fontweight='bold',
    )
    fig_ct.subplots_adjust(
        left=0.105, right=0.965, bottom=0.135, top=0.910,
        wspace=0.25, hspace=0.48,
    )
    corr_ct_path = _save_step08_matplotlib(
        fig_ct,
        STEP08_SCORE_CORRELATION_BY_CELLTYPE_FIG,
        "Step08-2 activation/exhaustion correlation by T cell type",
        bbox_inches=None
    )
    if corr_ct_path is not None:
        plt.show()
        print(f"  ✅ T-cell-type-specific correlation figure saved: {corr_ct_path}")

# The active Step08 scope ends at canonical Step08-03.
# Optional distance-, motif-, and tissue-stratified renderers are intentionally excluded.

print('✅ [Step 08-2] Activation/Exhaustion correlation analysis complete!')

if STEP08_MODE == 'full_run':
    _analysis_table_names_084 = [
        STEP08_RL_LONGFORM_CSV,
        STEP08_RL_SUMMARY_CSV,
        STEP08_SCORE_CORRELATION_CSV,
    ]
    _analysis_table_paths_084 = [
        os.path.realpath(_step08_table_path(name)) for name in _analysis_table_names_084
    ]
    _missing_tables_084 = [path for path in _analysis_table_paths_084 if not os.path.isfile(path)]
    if _missing_tables_084:
        raise RuntimeError(f'[Step 08] full-run analysis tables are incomplete: {_missing_tables_084}')

    _figure_stems_084 = [
        'Final_mv1_Step08-01_RL_Interaction_Heatmap',
        'Final_mv1_Step08-02_RL_Grouped_By_CellCellPair_AllSignificant',
        'Final_mv1_Step08-03_Activation_Exhaustion_Correlation_ByTcellType',
    ]
    _figure_paths_084 = [
        os.path.realpath(os.path.join(STEP08_RUN_FIG_DIR, f'{stem}.{suffix}'))
        for stem in _figure_stems_084
        for suffix in ('png', 'pptx')
    ]
    _missing_figures_084 = [path for path in _figure_paths_084 if not os.path.isfile(path)]
    if _missing_figures_084:
        raise RuntimeError(f'[Step 08] full-run figure bundle is incomplete: {_missing_figures_084}')

    _all_t_tables_084 = []
    for _table_path_084 in _analysis_table_paths_084[2:]:
        _frame_084 = pd.read_csv(_table_path_084)
        if 'T_Subset' in _frame_084.columns and _frame_084['T_Subset'].astype(str).eq('All_T_cells').any():
            _all_t_tables_084.append(_table_path_084)
    if STEP08_DROP_ALL_T and _all_t_tables_084:
        raise RuntimeError(f'[Step 08] All_T_cells survived the drop contract: {_all_t_tables_084}')
    if STEP08_INCLUDE_TPR:
        _missing_tpr_tables_084 = []
        for _table_path_084 in _analysis_table_paths_084[2:]:
            _frame_084 = pd.read_csv(_table_path_084)
            if 'T_Subset' in _frame_084.columns and not _frame_084['T_Subset'].astype(str).eq('Tpr').any():
                _missing_tpr_tables_084.append(_table_path_084)
        if _missing_tpr_tables_084:
            raise RuntimeError(f'[Step 08] Tpr is missing from subtype tables: {_missing_tpr_tables_084}')

    _count_rows_084 = []
    for _branch_084, _counts_084 in [
        ('R-L proximity cohort', STEP08_RL_TISSUE_COUNTS),
        ('Activation/exhaustion score cohort', STEP08_SCORE_TISSUE_COUNTS),
    ]:
        for _tissue_084 in STEP08_EXACT_DLBCL_TISSUES:
            _count_rows_084.append({
                'Branch': _branch_084,
                'Run_Tissue_name': _tissue_084,
                'N_cells': int(_counts_084[_tissue_084]),
            })
    _count_path_084 = _step08_table_path('Step08_Tissue_Cell_Counts.csv')
    _step08_write_csv(pd.DataFrame(_count_rows_084), _count_path_084, index=False)

    _protected_changed_084 = []
    for _protected_path_084, _before_hash_084 in STEP08_PROTECTED_BASELINE.items():
        if not os.path.isfile(_protected_path_084):
            _protected_changed_084.append((_protected_path_084, _before_hash_084, None))
            continue
        _after_hash_084 = _step08_sha256(_protected_path_084)
        if _after_hash_084 != _before_hash_084:
            _protected_changed_084.append(
                (_protected_path_084, _before_hash_084, _after_hash_084)
            )
    if _protected_changed_084:
        raise RuntimeError(f'[Step 08] protected canonical artifacts changed: {_protected_changed_084}')

    _hash_subjects_084 = sorted(set(
        _analysis_table_paths_084
        + [os.path.realpath(_count_path_084)]
        + _figure_paths_084
    ))
    _hash_rows_084 = [{
        'Relative_Path': os.path.relpath(path, FINAL_WORKSPACE_DIR),
        'Size_Bytes': int(os.path.getsize(path)),
        'SHA256': _step08_sha256(path),
    } for path in _hash_subjects_084]
    _hash_path_084 = _step08_table_path('Step08_Output_Hashes.csv')
    _step08_write_csv(pd.DataFrame(_hash_rows_084), _hash_path_084, index=False)

    _manifest_084 = {
        'status': 'complete',
        'run_id': STEP08_RUN_ID,
        'mode': STEP08_MODE,
        'cohort': STEP08_COHORT,
        'exact_tissues': list(STEP08_EXACT_DLBCL_TISSUES),
        'include_DLBCL1_02': True,
        'include_Tpr': bool(STEP08_INCLUDE_TPR),
        'drop_All_T_cells': bool(STEP08_DROP_ALL_T),
        'parameters': {
            'spatial_radius_um': 50.0,
            'coordinate_um_per_unit': STEP08_COORD_UM_PER_UNIT,
            'n_permutations': STEP08_N_PERM,
            'minimum_neighbor_pairs': STEP08_MIN_NEIGHBOR_PAIRS,
            'random_seed': int(globals().get('RANDOM_SEED', 42)),
            'recompute_T_to_B_distance': bool(STEP08_RECOMPUTE_TB_DISTANCE),
        },
        'gpu': {
            'required': bool(STEP08_REQUIRE_GPU),
            'available': bool(STEP08_GPU_AVAILABLE),
            'disabled_reason': STEP08_GPU_DISABLED_REASON,
        },
        'input_hashes': {
            os.path.relpath(path, FINAL_WORKSPACE_DIR): digest
            for path, digest in STEP08_PROTECTED_BASELINE.items()
            if path == os.path.realpath(CHECKPOINT_STEP06)
        },
        'cohort_counts': {
            'rl_by_tissue': STEP08_RL_TISSUE_COUNTS,
            'score_by_tissue': STEP08_SCORE_TISSUE_COUNTS,
            'score_by_subtype': STEP08_SCORE_SUBTYPE_COUNTS,
        },
        'outputs': [row['Relative_Path'] for row in _hash_rows_084],
        'output_hash_table': os.path.relpath(_hash_path_084, FINAL_WORKSPACE_DIR),
        'protected_canonical_artifacts_checked': len(STEP08_PROTECTED_BASELINE),
        'protected_canonical_artifacts_unchanged': True,
        'elapsed_seconds': round(time.time() - STEP08_RUN_STARTED, 3),
        'figure_editability_contract': (
            'PNG plus native-shape PPTX; dense Step08-03 scatter clouds remain '
            'raster layers while axes, lines, and annotations remain editable shapes.'
        ),
    }
    _manifest_path_084 = _step08_table_path('Step08_Run_Manifest.json')
    _step08_write_json(_manifest_084, _manifest_path_084)
    print(
        f'✅ [Step 08] isolated full-run bundle complete: {STEP08_RUN_ID} '
        f'({len(_analysis_table_paths_084)} analysis tables, '
        f'{len(_figure_paths_084)} figure files)'
    )
