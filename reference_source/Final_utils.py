# %%
# ==============================================================================
# [Step 00-1] Load required libraries (Packages)
# ==============================================================================
import os, gc, time, warnings, sys, pickle, joblib, subprocess, tempfile, math, colorsys, builtins, uuid, zipfile, re
from contextlib import contextmanager
from pathlib import Path as _FinalPath
from xml.etree import ElementTree as _FinalElementTree

# Set the Final-local cache path before importing packages to prevent numba/scanpy
# source-locator cache failures in the SpaceFlow Python 3.13 environment.
_FINAL_CACHE_ROOT = '__ROIDER_ROOT__/Analysis/python/Final/environment/cache'
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(_FINAL_CACHE_ROOT, 'numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_FINAL_CACHE_ROOT, 'mpl_cache'))
os.environ.setdefault('XDG_CACHE_HOME', os.path.join(_FINAL_CACHE_ROOT, 'xdg_cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')
for _cache_path in (os.environ['NUMBA_CACHE_DIR'], os.environ['MPLCONFIGDIR'], os.environ['XDG_CACHE_HOME']):
    os.makedirs(_cache_path, exist_ok=True)

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import scipy.stats as stats
from scipy.spatial import cKDTree
from scipy.stats import (
    chisquare, f_oneway, hypergeom, mannwhitneyu, spearmanr, zscore
)
from tqdm.auto import tqdm
from collections import Counter, defaultdict
from itertools import combinations as _combs

# ML & Metrics (CPU/Sklearn)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, recall_score, f1_score,
    silhouette_score, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import NMF

# GPU acceleration libraries
# GPU/RAPIDS imports can delay kernel startup by nearly a minute during Figure/table export.
# These export scripts set FINAL_LIGHT_IMPORTS=1 in the first cell to skip heavy imports.
FINAL_LIGHT_IMPORTS = os.environ.get('FINAL_LIGHT_IMPORTS', '0') == '1'
if not FINAL_LIGHT_IMPORTS:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import cupy as cp
    import xgboost as xgb
    import rapids_singlecell as rsc
    from cuml.cluster import KMeans as cuKMeans
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.neighbors import NearestNeighbors as cuNN
else:
    torch = nn = F = cp = xgb = rsc = cuKMeans = cuRF = cuNN = None

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import IPython
import IPython.display as display
from editable_pptx import (
    validate_final_figure_canvas,
    _validate_staged_final_png,
    _validate_staged_final_pptx,
    _new_final_pair_transaction_paths,
    _promote_staged_final_pair,
    _register_matplotlib_arial_fonts,
    _validate_matplotlib_arial_font_contract,
    save_matplotlib_figure_as_editable_pptx,
)

# Final v2 is a batch Figure export workspace. Disable GUI show calls unrelated to
# notebook inline display or GPU computation by default; restore them via the environment when needed.
FINAL_SUPPRESS_INTERACTIVE_SHOW = os.environ.get('FINAL_SUPPRESS_INTERACTIVE_SHOW', '1') != '0'
if FINAL_SUPPRESS_INTERACTIVE_SHOW:
    plt.ioff()
    plt.show = lambda *args, **kwargs: None

print("✅ [Step 00-1] Libraries loaded")

# %%
# ==============================================================================
# [Step 00-2] Define pipeline utility functions
# Contents: distances, label mapping, sampling, palettes, KNN guardrails,
#           MLP spawn helpers, tissue naming, and feature importance logging
# ==============================================================================

# ── 1. Basic distance and lineage mapping functions ───────────────────────────
SPILLOVER_RADIUS_PX = 20

def spillover_weight(distances):
    """Compute distance-based spillover weights."""
    weights = np.exp(-(distances ** 2) / (2 * (SPILLOVER_RADIUS_PX * 0.5) ** 2))
    weights[distances > SPILLOVER_RADIUS_PX] = 0.0001
    return weights

def map_subtype(x):
    """Map detailed cell subtypes."""
    s = str(x).strip()
    if s == 'na': return 'na'
    if s in ['CD8T', 'TTOX', 'CD8', 'TTOXNaive', 'TTOX_exh']: return 'CD8T'
    if s in ['Th', 'CD4T', 'CD4TNaive', 'T_H', 'T_H_Naive']: return 'Th'
    if s in ['Tfh', 'TFH', 'T_FH']: return 'Tfh'
    if s in ['Treg', 'T_REG']: return 'Treg'
    if s in ['Tpr', 'TPR', 'T_Pr', 'T_pr']: return 'Tpr'
    if s in ['CD4_CTL', 'CTL']: return 'CD4_CTL'
    if s in ['B', 'B cells']: return 'B cell'
    if s in ['PC', 'Plasma cells']: return 'Plasma cell'
    if s in ['NK', 'NKT', 'NK cells', 'NK/NKT']: return 'NK cell'
    if s in ['Macro', 'Macrophages', 'Histiocytes']: return 'Macrophage'
    if s in ['DC', 'Dendritic cells']: return 'DC'
    if s in ['Stromal', 'Stromal cells']: return 'Stromal'
    if s in ['FDC']: return 'FDC'
    return 'Myeloid'

def refine_broad_for_subtyping(x):
    """Convert subtypes to broad lineages."""
    s = str(x).strip()
    if s in ['na', 'Unknown']: return s
    if s in ['CD8T', 'Th', 'Tfh', 'Treg', 'Tpr', 'CD4_CTL']: return 'T cells'
    if s in ['B cell', 'Plasma cell']: return 'B cells'
    if s in ['NK cell']: return 'NK cells'
    if s in ['Stromal', 'FDC']: return 'Stromal cells'
    return 'Myeloid cells'

def map_broad(x):
    """Convert source labels directly to broad lineages."""
    return refine_broad_for_subtyping(map_subtype(x))

# ── 2. Data sampling functions ───────────────────────────────────────────────
def stratified_sample(X, y, n_samples_per_class, random_state=42):
    """Perform stratified sampling for class-balanced training."""
    rng = np.random.default_rng(random_state)
    unique_y = np.unique(y)
    indices = []
    for cls in unique_y:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) == 0: continue
        replace = len(cls_indices) < n_samples_per_class
        s_idx = rng.choice(cls_indices, n_samples_per_class, replace=replace)
        indices.extend(s_idx)
    return np.array(indices)

def dynamic_stratified_sample(X, y_enc, max_cells_per_class=50_000, random_state=42):
    """Sample up to N observations per class (subtypes only)."""
    rng = np.random.default_rng(random_state)
    unique_classes, counts = np.unique(y_enc, return_counts=True)
    sampled_indices = []
    if len(unique_classes) == 0: return np.array([])
    for cls, count in zip(unique_classes, counts):
        target = min(count, max_cells_per_class)
        cls_idx = np.where(y_enc == cls)[0]
        sampled_indices.extend(rng.choice(cls_idx, target, replace=False))
    return np.array(sampled_indices)

# ── 3. Global color palette (consistent across all Figures) ──────────────────
LINEAGE_COLORS = {
    'T cells': '#C0392B', 'B cells': '#2471A3', 'NK cells': '#D4AC0D',
    'Myeloid cells': '#6C3483', 'Stromal cells': '#27AE60',
    'Unknown': '#95A5A6', 'na': '#BDC3C7'
}

CELL_COLORS = {
    'T cells': '#C0392B', 'CD8T': '#E74C3C', 'Th': '#A93226',
    'Tfh': '#F1948A', 'Treg': '#922B21', 'Tpr': '#D98880', 'CD4_CTL': '#FF5733',
    'B cells': '#2471A3', 'B cell': '#2E86C1', 'Plasma cell': '#1B4F72',
    'NK cells': '#D4AC0D', 'NK cell': '#F1C40F',
    'Myeloid cells': '#6C3483', 'Myeloid': '#7D3C98',
    'Macrophage': '#8E44AD', 'DC': '#A569BD',
    'Stromal cells': '#27AE60', 'Stromal': '#58D68D', 'FDC': '#ABEBC6',
    'Unknown': '#95A5A6', 'na': '#BDC3C7'
}

# Disease display order (fixed across visualizations)
DISEASE_ORDER = ['DLBCL', 'FL', 'MZL', 'MCL', 'rLN']

def get_color(cell_type):
    return CELL_COLORS.get(str(cell_type).strip(), '#AAAAAA')

def get_color_list(cell_type_series):
    return [get_color(ct) for ct in cell_type_series]

# ── 4. Marker-aware KNN guardrail (spillover prevention) ──────────────────────
_LINEAGE_GROUP = {
    'CD8T':'T','Th':'T','Tfh':'T','Treg':'T','Tpr':'T','CD4_CTL':'T','T cells':'T',
    'B cell':'B','Plasma cell':'B','B cells':'B',
    'NK cell':'NK','NK cells':'NK',
    'Macrophage':'M','DC':'M','Myeloid':'M','Myeloid cells':'M',
    'Stromal':'S','FDC':'S','Stromal cells':'S'
}
_LINEAGE_MARKERS = {'T':'CD3', 'B':'CD20', 'NK':'CD56', 'M':'CD68', 'S':'PDPN'}
SPILLOVER_MARKER_THRESH = 1.0

def knn_lineage_guard(winner, orig, cell_expr=None, marker_col_idx=None):
    orig_s, winner_s = str(orig).strip(), str(winner).strip()
    if orig_s in ('Unknown', 'na') or winner_s in ('Unknown', 'na'): return True
    orig_g, win_g = _LINEAGE_GROUP.get(orig_s, 'X'), _LINEAGE_GROUP.get(winner_s, 'Y')
    if orig_g == win_g: return True
    if cell_expr is not None and marker_col_idx is not None:
        m_name = _LINEAGE_MARKERS.get(orig_g)
        if m_name in marker_col_idx:
            if float(cell_expr[marker_col_idx[m_name]]) > SPILLOVER_MARKER_THRESH: return False
            return True
    return False

# ── 5. Helper for isolated MLP GPU execution via spawn ────────────────────────
def spawn_mlp(
    X_train, y_train, X_test, n_classes, model_output_path=None,
    random_seed=42
):
    """Isolate MLP training/prediction in an external process and optionally save the model."""
    tmp_dir = tempfile.gettempdir()
    in_file = os.path.join(tmp_dir, f'mlp_in_{os.getpid()}_{np.random.randint(1000)}.npz')
    out_file = os.path.join(tmp_dir, f'mlp_out_{os.getpid()}_{np.random.randint(1000)}.npy')
    try:
        np.savez(in_file, X_train=X_train, y_train=y_train, X_test=X_test)
        local_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else ''
        script_candidates = [
            os.path.join(local_dir, '_mlp_subprocess_mv1.py'),
            os.path.join(globals().get('FINAL_WORKSPACE_DIR', '__ROIDER_ROOT__/Analysis/python/Final'),
                         'scripts', '_mlp_subprocess_mv1.py'),
        ]
        script_path = next((p for p in script_candidates if os.path.exists(p)), script_candidates[-1])
        command = ['python3', script_path, in_file, out_file, str(n_classes)]
        if model_output_path is not None:
            model_output_path = os.fspath(model_output_path)
            os.makedirs(os.path.dirname(os.path.abspath(model_output_path)), exist_ok=True)
        command.extend([
            model_output_path if model_output_path is not None else '__NONE__',
            str(int(random_seed)),
        ])
        subprocess.run(command, check=True)
        preds = np.load(out_file)
    finally:
        if os.path.exists(in_file): os.remove(in_file)
        if os.path.exists(out_file): os.remove(out_file)
    return preds

def spawn_mlp_predict_batched(
    X_train, y_train, X_pred_full, n_classes, chunk_size=None,
    model_output_path=None, random_seed=42
):
    """Wrap MLP spawn; one call suffices because the internal script handles batching."""
    return spawn_mlp(
        X_train, y_train, X_pred_full, n_classes,
        model_output_path=model_output_path,
        random_seed=random_seed
    )

# ── 6. Run_Tissue_name conversion ────────────────────────────────────────────
def rename_tissue_samples(obs_df):
    """
    Convert Run_Tissue_name to a compact form.
    Rules:
      - Number PatientID values within each Disease group (DLBCL1, DLBCL2, ...).
      - Number samples from the same PatientID with _01, _02, etc.
      - Example: 'DLBCL, non-GCB 191_3reg001' → 'DLBCL1_01'
    """
    obs_df = obs_df.copy()
    new_names = obs_df['Run_Tissue_name'].copy()

    # Map disease abbreviations.
    disease_abbr = {}
    for d in obs_df['Disease'].unique():
        d_str = str(d).strip()
        if 'DLBCL' in d_str: disease_abbr[d] = 'DLBCL'
        elif 'FL' in d_str: disease_abbr[d] = 'FL'
        elif 'MZL' in d_str: disease_abbr[d] = 'MZL'
        elif 'MCL' in d_str: disease_abbr[d] = 'MCL'
        elif 'rLN' in d_str or 'RLN' in d_str or 'LN' in d_str: disease_abbr[d] = 'rLN'
        else: disease_abbr[d] = d_str[:4]

    # Process each disease.
    for disease in obs_df['Disease'].unique():
        abbr = disease_abbr[disease]
        d_mask = obs_df['Disease'] == disease
        patients = obs_df.loc[d_mask, 'PatientID'].unique()

        # Assign numbers by PatientID.
        patient_num = {pid: idx + 1 for idx, pid in enumerate(sorted(patients))}

        for pid in patients:
            p_mask = d_mask & (obs_df['PatientID'] == pid)
            tissues = obs_df.loc[p_mask, 'Run_Tissue_name'].unique()

            for sample_idx, tissue in enumerate(sorted(tissues), 1):
                t_mask = obs_df['Run_Tissue_name'] == tissue
                new_name = f"{abbr}{patient_num[pid]}_{sample_idx:02d}"
                new_names[t_mask] = new_name

    obs_df['Run_Tissue_name'] = new_names
    # Also abbreviate Disease for categorical visualization.
    obs_df['Disease'] = obs_df['Disease'].map(disease_abbr)
    return obs_df

# ── 7. Feature importance logging utilities ─────────────────────────────────
def record_feature_importance(log_dict, model_name, feature_names, importance_values):
    """
    Record all feature importance values.
    Args:
        log_dict: Global store such as {'RF': [], 'XGB': [], ...}.
        model_name: Model name ('RF', 'XGB', 'ENet', 'MLP').
        feature_names: List of feature names.
        importance_values: Array of importance values.
    """
    record = {name: float(val) for name, val in zip(feature_names, importance_values)}
    log_dict[model_name].append(record)

def summarize_feature_importance(log_dict, feature_names, top_n=10):
    """Summarize how often each model selects each feature in its top N."""
    print("\n" + "=" * 70)
    print("📊 Feature importance summary (top-10 selection frequency)")
    print("=" * 70)
    for model_name, records in log_dict.items():
        if not records:
            continue
        n_iters = len(records)
        # Count appearances in the top 10 across iterations.
        top_counts = defaultdict(int)
        for rec in records:
            sorted_features = sorted(rec.items(), key=lambda x: abs(x[1]), reverse=True)
            for fname, _ in sorted_features[:top_n]:
                top_counts[fname] += 1
        # Sort by descending frequency.
        sorted_counts = sorted(top_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🔹 [{model_name}] ({n_iters} training iterations)")
        for fname, cnt in sorted_counts[:top_n]:
            print(f"    {fname:>15s}: {cnt:3d}/{n_iters} ({cnt/n_iters*100:.1f}%)")

# ── 8. Debugging helpers ────────────────────────────────────────────────────
def print_status(msg):
    """Print a debug message with GPU memory status."""
    used_mem = cp.get_default_memory_pool().used_bytes() / (1024**2)
    print(f"  🔍 [LOG] {msg} | CuPy VRAM: {used_mem:.1f} MB", flush=True)

print("✅ [Step 00-2] Utility functions loaded")

# %%
# ==============================================================================
# [Step 00-3] Environment and path configuration
# ==============================================================================
warnings.filterwarnings('ignore')
print("🔹 [Step 00-3] Initializing environment and paths...")

# 1. Font and visualization settings
# Final Figures use Arial only. Resolve the actual Regular/Bold faces again
# immediately before saving to prevent silent font substitution.
_register_matplotlib_arial_fonts()
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42

# 2. Path settings
BASE_DIR = '__ROIDER_ROOT__'
SOURCE_ANALYSIS_DIR = os.path.join(BASE_DIR, 'Analysis/python/V8')
V9_ANALYSIS_DIR = os.path.join(BASE_DIR, 'Analysis/python/V9')
FINAL_WORKSPACE_DIR = os.path.join(BASE_DIR, 'Analysis/python/Final')
# Place all new outputs from the relocated V9 code under the Final workspace.
ANALYSIS_DIR = FINAL_WORKSPACE_DIR
SPATIAL_DIR  = os.path.join(BASE_DIR, 'Multimodal_2024/Spatial_data')

# V8 is used only as the Step00 original-source fallback. Read completed V9 results
# from the local snapshot copied into Final/checkpoints.
SOURCE_CKPT_DIR = os.path.join(SOURCE_ANALYSIS_DIR, 'checkpoints/mv4')
V9_SOURCE_CKPT_DIR = os.path.join(V9_ANALYSIS_DIR, 'checkpoints/mv1')
FINAL_CHECKPOINT_ROOT = os.path.join(FINAL_WORKSPACE_DIR, 'checkpoints')
FINAL_LOCAL_CKPT_DIR = os.path.join(FINAL_CHECKPOINT_ROOT, 'mv1')
# Preserve legacy Step variable names while fixing references to the local snapshot.
V9_CKPT_DIR = FINAL_LOCAL_CKPT_DIR
CKPT_DIR = V9_CKPT_DIR
FINAL_TABLE_DIR = os.path.join(FINAL_WORKSPACE_DIR, 'tables/mv1')
# Figure-only mode blocks writes. Even if the guard is disabled, fix the checkpoint
# write root to the Final-local snapshot rather than the V9 source.
FINAL_CKPT_WRITE_DIR = FINAL_LOCAL_CKPT_DIR
FINAL_TRANSFER_ROOT = os.path.join(FINAL_CHECKPOINT_ROOT, 'transfer_models')
FINAL_SCRNA_CHECKPOINT_ROOT = os.path.join(FINAL_CHECKPOINT_ROOT, 'scRNA')
FIG_DIR = os.path.join(FINAL_WORKSPACE_DIR, 'figures/v2')
FINAL_FIGURE_ONLY_WRITE_GUARD = True
FINAL_FIGURE_EXTENSIONS = {'.png', '.pptx', '.svg', '.tif', '.tiff', '.jpg', '.jpeg', '.html'}
# Canonical Figure stem registry. Figure-only entry points render their complete
# active Figure set; individual revisions are reviewed under isolated temporary
# names and promoted only after approval. The registry remains a write-safety
# boundary and is not an execution-time target selector.
FINAL_FIGURE_STEM_ALIASES = {
    'Final_Figure1A_GraphicalAbstract': 'Final_Figure1A_GraphicalAbstract',
    'Final_mv1_Step01_Integrated_Summary_Vertical_Donuts_Final': 'Final_mv1_Step01-01_Integrated_Summary_Vertical_Donuts_Final',
    'Final_Figure2A-B_Broad_Lineage_Ensemble_Performance_and_Feature_Importance': 'Final_mv1_Step02-01_Broad_Lineage_Ensemble_Performance_and_Feature_Importance',
    'Final_Figure2C_A_Candidate_Confidence': 'Final_mv1_Step02-02_Candidate_Confidence',
    'Final_Figure2C_B_Prediction_Transition': 'Final_mv1_Step02-03_Prediction_Transition',
    'Final_Figure2C_D_Rescue_Decision': 'Final_mv1_Step02-04_Rescue_Decision',
    'Final_mv1_Step02-04_Broad_CM_Before_After': 'Final_mv1_Step02-05_Broad_CM_Before_After',
    'Final_mv1_Step02-04_Rescue_Changed_CM': 'Final_mv1_Step02-06_Rescue_Changed_CM',
    'Final_mv1_Step02-04_Rescue_Eval': 'Final_mv1_Step02-07_Rescue_Eval',
    'Final_mv1_Step02-04_Rescue_Example_01_B_cells_to_Myeloid_cells_DLBCL4_02': 'Final_mv1_Step02-08_Rescue_Example_01_B_cells_to_Myeloid_cells_DLBCL4_02',
    'Final_mv1_Step02-04_Rescue_Example_Combined_02_03': 'Final_mv1_Step02-09_Rescue_Example_Combined_02_03',
    'Final_mv1_Step02-Supp_OOB_PerLineage_BAcc_F1': 'Final_mv1_Step02-10_OOB_PerLineage_BAcc_F1',
    'DLBCL_Broad_Marker_Response': 'Final_mv1_Step02-01B_DLBCL_Broad_Marker_Response',
    'Final_mv1_Step03-06_CM_Baseline_Selected': 'Final_mv1_Step03-01_CM_Baseline_Selected',
    'Final_mv1_Step03-06_CM_Single_ML_Models': 'Final_mv1_Step03-02_CM_Single_ML_Models',
    'Final_mv1_Step03-06_Lineage_Performance_Compact': 'Final_mv1_Step03-03_Lineage_Performance_Compact',
    'Final_mv1_Step03-07_DLBCL_PerLineage_BAcc_F1': 'Final_mv1_Step03-04_DLBCL_PerLineage_BAcc_F1',
    'Final_mv1_Step04B-03A_CD4CTL_GMM_GRZB_Distribution': 'Final_mv1_Step04-01_CD4CTL_GMM_GRZB_Distribution',
    'Final_mv1_Step04B-03B_CD4CTL_GMM_High_Component_Posterior': 'Final_mv1_Step04-02_CD4CTL_GMM_High_Component_Posterior',
    'Final_mv1_Step04B-03C_CD4CTL_GMM_Cumulative_Composition': 'Final_mv1_Step04-03_CD4CTL_GMM_Cumulative_Composition',
    'Final_mv1_Step04B-03D_CD4CTL_GMM_Marker_Profile': 'Final_mv1_Step04-04_CD4CTL_GMM_Marker_Profile',
    'Final_mv1_Step04B-04-5_Combined_Composite': 'Final_mv1_Step04-05_Combined_Composite',
    'Final_mv1_Step04B-07_Subtype_Validation_Performance': 'Final_mv1_Step04-06_Subtype_Validation_Performance',
    'Final_mv1_Step04B-08_DLBCL_Benchmark_Final': 'Final_mv1_Step04-07_DLBCL_Benchmark_Final',
    'Final_mv1_Step04B-09_MarkerReliability_Composite': 'Final_mv1_Step04-08_MarkerReliability_Composite',
    'DLBCL_Subtype_Marker_Response': 'Final_mv1_Step04-08_MarkerReliability_Composite',
    'Final_Step04B_ALE_DLBCL_Staircase_Marker_Profiles_B_cells': 'Final_mv1_Step04-09_Purity_ColorSynced_B_cells',
    'Final_Step04B_ALE_DLBCL_Staircase_Marker_Profiles_Myeloid_cells': 'Final_mv1_Step04-10_Purity_ColorSynced_Myeloid_cells',
    'Final_Step04B_ALE_DLBCL_Staircase_Marker_Profiles_Stromal_cells': 'Final_mv1_Step04-11_Purity_ColorSynced_Stromal_cells',
    'Final_Step04B_ALE_DLBCL_Staircase_Marker_Profiles_T_cells': 'Final_mv1_Step04-12_Purity_ColorSynced_T_cells',
    'Final_mv1_Step04B-09_Purity_ColorSynced_B_cells': 'Final_mv1_Step04-09_Purity_ColorSynced_B_cells',
    'Final_mv1_Step04B-09_Purity_ColorSynced_Myeloid_cells': 'Final_mv1_Step04-10_Purity_ColorSynced_Myeloid_cells',
    'Final_mv1_Step04B-09_Purity_ColorSynced_Stromal_cells': 'Final_mv1_Step04-11_Purity_ColorSynced_Stromal_cells',
    'Final_mv1_Step04B-09_Purity_ColorSynced_T_cells': 'Final_mv1_Step04-12_Purity_ColorSynced_T_cells',
    'Final_mv1_Step04B-12_GT_vs_ML_Confusion_Heatmap': 'Final_mv1_Step04-13_GT_vs_ML_Confusion_Heatmap',
    'Final_MV1_Step062B_TtoB_Tcell_Distance_to_Bstate': 'Final_mv1_Step06-01_TtoB_Tcell_Distance_to_Bstate',
    'Final_MV1_Step062D_TtoB_Functional_Gradients': 'Final_mv1_Step06-02_TtoB_Functional_Gradients',
    'Final_MV1_Step065B_Tcell_Bstate_Median_Distance_Heatmap': 'Final_mv1_Step06-03_Tcell_Bstate_Median_Distance_Heatmap',
    'Final_MV1_Step065C_NearestOnly_Tcell_To_Bstate_Composition': 'Final_mv1_Step06-04_NearestOnly_Tcell_To_Bstate_Composition',
    'Final_MV1_Step07A_TcellAnchored_DRB_Motif_Composition': 'Final_mv1_Step07-01_TcellAnchored_DRB_Motif_Composition',
    'Final_MV1_Step07B_TcellAnchored_Tsubtype_Composition_ByMotif': 'Final_mv1_Step07-02_TcellAnchored_Tsubtype_Composition_ByMotif',
    'Final_MV1_Step07C_TcellAnchored_FunctionalScores_ByMotif': 'Final_mv1_Step07-03_TcellAnchored_FunctionalScores_ByMotif',
    'Final_MV1_Step07C2_TcellAnchored_MotifPairwiseStats_ByTSubtype': 'Final_mv1_Step07-04_TcellAnchored_MotifPairwiseStats_ByTSubtype',
    'Final_MV1_Step07D_TcellAnchored_FunctionalScore_Heatmap_BySubtype': 'Final_mv1_Step07-05_TcellAnchored_FunctionalScore_Heatmap_BySubtype',
    'Final_MV1_Step07G_TcellAnchored_DRB_Motif_TIFF_Multiscale_Map': 'Final_mv1_Step07-06_TcellAnchored_DRB_Motif_TIFF_Multiscale_Map',
    'Final_MV1_Step081_RL_Interaction_Heatmap': 'Final_mv1_Step08-01_RL_Interaction_Heatmap',
    'Final_MV1_Step081b_RL_Grouped_By_CellCellPair_AllSignificant': 'Final_mv1_Step08-02_RL_Grouped_By_CellCellPair_AllSignificant',
    'Final_MV1_Step082_Activation_Exhaustion_Correlation_ByTcellType': 'Final_mv1_Step08-03_Activation_Exhaustion_Correlation_ByTcellType',
    'GSE182434_marker_only_vs_dlbcl_ale_direction_weighted_performance': 'Final_mv1_Step12-01_GSE182434_marker_only_vs_dlbcl_ale_direction_weighted_performance',
    'GSE182434_marker_only_vs_dlbcl_ale_direction_weighted_3x3_umap_summary': 'Final_mv1_Step12-02_GSE182434_marker_only_vs_dlbcl_ale_direction_weighted_3x3_umap_summary',
    'Roider_marker_only_vs_dlbcl_ale_direction_weighted_performance': 'Final_mv1_Step12-03_Roider_marker_only_vs_dlbcl_ale_direction_weighted_performance',
    'Roider_marker_only_vs_dlbcl_ale_direction_weighted_3x3_umap_summary': 'Final_mv1_Step12-04_Roider_marker_only_vs_dlbcl_ale_direction_weighted_3x3_umap_summary',
    'GSE182434_marker_sign_importance_ablation_final_performance': 'Final_mv1_Step12-01_GSE182434_marker_sign_importance_ablation_performance',
    'Roider_marker_sign_importance_ablation_final_performance': 'Final_mv1_Step12-03_Roider_marker_sign_importance_ablation_performance',
    'GSE182434_marker_sign_importance_ablation_performance': 'Final_mv1_Step11B-01_GSE182434_marker_sign_importance_ablation_performance',
    'Roider_marker_sign_importance_ablation_performance': 'Final_mv1_Step11B-02_Roider_marker_sign_importance_ablation_performance',
    'scRNA_marker_sign_importance_A4_portrait_2x2_summary': 'Final_mv1_Step12-05_scRNA_marker_sign_importance_A4_portrait_2x2_summary',
}
FINAL_FIGURE_EXPORT_STEMS = set(FINAL_FIGURE_STEM_ALIASES.values())
FINAL_A4_OUTPUT_CANVAS = os.environ.get('FINAL_A4_OUTPUT_CANVAS', '0') != '0'
FINAL_A4_SHORT_INCH = float(os.environ.get('FINAL_A4_SHORT_INCH', '8.27'))
FINAL_A4_LONG_INCH = float(os.environ.get('FINAL_A4_LONG_INCH', '11.69'))
FINAL_A4_KEEP_TIGHT_BBOX = os.environ.get('FINAL_A4_KEEP_TIGHT_BBOX', '0') == '1'

# Step09-12 scRNA validation scripts share the V8/Final cache and path conventions.
SCRNA_SOURCE_ANALYSIS_DIR = os.path.join(BASE_DIR, 'scRNAseq/Analysis')
SCRNA_FINAL_SCRIPT_DIR = os.path.join(FINAL_WORKSPACE_DIR, 'scripts')
SCRNA_FINAL_TABLE_ROOT = os.path.join(FINAL_WORKSPACE_DIR, 'tables/scRNA')
SCRNA_FINAL_FIGURE_ROOT = os.path.join(FINAL_WORKSPACE_DIR, 'figures/v2/scRNA')
SCRNA_CACHE_ROOT = _FINAL_CACHE_ROOT

os.makedirs(FINAL_TABLE_DIR, exist_ok=True)
os.makedirs(FINAL_CHECKPOINT_ROOT, exist_ok=True)
os.makedirs(FINAL_LOCAL_CKPT_DIR, exist_ok=True)
os.makedirs(SCRNA_FINAL_TABLE_ROOT, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


def setup_scrna_final_environment(reload_utils=True, write_audit_tables=False):
    """Prepare shared paths, caches, and direct-transfer utilities for Final scRNA scripts."""
    os.environ['NUMBA_CACHE_DIR'] = os.path.join(SCRNA_CACHE_ROOT, 'numba_cache')
    os.environ['MPLCONFIGDIR'] = os.path.join(SCRNA_CACHE_ROOT, 'mpl_cache')
    os.environ['XDG_CACHE_HOME'] = os.path.join(SCRNA_CACHE_ROOT, 'xdg_cache')
    os.environ['SCRNA_WRITE_AUDIT_TABLES'] = '1' if write_audit_tables else '0'
    for cache_path in (os.environ['NUMBA_CACHE_DIR'], os.environ['MPLCONFIGDIR'], os.environ['XDG_CACHE_HOME']):
        os.makedirs(cache_path, exist_ok=True)

    for path in (SCRNA_SOURCE_ANALYSIS_DIR, SCRNA_FINAL_SCRIPT_DIR):
        if path in sys.path:
            sys.path.remove(path)
    for path in (SCRNA_SOURCE_ANALYSIS_DIR, SCRNA_FINAL_SCRIPT_DIR):
        sys.path.insert(0, path)

    import importlib
    import direct_transfer_validation_utils as dtv
    if reload_utils:
        dtv = importlib.reload(dtv)
    return dtv


def _as_path_string(path_like):
    """Convert a PathLike object or string to a comparable string."""
    try:
        return os.fspath(path_like)
    except TypeError:
        return path_like


def _final_figure_path(path_like, *, preserve_target_directory=False):
    """Return the canonical Figure path, optionally preserving a safe subdirectory.

    The default keeps the established flat ``figures/v2`` layout.  A caller may
    preserve an explicitly supplied directory only when its resolved parent is
    ``figures/v2`` itself or a descendant; absolute/relative path traversal outside
    that root is rejected before any directory is created.
    """
    path_str = _as_path_string(path_like)
    if not isinstance(path_str, str):
        return path_like
    root, ext = os.path.splitext(path_str)
    if ext.lower() not in {'.png', '.pptx', '.svg', '.tif', '.tiff', '.jpg', '.jpeg'}:
        return path_like
    if preserve_target_directory:
        requested_parent = _FinalPath(path_str).expanduser().resolve().parent
        figure_root = _FinalPath(FIG_DIR).resolve()
        try:
            requested_parent.relative_to(figure_root)
        except ValueError as exc:
            raise ValueError(
                'preserve_target_directory=True requires a target under figures/v2: '
                f'{requested_parent}'
            ) from exc
        directory = os.fspath(requested_parent)
    else:
        directory = FIG_DIR
    basename = os.path.basename(path_str)
    stem, ext = os.path.splitext(basename)
    stem = _normalized_final_figure_stem(stem)
    basename = f'{stem}{ext}'
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, basename)


def _normalized_final_figure_stem(path_like):
    """Normalize a path/name/stem through the canonical Final alias table."""
    path_str = _as_path_string(path_like)
    if not isinstance(path_str, str):
        return path_like
    stem = os.path.splitext(os.path.basename(path_str.strip()))[0]
    # Some historical call sites use intentionally short aliases without the
    # ``Final_`` prefix. Resolve those exact registered aliases before adding
    # the canonical prefix; otherwise entries such as the scRNA dataset stems
    # become unreachable as ``Final_<alias>``.
    if stem in FINAL_FIGURE_STEM_ALIASES:
        return FINAL_FIGURE_STEM_ALIASES[stem]
    if not stem.startswith('Final_'):
        stem = f'Final_{stem}'
    return FINAL_FIGURE_STEM_ALIASES.get(stem, stem)


def _final_figure_stem_registered(path_like):
    """Return whether a Figure path resolves to a registered canonical stem."""
    path_str = _as_path_string(path_like)
    if not isinstance(path_str, str):
        raise TypeError(f'Figure path must be path-like: {path_like!r}')
    stem = _normalized_final_figure_stem(path_str)
    return stem in FINAL_FIGURE_EXPORT_STEMS


def _final_a4_savefig_kwargs(kwargs):
    """Adjust save options so bbox_inches='tight' does not crop the A4 canvas again."""
    if not FINAL_A4_OUTPUT_CANVAS or FINAL_A4_KEEP_TIGHT_BBOX:
        return kwargs
    save_kwargs = dict(kwargs)
    if save_kwargs.get('bbox_inches') == 'tight':
        save_kwargs.pop('bbox_inches', None)
        save_kwargs.pop('pad_inches', None)
    return save_kwargs


def _path_is_under(path_str, root_str):
    """Safely check whether path_str is under root_str."""
    try:
        abs_path = os.path.abspath(path_str)
        abs_root = os.path.abspath(root_str)
        return abs_path == abs_root or abs_path.startswith(abs_root + os.sep)
    except Exception:
        return False


def _is_figure_like_path(path_like):
    """Identify Figure file extensions permitted for saving in figure-only mode."""
    path_str = _as_path_string(path_like)
    if not isinstance(path_str, str):
        return False
    return os.path.splitext(path_str)[1].lower() in FINAL_FIGURE_EXTENSIONS


def _is_protected_output_path(path_like):
    """Protect every Roider workspace path from non-Figure writes in Final mode."""
    path_str = _as_path_string(path_like)
    if not isinstance(path_str, str):
        return False
    protected_roots = [BASE_DIR]
    return any(_path_is_under(path_str, root) for root in protected_roots)


_FINAL_EXACT_DERIVED_WRITE_SCOPES = []


@contextmanager
def allow_exact_final_derived_writes(paths, *, owner):
    """Temporarily allow only explicitly enumerated Final derived file paths.

    This is intentionally not a directory allowlist and does not authorize
    checkpoint/model/ALE/motif writes.  The caller remains responsible for
    staging validation and atomic promotion of the registered derived bundle.
    """

    if not owner or not isinstance(owner, str):
        raise ValueError('A non-empty derived-write owner is required')
    resolved = set()
    for path_like in paths:
        path_str = _as_path_string(path_like)
        if not isinstance(path_str, str):
            raise TypeError(f'Derived write target must be path-like: {path_like!r}')
        target = os.path.realpath(os.path.abspath(path_str))
        if not _path_is_under(target, FINAL_WORKSPACE_DIR):
            raise ValueError(f'Derived write target escapes Final: {target}')
        resolved.add(target)
    if not resolved:
        raise ValueError('Derived write scope cannot be empty')
    scope = {'owner': owner, 'paths': frozenset(resolved)}
    _FINAL_EXACT_DERIVED_WRITE_SCOPES.append(scope)
    try:
        yield scope
    finally:
        if not _FINAL_EXACT_DERIVED_WRITE_SCOPES or _FINAL_EXACT_DERIVED_WRITE_SCOPES[-1] is not scope:
            raise RuntimeError('Final derived-write scopes were exited out of order')
        _FINAL_EXACT_DERIVED_WRITE_SCOPES.pop()


def _is_exact_allowed_derived_write(path_like):
    path_str = _as_path_string(path_like)
    if not isinstance(path_str, str):
        return False
    resolved = os.path.realpath(os.path.abspath(path_str))
    return any(
        resolved in scope['paths']
        for scope in reversed(_FINAL_EXACT_DERIVED_WRITE_SCOPES)
    )


def _should_block_nonfigure_write(path_like):
    """Fail protected writes except registered canonical Figure files."""
    if not FINAL_FIGURE_ONLY_WRITE_GUARD:
        return False
    if not _is_protected_output_path(path_like):
        return False
    if _is_exact_allowed_derived_write(path_like):
        return False
    if not _is_figure_like_path(path_like):
        return True
    path_str = _as_path_string(path_like)
    if not isinstance(path_str, str) or not _path_is_under(path_str, FIG_DIR):
        return True
    basename = os.path.basename(path_str)
    direct_transaction = re.fullmatch(
        r'\.final-pair-[0-9a-f]{32}\.(?:stage|backup)\.(?:png|pptx)',
        basename,
    )
    editable_sidecar = re.fullmatch(
        r'\.\.final-pair-[0-9a-f]{32}\.stage\.[0-9a-f]{32}'
        r'(?:\.(?:fallback|native))?\.(?:svg|png|pptx)',
        basename,
    )
    editable_conversion_sidecar = re.fullmatch(
        r'\.\.\.final-pair-[0-9a-f]{32}\.stage\.[0-9a-f]{32}'
        r'\.native\.[0-9a-f]{32}(?:\.overlay[0-9]{2})?\.(?:svg|png|pptx)',
        basename,
    )
    if direct_transaction or editable_sidecar or editable_conversion_sidecar:
        return False
    raw_stem = os.path.splitext(basename)[0]
    normalized_stem = _normalized_final_figure_stem(raw_stem)
    return raw_stem != normalized_stem or not _final_figure_stem_registered(path_str)


def _raise_final_nonfigure_write(api_name, path_like):
    """Fail closed so callers cannot print a false save-success message."""
    target = _as_path_string(path_like)
    raise PermissionError(
        f'Final Figure-only write guard blocked protected non-Figure write '
        f'via {api_name}: {target}'
    )


_ORIG_OPEN = builtins.open


def _final_safe_open(file, mode='r', *args, **kwargs):
    """Fail closed on protected non-Figure writes instead of returning a no-op."""
    write_mode = any(flag in mode for flag in ('w', 'a', 'x', '+'))
    if write_mode:
        if _should_block_nonfigure_write(file):
            _raise_final_nonfigure_write('open', file)
        parent = os.path.dirname(os.path.abspath(os.fspath(file))) if isinstance(_as_path_string(file), str) else None
        if parent:
            os.makedirs(parent, exist_ok=True)
    return _ORIG_OPEN(file, mode, *args, **kwargs)


builtins.open = _final_safe_open

_ORIG_PATH_OPEN = _FinalPath.open


def _final_safe_path_open(self, mode='r', *args, **kwargs):
    """Apply the same fail-closed policy to pathlib, which uses io.open."""
    if any(flag in str(mode) for flag in ('w', 'a', 'x', '+')):
        if _should_block_nonfigure_write(self):
            _raise_final_nonfigure_write('Path.open', self)
    return _ORIG_PATH_OPEN(self, mode, *args, **kwargs)


_FinalPath.open = _final_safe_path_open

_ORIG_DF_TO_CSV = pd.DataFrame.to_csv
_ORIG_SERIES_TO_CSV = pd.Series.to_csv


def _final_df_to_csv(self, path_or_buf=None, *args, **kwargs):
    if path_or_buf is not None:
        if _should_block_nonfigure_write(path_or_buf):
            _raise_final_nonfigure_write('DataFrame.to_csv', path_or_buf)
    return _ORIG_DF_TO_CSV(self, path_or_buf, *args, **kwargs)


def _final_series_to_csv(self, path_or_buf=None, *args, **kwargs):
    if path_or_buf is not None:
        if _should_block_nonfigure_write(path_or_buf):
            _raise_final_nonfigure_write('Series.to_csv', path_or_buf)
    return _ORIG_SERIES_TO_CSV(self, path_or_buf, *args, **kwargs)


pd.DataFrame.to_csv = _final_df_to_csv
pd.Series.to_csv = _final_series_to_csv

_ORIG_JOBLIB_DUMP = joblib.dump


def _final_joblib_dump(value, filename, *args, **kwargs):
    if _should_block_nonfigure_write(filename):
        _raise_final_nonfigure_write('joblib.dump', filename)
    parent = os.path.dirname(os.path.abspath(os.fspath(filename))) if isinstance(_as_path_string(filename), str) else None
    if parent:
        os.makedirs(parent, exist_ok=True)
    return _ORIG_JOBLIB_DUMP(value, filename, *args, **kwargs)


joblib.dump = _final_joblib_dump

_ORIG_WRITE_H5AD = ad.AnnData.write_h5ad


def _final_write_h5ad(self, filename=None, *args, **kwargs):
    if filename is not None:
        if _should_block_nonfigure_write(filename):
            _raise_final_nonfigure_write('AnnData.write_h5ad', filename)
        parent = os.path.dirname(os.path.abspath(os.fspath(filename))) if isinstance(_as_path_string(filename), str) else None
        if parent:
            os.makedirs(parent, exist_ok=True)
    return _ORIG_WRITE_H5AD(self, filename, *args, **kwargs)


ad.AnnData.write_h5ad = _final_write_h5ad

_ORIG_PLT_SAVEFIG = plt.savefig
_ORIG_FIG_SAVEFIG = matplotlib.figure.Figure.savefig


def _final_plt_savefig(fname, *args, **kwargs):
    path_str = _as_path_string(fname)
    if isinstance(path_str, str) and os.path.splitext(path_str)[1].lower() == '.pptx':
        raise ValueError(
            'Standalone PPTX savefig is disabled. Use save_final_figure_pair().'
        )
    if not _final_figure_stem_registered(fname):
        raise PermissionError(
            'Cannot save an unregistered Final Figure stem: '
            f'{os.path.basename(os.fspath(fname))}'
        )
    if isinstance(path_str, str) and os.path.splitext(path_str)[1].lower() == '.pdf':
        raise ValueError('Final v2 PDF export is disabled. Use the PNG/PPTX pair helper.')
    final_path = _final_figure_path(fname)
    save_kwargs = _final_a4_savefig_kwargs(kwargs)
    return _ORIG_PLT_SAVEFIG(final_path, *args, **save_kwargs)


def _final_fig_savefig(self, fname, *args, **kwargs):
    path_str = _as_path_string(fname)
    if isinstance(path_str, str) and os.path.splitext(path_str)[1].lower() == '.pptx':
        raise ValueError(
            'Standalone PPTX savefig is disabled. Use save_final_figure_pair().'
        )
    if not _final_figure_stem_registered(fname):
        raise PermissionError(
            'Cannot save an unregistered Final Figure stem: '
            f'{os.path.basename(os.fspath(fname))}'
        )
    if isinstance(path_str, str) and os.path.splitext(path_str)[1].lower() == '.pdf':
        raise ValueError('Final v2 PDF export is disabled. Use the PNG/PPTX pair helper.')
    final_path = _final_figure_path(fname)
    save_kwargs = _final_a4_savefig_kwargs(kwargs)
    return _ORIG_FIG_SAVEFIG(self, final_path, *args, **save_kwargs)


def save_final_figure_pair(fig, png_path, pptx_path=None, *, dpi=300,
                           facecolor='white', auxiliary_tight=False,
                           pad_inches=0.02, validate=True,
                           preserve_target_directory=False,
                           pptx_rasterize_dense_layers=False,
                           pptx_dense_layer_point_threshold=100,
                           pptx_dense_layer_mode='full_canvas',
                           show=False, close=False):
    """Save the same designed canvas as a PNG/PPTX pair with near-atomic replacement.

    Main manuscript Figures use auxiliary_tight=False. This function does not
    resize the Figure or use a tight bounding box to correct the target aspect ratio.
    Save and validate both formats at unique hidden paths in the target directory
    before promoting them together. By default, existing targets are replaced after
    a pair-aware backup; failures restore the original pair, individual file, or absence.
    Raster overlays for dense Matplotlib artists in PPTX apply only to Figures
    that explicitly opt in; other callers retain the fully native default.
    """
    if pptx_path is None:
        pptx_path = os.path.splitext(os.fspath(png_path))[0] + '.pptx'
    if validate is not True:
        raise ValueError(
            'Canonical Final Figure pairs require validate=True. '
            'Layout validation cannot be disabled.'
        )
    if not _final_figure_stem_registered(png_path):
        raise PermissionError(
            'Cannot save an unregistered Final Figure stem as a PNG/PPTX pair: '
            f'{os.path.basename(os.fspath(png_path))}'
        )
    png_path = _FinalPath(_final_figure_path(
        png_path, preserve_target_directory=preserve_target_directory,
    ))
    pptx_path = _FinalPath(_final_figure_path(
        pptx_path, preserve_target_directory=preserve_target_directory,
    ))
    if png_path.suffix.lower() != '.png' or pptx_path.suffix.lower() != '.pptx':
        raise ValueError(f'Figure pair extensions must be PNG/PPTX: {png_path}, {pptx_path}')
    if png_path.parent.resolve() != pptx_path.parent.resolve():
        raise ValueError(f'Figure pair files must share a target directory: {png_path}, {pptx_path}')
    if png_path.stem != pptx_path.stem:
        raise ValueError(f'Figure pair files must share a stem: {png_path}, {pptx_path}')
    png_path.parent.mkdir(parents=True, exist_ok=True)
    original_exists = (png_path.exists(), pptx_path.exists())
    _validate_matplotlib_arial_font_contract(fig)
    validate_final_figure_canvas(fig)
    original_size = tuple(float(v) for v in fig.get_size_inches())
    save_kwargs = {'facecolor': facecolor}
    if auxiliary_tight:
        save_kwargs.update({'bbox_inches': 'tight', 'pad_inches': pad_inches})
    transaction_paths = _new_final_pair_transaction_paths(png_path, pptx_path)
    try:
        _ORIG_FIG_SAVEFIG(
            fig, transaction_paths['png_stage'], dpi=dpi, **save_kwargs,
        )
        _validate_staged_final_png(transaction_paths['png_stage'])
        save_matplotlib_figure_as_editable_pptx(
            fig,
            transaction_paths['pptx_stage'],
            png_fallback_path=transaction_paths['png_stage'],
            title=png_path.stem,
            validate=True,
            convert_to_native_shapes=True,
            fallback_to_vector_object=False,
            vectorize_text_as_paths=True,
            rasterize_dense_layers=pptx_rasterize_dense_layers,
            dense_layer_point_threshold=pptx_dense_layer_point_threshold,
            dense_layer_mode=pptx_dense_layer_mode,
            dpi=dpi,
            **save_kwargs,
        )
        _validate_staged_final_pptx(transaction_paths['pptx_stage'])
        final_size = tuple(float(v) for v in fig.get_size_inches())
        if not np.allclose(original_size, final_size, rtol=0, atol=1e-9):
            raise RuntimeError(
                f'Figure canvas changed during saving: {original_size} -> {final_size}'
            )
        _promote_staged_final_pair(
            transaction_paths, png_path, pptx_path, original_exists,
        )
    finally:
        for key in ('png_stage', 'pptx_stage'):
            try:
                transaction_paths[key].unlink(missing_ok=True)
            except OSError as cleanup_exc:
                warnings.warn(
                    f"Failed to clean up unique Figure pair staging path: {transaction_paths[key]}: "
                    f"{cleanup_exc}", RuntimeWarning,
                )
    if show:
        plt.figure(fig.number)
        plt.show()
    if close:
        plt.close(fig)
    return os.fspath(png_path), os.fspath(pptx_path)


plt.savefig = _final_plt_savefig
matplotlib.figure.Figure.savefig = _final_fig_savefig

# 3. Shared file paths
H5AD_PATH    = os.path.join(SOURCE_ANALYSIS_DIR, 'checkpoints/step00_Raw_data.h5ad')
ANNOT_PATH   = os.path.join(SPATIAL_DIR, 'cells_annotation.csv')
EXPR_PATH    = os.path.join(SPATIAL_DIR, 'cells_expression.csv')
METADATA_PATH = os.path.join(SPATIAL_DIR, 'cells_metadata.csv')

# 4. Global hyperparameters
RANDOM_SEED   = 42
N_ITER        = 100         # Number of ensemble iterations
CELLS_PER_CLASS = 300_000   # Maximum samples per class during CV
CV_TEST_SIZE  = 0.3         # Training/validation split (7:3)
EARLY_STOP_PATIENCE = 5     # Early stopping patience for the overall loop

# Four model keys (CatBoost removed)
MODEL_KEYS = ['RF', 'XGB', 'ENet', 'MLP']
N_COMBOS = sum(1 for r in range(1, len(MODEL_KEYS)+1) for _ in _combs(MODEL_KEYS, r))  # 15

print(f"🔹 Analysis directory: {ANALYSIS_DIR}")
print(f"🔹 Checkpoints: {CKPT_DIR}")
print(f"🔹 Figure: {FIG_DIR}")
print(f"🔹 [Settings] SEED={RANDOM_SEED}, N_ITER={N_ITER}, Split=70:30, models={MODEL_KEYS}")
print(f"🔹 Number of combinations: {N_COMBOS} (C(4,1)+C(4,2)+C(4,3)+C(4,4))")

# Preserve Python's default wildcard-import surface while exposing the one
# compatibility helper used directly by the canonical Step02 entry point.
__all__ = [name for name in globals() if not name.startswith('_')]
__all__.append('_final_figure_stem_registered')
