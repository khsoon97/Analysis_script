"""Shared scRNA preparation plus retained historical transfer utilities.

The active Final Step09–12 fixed-signature path consumes only the protected
CODEX proxy H5AD inputs and saved fixed-signature tables. Fitted-classifier
transfer helpers and `02_*_direct_codex_transfer_predictions.h5ad` paths remain
in this module solely as an inactive historical workflow; they are not current
fixed-signature inputs.
"""

from pathlib import Path
import importlib
import json
import os
import shutil
import sys
import textwrap
import warnings

# Fix scanpy/numba import caches under Final in the SpaceFlow Python 3.13 environment.
_FINAL_CACHE_ROOT = "__ROIDER_ROOT__/Analysis/python/Final/environment/cache"
os.environ.setdefault("NUMBA_CACHE_DIR", f"{_FINAL_CACHE_ROOT}/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", f"{_FINAL_CACHE_ROOT}/mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", f"{_FINAL_CACHE_ROOT}/xdg_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
for _cache_path in (os.environ["NUMBA_CACHE_DIR"], os.environ["MPLCONFIGDIR"], os.environ["XDG_CACHE_HOME"]):
    Path(_cache_path).mkdir(parents=True, exist_ok=True)

import anndata as ad
import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import pandas.core.generic as _pd_generic
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import joblib.numpy_pickle as _joblib_numpy_pickle
from pandas.api.types import is_string_dtype
from scipy.spatial import cKDTree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm
from editable_pptx import save_matplotlib_figure_as_editable_pptx

BASE_DIR = Path("__ROIDER_ROOT__/scRNAseq")
ROIDER_DIR = BASE_DIR / "Roider_VRJUNV"
ANALYSIS_DIR = BASE_DIR / "Analysis"
V9_ANALYSIS_DIR = Path("__ROIDER_ROOT__/Analysis/python/V9")
FINAL_ANALYSIS_DIR = Path("__ROIDER_ROOT__/Analysis/python/Final")
FINAL_CHECKPOINT_ROOT = FINAL_ANALYSIS_DIR / "checkpoints"
OUT_DIR = FINAL_ANALYSIS_DIR / "tables" / "scRNA" / "scRNA_Roider"
FIG_DIR = FINAL_ANALYSIS_DIR / "figures" / "v2" / "scRNA" / "Roider"
GSE182434_DIR = BASE_DIR / "GSE182434"
GSE182434_RAW_MATRIX = GSE182434_DIR / "raw" / "GSE182434_raw_count_matrix.txt.gz"
GSE182434_CELL_ANNOTATION = GSE182434_DIR / "raw" / "GSE182434_cell_annotation.txt.gz"
GSE182434_OUT_DIR = FINAL_ANALYSIS_DIR / "tables" / "scRNA" / "scRNA_GSE182434"
GSE182434_FIG_DIR = FINAL_ANALYSIS_DIR / "figures" / "v2" / "scRNA" / "GSE182434"
FINAL_A4_OUTPUT_CANVAS = os.environ.get("FINAL_A4_OUTPUT_CANVAS", "0") != "0"
FINAL_A4_SHORT_INCH = float(os.environ.get("FINAL_A4_SHORT_INCH", "8.27"))
FINAL_A4_LONG_INCH = float(os.environ.get("FINAL_A4_LONG_INCH", "11.69"))
FINAL_A4_KEEP_TIGHT_BBOX = os.environ.get("FINAL_A4_KEEP_TIGHT_BBOX", "0") == "1"

V9_SOURCE_CKPT_DIR = V9_ANALYSIS_DIR / "checkpoints" / "mv1"
# Retain variable names for compatibility; actual references point to snapshots copied into Final.
V9_CKPT_DIR = FINAL_CHECKPOINT_ROOT / "mv1"
V9_TRANSFER_ROOT = FINAL_CHECKPOINT_ROOT / "transfer_models"
CODEX_STEP02_H5AD = V9_CKPT_DIR / "mv1_step02.h5ad"
CODEX_STEP04B_H5AD = V9_CKPT_DIR / "mv1_step04B_pan_cell.h5ad"
CODEX_MODELS_BROAD = V9_CKPT_DIR / "mv1_models_broad.pkl"
SOURCE_DIRECT_OUT_DIR = ANALYSIS_DIR / "outputs" / "direct_codex_transfer"
SOURCE_GSE182434_DIRECT_OUT_DIR = ANALYSIS_DIR / "outputs" / "direct_codex_transfer_gse182434"
FINAL_SCRNA_CACHE_DIR = FINAL_CHECKPOINT_ROOT / "scRNA"
ROIDER_CACHE_DIR = FINAL_SCRNA_CACHE_DIR / "Roider"
GSE182434_CACHE_DIR = FINAL_SCRNA_CACHE_DIR / "GSE182434"
DIRECT_MODEL_PACKAGE = ROIDER_CACHE_DIR / "direct_transfer_broad_model_package.joblib"
DIRECT_TCELL_SUBTYPE_MODEL_PACKAGE = ROIDER_CACHE_DIR / "direct_transfer_tcell_subtype_model_package.joblib"

RAW_H5AD = ROIDER_CACHE_DIR / "01_roider_raw_counts_with_reference.h5ad"
PROCESSED_H5AD = ROIDER_CACHE_DIR / "01_roider_processed_umap.h5ad"
FEATURE_H5AD = ROIDER_CACHE_DIR / "01_roider_codex_proxy_features.h5ad"
PREDICTED_H5AD = ROIDER_CACHE_DIR / "02_roider_direct_codex_transfer_predictions.h5ad"
GSE182434_FEATURE_H5AD = GSE182434_CACHE_DIR / "01_gse182434_codex_proxy_features.h5ad"
GSE182434_PREDICTED_H5AD = GSE182434_CACHE_DIR / "02_gse182434_direct_codex_transfer_predictions.h5ad"
GSE182434_QC_H5AD = GSE182434_CACHE_DIR / "00_gse182434_qc_processed_umap.h5ad"
CACHE_STAGE_KEY = "direct_transfer_cache_stage"
ROIDER_RAW_CACHE_STAGE = "raw_with_reference_curated_v2"
ROIDER_PROCESSED_CACHE_STAGE = "processed_umap_curated_v2"
ROIDER_FEATURE_CACHE_STAGE = "codex_proxy_features_curated_v2"
ROIDER_PREDICTION_CACHE_STAGE = "direct_predictions_curated_v4_broad_tcell_cd4ctl_annotation"
SCRNA_FIGURE_ONLY_WRITE_GUARD = os.environ.get("SCRNA_FIGURE_ONLY_WRITE_GUARD", "1") == "1"
SCRNA_WRITE_AUDIT_TABLES = os.environ.get("SCRNA_WRITE_AUDIT_TABLES", "0") == "1"
SCRNA_FIGURE_EXTENSIONS = {".png", ".pptx", ".svg", ".tif", ".tiff", ".jpg", ".jpeg", ".html"}
SCRNA_TABLE_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".json", ".txt")
SCRNA_FINAL_STEM_ALIASES = {
    "Final_scRNA_GSE182434_direct_transfer_3x3_umap_summary": "Final_mv1_Step12-01_GSE182434_direct_transfer_3x3_umap_summary",
    "Final_scRNA_GSE182434_direct_transfer_performance_comparison": "Final_mv1_Step12-02_GSE182434_direct_transfer_performance_comparison",
    "Final_scRNA_Roider_direct_transfer_3x3_umap_summary": "Final_mv1_Step12-03_Roider_direct_transfer_3x3_umap_summary",
    "Final_scRNA_Roider_direct_transfer_performance_comparison": "Final_mv1_Step12-04_Roider_direct_transfer_performance_comparison",
}

if os.environ.get("FINAL_SUPPRESS_INTERACTIVE_SHOW", "1") != "0":
    plt.ioff()
    plt.show = lambda *args, **kwargs: None


def _scrna_path_is_under(path_like, root):
    """Safely check whether path_like is under root."""
    try:
        path = Path(path_like).resolve()
        root = Path(root).resolve()
        return path == root or root in path.parents
    except Exception:
        return False


def _scrna_is_figure_path(path_like):
    """Identify scRNA Figure output files by extension."""
    try:
        return Path(path_like).suffix.lower() in SCRNA_FIGURE_EXTENSIONS
    except Exception:
        return False


def _scrna_is_table_path(path_like):
    """Identify Final table/audit output files by extension."""
    try:
        return Path(path_like).name.lower().endswith(SCRNA_TABLE_SUFFIXES)
    except Exception:
        return False


def _scrna_is_final_table_path(path_like):
    """Identify audit tables under the Final scRNA tables directory."""
    return (
        _scrna_is_table_path(path_like)
        and (
            _scrna_path_is_under(path_like, OUT_DIR)
            or _scrna_path_is_under(path_like, GSE182434_OUT_DIR)
        )
    )


def _scrna_is_protected_output_path(path_like):
    """Identify scRNA output paths whose updates are blocked in figure-only mode."""
    protected_roots = [
        V9_ANALYSIS_DIR,
        FINAL_ANALYSIS_DIR / "tables",
        FINAL_ANALYSIS_DIR / "checkpoints",
        OUT_DIR,
        GSE182434_OUT_DIR,
        SOURCE_DIRECT_OUT_DIR,
        SOURCE_GSE182434_DIRECT_OUT_DIR,
        FIG_DIR,
        GSE182434_FIG_DIR,
    ]
    return any(_scrna_path_is_under(path_like, root) for root in protected_roots)


def _scrna_should_block_nonfigure_write(path_like):
    """Return whether Figure-only mode must reject a protected non-Figure write."""
    return (
        SCRNA_FIGURE_ONLY_WRITE_GUARD
        and _scrna_is_protected_output_path(path_like)
        and not _scrna_is_figure_path(path_like)
        and not (SCRNA_WRITE_AUDIT_TABLES and _scrna_is_final_table_path(path_like))
    )


def _scrna_is_guarded_callable(func):
    """Check whether this is an scRNA guard installed by a previous reload."""
    return (
        getattr(func, "_final_scrna_guarded", False)
        or getattr(func, "__name__", "") in {
            "_scrna_guarded_df_to_csv",
            "_scrna_guarded_series_to_csv",
            "_scrna_guarded_joblib_dump",
        }
    )


if _scrna_is_guarded_callable(getattr(pd.DataFrame, "_final_scrna_original_to_csv", None)):
    pd.DataFrame._final_scrna_original_to_csv = _pd_generic.NDFrame.to_csv
if _scrna_is_guarded_callable(getattr(pd.Series, "_final_scrna_original_to_csv", None)):
    pd.Series._final_scrna_original_to_csv = _pd_generic.NDFrame.to_csv
if _scrna_is_guarded_callable(getattr(joblib, "_final_scrna_original_dump", None)):
    joblib._final_scrna_original_dump = _joblib_numpy_pickle.dump

if _scrna_is_guarded_callable(pd.DataFrame.to_csv):
    pd.DataFrame.to_csv = _pd_generic.NDFrame.to_csv
if _scrna_is_guarded_callable(pd.Series.to_csv):
    pd.Series.to_csv = _pd_generic.NDFrame.to_csv
if _scrna_is_guarded_callable(joblib.dump):
    joblib.dump = _joblib_numpy_pickle.dump

if not hasattr(pd.DataFrame, "_final_scrna_original_to_csv"):
    pd.DataFrame._final_scrna_original_to_csv = pd.DataFrame.to_csv
if not hasattr(pd.Series, "_final_scrna_original_to_csv"):
    pd.Series._final_scrna_original_to_csv = pd.Series.to_csv
if not hasattr(joblib, "_final_scrna_original_dump"):
    joblib._final_scrna_original_dump = joblib.dump

_ORIG_SCRNA_DF_TO_CSV = pd.DataFrame._final_scrna_original_to_csv
_ORIG_SCRNA_SERIES_TO_CSV = pd.Series._final_scrna_original_to_csv
_ORIG_SCRNA_JOBLIB_DUMP = joblib._final_scrna_original_dump


def _scrna_guarded_df_to_csv(self, path_or_buf=None, *args, **kwargs):
    if path_or_buf is not None and _scrna_should_block_nonfigure_write(path_or_buf):
        raise PermissionError(
            f"scRNA Figure-only guard blocked protected CSV write: {path_or_buf}"
        )
    return _ORIG_SCRNA_DF_TO_CSV(self, path_or_buf, *args, **kwargs)


def _scrna_guarded_series_to_csv(self, path_or_buf=None, *args, **kwargs):
    if path_or_buf is not None and _scrna_should_block_nonfigure_write(path_or_buf):
        raise PermissionError(
            f"scRNA Figure-only guard blocked protected CSV write: {path_or_buf}"
        )
    return _ORIG_SCRNA_SERIES_TO_CSV(self, path_or_buf, *args, **kwargs)


def _scrna_guarded_joblib_dump(value, filename, *args, **kwargs):
    if _scrna_should_block_nonfigure_write(filename):
        raise PermissionError(
            f"scRNA Figure-only guard blocked protected joblib write: {filename}"
        )
    return _ORIG_SCRNA_JOBLIB_DUMP(value, filename, *args, **kwargs)


_scrna_guarded_df_to_csv._final_scrna_guarded = True
_scrna_guarded_series_to_csv._final_scrna_guarded = True
_scrna_guarded_joblib_dump._final_scrna_guarded = True

pd.DataFrame.to_csv = _scrna_guarded_df_to_csv
pd.Series.to_csv = _scrna_guarded_series_to_csv
joblib.dump = _scrna_guarded_joblib_dump


def install_sklearn_loss_pickle_compat():
    """Map saved `_loss` pickle references from other scikit-learn versions to the current sklearn."""
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = importlib.import_module("sklearn._loss.loss")
    return sys.modules["_loss"]


def load_compatible_joblib(filename, *args, **kwargs):
    """Load existing joblib packages across environments with different sklearn internal module paths."""
    try:
        return joblib.load(filename, *args, **kwargs)
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "_loss":
            raise
        install_sklearn_loss_pickle_compat()
        return joblib.load(filename, *args, **kwargs)


def _final_scrna_figure_path(path_like):
    """Standardize scRNA validation Figures to the Final Figure directory and Final_ prefix."""
    path = Path(path_like)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".pptx", ".svg", ".tif", ".tiff", ".jpg", ".jpeg"}:
        return path_like
    resolved_parent = path.parent.resolve()
    path_parts = set(path.parts)
    if (
        resolved_parent == GSE182434_FIG_DIR.resolve()
        or _scrna_path_is_under(path, GSE182434_OUT_DIR)
        or "direct_codex_transfer_gse182434" in path_parts
    ):
        out_dir = GSE182434_FIG_DIR
        prefix = "Final_scRNA_GSE182434_"
    elif (
        resolved_parent == FIG_DIR.resolve()
        or _scrna_path_is_under(path, OUT_DIR)
        or "direct_codex_transfer" in path_parts
    ):
        out_dir = FIG_DIR
        prefix = "Final_scRNA_Roider_"
    else:
        out_dir = FIG_DIR
        prefix = "Final_scRNA_Roider_"
    name = path.name
    if not name.startswith("Final_"):
        name = f"{prefix}{name}"
    stem = Path(name).stem
    stem = SCRNA_FINAL_STEM_ALIASES.get(stem, stem)
    name = f"{stem}{path.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def resolve_final_scrna_figure_path(path_like):
    """Expose the actual Final save path of an scRNA Figure to external steps."""
    return _final_scrna_figure_path(path_like)


def _coerce_scrna_figure_to_a4_canvas(fig):
    """Set the scRNA Figure canvas to the A4 aspect ratio immediately before saving."""
    if not FINAL_A4_OUTPUT_CANVAS or fig is None:
        return
    try:
        width, height = fig.get_size_inches()
    except Exception:
        return
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        return

    if width >= height:
        target_size = (FINAL_A4_LONG_INCH, FINAL_A4_SHORT_INCH)
    else:
        target_size = (FINAL_A4_SHORT_INCH, FINAL_A4_LONG_INCH)
    fig.set_size_inches(*target_size, forward=True)


def _final_scrna_a4_savefig_kwargs(kwargs):
    """Adjust save options so bbox_inches='tight' does not crop the A4 canvas again."""
    if not FINAL_A4_OUTPUT_CANVAS or FINAL_A4_KEEP_TIGHT_BBOX:
        return kwargs
    save_kwargs = dict(kwargs)
    if save_kwargs.get("bbox_inches") == "tight":
        save_kwargs.pop("bbox_inches", None)
        save_kwargs.pop("pad_inches", None)
    return save_kwargs


def _final_scrna_plt_savefig(fname, *args, **kwargs):
    """Save the current Figure to its Final path without recursion across repeated reloads."""
    _coerce_scrna_figure_to_a4_canvas(plt.gcf())
    return plt.gcf().savefig(_final_scrna_figure_path(fname), *args, **_final_scrna_a4_savefig_kwargs(kwargs))


def _final_scrna_fig_savefig(self, fname, *args, **kwargs):
    """Avoid recursion from wrapping the original Matplotlib savefig wrapper again."""
    if args:
        raise TypeError("Figure.savefig does not support positional arguments after fname.")
    path = _final_scrna_figure_path(fname)
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError("Final scRNA PDF export is disabled. Use the PNG/PPTX helper.")
    _coerce_scrna_figure_to_a4_canvas(self)
    kwargs = _final_scrna_a4_savefig_kwargs(kwargs).copy()
    if suffix == ".pptx":
        return save_matplotlib_figure_as_editable_pptx(self, path, **kwargs)
    transparent = kwargs.pop("transparent", None)
    kwargs.setdefault("dpi", matplotlib.rcParams["savefig.dpi"])

    if transparent is None:
        transparent = matplotlib.rcParams["savefig.transparent"]
    if transparent:
        kwargs.setdefault("facecolor", "none")
        kwargs.setdefault("edgecolor", "none")
    else:
        kwargs.setdefault("facecolor", matplotlib.rcParams["savefig.facecolor"])
        kwargs.setdefault("edgecolor", matplotlib.rcParams["savefig.edgecolor"])

    return self.canvas.print_figure(path, **kwargs)


_final_scrna_plt_savefig._final_scrna_guarded = True
_final_scrna_fig_savefig._final_scrna_guarded = True
plt.savefig = _final_scrna_plt_savefig
matplotlib.figure.Figure.savefig = _final_scrna_fig_savefig


def _save_scrna_figure_png_pptx(fig, png_path, *, dpi=350, **save_kwargs):
    """Save the same scRNA figure canvas as PNG and native-shape PPTX."""
    _coerce_scrna_figure_to_a4_canvas(fig)
    save_kwargs = _final_scrna_a4_savefig_kwargs(save_kwargs).copy()
    final_png_path = Path(_final_scrna_figure_path(png_path))
    final_pptx_path = Path(_final_scrna_figure_path(final_png_path.with_suffix(".pptx")))
    fig.savefig(final_png_path, dpi=dpi, **save_kwargs)
    save_matplotlib_figure_as_editable_pptx(
        fig,
        final_pptx_path,
        png_fallback_path=final_png_path,
        dpi=dpi,
        **save_kwargs,
    )
    return final_png_path, final_pptx_path

SAMPLE_SHEET = ROIDER_DIR / "sample_sheet.csv"
B_ANNOTATION_RAW = ROIDER_DIR / "AnnotationBcells.csv"
T_ANNOTATION_RAW = ROIDER_DIR / "Annotation_Tcells.csv"
ROIDER_CURATED_ANNOTATION_DIR = ROIDER_DIR / "curated_annotations"
B_ANNOTATION_CURATED = ROIDER_CURATED_ANNOTATION_DIR / "AnnotationBcells_curated.csv"
T_ANNOTATION_CURATED = ROIDER_CURATED_ANNOTATION_DIR / "Annotation_Tcells_curated.csv"
USE_CURATED_ROIDER_ANNOTATIONS = os.environ.get("USE_CURATED_ROIDER_ANNOTATIONS", "1") != "0"
B_ANNOTATION = B_ANNOTATION_CURATED if USE_CURATED_ROIDER_ANNOTATIONS else B_ANNOTATION_RAW
T_ANNOTATION = T_ANNOTATION_CURATED if USE_CURATED_ROIDER_ANNOTATIONS else T_ANNOTATION_RAW

CODEX_BROAD_LABEL_ORDER = ["B cells", "Myeloid cells", "NK cells", "Stromal cells", "T cells"]
DISPLAY_BROAD_LABEL_ORDER = ["B_cells", "Myeloid_cells", "NK_cells", "Stromal_cells", "T_cells"]
# T-cell subtype labels that can be quantitatively evaluated against the external scRNA reference.
T_SUBTYPE_LABEL_ORDER = ["TH", "TFH", "TREG", "CD8T"]
# Include Tpr/CD4_CTL in CODEX classifier annotation/projection, but restrict metrics to the evaluation label set above.
T_SUBTYPE_ANNOTATION_LABEL_ORDER = ["TH", "TFH", "TPR", "TREG", "CD8T", "CD4_CTL"]
T_SUBTYPE_DISPLAY_ORDER = T_SUBTYPE_ANNOTATION_LABEL_ORDER + ["Non_T_cells", "Missing", "Not_evaluated"]

BROAD_DISPLAY_MAP = {
    "B cells": "B_cells",
    "Myeloid cells": "Myeloid_cells",
    "NK cells": "NK_cells",
    "Stromal cells": "Stromal_cells",
    "T cells": "T_cells",
    "B_cells": "B_cells",
    "Myeloid_cells": "Myeloid_cells",
    "NK_cells": "NK_cells",
    "Stromal_cells": "Stromal_cells",
    "T_cells": "T_cells",
}
DISPLAY_TO_CODEX_BROAD_MAP = {v: k for k, v in BROAD_DISPLAY_MAP.items() if k in CODEX_BROAD_LABEL_ORDER}

BROAD_COLOR_MAP = {
    "B_cells": "#4E79A7",
    "T_cells": "#E15759",
    "Myeloid_cells": "#59A14F",
    "Stromal_cells": "#B07AA1",
    "NK_cells": "#F28E2B",
    "Missing": "#C7C7C7",
    "Not_evaluated": "#D0D0D0",
    "Low_confidence": "#6F6F6F",
}

T_SUBTYPE_COLOR_MAP = {
    "TH": "#4E79A7",
    "TFH": "#59A14F",
    "TPR": "#EDC948",
    "TREG": "#F28E2B",
    "CD8T": "#E15759",
    "CD4_CTL": "#7E57C2",
    "Non_T_cells": "#D0D0D0",
    "Missing": "#8C8C8C",
    "Not_evaluated": "#D0D0D0",
    "Low_confidence": "#6F6F6F",
}

GSE182434_CELLTYPE_TO_DISPLAY = {
    "B cells": "B_cells",
    "Plasma cells": "B_cells",
    "T cells CD8": "T_cells",
    "T cells CD4": "T_cells",
    "Tregs": "T_cells",
    "TFH": "T_cells",
    "Monocytes and Macrophages": "Myeloid_cells",
    "NK cells": "NK_cells",
}

GSE182434_CELLTYPE_TO_T_SUBTYPE = {
    "T cells CD8": "CD8T",
    "T cells CD4": "TH",
    "Tregs": "TREG",
    "TFH": "TFH",
}

GSE182434_CANDIDATE_TO_DISPLAY = {
    "B_cells": "B_cells",
    "T_cells": "T_cells",
    "Myeloid": "Myeloid_cells",
    "Myeloid_cells": "Myeloid_cells",
    "NK": "NK_cells",
    "NK_cells": "NK_cells",
}

T_SUBTYPE_DISPLAY_MAP = {
    "TH": "TH",
    "Th": "TH",
    "TFH": "TFH",
    "Tfh": "TFH",
    "TPR": "TPR",
    "Tpr": "TPR",
    "TREG": "TREG",
    "Treg": "TREG",
    "Tregs": "TREG",
    "TTOX": "CD8T",
    "CD8T": "CD8T",
    "CD4_CTL": "CD4_CTL",
    "CTL": "CD4_CTL",
}

CODEX_T_SUBTYPE_TO_DISPLAY = {
    "Th": "TH",
    "Tfh": "TFH",
    "Tpr": "TPR",
    "Treg": "TREG",
    "CD8T": "CD8T",
    "CD4_CTL": "CD4_CTL",
}

BROAD_MARKER_SETS = {
    "B_cells": ["CD20", "PAX5", "CD79a", "CD21"],
    "T_cells": ["CD3", "CD7", "CD5", "CD4", "CD8"],
    "Myeloid_cells": ["CD14", "CD68", "CD163", "CD11c", "CD11b"],
    "Stromal_cells": ["CD34", "CD31", "CD90", "PDPN"],
    "NK_cells": ["CD56", "CD57", "CD16"],
}

T_SUBTYPE_MARKER_SETS = {
    "TH": ["CD4"],
    "TFH": ["CXCR5", "PD1", "ICOS", "CXCL13", "BCL6"],
    "TREG": ["FOXP3", "CD25", "CTLA4"],
    "CD8T": ["CD8", "GRZB"],
    "CD4_CTL": ["CD4", "GRZB"],
}

CODEX_TO_SCRNA_GENES = {
    "BCL6": ["BCL6"],
    "GATA3": ["GATA3"],
    "CXCR5": ["CXCR5"],
    "Tbet": ["TBX21"],
    "CD62L": ["SELL"],
    "TIM3": ["HAVCR2"],
    "CD134": ["TNFRSF4"],
    "IL10": ["IL10"],
    "CTLA4": ["CTLA4"],
    "CXCL13": ["CXCL13"],
    "PD1": ["PDCD1"],
    "LAG3": ["LAG3"],
    "CD45RO": ["PTPRC"],
    "ICOS": ["ICOS"],
    "CD69": ["CD69"],
    "CD45RA": ["PTPRC"],
    "GRZB": ["GZMB"],
    "CD44": ["CD44"],
    "PAX5": ["PAX5"],
    "FOXP3": ["FOXP3"],
    "CD45": ["PTPRC"],
    "CD20": ["MS4A1"],
    "CD79a": ["CD79A"],
    "CD21": ["CR2"],
    "CD38": ["CD38"],
    "lambda": ["IGLC1", "IGLC2", "IGLC3", "IGLC6", "IGLC7"],
    "kappa": ["IGKC", "IGKV1-5", "IGKV1-39", "IGKV3-20"],
    "CD3": ["CD3D", "CD3E", "CD3G"],
    "CD7": ["CD7"],
    "CD5": ["CD5"],
    "CD4": ["CD4"],
    "CD8": ["CD8A", "CD8B"],
    "CD56": ["NCAM1"],
    "CD57": ["B3GAT1"],
    "CD25": ["IL2RA"],
    "CD11b": ["ITGAM"],
    "CD15": ["FUT4"],
    "CD14": ["CD14"],
    "CD16": ["FCGR3A", "FCGR3B"],
    "CD68": ["CD68"],
    "CD163": ["CD163"],
    "CD206": ["MRC1"],
    "CD11c": ["ITGAX"],
    "MCT": ["TPSAB1", "TPSB2", "CPA3"],
    "CD34": ["CD34"],
    "CD31": ["PECAM1"],
    "CD90": ["THY1"],
    "PDPN": ["PDPN"],
    "HLA-DR": ["HLA-DRA", "HLA-DRB1", "HLA-DRB5"],
    "Ki67": ["MKI67"],
}

ROIDER_ORIGINAL_PAPER_QC_RULES = {
    "DLBCL1": {"low_n_gene": 200, "high_n_gene": 4000, "max_pc_mito": 0.05},
    "DLBCL2": {"low_n_gene": 200, "high_n_gene": 3000, "max_pc_mito": 0.05},
    "DLBCL3": {"low_n_gene": 200, "high_n_gene": 5000, "max_pc_mito": 0.05},
    "tFL1": {"low_n_gene": 200, "high_n_gene": 2500, "max_pc_mito": 0.05},
    "tFL2": {"low_n_gene": 600, "high_n_gene": 2500, "max_pc_mito": 0.05},
    "FL1": {"low_n_gene": 200, "high_n_gene": 2500, "max_pc_mito": 0.05},
    "FL2": {"low_n_gene": 200, "high_n_gene": 2500, "max_pc_mito": 0.05},
    "FL3": {"low_n_gene": 200, "high_n_gene": 2000, "max_pc_mito": 0.05},
    "FL4": {"low_n_gene": 600, "high_n_gene": 2500, "max_pc_mito": 0.05},
    "rLN1": {"low_n_gene": 200, "high_n_gene": 2500, "max_pc_mito": 0.05},
    "rLN2": {"low_n_gene": 200, "high_n_gene": 2500, "max_pc_mito": 0.05},
    "rLN3": {"low_n_gene": 200, "high_n_gene": 3000, "max_pc_mito": 0.05},
}


def _clean_roider_annotation_table(df, source):
    """Standardize Sample/Barcode/Population fields in Roider annotation CSV files."""
    required = {"Sample", "Barcode", "Population"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required {source} annotation columns are missing: {sorted(missing)}")
    out = df.loc[:, ["Sample", "Barcode", "Population"]].copy()
    out["Sample"] = out["Sample"].astype(str).str.strip()
    out["Barcode"] = out["Barcode"].astype(str).str.strip().str.split("-").str[0]
    out["Population"] = out["Population"].astype(str).str.strip()
    out.loc[out["Population"].isin(["", "nan", "None", "<NA>"]), "Population"] = np.nan
    out["annotation_source"] = source
    out["Sample_Barcode"] = out["Sample"] + "_" + out["Barcode"]
    return out


def ensure_roider_curated_annotations(force=False):
    """Preserve original Roider annotations and save curated annotations excluding conflicts."""
    if not USE_CURATED_ROIDER_ANNOTATIONS:
        return {
            "b_annotation": B_ANNOTATION_RAW,
            "t_annotation": T_ANNOTATION_RAW,
            "summary": None,
            "conflicts": None,
        }
    summary_path = ROIDER_CURATED_ANNOTATION_DIR / "annotation_curation_summary.csv"
    conflict_path = ROIDER_CURATED_ANNOTATION_DIR / "annotation_conflicting_B_T_keys.csv"
    if not force and B_ANNOTATION_CURATED.exists() and T_ANNOTATION_CURATED.exists() and summary_path.exists():
        return {
            "b_annotation": B_ANNOTATION_CURATED,
            "t_annotation": T_ANNOTATION_CURATED,
            "summary": summary_path,
            "conflicts": conflict_path,
        }

    ROIDER_CURATED_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    b_raw = _clean_roider_annotation_table(pd.read_csv(B_ANNOTATION_RAW, sep=";"), "B_annotation")
    t_raw = _clean_roider_annotation_table(pd.read_csv(T_ANNOTATION_RAW, sep=";"), "T_annotation")
    anno = pd.concat([b_raw, t_raw], ignore_index=True, sort=False)

    exact_duplicate_mask = anno.duplicated(["annotation_source", "Sample_Barcode", "Population"], keep="first")
    anno_dedup = anno.loc[~exact_duplicate_mask].copy()
    key_summary = (
        anno_dedup
        .groupby("Sample_Barcode", observed=False)
        .agg(
            n_rows=("Sample_Barcode", "size"),
            n_sources=("annotation_source", "nunique"),
            n_populations=("Population", "nunique"),
            source_set=("annotation_source", lambda s: "|".join(sorted(set(map(str, s))))),
            population_set=("Population", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    conflict_keys = set(
        key_summary.loc[
            (key_summary["n_sources"] > 1) | (key_summary["n_populations"] > 1),
            "Sample_Barcode",
        ]
    )
    conflicts = anno_dedup.loc[anno_dedup["Sample_Barcode"].isin(conflict_keys)].copy()
    curated = anno_dedup.loc[~anno_dedup["Sample_Barcode"].isin(conflict_keys)].copy()

    curated_b = curated.loc[curated["annotation_source"].eq("B_annotation"), ["Sample", "Barcode", "Population"]].copy()
    curated_t = curated.loc[curated["annotation_source"].eq("T_annotation"), ["Sample", "Barcode", "Population"]].copy()
    curated_b.to_csv(B_ANNOTATION_CURATED, sep=";", index=False)
    curated_t.to_csv(T_ANNOTATION_CURATED, sep=";", index=False)
    conflicts.to_csv(conflict_path, sep=";", index=False)

    summary = pd.DataFrame(
        [{
            "b_raw_path": str(B_ANNOTATION_RAW),
            "t_raw_path": str(T_ANNOTATION_RAW),
            "b_curated_path": str(B_ANNOTATION_CURATED),
            "t_curated_path": str(T_ANNOTATION_CURATED),
            "n_raw_annotation_rows": int(len(anno)),
            "n_raw_unique_keys": int(anno["Sample_Barcode"].nunique()),
            "n_exact_duplicate_rows_removed": int(exact_duplicate_mask.sum()),
            "n_conflict_keys_excluded": int(len(conflict_keys)),
            "n_conflict_rows_excluded": int(len(conflicts)),
            "n_curated_annotation_rows": int(len(curated_b) + len(curated_t)),
            "n_curated_unique_keys": int(curated["Sample_Barcode"].nunique()),
            "curation_rule": "strip Sample/Barcode/Population; remove exact duplicate rows; exclude keys assigned by both B and T annotations or assigned to multiple populations",
        }]
    )
    summary.to_csv(summary_path, index=False)
    conflicts.groupby(["annotation_source", "Population"], dropna=False, observed=False).size().reset_index(name="n").to_csv(
        ROIDER_CURATED_ANNOTATION_DIR / "annotation_conflict_population_counts.csv",
        index=False,
    )
    return {
        "b_annotation": B_ANNOTATION_CURATED,
        "t_annotation": T_ANNOTATION_CURATED,
        "summary": summary_path,
        "conflicts": conflict_path,
    }


def ensure_direct_transfer_dirs(out_dir=OUT_DIR):
    """Create the set of direct-transfer output directories."""
    out_dir = Path(out_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GSE182434_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if SCRNA_FIGURE_ONLY_WRITE_GUARD:
        if SCRNA_WRITE_AUDIT_TABLES:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "tables").mkdir(parents=True, exist_ok=True)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    ROIDER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GSE182434_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_output_dirs():
    """Create Roider direct-transfer output directories."""
    if not SCRNA_FIGURE_ONLY_WRITE_GUARD:
        ensure_roider_curated_annotations()
    return ensure_direct_transfer_dirs(OUT_DIR)


def ensure_gse182434_output_dirs():
    """Create GSE182434 direct-transfer output directories."""
    return ensure_direct_transfer_dirs(GSE182434_OUT_DIR)


def ensure_output_table_file(out_dir, filename, fallback_dirs=None):
    """Ensure auxiliary CSV files exist under standard Final tables paths even when using caches."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    target = out_dir / "tables" / filename
    if target.exists():
        return target

    candidates = [out_dir / filename]
    for fallback_dir in fallback_dirs or []:
        fallback_dir = Path(fallback_dir)
        candidates.extend([fallback_dir / "tables" / filename, fallback_dir / filename])

    for candidate in candidates:
        if candidate.exists():
            if SCRNA_WRITE_AUDIT_TABLES:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target)
                return target
            return candidate
    return None


def make_h5ad_string_safe(a):
    """Convert pandas nullable strings to h5ad-compatible objects before saving AnnData."""
    for attr in ["obs", "var"]:
        df = getattr(a, attr)
        for col in list(df.columns):
            series = df[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                categories = series.cat.categories
                if is_string_dtype(categories.dtype):
                    df[col] = series.astype(object).where(series.notna(), np.nan)
            elif is_string_dtype(series.dtype):
                df[col] = series.astype(object).where(series.notna(), np.nan)
    return a


def write_h5ad_safe(a, path, cache_stage=None):
    """Save h5ad while avoiding nullable-string errors."""
    if _scrna_should_block_nonfigure_write(path):
        raise PermissionError(
            f"scRNA Figure-only guard blocked protected H5AD write: {path}"
        )
    if cache_stage is not None:
        a.uns[CACHE_STAGE_KEY] = cache_stage
    make_h5ad_string_safe(a)
    a.write_h5ad(path)


def read_h5ad_cache(path, expected_stage=None):
    """Read only caches marked complete; return None to rebuild incomplete caches."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        a = sc.read_h5ad(path)
    except Exception as e:
        print(f"Could not read the existing cache; rebuilding: {path} ({type(e).__name__}: {e})")
        return None
    if expected_stage is not None and a.uns.get(CACHE_STAGE_KEY) != expected_stage:
        print(f"Ignoring and rebuilding a cache without a completion marker: {path}")
        return None
    return a


def to_numpy_1d(x):
    """Convert sparse or matrix results to a one-dimensional array."""
    if hasattr(x, "A1"):
        return x.A1
    return np.asarray(x).ravel()


def read_one_roider_sample(sample_row):
    """Read one sample's 10x matrix and attach metadata."""
    sample = sample_row.SampleName
    matrix_dir = ROIDER_DIR / sample / sample
    a = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", make_unique=True, cache=False)
    clean_barcodes = [bc.split("-")[0] for bc in a.obs_names]
    a.obs_names = [f"{sample}_{bc}" for bc in clean_barcodes]
    a.obs["Sample"] = sample
    a.obs["Barcode"] = clean_barcodes
    a.obs["Sample_Barcode"] = a.obs_names
    for col in ["Tissue", "Sex", "Entity", "Diagnosis", "Subclassification", "Batch", "CellrangerVersion"]:
        if hasattr(sample_row, col):
            a.obs[col] = getattr(sample_row, col)
    return a


def load_roider_raw(force_reload=False):
    """Read Roider 10x matrices or load the raw cache."""
    ensure_output_dirs()
    sample_sheet = pd.read_csv(SAMPLE_SHEET)
    sample_sheet["matrix_dir"] = sample_sheet["SampleName"].apply(lambda s: ROIDER_DIR / s / s)
    sample_sheet["matrix_exists"] = sample_sheet["matrix_dir"].apply(lambda p: (p / "matrix.mtx").exists())
    sample_sheet.to_csv(OUT_DIR / "tables" / "sample_sheet_checked.csv", index=False)
    if not sample_sheet["matrix_exists"].all():
        missing = sample_sheet.loc[~sample_sheet["matrix_exists"], ["SampleName", "matrix_dir"]]
        raise FileNotFoundError("matrix.mtx was not found for some samples.\n" + missing.to_string(index=False))

    if not force_reload:
        a = read_h5ad_cache(RAW_H5AD, expected_stage=ROIDER_RAW_CACHE_STAGE)
        if a is not None:
            attach_reference_annotations(a)
            return a, sample_sheet
        print("Raw cache is missing or incomplete; rebuilding.")
    else:
        print("FORCE_RELOAD_RAW=True: rebuilding the raw cache.")

    adatas = []
    for row in tqdm(list(sample_sheet.itertuples(index=False)), desc="Load Roider 10x samples"):
        adatas.append(read_one_roider_sample(row))
    a = ad.concat(adatas, join="outer", merge="same", index_unique=None)
    a.obs_names_make_unique()
    attach_reference_annotations(a)
    write_h5ad_safe(a, RAW_H5AD, cache_stage=ROIDER_RAW_CACHE_STAGE)
    return a, sample_sheet


def _clean_annotation_population(df):
    """Clean the annotation Population column without nullable strings."""
    out = df.copy()
    out["Population"] = out["Population"].astype(str).str.strip()
    out.loc[out["Population"].isin(["", "nan", "None", "<NA>"]), "Population"] = np.nan
    return out


def _combine_reference_labels(b_ref, t_ref):
    """Combine B/T reference labels using object dtype."""
    b = pd.Series(b_ref, index=b_ref.index, dtype=object)
    t = pd.Series(t_ref, index=t_ref.index, dtype=object)
    out = b.where(pd.notna(b), t)
    conflict = b.notna() & t.notna() & b.astype(str).ne(t.astype(str))
    out.loc[conflict] = np.nan
    return out


def _map_or_nan(series, mapping):
    """Leave unmatched mapping results as np.nan."""
    return series.map(lambda x: mapping.get(str(x), np.nan) if pd.notna(x) else np.nan).astype(object)


def _map_population_display(series):
    """Normalize TTOX to CD8T for analytical display."""
    return series.map(lambda x: T_SUBTYPE_DISPLAY_MAP.get(str(x), str(x)) if pd.notna(x) else np.nan).astype(object)


def attach_reference_annotations(a):
    """Attach B/T annotation CSV files to obs and create broad/subtype references."""
    ensure_roider_curated_annotations()
    b_anno = _clean_roider_annotation_table(pd.read_csv(B_ANNOTATION, sep=";"), "B_annotation")
    t_anno = _clean_roider_annotation_table(pd.read_csv(T_ANNOTATION, sep=";"), "T_annotation")

    a.obs["B_reference"] = a.obs["Sample_Barcode"].map(b_anno.set_index("Sample_Barcode")["Population"].to_dict())
    a.obs["T_reference"] = a.obs["Sample_Barcode"].map(t_anno.set_index("Sample_Barcode")["Population"].to_dict())
    b_ref = pd.Series(a.obs["B_reference"], index=a.obs.index, dtype=object)
    t_ref = pd.Series(a.obs["T_reference"], index=a.obs.index, dtype=object)
    conflict = b_ref.notna() & t_ref.notna() & b_ref.astype(str).ne(t_ref.astype(str))
    conflict_path = ROIDER_CURATED_ANNOTATION_DIR / "annotation_conflicting_B_T_keys.csv"
    if USE_CURATED_ROIDER_ANNOTATIONS and conflict_path.exists():
        conflict_keys = pd.read_csv(conflict_path, sep=";")["Sample_Barcode"].astype(str)
        conflict = conflict | a.obs["Sample_Barcode"].astype(str).isin(set(conflict_keys))
    a.obs["reference_population_conflict"] = conflict.astype(bool).to_numpy()
    raw_pop = _combine_reference_labels(a.obs["B_reference"], a.obs["T_reference"])
    a.obs["reference_population_raw"] = raw_pop
    a.obs["reference_population"] = _map_population_display(raw_pop)
    a.obs["reference_tcell_subtype"] = _map_or_nan(a.obs["reference_population"], {k: k for k in T_SUBTYPE_LABEL_ORDER})
    a.obs["reference_tcell_subtype_display"] = a.obs["reference_tcell_subtype"].astype(object)

    broad_map = {
        "MalignantB": "B_cells",
        "HealthyB": "B_cells",
        "Myeloid": "Myeloid_cells",
        "TH": "T_cells",
        "TFH": "T_cells",
        "TREG": "T_cells",
        "TTOX": "T_cells",
        "CD8T": "T_cells",
    }
    a.obs["reference_broad_display"] = _map_or_nan(raw_pop, broad_map)
    a.obs["reference_broad_codex"] = _map_or_nan(a.obs["reference_broad_display"], DISPLAY_TO_CODEX_BROAD_MAP)
    summary = (
        a.obs[[
            "Sample",
            "reference_population_raw",
            "reference_population",
            "reference_broad_display",
            "reference_broad_codex",
            "reference_tcell_subtype",
        ]]
        .value_counts(dropna=False)
        .reset_index(name="n")
    )
    summary.to_csv(OUT_DIR / "tables" / "reference_annotation_counts.csv", index=False)
    return summary


def audit_roider_reference_missing(a, out_dir=OUT_DIR, prefix="roider"):
    """Check whether missing Roider references reflect QC or annotation-join issues."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    attach_reference_annotations(a)
    b_anno = _clean_roider_annotation_table(pd.read_csv(B_ANNOTATION, sep=";"), "B_annotation")
    t_anno = _clean_roider_annotation_table(pd.read_csv(T_ANNOTATION, sep=";"), "T_annotation")
    annotation_ids = set(b_anno["Sample_Barcode"]) | set(t_anno["Sample_Barcode"])

    missing = a.obs["reference_broad_display"].isna()
    sample_barcode = a.obs["Sample_Barcode"].astype(str) if "Sample_Barcode" in a.obs else pd.Series(a.obs_names, index=a.obs.index, dtype=object)
    has_annotation_match = sample_barcode.isin(annotation_ids)
    conflict = a.obs["reference_population_conflict"].astype(bool) if "reference_population_conflict" in a.obs else pd.Series(False, index=a.obs.index)
    reason = pd.Series("reference_labeled", index=a.obs.index, dtype=object)
    reason.loc[missing & ~has_annotation_match] = "no_annotation_barcode_match"
    reason.loc[missing & has_annotation_match] = "annotation_matched_but_unmapped"
    reason.loc[missing & conflict] = "annotation_b_t_conflict_excluded"

    summary = pd.DataFrame(
        [{
            "n_cells": int(a.n_obs),
            "n_reference_missing": int(missing.sum()),
            "reference_missing_fraction": float(missing.mean()),
            "n_missing_no_annotation_barcode_match": int((missing & ~has_annotation_match).sum()),
            "n_missing_annotation_matched_but_unmapped": int((missing & has_annotation_match).sum()),
            "n_missing_annotation_b_t_conflict_excluded": int((missing & conflict).sum()),
            "b_annotation_unique_barcodes": int(b_anno["Sample_Barcode"].nunique()),
            "t_annotation_unique_barcodes": int(t_anno["Sample_Barcode"].nunique()),
            "b_annotation_path": str(B_ANNOTATION),
            "t_annotation_path": str(T_ANNOTATION),
        }]
    )
    summary.to_csv(out_dir / "tables" / f"{prefix}_reference_missing_audit_summary.csv", index=False)

    audit_obs = a.obs.copy()
    audit_obs["reference_missing"] = missing
    audit_obs["has_annotation_barcode_match"] = has_annotation_match.astype(bool).to_numpy()
    audit_obs["reference_population_conflict"] = conflict.astype(bool).to_numpy()
    audit_obs["reference_missing_reason"] = reason
    audit_obs["missing_no_annotation_barcode_match"] = (missing & ~has_annotation_match).astype(bool).to_numpy()
    audit_obs["missing_annotation_matched_but_unmapped"] = (missing & has_annotation_match).astype(bool).to_numpy()
    audit_obs["missing_annotation_b_t_conflict_excluded"] = (missing & conflict).astype(bool).to_numpy()
    if "Sample" in audit_obs:
        by_sample = (
            audit_obs
            .groupby("Sample", observed=False)
            .agg(
                n_cells=("reference_missing", "size"),
                n_reference_missing=("reference_missing", "sum"),
                reference_missing_fraction=("reference_missing", "mean"),
                n_missing_no_annotation_barcode_match=("missing_no_annotation_barcode_match", "sum"),
                n_missing_annotation_matched_but_unmapped=("missing_annotation_matched_but_unmapped", "sum"),
            )
            .reset_index()
            .sort_values("reference_missing_fraction", ascending=False)
        )
        by_sample.to_csv(out_dir / "tables" / f"{prefix}_reference_missing_audit_by_sample.csv", index=False)

    qc_cols = [c for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"] if c in audit_obs]
    if qc_cols:
        qc_summary = audit_obs.groupby("reference_missing", observed=False)[qc_cols].agg(["count", "median", "mean", "min", "max"])
        qc_summary.to_csv(out_dir / "tables" / f"{prefix}_reference_missing_qc_summary.csv")

    missing_cells_cols = [
        c for c in [
            "Sample", "Barcode", "Sample_Barcode", "reference_missing_reason",
            "B_reference", "T_reference", "reference_population_raw",
            "n_genes_by_counts", "total_counts", "pct_counts_mt",
        ] if c in audit_obs
    ]
    audit_obs.loc[missing, missing_cells_cols].to_csv(
        out_dir / "tables" / f"{prefix}_reference_missing_cells.csv.gz",
        index=True,
        compression="gzip",
    )
    return summary


def _roider_annotation_sample_barcode(df):
    """Align Sample/Barcode in Roider annotations to the raw 10x key format."""
    sample = df["Sample"].astype(str).str.strip()
    barcode = df["Barcode"].astype(str).str.strip().str.split("-").str[0]
    return sample + "_" + barcode


def audit_roider_annotation_key_coverage(out_dir=OUT_DIR, prefix="roider"):
    """Audit actual matches between Roider 10x barcodes and B/T annotation keys."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    sample_sheet = pd.read_csv(SAMPLE_SHEET)

    raw_rows = []
    for row in tqdm(list(sample_sheet.itertuples(index=False)), desc="Audit Roider raw barcode keys"):
        sample = str(row.SampleName)
        barcode_path = ROIDER_DIR / sample / sample / "barcodes.tsv"
        if not barcode_path.exists():
            raw_rows.append(
                {
                    "Sample": sample,
                    "Barcode": np.nan,
                    "Sample_Barcode": np.nan,
                    "barcode_file_exists": False,
                }
            )
            continue
        barcodes = pd.read_csv(barcode_path, header=None, sep="\t")[0].astype(str).str.split("-").str[0]
        raw_rows.extend(
            {
                "Sample": sample,
                "Barcode": barcode,
                "Sample_Barcode": f"{sample}_{barcode}",
                "barcode_file_exists": True,
            }
            for barcode in barcodes
        )
    raw_keys = pd.DataFrame(raw_rows)

    ensure_roider_curated_annotations()
    b_anno = _clean_roider_annotation_table(pd.read_csv(B_ANNOTATION, sep=";"), "B_annotation")
    t_anno = _clean_roider_annotation_table(pd.read_csv(T_ANNOTATION, sep=";"), "T_annotation")
    anno = pd.concat([b_anno, t_anno], ignore_index=True, sort=False)

    raw_key_set = set(raw_keys.loc[raw_keys["barcode_file_exists"], "Sample_Barcode"].dropna())
    anno_key_set = set(anno["Sample_Barcode"].dropna())
    raw_keys["has_annotation"] = raw_keys["Sample_Barcode"].isin(anno_key_set)
    anno["matches_raw_10x_barcode"] = anno["Sample_Barcode"].isin(raw_key_set)

    broad_map = {
        "MalignantB": "B_cells",
        "HealthyB": "B_cells",
        "Myeloid": "Myeloid_cells",
        "TH": "T_cells",
        "TFH": "T_cells",
        "TREG": "T_cells",
        "TTOX": "T_cells",
        "CD8T": "T_cells",
    }
    anno["reference_broad_display"] = _map_or_nan(anno["Population"], broad_map)
    anno["population_mapped_to_broad"] = anno["reference_broad_display"].notna()

    summary = pd.DataFrame(
        [{
            "n_raw_10x_cells": int(raw_keys["Sample_Barcode"].notna().sum()),
            "n_annotation_rows": int(len(anno)),
            "n_unique_annotation_keys": int(anno["Sample_Barcode"].nunique()),
            "n_duplicate_annotation_key_rows": int(len(anno) - anno["Sample_Barcode"].nunique()),
            "n_annotation_rows_matching_raw_10x": int(anno["matches_raw_10x_barcode"].sum()),
            "n_annotation_rows_not_matching_raw_10x": int((~anno["matches_raw_10x_barcode"]).sum()),
            "n_raw_cells_with_any_annotation": int(raw_keys["has_annotation"].sum()),
            "n_raw_cells_without_annotation": int((~raw_keys["has_annotation"]).sum()),
            "raw_cells_without_annotation_fraction": float((~raw_keys["has_annotation"]).mean()),
            "n_annotation_rows_unmapped_population": int((~anno["population_mapped_to_broad"]).sum()),
            "unmapped_population_values": "|".join(sorted(anno.loc[~anno["population_mapped_to_broad"], "Population"].dropna().astype(str).unique())),
        }]
    )

    raw_by_sample = (
        raw_keys
        .groupby("Sample", observed=False)
        .agg(
            n_raw_10x_cells=("Sample_Barcode", "count"),
            n_raw_cells_with_annotation=("has_annotation", "sum"),
        )
        .reset_index()
    )
    raw_by_sample["n_raw_cells_without_annotation"] = raw_by_sample["n_raw_10x_cells"] - raw_by_sample["n_raw_cells_with_annotation"]
    raw_by_sample["raw_cells_without_annotation_fraction"] = np.divide(
        raw_by_sample["n_raw_cells_without_annotation"],
        raw_by_sample["n_raw_10x_cells"],
        out=np.zeros(len(raw_by_sample), dtype=float),
        where=raw_by_sample["n_raw_10x_cells"].to_numpy() > 0,
    )
    anno_by_sample = (
        anno
        .groupby(["Sample", "annotation_source"], observed=False)
        .agg(
            n_annotation_rows=("Sample_Barcode", "count"),
            n_annotation_rows_matching_raw_10x=("matches_raw_10x_barcode", "sum"),
        )
        .reset_index()
    )
    by_sample = raw_by_sample.merge(anno_by_sample, on="Sample", how="outer").sort_values(
        ["raw_cells_without_annotation_fraction", "Sample"],
        ascending=[False, True],
    )

    by_population = (
        anno
        .groupby(["annotation_source", "Population", "reference_broad_display"], dropna=False, observed=False)
        .agg(
            n_annotation_rows=("Sample_Barcode", "count"),
            n_matching_raw_10x=("matches_raw_10x_barcode", "sum"),
        )
        .reset_index()
        .sort_values("n_annotation_rows", ascending=False)
    )

    summary.to_csv(out_dir / "tables" / f"{prefix}_annotation_key_coverage_summary.csv", index=False)
    by_sample.to_csv(out_dir / "tables" / f"{prefix}_annotation_key_coverage_by_sample.csv", index=False)
    by_population.to_csv(out_dir / "tables" / f"{prefix}_annotation_population_mapping.csv", index=False)
    return {"summary": summary, "by_sample": by_sample, "by_population": by_population}


def export_marker_base_marker_sets(out_dir=OUT_DIR):
    """Save marker sets and scRNA gene proxies actually used for Marker Base predictions."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    rows = []
    for task, marker_sets in [
        ("broad", BROAD_MARKER_SETS),
        ("tcell_subtype", T_SUBTYPE_MARKER_SETS),
    ]:
        for label, markers in marker_sets.items():
            for marker in markers:
                rows.append(
                    {
                        "task": task,
                        "label": label,
                        "codex_marker": marker,
                        "feature_col": _feature_col(marker),
                        "requested_genes": "|".join(CODEX_TO_SCRNA_GENES.get(marker, [marker])),
                    }
                )
    marker_sets = pd.DataFrame(rows)
    marker_sets.to_csv(out_dir / "tables" / "marker_base_marker_sets.csv", index=False)
    return marker_sets


def export_codex_model_coefficient_importance(package_path, task, out_dir=OUT_DIR, prefix=None):
    """Save class-specific direct-transfer classifier coefficients as marker-importance proxies."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    package = load_compatible_joblib(package_path)
    model = package["model"]
    classifier = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
    coef = getattr(classifier, "coef_", None)
    if coef is None:
        raise RuntimeError(f"Classifier coefficients not found: {package_path}")

    label_encoder = package["label_encoder"]
    feature_names = list(package.get("feature_names", []))
    feature_columns = list(package.get("feature_columns", [_feature_col(m) for m in feature_names]))
    rows = []
    for class_idx, raw_class in enumerate(label_encoder.classes_):
        if task == "broad":
            display_class = BROAD_DISPLAY_MAP.get(raw_class, raw_class)
        else:
            display_class = CODEX_T_SUBTYPE_TO_DISPLAY.get(raw_class, raw_class)
        for feature_idx, marker in enumerate(feature_names):
            rows.append(
                {
                    "task": task,
                    "class_raw": raw_class,
                    "class_display": display_class,
                    "codex_marker": marker,
                    "feature_col": feature_columns[feature_idx] if feature_idx < len(feature_columns) else _feature_col(marker),
                    "coefficient": float(coef[class_idx, feature_idx]),
                    "abs_coefficient": float(abs(coef[class_idx, feature_idx])),
                    "requested_genes": "|".join(CODEX_TO_SCRNA_GENES.get(marker, [marker])),
                }
            )
    importance = pd.DataFrame(rows)
    if prefix is None:
        prefix = f"codex_ml_{task}_coefficient_importance"
    importance.to_csv(out_dir / "tables" / f"{prefix}.csv", index=False)
    overall = (
        importance.groupby(["task", "codex_marker", "requested_genes"], as_index=False)["abs_coefficient"]
        .mean()
        .rename(columns={"abs_coefficient": "mean_abs_coefficient"})
        .sort_values("mean_abs_coefficient", ascending=False)
    )
    overall.to_csv(out_dir / "tables" / f"{prefix}_overall.csv", index=False)
    top_by_class = (
        importance.sort_values(["class_display", "abs_coefficient"], ascending=[True, False])
        .groupby("class_display", observed=False)
        .head(8)
    )
    top_by_class.to_csv(out_dir / "tables" / f"{prefix}_top_by_class.csv", index=False)
    return {"by_class": importance, "overall": overall, "top_by_class": top_by_class}


def export_all_marker_and_importance_tables(out_dir=OUT_DIR):
    """Update marker-set and CODEX ML coefficient-importance tables for the PPT/README."""
    marker_sets = export_marker_base_marker_sets(out_dir=out_dir)
    broad = export_codex_model_coefficient_importance(
        DIRECT_MODEL_PACKAGE,
        task="broad",
        out_dir=out_dir,
        prefix="codex_ml_broad_coefficient_importance",
    )
    tcell = export_codex_model_coefficient_importance(
        DIRECT_TCELL_SUBTYPE_MODEL_PACKAGE,
        task="tcell_subtype",
        out_dir=out_dir,
        prefix="codex_ml_tcell_subtype_coefficient_importance",
    )
    return {"marker_sets": marker_sets, "broad": broad, "tcell_subtype": tcell}


def _add_original_paper_qc_columns(obs, n_gene_col, pct_mito_col):
    """Annotate passage of sample-specific nGene/pc.mito thresholds from the original Roider Rmd."""
    obs = obs.copy()
    rules = pd.DataFrame.from_dict(ROIDER_ORIGINAL_PAPER_QC_RULES, orient="index")
    obs["paper_qc_low_gene"] = obs["Sample"].astype(str).map(rules["low_n_gene"]).astype(float)
    obs["paper_qc_high_gene"] = obs["Sample"].astype(str).map(rules["high_n_gene"]).astype(float)
    obs["paper_qc_max_pct_mito"] = obs["Sample"].astype(str).map(rules["max_pc_mito"]).astype(float) * 100.0
    n_genes = pd.to_numeric(obs[n_gene_col], errors="coerce")
    pct_mito = pd.to_numeric(obs[pct_mito_col], errors="coerce")
    obs["paper_qc_pass"] = (
        n_genes.ge(obs["paper_qc_low_gene"])
        & n_genes.le(obs["paper_qc_high_gene"])
        & pct_mito.le(obs["paper_qc_max_pct_mito"])
    )
    return obs


def _calculate_raw_qc_obs(a):
    """Calculate nGene, total counts, and mitochondrial percentage from the raw count matrix."""
    x = a.X
    if sp.issparse(x):
        n_genes = to_numpy_1d((x > 0).sum(axis=1))
        total_counts = to_numpy_1d(x.sum(axis=1))
    else:
        arr = np.asarray(x)
        n_genes = (arr > 0).sum(axis=1)
        total_counts = arr.sum(axis=1)

    mt_mask = np.asarray(pd.Index(a.var_names).astype(str).str.startswith("MT-"), dtype=bool)
    if mt_mask.any():
        if sp.issparse(x):
            mt_counts = to_numpy_1d(x[:, mt_mask].sum(axis=1))
        else:
            mt_counts = np.asarray(x[:, mt_mask]).sum(axis=1)
    else:
        mt_counts = np.zeros(a.n_obs, dtype=float)
    pct_mt = np.divide(mt_counts, total_counts, out=np.zeros_like(mt_counts, dtype=float), where=total_counts != 0) * 100.0

    obs = a.obs.copy()
    obs["n_genes_by_counts_calc"] = n_genes
    obs["total_counts_calc"] = total_counts
    obs["pct_counts_mt_calc"] = pct_mt
    return obs


def _summarize_missing_paper_qc(obs, stage, n_gene_col, total_col, pct_mito_col):
    """Cross-tabulate missing references against passage of the original study's QC criteria."""
    rows = []
    for (missing, paper_pass), sub in obs.groupby(["reference_missing", "paper_qc_pass"], dropna=False, observed=False):
        rows.append(
            {
                "dataset_stage": stage,
                "reference_missing": bool(missing),
                "paper_qc_pass": bool(paper_pass),
                "n_cells": int(len(sub)),
                "fraction_of_stage": float(len(sub) / len(obs)),
                "n_genes_median": float(pd.to_numeric(sub[n_gene_col], errors="coerce").median()),
                "total_counts_median": float(pd.to_numeric(sub[total_col], errors="coerce").median()),
                "pct_mt_median": float(pd.to_numeric(sub[pct_mito_col], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def audit_roider_missing_against_original_paper_qc(
    raw_h5ad=RAW_H5AD,
    predicted_h5ad=PREDICTED_H5AD,
    out_dir=OUT_DIR,
):
    """Test whether missing Roider references fail original Rmd QC criteria at raw/processed stages."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    pd.DataFrame.from_dict(ROIDER_ORIGINAL_PAPER_QC_RULES, orient="index").rename_axis("Sample").reset_index().to_csv(
        out_dir / "tables" / "roider_original_paper_qc_rules_from_rmd.csv",
        index=False,
    )

    raw = sc.read_h5ad(raw_h5ad)
    attach_reference_annotations(raw)
    raw_obs = _calculate_raw_qc_obs(raw)
    raw_obs["reference_missing"] = raw_obs["reference_broad_display"].isna()
    raw_obs = _add_original_paper_qc_columns(raw_obs, "n_genes_by_counts_calc", "pct_counts_mt_calc")

    predicted = ad.read_h5ad(predicted_h5ad, backed="r")
    predicted_obs = predicted.obs.copy()
    predicted.file.close()
    predicted_obs["reference_missing"] = predicted_obs["reference_broad_display"].isna()
    predicted_obs = _add_original_paper_qc_columns(predicted_obs, "n_genes_by_counts", "pct_counts_mt")

    summary = pd.concat(
        [
            _summarize_missing_paper_qc(raw_obs, "raw_loaded", "n_genes_by_counts_calc", "total_counts_calc", "pct_counts_mt_calc"),
            _summarize_missing_paper_qc(predicted_obs, "processed_predicted", "n_genes_by_counts", "total_counts", "pct_counts_mt"),
        ],
        ignore_index=True,
    )
    summary.to_csv(out_dir / "tables" / "roider_missing_original_paper_qc_audit_summary.csv", index=False)

    sample_rows = []
    for stage, obs, n_gene_col, total_col, pct_mito_col in [
        ("raw_loaded", raw_obs, "n_genes_by_counts_calc", "total_counts_calc", "pct_counts_mt_calc"),
        ("processed_predicted", predicted_obs, "n_genes_by_counts", "total_counts", "pct_counts_mt"),
    ]:
        for sample_name, sub in obs.groupby("Sample", observed=False):
            missing = sub["reference_missing"]
            paper_pass = sub["paper_qc_pass"]
            sample_rows.append(
                {
                    "dataset_stage": stage,
                    "Sample": sample_name,
                    "n_cells": int(len(sub)),
                    "n_reference_missing": int(missing.sum()),
                    "reference_missing_fraction": float(missing.mean()),
                    "n_paper_qc_fail": int((~paper_pass).sum()),
                    "paper_qc_fail_fraction": float((~paper_pass).mean()),
                    "n_missing_paper_qc_fail": int((missing & ~paper_pass).sum()),
                    "missing_paper_qc_fail_fraction_among_missing": float(((missing & ~paper_pass).sum() / missing.sum()) if missing.sum() else 0.0),
                    "n_missing_paper_qc_pass": int((missing & paper_pass).sum()),
                    "missing_n_genes_median": float(pd.to_numeric(sub.loc[missing, n_gene_col], errors="coerce").median()) if missing.any() else np.nan,
                    "missing_total_counts_median": float(pd.to_numeric(sub.loc[missing, total_col], errors="coerce").median()) if missing.any() else np.nan,
                    "missing_pct_mt_median": float(pd.to_numeric(sub.loc[missing, pct_mito_col], errors="coerce").median()) if missing.any() else np.nan,
                }
            )
    by_sample = pd.DataFrame(sample_rows)
    by_sample.to_csv(out_dir / "tables" / "roider_missing_original_paper_qc_audit_by_sample.csv", index=False)

    rln2_rows = []
    for stage, obs, n_gene_col, total_col in [
        ("raw_loaded", raw_obs, "n_genes_by_counts_calc", "total_counts_calc"),
        ("processed_predicted", predicted_obs, "n_genes_by_counts", "total_counts"),
    ]:
        sub = obs[obs["Sample"].astype(str).eq("rLN2")]
        missing = sub["reference_missing"]
        paper_pass = sub["paper_qc_pass"]
        rln2_rows.append(
            {
                "dataset_stage": stage,
                "n_rLN2_cells": int(len(sub)),
                "n_rLN2_missing": int(missing.sum()),
                "rLN2_missing_fraction": float(missing.mean()) if len(sub) else 0.0,
                "n_rLN2_missing_paper_qc_fail": int((missing & ~paper_pass).sum()),
                "n_rLN2_missing_paper_qc_pass": int((missing & paper_pass).sum()),
                "missing_n_genes_median": float(pd.to_numeric(sub.loc[missing, n_gene_col], errors="coerce").median()) if missing.any() else np.nan,
                "missing_total_counts_median": float(pd.to_numeric(sub.loc[missing, total_col], errors="coerce").median()) if missing.any() else np.nan,
            }
        )
    rln2 = pd.DataFrame(rln2_rows)
    rln2.to_csv(out_dir / "tables" / "roider_rln2_missing_original_annotation_note_audit.csv", index=False)
    return {"summary": summary, "by_sample": by_sample, "rln2": rln2}


def attach_gse182434_reference_annotations(a):
    """Normalize GSE182434 CellType annotations to CODEX broad-label format."""
    ensure_gse182434_output_dirs()
    if "CellType" not in a.obs and "codex_broad_candidate" not in a.obs:
        raise KeyError("GSE182434 AnnData lacks a CellType or codex_broad_candidate column.")

    if "CellType" in a.obs:
        raw_pop = a.obs["CellType"].astype(object).where(pd.notna(a.obs["CellType"]), np.nan)
        display = _map_or_nan(raw_pop, GSE182434_CELLTYPE_TO_DISPLAY)
        source_col = "CellType"
    else:
        raw_pop = a.obs["codex_broad_candidate"].astype(object).where(pd.notna(a.obs["codex_broad_candidate"]), np.nan)
        display = _map_or_nan(raw_pop, GSE182434_CANDIDATE_TO_DISPLAY)
        source_col = "codex_broad_candidate"

    if "codex_broad_candidate" in a.obs:
        candidate = a.obs["codex_broad_candidate"].astype(object).where(pd.notna(a.obs["codex_broad_candidate"]), np.nan)
        a.obs["gse182434_broad_candidate_display"] = _map_or_nan(candidate, GSE182434_CANDIDATE_TO_DISPLAY)

    a.obs["reference_population_raw"] = raw_pop
    a.obs["reference_population"] = raw_pop
    a.obs["reference_broad_display"] = display
    a.obs["reference_broad_codex"] = _map_or_nan(display, DISPLAY_TO_CODEX_BROAD_MAP)
    a.obs["reference_tcell_subtype"] = _map_or_nan(raw_pop, GSE182434_CELLTYPE_TO_T_SUBTYPE)
    a.obs["reference_tcell_subtype_display"] = a.obs["reference_tcell_subtype"].astype(object)
    a.uns["gse182434_reference_source_column"] = source_col

    summary_cols = ["reference_population_raw", "reference_broad_display", "reference_broad_codex", "reference_tcell_subtype"]
    for col in ["Sample", "Patient", "TumorNormal", "Tissue", "COO"]:
        if col in a.obs and col not in summary_cols:
            summary_cols.insert(0, col)
    summary = a.obs[summary_cols].value_counts(dropna=False).reset_index(name="n")
    summary.to_csv(GSE182434_OUT_DIR / "tables" / "reference_annotation_counts.csv", index=False)
    return summary


def audit_gse182434_reference_missing(a, out_dir=GSE182434_OUT_DIR, prefix="gse182434", include_raw_qc=True):
    """Separate and save missing GSE182434 references due to QC filtering versus annotation mapping."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    if "reference_broad_codex" not in a.obs or "reference_tcell_subtype" not in a.obs:
        attach_gse182434_reference_annotations(a)

    obs = a.obs.copy()
    source_col = a.uns.get("gse182434_reference_source_column")
    if source_col not in obs:
        source_col = "CellType" if "CellType" in obs else "codex_broad_candidate" if "codex_broad_candidate" in obs else "reference_population_raw"
    source_label = obs[source_col].astype(object).where(pd.notna(obs[source_col]), np.nan)

    broad_eval = obs["reference_broad_codex"].isin(CODEX_BROAD_LABEL_ORDER)
    t_eval = obs["reference_tcell_subtype"].isin(T_SUBTYPE_LABEL_ORDER)
    raw_missing = source_label.isna()
    non_t = obs["reference_broad_codex"].isin([label for label in CODEX_BROAD_LABEL_ORDER if label != "T cells"])
    broad_t = obs["reference_broad_codex"].eq("T cells")

    broad_reason = pd.Series("evaluable_broad_label", index=obs.index, dtype=object)
    broad_reason.loc[raw_missing] = "source_annotation_missing"
    broad_reason.loc[(~raw_missing) & (~broad_eval)] = "source_annotation_present_but_unmapped_to_codex_broad"

    t_reason = pd.Series("evaluable_tcell_subtype", index=obs.index, dtype=object)
    t_reason.loc[raw_missing] = "source_annotation_missing"
    t_reason.loc[(~raw_missing) & (~broad_eval)] = "source_annotation_present_but_unmapped_to_codex_broad"
    t_reason.loc[non_t] = "non_t_cell_not_in_subtype_evaluation"
    t_reason.loc[broad_t & (~t_eval)] = "t_cell_annotation_present_but_subtype_unmapped"

    audit = pd.DataFrame(
        {
            "cell_id": obs.index.astype(str),
            "source_column": source_col,
            "source_label": source_label.to_numpy(dtype=object),
            "reference_broad_codex": obs["reference_broad_codex"].to_numpy(dtype=object),
            "reference_tcell_subtype": obs["reference_tcell_subtype"].to_numpy(dtype=object),
            "broad_missing_reason": broad_reason.to_numpy(dtype=object),
            "tcell_subtype_missing_reason": t_reason.to_numpy(dtype=object),
        },
        index=obs.index,
    )
    for col in ["Sample", "Patient", "TumorNormal", "Tissue", "COO"]:
        if col in obs:
            audit[col] = obs[col].astype(object).to_numpy()

    summary_values = {
        "dataset": "GSE182434 scRNA",
        "dataset_stage": "qc_processed_current",
        "source_column": source_col,
        "n_cells": int(len(obs)),
        "n_broad_reference_evaluable": int(broad_eval.sum()),
        "n_broad_reference_missing_or_unmapped": int((~broad_eval).sum()),
        "n_broad_source_annotation_missing": int((broad_reason == "source_annotation_missing").sum()),
        "n_broad_annotation_present_but_unmapped": int((broad_reason == "source_annotation_present_but_unmapped_to_codex_broad").sum()),
        "n_tcell_subtype_reference_evaluable": int(t_eval.sum()),
        "n_tcell_subtype_missing_or_unmapped": int((~t_eval).sum()),
        "n_tcell_non_t_excluded_from_subtype": int((t_reason == "non_t_cell_not_in_subtype_evaluation").sum()),
        "n_tcell_broad_unmapped_or_others": int((t_reason == "source_annotation_present_but_unmapped_to_codex_broad").sum()),
        "n_tcell_source_annotation_missing": int((t_reason == "source_annotation_missing").sum()),
        "n_tcell_annotation_present_but_subtype_unmapped": int((t_reason == "t_cell_annotation_present_but_subtype_unmapped").sum()),
    }

    qc_path = out_dir / "tables" / "qc_filtering_summary.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        if not qc.empty:
            for col in ["n_cells_before_qc", "n_cells_after_qc", "removed_cells"]:
                if col in qc:
                    summary_values[col] = qc[col].iloc[0]

    raw_summary = None
    if include_raw_qc:
        try:
            raw_annotation = load_gse182434_cell_annotation().set_index("ID")
            raw_obs = raw_annotation.copy()
            raw_source_col = "CellType" if "CellType" in raw_obs else None
            if raw_source_col is not None:
                raw_label = raw_obs[raw_source_col].astype(object).where(pd.notna(raw_obs[raw_source_col]), np.nan)
                raw_display = _map_or_nan(raw_label, GSE182434_CELLTYPE_TO_DISPLAY)
                raw_broad = _map_or_nan(raw_display, DISPLAY_TO_CODEX_BROAD_MAP)
                raw_tcell = _map_or_nan(raw_label, GSE182434_CELLTYPE_TO_T_SUBTYPE)
                raw_kept = raw_obs.index.isin(obs.index)
                raw_summary = pd.DataFrame(
                    [{
                        "dataset": "GSE182434 scRNA",
                        "dataset_stage": "raw_annotation_text",
                        "source_annotation": str(GSE182434_CELL_ANNOTATION),
                        "source_column": raw_source_col,
                        "n_cells": int(len(raw_obs)),
                        "n_cells_kept_after_current_qc": int(raw_kept.sum()),
                        "n_cells_removed_by_current_qc": int((~raw_kept).sum()),
                        "n_broad_reference_evaluable": int(raw_broad.isin(CODEX_BROAD_LABEL_ORDER).sum()),
                        "n_broad_reference_missing_or_unmapped": int((~raw_broad.isin(CODEX_BROAD_LABEL_ORDER)).sum()),
                        "n_source_annotation_missing": int(raw_label.isna().sum()),
                        "n_annotation_present_but_unmapped_to_codex_broad": int(((~raw_label.isna()) & (~raw_broad.isin(CODEX_BROAD_LABEL_ORDER))).sum()),
                        "n_tcell_subtype_reference_evaluable": int(raw_tcell.isin(T_SUBTYPE_LABEL_ORDER).sum()),
                    }]
                )
                raw_by_label = pd.DataFrame(
                    {
                        "source_label": raw_label,
                        "reference_broad_codex": raw_broad,
                        "reference_tcell_subtype": raw_tcell,
                        "kept_after_current_qc": raw_kept,
                    }
                ).value_counts(dropna=False).reset_index(name="n")
                raw_by_label.to_csv(out_dir / "tables" / f"{prefix}_raw_vs_qc_reference_missing_by_source_label.csv", index=False)
        except Exception as exc:
            raw_summary = pd.DataFrame(
                [{
                    "dataset": "GSE182434 scRNA",
                    "dataset_stage": "raw_annotation_text",
                    "raw_qc_audit_error": str(exc),
                }]
            )

    summary = pd.DataFrame([summary_values])
    if raw_summary is not None:
        summary = pd.concat([summary, raw_summary], ignore_index=True, sort=False)

    group_cols = ["source_column", "source_label", "reference_broad_codex", "reference_tcell_subtype", "broad_missing_reason", "tcell_subtype_missing_reason"]
    by_raw_label = audit[group_cols].value_counts(dropna=False).reset_index(name="n")
    sample_cols = [col for col in ["Sample", "Patient", "TumorNormal", "Tissue", "COO"] if col in audit]
    if sample_cols:
        by_sample_broad = audit[sample_cols + ["broad_missing_reason"]].value_counts(dropna=False).reset_index(name="n")
        by_sample_broad["task"] = "broad"
        by_sample_broad = by_sample_broad.rename(columns={"broad_missing_reason": "reason"})
        by_sample_tcell = audit[sample_cols + ["tcell_subtype_missing_reason"]].value_counts(dropna=False).reset_index(name="n")
        by_sample_tcell["task"] = "tcell_subtype"
        by_sample_tcell = by_sample_tcell.rename(columns={"tcell_subtype_missing_reason": "reason"})
        by_sample = pd.concat([by_sample_broad, by_sample_tcell], ignore_index=True, sort=False)
    else:
        by_sample = pd.DataFrame(columns=["task", "reason", "n"])

    summary.to_csv(out_dir / "tables" / f"{prefix}_reference_missing_audit_summary.csv", index=False)
    by_raw_label.to_csv(out_dir / "tables" / f"{prefix}_reference_missing_audit_by_source_label.csv", index=False)
    by_sample.to_csv(out_dir / "tables" / f"{prefix}_reference_missing_audit_by_sample.csv", index=False)
    return {"summary": summary, "by_source_label": by_raw_label, "by_sample": by_sample, "cells": audit}


def load_gse182434_preprocessed():
    """Reconstruct QC/UMAP processing from GSE182434 raw text."""
    return preprocess_gse182434_for_transfer(force_reprocess=True)


def load_gse182434_cell_annotation():
    """Read the original GSE182434 cell-annotation text."""
    if not GSE182434_CELL_ANNOTATION.exists():
        raise FileNotFoundError(f"Original GSE182434 cell annotations not found: {GSE182434_CELL_ANNOTATION}")
    annotation = pd.read_csv(GSE182434_CELL_ANNOTATION, sep="\t")
    annotation.columns = [str(col).strip() for col in annotation.columns]
    if "ID" not in annotation.columns:
        raise ValueError(f"GSE182434 annotations lack an ID column: {GSE182434_CELL_ANNOTATION}")
    annotation["ID"] = annotation["ID"].astype(str)
    return annotation


def load_gse182434_raw_counts_from_text():
    """Create AnnData from GSE182434 raw count and annotation text files."""
    if not GSE182434_RAW_MATRIX.exists():
        raise FileNotFoundError(f"GSE182434 raw count matrix not found: {GSE182434_RAW_MATRIX}")

    print(f"Loading GSE182434 raw count matrix: {GSE182434_RAW_MATRIX}")
    counts = pd.read_csv(GSE182434_RAW_MATRIX, sep="\t", index_col=0)
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)

    x = sp.csr_matrix(counts.T.to_numpy(dtype=np.float32, copy=False))
    a = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=pd.Index(counts.columns, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(counts.index, name="gene")),
    )
    a.obs_names_make_unique()
    a.var_names_make_unique()

    annotation = load_gse182434_cell_annotation().set_index("ID")
    a.obs = a.obs.join(annotation, how="left")
    a.uns["raw_source_matrix"] = str(GSE182434_RAW_MATRIX)
    a.uns["raw_source_annotation"] = str(GSE182434_CELL_ANNOTATION)
    return a


def _gse182434_full_gene_source(a):
    """Recover the full-gene log-normalized matrix from GSE182434 h5ad."""
    if a.raw is not None and a.raw.shape[1] > a.n_vars:
        full = a.raw.to_adata()
        full.obs = a.obs.copy()
        full.uns = a.uns.copy()
        return full
    return a.copy()


def preprocess_gse182434_for_transfer(force_reprocess=False):
    """Recompute QC filtering, normalization, and PCA/UMAP from original GSE182434 count text."""
    ensure_gse182434_output_dirs()
    if not force_reprocess:
        out = read_h5ad_cache(GSE182434_QC_H5AD, expected_stage="gse182434_qc_umap_v1")
        if out is not None:
            for filename in ["qc_filtering_summary.csv", "qc_before_filtering_by_sample.csv"]:
                ensure_output_table_file(
                    GSE182434_OUT_DIR,
                    filename,
                    fallback_dirs=[SOURCE_GSE182434_DIRECT_OUT_DIR, GSE182434_QC_H5AD.parent],
                )
            summary = attach_gse182434_reference_annotations(out)
            audit_gse182434_reference_missing(out, out_dir=GSE182434_OUT_DIR, prefix="gse182434_qc")
            return out, summary

    source_path = GSE182434_RAW_MATRIX
    a0 = load_gse182434_raw_counts_from_text()
    attach_gse182434_reference_annotations(a0)

    required_qc_cols = {"n_genes_by_counts", "total_counts", "pct_counts_mt"}
    if not required_qc_cols.issubset(a0.obs.columns):
        add_qc_metrics(a0)

    qc_before = a0.obs.groupby("Sample")[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe()
    qc_before.to_csv(GSE182434_OUT_DIR / "tables" / "qc_before_filtering_by_sample.csv")

    sample_key = a0.obs["Sample"].astype("string") if "Sample" in a0.obs else pd.Series("GSE182434", index=a0.obs.index, dtype="string")
    high_gene_cutoff = a0.obs.groupby(sample_key)["n_genes_by_counts"].quantile(0.995).astype(float)
    a0.obs["sample_high_gene_cutoff"] = sample_key.map(high_gene_cutoff).astype("float64")
    a0.obs["sample_high_gene_cutoff"] = a0.obs["sample_high_gene_cutoff"].fillna(float(a0.obs["n_genes_by_counts"].quantile(0.995)))
    qc_mask = (
        (a0.obs["n_genes_by_counts"] >= 200)
        & (a0.obs["pct_counts_mt"] <= 5)
        & (a0.obs["n_genes_by_counts"] <= a0.obs["sample_high_gene_cutoff"])
    )
    qc_summary = pd.DataFrame(
        [{
            "input_raw_matrix": str(source_path),
            "input_cell_annotation": str(GSE182434_CELL_ANNOTATION),
            "n_cells_before_qc": int(a0.n_obs),
            "n_cells_after_qc": int(qc_mask.sum()),
            "removed_cells": int((~qc_mask).sum()),
            "min_n_genes_by_counts": 200,
            "max_pct_counts_mt": 5,
            "high_gene_cutoff_rule": "per-sample n_genes_by_counts 99.5 percentile",
        }]
    )
    qc_summary.to_csv(GSE182434_OUT_DIR / "tables" / "qc_filtering_summary.csv", index=False)

    a = a0[qc_mask.to_numpy()].copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a.copy()
    hvg_kwargs = {"n_top_genes": 3000}
    if "Sample" in a.obs:
        hvg_kwargs["batch_key"] = "Sample"
    sc.pp.highly_variable_genes(a, **hvg_kwargs)
    a = a[:, np.asarray(a.var["highly_variable"]).astype(bool)].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=40)
    sc.tl.umap(a, random_state=0)
    try:
        sc.tl.leiden(a, resolution=0.8, key_added="leiden_0_8")
    except Exception as e:
        print(f"Skipping Leiden clustering: {e}")
    summary = attach_gse182434_reference_annotations(a)
    audit_gse182434_reference_missing(a, out_dir=GSE182434_OUT_DIR, prefix="gse182434_qc")
    a.uns["gse182434_qc_summary"] = qc_summary.iloc[0].to_dict()
    a.uns["gse182434_processing_note"] = (
        "QC was calculated directly from GSE182434 raw count text. QC-passing cells were normalized/log1p-transformed, then the full-gene raw layer and HVG-based UMAP were recomputed."
    )
    write_h5ad_safe(a, GSE182434_QC_H5AD, cache_stage="gse182434_qc_umap_v1")
    return a, summary


def add_qc_metrics(a):
    """Calculate QC metrics directly from a sparse matrix rather than Scanpy QC helpers."""
    gene_upper = pd.Index(a.var_names).str.upper()
    mt_mask = np.asarray(gene_upper.str.startswith("MT-"), dtype=bool)
    ribo_mask = np.asarray(gene_upper.str.startswith(("RPS", "RPL")), dtype=bool)
    x = a.X
    if sp.issparse(x):
        total_counts = to_numpy_1d(x.sum(axis=1))
        n_genes = to_numpy_1d((x > 0).sum(axis=1))
        mt_counts = to_numpy_1d(x[:, mt_mask].sum(axis=1)) if mt_mask.any() else np.zeros(a.n_obs)
        ribo_counts = to_numpy_1d(x[:, ribo_mask].sum(axis=1)) if ribo_mask.any() else np.zeros(a.n_obs)
    else:
        x_arr = np.asarray(x)
        total_counts = x_arr.sum(axis=1)
        n_genes = (x_arr > 0).sum(axis=1)
        mt_counts = x_arr[:, mt_mask].sum(axis=1) if mt_mask.any() else np.zeros(a.n_obs)
        ribo_counts = x_arr[:, ribo_mask].sum(axis=1) if ribo_mask.any() else np.zeros(a.n_obs)
    a.obs["n_genes_by_counts"] = n_genes.astype(int)
    a.obs["total_counts"] = total_counts.astype(float)
    a.obs["pct_counts_mt"] = np.divide(mt_counts, total_counts, out=np.zeros_like(total_counts, dtype=float), where=total_counts > 0) * 100
    a.obs["pct_counts_ribo"] = np.divide(ribo_counts, total_counts, out=np.zeros_like(total_counts, dtype=float), where=total_counts > 0) * 100


def preprocess_roider_for_transfer(a, force_reprocess=False):
    """Perform QC, normalization, and UMAP; save the state before creating CODEX marker proxies."""
    ensure_output_dirs()
    if not force_reprocess:
        out = read_h5ad_cache(PROCESSED_H5AD, expected_stage=ROIDER_PROCESSED_CACHE_STAGE)
        if out is not None:
            attach_reference_annotations(out)
            return out

    add_qc_metrics(a)
    qc_before = a.obs.groupby("Sample")[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe()
    qc_before.to_csv(OUT_DIR / "tables" / "qc_before_filtering_by_sample.csv")

    sample_key = a.obs["Sample"].astype("string")
    high_gene_cutoff = a.obs.groupby(sample_key)["n_genes_by_counts"].quantile(0.995).astype(float)
    a.obs["sample_high_gene_cutoff"] = sample_key.map(high_gene_cutoff).astype("float64")
    a.obs["sample_high_gene_cutoff"] = a.obs["sample_high_gene_cutoff"].fillna(float(a.obs["n_genes_by_counts"].quantile(0.995)))
    qc_mask = (
        (a.obs["n_genes_by_counts"] >= 200)
        & (a.obs["pct_counts_mt"] <= 5)
        & (a.obs["n_genes_by_counts"] <= a.obs["sample_high_gene_cutoff"])
    )
    a = a[qc_mask].copy()

    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    sc.pp.highly_variable_genes(a, n_top_genes=3000, batch_key="Sample")
    a = a[:, np.asarray(a.var["highly_variable"]).astype(bool)].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=40)
    sc.tl.umap(a, random_state=0)
    try:
        sc.tl.leiden(a, resolution=0.8, key_added="leiden_0_8")
    except Exception as e:
        print(f"Skipping Leiden clustering: {e}")
    attach_reference_annotations(a)
    write_h5ad_safe(a, PROCESSED_H5AD, cache_stage=ROIDER_PROCESSED_CACHE_STAGE)
    return a


def get_source_for_gene_features(a):
    """Prefer the raw log-normalized full-gene matrix for marker-proxy calculations."""
    return a.raw if a.raw is not None else a


def build_codex_gene_proxy_coverage(a, feature_names):
    """Quickly tabulate only the matching status of CODEX markers and scRNA gene proxies."""
    source = get_source_for_gene_features(a)
    var_names = pd.Index(source.var_names)
    upper_to_name = {}
    for gene in var_names:
        upper_to_name.setdefault(str(gene).upper(), str(gene))

    coverage_rows = []
    for marker in feature_names:
        requested = CODEX_TO_SCRNA_GENES.get(marker, [marker])
        present = [upper_to_name[g.upper()] for g in requested if g.upper() in upper_to_name]
        coverage_rows.append(
            {
                "codex_marker": marker,
                "feature_col": f"codex_{marker}",
                "requested_genes": "|".join(requested),
                "present_genes": "|".join(present),
                "covered": bool(present),
                "n_present_genes": len(present),
            }
        )
    return pd.DataFrame(coverage_rows)


def _codex_proxy_coverage_candidates(out_dir, fallback_dirs=None):
    """Search for coverage CSVs in standard Final, legacy flat, then source locations."""
    candidates = [
        Path(out_dir) / "tables" / "codex_gene_proxy_coverage.csv",
        Path(out_dir) / "codex_gene_proxy_coverage.csv",
    ]
    for fallback_dir in fallback_dirs or []:
        fallback_dir = Path(fallback_dir)
        candidates.extend(
            [
                fallback_dir / "tables" / "codex_gene_proxy_coverage.csv",
                fallback_dir / "codex_gene_proxy_coverage.csv",
            ]
        )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates


def read_or_create_codex_proxy_coverage(a, feature_names, out_dir, fallback_dirs=None):
    """Ensure a coverage CSV at the standard Final location even when using a cache."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    target = out_dir / "tables" / "codex_gene_proxy_coverage.csv"
    for candidate in _codex_proxy_coverage_candidates(out_dir, fallback_dirs=fallback_dirs):
        if candidate.exists():
            coverage = pd.read_csv(candidate)
            if candidate.resolve() != target.resolve():
                coverage.to_csv(target, index=False)
            return coverage

    if a is None:
        raise FileNotFoundError(f"CODEX gene-proxy coverage CSV not found: {target}")
    coverage = build_codex_gene_proxy_coverage(a, feature_names)
    coverage.to_csv(target, index=False)
    return coverage


def add_codex_proxy_features(
    a,
    feature_names,
    force_recompute=False,
    feature_h5ad=None,
    out_dir=None,
    attach_reference_func=attach_reference_annotations,
    cache_stage=ROIDER_FEATURE_CACHE_STAGE,
):
    """Add scRNA gene-proxy features to obs in CODEX marker-name order."""
    # Resolve mutable run destinations at call time.  Import-time defaults would
    # retain the canonical Final paths after Step09 rebinds this module to an
    # isolated noncanonical run.
    feature_h5ad = FEATURE_H5AD if feature_h5ad is None else feature_h5ad
    out_dir = OUT_DIR if out_dir is None else out_dir
    out_dir = ensure_direct_transfer_dirs(out_dir)
    feature_h5ad = Path(feature_h5ad)
    if not force_recompute:
        out = read_h5ad_cache(feature_h5ad, expected_stage=cache_stage)
        if out is not None:
            if attach_reference_func is not None:
                attach_reference_func(out)
            coverage = read_or_create_codex_proxy_coverage(
                out,
                feature_names,
                out_dir,
                fallback_dirs=[SOURCE_DIRECT_OUT_DIR, feature_h5ad.parent],
            )
            return out, coverage

    source = get_source_for_gene_features(a)
    var_names = pd.Index(source.var_names)
    upper_to_name = {}
    for gene in var_names:
        upper_to_name.setdefault(str(gene).upper(), str(gene))

    coverage_rows = []
    for marker in tqdm(feature_names, desc="CODEX marker -> scRNA proxy feature"):
        requested = CODEX_TO_SCRNA_GENES.get(marker, [marker])
        present = [upper_to_name[g.upper()] for g in requested if g.upper() in upper_to_name]
        col = f"codex_{marker}"
        if present:
            x = source[:, present].X
            if sp.issparse(x):
                vals = to_numpy_1d(x.mean(axis=1))
            else:
                vals = np.asarray(x).mean(axis=1)
            a.obs[col] = vals.astype(float)
        else:
            a.obs[col] = 0.0
        coverage_rows.append(
            {
                "codex_marker": marker,
                "feature_col": col,
                "requested_genes": "|".join(requested),
                "present_genes": "|".join(present),
                "covered": bool(present),
                "n_present_genes": len(present),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out_dir / "tables" / "codex_gene_proxy_coverage.csv", index=False)
    write_h5ad_safe(a, feature_h5ad, cache_stage=cache_stage)
    return a, coverage


def prepare_roider_feature_from_raw(
    feature_names,
    force_reload_raw=True,
    force_reprocess=True,
    force_recompute_features=True,
    *,
    feature_h5ad=None,
    out_dir=None,
):
    """Rebuild QC/UMAP and CODEX proxy features from raw Roider 10x data."""
    adata_raw, sample_sheet = load_roider_raw(force_reload=force_reload_raw)
    reference_summary = attach_reference_annotations(adata_raw)
    adata = preprocess_roider_for_transfer(adata_raw, force_reprocess=force_reprocess)
    adata_feature, coverage = add_codex_proxy_features(
        adata,
        feature_names=feature_names,
        force_recompute=force_recompute_features,
        feature_h5ad=feature_h5ad,
        out_dir=out_dir,
    )
    return adata_feature, coverage, sample_sheet, reference_summary


def apply_roider_direct_predictions(
    adata,
    package,
    tcell_package,
    min_direct_confidence=0.0,
    min_tcell_subtype_confidence=0.0,
):
    """Attach marker-base and CODEX ML predictions to Roider scRNA feature AnnData."""
    attach_reference_annotations(adata)
    adata = add_marker_based_predictions(adata)
    adata = apply_direct_broad_classifier(adata, package, min_confidence=min_direct_confidence)
    adata = apply_direct_tcell_subtype_classifier(
        adata,
        tcell_package,
        min_confidence=min_tcell_subtype_confidence,
    )
    adata = add_display_ready_columns(adata)
    adata = add_unified_prediction_scores(adata)
    return adata


def add_gse182434_codex_proxy_features(feature_names, force_recompute=False, a=None):
    """Create CODEX marker gene-proxy features for GSE182434."""
    ensure_gse182434_output_dirs()
    if not force_recompute:
        out = read_h5ad_cache(GSE182434_FEATURE_H5AD, expected_stage="gse182434_codex_proxy_features_v2_qc")
        if out is not None:
            attach_gse182434_reference_annotations(out)
            coverage = read_or_create_codex_proxy_coverage(
                out,
                feature_names,
                GSE182434_OUT_DIR,
                fallback_dirs=[SOURCE_GSE182434_DIRECT_OUT_DIR, GSE182434_FEATURE_H5AD.parent],
            )
            return out, coverage
    if a is None:
        a, _ = load_gse182434_preprocessed()
    return add_codex_proxy_features(
        a,
        feature_names=feature_names,
        force_recompute=True,
        feature_h5ad=GSE182434_FEATURE_H5AD,
        out_dir=GSE182434_OUT_DIR,
        attach_reference_func=attach_gse182434_reference_annotations,
        cache_stage="gse182434_codex_proxy_features_v2_qc",
    )


def _feature_col(marker):
    """Return CODEX marker-proxy column names."""
    return f"codex_{marker}"


def _safe_zscore(vals):
    """Convert one-dimensional values to z-scores for combining marker scores."""
    vals = np.asarray(vals, dtype=float)
    mean = np.nanmean(vals)
    std = np.nanstd(vals)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(vals, dtype=float)
    return (vals - mean) / std


def _marker_group_score(a, markers):
    """Standardize multiple marker proxies and combine them as a mean score."""
    score_parts = []
    for marker in markers:
        col = _feature_col(marker)
        if col in a.obs:
            score_parts.append(_safe_zscore(pd.to_numeric(a.obs[col], errors="coerce").fillna(0.0).to_numpy()))
    if not score_parts:
        return np.zeros(a.n_obs, dtype=float)
    return np.vstack(score_parts).mean(axis=0)


def add_hladr_expression(a):
    """Add an HLA-DR expression proxy for the 3x3 UMAP."""
    col = _feature_col("HLA-DR")
    if col in a.obs:
        a.obs["expr_HLADR"] = pd.to_numeric(a.obs[col], errors="coerce").fillna(0.0).astype(float)
    else:
        a.obs["expr_HLADR"] = 0.0
    return a


def add_marker_based_predictions(a):
    """Create broad and T-cell subtype marker-base labels from scRNA gene-proxy features."""
    broad_scores = {}
    for label, markers in BROAD_MARKER_SETS.items():
        score = _marker_group_score(a, markers)
        broad_scores[label] = score
        a.obs[f"score_marker_broad_{label}"] = score.astype(float)
    broad_order = list(BROAD_MARKER_SETS.keys())
    broad_matrix = np.vstack([broad_scores[label] for label in broad_order]).T
    broad_idx = broad_matrix.argmax(axis=1)
    broad_pred = np.asarray(broad_order, dtype=object)[broad_idx]
    a.obs["pred_broad_marker_display"] = broad_pred.astype(object)
    a.obs["pred_broad_marker_codex"] = pd.Series(broad_pred, index=a.obs.index).map(DISPLAY_TO_CODEX_BROAD_MAP).astype(object)
    a.obs["pred_broad_marker_confidence"] = broad_matrix.max(axis=1).astype(float)
    a.obs["score_marker_broad_max"] = broad_matrix.max(axis=1).astype(float)
    if broad_matrix.shape[1] > 1:
        sorted_broad = np.sort(broad_matrix, axis=1)
        a.obs["score_marker_broad_margin"] = (sorted_broad[:, -1] - sorted_broad[:, -2]).astype(float)
    else:
        a.obs["score_marker_broad_margin"] = 0.0

    subtype_scores = {}
    for label, markers in T_SUBTYPE_MARKER_SETS.items():
        score = _marker_group_score(a, markers)
        subtype_scores[label] = score
        a.obs[f"score_marker_tcell_{label}"] = score.astype(float)
    subtype_order = list(T_SUBTYPE_MARKER_SETS.keys())
    subtype_matrix = np.vstack([subtype_scores[label] for label in subtype_order]).T
    subtype_idx = subtype_matrix.argmax(axis=1)
    subtype_pred = np.asarray(subtype_order, dtype=object)[subtype_idx]
    a.obs["pred_tcell_subtype_marker"] = subtype_pred.astype(object)
    marker_subtype_display = pd.Series(subtype_pred, index=a.obs.index, dtype=object)
    marker_subtype_display.loc[a.obs["pred_broad_marker_display"].astype(str) != "T_cells"] = "Non_T_cells"
    a.obs["pred_tcell_subtype_marker_display"] = marker_subtype_display.astype(object)
    a.obs["pred_tcell_subtype_marker_confidence"] = subtype_matrix.max(axis=1).astype(float)
    a.obs["score_marker_tcell_max"] = subtype_matrix.max(axis=1).astype(float)
    if subtype_matrix.shape[1] > 1:
        sorted_subtype = np.sort(subtype_matrix, axis=1)
        a.obs["score_marker_tcell_margin"] = (sorted_subtype[:, -1] - sorted_subtype[:, -2]).astype(float)
    else:
        a.obs["score_marker_tcell_margin"] = 0.0
    add_hladr_expression(a)
    return a


def _add_score_max_and_margin(a, score_cols, max_col, margin_col):
    """Calculate the maximum and top-two margin from class-specific scores."""
    present = [col for col in score_cols if col in a.obs]
    if not present:
        return
    matrix = a.obs[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    a.obs[max_col] = matrix.max(axis=1).astype(float)
    if matrix.shape[1] > 1:
        sorted_scores = np.sort(matrix, axis=1)
        a.obs[margin_col] = (sorted_scores[:, -1] - sorted_scores[:, -2]).astype(float)
    else:
        a.obs[margin_col] = 0.0


def add_unified_prediction_scores(a):
    """Add Marker Base and CODEX ML model scores using consistent naming conventions."""
    _add_score_max_and_margin(
        a,
        [f"score_marker_broad_{label}" for label in BROAD_MARKER_SETS],
        "score_marker_broad_max",
        "score_marker_broad_margin",
    )
    _add_score_max_and_margin(
        a,
        [f"score_marker_tcell_{label}" for label in T_SUBTYPE_MARKER_SETS],
        "score_marker_tcell_max",
        "score_marker_tcell_margin",
    )
    if "pred_broad_marker_confidence" in a.obs and "score_marker_broad_max" not in a.obs:
        a.obs["score_marker_broad_max"] = pd.to_numeric(a.obs["pred_broad_marker_confidence"], errors="coerce")
    if "pred_tcell_subtype_marker_confidence" in a.obs and "score_marker_tcell_max" not in a.obs:
        a.obs["score_marker_tcell_max"] = pd.to_numeric(a.obs["pred_tcell_subtype_marker_confidence"], errors="coerce")

    for label in DISPLAY_BROAD_LABEL_ORDER:
        prob_col = f"prob_direct_{label}"
        score_col = f"score_codex_ml_broad_{label}"
        if prob_col in a.obs:
            a.obs[score_col] = pd.to_numeric(a.obs[prob_col], errors="coerce").fillna(0.0).astype(float)
    if "pred_broad_direct_confidence" in a.obs:
        a.obs["score_codex_ml_broad_max"] = pd.to_numeric(a.obs["pred_broad_direct_confidence"], errors="coerce").fillna(0.0).astype(float)

    for label in T_SUBTYPE_ANNOTATION_LABEL_ORDER:
        prob_col = f"prob_direct_tcell_{label}"
        score_col = f"score_codex_ml_tcell_{label}"
        if prob_col in a.obs:
            a.obs[score_col] = pd.to_numeric(a.obs[prob_col], errors="coerce").fillna(0.0).astype(float)
    if "pred_tcell_subtype_direct_confidence" in a.obs:
        a.obs["score_codex_ml_tcell_max"] = pd.to_numeric(a.obs["pred_tcell_subtype_direct_confidence"], errors="coerce").fillna(0.0).astype(float)
    return a


def _score_column_specs(a):
    """Define groups of score columns to save."""
    specs = [
        ("Marker Base", "broad", "zscore_mean", {label: f"score_marker_broad_{label}" for label in BROAD_MARKER_SETS}),
        ("CODEX ML model", "broad", "probability", {label: f"score_codex_ml_broad_{label}" for label in DISPLAY_BROAD_LABEL_ORDER}),
        ("Marker Base", "tcell_subtype", "zscore_mean", {label: f"score_marker_tcell_{label}" for label in T_SUBTYPE_MARKER_SETS}),
        ("CODEX ML model", "tcell_subtype", "probability", {label: f"score_codex_ml_tcell_{label}" for label in T_SUBTYPE_ANNOTATION_LABEL_ORDER}),
    ]
    out = []
    for method, task, score_scale, col_map in specs:
        present = {label: col for label, col in col_map.items() if col in a.obs}
        if present:
            out.append((method, task, score_scale, present))
    return out


def save_prediction_score_tables(a, out_dir, dataset_name, prefix="prediction_scores"):
    """Save cell-level marker/CODEX ML score tables and summary tables."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    add_unified_prediction_scores(a)

    metadata_cols = [
        "Sample", "Barcode", "Sample_Barcode", "Patient", "CellType",
        "reference_population", "reference_population_raw",
        "reference_broad_display", "reference_broad_codex", "reference_tcell_subtype",
        "reference_evaluable_broad", "reference_evaluable_tcell_subtype",
        "pred_broad_marker_display", "pred_broad_marker_codex", "pred_broad_marker_confidence",
        "pred_broad_direct_display", "pred_broad_direct_codex", "pred_broad_direct_confidence",
        "pred_broad_marker_display_eval", "pred_broad_direct_display_eval",
        "pred_tcell_subtype_marker", "pred_tcell_subtype_marker_display", "pred_tcell_subtype_marker_confidence",
        "pred_tcell_subtype_direct", "pred_tcell_subtype_direct_display", "pred_tcell_subtype_direct_confidence",
        "pred_tcell_subtype_marker_display_eval", "pred_tcell_subtype_direct_display_eval",
        "expr_HLADR",
    ]
    score_cols = sorted([
        col for col in a.obs.columns
        if col.startswith("score_marker_") or col.startswith("score_codex_ml_")
    ])
    wide_cols = [c for c in metadata_cols + score_cols if c in a.obs]
    wide = a.obs[wide_cols].copy()
    wide.insert(0, "cell_id", a.obs_names.astype(str))
    wide.insert(0, "dataset", dataset_name)
    wide_path = out_dir / "tables" / f"{prefix}_cell_scores.csv.gz"
    wide.to_csv(wide_path, index=False, compression="gzip")

    long_frames = []
    summary_rows = []
    for method, task, score_scale, col_map in _score_column_specs(a):
        frame = pd.DataFrame({
            "dataset": dataset_name,
            "cell_id": a.obs_names.astype(str),
        })
        for label, col in col_map.items():
            frame[label] = pd.to_numeric(a.obs[col], errors="coerce").astype(float).to_numpy()
            vals = frame[label].dropna()
            summary_rows.append({
                "dataset": dataset_name,
                "method": method,
                "task": task,
                "label": label,
                "score_scale": score_scale,
                "score_column": col,
                "n_cells_scored": int(vals.shape[0]),
                "score_mean": float(vals.mean()) if len(vals) else np.nan,
                "score_median": float(vals.median()) if len(vals) else np.nan,
                "score_p90": float(vals.quantile(0.9)) if len(vals) else np.nan,
            })
        long = frame.melt(
            id_vars=["dataset", "cell_id"],
            var_name="label",
            value_name="score",
        )
        long["method"] = method
        long["task"] = task
        long["score_scale"] = score_scale
        long_frames.append(long[["dataset", "cell_id", "method", "task", "label", "score_scale", "score"]])

    long_path = out_dir / "tables" / f"{prefix}_long_scores.csv.gz"
    if long_frames:
        pd.concat(long_frames, ignore_index=True).to_csv(long_path, index=False, compression="gzip")
    else:
        pd.DataFrame(columns=["dataset", "cell_id", "method", "task", "label", "score_scale", "score"]).to_csv(
            long_path,
            index=False,
            compression="gzip",
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "tables" / f"{prefix}_score_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {
        "cell_scores": wide_path,
        "long_scores": long_path,
        "score_summary": summary_path,
    }


def _available_obs_columns(a, columns):
    """Return only columns present in AnnData obs, in order."""
    return [col for col in columns if col in a.obs]


def _series_with_missing(series):
    """Explicitly label missing values as Missing in crosstab/value_counts."""
    return pd.Series(series, index=series.index, dtype=object).where(pd.notna(series), "Missing").astype(str)


def _value_counts_table(obs, columns, dataset_name, table_name):
    """Create a count table for the specified column combination."""
    if not columns:
        return pd.DataFrame(columns=["dataset", "table", "n"])
    out = (
        obs[columns]
        .value_counts(dropna=False)
        .reset_index(name="n")
    )
    out.insert(0, "table", table_name)
    out.insert(0, "dataset", dataset_name)
    return out


def _crosstab_table(obs, truth_col, pred_col, dataset_name, task, method):
    """Convert a truth/prediction label cross-tabulation to a long-form table."""
    if truth_col not in obs or pred_col not in obs:
        return pd.DataFrame(columns=["dataset", "task", "method", "truth_col", "pred_col", "truth_label", "pred_label", "n"])
    tab = pd.crosstab(
        _series_with_missing(obs[truth_col]),
        _series_with_missing(obs[pred_col]),
        dropna=False,
    )
    out = tab.stack().reset_index()
    out.columns = ["truth_label", "pred_label", "n"]
    out.insert(0, "pred_col", pred_col)
    out.insert(0, "truth_col", truth_col)
    out.insert(0, "method", method)
    out.insert(0, "task", task)
    out.insert(0, "dataset", dataset_name)
    return out


def audit_reference_and_prediction_labels(
    a,
    dataset_name,
    out_dir,
    prefix,
    source_label_cols=None,
):
    """Create tables auditing reference-label normalization and prediction-label agreement."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    source_label_cols = source_label_cols or [
        "Sample",
        "CellType",
        "reference_population_raw",
        "reference_population",
    ]
    reference_cols = [
        "reference_broad_display",
        "reference_broad_codex",
        "reference_tcell_subtype",
        "reference_tcell_subtype_display",
    ]
    prediction_cols = [
        "pred_broad_marker_display",
        "pred_broad_marker_codex",
        "pred_broad_direct_display",
        "pred_broad_direct_codex",
        "pred_broad_marker_display_eval",
        "pred_broad_direct_display_eval",
        "pred_tcell_subtype_marker",
        "pred_tcell_subtype_direct",
        "pred_tcell_subtype_marker_display",
        "pred_tcell_subtype_direct_display",
        "pred_tcell_subtype_marker_display_eval",
        "pred_tcell_subtype_direct_display_eval",
    ]
    obs = a.obs.copy()
    label_count_cols = _available_obs_columns(a, source_label_cols + reference_cols)
    prediction_count_cols = _available_obs_columns(a, source_label_cols[:2] + reference_cols + prediction_cols)

    reference_counts = _value_counts_table(obs, label_count_cols, dataset_name, "reference_label_mapping")
    prediction_counts = _value_counts_table(obs, prediction_count_cols, dataset_name, "reference_prediction_mapping")

    summary_rows = []
    for task, truth_col, labels in [
        ("broad", "reference_broad_codex", CODEX_BROAD_LABEL_ORDER),
        ("tcell_subtype", "reference_tcell_subtype", T_SUBTYPE_LABEL_ORDER),
    ]:
        if truth_col in obs:
            truth_vals = _series_with_missing(obs[truth_col])
            reference_mask = obs[truth_col].isin(labels)
            missing_mask = obs[truth_col].isna()
            present = [label for label in labels if label in set(truth_vals)]
            absent = [label for label in labels if label not in set(truth_vals)]
            non_t_excluded = np.nan
            broad_t_without_subtype = np.nan
            if task == "tcell_subtype" and "reference_broad_codex" in obs:
                broad_vals = obs["reference_broad_codex"]
                non_t_excluded = int(broad_vals.isin([label for label in CODEX_BROAD_LABEL_ORDER if label != "T cells"]).sum())
                broad_t_without_subtype = int((broad_vals.eq("T cells") & (~reference_mask)).sum())
            summary_rows.append({
                "dataset": dataset_name,
                "task": task,
                "truth_col": truth_col,
                "expected_labels": "|".join(labels),
                "reference_present_labels": "|".join(present),
                "reference_absent_expected_labels": "|".join(absent),
                "n_reference_evaluable": int(reference_mask.sum()),
                "n_reference_missing_or_unmapped": int((~reference_mask).sum()),
                "n_reference_missing": int(missing_mask.sum()),
                "n_reference_unmapped_not_missing": int(((~reference_mask) & (~missing_mask)).sum()),
                "n_non_t_cells_excluded_from_subtype": non_t_excluded,
                "n_broad_t_cells_without_subtype_reference": broad_t_without_subtype,
                "evaluation_rule": "Only cells with reference labels in the expected label set are included in BAcc/Macro F1.",
            })
    label_summary = pd.DataFrame(summary_rows)

    crosstab_specs = [
        ("broad", "Marker Base display", "reference_broad_display", "pred_broad_marker_display_eval"),
        ("broad", "CODEX ML display", "reference_broad_display", "pred_broad_direct_display_eval"),
        ("broad", "Marker Base CODEX", "reference_broad_codex", "pred_broad_marker_codex"),
        ("broad", "CODEX ML CODEX", "reference_broad_codex", "pred_broad_direct_codex"),
        ("tcell_subtype", "Marker Base", "reference_tcell_subtype", "pred_tcell_subtype_marker"),
        ("tcell_subtype", "CODEX ML", "reference_tcell_subtype", "pred_tcell_subtype_direct"),
        ("tcell_subtype", "Marker Base display", "reference_tcell_subtype_display", "pred_tcell_subtype_marker_display_eval"),
        ("tcell_subtype", "CODEX ML display", "reference_tcell_subtype_display", "pred_tcell_subtype_direct_display_eval"),
    ]
    crosstabs = []
    for task, method, truth_col, pred_col in crosstab_specs:
        crosstabs.append(_crosstab_table(obs, truth_col, pred_col, dataset_name, task, method))
    label_crosstab = pd.concat(crosstabs, ignore_index=True) if crosstabs else pd.DataFrame()

    table_dir = Path(out_dir) / "tables"
    reference_counts.to_csv(table_dir / f"{prefix}_reference_label_counts.csv", index=False)
    prediction_counts.to_csv(table_dir / f"{prefix}_prediction_label_counts.csv", index=False)
    label_summary.to_csv(table_dir / f"{prefix}_label_summary.csv", index=False)
    label_crosstab.to_csv(table_dir / f"{prefix}_label_crosstab.csv", index=False)

    return {
        "reference_counts": reference_counts,
        "prediction_counts": prediction_counts,
        "label_summary": label_summary,
        "label_crosstab": label_crosstab,
    }


def _nearest_labeled_majority(coords, source_idx, target_idx, labels, n_neighbors=30):
    """Calculate the UMAP nearest-neighbor majority source label for each target cell."""
    if len(source_idx) == 0 or len(target_idx) == 0:
        return pd.DataFrame(index=np.asarray(target_idx, dtype=int))
    k = min(int(n_neighbors), len(source_idx))
    tree = cKDTree(coords[source_idx])
    dist, nbr = tree.query(coords[target_idx], k=k)
    if k == 1:
        dist = dist[:, None]
        nbr = nbr[:, None]
    neighbor_labels = pd.Series(labels).iloc[source_idx[nbr.reshape(-1)]].astype(str).to_numpy().reshape(nbr.shape)
    rows = []
    for i in range(len(target_idx)):
        vals, counts = np.unique(neighbor_labels[i], return_counts=True)
        order = np.argsort(counts)[::-1]
        rows.append({
            "majority_label": vals[order[0]],
            "majority_fraction": counts[order[0]] / float(k),
            "median_distance": float(np.median(dist[i])),
        })
    return pd.DataFrame(rows, index=np.asarray(target_idx, dtype=int))


def infer_roider_missing_cell_identity(a, out_dir=OUT_DIR, prefix="roider_predicted", n_neighbors=30):
    """Infer identities of Reference Missing cells from nearby UMAP references and predictions."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    if "X_umap" not in a.obsm:
        raise KeyError("Cannot audit Missing identities by UMAP neighbors without adata.obsm['X_umap'].")
    attach_reference_annotations(a)
    add_display_ready_columns(a)
    add_unified_prediction_scores(a)

    obs = a.obs.copy()
    missing = obs["reference_broad_display"].isna().to_numpy()
    missing_idx = np.where(missing)[0]
    coords = a.obsm["X_umap"]

    labeled_broad_idx = np.where(obs["reference_broad_display"].notna().to_numpy())[0]
    broad_neighbor = _nearest_labeled_majority(
        coords,
        labeled_broad_idx,
        missing_idx,
        obs["reference_broad_display"],
        n_neighbors=n_neighbors,
    )
    broad_neighbor.index = obs.index[missing_idx]

    labeled_pop_idx = np.where(obs["reference_population_raw"].notna().to_numpy())[0]
    pop_neighbor = _nearest_labeled_majority(
        coords,
        labeled_pop_idx,
        missing_idx,
        obs["reference_population_raw"],
        n_neighbors=n_neighbors,
    )
    pop_neighbor.index = obs.index[missing_idx]

    base_cols = [
        "Sample", "Barcode", "Sample_Barcode",
        "pred_broad_direct_display", "pred_broad_direct_confidence",
        "pred_broad_marker_display", "pred_broad_marker_confidence",
        "pred_tcell_subtype_direct_display", "pred_tcell_subtype_direct_confidence",
        "score_codex_ml_broad_B_cells", "score_codex_ml_broad_T_cells", "score_codex_ml_broad_Myeloid_cells",
        "score_marker_broad_B_cells", "score_marker_broad_T_cells", "expr_HLADR",
    ]
    cell_df = obs.loc[missing, [c for c in base_cols if c in obs]].copy()
    cell_df["umap_neighbor_majority_broad"] = broad_neighbor["majority_label"]
    cell_df["umap_neighbor_majority_broad_fraction"] = broad_neighbor["majority_fraction"]
    cell_df["umap_neighbor_broad_median_distance"] = broad_neighbor["median_distance"]
    cell_df["nearest_reference_population_raw"] = pop_neighbor["majority_label"]
    cell_df["nearest_reference_population_fraction"] = pop_neighbor["majority_fraction"]
    cell_df["nearest_reference_population_median_distance"] = pop_neighbor["median_distance"]

    direct = cell_df.get("pred_broad_direct_display", pd.Series("Missing", index=cell_df.index)).astype(str)
    neighbor = cell_df["umap_neighbor_majority_broad"].astype(str)
    consistent = direct.eq(neighbor)
    cell_df["inferred_missing_identity"] = np.where(
        consistent,
        neighbor,
        np.where(
            cell_df["umap_neighbor_majority_broad_fraction"].astype(float) >= 0.8,
            neighbor + "_by_UMAP_neighbor",
            direct + "_by_CODEX_ML",
        ),
    )
    cell_df["identity_support"] = np.where(
        consistent,
        "CODEX_ML_and_UMAP_neighbor_agree",
        np.where(
            cell_df["umap_neighbor_majority_broad_fraction"].astype(float) >= 0.8,
            "UMAP_neighbor_high_fraction",
            "discordant_or_low_neighbor_fraction",
        ),
    )

    cell_path = out_dir / "tables" / f"{prefix}_missing_identity_cell_level.csv.gz"
    cell_df.to_csv(cell_path, index=True, compression="gzip")

    summary_rows = []
    for name, col in [
        ("codex_ml_prediction", "pred_broad_direct_display"),
        ("marker_base_prediction", "pred_broad_marker_display"),
        ("umap_neighbor_broad", "umap_neighbor_majority_broad"),
        ("nearest_reference_population", "nearest_reference_population_raw"),
        ("inferred_missing_identity", "inferred_missing_identity"),
        ("identity_support", "identity_support"),
    ]:
        if col not in cell_df:
            continue
        counts = cell_df[col].value_counts(dropna=False)
        fracs = cell_df[col].value_counts(normalize=True, dropna=False)
        for label, n in counts.items():
            summary_rows.append({
                "summary_type": name,
                "label": label,
                "n_missing_cells": int(n),
                "fraction_missing_cells": float(fracs.loc[label]),
            })
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "tables" / f"{prefix}_missing_identity_summary.csv"
    summary.to_csv(summary_path, index=False)

    crosstab = pd.crosstab(
        cell_df["umap_neighbor_majority_broad"],
        cell_df["pred_broad_direct_display"],
        dropna=False,
    )
    crosstab_path = out_dir / "tables" / f"{prefix}_missing_identity_umap_neighbor_vs_codex_ml.csv"
    crosstab.to_csv(crosstab_path)

    by_sample = (
        cell_df
        .groupby(["Sample", "inferred_missing_identity"], observed=False)
        .size()
        .reset_index(name="n_missing_cells")
        .sort_values(["Sample", "n_missing_cells"], ascending=[True, False])
        if "Sample" in cell_df else pd.DataFrame()
    )
    by_sample_path = out_dir / "tables" / f"{prefix}_missing_identity_by_sample.csv"
    by_sample.to_csv(by_sample_path, index=False)

    return {
        "cell_level": cell_path,
        "summary": summary_path,
        "umap_neighbor_vs_codex_ml": crosstab_path,
        "by_sample": by_sample_path,
        "cell_df": cell_df,
        "summary_df": summary,
        "crosstab": crosstab,
        "by_sample_df": by_sample,
    }


def save_missing_identity_umap(a, identity_df, path, point_size=7):
    """Highlight inferred identities of Reference Missing cells on UMAP and save."""
    path = Path(path)
    if "X_umap" not in a.obsm:
        raise KeyError("adata.obsm['X_umap'] is missing.")
    coords = a.obsm["X_umap"]
    all_index = pd.Index(a.obs_names)
    missing_index = pd.Index(identity_df.index)
    missing_pos = all_index.get_indexer(missing_index)
    valid = missing_pos >= 0
    missing_pos = missing_pos[valid]
    plot_df = identity_df.iloc[np.where(valid)[0]].copy()
    vals, categories = _ordered_categories(
        plot_df["inferred_missing_identity"],
        [
            "B_cells",
            "B_cells_by_UMAP_neighbor",
            "B_cells_by_CODEX_ML",
            "T_cells",
            "T_cells_by_UMAP_neighbor",
            "T_cells_by_CODEX_ML",
            "Myeloid_cells",
            "Myeloid_cells_by_UMAP_neighbor",
            "Myeloid_cells_by_CODEX_ML",
        ],
    )
    palette = _palette_for_categories(categories, {
        "B_cells": BROAD_COLOR_MAP["B_cells"],
        "B_cells_by_UMAP_neighbor": "#86BCB6",
        "B_cells_by_CODEX_ML": "#1F4E79",
        "T_cells": BROAD_COLOR_MAP["T_cells"],
        "T_cells_by_UMAP_neighbor": "#F28E8B",
        "T_cells_by_CODEX_ML": "#A23E3E",
        "Myeloid_cells": BROAD_COLOR_MAP["Myeloid_cells"],
        "Myeloid_cells_by_UMAP_neighbor": "#8CD17D",
        "Myeloid_cells_by_CODEX_ML": "#2E7D32",
    })

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(coords[:, 0], coords[:, 1], s=max(point_size * 0.5, 1), c="#D0D0D0", alpha=0.22, linewidths=0)
    for cat in categories:
        idx = vals.eq(cat).to_numpy()
        ax.scatter(
            coords[missing_pos[idx], 0],
            coords[missing_pos[idx], 1],
            s=point_size,
            c=palette.get(cat, "#808080"),
            label=cat,
            alpha=0.9,
            linewidths=0,
        )
    ax.set_title("Roider reference Missing cell identity audit", fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP1", fontsize=12, fontweight="bold")
    ax.set_ylabel("UMAP2", fontsize=12, fontweight="bold")
    ax.legend(markerscale=2.3, fontsize=8, title="Inferred identity", title_fontsize=9, frameon=True, loc="best")
    fig.tight_layout()
    _save_scrna_figure_png_pptx(fig, path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def add_evaluation_masked_prediction_columns(a, not_evaluated_label="Not_evaluated"):
    """Exclude cells without reference labels from annotation in validation displays."""
    if "reference_broad_display" in a.obs:
        broad_eval_mask = a.obs["reference_broad_display"].notna()
    else:
        broad_eval_mask = pd.Series(False, index=a.obs.index)
    a.obs["reference_evaluable_broad"] = broad_eval_mask.astype(bool).to_numpy()

    for src, dst in [
        ("pred_broad_marker_display", "pred_broad_marker_display_eval"),
        ("pred_broad_direct_display", "pred_broad_direct_display_eval"),
    ]:
        if src in a.obs:
            vals = a.obs[src].astype(object).copy()
            vals.loc[~broad_eval_mask] = not_evaluated_label
            a.obs[dst] = vals.astype(object)

    if "reference_tcell_subtype" in a.obs:
        subtype_eval_mask = a.obs["reference_tcell_subtype"].isin(T_SUBTYPE_LABEL_ORDER)
    else:
        subtype_eval_mask = pd.Series(False, index=a.obs.index)
    a.obs["reference_evaluable_tcell_subtype"] = subtype_eval_mask.astype(bool).to_numpy()

    for src, dst in [
        ("pred_tcell_subtype_marker_display", "pred_tcell_subtype_marker_display_eval"),
        ("pred_tcell_subtype_direct_display", "pred_tcell_subtype_direct_display_eval"),
    ]:
        if src in a.obs:
            vals = a.obs[src].astype(object).copy()
            vals.loc[~subtype_eval_mask] = not_evaluated_label
            a.obs[dst] = vals.astype(object)
    return a


def add_display_ready_columns(a):
    """Add Missing/Non-T display columns for UMAP panels."""
    if "reference_tcell_subtype_display" in a.obs:
        ref_sub = a.obs["reference_tcell_subtype_display"].astype(object).copy()
    elif "reference_tcell_subtype" in a.obs:
        ref_sub = a.obs["reference_tcell_subtype"].astype(object).copy()
    else:
        ref_sub = pd.Series(np.nan, index=a.obs.index, dtype=object)
    if "reference_broad_display" in a.obs:
        broad_vals = a.obs["reference_broad_display"]
        non_t_mask = broad_vals.notna() & (broad_vals.astype(str) != "T_cells")
        ref_sub.loc[non_t_mask & pd.isna(ref_sub)] = "Non_T_cells"
    a.obs["reference_tcell_subtype_display"] = ref_sub.astype(object)

    for col in ["pred_tcell_subtype_marker_display", "pred_tcell_subtype_direct_display"]:
        if col not in a.obs:
            a.obs[col] = pd.Series("Missing", index=a.obs.index, dtype=object)
    add_evaluation_masked_prediction_columns(a)
    add_hladr_expression(a)
    return a


def inspect_codex_broad_artifact(model_path=CODEX_MODELS_BROAD):
    """Check V9 broad metadata artifacts and separately saved frozen ensembles."""
    ensure_output_dirs()
    obj = load_compatible_joblib(model_path)
    has_predictor = False
    predictor_keys = []
    for key, val in obj.items() if isinstance(obj, dict) else []:
        if hasattr(val, "predict_proba") or hasattr(val, "predict"):
            has_predictor = True
            predictor_keys.append(str(key))
    info = {
        "artifact_path": str(model_path),
        "artifact_type": type(obj).__name__,
        "keys": list(obj.keys()) if isinstance(obj, dict) else [],
        "classes": list(obj["le_broad"].classes_) if isinstance(obj, dict) and "le_broad" in obj else [],
        "feature_names": list(obj.get("feature_names", [])) if isinstance(obj, dict) else [],
        "target_combo": obj.get("target_combo") if isinstance(obj, dict) else None,
        "has_predict_proba_model_object": has_predictor,
        "predictor_keys": predictor_keys,
        "direct_transfer_note": (
            "mv1_models_broad.pkl is for checking feature/label metadata. "
            "Step11 inference uses iteration-specific scalers and frozen models in checkpoints/transfer_models/broad."
        ),
    }
    with open(OUT_DIR / "tables" / "codex_broad_artifact_inspection.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info, obj


def _read_first_existing_csv(paths):
    """Read the first existing CSV among candidate paths."""
    for path in paths:
        path = Path(path)
        if path.exists():
            return pd.read_csv(path), path
    return None, None


def _summarize_direct_package(package_path, task, artifact_features):
    """Summarize direct-transfer package features, classes, and coefficients."""
    package_path = Path(package_path)
    row = {
        "task": task,
        "package_path": str(package_path),
        "package_exists": bool(package_path.exists()),
        "model_type": np.nan,
        "classes": np.nan,
        "n_classes": 0,
        "n_features": 0,
        "feature_order_matches_v8_artifact": False,
        "feature_set_matches_v8_artifact": False,
        "has_coefficients": False,
        "training_source": np.nan,
        "holdout_balanced_accuracy": np.nan,
        "holdout_macro_f1": np.nan,
    }
    if not package_path.exists():
        return row, None

    package = load_compatible_joblib(package_path)
    feature_names = list(package.get("feature_names", []))
    label_encoder = package.get("label_encoder")
    classes = list(label_encoder.classes_) if label_encoder is not None else []
    model = package.get("model")
    classifier = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
    holdout = package.get("holdout_metrics")
    if isinstance(holdout, pd.DataFrame) and not holdout.empty:
        row["holdout_balanced_accuracy"] = float(holdout["balanced_accuracy"].iloc[0])
        row["holdout_macro_f1"] = float(holdout["macro_f1"].iloc[0])

    row.update({
        "model_type": type(model).__name__,
        "classes": "|".join(map(str, classes)),
        "n_classes": len(classes),
        "n_features": len(feature_names),
        "feature_order_matches_v8_artifact": feature_names == list(artifact_features),
        "feature_set_matches_v8_artifact": set(feature_names) == set(artifact_features),
        "has_coefficients": getattr(classifier, "coef_", None) is not None,
        "training_source": package.get("training_source", np.nan),
    })
    return row, package


def _read_frozen_manifest(stage_dir):
    """Read the V9 frozen-stage manifest and verify completion."""
    stage_dir = Path(stage_dir)
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"V9 frozen manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"V9 frozen stage is incomplete: {manifest_path}; "
            f"status={manifest.get('status')}"
        )
    if not manifest.get("selected_combo") or not manifest.get("selected_iterations"):
        raise RuntimeError(
            f"Selected combo/iterations are absent from the manifest: {manifest_path}"
        )
    return manifest


def _summarize_frozen_stage(stage_dir, task, artifact_features):
    """Summarize V9 frozen ensemble features, classes, and selected models."""
    stage_dir = Path(stage_dir)
    manifest = _read_frozen_manifest(stage_dir)
    feature_names = list(manifest.get("feature_names", []))
    classes = list(map(str, manifest.get("classes", [])))
    combo = str(manifest.get("selected_combo"))
    selected_keys = [key.strip() for key in combo.split("+") if key.strip()]
    selected_iterations = int(manifest.get("selected_iterations", 0))
    available = manifest.get("available_iterations", {})
    scaler_iterations = set(map(int, manifest.get("scaler_iterations", [])))
    available_counts = {
        key: len([i for i in available.get(key, []) if int(i) in scaler_iterations])
        for key in selected_keys
    }
    if any(count < selected_iterations for count in available_counts.values()):
        raise RuntimeError(
            f"Fewer saved models than selected iterations: {stage_dir}; "
            f"selected={selected_iterations}, available={available_counts}"
        )
    row = {
        "task": task,
        "package_path": str(stage_dir / "manifest.json"),
        "package_exists": True,
        "model_type": "V9 frozen ensemble",
        "classes": "|".join(classes),
        "n_classes": len(classes),
        "n_features": len(feature_names),
        "feature_order_matches_v8_artifact": feature_names == list(artifact_features),
        "feature_set_matches_v8_artifact": set(feature_names) == set(artifact_features),
        "has_coefficients": False,
        "training_source": manifest.get("metadata", {}).get("source_code", np.nan),
        "holdout_balanced_accuracy": np.nan,
        "holdout_macro_f1": np.nan,
        "status": manifest.get("status"),
        "selected_combo": combo,
        "selected_iterations": selected_iterations,
        "probability_rule": manifest.get("probability_rule", ""),
        "rescue_applied": False,
    }
    package_view = {
        "feature_names": feature_names,
        "feature_columns": [_feature_col(marker) for marker in feature_names],
        "classes": classes,
        "manifest": manifest,
    }
    return row, package_view


def inspect_v9_frozen_transfer_package(transfer_root=V9_TRANSFER_ROOT):
    """Validate the broad/T-cell frozen ensemble contracts for Step11."""
    transfer_root = Path(transfer_root)
    broad = _read_frozen_manifest(transfer_root / "broad")
    tcell = _read_frozen_manifest(transfer_root / "subtype" / "T_cells")
    broad_features = list(broad.get("feature_names", []))
    tcell_features = list(tcell.get("feature_names", []))
    if broad_features != tcell_features:
        raise RuntimeError("Feature order differs between broad and T-cell subtype frozen manifests.")
    summary = pd.DataFrame([
        {
            "stage": "broad",
            "manifest": str(transfer_root / "broad" / "manifest.json"),
            "status": broad.get("status"),
            "selected_combo": broad.get("selected_combo"),
            "selected_iterations": broad.get("selected_iterations"),
            "n_features": len(broad_features),
            "classes": "|".join(map(str, broad.get("classes", []))),
            "rescue_applied": False,
        },
        {
            "stage": "T-cell subtype",
            "manifest": str(transfer_root / "subtype" / "T_cells" / "manifest.json"),
            "status": tcell.get("status"),
            "selected_combo": tcell.get("selected_combo"),
            "selected_iterations": tcell.get("selected_iterations"),
            "n_features": len(tcell_features),
            "classes": "|".join(map(str, tcell.get("classes", []))),
            "rescue_applied": False,
        },
    ])
    return {
        "transfer_root": transfer_root,
        "feature_names": broad_features,
        "broad_manifest": broad,
        "tcell_manifest": tcell,
        "summary": summary,
    }


def audit_direct_transfer_alignment(artifact_info=None, coverage=None, out_dir=OUT_DIR, dataset_label="Roider"):
    """Check feature/class alignment between scRNA proxies and the V9 frozen ensemble."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    if artifact_info is None:
        artifact_info, _ = inspect_codex_broad_artifact(CODEX_MODELS_BROAD)
    artifact_features = list(artifact_info.get("feature_names", []))

    if coverage is None:
        coverage, coverage_path = _read_first_existing_csv([
            out_dir / "tables" / "codex_gene_proxy_coverage.csv",
            SOURCE_DIRECT_OUT_DIR / "tables" / "codex_gene_proxy_coverage.csv",
        ])
    else:
        coverage = coverage.copy()
        coverage_path = "provided_dataframe"

    coverage_map = {}
    if isinstance(coverage, pd.DataFrame) and "codex_marker" in coverage.columns:
        coverage_map = coverage.set_index("codex_marker").to_dict(orient="index")
    n_covered = int(pd.Series([bool(v.get("covered", False)) for v in coverage_map.values()]).sum()) if coverage_map else 0

    package_rows = []
    packages = {}
    for task, path in [
        ("broad", V9_TRANSFER_ROOT / "broad"),
        ("tcell_subtype", V9_TRANSFER_ROOT / "subtype" / "T_cells"),
    ]:
        row, package = _summarize_frozen_stage(path, task, artifact_features)
        package_rows.append(row)
        packages[task] = package
    package_audit = pd.DataFrame(package_rows)

    feature_rows = []
    for i, marker in enumerate(artifact_features):
        cov = coverage_map.get(marker, {})
        row = {
            "feature_index": i,
            "codex_marker": marker,
            "requested_genes": cov.get("requested_genes", "|".join(CODEX_TO_SCRNA_GENES.get(marker, [marker]))),
            "present_genes": cov.get("present_genes", ""),
            "scrna_proxy_covered": bool(cov.get("covered", False)),
        }
        for task, package in packages.items():
            feature_names = list(package.get("feature_names", [])) if package is not None else []
            feature_columns = list(package.get("feature_columns", [])) if package is not None else []
            row[f"{task}_feature_at_same_index"] = feature_names[i] if i < len(feature_names) else np.nan
            row[f"{task}_feature_order_match"] = bool(i < len(feature_names) and feature_names[i] == marker)
            row[f"{task}_feature_col"] = feature_columns[i] if i < len(feature_columns) else np.nan
        feature_rows.append(row)
    feature_audit = pd.DataFrame(feature_rows)

    if str(dataset_label).lower().startswith("gse"):
        broad_reference_source = "GSE182434 CellType"
        broad_raw_mapping = "B cells/Plasma cells->B_cells; T cells CD4/CD8/Tregs/TFH->T_cells; Monocytes and Macrophages->Myeloid_cells; NK cells->NK_cells"
        subtype_reference_source = "GSE182434 CellType"
        subtype_raw_mapping = "T cells CD4->TH; TFH->TFH; Tregs->TREG; T cells CD8->CD8T"
        label_detail = "GSE182434 subtype metrics use TH/TFH/TREG/CD8T only; TPR/CD4_CTL remain annotation labels because they are not external scRNA reference classes."
    else:
        broad_reference_source = "Roider AnnotationBcells/Annotation_Tcells Population"
        broad_raw_mapping = "MalignantB/HealthyB->B_cells; Myeloid->Myeloid_cells; TH/TFH/TREG/TTOX/CD8T->T_cells"
        subtype_reference_source = "Roider T-cell Population"
        subtype_raw_mapping = "TH->TH; TFH->TFH; TREG->TREG; TTOX->CD8T"
        label_detail = "Roider subtype metrics use TH/TFH/TREG/CD8T only; TTOX is evaluated as CD8T, while TPR/CD4_CTL remain annotation labels."

    broad_classes = set(str(package_audit.loc[package_audit["task"].eq("broad"), "classes"].iloc[0]).split("|"))
    tcell_raw_classes = set(str(package_audit.loc[package_audit["task"].eq("tcell_subtype"), "classes"].iloc[0]).split("|"))
    tcell_classes = {CODEX_T_SUBTYPE_TO_DISPLAY.get(label, label) for label in tcell_raw_classes}

    label_rows = [
        {
            "dataset": dataset_label,
            "task": "broad",
            "reference_source": broad_reference_source,
            "raw_label_mapping": broad_raw_mapping,
            "evaluation_truth_col": "reference_broad_codex",
            "evaluation_labels": "|".join(CODEX_BROAD_LABEL_ORDER),
            "annotation_labels": "|".join(CODEX_BROAD_LABEL_ORDER),
            "package_classes": package_audit.loc[package_audit["task"].eq("broad"), "classes"].iloc[0],
            "class_set_matches_evaluation_labels": broad_classes == set(CODEX_BROAD_LABEL_ORDER),
            "class_set_contains_evaluation_labels": broad_classes.issuperset(set(CODEX_BROAD_LABEL_ORDER)),
            "class_set_matches_annotation_labels": broad_classes == set(CODEX_BROAD_LABEL_ORDER),
        },
        {
            "dataset": dataset_label,
            "task": "tcell_subtype",
            "reference_source": subtype_reference_source,
            "raw_label_mapping": subtype_raw_mapping,
            "evaluation_truth_col": "reference_tcell_subtype",
            "evaluation_labels": "|".join(T_SUBTYPE_LABEL_ORDER),
            "annotation_labels": "|".join(T_SUBTYPE_ANNOTATION_LABEL_ORDER),
            "package_classes": package_audit.loc[package_audit["task"].eq("tcell_subtype"), "classes"].iloc[0],
            "class_set_matches_evaluation_labels": tcell_classes == set(T_SUBTYPE_LABEL_ORDER),
            "class_set_contains_evaluation_labels": tcell_classes.issuperset(set(T_SUBTYPE_LABEL_ORDER)),
            "class_set_matches_annotation_labels": tcell_classes == set(T_SUBTYPE_ANNOTATION_LABEL_ORDER),
        },
    ]
    label_audit = pd.DataFrame(label_rows)

    metric_frames = []
    for task, filename in [
        ("broad", "broad_lineage_method_metrics.csv"),
        ("tcell_subtype", "tcell_subtype_method_metrics.csv"),
    ]:
        fallback_dir = SOURCE_GSE182434_DIRECT_OUT_DIR if str(dataset_label).lower().startswith("gse") else SOURCE_DIRECT_OUT_DIR
        metrics, metric_path = _read_first_existing_csv([
            out_dir / "tables" / filename,
            fallback_dir / "tables" / filename,
        ])
        if metrics is None:
            continue
        metrics = metrics.copy()
        metrics.insert(0, "task", task)
        metrics["metric_path"] = str(metric_path)
        if "macro_f1_model_label_space" in metrics.columns:
            metrics["macro_f1_drop_from_absent_reference_classes"] = (
                pd.to_numeric(metrics["macro_f1"], errors="coerce")
                - pd.to_numeric(metrics["macro_f1_model_label_space"], errors="coerce")
            )
        elif "macro_f1_reference_present_classes" in metrics.columns:
            metrics["macro_f1_drop_from_absent_reference_classes"] = (
                pd.to_numeric(metrics["macro_f1_reference_present_classes"], errors="coerce")
                - pd.to_numeric(metrics["macro_f1"], errors="coerce")
            )
        metric_frames.append(metrics)
    metric_audit = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()

    summary_rows = [
        {
            "audit_item": "V9 frozen ensemble manifests",
            "status": "OK" if package_audit["status"].eq("complete").all() else "CHECK",
            "detail": "Step11 inference uses the selected combo/iterations saved under checkpoints/transfer_models; no scRNA retraining and no spatial rescue.",
        },
        {
            "audit_item": "scRNA proxy marker coverage",
            "status": "OK" if n_covered == len(artifact_features) else "CHECK",
            "detail": f"{n_covered}/{len(artifact_features)} CODEX markers have scRNA gene proxies; source={coverage_path}",
        },
        {
            "audit_item": "Broad package feature order",
            "status": "OK" if bool(package_audit.loc[package_audit["task"].eq("broad"), "feature_order_matches_v8_artifact"].iloc[0]) else "CHECK",
            "detail": "broad frozen manifest feature order compared with mv1 broad artifact feature_names",
        },
        {
            "audit_item": "T-cell subtype package feature order",
            "status": "OK" if bool(package_audit.loc[package_audit["task"].eq("tcell_subtype"), "feature_order_matches_v8_artifact"].iloc[0]) else "CHECK",
            "detail": "T-cell subtype frozen manifest feature order compared with mv1 broad artifact feature_names",
        },
        {
            "audit_item": "Label harmonization",
            "status": "OK" if label_audit["class_set_contains_evaluation_labels"].all() else "CHECK",
            "detail": label_detail,
        },
    ]
    summary = pd.DataFrame(summary_rows)

    for name, frame in [
        ("direct_transfer_alignment_summary", summary),
        ("direct_transfer_package_audit", package_audit),
        ("direct_transfer_feature_audit", feature_audit),
        ("direct_transfer_label_audit", label_audit),
        ("direct_transfer_metric_audit", metric_audit),
    ]:
        frame.to_csv(out_dir / "tables" / f"{name}.csv", index=False)

    return {
        "summary": summary,
        "package_audit": package_audit,
        "feature_audit": feature_audit,
        "label_audit": label_audit,
        "metric_audit": metric_audit,
    }


def make_valid_label_mask(labels):
    """Exclude na/nan/unknown labels."""
    s = pd.Series(labels).astype("string").str.strip()
    invalid = s.isna() | s.str.lower().isin({"", "na", "nan", "none", "unknown", "unassigned"})
    return ~invalid.to_numpy(), s


def load_codex_reference_training_data(feature_names, max_cells_per_class=50000, random_state=0):
    """Sample reference cells for broad-classifier training from the CODEX Step02 checkpoint."""
    adata = sc.read_h5ad(CODEX_STEP02_H5AD)
    x = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    var_names = list(adata.var_names)
    if var_names != list(feature_names):
        feature_to_idx = {f: i for i, f in enumerate(var_names)}
        missing = [f for f in feature_names if f not in feature_to_idx]
        if missing:
            raise KeyError(f"Required features are missing from the CODEX checkpoint: {missing[:10]}")
        x = x[:, [feature_to_idx[f] for f in feature_names]]

    disease = adata.obs["Disease"].astype("string").str.upper()
    ref_mask = ~disease.eq("DLBCL").to_numpy()
    valid_mask, labels = make_valid_label_mask(adata.obs.loc[ref_mask, "GT_Lineage_Broad"].values)
    x_ref = np.asarray(x[ref_mask][valid_mask], dtype=np.float32)
    y_ref = labels[valid_mask].astype(str).to_numpy()
    groups = adata.obs.loc[ref_mask, "PatientID"].astype("string").to_numpy()[valid_mask]

    rng = np.random.default_rng(random_state)
    keep_idx = []
    for label in np.unique(y_ref):
        idx = np.where(y_ref == label)[0]
        if max_cells_per_class is not None and len(idx) > max_cells_per_class:
            idx = rng.choice(idx, size=max_cells_per_class, replace=False)
        keep_idx.extend(idx.tolist())
    keep_idx = np.asarray(sorted(keep_idx))
    meta = pd.DataFrame({"label": y_ref[keep_idx], "PatientID": groups[keep_idx]})
    meta.value_counts(["label"], dropna=False).rename_axis("label").reset_index(name="n").to_csv(
        OUT_DIR / "tables" / "codex_training_label_counts.csv", index=False
    )
    meta.value_counts(["label", "PatientID"], dropna=False).rename_axis(["label", "PatientID"]).reset_index(name="n").to_csv(
        OUT_DIR / "tables" / "codex_training_label_patient_counts.csv", index=False
    )
    return x_ref[keep_idx], y_ref[keep_idx], groups[keep_idx]


def train_direct_broad_classifier(feature_names, max_cells_per_class=50000, random_state=0, holdout_check=True):
    """Train a broad classifier on CODEX references and save it as a direct-transfer package."""
    ensure_output_dirs()
    x, y, groups = load_codex_reference_training_data(
        feature_names=feature_names,
        max_cells_per_class=max_cells_per_class,
        random_state=random_state,
    )
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    l1_ratio=0.15,
                    alpha=1e-4,
                    max_iter=1000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    holdout_metrics = None
    if holdout_check:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        train_idx, test_idx = next(splitter.split(x, y_enc, groups=groups))
        model.fit(x[train_idx], y_enc[train_idx])
        pred = model.predict(x[test_idx])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            bacc = balanced_accuracy_score(y_enc[test_idx], pred)
        macro_f1 = f1_score(y_enc[test_idx], pred, average="macro", zero_division=0)
        holdout_metrics = pd.DataFrame(
            [{"dataset": "CODEX reference holdout", "n_cells": len(test_idx), "balanced_accuracy": bacc, "macro_f1": macro_f1}]
        )
        holdout_metrics.to_csv(OUT_DIR / "tables" / "codex_reference_holdout_metrics.csv", index=False)

    model.fit(x, y_enc)
    package = {
        "model": model,
        "label_encoder": le,
        "feature_names": list(feature_names),
        "feature_columns": [f"codex_{m}" for m in feature_names],
        "training_source": str(CODEX_STEP02_H5AD),
        "max_cells_per_class": max_cells_per_class,
        "holdout_metrics": holdout_metrics,
        "direct_transfer_warning": (
            "This package applies a classifier trained on CODEX protein-intensity features to scRNA gene-proxy features. "
            "Cross-modal scale/domain shift may therefore be a key source of reduced performance."
        ),
    }
    joblib.dump(package, DIRECT_MODEL_PACKAGE)
    return package


def load_codex_tcell_subtype_training_data(feature_names, max_cells_per_class=50000, random_state=0):
    """Sample T-cell subtype classifier reference cells from the CODEX Step04B checkpoint."""
    adata = sc.read_h5ad(CODEX_STEP04B_H5AD)
    x = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    var_names = list(adata.var_names)
    if var_names != list(feature_names):
        feature_to_idx = {f: i for i, f in enumerate(var_names)}
        missing = [f for f in feature_names if f not in feature_to_idx]
        if missing:
            raise KeyError(f"Required features are missing from the CODEX Step04B checkpoint: {missing[:10]}")
        x = x[:, [feature_to_idx[f] for f in feature_names]]

    disease = adata.obs["Disease"].astype("string").str.upper()
    broad = adata.obs["Modeling_Broad"].astype("string")
    ref_mask = ((~disease.eq("DLBCL")) & broad.eq("T cells")).fillna(False).to_numpy()
    raw_labels = adata.obs.loc[ref_mask, "GT_Subtype"].astype("string")
    mapped_labels = raw_labels.map(lambda x: CODEX_T_SUBTYPE_TO_DISPLAY.get(str(x), np.nan) if pd.notna(x) else np.nan)
    valid_mask = mapped_labels.isin(T_SUBTYPE_ANNOTATION_LABEL_ORDER).to_numpy()
    x_ref = np.asarray(x[ref_mask][valid_mask], dtype=np.float32)
    y_ref = mapped_labels[valid_mask].astype(str).to_numpy()
    groups = adata.obs.loc[ref_mask, "PatientID"].astype("string").to_numpy()[valid_mask]

    rng = np.random.default_rng(random_state)
    keep_idx = []
    for label in np.unique(y_ref):
        idx = np.where(y_ref == label)[0]
        if max_cells_per_class is not None and len(idx) > max_cells_per_class:
            idx = rng.choice(idx, size=max_cells_per_class, replace=False)
        keep_idx.extend(idx.tolist())
    keep_idx = np.asarray(sorted(keep_idx))
    meta = pd.DataFrame({"label": y_ref[keep_idx], "PatientID": groups[keep_idx]})
    meta.value_counts(["label"], dropna=False).rename_axis("label").reset_index(name="n").to_csv(
        OUT_DIR / "tables" / "codex_tcell_subtype_training_label_counts.csv", index=False
    )
    meta.value_counts(["label", "PatientID"], dropna=False).rename_axis(["label", "PatientID"]).reset_index(name="n").to_csv(
        OUT_DIR / "tables" / "codex_tcell_subtype_training_label_patient_counts.csv", index=False
    )
    return x_ref[keep_idx], y_ref[keep_idx], groups[keep_idx]


def train_direct_tcell_subtype_classifier(feature_names, max_cells_per_class=50000, random_state=0, holdout_check=True):
    """Train a subtype classifier on CODEX reference T cells and save it as a direct-transfer package."""
    ensure_output_dirs()
    x, y, groups = load_codex_tcell_subtype_training_data(
        feature_names=feature_names,
        max_cells_per_class=max_cells_per_class,
        random_state=random_state,
    )
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    l1_ratio=0.15,
                    alpha=1e-4,
                    max_iter=1000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    holdout_metrics = None
    if holdout_check:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        train_idx, test_idx = next(splitter.split(x, y_enc, groups=groups))
        model.fit(x[train_idx], y_enc[train_idx])
        pred = model.predict(x[test_idx])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            bacc = balanced_accuracy_score(y_enc[test_idx], pred)
        macro_f1 = f1_score(y_enc[test_idx], pred, average="macro", zero_division=0)
        holdout_metrics = pd.DataFrame(
            [{"dataset": "CODEX reference T-cell subtype holdout", "n_cells": len(test_idx), "balanced_accuracy": bacc, "macro_f1": macro_f1}]
        )
        holdout_metrics.to_csv(OUT_DIR / "tables" / "codex_tcell_subtype_reference_holdout_metrics.csv", index=False)

    model.fit(x, y_enc)
    package = {
        "model": model,
        "label_encoder": le,
        "feature_names": list(feature_names),
        "feature_columns": [f"codex_{m}" for m in feature_names],
        "training_source": str(CODEX_STEP04B_H5AD),
        "max_cells_per_class": max_cells_per_class,
        "annotation_labels": list(T_SUBTYPE_ANNOTATION_LABEL_ORDER),
        "evaluation_labels": list(T_SUBTYPE_LABEL_ORDER),
        "holdout_metrics": holdout_metrics,
        "direct_transfer_warning": (
            "This package applies the CODEX T-cell subtype classifier to scRNA gene-proxy features. "
            "CD4_CTL is included in annotations but is absent from the external scRNA reference; "
            "BAcc/Macro F1 are therefore calculated only in the comparable TH/TFH/TREG/CD8T label space."
        ),
    }
    joblib.dump(package, DIRECT_TCELL_SUBTYPE_MODEL_PACKAGE)
    return package


def _frozen_feature_frame(a, feature_names):
    """Convert scRNA proxies in AnnData obs to a numeric matrix in manifest feature order."""
    feature_names = list(feature_names)
    feature_cols = [_feature_col(marker) for marker in feature_names]
    missing = [col for col in feature_cols if col not in a.obs]
    if missing:
        raise KeyError(
            "Required V9 frozen ensemble proxy features are missing from scRNA obs: "
            f"{missing[:10]}"
        )
    numeric = a.obs[feature_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    invalid = ~np.isfinite(values)
    if invalid.any():
        invalid_counts = invalid.sum(axis=0)
        bad = [
            f"{feature_names[i]}={int(count)}"
            for i, count in enumerate(invalid_counts)
            if count
        ]
        raise ValueError(
            "V9 frozen ensemble input contains NaN/inf. Stopping without arbitrary zero imputation: "
            + ", ".join(bad[:10])
        )
    return pd.DataFrame(values, index=a.obs_names, columns=feature_names)


def apply_v9_frozen_broad_classifier(
    a,
    transfer_root=V9_TRANSFER_ROOT,
    min_confidence=0.0,
    device="auto",
    batch_size=8192,
    show_progress=True,
):
    """Apply the broad ensemble selected and saved in V9 Step02 without retraining."""
    from V9_transfer_utils import predict_frozen_broad

    package_info = inspect_v9_frozen_transfer_package(transfer_root)
    feature_names = package_info["feature_names"]
    x = _frozen_feature_frame(a, feature_names)
    result = predict_frozen_broad(
        x,
        transfer_root,
        feature_names=feature_names,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
    )
    probs = result["probabilities"]
    pred_codex = np.asarray(result["labels"], dtype=object)
    conf = probs.max(axis=1).to_numpy(dtype=float)
    pred_display = pd.Series(pred_codex, index=a.obs.index).map(BROAD_DISPLAY_MAP).astype(object)
    if pred_display.isna().any():
        unknown = sorted(pd.Series(pred_codex[pred_display.isna().to_numpy()]).astype(str).unique())
        raise ValueError(f"Unknown V9 broad classes: {unknown}")
    pred_display.loc[conf < min_confidence] = "Low_confidence"
    a.obs["pred_broad_direct_codex"] = pred_codex
    a.obs["pred_broad_direct_display"] = pred_display
    a.obs["pred_broad_direct_confidence"] = conf
    for raw_class in result["classes"]:
        display_class = BROAD_DISPLAY_MAP.get(str(raw_class), str(raw_class))
        class_probs = probs[str(raw_class)].to_numpy(dtype=float)
        a.obs[f"prob_direct_{display_class}"] = class_probs
        a.obs[f"score_codex_ml_broad_{display_class}"] = class_probs
    a.obs["score_codex_ml_broad_max"] = conf
    a.uns["v9_frozen_broad_inference"] = {
        "manifest": str(Path(transfer_root) / "broad" / "manifest.json"),
        "selected_combo": result["selected_combo"],
        "selected_iterations": int(result["manifest"]["selected_iterations"]),
        "model_iterations": {key: list(map(int, vals)) for key, vals in result["model_iterations"].items()},
        "probability_rule": result["manifest"].get("probability_rule", ""),
        "spatial_rescue_applied": False,
        "scrna_retraining_applied": False,
    }
    return a


def apply_v9_frozen_tcell_subtype_classifier(
    a,
    transfer_root=V9_TRANSFER_ROOT,
    min_confidence=0.0,
    gate_col="pred_broad_direct_codex",
    device="auto",
    batch_size=8192,
    show_progress=True,
):
    """Apply the Step04B T-subtype ensemble only to cells predicted as T cells by the broad frozen model."""
    from V9_transfer_utils import predict_frozen_stage

    if gate_col not in a.obs:
        raise KeyError(f"Broad prediction column required for T-cell subtype routing is missing: {gate_col}")
    gate_values = a.obs[gate_col].astype(str)
    t_mask = gate_values.eq("T cells").to_numpy()
    pred_subtype = np.full(a.n_obs, "Non_T_cells", dtype=object)
    pred_display = np.full(a.n_obs, "Non_T_cells", dtype=object)
    confidence = np.zeros(a.n_obs, dtype=float)
    for label in T_SUBTYPE_ANNOTATION_LABEL_ORDER:
        a.obs[f"prob_direct_tcell_{label}"] = 0.0
        a.obs[f"score_codex_ml_tcell_{label}"] = 0.0

    package_info = inspect_v9_frozen_transfer_package(transfer_root)
    tcell_manifest = package_info["tcell_manifest"]
    if t_mask.any():
        feature_names = package_info["feature_names"]
        x = _frozen_feature_frame(a, feature_names)
        result = predict_frozen_stage(
            x.loc[t_mask],
            Path(transfer_root) / "subtype" / "T_cells",
            feature_names=feature_names,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )
        raw_labels = np.asarray(result["labels"], dtype=object)
        mapped_labels = np.asarray(
            [CODEX_T_SUBTYPE_TO_DISPLAY.get(str(label), str(label)) for label in raw_labels],
            dtype=object,
        )
        unknown = sorted(set(map(str, mapped_labels)) - set(T_SUBTYPE_ANNOTATION_LABEL_ORDER))
        if unknown:
            raise ValueError(f"Unknown V9 T-cell subtype classes: {unknown}")
        routed_confidence = result["probabilities"].max(axis=1).to_numpy(dtype=float)
        routed_display = mapped_labels.copy()
        routed_display[routed_confidence < min_confidence] = "Low_confidence"
        pred_subtype[t_mask] = mapped_labels
        pred_display[t_mask] = routed_display
        confidence[t_mask] = routed_confidence
        for raw_class in result["classes"]:
            display_class = CODEX_T_SUBTYPE_TO_DISPLAY.get(str(raw_class), str(raw_class))
            class_probs = result["probabilities"][str(raw_class)].to_numpy(dtype=float)
            a.obs.loc[t_mask, f"prob_direct_tcell_{display_class}"] = class_probs
            a.obs.loc[t_mask, f"score_codex_ml_tcell_{display_class}"] = class_probs
        model_iterations = {
            key: list(map(int, vals)) for key, vals in result["model_iterations"].items()
        }
        selected_combo = result["selected_combo"]
    else:
        model_iterations = {}
        selected_combo = str(tcell_manifest["selected_combo"])

    a.obs["pred_tcell_subtype_direct"] = pred_subtype
    a.obs["pred_tcell_subtype_direct_display"] = pred_display
    a.obs["pred_tcell_subtype_direct_confidence"] = confidence
    a.obs["score_codex_ml_tcell_max"] = confidence
    a.uns["v9_frozen_tcell_subtype_inference"] = {
        "manifest": str(Path(transfer_root) / "subtype" / "T_cells" / "manifest.json"),
        "selected_combo": selected_combo,
        "selected_iterations": int(tcell_manifest["selected_iterations"]),
        "model_iterations": model_iterations,
        "routing_column": gate_col,
        "routing_label": "T cells",
        "n_routed_cells": int(t_mask.sum()),
        "spatial_rescue_applied": False,
        "scrna_retraining_applied": False,
    }
    return a


def apply_roider_v9_frozen_predictions(
    adata,
    transfer_root=V9_TRANSFER_ROOT,
    min_direct_confidence=0.0,
    min_tcell_subtype_confidence=0.0,
    device="auto",
    batch_size=8192,
    show_progress=True,
):
    """Attach marker-baseline and V9 frozen broad/subtype annotations to Roider scRNA."""
    attach_reference_annotations(adata)
    adata = add_marker_based_predictions(adata)
    adata = apply_v9_frozen_broad_classifier(
        adata,
        transfer_root=transfer_root,
        min_confidence=min_direct_confidence,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    adata = apply_v9_frozen_tcell_subtype_classifier(
        adata,
        transfer_root=transfer_root,
        min_confidence=min_tcell_subtype_confidence,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    adata = add_display_ready_columns(adata)
    adata = add_unified_prediction_scores(adata)
    return adata


def apply_direct_broad_classifier(a, package, min_confidence=0.0):
    """Apply the trained CODEX broad classifier directly to scRNA proxy features."""
    feature_cols = package["feature_columns"]
    missing = [c for c in feature_cols if c not in a.obs]
    if missing:
        raise KeyError(f"Required proxy features are missing from scRNA obs: {missing[:10]}")
    x = a.obs[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    model = package["model"]
    le = package["label_encoder"]
    probs = model.predict_proba(x)
    pred_idx = probs.argmax(axis=1)
    pred_codex = le.inverse_transform(pred_idx)
    conf = probs.max(axis=1)
    pred_display = pd.Series(pred_codex, index=a.obs.index).map(BROAD_DISPLAY_MAP).astype(object)
    pred_display.loc[conf < min_confidence] = "Low_confidence"
    a.obs["pred_broad_direct_codex"] = pred_codex
    a.obs["pred_broad_direct_display"] = pred_display
    a.obs["pred_broad_direct_confidence"] = conf.astype(float)
    for class_idx, class_name in enumerate(le.classes_):
        display_name = BROAD_DISPLAY_MAP.get(class_name, class_name)
        a.obs[f"prob_direct_{display_name}"] = probs[:, class_idx].astype(float)
        a.obs[f"score_codex_ml_broad_{display_name}"] = probs[:, class_idx].astype(float)
    a.obs["score_codex_ml_broad_max"] = conf.astype(float)
    return a


def apply_direct_tcell_subtype_classifier(a, package, min_confidence=0.0, gate_col="pred_broad_direct_display"):
    """Apply the trained CODEX T-cell subtype classifier directly to scRNA proxy features."""
    feature_cols = package["feature_columns"]
    missing = [c for c in feature_cols if c not in a.obs]
    if missing:
        raise KeyError(f"Required proxy features are missing from scRNA obs: {missing[:10]}")
    x = a.obs[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    model = package["model"]
    le = package["label_encoder"]
    probs = model.predict_proba(x)
    pred_idx = probs.argmax(axis=1)
    pred_subtype = le.inverse_transform(pred_idx)
    conf = probs.max(axis=1)
    pred_display = pd.Series(pred_subtype, index=a.obs.index, dtype=object)
    pred_display.loc[conf < min_confidence] = "Low_confidence"
    if gate_col is not None and gate_col in a.obs:
        pred_display.loc[a.obs[gate_col].astype(str) != "T_cells"] = "Non_T_cells"
    a.obs["pred_tcell_subtype_direct"] = pred_subtype.astype(object)
    a.obs["pred_tcell_subtype_direct_display"] = pred_display.astype(object)
    a.obs["pred_tcell_subtype_direct_confidence"] = conf.astype(float)
    for class_idx, class_name in enumerate(le.classes_):
        a.obs[f"prob_direct_tcell_{class_name}"] = probs[:, class_idx].astype(float)
        a.obs[f"score_codex_ml_tcell_{class_name}"] = probs[:, class_idx].astype(float)
    a.obs["score_codex_ml_tcell_max"] = conf.astype(float)
    return a


def evaluate_direct_transfer(
    a,
    truth_col="reference_broad_codex",
    pred_col="pred_broad_direct_codex",
    dataset_name="Roider scRNA reference-labeled cells",
    out_dir=OUT_DIR,
    table_prefix="direct_transfer_broad",
):
    """Calculate direct-transfer performance in scRNA cells with reference labels."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    raw_df = a.obs[[truth_col, pred_col, "pred_broad_direct_confidence"]].copy()
    reference_mask = raw_df[truth_col].isin(CODEX_BROAD_LABEL_ORDER)
    eval_df = raw_df.loc[reference_mask].dropna(subset=[truth_col, pred_col]).copy()
    if eval_df.empty:
        raise ValueError(f"No evaluable reference labels: truth_col={truth_col}")
    y_true = eval_df[truth_col].astype(str)
    y_pred = eval_df[pred_col].astype(str)
    labels = CODEX_BROAD_LABEL_ORDER
    present_labels = [label for label in labels if label in set(y_true)]
    absent_reference_labels = [label for label in labels if label not in set(y_true)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1_present = f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0) if present_labels else np.nan
    macro_f1_model_space = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    metrics = pd.DataFrame(
        [{
            "dataset": dataset_name,
            "n_cells": len(eval_df),
            "n_reference_missing_or_unmapped": int((~reference_mask).sum()),
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1_present,
            "macro_f1_reference_present_classes": macro_f1_present,
            "macro_f1_model_label_space": macro_f1_model_space,
            "reference_present_classes": "|".join(present_labels),
            "reference_absent_classes_excluded_from_macro": "|".join(absent_reference_labels),
            "mean_confidence": float(eval_df["pred_broad_direct_confidence"].mean()),
        }]
    )
    cm = pd.crosstab(y_true, y_pred).reindex(index=present_labels, columns=labels, fill_value=0)
    report = pd.DataFrame(classification_report(y_true, y_pred, labels=present_labels, output_dict=True, zero_division=0)).T
    metrics.to_csv(out_dir / "tables" / f"{table_prefix}_metrics.csv", index=False)
    cm.to_csv(out_dir / "tables" / f"{table_prefix}_confusion_matrix.csv")
    report.to_csv(out_dir / "tables" / f"{table_prefix}_classification_report.csv")
    return metrics, cm, report


def evaluate_prediction_table(
    a,
    truth_col,
    pred_col,
    labels,
    dataset_name,
    method_name,
    out_dir,
    table_prefix,
    drop_pred_outside_labels=False,
):
    """Save BAcc, Macro F1, and confusion matrices for specified truth/prediction columns."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    labels = list(labels)
    raw_df = a.obs[[truth_col, pred_col]].copy()
    reference_mask = raw_df[truth_col].isin(labels)
    eval_df = raw_df.loc[reference_mask].dropna(subset=[truth_col, pred_col]).copy()
    pred_outside_labels = []
    n_pred_outside_eval_label_space = 0
    if drop_pred_outside_labels:
        pred_in_label_mask = eval_df[pred_col].isin(labels)
        pred_outside = eval_df.loc[~pred_in_label_mask, pred_col].astype(str)
        pred_outside_labels = sorted(pred_outside.unique().tolist())
        n_pred_outside_eval_label_space = int((~pred_in_label_mask).sum())
        eval_df = eval_df.loc[pred_in_label_mask].copy()
    if eval_df.empty:
        raise ValueError(f"No evaluable reference labels: truth_col={truth_col}, pred_col={pred_col}")
    y_true = eval_df[truth_col].astype(str)
    y_pred = eval_df[pred_col].astype(str)
    present_labels = [label for label in labels if label in set(y_true)]
    absent_reference_labels = [label for label in labels if label not in set(y_true)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1_present = f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0) if present_labels else np.nan
    macro_f1_model_space = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    metrics = pd.DataFrame(
        [{
            "dataset": dataset_name,
            "method": method_name,
            "n_cells": len(eval_df),
            "n_reference_missing_or_unmapped": int((~reference_mask).sum()),
            "n_pred_outside_eval_label_space": n_pred_outside_eval_label_space,
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1_present,
            "macro_f1_reference_present_classes": macro_f1_present,
            "macro_f1_model_label_space": macro_f1_model_space,
            "reference_present_classes": "|".join(present_labels),
            "reference_absent_classes_excluded_from_macro": "|".join(absent_reference_labels),
            "pred_classes_excluded_from_metric": "|".join(pred_outside_labels),
        }]
    )
    cm = pd.crosstab(y_true, y_pred).reindex(index=present_labels, columns=labels, fill_value=0)
    report = pd.DataFrame(classification_report(y_true, y_pred, labels=present_labels, output_dict=True, zero_division=0)).T
    metrics.to_csv(out_dir / "tables" / f"{table_prefix}_metrics.csv", index=False)
    cm.to_csv(out_dir / "tables" / f"{table_prefix}_confusion_matrix.csv")
    report.to_csv(out_dir / "tables" / f"{table_prefix}_classification_report.csv")
    return metrics, cm, report


def evaluate_broad_lineage_methods(a, dataset_name, out_dir):
    """Save evaluation results for broad-lineage marker-base and direct CODEX ML models."""
    method_specs = [
        ("Marker Base", "pred_broad_marker_codex", "marker_broad_lineage"),
        ("CODEX ML model", "pred_broad_direct_codex", "codex_ml_broad_lineage"),
    ]
    rows = []
    cms = {}
    reports = {}
    for method_name, pred_col, prefix in method_specs:
        metrics, cm, report = evaluate_prediction_table(
            a,
            truth_col="reference_broad_codex",
            pred_col=pred_col,
            labels=CODEX_BROAD_LABEL_ORDER,
            dataset_name=dataset_name,
            method_name=method_name,
            out_dir=out_dir,
            table_prefix=prefix,
        )
        rows.append(metrics)
        cms[method_name] = cm
        reports[method_name] = report
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(Path(out_dir) / "tables" / "broad_lineage_method_metrics.csv", index=False)
    return combined, cms, reports


def evaluate_tcell_subtype_methods(a, dataset_name, out_dir):
    """Save evaluation results for T-cell subtype marker-base and direct CODEX ML models."""
    method_specs = [
        ("Marker Base", "pred_tcell_subtype_marker", "marker_tcell_subtype"),
        ("CODEX ML model", "pred_tcell_subtype_direct", "direct_tcell_subtype"),
    ]
    rows = []
    cms = {}
    reports = {}
    for method_name, pred_col, prefix in method_specs:
        metrics, cm, report = evaluate_prediction_table(
            a,
            truth_col="reference_tcell_subtype",
            pred_col=pred_col,
            labels=T_SUBTYPE_LABEL_ORDER,
            dataset_name=dataset_name,
            method_name=method_name,
            out_dir=out_dir,
            table_prefix=prefix,
            # Evaluate only shared subtype labels present in the scRNA reference.
            # Exclude predictions outside the reference label space (CD4_CTL, Tpr,
            # Non_T_cells, etc.) from subtype metrics, as in the existing Final analysis.
            drop_pred_outside_labels=True,
        )
        rows.append(metrics)
        cms[method_name] = cm
        reports[method_name] = report
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(Path(out_dir) / "tables" / "tcell_subtype_method_metrics.csv", index=False)
    return combined, cms, reports


def save_broad_lineage_evaluation_figures(a, out_dir, dataset_name, save_confusion=False):
    """Save broad-lineage evaluation tables and Marker/CODEX ML performance bar plots."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    metrics, cms, reports = evaluate_broad_lineage_methods(a, dataset_name=dataset_name, out_dir=out_dir)
    save_method_metric_plot(
        metrics,
        out_dir / "figures" / "broad_lineage_bacc_macro_f1.png",
        f"{dataset_name}: broad lineage",
    )
    if save_confusion:
        for method, cm in cms.items():
            safe_method = method.lower().replace(" ", "_")
            save_confusion_heatmap(
                cm,
                f"{dataset_name}: {method} broad lineage",
                out_dir / "figures" / f"broad_lineage_{safe_method}_confusion.png",
            )
    return metrics, cms, reports


def apply_gse182434_direct_broad_classifier(
    package,
    min_confidence=0.0,
    force_recompute_features=False,
    tcell_subtype_package=None,
    tcell_min_confidence=0.0,
):
    """Apply direct CODEX broad/subtype classifiers to GSE182434 and save metrics."""
    a, coverage = add_gse182434_codex_proxy_features(
        package["feature_names"],
        force_recompute=force_recompute_features,
    )
    add_marker_based_predictions(a)
    a = apply_direct_broad_classifier(a, package, min_confidence=min_confidence)
    if tcell_subtype_package is not None:
        a = apply_direct_tcell_subtype_classifier(a, tcell_subtype_package, min_confidence=tcell_min_confidence)
    add_display_ready_columns(a)
    add_unified_prediction_scores(a)
    save_prediction_score_tables(
        a,
        out_dir=GSE182434_OUT_DIR,
        dataset_name="GSE182434 scRNA",
        prefix="gse182434_prediction_scores",
    )
    metrics, cm, report = evaluate_direct_transfer(
        a,
        dataset_name="GSE182434 scRNA CellType-labeled cells",
        out_dir=GSE182434_OUT_DIR,
        table_prefix="direct_transfer_broad",
    )
    broad_eval = evaluate_broad_lineage_methods(
        a,
        dataset_name="GSE182434 scRNA CellType-labeled cells",
        out_dir=GSE182434_OUT_DIR,
    )
    subtype_eval = None
    if tcell_subtype_package is not None:
        subtype_eval = evaluate_tcell_subtype_methods(
            a,
            dataset_name="GSE182434 scRNA CellType-labeled T cells",
            out_dir=GSE182434_OUT_DIR,
        )
    write_h5ad_safe(a, GSE182434_PREDICTED_H5AD, cache_stage="gse182434_direct_predictions_v3_broad_tcell_cd4ctl_annotation")
    return a, metrics, cm, report, coverage, subtype_eval, broad_eval


def save_confusion_heatmap(cm, title, path):
    """Save a confusion-matrix heatmap."""
    path = Path(path)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("Prediction", fontsize=14, fontweight="bold")
    ax.set_ylabel("Reference", fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save_scrna_figure_png_pptx(fig, path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(metrics, path):
    """Save BAcc and Macro F1 bar plots."""
    path = Path(path)
    plot_df = metrics.melt(value_vars=["balanced_accuracy", "macro_f1"], var_name="metric", value_name="score")
    plot_df["metric"] = plot_df["metric"].map({"balanced_accuracy": "BAcc", "macro_f1": "Macro F1"})
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.barplot(data=plot_df, x="metric", y="score", palette=["#4E79A7", "#E15759"], ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Direct CODEX classifier transfer", fontsize=16, fontweight="bold")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=11, padding=3)
    fig.tight_layout()
    _save_scrna_figure_png_pptx(fig, path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_method_metric_plot(metrics, path, title):
    """Save BAcc and Macro F1 comparisons across methods."""
    path = Path(path)
    plot_df = metrics.melt(
        id_vars=["method"],
        value_vars=["balanced_accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map({"balanced_accuracy": "BAcc", "macro_f1": "Macro F1"})
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.barplot(data=plot_df, x="method", y="score", hue="metric", palette=["#4E79A7", "#E15759"], ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(title="", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=10, padding=3)
    fig.tight_layout()
    _save_scrna_figure_png_pptx(fig, path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_scrna_performance_comparison_plot(
    broad_metrics,
    subtype_metrics,
    path,
    title,
    figsize=(11.69, 8.27),
    layout="horizontal",
    font_scale=1.0,
    force_a4_canvas=None,
    bar_width_scale=0.92,
):
    """Save broad-lineage and T-cell subtype performance comparisons on the specified canvas."""
    path = Path(path)
    plot_parts = []
    for task_label, metrics in [
        ("Broad lineage", broad_metrics),
        ("T-cell subtype", subtype_metrics),
    ]:
        if metrics is None or len(metrics) == 0:
            continue
        task_df = metrics.copy()
        task_df["task"] = task_label
        plot_parts.append(task_df)
    if not plot_parts:
        raise ValueError("No metric table is available for the performance comparison plot.")

    plot_df = pd.concat(plot_parts, ignore_index=True).melt(
        id_vars=["task", "method"],
        value_vars=["balanced_accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map({"balanced_accuracy": "BAcc", "macro_f1": "Macro F1"})
    task_order = [task for task in ["Broad lineage", "T-cell subtype"] if task in set(plot_df["task"])]
    if layout == "vertical":
        n_rows, n_cols = len(task_order), 1
    else:
        n_rows, n_cols = 1, len(task_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axis_label_font = 10.5 * font_scale
    panel_title_font = 12.0 * font_scale
    tick_font = 9.0 * font_scale
    legend_font = 9.0 * font_scale
    value_font = 8.5 * font_scale
    suptitle_font = 12.0 * font_scale if layout == "vertical" else 13.5 * font_scale
    display_title = '' if title is None else str(title).strip()
    if layout == "vertical":
        display_title = "\n".join(textwrap.wrap(display_title, width=32))
    legend_handles = None
    legend_labels = None
    metric_order = ["BAcc", "Macro F1"]
    metric_colors = {"BAcc": "#4E79A7", "Macro F1": "#E15759"}
    preferred_method_order = ["Marker Base", "CODEX ML model"]
    for ax, task_label in zip(axes.ravel(), task_order):
        task_plot = plot_df.loc[plot_df["task"].eq(task_label)].copy()
        method_order = [m for m in preferred_method_order if m in set(task_plot["method"])]
        method_order += [m for m in task_plot["method"].drop_duplicates() if m not in method_order]
        group_spacing = 0.68
        base_bar_width = 0.25
        bar_width = base_bar_width * bar_width_scale
        method_positions = np.arange(len(method_order), dtype=float) * group_spacing
        score_lookup = task_plot.set_index(["method", "metric"])["score"].to_dict()
        for metric_idx, metric in enumerate(metric_order):
            offset = (metric_idx - (len(metric_order) - 1) / 2) * bar_width
            heights = [score_lookup.get((method, metric), np.nan) for method in method_order]
            bars = ax.bar(
                method_positions + offset,
                heights,
                width=bar_width,
                color=metric_colors[metric],
                edgecolor="white",
                linewidth=0.8,
                label=metric,
            )
            for bar in bars:
                height = bar.get_height()
                if np.isfinite(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.014,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=value_font,
                    )
        ax.set_ylim(0, 1.12)
        if method_order:
            ax.set_xlim(method_positions[0] - 0.30, method_positions[-1] + 0.30)
        ax.set_xlabel("")
        ax.set_ylabel("Score", fontsize=axis_label_font, fontweight="bold")
        ax.set_title(task_label, fontsize=panel_title_font, fontweight="bold", pad=8)
        ax.tick_params(axis="both", labelsize=tick_font)
        if legend_handles is None:
            legend_handles = [Patch(facecolor=metric_colors[m], label=m) for m in metric_order]
            legend_labels = metric_order
        wrapped_labels = [
            "\n".join(textwrap.wrap(method, width=11)) for method in method_order
        ]
        ax.set_xticks(method_positions)
        ax.set_xticklabels(wrapped_labels)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    if legend_handles and legend_labels:
        # Vertical manuscript panels use a dedicated top band for the shared
        # legend.  Keeping it above the first panel title prevents the legend
        # from colliding with "Broad lineage" on compact canvases.
        if layout == "vertical":
            legend_y = 0.985 if not display_title else 0.905
        else:
            legend_y = 0.855
        fig.legend(
            legend_handles,
            legend_labels,
            title="",
            frameon=False,
            fontsize=legend_font,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(legend_labels),
            borderaxespad=0.0,
        )
    if display_title:
        fig.suptitle(display_title, fontsize=suptitle_font, fontweight="bold", y=0.985 if layout == "vertical" else 0.955)
    if layout == "vertical":
        fig.subplots_adjust(
            left=0.145, right=0.985, bottom=0.095,
            top=0.80 if display_title else 0.86,
            hspace=0.42
        )
    else:
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.76, wspace=0.16)
    global FINAL_A4_OUTPUT_CANVAS
    old_force_a4 = FINAL_A4_OUTPUT_CANVAS
    if force_a4_canvas is not None:
        FINAL_A4_OUTPUT_CANVAS = bool(force_a4_canvas)
    try:
        _save_scrna_figure_png_pptx(fig, path, dpi=350)
    finally:
        FINAL_A4_OUTPUT_CANVAS = old_force_a4
    plt.close(fig)


def _ordered_categories(vals, base_order):
    """Prioritize the specified order and append remaining categories."""
    vals = pd.Series(vals).astype(object).where(pd.notna(vals), "Missing").astype(str)
    present = set(vals)
    ordered = [c for c in base_order if c in present]
    ordered += [c for c in vals.drop_duplicates().tolist() if c not in ordered]
    return vals, ordered


def _palette_for_categories(categories, preferred=None):
    """Create a color mapping for categories."""
    preferred = preferred or {}
    palette = {}
    fallback = sns.color_palette("tab20", max(len(categories), 1)).as_hex()
    j = 0
    for cat in categories:
        if cat in preferred:
            palette[cat] = preferred[cat]
        else:
            palette[cat] = fallback[j % len(fallback)]
            j += 1
    return palette


def _plot_categorical_umap(
    ax,
    coords,
    vals,
    title,
    legend_title,
    order,
    palette,
    point_size,
    legend_loc="best",
    category_renames=None,
    legend_outside=False,
    font_scale=1.0,
    draw_legend=True,
    panel_title_scale=1.0,
    axis_label_scale=1.0,
    legend_scale=1.0,
):
    """Draw a categorical UMAP panel."""
    category_renames = category_renames or {}
    if category_renames:
        vals = pd.Series(vals).astype(object).replace(category_renames)
        order = [category_renames.get(cat, cat) for cat in order]
        palette = {category_renames.get(cat, cat): color for cat, color in (palette or {}).items()}
    vals, categories = _ordered_categories(vals, order)
    palette = _palette_for_categories(categories, palette)
    draw_point_size = min(float(point_size), 4.0)
    for cat in categories:
        idx = vals.eq(cat).to_numpy()
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=draw_point_size,
            c=palette.get(cat, "#808080"),
            label=cat,
            alpha=0.78,
            linewidths=0,
        )
    ax.set_title(title, fontsize=11.1 * font_scale * panel_title_scale, fontweight="bold", pad=7)
    ax.set_xlabel("UMAP1", fontsize=8.2 * font_scale * axis_label_scale, fontweight="bold", labelpad=2)
    ax.set_ylabel("UMAP2", fontsize=8.2 * font_scale * axis_label_scale, fontweight="bold", labelpad=2)
    ax.tick_params(axis="both", labelsize=7.4 * font_scale * axis_label_scale, pad=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    if len(categories) <= 6:
        legend_ncol = 1
    elif len(categories) <= 9:
        legend_ncol = 2
    elif len(categories) <= 16:
        legend_ncol = 3
    else:
        legend_ncol = 4
    legend_kwargs = {
        "markerscale": 2.85 * legend_scale,
        "fontsize": 7.9 * font_scale * legend_scale,
        "title": legend_title,
        "title_fontsize": 8.4 * font_scale * legend_scale,
        "frameon": True,
        "framealpha": 0.90,
        "borderpad": 0.42 * legend_scale,
        "labelspacing": 0.34 * legend_scale,
        "handletextpad": 0.44 * legend_scale,
        "columnspacing": 0.78 * legend_scale,
        "ncol": legend_ncol,
    }
    if draw_legend:
        if legend_outside:
            if str(legend_loc).startswith("upper"):
                legend_kwargs.update({"loc": "upper left", "bbox_to_anchor": (1.01, 1.0), "borderaxespad": 0.0})
            elif str(legend_loc).startswith("lower"):
                legend_kwargs.update({"loc": "lower left", "bbox_to_anchor": (1.01, 0.0), "borderaxespad": 0.0})
            else:
                legend_kwargs.update({"loc": "center left", "bbox_to_anchor": (1.01, 0.5), "borderaxespad": 0.0})
        else:
            legend_kwargs.update({"loc": legend_loc})
        ax.legend(
            **legend_kwargs,
        )
    return {
        "title": legend_title,
        "categories": categories,
        "palette": palette,
        "ncol": legend_ncol,
    }


def _default_legend_loc(legend_loc):
    """Obtain the shared legend position from panel-specific legend settings."""
    if isinstance(legend_loc, dict):
        return legend_loc.get("_default", "best")
    return legend_loc


def _compute_umap_view(coords, legend_loc="best", base_pad_frac=0.045):
    """Calculate common axis limits for all UMAP panels."""
    coords = np.asarray(coords, dtype=float)
    finite_mask = np.isfinite(coords).all(axis=1)
    coords = coords[finite_mask]
    if coords.size == 0:
        return None, None

    x0, x1 = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    y0, y1 = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    x_span = max(x1 - x0, 1e-6)
    y_span = max(y1 - y0, 1e-6)
    span = max(x_span, y_span)
    x_center = (x0 + x1) / 2.0
    y_center = (y0 + y1) / 2.0
    half = span * (0.5 + base_pad_frac)
    x0, x1 = x_center - half, x_center + half
    y0, y1 = y_center - half, y_center + half

    loc = str(legend_loc).lower()
    legend_x_pad = span * 0.16
    legend_y_pad = span * 0.13
    if "right" in loc:
        x1 += legend_x_pad
    elif "left" in loc:
        x0 -= legend_x_pad
    if "upper" in loc:
        y1 += legend_y_pad
    elif "lower" in loc:
        y0 -= legend_y_pad
    return (x0, x1), (y0, y1)


def _apply_umap_view(ax, xlim, ylim):
    """Apply shared axes to keep UMAP coordinates consistent across panels."""
    if xlim is None or ylim is None:
        return
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)


def _plot_continuous_umap(
    ax,
    coords,
    vals,
    title,
    point_size,
    font_scale=1.0,
    panel_title_scale=1.0,
    axis_label_scale=1.0,
    colorbar_scale=1.0,
    colorbar_fraction=0.052,
    colorbar_pad=0.055,
    colorbar_tick_side="left",
):
    """Draw a continuous UMAP panel."""
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    draw_point_size = min(float(point_size), 4.0)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=draw_point_size, c=vals, cmap="viridis", alpha=0.82, linewidths=0)
    ax.set_title(title, fontsize=11.1 * font_scale * panel_title_scale, fontweight="bold", pad=7)
    ax.set_xlabel("UMAP1", fontsize=8.2 * font_scale * axis_label_scale, fontweight="bold", labelpad=2)
    ax.set_ylabel("UMAP2", fontsize=8.2 * font_scale * axis_label_scale, fontweight="bold", labelpad=2)
    ax.tick_params(axis="both", labelsize=7.4 * font_scale * axis_label_scale, pad=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    colorbar_tick_side = str(colorbar_tick_side).lower()
    if colorbar_tick_side not in {"left", "right"}:
        colorbar_tick_side = "right"
    # Attaching a colorbar directly to ax reduces the width of that UMAP panel.
    # Use a separate inset axis to keep all panels in the 3x3 UMAP equally sized.
    colorbar_width = max(0.018, float(colorbar_fraction))
    colorbar_x = 1.0 + float(colorbar_pad)
    cax = inset_axes(
        ax,
        width=f"{colorbar_width * 100:.2f}%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(colorbar_x, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.yaxis.set_ticks_position(colorbar_tick_side)
    cbar.ax.yaxis.set_label_position(colorbar_tick_side)
    cbar.ax.tick_params(labelsize=9.0 * font_scale * colorbar_scale, pad=1)
    cbar.set_label("")


def _metadata_celltype_column(a):
    """Select the column for the metadata cell-type panel."""
    for col in ["CellType", "reference_population", "reference_population_raw", "leiden_0_8"]:
        if col in a.obs:
            return col
    return None


def _umap_reference_mask(a, task="broad"):
    """Create a mask excluding cells without ground-truth/reference labels from UMAP."""
    if task == "tcell_subtype":
        if "reference_tcell_subtype" not in a.obs:
            return np.zeros(a.n_obs, dtype=bool)
        return a.obs["reference_tcell_subtype"].isin(T_SUBTYPE_LABEL_ORDER).to_numpy(dtype=bool)
    if "reference_broad_display" not in a.obs:
        return np.zeros(a.n_obs, dtype=bool)
    return a.obs["reference_broad_display"].notna().to_numpy(dtype=bool)


def save_nine_umap_panel(
    a,
    path,
    title="Direct CODEX classifier transfer",
    point_size=5,
    panel_order=None,
    legend_loc="best",
    category_renames=None,
    legend_outside=False,
    figsize=(19.5, 17.2),
    force_a4_canvas=None,
    font_scale=1.0,
    row_shared_legends=False,
    share_axis_labels=False,
    shared_legend_position="right",
    panel_title_scale=1.0,
    axis_label_scale=1.0,
    legend_scale=1.0,
    colorbar_scale=1.0,
    colorbar_fraction=0.052,
    colorbar_pad=0.055,
    colorbar_tick_side="left",
    suptitle_scale=1.0,
    right_legend_width=0.55,
    right_legend_wspace=0.035,
    right_legend_anchor_x=0.035,
    show_umap_ticks=True,
):
    """Save reference/marker-base/CODEX ML, metadata, and HLA-DR as a 3x3 UMAP."""
    path = Path(path)
    if "X_umap" not in a.obsm:
        raise KeyError("adata.obsm['X_umap'] is missing.")
    add_display_ready_columns(a)
    coords = a.obsm["X_umap"]
    shared_legend_position = str(shared_legend_position).lower()
    if shared_legend_position not in {"right", "bottom", "inside_last"}:
        raise ValueError("shared_legend_position must be 'right', 'bottom', or 'inside_last'.")
    common_legend_loc = _default_legend_loc(legend_loc) if shared_legend_position == "inside_last" else "best"
    common_xlim, common_ylim = _compute_umap_view(coords, legend_loc=common_legend_loc)
    sample_col = "Sample" if "Sample" in a.obs else None
    celltype_col = _metadata_celltype_column(a)

    panel_defs = {
        "metadata_sample": ("categorical", sample_col, "Metadata Sample", "Metadata Sample", [], {}, "broad"),
        "metadata_celltype": ("categorical", celltype_col, "Metadata Celltype", "Metadata Celltype", [], {}, "broad"),
        "hladr": ("continuous", "expr_HLADR", "HLA-DR Expression", "HLA-DR", [], {}, "broad"),
        "reference_broad": ("categorical", "reference_broad_display", "Reference Broad label", "Reference Broad", DISPLAY_BROAD_LABEL_ORDER, BROAD_COLOR_MAP, "broad"),
        "marker_broad": ("categorical", "pred_broad_marker_display_eval", "Marker Base", "Marker Broad", DISPLAY_BROAD_LABEL_ORDER + ["Low_confidence"], BROAD_COLOR_MAP, "broad"),
        "codex_broad": ("categorical", "pred_broad_direct_display_eval", "CODEX ML model", "CODEX Broad", DISPLAY_BROAD_LABEL_ORDER + ["Low_confidence"], BROAD_COLOR_MAP, "broad"),
        "reference_tcell": ("categorical", "reference_tcell_subtype_display", "Reference T cell subtype", "Reference T subtype", T_SUBTYPE_DISPLAY_ORDER, T_SUBTYPE_COLOR_MAP, "tcell_subtype"),
        "marker_tcell": ("categorical", "pred_tcell_subtype_marker_display_eval", "Marker Base", "Marker T subtype", T_SUBTYPE_DISPLAY_ORDER + ["Low_confidence"], T_SUBTYPE_COLOR_MAP, "tcell_subtype"),
        "codex_tcell": ("categorical", "pred_tcell_subtype_direct_display_eval", "CODEX ML model", "CODEX T subtype", T_SUBTYPE_DISPLAY_ORDER + ["Low_confidence"], T_SUBTYPE_COLOR_MAP, "tcell_subtype"),
    }
    if panel_order is None:
        panel_order = [
            "reference_broad",
            "marker_broad",
            "codex_broad",
            "reference_tcell",
            "marker_tcell",
            "codex_tcell",
            "metadata_sample",
            "metadata_celltype",
            "hladr",
        ]
    panels = [(panel_key, panel_defs[panel_key]) for panel_key in panel_order]
    category_renames = category_renames or {}
    shared_legend_groups = {
        "broad": {
            "keys": {"reference_broad", "marker_broad", "codex_broad"},
            "title": "Broad label",
            "order": DISPLAY_BROAD_LABEL_ORDER + ["Low_confidence"],
        },
        "tcell_subtype": {
            "keys": {"reference_tcell", "marker_tcell", "codex_tcell"},
            "title": "T cell subtype",
            "order": T_SUBTYPE_DISPLAY_ORDER + ["Low_confidence"],
        },
    }
    shared_legend_group_by_row = {}
    if row_shared_legends:
        for row_idx in range(3):
            row_keys = {panel_key for panel_key, _ in panels[row_idx * 3 : (row_idx + 1) * 3]}
            for group_name, group_spec in shared_legend_groups.items():
                if row_keys == group_spec["keys"]:
                    shared_legend_group_by_row[row_idx] = group_name
                    break
    shared_legend_data = {}
    metadata_legend_data = []

    shared_legend_axes = {}
    bottom_legend_ax = None
    if row_shared_legends and shared_legend_position == "bottom":
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            4,
            3,
            width_ratios=[1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0, 1.0, 0.72],
            hspace=0.14,
            wspace=0.06,
        )
        axes = np.empty((3, 3), dtype=object)
        for row_idx in range(3):
            for col_idx in range(3):
                axes[row_idx, col_idx] = fig.add_subplot(grid[row_idx, col_idx])
        bottom_legend_ax = fig.add_subplot(grid[3, :])
        bottom_legend_ax.set_axis_off()
    elif row_shared_legends and shared_legend_position == "right":
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            3,
            4,
            width_ratios=[1.0, 1.0, 1.0, float(right_legend_width)],
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.16,
            wspace=float(right_legend_wspace),
        )
        axes = np.empty((3, 3), dtype=object)
        for row_idx in range(3):
            for col_idx in range(3):
                axes[row_idx, col_idx] = fig.add_subplot(grid[row_idx, col_idx])
        right_legend_ax = fig.add_subplot(grid[:, 3])
        shared_legend_axes = {
            "right": right_legend_ax,
            0: right_legend_ax,
            1: right_legend_ax,
            2: right_legend_ax,
        }
        for legend_ax in shared_legend_axes.values():
            legend_ax.set_axis_off()
    else:
        fig, axes = plt.subplots(3, 3, figsize=figsize, squeeze=False)
    for panel_idx, (ax, (panel_key, (kind, col, panel_title, legend_title, order, palette, mask_task))) in enumerate(zip(axes.ravel(), panels)):
        row = panel_idx // 3
        col_idx = panel_idx % 3
        if col is None or col not in a.obs:
            ax.text(0.5, 0.5, "Missing metadata", ha="center", va="center", fontsize=12 * font_scale, fontweight="bold")
            ax.set_axis_off()
            continue
        mask = _umap_reference_mask(a, mask_task)
        if not np.any(mask):
            ax.text(0.5, 0.5, "No GT/reference-labeled cells", ha="center", va="center", fontsize=12 * font_scale, fontweight="bold")
            ax.set_axis_off()
            continue
        panel_coords = coords[mask]
        panel_vals = a.obs.loc[mask, col]
        if kind == "continuous":
            _plot_continuous_umap(
                ax,
                panel_coords,
                panel_vals,
                panel_title,
                point_size,
                font_scale=font_scale,
                panel_title_scale=panel_title_scale,
                axis_label_scale=axis_label_scale,
                colorbar_scale=colorbar_scale,
                colorbar_fraction=colorbar_fraction,
                colorbar_pad=colorbar_pad,
                colorbar_tick_side=colorbar_tick_side,
            )
        else:
            row_legend_group = shared_legend_group_by_row.get(row)
            use_metadata_external_legend = bool(
                row_shared_legends
                and shared_legend_position in {"right", "bottom"}
                and row == 0
                and panel_key in {"metadata_sample", "metadata_celltype"}
            )
            draw_panel_legend = row_legend_group is None and not use_metadata_external_legend
            if isinstance(legend_loc, dict):
                panel_legend_loc = legend_loc.get(panel_key, legend_loc.get("_default", "best"))
            else:
                panel_legend_loc = legend_loc
            panel_legend_outside = legend_outside.get(panel_key, False) if isinstance(legend_outside, dict) else bool(legend_outside)
            panel_category_renames = category_renames.get(panel_key, category_renames.get(col, {}))
            legend_info = _plot_categorical_umap(
                ax,
                panel_coords,
                panel_vals,
                panel_title,
                legend_title,
                order,
                palette,
                point_size,
                legend_loc=panel_legend_loc,
                category_renames=panel_category_renames,
                legend_outside=panel_legend_outside,
                font_scale=font_scale,
                draw_legend=draw_panel_legend,
                panel_title_scale=panel_title_scale,
                axis_label_scale=axis_label_scale,
                legend_scale=legend_scale,
            )
            if use_metadata_external_legend:
                metadata_legend_data.append(legend_info)
            elif row_legend_group is not None:
                row_data = shared_legend_data.setdefault(
                    row,
                    {
                        "group": row_legend_group,
                        "categories": [],
                        "palette": {},
                    },
                )
                for cat in legend_info["categories"]:
                    if cat not in row_data["categories"]:
                        row_data["categories"].append(cat)
                row_data["palette"].update(legend_info["palette"])
        _apply_umap_view(ax, common_xlim, common_ylim)
        if share_axis_labels or row < 2:
            ax.set_xlabel("")
        if share_axis_labels or col_idx > 0:
            ax.set_ylabel("")
        if show_umap_ticks:
            if share_axis_labels:
                ax.tick_params(labelbottom=row == 2, labelleft=col_idx == 0)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="both", length=0)
    if share_axis_labels:
        xlabel_y = 0.24 if row_shared_legends and shared_legend_position == "bottom" else 0.018
        fig.supxlabel("UMAP1", fontsize=8.6 * font_scale * axis_label_scale, fontweight="bold", y=xlabel_y)
        fig.supylabel("UMAP2", fontsize=8.6 * font_scale * axis_label_scale, fontweight="bold", x=0.018)
    if row_shared_legends and shared_legend_position == "bottom" and bottom_legend_ax is not None:
        bottom_legend_ax.set_xlim(0, 1)
        bottom_legend_ax.set_ylim(0, 1)

        def _ordered_shared_legend(row_idx):
            row_data = shared_legend_data.get(row_idx)
            if row_data is None:
                return None
            group_spec = shared_legend_groups[row_data["group"]]
            seen = set(row_data["categories"])
            categories = [cat for cat in group_spec["order"] if cat in seen]
            categories += [cat for cat in row_data["categories"] if cat not in set(categories)]
            return {
                "title": group_spec["title"],
                "categories": categories,
                "palette": row_data["palette"],
            }

        bottom_specs = []
        if len(metadata_legend_data) > 0:
            bottom_specs.append((metadata_legend_data[0], (0.00, 0.98), 4, 5.0, 5.6))
        if len(metadata_legend_data) > 1:
            bottom_specs.append((metadata_legend_data[1], (0.57, 0.98), 2, 5.0, 5.6))
        broad_legend = _ordered_shared_legend(1)
        if broad_legend is not None:
            bottom_specs.append((broad_legend, (0.00, 0.38), 5, 5.5, 6.1))
        tcell_legend = _ordered_shared_legend(2)
        if tcell_legend is not None:
            bottom_specs.append((tcell_legend, (0.57, 0.38), 5, 5.5, 6.1))

        for legend_info, anchor, ncol, label_size, title_size in bottom_specs:
            categories = legend_info["categories"]
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=(label_size + 0.2) * font_scale,
                    markerfacecolor=legend_info["palette"].get(cat, "#808080"),
                    markeredgewidth=0,
                    label=cat,
                )
                for cat in categories
            ]
            legend_obj = bottom_legend_ax.legend(
                handles=handles,
                labels=categories,
                title=legend_info["title"],
                loc="upper left",
                bbox_to_anchor=anchor,
                bbox_transform=bottom_legend_ax.transAxes,
                ncol=min(ncol, max(1, len(categories))),
                frameon=True,
                framealpha=0.92,
                fontsize=label_size * font_scale,
                title_fontsize=title_size * font_scale,
                borderpad=0.25,
                labelspacing=0.26,
                handletextpad=0.3,
                columnspacing=0.58,
            )
            bottom_legend_ax.add_artist(legend_obj)
    if row_shared_legends and shared_legend_position == "inside_last":
        for row_idx, row_data in shared_legend_data.items():
            if row_idx >= axes.shape[0]:
                continue
            target_panel_key = panels[row_idx * 3 + 2][0]
            if isinstance(legend_loc, dict):
                target_legend_loc = legend_loc.get(target_panel_key, legend_loc.get("_default", "best"))
            else:
                target_legend_loc = legend_loc
            group_spec = shared_legend_groups[row_data["group"]]
            seen = set(row_data["categories"])
            categories = [cat for cat in group_spec["order"] if cat in seen]
            categories += [cat for cat in row_data["categories"] if cat not in set(categories)]
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=7.6 * font_scale * legend_scale,
                    markerfacecolor=row_data["palette"].get(cat, "#808080"),
                    markeredgewidth=0,
                    label=cat,
                )
                for cat in categories
            ]
            axes[row_idx, 2].legend(
                handles=handles,
                labels=categories,
                title=group_spec["title"],
                loc=target_legend_loc,
                ncol=1,
                frameon=True,
                framealpha=0.92,
                fontsize=7.8 * font_scale * legend_scale,
                title_fontsize=8.4 * font_scale * legend_scale,
                borderpad=0.42 * legend_scale,
                labelspacing=0.36 * legend_scale,
                handletextpad=0.46 * legend_scale,
                columnspacing=0.78 * legend_scale,
            )
    if row_shared_legends and shared_legend_position == "right" and metadata_legend_data:
        metadata_legend_target = shared_legend_axes.get("right", shared_legend_axes.get(0))
        if metadata_legend_target is not None:
            for idx, legend_info in enumerate(metadata_legend_data[:2]):
                categories = legend_info["categories"]
                handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markersize=5.0 * font_scale * legend_scale,
                        markerfacecolor=legend_info["palette"].get(cat, "#808080"),
                        markeredgewidth=0,
                        label=cat,
                    )
                    for cat in categories
                ]
                loc = "upper left"
                bbox_y = 0.980 if idx == 0 else 0.680
                ncol = 1
                legend_obj = metadata_legend_target.legend(
                    handles=handles,
                    labels=categories,
                    title=legend_info["title"],
                    loc=loc,
                    bbox_to_anchor=(float(right_legend_anchor_x), bbox_y),
                    ncol=ncol,
                    frameon=True,
                    framealpha=0.92,
                    fontsize=6.8 * font_scale * legend_scale,
                    title_fontsize=7.3 * font_scale * legend_scale,
                    borderpad=0.22 * legend_scale,
                    labelspacing=0.24 * legend_scale,
                    handletextpad=0.30 * legend_scale,
                    columnspacing=0.50 * legend_scale,
                )
                metadata_legend_target.add_artist(legend_obj)
    if row_shared_legends and shared_legend_position == "right":
        shared_legend_iter = shared_legend_data.items()
    else:
        shared_legend_iter = []
    for row_idx, row_data in shared_legend_iter:
        group_spec = shared_legend_groups[row_data["group"]]
        seen = set(row_data["categories"])
        categories = [cat for cat in group_spec["order"] if cat in seen]
        categories += [cat for cat in row_data["categories"] if cat not in set(categories)]
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5.4 * font_scale * legend_scale,
                markerfacecolor=row_data["palette"].get(cat, "#808080"),
                markeredgewidth=0,
                label=cat,
            )
            for cat in categories
        ]
        legend_target = shared_legend_axes.get(row_idx)
        if shared_legend_position == "right" and "right" in shared_legend_axes:
            legend_target = shared_legend_axes["right"]
            legend_anchor_y = 0.405 if row_idx == 1 else 0.175
            legend_loc = "upper left"
        else:
            legend_anchor_y = 0.5
            legend_loc = "center left"
        legend_kwargs = {
            "handles": handles,
            "labels": categories,
            "title": group_spec["title"],
            "loc": legend_loc,
            "bbox_to_anchor": (float(right_legend_anchor_x), legend_anchor_y),
            "ncol": 1,
            "frameon": True,
            "framealpha": 0.92,
            "fontsize": 7.0 * font_scale * legend_scale,
            "title_fontsize": 7.6 * font_scale * legend_scale,
            "borderpad": 0.32 * legend_scale,
            "labelspacing": 0.36 * legend_scale,
            "handletextpad": 0.35 * legend_scale,
            "columnspacing": 0.60 * legend_scale,
        }
        if legend_target is not None:
            legend_obj = legend_target.legend(**legend_kwargs)
            legend_target.add_artist(legend_obj)
        else:
            fig.legend(
                **legend_kwargs,
                bbox_to_anchor=(0.5, 0.5),
            )
    has_suptitle = bool(str(title).strip())
    if has_suptitle:
        fig.suptitle(title, fontsize=13.5 * font_scale * suptitle_scale, fontweight="bold", y=0.975)
    # Even without a figure-wide title, panel headings extend above their axes.
    # Keep real top-canvas headroom so the first-row titles are never clipped.
    layout_top = 0.92 if has_suptitle else 0.95
    if legend_outside:
        fig.subplots_adjust(left=0.05, right=0.91, bottom=0.06, top=layout_top, wspace=0.42, hspace=0.42)
    elif row_shared_legends and shared_legend_position == "bottom":
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=layout_top)
    elif row_shared_legends and shared_legend_position == "inside_last":
        inside_top = 0.935 if has_suptitle else 0.985
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.055, top=inside_top, wspace=0.045, hspace=0.11)
    elif row_shared_legends:
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=layout_top)
    else:
        default_top = 0.925 if has_suptitle else 0.98
        fig.subplots_adjust(left=0.04, right=0.985, bottom=0.055, top=default_top, wspace=0.13, hspace=0.28)
    global FINAL_A4_OUTPUT_CANVAS
    old_force_a4 = FINAL_A4_OUTPUT_CANVAS
    if force_a4_canvas is not None:
        FINAL_A4_OUTPUT_CANVAS = bool(force_a4_canvas)
    try:
        _save_scrna_figure_png_pptx(fig, path, dpi=350)
    finally:
        FINAL_A4_OUTPUT_CANVAS = old_force_a4
    plt.close(fig)


def save_scrna_abcd_image_panel(
    panel_specs,
    path,
    title=None,
    figsize=(11.69, 8.27),
    dpi=350,
    nrows=2,
    ncols=2,
    width_ratios=None,
    height_ratios=None,
    wspace=0.025,
    hspace=0.05,
):
    """Combine saved scRNA Figures using the specified grid and aspect ratio."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gridspec_kw = {"wspace": wspace, "hspace": hspace}
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = width_ratios
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = height_ratios
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
    for ax, spec in zip(axes.ravel(), panel_specs):
        label = spec.get("label", "")
        image_path = Path(spec["path"])
        if not image_path.exists():
            raise FileNotFoundError(f"ABCD panel source Figure not found: {image_path}")
        img = plt.imread(str(image_path))
        ax.imshow(img, aspect="auto")
        ax.set_axis_off()
        ax.text(
            0.012,
            0.985,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8},
        )
    for ax in axes.ravel()[len(panel_specs):]:
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.988)
        top = 0.955
    else:
        top = 0.99
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.012, top=top, wspace=wspace, hspace=hspace)
    _save_scrna_figure_png_pptx(fig, path, dpi=dpi)
    plt.close(fig)


def save_tcell_subtype_evaluation_figures(a, out_dir, dataset_name, save_confusion=False):
    """Save T-cell subtype evaluation tables and performance comparison bar plots."""
    out_dir = ensure_direct_transfer_dirs(out_dir)
    metrics, cms, reports = evaluate_tcell_subtype_methods(a, dataset_name=dataset_name, out_dir=out_dir)
    save_method_metric_plot(
        metrics,
        out_dir / "figures" / "tcell_subtype_bacc_macro_f1.png",
        f"{dataset_name}: T-cell subtype",
    )
    if save_confusion:
        for method, cm in cms.items():
            safe_method = method.lower().replace(" ", "_")
            save_confusion_heatmap(
                cm,
                f"{dataset_name}: {method} T-cell subtype",
                out_dir / "figures" / f"tcell_subtype_{safe_method}_confusion.png",
            )
    return metrics, cms, reports


def save_umap_panels(a, columns, titles, path, point_size=6):
    """Save UMAP panels for references and direct-transfer predictions."""
    path = Path(path)
    if "X_umap" not in a.obsm:
        raise KeyError("adata.obsm['X_umap'] is missing.")
    coords = a.obsm["X_umap"]
    fig, axes = plt.subplots(1, len(columns), figsize=(8 * len(columns), 7), squeeze=False)
    for ax, col, title in zip(axes[0], columns, titles):
        vals = a.obs[col].astype("string").fillna("Missing")
        categories = [c for c in DISPLAY_BROAD_LABEL_ORDER + ["Low_confidence", "Missing"] if c in set(vals)]
        categories += [c for c in pd.Series(vals).drop_duplicates() if c not in categories]
        for cat in categories:
            idx = vals.eq(cat).to_numpy()
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=point_size,
                c=BROAD_COLOR_MAP.get(cat, "#808080"),
                label=cat,
                alpha=0.78,
                linewidths=0,
            )
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("UMAP1", fontsize=13, fontweight="bold")
        ax.set_ylabel("UMAP2", fontsize=13, fontweight="bold")
        ax.legend(markerscale=2.5, fontsize=10, title=col, title_fontsize=10, frameon=True, loc="best")
    fig.tight_layout(w_pad=3.0)
    _save_scrna_figure_png_pptx(fig, path, dpi=350, bbox_inches="tight")
    plt.close(fig)
