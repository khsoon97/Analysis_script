#!/usr/bin/env python3
"""Compare frozen marker-only and marker+ALE-direction/importance scRNA signatures.

This Final derived-output backend does not retrain or transfer a classifier object. It
compares two scores on identical fixed marker support within every class:

* Marker only: one positive equal vote per stable covered proxy.
* Marker + ALE direction × importance: class-normalized permutation-importance magnitude
  with the pinned patient-equal DLBCL q10-to-q90 first-order ALE direction,
  normalized by signed weight L1 mass after dataset proxy coverage.

Direction must be consistent in at least 0.8 of paired iterations. Paired
positive top-20 frequency and primary-versus-secondary agreement are exported
as sensitivity metadata only; neither selects a favorable sign. Broad and
T-cell subtype methods use identical reference-derived masks within task;
predictions outside the evaluable reference label space remain errors.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid
import warnings
import zipfile

from editable_pptx import _configure_font

import anndata as ad
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)


WORKFLOW_ROOT = Path(__file__).resolve().parent
FINAL_ROOT = WORKFLOW_ROOT.parent
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))
FINAL_SCRIPTS_ROOT = FINAL_ROOT / "scripts"
if str(FINAL_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(FINAL_SCRIPTS_ROOT))

RESOLVER_PATH = WORKFLOW_ROOT / "final_fixed_signature_inputs.py"
RESOLVER_CORE_PATH = WORKFLOW_ROOT / "final_fixed_signature_input_core.py"
CANONICAL_RUN_ID = "step02_step04b_dlbcl_ale_signature_scrna_seed42_20260813_01"
CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID = "sign_only_equal_magnitude_20260824_172147"
CANONICAL_THREE_ARM_FIGURE_MANIFEST = (
    "figure_render_manifest_marker_sign_importance_20260824.json"
)
CANONICAL_A4_COMPOSITE_MANIFEST = (
    "figure_render_manifest_marker_sign_importance_A4_portrait_2x2_20260825.json"
)
COMPLETED_BUNDLE_ROOT = (
    FINAL_ROOT / "tables" / "scRNA" / "fixed_signature" / CANONICAL_RUN_ID
)
RUNS_ROOT = FINAL_ROOT / "tables" / "scRNA" / "fixed_signature"
THREE_ARM_SENSITIVITY_ROOT = RUNS_ROOT / "sensitivity"
FIGURE_REEXPORT_ROOT = FINAL_ROOT / "figures" / "v2" / "scRNA" / "reexports"
A4_LAYOUT_REFERENCE_PPTX = FINAL_ROOT / "figures" / "Figure_\uc218\uc815.pptx"
A4_LAYOUT_REFERENCE_SLIDE_EMU = (6858000, 9906000)
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DEFAULT_SEED = 42
RUN_CONTRACT_VERSION = 2
SUPPORTED_RUN_CONTRACT_VERSIONS = frozenset({1, RUN_CONTRACT_VERSION})
CANONICAL_DERIVED_CONTRACT_VERSION = 1
PREPARED_INPUT_CONTRACT_VERSION = 1
ACTIVE_RUNTIME_PROVENANCE_CONTRACT_VERSION = 1
ANALYSIS_FAMILY = "frozen_fixed_signature"
EXPECTED_DATASETS = ("Roider", "GSE182434")
FIGURE_DATASET_DISPLAY_LABELS = {
    "Roider": "VRJUNV",
    "GSE182434": "GSE182434",
}
EXPECTED_TASK_TRUTH = {
    "broad": "reference_broad_codex",
    "tcell_subtype": "reference_tcell_subtype",
}
EXPECTED_ROUTING = "fixed_reference_target_mask"
EXPECTED_DIRECTION_METRIC = "patient-equal first-order ALE q10-to-q90 contrast"


def _canonical_derived_table_names() -> tuple[str, ...]:
    names = {
        "step02_broad_directional_signature_features.csv",
        "step04b_tcell_directional_signature_features.csv",
        "combined_fixed_class_direction_signature_features.csv",
        "pooled_cell_metrics.csv",
        "sample_patient_metrics.csv",
    }
    for dataset in EXPECTED_DATASETS:
        names.update(
            {
                f"{dataset}_signature_proxy_weights.csv",
                f"{dataset}_signature_proxy_coverage_audit.csv",
                f"{dataset}_cell_predictions.csv.gz",
                f"{dataset}_class_scores.csv.gz",
                f"{dataset}_evaluation_mask_audit.csv.gz",
            }
        )
        for task in EXPECTED_TASK_TRUTH:
            for method_id in ("equal_weight", "direction_aware"):
                prefix = f"{dataset}_{task}_{method_id}"
                names.add(f"{prefix}_confusion_matrix.csv")
                names.add(f"{prefix}_classification_report.csv")
    return tuple(sorted(names))


def _existing_final_utils_write_scope(paths: list[Path], *, owner: str):
    """Use an already-installed Final guard without importing it into scoring."""

    final_utils = sys.modules.get("Final_utils")
    if final_utils is None:
        return nullcontext()
    scope_factory = getattr(final_utils, "allow_exact_final_derived_writes", None)
    if scope_factory is None:
        raise RuntimeError(
            "Loaded Final_utils lacks exact derived-write scope support; restart the kernel"
        )
    return scope_factory(paths, owner=owner)


def _write_staged_derived_csv(
    frame,
    path: Path,
    *,
    tables_root: Path,
    **kwargs,
) -> None:
    path = Path(path)
    if path.parent.resolve() != tables_root.resolve():
        raise ValueError(f"Derived table escapes its staging table root: {path}")
    if path.name not in _canonical_derived_table_names():
        raise ValueError(f"Unregistered canonical derived table: {path.name}")
    with _existing_final_utils_write_scope([path], owner="Step11 fixed-signature derived CSV"):
        frame.to_csv(path, **kwargs)


def _write_exact_derived_json(path: Path, payload: dict[str, object], *, owner: str) -> None:
    path = Path(path)
    with _existing_final_utils_write_scope([path], owner=owner):
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

# Supplementary Figure 5 panel order is fixed to performance then UMAP within
# each dataset.  These stems replace the former direct-transfer Step12 files.
FINAL_SCRNA_FIGURE_STEMS = {
    ("GSE182434", "performance"): (
        "Final_mv1_Step12-01_GSE182434_"
        "marker_sign_importance_ablation_performance"
    ),
    ("GSE182434", "3x3_umap_summary"): (
        "Final_mv1_Step12-02_GSE182434_"
        "marker_only_vs_dlbcl_ale_direction_weighted_3x3_umap_summary"
    ),
    ("Roider", "performance"): (
        "Final_mv1_Step12-03_Roider_"
        "marker_sign_importance_ablation_performance"
    ),
    ("Roider", "3x3_umap_summary"): (
        "Final_mv1_Step12-04_Roider_"
        "marker_only_vs_dlbcl_ale_direction_weighted_3x3_umap_summary"
    ),
}

FINAL_SCRNA_A4_COMPOSITE_STEM = (
    "Final_mv1_Step12-05_scRNA_marker_sign_importance_A4_portrait_2x2_summary"
)

# Kept only for isolated historical two-arm rerenders. Final Step12 performance
# is supplied by the pinned Step11B three-arm sensitivity bundle above.
LEGACY_TWO_ARM_PERFORMANCE_STEMS = {
    "GSE182434": (
        "Final_mv1_Step12-01_GSE182434_"
        "marker_only_vs_dlbcl_ale_direction_weighted_performance"
    ),
    "Roider": (
        "Final_mv1_Step12-03_Roider_"
        "marker_only_vs_dlbcl_ale_direction_weighted_performance"
    ),
}

THREE_ARM_METHOD_SPECS = {
    "marker_only": {"display": "Marker only"},
    "collapsed_sign_only": {"display": "ALE sign only"},
    "sign_plus_importance": {"display": "ALE sign × PFI weight"},
}

# Figure-only aliases keep the protected table and calculation labels intact
# while presenting the three ablation arms in plain language.
FIGURE_THREE_ARM_LABELS = {
    "marker_only": "Marker only",
    "collapsed_sign_only": "Direction only",
    "sign_plus_importance": "Direction + importance",
}
FIGURE_DIRECTION_IMPORTANCE_LABEL = FIGURE_THREE_ARM_LABELS[
    "sign_plus_importance"
]
INDIVIDUAL_PERFORMANCE_CANVAS_INCHES = (3.2, 6.9)
INDIVIDUAL_UMAP_CANVAS_INCHES = (7.5, 6.9)
SCRNA_SCORE_BAR_WIDTH = 0.30
SCRNA_SCORE_BAR_CENTER_OFFSETS = (-0.22, 0.22)
A4_SCORE_BAR_WIDTH = 0.30
A4_SCORE_BAR_CENTER_OFFSETS = (-0.27, 0.27)
A4_SCORE_GRID_HEIGHT_RATIOS = (0.24, 0.13, 1.0, 0.34, 1.0)
A4_SCORE_HEADER_TITLE_Y = 0.82
A4_SCORE_METRIC_LEGEND_ANCHOR_Y = 0.00
SCRNA_UMAP_COLORBAR_COLUMN_RATIO = 0.055
A4_SCORE_UMAP_WIDTH_RATIOS = (0.9, 2.1)
A4_OUTER_WSPACE = 0.075
A4_OUTER_HSPACE = 0.10
A4_UMAP_GRID_WSPACE = 0.065
A4_UMAP_GRID_HSPACE = 0.25
A4_UMAP_LEGEND_COLUMN_RATIO = 0.92
A4_UMAP_LEGEND_MARKER_SIZE = 6.8
A4_UMAP_LEGEND_LABEL_FONTSIZE = 8.7
A4_UMAP_LEGEND_TITLE_FONTSIZE = 10.0
A4_UMAP_LEGEND_GROUP_HSPACE = 0.16
A4_UMAP_LEGEND_LABELSPACING = 0.10
A4_UMAP_LEGEND_HANDLETEXTPAD = 0.30
A4_UMAP_LEGEND_WRAP_WIDTH = 13
A4_UMAP_LEGEND_LEFT_INSET = 0.10
A4_TYPOGRAPHY_POINTS = {
    "panel_header": 12.0,
    "score_metric_legend": 10.0,
    "score_value_annotation": 8.2,
    "score_x_tick": 9.8,
    "score_y_tick": 10.5,
    "score_y_label": 12.0,
    "score_task_title": 11.5,
    "umap_subplot_title": 10.0,
    "umap_colorbar_tick": 9.0,
    "umap_legend_label": A4_UMAP_LEGEND_LABEL_FONTSIZE,
    "umap_legend_title": A4_UMAP_LEGEND_TITLE_FONTSIZE,
}

DEFAULT_STEP02_RUN_ID = (
    "step02_broad_xgb_mlp_fast_chunk_paired_directional_attribution_seed42_20260810_134034"
)
DEFAULT_STEP04B_RUN_ID = (
    "step04b_joint_xgb_mlp_fast_chunk_paired_directional_attribution_seed42_20260810_224723"
)

DATASETS = {
    "Roider": {
        "h5ad": FINAL_ROOT
        / "checkpoints"
        / "scRNA"
        / "Roider"
        / "01_roider_codex_proxy_features.h5ad",
        "group_col": "Sample",
    },
    "GSE182434": {
        "h5ad": FINAL_ROOT
        / "checkpoints"
        / "scRNA"
        / "GSE182434"
        / "01_gse182434_codex_proxy_features.h5ad",
        "group_col": "Patient",
    },
}

# Immutable digest pins transcribed from the protected completed fixed-signature
# run manifest.  Default Step09/10 reuse validates both this code-owned registry
# and that authoritative historical record before trusting the canonical proxy.
CANONICAL_PROXY_INPUT_RECORDS = {
    "Roider": {
        "path_relative_to_final": (
            "checkpoints/scRNA/Roider/01_roider_codex_proxy_features.h5ad"
        ),
        "size_bytes": 1_496_140_275,
        "sha256": "5d454ddd0bfb558050140a7fd853dc36d3b89141558b849acae5fc532f003bab",
        "manifest_key": "Roider_proxy_h5ad",
    },
    "GSE182434": {
        "path_relative_to_final": (
            "checkpoints/scRNA/GSE182434/01_gse182434_codex_proxy_features.h5ad"
        ),
        "size_bytes": 666_190_417,
        "sha256": "04c847593a21de1f107fd2e4d1d7f0bff290340428da1cc34d8eddf697085af2",
        "manifest_key": "GSE182434_proxy_h5ad",
    },
}

TASK_CLASS_ORDER = {
    "broad": ["B cells", "Myeloid cells", "NK cells", "Stromal cells", "T cells"],
    "tcell_subtype": ["TH", "TFH", "TREG", "CD8T", "CD4_CTL", "TPR"],
}

# Stable internal IDs keep the retained completed run readable.  The explicit
# analysis labels are the manuscript-facing comparison requested for new runs.
METHOD_SPECS = {
    "equal_weight": {
        "analysis_label": "marker_only",
        "display": "Marker only",
        "weight_col": "equal_weight_proxy_weight",
    },
    "direction_aware": {
        "analysis_label": "marker_ale_direction_weighted",
        "display": "Marker + ALE direction × importance",
        "weight_col": "direction_aware_proxy_weight",
    },
}

FIGURE_SCORING_TITLE = "CODEX ML-derived marker-signature scoring"
FIGURE_SCORING_SUBTITLE = (
    "Frozen signatures; no scRNA fitting or paired same-cell comparison"
)

CLASS_DISPLAY = {
    **{label: label for label in TASK_CLASS_ORDER["broad"]},
    **{label: label for label in TASK_CLASS_ORDER["tcell_subtype"]},
}

BROAD_PALETTE = {
    "B cells": "#4E79A7",
    "Myeloid cells": "#59A14F",
    "NK cells": "#F28E2B",
    "Stromal cells": "#B07AA1",
    "T cells": "#E15759",
    "Unassigned": "#6F6F6F",
}
T_PALETTE = {
    "TH": "#4E79A7",
    "TFH": "#59A14F",
    "TREG": "#F28E2B",
    "CD8T": "#E15759",
    "CD4_CTL": "#7E57C2",
    "TPR": "#EDC948",
    "Unassigned": "#6F6F6F",
}


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    truth_col: str
    reference_labels: tuple[str, ...]
    model_labels: tuple[str, ...]
    truth: pd.Series
    description: str


def _load_resolver():
    spec = importlib.util.spec_from_file_location("step04b_weight_resolver", RESOLVER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Step04B resolver: {RESOLVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_immutable_file_digest(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
    verify_hash: bool = True,
) -> dict[str, object]:
    """Fail closed unless one immutable input matches its pinned byte contract."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Immutable {label} is missing: {path}")
    observed_size = int(path.stat().st_size)
    if observed_size != int(expected_size):
        raise RuntimeError(
            f"Immutable {label} size mismatch: {observed_size} != {expected_size}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)) is None:
        raise RuntimeError(f"Immutable {label} has an invalid pinned SHA-256")
    observed_sha256 = _sha256(path) if verify_hash else None
    if verify_hash and observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"Immutable {label} SHA-256 mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )
    return {
        "path": str(path),
        "size_bytes": observed_size,
        "expected_sha256": str(expected_sha256),
        "observed_sha256": observed_sha256,
        "sha256_verified": bool(verify_hash),
    }


def _normalise_hash_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _semantic_rows_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = [_normalise_hash_value(value) for value in row]
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _sequence_sha256(values) -> str:
    return _semantic_rows_sha256((position, value) for position, value in enumerate(values))


def _truth_semantic_sha256(obs: pd.DataFrame) -> str:
    required = ["reference_broad_codex", "reference_tcell_subtype"]
    missing = [column for column in required if column not in obs.columns]
    if missing:
        raise KeyError(f"Prepared scRNA input lacks truth columns: {missing}")
    return _semantic_rows_sha256(
        (
            position,
            cell_id,
            broad,
            subtype,
        )
        for position, (cell_id, broad, subtype) in enumerate(
            zip(
                obs.index.astype(str),
                obs["reference_broad_codex"],
                obs["reference_tcell_subtype"],
            )
        )
    )


def _coverage_semantic_sha256(coverage: pd.DataFrame) -> str:
    columns = [
        "codex_marker",
        "feature_col",
        "requested_genes",
        "present_genes",
        "covered",
        "n_present_genes",
    ]
    missing = [column for column in columns if column not in coverage.columns]
    if missing:
        raise KeyError(f"Prepared proxy coverage lacks columns: {missing}")
    return _semantic_rows_sha256(
        (position, *(row[column] for column in columns))
        for position, (_, row) in enumerate(coverage.loc[:, columns].iterrows())
    )


def _lineage_gate(task_id: str) -> str:
    if task_id == "tcell_subtype":
        return "reference_broad_codex == 'T cells'"
    if task_id == "broad":
        return "reference broad truth in evaluable label space"
    raise KeyError(f"Unknown fixed-signature task: {task_id}")


def _evaluation_semantic_sha256(
    dataset: str,
    task_id: str,
    cell_ids: pd.Index,
    truth: pd.Series,
    included: pd.Series,
) -> str:
    gate = _lineage_gate(task_id)
    return _semantic_rows_sha256(
        (
            dataset,
            task_id,
            position,
            cell_id,
            truth_value,
            bool(is_included),
            gate,
        )
        for position, (cell_id, truth_value, is_included) in enumerate(
            zip(cell_ids.astype(str), truth, included)
        )
    )


def _relative_to_final(path: Path) -> str:
    try:
        return path.resolve().relative_to(FINAL_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path escapes the Final workspace: {path}") from exc


def _resolve_final_relative(relative_path: str) -> Path:
    candidate = (FINAL_ROOT / str(relative_path)).resolve()
    try:
        candidate.relative_to(FINAL_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes the Final workspace: {relative_path}") from exc
    return candidate


def _require_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} is outside its isolated run root: {resolved}") from exc
    return resolved


def require_new_run_id(run_id: str | None) -> str:
    if (
        not run_id
        or RUN_NAME.fullmatch(run_id) is None
        or run_id == CANONICAL_RUN_ID
    ):
        raise ValueError("A safe noncanonical fixed-signature run ID is required")
    return run_id


def validate_canonical_proxy_inputs(
    datasets: tuple[str, ...] | list[str] = EXPECTED_DATASETS,
    *,
    verify_hashes: bool = True,
) -> dict[str, dict[str, object]]:
    """Verify canonical proxies against an immutable manifest plus digest pins."""

    requested = tuple(datasets)
    if len(set(requested)) != len(requested) or not set(requested).issubset(
        EXPECTED_DATASETS
    ):
        raise ValueError(f"Unexpected canonical proxy dataset selection: {requested}")
    authoritative_manifest_path = (
        COMPLETED_BUNDLE_ROOT / "manifests" / "run_manifest.json"
    )
    if not authoritative_manifest_path.is_file():
        raise FileNotFoundError(
            f"Authoritative canonical proxy digest record is missing: "
            f"{authoritative_manifest_path}"
        )
    authoritative_manifest = json.loads(
        authoritative_manifest_path.read_text(encoding="utf-8")
    )
    if (
        authoritative_manifest.get("status") != "complete"
        or authoritative_manifest.get("run_id") != CANONICAL_RUN_ID
    ):
        raise RuntimeError("Authoritative canonical proxy digest record is invalid")
    large_inputs = authoritative_manifest.get("large_inputs", {})
    reports: dict[str, dict[str, object]] = {}
    for dataset in requested:
        expected = CANONICAL_PROXY_INPUT_RECORDS[dataset]
        record = large_inputs.get(expected["manifest_key"], {})
        if (
            record.get("sha256") != expected["sha256"]
            or int(record.get("size_bytes_before", -1)) != expected["size_bytes"]
            or int(record.get("size_bytes_after", -1)) != expected["size_bytes"]
            or record.get("input_unchanged") is not True
            or record.get("sha256_skipped") is not False
        ):
            raise RuntimeError(
                f"{dataset} authoritative proxy digest record differs from the immutable pin"
            )
        canonical_path = _resolve_final_relative(expected["path_relative_to_final"])
        configured_path = Path(DATASETS[dataset]["h5ad"]).resolve()
        if configured_path != canonical_path:
            raise RuntimeError(
                f"{dataset} default proxy path is not canonical: "
                f"{configured_path} != {canonical_path}"
            )
        recorded_path = Path(str(record.get("path", ""))).resolve()
        if recorded_path != canonical_path:
            raise RuntimeError(
                f"{dataset} authoritative proxy path differs from the canonical path"
            )
        report = _validate_immutable_file_digest(
            canonical_path,
            expected_size=int(expected["size_bytes"]),
            expected_sha256=str(expected["sha256"]),
            label=f"{dataset} canonical proxy H5AD",
            verify_hash=verify_hashes,
        )
        report.update(
            {
                "authoritative_manifest": str(authoritative_manifest_path),
                "authoritative_manifest_key": str(expected["manifest_key"]),
                "authoritative_record_matches_pin": True,
            }
        )
        reports[dataset] = report
    return reports


def _build_active_runtime_provenance(
    *,
    dependencies: dict[str, Path],
    resolved_font_family: str | None,
    arial_regular: Path | None = None,
    arial_bold: Path | None = None,
) -> dict[str, object]:
    """Describe only current executable dependencies, never historical paths."""

    relative_dependencies = {
        name: _relative_to_final(Path(path)) for name, path in dependencies.items()
    }
    contract = {
        "contract_version": ACTIVE_RUNTIME_PROVENANCE_CONTRACT_VERSION,
        "active_dependency_paths_relative_to_final": relative_dependencies,
        "review_outputs_executable_dependency": False,
        "historical_source_path_strings_role": "immutable provenance metadata only",
        "figure_font_contract": "strict Arial Regular/Bold; no fallback family",
        "resolved_font_family": resolved_font_family,
        "resolved_font_files": {
            "regular": str(Path(arial_regular).resolve()) if arial_regular else None,
            "bold": str(Path(arial_bold).resolve()) if arial_bold else None,
        },
    }
    _validate_active_runtime_provenance(contract)
    return contract


def _validate_active_runtime_provenance(contract: dict[str, object]) -> None:
    """Reject new active provenance that reintroduces retired paths or fonts."""

    if contract.get("contract_version") != ACTIVE_RUNTIME_PROVENANCE_CONTRACT_VERSION:
        raise RuntimeError("Active runtime provenance contract version mismatch")
    if contract.get("review_outputs_executable_dependency") is not False:
        raise RuntimeError("Active runtime provenance claims a review_outputs dependency")
    if contract.get("figure_font_contract") != (
        "strict Arial Regular/Bold; no fallback family"
    ):
        raise RuntimeError("Active runtime provenance does not require strict Arial")
    family = contract.get("resolved_font_family")
    if family not in (None, "Arial"):
        raise RuntimeError(f"Active runtime provenance resolved a non-Arial font: {family!r}")
    serialized = json.dumps(contract, ensure_ascii=False).casefold()
    if "helvetica" in serialized:
        raise RuntimeError("New active provenance must not claim Helvetica or its fallback")
    dependencies = contract.get("active_dependency_paths_relative_to_final", {})
    if not isinstance(dependencies, dict) or not dependencies:
        raise RuntimeError("Active runtime provenance dependency set is missing")
    for name, relative in dependencies.items():
        relative_text = str(relative).replace("\\", "/")
        if "review_outputs" in Path(relative_text).parts:
            raise RuntimeError(
                f"Active runtime dependency {name!r} resolves through review_outputs"
            )
        resolved = _resolve_final_relative(relative_text)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Active runtime provenance dependency is missing: {name}={resolved}"
            )


def _validate_generated_manifest_provenance(manifest: dict[str, object]) -> None:
    """Validate provenance written by current Step11/12 code only."""

    contract = manifest.get("active_runtime_provenance")
    if not isinstance(contract, dict):
        raise RuntimeError("Current generated manifest lacks active runtime provenance")
    _validate_active_runtime_provenance(contract)
    if "helvetica" in json.dumps(manifest, ensure_ascii=False).casefold():
        raise RuntimeError("Current generated manifest contains a Helvetica claim")
    for name, record in manifest.get("source_hashes", {}).items():
        if not isinstance(record, dict) or "path" not in record:
            raise RuntimeError(f"Generated source provenance record is invalid: {name}")
        source_path = Path(str(record["path"])).resolve()
        try:
            relative = source_path.relative_to(FINAL_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Generated active source provenance escapes Final: {source_path}"
            ) from exc
        if "review_outputs" in relative.parts:
            raise RuntimeError(
                f"Generated active source provenance uses review_outputs: {source_path}"
            )


def write_prepared_input_manifest(
    manifest_path: Path,
    *,
    stage: str,
    run_id: str,
    dataset: str,
    feature_h5ad: Path,
    coverage_csv: Path,
    signature_table: Path,
    feature_names: list[str],
    adata,
) -> dict[str, object]:
    """Write one immutable Step09/10 handoff after verifying isolated paths."""

    run_id = require_new_run_id(run_id)
    if dataset not in EXPECTED_DATASETS:
        raise ValueError(f"Unsupported prepared scRNA dataset: {dataset}")
    expected_stage = {"Roider": "Step09", "GSE182434": "Step10"}[dataset]
    if stage != expected_stage:
        raise ValueError(f"Prepared input stage mismatch: {stage} != {expected_stage}")
    checkpoint_root = (
        FINAL_ROOT / "checkpoints" / "scRNA" / "recalculation_inputs" / run_id / dataset
    )
    table_root = (
        FINAL_ROOT
        / "tables"
        / "scRNA"
        / "fixed_signature_recalculation_inputs"
        / run_id
        / dataset
    )
    feature_h5ad = _require_within(feature_h5ad, checkpoint_root, label="feature_h5ad")
    coverage_csv = _require_within(coverage_csv, table_root, label="coverage_csv")
    manifest_path = _require_within(manifest_path, table_root, label="handoff manifest")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared-input manifest: {manifest_path}")
    for path in (feature_h5ad, coverage_csv, signature_table):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Prepared-input artifact is missing or empty: {path}")
    coverage = pd.read_csv(coverage_csv)
    coverage_markers = coverage["codex_marker"].astype(str).tolist()
    if coverage_markers != list(map(str, feature_names)):
        raise RuntimeError(f"{dataset} coverage marker order differs from the prepared signature")
    if int(adata.n_obs) != len(adata.obs.index):
        raise RuntimeError(f"{dataset} AnnData observation count is inconsistent")
    umap_shape = list(np.asarray(adata.obsm["X_umap"]).shape)
    if umap_shape != [int(adata.n_obs), 2]:
        raise RuntimeError(f"{dataset} prepared UMAP shape is invalid: {umap_shape}")
    manifest = {
        "contract_version": PREPARED_INPUT_CONTRACT_VERSION,
        "analysis_family": ANALYSIS_FAMILY,
        "status": "complete",
        "stage": stage,
        "run_id": run_id,
        "dataset": dataset,
        "feature_h5ad": {
            "path_relative_to_final": _relative_to_final(feature_h5ad),
            "size_bytes": int(feature_h5ad.stat().st_size),
            "sha256": _sha256(feature_h5ad),
        },
        "coverage_csv": {
            "path_relative_to_final": _relative_to_final(coverage_csv),
            "size_bytes": int(coverage_csv.stat().st_size),
            "sha256": _sha256(coverage_csv),
            "semantic_sha256": _coverage_semantic_sha256(coverage),
            "n_rows": int(len(coverage)),
        },
        "signature_source_run_id": CANONICAL_RUN_ID,
        "signature_table": {
            "path_relative_to_final": _relative_to_final(signature_table),
            "size_bytes": int(signature_table.stat().st_size),
            "sha256": _sha256(signature_table),
        },
        "signature_feature_order_sha256": _sequence_sha256(feature_names),
        "n_obs": int(adata.n_obs),
        "umap_shape": umap_shape,
        "ordered_cell_id_sha256": _sequence_sha256(adata.obs.index.astype(str)),
        "truth_semantic_sha256": _truth_semantic_sha256(adata.obs),
        "canonical_overwritten": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_prepared_input_manifests(
    run_id: str,
    manifest_paths: dict[str, Path],
    *,
    verify_hashes: bool = True,
) -> dict[str, dict[str, object]]:
    """Validate the paired Step09/10 manifests and return their pinned paths."""

    run_id = require_new_run_id(run_id)
    if set(manifest_paths) != set(EXPECTED_DATASETS):
        raise RuntimeError("Step11 requires exactly one Roider and one GSE182434 handoff")
    reports: dict[str, dict[str, object]] = {}
    for dataset in EXPECTED_DATASETS:
        table_root = (
            FINAL_ROOT
            / "tables"
            / "scRNA"
            / "fixed_signature_recalculation_inputs"
            / run_id
            / dataset
        )
        checkpoint_root = (
            FINAL_ROOT
            / "checkpoints"
            / "scRNA"
            / "recalculation_inputs"
            / run_id
            / dataset
        )
        manifest_path = _require_within(
            Path(manifest_paths[dataset]), table_root, label=f"{dataset} handoff manifest"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prepared-input manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "contract_version": PREPARED_INPUT_CONTRACT_VERSION,
            "analysis_family": ANALYSIS_FAMILY,
            "status": "complete",
            "stage": {"Roider": "Step09", "GSE182434": "Step10"}[dataset],
            "run_id": run_id,
            "dataset": dataset,
            "signature_source_run_id": CANONICAL_RUN_ID,
            "canonical_overwritten": False,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"{dataset} prepared-input manifest mismatch: {key}")
        feature_record = manifest.get("feature_h5ad", {})
        coverage_record = manifest.get("coverage_csv", {})
        signature_record = manifest.get("signature_table", {})
        feature_h5ad = _require_within(
            _resolve_final_relative(feature_record.get("path_relative_to_final", "")),
            checkpoint_root,
            label=f"{dataset} feature_h5ad",
        )
        coverage_csv = _require_within(
            _resolve_final_relative(coverage_record.get("path_relative_to_final", "")),
            table_root,
            label=f"{dataset} coverage_csv",
        )
        signature_table = _resolve_final_relative(
            signature_record.get("path_relative_to_final", "")
        )
        for label, path, record in (
            ("feature_h5ad", feature_h5ad, feature_record),
            ("coverage_csv", coverage_csv, coverage_record),
            ("signature_table", signature_table, signature_record),
        ):
            if not path.is_file() or int(path.stat().st_size) != int(record.get("size_bytes", -1)):
                raise RuntimeError(f"{dataset} prepared-input size mismatch: {label}")
            if verify_hashes and _sha256(path) != record.get("sha256"):
                raise RuntimeError(f"{dataset} prepared-input hash mismatch: {label}")
        coverage = pd.read_csv(coverage_csv)
        if int(len(coverage)) != int(coverage_record.get("n_rows", -1)):
            raise RuntimeError(f"{dataset} prepared coverage row count mismatch")
        if _coverage_semantic_sha256(coverage) != coverage_record.get("semantic_sha256"):
            raise RuntimeError(f"{dataset} prepared coverage semantic hash mismatch")
        coverage_order_hash = _sequence_sha256(coverage["codex_marker"].astype(str))
        if coverage_order_hash != manifest.get("signature_feature_order_sha256"):
            raise RuntimeError(f"{dataset} prepared signature feature order mismatch")
        reports[dataset] = {
            "manifest_path": manifest_path,
            "manifest_sha256": _sha256(manifest_path),
            "feature_h5ad": feature_h5ad,
            "feature_h5ad_sha256": feature_record.get("sha256"),
            "coverage_csv": coverage_csv,
            "coverage_csv_sha256": coverage_record.get("sha256"),
            "signature_feature_order_sha256": coverage_order_hash,
            "n_obs": int(manifest.get("n_obs", -1)),
            "umap_shape": manifest.get("umap_shape"),
            "ordered_cell_id_sha256": manifest.get("ordered_cell_id_sha256"),
            "truth_semantic_sha256": manifest.get("truth_semantic_sha256"),
        }
    return reports


def _safe_run_root(run_id: str) -> tuple[Path, Path]:
    if RUN_NAME.fullmatch(run_id) is None:
        raise ValueError(f"Unsafe scRNA validation run ID: {run_id!r}")
    final_root = (RUNS_ROOT / run_id).resolve()
    partial_root = (RUNS_ROOT / f"{run_id}.partial").resolve()
    expected_parent = RUNS_ROOT.resolve()
    if final_root.parent != expected_parent or partial_root.parent != expected_parent:
        raise ValueError("scRNA validation output escapes the review run root")
    if final_root.exists() or partial_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing validation run: {final_root}"
        )
    return final_root, partial_root


def _canonical_derived_staging_root() -> tuple[Path, Path]:
    """Return the canonical run and a unique same-filesystem staging directory."""

    final_root = COMPLETED_BUNDLE_ROOT.resolve()
    if final_root.parent != RUNS_ROOT.resolve():
        raise RuntimeError("Canonical fixed-signature root escapes RUNS_ROOT")
    token = uuid.uuid4().hex
    partial_root = RUNS_ROOT / f".{CANONICAL_RUN_ID}.{token}.partial"
    if partial_root.exists():
        raise FileExistsError(f"Canonical derived staging root already exists: {partial_root}")
    return final_root, partial_root


def _commit_canonical_derived_run(partial_root: Path, final_root: Path) -> None:
    """Promote a validated derived bundle and roll back ordinary failures.

    The protected proxy H5AD and frozen Step02/04B inputs are outside
    ``final_root``.  Only the canonical derived table/manifest bundle is
    replaced here.
    """

    if partial_root.parent.resolve() != RUNS_ROOT.resolve():
        raise ValueError(f"Unexpected canonical staging parent: {partial_root}")
    if final_root.resolve() != COMPLETED_BUNDLE_ROOT.resolve():
        raise ValueError(f"Unexpected canonical target: {final_root}")
    if not partial_root.is_dir():
        raise FileNotFoundError(partial_root)
    backup_root = RUNS_ROOT / f".{CANONICAL_RUN_ID}.{uuid.uuid4().hex}.backup"
    transaction_path = RUNS_ROOT / f".{CANONICAL_RUN_ID}.transaction.json"
    if transaction_path.exists() or backup_root.exists():
        raise RuntimeError("A canonical derived transaction is already pending")
    transaction = {
        "status": "prepared",
        "canonical_run_id": CANONICAL_RUN_ID,
        "target": str(final_root),
        "staging": str(partial_root),
        "backup": str(backup_root),
    }
    _write_exact_derived_json(
        transaction_path,
        transaction,
        owner="Step11 canonical derived transaction",
    )
    try:
        if final_root.exists():
            os.replace(final_root, backup_root)
        transaction["status"] = "previous_moved_to_backup"
        _write_exact_derived_json(
            transaction_path,
            transaction,
            owner="Step11 canonical derived transaction",
        )
        os.replace(partial_root, final_root)
        transaction["status"] = "candidate_promoted"
        _write_exact_derived_json(
            transaction_path,
            transaction,
            owner="Step11 canonical derived transaction",
        )
        validate_fixed_signature_run(CANONICAL_RUN_ID, verify_artifacts=True)
    except BaseException:
        if final_root.exists():
            shutil.rmtree(final_root)
        if backup_root.exists():
            os.replace(backup_root, final_root)
        transaction_path.unlink(missing_ok=True)
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)
    transaction_path.unlink(missing_ok=True)


def _strict_bool_series(series: pd.Series, *, label: str) -> pd.Series:
    mapped = series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else {"true": True, "false": False}.get(str(value).strip().lower())
    )
    if mapped.isna().any():
        raise RuntimeError(f"{label} contains non-boolean values")
    return mapped.astype(bool)


def _table_artifacts(root: Path) -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted((root / "tables").rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            artifacts[relative] = {
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
    return artifacts


def _validate_fixed_signature_table_semantics(
    run_root: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    tables_root = run_root / "tables"
    pooled_path = tables_root / "pooled_cell_metrics.csv"
    grouped_path = tables_root / "sample_patient_metrics.csv"
    if not pooled_path.is_file() or not grouped_path.is_file():
        raise FileNotFoundError("Fixed-signature metric tables are incomplete")
    pooled = pd.read_csv(pooled_path)
    grouped = pd.read_csv(grouped_path)
    required_metric_columns = {
        "dataset",
        "task",
        "method",
        "scope",
        "truth_col",
        "routing",
        "evaluation_mask_sha256",
        "n_total_dataset_cells",
        "n_reference_excluded",
        "n_cells",
        "n_pred_outside_reference_label_space",
        "outside_predictions_dropped",
    }
    missing_metric_columns = sorted(required_metric_columns - set(pooled.columns))
    if missing_metric_columns:
        raise RuntimeError(f"Pooled metrics lack contract columns: {missing_metric_columns}")
    methods = tuple(METHOD_SPECS)
    expected_rows = {
        (dataset, task, method)
        for dataset in EXPECTED_DATASETS
        for task in EXPECTED_TASK_TRUTH
        for method in methods
    }
    observed_rows = set(
        pooled.loc[:, ["dataset", "task", "method"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if observed_rows != expected_rows or len(pooled) != len(expected_rows):
        raise RuntimeError("Fixed-signature pooled metric dataset/task/method set is not exact")
    if set(pooled["scope"].astype(str)) != {"pooled_cells"}:
        raise RuntimeError("Fixed-signature pooled metric scope is not pooled_cells")
    if set(pooled["routing"].astype(str)) != {EXPECTED_ROUTING}:
        raise RuntimeError("Fixed-signature routing is not the shared reference mask")
    if _strict_bool_series(
        pooled["outside_predictions_dropped"], label="outside_predictions_dropped"
    ).any():
        raise RuntimeError("Outside-label predictions must remain denominator errors")
    required_grouped_columns = {
        "dataset",
        "task",
        "method",
        "evaluation_mask_sha256",
        "outside_predictions_dropped",
    }
    missing_grouped_columns = sorted(required_grouped_columns - set(grouped.columns))
    if missing_grouped_columns:
        raise RuntimeError(
            f"Sample/patient metrics lack contract columns: {missing_grouped_columns}"
        )
    grouped_combinations = set(
        grouped.loc[:, ["dataset", "task", "method"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if grouped_combinations != expected_rows:
        raise RuntimeError("Sample/patient metric dataset/task/method set is not exact")
    if _strict_bool_series(
        grouped["outside_predictions_dropped"],
        label="grouped outside_predictions_dropped",
    ).any():
        raise RuntimeError("Grouped metrics drop outside-label predictions")
    manifest_datasets = manifest.get("datasets", {})
    if set(manifest_datasets) != set(EXPECTED_DATASETS):
        raise RuntimeError("Run manifest dataset set is incomplete or unexpected")

    semantic_hashes: dict[str, dict[str, str]] = {}
    for dataset in EXPECTED_DATASETS:
        dataset_record = manifest_datasets[dataset]
        predictions_path = tables_root / f"{dataset}_cell_predictions.csv.gz"
        mask_path = tables_root / f"{dataset}_evaluation_mask_audit.csv.gz"
        scores_path = tables_root / f"{dataset}_class_scores.csv.gz"
        for path in (predictions_path, mask_path, scores_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Fixed-signature table is missing or empty: {path}")
        predictions = pd.read_csv(predictions_path, dtype={"cell_id": str})
        mask_audit = pd.read_csv(mask_path, dtype={"cell_id": str})
        if predictions["cell_id"].duplicated().any():
            raise RuntimeError(f"{dataset} prediction cell IDs are not unique")
        dataset_cells = int(dataset_record.get("cells", -1))
        if len(predictions) != dataset_cells:
            raise RuntimeError(f"{dataset} prediction count differs from the run manifest")
        if set(mask_audit["task"].astype(str)) != set(EXPECTED_TASK_TRUTH):
            raise RuntimeError(f"{dataset} evaluation-mask task set is not exact")
        semantic_hashes[dataset] = {}
        broad_truth = predictions["reference_broad_codex"].astype("string")
        subtype_truth = predictions["reference_tcell_subtype"].astype("string").where(
            broad_truth.eq("T cells"), pd.NA
        )
        task_truth = {"broad": broad_truth, "tcell_subtype": subtype_truth}
        reference_labels = {
            "broad": tuple(TASK_CLASS_ORDER["broad"]),
            "tcell_subtype": ("TH", "TFH", "TREG", "CD8T"),
        }
        for task_id, truth_col in EXPECTED_TASK_TRUTH.items():
            task_mask = mask_audit.loc[mask_audit["task"].astype(str).eq(task_id)].copy()
            if len(task_mask) != dataset_cells:
                raise RuntimeError(f"{dataset}/{task_id} mask row count is not one per cell")
            if task_mask["cell_id"].astype(str).tolist() != predictions["cell_id"].tolist():
                raise RuntimeError(f"{dataset}/{task_id} mask rows do not match prediction order")
            if set(task_mask["truth_col"].astype(str)) != {truth_col}:
                raise RuntimeError(f"{dataset}/{task_id} truth column contract changed")
            if set(task_mask["lineage_gate"].astype(str)) != {_lineage_gate(task_id)}:
                raise RuntimeError(f"{dataset}/{task_id} lineage gate contract changed")
            included = _strict_bool_series(
                task_mask["included_in_both_methods"],
                label=f"{dataset}/{task_id} included_in_both_methods",
            )
            expected_truth = task_truth[task_id].reset_index(drop=True)
            expected_included = expected_truth.isin(reference_labels[task_id])
            if not included.reset_index(drop=True).equals(expected_included.astype(bool)):
                raise RuntimeError(f"{dataset}/{task_id} mask is not derived from reference truth")
            audit_truth = task_mask["reference_truth"].astype("string").reset_index(drop=True)
            if not audit_truth.fillna("<NA>").equals(expected_truth.fillna("<NA>")):
                raise RuntimeError(f"{dataset}/{task_id} mask truth values differ from predictions")
            mask_hash = _evaluation_mask_sha256(
                pd.Index(predictions.loc[expected_included.to_numpy(), "cell_id"].astype(str))
            )
            if set(task_mask["evaluation_mask_sha256"].astype(str)) != {mask_hash}:
                raise RuntimeError(f"{dataset}/{task_id} mask cell-ID hash mismatch")
            semantic_hash = _evaluation_semantic_sha256(
                dataset,
                task_id,
                pd.Index(predictions["cell_id"].astype(str)),
                expected_truth,
                expected_included,
            )
            semantic_hashes[dataset][task_id] = semantic_hash
            if "evaluation_semantic_sha256" in task_mask.columns:
                if set(task_mask["evaluation_semantic_sha256"].astype(str)) != {semantic_hash}:
                    raise RuntimeError(f"{dataset}/{task_id} semantic mask hash mismatch")
            elif manifest.get("run_contract_version") in SUPPORTED_RUN_CONTRACT_VERSIONS:
                raise RuntimeError(f"{dataset}/{task_id} semantic mask hash is missing")
            recorded_task = dataset_record.get("evaluable_task_cells", {})
            recorded_mask = dataset_record.get("evaluation_mask_sha256", {})
            recorded_semantic = dataset_record.get("evaluation_semantic_sha256", {})
            n_evaluable = int(expected_included.sum())
            if int(recorded_task.get(task_id, -1)) != n_evaluable:
                raise RuntimeError(f"{dataset}/{task_id} manifest denominator mismatch")
            if recorded_mask.get(task_id) != mask_hash:
                raise RuntimeError(f"{dataset}/{task_id} manifest mask hash mismatch")
            if manifest.get("run_contract_version") in SUPPORTED_RUN_CONTRACT_VERSIONS:
                if recorded_semantic.get(task_id) != semantic_hash:
                    raise RuntimeError(f"{dataset}/{task_id} manifest semantic hash mismatch")
            grouped_rows = grouped.loc[
                grouped["dataset"].astype(str).eq(dataset)
                & grouped["task"].astype(str).eq(task_id)
            ]
            if set(grouped_rows["method"].astype(str)) != set(methods):
                raise RuntimeError(f"{dataset}/{task_id} grouped methods are incomplete")
            if set(grouped_rows["evaluation_mask_sha256"].astype(str)) != {mask_hash}:
                raise RuntimeError(f"{dataset}/{task_id} grouped mask hash mismatch")
            if "evaluation_semantic_sha256" in grouped_rows.columns:
                if set(grouped_rows["evaluation_semantic_sha256"].astype(str)) != {
                    semantic_hash
                }:
                    raise RuntimeError(f"{dataset}/{task_id} grouped semantic hash mismatch")
            elif manifest.get("run_contract_version") in SUPPORTED_RUN_CONTRACT_VERSIONS:
                raise RuntimeError(f"{dataset}/{task_id} grouped semantic hash is missing")
            metric_rows = pooled.loc[
                pooled["dataset"].astype(str).eq(dataset)
                & pooled["task"].astype(str).eq(task_id)
            ].copy()
            if set(metric_rows["method"].astype(str)) != set(methods) or len(metric_rows) != len(methods):
                raise RuntimeError(f"{dataset}/{task_id} methods are incomplete")
            if metric_rows["evaluation_mask_sha256"].astype(str).nunique() != 1:
                raise RuntimeError(f"{dataset}/{task_id} methods use different mask hashes")
            if set(metric_rows["evaluation_mask_sha256"].astype(str)) != {mask_hash}:
                raise RuntimeError(f"{dataset}/{task_id} metric mask hash mismatch")
            if "evaluation_semantic_sha256" in metric_rows.columns:
                if set(metric_rows["evaluation_semantic_sha256"].astype(str)) != {semantic_hash}:
                    raise RuntimeError(f"{dataset}/{task_id} metric semantic hash mismatch")
            elif manifest.get("run_contract_version") in SUPPORTED_RUN_CONTRACT_VERSIONS:
                raise RuntimeError(f"{dataset}/{task_id} metric semantic hash is missing")
            for _, metric_row in metric_rows.iterrows():
                if int(metric_row["n_cells"]) != n_evaluable:
                    raise RuntimeError(f"{dataset}/{task_id} method denominator mismatch")
                if int(metric_row["n_total_dataset_cells"]) != dataset_cells:
                    raise RuntimeError(f"{dataset}/{task_id} total cell count mismatch")
                if int(metric_row["n_reference_excluded"]) != dataset_cells - n_evaluable:
                    raise RuntimeError(f"{dataset}/{task_id} excluded-cell count mismatch")
                method = str(metric_row["method"])
                prediction_column = f"pred_{task_id}_{method}"
                if prediction_column not in predictions.columns:
                    raise RuntimeError(f"{dataset} lacks prediction column {prediction_column}")
                method_prediction = predictions.loc[
                    expected_included.to_numpy(), prediction_column
                ].astype(str)
                allowed = set(TASK_CLASS_ORDER[task_id]) | {"Unassigned"}
                unexpected = sorted(set(method_prediction) - allowed)
                if unexpected:
                    raise RuntimeError(
                        f"{dataset}/{task_id}/{method} has labels outside the model space: {unexpected}"
                    )
                outside_count = int((~method_prediction.isin(reference_labels[task_id])).sum())
                if int(metric_row["n_pred_outside_reference_label_space"]) != outside_count:
                    raise RuntimeError(
                        f"{dataset}/{task_id}/{method} outside-label count mismatch"
                    )
        scores = pd.read_csv(scores_path, usecols=["cell_id", "task", "class"])
        if set(scores["task"].astype(str)) != set(EXPECTED_TASK_TRUTH):
            raise RuntimeError(f"{dataset} score table task set is not exact")
        if set(scores["cell_id"].astype(str)) != set(predictions["cell_id"]):
            raise RuntimeError(f"{dataset} score table contains unknown cell IDs")
        if scores.duplicated(["cell_id", "task", "class"]).any():
            raise RuntimeError(f"{dataset} score table contains duplicate cell/task/class rows")
        expected_score_rows = dataset_cells * sum(
            len(TASK_CLASS_ORDER[task_id]) for task_id in EXPECTED_TASK_TRUTH
        )
        if len(scores) != expected_score_rows:
            raise RuntimeError(f"{dataset} score table is incomplete")
        for task_id in EXPECTED_TASK_TRUTH:
            observed_classes = set(
                scores.loc[scores["task"].astype(str).eq(task_id), "class"].astype(str)
            )
            if observed_classes != set(TASK_CLASS_ORDER[task_id]):
                raise RuntimeError(f"{dataset}/{task_id} score class set is not exact")
    return {
        "pooled_metric_rows": int(len(pooled)),
        "datasets": list(EXPECTED_DATASETS),
        "tasks": list(EXPECTED_TASK_TRUTH),
        "methods": list(METHOD_SPECS),
        "evaluation_semantic_sha256": semantic_hashes,
    }


def validate_fixed_signature_run(
    run_id: str,
    *,
    verify_artifacts: bool = True,
) -> dict[str, object]:
    """Validate one canonical or noncanonical run before reuse or rendering."""

    if RUN_NAME.fullmatch(run_id) is None:
        raise ValueError(f"Unsafe fixed-signature run ID: {run_id!r}")
    run_root = (RUNS_ROOT / run_id).resolve()
    if run_root.parent != RUNS_ROOT.resolve():
        raise ValueError("Fixed-signature run escapes the Final run root")
    manifest_path = run_root / "manifests" / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Fixed-signature run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "run_id": run_id,
        "direction_source": "dlbcl_ale",
        "direction_metric": EXPECTED_DIRECTION_METRIC,
        "source_artifacts_read_only": True,
        "source_hashes_verified_unchanged": True,
        "models_retrained": False,
        "classifier_objects_transferred_to_scRNA": False,
        "scRNA_labels_used_for_training": False,
        "outside_predictions_dropped": False,
        "evaluation_mask_contract": (
            "one fixed reference-derived mask per dataset/task shared by both methods"
        ),
        "tcell_subtype_lineage_gate": "reference_broad_codex == 'T cells'",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Fixed-signature run manifest mismatch: {key}")
    if manifest.get("evaluation_truth_columns") != EXPECTED_TASK_TRUTH:
        raise RuntimeError("Fixed-signature truth-column contract changed")
    if float(manifest.get("direction_consistency_threshold", -1)) != 0.8:
        raise RuntimeError("Fixed-signature direction-consistency threshold changed")
    if list(manifest.get("upstream_iterations", [])) != list(range(1, 101)):
        raise RuntimeError("Fixed-signature upstream iteration set is not 1..100")
    if int(manifest.get("upstream_top_k", -1)) != 20:
        raise RuntimeError("Fixed-signature upstream top-k contract changed")
    random_seed = int(manifest.get("random_seed", -1))
    if (
        int(manifest.get("upstream_step02_seed", -2)) != random_seed
        or int(manifest.get("upstream_step04b_seed", -3)) != random_seed
    ):
        raise RuntimeError("Fixed-signature upstream seed contract diverged")
    if not manifest.get("step02_run_id") or not manifest.get("step04b_run_id"):
        raise RuntimeError("Fixed-signature upstream run IDs are missing")
    expected_methods = [
        (method_id, spec["analysis_label"])
        for method_id, spec in METHOD_SPECS.items()
    ]
    observed_methods = [
        (item.get("method_id"), item.get("analysis_label"))
        for item in manifest.get("primary_methods", [])
    ]
    if observed_methods != expected_methods:
        raise RuntimeError("Fixed-signature run method contract is not exact")

    artifacts_verified = 0
    historical_final_copy_artifacts_superseded = 0
    canonical_performance_artifacts_verified = 0
    canonical_umap_artifacts_verified = 0
    if run_id == CANONICAL_RUN_ID:
        if manifest.get("canonical_derived_overwrite") is True:
            if (
                manifest.get("run_contract_version") != RUN_CONTRACT_VERSION
                or not isinstance(manifest.get("active_runtime_provenance"), dict)
                or manifest.get("canonical_derived_contract_version")
                != CANONICAL_DERIVED_CONTRACT_VERSION
                or manifest.get("protected_upstream_reused") is not True
                or manifest.get("raw_preprocessing_executed") is not False
                or manifest.get("proxy_features_recomputed") is not False
            ):
                raise RuntimeError("Canonical derived-generation contract is invalid")
            _validate_generated_manifest_provenance(manifest)
            recorded_artifacts = manifest.get("table_artifacts", {})
            actual_table_paths = {
                path.relative_to(run_root).as_posix()
                for path in (run_root / "tables").rglob("*")
                if path.is_file()
            }
            if set(recorded_artifacts) != actual_table_paths:
                raise RuntimeError(
                    "Canonical derived table set differs from its generation manifest"
                )
            if verify_artifacts:
                for relative, record in recorded_artifacts.items():
                    path = _require_within(
                        run_root / relative,
                        run_root / "tables",
                        label="canonical derived table",
                    )
                    if (
                        not path.is_file()
                        or int(path.stat().st_size) != int(record.get("size_bytes", -1))
                        or _sha256(path) != record.get("sha256")
                    ):
                        raise RuntimeError(
                            f"Canonical derived table hash mismatch: {relative}"
                        )
                    artifacts_verified += 1
        else:
            copy_manifest_path = run_root / "manifests" / "final_copy_manifest.json"
            if not copy_manifest_path.is_file():
                raise FileNotFoundError("Canonical Final copy manifest is missing")
            copy_manifest = json.loads(copy_manifest_path.read_text(encoding="utf-8"))
            if (
                copy_manifest.get("status") != "complete_exact_copy"
                or copy_manifest.get("run_id") != CANONICAL_RUN_ID
                or copy_manifest.get("all_hashes_match") is not True
                or copy_manifest.get("calculation_executed") is not False
                or copy_manifest.get("figure_regenerated") is not False
            ):
                raise RuntimeError("Canonical Final copy manifest is invalid")
            if verify_artifacts:
                legacy_performance_relatives = {
                    (
                        f"figures/v2/scRNA/{dataset}/"
                        f"{LEGACY_TWO_ARM_PERFORMANCE_STEMS[dataset]}.{suffix}"
                    )
                    for dataset in EXPECTED_DATASETS
                    for suffix in ("png", "pptx")
                }
                superseded_umap_relatives = {
                    (
                        f"figures/v2/scRNA/{dataset}/"
                        f"{FINAL_SCRNA_FIGURE_STEMS[(dataset, '3x3_umap_summary')]}"
                        f".{suffix}"
                    )
                    for dataset in EXPECTED_DATASETS
                    for suffix in ("png", "pptx")
                }
                expected_superseded_copy_relatives = (
                    legacy_performance_relatives | superseded_umap_relatives
                )
                superseded_final_copy_items: list[dict[str, object]] = []
                for item in copy_manifest.get("copied_files", []):
                    relative = str(item.get("destination_relative", ""))
                    if relative in expected_superseded_copy_relatives:
                        superseded_final_copy_items.append(item)
                        continue
                    path = _resolve_final_relative(relative)
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Canonical Final copy artifact is missing: {path}"
                        )
                    if _sha256(path) != item.get("sha256"):
                        raise RuntimeError(f"Canonical Final copy hash mismatch: {path}")
                    artifacts_verified += 1
                observed_superseded = {
                    str(item.get("destination_relative", ""))
                    for item in superseded_final_copy_items
                }
                if observed_superseded != expected_superseded_copy_relatives:
                    raise RuntimeError(
                        "Canonical Final copy supersession differs from the exact "
                        "legacy-performance plus replaced-UMAP eight-file set"
                    )
                remaining_legacy = [
                    relative
                    for relative in legacy_performance_relatives
                    if _resolve_final_relative(relative).exists()
                ]
                if remaining_legacy:
                    raise RuntimeError(
                        "Superseded two-arm performance copies remain: "
                        + ", ".join(sorted(remaining_legacy))
                    )
                canonical_manifest_path = (
                    COMPLETED_BUNDLE_ROOT
                    / "manifests"
                    / CANONICAL_THREE_ARM_FIGURE_MANIFEST
                )
                canonical_manifest = json.loads(
                    canonical_manifest_path.read_text(encoding="utf-8")
                )
                if (
                    canonical_manifest.get("status")
                    != "figures_rendered_from_completed_results"
                    or canonical_manifest.get("performance_sensitivity_run_id")
                    != CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID
                    or canonical_manifest.get("performance_contract")
                    != "canonical_compatible"
                    or canonical_manifest.get("canonical_figures_replaced") is not True
                    or canonical_manifest.get("canonical_figures_added") is not False
                ):
                    raise RuntimeError(
                        "Canonical Step12 individual Figure replacement contract changed"
                    )
                figure_artifacts = canonical_manifest.get("figure_artifacts", {})
                figure_sources = canonical_manifest.get("figure_sources", {})
                expected_current_relatives = {
                    (
                        f"figures/v2/scRNA/{dataset}/"
                        f"{FINAL_SCRNA_FIGURE_STEMS[(dataset, figure_kind)]}.{suffix}"
                    )
                    for dataset in EXPECTED_DATASETS
                    for figure_kind in ("performance", "3x3_umap_summary")
                    for suffix in ("png", "pptx")
                }
                if (
                    set(figure_artifacts) != expected_current_relatives
                    or set(figure_sources) != expected_current_relatives
                ):
                    raise RuntimeError(
                        "Canonical Step12 individual Figure artifact set is not exact"
                    )
                allowed_figure_states = {
                    "generated",
                    "reused_without_regeneration",
                }
                for dataset in EXPECTED_DATASETS:
                    for figure_kind in ("performance", "3x3_umap_summary"):
                        expected_source_run = (
                            CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID
                            if figure_kind == "performance"
                            else CANONICAL_RUN_ID
                        )
                        stem = FINAL_SCRNA_FIGURE_STEMS[(dataset, figure_kind)]
                        for suffix in ("png", "pptx"):
                            relative = f"figures/v2/scRNA/{dataset}/{stem}.{suffix}"
                            record = figure_artifacts.get(relative)
                            source = figure_sources.get(relative)
                            path = _resolve_final_relative(relative)
                            if not isinstance(record, dict) or not isinstance(source, dict):
                                raise RuntimeError(
                                    f"Canonical Step12 Figure record is missing: {relative}"
                                )
                            record_state = record.get("state")
                            if (
                                record_state not in allowed_figure_states
                                or source.get("state") != record_state
                                or source.get("dataset") != dataset
                                or source.get("figure_kind") != figure_kind
                                or source.get("source_run_id") != expected_source_run
                            ):
                                raise RuntimeError(
                                    f"Canonical Step12 Figure source changed: {relative}"
                                )
                            if (
                                not path.is_file()
                                or int(path.stat().st_size)
                                != int(record.get("size_bytes", -1))
                                or _sha256(path) != record.get("sha256")
                            ):
                                raise RuntimeError(
                                    f"Canonical Step12 Figure hash mismatch: {path}"
                                )
                            if figure_kind == "performance":
                                canonical_performance_artifacts_verified += 1
                            else:
                                canonical_umap_artifacts_verified += 1
                generated_figures = canonical_manifest.get("generated_figures")
                reused_figures = canonical_manifest.get("reused_figures")
                if not isinstance(generated_figures, list) or not isinstance(
                    reused_figures, list
                ):
                    raise RuntimeError(
                        "Canonical Step12 generated/reused Figure lists are missing"
                    )

                def _canonical_manifest_relatives(values: list[object]) -> set[str]:
                    relatives: set[str] = set()
                    for value in values:
                        path = Path(str(value))
                        relative = (
                            _relative_to_final(path.resolve())
                            if path.is_absolute()
                            else str(path)
                        )
                        if relative in relatives:
                            raise RuntimeError(
                                "Canonical Step12 Figure list contains duplicates"
                            )
                        relatives.add(relative)
                    return relatives

                generated_relatives = _canonical_manifest_relatives(generated_figures)
                reused_relatives = _canonical_manifest_relatives(reused_figures)
                expected_generated_relatives = {
                    relative
                    for relative, record in figure_artifacts.items()
                    if record.get("state") == "generated"
                }
                expected_reused_relatives = {
                    relative
                    for relative, record in figure_artifacts.items()
                    if record.get("state") == "reused_without_regeneration"
                }
                if (
                    generated_relatives & reused_relatives
                    or generated_relatives != expected_generated_relatives
                    or reused_relatives != expected_reused_relatives
                    or generated_relatives | reused_relatives
                    != expected_current_relatives
                ):
                    raise RuntimeError(
                        "Canonical Step12 generated/reused Figure partition changed"
                    )
                if expected_reused_relatives and canonical_manifest.get(
                    "promotion_contract", {}
                ).get("unselected_canonical_pairs_byte_checked") is not True:
                    raise RuntimeError(
                        "Canonical Step12 reused Figure byte-check contract changed"
                    )
                historical_final_copy_artifacts_superseded = len(
                    superseded_final_copy_items
                )
    else:
        contract_version = manifest.get("run_contract_version")
        if contract_version not in SUPPORTED_RUN_CONTRACT_VERSIONS:
            raise RuntimeError("Noncanonical fixed-signature run contract version is unsupported")
        if manifest.get("analysis_family") != ANALYSIS_FAMILY:
            raise RuntimeError("Noncanonical run analysis family is not fixed-signature")
        if contract_version == RUN_CONTRACT_VERSION:
            _validate_generated_manifest_provenance(manifest)
        recorded_artifacts = manifest.get("table_artifacts", {})
        actual_table_paths = {
            path.relative_to(run_root).as_posix()
            for path in (run_root / "tables").rglob("*")
            if path.is_file()
        }
        if set(recorded_artifacts) != actual_table_paths:
            raise RuntimeError("Noncanonical run table artifact set differs from its manifest")
        if verify_artifacts:
            for relative, record in recorded_artifacts.items():
                path = _require_within(run_root / relative, run_root / "tables", label="run table")
                if (
                    not path.is_file()
                    or int(path.stat().st_size) != int(record.get("size_bytes", -1))
                    or _sha256(path) != record.get("sha256")
                ):
                    raise RuntimeError(f"Noncanonical run table hash mismatch: {relative}")
                artifacts_verified += 1
        handoff_records = manifest.get("input_handoffs", {})
        if set(handoff_records) != set(EXPECTED_DATASETS):
            raise RuntimeError("Noncanonical run does not record both Step09/10 handoffs")
        handoff_paths = {
            dataset: _resolve_final_relative(
                handoff_records[dataset].get("manifest_path_relative_to_final", "")
            )
            for dataset in EXPECTED_DATASETS
        }
        handoff_reports = validate_prepared_input_manifests(
            run_id, handoff_paths, verify_hashes=verify_artifacts
        )
        for dataset, report in handoff_reports.items():
            record = handoff_records[dataset]
            if report["manifest_sha256"] != record.get("manifest_sha256"):
                raise RuntimeError(f"{dataset} handoff manifest hash changed after scoring")
            if report["feature_h5ad_sha256"] != record.get("feature_h5ad_sha256"):
                raise RuntimeError(f"{dataset} handoff feature hash differs from the run")
            if report["coverage_csv_sha256"] != record.get("coverage_csv_sha256"):
                raise RuntimeError(f"{dataset} handoff coverage hash differs from the run")
            if (
                report["signature_feature_order_sha256"]
                != record.get("signature_feature_order_sha256")
            ):
                raise RuntimeError(f"{dataset} handoff feature-order digest changed")
            large_input = manifest.get("large_inputs", {}).get(
                f"{dataset}_proxy_h5ad", {}
            )
            source_coverage = manifest.get("source_hashes", {}).get(
                f"{dataset}_coverage", {}
            )
            if large_input.get("sha256") != record.get("feature_h5ad_sha256"):
                raise RuntimeError(f"{dataset} scored H5AD differs from its Step09/10 handoff")
            if source_coverage.get("sha256") != record.get("coverage_csv_sha256"):
                raise RuntimeError(f"{dataset} scored coverage differs from its Step09/10 handoff")

    semantic_report = _validate_fixed_signature_table_semantics(run_root, manifest)
    return {
        "workflow": "Final fixed marker-only versus DLBCL-ALE direction × importance",
        "run_id": run_id,
        "run_root": str(run_root),
        "status": "complete",
        "analysis_family": ANALYSIS_FAMILY,
        "artifacts_verified": artifacts_verified,
        "historical_final_copy_artifacts_superseded": (
            historical_final_copy_artifacts_superseded
        ),
        "canonical_performance_artifacts_verified": (
            canonical_performance_artifacts_verified
        ),
        "canonical_umap_artifacts_verified": canonical_umap_artifacts_verified,
        "canonical_individual_figure_artifacts_verified": (
            canonical_performance_artifacts_verified
            + canonical_umap_artifacts_verified
        ),
        "final_copy_inventory_accounted_for": (
            artifacts_verified + historical_final_copy_artifacts_superseded == 41
            if run_id == CANONICAL_RUN_ID
            and manifest.get("canonical_derived_overwrite") is not True
            and verify_artifacts
            else None
        ),
        "semantic_contract_verified": True,
        **semantic_report,
    }


def validate_completed_run(
    *,
    verify_input_hashes: bool = True,
    canonical_proxy_datasets: tuple[str, ...] | list[str] = EXPECTED_DATASETS,
) -> dict[str, object]:
    """Validate the canonical derived bundle plus its immutable proxy inputs."""

    report = validate_fixed_signature_run(
        CANONICAL_RUN_ID,
        verify_artifacts=verify_input_hashes,
    )
    proxy_reports = validate_canonical_proxy_inputs(
        canonical_proxy_datasets,
        verify_hashes=verify_input_hashes,
    )
    report.update(
        {
            "reuse_only": True,
            "heavy_calculation_executed": False,
            "canonical_proxy_inputs": proxy_reports,
            "canonical_proxy_input_hashes_verified": bool(verify_input_hashes),
        }
    )
    return report


def validate_three_arm_sensitivity_run(
    run_id: str = CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID,
    *,
    verify_artifacts: bool = True,
) -> dict[str, object]:
    """Validate the pinned Step11B bundle used by Final Step12 performance panels."""

    if run_id != CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID:
        raise ValueError(
            "Final Step12 accepts only the code-pinned three-arm sensitivity run: "
            f"{CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID}"
        )
    run_root = (THREE_ARM_SENSITIVITY_ROOT / run_id).resolve()
    if run_root.parent != THREE_ARM_SENSITIVITY_ROOT.resolve():
        raise ValueError("Three-arm sensitivity run escapes its pinned root")
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Three-arm sensitivity manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "status": "complete",
        "analysis_family": "fixed_signature_three_arm_sensitivity",
        "run_id": run_id,
        "source_run_id": CANONICAL_RUN_ID,
        "canonical_bundle_modified": False,
        "umap_read": False,
        "umap_modified": False,
        "scrna_fitting": False,
        "model_training": False,
        "ale_recalculation": False,
        "permutation_importance_recalculation": False,
        "outside_predictions_dropped": False,
        "resolved_font_family": "Arial",
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Three-arm sensitivity manifest mismatch: {key}")
    if manifest.get("contracts") != ["canonical_compatible", "strict_conflict_free"]:
        raise RuntimeError("Three-arm sensitivity contract order changed")
    if set(manifest.get("methods", {})) != set(THREE_ARM_METHOD_SPECS):
        raise RuntimeError("Three-arm sensitivity method set changed")

    figure_root = Path(str(manifest.get("output_roots", {}).get("figures", ""))).resolve()
    expected_figure_root = (
        FIGURE_REEXPORT_ROOT / CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID
    ).resolve()
    if figure_root != expected_figure_root:
        raise RuntimeError("Three-arm sensitivity Figure root differs from its pinned path")
    if Path(str(manifest.get("output_roots", {}).get("tables", ""))).resolve() != run_root:
        raise RuntimeError("Three-arm sensitivity table root differs from its pinned path")

    artifacts_verified = 0
    table_artifacts_verified = 0
    historical_figure_artifacts_verified = 0
    missing_historical_figure_records: list[dict[str, object]] = []
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or len(artifacts) != 88:
        raise RuntimeError("Three-arm sensitivity artifact inventory is not the expected 88")
    for record in artifacts:
        kind = record.get("kind")
        base = run_root if kind == "table_bundle" else figure_root if kind == "figure_bundle" else None
        if base is None:
            raise RuntimeError(f"Unexpected three-arm artifact kind: {kind!r}")
        artifact_path = _require_within(
            base / str(record.get("relative_path", "")),
            base,
            label="three-arm artifact",
        )
        if not artifact_path.is_file() and kind == "figure_bundle":
            missing_historical_figure_records.append(record)
            continue
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Three-arm artifact is missing: {artifact_path}")
        if int(artifact_path.stat().st_size) != int(record.get("size_bytes", -1)):
            raise RuntimeError(f"Three-arm artifact size mismatch: {artifact_path}")
        if verify_artifacts and _sha256(artifact_path) != record.get("sha256"):
            raise RuntimeError(f"Three-arm artifact hash mismatch: {artifact_path}")
        artifacts_verified += 1
        if kind == "table_bundle":
            table_artifacts_verified += 1
        else:
            historical_figure_artifacts_verified += 1

    canonical_performance_artifacts_verified = 0
    historical_figure_artifacts_superseded = 0
    if missing_historical_figure_records:
        expected_historical = {
            f"Final_mv1_Step11B-01_GSE182434_marker_sign_importance_ablation_performance.{ext}"
            for ext in ("png", "pptx")
        } | {
            f"Final_mv1_Step11B-02_Roider_marker_sign_importance_ablation_performance.{ext}"
            for ext in ("png", "pptx")
        }
        missing_historical = {
            str(record.get("relative_path", ""))
            for record in missing_historical_figure_records
        }
        if historical_figure_artifacts_verified or missing_historical != expected_historical:
            raise RuntimeError(
                "Step11B historical Figure cleanup is partial or differs from the exact "
                "superseded four-file bundle"
            )
        canonical_manifest_path = (
            COMPLETED_BUNDLE_ROOT / "manifests" / CANONICAL_THREE_ARM_FIGURE_MANIFEST
        )
        if not canonical_manifest_path.is_file():
            raise FileNotFoundError(
                "Canonical Step12 performance supersession manifest is missing: "
                f"{canonical_manifest_path}"
            )
        canonical_manifest = json.loads(
            canonical_manifest_path.read_text(encoding="utf-8")
        )
        if (
            canonical_manifest.get("status")
            != "figures_rendered_from_completed_results"
            or canonical_manifest.get("performance_sensitivity_run_id") != run_id
            or canonical_manifest.get("performance_contract") != "canonical_compatible"
        ):
            raise RuntimeError("Canonical Step12 performance supersession contract changed")
        figure_artifacts = canonical_manifest.get("figure_artifacts", {})
        figure_sources = canonical_manifest.get("figure_sources", {})
        for dataset in EXPECTED_DATASETS:
            stem = FINAL_SCRNA_FIGURE_STEMS[(dataset, "performance")]
            for suffix in ("png", "pptx"):
                relative = f"figures/v2/scRNA/{dataset}/{stem}.{suffix}"
                artifact_record = figure_artifacts.get(relative)
                source_record = figure_sources.get(relative)
                if not isinstance(artifact_record, dict) or not isinstance(
                    source_record, dict
                ):
                    raise RuntimeError(
                        f"Canonical Step12 performance record is missing: {relative}"
                    )
                if (
                    source_record.get("dataset") != dataset
                    or source_record.get("figure_kind") != "performance"
                    or source_record.get("source_run_id") != run_id
                ):
                    raise RuntimeError(
                        f"Canonical Step12 performance source changed: {relative}"
                    )
                artifact_path = _resolve_final_relative(relative)
                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"Canonical Step12 performance artifact is missing: {artifact_path}"
                    )
                if int(artifact_path.stat().st_size) != int(
                    artifact_record.get("size_bytes", -1)
                ):
                    raise RuntimeError(
                        f"Canonical Step12 performance size mismatch: {artifact_path}"
                    )
                if verify_artifacts and _sha256(artifact_path) != artifact_record.get(
                    "sha256"
                ):
                    raise RuntimeError(
                        f"Canonical Step12 performance hash mismatch: {artifact_path}"
                    )
                canonical_performance_artifacts_verified += 1
        historical_figure_artifacts_superseded = len(
            missing_historical_figure_records
        )

    for relative, expected_hash in manifest.get("source_hashes", {}).items():
        source_path = _resolve_final_relative(str(relative))
        if not source_path.is_file():
            raise FileNotFoundError(f"Three-arm source dependency is missing: {source_path}")
        if verify_artifacts and _sha256(source_path) != expected_hash:
            raise RuntimeError(f"Three-arm source dependency hash mismatch: {source_path}")

    metrics_path = run_root / "tables" / "three_arm_pooled_cell_metrics.csv"
    deltas_path = run_root / "tables" / "three_arm_pairwise_metric_deltas.csv"
    alias_path = run_root / "tables" / "alias_direction_conflict_audit.csv"
    metrics = pd.read_csv(metrics_path)
    expected_keys = {
        (contract, dataset, task, method)
        for contract in ("canonical_compatible", "strict_conflict_free")
        for dataset in EXPECTED_DATASETS
        for task in EXPECTED_TASK_TRUTH
        for method in THREE_ARM_METHOD_SPECS
    }
    observed_keys = set(
        metrics.loc[:, ["ablation_contract", "dataset", "task", "method"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if len(metrics) != 24 or observed_keys != expected_keys:
        raise RuntimeError("Three-arm pooled metric key set is not exact")
    if set(metrics["scope"].astype(str)) != {"pooled_cells"}:
        raise RuntimeError("Three-arm pooled metric scope changed")
    if set(metrics["routing"].astype(str)) != {EXPECTED_ROUTING}:
        raise RuntimeError("Three-arm routing changed")
    if _strict_bool_series(
        metrics["outside_predictions_dropped"],
        label="three-arm outside_predictions_dropped",
    ).any():
        raise RuntimeError("Three-arm metrics drop outside-label predictions")
    for _, frame in metrics.groupby(["ablation_contract", "dataset", "task"], sort=False):
        for column in (
            "n_cells",
            "n_total_dataset_cells",
            "n_reference_excluded",
            "truth_col",
            "routing",
            "evaluation_mask_sha256",
            "evaluation_semantic_sha256",
            "reference_present_classes",
            "model_label_space",
        ):
            if frame[column].nunique(dropna=False) != 1:
                raise RuntimeError(f"Three-arm shared comparison contract changed: {column}")

    canonical_metrics = pd.read_csv(COMPLETED_BUNDLE_ROOT / "tables" / "pooled_cell_metrics.csv")
    compatibility = metrics.loc[metrics["ablation_contract"].eq("canonical_compatible")]
    method_map = {"marker_only": "equal_weight", "sign_plus_importance": "direction_aware"}
    equality_columns = (
        "balanced_accuracy",
        "macro_f1_reference_present_classes",
        "fraction_pred_outside_reference_label_space",
        "n_pred_outside_reference_label_space",
        "n_cells",
        "evaluation_mask_sha256",
    )
    for sensitivity_method, canonical_method in method_map.items():
        left = compatibility.loc[compatibility["method"].eq(sensitivity_method)].set_index(
            ["dataset", "task"]
        )
        right = canonical_metrics.loc[canonical_metrics["method"].eq(canonical_method)].set_index(
            ["dataset", "task"]
        )
        if not left.index.equals(right.index):
            left = left.sort_index()
            right = right.sort_index()
        if not left.index.equals(right.index):
            raise RuntimeError("Step11B compatibility endpoints do not match Step11 keys")
        for column in equality_columns:
            if pd.api.types.is_numeric_dtype(left[column]):
                if not np.allclose(
                    left[column].to_numpy(dtype=float),
                    right[column].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"Step11B {sensitivity_method} differs from Step11 {canonical_method}: {column}"
                    )
            elif not left[column].astype(str).equals(right[column].astype(str)):
                raise RuntimeError(
                    f"Step11B {sensitivity_method} differs from Step11 {canonical_method}: {column}"
                )

    source_semantics = manifest.get("source_validation", {}).get(
        "evaluation_semantic_sha256", {}
    )
    for (dataset, task), frame in compatibility.groupby(["dataset", "task"], sort=False):
        observed = frame["evaluation_semantic_sha256"].astype(str).unique()
        expected = str(source_semantics.get(dataset, {}).get(task, ""))
        if len(observed) != 1 or observed[0] != expected:
            raise RuntimeError(
                f"Step11B evaluation semantic hash differs from Step11: {dataset}/{task}"
            )

    deltas = pd.read_csv(deltas_path)
    strict = deltas.loc[
        deltas["ablation_contract"].eq("strict_conflict_free")
        & deltas["metric"].isin(
            ["balanced_accuracy", "macro_f1_reference_present_classes"]
        )
    ]
    if len(strict) != 16 or not strict["delta"].astype(float).gt(0).all():
        raise RuntimeError("Strict conflict-free BAcc/F1 stepwise deltas are not all positive")
    aliases = pd.read_csv(alias_path)
    if len(aliases) != 4 or set(aliases["proxy_group"].astype(str)) != {"PTPRC"}:
        raise RuntimeError("Expected four PTPRC alias-conflict audit rows")

    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "deltas_path": str(deltas_path),
        "deltas_sha256": _sha256(deltas_path),
        "alias_audit_path": str(alias_path),
        "alias_audit_sha256": _sha256(alias_path),
        "figure_root": str(figure_root),
        "artifacts_verified": artifacts_verified,
        "table_artifacts_verified": table_artifacts_verified,
        "historical_figure_artifacts_verified": historical_figure_artifacts_verified,
        "historical_figure_artifacts_superseded": historical_figure_artifacts_superseded,
        "canonical_performance_artifacts_verified": canonical_performance_artifacts_verified,
        "artifact_inventory_accounted_for": (
            artifacts_verified + historical_figure_artifacts_superseded == 88
        ),
        "endpoint_equivalence_verified": True,
        "strict_stepwise_deltas_positive": True,
        "pooled_metrics": metrics,
    }


def validate_final_source_inputs() -> dict[str, object]:
    """Verify the Final-owned Step02/04B inputs used by a future new run."""

    manifest_path = (
        FINAL_ROOT / "tables" / "scRNA" / "fixed_signature" / "source_inputs"
        / "source_inputs_copy_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_exact_copy"
        or manifest.get("all_hashes_match") is not True
        or manifest.get("calculation_executed") is not False
    ):
        raise RuntimeError("Final fixed-signature source-input copy contract is invalid")
    for item in manifest.get("copied_files", []):
        path = FINAL_ROOT / item["destination_relative"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"Final source-input hash mismatch: {path}")
    return {
        "status": "complete",
        "manifest": str(manifest_path),
        "files_verified": len(manifest.get("copied_files", [])),
    }


def _zscore(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    if not np.isfinite(std) or std <= 0:
        return np.zeros(len(array), dtype=float)
    return (array - mean) / std


def _load_dataset_obs(
    dataset: str,
) -> tuple[pd.DataFrame, np.ndarray, Path, dict[str, object]]:
    spec = DATASETS[dataset]
    h5ad_path = Path(spec["h5ad"])
    if not h5ad_path.is_file():
        raise FileNotFoundError(h5ad_path)
    before = h5ad_path.stat()
    input_sha256 = _sha256(h5ad_path)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = backed.obs.copy()
        if "X_umap" not in backed.obsm:
            raise KeyError(f"Saved UMAP is missing: {h5ad_path}")
        umap = np.asarray(backed.obsm["X_umap"], dtype=float).copy()
    finally:
        backed.file.close()
    if umap.shape != (len(obs), 2):
        raise RuntimeError(f"Unexpected UMAP shape for {dataset}: {umap.shape}")
    after = h5ad_path.stat()
    unchanged = (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    if not unchanged:
        raise RuntimeError(f"Read-only scRNA input changed while reading: {h5ad_path}")
    obs.index = obs.index.astype(str)
    return (
        obs,
        umap,
        h5ad_path,
        {
            "path": str(h5ad_path),
            "size_bytes_before": int(before.st_size),
            "size_bytes_after": int(after.st_size),
            "mtime_ns_before": int(before.st_mtime_ns),
            "mtime_ns_after": int(after.st_mtime_ns),
            "input_unchanged": True,
            "sha256": input_sha256,
            "sha256_skipped": False,
        },
    )


def _proxy_zscores(obs: pd.DataFrame, proxy_weights: pd.DataFrame) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for proxy_group, group in proxy_weights.groupby("proxy_group", sort=False):
        feature_cols = sorted(
            {
                value
                for joined in group["feature_cols"].astype(str)
                for value in joined.split("|")
                if value
            }
        )
        missing = [column for column in feature_cols if column not in obs.columns]
        if missing:
            raise KeyError(f"scRNA proxy columns are missing for {proxy_group}: {missing}")
        if not feature_cols:
            raise RuntimeError(f"No feature column resolves proxy group {proxy_group}")
        reference = pd.to_numeric(obs[feature_cols[0]], errors="coerce").fillna(0.0).to_numpy(
            dtype=float
        )
        for column in feature_cols[1:]:
            candidate = pd.to_numeric(obs[column], errors="coerce").fillna(0.0).to_numpy(
                dtype=float
            )
            if not np.allclose(reference, candidate, atol=1e-10, rtol=0.0):
                raise RuntimeError(
                    f"CODEX marker columns sharing {proxy_group} are not identical: "
                    f"{feature_cols}"
                )
        cache[str(proxy_group)] = _zscore(obs[feature_cols[0]])
    return cache


def score_dataset(
    obs: pd.DataFrame,
    proxy_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-stage predictions and long class scores without modifying the h5ad."""

    zscores = _proxy_zscores(obs, proxy_weights)
    predictions = pd.DataFrame(index=obs.index)
    score_tables: list[pd.DataFrame] = []
    for task, class_order in TASK_CLASS_ORDER.items():
        task_weights = proxy_weights.loc[proxy_weights["task"].eq(task)].copy()
        observed_classes = set(task_weights["class"].astype(str))
        if observed_classes != set(class_order):
            raise RuntimeError(
                f"Signature class mismatch for {task}: expected {class_order}, "
                f"observed {sorted(observed_classes)}"
            )
        method_matrices: dict[str, np.ndarray] = {}
        for method, method_spec in METHOD_SPECS.items():
            weight_col = str(method_spec["weight_col"])
            class_scores: list[np.ndarray] = []
            for class_name in class_order:
                class_rows = task_weights.loc[task_weights["class"].eq(class_name)]
                score = np.zeros(len(obs), dtype=float)
                for row in class_rows.itertuples(index=False):
                    score += zscores[str(row.proxy_group)] * float(getattr(row, weight_col))
                class_scores.append(score)
            matrix = np.column_stack(class_scores)
            method_matrices[method] = matrix
            pred = np.asarray(class_order, dtype=object)[np.argmax(matrix, axis=1)]
            sorted_scores = np.sort(matrix, axis=1)
            margin = sorted_scores[:, -1] - sorted_scores[:, -2]
            tied = np.isclose(margin, 0.0, atol=1e-12, rtol=0.0)
            pred[tied] = "Unassigned"
            predictions[f"pred_{task}_{method}"] = pred
            predictions[f"score_max_{task}_{method}"] = matrix.max(axis=1)
            predictions[f"score_margin_{task}_{method}"] = margin
        for class_index, class_name in enumerate(class_order):
            score_tables.append(
                pd.DataFrame(
                    {
                        "cell_id": obs.index,
                        "task": task,
                        "class": class_name,
                        "equal_weight_score": method_matrices["equal_weight"][:, class_index],
                        "direction_aware_score": method_matrices["direction_aware"][:,
                            class_index],
                    }
                )
            )
    return predictions, pd.concat(score_tables, ignore_index=True)


def _make_tasks(dataset: str, obs: pd.DataFrame) -> list[EvaluationTask]:
    if dataset not in DATASETS:
        raise KeyError(dataset)
    required = {"reference_broad_codex", "reference_tcell_subtype"}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise KeyError(f"{dataset} lacks existing V2 reference target columns: {missing}")
    broad_truth = obs["reference_broad_codex"].astype("string")
    tcell_truth_raw = obs["reference_tcell_subtype"].astype("string")
    # Subtype validation is explicitly restricted by the independent broad
    # reference lineage; subtype text outside reference T cells never enters
    # the metric denominator.
    tcell_truth = tcell_truth_raw.where(broad_truth.eq("T cells"), pd.NA)
    return [
        EvaluationTask(
            task_id="broad",
            truth_col="reference_broad_codex",
            reference_labels=tuple(TASK_CLASS_ORDER["broad"]),
            model_labels=tuple(TASK_CLASS_ORDER["broad"] + ["Unassigned"]),
            truth=broad_truth,
            description=(
                "Existing V2 broad reference target space; reference-mapped cells only"
            ),
        ),
        EvaluationTask(
            task_id="tcell_subtype",
            truth_col="reference_tcell_subtype",
            reference_labels=("TH", "TFH", "TREG", "CD8T"),
            model_labels=tuple(TASK_CLASS_ORDER["tcell_subtype"] + ["Unassigned"]),
            truth=tcell_truth,
            description=(
                "Existing V2 evaluable T-cell target space gated by "
                "reference_broad_codex == 'T cells'; Roider TTOX is already represented "
                "as CD8T in the saved reference_tcell_subtype column"
            ),
        ),
    ]


def _display_predictions(values: pd.Series) -> pd.Series:
    return values.astype(str).map(CLASS_DISPLAY).fillna(values.astype(str))


def _evaluation_mask_sha256(cell_ids: pd.Index) -> str:
    digest = hashlib.sha256()
    for cell_id in cell_ids.astype(str):
        digest.update(cell_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _evaluate_vectors(
    y_true: pd.Series,
    y_pred: pd.Series,
    reference_labels: tuple[str, ...],
    model_labels: tuple[str, ...],
) -> dict[str, object]:
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    present = [label for label in reference_labels if label in set(y_true)]
    if not present:
        raise RuntimeError("No evaluable reference class remains")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_present = float(
        f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    )
    macro_model = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(model_labels),
            average="macro",
            zero_division=0,
        )
    )
    outside = ~y_pred.isin(reference_labels)
    return {
        "n_cells": int(len(y_true)),
        "balanced_accuracy": bacc,
        "macro_f1_reference_present_classes": macro_present,
        "macro_f1_model_label_space": macro_model,
        "macro_f1_model_label_space_role": "secondary_diagnostic",
        "reference_present_classes": "|".join(present),
        "model_label_space": "|".join(model_labels),
        "n_pred_outside_reference_label_space": int(outside.sum()),
        "fraction_pred_outside_reference_label_space": float(outside.mean()),
        "outside_predictions_dropped": False,
    }


def evaluate_dataset(
    dataset: str,
    obs: pd.DataFrame,
    predictions: pd.DataFrame,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    pooled_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    confusion_tables: dict[str, pd.DataFrame] = {}
    report_tables: dict[str, pd.DataFrame] = {}
    for task in _make_tasks(dataset, obs):
        truth = task.truth
        mask = truth.isin(task.reference_labels)
        if not mask.any():
            raise RuntimeError(f"No evaluable cells for {dataset}/{task.task_id}")
        mask_hash = _evaluation_mask_sha256(obs.index[mask])
        semantic_hash = _evaluation_semantic_sha256(
            dataset,
            task.task_id,
            obs.index,
            truth,
            mask,
        )
        for method in METHOD_SPECS:
            raw_pred = predictions.loc[mask, f"pred_{task.task_id}_{method}"]
            pred = _display_predictions(raw_pred)
            y_true = truth.loc[mask].astype(str)
            metrics = _evaluate_vectors(
                y_true,
                pred,
                task.reference_labels,
                task.model_labels,
            )
            pooled_rows.append(
                {
                    "dataset": dataset,
                    "task": task.task_id,
                    "method": method,
                    "scope": "pooled_cells",
                    "truth_col": task.truth_col,
                    "routing": "fixed_reference_target_mask",
                    "evaluation_mask_sha256": mask_hash,
                    "evaluation_semantic_sha256": semantic_hash,
                    "n_total_dataset_cells": int(len(obs)),
                    "n_reference_excluded": int((~mask).sum()),
                    "task_description": task.description,
                    **metrics,
                }
            )
            confusion = pd.crosstab(y_true, pred).reindex(
                index=list(task.reference_labels),
                columns=list(task.model_labels),
                fill_value=0,
            )
            confusion.index.name = "reference"
            confusion.columns.name = "predicted"
            key = f"{dataset}_{task.task_id}_{method}"
            confusion_tables[key] = confusion
            report_tables[key] = pd.DataFrame(
                classification_report(
                    y_true,
                    pred,
                    labels=[
                        label
                        for label in task.reference_labels
                        if label in set(y_true)
                    ],
                    output_dict=True,
                    zero_division=0,
                )
            ).T
            if group_col not in obs.columns:
                raise KeyError(f"Grouping column is missing for {dataset}: {group_col}")
            groups = obs.loc[mask, group_col].astype("string").fillna("Missing")
            grouped_frame = pd.DataFrame(
                {"truth": y_true, "prediction": pred, "group": groups},
                index=y_true.index,
            )
            for group_name, frame in grouped_frame.groupby("group", sort=True):
                group_metrics = _evaluate_vectors(
                    frame["truth"],
                    frame["prediction"],
                    task.reference_labels,
                    task.model_labels,
                )
                group_rows.append(
                    {
                        "dataset": dataset,
                        "task": task.task_id,
                        "method": method,
                        "group_column": group_col,
                        "group": str(group_name),
                        "truth_col": task.truth_col,
                        "evaluation_mask_sha256": mask_hash,
                        "evaluation_semantic_sha256": semantic_hash,
                        **group_metrics,
                    }
                )
    return (
        pd.DataFrame(pooled_rows),
        pd.DataFrame(group_rows),
        confusion_tables,
        report_tables,
    )


def _save_review_figure_pair(
    fig,
    output_path: Path,
    *,
    rasterize_dense_layers: bool = False,
    dense_layer_point_threshold: int = 100,
    dense_layer_mode: str = "full_canvas",
) -> tuple[Path | None, Path | None]:
    """Save or skip one canonical Figure pair through the shared target policy."""

    from Final_utils import save_final_figure_pair

    png_path = Path(output_path)
    if png_path.suffix.lower() != ".png":
        raise ValueError(f"Final Figure output must be a PNG path: {png_path}")
    png, pptx = save_final_figure_pair(
        fig,
        png_path,
        png_path.with_suffix(".pptx"),
        dpi=300,
        facecolor="white",
        preserve_target_directory=True,
        pptx_rasterize_dense_layers=rasterize_dense_layers,
        pptx_dense_layer_point_threshold=dense_layer_point_threshold,
        pptx_dense_layer_mode=dense_layer_mode,
    )
    return (
        Path(png) if png is not None else None,
        Path(pptx) if pptx is not None else None,
    )


def _scatter_categorical(
    ax,
    umap: np.ndarray,
    labels: pd.Series,
    mask: np.ndarray,
    order: list[str],
    palette: dict[str, str],
    size: float = 1.2,
) -> None:
    ax.scatter(umap[:, 0], umap[:, 1], s=0.6, c="#E1E1E1", alpha=0.22, linewidths=0)
    values = labels.astype("string")
    for label in order:
        current = mask & values.eq(label).fillna(False).to_numpy(dtype=bool)
        if current.any():
            ax.scatter(
                umap[current, 0],
                umap[current, 1],
                s=size,
                color=palette.get(label, "#666666"),
                alpha=0.78,
                linewidths=0,
                label=label,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _categorical_legend_handles(
    order: list[str],
    palette: dict[str, str],
    *,
    markersize: float,
) -> list[plt.Line2D]:
    """Build display-only categorical handles with consistent marker geometry."""

    display = {"Monocytes and Macrophages": "Mono and Macro"}
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[label],
            markeredgecolor="none",
            markersize=markersize,
            label=display.get(label, label),
        )
        for label in order
        if label != "Unassigned" and label in palette
    ]


def _add_stacked_categorical_legends(
    fig,
    parent_spec,
    groups: list[tuple[list[plt.Line2D], str]],
    *,
    fontsize: float,
    title_fontsize: float,
    hspace: float,
    labelspacing: float,
    handletextpad: float,
    alignment: str = "center",
    anchor_x: float | None = None,
) -> None:
    """Place categorical legends in dedicated, item-proportional stacked axes."""

    if alignment not in {"left", "center", "right"}:
        raise ValueError(f"Unsupported legend alignment: {alignment!r}")
    if anchor_x is not None and not 0.0 <= anchor_x <= 1.0:
        raise ValueError(f"Legend anchor_x must be within [0, 1]: {anchor_x}")
    height_ratios = [
        max(
            3.0,
            sum(str(handle.get_label()).count("\n") + 1 for handle in handles)
            + 1.5,
        )
        for handles, _ in groups
    ]
    legend_grid = parent_spec.subgridspec(
        len(groups),
        1,
        height_ratios=height_ratios,
        hspace=hspace,
    )
    for row, (handles, title) in enumerate(groups):
        legend_ax = fig.add_subplot(legend_grid[row, 0])
        legend_ax.axis("off")
        if not handles:
            continue
        legend_location = f"upper {alignment}"
        legend_anchor_x = (
            {"left": 0.0, "center": 0.5, "right": 1.0}[alignment]
            if anchor_x is None
            else anchor_x
        )
        legend = legend_ax.legend(
            handles=handles,
            loc=legend_location,
            bbox_to_anchor=(legend_anchor_x, 1.0),
            frameon=False,
            borderaxespad=0,
            labelspacing=labelspacing,
            handletextpad=handletextpad,
            fontsize=fontsize,
            title=title,
            title_fontsize=title_fontsize,
            alignment=alignment,
        )
        legend.get_title().set_fontweight("bold")


def save_v9_umap_summary(
    *,
    dataset: str,
    obs: pd.DataFrame,
    umap: np.ndarray,
    predictions: pd.DataFrame,
    output_path: Path,
) -> tuple[Path | None, Path | None]:
    """Preserve the existing dataset-specific V9 3x3 fixed-signature layout."""

    tasks = {task.task_id: task for task in _make_tasks(dataset, obs)}
    broad = tasks["broad"]
    tcell = tasks["tcell_subtype"]
    broad_mask = broad.truth.isin(broad.reference_labels).to_numpy(dtype=bool)
    t_mask = tcell.truth.isin(tcell.reference_labels).to_numpy(dtype=bool)
    metadata_col = str(DATASETS[dataset]["group_col"])
    metadata = obs[metadata_col].astype("string")
    celltype_col = (
        "reference_population_raw" if "reference_population_raw" in obs else broad.truth_col
    )
    celltype = obs[celltype_col].astype("string")
    metadata_order = sorted(metadata[broad_mask].dropna().unique())
    celltype_order = sorted(celltype[broad_mask].dropna().unique())
    metadata_colors = {
        value: plt.cm.tab20(index % 20) for index, value in enumerate(metadata_order)
    }
    celltype_colors = {
        value: plt.cm.tab20(index % 20) for index, value in enumerate(celltype_order)
    }
    fig = plt.figure(figsize=INDIVIDUAL_UMAP_CANVAS_INCHES, facecolor="white")
    grid = fig.add_gridspec(
        3,
        5,
        width_ratios=[1, 1, 1, SCRNA_UMAP_COLORBAR_COLUMN_RATIO, 0.92],
        wspace=0.07,
        hspace=0.22,
    )
    axes = [[fig.add_subplot(grid[row, column]) for column in range(3)] for row in range(3)]
    _scatter_categorical(
        axes[0][0], umap, metadata, broad_mask, metadata_order, metadata_colors
    )
    axes[0][0].set_title(
        f"Metadata {metadata_col}", fontsize=11.5, pad=4, fontweight="bold"
    )
    _scatter_categorical(
        axes[0][1], umap, celltype, broad_mask, celltype_order, celltype_colors
    )
    axes[0][1].set_title(
        "Metadata cell type", fontsize=11.5, pad=4, fontweight="bold"
    )
    hladr = pd.to_numeric(
        obs.get("codex_HLA-DR", pd.Series(0.0, index=obs.index)), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    axes[0][2].scatter(
        umap[:, 0], umap[:, 1], s=0.6, c="#E1E1E1", alpha=0.2, linewidths=0
    )
    scatter = axes[0][2].scatter(
        umap[broad_mask, 0],
        umap[broad_mask, 1],
        s=1.2,
        c=hladr[broad_mask],
        cmap="viridis",
        linewidths=0,
    )
    axes[0][2].set_title(
        "HLA-DR RNA proxy", fontsize=11.5, pad=4, fontweight="bold"
    )
    axes[0][2].set_xticks([])
    axes[0][2].set_yticks([])
    colorbar_ax = fig.add_subplot(grid[0, 3])
    colorbar = fig.colorbar(scatter, cax=colorbar_ax)
    colorbar.set_label("RNA proxy signal", fontsize=10.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9.5)

    broad_order = list(broad.model_labels)
    _scatter_categorical(
        axes[1][0], umap, broad.truth, broad_mask, broad_order, BROAD_PALETTE
    )
    axes[1][0].set_title(
        f"Broad scRNA reference\n(n={int(broad_mask.sum()):,})",
        fontsize=11.5,
        pad=4,
        fontweight="bold",
    )
    _scatter_categorical(
        axes[1][1],
        umap,
        predictions["pred_broad_equal_weight"],
        broad_mask,
        broad_order,
        BROAD_PALETTE,
    )
    axes[1][1].set_title(
        "Marker only", fontsize=11.5, pad=4, fontweight="bold"
    )
    _scatter_categorical(
        axes[1][2],
        umap,
        predictions["pred_broad_direction_aware"],
        broad_mask,
        broad_order,
        BROAD_PALETTE,
    )
    axes[1][2].set_title(
        FIGURE_DIRECTION_IMPORTANCE_LABEL,
        fontsize=11.5,
        pad=4,
        fontweight="bold",
    )

    t_order = list(tcell.model_labels)
    _scatter_categorical(axes[2][0], umap, tcell.truth, t_mask, t_order, T_PALETTE)
    axes[2][0].set_title(
        f"T-cell scRNA reference\n(n={int(t_mask.sum()):,})",
        fontsize=11.5,
        pad=4,
        fontweight="bold",
    )
    _scatter_categorical(
        axes[2][1],
        umap,
        predictions["pred_tcell_subtype_equal_weight"],
        t_mask,
        t_order,
        T_PALETTE,
    )
    axes[2][1].set_title(
        "Marker only", fontsize=11.5, pad=4, fontweight="bold"
    )
    _scatter_categorical(
        axes[2][2],
        umap,
        predictions["pred_tcell_subtype_direction_aware"],
        t_mask,
        t_order,
        T_PALETTE,
    )
    axes[2][2].set_title(
        FIGURE_DIRECTION_IMPORTANCE_LABEL,
        fontsize=11.5,
        pad=4,
        fontweight="bold",
    )

    legend_groups = [
        (
            _categorical_legend_handles(
                metadata_order, metadata_colors, markersize=7.0
            ),
            f"Metadata {metadata_col}",
        ),
        (
            _categorical_legend_handles(
                celltype_order, celltype_colors, markersize=7.0
            ),
            "Metadata cell type",
        ),
        (
            _categorical_legend_handles(
                broad_order, BROAD_PALETTE, markersize=7.0
            ),
            "Broad label",
        ),
        (
            _categorical_legend_handles(t_order, T_PALETTE, markersize=7.0),
            "T-cell subtype",
        ),
    ]
    _add_stacked_categorical_legends(
        fig,
        grid[:, 4],
        legend_groups,
        fontsize=10.0,
        title_fontsize=10.5,
        hspace=0.10,
        labelspacing=0.12,
        handletextpad=0.30,
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.925, bottom=0.055)
    umap_block_center_x = (
        axes[2][0].get_position().x0 + axes[2][2].get_position().x1
    ) / 2
    umap_block_center_y = (
        axes[2][0].get_position().y0 + axes[0][0].get_position().y1
    ) / 2
    fig.text(
        umap_block_center_x,
        0.015,
        "UMAP1",
        ha="center",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.015,
        umap_block_center_y,
        "UMAP2",
        ha="center",
        va="center",
        rotation=90,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.suptitle(
        f"{FIGURE_DATASET_DISPLAY_LABELS[dataset]}: CODEX ML marker-signature validation",
        y=0.985,
        fontsize=14.0,
        fontweight="bold",
    )
    try:
        return _save_review_figure_pair(
            fig,
            output_path,
            rasterize_dense_layers=True,
            dense_layer_point_threshold=100,
            dense_layer_mode="axis_local",
        )
    finally:
        plt.close(fig)


def save_v9_performance_pair(
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
) -> tuple[Path | None, Path | None]:
    methods = list(METHOD_SPECS)
    direction_label = METHOD_SPECS["direction_aware"]["display"]
    labels = [
        METHOD_SPECS["equal_weight"]["display"],
        direction_label.replace(" + ", " +\n", 1).replace(" × ", " ×\n", 1),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(4.1, 6.9), facecolor="white")
    task_frames: dict[str, pd.DataFrame] = {}
    for ax, task, task_title in zip(
        axes,
        ["broad", "tcell_subtype"],
        ["Broad lineage", "T-cell subtype"],
    ):
        frame = metrics.loc[metrics["task"].eq(task)].set_index("method").reindex(methods)
        task_frames[task] = frame
        n_values = frame["n_cells"].dropna().astype(int).unique()
        if len(n_values) != 1:
            raise RuntimeError(f"{task} Figure metrics have inconsistent cell counts")
        x = np.arange(len(methods))
        width = 0.34
        for offset, column, metric_label, color in [
            (-0.5, "balanced_accuracy", "Balanced accuracy", "#4E79A7"),
            (
                0.5,
                "macro_f1_reference_present_classes",
                "Macro F1 (truth-present)",
                "#E15759",
            ),
        ]:
            bars = ax.bar(
                x + offset * width,
                frame[column].to_numpy(dtype=float),
                width,
                color=color,
                label=metric_label,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.018,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.12)
        ax.set_title(
            f"{task_title} vs scRNA reference\n(n={int(n_values[0]):,} cells)",
            pad=8,
            fontsize=10.8,
        )
        ax.spines[["top", "right"]].set_visible(False)
    reference_classes = {}
    for task, frame in task_frames.items():
        values = frame["reference_present_classes"].dropna().astype(str).unique()
        if len(values) != 1:
            raise RuntimeError(f"{task} Figure metrics have inconsistent reference classes")
        reference_classes[task] = (
            values[0].replace(" cells", "").replace("|", "/")
        )
    subtype_outside = task_frames["tcell_subtype"][
        "fraction_pred_outside_reference_label_space"
    ].astype(float)
    if subtype_outside.isna().any():
        raise RuntimeError("Subtype Figure metrics lack outside-label fractions")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    display_title = title.replace(": ", ":\n", 1)
    fig.suptitle(display_title, y=0.992, fontsize=11.5, fontweight="bold")
    fig.text(
        0.5,
        0.922,
        FIGURE_SCORING_SUBTITLE,
        ha="center",
        va="top",
        fontsize=8.0,
    )
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=2,
        fontsize=7.5,
    )
    footer = (
        "Reference classes: broad "
        f"{reference_classes['broad']}; subtype {reference_classes['tcell_subtype']}.\n"
        "Subtype outside-label calls retained as errors:\n"
        f"Marker only {subtype_outside['equal_weight']:.1%}; "
        f"ALE direction × importance {subtype_outside['direction_aware']:.1%}.\n"
        "Pooled-cell descriptive metrics; no patient-level inference."
    )
    fig.text(
        0.5,
        0.018,
        footer,
        ha="center",
        va="bottom",
        fontsize=6.8,
        linespacing=1.2,
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.79, bottom=0.19, hspace=0.48)
    try:
        return _save_review_figure_pair(fig, output_path)
    finally:
        plt.close(fig)


def save_three_arm_performance_pair(
    dataset: str,
    metrics: pd.DataFrame,
    output_path: Path,
) -> tuple[Path | None, Path | None]:
    """Render the pinned marker/sign/PFI compatibility ablation."""

    methods = list(THREE_ARM_METHOD_SPECS)
    labels = ["Marker\nonly", "Direction\nonly", "Direction +\nimportance"]
    frame_all = metrics.loc[
        metrics["ablation_contract"].eq("canonical_compatible")
        & metrics["dataset"].eq(dataset)
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=INDIVIDUAL_PERFORMANCE_CANVAS_INCHES,
        facecolor="white",
    )
    for ax, task, task_title in zip(
        axes,
        ["broad", "tcell_subtype"],
        ["Broad lineage", "T-cell subtype"],
    ):
        frame = frame_all.loc[frame_all["task"].eq(task)].set_index("method").reindex(methods)
        required = [
            "n_cells",
            "balanced_accuracy",
            "macro_f1_reference_present_classes",
            "fraction_pred_outside_reference_label_space",
        ]
        if frame[required].isna().any().any():
            raise RuntimeError(f"Incomplete three-arm Figure metrics for {dataset}/{task}")
        n_values = frame["n_cells"].astype(int).unique()
        if len(n_values) != 1:
            raise RuntimeError(f"Inconsistent three-arm denominator for {dataset}/{task}")
        x = np.arange(len(methods))
        score_metric_specs = [
            ("balanced_accuracy", "BAcc", "#4E79A7"),
            (
                "macro_f1_reference_present_classes",
                "Macro F1",
                "#E15759",
            ),
        ]
        for center_offset, (column, metric_label, color) in zip(
            SCRNA_SCORE_BAR_CENTER_OFFSETS,
            score_metric_specs,
        ):
            bars = ax.bar(
                x + center_offset,
                frame[column].to_numpy(dtype=float),
                SCRNA_SCORE_BAR_WIDTH,
                color=color,
                label=metric_label,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.018,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                )
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title(
            f"{task_title} vs scRNA reference\n(n={int(n_values[0]):,} cells)",
            pad=6,
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_ylabel("Score", fontsize=12.0)
        ax.tick_params(axis="x", labelsize=11.0)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        f"{FIGURE_DATASET_DISPLAY_LABELS[dataset]}\nMarker-signature ablation",
        y=0.985,
        fontsize=12.2,
        fontweight="bold",
    )
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        fontsize=10.0,
    )
    # Keep the established axes and legend geometry so the cleaned PNG and
    # native-shape PPTX remain the same canvas after explanatory prose is
    # removed from the individual performance panels.
    fig.subplots_adjust(left=0.19, right=0.98, top=0.82, bottom=0.075, hspace=0.32)
    try:
        return _save_review_figure_pair(fig, output_path)
    finally:
        plt.close(fig)


def _composite_label_handles(
    order: list[str],
    palette: dict[str, str],
    *,
    markersize: float = 4.2,
) -> list[plt.Line2D]:
    """Build compact label handles for the A4 composite legends."""

    handles = _categorical_legend_handles(
        order,
        palette,
        markersize=markersize,
    )
    for handle in handles:
        label = str(handle.get_label())
        words = label.split()
        if len(label) <= A4_UMAP_LEGEND_WRAP_WIDTH or len(words) < 2:
            continue
        split_at = min(
            range(1, len(words)),
            key=lambda index: max(
                len(" ".join(words[:index])),
                len(" ".join(words[index:])),
            ),
        )
        handle.set_label(
            " ".join(words[:split_at]) + "\n" + " ".join(words[split_at:])
        )
    return handles


def _read_a4_layout_reference() -> tuple[tuple[float, float], dict[str, object]]:
    """Read and code-pin the actual slide canvas of the layout reference PPTX."""

    reference = A4_LAYOUT_REFERENCE_PPTX.resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"A4 layout reference is missing: {reference}")
    with zipfile.ZipFile(reference, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"A4 layout reference PPTX is corrupt: {reference}")
        presentation_xml = archive.read("ppt/presentation.xml").decode("utf-8")
    match = re.search(
        r'<p:sldSz cx="(\d+)" cy="(\d+)"(?: type="([^"]+)")?',
        presentation_xml,
    )
    if match is None:
        raise RuntimeError(f"A4 layout reference slide size is missing: {reference}")
    slide_emu = (int(match.group(1)), int(match.group(2)))
    if slide_emu != A4_LAYOUT_REFERENCE_SLIDE_EMU:
        raise RuntimeError(
            "Layout reference PPTX slide size changed: "
            f"{slide_emu} != {A4_LAYOUT_REFERENCE_SLIDE_EMU}"
        )
    slide_type = match.group(3)
    if slide_type != "A4":
        raise RuntimeError(f"Layout reference PPTX is no longer typed as A4: {slide_type!r}")
    canvas_inches = tuple(value / 914400.0 for value in slide_emu)
    return (
        canvas_inches,
        {
            "path": str(reference),
            "slide_size_emu": list(slide_emu),
            "slide_size_inches": list(canvas_inches),
            "slide_type": slide_type,
            "orientation": "portrait",
            "sha256": _sha256(reference),
        },
    )


def save_scrna_a4_2x2_composite(
    *,
    dataset_payloads: dict[str, dict[str, object]],
    metrics: pd.DataFrame,
    canvas_inches: tuple[float, float],
    output_path: Path,
) -> tuple[Path | None, Path | None]:
    """Render the Final scRNA summary on the reference PPT A4 portrait canvas.

    Rows are GSE182434 then Roider. Each row places the score block beside the
    validation UMAP block at an outer width ratio of 0.9:2.1. The composite omits
    explanatory subtitle/footer prose and reuses only saved metrics,
    predictions, and UMAP coordinates.
    """

    dataset_order = ("GSE182434", "Roider")
    if set(dataset_payloads) != set(dataset_order):
        raise ValueError("A4 composite requires exact GSE182434 and Roider payloads")
    if len(canvas_inches) != 2 or canvas_inches[0] >= canvas_inches[1]:
        raise ValueError(f"A4 composite requires a portrait canvas: {canvas_inches}")
    methods = list(THREE_ARM_METHOD_SPECS)
    method_labels = ["Marker\nonly", "Direction\nonly", "Direction+\nimportance"]
    metric_specs = (
        ("balanced_accuracy", "BAcc", "#4E79A7"),
        (
            "macro_f1_reference_present_classes",
            "Macro F1",
            "#E15759",
        ),
    )
    fig = plt.figure(figsize=canvas_inches, facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=A4_SCORE_UMAP_WIDTH_RATIOS,
        height_ratios=[1, 1],
        left=0.072,
        right=0.978,
        top=0.988,
        bottom=0.045,
        wspace=A4_OUTER_WSPACE,
        hspace=A4_OUTER_HSPACE,
    )
    panel_letters = (("A", "B"), ("C", "D"))

    for row, dataset in enumerate(dataset_order):
        payload = dataset_payloads[dataset]
        obs = payload["obs"]
        umap = payload["umap"]
        predictions = payload["predictions"]
        if not isinstance(obs, pd.DataFrame) or not isinstance(
            predictions, pd.DataFrame
        ):
            raise TypeError(f"Invalid A4 composite table payload for {dataset}")
        umap = np.asarray(umap, dtype=float)
        if umap.shape != (len(obs), 2) or not predictions.index.equals(obs.index):
            raise RuntimeError(f"A4 composite row alignment changed for {dataset}")

        score_grid = outer[row, 0].subgridspec(
            5,
            1,
            height_ratios=A4_SCORE_GRID_HEIGHT_RATIOS,
            hspace=0.0,
        )
        score_header = fig.add_subplot(score_grid[0, 0])
        score_header.axis("off")
        score_header.text(
            0.0,
            A4_SCORE_HEADER_TITLE_Y,
            panel_letters[row][0],
            ha="left",
            va="center",
            fontsize=A4_TYPOGRAPHY_POINTS["panel_header"],
            fontweight="bold",
        )
        score_header.text(
            0.5,
            A4_SCORE_HEADER_TITLE_Y,
            f"{FIGURE_DATASET_DISPLAY_LABELS[dataset]} — Score",
            ha="center",
            va="center",
            fontsize=A4_TYPOGRAPHY_POINTS["panel_header"],
            fontweight="bold",
        )
        score_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
            for _, _, color in metric_specs
        ]
        score_header.legend(
            score_handles,
            [label for _, label, _ in metric_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, A4_SCORE_METRIC_LEGEND_ANCHOR_Y),
            frameon=False,
            ncol=2,
            borderaxespad=0,
            columnspacing=0.9,
            handlelength=1.2,
            handletextpad=0.35,
            fontsize=A4_TYPOGRAPHY_POINTS["score_metric_legend"],
        )
        score_frame = metrics.loc[
            metrics["ablation_contract"].eq("canonical_compatible")
            & metrics["dataset"].eq(dataset)
        ]
        for score_row, task, task_title in (
            (2, "broad", "Broad lineage"),
            (4, "tcell_subtype", "T-cell subtype"),
        ):
            ax = fig.add_subplot(score_grid[score_row, 0])
            frame = (
                score_frame.loc[score_frame["task"].eq(task)]
                .set_index("method")
                .reindex(methods)
            )
            required = [
                "n_cells",
                "balanced_accuracy",
                "macro_f1_reference_present_classes",
            ]
            if frame[required].isna().any().any():
                raise RuntimeError(f"Incomplete A4 score metrics for {dataset}/{task}")
            n_values = frame["n_cells"].astype(int).unique()
            if len(n_values) != 1:
                raise RuntimeError(f"Inconsistent A4 denominator for {dataset}/{task}")
            x = np.arange(len(methods))
            for center_offset, (column, _, color) in zip(
                A4_SCORE_BAR_CENTER_OFFSETS,
                metric_specs,
            ):
                bars = ax.bar(
                    x + center_offset,
                    frame[column].to_numpy(dtype=float),
                    A4_SCORE_BAR_WIDTH,
                    color=color,
                )
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.018,
                        f"{bar.get_height():.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=A4_TYPOGRAPHY_POINTS["score_value_annotation"],
                    )
            ax.set_xticks(
                x,
                method_labels,
                fontsize=A4_TYPOGRAPHY_POINTS["score_x_tick"],
            )
            ax.set_ylim(0, 1.10)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(
                axis="y",
                labelsize=A4_TYPOGRAPHY_POINTS["score_y_tick"],
            )
            ax.set_ylabel(
                "Score",
                fontsize=A4_TYPOGRAPHY_POINTS["score_y_label"],
                labelpad=1.0,
            )
            ax.set_title(
                task_title,
                loc="center",
                pad=3,
                fontsize=A4_TYPOGRAPHY_POINTS["score_task_title"],
                fontweight="bold",
            )
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        tasks = {task.task_id: task for task in _make_tasks(dataset, obs)}
        broad = tasks["broad"]
        tcell = tasks["tcell_subtype"]
        broad_mask = broad.truth.isin(broad.reference_labels).to_numpy(dtype=bool)
        tcell_mask = tcell.truth.isin(tcell.reference_labels).to_numpy(dtype=bool)
        metadata_col = str(DATASETS[dataset]["group_col"])
        metadata = obs[metadata_col].astype("string")
        celltype_col = (
            "reference_population_raw"
            if "reference_population_raw" in obs
            else broad.truth_col
        )
        celltype = obs[celltype_col].astype("string")
        metadata_order = sorted(metadata[broad_mask].dropna().unique())
        celltype_order = sorted(celltype[broad_mask].dropna().unique())
        metadata_colors = {
            value: plt.cm.tab20(index % 20)
            for index, value in enumerate(metadata_order)
        }
        celltype_colors = {
            value: plt.cm.tab20(index % 20)
            for index, value in enumerate(celltype_order)
        }
        umap_grid = outer[row, 1].subgridspec(
            4,
            5,
            height_ratios=[0.20, 1, 1, 1],
            width_ratios=[
                1,
                1,
                1,
                SCRNA_UMAP_COLORBAR_COLUMN_RATIO,
                A4_UMAP_LEGEND_COLUMN_RATIO,
            ],
            wspace=A4_UMAP_GRID_WSPACE,
            hspace=A4_UMAP_GRID_HSPACE,
        )
        umap_header = fig.add_subplot(umap_grid[0, :3])
        umap_header.axis("off")
        umap_header.text(
            0.0,
            0.70,
            panel_letters[row][1],
            ha="left",
            va="center",
            fontsize=A4_TYPOGRAPHY_POINTS["panel_header"],
            fontweight="bold",
        )
        umap_header.text(
            0.5,
            0.70,
            f"{FIGURE_DATASET_DISPLAY_LABELS[dataset]} — UMAP",
            ha="center",
            va="center",
            fontsize=A4_TYPOGRAPHY_POINTS["panel_header"],
            fontweight="bold",
        )
        umap_axes = [
            [fig.add_subplot(umap_grid[plot_row, column]) for column in range(3)]
            for plot_row in (1, 2, 3)
        ]
        broad_order = list(broad.model_labels)
        tcell_order = list(tcell.model_labels)
        _scatter_categorical(
            umap_axes[0][0],
            umap,
            metadata,
            broad_mask,
            metadata_order,
            metadata_colors,
            size=0.9,
        )
        _scatter_categorical(
            umap_axes[0][1],
            umap,
            celltype,
            broad_mask,
            celltype_order,
            celltype_colors,
            size=0.9,
        )
        hladr = pd.to_numeric(
            obs.get("codex_HLA-DR", pd.Series(0.0, index=obs.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        umap_axes[0][2].scatter(
            umap[:, 0],
            umap[:, 1],
            s=0.5,
            c="#E1E1E1",
            alpha=0.2,
            linewidths=0,
        )
        hladr_scatter = umap_axes[0][2].scatter(
            umap[broad_mask, 0],
            umap[broad_mask, 1],
            s=0.9,
            c=hladr[broad_mask],
            cmap="viridis",
            linewidths=0,
        )
        umap_axes[0][2].set_xticks([])
        umap_axes[0][2].set_yticks([])
        for spine in umap_axes[0][2].spines.values():
            spine.set_linewidth(0.8)
        colorbar_ax = fig.add_subplot(umap_grid[1, 3])
        colorbar = fig.colorbar(hladr_scatter, cax=colorbar_ax)
        colorbar.ax.tick_params(
            labelsize=A4_TYPOGRAPHY_POINTS["umap_colorbar_tick"],
            length=1.3,
        )

        _scatter_categorical(
            umap_axes[1][0],
            umap,
            broad.truth,
            broad_mask,
            broad_order,
            BROAD_PALETTE,
            size=0.9,
        )
        _scatter_categorical(
            umap_axes[1][1],
            umap,
            predictions["pred_broad_equal_weight"],
            broad_mask,
            broad_order,
            BROAD_PALETTE,
            size=0.9,
        )
        _scatter_categorical(
            umap_axes[1][2],
            umap,
            predictions["pred_broad_direction_aware"],
            broad_mask,
            broad_order,
            BROAD_PALETTE,
            size=0.9,
        )
        _scatter_categorical(
            umap_axes[2][0],
            umap,
            tcell.truth,
            tcell_mask,
            tcell_order,
            T_PALETTE,
            size=0.9,
        )
        _scatter_categorical(
            umap_axes[2][1],
            umap,
            predictions["pred_tcell_subtype_equal_weight"],
            tcell_mask,
            tcell_order,
            T_PALETTE,
            size=0.9,
        )
        _scatter_categorical(
            umap_axes[2][2],
            umap,
            predictions["pred_tcell_subtype_direction_aware"],
            tcell_mask,
            tcell_order,
            T_PALETTE,
            size=0.9,
        )
        for ax, title in zip(
            umap_axes[0],
            (metadata_col, "Reference\ncell type", "HLA-DR RNA\nproxy"),
        ):
            ax.set_title(
                title,
                fontsize=A4_TYPOGRAPHY_POINTS["umap_subplot_title"],
                pad=1.8,
                fontweight="bold",
                linespacing=0.86,
            )
        umap_titles = (
            ("Broad\nreference", "Marker\nonly", "Direction+\nimportance"),
            ("T-cell\nreference", "Marker\nonly", "Direction+\nimportance"),
        )
        for plot_row, titles in enumerate(umap_titles, start=1):
            for column, title in enumerate(titles):
                umap_axes[plot_row][column].set_title(
                    title,
                    fontsize=A4_TYPOGRAPHY_POINTS["umap_subplot_title"],
                    pad=1.8,
                    fontweight="bold",
                    linespacing=0.86,
                )
        legend_groups = [
            (
                _composite_label_handles(
                    metadata_order,
                    metadata_colors,
                    markersize=A4_UMAP_LEGEND_MARKER_SIZE,
                ),
                metadata_col,
            ),
            (
                _composite_label_handles(
                    celltype_order,
                    celltype_colors,
                    markersize=A4_UMAP_LEGEND_MARKER_SIZE,
                ),
                "Cell type",
            ),
            (
                _composite_label_handles(
                    broad_order,
                    BROAD_PALETTE,
                    markersize=A4_UMAP_LEGEND_MARKER_SIZE,
                ),
                "Broad",
            ),
            (
                _composite_label_handles(
                    tcell_order,
                    T_PALETTE,
                    markersize=A4_UMAP_LEGEND_MARKER_SIZE,
                ),
                "T-cell",
            ),
        ]
        _add_stacked_categorical_legends(
            fig,
            umap_grid[1:, 4],
            legend_groups,
            fontsize=A4_UMAP_LEGEND_LABEL_FONTSIZE,
            title_fontsize=A4_UMAP_LEGEND_TITLE_FONTSIZE,
            hspace=A4_UMAP_LEGEND_GROUP_HSPACE,
            labelspacing=A4_UMAP_LEGEND_LABELSPACING,
            handletextpad=A4_UMAP_LEGEND_HANDLETEXTPAD,
            alignment="left",
            anchor_x=A4_UMAP_LEGEND_LEFT_INSET,
        )

    try:
        return _save_review_figure_pair(
            fig,
            output_path,
            rasterize_dense_layers=True,
            dense_layer_point_threshold=100,
            dense_layer_mode="axis_local",
        )
    finally:
        plt.close(fig)


def _snapshot_artifact(path: Path) -> dict[str, object]:
    """Capture exact initial presence/bytes for transaction conflict detection."""

    path = Path(path).resolve()
    if not path.exists():
        return {"exists": False, "size_bytes": None, "sha256": None}
    if not path.is_file():
        raise RuntimeError(f"Transactional artifact target is not a file: {path}")
    return {
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _artifact_matches_snapshot(path: Path, snapshot: dict[str, object]) -> bool:
    path = Path(path).resolve()
    if path.exists() != bool(snapshot["exists"]):
        return False
    if not path.exists():
        return True
    return (
        path.is_file()
        and int(path.stat().st_size) == int(snapshot["size_bytes"])
        and _sha256(path) == snapshot["sha256"]
    )


def _promote_staged_artifact_set(
    promotions: list[tuple[Path, Path]],
    *,
    initial_target_states: dict[Path, dict[str, object]],
    failure_injector=None,
) -> None:
    """Promote an ordered artifact set and restore every initial target on failure.

    The caller supplies the render manifest last, so no completed manifest can
    become visible before every selected PNG/PPTX has been promoted.
    ``failure_injector`` is reserved for lightweight synthetic rollback tests.
    """

    if not promotions:
        raise ValueError("Transactional artifact promotion cannot be empty")
    normalized = [(Path(stage).resolve(), Path(target).resolve()) for stage, target in promotions]
    targets = [target for _, target in normalized]
    if len(set(targets)) != len(targets):
        raise ValueError("Transactional artifact targets are not unique")
    normalized_states = {
        Path(path).resolve(): state for path, state in initial_target_states.items()
    }
    if set(normalized_states) != set(targets):
        raise RuntimeError("Initial transaction snapshot does not match the target set")
    token = uuid.uuid4().hex
    states = []
    for stage, target in normalized:
        if not stage.is_file() or stage.stat().st_size <= 0:
            raise RuntimeError(f"Staged transaction artifact is missing or empty: {stage}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if stage.stat().st_dev != target.parent.stat().st_dev:
            raise RuntimeError(f"Staged artifact is not on the target filesystem: {stage}")
        backup = target.with_name(f".{target.name}.step12-{token}.backup")
        if backup.exists():
            raise FileExistsError(f"Step12 transaction backup already exists: {backup}")
        states.append(
            {
                "stage": stage,
                "target": target,
                "backup": backup,
                "initial": normalized_states[target],
                "staged": _snapshot_artifact(stage),
                "backed_up": False,
                "promoted": False,
            }
        )
    try:
        changed = [
            str(state["target"])
            for state in states
            if not _artifact_matches_snapshot(state["target"], state["initial"])
        ]
        if changed:
            raise RuntimeError(
                "Step12 canonical target changed during staging: " + ", ".join(changed)
            )
        for index, state in enumerate(states):
            if state["initial"]["exists"]:
                os.replace(state["target"], state["backup"])
                state["backed_up"] = True
            if failure_injector is not None:
                failure_injector("after_backup", index, state["target"])
        for index, state in enumerate(states):
            if state["target"].exists():
                raise FileExistsError(
                    f"Step12 target reappeared before promotion: {state['target']}"
                )
            os.replace(state["stage"], state["target"])
            state["promoted"] = True
            if not _artifact_matches_snapshot(state["target"], state["staged"]):
                raise RuntimeError(
                    f"Step12 promoted artifact differs from staging: {state['target']}"
                )
            if failure_injector is not None:
                failure_injector("after_promote", index, state["target"])
    except BaseException as promotion_error:
        rollback_errors = []
        for state in reversed(states):
            if state["promoted"] and state["target"].exists():
                try:
                    state["target"].unlink()
                except BaseException as exc:
                    rollback_errors.append(
                        f"remove promoted {state['target']}: {exc}"
                    )
        for state in states:
            if state["backup"].exists():
                try:
                    if state["target"].exists():
                        raise FileExistsError(
                            f"rollback target already exists: {state['target']}"
                        )
                    os.replace(state["backup"], state["target"])
                except BaseException as exc:
                    rollback_errors.append(
                        f"restore original {state['target']}: {exc}"
                    )
        for state in states:
            if not _artifact_matches_snapshot(state["target"], state["initial"]):
                rollback_errors.append(
                    f"restored target differs from initial state: {state['target']}"
                )
        if rollback_errors:
            raise RuntimeError(
                "Step12 promotion failed and rollback was incomplete; backups are preserved: "
                + "; ".join(rollback_errors)
            ) from promotion_error
        raise
    for state in states:
        if state["backup"].exists():
            try:
                state["backup"].unlink()
            except OSError as exc:
                warnings.warn(
                    f"Step12 committed backup cleanup failed: {state['backup']}: {exc}",
                    RuntimeWarning,
                )


def render_figures_from_completed_run(
    source_run_id: str,
    *,
    export_tag: str | None,
    datasets: list[str],
    arial_regular: Path,
    arial_bold: Path,
    overwrite_canonical: bool = False,
    performance_sensitivity_run_id: str = CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID,
) -> dict[str, object]:
    """Render Step12 from mixed immutable sources without recalculating metrics.

    UMAP panels use the completed two-arm Step11 predictions. Performance panels
    use the pinned Step11B marker/sign/PFI table. A canonical run regenerates the
    complete four-pair active family as one validated promotion unit.
    """

    if RUN_NAME.fullmatch(source_run_id or "") is None:
        raise ValueError(f"Unsafe completed source run id: {source_run_id!r}")
    if len(set(datasets)) != len(datasets) or not set(datasets).issubset(EXPECTED_DATASETS):
        raise ValueError(f"Unexpected Figure dataset selection: {datasets}")
    if overwrite_canonical:
        if source_run_id != CANONICAL_RUN_ID or export_tag is not None:
            raise ValueError(
                "Canonical Figure replacement requires the canonical Step11 run and no export tag"
            )
        if set(datasets) != set(EXPECTED_DATASETS):
            raise ValueError(
                "Canonical Step12 replacement requires the complete two-dataset Figure family"
            )
    elif RUN_NAME.fullmatch(export_tag or "") is None:
        raise ValueError(f"Unsafe Figure export tag: {export_tag!r}")

    source_root = (RUNS_ROOT / source_run_id).resolve()
    if source_root.parent != RUNS_ROOT.resolve():
        raise ValueError("Figure source escapes the Final fixed-signature run root")
    source_manifest_path = source_root / "manifests" / "run_manifest.json"
    source_contract = validate_fixed_signature_run(source_run_id, verify_artifacts=True)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    from Final_utils import (
        _final_figure_stem_registered,
        _validate_staged_final_png,
        _validate_staged_final_pptx,
    )

    figure_specs = [
        (dataset, figure_kind, FINAL_SCRNA_FIGURE_STEMS[(dataset, figure_kind)])
        for dataset in datasets
        for figure_kind in ("performance", "3x3_umap_summary")
    ]
    unregistered_stems = [
        stem
        for _, _, stem in figure_specs
        if not _final_figure_stem_registered(Path(f"{stem}.png"))
    ]
    if unregistered_stems:
        raise RuntimeError(
            "Step12 active Figure family contains unregistered stems: "
            + ", ".join(sorted(unregistered_stems))
        )
    active_keys = {
        (dataset, figure_kind) for dataset, figure_kind, _ in figure_specs
    }
    performance_keys = {
        key for key in active_keys if key[1] == "performance"
    }
    if overwrite_canonical and performance_keys:
        required_performance = {(dataset, "performance") for dataset in datasets}
        if performance_keys != required_performance:
            raise RuntimeError(
                "Canonical three-arm performance promotion requires both dataset pairs together"
            )
    sensitivity_contract = None
    three_arm_metrics = None
    if performance_keys:
        sensitivity_contract = validate_three_arm_sensitivity_run(
            performance_sensitivity_run_id,
            verify_artifacts=True,
        )
        three_arm_metrics = sensitivity_contract["pooled_metrics"]

    if overwrite_canonical:
        final_root = FINAL_ROOT / "figures" / "v2" / "scRNA"
        manifest_path = source_root / "manifests" / CANONICAL_THREE_ARM_FIGURE_MANIFEST
        active_targets = {
            (dataset, figure_kind): (
                final_root / dataset / f"{stem}.png",
                final_root / dataset / f"{stem}.pptx",
            )
            for dataset, figure_kind, stem in figure_specs
        }
        missing_active = [
            path for pair in active_targets.values() for path in pair if not path.is_file()
        ]
        if missing_active or not manifest_path.is_file():
            missing = [str(path) for path in missing_active]
            if not manifest_path.is_file():
                missing.append(str(manifest_path))
            raise FileNotFoundError(
                "Canonical Step12 replacement requires the existing active pairs "
                "and current manifest: " + ", ".join(missing)
            )
        transaction_token = uuid.uuid4().hex
        partial_root = final_root / f".step12-{transaction_token}.partial"
        if partial_root.parent.resolve() != final_root.resolve() or partial_root.exists():
            raise RuntimeError(f"Unsafe or occupied Step12 staging root: {partial_root}")
        canonical_targets = [
            path for pair in active_targets.values() for path in pair
        ] + [manifest_path]
        initial_target_states = {
            path.resolve(): _snapshot_artifact(path) for path in canonical_targets
        }
        partial_root.mkdir(parents=False, exist_ok=False)
    else:
        final_root = FIGURE_REEXPORT_ROOT / str(export_tag)
        partial_root = final_root.with_name(f".{export_tag}.partial")
        if final_root.exists() or partial_root.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing Figure-only export: {final_root}"
            )
        partial_root.mkdir(parents=True, exist_ok=False)
        manifest_path = final_root / "render_manifest.json"
        active_targets = {}
        initial_target_states = {}

    generated_paths: list[Path] = []
    input_hashes: dict[str, str] = {}
    try:
        font_family = _configure_font(arial_regular, arial_bold)
        for dataset in datasets:
            output_root = partial_root / dataset if overwrite_canonical else partial_root
            output_root.mkdir(parents=True, exist_ok=True)
            if (dataset, "3x3_umap_summary") in active_keys:
                print(
                    f"Figure-only reuse: loading read-only {dataset} metadata/UMAP...",
                    flush=True,
                )
                recorded_audit = source_manifest["large_inputs"][f"{dataset}_proxy_h5ad"]
                DATASETS[dataset]["h5ad"] = Path(recorded_audit["path"]).resolve()
                obs, umap, _, h5ad_audit = _load_dataset_obs(dataset)
                if h5ad_audit["sha256"] != recorded_audit["sha256"]:
                    raise RuntimeError(
                        f"{dataset} proxy h5ad no longer matches the completed run"
                    )
                input_hashes[dataset] = str(h5ad_audit["sha256"])
                predictions = pd.read_csv(
                    source_root / "tables" / f"{dataset}_cell_predictions.csv.gz",
                    dtype={"cell_id": str},
                ).set_index("cell_id")
                if not predictions.index.equals(obs.index):
                    raise RuntimeError(
                        f"{dataset} saved prediction rows do not match the immutable h5ad"
                    )
                umap_outputs = save_v9_umap_summary(
                    dataset=dataset,
                    obs=obs,
                    umap=umap,
                    predictions=predictions,
                    output_path=(
                        output_root
                        / f"{FINAL_SCRNA_FIGURE_STEMS[(dataset, '3x3_umap_summary')]}.png"
                    ),
                )
                generated_paths.extend(Path(path) for path in umap_outputs if path is not None)
            if (dataset, "performance") in active_keys:
                if three_arm_metrics is None:
                    raise RuntimeError("Three-arm performance metrics were not validated")
                performance_outputs = save_three_arm_performance_pair(
                    dataset,
                    three_arm_metrics,
                    output_root
                    / f"{FINAL_SCRNA_FIGURE_STEMS[(dataset, 'performance')]}.png",
                )
                generated_paths.extend(
                    Path(path) for path in performance_outputs if path is not None
                )

        expected_generated_count = 2 * len(active_keys)
        if len(generated_paths) != expected_generated_count:
            raise RuntimeError(
                f"Expected {expected_generated_count} active Figure files, "
                f"generated {len(generated_paths)}"
            )
        for path in generated_paths:
            if path.suffix.lower() == ".png":
                _validate_staged_final_png(path)
            elif path.suffix.lower() == ".pptx":
                _validate_staged_final_pptx(path)
            else:
                raise RuntimeError(f"Unexpected Step12 Figure artifact: {path}")

        if overwrite_canonical:
            expected_targets = {
                path.resolve() for pair in active_targets.values() for path in pair
            }
            staged_to_target = {
                path.resolve(): (final_root / path.relative_to(partial_root)).resolve()
                for path in generated_paths
            }
            if set(staged_to_target.values()) != expected_targets:
                raise RuntimeError(
                    "Staged Step12 Figure set differs from the complete active family"
                )
            generated_target_to_stage = {
                target: stage for stage, target in staged_to_target.items()
            }
            figure_artifacts: dict[str, dict[str, object]] = {}
            figure_sources: dict[str, dict[str, object]] = {}
            generated_targets: list[str] = []
            reused_targets: list[str] = []
            for key, pair in active_targets.items():
                dataset, figure_kind = key
                for target in pair:
                    resolved_target = target.resolve()
                    relative = _relative_to_final(resolved_target)
                    if resolved_target not in generated_target_to_stage:
                        raise RuntimeError(
                            f"Active canonical Figure was not staged: {resolved_target}"
                        )
                    artifact = generated_target_to_stage[resolved_target]
                    state = "generated"
                    generated_targets.append(str(resolved_target))
                    figure_artifacts[relative] = {
                        "size_bytes": int(artifact.stat().st_size),
                        "sha256": _sha256(artifact),
                        "state": state,
                    }
                    figure_sources[relative] = {
                        "dataset": dataset,
                        "figure_kind": figure_kind,
                        "state": state,
                        "source_run_id": (
                            source_run_id
                            if figure_kind == "3x3_umap_summary"
                            else performance_sensitivity_run_id
                        ),
                    }
            manifest_figure_paths = [
                str(path.resolve())
                for pair in active_targets.values()
                for path in pair
            ]
        else:
            staged_to_target = {}
            figure_artifacts = {
                path.name: {
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                    "state": "generated",
                }
                for path in generated_paths
            }
            figure_sources = {
                path.name: {
                    "state": "generated",
                    "source_run_id": (
                        performance_sensitivity_run_id
                        if "ablation_performance" in path.name
                        else source_run_id
                    ),
                }
                for path in generated_paths
            }
            generated_targets = [path.name for path in generated_paths]
            reused_targets = []
            manifest_figure_paths = [path.name for path in generated_paths]

        runtime_dependencies = {
            "umap_source_run_root": source_root,
            "umap_source_manifest": source_manifest_path,
        }
        if performance_keys and sensitivity_contract is not None:
            runtime_dependencies.update(
                {
                    "performance_sensitivity_run_root": Path(
                        str(sensitivity_contract["run_root"])
                    ),
                    "performance_sensitivity_manifest": Path(
                        str(sensitivity_contract["manifest_path"])
                    ),
                    "performance_pooled_metrics": Path(
                        str(sensitivity_contract["metrics_path"])
                    ),
                    "performance_pairwise_deltas": Path(
                        str(sensitivity_contract["deltas_path"])
                    ),
                    "performance_alias_audit": Path(
                        str(sensitivity_contract["alias_audit_path"])
                    ),
                }
            )
        for dataset, figure_kind in active_keys:
            if figure_kind == "3x3_umap_summary":
                runtime_dependencies[f"{dataset}_predictions"] = (
                    source_root / "tables" / f"{dataset}_cell_predictions.csv.gz"
                )
                runtime_dependencies[f"{dataset}_proxy_h5ad"] = Path(
                    DATASETS[dataset]["h5ad"]
                )
        active_runtime_provenance = _build_active_runtime_provenance(
            dependencies=runtime_dependencies,
            resolved_font_family=font_family,
            arial_regular=arial_regular,
            arial_bold=arial_bold,
        )
        source_files = {
            "umap_source_manifest": source_manifest_path,
            "umap_source_pooled_metrics": source_root / "tables" / "pooled_cell_metrics.csv",
        }
        if performance_keys and sensitivity_contract is not None:
            source_files.update(
                {
                    "performance_sensitivity_manifest": Path(
                        str(sensitivity_contract["manifest_path"])
                    ),
                    "performance_pooled_metrics": Path(
                        str(sensitivity_contract["metrics_path"])
                    ),
                    "performance_pairwise_deltas": Path(
                        str(sensitivity_contract["deltas_path"])
                    ),
                    "performance_alias_audit": Path(
                        str(sensitivity_contract["alias_audit_path"])
                    ),
                }
            )
        render_manifest = {
            "status": "figures_rendered_from_completed_results",
            "analysis_family": "fixed_signature_mixed_source_figure_export",
            "umap_source_run_id": source_run_id,
            "performance_sensitivity_run_id": performance_sensitivity_run_id,
            "performance_contract": "canonical_compatible",
            "source_run_root": str(source_root),
            "source_run_read_only": True,
            "source_run_tables_read_only": True,
            "source_manifest_status": source_manifest["status"],
            "source_semantic_contract_verified": source_contract[
                "semantic_contract_verified"
            ],
            "source_table_artifacts_verified": source_contract["artifacts_verified"],
            "performance_sensitivity_artifacts_verified": (
                sensitivity_contract["artifacts_verified"]
                if sensitivity_contract is not None
                else None
            ),
            "performance_endpoint_equivalence_verified": (
                sensitivity_contract["endpoint_equivalence_verified"]
                if sensitivity_contract is not None
                else None
            ),
            "performance_strict_stepwise_deltas_positive": (
                sensitivity_contract["strict_stepwise_deltas_positive"]
                if sensitivity_contract is not None
                else None
            ),
            "source_input_sha256_verified": input_hashes,
            "models_retrained": False,
            "scrna_fitting": False,
            "ale_recalculated": False,
            "permutation_importance_recalculated": False,
            "scores_recomputed": False,
            "metrics_recomputed": False,
            "analysis_tables_written": False,
            "umap_read": any(key[1] == "3x3_umap_summary" for key in active_keys),
            "umap_regenerated": any(
                key[1] == "3x3_umap_summary" for key in active_keys
            ),
            "umap_modified": any(key[1] == "3x3_umap_summary" for key in active_keys),
            "performance_figures_regenerated": bool(performance_keys),
            "canonical_figures_replaced": bool(overwrite_canonical),
            "canonical_figures_added": False,
            "font_family": font_family,
            "figure_pair_contract": (
                "same-canvas PNG plus strict PowerPoint-native editable PPTX; "
                "no raster/vector fallback deck"
            ),
            "display_contract": {
                "dataset_display_labels": dict(FIGURE_DATASET_DISPLAY_LABELS),
                "performance_methods": list(FIGURE_THREE_ARM_LABELS.values()),
                "performance_metrics": ["BAcc", "Macro F1"],
                "umap_direction_importance_alias": FIGURE_DIRECTION_IMPORTANCE_LABEL,
                "individual_performance_canvas_inches": list(
                    INDIVIDUAL_PERFORMANCE_CANVAS_INCHES
                ),
                "individual_umap_canvas_inches": list(INDIVIDUAL_UMAP_CANVAS_INCHES),
                "performance_explanatory_subtitle_removed": True,
                "performance_explanatory_footer_removed": True,
                "umap_explanatory_subtitle_removed": True,
                "umap_legends_use_item_proportional_stacked_axes": True,
                "umap_pptx_dense_layer_mode": "axis_local",
                "umap_pptx_full_slide_dense_overlay": False,
                "score_bar_width": SCRNA_SCORE_BAR_WIDTH,
                "score_bar_center_offsets": list(SCRNA_SCORE_BAR_CENTER_OFFSETS),
                "umap_colorbar_column_ratio": SCRNA_UMAP_COLORBAR_COLUMN_RATIO,
                "alignment_contract": {
                    "figure_and_subplot_titles": "centered over associated plot blocks",
                    "bar_value_annotations": "centered over bars",
                    "umap_axis_labels": "centered on the rendered 3x3 plot block",
                    "umap_colorbar": "dedicated column preserves equal UMAP plot widths",
                    "categorical_legend_blocks": "centered in dedicated legend axes",
                },
            },
            "active_runtime_provenance": active_runtime_provenance,
            "source_hashes": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in source_files.items()
            },
            "figure_artifacts": figure_artifacts,
            "figure_sources": figure_sources,
            "generated_figures": generated_targets,
            "reused_figures": reused_targets,
            "promotion_contract": {
                "complete_active_pairs_and_manifest_atomic": bool(overwrite_canonical),
                "manifest_promoted_last": bool(overwrite_canonical),
                "rollback_restores_initial_bytes_and_presence": bool(overwrite_canonical),
                "active_pairs_required_initially_absent": False,
                "active_pairs_required_initially_present": bool(overwrite_canonical),
            },
            "figures": manifest_figure_paths,
            "output_directory": str(final_root),
        }
        _validate_generated_manifest_provenance(render_manifest)
        if overwrite_canonical:
            staged_manifest_path = partial_root / "render_manifest.json"
            _write_exact_derived_json(
                staged_manifest_path,
                render_manifest,
                owner="Step12 staged mixed-source Figure render manifest",
            )
            promotions = sorted(
                ((stage, target) for stage, target in staged_to_target.items()),
                key=lambda pair: str(pair[1]),
            )
            promotions.append((staged_manifest_path, manifest_path))
            _promote_staged_artifact_set(
                promotions,
                initial_target_states=initial_target_states,
            )
            shutil.rmtree(partial_root)
        else:
            _write_exact_derived_json(
                partial_root / "render_manifest.json",
                render_manifest,
                owner="Step12 mixed-source Figure re-export manifest",
            )
            partial_root.rename(final_root)
            render_manifest["figures"] = [
                str(final_root / Path(path).name) for path in render_manifest["figures"]
            ]
    except BaseException:
        shutil.rmtree(partial_root, ignore_errors=True)
        raise
    return render_manifest


def render_a4_composite_from_completed_run(
    source_run_id: str = CANONICAL_RUN_ID,
    *,
    performance_sensitivity_run_id: str = CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID,
    arial_regular: Path,
    arial_bold: Path,
) -> dict[str, object]:
    """Replace the canonical A4 2x2 composite from saved Step11/Step11B results."""

    if source_run_id != CANONICAL_RUN_ID:
        raise ValueError("A4 composite accepts only the code-pinned Step11 run")
    if performance_sensitivity_run_id != CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID:
        raise ValueError("A4 composite accepts only the code-pinned Step11B run")

    source_root = (RUNS_ROOT / source_run_id).resolve()
    source_manifest_path = source_root / "manifests" / "run_manifest.json"
    source_contract = validate_fixed_signature_run(source_run_id, verify_artifacts=True)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    sensitivity_contract = validate_three_arm_sensitivity_run(
        performance_sensitivity_run_id,
        verify_artifacts=True,
    )
    canvas_inches, layout_reference = _read_a4_layout_reference()
    from Final_utils import (
        _final_figure_stem_registered,
        _validate_staged_final_png,
        _validate_staged_final_pptx,
    )

    if not _final_figure_stem_registered(
        Path(f"{FINAL_SCRNA_A4_COMPOSITE_STEM}.png")
    ):
        raise RuntimeError(
            "Step12 A4 composite stem is not registered: "
            f"{FINAL_SCRNA_A4_COMPOSITE_STEM}"
        )

    final_root = (FINAL_ROOT / "figures" / "v2" / "scRNA").resolve()
    target_png = final_root / f"{FINAL_SCRNA_A4_COMPOSITE_STEM}.png"
    target_pptx = final_root / f"{FINAL_SCRNA_A4_COMPOSITE_STEM}.pptx"
    manifest_path = source_root / "manifests" / CANONICAL_A4_COMPOSITE_MANIFEST
    targets = [target_png, target_pptx, manifest_path]
    missing_targets = [str(path) for path in targets if not path.is_file()]
    if missing_targets:
        raise FileNotFoundError(
            "Canonical A4 composite replacement requires the existing PNG/PPTX "
            "pair and current manifest: " + ", ".join(missing_targets)
        )
    initial_target_states = {
        path.resolve(): _snapshot_artifact(path) for path in targets
    }
    partial_root = final_root / f".step12-a4-{uuid.uuid4().hex}.partial"
    if partial_root.parent.resolve() != final_root or partial_root.exists():
        raise RuntimeError(f"Unsafe or occupied A4 staging root: {partial_root}")
    partial_root.mkdir(parents=False, exist_ok=False)

    dataset_payloads: dict[str, dict[str, object]] = {}
    input_hashes: dict[str, str] = {}
    runtime_dependencies: dict[str, Path] = {
        "umap_source_run_root": source_root,
        "umap_source_manifest": source_manifest_path,
        "performance_sensitivity_run_root": Path(
            str(sensitivity_contract["run_root"])
        ),
        "performance_sensitivity_manifest": Path(
            str(sensitivity_contract["manifest_path"])
        ),
        "performance_pooled_metrics": Path(
            str(sensitivity_contract["metrics_path"])
        ),
        "performance_pairwise_deltas": Path(
            str(sensitivity_contract["deltas_path"])
        ),
        "performance_alias_audit": Path(
            str(sensitivity_contract["alias_audit_path"])
        ),
        "a4_layout_reference_pptx": A4_LAYOUT_REFERENCE_PPTX,
    }
    try:
        font_family = _configure_font(arial_regular, arial_bold)
        for dataset in ("GSE182434", "Roider"):
            print(
                f"A4 Figure-only reuse: loading read-only {dataset} metadata/UMAP...",
                flush=True,
            )
            recorded_audit = source_manifest["large_inputs"][f"{dataset}_proxy_h5ad"]
            DATASETS[dataset]["h5ad"] = Path(recorded_audit["path"]).resolve()
            obs, umap, h5ad_path, h5ad_audit = _load_dataset_obs(dataset)
            if h5ad_audit["sha256"] != recorded_audit["sha256"]:
                raise RuntimeError(
                    f"{dataset} proxy h5ad no longer matches the completed run"
                )
            predictions_path = (
                source_root / "tables" / f"{dataset}_cell_predictions.csv.gz"
            )
            predictions = pd.read_csv(
                predictions_path,
                dtype={"cell_id": str},
            ).set_index("cell_id")
            if not predictions.index.equals(obs.index):
                raise RuntimeError(
                    f"{dataset} saved prediction rows do not match the immutable h5ad"
                )
            dataset_payloads[dataset] = {
                "obs": obs,
                "umap": umap,
                "predictions": predictions,
            }
            input_hashes[dataset] = str(h5ad_audit["sha256"])
            runtime_dependencies[f"{dataset}_predictions"] = predictions_path
            runtime_dependencies[f"{dataset}_proxy_h5ad"] = h5ad_path

        staged_outputs = save_scrna_a4_2x2_composite(
            dataset_payloads=dataset_payloads,
            metrics=sensitivity_contract["pooled_metrics"],
            canvas_inches=canvas_inches,
            output_path=partial_root / f"{FINAL_SCRNA_A4_COMPOSITE_STEM}.png",
        )
        generated_paths = [Path(path) for path in staged_outputs if path is not None]
        if len(generated_paths) != 2:
            raise RuntimeError(
                f"Expected one A4 PNG/PPTX pair, generated {len(generated_paths)} files"
            )
        staged_png = next(path for path in generated_paths if path.suffix == ".png")
        staged_pptx = next(path for path in generated_paths if path.suffix == ".pptx")
        _validate_staged_final_png(staged_png)
        _validate_staged_final_pptx(staged_pptx)
        staged_to_target = {
            staged_png.resolve(): target_png.resolve(),
            staged_pptx.resolve(): target_pptx.resolve(),
        }
        active_runtime_provenance = _build_active_runtime_provenance(
            dependencies=runtime_dependencies,
            resolved_font_family=font_family,
            arial_regular=arial_regular,
            arial_bold=arial_bold,
        )
        source_files = {
            "umap_source_manifest": source_manifest_path,
            "performance_sensitivity_manifest": Path(
                str(sensitivity_contract["manifest_path"])
            ),
            "performance_pooled_metrics": Path(
                str(sensitivity_contract["metrics_path"])
            ),
            "performance_pairwise_deltas": Path(
                str(sensitivity_contract["deltas_path"])
            ),
            "performance_alias_audit": Path(
                str(sensitivity_contract["alias_audit_path"])
            ),
            "a4_layout_reference_pptx": A4_LAYOUT_REFERENCE_PPTX,
        }
        figure_artifacts = {
            _relative_to_final(target): {
                "size_bytes": int(stage.stat().st_size),
                "sha256": _sha256(stage),
                "state": "generated",
            }
            for stage, target in staged_to_target.items()
        }
        render_manifest = {
            "status": "a4_composite_rendered_from_completed_results",
            "analysis_family": "fixed_signature_mixed_source_figure_export",
            "umap_source_run_id": source_run_id,
            "performance_sensitivity_run_id": performance_sensitivity_run_id,
            "performance_contract": "canonical_compatible",
            "source_run_root": str(source_root),
            "source_run_read_only": True,
            "source_run_tables_read_only": True,
            "source_semantic_contract_verified": source_contract[
                "semantic_contract_verified"
            ],
            "source_table_artifacts_verified": source_contract["artifacts_verified"],
            "performance_sensitivity_artifacts_verified": sensitivity_contract[
                "artifacts_verified"
            ],
            "performance_endpoint_equivalence_verified": sensitivity_contract[
                "endpoint_equivalence_verified"
            ],
            "source_input_sha256_verified": input_hashes,
            "models_retrained": False,
            "scrna_fitting": False,
            "ale_recalculated": False,
            "permutation_importance_recalculated": False,
            "scores_recomputed": False,
            "metrics_recomputed": False,
            "analysis_tables_written": False,
            "umap_coordinates_reused": True,
            "umap_coordinates_modified": False,
            "figure_regenerated": True,
            "canonical_figures_replaced": True,
            "canonical_figures_added": False,
            "font_family": font_family,
            "canvas_inches": list(canvas_inches),
            "canvas_orientation": "A4 portrait",
            "layout_reference": layout_reference,
            "outer_grid": {
                "rows": ["GSE182434", "Roider"],
                "columns": ["Score", "UMAP"],
                "score_to_umap_width_ratio": list(A4_SCORE_UMAP_WIDTH_RATIOS),
            },
            "display_contract": {
                "dataset_display_labels": dict(FIGURE_DATASET_DISPLAY_LABELS),
                "score_methods": list(FIGURE_THREE_ARM_LABELS.values()),
                "performance_metrics": ["BAcc", "Macro F1"],
                "umap_direction_importance_alias": FIGURE_DIRECTION_IMPORTANCE_LABEL,
                "umap_legends_use_item_proportional_stacked_axes": True,
                "umap_pptx_dense_layer_mode": "axis_local",
                "umap_pptx_full_slide_dense_overlay": False,
                "umap_legend_column_ratio": A4_UMAP_LEGEND_COLUMN_RATIO,
                "umap_legend_marker_size": A4_UMAP_LEGEND_MARKER_SIZE,
                "umap_legend_label_fontsize": A4_UMAP_LEGEND_LABEL_FONTSIZE,
                "umap_legend_title_fontsize": A4_UMAP_LEGEND_TITLE_FONTSIZE,
                "typography_points": dict(A4_TYPOGRAPHY_POINTS),
                "score_bar_width": A4_SCORE_BAR_WIDTH,
                "score_bar_center_offsets": list(A4_SCORE_BAR_CENTER_OFFSETS),
                "score_grid_height_ratios": list(A4_SCORE_GRID_HEIGHT_RATIOS),
                "score_header_title_y": A4_SCORE_HEADER_TITLE_Y,
                "score_metric_legend_anchor_y": A4_SCORE_METRIC_LEGEND_ANCHOR_Y,
                "outer_wspace": A4_OUTER_WSPACE,
                "outer_hspace": A4_OUTER_HSPACE,
                "umap_grid_wspace": A4_UMAP_GRID_WSPACE,
                "umap_grid_hspace": A4_UMAP_GRID_HSPACE,
                "umap_legend_group_hspace": A4_UMAP_LEGEND_GROUP_HSPACE,
                "umap_legend_labelspacing": A4_UMAP_LEGEND_LABELSPACING,
                "umap_legend_handletextpad": A4_UMAP_LEGEND_HANDLETEXTPAD,
                "umap_legend_alignment": "left",
                "umap_legend_wrap_width": A4_UMAP_LEGEND_WRAP_WIDTH,
                "umap_legend_left_inset": A4_UMAP_LEGEND_LEFT_INSET,
                "score_task_denominator_annotations_removed": True,
                "direction_importance_title_line_count": 2,
                "umap_colorbar_column_ratio": SCRNA_UMAP_COLORBAR_COLUMN_RATIO,
                "alignment_contract": {
                    "panel_letters": "left anchored in each header block",
                    "panel_titles": "centered over associated plot blocks",
                    "score_metric_legend": "centered over the score block",
                    "score_task_titles": "centered over score axes",
                    "bar_value_annotations": "centered over bars",
                    "umap_subplot_titles": "centered over UMAP axes",
                    "umap_colorbar": "dedicated column preserves equal UMAP plot widths",
                    "categorical_legend_blocks": (
                        "left aligned to one shared legend-column anchor"
                    ),
                },
                "explanatory_subtitle_removed": True,
                "explanatory_footer_removed": True,
            },
            "figure_pair_contract": (
                "same-canvas PNG plus PowerPoint-native editable PPTX; "
                "dense UMAP scatter layers are axis-local selectable pictures, "
                "never one full-slide background overlay"
            ),
            "active_runtime_provenance": active_runtime_provenance,
            "source_hashes": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in source_files.items()
            },
            "figure_artifacts": figure_artifacts,
            "figure_sources": {
                relative: {
                    "figure_kind": "A4_2x2_composite",
                    "umap_source_run_id": source_run_id,
                    "performance_sensitivity_run_id": performance_sensitivity_run_id,
                }
                for relative in figure_artifacts
            },
            "generated_figures": [str(target_png), str(target_pptx)],
            "promotion_contract": {
                "figure_pair_and_manifest_atomic": True,
                "manifest_promoted_last": True,
                "rollback_restores_initial_bytes_and_presence": True,
                "targets_required_initially_absent": False,
                "targets_required_initially_present": True,
            },
            "figures": [str(target_png), str(target_pptx)],
            "output_directory": str(final_root),
        }
        _validate_generated_manifest_provenance(render_manifest)
        staged_manifest_path = partial_root / CANONICAL_A4_COMPOSITE_MANIFEST
        _write_exact_derived_json(
            staged_manifest_path,
            render_manifest,
            owner="Step12 staged A4 composite render manifest",
        )
        promotions = sorted(
            staged_to_target.items(),
            key=lambda pair: str(pair[1]),
        )
        promotions.append((staged_manifest_path, manifest_path))
        _promote_staged_artifact_set(
            promotions,
            initial_target_states=initial_target_states,
        )
        shutil.rmtree(partial_root)
    except BaseException:
        shutil.rmtree(partial_root, ignore_errors=True)
        raise
    return render_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=False)
    parser.add_argument("--step02-run-id", default=DEFAULT_STEP02_RUN_ID)
    parser.add_argument("--step04b-run-id", default=DEFAULT_STEP04B_RUN_ID)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--write-figures", action="store_true")
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help=(
            "Explicitly authorize scoring/metrics and a new run. Without this flag, "
            "the backend only validates and reuses the pinned completed result."
        ),
    )
    parser.add_argument(
        "--replace-canonical-derived",
        action="store_true",
        help=(
            "Recompute only fixed-signature/Z-score/annotation/metric derived outputs "
            "from protected canonical proxy inputs, then transactionally replace the "
            "canonical derived table/manifest bundle. Raw preprocessing, model training, "
            "ALE, and proxy generation remain disabled."
        ),
    )
    parser.add_argument(
        "--render-completed-figures",
        action="store_true",
        help="Render Figures only from the pinned completed prediction/metric tables.",
    )
    parser.add_argument(
        "--reuse-run-id",
        default=CANONICAL_RUN_ID,
        help="Explicit pinned completed source run; latest-run lookup is forbidden.",
    )
    parser.add_argument(
        "--figure-export-tag",
        help="Required new tag below figure_reexports/ for Figure-only rendering.",
    )
    parser.add_argument(
        "--roider-input-manifest",
        type=Path,
        help="Explicit completed Step09 handoff manifest for a new run.",
    )
    parser.add_argument(
        "--gse182434-input-manifest",
        type=Path,
        help="Explicit completed Step10 handoff manifest for a new run.",
    )
    parser.add_argument(
        "--roider-feature-h5ad",
        type=Path,
        help="Deprecated compatibility check; the Step09 manifest is authoritative.",
    )
    parser.add_argument(
        "--gse182434-feature-h5ad",
        type=Path,
        help="Deprecated compatibility check; the Step10 manifest is authoritative.",
    )
    parser.add_argument(
        "--roider-coverage",
        type=Path,
        help="Deprecated compatibility check; the Step09 manifest is authoritative.",
    )
    parser.add_argument(
        "--gse182434-coverage",
        type=Path,
        help="Deprecated compatibility check; the Step10 manifest is authoritative.",
    )
    parser.add_argument(
        "--arial-regular",
        type=Path,
        default=Path("__FONT_ROOT__/arial.ttf"),
    )
    parser.add_argument(
        "--arial-bold",
        type=Path,
        default=Path("__FONT_ROOT__/arialbd.ttf"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.force_recalculate and args.replace_canonical_derived:
        raise ValueError(
            "Choose either a new isolated run or canonical derived replacement, not both"
        )
    if args.replace_canonical_derived and args.write_figures:
        raise ValueError(
            "Canonical derived tables are committed in Step11; Step12 renders Figures "
            "from that validated generation"
        )
    if args.render_completed_figures:
        if (
            args.force_recalculate
            or args.replace_canonical_derived
            or args.preflight_only
            or args.run_id
        ):
            raise ValueError(
                "Figure-only reuse cannot be combined with recalculation/preflight flags"
            )
        if not args.figure_export_tag:
            raise ValueError("--figure-export-tag is required for Figure-only rendering")
        report = render_figures_from_completed_run(
            args.reuse_run_id,
            export_tag=args.figure_export_tag,
            datasets=list(args.datasets),
            arial_regular=args.arial_regular,
            arial_bold=args.arial_bold,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if (
        not args.force_recalculate
        and not args.replace_canonical_derived
        and not args.preflight_only
    ):
        if args.run_id or args.write_figures or args.figure_export_tag:
            raise ValueError(
                "Heavy execution options require --force-recalculate; Figure-only "
                "export requires --render-completed-figures"
            )
        report = validate_completed_run(verify_input_hashes=True)
        report["status"] = "verified_existing_figures_reused"
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    handoff_reports: dict[str, dict[str, object]] = {}
    if args.force_recalculate and not args.run_id:
        raise ValueError("--force-recalculate requires an explicit new --run-id")
    if args.replace_canonical_derived:
        if args.run_id not in (None, CANONICAL_RUN_ID):
            raise ValueError(
                "--replace-canonical-derived only accepts the canonical run ID"
            )
        if set(args.datasets) != set(EXPECTED_DATASETS):
            raise ValueError(
                "Canonical derived replacement requires both scRNA datasets"
            )
    if args.force_recalculate:
        require_new_run_id(args.run_id)
        if set(args.datasets) != set(EXPECTED_DATASETS):
            raise ValueError("A new fixed-signature run requires both Step09/10 datasets")
        manifest_paths = {
            "Roider": args.roider_input_manifest,
            "GSE182434": args.gse182434_input_manifest,
        }
        if any(path is None for path in manifest_paths.values()):
            raise ValueError("A new run requires both completed Step09/10 handoff manifests")
        handoff_reports = validate_prepared_input_manifests(
            args.run_id,
            {dataset: Path(path) for dataset, path in manifest_paths.items()},
            verify_hashes=True,
        )
        compatibility_paths = {
            "Roider": (args.roider_feature_h5ad, args.roider_coverage),
            "GSE182434": (args.gse182434_feature_h5ad, args.gse182434_coverage),
        }
        for dataset, report in handoff_reports.items():
            feature_override, coverage_override = compatibility_paths[dataset]
            if feature_override is not None and Path(feature_override).resolve() != report["feature_h5ad"]:
                raise RuntimeError(f"{dataset} feature override differs from its handoff manifest")
            if coverage_override is not None and Path(coverage_override).resolve() != report["coverage_csv"]:
                raise RuntimeError(f"{dataset} coverage override differs from its handoff manifest")
            DATASETS[dataset]["h5ad"] = report["feature_h5ad"]
    if args.force_recalculate or args.replace_canonical_derived or args.preflight_only:
        validate_final_source_inputs()
    resolver = _load_resolver()
    if args.force_recalculate:
        for dataset, report in handoff_reports.items():
            resolver.COVERAGE_PATHS[dataset] = report["coverage_csv"]
    direction_label = "patient-equal first-order ALE q10-to-q90 contrast"
    direction_short_label = "DLBCL ALE q10-to-q90"
    figure_comparison_title = FIGURE_SCORING_TITLE
    step02_loader = resolver.load_ale_directional_step02_broad_class_weights
    step04b_loader = resolver.load_ale_directional_step04b_class_weights
    METHOD_SPECS["direction_aware"]["display"] = (
        "Marker + ALE direction × importance"
    )
    print(
        f"Resolving pinned completed Step02 broad importance + {direction_short_label}...",
        flush=True,
    )
    step02, broad_weights = step02_loader(args.step02_run_id)
    broad_support = resolver.build_step02_broad_directional_signature_support(
        broad_weights
    )
    print(
        f"Resolving pinned completed Step04B T-cell importance + {direction_short_label}...",
        flush=True,
    )
    step04b, subtype_weights = step04b_loader(args.step04b_run_id)
    if (
        int(args.random_seed) != int(step02.manifest["random_seed"])
        or int(args.random_seed) != int(step04b.manifest["random_seed"])
    ):
        raise RuntimeError(
            "Requested provenance seed does not match both pinned upstream runs: "
            f"requested={args.random_seed}, Step02={step02.manifest['random_seed']}, "
            f"Step04B={step04b.manifest['random_seed']}"
        )
    tcell_support = resolver.build_step04b_tcell_directional_signature_support(
        subtype_weights
    )
    support = pd.concat([broad_support, tcell_support], ignore_index=True, sort=False)
    scoring_feature_names = (
        support.sort_values(["feature_position", "feature"], kind="mergesort")["feature"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    scoring_feature_order_sha256 = _sequence_sha256(scoring_feature_names)
    if args.force_recalculate:
        for dataset, report in handoff_reports.items():
            if report["signature_feature_order_sha256"] != scoring_feature_order_sha256:
                raise RuntimeError(
                    f"{dataset} Step09/10 feature order differs from Step11 pinned support"
                )
    expected_support = {
        "broad": set(TASK_CLASS_ORDER["broad"]),
        "tcell_subtype": set(TASK_CLASS_ORDER["tcell_subtype"]),
    }
    for task, expected_classes in expected_support.items():
        observed = set(support.loc[support["task"].eq(task), "class"].astype(str))
        if observed != expected_classes:
            raise RuntimeError(
                f"Canonical support mismatch for {task}: {sorted(observed)} != "
                f"{sorted(expected_classes)}"
            )
    direction_sensitivity = {
        task: {
            "stable_support_rows": int(len(frame)),
            "positive_direction_rows": int(
                frame["secondary_stable_direction_sign"].eq(1).sum()
            ),
            "negative_direction_rows": int(
                frame["secondary_stable_direction_sign"].eq(-1).sum()
            ),
            "primary_secondary_agreement_sensitivity_rows": int(
                frame["primary_secondary_agreement_sensitivity_gate"].sum()
            ),
            "agreement_used_for_selection": False,
        }
        for task, frame in support.groupby("task", sort=True)
    }

    source_artifacts = {
        "step02_manifest": step02.root / "manifests" / "run_manifest.json",
        "step02_joint_importance": step02.root
        / "tables"
        / "class_feature_joint_importance_summary.csv",
        "step02_paired_frequency": step02.root
        / "tables"
        / "paired_iteration_top20_attribution_frequency.csv",
        "step02_paired_reconciliation": step02.root
        / "tables"
        / "paired_iteration_vs_ensemble_reconciliation.csv",
        "step04b_manifest": step04b.root / "manifests" / "run_manifest.json",
        "step04b_joint_importance": step04b.root
        / "tables"
        / "class_feature_joint_importance_summary.csv",
        "step04b_paired_frequency": step04b.root
        / "tables"
        / "paired_iteration_top20_attribution_frequency.csv",
        "step04b_paired_reconciliation": step04b.root
        / "tables"
        / "paired_iteration_vs_ensemble_reconciliation.csv",
        "resolver": RESOLVER_PATH,
        "resolver_core": RESOLVER_CORE_PATH,
        "backend": Path(__file__).resolve(),
    }
    source_artifacts.update(
        {
            "step02_ale_manifest": resolver.STEP02_ALE_ROOT
            / "manifests"
            / "run_manifest.json",
            "step02_ale_recovery_manifest": resolver.STEP02_ALE_ROOT
            / "manifests"
            / "finalization_recovery_manifest.json",
            "step02_ale_summary": resolver.STEP02_ALE_ROOT
            / "tables"
            / "class_feature_marker_response_summary.csv",
            "step04b_ale_manifest": resolver.STEP04B_ALE_ROOT
            / "manifests"
            / "run_manifest.json",
            "step04b_ale_summary": resolver.STEP04B_ALE_ROOT
            / "tables"
            / "class_feature_marker_response_summary.csv",
        }
    )
    notebook_path = (
        FINAL_ROOT / "notebooks" / "Final_step11_scRNA_03_direct_transfer_validation.ipynb"
    )
    for dataset in args.datasets:
        source_artifacts[f"{dataset}_coverage"] = resolver.COVERAGE_PATHS[dataset]
    source_hashes_before = {
        name: {"path": str(path), "sha256": _sha256(Path(path))}
        for name, path in source_artifacts.items()
    }

    dataset_inputs: dict[str, dict[str, object]] = {}
    for dataset in args.datasets:
        print(f"Loading read-only {dataset} proxy input and SHA-256...", flush=True)
        proxy, audit = resolver.collapse_directional_signature_support_to_proxy_groups(
            support,
            resolver.COVERAGE_PATHS[dataset],
            dataset,
        )
        obs, umap, h5ad_path, h5ad_audit = _load_dataset_obs(dataset)
        if args.force_recalculate:
            handoff = handoff_reports[dataset]
            if int(len(obs)) != int(handoff["n_obs"]):
                raise RuntimeError(f"{dataset} handoff observation count changed")
            if list(umap.shape) != list(handoff["umap_shape"]):
                raise RuntimeError(f"{dataset} handoff UMAP shape changed")
            if _sequence_sha256(obs.index.astype(str)) != handoff["ordered_cell_id_sha256"]:
                raise RuntimeError(f"{dataset} handoff cell order changed")
            if _truth_semantic_sha256(obs) != handoff["truth_semantic_sha256"]:
                raise RuntimeError(f"{dataset} handoff truth semantics changed")
        required_cols = {
            value
            for joined in proxy["feature_cols"].astype(str)
            for value in joined.split("|")
            if value
        }
        missing = sorted(required_cols - set(obs.columns))
        if missing:
            raise KeyError(f"{dataset} is missing required proxy columns: {missing}")
        _proxy_zscores(obs, proxy)
        tasks = _make_tasks(dataset, obs)
        task_counts = {
            task.task_id: int(task.truth.isin(task.reference_labels).sum())
            for task in tasks
        }
        task_mask_hashes = {
            task.task_id: _evaluation_mask_sha256(
                obs.index[task.truth.isin(task.reference_labels)]
            )
            for task in tasks
        }
        task_semantic_hashes = {
            task.task_id: _evaluation_semantic_sha256(
                dataset,
                task.task_id,
                obs.index,
                task.truth,
                task.truth.isin(task.reference_labels),
            )
            for task in tasks
        }
        dataset_inputs[dataset] = {
            "proxy": proxy,
            "audit": audit,
            "obs": obs,
            "umap": umap,
            "h5ad": h5ad_path,
            "h5ad_audit": h5ad_audit,
            "task_counts": task_counts,
            "task_mask_hashes": task_mask_hashes,
            "task_semantic_hashes": task_semantic_hashes,
        }
    preflight = {
        "status": "ready",
        "step02_run_id": step02.run_id,
        "step02_status": step02.manifest["status"],
        "step02_input_unchanged": step02.manifest["input_unchanged"],
        "step04b_run_id": step04b.run_id,
        "step04b_status": step04b.manifest["status"],
        "step04b_input_unchanged": step04b.manifest["input_unchanged"],
        "selected_signature_features": {
            "Step02_broad": int(len(broad_support)),
            "Step04B_tcell_subtype": int(len(tcell_support)),
        },
        "classes": {
            task: sorted(classes) for task, classes in expected_support.items()
        },
        "datasets": {
            dataset: {
                "cells": int(len(payload["obs"])),
                "proxy_rows": int(len(payload["proxy"])),
                "evaluable_task_cells": payload["task_counts"],
                "evaluation_mask_sha256": payload["task_mask_hashes"],
                "evaluation_semantic_sha256": payload["task_semantic_hashes"],
            }
            for dataset, payload in dataset_inputs.items()
        },
        "broad_lineage_direction_aware_validation_available": True,
        "tcell_subtype_direction_aware_validation_available": True,
        "direction_consistency_threshold": 0.8,
        "direction_source": "dlbcl_ale",
        "direction_metric": direction_label,
        "direction_scope": (
            "all routed DLBCL cells; patient-equal local effects within each frozen pair"
        ),
        "direction_sensitivity": direction_sensitivity,
        "primary_secondary_agreement_used_for_selection": False,
        "positive_top20_frequency_role": "metadata_only",
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if args.replace_canonical_derived:
        run_id = CANONICAL_RUN_ID
        final_root, partial_root = _canonical_derived_staging_root()
    else:
        run_id = args.run_id or datetime.now().strftime(
            "step02_step04b_dlbcl_ale_signature_scrna_seed42_%Y%m%d_%H%M%S"
        )
        final_root, partial_root = _safe_run_root(run_id)
    tables_root = partial_root / "tables"
    figures_root = partial_root / "figures"
    manifests_root = partial_root / "manifests"
    code_root = partial_root / "code"
    required_roots = [tables_root, manifests_root]
    if not args.replace_canonical_derived:
        required_roots.extend([figures_root, code_root])
    for path in required_roots:
        path.mkdir(parents=True, exist_ok=False)
    if not args.replace_canonical_derived:
        shutil.copy2(Path(__file__).resolve(), code_root / Path(__file__).name)
        shutil.copy2(RESOLVER_PATH, code_root / RESOLVER_PATH.name)
        shutil.copy2(RESOLVER_CORE_PATH, code_root / RESOLVER_CORE_PATH.name)
        if notebook_path.is_file():
            shutil.copy2(notebook_path, code_root / notebook_path.name)
    _write_staged_derived_csv(
        broad_support,
        tables_root / "step02_broad_directional_signature_features.csv",
        tables_root=tables_root,
        index=False,
    )
    _write_staged_derived_csv(
        tcell_support,
        tables_root / "step04b_tcell_directional_signature_features.csv",
        tables_root=tables_root,
        index=False,
    )
    _write_staged_derived_csv(
        support,
        tables_root / "combined_fixed_class_direction_signature_features.csv",
        tables_root=tables_root,
        index=False,
    )
    all_metrics: list[pd.DataFrame] = []
    all_group_metrics: list[pd.DataFrame] = []
    large_inputs: dict[str, dict[str, object]] = {}
    resolved_figure_font: str | None = None
    for dataset, payload in dataset_inputs.items():
        print(f"Scoring {dataset}: Broad and T-cell subtype...", flush=True)
        proxy = payload["proxy"]
        audit = payload["audit"]
        obs = payload["obs"]
        umap = payload["umap"]
        predictions, score_long = score_dataset(obs, proxy)
        predictions.insert(0, "cell_id", obs.index)
        prediction_export = predictions.copy()
        for column in [
            "reference_population_raw",
            "reference_broad_codex",
            "reference_tcell_subtype",
        ]:
            if column in obs:
                prediction_export[column] = obs[column].astype("string").to_numpy()
        prediction_export["UMAP1"] = umap[:, 0]
        prediction_export["UMAP2"] = umap[:, 1]
        _write_staged_derived_csv(
            proxy,
            tables_root / f"{dataset}_signature_proxy_weights.csv",
            tables_root=tables_root,
            index=False,
        )
        _write_staged_derived_csv(
            audit,
            tables_root / f"{dataset}_signature_proxy_coverage_audit.csv",
            tables_root=tables_root,
            index=False,
        )
        _write_staged_derived_csv(
            prediction_export,
            tables_root / f"{dataset}_cell_predictions.csv.gz",
            tables_root=tables_root,
            index=False,
            compression="gzip",
        )
        _write_staged_derived_csv(
            score_long,
            tables_root / f"{dataset}_class_scores.csv.gz",
            tables_root=tables_root,
            index=False,
            compression="gzip",
        )
        mask_rows: list[pd.DataFrame] = []
        for task in _make_tasks(dataset, obs):
            included = task.truth.isin(task.reference_labels)
            mask_rows.append(
                pd.DataFrame(
                    {
                        "cell_id": obs.index,
                        "task": task.task_id,
                        "truth_col": task.truth_col,
                        "reference_truth": task.truth,
                        "included_in_both_methods": included.to_numpy(dtype=bool),
                        "lineage_gate": _lineage_gate(task.task_id),
                        "evaluation_mask_sha256": payload["task_mask_hashes"][task.task_id],
                        "evaluation_semantic_sha256": payload["task_semantic_hashes"][task.task_id],
                    }
                )
            )
        _write_staged_derived_csv(
            pd.concat(mask_rows, ignore_index=True),
            tables_root / f"{dataset}_evaluation_mask_audit.csv.gz",
            tables_root=tables_root,
            index=False,
            compression="gzip",
        )
        metrics, group_metrics, confusions, reports = evaluate_dataset(
            dataset,
            obs,
            predictions.set_index("cell_id"),
            str(DATASETS[dataset]["group_col"]),
        )
        denominator_audit = metrics.groupby(["dataset", "task"]).agg(
            mask_hashes=("evaluation_mask_sha256", "nunique"),
            semantic_hashes=("evaluation_semantic_sha256", "nunique"),
            cell_denominators=("n_cells", "nunique"),
            methods=("method", "nunique"),
        )
        if not (
            denominator_audit["mask_hashes"].eq(1).all()
            and denominator_audit["semantic_hashes"].eq(1).all()
            and denominator_audit["cell_denominators"].eq(1).all()
            and denominator_audit["methods"].eq(2).all()
        ):
            raise RuntimeError(
                "Marker-only and marker+ALE-direction/importance denominators "
                f"diverged for {dataset}"
            )
        all_metrics.append(metrics)
        all_group_metrics.append(group_metrics)
        for key, table in confusions.items():
            _write_staged_derived_csv(
                table,
                tables_root / f"{key}_confusion_matrix.csv",
                tables_root=tables_root,
            )
        for key, table in reports.items():
            _write_staged_derived_csv(
                table,
                tables_root / f"{key}_classification_report.csv",
                tables_root=tables_root,
            )
        if args.write_figures:
            current_font = _configure_font(
                args.arial_regular, args.arial_bold
            )
            if resolved_figure_font is None:
                resolved_figure_font = current_font
            elif resolved_figure_font != current_font:
                raise RuntimeError(
                    "Resolved Figure font changed between scRNA datasets"
                )
            save_v9_umap_summary(
                dataset=dataset,
                obs=obs,
                umap=umap,
                predictions=predictions.set_index("cell_id"),
                output_path=figures_root
                / f"{FINAL_SCRNA_FIGURE_STEMS[(dataset, '3x3_umap_summary')]}.png",
            )
            save_v9_performance_pair(
                metrics,
                figures_root
                / f"{LEGACY_TWO_ARM_PERFORMANCE_STEMS[dataset]}.png",
                f"{FIGURE_DATASET_DISPLAY_LABELS[dataset]}: {figure_comparison_title}",
            )
        large_inputs[f"{dataset}_proxy_h5ad"] = payload["h5ad_audit"]
    pooled_metrics = pd.concat(all_metrics, ignore_index=True)
    grouped_metrics = pd.concat(all_group_metrics, ignore_index=True)
    _write_staged_derived_csv(
        pooled_metrics,
        tables_root / "pooled_cell_metrics.csv",
        tables_root=tables_root,
        index=False,
    )
    _write_staged_derived_csv(
        grouped_metrics,
        tables_root / "sample_patient_metrics.csv",
        tables_root=tables_root,
        index=False,
    )
    source_hashes_after = {
        name: {"path": str(path), "sha256": _sha256(Path(path))}
        for name, path in source_artifacts.items()
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("A pinned small source artifact changed during scRNA validation")
    for dataset, payload in dataset_inputs.items():
        current = Path(payload["h5ad"]).stat()
        audit = payload["h5ad_audit"]
        if (
            int(current.st_size) != int(audit["size_bytes_before"])
            or int(current.st_mtime_ns) != int(audit["mtime_ns_before"])
        ):
            raise RuntimeError(f"Read-only {dataset} h5ad changed during validation")
    input_handoffs = {
        dataset: {
            "manifest_path_relative_to_final": _relative_to_final(report["manifest_path"]),
            "manifest_sha256": report["manifest_sha256"],
            "feature_h5ad_sha256": report["feature_h5ad_sha256"],
            "coverage_csv_sha256": report["coverage_csv_sha256"],
            "signature_feature_order_sha256": report[
                "signature_feature_order_sha256"
            ],
        }
        for dataset, report in handoff_reports.items()
    }
    active_runtime_dependencies = {
        **source_artifacts,
        "source_inputs_copy_manifest": (
            FINAL_ROOT
            / "tables"
            / "scRNA"
            / "fixed_signature"
            / "source_inputs"
            / "source_inputs_copy_manifest.json"
        ),
        **{
            f"{dataset}_proxy_h5ad": Path(payload["h5ad"])
            for dataset, payload in dataset_inputs.items()
        },
    }
    active_runtime_provenance = _build_active_runtime_provenance(
        dependencies=active_runtime_dependencies,
        resolved_font_family=resolved_figure_font,
        arial_regular=args.arial_regular,
        arial_bold=args.arial_bold,
    )
    manifest = {
        **preflight,
        "status": "complete",
        "run_id": run_id,
        "run_contract_version": RUN_CONTRACT_VERSION,
        "canonical_derived_overwrite": bool(args.replace_canonical_derived),
        "canonical_derived_contract_version": (
            CANONICAL_DERIVED_CONTRACT_VERSION
            if args.replace_canonical_derived
            else None
        ),
        "protected_upstream_reused": True,
        "raw_preprocessing_executed": False,
        "proxy_features_recomputed": False,
        "analysis_family": ANALYSIS_FAMILY,
        "created_at": datetime.now().astimezone().isoformat(),
        "random_seed": int(args.random_seed),
        "upstream_step02_seed": int(step02.manifest["random_seed"]),
        "upstream_step04b_seed": int(step04b.manifest["random_seed"]),
        "upstream_iterations": list(range(1, 101)),
        "upstream_top_k": 20,
        "scoring_randomness": False,
        "scoring_seed": None,
        "analysis_type": (
            "cross-modal frozen marker-only versus DLBCL-pinned permutation-importance "
            f"plus {direction_label} signature validation"
        ),
        "primary_methods": [
            {
                "method_id": method_id,
                "analysis_label": method_spec["analysis_label"],
                "display": method_spec["display"],
            }
            for method_id, method_spec in METHOD_SPECS.items()
        ],
        "models_retrained": False,
        "classifier_objects_transferred_to_scRNA": False,
        "scRNA_labels_used_for_training": False,
        "source_artifacts_read_only": True,
        "source_hashes_verified_unchanged": True,
        "large_input_sha256_recorded": True,
        "step02_broad_signature_support_contract": (
            f"{direction_label} direction consistent in >=0.8 of frozen pairs/support; "
            "DLBCL class_normalized_importance supplies magnitude"
        ),
        "step04b_tcell_signature_support_contract": (
            f"{direction_label} direction consistent in >=0.8 of frozen pairs/support; "
            "DLBCL class_normalized_importance supplies magnitude"
        ),
        "equal_weight_contract": (
            "one positive equal vote per unique covered proxy on identical stable support"
        ),
        "direction_aware_contract": (
            f"DLBCL-pinned {direction_label} sign times DLBCL class_normalized_importance "
            "magnitude; signed weights normalized by sum(abs(weight)) after proxy coverage"
        ),
        "tcell_subtype_lineage_gate": "reference_broad_codex == 'T cells'",
        "generic_rank_contract": (
            "saved rank_within_class is specificity_rank_within_class; an explicit "
            "magnitude_rank_within_class is exported"
        ),
        "paired_frequency_role": "metadata only; never multiplied into score",
        "primary_secondary_agreement_role": (
            "for ALE runs, compatibility alias for pair/support direction concordance; "
            "not an independent significance measure"
        ),
        "evaluation_truth_columns": {
            "broad": "reference_broad_codex",
            "tcell_subtype": "reference_tcell_subtype",
        },
        "evaluation_mask_contract": "one fixed reference-derived mask per dataset/task shared by both methods",
        "primary_metric_contract": (
            "BAcc and Macro F1 over reference-present truth classes"
        ),
        "secondary_metric_contract": "model-label-space Macro F1 is diagnostic only",
        "outside_predictions_dropped": False,
        "figure_composition": (
            "dataset-specific V9-style 3x3 UMAP summary plus two-panel performance comparison"
        ),
        "figure_font_contract": (
            "strict Windows Arial Regular/Bold with exact resolved-font verification"
        ),
        "resolved_figure_font": resolved_figure_font,
        "figure_save_contract": (
            "same-canvas PNG plus strict PowerPoint-native editable PPTX; "
            "fallback_to_vector_object=False"
        ),
        "source_hashes": source_hashes_before,
        "large_inputs": large_inputs,
        "input_handoffs": input_handoffs,
        "active_runtime_provenance": active_runtime_provenance,
        "table_hash_contract": "SHA-256 of exact saved file bytes",
        "table_artifacts": _table_artifacts(partial_root),
        "output_files_before_manifest": sorted(
            str(path.relative_to(partial_root))
            for path in partial_root.rglob("*")
            if path.is_file()
        ),
    }
    _validate_generated_manifest_provenance(manifest)
    _write_exact_derived_json(
        manifests_root / "run_manifest.json",
        manifest,
        owner="Step11 fixed-signature run manifest",
    )
    _validate_fixed_signature_table_semantics(partial_root, manifest)
    for relative, record in manifest["table_artifacts"].items():
        staged = _require_within(
            partial_root / relative,
            tables_root,
            label="staged derived table",
        )
        if (
            not staged.is_file()
            or int(staged.stat().st_size) != int(record["size_bytes"])
            or _sha256(staged) != record["sha256"]
        ):
            raise RuntimeError(f"Staged derived table validation failed: {relative}")
    if args.replace_canonical_derived:
        _commit_canonical_derived_run(partial_root, final_root)
    else:
        partial_root.rename(final_root)
    completed_report = validate_fixed_signature_run(run_id, verify_artifacts=True)
    print(
        json.dumps(
            {
                **preflight,
                "status": "complete",
                "output_root": str(final_root),
                "artifacts_verified": completed_report["artifacts_verified"],
                "semantic_contract_verified": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
