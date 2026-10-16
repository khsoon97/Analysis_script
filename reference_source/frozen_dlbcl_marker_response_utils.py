#!/usr/bin/env python3
"""Final-owned frozen XGB+MLP DLBCL marker-response analysis.

This module computes support-restricted first-order accumulated local effects
(ALE) from the completed Step02 broad and Step04B subtype model packages.  It
never fits a model and never writes to a checkpoint, fitted-model directory,
or completed attribution run.  The caller supplies one fixed review-output
root; an execution fails closed when that root already exists.

The ``fast`` profile changes only how many feature endpoint scenarios are
stacked in one inference request.  It retains every eligible DLBCL cell, all
50 features, all 100 iteration-matched XGB+MLP pairs, the same quantile edges,
the same cell chunks, and the same patient-first reductions as ``exact``.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Sequence

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb


SCRIPTS_ROOT = Path(__file__).resolve().parent
FINAL_ROOT = SCRIPTS_ROOT.parent
import class_specific_weight_extraction_utils as frozen_utils  # noqa: E402


ITERATIONS = tuple(range(1, 101))
MODEL_KEYS = ("XGB", "MLP")
FEATURE_COUNT = 50
QUANTILE_BINS = 10
CONSISTENCY_THRESHOLD = 0.80
ZERO_TOLERANCE = 1e-12
CELL_CHUNK_SIZE = 500_000
INFERENCE_BATCH_SIZE = 8_192
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 6, "mtime": 0}


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    features_per_request: int
    cell_chunk_size: int = CELL_CHUNK_SIZE
    inference_batch_size: int = INFERENCE_BATCH_SIZE


PROFILES = {
    "exact": ExecutionProfile(name="exact", features_per_request=1),
    "fast": ExecutionProfile(name="fast", features_per_request=25),
}


@dataclass(frozen=True)
class FrozenResponseStage:
    stage_key: str
    lineage: str
    checkpoint_path: Path
    model_root: Path
    source_run_root: Path
    output_root: Path
    route_column: str | None
    route_value: str | None
    saved_prediction_column: str
    expected_classes: tuple[str, ...]
    feature_count: int = FEATURE_COUNT


@dataclass(frozen=True)
class ResponseRunConfig:
    workflow: str
    stages: tuple[FrozenResponseStage, ...]
    execution_profile: str = "fast"
    quantile_bins: int = QUANTILE_BINS
    consistency_threshold: float = CONSISTENCY_THRESHOLD
    zero_tolerance: float = ZERO_TOLERANCE
    random_seed: int = 42


@dataclass(frozen=True)
class FeatureSupport:
    feature: str
    feature_position: int
    edges: np.ndarray
    bin_codes: np.ndarray
    patient_bin_counts: np.ndarray
    q10: float
    q90: float
    central_bin_mask: np.ndarray
    constant_or_flat_support: bool


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [json_ready(item) for item in list(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"dlbcl_marker_response_{path.parent.parent.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(path, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _close_file_logger(logger: logging.Logger) -> None:
    """Release the run.log handle before renaming a directory on drvfs/NTFS."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


def _available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return 0
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if value:
            values[key] = int(value.strip().split()[0]) * 1024
    return int(values.get("MemAvailable", 0))


def memory_plan(
    n_cells: int,
    n_features: int,
    n_classes: int,
    profile: ExecutionProfile,
) -> dict[str, Any]:
    chunk_cells = min(int(n_cells), profile.cell_chunk_size)
    scenario_rows = 2 * profile.features_per_request * chunk_cells
    feature_matrix_bytes = scenario_rows * n_features * 4
    probability_bytes = scenario_rows * n_classes * (4 + 4 + 8)
    # Parent raw/scaled matrices, RAM-backed exchange input, worker input, and
    # probability buffers.  The multiplier is deliberately conservative.
    estimated_peak = int(feature_matrix_bytes * 4.5 + probability_bytes * 1.5)
    available = _available_memory_bytes()
    return {
        "profile": profile.name,
        "eligible_cells": int(n_cells),
        "cell_chunk_size": int(chunk_cells),
        "features_per_request": int(profile.features_per_request),
        "scenario_rows_per_request": int(scenario_rows),
        "inference_batch_size": int(profile.inference_batch_size),
        "estimated_peak_host_bytes": estimated_peak,
        "estimated_peak_host_gib": estimated_peak / (1024**3),
        "available_host_bytes": available,
        "available_host_gib": available / (1024**3) if available else None,
        "passes_50_percent_available_memory_guard": bool(
            not available or estimated_peak <= available * 0.50
        ),
    }


def _manifest(stage: FrozenResponseStage) -> dict[str, Any]:
    manifest_path = stage.model_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete frozen model package: {manifest_path}")
    if manifest.get("selected_combo") != "XGB+MLP":
        raise RuntimeError(f"Expected XGB+MLP package: {manifest_path}")
    if int(manifest.get("selected_iterations", -1)) != len(ITERATIONS):
        raise RuntimeError(f"Expected exactly 100 selected iterations: {manifest_path}")
    if tuple(map(str, manifest.get("classes", []))) != stage.expected_classes:
        raise RuntimeError(
            f"Class order mismatch for {stage.stage_key}: "
            f"{manifest.get('classes')} != {stage.expected_classes}"
        )
    features = tuple(map(str, manifest.get("feature_names", [])))
    if len(features) != stage.feature_count or len(set(features)) != stage.feature_count:
        raise RuntimeError(
            f"Expected {stage.feature_count} unique features for {stage.stage_key}"
        )
    available = set(map(int, manifest["available_iterations"]["XGB"]))
    available &= set(map(int, manifest["available_iterations"]["MLP"]))
    available &= set(map(int, manifest.get("scaler_iterations", [])))
    if tuple(sorted(value for value in available if value <= 100)) != ITERATIONS:
        raise RuntimeError(f"Incomplete scaler/XGB/MLP iteration set: {stage.model_root}")
    for iteration in ITERATIONS:
        iteration_root = frozen_utils.iteration_dir(stage.model_root, iteration)
        for filename in ("iteration.json", "scaler.joblib", "xgb_model.json", "mlp_model.pt"):
            if not (iteration_root / filename).is_file():
                raise FileNotFoundError(iteration_root / filename)
    return manifest


def _source_tables(stage: FrozenResponseStage) -> tuple[Path, Path, Path]:
    manifest_path = stage.source_run_root / "manifests" / "run_manifest.json"
    importance_path = stage.source_run_root / "tables" / "class_feature_joint_importance_summary.csv"
    frequency_path = stage.source_run_root / "tables" / "paired_iteration_top20_attribution_frequency.csv"
    for path in (manifest_path, importance_path, frequency_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete" or not source_manifest.get("input_unchanged"):
        raise RuntimeError(f"Source attribution run is not complete and immutable: {manifest_path}")
    return manifest_path, importance_path, frequency_path


def _canonical_input_audit(stage: FrozenResponseStage) -> pd.DataFrame:
    path = stage.source_run_root / "tables" / "selected_read_only_input_audit.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    audit = pd.read_csv(path)
    required = {"path", "size", "mtime_ns", "sha256"}
    if not required.issubset(audit.columns):
        raise RuntimeError(f"Malformed canonical input audit: {path}")
    return audit


def _verify_canonical_inputs(stage: FrozenResponseStage) -> pd.DataFrame:
    audit = _canonical_input_audit(stage)
    relevant_prefixes = (
        str(stage.checkpoint_path.resolve()),
        str(stage.model_root.resolve()) + os.sep,
    )
    selected = audit[
        audit["path"].astype(str).map(
            lambda value: value == relevant_prefixes[0] or value.startswith(relevant_prefixes[1])
        )
    ].copy()
    if selected.empty:
        raise RuntimeError(f"Canonical audit contains no inputs for {stage.stage_key}")
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        path = Path(str(row.path))
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        same_stat = int(stat.st_size) == int(row.size) and int(stat.st_mtime_ns) == int(row.mtime_ns)
        hash_verified = False
        if not same_stat:
            observed_hash = sha256_file(path)
            hash_verified = observed_hash == str(row.sha256)
            if not hash_verified:
                raise RuntimeError(f"Read-only input changed from canonical audit: {path}")
        records.append(
            {
                "stage": stage.stage_key,
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "canonical_sha256": str(row.sha256),
                "verification": "size_mtime_match" if same_stat else "sha256_match",
                "hash_recomputed": bool(not same_stat),
                "verified": True,
            }
        )
    return pd.DataFrame(records)


def _read_stage_cells(
    stage: FrozenResponseStage,
    features: Sequence[str],
) -> tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not stage.checkpoint_path.is_file():
        raise FileNotFoundError(stage.checkpoint_path)
    adata = ad.read_h5ad(stage.checkpoint_path, backed="r")
    required_obs = {"Disease", "PatientID", stage.saved_prediction_column}
    if stage.route_column is not None:
        required_obs.add(stage.route_column)
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    if missing_obs:
        adata.file.close()
        raise KeyError(f"Missing obs columns in {stage.checkpoint_path}: {missing_obs}")
    disease = adata.obs["Disease"].astype("string").str.strip()
    exact_dlbcl = disease.eq("DLBCL")
    contains_dlbcl = disease.str.contains("DLBCL", na=False)
    if not np.array_equal(
        np.flatnonzero(exact_dlbcl.to_numpy(dtype=bool)),
        np.flatnonzero(contains_dlbcl.to_numpy(dtype=bool)),
    ):
        adata.file.close()
        raise RuntimeError(
            "Disease exact-equality and legacy contains('DLBCL') masks differ; "
            "routing contract requires explicit review"
        )
    mask = exact_dlbcl
    if stage.route_column is not None:
        mask = mask & adata.obs[stage.route_column].astype("string").eq(stage.route_value)
    selected_idx = np.flatnonzero(mask.to_numpy(dtype=bool))
    if not len(selected_idx):
        adata.file.close()
        raise RuntimeError(f"No eligible DLBCL cells for {stage.stage_key}")
    var_names = list(map(str, adata.var_names))
    missing_features = [feature for feature in features if feature not in var_names]
    if missing_features:
        adata.file.close()
        raise KeyError(f"Missing frozen features for {stage.stage_key}: {missing_features}")
    feature_idx = np.asarray([var_names.index(feature) for feature in features], dtype=int)
    backed_matrix = adata.X
    full_matrix_bytes = int(
        np.prod(backed_matrix.shape, dtype=np.int64)
        * np.dtype(getattr(backed_matrix, "dtype", np.float32)).itemsize
    )
    use_full_sequential_read = bool(
        len(selected_idx) >= 50_000
        and not sp.issparse(backed_matrix)
        and full_matrix_bytes <= int(0.25 * _available_memory_bytes())
    )
    if use_full_sequential_read:
        full_matrix = np.asarray(backed_matrix[:], dtype=np.float32)
        selected_matrix = np.asarray(full_matrix[selected_idx], dtype=np.float32)
        x_raw = np.ascontiguousarray(selected_matrix[:, feature_idx], dtype=np.float32)
        del selected_matrix, full_matrix
        matrix_read_strategy = "full_dense_sequential_then_memory_row_selection"
    else:
        x_raw = frozen_utils.read_backed_rows(adata, selected_idx, feature_idx)
        matrix_read_strategy = "backed_selected_row_read"
    patients = (
        adata.obs.iloc[selected_idx]["PatientID"].astype("string").to_numpy(dtype=str)
    )
    saved_prediction = (
        adata.obs.iloc[selected_idx][stage.saved_prediction_column]
        .astype("string")
        .to_numpy(dtype=str)
    )
    audit = {
        "stage": stage.stage_key,
        "lineage": stage.lineage,
        "checkpoint": str(stage.checkpoint_path.resolve()),
        "total_checkpoint_cells": int(adata.n_obs),
        "eligible_dlbcl_cells": int(len(selected_idx)),
        "patients": int(pd.Series(patients).nunique()),
        "route_column": stage.route_column,
        "route_value": stage.route_value,
        "saved_prediction_column": stage.saved_prediction_column,
        "exact_equals_contains_dlbcl_mask": True,
        "matrix_read_strategy": matrix_read_strategy,
        "full_matrix_bytes": full_matrix_bytes,
    }
    return adata, x_raw, patients, saved_prediction, audit


def _feature_supports(
    features: Sequence[str],
    x_raw: np.ndarray,
    patient_codes: np.ndarray,
    n_patients: int,
    quantile_bins: int,
    zero_tolerance: float,
) -> list[FeatureSupport]:
    quantiles = np.linspace(0.0, 1.0, quantile_bins + 1)
    supports: list[FeatureSupport] = []
    for feature_position, feature in enumerate(features):
        values = np.asarray(x_raw[:, feature_position], dtype=np.float64)
        raw_edges = np.quantile(values, quantiles, method="linear")
        edges = np.unique(raw_edges)
        q10 = float(np.quantile(values, 0.10, method="linear"))
        q90 = float(np.quantile(values, 0.90, method="linear"))
        constant = len(edges) < 2 or abs(q90 - q10) <= zero_tolerance
        if len(edges) < 2:
            edges = np.asarray([float(values[0]), float(values[0])], dtype=np.float64)
            bin_codes = np.zeros(len(values), dtype=np.int16)
            central_mask = np.asarray([False], dtype=bool)
        else:
            bin_codes = np.searchsorted(edges, values, side="right") - 1
            bin_codes = np.clip(bin_codes, 0, len(edges) - 2).astype(np.int16)
            central_mask = (
                (edges[:-1] >= q10 - zero_tolerance)
                & (edges[1:] <= q90 + zero_tolerance)
            )
            if not central_mask.any():
                constant = True
        n_bins = len(edges) - 1
        combined = patient_codes.astype(np.int64) * n_bins + bin_codes.astype(np.int64)
        counts = np.bincount(combined, minlength=n_patients * n_bins).reshape(
            n_patients, n_bins
        )
        supports.append(
            FeatureSupport(
                feature=str(feature),
                feature_position=int(feature_position),
                edges=np.asarray(edges, dtype=np.float64),
                bin_codes=bin_codes,
                patient_bin_counts=counts.astype(np.int64),
                q10=q10,
                q90=q90,
                central_bin_mask=central_mask,
                constant_or_flat_support=bool(constant),
            )
        )
    return supports


def _support_table(
    stage: FrozenResponseStage,
    supports: Sequence[FeatureSupport],
    patient_order: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for support in supports:
        for bin_position in range(len(support.edges) - 1):
            counts = support.patient_bin_counts[:, bin_position]
            rows.append(
                {
                    "stage": stage.stage_key,
                    "lineage": stage.lineage,
                    "feature": support.feature,
                    "feature_position": support.feature_position,
                    "bin": int(bin_position),
                    "lower_raw_value": float(support.edges[bin_position]),
                    "upper_raw_value": float(support.edges[bin_position + 1]),
                    "q10_raw_value": support.q10,
                    "q90_raw_value": support.q90,
                    "inside_q10_q90_contrast": bool(support.central_bin_mask[bin_position]),
                    "n_cells": int(counts.sum()),
                    "n_patients": int(np.count_nonzero(counts)),
                    "patient_count_total": int(len(patient_order)),
                    "constant_or_flat_support": support.constant_or_flat_support,
                }
            )
    return pd.DataFrame(rows)


def _load_iteration_pair(
    stage: FrozenResponseStage,
    iteration: int,
) -> tuple[Any, xgb.XGBClassifier, Path]:
    iteration_root = frozen_utils.iteration_dir(stage.model_root, iteration)
    scaler = joblib.load(iteration_root / "scaler.joblib")
    model = xgb.XGBClassifier(device="cpu")
    model.load_model(str(iteration_root / "xgb_model.json"))
    model.get_booster().set_param({"device": "cpu"})
    return scaler, model, iteration_root / "mlp_model.pt"


def _pair_probabilities(
    scaler: Any,
    xgb_model: xgb.XGBClassifier,
    mlp_path: Path,
    x_raw: np.ndarray,
    n_classes: int,
    worker: Any,
    batch_size: int,
    overlap_pool: ThreadPoolExecutor,
    scenario_feature_count: int = 1,
) -> np.ndarray:
    if scenario_feature_count < 1 or len(x_raw) % scenario_feature_count:
        raise ValueError(
            "scenario_feature_count must evenly partition the stacked scenario rows"
        )
    x_scaled = np.ascontiguousarray(scaler.transform(x_raw), dtype=np.float32)
    rows_per_feature = len(x_scaled) // scenario_feature_count

    def predict_mlp_with_exact_boundaries() -> np.ndarray:
        segments = []
        for feature_position in range(scenario_feature_count):
            start = feature_position * rows_per_feature
            stop = start + rows_per_feature
            segments.append(worker.predict(mlp_path, x_scaled[start:stop], batch_size))
        return np.concatenate(segments, axis=0)

    mlp_future = overlap_pool.submit(predict_mlp_with_exact_boundaries)
    xgb_probability = frozen_utils._align_probability_columns(
        xgb_model.predict_proba(x_scaled),
        getattr(xgb_model, "classes_", None),
        n_classes,
    )
    mlp_probability = frozen_utils._align_probability_columns(
        mlp_future.result(), None, n_classes
    )
    return (
        np.asarray(xgb_probability, dtype=np.float64)
        + np.asarray(mlp_probability, dtype=np.float64)
    ) / 2.0


def _scenario_stack(
    x_chunk: np.ndarray,
    supports: Sequence[FeatureSupport],
    cell_start: int,
    cell_stop: int,
) -> np.ndarray:
    n_cells = cell_stop - cell_start
    stacked = np.tile(x_chunk, (2 * len(supports), 1))
    for local_position, support in enumerate(supports):
        codes = support.bin_codes[cell_start:cell_stop]
        lower = support.edges[codes]
        upper = support.edges[codes + 1]
        lower_block = stacked[(2 * local_position) * n_cells : (2 * local_position + 1) * n_cells]
        upper_block = stacked[(2 * local_position + 1) * n_cells : (2 * local_position + 2) * n_cells]
        lower_block[:, support.feature_position] = lower
        upper_block[:, support.feature_position] = upper
    return np.ascontiguousarray(stacked, dtype=np.float32)


def _patient_bin_means(
    sums: np.ndarray,
    support: FeatureSupport,
) -> np.ndarray:
    counts = support.patient_bin_counts
    return np.divide(
        sums,
        counts[:, :, None],
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts[:, :, None] > 0,
    )


def _patient_equal_bin_effect(patient_bin_mean: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(patient_bin_mean, axis=0)


def _centered_ale(bin_effect: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    edge_values = np.vstack(
        [np.zeros((1, bin_effect.shape[1]), dtype=np.float64), np.cumsum(bin_effect, axis=0)]
    )
    midpoint_values = (edge_values[:-1] + edge_values[1:]) / 2.0
    weights = np.asarray(bin_counts, dtype=np.float64)
    if weights.sum() <= 0:
        return midpoint_values
    center = np.sum(midpoint_values * weights[:, None], axis=0) / weights.sum()
    return midpoint_values - center[None, :]


def _direction_symbol(
    mean_response: float,
    pair_consistency: float,
    support_concordance: float,
    threshold: float,
    tolerance: float,
) -> tuple[str, str, float, float]:
    if abs(mean_response) <= tolerance:
        return "±", "flat_or_ambiguous", 0.0, 0.45
    consistency = min(pair_consistency, support_concordance)
    alpha = float(np.clip(0.45 + 0.50 * (consistency - 0.80) / 0.20, 0.45, 0.95))
    if pair_consistency < threshold or support_concordance < threshold:
        return "±", "nonmonotonic_or_pair_unstable", consistency, alpha
    if mean_response > 0:
        return "+", "higher_marker_increases_probability", consistency, alpha
    return "−", "higher_marker_decreases_probability", consistency, alpha


def _source_join(
    stage: FrozenResponseStage,
    response_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, importance_path, frequency_path = _source_tables(stage)
    importance = pd.read_csv(importance_path)
    frequency = pd.read_csv(frequency_path)
    keys = ["stage", "lineage", "class", "feature"]
    importance_columns = keys + [
        "signed_probability_drop_in_baseline_class",
        "class_normalized_importance",
        "class_normalized_importance_percent",
        "rank_within_class",
    ]
    frequency_columns = keys + [
        "positive_top_k_count",
        "positive_top_k_attribution_frequency",
        "positive_top_k_attribution_frequency_percent",
        "n_iterations",
        "top_k",
    ]
    joined = response_summary.merge(
        importance[importance_columns], on=keys, how="left", validate="one_to_one"
    ).merge(
        frequency[frequency_columns].rename(
            columns={
                "n_iterations": "importance_frequency_n_iterations",
                "top_k": "importance_frequency_top_k",
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    required = [
        "class_normalized_importance_percent",
        "positive_top_k_attribution_frequency_percent",
    ]
    joined["source_join_complete"] = ~joined[required].isna().any(axis=1)
    audit = joined[keys + ["source_join_complete"]].copy()
    if not bool(audit["source_join_complete"].all()):
        raise RuntimeError(
            f"ALE response did not key-join to pinned importance/frequency for {stage.stage_key}"
        )
    joined["importance_source_run"] = str(stage.source_run_root.resolve())
    joined["response_scope"] = "all eligible DLBCL cells in the stage routing cohort"
    joined["importance_scope"] = "fixed baseline-predicted class cells"
    return joined, audit


def _prepare_output(config: ResponseRunConfig) -> dict[str, Path]:
    output_roots = {stage.output_root.resolve() for stage in config.stages}
    if len(output_roots) != 1:
        raise ValueError("Every stage in one execution must share one fixed output root")
    output_root = next(iter(output_roots))
    review_root = output_root.parent.resolve()
    output_root.relative_to(review_root)
    if output_root.exists():
        raise FileExistsError(
            f"Fixed Final recalculation output already exists and will not be overwritten: {output_root}"
        )
    staging_root = output_root.with_name(f".{output_root.name}.staging")
    if staging_root.exists():
        raise FileExistsError(
            "Incomplete staging output already exists and requires explicit review: "
            f"{staging_root}"
        )
    paths = {"run": staging_root, "final": output_root}
    for name in ("code", "logs", "manifests", "tables", "Figure"):
        path = staging_root / name
        path.mkdir(parents=True, exist_ok=False)
        paths[name] = path
    return paths


def _copy_code_snapshot(paths: dict[str, Path], caller_paths: Iterable[Path]) -> None:
    sources = [Path(__file__).resolve(), *map(Path, caller_paths)]
    for source in sources:
        if source.is_file():
            shutil.copy2(source, paths["code"] / source.name)


def preflight(config: ResponseRunConfig) -> dict[str, Any]:
    if config.execution_profile not in PROFILES:
        raise ValueError(f"Unknown execution profile: {config.execution_profile}")
    profile = PROFILES[config.execution_profile]
    stage_reports: list[dict[str, Any]] = []
    for stage in config.stages:
        manifest = _manifest(stage)
        _source_tables(stage)
        adata, x_raw, patients, _, audit = _read_stage_cells(
            stage, list(map(str, manifest["feature_names"]))
        )
        try:
            plan = memory_plan(
                n_cells=len(x_raw),
                n_features=x_raw.shape[1],
                n_classes=len(stage.expected_classes),
                profile=profile,
            )
            if not plan["passes_50_percent_available_memory_guard"]:
                raise MemoryError(
                    f"Fast scenario plan exceeds 50% of available RAM for {stage.stage_key}: {plan}"
                )
            stage_reports.append(
                {
                    **audit,
                    "classes": list(stage.expected_classes),
                    "features": int(x_raw.shape[1]),
                    "iterations": len(ITERATIONS),
                    "quantile_bins": config.quantile_bins,
                    "memory_plan": plan,
                }
            )
        finally:
            adata.file.close()
            del x_raw
    return {
        "workflow": config.workflow,
        "execution_requested": False,
        "retraining": False,
        "execution_profile": asdict(profile),
        "result_invariant_contract": {
            "iterations": len(ITERATIONS),
            "features_per_stage": FEATURE_COUNT,
            "eligible_cell_sampling": "none",
            "quantile_bins": config.quantile_bins,
            "cell_chunk_size_same_in_exact_and_fast": CELL_CHUNK_SIZE,
            "patient_first_aggregation": True,
            "fast_only_changes_features_stacked_per_request": True,
        },
        "stages": stage_reports,
    }


def synthetic_exact_fast_smoke() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    n_cells, n_features, n_classes = 37, 30, 3
    x_raw = rng.normal(size=(n_cells, n_features)).astype(np.float32)
    patients = np.asarray(["P1", "P2", "P3", "P4"] * 10)[:n_cells]
    patient_order = sorted(pd.unique(patients))
    patient_lookup = {patient: index for index, patient in enumerate(patient_order)}
    patient_codes = np.asarray([patient_lookup[value] for value in patients], dtype=np.int16)
    supports = _feature_supports(
        [f"F{index}" for index in range(n_features)],
        x_raw,
        patient_codes,
        len(patient_order),
        quantile_bins=5,
        zero_tolerance=ZERO_TOLERANCE,
    )

    def fake_predict(stacked: np.ndarray) -> np.ndarray:
        logits = np.stack(
            [stacked[:, 0] - stacked[:, 1], stacked[:, 2] + 0.2 * stacked[:, 3], stacked[:, 4]],
            axis=1,
        )
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float64)
        return exp / exp.sum(axis=1, keepdims=True)

    def reduce(feature_group_size: int) -> np.ndarray:
        result = np.zeros((n_features, len(patient_order), 5, n_classes), dtype=np.float64)
        for cell_start in range(0, n_cells, 11):
            cell_stop = min(cell_start + 11, n_cells)
            x_chunk = x_raw[cell_start:cell_stop]
            for feature_start in range(0, n_features, feature_group_size):
                group = supports[feature_start : feature_start + feature_group_size]
                stack = _scenario_stack(x_chunk, group, cell_start, cell_stop)
                probability = fake_predict(stack).reshape(len(group), 2, cell_stop - cell_start, n_classes)
                delta = probability[:, 1] - probability[:, 0]
                for local_position, support in enumerate(group):
                    codes = (
                        patient_codes[cell_start:cell_stop].astype(np.int64) * 5
                        + support.bin_codes[cell_start:cell_stop].astype(np.int64)
                    )
                    flat = result[support.feature_position].reshape(-1, n_classes)
                    np.add.at(flat, codes, delta[local_position])
        return result

    exact = reduce(1)
    fast_features_per_request = min(PROFILES["fast"].features_per_request, n_features)
    fast = reduce(fast_features_per_request)
    return {
        "status": "pass" if np.array_equal(exact, fast) else "fail",
        "array_equal": bool(np.array_equal(exact, fast)),
        "max_absolute_difference": float(np.max(np.abs(exact - fast))),
        "exact_features_per_request": 1,
        "fast_features_per_request": fast_features_per_request,
        "cell_chunk_order_identical": True,
    }


def _stage_response(
    stage: FrozenResponseStage,
    config: ResponseRunConfig,
    profile: ExecutionProfile,
    paths: dict[str, Path],
    worker: Any,
    overlap_pool: ThreadPoolExecutor,
    patient_writer: csv.DictWriter,
    logger: logging.Logger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    manifest = _manifest(stage)
    features = list(map(str, manifest["feature_names"]))
    classes = list(stage.expected_classes)
    adata, x_raw, patients, saved_prediction, stage_audit = _read_stage_cells(
        stage, features
    )
    try:
        patient_order = [str(value) for value in sorted(pd.unique(patients))]
        patient_lookup = {patient: index for index, patient in enumerate(patient_order)}
        patient_codes = np.asarray([patient_lookup[str(value)] for value in patients], dtype=np.int16)
        supports = _feature_supports(
            features,
            x_raw,
            patient_codes,
            len(patient_order),
            config.quantile_bins,
            config.zero_tolerance,
        )
        support_table = _support_table(stage, supports, patient_order)
        n_cells = len(x_raw)
        n_classes = len(classes)
        plan = memory_plan(n_cells, len(features), n_classes, profile)
        if not plan["passes_50_percent_available_memory_guard"]:
            raise MemoryError(f"Memory guard failed for {stage.stage_key}: {plan}")
        logger.info(
            "%s: %d cells, %d patients, %d classes, %d features; %s profile, %d features/request",
            stage.lineage,
            n_cells,
            len(patient_order),
            n_classes,
            len(features),
            profile.name,
            profile.features_per_request,
        )

        pair_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        baseline_sum = np.zeros((n_cells, n_classes), dtype=np.float64)
        for iteration in ITERATIONS:
            scaler, xgb_model, mlp_path = _load_iteration_pair(stage, iteration)
            baseline_sum += _pair_probabilities(
                scaler,
                xgb_model,
                mlp_path,
                x_raw,
                n_classes,
                worker,
                profile.inference_batch_size,
                overlap_pool,
            )
            feature_sums = [
                np.zeros(
                    (len(patient_order), len(support.edges) - 1, n_classes),
                    dtype=np.float64,
                )
                for support in supports
            ]
            for cell_start in range(0, n_cells, profile.cell_chunk_size):
                cell_stop = min(cell_start + profile.cell_chunk_size, n_cells)
                x_chunk = x_raw[cell_start:cell_stop]
                for feature_start in range(0, len(features), profile.features_per_request):
                    feature_group = supports[
                        feature_start : feature_start + profile.features_per_request
                    ]
                    stacked = _scenario_stack(x_chunk, feature_group, cell_start, cell_stop)
                    probability = _pair_probabilities(
                        scaler,
                        xgb_model,
                        mlp_path,
                        stacked,
                        n_classes,
                        worker,
                        profile.inference_batch_size,
                        overlap_pool,
                        scenario_feature_count=len(feature_group),
                    ).reshape(len(feature_group), 2, cell_stop - cell_start, n_classes)
                    delta = probability[:, 1] - probability[:, 0]
                    for local_position, support in enumerate(feature_group):
                        n_bins = len(support.edges) - 1
                        group_codes = (
                            patient_codes[cell_start:cell_stop].astype(np.int64) * n_bins
                            + support.bin_codes[cell_start:cell_stop].astype(np.int64)
                        )
                        flat = feature_sums[support.feature_position].reshape(-1, n_classes)
                        np.add.at(flat, group_codes, delta[local_position])

            for support in supports:
                patient_bin_mean = _patient_bin_means(
                    feature_sums[support.feature_position], support
                )
                bin_effect = _patient_equal_bin_effect(patient_bin_mean)
                total_bin_counts = support.patient_bin_counts.sum(axis=0)
                ale_curve = _centered_ale(bin_effect, total_bin_counts)
                central = support.central_bin_mask
                response = (
                    np.sum(bin_effect[central], axis=0)
                    if central.any() and not support.constant_or_flat_support
                    else np.zeros(n_classes, dtype=np.float64)
                )
                for patient_position, patient in enumerate(patient_order):
                    for bin_position in range(len(support.edges) - 1):
                        count = int(support.patient_bin_counts[patient_position, bin_position])
                        for class_position, class_name in enumerate(classes):
                            value = patient_bin_mean[patient_position, bin_position, class_position]
                            patient_writer.writerow(
                                {
                                    "stage": stage.stage_key,
                                    "lineage": stage.lineage,
                                    "iteration": iteration,
                                    "patient": patient,
                                    "class": class_name,
                                    "feature": support.feature,
                                    "feature_position": support.feature_position,
                                    "bin": bin_position,
                                    "lower_raw_value": support.edges[bin_position],
                                    "upper_raw_value": support.edges[bin_position + 1],
                                    "n_cells": count,
                                    "local_probability_change": value if np.isfinite(value) else "",
                                }
                            )
                for bin_position in range(len(support.edges) - 1):
                    for class_position, class_name in enumerate(classes):
                        curve_rows.append(
                            {
                                "stage": stage.stage_key,
                                "lineage": stage.lineage,
                                "iteration": iteration,
                                "class": class_name,
                                "feature": support.feature,
                                "feature_position": support.feature_position,
                                "bin": bin_position,
                                "lower_raw_value": float(support.edges[bin_position]),
                                "upper_raw_value": float(support.edges[bin_position + 1]),
                                "n_cells": int(total_bin_counts[bin_position]),
                                "n_patients": int(
                                    np.count_nonzero(support.patient_bin_counts[:, bin_position])
                                ),
                                "patient_equal_local_probability_change": float(
                                    bin_effect[bin_position, class_position]
                                ),
                                "centered_ale_probability": float(
                                    ale_curve[bin_position, class_position]
                                ),
                                "inside_q10_q90_contrast": bool(central[bin_position]),
                            }
                        )
                for class_position, class_name in enumerate(classes):
                    pair_rows.append(
                        {
                            "stage": stage.stage_key,
                            "lineage": stage.lineage,
                            "iteration": iteration,
                            "class": class_name,
                            "feature": support.feature,
                            "feature_position": support.feature_position,
                            "q10_raw_value": support.q10,
                            "q90_raw_value": support.q90,
                            "ale_q10_to_q90_probability_change": float(
                                response[class_position]
                            ),
                            "constant_or_flat_support": support.constant_or_flat_support,
                        }
                    )
            if iteration == 1 or iteration % 10 == 0:
                logger.info("%s ALE pair %d/100 complete", stage.lineage, iteration)
            del scaler, xgb_model

        baseline_probability = baseline_sum / len(ITERATIONS)
        baseline_prediction = np.asarray(classes, dtype=object)[
            baseline_probability.argmax(axis=1)
        ].astype(str)
        agreement = pd.DataFrame(
            {
                "stage": [stage.stage_key],
                "lineage": [stage.lineage],
                "eligible_dlbcl_cells": [n_cells],
                "patients": [len(patient_order)],
                "saved_prediction_column": [stage.saved_prediction_column],
                "saved_prediction_agreement": [
                    float(np.mean(baseline_prediction == saved_prediction))
                ],
            }
        )
        worker.clear_model_cache()
        pair_df = pd.DataFrame(pair_rows)
        curve_df = pd.DataFrame(curve_rows)

        summary_rows: list[dict[str, Any]] = []
        grouped_pairs = pair_df.groupby(
            ["stage", "lineage", "class", "feature", "feature_position"],
            sort=False,
            observed=True,
        )
        for keys, group in grouped_pairs:
            stage_key, lineage, class_name, feature, feature_position = keys
            responses = group["ale_q10_to_q90_probability_change"].to_numpy(dtype=float)
            mean_response = float(np.mean(responses))
            if mean_response > config.zero_tolerance:
                pair_consistency = float(np.mean(responses > config.zero_tolerance))
            elif mean_response < -config.zero_tolerance:
                pair_consistency = float(np.mean(responses < -config.zero_tolerance))
            else:
                pair_consistency = 0.0
            feature_curve = curve_df[
                (curve_df["class"] == class_name)
                & (curve_df["feature"] == feature)
                & curve_df["inside_q10_q90_contrast"]
            ]
            local_mean = (
                feature_curve.groupby("bin", sort=True)[
                    "patient_equal_local_probability_change"
                ]
                .mean()
                .to_numpy(dtype=float)
            )
            magnitude = float(np.sum(np.abs(local_mean)))
            if magnitude <= config.zero_tolerance or abs(mean_response) <= config.zero_tolerance:
                support_concordance = 0.0
            else:
                matching = local_mean > config.zero_tolerance if mean_response > 0 else local_mean < -config.zero_tolerance
                support_concordance = float(np.sum(np.abs(local_mean[matching])) / magnitude)
            symbol, label, consistency, alpha = _direction_symbol(
                mean_response,
                pair_consistency,
                support_concordance,
                config.consistency_threshold,
                config.zero_tolerance,
            )
            support = supports[int(feature_position)]
            summary_rows.append(
                {
                    "stage": stage_key,
                    "lineage": lineage,
                    "class": class_name,
                    "feature": feature,
                    "feature_position": int(feature_position),
                    "mean_ale_q10_to_q90_probability_change": mean_response,
                    "sd_ale_q10_to_q90_probability_change": float(
                        np.std(responses, ddof=1)
                    ),
                    "pair_sign_consistency": pair_consistency,
                    "support_sign_concordance": support_concordance,
                    "direction_consistency": consistency,
                    "direction_consistency_threshold": config.consistency_threshold,
                    "direction_symbol": symbol,
                    "direction_label": label,
                    "figure_alpha": alpha,
                    "q10_raw_value": support.q10,
                    "q90_raw_value": support.q90,
                    "n_iterations": len(responses),
                    "n_patients": len(patient_order),
                    "n_cells": n_cells,
                    "response_metric": "patient-equal first-order ALE q10-to-q90 contrast",
                    "response_is_causal": False,
                    "alpha_is_p_value": False,
                }
            )
        response_summary = pd.DataFrame(summary_rows)
        joined_summary, join_audit = _source_join(stage, response_summary)
        stage_report = {
            **stage_audit,
            "features": len(features),
            "classes": classes,
            "quantile_bins_requested": config.quantile_bins,
            "execution_profile": asdict(profile),
            "memory_plan": plan,
            "response_rows": int(len(joined_summary)),
            "pair_response_rows": int(len(pair_df)),
            "curve_rows": int(len(curve_df)),
        }
        return support_table, pair_df, curve_df, joined_summary, agreement, join_audit, stage_report
    finally:
        adata.file.close()
        del x_raw


def run(
    config: ResponseRunConfig,
    caller_paths: Iterable[Path] = (),
) -> Path:
    if config.execution_profile not in PROFILES:
        raise ValueError(config.execution_profile)
    if config.quantile_bins != QUANTILE_BINS:
        raise ValueError(f"quantile_bins is locked to {QUANTILE_BINS}")
    if not math.isclose(config.consistency_threshold, CONSISTENCY_THRESHOLD):
        raise ValueError(f"consistency_threshold is locked to {CONSISTENCY_THRESHOLD}")
    if config.random_seed != 42:
        raise ValueError("random_seed is locked to 42")
    smoke = synthetic_exact_fast_smoke()
    if smoke["status"] != "pass":
        raise RuntimeError(f"Synthetic exact-fast grouping smoke failed: {smoke}")

    paths = _prepare_output(config)
    logger = _setup_logger(paths["logs"] / "run.log")
    _copy_code_snapshot(paths, caller_paths)
    manifest_path = paths["manifests"] / "run_manifest.json"
    profile = PROFILES[config.execution_profile]
    manifest: dict[str, Any] = {
        "status": "running",
        "workflow": config.workflow,
        "output_root": str(paths["final"].resolve()),
        "temporary_staging_root": str(paths["run"].resolve()),
        "retraining": False,
        "source_models_read_only": True,
        "source_checkpoints_read_only": True,
        "source_attribution_tables_read_only": True,
        "config": json_ready(asdict(config)),
        "execution_profile": asdict(profile),
        "synthetic_exact_fast_smoke": smoke,
        "direction_contract": {
            "metric": "patient-equal first-order ALE q10-to-q90 contrast",
            "symbol_positive": "higher marker value increases class probability",
            "symbol_negative": "higher marker value decreases class probability",
            "symbol_ambiguous": "flat, nonmonotonic, or below consistency threshold",
            "consistency": "minimum of pair sign consistency and support sign concordance",
            "alpha": "descriptive consistency mapping; not a p-value or confidence interval",
        },
    }
    write_json(manifest_path, manifest)
    worker = None
    try:
        input_audits = [_verify_canonical_inputs(stage) for stage in config.stages]
        input_audit = pd.concat(input_audits, ignore_index=True)
        input_audit.to_csv(paths["tables"] / "selected_read_only_input_audit.csv", index=False)
        worker = frozen_utils.CudaMLPWorker(
            config.random_seed,
            Path("/dev/shm") / f"roider_{paths['run'].name}",
        )
        manifest["cuda_worker"] = worker.info
        write_json(manifest_path, manifest)

        support_frames: list[pd.DataFrame] = []
        pair_frames: list[pd.DataFrame] = []
        curve_frames: list[pd.DataFrame] = []
        summary_frames: list[pd.DataFrame] = []
        agreement_frames: list[pd.DataFrame] = []
        join_frames: list[pd.DataFrame] = []
        stage_reports: list[dict[str, Any]] = []
        patient_path = paths["tables"] / "paired_iteration_patient_bin_response.csv.gz"
        patient_fields = [
            "stage",
            "lineage",
            "iteration",
            "patient",
            "class",
            "feature",
            "feature_position",
            "bin",
            "lower_raw_value",
            "upper_raw_value",
            "n_cells",
            "local_probability_change",
        ]
        with gzip.open(patient_path, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
            patient_writer = csv.DictWriter(handle, fieldnames=patient_fields)
            patient_writer.writeheader()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="roider_mlp_cuda") as overlap_pool:
                for stage in config.stages:
                    (
                        support,
                        pair,
                        curve,
                        summary,
                        agreement,
                        join_audit,
                        stage_report,
                    ) = _stage_response(
                        stage,
                        config,
                        profile,
                        paths,
                        worker,
                        overlap_pool,
                        patient_writer,
                        logger,
                    )
                    support_frames.append(support)
                    pair_frames.append(pair)
                    curve_frames.append(curve)
                    summary_frames.append(summary)
                    agreement_frames.append(agreement)
                    join_frames.append(join_audit)
                    stage_reports.append(stage_report)

        outputs = {
            "ale_bin_support.csv": pd.concat(support_frames, ignore_index=True),
            "paired_iteration_class_feature_response.csv.gz": pd.concat(
                pair_frames, ignore_index=True
            ),
            "paired_iteration_ale_curve.csv.gz": pd.concat(curve_frames, ignore_index=True),
            "class_feature_marker_response_summary.csv": pd.concat(
                summary_frames, ignore_index=True
            ),
            "source_importance_frequency_join_audit.csv": pd.concat(
                join_frames, ignore_index=True
            ),
            "baseline_saved_annotation_agreement_audit.csv": pd.concat(
                agreement_frames, ignore_index=True
            ),
        }
        for filename, frame in outputs.items():
            compression = GZIP_COMPRESSION if filename.endswith(".gz") else None
            frame.to_csv(paths["tables"] / filename, index=False, compression=compression)

        # Existing inputs must retain the same size and mtime throughout execution.
        for row in input_audit.itertuples(index=False):
            stat = Path(row.path).stat()
            if int(stat.st_size) != int(row.size) or int(stat.st_mtime_ns) != int(row.mtime_ns):
                raise RuntimeError(f"Read-only input metadata changed during execution: {row.path}")
        manifest.update(
            {
                "status": "complete",
                "input_unchanged": True,
                "stage_reports": stage_reports,
                "output_tables": sorted(path.name for path in paths["tables"].iterdir()),
                "figure_directory": str((paths["final"] / "Figure").resolve()),
                "figure_generated": False,
            }
        )
        write_json(manifest_path, manifest)
        logger.info("FINALIZING: %s", paths["final"])
        _close_file_logger(logger)
        os.replace(paths["run"], paths["final"])
        logger.info("COMPLETE: %s", paths["final"])
        return paths["final"]
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "input_unchanged": False,
                "error": repr(exc),
            }
        )
        write_json(manifest_path, manifest)
        logger.exception("FAILED")
        raise
    finally:
        if worker is not None:
            worker.close()
